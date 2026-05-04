"""expINJ3: Three complementary experiments in one script.

Context (2026-04-14):
    INJ2F variant F2 showed 92% math content when replacing full KV cache.
    Literature (Dumas et al. 2024, Wendler et al. 2024) shows:
    - Models have language-agnostic concept spaces in middle layers
    - Mean concept vectors averaged across languages can improve transfer
    - Logit lens reveals English-biased intermediate representations

    This script runs three experiments that connect our findings to the literature:

    PART 1: TUNED LENS (connects to Wendler "Do Llamas Work in English?")
        Project hidden states at each layer through the unembedding matrix (lm_head).
        For each math problem in EN and ZH, what tokens does the model "think about"
        at each layer? If intermediate layers predict English tokens regardless of
        input language, that confirms the shared English-biased concept space.

    PART 2: MEAN-VECTOR KV INJECTION (connects to Dumas et al.)
        Average EN and ZH KV caches for the same problem. Inject the mean into a
        describe prompt using F2 approach. Compare to single-language F2.
        Dumas found mean vectors IMPROVE translation. Does mean KV improve description?

    PART 3: REVERSE INJECTION (novel, tests bidirectional h'∘f∘h)
        Inject describe-prompt KV cache into a math prompt. Does the model try to
        describe/explain instead of solving? If so, h and h' are truly separable:
        you can swap the "render" function while keeping the "compute" state.
        If it fails, the rendering function is more tightly coupled than h∘f∘h implies.

    Model: Qwen2.5-3B on RayGun. Estimated: ~15 min total.
"""
import json
import time
import copy
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_GEN = 256

PROBLEMS = [
    {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22",
     "answer": "5", "category": "algebra"},
    {"en": "What is the area of a triangle with base 10 and height 7?",
     "zh": "底边为10、高为7的三角形面积是多少？",
     "answer": "35", "category": "geometry"},
    {"en": "Find the GCD of 84 and 120", "zh": "求84和120的最大公约数",
     "answer": "12", "category": "number_theory"},
]

DESCRIBE_EN = "Describe in detail what mathematical operation or reasoning is being performed:"
DESCRIBE_ZH = "详细描述正在进行什么数学运算或推理："


def build_chat_prompt(tokenizer, text):
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def detect_lang(text):
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if cjk > len(text) * 0.1 else "en"


def encode_with_cache(model, tokenizer, text, device):
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    return {
        "past_kv": outputs.past_key_values,
        "hidden_states": outputs.hidden_states,
        "logits": outputs.logits,
        "input_ids": inputs["input_ids"],
        "seq_len": inputs["input_ids"].shape[1],
    }


def manual_generate(model, tokenizer, cache, first_logits, max_gen=MAX_GEN):
    """Manual greedy decode from a cache + first-token logits."""
    kv = copy.deepcopy(cache)
    next_token = first_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    eos = tokenizer.eos_token_id

    for _ in range(max_gen - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tid = next_token.item()
        if tid == eos:
            break
        generated.append(tid)

    return tokenizer.decode(generated, skip_special_tokens=True)


def swap_kv(target_kv, source_kv, target_seq_len, source_seq_len, n_layers=N_LAYERS):
    """Deep copy target_kv, replace all layers' K/V with source_kv (truncated/padded)."""
    hybrid = copy.deepcopy(target_kv)
    min_seq = min(target_seq_len, source_seq_len)

    for i in range(n_layers):
        mk = source_kv.layers[i].keys[:, :, :min_seq, :].clone()
        mv = source_kv.layers[i].values[:, :, :min_seq, :].clone()
        if min_seq < target_seq_len:
            pad = target_seq_len - min_seq
            mk = torch.cat([mk, mk[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
            mv = torch.cat([mv, mv[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
        hybrid.layers[i].keys = mk
        hybrid.layers[i].values = mv
    return hybrid


# ========================================================================
# PART 1: TUNED LENS
# ========================================================================

def run_tuned_lens(model, tokenizer, device, caches):
    """Project hidden states at each layer through lm_head, get top tokens."""
    print(f"\n{'#' * 70}", flush=True)
    print(f"  PART 1: TUNED LENS", flush=True)
    print(f"{'#' * 70}", flush=True)

    lm_head = model.lm_head  # Projection from hidden to vocab
    norm = model.model.norm  # Final RMSNorm (applied before lm_head)

    results = {}

    for pi, problem in enumerate(PROBLEMS):
        prob_results = {}
        for lang in ["en", "zh"]:
            cache = caches[f"p{pi}_{lang}"]
            hidden_states = cache["hidden_states"]  # [embed, L0, L1, ..., L35]

            layer_tokens = {}
            for L in range(N_LAYERS):
                # hidden_states[L+1] is the output of layer L (index 0 is embedding)
                h = hidden_states[L + 1]  # (batch, seq, d)
                h_last = h[:, -1, :]  # Last token (batch, d)

                # Apply final norm + lm_head (logit lens)
                with torch.no_grad():
                    normed = norm(h_last.float())
                    logits = lm_head(normed.half())  # (batch, vocab)

                # Top-5 tokens
                probs = F.softmax(logits.float(), dim=-1)
                top_vals, top_ids = probs.topk(5, dim=-1)

                top_tokens = []
                for k in range(5):
                    tid = top_ids[0, k].item()
                    tok_str = tokenizer.decode([tid])
                    prob = top_vals[0, k].item()
                    top_tokens.append({"token": tok_str, "id": tid, "prob": prob})

                layer_tokens[f"L{L}"] = top_tokens

            prob_results[lang] = layer_tokens
            # Print summary for key layers
            for L in [0, 9, 15, 18, 22, 27, 30, 33, 35]:
                toks = layer_tokens[f"L{L}"]
                tok_str = " | ".join(f"{t['token']!r}({t['prob']:.3f})" for t in toks[:3])
                print(f"  p{pi}_{lang} L{L:02d}: {tok_str}", flush=True)
            print(flush=True)

        # Check: are EN and ZH layers predicting the SAME tokens?
        en_layers = prob_results["en"]
        zh_layers = prob_results["zh"]
        agreement = {}
        for L in range(N_LAYERS):
            en_top = set(t["id"] for t in en_layers[f"L{L}"][:3])
            zh_top = set(t["id"] for t in zh_layers[f"L{L}"][:3])
            overlap = len(en_top & zh_top)
            agreement[f"L{L}"] = overlap / 3.0

        # Find layer where agreement peaks
        peak_L = max(agreement, key=agreement.get)
        print(f"  p{pi} EN/ZH top-3 agreement peaks at {peak_L} ({agreement[peak_L]:.2f})", flush=True)

        prob_results["agreement"] = agreement
        results[f"problem_{pi}"] = prob_results

    return results


# ========================================================================
# PART 2: MEAN-VECTOR KV INJECTION
# ========================================================================

def run_mean_kv(model, tokenizer, device, caches, desc_cache):
    """Average EN+ZH KV caches, inject into describe prompt, compare to single-lang."""
    print(f"\n{'#' * 70}", flush=True)
    print(f"  PART 2: MEAN-VECTOR KV INJECTION", flush=True)
    print(f"{'#' * 70}", flush=True)

    results = {}

    for pi, problem in enumerate(PROBLEMS):
        print(f"\n--- Problem {pi}: {problem['en'][:50]}... ---", flush=True)
        prob_results = {}

        en_cache = caches[f"p{pi}_en"]
        zh_cache = caches[f"p{pi}_zh"]
        desc = desc_cache["en"]

        # Build mean KV cache: average EN and ZH
        min_seq = min(en_cache["seq_len"], zh_cache["seq_len"])
        mean_kv = copy.deepcopy(en_cache["past_kv"])
        for i in range(N_LAYERS):
            ek = en_cache["past_kv"].layers[i].keys[:, :, :min_seq, :]
            zk = zh_cache["past_kv"].layers[i].keys[:, :, :min_seq, :]
            ev = en_cache["past_kv"].layers[i].values[:, :, :min_seq, :]
            zv = zh_cache["past_kv"].layers[i].values[:, :, :min_seq, :]
            mean_kv.layers[i].keys = ((ek + zk) / 2.0).clone()
            mean_kv.layers[i].values = ((ev + zv) / 2.0).clone()
        mean_cache_obj = {"past_kv": mean_kv, "seq_len": min_seq}

        # Generate from each: EN-only, ZH-only, Mean
        for label, src_cache in [("en_only", en_cache), ("zh_only", zh_cache),
                                  ("mean_en_zh", mean_cache_obj)]:
            hybrid = swap_kv(desc["past_kv"], src_cache["past_kv"],
                             desc["seq_len"], src_cache["seq_len"])
            out = manual_generate(model, tokenizer, hybrid, desc["logits"])
            lang = detect_lang(out)
            has_answer = problem["answer"] in out
            prob_results[label] = {
                "output": out, "lang": lang, "has_answer": has_answer,
            }
            print(f"  {label}: lang={lang} ans={has_answer} | {out[:120]}", flush=True)

        results[f"problem_{pi}"] = prob_results

    return results


# ========================================================================
# PART 3: REVERSE INJECTION
# ========================================================================

def run_reverse_injection(model, tokenizer, device, caches, desc_cache):
    """Inject describe KV into math prompt. Does model describe instead of solve?"""
    print(f"\n{'#' * 70}", flush=True)
    print(f"  PART 3: REVERSE INJECTION", flush=True)
    print(f"{'#' * 70}", flush=True)

    results = {}

    for pi, problem in enumerate(PROBLEMS):
        print(f"\n--- Problem {pi}: {problem['en'][:50]}... ---", flush=True)
        prob_results = {}

        en_math = caches[f"p{pi}_en"]
        zh_math = caches[f"p{pi}_zh"]
        desc_en = desc_cache["en"]
        desc_zh = desc_cache["zh"]

        # Control: normal math generation (no injection)
        ctrl_out = manual_generate(model, tokenizer, en_math["past_kv"], en_math["logits"])
        prob_results["control_en_math"] = {"output": ctrl_out}
        print(f"  CTRL[en_math]: {ctrl_out[:120]}", flush=True)

        # Reverse: inject EN describe KV into EN math prompt
        hybrid = swap_kv(en_math["past_kv"], desc_en["past_kv"],
                         en_math["seq_len"], desc_en["seq_len"])
        out = manual_generate(model, tokenizer, hybrid, en_math["logits"])
        prob_results["en_desc_kv_into_en_math"] = {"output": out, "lang": detect_lang(out)}
        print(f"  REV[en_desc→en_math]: {out[:120]}", flush=True)

        # Reverse: inject ZH describe KV into EN math prompt
        hybrid = swap_kv(en_math["past_kv"], desc_zh["past_kv"],
                         en_math["seq_len"], desc_zh["seq_len"])
        out = manual_generate(model, tokenizer, hybrid, en_math["logits"])
        prob_results["zh_desc_kv_into_en_math"] = {"output": out, "lang": detect_lang(out)}
        print(f"  REV[zh_desc→en_math]: {out[:120]}", flush=True)

        # Reverse: inject EN describe KV into ZH math prompt
        hybrid = swap_kv(zh_math["past_kv"], desc_en["past_kv"],
                         zh_math["seq_len"], desc_en["seq_len"])
        out = manual_generate(model, tokenizer, hybrid, zh_math["logits"])
        prob_results["en_desc_kv_into_zh_math"] = {"output": out, "lang": detect_lang(out)}
        print(f"  REV[en_desc→zh_math]: {out[:120]}", flush=True)

        # Cross-check: inject EN math KV into EN describe (this is F2, our known-good)
        hybrid = swap_kv(desc_en["past_kv"], en_math["past_kv"],
                         desc_en["seq_len"], en_math["seq_len"])
        out = manual_generate(model, tokenizer, hybrid, desc_en["logits"])
        prob_results["f2_en_math_into_en_desc"] = {"output": out, "lang": detect_lang(out)}
        print(f"  F2[en_math→en_desc]: {out[:120]}", flush=True)

        results[f"problem_{pi}"] = prob_results

    return results


# ========================================================================
# MAIN
# ========================================================================

def main():
    device = "cuda"
    print(f"{'#' * 70}", flush=True)
    print(f"  EXP INJ3: Tuned Lens + Mean KV + Reverse Injection", flush=True)
    print(f"{'#' * 70}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(flush=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Pre-encode everything
    print("\nPre-encoding...", flush=True)
    caches = {}
    for pi, problem in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            key = f"p{pi}_{lang}"
            caches[key] = encode_with_cache(model, tokenizer, problem[lang], device)
            print(f"  {key}: {caches[key]['seq_len']} tokens", flush=True)

    desc_cache = {}
    for lang, text in [("en", DESCRIBE_EN), ("zh", DESCRIBE_ZH)]:
        desc_cache[lang] = encode_with_cache(model, tokenizer, text, device)
        print(f"  desc_{lang}: {desc_cache[lang]['seq_len']} tokens", flush=True)

    # Run all three parts
    results = {}

    t1 = time.time()
    results["tuned_lens"] = run_tuned_lens(model, tokenizer, device, caches)
    print(f"\n  Part 1 done in {time.time() - t1:.0f}s", flush=True)

    t2 = time.time()
    results["mean_kv"] = run_mean_kv(model, tokenizer, device, caches, desc_cache)
    print(f"\n  Part 2 done in {time.time() - t2:.0f}s", flush=True)

    t3 = time.time()
    results["reverse_injection"] = run_reverse_injection(model, tokenizer, device, caches, desc_cache)
    print(f"\n  Part 3 done in {time.time() - t3:.0f}s", flush=True)

    wall = time.time() - t0
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"  DONE — {wall:.0f}s ({wall/60:.1f}min)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / "expINJ3_trio.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "INJ3_trio",
            "model": MODEL_NAME,
            "parts": ["tuned_lens", "mean_kv", "reverse_injection"],
            "results": results,
            "wall_time_s": wall,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
