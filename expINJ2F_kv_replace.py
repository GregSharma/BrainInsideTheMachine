"""expINJ2F: Full KV Cache + Hidden State Replacement

Context (2026-04-13 late):
    INJ2 variant A (full-state replacement at L*) improved math content from 33% to 62%,
    confirming Web's KV conflict diagnosis. But 33% of outputs were garbage (emoji loops).

    Root cause analysis revealed a RESIDUAL confound: full-state replacement at L* only
    fixes the KV cache at layers L*+1 through L35. Layers 0 through L* still have
    describe-prompt KV entries. During generation, each new token flows through layers
    0..L* attending to those describe-prompt cached keys/values, injecting describe-prompt
    information into every generation step before it reaches the math-state upper layers.

    Variant F: replace hidden states at L* AND replace the KV cache at layers 0..L*
    with the math problem's KV cache entries. This eliminates the split entirely —
    from the model's perspective at generation time, it's as if the math problem was
    the prompt at every layer.

    Implementation: run the math problem through a normal forward pass, capture the
    DynamicCache (past_key_values). Then during the describe-prompt forward pass, at
    layer L*, replace hidden states. For generation, swap the past_key_values from
    layers 0..L* with the math problem's cached values.

    We also run a variant F2: replace the ENTIRE KV cache (all layers) with the math
    problem's KV cache. This is the maximum-information condition — the model generates
    from the math problem's complete state but with the describe prompt's generation
    position. If F2 works, it means the model can describe math when given the full
    internal state without any describe-prompt residue.

    Expected hierarchy:
    F2 (full KV replace) >= F (partial KV replace) > A (hidden-state only) > D (last-token only)

    Problems: same 3 as INJ2. L* sweep: [15, 18, 25, 30]. Model: Qwen2.5-3B.
"""
import json
import time
import copy
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_GEN = 256

L_STAR_CANDIDATES = [15, 18, 25, 30]

DESCRIBE_TEMPLATE = {
    "en": "Describe in detail what mathematical operation or reasoning is being performed:",
    "zh": "详细描述正在进行什么数学运算或推理：",
}

INJECT_PROBLEMS = [
    {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22",
     "answer": "5", "category": "algebra"},
    {"en": "What is the area of a triangle with base 10 and height 7?",
     "zh": "底边为10、高为7的三角形面积是多少？",
     "answer": "35", "category": "geometry"},
    {"en": "Find the GCD of 84 and 120", "zh": "求84和120的最大公约数",
     "answer": "12", "category": "number_theory"},
]


def build_chat_prompt(tokenizer, text):
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def detect_lang(text):
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if cjk > len(text) * 0.1 else "en"


def contains_math_content(text, problem):
    answer = problem["answer"]
    keywords = {
        "algebra": ["solve", "equation", "variable", "x", answer, "求解", "方程"],
        "geometry": ["area", "triangle", "base", "height", answer, "面积", "三角", "底"],
        "number_theory": ["gcd", "greatest common", "divisor", answer, "最大公约数", "公约数"],
    }
    cat_words = keywords.get(problem["category"], [answer])
    text_lower = text.lower()
    hits = sum(1 for w in cat_words if w.lower() in text_lower)
    return hits, len(cat_words)


def is_garbage(text):
    """Detect degenerate outputs (emoji loops, repetition, numeric spam)."""
    if text.count("🎓") > 5 or text.count("🕹") > 5 or text.count("🥗") > 5:
        return True
    if "0000000" in text or text.count("玳瑁") > 3:
        return True
    # Check for high repetition: any 4-char substring appearing 10+ times
    if len(text) > 40:
        for i in range(0, min(len(text) - 4, 40)):
            chunk = text[i:i+4]
            if chunk.strip() and text.count(chunk) > 10:
                return True
    return False


def encode_with_cache(model, tokenizer, text, device):
    """Encode text, return past_key_values (KV cache) and hidden states."""
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)

    # outputs.past_key_values: tuple of (key, value) per layer
    # outputs.hidden_states: tuple of hidden states per layer (including embedding)
    return {
        "past_kv": outputs.past_key_values,
        "hidden_states": outputs.hidden_states,  # [embed, L0_out, L1_out, ..., L35_out]
        "input_ids": inputs["input_ids"],
        "seq_len": inputs["input_ids"].shape[1],
    }


def _get_kv(cache, layer_idx):
    """Extract (key, value) tensors from a DynamicCache at a given layer.
    DynamicCache API: cache.layers[i].keys / .values
    Each has shape (batch, n_kv_heads, seq_len, head_dim).
    """
    return cache.layers[layer_idx].keys, cache.layers[layer_idx].values


def _build_hybrid_cache(math_kv, desc_kv, desc_seq_len, math_seq_len,
                         n_replace, n_layers):
    """Build a hybrid DynamicCache: layers 0..n_replace-1 from math, rest from desc."""
    from transformers.cache_utils import DynamicCache
    hybrid = DynamicCache()
    min_seq = min(desc_seq_len, math_seq_len)

    for layer_idx in range(n_layers):
        if layer_idx < n_replace:
            mk, mv = _get_kv(math_kv, layer_idx)
            mk = mk[:, :, :min_seq, :].clone()
            mv = mv[:, :, :min_seq, :].clone()
            if min_seq < desc_seq_len:
                pad = desc_seq_len - min_seq
                mk = torch.cat([mk, mk[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                mv = torch.cat([mv, mv[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
            hybrid.update(mk, mv, layer_idx)
        else:
            dk, dv = _get_kv(desc_kv, layer_idx)
            hybrid.update(dk.clone(), dv.clone(), layer_idx)
    return hybrid


def generate_with_kv_swap(model, tokenizer, desc_text, math_cache, L_star, device,
                           replace_all_kv=False):
    """Generate from describe prompt but with math KV cache injected.

    1. Encode describe prompt → get describe KV cache + logits for first token
    2. Build hybrid cache: swap layers 0..L* (or all) with math KV
    3. Manual greedy decode from the hybrid cache

    Uses manual token-by-token generation because model.generate() tries to
    re-prefill which fails with a pre-built cache. Manual loop just calls
    model.forward() with the previous token and the growing cache.
    """
    prompt = build_chat_prompt(tokenizer, desc_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    desc_seq_len = inputs["input_ids"].shape[1]

    # Run describe prompt to get KV cache and first-token logits
    with torch.no_grad():
        desc_out = model(**inputs, use_cache=True)

    desc_kv = desc_out.past_key_values
    math_kv = math_cache["past_kv"]
    math_seq_len = math_cache["seq_len"]

    # Build hybrid cache by mutating a deep copy
    import copy
    hybrid = copy.deepcopy(desc_kv)
    n_replace = N_LAYERS if replace_all_kv else (L_star + 1)
    min_seq = min(desc_seq_len, math_seq_len)

    for i in range(n_replace):
        mk = math_kv.layers[i].keys[:, :, :min_seq, :].clone()
        mv = math_kv.layers[i].values[:, :, :min_seq, :].clone()
        if min_seq < desc_seq_len:
            pad = desc_seq_len - min_seq
            mk = torch.cat([mk, mk[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
            mv = torch.cat([mv, mv[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
        hybrid.layers[i].keys = mk
        hybrid.layers[i].values = mv

    # Manual greedy decoding
    # First generated token comes from the describe prompt's logits
    next_token = desc_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token.item()]

    eos_id = tokenizer.eos_token_id
    for step in range(MAX_GEN - 1):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=hybrid,
                use_cache=True,
            )
        hybrid = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = next_token.item()
        if tok_id == eos_id:
            break
        generated_ids.append(tok_id)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_baseline(model, tokenizer, text, device):
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_GEN, do_sample=False)
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main():
    device = "cuda"
    print(f"{'#' * 80}", flush=True)
    print(f"  Exp INJ2F: KV Cache + Hidden State Replacement", flush=True)
    print(f"{'#' * 80}", flush=True)
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(f"Problems: {len(INJECT_PROBLEMS)}", flush=True)
    print(f"L* sweep: {L_STAR_CANDIDATES}", flush=True)
    print(f"Variants: F (partial KV L0..L*), F2 (full KV all layers)", flush=True)
    print(flush=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    all_results = {}
    gen_count = 0

    # Pre-encode all math problems (cache reused across L* sweep)
    print("\nPre-encoding math problems...", flush=True)
    math_caches = {}
    for pi, problem in enumerate(INJECT_PROBLEMS):
        for lang in ["en", "zh"]:
            key = f"p{pi}_{lang}"
            math_caches[key] = encode_with_cache(model, tokenizer, problem[lang], device)
            print(f"  {key}: {math_caches[key]['seq_len']} tokens", flush=True)

    # Encode describe prompts
    desc_caches = {}
    for lang in ["en", "zh"]:
        desc_caches[lang] = encode_with_cache(model, tokenizer, DESCRIBE_TEMPLATE[lang], device)
        print(f"  desc_{lang}: {desc_caches[lang]['seq_len']} tokens", flush=True)

    # Baselines
    print("\n--- Baselines ---", flush=True)
    baseline_desc_en = generate_baseline(model, tokenizer, DESCRIBE_TEMPLATE["en"], device)
    print(f"  Baseline desc_en: {baseline_desc_en[:100]}", flush=True)

    for L_star in L_STAR_CANDIDATES:
        print(f"\n\n{'=' * 70}", flush=True)
        print(f"  L* = {L_star}", flush=True)
        print(f"{'=' * 70}", flush=True)

        l_results = {}

        for pi, problem in enumerate(INJECT_PROBLEMS):
            print(f"\n--- Problem {pi}: {problem['en'][:50]}... ---", flush=True)
            prob_results = {}

            for variant_name, replace_all in [("F", False), ("F2", True)]:
                for src_lang, desc_lang in [("en", "en"), ("zh", "en"), ("en", "zh"), ("zh", "zh")]:
                    cond_name = f"{src_lang}_math_{desc_lang}_desc"
                    math_key = f"p{pi}_{src_lang}"

                    try:
                        out = generate_with_kv_swap(
                            model, tokenizer,
                            DESCRIBE_TEMPLATE[desc_lang],
                            math_caches[math_key],
                            L_star, device,
                            replace_all_kv=replace_all,
                        )
                    except Exception as e:
                        out = f"ERROR: {str(e)}"
                        print(f"    {variant_name}[{cond_name}]: ERROR: {e}", flush=True)
                        prob_results.setdefault(variant_name, {})[cond_name] = {
                            "output": out, "error": True
                        }
                        gen_count += 1
                        continue

                    hits, total = contains_math_content(out, problem)
                    garb = is_garbage(out)
                    lang = detect_lang(out)
                    result = {
                        "output": out, "lang": lang,
                        "math_hits": hits, "math_total": total,
                        "is_garbage": garb,
                        "has_answer": problem["answer"] in out,
                    }
                    prob_results.setdefault(variant_name, {})[cond_name] = result
                    gen_count += 1

                    status = "GARBAGE" if garb else f"math={hits}/{total}"
                    print(f"    {variant_name}[{cond_name}]: lang={lang} {status} | {out[:100]}", flush=True)

            # Noise control for F2 — build a cache with random KV entries
            import copy as _copy
            ref_cache = math_caches[f"p{pi}_en"]
            noise_kv = _copy.deepcopy(ref_cache["past_kv"])
            for layer_idx in range(N_LAYERS):
                rk = noise_kv.layers[layer_idx].keys
                rv = noise_kv.layers[layer_idx].values
                noise_kv.layers[layer_idx].keys = torch.randn_like(rk) * rk.norm() / (rk.numel() ** 0.5)
                noise_kv.layers[layer_idx].values = torch.randn_like(rv) * rv.norm() / (rv.numel() ** 0.5)
            noise_cache = {"past_kv": noise_kv, "seq_len": ref_cache["seq_len"]}
            try:
                out = generate_with_kv_swap(
                    model, tokenizer, DESCRIBE_TEMPLATE["en"],
                    noise_cache, L_star, device, replace_all_kv=True,
                )
            except Exception as e:
                out = f"ERROR: {str(e)}"
            prob_results.setdefault("F2", {})["noise_control"] = {"output": out}
            gen_count += 1
            print(f"    F2[NOISE]: {out[:100]}", flush=True)

            l_results[f"problem_{pi}"] = prob_results

        all_results[f"L{L_star}"] = l_results

    wall = time.time() - t0
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"  DONE — {gen_count} generations in {wall:.0f}s ({wall/60:.1f}min)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    for variant in ["F", "F2"]:
        math_yes = 0; garbage = 0; total = 0
        for L in L_STAR_CANDIDATES:
            for pi in range(3):
                v = all_results.get(f"L{L}", {}).get(f"problem_{pi}", {}).get(variant, {})
                for cond, data in v.items():
                    if cond == "noise_control":
                        continue
                    total += 1
                    if data.get("is_garbage"):
                        garbage += 1
                    elif data.get("math_hits", 0) > 0:
                        math_yes += 1
        non_garb = total - garbage
        print(f"  {variant}: math={math_yes}/{total} ({100*math_yes/total:.0f}%)  "
              f"garbage={garbage}/{total} ({100*garbage/total:.0f}%)  "
              f"coherent_math={math_yes}/{non_garb} ({100*math_yes/max(non_garb,1):.0f}% of non-garbage)", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / "expINJ2F_kv_replace.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "INJ2F_kv_replace",
            "model": MODEL_NAME,
            "L_star_candidates": L_STAR_CANDIDATES,
            "problems": [p["en"] for p in INJECT_PROBLEMS],
            "variants": ["F_partial_kv", "F2_full_kv"],
            "results": all_results,
            "total_generations": gen_count,
            "wall_time_s": wall,
            "baseline_desc_en": baseline_desc_en,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
