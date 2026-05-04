"""expH1c: Logits Bias + Burst Injection + Gen-Time Hidden States

Three approaches to the bun inversion that avoid the fixed-point
repetition problem from H1b:

A) LOGITS BIAS: At each gen step, add math encoding logits as constant
   bias to generation logits. Nudges token selection toward math-content
   tokens while describe KV keeps the mode. No hooks needed.

B) BURST INJECTION: All-layer pre-hooks for first N tokens only, then
   free generation. Seeds KV cache with math entries, avoids repetition
   from continuous injection.

C) GEN-TIME HIDDEN STATES: Capture hidden states from math problem's
   first GENERATION step (not encoding). These are in generation-time
   distribution, should be more compatible with describe generation.
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

DESCRIBE_GENERIC = {
    "en": "Describe what mathematical reasoning is being performed:",
    "zh": "描述正在进行什么数学推理：",
}

PROBLEMS = [
    {
        "en": "Solve for x: 3x + 7 = 22",
        "zh": "求解x：3x + 7 = 22",
        "answer": "5", "category": "algebra",
        "indicators": ["equation", "variable", "x =", "isolat", "subtract", "divide",
                        "linear", "inverse", "3x", "22", "15", "方程", "求解"],
    },
    {
        "en": "What is the area of a triangle with base 10 and height 7?",
        "zh": "底边为10、高为7的三角形面积是多少？",
        "answer": "35", "category": "geometry",
        "indicators": ["triangle", "area", "base", "height", "formula",
                        "multiply", "divide by 2", "half", "10", "35",
                        "三角", "面积", "底", "高"],
    },
    {
        "en": "Find the GCD of 84 and 120",
        "zh": "求84和120的最大公约数",
        "answer": "12", "category": "number_theory",
        "indicators": ["gcd", "greatest common", "divisor", "euclidean",
                        "remainder", "84", "120", "最大公约数", "辗转"],
    },
]


def build_chat_prompt(tokenizer, text):
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def is_garbage(text):
    if len(text.strip()) < 5:
        return True
    for emoji in ["🎓", "🕹", "🥗"]:
        if text.count(emoji) > 5:
            return True
    if "0000000" in text or text.count("玳瑁") > 3:
        return True
    if len(text) > 60:
        for i in range(0, min(len(text) - 8, 40)):
            chunk = text[i:i + 8]
            if chunk.strip() and text.count(chunk) > 6:
                return True
    return False


def score_match(text, problem):
    text_lower = text.lower()
    hits = sum(1 for ind in problem["indicators"] if ind.lower() in text_lower)
    return {
        "hits": hits,
        "total": len(problem["indicators"]),
        "frac": round(hits / max(len(problem["indicators"]), 1), 3),
        "has_answer": problem["answer"] in text,
    }


def score_all(text, problems):
    scores = [score_match(text, p) for p in problems]
    best_idx = max(range(len(scores)), key=lambda i: scores[i]["frac"])
    return scores, best_idx


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


def get_math_logit_bias(model, math_cache):
    """Extract logits from math encoding's last token hidden state."""
    # hidden_states[-1] is the output of the last layer
    h_final = math_cache["hidden_states"][-1][:, -1:, :]  # (1, 1, d)
    with torch.no_grad():
        normed = model.model.norm(h_final.float())
        logits = model.lm_head(normed.half())  # (1, 1, vocab)
    return logits.squeeze(0)  # (1, vocab)


def capture_gen_hidden_states(model, math_cache):
    """Capture hidden states from math problem's first generation step.

    Run one generation step on the math encoding's KV cache.
    Return per-layer hidden states for the generated token.
    """
    kv = copy.deepcopy(math_cache["past_kv"])
    first_token = math_cache["logits"][:, -1, :].argmax(dim=-1, keepdim=True)

    with torch.no_grad():
        out = model(
            input_ids=first_token,
            past_key_values=kv,
            use_cache=True,
            output_hidden_states=True,
        )

    # out.hidden_states[L] has shape (1, 1, d) for each layer
    gen_h = [out.hidden_states[L].clone() for L in range(N_LAYERS + 1)]
    return gen_h  # gen_h[0]=embedding, gen_h[L+1]=output of layer L


# ==================================================================
# VARIANT A: Logits Bias
# ==================================================================
def run_variant_a(model, tokenizer, desc_cache, math_bias, beta, max_gen=MAX_GEN):
    """Generate with describe KV + constant math logit bias at each step."""
    kv = copy.deepcopy(desc_cache["past_kv"])
    # First token from BLENDED logits
    blended = desc_cache["logits"][:, -1, :] + beta * math_bias
    next_token = blended.argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    eos = tokenizer.eos_token_id

    for _ in range(max_gen - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        # Blend logits: describe generation + math bias
        blended = out.logits[:, -1, :] + beta * math_bias
        next_token = blended.argmax(dim=-1, keepdim=True)
        tid = next_token.item()
        if tid == eos:
            break
        generated.append(tid)

    return tokenizer.decode(generated, skip_special_tokens=True)


# ==================================================================
# VARIANT B: Burst Injection
# ==================================================================
def run_variant_b(model, tokenizer, desc_cache, math_h_layers, N_burst, max_gen=MAX_GEN):
    """All-layer injection for first N tokens, then free generation."""
    kv = copy.deepcopy(desc_cache["past_kv"])
    next_token = desc_cache["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    eos = tokenizer.eos_token_id
    step_counter = [0]

    def make_hook(L):
        target = math_h_layers[L]

        def hook(module, args, kwargs):
            if step_counter[0] >= N_burst:
                return args, kwargs
            new_args = list(args)
            new_args[0] = target
            return tuple(new_args), kwargs

        return hook

    handles = []
    for L in range(N_LAYERS):
        h = model.model.layers[L].register_forward_pre_hook(make_hook(L), with_kwargs=True)
        handles.append(h)

    try:
        for _ in range(max_gen - 1):
            with torch.no_grad():
                out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            step_counter[0] += 1
            tid = next_token.item()
            if tid == eos:
                break
            generated.append(tid)
    finally:
        for h in handles:
            h.remove()

    return tokenizer.decode(generated, skip_special_tokens=True)


# ==================================================================
# VARIANT C: Gen-Time Hidden States
# ==================================================================
def run_variant_c(model, tokenizer, desc_cache, gen_h, L_star, max_gen=MAX_GEN):
    """Single-layer injection using generation-time hidden states."""
    kv = copy.deepcopy(desc_cache["past_kv"])
    next_token = desc_cache["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    eos = tokenizer.eos_token_id
    step_counter = [0]

    target = gen_h[L_star]  # gen_h[L] = input to layer L during gen step

    def hook(module, args, kwargs):
        if step_counter[0] > 0:
            return args, kwargs
        new_args = list(args)
        new_args[0] = target
        return tuple(new_args), kwargs

    handle = model.model.layers[L_star].register_forward_pre_hook(hook, with_kwargs=True)

    try:
        for _ in range(max_gen - 1):
            with torch.no_grad():
                out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            step_counter[0] += 1
            tid = next_token.item()
            if tid == eos:
                break
            generated.append(tid)
    finally:
        handle.remove()

    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Encode all prompts
    print("Encoding...", flush=True)

    desc_caches = {}
    for lang in ["en", "zh"]:
        desc_caches[lang] = encode_with_cache(
            model, tokenizer, DESCRIBE_GENERIC[lang], device
        )

    math_caches = {}
    math_biases = {}
    math_gen_h = {}
    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            key = f"p{pi}_{lang}"
            cache = encode_with_cache(model, tokenizer, prob[lang], device)
            # Per-layer hidden states (last token, input to each layer)
            h_layers = [cache["hidden_states"][L][:, -1:, :].clone() for L in range(N_LAYERS)]
            math_caches[key] = {"h_layers": h_layers, "cache": cache}
            # Logit bias
            math_biases[key] = get_math_logit_bias(model, cache)
            # Gen-time hidden states
            math_gen_h[key] = capture_gen_hidden_states(model, cache)

            # Show what the math encoding predicts
            top5 = math_biases[key].topk(5, dim=-1)
            tokens = [tokenizer.decode([t]) for t in top5.indices[0].tolist()]
            print(f"  {key}: top logit tokens = {tokens}")

    print()
    all_results = {}

    # ==================================================================
    # VARIANT A: Logits Bias — beta sweep
    # ==================================================================
    print(f"{'=' * 70}")
    print(f"  VARIANT A: LOGITS BIAS (beta sweep)")
    print(f"  describe KV + beta * math_encoding_logits at each step")
    print(f"{'=' * 70}\n")

    variant_a = []
    BETA_SWEEP = [0.1, 0.3, 0.5, 1.0, 2.0]

    for beta in BETA_SWEEP:
        print(f"  --- beta = {beta} ---")
        for pi, prob in enumerate(PROBLEMS):
            bias = math_biases[f"p{pi}_en"]
            desc_cache = desc_caches["en"]

            text = run_variant_a(model, tokenizer, desc_cache, bias, beta)
            garbage = is_garbage(text)
            scores, best_idx = score_all(text, PROBLEMS)

            result = {
                "beta": beta,
                "problem_idx": pi,
                "category": prob["category"],
                "output": text[:500],
                "is_garbage": garbage,
                "scores": scores,
                "best_idx": best_idx,
                "correct": best_idx == pi and scores[pi]["frac"] > 0.15,
            }
            variant_a.append(result)

            tag = f"p{pi}({prob['category']})"
            status = "GARBAGE" if garbage else ("CORRECT" if result["correct"] else "WRONG")
            print(f"    {tag} [{status}] → {PROBLEMS[best_idx]['category']}({scores[best_idx]['frac']:.0%}): {text[:100]}...")

        lr = [r for r in variant_a if r["beta"] == beta]
        c = sum(1 for r in lr if r["correct"])
        g = sum(1 for r in lr if r["is_garbage"])
        print(f"  beta={beta}: correct={c}/{len(lr)}, garbage={g}/{len(lr)}\n")

    all_results["variant_a_logits_bias"] = variant_a

    # ==================================================================
    # VARIANT B: Burst Injection — N_burst sweep
    # ==================================================================
    print(f"{'=' * 70}")
    print(f"  VARIANT B: BURST INJECTION (N_burst sweep)")
    print(f"  all-layer inject for first N tokens, then free")
    print(f"{'=' * 70}\n")

    variant_b = []
    N_BURST_SWEEP = [1, 3, 5, 10, 20]

    for N_burst in N_BURST_SWEEP:
        print(f"  --- N_burst = {N_burst} ---")
        for pi, prob in enumerate(PROBLEMS):
            math_h = math_caches[f"p{pi}_en"]["h_layers"]
            desc_cache = desc_caches["en"]

            text = run_variant_b(model, tokenizer, desc_cache, math_h, N_burst)
            garbage = is_garbage(text)
            scores, best_idx = score_all(text, PROBLEMS)

            result = {
                "N_burst": N_burst,
                "problem_idx": pi,
                "category": prob["category"],
                "output": text[:500],
                "is_garbage": garbage,
                "scores": scores,
                "best_idx": best_idx,
                "correct": best_idx == pi and scores[pi]["frac"] > 0.15,
            }
            variant_b.append(result)

            tag = f"p{pi}({prob['category']})"
            status = "GARBAGE" if garbage else ("CORRECT" if result["correct"] else "WRONG")
            print(f"    {tag} [{status}]: {text[:100]}...")

        lr = [r for r in variant_b if r["N_burst"] == N_burst]
        c = sum(1 for r in lr if r["correct"])
        g = sum(1 for r in lr if r["is_garbage"])
        print(f"  N={N_burst}: correct={c}/{len(lr)}, garbage={g}/{len(lr)}\n")

    all_results["variant_b_burst"] = variant_b

    # ==================================================================
    # VARIANT C: Gen-Time Hidden States — L* sweep
    # ==================================================================
    print(f"{'=' * 70}")
    print(f"  VARIANT C: GEN-TIME HIDDEN STATES (L* sweep)")
    print(f"  inject math gen-step hidden states (not encoding) at single layer")
    print(f"{'=' * 70}\n")

    variant_c = []
    L_SWEEP = [10, 15, 18, 22, 27, 33]

    for L_star in L_SWEEP:
        print(f"  --- L* = {L_star} ---")
        for pi, prob in enumerate(PROBLEMS):
            gen_h = math_gen_h[f"p{pi}_en"]
            desc_cache = desc_caches["en"]

            text = run_variant_c(model, tokenizer, desc_cache, gen_h, L_star)
            garbage = is_garbage(text)
            scores, best_idx = score_all(text, PROBLEMS)

            result = {
                "L_star": L_star,
                "problem_idx": pi,
                "category": prob["category"],
                "output": text[:500],
                "is_garbage": garbage,
                "scores": scores,
                "best_idx": best_idx,
                "correct": best_idx == pi and scores[pi]["frac"] > 0.15,
            }
            variant_c.append(result)

            tag = f"p{pi}({prob['category']})"
            status = "GARBAGE" if garbage else ("CORRECT" if result["correct"] else "WRONG")
            print(f"    {tag} [{status}]: {text[:80]}...")

        lr = [r for r in variant_c if r["L_star"] == L_star]
        c = sum(1 for r in lr if r["correct"])
        g = sum(1 for r in lr if r["is_garbage"])
        print(f"  L*={L_star}: correct={c}/{len(lr)}, garbage={g}/{len(lr)}\n")

    all_results["variant_c_gentime"] = variant_c

    # ==================================================================
    # VARIANT D: Cross-problem logits bias (the money test)
    # ==================================================================
    print(f"{'=' * 70}")
    print(f"  VARIANT D: CROSS-PROBLEM LOGITS BIAS")
    print(f"  inject problem B's logits, does output reference problem B?")
    print(f"{'=' * 70}\n")

    # Use best beta from variant A
    a_by_beta = {}
    for beta in BETA_SWEEP:
        lr = [r for r in variant_a if r["beta"] == beta]
        a_by_beta[beta] = sum(1 for r in lr if r["correct"])
    best_beta = max(BETA_SWEEP, key=lambda b: a_by_beta[b])
    if a_by_beta[best_beta] == 0:
        best_beta = 0.3  # Default if none correct
    print(f"  Using beta={best_beta}\n")

    variant_d = []
    for pi_inject in range(len(PROBLEMS)):
        bias = math_biases[f"p{pi_inject}_en"]
        desc_cache = desc_caches["en"]

        text = run_variant_a(model, tokenizer, desc_cache, bias, best_beta)
        garbage = is_garbage(text)
        scores, best_idx = score_all(text, PROBLEMS)

        result = {
            "beta": best_beta,
            "injected_idx": pi_inject,
            "injected_category": PROBLEMS[pi_inject]["category"],
            "output": text[:500],
            "is_garbage": garbage,
            "scores": scores,
            "best_idx": best_idx,
            "describes_injected": best_idx == pi_inject and scores[pi_inject]["frac"] > 0.15,
        }
        variant_d.append(result)

        inj = PROBLEMS[pi_inject]["category"]
        best = PROBLEMS[best_idx]["category"]
        hit = "HIT" if result["describes_injected"] else "MISS"
        print(f"  inject={inj} [{hit}] → best={best}({scores[best_idx]['frac']:.0%})")
        print(f"    {text[:150]}...")
        print()

    all_results["variant_d_cross_problem_bias"] = variant_d

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0

    summary = {
        "elapsed_s": round(elapsed, 1),
        "variant_a_best_beta": best_beta,
        "variant_a_by_beta": {
            str(b): {
                "correct": sum(1 for r in variant_a if r["beta"] == b and r["correct"]),
                "garbage": sum(1 for r in variant_a if r["beta"] == b and r["is_garbage"]),
            } for b in BETA_SWEEP
        },
        "variant_b_by_n": {
            str(n): {
                "correct": sum(1 for r in variant_b if r["N_burst"] == n and r["correct"]),
                "garbage": sum(1 for r in variant_b if r["N_burst"] == n and r["is_garbage"]),
            } for n in N_BURST_SWEEP
        },
        "variant_c_by_l": {
            str(L): {
                "correct": sum(1 for r in variant_c if r["L_star"] == L and r["correct"]),
                "garbage": sum(1 for r in variant_c if r["L_star"] == L and r["is_garbage"]),
            } for L in L_SWEEP
        },
        "variant_d_hits": sum(1 for r in variant_d if r["describes_injected"]),
    }
    all_results["summary"] = summary

    out_path = OUTPUT_DIR / "expH1c_logits_bias.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  H1c — COMPLETE in {elapsed:.1f}s")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 70}\n")

    # Summary table
    print("  VARIANT A (logits bias):")
    for b in BETA_SWEEP:
        s = summary["variant_a_by_beta"][str(b)]
        print(f"    beta={b}: correct={s['correct']}/3, garbage={s['garbage']}/3")

    print("\n  VARIANT B (burst inject):")
    for n in N_BURST_SWEEP:
        s = summary["variant_b_by_n"][str(n)]
        print(f"    N={n}: correct={s['correct']}/3, garbage={s['garbage']}/3")

    print("\n  VARIANT C (gen-time h-states):")
    for L in L_SWEEP:
        s = summary["variant_c_by_l"][str(L)]
        print(f"    L*={L}: correct={s['correct']}/3, garbage={s['garbage']}/3")

    print(f"\n  VARIANT D (cross-problem): hits={summary['variant_d_hits']}/3")


if __name__ == "__main__":
    main()
