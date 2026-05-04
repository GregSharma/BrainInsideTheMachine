"""expH1: Residual Stream Injection — The Bun Inversion

Generic describe prompt (NO problem text). Describe KV at ALL layers.
Math hidden states injected into the residual stream during generation.

If the model describes math it never saw in text → h' ∘ f confirmed.

The token walks past describe-mode shelves, but the token IS the math
computation. The shelves say "describe what you are." The token answers.

Variants:
  A (inject_propagate): Pre-hook at layer L* replaces hidden state with
     h_math[L*]. Layer L* computes attention(math_Q, desc_K/V) + MLP.
     Result propagates through L*+1..35 with describe KV. L* sweep.
  B (additive_delta): At every layer, add alpha*(h_math[L]-h_desc[L])
     to current hidden state. Nudges describe computation toward math.
  C (noise_control): Same as A but with random vectors (norm-matched).
  D (cross_problem): Inject problem B's states, does model describe B?
  E (cross_lingual): Inject ZH math, describe in EN. Language crossover.
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

# GENERIC describe prompt — NO problem text
DESCRIBE_GENERIC = {
    "en": "Describe what mathematical reasoning is being performed:",
    "zh": "描述正在进行什么数学推理：",
}

PROBLEMS = [
    {
        "en": "Solve for x: 3x + 7 = 22",
        "zh": "求解x：3x + 7 = 22",
        "answer": "5", "category": "algebra",
        "indicators_en": ["equation", "variable", "x", "isolat", "subtract", "divide",
                          "linear", "inverse", "3x", "22", "15"],
        "indicators_zh": ["方程", "变量", "求解", "减", "除", "线性", "3x", "22"],
    },
    {
        "en": "What is the area of a triangle with base 10 and height 7?",
        "zh": "底边为10、高为7的三角形面积是多少？",
        "answer": "35", "category": "geometry",
        "indicators_en": ["triangle", "area", "base", "height", "formula",
                          "multiply", "divide by 2", "10", "7", "35"],
        "indicators_zh": ["三角", "面积", "底", "高", "公式", "10", "7", "35"],
    },
    {
        "en": "Find the GCD of 84 and 120",
        "zh": "求84和120的最大公约数",
        "answer": "12", "category": "number_theory",
        "indicators_en": ["gcd", "greatest common", "divisor", "euclidean",
                          "remainder", "84", "120", "12"],
        "indicators_zh": ["最大公约数", "辗转", "欧几里", "余数", "84", "120", "12"],
    },
]

L_STAR_SWEEP = [5, 10, 15, 18, 22, 27, 33]
ALPHA_SWEEP = [0.25, 0.5, 1.0, 2.0]


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
    # Repetition check — but use 8-char window to avoid false positives on common words
    if len(text) > 60:
        for i in range(0, min(len(text) - 8, 40)):
            chunk = text[i:i+8]
            if chunk.strip() and text.count(chunk) > 6:
                return True
    return False


def score_match(text, problem, lang="en"):
    """Score how much the output matches a specific problem."""
    text_lower = text.lower()
    key = f"indicators_{lang}" if f"indicators_{lang}" in problem else "indicators_en"
    indicators = problem.get(key, problem["indicators_en"])
    hits = sum(1 for ind in indicators if ind.lower() in text_lower)
    return {
        "hits": hits,
        "total": len(indicators),
        "frac": round(hits / max(len(indicators), 1), 3),
        "has_answer": problem["answer"] in text,
    }


def score_all_problems(text, problems, lang="en"):
    """Score output against ALL problems — identifies which problem (if any) is described."""
    scores = []
    for pi, p in enumerate(problems):
        s = score_match(text, p, lang)
        s["problem_idx"] = pi
        s["category"] = p["category"]
        scores.append(s)
    best = max(scores, key=lambda s: s["frac"])
    return scores, best


def encode_with_cache(model, tokenizer, text, device):
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    return {
        "past_kv": outputs.past_key_values,
        "hidden_states": outputs.hidden_states,  # [embed, L0_out, ..., L35_out]
        "logits": outputs.logits,
        "input_ids": inputs["input_ids"],
        "seq_len": inputs["input_ids"].shape[1],
    }


def generate_with_injection(model, tokenizer, desc_cache, math_hidden_states,
                            inject_layers, mode="replace", alpha=1.0,
                            inject_first_only=True):
    """Generate from describe prompt with math hidden states injected.

    Args:
        desc_cache: encoded describe prompt (KV + logits)
        math_hidden_states: list of (1, 1, d) tensors, one per layer input
                           math_hidden_states[L] = hidden_states[L][:, -1:, :]
        inject_layers: which layers to inject at
        mode: "replace" (set to math state) or "additive" (add delta)
        alpha: scaling for additive mode
        inject_first_only: only inject during first generated token
    """
    kv = copy.deepcopy(desc_cache["past_kv"])
    next_token = desc_cache["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    eos = tokenizer.eos_token_id

    step_counter = [0]  # mutable for closure

    def make_hook(layer_idx):
        target_h = math_hidden_states[layer_idx]  # (1, 1, d)

        def hook(module, args, kwargs):
            if inject_first_only and step_counter[0] > 0:
                return args, kwargs

            new_args = list(args)
            if mode == "replace":
                new_args[0] = target_h
            elif mode == "additive":
                new_args[0] = args[0] + alpha * target_h  # target_h is delta here
            return tuple(new_args), kwargs

        return hook

    # Register pre-hooks
    handles = []
    for L in inject_layers:
        h = model.model.layers[L].register_forward_pre_hook(make_hook(L), with_kwargs=True)
        handles.append(h)

    try:
        for step in range(MAX_GEN - 1):
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


def generate_baseline(model, tokenizer, text, device):
    """Standard generation without injection."""
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_GEN, do_sample=False)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


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
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # ================================================================
    # ENCODE ALL PROMPTS
    # ================================================================
    print("Encoding prompts...", flush=True)

    # Generic describe prompts (no problem text)
    desc_caches = {}
    for lang in ["en", "zh"]:
        desc_caches[lang] = encode_with_cache(
            model, tokenizer, DESCRIBE_GENERIC[lang], device
        )
        print(f"  Describe ({lang}): seq_len={desc_caches[lang]['seq_len']}")

    # Math problem encodings
    math_caches = {}
    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            cache = encode_with_cache(model, tokenizer, prob[lang], device)
            # Extract last-token hidden states at each layer input
            # hidden_states[L] = output of layer L-1 = input to layer L
            # hidden_states[0] = embedding output = input to layer 0
            h_last = []
            for L in range(N_LAYERS):
                h_last.append(cache["hidden_states"][L][:, -1:, :].clone())
            math_caches[f"p{pi}_{lang}"] = {
                "h_layers": h_last,  # h_layers[L] = input to layer L, last token
                "seq_len": cache["seq_len"],
                "norms": [h.norm().item() for h in h_last],
            }
            print(f"  Math p{pi}_{lang}: seq_len={cache['seq_len']}, "
                  f"norm_range=[{min(math_caches[f'p{pi}_{lang}']['norms']):.1f}, "
                  f"{max(math_caches[f'p{pi}_{lang}']['norms']):.1f}]")

    # Describe prompt hidden states (for additive delta)
    desc_h = {}
    for lang in ["en", "zh"]:
        cache = desc_caches[lang]
        h_last = []
        for L in range(N_LAYERS):
            h_last.append(cache["hidden_states"][L][:, -1:, :].clone())
        desc_h[lang] = h_last

    print(f"\nAll prompts encoded.\n")

    # ================================================================
    # BASELINE: What does the generic describe prompt produce on its own?
    # ================================================================
    print(f"{'='*70}")
    print(f"  BASELINE: Generic describe prompt (no injection)")
    print(f"{'='*70}\n")

    baselines = {}
    for lang in ["en", "zh"]:
        text = generate_baseline(model, tokenizer, DESCRIBE_GENERIC[lang], device)
        baselines[lang] = text
        print(f"  {lang}: {text[:200]}...")
        print()

    all_results = {"baselines": baselines}

    # ================================================================
    # VARIANT A: inject_propagate — L* sweep
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  VARIANT A: INJECT & PROPAGATE (L* sweep)")
    print(f"  Inject h_math[L*] at layer L* input, propagate through L*..35")
    print(f"{'='*70}\n")

    variant_a = []
    for L_star in L_STAR_SWEEP:
        print(f"  --- L* = {L_star} ---")
        for pi, prob in enumerate(PROBLEMS):
            math_h = math_caches[f"p{pi}_en"]["h_layers"]
            desc_cache = desc_caches["en"]

            output = generate_with_injection(
                model, tokenizer, desc_cache, math_h,
                inject_layers=[L_star], mode="replace",
                inject_first_only=True,
            )

            garbage = is_garbage(output)
            scores, best = score_all_problems(output, PROBLEMS, "en")

            result = {
                "L_star": L_star,
                "problem_idx": pi,
                "injected_problem": prob["category"],
                "math_lang": "en",
                "desc_lang": "en",
                "output": output[:500],
                "is_garbage": garbage,
                "scores": scores,
                "best_match": best,
                "identified_correct": best["problem_idx"] == pi and best["frac"] > 0.1,
            }
            variant_a.append(result)

            tag = f"p{pi}({prob['category']})"
            match_str = f"→ best={PROBLEMS[best['problem_idx']]['category']}({best['frac']:.0%})"
            status = "GARBAGE" if garbage else ("CORRECT" if result["identified_correct"] else "WRONG/VAGUE")
            print(f"    {tag} [{status}] {match_str}: {output[:100]}...")

        # Per-L* summary
        lr = [r for r in variant_a if r["L_star"] == L_star]
        correct = sum(1 for r in lr if r["identified_correct"])
        garbage = sum(1 for r in lr if r["is_garbage"])
        print(f"  L*={L_star}: correct={correct}/{len(lr)}, garbage={garbage}/{len(lr)}\n")

    all_results["variant_a_inject_propagate"] = variant_a

    # ================================================================
    # VARIANT B: additive_delta — alpha sweep
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  VARIANT B: ADDITIVE DELTA (alpha sweep)")
    print(f"  Add alpha*(h_math[L] - h_desc[L]) at all layers, first token")
    print(f"{'='*70}\n")

    variant_b = []
    for alpha in ALPHA_SWEEP:
        print(f"  --- alpha = {alpha} ---")
        for pi, prob in enumerate(PROBLEMS):
            math_h = math_caches[f"p{pi}_en"]["h_layers"]
            desc_cache = desc_caches["en"]
            d_h = desc_h["en"]

            # Compute deltas
            deltas = [math_h[L] - d_h[L] for L in range(N_LAYERS)]

            output = generate_with_injection(
                model, tokenizer, desc_cache, deltas,
                inject_layers=list(range(N_LAYERS)), mode="additive", alpha=alpha,
                inject_first_only=True,
            )

            garbage = is_garbage(output)
            scores, best = score_all_problems(output, PROBLEMS, "en")

            result = {
                "alpha": alpha,
                "problem_idx": pi,
                "injected_problem": prob["category"],
                "output": output[:500],
                "is_garbage": garbage,
                "scores": scores,
                "best_match": best,
                "identified_correct": best["problem_idx"] == pi and best["frac"] > 0.1,
            }
            variant_b.append(result)

            tag = f"p{pi}({prob['category']})"
            status = "GARBAGE" if garbage else ("CORRECT" if result["identified_correct"] else "WRONG/VAGUE")
            print(f"    {tag} [{status}]: {output[:100]}...")

        lr = [r for r in variant_b if r["alpha"] == alpha]
        correct = sum(1 for r in lr if r["identified_correct"])
        garbage = sum(1 for r in lr if r["is_garbage"])
        print(f"  alpha={alpha}: correct={correct}/{len(lr)}, garbage={garbage}/{len(lr)}\n")

    all_results["variant_b_additive_delta"] = variant_b

    # ================================================================
    # VARIANT C: noise_control — random vectors at L*=18
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  VARIANT C: NOISE CONTROL (random, norm-matched, L*=18)")
    print(f"{'='*70}\n")

    variant_c = []
    L_noise = 18
    for pi, prob in enumerate(PROBLEMS):
        math_norm = math_caches[f"p{pi}_en"]["norms"][L_noise]

        # Random direction, scaled to match math norm
        noise_h = torch.randn(1, 1, D_MODEL, dtype=torch.float16, device=device)
        noise_h = noise_h / noise_h.norm() * math_norm

        noise_layers = [None] * N_LAYERS
        noise_layers[L_noise] = noise_h

        desc_cache = desc_caches["en"]
        output = generate_with_injection(
            model, tokenizer, desc_cache, noise_layers,
            inject_layers=[L_noise], mode="replace",
            inject_first_only=True,
        )

        garbage = is_garbage(output)
        scores, best = score_all_problems(output, PROBLEMS, "en")

        result = {
            "L_star": L_noise,
            "problem_idx": pi,
            "injected": "noise",
            "noise_norm": round(math_norm, 2),
            "output": output[:500],
            "is_garbage": garbage,
            "scores": scores,
            "best_match": best,
        }
        variant_c.append(result)

        tag = f"p{pi}_noise"
        print(f"    {tag} [{'GARBAGE' if garbage else 'OK'}]: {output[:100]}...")

    all_results["variant_c_noise_control"] = variant_c
    garbage_c = sum(1 for r in variant_c if r["is_garbage"])
    print(f"  Noise: garbage={garbage_c}/{len(variant_c)}\n")

    # ================================================================
    # VARIANT D: cross_problem — inject problem B, does model describe B?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  VARIANT D: CROSS-PROBLEM INJECTION (inject B, describe B?)")
    print(f"{'='*70}\n")

    variant_d = []
    L_cross = 18
    for pi_inject in range(len(PROBLEMS)):
        for pi_target in range(len(PROBLEMS)):
            if pi_inject == pi_target:
                continue

            math_h = math_caches[f"p{pi_inject}_en"]["h_layers"]
            desc_cache = desc_caches["en"]

            output = generate_with_injection(
                model, tokenizer, desc_cache, math_h,
                inject_layers=[L_cross], mode="replace",
                inject_first_only=True,
            )

            garbage = is_garbage(output)
            scores, best = score_all_problems(output, PROBLEMS, "en")

            # Check: does it describe the INJECTED problem?
            inject_score = scores[pi_inject]

            result = {
                "L_star": L_cross,
                "injected_idx": pi_inject,
                "injected_problem": PROBLEMS[pi_inject]["category"],
                "output": output[:500],
                "is_garbage": garbage,
                "scores": scores,
                "best_match": best,
                "describes_injected": best["problem_idx"] == pi_inject and best["frac"] > 0.1,
                "inject_score_frac": inject_score["frac"],
            }
            variant_d.append(result)

            inj_cat = PROBLEMS[pi_inject]["category"]
            best_cat = PROBLEMS[best["problem_idx"]]["category"]
            hit = "HIT" if result["describes_injected"] else "MISS"
            print(f"    inject={inj_cat} [{hit}] → best={best_cat}({best['frac']:.0%}): {output[:100]}...")

    all_results["variant_d_cross_problem"] = variant_d
    hits_d = sum(1 for r in variant_d if r["describes_injected"])
    print(f"  Cross-problem: hits={hits_d}/{len(variant_d)}\n")

    # ================================================================
    # VARIANT E: cross_lingual — inject ZH math, describe in EN (and vice versa)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  VARIANT E: CROSS-LINGUAL INJECTION")
    print(f"{'='*70}\n")

    variant_e = []
    L_cross_l = 18
    for pi, prob in enumerate(PROBLEMS):
        for math_lang in ["en", "zh"]:
            for desc_lang in ["en", "zh"]:
                math_h = math_caches[f"p{pi}_{math_lang}"]["h_layers"]
                desc_cache = desc_caches[desc_lang]

                output = generate_with_injection(
                    model, tokenizer, desc_cache, math_h,
                    inject_layers=[L_cross_l], mode="replace",
                    inject_first_only=True,
                )

                garbage = is_garbage(output)
                scores_en, best_en = score_all_problems(output, PROBLEMS, "en")
                scores_zh, best_zh = score_all_problems(output, PROBLEMS, "zh")
                # Use whichever language scores higher
                best = best_en if best_en["frac"] >= best_zh["frac"] else best_zh

                result = {
                    "L_star": L_cross_l,
                    "problem_idx": pi,
                    "injected_problem": prob["category"],
                    "math_lang": math_lang,
                    "desc_lang": desc_lang,
                    "condition": f"{math_lang}_math→{desc_lang}_desc",
                    "output": output[:500],
                    "is_garbage": garbage,
                    "best_match": best,
                    "identified_correct": best["problem_idx"] == pi and best["frac"] > 0.1,
                }
                variant_e.append(result)

                tag = f"p{pi}_{math_lang}→{desc_lang}"
                status = "GARBAGE" if garbage else ("CORRECT" if result["identified_correct"] else "WRONG/VAGUE")
                print(f"    {tag} [{status}]: {output[:80]}...")

    all_results["variant_e_cross_lingual"] = variant_e
    for cond in ["en_math→en_desc", "en_math→zh_desc", "zh_math→en_desc", "zh_math→zh_desc"]:
        lr = [r for r in variant_e if r["condition"] == cond]
        correct = sum(1 for r in lr if r["identified_correct"])
        garbage = sum(1 for r in lr if r["is_garbage"])
        print(f"  {cond}: correct={correct}/{len(lr)}, garbage={garbage}/{len(lr)}")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0

    # Best L* from variant A
    a_by_l = {}
    for L in L_STAR_SWEEP:
        lr = [r for r in variant_a if r["L_star"] == L]
        a_by_l[L] = {
            "correct": sum(1 for r in lr if r["identified_correct"]),
            "garbage": sum(1 for r in lr if r["is_garbage"]),
            "n": len(lr),
        }

    # Best alpha from variant B
    b_by_a = {}
    for alpha in ALPHA_SWEEP:
        lr = [r for r in variant_b if r["alpha"] == alpha]
        b_by_a[alpha] = {
            "correct": sum(1 for r in lr if r["identified_correct"]),
            "garbage": sum(1 for r in lr if r["is_garbage"]),
            "n": len(lr),
        }

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "variant_a_by_lstar": {str(k): v for k, v in a_by_l.items()},
        "variant_b_by_alpha": {str(k): v for k, v in b_by_a.items()},
        "variant_c_noise_garbage": garbage_c,
        "variant_d_cross_problem_hits": hits_d,
        "variant_d_total": len(variant_d),
        "variant_e_correct_by_condition": {
            cond: sum(1 for r in variant_e if r["condition"] == cond and r["identified_correct"])
            for cond in ["en_math→en_desc", "en_math→zh_desc", "zh_math→en_desc", "zh_math→zh_desc"]
        },
    }
    all_results["summary"] = summary

    out_path = OUTPUT_DIR / "expH1_bun_inversion.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  H1 BUN INVERSION — COMPLETE in {elapsed:.1f}s")
    print(f"  Saved: {out_path}")
    print(f"{'='*70}\n")

    print(f"  VARIANT A (inject_propagate, L* sweep):")
    print(f"  {'L*':>4} | {'correct':>7} | {'garbage':>7}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}")
    for L in L_STAR_SWEEP:
        s = a_by_l[L]
        print(f"  {L:>4} | {s['correct']:>4}/{s['n']} | {s['garbage']:>4}/{s['n']}")

    print(f"\n  VARIANT B (additive, alpha sweep):")
    for alpha in ALPHA_SWEEP:
        s = b_by_a[alpha]
        print(f"    alpha={alpha}: correct={s['correct']}/{s['n']}, garbage={s['garbage']}/{s['n']}")

    print(f"\n  VARIANT C (noise): garbage={garbage_c}/{len(variant_c)}")
    print(f"  VARIANT D (cross-problem): hits={hits_d}/{len(variant_d)}")

    if hits_d > 0:
        print(f"\n  *** CROSS-PROBLEM IDENTIFICATION DETECTED ***")
        print(f"  The model described injected problems it never saw in text.")
        print(f"  This is h' ∘ f — the bun inversion.")


if __name__ == "__main__":
    main()
