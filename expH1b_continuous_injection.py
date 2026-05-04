"""expH1b: Continuous Residual Stream Injection — The Real Bun Inversion

H1 showed single-layer, first-token injection is too weak. The describe
KV overwhelms the math signal in one layer of attention.

H1b injects the math hidden state at EVERY layer during EVERY generation
step. Crucially: the KV cache accumulates math-derived K/V entries over
generation steps. At layer L, each step adds K=W_K*h_math[L] and
V=W_V*h_math[L] to the cache. After N steps, attention sees:
  ~25 describe entries + N math entries (identical, so effectively 1 weighted entry)

Prediction: the output transitions from describe-mode (early) to math-mode
(late) as math K/V entries accumulate. Phase transition around N=25 tokens
when math entries equal describe entries.

Variants:
  A (all_layers_every): Inject h_math at ALL 36 layers, every step.
  B (upper_free): Inject h_math at layers 0..L*, layers L*+1..35 free.
     The upper layers see: injected residual from L* + mixed KV cache.
  C (noise_all): Random vectors at all layers, every step (control).
  D (cross_problem): Inject problem B's states continuously.
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
MAX_GEN = 300  # Longer generation to see phase transition

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


def continuous_generate(model, tokenizer, desc_cache, inject_h_layers,
                        inject_layer_set, max_gen=MAX_GEN):
    """Generate with continuous injection at specified layers.

    inject_h_layers[L]: (1, 1, d) tensor to inject at layer L's input.
    inject_layer_set: set of layer indices to inject at.

    Injects at EVERY generation step (not just first).
    Returns: full text + per-token text for phase analysis.
    """
    kv = copy.deepcopy(desc_cache["past_kv"])
    next_token = desc_cache["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = [next_token.item()]
    eos = tokenizer.eos_token_id

    # Register hooks — active on EVERY step
    def make_hook(L):
        target = inject_h_layers[L]

        def hook(module, args, kwargs):
            new_args = list(args)
            new_args[0] = target
            return tuple(new_args), kwargs

        return hook

    handles = []
    for L in inject_layer_set:
        h = model.model.layers[L].register_forward_pre_hook(make_hook(L), with_kwargs=True)
        handles.append(h)

    try:
        for step in range(max_gen - 1):
            with torch.no_grad():
                out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tid = next_token.item()
            if tid == eos:
                break
            generated_ids.append(tid)
    finally:
        for h in handles:
            h.remove()

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Decode in windows for phase analysis
    windows = {}
    checkpoints = [25, 50, 75, 100, 150, 200, 250]
    for cp in checkpoints:
        if cp <= len(generated_ids):
            windows[str(cp)] = tokenizer.decode(generated_ids[:cp], skip_special_tokens=True)

    return full_text, windows, len(generated_ids)


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

    # Encode prompts
    print("Encoding prompts...", flush=True)

    desc_caches = {}
    for lang in ["en", "zh"]:
        desc_caches[lang] = encode_with_cache(
            model, tokenizer, DESCRIBE_GENERIC[lang], device
        )
        print(f"  Describe ({lang}): seq_len={desc_caches[lang]['seq_len']}")

    math_caches = {}
    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            cache = encode_with_cache(model, tokenizer, prob[lang], device)
            h_layers = []
            for L in range(N_LAYERS):
                h_layers.append(cache["hidden_states"][L][:, -1:, :].clone())
            math_caches[f"p{pi}_{lang}"] = {
                "h_layers": h_layers,
                "norms": [h.norm().item() for h in h_layers],
            }
            print(f"  Math p{pi}_{lang}: norm_range=[{min(math_caches[f'p{pi}_{lang}']['norms']):.1f}, "
                  f"{max(math_caches[f'p{pi}_{lang}']['norms']):.1f}]")

    print()

    all_results = {}

    # ==================================================================
    # VARIANT A: All-layer, every-token continuous injection
    # ==================================================================
    print(f"{'=' * 70}")
    print(f"  VARIANT A: ALL LAYERS, EVERY TOKEN")
    print(f"  (inject h_math at all 36 layers, every generation step)")
    print(f"{'=' * 70}\n")

    variant_a = []
    all_layer_set = set(range(N_LAYERS))

    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            math_h = math_caches[f"p{pi}_{lang}"]["h_layers"]
            desc_cache = desc_caches[lang]

            text, windows, n_tokens = continuous_generate(
                model, tokenizer, desc_cache, math_h,
                inject_layer_set=all_layer_set, max_gen=MAX_GEN,
            )

            garbage = is_garbage(text)
            scores = [score_match(text, p) for p in PROBLEMS]
            best_idx = max(range(len(scores)), key=lambda i: scores[i]["frac"])

            # Phase analysis: score each window
            window_scores = {}
            for cp, wtext in windows.items():
                ws = score_match(wtext, prob)
                window_scores[cp] = ws

            result = {
                "problem_idx": pi,
                "category": prob["category"],
                "math_lang": lang,
                "desc_lang": lang,
                "n_tokens": n_tokens,
                "output": text[:600],
                "is_garbage": garbage,
                "scores": scores,
                "best_match_idx": best_idx,
                "best_match_cat": PROBLEMS[best_idx]["category"],
                "best_frac": scores[best_idx]["frac"],
                "correct": best_idx == pi and scores[pi]["frac"] > 0.15,
                "window_scores": window_scores,
            }
            variant_a.append(result)

            tag = f"p{pi}_{lang}({prob['category']})"
            status = "GARBAGE" if garbage else ("CORRECT" if result["correct"] else "WRONG")
            print(f"  {tag} [{status}] {n_tokens}tok best={PROBLEMS[best_idx]['category']}({scores[best_idx]['frac']:.0%})")
            print(f"    First 100: {text[:100]}...")
            if len(text) > 200:
                print(f"    Last 100:  ...{text[-100:]}")
            # Phase transition?
            for cp in ["25", "50", "100", "200"]:
                if cp in window_scores:
                    ws = window_scores[cp]
                    print(f"    @{cp}tok: hits={ws['hits']}/{ws['total']} ans={ws['has_answer']}")
            print()

    all_results["variant_a_all_layers"] = variant_a
    correct_a = sum(1 for r in variant_a if r["correct"])
    garbage_a = sum(1 for r in variant_a if r["is_garbage"])
    print(f"  VARIANT A: correct={correct_a}/{len(variant_a)}, garbage={garbage_a}/{len(variant_a)}\n")

    # ==================================================================
    # VARIANT B: Lower-layer inject, upper layers free
    # Inject at 0..L*, layers L*+1..35 process freely with mixed KV
    # ==================================================================
    print(f"{'=' * 70}")
    print(f"  VARIANT B: LOWER-LAYER INJECT, UPPER FREE (L* sweep)")
    print(f"  (inject at 0..L*, upper layers process mixed KV + propagated state)")
    print(f"{'=' * 70}\n")

    variant_b = []
    L_STAR_SWEEP = [10, 15, 18, 22, 27]

    for L_star in L_STAR_SWEEP:
        print(f"  --- L* = {L_star} (inject 0..{L_star}, free {L_star + 1}..35) ---")
        layer_set = set(range(L_star + 1))

        for pi, prob in enumerate(PROBLEMS):
            math_h = math_caches[f"p{pi}_en"]["h_layers"]
            desc_cache = desc_caches["en"]

            text, windows, n_tokens = continuous_generate(
                model, tokenizer, desc_cache, math_h,
                inject_layer_set=layer_set, max_gen=MAX_GEN,
            )

            garbage = is_garbage(text)
            scores = [score_match(text, p) for p in PROBLEMS]
            best_idx = max(range(len(scores)), key=lambda i: scores[i]["frac"])

            result = {
                "L_star": L_star,
                "problem_idx": pi,
                "category": prob["category"],
                "n_tokens": n_tokens,
                "output": text[:600],
                "is_garbage": garbage,
                "scores": scores,
                "best_match_idx": best_idx,
                "best_frac": scores[best_idx]["frac"],
                "correct": best_idx == pi and scores[pi]["frac"] > 0.15,
            }
            variant_b.append(result)

            tag = f"p{pi}({prob['category']})"
            status = "GARBAGE" if garbage else ("CORRECT" if result["correct"] else "WRONG")
            print(f"    {tag} [{status}] {n_tokens}tok: {text[:120]}...")

        lr = [r for r in variant_b if r["L_star"] == L_star]
        correct = sum(1 for r in lr if r["correct"])
        garbage = sum(1 for r in lr if r["is_garbage"])
        print(f"  L*={L_star}: correct={correct}/{len(lr)}, garbage={garbage}/{len(lr)}\n")

    all_results["variant_b_upper_free"] = variant_b

    # ==================================================================
    # VARIANT C: Noise control (all layers, every token)
    # ==================================================================
    print(f"{'=' * 70}")
    print(f"  VARIANT C: NOISE CONTROL (all layers, every token)")
    print(f"{'=' * 70}\n")

    variant_c = []
    for pi, prob in enumerate(PROBLEMS):
        norms = math_caches[f"p{pi}_en"]["norms"]
        noise_h = []
        for L in range(N_LAYERS):
            n = torch.randn(1, 1, D_MODEL, dtype=torch.float16, device=device)
            n = n / n.norm() * norms[L]
            noise_h.append(n)

        text, windows, n_tokens = continuous_generate(
            model, tokenizer, desc_caches["en"], noise_h,
            inject_layer_set=all_layer_set, max_gen=MAX_GEN,
        )

        garbage = is_garbage(text)
        result = {
            "problem_idx": pi,
            "injected": "noise",
            "n_tokens": n_tokens,
            "output": text[:400],
            "is_garbage": garbage,
        }
        variant_c.append(result)
        print(f"  noise_p{pi} [{'GARBAGE' if garbage else 'OK'}] {n_tokens}tok: {text[:120]}...")

    all_results["variant_c_noise"] = variant_c
    print()

    # ==================================================================
    # VARIANT D: Cross-problem (all layers, every token)
    # ==================================================================
    print(f"{'=' * 70}")
    print(f"  VARIANT D: CROSS-PROBLEM (inject B, does model describe B?)")
    print(f"{'=' * 70}\n")

    variant_d = []
    for pi_inject in range(len(PROBLEMS)):
        math_h = math_caches[f"p{pi_inject}_en"]["h_layers"]
        desc_cache = desc_caches["en"]

        text, windows, n_tokens = continuous_generate(
            model, tokenizer, desc_cache, math_h,
            inject_layer_set=all_layer_set, max_gen=MAX_GEN,
        )

        garbage = is_garbage(text)
        scores = [score_match(text, p) for p in PROBLEMS]
        best_idx = max(range(len(scores)), key=lambda i: scores[i]["frac"])

        result = {
            "injected_idx": pi_inject,
            "injected_category": PROBLEMS[pi_inject]["category"],
            "n_tokens": n_tokens,
            "output": text[:600],
            "is_garbage": garbage,
            "scores": scores,
            "best_match_idx": best_idx,
            "best_match_cat": PROBLEMS[best_idx]["category"],
            "describes_injected": best_idx == pi_inject and scores[pi_inject]["frac"] > 0.15,
        }
        variant_d.append(result)

        inj = PROBLEMS[pi_inject]["category"]
        best = PROBLEMS[best_idx]["category"]
        hit = "HIT" if result["describes_injected"] else "MISS"
        print(f"  inject={inj} [{hit}] → best={best}({scores[best_idx]['frac']:.0%})")
        print(f"    {text[:150]}...")
        print()

    all_results["variant_d_cross_problem"] = variant_d
    hits_d = sum(1 for r in variant_d if r["describes_injected"])

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "variant_a_correct": correct_a,
        "variant_a_total": len(variant_a),
        "variant_a_garbage": garbage_a,
        "variant_b_by_lstar": {
            str(L): {
                "correct": sum(1 for r in variant_b if r["L_star"] == L and r["correct"]),
                "garbage": sum(1 for r in variant_b if r["L_star"] == L and r["is_garbage"]),
            } for L in L_STAR_SWEEP
        },
        "variant_c_garbage": sum(1 for r in variant_c if r["is_garbage"]),
        "variant_d_hits": hits_d,
    }
    all_results["summary"] = summary

    out_path = OUTPUT_DIR / "expH1b_continuous_injection.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  H1b CONTINUOUS INJECTION — COMPLETE in {elapsed:.1f}s")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 70}\n")

    print(f"  VARIANT A (all layers, every token): correct={correct_a}/{len(variant_a)}")
    print(f"  VARIANT B (lower inject, upper free):")
    for L in L_STAR_SWEEP:
        lr = [r for r in variant_b if r["L_star"] == L]
        c = sum(1 for r in lr if r["correct"])
        g = sum(1 for r in lr if r["is_garbage"])
        print(f"    L*={L}: correct={c}/{len(lr)}, garbage={g}/{len(lr)}")
    print(f"  VARIANT C (noise): garbage={sum(1 for r in variant_c if r['is_garbage'])}/{len(variant_c)}")
    print(f"  VARIANT D (cross-problem): hits={hits_d}/{len(variant_d)}")


if __name__ == "__main__":
    main()
