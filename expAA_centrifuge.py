"""
Exp AA: The Centrifuge
======================
Hypothesis: The adversarial phase (L9-L17) is where language and math are maximally fused.
Each layer contributes a fresh language-biased innovation (R²=0.03, 97% novel).

If we ALTERNATE the language direction on every layer within L9-L17:
  L9: +scale (EN bias)
  L10: -scale (ZH bias)
  L11: +scale
  ...

Then by linearity of the residual stream:
  - Language components (same direction, alternating sign) → CANCEL
  - Math components (orthogonal to language direction) → ACCUMULATE undisturbed

The munchkin stays. The donut spins away.

Prediction A (concentrate): alternating flip > sustained flip > baseline
  The cancellation isolates pure math signal → best efficiency

Prediction B (collapse): alternating flip destroys coherence
  The oscillation disrupts the computation itself

Prediction C (neutral): alternating ≈ baseline
  Language and math are not separable even layer-by-layer

Controls:
  - Baseline (no intervention)
  - Sustained flip (all L9-L17 same direction, scale=-1.0) — our established result
  - Alternating flip (L9-L17, alternating ±1.0 per layer)
  - Random alternating (random signs each layer, not language direction) — specificity check

Metric: FAT (first appearance token) + accuracy at 128 tokens, N=20 problems
"""

import json, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
SCALE = 1.0
ADVERSARIAL_LAYERS = list(range(9, 18))  # L9-L17 inclusive

PROBLEMS = [
    {"prompt": "Calculate 664 + 124.", "answer": "788"},
    {"prompt": "Calculate 769 + 291.", "answer": "1060"},
    {"prompt": "Find the value of C(9, 5).", "answer": "126"},
    {"prompt": "Find the value of C(9, 4).", "answer": "126"},
    {"prompt": "What is the remainder when 1014 is divided by 17?", "answer": "11"},
    {"prompt": "What is the remainder when 1154 is divided by 5?", "answer": "4"},
    {"prompt": "A rectangle has length 34 and width 35. Find its area.", "answer": "1190"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60"},
    {"prompt": "An arithmetic sequence has first term 9 and common difference 5. Find sum of first 24 terms.", "answer": "1596"},
    {"prompt": "An arithmetic sequence has first term 7 and common difference 6. Find sum of first 11 terms.", "answer": "407"},
    {"prompt": "计算 664 + 124 的值。", "answer": "788"},
    {"prompt": "计算 769 + 291 的值。", "answer": "1060"},
    {"prompt": "求组合数 C(9, 5) 的值。", "answer": "126"},
    {"prompt": "求组合数 C(9, 4) 的值。", "answer": "126"},
    {"prompt": "1014 除以 17 的余数是多少？", "answer": "11"},
    {"prompt": "1154 除以 5 的余数是多少？", "answer": "4"},
    {"prompt": "一个长方形的长为 34，宽为 35，求其面积。", "answer": "1190"},
    {"prompt": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60"},
    {"prompt": "等差数列首项为 9，公差为 5，求前 24 项之和。", "answer": "1596"},
    {"prompt": "等差数列首项为 7，公差为 6，求前 11 项之和。", "answer": "407"},
]


def get_lang_direction(model, tok, n_problems=80):
    """Fit zh/en mean-difference direction at each adversarial layer."""
    import random
    rng = random.Random(42)
    pairs = []
    cats = [
        ("Calculate {} + {}.", "计算 {} + {} 的值。", lambda: (rng.randint(10,999), rng.randint(10,999))),
        ("Find C({}, {}).", "求组合数 C({}, {}) 的值。", lambda: (rng.randint(5,15), rng.randint(1,4))),
        ("Remainder when {} divided by {}?", "{} 除以 {} 的余数？", lambda: (rng.randint(50,999), rng.randint(3,20))),
    ]
    for _ in range(n_problems // 3 + 1):
        for en_t, zh_t, gen in cats:
            a, b = gen()
            pairs.append((en_t.format(a, b), zh_t.format(a, b)))
    pairs = pairs[:n_problems]

    layer_dirs = {}
    hooks = []
    en_acts = {l: [] for l in ADVERSARIAL_LAYERS}
    zh_acts = {l: [] for l in ADVERSARIAL_LAYERS}

    def make_hook(layer_idx, store):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[layer_idx].append(h[:, -1, :].detach().float().cpu())
        return hook

    for l in ADVERSARIAL_LAYERS:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l, en_acts)))

    for en_p, _ in pairs:
        inp = tok(en_p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inp)

    for h in hooks: h.remove()
    hooks = []
    for l in ADVERSARIAL_LAYERS:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l, zh_acts)))

    for _, zh_p in pairs:
        inp = tok(zh_p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inp)

    for h in hooks: h.remove()

    for l in ADVERSARIAL_LAYERS:
        en_mean = torch.stack(en_acts[l]).mean(0).squeeze()
        zh_mean = torch.stack(zh_acts[l]).mean(0).squeeze()
        diff = zh_mean - en_mean
        layer_dirs[l] = (diff / diff.norm()).to(DEVICE).to(torch.bfloat16)

    return layer_dirs


def run_condition(model, tok, lang_dirs, condition, max_tokens=128):
    """Run all 20 problems under a given condition.
    condition: 'baseline' | 'sustained' | 'alternating' | 'random_alt'
    """
    results = []

    # Build per-layer scales
    if condition == "baseline":
        layer_scales = {l: 0.0 for l in ADVERSARIAL_LAYERS}
    elif condition == "sustained":
        layer_scales = {l: -SCALE for l in ADVERSARIAL_LAYERS}
    elif condition == "alternating":
        # Even index in adversarial list → +scale, odd → -scale
        layer_scales = {}
        for i, l in enumerate(ADVERSARIAL_LAYERS):
            layer_scales[l] = +SCALE if i % 2 == 0 else -SCALE
    elif condition == "random_alt":
        torch.manual_seed(42)
        layer_scales = {l: float(torch.sign(torch.randn(1)).item()) * SCALE
                        for l in ADVERSARIAL_LAYERS}

    # Hook: add scale * lang_dir to MLP output at each adversarial layer
    hooks = []
    def make_hook(layer_idx):
        s = layer_scales[layer_idx]
        d = lang_dirs[layer_idx]
        def hook(module, inp, out):
            if s == 0.0:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h + s * d.unsqueeze(0).unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    for l in ADVERSARIAL_LAYERS:
        hooks.append(model.model.layers[l].mlp.register_forward_hook(make_hook(l)))

    for prob in PROBLEMS:
        inp = tok(prob["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        answer = prob["answer"]

        # FAT: first token index where answer appears
        tokens = tok.convert_ids_to_tokens(out[0][inp["input_ids"].shape[1]:].tolist())
        gen_text = ""
        fat = None
        for ti, tok_id in enumerate(out[0][inp["input_ids"].shape[1]:].tolist()):
            gen_text += tok.decode([tok_id], skip_special_tokens=True)
            if answer in gen_text and fat is None:
                fat = ti + 1

        correct = answer in gen
        results.append({
            "prompt": prob["prompt"][:50],
            "answer": answer,
            "correct": correct,
            "fat": fat,
            "gen_preview": gen[:80],
        })

    for h in hooks: h.remove()
    acc = sum(r["correct"] for r in results)
    fats = [r["fat"] for r in results if r["fat"] is not None]
    median_fat = float(np.median(fats)) if fats else max_tokens
    return acc, median_fat, results


def main():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    print("Fitting language directions at adversarial layers L9-L17...")
    lang_dirs = get_lang_direction(model, tok)
    print(f"  Done. {len(lang_dirs)} layer directions fitted.")

    output = {
        "experiment": "AA: Centrifuge — Layer-by-Layer Alternating Language Flip",
        "adversarial_layers": ADVERSARIAL_LAYERS,
        "scale": SCALE,
        "n_problems": len(PROBLEMS),
        "conditions": {},
    }

    conditions = ["baseline", "sustained", "alternating", "random_alt"]
    t0 = time.time()

    for cond in conditions:
        print(f"\n--- {cond} ---")
        acc, med_fat, results = run_condition(model, tok, lang_dirs, cond)
        output["conditions"][cond] = {
            "accuracy": acc,
            "median_fat": med_fat,
            "n": len(PROBLEMS),
            "results": results,
        }
        print(f"  acc={acc}/{len(PROBLEMS)}  median_FAT={med_fat:.1f}")
        # Print per-problem snapshot
        for r in results[:4]:
            print(f"    {'✓' if r['correct'] else '✗'} [{r['answer']}] FAT={r['fat']} | {r['gen_preview'][:60]}")

    output["runtime_seconds"] = time.time() - t0

    # Summary comparison
    print(f"\n=== CENTRIFUGE SUMMARY ===")
    for c in conditions:
        d = output["conditions"][c]
        print(f"  {c:15s}: acc={d['accuracy']:2d}/20  median_FAT={d['median_fat']:6.1f}")

    with open("output/expAA_centrifuge.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → output/expAA_centrifuge.json")
    print(f"Runtime: {output['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
