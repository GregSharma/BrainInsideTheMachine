"""
Exp BQ3: Lyapunov-Guided Layer Pruning — Causal Validation

The causal test: can the Gram matrix dynamics (BQ/BQ2) predict which
layers are redundant for math generation?

Rankings from BQ/BQ2:
  ΔG bottom-5 (smallest Frobenius perturbation): layers 21, 24, 5, 25, 19
  Lyapunov bottom-5 (smallest spectral redistribution): layers 24, 19, 21, 5, 17
  Overlap: {5, 19, 21, 24} — 4/5 agree (all cooperative phase)

  ΔG top-5 (destructive): layers 1, 7, 12, 16, 35
  Lyapunov top-5 (destructive): layers 1, 2, 3, 16, 35

Conditions:
  1. Baseline (no skip)
  2. ΔG-guided skip at k=3,5,8
  3. Lyapunov-guided skip at k=3,5,8
  4. Random skip at k=3,5,8 (3 seeds each)
  5. Destructive: ΔG top-5, Lyapunov top-5

Skip mechanism: forward hook on model.model.layers[L] → identity.
"""

import torch
import json
import time
import math
import re
import numpy as np
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUT_PATH = "output/expBQ3_lyapunov_pruning.json"
MAX_NEW_TOKENS = 128
SEED = 42

# Layer rankings from BQ/BQ2 (0-indexed transformer layers)
DG_RANKING = [21, 24, 5, 25, 19, 22, 14, 13, 17, 10]
LYAP_RANKING = [24, 19, 21, 5, 17, 10, 15, 29, 13, 11]
DG_TOP5 = [1, 7, 12, 16, 35]
LYAP_TOP5 = [1, 2, 3, 16, 35]

RANDOM_SEEDS = [42, 123, 777]
SKIP_COUNTS = [3, 5, 8]
RANDOM_POOL = list(range(1, 35))

# ── Problem generation (identical to BO) ──

TEMPLATES = {
    'zh': {
        'arithmetic_plus': "计算 {a} + {b} 的值。",
        'arithmetic_times': "计算 {a} × {b} 的值。",
        'combinatorics': "求组合数 C({n}, {k}) 的值。",
        'modular': "{a} 除以 {b} 的余数是多少？",
        'geometry': "一个长方形的长为 {w}，宽为 {h}，求其面积。",
    },
    'en': {
        'arithmetic_plus': "Calculate {a} + {b}.",
        'arithmetic_times': "Calculate {a} × {b}.",
        'combinatorics': "Find the value of C({n}, {k}).",
        'modular': "What is the remainder when {a} is divided by {b}?",
        'geometry': "A rectangle has length {w} and width {h}. Find its area.",
    },
}


def generate_test_problems(n_test=4):
    rng = pyrandom.Random(SEED)
    cats = []

    per_cat = 200 // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        ans = a + b if op == "plus" else a * b
        prompts = {}
        for lang in ['en', 'zh']:
            prompts[lang] = TEMPLATES[lang][f'arithmetic_{op}'].format(a=a, b=b)
        cats.append(('arithmetic', ans, prompts))

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        ans = math.comb(n_val, k_val)
        prompts = {}
        for lang in ['en', 'zh']:
            prompts[lang] = TEMPLATES[lang]['combinatorics'].format(n=n_val, k=k_val)
        cats.append(('combinatorics', ans, prompts))

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        ans = a % b
        prompts = {}
        for lang in ['en', 'zh']:
            prompts[lang] = TEMPLATES[lang]['modular'].format(a=a, b=b)
        cats.append(('modular', ans, prompts))

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        ans = w * h
        prompts = {}
        for lang in ['en', 'zh']:
            prompts[lang] = TEMPLATES[lang]['geometry'].format(w=w, h=h)
        cats.append(('geometry', ans, prompts))

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        cats.append(('sequences', ans, {
            'en': f"An arithmetic sequence: first term {a1}, common difference {d}. Sum of first {n_terms} terms?",
            'zh': f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
        }))

    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats)))
    rng2.shuffle(indices)
    cats = [cats[i] for i in indices]

    by_cat = {}
    for cat, ans, prompts in cats:
        if cat not in by_cat:
            by_cat[cat] = []
        if len(by_cat[cat]) < n_test:
            by_cat[cat].append((ans, prompts))

    test_set = []
    for cat in by_cat:
        for ans, prompts in by_cat[cat]:
            test_set.append({'category': cat, 'answer': ans, 'en': prompts['en'], 'zh': prompts['zh']})
    return test_set


def check_answer(text, correct_answer):
    target = str(correct_answer)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return target in numbers


# ── Layer skip hooks ──

class LayerSkipHook:
    def __init__(self):
        self.handles = []

    def skip_layers(self, model, layer_indices):
        self.remove()
        for idx in layer_indices:
            if 0 <= idx < len(model.model.layers):
                handle = model.model.layers[idx].register_forward_hook(self._identity_hook)
                self.handles.append(handle)

    @staticmethod
    def _identity_hook(module, input, output):
        # Transformer block output: (hidden_states, ...) tuple
        if isinstance(output, tuple):
            return (input[0],) + output[1:]
        return input[0]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ── Run one condition ──

def run_condition(model, tokenizer, problems, skip_layers, label, device):
    hook = LayerSkipHook()
    if skip_layers:
        hook.skip_layers(model, skip_layers)

    results = {
        "label": label,
        "skip_layers": sorted(skip_layers) if skip_layers else [],
        "n_skipped": len(skip_layers) if skip_layers else 0,
    }

    for lang in ["en", "zh"]:
        correct = 0
        for p in problems:
            prompt = p[lang]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if check_answer(generated, p["answer"]):
                correct += 1
        results[f"{lang}_correct"] = correct
        results[f"{lang}_total"] = len(problems)

    hook.remove()
    return results


# ── Main ──

def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Model loaded: {n_layers} layers")

    problems = generate_test_problems(n_test=4)
    print(f"Test set: {len(problems)} problems")

    all_results = []
    step = 0
    total_steps = 1 + 3 + 3 + 9 + 2  # baseline + dg + lyap + random + destructive = 18

    # 1. Baseline
    step += 1
    print(f"\n[{step}/{total_steps}] Baseline...")
    r = run_condition(model, tokenizer, problems, [], "baseline", device)
    all_results.append(r)
    baseline_en, baseline_zh = r['en_correct'], r['zh_correct']
    print(f"  EN: {baseline_en}/{r['en_total']}, ZH: {baseline_zh}/{r['zh_total']}")

    # 2. ΔG-guided skip
    for k in SKIP_COUNTS:
        step += 1
        layers = DG_RANKING[:k]
        print(f"\n[{step}/{total_steps}] ΔG skip k={k}: {layers}...")
        r = run_condition(model, tokenizer, problems, layers, f"dg_skip_{k}", device)
        all_results.append(r)
        print(f"  EN: {r['en_correct']}/{r['en_total']}, ZH: {r['zh_correct']}/{r['zh_total']}")

    # 3. Lyapunov-guided skip
    for k in SKIP_COUNTS:
        step += 1
        layers = LYAP_RANKING[:k]
        print(f"\n[{step}/{total_steps}] Lyapunov skip k={k}: {layers}...")
        r = run_condition(model, tokenizer, problems, layers, f"lyap_skip_{k}", device)
        all_results.append(r)
        print(f"  EN: {r['en_correct']}/{r['en_total']}, ZH: {r['zh_correct']}/{r['zh_total']}")

    # 4. Random skip (3 seeds each k)
    for k in SKIP_COUNTS:
        for seed in RANDOM_SEEDS:
            step += 1
            rng = np.random.RandomState(seed)
            layers = sorted(rng.choice(RANDOM_POOL, size=k, replace=False).tolist())
            print(f"\n[{step}/{total_steps}] Random skip k={k} seed={seed}: {layers}...")
            r = run_condition(model, tokenizer, problems, layers, f"random_{k}_s{seed}", device)
            all_results.append(r)
            print(f"  EN: {r['en_correct']}/{r['en_total']}, ZH: {r['zh_correct']}/{r['zh_total']}")

    # 5. Destructive skip
    step += 1
    print(f"\n[{step}/{total_steps}] Destructive: ΔG top-5 = {DG_TOP5}...")
    r = run_condition(model, tokenizer, problems, DG_TOP5, "dg_destructive_5", device)
    all_results.append(r)
    print(f"  EN: {r['en_correct']}/{r['en_total']}, ZH: {r['zh_correct']}/{r['zh_total']}")

    step += 1
    print(f"\n[{step}/{total_steps}] Destructive: Lyapunov top-5 = {LYAP_TOP5}...")
    r = run_condition(model, tokenizer, problems, LYAP_TOP5, "lyap_destructive_5", device)
    all_results.append(r)
    print(f"  EN: {r['en_correct']}/{r['en_total']}, ZH: {r['zh_correct']}/{r['zh_total']}")

    elapsed = time.time() - t0

    # ── Compile and save ──
    output = {
        "experiment": "BQ3",
        "name": "Lyapunov-Guided Layer Pruning",
        "method": "Skip layers by ΔG Frobenius ranking (BQ) and Lyapunov magnitude ranking (BQ2). "
                  "Controls: random skip, destructive skip (top-ΔG and top-Lyapunov layers).",
        "model": MODEL_NAME,
        "n_problems": len(problems),
        "max_new_tokens": MAX_NEW_TOKENS,
        "baseline": {"en": baseline_en, "zh": baseline_zh},
        "dg_ranking": DG_RANKING,
        "lyap_ranking": LYAP_RANKING,
        "dg_destructive": DG_TOP5,
        "lyap_destructive": LYAP_TOP5,
        "results": all_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH} in {elapsed:.1f}s")

    # ── Summary table ──
    print("\n" + "="*70)
    print("SUMMARY: Lyapunov-Guided Layer Pruning")
    print("="*70)
    print(f"Baseline: EN={baseline_en}/20, ZH={baseline_zh}/20\n")
    print(f"{'Condition':<25s} {'Skip':>4s} {'EN':>4s} {'ZH':>4s}  {'ΔEN':>5s} {'ΔZH':>5s}  Layers")
    print("-"*80)
    for r in all_results:
        lab = r['label']
        n = r['n_skipped']
        en = r['en_correct']
        zh = r['zh_correct']
        d_en = en - baseline_en
        d_zh = zh - baseline_zh
        layers = r['skip_layers']
        ls = str(layers) if len(str(layers)) < 30 else str(layers[:5]) + f"..+{len(layers)-5}"
        print(f"{lab:<25s} {n:>4d} {en:>4d} {zh:>4d}  {d_en:>+5d} {d_zh:>+5d}  {ls}")

    # Random averages
    print("\n--- Random skip averages ---")
    for k in SKIP_COUNTS:
        rand_results = [r for r in all_results if r['label'].startswith(f'random_{k}_')]
        avg_en = np.mean([r['en_correct'] for r in rand_results])
        avg_zh = np.mean([r['zh_correct'] for r in rand_results])
        print(f"  Random k={k}: EN={avg_en:.1f}, ZH={avg_zh:.1f}")


if __name__ == "__main__":
    main()
