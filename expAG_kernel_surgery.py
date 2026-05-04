"""
Exp AG: Kernel Weight Surgery — Bake the language kernel into W_down

The insight: P = I - U U^T (kernel projector, 2048x2048) strips language from
the MLP output. Instead of hooking at inference, modify W_down permanently:

    W_down_new = P @ W_down_old    (per layer, L9-L26)

This removes the MLP's ability to output into language dimensions.
Language identity survives via skip connections (residual stream carries PC0).

Test: load modified model, generate on standard 20-problem test set (EN + ZH),
compare accuracy + FAT (first-answer-token) vs unmodified baseline.

Why SGD didn't find this:
1. Training loss rewards verbose scaffolding — language "noise" IS the target
2. Kernel requires 7 simultaneous languages — SGD sees one per example
3. Cross-layer coordination problem — 18 layers must change together
"""

import json
import numpy as np
import torch
import re
import copy
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom
import gc

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SEED = 42
LANGUAGES = ['zh', 'en', 'es', 'ar', 'ja', 'ko', 'sw']
SURGERY_LAYERS = list(range(9, 27))  # L9-L26
N_LANG_DIMS = 10  # Same as Exp AB

# ── Problem generation (same seed as all experiments) ──

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
    """First 4 per category = 20 test problems, same as all experiments."""
    import math
    rng = pyrandom.Random(SEED)
    problems = []
    per_cat = 200 // 5

    cats = []
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        ans = a + b if op == "plus" else a * b
        prompts = {}
        for lang in ['en', 'zh']:
            key = f'arithmetic_{op}'
            prompts[lang] = TEMPLATES[lang][key].format(a=a, b=b)
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
        prompts = {}
        # No sequences template in test set — use geometry slot count
        # Actually we need it:
        cats.append(('sequences', ans, {
            'en': f"An arithmetic sequence: first term {a1}, common difference {d}. Sum of first {n_terms} terms?",
            'zh': f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
        }))

    # Shuffle with same seed
    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats)))
    rng2.shuffle(indices)
    cats = [cats[i] for i in indices]

    # Take first n_test per category
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


def compute_kernel_projectors(n_lang_dims=10):
    """Recompute kernel projectors from cached multilingual activations."""
    cache_path = OUTPUT_DIR / "multilingual_all_layers.npz"
    assert cache_path.exists(), f"Need {cache_path} from Exp AB"

    print("Loading cached multilingual activations...")
    data = np.load(cache_path, allow_pickle=True)
    n_layers = 36
    d = 2048

    projectors = {}  # layer -> (2048, 2048) float32
    lang_axes = {}   # layer -> (n_lang_dims, 2048) float32

    for l in SURGERY_LAYERS:
        # Stack per-language activations: (200, 7, 2048)
        per_lang = np.stack([data[f"{lang}_L{l}"] for lang in LANGUAGES], axis=1)
        prob_means = per_lang.mean(axis=1, keepdims=True)
        deviations = (per_lang - prob_means).reshape(-1, d)  # (1400, 2048)

        U, S, Vt = np.linalg.svd(deviations, full_matrices=False)
        axes = Vt[:n_lang_dims]  # (n_lang_dims, 2048)

        U_mat = axes.T  # (2048, n_lang_dims)
        P = np.eye(d, dtype=np.float32) - (U_mat @ U_mat.T).astype(np.float32)

        projectors[l] = P
        lang_axes[l] = axes

        var_explained = (S[:n_lang_dims]**2).sum() / (S**2).sum()
        print(f"  L{l}: lang variance captured = {var_explained:.3f}")

    return projectors, lang_axes


def apply_kernel_surgery(model, projectors):
    """Modify W_down at surgery layers: W_new = P @ W_old"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for l in SURGERY_LAYERS:
        P = torch.tensor(projectors[l], dtype=dtype, device=device)
        W_old = model.model.layers[l].mlp.down_proj.weight.data  # (2048, 11008)

        # W_down maps 11008 -> 2048. Each column of W_old.T is a 2048-dim output.
        # P @ W_old: project each output vector into the kernel.
        W_new = P @ W_old  # (2048, 2048) @ (2048, 11008) = (2048, 11008)

        model.model.layers[l].mlp.down_proj.weight.data = W_new
        print(f"  L{l}: W_down modified. Norm change: {W_old.norm():.1f} -> {W_new.norm():.1f}")

    return model


def check_answer(text, correct_answer):
    """Check if correct answer appears in generated text."""
    target = str(correct_answer)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return target in numbers


def find_first_answer_token(text, correct_answer):
    """Find position of first token containing the answer."""
    target = str(correct_answer)
    tokens = text.split()
    for i, tok in enumerate(tokens):
        nums = re.findall(r'-?\d+\.?\d*', tok)
        if target in nums:
            return i
    return -1


def run_generation_test(model, tokenizer, test_problems, max_new_tokens=128, label=""):
    """Run generation on test set, return accuracy and FAT metrics."""
    results = []
    for lang in ['en', 'zh']:
        correct = 0
        fats = []
        outputs = []
        for prob in test_problems:
            prompt = prob[lang]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
            gen_text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            ok = check_answer(gen_text, prob['answer'])
            fat = find_first_answer_token(gen_text, prob['answer'])
            if ok:
                correct += 1
            fats.append(fat)
            outputs.append({
                'prompt': prompt[:80],
                'answer': prob['answer'],
                'correct': ok,
                'fat': fat,
                'gen': gen_text[:200],
            })

        valid_fats = [f for f in fats if f >= 0]
        mean_fat = np.mean(valid_fats) if valid_fats else -1

        results.append({
            'lang': lang,
            'label': label,
            'correct': correct,
            'total': len(test_problems),
            'accuracy': correct / len(test_problems),
            'mean_fat': float(mean_fat),
            'outputs': outputs,
        })
        print(f"  [{label}] {lang.upper()}: {correct}/{len(test_problems)} correct, mean FAT={mean_fat:.1f}")

    return results


def detect_output_language(text):
    """Heuristic: count CJK characters vs latin."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if cjk > latin:
        return 'zh'
    elif latin > 0:
        return 'en'
    return 'unknown'


def main():
    print("="*70)
    print("EXP AG: KERNEL WEIGHT SURGERY")
    print("Bake language kernel into W_down at L9-L26")
    print("="*70)

    # ── Step 1: Compute kernel projectors ──
    print("\n[1] Computing kernel projectors from 7-language data...")
    projectors, lang_axes = compute_kernel_projectors(N_LANG_DIMS)

    # ── Step 2: Load model ──
    print(f"\n[2] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    # ── Step 3: Baseline generation ──
    print("\n[3] Generating test problems...")
    test_problems = generate_test_problems(n_test=4)
    print(f"  {len(test_problems)} test problems")

    print("\n[4] Running BASELINE generation...")
    baseline_results = run_generation_test(model, tokenizer, test_problems, label="baseline")

    # ── Step 4: Apply kernel surgery ──
    print(f"\n[5] Applying kernel surgery to W_down at L{SURGERY_LAYERS[0]}-L{SURGERY_LAYERS[-1]}...")
    model = apply_kernel_surgery(model, projectors)

    # ── Step 5: Post-surgery generation ──
    print("\n[6] Running SURGERY generation...")
    surgery_results = run_generation_test(model, tokenizer, test_problems, label="surgery")

    # ── Step 6: Language preservation check ──
    print("\n[7] Checking language preservation...")
    for res in surgery_results:
        lang_counts = {'zh': 0, 'en': 0, 'unknown': 0}
        for out in res['outputs']:
            detected = detect_output_language(out['gen'])
            lang_counts[detected] += 1
        res['lang_preservation'] = lang_counts
        print(f"  {res['lang'].upper()} prompts -> output languages: {lang_counts}")

    # ── Step 7: Summary ──
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Condition':>12} | {'EN acc':>7} {'EN FAT':>7} | {'ZH acc':>7} {'ZH FAT':>7}")
    print("-"*55)
    for b, s in zip(baseline_results, surgery_results):
        if b['lang'] == 'en':
            pass  # printed in pairs below

    for lang in ['en', 'zh']:
        b = [r for r in baseline_results if r['lang'] == lang][0]
        s = [r for r in surgery_results if r['lang'] == lang][0]
        print(f"  baseline   | {lang.upper()} {b['correct']}/{b['total']}  FAT={b['mean_fat']:5.1f}")
        print(f"  surgery    | {lang.upper()} {s['correct']}/{s['total']}  FAT={s['mean_fat']:5.1f}")

    # ── Save results ──
    all_results = {
        'experiment': 'AG_kernel_surgery',
        'surgery_layers': SURGERY_LAYERS,
        'n_lang_dims': N_LANG_DIMS,
        'n_test_problems': len(test_problems),
        'max_new_tokens': 128,
        'baseline': baseline_results,
        'surgery': surgery_results,
        'hypothesis': 'Baking kernel into W_down removes language interference from MLP, '
                      'preserving accuracy while potentially improving efficiency (FAT)',
        'sgd_argument': [
            'SGD optimizes next-token loss which rewards verbose scaffolding',
            'Kernel requires 7 simultaneous languages — SGD sees one per example',
            'Cross-layer coordination: 18 layers must change together',
        ],
    }

    # Stringify non-serializable
    for key in ['baseline', 'surgery']:
        for r in all_results[key]:
            for out in r['outputs']:
                out['correct'] = bool(out['correct'])

    out_path = OUTPUT_DIR / "expAG_kernel_surgery.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
