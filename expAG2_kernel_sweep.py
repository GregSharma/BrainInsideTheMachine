"""
Exp AG2: Kernel Surgery Dimension Sweep

AG showed 10D kernel surgery: EN 3→8 (+167%), ZH 9→9 (preserved), FAT 3-4x faster.
But ZH outputs degenerate qualitatively (repetition, hallucination).

Hypothesis: 10D is too aggressive. Some of those dims carry ZH math content
(confirmed by Exp X asymmetry). Sweep: 1D, 3D, 5D, 10D surgery.
Also test: flip_1d surgery (replace P with reflection I - 2uu^T on dim 1 only).
"""

import json
import numpy as np
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom
import copy

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SEED = 42
LANGUAGES = ['zh', 'en', 'es', 'ar', 'ja', 'ko', 'sw']
SURGERY_LAYERS = list(range(9, 27))

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


def compute_lang_axes(n_dims=10):
    """Get language axes from cached multilingual data."""
    cache_path = OUTPUT_DIR / "multilingual_all_layers.npz"
    data = np.load(cache_path, allow_pickle=True)
    d = 2048
    axes = {}  # layer -> (n_dims, 2048)
    for l in SURGERY_LAYERS:
        per_lang = np.stack([data[f"{lang}_L{l}"] for lang in LANGUAGES], axis=1)
        prob_means = per_lang.mean(axis=1, keepdims=True)
        deviations = (per_lang - prob_means).reshape(-1, d)
        _, S, Vt = np.linalg.svd(deviations, full_matrices=False)
        axes[l] = Vt[:n_dims].astype(np.float32)
    return axes


def make_projector(axes, k):
    """P = I - U_k U_k^T, projecting out top k language dims."""
    d = axes.shape[1]
    U = axes[:k].T  # (2048, k)
    return np.eye(d, dtype=np.float32) - (U @ U.T)


def make_flip_projector(axes):
    """Flip along top-1 language axis: I - 2 u u^T (reflection)."""
    d = axes.shape[1]
    u = axes[0:1].T  # (2048, 1)
    return np.eye(d, dtype=np.float32) - 2.0 * (u @ u.T)


def apply_surgery(model, axes_by_layer, surgery_type, k=10):
    """Apply surgery to W_down. Returns modified model (in-place)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for l in SURGERY_LAYERS:
        if surgery_type == 'flip_1d':
            P = make_flip_projector(axes_by_layer[l])
        else:
            P = make_projector(axes_by_layer[l], k)

        P_t = torch.tensor(P, dtype=dtype, device=device)
        W_old = model.model.layers[l].mlp.down_proj.weight.data
        model.model.layers[l].mlp.down_proj.weight.data = P_t @ W_old

    return model


def check_answer(text, correct_answer):
    target = str(correct_answer)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return target in numbers


def find_fat(text, correct_answer):
    target = str(correct_answer)
    tokens = text.split()
    for i, tok in enumerate(tokens):
        nums = re.findall(r'-?\d+\.?\d*', tok)
        if target in nums:
            return i
    return -1


def detect_lang(text):
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return 'zh' if cjk > latin else ('en' if latin > 0 else 'unk')


def run_test(model, tokenizer, problems, label="", max_new=128):
    results = {}
    for lang in ['en', 'zh']:
        correct = 0
        fats = []
        lang_pres = {'zh': 0, 'en': 0, 'unk': 0}
        samples = []
        for prob in problems:
            inputs = tokenizer(prob[lang], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            ok = check_answer(gen, prob['answer'])
            fat = find_fat(gen, prob['answer'])
            if ok:
                correct += 1
            fats.append(fat)
            dl = detect_lang(gen)
            lang_pres[dl] += 1
            samples.append({'ans': prob['answer'], 'ok': ok, 'fat': fat, 'gen': gen[:120]})

        valid_fats = [f for f in fats if f >= 0]
        mfat = float(np.mean(valid_fats)) if valid_fats else -1
        results[lang] = {
            'correct': correct, 'total': len(problems), 'mean_fat': mfat,
            'lang_pres': lang_pres, 'samples': samples
        }
    return results


def main():
    print("=" * 70)
    print("EXP AG2: KERNEL SURGERY DIMENSION SWEEP")
    print("=" * 70)

    # Compute language axes (max 10 dims)
    print("\n[1] Computing language axes...")
    all_axes = compute_lang_axes(n_dims=10)

    # Generate test problems
    test_problems = generate_test_problems(n_test=4)
    print(f"  {len(test_problems)} test problems")

    # Conditions to test
    conditions = [
        ('baseline', None, None),
        ('project_1d', 'project', 1),
        ('project_3d', 'project', 3),
        ('project_5d', 'project', 5),
        ('project_10d', 'project', 10),
        ('flip_1d', 'flip_1d', None),
    ]

    all_results = {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    for cond_name, surgery_type, k in conditions:
        print(f"\n{'='*50}")
        print(f"  Condition: {cond_name}")
        print(f"{'='*50}")

        # Reload fresh model each time
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        model.eval()

        if surgery_type is not None:
            print(f"  Applying {cond_name} surgery...")
            apply_surgery(model, all_axes, surgery_type, k=k)

        res = run_test(model, tokenizer, test_problems, label=cond_name)
        all_results[cond_name] = res

        for lang in ['en', 'zh']:
            r = res[lang]
            print(f"  {lang.upper()}: {r['correct']}/{r['total']} correct, "
                  f"FAT={r['mean_fat']:.1f}, lang={r['lang_pres']}")

        del model
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Condition':>14} | {'EN':>4} {'EN_FAT':>7} {'EN_lang':>8} | {'ZH':>4} {'ZH_FAT':>7} {'ZH_lang':>8}")
    print("-" * 70)
    for cond_name, _, _ in conditions:
        r = all_results[cond_name]
        en, zh = r['en'], r['zh']
        en_lp = f"{en['lang_pres'].get('en',0)}/{en['total']}"
        zh_lp = f"{zh['lang_pres'].get('zh',0)}/{zh['total']}"
        print(f"  {cond_name:>12} | {en['correct']:>3}/{en['total']} {en['mean_fat']:>7.1f} {en_lp:>8} | "
              f"{zh['correct']:>3}/{zh['total']} {zh['mean_fat']:>7.1f} {zh_lp:>8}")

    # Save
    out_path = OUTPUT_DIR / "expAG2_kernel_sweep.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
