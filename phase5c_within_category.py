"""Phase 5C: Within-Category Cross-Lingual Probe — The Reasoning Test.

Run with: MPLBACKEND=Agg .venv/bin/python phase5c_within_category.py

The killer test: within each category, all problems have identical template
structure. The ONLY thing that varies is the mathematical content (the actual
numbers). If contrastive Z enables cross-lingual problem identification
within a category better than random subspaces, that's reasoning content.

Design:
  1. Generate 200 problems (5 categories x 40), split train/test
  2. Build contrastive Z from train activations
  3. For EACH category separately on held-out test:
     - 20 problems per category in test set
     - Train probe on zh activations to identify which problem (20 classes)
     - Test on en activations (cross-lingual transfer)
     - Compare vs 100 random subspaces
  4. Also pool across categories for overall within-category score

This eliminates template surface features as an explanation.
"""

import json
import random as pyrandom
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_model_dims

MODEL_NAME = "Qwen/Qwen2.5-3B"
LAYER = 32
K_VALUES = [20, 50]
N_RANDOM = 100
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42


def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            ans = a + b
            zh = f"计算 {a} + {b} 的值。"
            en = f"Calculate {a} + {b}."
        else:
            ans = a * b
            zh = f"计算 {a} × {b} 的值。"
            en = f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "answer": str(ans), "category": 0})

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        from math import comb
        ans = comb(n_val, k_val)
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "answer": str(ans), "category": 1})

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        ans = a % b
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "answer": str(ans), "category": 2})

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        ans = w * h
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "answer": str(ans), "category": 3})

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        problems.append({"zh": zh, "en": en, "answer": str(ans), "category": 4})

    rng.shuffle(problems)
    return problems


def generate_random_basis(d, k, rng):
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].T


def build_contrastive_z(zh_means, en_means, k, var_threshold=0.90):
    N, d = zh_means.shape
    diffs = zh_means - en_means
    diffs_centered = diffs - diffs.mean(axis=0)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)

    var_explained = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)
    n_lang = max(1, int(np.searchsorted(cumvar, var_threshold) + 1))
    n_lang = min(n_lang, len(S))

    lang_dirs = Vt[:n_lang]
    proj_out = np.eye(d, dtype=np.float32) - lang_dirs.T @ lang_dirs

    all_means = np.concatenate([zh_means, en_means], axis=0)
    projected = all_means @ proj_out.T
    projected_centered = projected - projected.mean(axis=0)
    _, _, Vt_proj = np.linalg.svd(projected_centered, full_matrices=False)

    actual_k = min(k, Vt_proj.shape[0])
    return Vt_proj[:actual_k].astype(np.float32), n_lang


def within_category_probe(zh_acts, en_acts, n_per_cat=20):
    """Train probe on zh to identify problem, test on en. Returns accuracy.

    zh_acts, en_acts: (n_per_cat, k) — activations for one category.
    Labels: 0..n_per_cat-1 (problem identity).
    """
    labels = np.arange(len(zh_acts))
    if len(labels) < 3:
        return 0.0

    s_zh = StandardScaler().fit(zh_acts)
    s_en = StandardScaler().fit(en_acts)

    clf = RidgeClassifier(alpha=1.0)
    clf.fit(s_zh.transform(zh_acts), labels)
    return float(clf.score(s_en.transform(en_acts), labels))


def main():
    rng = np.random.default_rng(SEED)
    cat_names = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]

    print("=" * 70)
    print("PHASE 5C: WITHIN-CATEGORY CROSS-LINGUAL PROBE")
    print("The test: can Z distinguish problems with identical templates?")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    print(f"k values: {K_VALUES}, Random baselines: {N_RANDOM}")
    print("=" * 70)

    problems = generate_problems(200, seed=SEED)
    categories = np.array([p["category"] for p in problems])

    # Stratified train/test split
    train_idx, test_idx = [], []
    for cat in range(5):
        cat_indices = np.where(categories == cat)[0]
        np.random.default_rng(SEED).shuffle(cat_indices)
        half = len(cat_indices) // 2
        train_idx.extend(cat_indices[:half].tolist())
        test_idx.extend(cat_indices[half:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    cat_test = categories[test_idx]
    print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")

    # Load model & extract activations
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
    )
    model.eval()
    dims = get_model_dims(model)
    d = dims["d"]
    N = len(problems)

    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h_out.detach().cpu().squeeze(0)
        return hook

    hook_handle = model.model.layers[LAYER].register_forward_hook(make_hook("target"))

    zh_means = np.zeros((N, d), dtype=np.float32)
    en_means = np.zeros((N, d), dtype=np.float32)

    for i, prob in enumerate(tqdm(problems, desc="Chinese forward")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        zh_means[i] = activations["target"].float().numpy().mean(axis=0)

    for i, prob in enumerate(tqdm(problems, desc="English forward")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        en_means[i] = activations["target"].float().numpy().mean(axis=0)

    hook_handle.remove()
    del model
    torch.cuda.empty_cache()

    zh_train, zh_test = zh_means[train_idx], zh_means[test_idx]
    en_train, en_test = en_means[train_idx], en_means[test_idx]

    # Build contrastive Z from TRAIN only
    print("\n--- Building Contrastive Z from TRAIN ---")
    contrastive_z = {}
    for k in K_VALUES:
        z_basis, n_lang = build_contrastive_z(zh_train, en_train, k)
        contrastive_z[k] = z_basis
        print(f"  k={k}: removed {n_lang} language dirs, shape {z_basis.shape}")

    # Random baselines
    random_bases = {k: [generate_random_basis(d, k, rng) for _ in range(N_RANDOM)] for k in K_VALUES}

    # Within-category probe on held-out test
    all_results = {}
    for k in K_VALUES:
        print(f"\n{'='*70}")
        print(f"WITHIN-CATEGORY PROBE k={k}")
        print(f"{'='*70}")

        z_basis = contrastive_z[k]

        # Project test data
        zh_z = zh_test @ z_basis.T
        en_z = en_test @ z_basis.T

        cat_results = {}
        for cat in range(5):
            cat_mask = cat_test == cat
            zh_cat = zh_z[cat_mask]
            en_cat = en_z[cat_mask]
            n_cat = cat_mask.sum()
            chance = 1.0 / n_cat

            # Real Z accuracy
            z_acc = within_category_probe(zh_cat, en_cat)

            # Random subspace accuracies
            rand_accs = []
            for rb in random_bases[k]:
                zh_r = zh_test[cat_mask] @ rb.T
                en_r = en_test[cat_mask] @ rb.T
                rand_accs.append(within_category_probe(zh_r, en_r))
            rand_accs = np.array(rand_accs)
            pct = float(np.mean(rand_accs <= z_acc) * 100)

            cat_results[cat_names[cat]] = {
                "z_acc": z_acc, "random_mean": float(rand_accs.mean()),
                "random_std": float(rand_accs.std()), "percentile": pct,
                "chance": float(chance), "n_problems": int(n_cat),
                "random_dist": rand_accs.tolist(),
            }

            print(f"\n  {cat_names[cat]} ({n_cat} problems, chance={chance:.0%}):")
            print(f"    Z accuracy:      {z_acc:.0%}")
            print(f"    Random mean:     {rand_accs.mean():.0%} ± {rand_accs.std():.0%}")
            print(f"    Percentile:      {pct:.0f}%")
            if pct >= 95:
                print(f"    *** SIGNIFICANT — Z carries mathematical content ***")

        # Pooled: average across categories
        pool_z = np.mean([cat_results[c]["z_acc"] for c in cat_names])
        pool_rand = np.mean([cat_results[c]["random_mean"] for c in cat_names])
        # For pooled percentile: average per-random-draw across categories
        pooled_rand_draws = np.zeros(N_RANDOM)
        for cat in range(5):
            pooled_rand_draws += np.array(cat_results[cat_names[cat]]["random_dist"])
        pooled_rand_draws /= 5
        pooled_pct = float(np.mean(pooled_rand_draws <= pool_z) * 100)

        cat_results["pooled"] = {
            "z_acc": float(pool_z), "random_mean": float(pool_rand),
            "percentile": pooled_pct,
        }

        print(f"\n  POOLED across categories:")
        print(f"    Z mean accuracy: {pool_z:.0%}")
        print(f"    Random mean:     {pool_rand:.0%}")
        print(f"    Percentile:      {pooled_pct:.0f}%")

        all_results[f"k{k}"] = cat_results

    # --- Plotting ---
    fig, axes = plt.subplots(len(K_VALUES), 5, figsize=(25, 5 * len(K_VALUES)))
    if len(K_VALUES) == 1:
        axes = axes[np.newaxis, :]

    for row, k in enumerate(K_VALUES):
        for col, cat in enumerate(cat_names):
            ax = axes[row, col]
            res = all_results[f"k{k}"][cat]
            ax.hist(res["random_dist"], bins=20, alpha=0.6, color='gray', label='Random')
            ax.axvline(res["z_acc"], color='blue', lw=2,
                       label=f'Contr-Z ({res["z_acc"]:.0%}, p={res["percentile"]:.0f}%)')
            ax.axvline(res["chance"], color='green', lw=1, ls=':', label=f'Chance ({res["chance"]:.0%})')
            ax.set_title(f"k={k}: {cat}\n(zh→en within-cat)")
            ax.legend(fontsize=7)
            ax.set_xlabel("Accuracy")

    plt.suptitle("Phase 5C: Within-Category Cross-Lingual Probe — Does Z Carry Math Content?",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5c_within_category.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {OUTPUT_DIR / 'phase5c_within_category.png'}")

    # Save JSON
    save_results = {}
    for kkey, kres in all_results.items():
        save_results[kkey] = {}
        for cat, cres in kres.items():
            save_results[kkey][cat] = {k2: v2 for k2, v2 in cres.items()
                                       if not k2.endswith("_dist")}
    with open(OUTPUT_DIR / "phase5c_within_category.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'phase5c_within_category.json'}")

    # --- VERDICT ---
    print(f"\n{'='*70}")
    print("PHASE 5C VERDICT — DOES Z CARRY MATHEMATICAL CONTENT?")
    print(f"{'='*70}")
    for k in K_VALUES:
        res = all_results[f"k{k}"]
        n_sig = sum(1 for c in cat_names if res[c]["percentile"] >= 95)
        print(f"\n  k={k}: {n_sig}/5 categories significant at 95th percentile")
        for cat in cat_names:
            r = res[cat]
            marker = "***" if r["percentile"] >= 95 else "   "
            print(f"    {marker} {cat:<15} Z={r['z_acc']:.0%}  Random={r['random_mean']:.0%}  p={r['percentile']:.0f}%")
        p = res["pooled"]
        print(f"    --- Pooled:       Z={p['z_acc']:.0%}  Random={p['random_mean']:.0%}  p={p['percentile']:.0f}%")

        if n_sig >= 3:
            print(f"    ==> Z CARRIES MATHEMATICAL CONTENT across multiple categories.")
        elif n_sig >= 1:
            print(f"    ==> PARTIAL signal — some categories show reasoning content.")
        else:
            print(f"    ==> Z does NOT carry mathematical content. Shared ≠ reasoning.")

    print("\nDone.")


if __name__ == "__main__":
    main()
