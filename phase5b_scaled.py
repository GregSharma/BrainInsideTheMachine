"""Phase 5B: Non-Circular Test with 200 Template Problems.

Run with: MPLBACKEND=Agg .venv/bin/python phase5b_scaled.py

Phase 5A was circular: contrastive Z optimizes for cross-lingual similarity,
then we tested cross-lingual similarity. This fixes that.

Design:
  1. Generate 200 paired zh/en math problems from templates (5 categories)
  2. Split 100 train / 100 test
  3. Build contrastive Z from TRAIN activations only
  4. Test on HELD-OUT activations:
     a) CKA(zh_test, en_test) in contrastive-Z vs random — still partially
        circular but now on unseen data, so tests generalization
     b) Probe: predict problem CATEGORY (5 classes, not 200) — non-circular,
        tests whether Z captures mathematical content
     c) Cross-lingual probe transfer: train on zh_test, predict en_test category
        — tests whether mathematical content transfers across languages in Z
  5. Compare all metrics vs 100 random subspaces of same dimension

The key test is (b) + (c): if contrastive Z enables better category prediction
AND better cross-lingual category transfer than random subspaces, that's
evidence of reasoning content, not just language removal.
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

from utils import get_attn_subspace, get_model_dims

MODEL_NAME = "Qwen/Qwen2.5-3B"
LAYER = 32
K_VALUES = [20, 50]
N_RANDOM = 100
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42

# ---------------------------------------------------------------------------
# Template-based problem generation
# ---------------------------------------------------------------------------

def generate_problems(n=200, seed=42):
    """Generate n paired zh/en math problems from 5 category templates.

    Categories (balanced: n/5 each):
      0: arithmetic — "compute A op B"
      1: combinatorics — "C(n,k)"
      2: modular — "A mod B"
      3: geometry — "area/perimeter of shape with params"
      4: sequences — "sum of first N terms of arithmetic/geometric seq"
    """
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    # Category 0: Arithmetic
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

    # Category 1: Combinatorics C(n,k)
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        from math import comb
        ans = comb(n_val, k_val)
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "answer": str(ans), "category": 1})

    # Category 2: Modular arithmetic
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        ans = a % b
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "answer": str(ans), "category": 2})

    # Category 3: Geometry (rectangle area)
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        ans = w * h
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "answer": str(ans), "category": 3})

    # Category 4: Arithmetic sequences — sum of first N terms
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


def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    if hsic_xx * hsic_yy == 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def generate_random_basis(d, k, rng):
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].T


def build_contrastive_z(zh_means, en_means, k, var_threshold=0.90):
    """Build contrastive Z from activation differences. Returns (basis, n_lang, varexp)."""
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
    z_basis = Vt_proj[:actual_k].astype(np.float32)
    return z_basis, n_lang, var_explained[:min(10, len(var_explained))].tolist()


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("PHASE 5B: NON-CIRCULAR TEST — 200 Problems, Train/Test Split")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    print(f"k values: {K_VALUES}, Random baselines: {N_RANDOM}")
    print("=" * 70)

    # Generate problems
    problems = generate_problems(200, seed=SEED)
    categories = np.array([p["category"] for p in problems])
    cat_names = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]
    print(f"\nGenerated {len(problems)} problems across {len(cat_names)} categories")
    for i, name in enumerate(cat_names):
        print(f"  {name}: {(categories == i).sum()}")

    # Train/test split (stratified)
    train_idx = []
    test_idx = []
    for cat in range(5):
        cat_indices = np.where(categories == cat)[0]
        np.random.default_rng(SEED).shuffle(cat_indices)
        half = len(cat_indices) // 2
        train_idx.extend(cat_indices[:half].tolist())
        test_idx.extend(cat_indices[half:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")

    # Load model & extract activations
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
    )
    model.eval()
    dims = get_model_dims(model)
    L, d, h, GQA = dims["L"], dims["d"], dims["h"], dims["GQA"]
    print(f"  L={L} d={d} h={h} GQA={GQA}")
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
        acts = activations["target"].float().numpy()
        zh_means[i] = acts.mean(axis=0)

    for i, prob in enumerate(tqdm(problems, desc="English forward")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        acts = activations["target"].float().numpy()
        en_means[i] = acts.mean(axis=0)

    hook_handle.remove()
    del model
    torch.cuda.empty_cache()

    print(f"\n  zh_means: {zh_means.shape}, en_means: {en_means.shape}")

    # Split activations
    zh_train, zh_test = zh_means[train_idx], zh_means[test_idx]
    en_train, en_test = en_means[train_idx], en_means[test_idx]
    cat_train, cat_test = categories[train_idx], categories[test_idx]

    # Build contrastive Z from TRAIN data only
    print("\n--- Building Contrastive Z from TRAIN split only ---")
    contrastive_z = {}
    contrastive_meta = {}
    for k in K_VALUES:
        z_basis, n_lang, lang_varexp = build_contrastive_z(zh_train, en_train, k)
        contrastive_z[k] = z_basis
        contrastive_meta[k] = {"n_lang_removed": n_lang, "lang_var_explained": lang_varexp}
        print(f"  k={k}: removed {n_lang} language dirs, Z shape {z_basis.shape}")

    # Generate random baselines
    print(f"\n--- Generating {N_RANDOM} random baselines ---")
    random_bases = {}
    for k in K_VALUES:
        random_bases[k] = [generate_random_basis(d, k, rng) for _ in range(N_RANDOM)]

    # Full-space baseline
    full_cka = linear_cka(zh_test, en_test)
    print(f"\n  Full-space CKA on test: {full_cka:.4f}")

    # Run tests on HELD-OUT test set
    all_results = {}
    for k in K_VALUES:
        print(f"\n{'='*70}")
        print(f"TESTING k={k} on HELD-OUT test set")
        print(f"{'='*70}")

        z_basis = contrastive_z[k]
        actual_k = z_basis.shape[0]

        # Project test data into Z
        zh_z = zh_test @ z_basis.T
        en_z = en_test @ z_basis.T

        # --- Test 1: CKA on held-out data ---
        cka_z = linear_cka(zh_z, en_z)
        cka_random = []
        for rb in random_bases[k]:
            cka_random.append(linear_cka(zh_test @ rb.T, en_test @ rb.T))
        cka_random = np.array(cka_random)
        cka_pct = float(np.mean(cka_random <= cka_z) * 100)
        print(f"\n  CKA(zh_test, en_test) in Z: {cka_z:.4f} (p={cka_pct:.0f}%)")
        print(f"  CKA random: mean={cka_random.mean():.4f}, std={cka_random.std():.4f}")

        # --- Test 2: Category prediction (NON-CIRCULAR) ---
        # Train on zh_test in Z, predict category. Compare vs random subspaces.
        # This tests whether Z captures MATHEMATICAL CONTENT, not just language similarity.
        scaler = StandardScaler().fit(zh_z)
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(scaler.transform(zh_z), cat_test)
        zh_cat_acc = float(clf.score(scaler.transform(zh_z), cat_test))

        # Also test on en (same probe, cross-lingual transfer of CATEGORY)
        scaler_en = StandardScaler().fit(en_z)
        en_cat_acc_self = float(RidgeClassifier(alpha=1.0).fit(
            scaler_en.transform(en_z), cat_test).score(scaler_en.transform(en_z), cat_test))

        # Cross-lingual category transfer: train on zh, test on en
        clf_xling = RidgeClassifier(alpha=1.0)
        clf_xling.fit(scaler.transform(zh_z), cat_test)
        xling_cat_acc = float(clf_xling.score(scaler_en.transform(en_z), cat_test))

        # Same for random subspaces
        cat_random_zh = []
        cat_random_xling = []
        for rb in random_bases[k]:
            zh_r = zh_test @ rb.T
            en_r = en_test @ rb.T
            s1 = StandardScaler().fit(zh_r)
            s2 = StandardScaler().fit(en_r)
            c = RidgeClassifier(alpha=1.0)
            c.fit(s1.transform(zh_r), cat_test)
            cat_random_zh.append(c.score(s1.transform(zh_r), cat_test))
            cat_random_xling.append(c.score(s2.transform(en_r), cat_test))

        cat_random_zh = np.array(cat_random_zh)
        cat_random_xling = np.array(cat_random_xling)
        cat_pct_zh = float(np.mean(cat_random_zh <= zh_cat_acc) * 100)
        cat_pct_xling = float(np.mean(cat_random_xling <= xling_cat_acc) * 100)

        print(f"\n  Category prediction (5 classes, chance=20%):")
        print(f"    zh→zh in Z: {zh_cat_acc:.0%} (p={cat_pct_zh:.0f}%, random={cat_random_zh.mean():.0%})")
        print(f"    en→en in Z: {en_cat_acc_self:.0%}")
        print(f"    zh→en in Z: {xling_cat_acc:.0%} (p={cat_pct_xling:.0f}%, random={cat_random_xling.mean():.0%})")

        # --- Test 3: Energy on held-out data ---
        zh_norms = np.linalg.norm(zh_test, axis=1) ** 2
        en_norms = np.linalg.norm(en_test, axis=1) ** 2
        zh_z_norms = np.linalg.norm(zh_z, axis=1) ** 2
        en_z_norms = np.linalg.norm(en_z, axis=1) ** 2
        zh_energy = (zh_z_norms / zh_norms).mean()
        en_energy = (en_z_norms / en_norms).mean()
        combined_energy = (zh_energy + en_energy) / 2
        expected = actual_k / d

        energy_random = []
        for rb in random_bases[k]:
            zh_r = zh_test @ rb.T
            en_r = en_test @ rb.T
            e = ((np.linalg.norm(zh_r, axis=1)**2 / zh_norms).mean() +
                 (np.linalg.norm(en_r, axis=1)**2 / en_norms).mean()) / 2
            energy_random.append(e)
        energy_random = np.array(energy_random)
        energy_pct = float(np.mean(energy_random <= combined_energy) * 100)

        print(f"\n  Energy: {combined_energy:.4f} (p={energy_pct:.0f}%, expected={expected:.4f})")
        print(f"    Concentration ratio: {combined_energy/expected:.1f}x")

        all_results[f"k{k}"] = {
            "cka": {"real": float(cka_z), "random_mean": float(cka_random.mean()),
                    "random_std": float(cka_random.std()), "percentile": cka_pct,
                    "random_dist": cka_random.tolist()},
            "category": {
                "zh_acc": zh_cat_acc, "en_acc": en_cat_acc_self,
                "xling_acc": xling_cat_acc,
                "random_zh_mean": float(cat_random_zh.mean()),
                "random_zh_std": float(cat_random_zh.std()),
                "random_xling_mean": float(cat_random_xling.mean()),
                "random_xling_std": float(cat_random_xling.std()),
                "pct_zh": cat_pct_zh, "pct_xling": cat_pct_xling,
                "random_zh_dist": cat_random_zh.tolist(),
                "random_xling_dist": cat_random_xling.tolist(),
            },
            "energy": {
                "combined": float(combined_energy), "expected": float(expected),
                "ratio": float(combined_energy / expected),
                "random_mean": float(energy_random.mean()),
                "percentile": energy_pct,
                "random_dist": energy_random.tolist(),
            },
            "meta": contrastive_meta[k],
        }

    # --- Plotting ---
    fig, axes = plt.subplots(len(K_VALUES), 3, figsize=(18, 5 * len(K_VALUES)))
    if len(K_VALUES) == 1:
        axes = axes[np.newaxis, :]

    for row, k in enumerate(K_VALUES):
        res = all_results[f"k{k}"]

        # CKA
        ax = axes[row, 0]
        ax.hist(res["cka"]["random_dist"], bins=20, alpha=0.6, color='gray', label='Random')
        ax.axvline(res["cka"]["real"], color='blue', lw=2, label=f'Contr-Z (p={res["cka"]["percentile"]:.0f}%)')
        ax.set_title(f"k={k}: CKA on HELD-OUT test")
        ax.legend(fontsize=8)

        # Category prediction (cross-lingual)
        ax = axes[row, 1]
        ax.hist(res["category"]["random_xling_dist"], bins=20, alpha=0.6, color='gray', label='Random')
        ax.axvline(res["category"]["xling_acc"], color='blue', lw=2,
                   label=f'Contr-Z (p={res["category"]["pct_xling"]:.0f}%)')
        ax.axvline(0.2, color='green', lw=1, ls=':', label='Chance (20%)')
        ax.set_title(f"k={k}: Category zh→en Transfer (NON-CIRCULAR)")
        ax.legend(fontsize=8)

        # Energy
        ax = axes[row, 2]
        ax.hist(res["energy"]["random_dist"], bins=20, alpha=0.6, color='gray', label='Random')
        ax.axvline(res["energy"]["combined"], color='blue', lw=2,
                   label=f'Contr-Z (p={res["energy"]["percentile"]:.0f}%)')
        ax.set_title(f"k={k}: Energy on HELD-OUT test")
        ax.legend(fontsize=8)

    plt.suptitle("Phase 5B: Non-Circular Test — Contrastive Z on Held-Out Data", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5b_scaled.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {OUTPUT_DIR / 'phase5b_scaled.png'}")

    # Save JSON (strip dists)
    save_results = {}
    for kkey, kres in all_results.items():
        save_results[kkey] = {}
        for test in kres:
            if test == "meta":
                save_results[kkey][test] = kres[test]
            else:
                save_results[kkey][test] = {k2: v2 for k2, v2 in kres[test].items()
                                            if not k2.endswith("_dist")}
    with open(OUTPUT_DIR / "phase5b_scaled.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'phase5b_scaled.json'}")

    # --- VERDICT ---
    print(f"\n{'='*70}")
    print("PHASE 5B VERDICT — NON-CIRCULAR TESTS")
    print(f"{'='*70}")
    for k in K_VALUES:
        res = all_results[f"k{k}"]
        print(f"\n  k={k}:")
        print(f"    CKA on held-out:          {res['cka']['real']:.4f} (p={res['cka']['percentile']:.0f}%)")
        print(f"    Category zh→zh:           {res['category']['zh_acc']:.0%} (p={res['category']['pct_zh']:.0f}%)")
        print(f"    Category zh→en (KEY):     {res['category']['xling_acc']:.0%} (p={res['category']['pct_xling']:.0f}%)")
        print(f"    Energy:                   {res['energy']['combined']:.4f} (p={res['energy']['percentile']:.0f}%, {res['energy']['ratio']:.1f}x)")

        cat_special = res["category"]["pct_xling"] >= 95
        cka_special = res["cka"]["percentile"] >= 95
        if cat_special:
            print(f"    ==> CATEGORY TRANSFER IS SPECIAL. Z carries reasoning content.")
        elif cka_special and not cat_special:
            print(f"    ==> CKA special but category isn't. Z captures shared structure but not reasoning.")
        else:
            print(f"    ==> Not special. Language-invariant ≠ reasoning-relevant.")

    print("\nDone.")


if __name__ == "__main__":
    main()
