"""Thread 2: Characterize PCs 2-5 at L32.

PC1 is the mean direction (zero alignment contribution).
PCs 2-5 carry 91% of cross-lingual alignment. What ARE they?

For each PC:
1. Per-problem loading — correlate with problem metadata
2. Per-category mean loading — do PCs separate categories?
3. Top/bottom problems — what's special about high vs low loading?
4. Cross-language consistency — does zh loading predict en loading?
5. Correlation with problem features (numerical values, prompt length, answer magnitude)
"""

import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import stats
import random as pyrandom

OUTPUT_DIR = Path("output")

CAT_NAMES = ['arithmetic', 'combinatorics', 'modular', 'geometry', 'sequences']


def generate_problems(n=200, seed=42):
    """Same deterministic problems as all scripts."""
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh = f"计算 {a} + {b} 的值。"
            en = f"Calculate {a} + {b}."
            answer = a + b
        else:
            zh = f"计算 {a} × {b} 的值。"
            en = f"Calculate {a} × {b}."
            answer = a * b
        problems.append({"zh": zh, "en": en, "category": 0, "answer": answer,
                          "a": a, "b": b, "prompt_len_zh": len(zh), "prompt_len_en": len(en)})

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        from math import comb
        answer = comb(n_val, k_val)
        problems.append({"zh": zh, "en": en, "category": 1, "answer": answer,
                          "a": n_val, "b": k_val, "prompt_len_zh": len(zh), "prompt_len_en": len(en)})

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        answer = a % b
        problems.append({"zh": zh, "en": en, "category": 2, "answer": answer,
                          "a": a, "b": b, "prompt_len_zh": len(zh), "prompt_len_en": len(en)})

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        answer = w * h
        problems.append({"zh": zh, "en": en, "category": 3, "answer": answer,
                          "a": w, "b": h, "prompt_len_zh": len(zh), "prompt_len_en": len(en)})

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        answer = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        problems.append({"zh": zh, "en": en, "category": 4, "answer": answer,
                          "a": a1, "b": d, "prompt_len_zh": len(zh), "prompt_len_en": len(en)})

    rng.shuffle(problems)
    return problems


def matched_vs_scrambled_z(zh, en, n_perms=1000):
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    rng = np.random.RandomState(42)
    scrambled = np.array([
        np.mean(np.sum(zh_u * en_u[rng.permutation(len(en_u))], axis=1))
        for _ in range(n_perms)
    ])
    z = (matched - scrambled.mean()) / scrambled.std()
    return float(z)


def main():
    print("Loading clean last-token data...")
    data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")
    problems = generate_problems(200, seed=42)
    categories = data["categories"]
    N = 200

    # Problem metadata arrays
    answers = np.array([p["answer"] for p in problems], dtype=float)
    log_answers = np.log1p(np.abs(answers))
    a_vals = np.array([p["a"] for p in problems], dtype=float)
    b_vals = np.array([p["b"] for p in problems], dtype=float)
    prompt_len_zh = np.array([p["prompt_len_zh"] for p in problems], dtype=float)
    prompt_len_en = np.array([p["prompt_len_en"] for p in problems], dtype=float)

    results = {}

    for target_layer in [32, 35]:
        print(f"\n{'='*70}")
        print(f"LAYER {target_layer}: PC CHARACTERIZATION")
        print(f"{'='*70}")

        zh = data[f"zh_L{target_layer}"]
        en = data[f"en_L{target_layer}"]

        # Unit normalize
        zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
        en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
        combined_u = np.vstack([zh_u, en_u])

        # Fit PCA on combined unit-normalized data
        pca = PCA(n_components=20)
        pca.fit(combined_u)

        # Project zh and en separately
        zh_proj = pca.transform(zh_u)  # (200, 20)
        en_proj = pca.transform(en_u)  # (200, 20)

        print(f"\nVariance explained: {pca.explained_variance_ratio_[:10]}")
        print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_[:10])}")

        # ========== 1. Per-PC alignment contribution ==========
        print(f"\n--- 1. Per-PC alignment contribution ---")
        # For each subset of PCs, compute matched-vs-scrambled z
        for k in [1, 2, 3, 5, 10, 20]:
            z = matched_vs_scrambled_z(zh_proj[:, :k], en_proj[:, :k])
            print(f"  PCs 1-{k}: z={z:.1f}")

        # Individual PC alignment contribution (leave-one-out)
        z_all20 = matched_vs_scrambled_z(zh_proj[:, :20], en_proj[:, :20], n_perms=500)
        print(f"\n  Leave-one-out (z drop when removing each PC from top-20):")
        for pc_idx in range(10):
            mask = list(range(20))
            mask.remove(pc_idx)
            z_without = matched_vs_scrambled_z(zh_proj[:, mask], en_proj[:, mask], n_perms=500)
            dz = z_all20 - z_without
            print(f"    Remove PC{pc_idx}: z drops by {dz:+.1f} (from {z_all20:.1f} to {z_without:.1f})")

        # ========== 2. Per-category PC loadings ==========
        print(f"\n--- 2. Per-category mean loadings (zh) ---")
        print(f"  {'Category':>15} | " + " | ".join([f"PC{i:d}" for i in range(6)]))
        print("  " + "-" * 80)

        pc_category_results = {}
        for cat in range(5):
            mask = categories == cat
            zh_cat_mean = zh_proj[mask].mean(axis=0)
            en_cat_mean = en_proj[mask].mean(axis=0)
            vals = " | ".join([f"{zh_cat_mean[i]:+.3f}" for i in range(6)])
            print(f"  {CAT_NAMES[cat]:>15} | {vals}")
            pc_category_results[CAT_NAMES[cat]] = {
                "zh_means": [float(x) for x in zh_cat_mean[:10]],
                "en_means": [float(x) for x in en_cat_mean[:10]],
            }

        # Category separability per PC (one-way ANOVA F-stat)
        print(f"\n  ANOVA F-stat per PC (higher = more category-separating):")
        for pc_idx in range(10):
            groups = [zh_proj[categories == cat, pc_idx] for cat in range(5)]
            f_stat, p_val = stats.f_oneway(*groups)
            marker = " ***" if p_val < 0.001 else (" **" if p_val < 0.01 else (" *" if p_val < 0.05 else ""))
            print(f"    PC{pc_idx}: F={f_stat:.1f}, p={p_val:.4f}{marker}")

        # ========== 3. Cross-language PC consistency ==========
        print(f"\n--- 3. Cross-language consistency (zh vs en loadings) ---")
        for pc_idx in range(10):
            r, p = stats.pearsonr(zh_proj[:, pc_idx], en_proj[:, pc_idx])
            print(f"    PC{pc_idx}: r={r:.3f}, p={p:.1e}")

        # ========== 4. Correlation with problem features ==========
        print(f"\n--- 4. Correlation with problem features (zh loadings) ---")
        features = {
            "log_answer": log_answers,
            "a_val": a_vals,
            "b_val": b_vals,
            "prompt_len": prompt_len_zh,
        }

        for pc_idx in range(6):
            print(f"\n    PC{pc_idx}:")
            for fname, fvals in features.items():
                r, p = stats.pearsonr(zh_proj[:, pc_idx], fvals)
                marker = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
                if abs(r) > 0.15 or p < 0.05:
                    print(f"      vs {fname:>12}: r={r:+.3f}, p={p:.3f}{marker}")

            # Also per-category correlation with answer magnitude
            for cat in range(5):
                mask = categories == cat
                if mask.sum() >= 10:
                    r, p = stats.pearsonr(zh_proj[mask, pc_idx], log_answers[mask])
                    if abs(r) > 0.3:
                        print(f"      vs log_answer ({CAT_NAMES[cat]}): r={r:+.3f}")

        # ========== 5. What do top/bottom problems look like? ==========
        print(f"\n--- 5. Top/bottom problems per PC ---")
        for pc_idx in range(5):
            order = np.argsort(zh_proj[:, pc_idx])
            top3 = order[-3:][::-1]
            bot3 = order[:3]
            print(f"\n    PC{pc_idx}:")
            print(f"      Top 3 (highest loading):")
            for idx in top3:
                p = problems[idx]
                print(f"        [{CAT_NAMES[p['category']]}] {p['en'][:60]}  (loading={zh_proj[idx, pc_idx]:.3f})")
            print(f"      Bottom 3 (lowest loading):")
            for idx in bot3:
                p = problems[idx]
                print(f"        [{CAT_NAMES[p['category']]}] {p['en'][:60]}  (loading={zh_proj[idx, pc_idx]:.3f})")

        # ========== 6. PC component structure in embedding space ==========
        print(f"\n--- 6. Top dimensions in each PC ---")
        for pc_idx in range(5):
            comp = pca.components_[pc_idx]
            top_dims = np.argsort(np.abs(comp))[::-1][:10]
            print(f"    PC{pc_idx}: top dims = {[(int(d), f'{comp[d]:+.4f}') for d in top_dims[:5]]}")

        # ========== 7. Is PC1 really the mean? ==========
        print(f"\n--- 7. PC1 analysis ---")
        mean_dir = combined_u.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)
        cos_pc0_mean = np.abs(np.dot(pca.components_[0], mean_dir))
        print(f"  cos(PC0, mean_direction) = {cos_pc0_mean:.4f}")

        # Language separability of PC0
        zh_pc0 = zh_proj[:, 0]
        en_pc0 = en_proj[:, 0]
        print(f"  PC0 mean: zh={zh_pc0.mean():.4f}, en={en_pc0.mean():.4f}")
        d_cohen = (zh_pc0.mean() - en_pc0.mean()) / np.sqrt((zh_pc0.std()**2 + en_pc0.std()**2) / 2)
        print(f"  Cohen's d (zh vs en on PC0): {d_cohen:.2f}")

        # Store results
        layer_key = f"L{target_layer}"
        results[layer_key] = {
            "variance_explained": [float(x) for x in pca.explained_variance_ratio_[:20]],
            "pc_category": pc_category_results,
            "cos_pc0_mean": float(cos_pc0_mean),
            "cohens_d_pc0_lang": float(d_cohen),
        }

    # Save
    outpath = OUTPUT_DIR / "pc_characterization.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
