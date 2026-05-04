"""Anatomy of monotonic alignment accumulation.

Questions:
1. Is alignment concentrated in a stable set of dimensions, or does the cast rotate?
2. Does the PCA subspace transfer across layers? (fit L15, test L32)
3. What do the top PCs look like — are they shared across layers?
4. Per-dimension alignment contribution: which dims carry the zh-en signal?
5. Cumulative variance curve: how many PCs needed at each layer?
"""

import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

OUTPUT_DIR = Path("output")


def main():
    print("Loading clean last-token data...")
    data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")

    N = 200
    d = 2048
    n_layers = 36

    results = {}

    # ========== 1. Per-dimension alignment contribution ==========
    print("\n=== 1. Per-dimension alignment contribution ===")
    # For each dimension, compute how much it contributes to matched-vs-scrambled
    # = mean(zh_i[d] * en_i[d]) - mean(zh_i[d] * en_j[d])
    # On unit-normalized data

    key_layers = [0, 8, 15, 22, 29, 35]
    for l in key_layers:
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]

        # Unit normalize
        zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
        en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

        # Per-dim matched cosine contribution
        matched_per_dim = np.mean(zh_u * en_u, axis=0)  # shape (2048,)

        # Per-dim scrambled (average over permutations)
        rng = np.random.RandomState(42)
        scrambled_per_dim = np.zeros(d)
        n_perm = 100
        for _ in range(n_perm):
            perm = rng.permutation(N)
            scrambled_per_dim += np.mean(zh_u * en_u[perm], axis=0)
        scrambled_per_dim /= n_perm

        # Gap per dim
        gap_per_dim = matched_per_dim - scrambled_per_dim

        # Top contributing dims
        top_pos = np.argsort(gap_per_dim)[::-1][:10]
        top_neg = np.argsort(gap_per_dim)[:10]

        total_gap = gap_per_dim.sum()
        top10_gap = gap_per_dim[top_pos].sum()

        print(f"\n  L{l}: total gap={total_gap:.4f}, top10 contribute {top10_gap/total_gap:.1%}")
        print(f"    Top 10 dims: {[(int(d), f'{gap_per_dim[d]:.5f}') for d in top_pos]}")

        # How many dims needed for 50%, 80%, 90% of gap?
        sorted_gaps = np.sort(np.abs(gap_per_dim))[::-1]
        cumgap = np.cumsum(sorted_gaps) / np.abs(gap_per_dim).sum()
        for frac in [0.5, 0.8, 0.9]:
            n_needed = np.searchsorted(cumgap, frac) + 1
            print(f"    Dims for {frac:.0%} of gap: {n_needed}")

        results[f"L{l}_top10_dims"] = [int(d) for d in top_pos]
        results[f"L{l}_top10_gap_frac"] = float(top10_gap / total_gap)

    # ========== 2. Dim stability across layers ==========
    print("\n=== 2. Are the same dims important across layers? ===")
    # Overlap of top-50 alignment dims between consecutive layers
    all_top50 = {}
    for l in range(n_layers):
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]
        zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
        en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
        matched_per_dim = np.mean(zh_u * en_u, axis=0)
        rng = np.random.RandomState(42)
        scrambled_per_dim = np.zeros(d)
        for _ in range(50):
            perm = rng.permutation(N)
            scrambled_per_dim += np.mean(zh_u * en_u[perm], axis=0)
        scrambled_per_dim /= 50
        gap = matched_per_dim - scrambled_per_dim
        all_top50[l] = set(np.argsort(np.abs(gap))[::-1][:50])

    # Overlap matrix (adjacent layers)
    print("\n  Adjacent layer overlap (top-50 dims):")
    for l in range(n_layers - 1):
        overlap = len(all_top50[l] & all_top50[l+1])
        if l % 5 == 0 or overlap < 30:
            print(f"    L{l}→L{l+1}: {overlap}/50")

    # Overlap with L35 (final)
    print("\n  Overlap with L35:")
    for l in [0, 5, 10, 15, 20, 25, 30]:
        overlap = len(all_top50[l] & all_top50[35])
        print(f"    L{l}: {overlap}/50")

    # ========== 3. PCA subspace transfer across layers ==========
    print("\n=== 3. PCA subspace transfer across layers ===")
    # Fit PCA at L15, project L32 data through it
    for fit_l, test_l in [(15, 32), (15, 35), (32, 15), (0, 35), (35, 0)]:
        zh_fit = data[f"zh_L{fit_l}"]
        en_fit = data[f"en_L{fit_l}"]
        combined_fit = np.vstack([zh_fit, en_fit])
        combined_fit_u = combined_fit / np.linalg.norm(combined_fit, axis=1, keepdims=True)

        pca = PCA(n_components=20)
        pca.fit(combined_fit_u)

        zh_test = data[f"zh_L{test_l}"]
        en_test = data[f"en_L{test_l}"]
        zh_test_u = zh_test / np.linalg.norm(zh_test, axis=1, keepdims=True)
        en_test_u = en_test / np.linalg.norm(en_test, axis=1, keepdims=True)

        zh_proj = pca.transform(zh_test_u)
        en_proj = pca.transform(en_test_u)

        zh_proj_u = zh_proj / np.linalg.norm(zh_proj, axis=1, keepdims=True)
        en_proj_u = en_proj / np.linalg.norm(en_proj, axis=1, keepdims=True)

        matched = np.mean(np.sum(zh_proj_u * en_proj_u, axis=1))
        rng = np.random.RandomState(42)
        scr_vals = []
        for _ in range(1000):
            perm = rng.permutation(N)
            scr_vals.append(np.mean(np.sum(zh_proj_u * en_proj_u[perm], axis=1)))
        scr_vals = np.array(scr_vals)
        z = (matched - scr_vals.mean()) / scr_vals.std()

        # Also measure variance captured
        zh_var = np.sum(pca.transform(zh_test_u) ** 2) / np.sum(zh_test_u ** 2)

        print(f"  Fit L{fit_l} → Test L{test_l}: z={z:.1f}, var_captured={zh_var:.3f}")

    # ========== 4. Alignment rate of change ==========
    print("\n=== 4. Alignment rate of change (Δz per layer) ===")
    z_scores = []
    for l in range(n_layers):
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]
        zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
        en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

        matched = np.mean(np.sum(zh_u * en_u, axis=1))
        rng = np.random.RandomState(42)
        scr = []
        for _ in range(500):
            perm = rng.permutation(N)
            scr.append(np.mean(np.sum(zh_u * en_u[perm], axis=1)))
        scr = np.array(scr)
        z = (matched - scr.mean()) / scr.std()
        z_scores.append(z)

    print("  Layer-by-layer z-scores and deltas:")
    for l in range(n_layers):
        delta = z_scores[l] - z_scores[l-1] if l > 0 else 0
        bar = "█" * int(z_scores[l])
        print(f"  L{l:2d}: z={z_scores[l]:5.1f} Δ={delta:+5.1f} {bar}")

    results["z_scores_by_layer"] = [float(z) for z in z_scores]

    # ========== 5. Cosine gap decomposition ==========
    print("\n=== 5. Cosine gap (matched - scrambled) across layers ===")
    for l in range(n_layers):
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]
        zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
        en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

        matched = np.mean(np.sum(zh_u * en_u, axis=1))
        rng = np.random.RandomState(42)
        perm = rng.permutation(N)
        scrambled = np.mean(np.sum(zh_u * en_u[perm], axis=1))
        gap = matched - scrambled
        print(f"  L{l:2d}: matched={matched:.4f}, scrambled={scrambled:.4f}, gap={gap:.4f}")

    # Save
    outpath = OUTPUT_DIR / "alignment_anatomy.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
