"""
Is the L8 difference matrix a tensor product: categories ⊗ language_directions?

If yes, then:
  diff_i = category_embedding(cat_i) ⊗ language_direction(cat_i)
  and the SVD left singular vectors should cleanly separate categories.

Also: investigate the combinatorics outlier (loading=332, 5x everything else).
"""

import numpy as np
from scipy.linalg import svd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

data = np.load('output/viz_activations.npz', allow_pickle=True)
categories = data['categories']
cat_names = ['arithmetic', 'sequences', 'combinatorics', 'modular', 'geometry']
N = 200

# ====================================================================
# PART 1: Do SVD left singular vectors correspond to categories?
# ====================================================================
print("=" * 70)
print("PART 1: SVD LEFT SINGULAR VECTORS vs CATEGORIES")
print("=" * 70)

for layer in [8, 32]:
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']
    diff = zh - en
    U, s, Vt = svd(diff, full_matrices=False)

    print(f"\nLayer {layer}:")
    print(f"  Top 8 singular values: {[f'{v:.1f}' for v in s[:8]]}")

    # U columns are the "problem loadings" on each SVD direction
    # If SVD direction k corresponds to category c, then U[:,k] should be
    # large only for problems in category c.

    print(f"\n  Loadings of top-6 SVD directions by category:")
    print(f"  {'Category':>15}", end="")
    for k in range(6):
        print(f" | SV{k:d} (s={s[k]:.0f})", end="")
    print()
    print("  " + "-" * 95)

    for cat in range(5):
        mask = categories == cat
        row = f"  {cat_names[cat]:>15}"
        for k in range(6):
            loadings = U[mask, k] * s[k]
            row += f" | {np.mean(loadings):>7.1f}±{np.std(loadings):>5.1f}"
        print(row)

    # K-means on top-5 U columns — does it recover categories?
    U_top5 = U[:, :5] * s[:5]  # scale by singular values
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    pred = kmeans.fit_predict(U_top5)
    ari = adjusted_rand_score(categories, pred)
    nmi = normalized_mutual_info_score(categories, pred)
    print(f"\n  K-means on top-5 SVD loadings → ARI={ari:.3f}, NMI={nmi:.3f}")
    print(f"  (ARI=1.0 means perfect category recovery)")

    # Which SVD direction best separates each category?
    print(f"\n  Best SVD direction per category (by |mean loading| / std):")
    for cat in range(5):
        mask = categories == cat
        best_k = -1
        best_score = -1
        for k in range(6):
            in_cat = np.abs(np.mean(U[mask, k] * s[k]))
            out_cat = np.abs(np.mean(U[~mask, k] * s[k]))
            score = (in_cat - out_cat) / (np.std(U[:, k] * s[k]) + 1e-10)
            if score > best_score:
                best_score = score
                best_k = k
        print(f"    {cat_names[cat]:>15}: SV{best_k} (separation score={best_score:.2f})")

# ====================================================================
# PART 2: The combinatorics outlier
# ====================================================================
print(f"\n{'=' * 70}")
print("PART 2: THE COMBINATORICS OUTLIER")
print("=" * 70)

zh8 = data['zh_L8']
en8 = data['en_L8']
diff8 = zh8 - en8
_, s8, Vt8 = svd(diff8, full_matrices=False)

# Find the outlier
combo_mask = categories == 2  # combinatorics
combo_indices = np.where(combo_mask)[0]

# Projection onto SV0
proj_sv0 = diff8 @ Vt8[0]  # (200,) — projection of each problem's diff onto top direction
combo_projs = proj_sv0[combo_mask]

# Find the extreme one
outlier_local = np.argmax(np.abs(combo_projs))
outlier_global = combo_indices[outlier_local]

print(f"\nProjection onto SV0 (top difference direction):")
print(f"  Overall: mean={np.mean(proj_sv0):.2f}, std={np.std(proj_sv0):.2f}")
print(f"  Combinatorics: mean={np.mean(combo_projs):.2f}, std={np.std(combo_projs):.2f}")
print(f"  Outlier: problem index {outlier_global}, projection = {combo_projs[outlier_local]:.2f}")
print(f"  Next largest: {np.sort(np.abs(combo_projs))[-2]:.2f}")

# How much of total difference energy is this ONE problem?
outlier_diff = diff8[outlier_global]
outlier_energy = np.sum(outlier_diff**2)
total_energy = np.sum(diff8**2)
print(f"\n  Outlier ||zh-en||² = {outlier_energy:.1f} ({outlier_energy/total_energy*100:.1f}% of total!)")
print(f"  ||zh-en|| = {np.linalg.norm(outlier_diff):.2f}")
print(f"  Mean ||zh-en|| for other problems: {np.mean(np.linalg.norm(diff8[np.arange(N) != outlier_global], axis=1)):.2f}")

# Does this outlier persist across layers?
print(f"\n  Outlier (problem {outlier_global}) displacement norm across layers:")
for layer in layers:
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']
    d = zh[outlier_global] - en[outlier_global]
    all_norms = np.linalg.norm(zh - en, axis=1)
    percentile = np.searchsorted(np.sort(all_norms), np.linalg.norm(d)) / N * 100
    print(f"    L{layer}: ||diff|| = {np.linalg.norm(d):.2f}, "
          f"percentile = {percentile:.0f}th, "
          f"ratio to median = {np.linalg.norm(d) / np.median(all_norms):.1f}x")

# ====================================================================
# PART 3: Rerun key analysis WITHOUT the outlier
# ====================================================================
print(f"\n{'=' * 70}")
print(f"PART 3: ANALYSIS WITHOUT OUTLIER (problem {outlier_global})")
print("=" * 70)

clean_mask = np.arange(N) != outlier_global

for layer in [8, 32]:
    zh = data[f'zh_L{layer}'][clean_mask]
    en = data[f'en_L{layer}'][clean_mask]
    diff = zh - en
    U, s, Vt = svd(diff, full_matrices=False)

    total_e = np.sum(s**2)
    cum = np.cumsum(s**2) / total_e

    # Mean offset fraction
    mean_off = diff.mean(0)
    mean_frac = (N-1) * np.sum(mean_off**2) / total_e

    print(f"\nLayer {layer} (N={N-1}):")
    print(f"  Top 8 singular values: {[f'{v:.1f}' for v in s[:8]]}")
    print(f"  Top-1 explains {cum[0]*100:.1f}%, Top-5 explains {cum[4]*100:.1f}%")
    print(f"  Mean offset fraction: {mean_frac*100:.1f}%")

    # Spectral gap
    ratios = s[:-1] / s[1:]
    gap_idx = np.argmax(ratios[:15])
    print(f"  Largest spectral gap at position {gap_idx}: ratio={ratios[gap_idx]:.1f}")

    # Within/between category displacement similarity
    cats_clean = categories[clean_mask]
    norms_d = np.linalg.norm(diff, axis=1, keepdims=True)
    diff_n = diff / (norms_d + 1e-10)
    cos_mat = diff_n @ diff_n.T

    within, between = [], []
    for i in range(len(cats_clean)):
        for j in range(i+1, len(cats_clean)):
            if cats_clean[i] == cats_clean[j]:
                within.append(cos_mat[i, j])
            else:
                between.append(cos_mat[i, j])
    print(f"  Within-cat cos: {np.mean(within):.4f}")
    print(f"  Between-cat cos: {np.mean(between):.4f}")

    # K-means recovery
    U5 = U[:, :5] * s[:5]
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    pred = km.fit_predict(U5)
    ari = adjusted_rand_score(cats_clean, pred)
    print(f"  K-means ARI on top-5 SVD: {ari:.3f}")

# ====================================================================
# PART 4: Rank-5 tensor product reconstruction
# ====================================================================
print(f"\n{'=' * 70}")
print("PART 4: RANK-5 TENSOR PRODUCT RECONSTRUCTION")
print("=" * 70)

for layer in [8, 32]:
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']
    diff = zh - en

    # Category-mean difference vectors (5 vectors of dim 2048)
    cat_means = np.array([diff[categories == c].mean(0) for c in range(5)])  # (5, 2048)

    # Reconstruct each problem's diff using its category mean
    diff_reconstructed = cat_means[categories]  # (200, 2048) — each problem gets its category mean

    # How good is the reconstruction?
    residual = diff - diff_reconstructed
    recon_r2 = 1 - np.sum(residual**2) / np.sum((diff - diff.mean(0))**2)
    recon_frac = 1 - np.sum(residual**2) / np.sum(diff**2)

    print(f"\nLayer {layer}:")
    print(f"  Category-mean reconstruction:")
    print(f"    Fraction of diff energy explained: {recon_frac*100:.1f}%")
    print(f"    R² (vs global mean): {recon_r2:.3f}")
    print(f"    Per-category residual RMS:")
    for cat in range(5):
        mask = categories == cat
        cat_res = np.sqrt(np.mean(np.sum(residual[mask]**2, axis=1)))
        cat_total = np.sqrt(np.mean(np.sum(diff[mask]**2, axis=1)))
        print(f"      {cat_names[cat]:>15}: residual={cat_res:.2f}, "
              f"total={cat_total:.2f}, "
              f"explained={1-cat_res**2/cat_total**2:.1%}")
