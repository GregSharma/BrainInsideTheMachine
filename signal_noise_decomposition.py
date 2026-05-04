"""Decompose the z-score paradox: gap decreases but z increases.

z = (matched - scrambled_mean) / scrambled_std

If z increases while gap decreases, scrambled_std must decrease even faster.
Why? Possible explanations:
1. Representations cluster tighter (less noise in cosine measurements)
2. Language-specific variance gets concentrated in fewer dims
3. Problem-specific signal gets refined vs diffuse noise

Let's measure all three components across layers and understand the mechanism.
"""

import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("output")
data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")

N = 200
n_layers = 36

print("=== Signal-Noise Decomposition across layers ===\n")
print(f"{'L':>3s} {'matched':>8s} {'scr_mean':>9s} {'gap':>8s} {'scr_std':>9s} {'z':>6s} {'norm_zh':>8s} {'norm_en':>8s} {'cos_spread':>10s}")

for l in range(n_layers):
    zh = data[f"zh_L{l}"]
    en = data[f"en_L{l}"]

    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

    # Matched
    matched_cos = np.sum(zh_u * en_u, axis=1)  # per-problem
    matched_mean = matched_cos.mean()
    matched_std = matched_cos.std()

    # Scrambled
    rng = np.random.RandomState(42)
    scr_means = []
    for _ in range(500):
        perm = rng.permutation(N)
        scr_means.append(np.mean(np.sum(zh_u * en_u[perm], axis=1)))
    scr_mean = np.mean(scr_means)
    scr_std = np.std(scr_means)

    gap = matched_mean - scr_mean
    z = gap / scr_std if scr_std > 0 else 0

    # Norms (pre-normalization)
    norm_zh = np.linalg.norm(zh, axis=1).mean()
    norm_en = np.linalg.norm(en, axis=1).mean()

    # Cosine spread: std of per-problem matched cosines
    print(f"{l:3d} {matched_mean:8.4f} {scr_mean:9.4f} {gap:8.4f} {scr_std:9.6f} {z:6.1f} {norm_zh:8.1f} {norm_en:8.1f} {matched_std:10.4f}")

# Detailed decomposition at key layers
print("\n=== Detailed decomposition ===\n")

for l in [0, 8, 15, 22, 35]:
    zh = data[f"zh_L{l}"]
    en = data[f"en_L{l}"]

    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

    # Per-problem matched cosine distribution
    matched_cos = np.sum(zh_u * en_u, axis=1)

    # Effective dimensionality: how many dims carry the variance?
    combined = np.vstack([zh_u, en_u])
    variances = np.var(combined, axis=0)
    sorted_var = np.sort(variances)[::-1]
    cum_var = np.cumsum(sorted_var) / sorted_var.sum()
    eff_dim = np.searchsorted(cum_var, 0.9) + 1

    # Intrinsic dimensionality (participation ratio)
    pr = (variances.sum() ** 2) / (np.sum(variances ** 2))

    # How many dims have > 1% of total variance
    n_significant = np.sum(variances / variances.sum() > 0.01)

    print(f"L{l}:")
    print(f"  Matched cosine: mean={matched_cos.mean():.4f}, std={matched_cos.std():.4f}, min={matched_cos.min():.4f}, max={matched_cos.max():.4f}")
    print(f"  Effective dim (90% var): {eff_dim}")
    print(f"  Participation ratio: {pr:.1f}")
    print(f"  Dims with >1% variance: {n_significant}")

    # What fraction of alignment gap comes from the PCA top-20?
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    pca.fit(combined)

    zh_proj = pca.transform(zh_u)
    en_proj = pca.transform(en_u)
    zh_resid = zh_u - pca.inverse_transform(zh_proj)
    en_resid = en_u - pca.inverse_transform(en_proj)

    # Gap in top-20
    zh_proj_u = zh_proj / np.linalg.norm(zh_proj, axis=1, keepdims=True)
    en_proj_u = en_proj / np.linalg.norm(en_proj, axis=1, keepdims=True)
    gap_top20 = np.mean(np.sum(zh_proj_u * en_proj_u, axis=1))

    # Gap in residual
    zh_resid_u = zh_resid / np.linalg.norm(zh_resid, axis=1, keepdims=True)
    en_resid_u = en_resid / np.linalg.norm(en_resid, axis=1, keepdims=True)
    gap_resid = np.mean(np.sum(zh_resid_u * en_resid_u, axis=1))

    rng = np.random.RandomState(42)
    scr_top20 = []
    scr_resid = []
    for _ in range(200):
        perm = rng.permutation(N)
        scr_top20.append(np.mean(np.sum(zh_proj_u * en_proj_u[perm], axis=1)))
        scr_resid.append(np.mean(np.sum(zh_resid_u * en_resid_u[perm], axis=1)))

    z_top20 = (gap_top20 - np.mean(scr_top20)) / np.std(scr_top20)
    z_resid = (gap_resid - np.mean(scr_resid)) / np.std(scr_resid)

    print(f"  Top-20 PCA: matched_cos={gap_top20:.4f}, z={z_top20:.1f}")
    print(f"  Residual (d-20): matched_cos={gap_resid:.4f}, z={z_resid:.1f}")
    print(f"  PCA var captured: {pca.explained_variance_ratio_.sum():.3f}")
    print()
