"""Investigate the new dominant variance dimensions in clean data.

Dims 1979, 43, 1783, 728, 1446 replaced dim 318 as top variance sources.
What are they? Language dims? Content dims? Norm-correlated?
How do they relate to cross-lingual alignment?
"""

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

OUTPUT_DIR = Path("output")
data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")

target_dims = [1979, 43, 1783, 728, 1446, 1874, 1819, 132, 465]
N = 200

print("=== New Dominant Dimensions: Deep Characterization ===\n")

for dim in target_dims:
    print(f"\n{'='*50}")
    print(f"  DIM {dim}")
    print(f"{'='*50}")

    # Track across layers
    zh_means = []
    en_means = []
    var_fracs = []

    for l in range(36):
        zh = data[f"zh_L{l}"][:, dim]
        en = data[f"en_L{l}"][:, dim]
        combined = np.concatenate([zh, en])
        zh_means.append(float(zh.mean()))
        en_means.append(float(en.mean()))
        total_var = np.sum(np.var(np.vstack([data[f"zh_L{l}"], data[f"en_L{l}"]]), axis=0))
        var_fracs.append(float(np.var(combined) / total_var))

    # Language separability
    zh_35 = data["zh_L35"][:, dim]
    en_35 = data["en_L35"][:, dim]
    gap = zh_35.mean() - en_35.mean()
    pooled_std = np.sqrt((zh_35.var() + en_35.var()) / 2)
    cohen_d = gap / pooled_std if pooled_std > 0 else 0

    # Norm correlation
    zh_full = data["zh_L15"]
    norms = np.linalg.norm(zh_full, axis=1)
    r = np.corrcoef(zh_full[:, dim], norms)[0, 1]

    # Category separability (is it content-specific?)
    categories = data["categories"]
    zh_32 = data["zh_L32"][:, dim]
    cat_means = [zh_32[categories == c].mean() for c in range(5)]
    cat_stds = [zh_32[categories == c].std() for c in range(5)]

    # Where does variance fraction peak?
    peak_layer = np.argmax(var_fracs)
    peak_frac = var_fracs[peak_layer]

    # Alignment contribution at L32
    zh_u = data["zh_L32"] / np.linalg.norm(data["zh_L32"], axis=1, keepdims=True)
    en_u = data["en_L32"] / np.linalg.norm(data["en_L32"], axis=1, keepdims=True)
    matched_dim = np.mean(zh_u[:, dim] * en_u[:, dim])
    rng = np.random.RandomState(42)
    scr_dim = np.mean([np.mean(zh_u[:, dim] * en_u[rng.permutation(N), dim]) for _ in range(100)])
    align_gap = matched_dim - scr_dim

    print(f"  Language: zh_mean={zh_35.mean():+8.2f}, en_mean={en_35.mean():+8.2f}, Cohen's d={cohen_d:+.2f}")
    print(f"  Norm corr (L15): r={r:+.3f}")
    print(f"  Var fraction peak: L{peak_layer} ({peak_frac:.4f})")
    print(f"  Alignment contrib (L32): gap={align_gap:.6f}")
    print(f"  Category means (L32): {[f'{m:.1f}' for m in cat_means]}")
    print(f"  Category stds (L32):  {[f'{s:.1f}' for s in cat_stds]}")

    # Lifecycle summary
    print(f"  Lifecycle (zh_mean): L0={zh_means[0]:+.1f}, L15={zh_means[15]:+.1f}, L35={zh_means[35]:+.1f}")
    print(f"  Lifecycle (en_mean): L0={en_means[0]:+.1f}, L15={en_means[15]:+.1f}, L35={en_means[35]:+.1f}")

# Category labels
print("\n\nCategory key: 0=arithmetic, 1=combinatorics, 2=modular, 3=geometry, 4=sequences")

# Cross-lingual alignment: which dims contribute most at different layers?
print("\n\n=== Alignment contribution ranking at key layers ===")
for l in [0, 15, 32, 35]:
    zh = data[f"zh_L{l}"]
    en = data[f"en_L{l}"]
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

    matched_per_dim = np.mean(zh_u * en_u, axis=0)
    rng = np.random.RandomState(42)
    scr_per_dim = np.zeros(2048)
    for _ in range(50):
        perm = rng.permutation(N)
        scr_per_dim += np.mean(zh_u * en_u[perm], axis=0)
    scr_per_dim /= 50
    gap_per_dim = matched_per_dim - scr_per_dim

    top5 = np.argsort(np.abs(gap_per_dim))[::-1][:10]
    print(f"\n  L{l}: top 10 alignment dims: {[(int(d), f'{gap_per_dim[d]:.5f}') for d in top5]}")

    # Check if target dims appear
    for td in target_dims:
        rank = np.where(np.argsort(np.abs(gap_per_dim))[::-1] == td)[0][0]
        if rank < 50:
            print(f"    dim {td}: rank {rank+1}/2048, gap={gap_per_dim[td]:.5f}")
