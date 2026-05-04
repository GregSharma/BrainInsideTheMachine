"""Test whether dim 1874 (the new alignment champion) masks deeper alignment.

Like the dim 318 analysis but on clean data. Does removing the top norm-correlated
dims reveal stronger alignment in the residual?
"""

import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("output")
data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")

N = 200


def z_score(zh, en, n_perms=500):
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    rng = np.random.RandomState(42)
    scr = [np.mean(np.sum(zh_u * en_u[rng.permutation(N)], axis=1)) for _ in range(n_perms)]
    scr = np.array(scr)
    return (matched - np.mean(scr)) / np.std(scr), matched, np.mean(scr)


# Test at key layers with progressively more dims removed
print("=== Effect of removing norm-correlated dims on alignment ===\n")

# Find norm-correlated dims at L15
zh_15 = data["zh_L15"]
norms = np.linalg.norm(zh_15, axis=1)
norm_corr = np.array([np.corrcoef(zh_15[:, d], norms)[0, 1] for d in range(2048)])
top_norm_dims = np.argsort(np.abs(norm_corr))[::-1]

print(f"Top 20 norm-correlated dims at L15:")
for i in range(20):
    d = top_norm_dims[i]
    print(f"  dim {d}: r={norm_corr[d]:+.3f}")

# Progressive masking
print(f"\n{'Dims removed':>15s} {'L0 z':>8s} {'L15 z':>8s} {'L32 z':>8s} {'L35 z':>8s}")

for n_remove in [0, 1, 5, 10, 20, 50, 100, 200]:
    mask = np.ones(2048, dtype=bool)
    if n_remove > 0:
        mask[top_norm_dims[:n_remove]] = False

    zs = []
    for l in [0, 15, 32, 35]:
        zh = data[f"zh_L{l}"][:, mask]
        en = data[f"en_L{l}"][:, mask]
        z, _, _ = z_score(zh, en, n_perms=300)
        zs.append(z)

    label = f"top {n_remove}" if n_remove > 0 else "none"
    print(f"{label:>15s} {zs[0]:8.1f} {zs[1]:8.1f} {zs[2]:8.1f} {zs[3]:8.1f}")

# Also try removing language dims instead
print(f"\n=== Effect of removing language dims ===")
zh_35 = data["zh_L35"]
en_35 = data["en_L35"]
lang_gap = zh_35.mean(axis=0) - en_35.mean(axis=0)
top_lang_dims = np.argsort(np.abs(lang_gap))[::-1]

print(f"Top 10 language dims at L35:")
for i in range(10):
    d = top_lang_dims[i]
    print(f"  dim {d}: gap={lang_gap[d]:+.1f}")

print(f"\n{'Lang dims removed':>20s} {'L0 z':>8s} {'L15 z':>8s} {'L32 z':>8s} {'L35 z':>8s}")

for n_remove in [0, 5, 10, 20, 50, 100]:
    mask = np.ones(2048, dtype=bool)
    if n_remove > 0:
        mask[top_lang_dims[:n_remove]] = False

    zs = []
    for l in [0, 15, 32, 35]:
        zh = data[f"zh_L{l}"][:, mask]
        en = data[f"en_L{l}"][:, mask]
        z, _, _ = z_score(zh, en, n_perms=300)
        zs.append(z)

    label = f"top {n_remove}" if n_remove > 0 else "none"
    print(f"{label:>20s} {zs[0]:8.1f} {zs[1]:8.1f} {zs[2]:8.1f} {zs[3]:8.1f}")

# The real question: what about removing BOTH norm and language dims?
print(f"\n=== Removing BOTH norm and language dims ===")
for n_norm, n_lang in [(20, 20), (50, 50), (100, 100)]:
    mask = np.ones(2048, dtype=bool)
    mask[top_norm_dims[:n_norm]] = False
    mask[top_lang_dims[:n_lang]] = False
    remaining = mask.sum()

    zs = []
    for l in [0, 15, 32, 35]:
        zh = data[f"zh_L{l}"][:, mask]
        en = data[f"en_L{l}"][:, mask]
        z, _, _ = z_score(zh, en, n_perms=300)
        zs.append(z)

    print(f"  norm={n_norm}, lang={n_lang} ({remaining} dims left): L0 z={zs[0]:.1f}, L15 z={zs[1]:.1f}, L32 z={zs[2]:.1f}, L35 z={zs[3]:.1f}")
