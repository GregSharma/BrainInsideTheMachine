"""
Anatomy of the Additive → Rotational Transition

Key question: WHY does language encoding change type with depth?
Hypothesis: input embedding difference is additive and low-rank.
The model gradually absorbs it into a high-rank rotation for output.

Tests:
1. SVD energy concentration: what fraction of ||zh-en||² is in top-k directions?
   → Should go from concentrated (additive) to diffuse (rotational)
2. Offset subspace stability: do the top difference directions persist across layers?
   → If stable = fixed language tag. If changing = active reorganization.
3. Decomposition: for each layer, separate the zh-en difference into
   "shared offset" (rank-1 mean) vs "problem-dependent rotation residual"
4. What are the offset directions? Project onto category centroids to interpret.
5. The absorption: do L8 offset directions become L32 rotation planes?
"""

import numpy as np
from scipy.linalg import svd, subspace_angles, orthogonal_procrustes
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

data = np.load('output/viz_activations.npz', allow_pickle=True)
layers = [8, 16, 24, 32, 34]
categories = data['categories']
N = 200

# ====================================================================
# TEST 1: SVD energy concentration of (zh - en) at each layer
# ====================================================================
print("=" * 70)
print("TEST 1: SVD ENERGY CONCENTRATION OF LANGUAGE DIFFERENCE")
print("=" * 70)

diff_svd_info = {}
diff_bases = {}  # store top-k right singular vectors for subspace comparison

for layer in layers:
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']
    diff = zh - en  # (200, 2048) — per-problem language difference

    U, s, Vt = svd(diff, full_matrices=False)
    total_energy = np.sum(s**2)
    cumulative = np.cumsum(s**2) / total_energy

    # Store basis
    diff_bases[layer] = Vt  # (200, 2048) — right singular vectors

    print(f"\nLayer {layer}:")
    print(f"  Total ||zh-en||² = {total_energy:.1f}")
    print(f"  Singular value spectrum (top 10): {[f'{v:.1f}' for v in s[:10]]}")
    for k in [1, 3, 5, 10, 20, 50]:
        print(f"  Top-{k:2d} directions explain {cumulative[k-1]*100:.1f}% of difference energy")

    # Effective rank (number of components for 90% energy)
    eff_rank_90 = np.searchsorted(cumulative, 0.90) + 1
    eff_rank_95 = np.searchsorted(cumulative, 0.95) + 1
    eff_rank_99 = np.searchsorted(cumulative, 0.99) + 1

    print(f"  Effective rank (90%): {eff_rank_90}")
    print(f"  Effective rank (95%): {eff_rank_95}")
    print(f"  Effective rank (99%): {eff_rank_99}")

    diff_svd_info[layer] = {
        'total_energy': float(total_energy),
        'eff_rank_90': int(eff_rank_90),
        'eff_rank_95': int(eff_rank_95),
        'eff_rank_99': int(eff_rank_99),
        'top1_pct': float(cumulative[0] * 100),
        'top5_pct': float(cumulative[4] * 100),
        'top10_pct': float(cumulative[9] * 100),
        'top20_pct': float(cumulative[19] * 100),
    }

# ====================================================================
# TEST 2: Offset subspace stability across layers
# ====================================================================
print(f"\n{'=' * 70}")
print("TEST 2: OFFSET SUBSPACE STABILITY")
print("=" * 70)
print("\nPrincipal angles between top-10 difference subspaces (degrees):")

k_sub = 10  # compare top-10 subspaces
header = f"{'':>6}"
for l2 in layers:
    header += f" | L{l2:>3}"
print(header)
print("-" * (8 + 7 * len(layers)))

for l1 in layers:
    row = f"L{l1:>3}  "
    V1 = diff_bases[l1][:k_sub].T  # (2048, 10)
    for l2 in layers:
        V2 = diff_bases[l2][:k_sub].T  # (2048, 10)
        angles = subspace_angles(V1, V2)
        mean_angle = np.degrees(np.mean(angles))
        row += f" | {mean_angle:5.1f}"
    print(row)

# Also compute the cosine similarity between the TOP-1 direction at each layer
print(f"\nCosine similarity of TOP-1 difference direction:")
for i, l1 in enumerate(layers):
    row = f"L{l1:>3}  "
    v1 = diff_bases[l1][0]  # top singular vector
    for l2 in layers:
        v2 = diff_bases[l2][0]
        cos_sim = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        row += f" | {cos_sim:5.3f}"
    print(row)

# ====================================================================
# TEST 3: Decompose difference into mean offset + problem-specific residual
# ====================================================================
print(f"\n{'=' * 70}")
print("TEST 3: MEAN OFFSET vs PROBLEM-SPECIFIC RESIDUAL")
print("=" * 70)

for layer in layers:
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']
    diff = zh - en  # (200, 2048)

    # Mean offset (rank-1 component)
    mean_offset = diff.mean(0)  # (2048,)
    mean_offset_energy = N * np.sum(mean_offset**2)  # energy from mean offset for N problems

    # Problem-specific residual
    residual = diff - mean_offset  # (200, 2048)
    residual_energy = np.sum(residual**2)

    total_energy = np.sum(diff**2)

    # What fraction is "shared" (mean offset) vs "problem-specific" (residual)?
    mean_frac = mean_offset_energy / total_energy
    resid_frac = residual_energy / total_energy

    # SVD of residual — how concentrated is the problem-specific part?
    _, s_res, _ = svd(residual, full_matrices=False)
    cum_res = np.cumsum(s_res**2) / np.sum(s_res**2)
    eff_rank_resid = np.searchsorted(cum_res, 0.90) + 1

    print(f"\nLayer {layer}:")
    print(f"  Mean offset energy: {mean_frac*100:.1f}% of total difference")
    print(f"  Problem-specific residual: {resid_frac*100:.1f}% of total difference")
    print(f"  ||mean_offset||: {np.linalg.norm(mean_offset):.2f}")
    print(f"  Residual effective rank (90%): {eff_rank_resid}")

    # Per-category: which categories have largest problem-specific residuals?
    cat_names = ['arithmetic', 'sequences', 'combinatorics', 'modular', 'geometry']
    print(f"  Per-category residual magnitude:")
    for cat in range(5):
        mask = categories == cat
        cat_resid = np.sqrt(np.mean(np.sum(residual[mask]**2, axis=1)))
        cat_mean_offset = np.sqrt(np.sum(diff[mask].mean(0)**2))
        print(f"    {cat_names[cat]:15s}: resid_rms={cat_resid:.2f}, cat_offset={cat_mean_offset:.2f}")

# ====================================================================
# TEST 4: Do L8 offset directions become L32 rotation planes?
# ====================================================================
print(f"\n{'=' * 70}")
print("TEST 4: ABSORPTION OF OFFSET INTO ROTATION")
print("=" * 70)

# At L8, the top difference directions capture language additively.
# At L32, the Procrustes rotation captures language rotationally.
# Question: are the L8 offset directions involved in the L32 rotation?

# Compute Procrustes rotation at L32
zh32 = data['zh_L32']
en32 = data['en_L32']
zh32_c = zh32 - zh32.mean(0)
en32_c = en32 - en32.mean(0)

# PCA for Procrustes
combined32 = np.vstack([zh32_c, en32_c])
pca32 = PCA(n_components=100, random_state=42)
pca32.fit(combined32)
zh32_p = pca32.transform(zh32_c)
en32_p = pca32.transform(en32_c)
R32, _ = orthogonal_procrustes(zh32_p, en32_p)

# Eigendecomposition of R32 to find rotation planes
eigvals = np.linalg.eigvals(R32)
angles_32 = np.abs(np.angle(eigvals))

# The rotation planes with largest angles are where zh→en mapping is most "rotational"
# Get eigenvectors
eigvals_r, eigvecs_r = np.linalg.eig(R32)
# Sort by angle magnitude
angle_order = np.argsort(np.abs(np.angle(eigvals_r)))[::-1]

# Top rotation planes in PCA space → project back to original space
top_rotation_dirs_pca = eigvecs_r[:, angle_order[:20]].real  # (100, 20)
top_rotation_dirs_orig = pca32.inverse_transform(top_rotation_dirs_pca.T)  # (20, 2048)
# Normalize
for i in range(20):
    top_rotation_dirs_orig[i] /= np.linalg.norm(top_rotation_dirs_orig[i]) + 1e-10

# L8 offset directions in original space
l8_offset_dirs = diff_bases[8][:10]  # (10, 2048)

# Compute how much of L8 offset lies in L32 rotation planes
print("\nOverlap between L8 offset dirs and L32 rotation planes:")
print("(How much of each L8 offset dir is captured by top-20 L32 rotation planes)")

for i in range(5):
    l8_dir = l8_offset_dirs[i]
    # Project onto L32 rotation plane span
    projections = top_rotation_dirs_orig @ l8_dir  # (20,)
    captured = np.sum(projections**2) / np.sum(l8_dir**2)
    print(f"  L8 offset dir {i}: {captured*100:.1f}% captured by L32 rotation planes")

# Reverse: how much of L32 rotation planes are in L8 offset space?
print("\nReverse: how much of L32 rotation planes lie in L8 offset subspace?")
l8_offset_space = l8_offset_dirs[:10].T  # (2048, 10)
for i in range(5):
    rot_dir = top_rotation_dirs_orig[i]
    proj = l8_offset_space @ (l8_offset_space.T @ rot_dir)
    captured = np.sum(proj**2) / np.sum(rot_dir**2)
    angle_deg = np.degrees(np.abs(np.angle(eigvals_r[angle_order[i]])))
    print(f"  L32 rotation plane {i} (angle={angle_deg:.1f}°): {captured*100:.1f}% in L8 offset space")

# ====================================================================
# TEST 5: Additivity index — one number per layer
# ====================================================================
print(f"\n{'=' * 70}")
print("TEST 5: ADDITIVITY INDEX")
print("=" * 70)
print("\nAdditivity = fraction of difference energy in top-10 SVD directions")
print("High = additive (few directions capture language)")
print("Low = rotational (language distributed across many directions)\n")

print(f"{'Layer':>6} | {'Top-10 energy':>13} | {'Eff rank 90%':>12} | {'Mean offset %':>13} | {'Interpretation':>20}")
print("-" * 80)
for layer in layers:
    info = diff_svd_info[layer]
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']
    diff = zh - en
    mean_offset = diff.mean(0)
    mean_frac = N * np.sum(mean_offset**2) / np.sum(diff**2) * 100

    interp = "ADDITIVE" if info['top10_pct'] > 80 else ("MIXED" if info['top10_pct'] > 60 else "ROTATIONAL")
    print(f"  L{layer:>3} | {info['top10_pct']:>11.1f}% | {info['eff_rank_90']:>12} | {mean_frac:>11.1f}% | {interp:>20}")

# ====================================================================
# SYNTHESIS
# ====================================================================
print(f"\n{'=' * 70}")
print("SYNTHESIS")
print("=" * 70)
print("""
The additive→rotational transition is quantified by SVD energy concentration.
If top-10 directions capture >80% of ||zh-en||², language is additive.
If they capture <60%, language is distributed/rotational.

The mean offset (rank-1) captures the "shared language tag" — the part
of language encoding that's the same for all problems. The residual
captures problem-specific language differences — how the model encodes
the SAME math differently depending on the language.

The transition tells us: early layers carry language as a tag (embedding
artifact). Late layers carry language as a coordinate frame (output
preparation). The model doesn't "remove" language — it transforms the
encoding from additive tag to rotational frame.
""")

# Save
results = {
    'svd_concentration': {f'L{l}': diff_svd_info[l] for l in layers},
}
with open('output/transition_anatomy_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to output/transition_anatomy_results.json")
