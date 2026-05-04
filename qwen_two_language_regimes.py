"""
Two Language Regimes: What do the L8 and L32 language directions encode?

The primary zh-en difference direction is:
  - STABLE L8-L24 (cos>0.99): "early language direction"
  - COMPLETELY DIFFERENT at L32-L34 (cos=0.15-0.28): "late language direction"
  - These are nearly orthogonal — two different things the model calls "language"

Questions:
1. What problems have extreme projections on each direction?
2. How do categories distribute along each direction?
3. Can we interpret what each direction encodes?
4. The per-problem displacement vectors: how does their structure change?
5. Is the L24→L32 transition sharp or gradual? (need to look at subspace angles
   between adjacent layers, but we only have L24 and L32)
"""

import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

data = np.load('output/viz_activations.npz', allow_pickle=True)
layers = [8, 16, 24, 32, 34]
categories = data['categories']
cat_names = ['arithmetic', 'sequences', 'combinatorics', 'modular', 'geometry']
N = 200

# Compute the primary difference direction at each layer
diff_dirs = {}
diff_mats = {}
for layer in layers:
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']
    diff = zh - en
    diff_mats[layer] = diff
    U, s, Vt = svd(diff, full_matrices=False)
    diff_dirs[layer] = {
        'top1': Vt[0],       # primary diff direction
        'top5': Vt[:5],      # top-5 diff directions
        'U': U,              # left singular vectors (problem loadings)
        's': s,              # singular values
    }

# ====================================================================
# ANALYSIS 1: Problem loadings on the primary difference direction
# ====================================================================
print("=" * 70)
print("ANALYSIS 1: PROBLEM LOADINGS ON PRIMARY DIFFERENCE DIRECTION")
print("=" * 70)

for layer in [8, 32]:
    d = diff_dirs[layer]
    # U[:,0] * s[0] gives the projection of each problem's diff onto the primary direction
    loadings = d['U'][:, 0] * d['s'][0]

    print(f"\nLayer {layer} — Primary diff direction (explains "
          f"{(d['s'][0]**2 / np.sum(d['s']**2))*100:.1f}% of diff energy)")

    # Per-category statistics
    print(f"\n  Per-category distribution of loadings:")
    for cat in range(5):
        mask = categories == cat
        cat_load = loadings[mask]
        print(f"    {cat_names[cat]:15s}: mean={np.mean(cat_load):>8.2f}, "
              f"std={np.std(cat_load):>7.2f}, "
              f"range=[{np.min(cat_load):>8.2f}, {np.max(cat_load):>8.2f}]")

    # Are loadings uniform (shared tag) or variable (problem-specific)?
    cv = np.std(loadings) / np.abs(np.mean(loadings)) if np.abs(np.mean(loadings)) > 1e-10 else float('inf')
    print(f"\n  Coefficient of variation: {cv:.3f}")
    print(f"  (CV<0.5 → mostly shared, CV>1.0 → mostly problem-specific)")

    # Sign consistency: do all problems have the same sign?
    pos = np.sum(loadings > 0)
    neg = np.sum(loadings < 0)
    print(f"  Sign: {pos} positive, {neg} negative "
          f"({'consistent' if min(pos, neg) < 20 else 'mixed'})")

# ====================================================================
# ANALYSIS 2: The displacement vector field
# ====================================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 2: DISPLACEMENT VECTOR FIELD STRUCTURE")
print("=" * 70)

for layer in [8, 32]:
    diff = diff_mats[layer]

    # Normalize each displacement vector
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    diff_normed = diff / (norms + 1e-10)

    # Pairwise cosine similarity of displacement vectors
    cos_sim = diff_normed @ diff_normed.T  # (200, 200)

    # Within-category vs between-category cosine similarity
    within_cos = []
    between_cos = []
    for i in range(N):
        for j in range(i+1, N):
            if categories[i] == categories[j]:
                within_cos.append(cos_sim[i, j])
            else:
                between_cos.append(cos_sim[i, j])

    print(f"\nLayer {layer}:")
    print(f"  ||displacement|| — mean={np.mean(norms):.2f}, std={np.std(norms):.2f}")
    print(f"  Displacement direction similarity:")
    print(f"    All pairs:     mean cos = {np.mean(cos_sim[np.triu_indices(N, k=1)]):.4f}")
    print(f"    Within-cat:    mean cos = {np.mean(within_cos):.4f}")
    print(f"    Between-cat:   mean cos = {np.mean(between_cos):.4f}")
    print(f"    Gap (within - between): {np.mean(within_cos) - np.mean(between_cos):.4f}")

    # Are displacements more aligned at L32 than L8?
    overall_alignment = np.mean(cos_sim[np.triu_indices(N, k=1)])
    print(f"    → {'HIGH alignment' if overall_alignment > 0.5 else 'LOW alignment'} "
          f"(all problems displaced in {'similar' if overall_alignment > 0.5 else 'different'} directions)")

# ====================================================================
# ANALYSIS 3: Category-specific language directions
# ====================================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 3: CATEGORY-SPECIFIC LANGUAGE DIRECTIONS")
print("=" * 70)

for layer in [8, 32, 34]:
    diff = diff_mats[layer]

    # Compute mean displacement per category
    cat_means = {}
    for cat in range(5):
        mask = categories == cat
        cat_means[cat] = diff[mask].mean(0)

    # Cosine similarity between category-specific language directions
    print(f"\nLayer {layer} — Category language direction similarity:")
    header = f"{'':>15}"
    for c in range(5):
        header += f" | {cat_names[c][:8]:>8}"
    print(header)
    print("-" * (17 + 11 * 5))

    for c1 in range(5):
        row = f"  {cat_names[c1]:>13}"
        for c2 in range(5):
            cos = np.dot(cat_means[c1], cat_means[c2]) / (
                np.linalg.norm(cat_means[c1]) * np.linalg.norm(cat_means[c2]) + 1e-10)
            row += f" | {cos:>8.3f}"
        print(row)

    # How similar are category language directions to the overall mean?
    overall_mean = diff.mean(0)
    print(f"\n  Category alignment with overall language direction:")
    for cat in range(5):
        cos = np.dot(cat_means[cat], overall_mean) / (
            np.linalg.norm(cat_means[cat]) * np.linalg.norm(overall_mean) + 1e-10)
        print(f"    {cat_names[cat]:15s}: cos = {cos:.3f}")

# ====================================================================
# ANALYSIS 4: The transition — what changes between L24 and L32?
# ====================================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 4: WHAT CHANGES BETWEEN L24 AND L32?")
print("=" * 70)

for pair in [(8, 16), (16, 24), (24, 32), (32, 34)]:
    l1, l2 = pair
    d1 = diff_mats[l1]
    d2 = diff_mats[l2]

    # How much does the displacement field change?
    delta_disp = d2 - d1  # change in (zh-en) from layer l1 to l2
    mean_change = np.linalg.norm(delta_disp.mean(0))
    per_problem_change = np.mean(np.linalg.norm(delta_disp, axis=1))

    # Correlation of displacement magnitudes
    norms1 = np.linalg.norm(d1, axis=1)
    norms2 = np.linalg.norm(d2, axis=1)
    norm_corr = np.corrcoef(norms1, norms2)[0, 1]

    # Direction change
    d1_normed = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-10)
    d2_normed = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-10)
    per_problem_cos = np.sum(d1_normed * d2_normed, axis=1)

    print(f"\nL{l1} → L{l2}:")
    print(f"  Mean displacement field change: {mean_change:.2f}")
    print(f"  Per-problem displacement change: {per_problem_change:.2f}")
    print(f"  Magnitude correlation: {norm_corr:.3f}")
    print(f"  Direction similarity (per-problem cos): "
          f"mean={np.mean(per_problem_cos):.3f}, "
          f"std={np.std(per_problem_cos):.3f}")

    # Per-category direction change
    print(f"  Per-category direction change (mean cos):")
    for cat in range(5):
        mask = categories == cat
        cat_cos = np.mean(per_problem_cos[mask])
        print(f"    {cat_names[cat]:15s}: {cat_cos:.3f}")

# ====================================================================
# ANALYSIS 5: Spectral gap — is there a clear "language subspace"?
# ====================================================================
print(f"\n{'=' * 70}")
print("ANALYSIS 5: SPECTRAL GAP IN DIFFERENCE MATRIX")
print("=" * 70)

for layer in layers:
    d = diff_dirs[layer]
    s = d['s']
    # Look for gaps in the singular value spectrum
    ratios = s[:-1] / s[1:]
    # Find the largest gap
    gap_idx = np.argmax(ratios[:20])  # only look at top 20

    print(f"\nLayer {layer}:")
    print(f"  Top 10 singular values: {[f'{v:.1f}' for v in s[:10]]}")
    print(f"  Consecutive ratios: {[f'{r:.1f}' for r in ratios[:10]]}")
    print(f"  Largest spectral gap at position {gap_idx}: "
          f"s[{gap_idx}]/s[{gap_idx+1}] = {ratios[gap_idx]:.1f}")
    print(f"  This suggests {gap_idx+1} 'language dimensions' "
          f"separated from {200-gap_idx-1} 'noise dimensions'")
