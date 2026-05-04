"""
Corrected Joint Diffusion Map — "One Tape" test done RIGHT.

Previous version stacked raw zh and en trajectories into one distance matrix.
This is wrong because:
1. zh and en vectors have a massive language offset — the kernel sees them as
   two separate clusters by construction, regardless of reasoning alignment.
2. Point-by-point fractional alignment is meaningless when token counts differ.

Correct approach:
A. Fit ridge map ω_{zh→en} on the 200 INPUT-PASS L32 activations (the same map
   that achieves 2.86% cocycle error). This is the "translator."
B. Apply ω to the zh GENERATION-TIME trajectory: zh_mapped = ω(zh_traj).
   Now zh_mapped lives in en-space — language offset removed.
C. Joint diffusion map on zh_mapped + en_traj.
D. Also: dynamic time warping (DTW) to align trajectories by content, not time.
E. Also: compare trajectory SHAPES independently (eigenvalue spectra, curvature).

If One Tape is true: zh_mapped and en_traj should OVERLAP in diffusion space.
If false: they diverge even after removing the language offset.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random as pyrandom
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- LOAD DATA ----------
print("Loading pre-computed data...")
multi = np.load('output/multilingual_activations.npz')
traj = np.load('output/h32_trajectories.npz')

zh_input = multi['zh']  # (200, 2048) — input-pass L32
en_input = multi['en']  # (200, 2048)

print(f"Input-pass activations: zh={zh_input.shape}, en={en_input.shape}")
print(f"Trajectories: {list(traj.keys())}")

# ---------- FIT RIDGE MAP ω_{zh→en} ----------
# This is the affine map that achieves 2.86% cocycle error.
# Ridge regression: en = zh @ W + b
# Use all 200 problems for fitting (same as cocycle analysis).

print("\nFitting ridge map ω_{zh→en}...")
ridge = Ridge(alpha=1.0)
ridge.fit(zh_input, en_input)
en_predicted = ridge.predict(zh_input)
r2 = r2_score(en_input, en_predicted)
print(f"  Ridge R² (zh→en, train): {r2:.4f}")

# Verify it matches cocycle quality
residual = np.linalg.norm(en_predicted - en_input) / np.linalg.norm(en_input)
print(f"  Residual norm ratio: {residual:.4f}")

# Also fit en→zh for bidirectional analysis
ridge_rev = Ridge(alpha=1.0)
ridge_rev.fit(en_input, zh_input)
r2_rev = r2_score(zh_input, ridge_rev.predict(en_input))
print(f"  Ridge R² (en→zh, train): {r2_rev:.4f}")


# ---------- DIFFUSION MAP ----------
def diffusion_map(X, n_components=3, epsilon=None, alpha=1.0):
    """Compute diffusion map embedding."""
    n = X.shape[0]
    D2 = squareform(pdist(X, 'sqeuclidean'))
    if epsilon is None:
        pairwise_dists = pdist(X, 'euclidean')
        epsilon = np.median(pairwise_dists) ** 2
    K = np.exp(-D2 / epsilon)
    q = K.sum(axis=1)
    if alpha > 0:
        q_alpha = q ** alpha
        K = K / np.outer(q_alpha, q_alpha)
    row_sums = K.sum(axis=1)
    T = K / row_sums[:, np.newaxis]
    D_sqrt = np.sqrt(row_sums)
    D_sqrt_inv = 1.0 / D_sqrt
    T_sym = (T * D_sqrt[np.newaxis, :]) * D_sqrt_inv[:, np.newaxis]
    T_sym = 0.5 * (T_sym + T_sym.T)
    n_eig = min(n_components + 1, n - 1)
    eigenvalues, eigenvectors = eigh(T_sym, subset_by_index=[n - n_eig, n - 1])
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[1:n_components+1]
    eigenvectors = eigenvectors[:, 1:n_components+1]
    coords = eigenvectors * D_sqrt_inv[:, np.newaxis]
    coords = coords * eigenvalues[np.newaxis, :]
    return coords, eigenvalues, epsilon


def compute_path_metrics(coords):
    """Compute path ratio and consecutive cosines."""
    n = coords.shape[0]
    if n < 3:
        return {'path_ratio': float('nan'), 'mean_consec_cos': float('nan')}
    steps = np.diff(coords, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    path_length = step_norms.sum()
    straight_line = np.linalg.norm(coords[-1] - coords[0])
    path_ratio = path_length / straight_line if straight_line > 1e-12 else float('inf')
    cosines = []
    for t in range(len(steps) - 1):
        d1, d2 = steps[t], steps[t + 1]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 > 1e-12 and n2 > 1e-12:
            cosines.append(float(np.dot(d1, d2) / (n1 * n2)))
    return {
        'path_ratio': float(path_ratio),
        'path_length': float(path_length),
        'straight_line': float(straight_line),
        'mean_consec_cos': float(np.mean(cosines)) if cosines else float('nan'),
        'frac_anticorrelated': float(np.mean([c < 0 for c in cosines])) if cosines else float('nan'),
        'n_steps': n,
    }


def dtw_distance(X, Y):
    """Dynamic time warping distance between two trajectories.
    X: (n, d), Y: (m, d). Returns DTW distance and alignment path."""
    n, m = X.shape[0], Y.shape[0]
    D = np.zeros((n + 1, m + 1))
    D[0, :] = np.inf
    D[:, 0] = np.inf
    D[0, 0] = 0.0
    # Cost matrix
    C = np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)  # (n, m)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i-1, j-1] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    # Backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [(D[i-1, j-1], i-1, j-1),
                       (D[i-1, j], i-1, j),
                       (D[i, j-1], i, j-1)]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()
    return D[n, m], path


# ---------- PROBLEMS ----------
def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh = f"计算 {a} + {b} 的值。"
            en = f"Calculate {a} + {b}."
        else:
            zh = f"计算 {a} × {b} 的值。"
            en = f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "category": 0})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "category": 1})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "category": 2})
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "category": 3})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        problems.append({"zh": zh, "en": en, "category": 4})
    rng.shuffle(problems)
    return problems

problems = generate_problems(200, seed=42)
CAT_NAMES = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]

# Map prob indices to categories
cat_examples = {}
for i, p in enumerate(problems):
    c = p['category']
    if c not in cat_examples:
        cat_examples[c] = i
    if len(cat_examples) == 5:
        break
test_indices = sorted(cat_examples.values())

# ---------- MAIN ANALYSIS ----------
all_results = {}

prob_keys = set()
for k in traj.keys():
    prob_keys.add(k.rsplit('_', 1)[0])  # e.g., 'prob0', 'prob1', etc.

for prob_key in sorted(prob_keys):
    zh_key = f"{prob_key}_zh"
    en_key = f"{prob_key}_en"

    if zh_key not in traj or en_key not in traj:
        continue

    h32_zh = traj[zh_key]  # (n_zh, 2048)
    h32_en = traj[en_key]  # (n_en, 2048)
    n_zh, n_en = h32_zh.shape[0], h32_en.shape[0]

    prob_idx = int(prob_key.replace('prob', ''))
    cat_name = CAT_NAMES[problems[prob_idx]['category']]

    print(f"\n{'='*70}")
    print(f"{prob_key} ({cat_name}): {n_zh} zh steps, {n_en} en steps")
    print(f"  zh: {problems[prob_idx]['zh']}")
    print(f"  en: {problems[prob_idx]['en']}")

    if n_zh < 10 or n_en < 10:
        print("  Skipping — too few steps")
        continue

    # ===== STEP 1: Apply ridge map to zh trajectory =====
    # ω_{zh→en}: maps zh generation-time h32 into en-space
    h32_zh_mapped = ridge.predict(h32_zh)  # (n_zh, 2048), now in en-space

    # Sanity check: how close are mapped zh points to actual en points?
    # (They shouldn't match point-for-point since reasoning paths differ,
    #  but the DISTRIBUTIONS should overlap if Z is shared)

    # ===== STEP 2: R^2048 metrics before and after mapping =====
    r2048_zh = compute_path_metrics(h32_zh)
    r2048_en = compute_path_metrics(h32_en)
    r2048_zh_mapped = compute_path_metrics(h32_zh_mapped)

    print(f"\n  R^2048 metrics:")
    print(f"    zh raw:    path={r2048_zh['path_ratio']:.1f}x, cos={r2048_zh['mean_consec_cos']:.3f}")
    print(f"    en raw:    path={r2048_en['path_ratio']:.1f}x, cos={r2048_en['mean_consec_cos']:.3f}")
    print(f"    zh mapped: path={r2048_zh_mapped['path_ratio']:.1f}x, cos={r2048_zh_mapped['mean_consec_cos']:.3f}")

    # ===== STEP 3: Pairwise distances between mapped-zh and en trajectories =====
    # Mean nearest-neighbor distance from mapped-zh to en
    from scipy.spatial.distance import cdist
    D_cross = cdist(h32_zh_mapped, h32_en, 'euclidean')  # (n_zh, n_en)
    nn_zh_to_en = D_cross.min(axis=1).mean()  # each zh mapped point's nearest en neighbor
    nn_en_to_zh = D_cross.min(axis=0).mean()  # each en point's nearest mapped-zh neighbor

    # Compare to within-trajectory distances
    D_en_self = pdist(h32_en, 'euclidean')
    D_zh_mapped_self = pdist(h32_zh_mapped, 'euclidean')
    within_en = np.mean(D_en_self)
    within_zh_mapped = np.mean(D_zh_mapped_self)

    print(f"\n  Cross-trajectory distances (after ridge mapping):")
    print(f"    NN zh_mapped→en: {nn_zh_to_en:.1f}")
    print(f"    NN en→zh_mapped: {nn_en_to_zh:.1f}")
    print(f"    Within en (mean pairwise): {within_en:.1f}")
    print(f"    Within zh_mapped (mean pairwise): {within_zh_mapped:.1f}")
    print(f"    NN / within_en ratio: {nn_zh_to_en / within_en:.3f}")
    print(f"    (< 0.5 = overlapping clouds; > 1.0 = fully separate)")

    # ===== STEP 4: Joint diffusion map on zh_mapped + en =====
    h32_joint = np.vstack([h32_zh_mapped, h32_en])  # (n_zh + n_en, 2048)
    n_joint = h32_joint.shape[0]

    print(f"\n  Joint diffusion map ({n_zh} mapped-zh + {n_en} en = {n_joint} points)...")
    n_comp = min(3, n_joint - 2)
    coords_joint, evals_joint, eps_joint = diffusion_map(h32_joint, n_components=n_comp)

    coords_zh = coords_joint[:n_zh]
    coords_en = coords_joint[n_zh:]

    # Metrics in diffusion space
    diff_zh = compute_path_metrics(coords_zh)
    diff_en = compute_path_metrics(coords_en)

    print(f"    Eigenvalues: {evals_joint}")
    print(f"    Diffusion zh_mapped: path={diff_zh['path_ratio']:.1f}x, cos={diff_zh['mean_consec_cos']:.3f}")
    print(f"    Diffusion en:        path={diff_en['path_ratio']:.1f}x, cos={diff_en['mean_consec_cos']:.3f}")

    # ===== STEP 5: DTW alignment =====
    print(f"\n  Dynamic Time Warping (zh_mapped vs en in R^2048)...")
    dtw_dist, dtw_path = dtw_distance(h32_zh_mapped, h32_en)

    # Normalized DTW: divide by path length
    dtw_norm = dtw_dist / (n_zh + n_en)
    # Compare to mean step size in each trajectory
    zh_step_mean = np.mean([np.linalg.norm(h32_zh_mapped[t+1] - h32_zh_mapped[t])
                            for t in range(n_zh - 1)])
    en_step_mean = np.mean([np.linalg.norm(h32_en[t+1] - h32_en[t])
                            for t in range(n_en - 1)])
    avg_step = (zh_step_mean + en_step_mean) / 2

    # DTW alignment quality: mean distance between aligned pairs
    aligned_dists = [np.linalg.norm(h32_zh_mapped[i] - h32_en[j]) for i, j in dtw_path]
    mean_aligned_dist = np.mean(aligned_dists)

    print(f"    DTW total distance: {dtw_dist:.1f}")
    print(f"    DTW normalized: {dtw_norm:.1f}")
    print(f"    Mean aligned-pair distance: {mean_aligned_dist:.1f}")
    print(f"    Mean step size: zh={zh_step_mean:.1f}, en={en_step_mean:.1f}")
    print(f"    Aligned dist / step size: {mean_aligned_dist / avg_step:.3f}")
    print(f"    (< 1.0 = aligned pairs are closer than one step; > 3.0 = unrelated)")

    # ===== STEP 6: DTW in diffusion coordinates =====
    if coords_zh.shape[1] >= 3 and coords_en.shape[1] >= 3:
        print(f"\n  DTW in diffusion coordinates...")
        dtw_dist_diff, dtw_path_diff = dtw_distance(coords_zh, coords_en)
        aligned_dists_diff = [np.linalg.norm(coords_zh[i] - coords_en[j])
                              for i, j in dtw_path_diff]
        mean_aligned_diff = np.mean(aligned_dists_diff)
        zh_diff_step = np.mean(np.linalg.norm(np.diff(coords_zh, axis=0), axis=1))
        en_diff_step = np.mean(np.linalg.norm(np.diff(coords_en, axis=0), axis=1))
        avg_diff_step = (zh_diff_step + en_diff_step) / 2
        print(f"    Mean aligned-pair distance: {mean_aligned_diff:.6f}")
        print(f"    Mean step size: zh={zh_diff_step:.6f}, en={en_diff_step:.6f}")
        print(f"    Aligned dist / step size: {mean_aligned_diff / avg_diff_step:.3f}")

    # ===== STEP 7: Also compare WITHOUT ridge map (control) =====
    # Stack raw zh + en (the wrong way) to show the difference
    h32_joint_raw = np.vstack([h32_zh, h32_en])
    coords_raw, evals_raw, eps_raw = diffusion_map(h32_joint_raw, n_components=n_comp)
    coords_zh_raw = coords_raw[:n_zh]
    coords_en_raw = coords_raw[n_zh:]

    # Mean centroid distance in diffusion coords (corrected vs raw)
    centroid_zh_corrected = coords_zh.mean(axis=0)
    centroid_en_corrected = coords_en.mean(axis=0)
    centroid_dist_corrected = np.linalg.norm(centroid_zh_corrected - centroid_en_corrected)
    spread_corrected = (np.mean(np.linalg.norm(coords_zh - centroid_zh_corrected, axis=1)) +
                        np.mean(np.linalg.norm(coords_en - centroid_en_corrected, axis=1))) / 2
    ratio_corrected = centroid_dist_corrected / spread_corrected

    centroid_zh_raw = coords_zh_raw.mean(axis=0)
    centroid_en_raw = coords_en_raw.mean(axis=0)
    centroid_dist_raw = np.linalg.norm(centroid_zh_raw - centroid_en_raw)
    spread_raw = (np.mean(np.linalg.norm(coords_zh_raw - centroid_zh_raw, axis=1)) +
                  np.mean(np.linalg.norm(coords_en_raw - centroid_en_raw, axis=1))) / 2
    ratio_raw = centroid_dist_raw / spread_raw

    print(f"\n  Alignment ratio comparison:")
    print(f"    RAW (no mapping):     centroid_dist={centroid_dist_raw:.4f}, "
          f"spread={spread_raw:.4f}, ratio={ratio_raw:.3f}")
    print(f"    CORRECTED (ridge):    centroid_dist={centroid_dist_corrected:.4f}, "
          f"spread={spread_corrected:.4f}, ratio={ratio_corrected:.3f}")
    print(f"    Improvement: {ratio_raw:.3f} → {ratio_corrected:.3f} "
          f"({'BETTER' if ratio_corrected < ratio_raw else 'WORSE'})")

    # ===== PLOT =====
    fig = plt.figure(figsize=(20, 8))

    # Left: Raw joint (wrong way)
    ax1 = fig.add_subplot(131, projection='3d')
    if coords_zh_raw.shape[1] >= 3:
        ax1.scatter(coords_zh_raw[:, 0], coords_zh_raw[:, 1], coords_zh_raw[:, 2],
                   c=np.linspace(0, 1, n_zh), cmap='Reds', s=10, alpha=0.6)
        ax1.scatter(coords_en_raw[:, 0], coords_en_raw[:, 1], coords_en_raw[:, 2],
                   c=np.linspace(0, 1, n_en), cmap='Blues', s=10, alpha=0.6)
        ax1.plot(coords_zh_raw[:, 0], coords_zh_raw[:, 1], coords_zh_raw[:, 2],
                'r-', alpha=0.2, linewidth=0.5)
        ax1.plot(coords_en_raw[:, 0], coords_en_raw[:, 1], coords_en_raw[:, 2],
                'b-', alpha=0.2, linewidth=0.5)
    ax1.set_title(f'RAW (no mapping)\nratio={ratio_raw:.3f}')
    ax1.set_xlabel('φ₁'); ax1.set_ylabel('φ₂'); ax1.set_zlabel('φ₃')

    # Middle: Corrected joint (with ridge map)
    ax2 = fig.add_subplot(132, projection='3d')
    if coords_zh.shape[1] >= 3:
        ax2.scatter(coords_zh[:, 0], coords_zh[:, 1], coords_zh[:, 2],
                   c=np.linspace(0, 1, n_zh), cmap='Reds', s=10, alpha=0.6)
        ax2.scatter(coords_en[:, 0], coords_en[:, 1], coords_en[:, 2],
                   c=np.linspace(0, 1, n_en), cmap='Blues', s=10, alpha=0.6)
        ax2.plot(coords_zh[:, 0], coords_zh[:, 1], coords_zh[:, 2],
                'r-', alpha=0.2, linewidth=0.5)
        ax2.plot(coords_en[:, 0], coords_en[:, 1], coords_en[:, 2],
                'b-', alpha=0.2, linewidth=0.5)
    ax2.set_title(f'CORRECTED (ridge ω_{{zh→en}})\nratio={ratio_corrected:.3f}')
    ax2.set_xlabel('φ₁'); ax2.set_ylabel('φ₂'); ax2.set_zlabel('φ₃')

    # Right: DTW alignment visualization (in R^2048, projected to 2D PCA)
    ax3 = fig.add_subplot(133)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    both = np.vstack([h32_zh_mapped, h32_en])
    both_2d = pca.fit_transform(both)
    zh_2d = both_2d[:n_zh]
    en_2d = both_2d[n_zh:]

    ax3.plot(zh_2d[:, 0], zh_2d[:, 1], 'r-', alpha=0.3, linewidth=0.5)
    ax3.plot(en_2d[:, 0], en_2d[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    ax3.scatter(zh_2d[:, 0], zh_2d[:, 1], c=np.linspace(0, 1, n_zh),
               cmap='Reds', s=8, alpha=0.6)
    ax3.scatter(en_2d[:, 0], en_2d[:, 1], c=np.linspace(0, 1, n_en),
               cmap='Blues', s=8, alpha=0.6)

    # Draw DTW alignment lines for a subset
    n_draw = min(30, len(dtw_path))
    step = max(1, len(dtw_path) // n_draw)
    for idx in range(0, len(dtw_path), step):
        i, j = dtw_path[idx]
        ax3.plot([zh_2d[i, 0], en_2d[j, 0]], [zh_2d[i, 1], en_2d[j, 1]],
                'g-', alpha=0.15, linewidth=0.5)

    ax3.set_title(f'PCA + DTW alignment\n'
                  f'aligned_dist/step={mean_aligned_dist/avg_step:.2f}')
    ax3.set_xlabel('PC1'); ax3.set_ylabel('PC2')

    fig.suptitle(f"{prob_key} ({cat_name}) — Corrected One Tape Test\n"
                 f"Red=Chinese (mapped), Blue=English", fontsize=14)
    fig.tight_layout()
    fig.savefig(f'output/diffusion_corrected_{prob_key}.png', dpi=150)
    plt.close(fig)
    print(f"\n  Saved: output/diffusion_corrected_{prob_key}.png")

    # Store results
    all_results[prob_key] = {
        'category': cat_name,
        'n_zh': n_zh,
        'n_en': n_en,
        'ridge_r2': float(r2),
        'r2048': {
            'zh_raw': r2048_zh,
            'en_raw': r2048_en,
            'zh_mapped': r2048_zh_mapped,
        },
        'cross_trajectory': {
            'nn_zh_to_en': float(nn_zh_to_en),
            'nn_en_to_zh': float(nn_en_to_zh),
            'within_en': float(within_en),
            'within_zh_mapped': float(within_zh_mapped),
            'nn_ratio': float(nn_zh_to_en / within_en),
        },
        'diffusion_joint': {
            'eigenvalues': evals_joint.tolist(),
            'epsilon': float(eps_joint),
            'ratio_raw': float(ratio_raw),
            'ratio_corrected': float(ratio_corrected),
        },
        'dtw_r2048': {
            'total_distance': float(dtw_dist),
            'normalized': float(dtw_norm),
            'mean_aligned_dist': float(mean_aligned_dist),
            'aligned_dist_over_step': float(mean_aligned_dist / avg_step),
        },
    }


# ---------- SUMMARY ----------
print("\n" + "=" * 90)
print("CORRECTED ONE TAPE TEST — SUMMARY")
print("=" * 90)

for prob_key, r in sorted(all_results.items()):
    print(f"\n  {prob_key} ({r['category']}):")
    print(f"    Ridge R² = {r['ridge_r2']:.4f}")
    print(f"    NN ratio (zh_mapped→en / within_en): {r['cross_trajectory']['nn_ratio']:.3f}")
    print(f"    Diffusion alignment: RAW={r['diffusion_joint']['ratio_raw']:.3f} → "
          f"CORRECTED={r['diffusion_joint']['ratio_corrected']:.3f}")
    print(f"    DTW aligned_dist/step: {r['dtw_r2048']['aligned_dist_over_step']:.3f}")

print("\n" + "=" * 90)
print("INTERPRETATION:")
print("  NN ratio < 0.5: trajectories overlap (One Tape supported)")
print("  NN ratio > 1.0: trajectories fully separate (One Tape rejected)")
print("  DTW dist/step < 1.0: aligned points within one step (strong alignment)")
print("  DTW dist/step > 3.0: aligned points far apart (weak alignment)")
print("  Diffusion ratio: CORRECTED < RAW means ridge map helps collapse languages")
print("=" * 90)

with open('output/diffusion_corrected_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: output/diffusion_corrected_results.json")
