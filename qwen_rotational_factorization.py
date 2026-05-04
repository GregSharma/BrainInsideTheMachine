"""
Rotational Factorization Analysis

ICA showed language and reasoning are NOT additively separable.
Hypothesis: language is encoded as a ROTATION of a shared reasoning manifold.
  h_zh = R_zh · f(problem)
  h_en = R_en · f(problem)
Procrustes finds R = R_en · R_zh^T.

Tests:
1. Rotation angle (||R - I||) should INCREASE with depth (languages rotate apart)
2. Subtractive removal (projecting out mean difference) should NOT help (not additive)
3. Rotational removal (Procrustes alignment) SHOULD help (rotational factorization)
4. The angular spectrum of R: how many rotation dimensions are needed?
5. Shared structure quality: after Procrustes, how congruent are the manifolds?
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes, svd, logm, norm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

data = np.load('output/viz_activations.npz', allow_pickle=True)
layers = [8, 16, 24, 32, 34]
categories = data['categories']
N = 200

results = {}

for layer in layers:
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']

    # Center
    zh_c = zh - zh.mean(0)
    en_c = en - en.mean(0)

    print(f"\n{'='*70}")
    print(f"LAYER {layer}")
    print(f"{'='*70}")

    # --- Test 1: Procrustes rotation analysis ---
    # Work in PCA-reduced space for numerical stability
    combined = np.vstack([zh_c, en_c])
    pca = PCA(n_components=100, random_state=42)
    pca.fit(combined)
    zh_pca = pca.transform(zh_c)
    en_pca = pca.transform(en_c)

    R, scale = orthogonal_procrustes(zh_pca, en_pca)

    # Rotation angle: ||R - I||_F measures total rotation
    rotation_magnitude = norm(R - np.eye(R.shape[0]), 'fro')
    # Normalized by sqrt(d) for comparison
    rotation_normalized = rotation_magnitude / np.sqrt(R.shape[0])

    # SVD of R to get rotation spectrum
    # For orthogonal R, singular values are all 1, but eigenvalues reveal rotation angles
    eigvals = np.linalg.eigvals(R)
    # Rotation angles from eigenvalue phases
    angles = np.abs(np.angle(eigvals))
    angles_sorted = np.sort(angles)[::-1]
    mean_angle = np.mean(angles)
    max_angle = np.max(angles)
    # How many eigenvalues have significant rotation (angle > 0.1 radians)?
    n_significant_rotations = np.sum(angles > 0.1)

    print(f"\n  Rotation Analysis:")
    print(f"    ||R - I||_F = {rotation_magnitude:.4f}")
    print(f"    ||R - I||_F / sqrt(d) = {rotation_normalized:.4f}")
    print(f"    Scale factor: {scale:.4f}")
    print(f"    Mean rotation angle: {mean_angle:.4f} rad ({np.degrees(mean_angle):.2f}°)")
    print(f"    Max rotation angle:  {max_angle:.4f} rad ({np.degrees(max_angle):.2f}°)")
    print(f"    Significant rotations (>0.1 rad): {n_significant_rotations}/100")
    print(f"    Top 5 angles (degrees): {[f'{np.degrees(a):.1f}' for a in angles_sorted[:5]]}")

    # --- Test 2: Additive vs Rotational removal ---
    # Method A: Subtractive (project out mean difference direction)
    mean_diff = (zh - en).mean(0)
    mean_diff_normed = mean_diff / (norm(mean_diff) + 1e-10)
    zh_sub = zh - np.outer(zh @ mean_diff_normed, mean_diff_normed)
    en_sub = en - np.outer(en @ mean_diff_normed, mean_diff_normed)

    nbrs_sub = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(en_sub)
    _, idx_sub = nbrs_sub.kneighbors(zh_sub)
    nn_sub = np.mean(idx_sub.flatten() == np.arange(N))

    # Method B: Project out top-k difference directions
    diffs = zh - en  # (200, 2048)
    U_diff, s_diff, Vt_diff = svd(diffs, full_matrices=False)
    for k_proj in [1, 5, 10, 20]:
        proj_dirs = Vt_diff[:k_proj].T  # (2048, k)
        # Project out these directions from both
        zh_proj = zh - zh @ proj_dirs @ proj_dirs.T
        en_proj = en - en @ proj_dirs @ proj_dirs.T

        nbrs_p = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(en_proj)
        _, idx_p = nbrs_p.kneighbors(zh_proj)
        nn_proj = np.mean(idx_p.flatten() == np.arange(N))

        if k_proj == 1:
            print(f"\n  Additive Removal (project out difference directions):")
        print(f"    Remove top-{k_proj:2d} diff dirs → NN accuracy: {nn_proj:.3f}")

    # Method C: Procrustes (rotational removal)
    zh_rot = zh_pca @ R
    nbrs_rot = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(en_pca)
    _, idx_rot = nbrs_rot.kneighbors(zh_rot)
    nn_rot = np.mean(idx_rot.flatten() == np.arange(N))

    print(f"\n  Rotational Removal (Procrustes):")
    print(f"    Procrustes-aligned NN accuracy: {nn_rot:.3f}")

    # --- Test 3: Hybrid — project out differences THEN Procrustes ---
    for k_proj in [5, 10, 20]:
        proj_dirs = Vt_diff[:k_proj].T
        zh_hybrid = zh - zh @ proj_dirs @ proj_dirs.T
        en_hybrid = en - en @ proj_dirs @ proj_dirs.T

        zh_hc = zh_hybrid - zh_hybrid.mean(0)
        en_hc = en_hybrid - en_hybrid.mean(0)

        # PCA on cleaned
        combined_h = np.vstack([zh_hc, en_hc])
        pca_h = PCA(n_components=min(50, combined_h.shape[1]), random_state=42)
        pca_h.fit(combined_h)
        zh_hp = pca_h.transform(zh_hc)
        en_hp = pca_h.transform(en_hc)

        R_h, _ = orthogonal_procrustes(zh_hp, en_hp)
        zh_hr = zh_hp @ R_h

        nbrs_h = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(en_hp)
        _, idx_h = nbrs_h.kneighbors(zh_hr)
        nn_hybrid = np.mean(idx_h.flatten() == np.arange(N))

        # R² after hybrid
        ss_res = np.sum((zh_hr - en_hp)**2)
        ss_tot = np.sum((en_hp - en_hp.mean(0))**2)
        r2_hybrid = 1 - ss_res / ss_tot

        if k_proj == 5:
            print(f"\n  Hybrid (subtract diff dirs + Procrustes):")
        print(f"    Remove {k_proj:2d} dirs + Procrustes → NN={nn_hybrid:.3f}, R²={r2_hybrid:.3f}")

    # --- Test 4: Angular spectrum — how many rotation dimensions matter? ---
    # Cumulatively apply partial rotation and measure NN improvement
    # Decompose R into rotation planes using Schur decomposition
    print(f"\n  Rotation dimensionality:")
    print(f"    Eigenvalue angle distribution:")
    angle_bins = [0, 5, 15, 30, 45, 90, 180]
    for i in range(len(angle_bins)-1):
        n_in_bin = np.sum((np.degrees(angles) >= angle_bins[i]) &
                          (np.degrees(angles) < angle_bins[i+1]))
        print(f"      {angle_bins[i]:3d}°-{angle_bins[i+1]:3d}°: {n_in_bin} eigenvalues")

    # --- Test 5: Per-problem rotation consistency ---
    # If h = R · f, then the RESIDUAL after Procrustes should be small and random
    zh_aligned = zh_pca @ R
    residuals = zh_aligned - en_pca
    per_problem_error = np.sqrt(np.sum(residuals**2, axis=1))
    signal_strength = np.sqrt(np.sum(en_pca**2, axis=1))
    relative_error = per_problem_error / signal_strength

    print(f"\n  Per-problem Procrustes residual:")
    print(f"    Mean relative error: {np.mean(relative_error):.4f}")
    print(f"    Std relative error:  {np.std(relative_error):.4f}")
    print(f"    Min: {np.min(relative_error):.4f}  Max: {np.max(relative_error):.4f}")

    # Per-category relative error
    unique_cats = np.unique(categories)
    print(f"    Per-category mean relative error:")
    cat_names = ['arithmetic', 'sequences', 'combinatorics', 'modular', 'geometry']
    for cat in unique_cats:
        mask = categories == cat
        cat_err = np.mean(relative_error[mask])
        name = cat_names[cat] if cat < len(cat_names) else f'cat_{cat}'
        print(f"      {name}: {cat_err:.4f} (n={mask.sum()})")

    results[f'L{layer}'] = {
        'rotation_magnitude': float(rotation_magnitude),
        'rotation_normalized': float(rotation_normalized),
        'mean_angle_deg': float(np.degrees(mean_angle)),
        'max_angle_deg': float(np.degrees(max_angle)),
        'n_significant_rotations': int(n_significant_rotations),
        'nn_procrustes': float(nn_rot),
        'mean_relative_error': float(np.mean(relative_error)),
    }

# --- CROSS-LAYER SUMMARY ---
print(f"\n{'='*70}")
print("CROSS-LAYER ROTATION SUMMARY")
print(f"{'='*70}")
print(f"\n{'Layer':>6} | {'||R-I||/√d':>10} | {'Mean angle':>10} | {'Max angle':>10} | {'#Sig rots':>9} | {'Proc NN':>8} | {'Rel err':>8}")
print("-" * 80)
for layer in layers:
    r = results[f'L{layer}']
    print(f"  L{layer:>3} | {r['rotation_normalized']:>10.4f} | {r['mean_angle_deg']:>9.2f}° | {r['max_angle_deg']:>9.2f}° | {r['n_significant_rotations']:>9} | {r['nn_procrustes']:>8.3f} | {r['mean_relative_error']:>8.4f}")

with open('output/rotational_factorization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to output/rotational_factorization_results.json")
