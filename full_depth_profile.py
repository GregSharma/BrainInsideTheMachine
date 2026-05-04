"""Full depth profile: every metric at every layer (0-35).

Find the exact transition layer for:
1. Procrustes R² (should show sigmoid)
2. Mean offset fraction (should jump)
3. Top-1 direction stability (should break)
4. Within-category vs between-category displacement cos
5. Category-mean reconstruction quality
6. Displacement field direction change between consecutive layers
"""

import numpy as np
from scipy.linalg import svd, orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import json

data = np.load('output/all_layers.npz', allow_pickle=True)
categories = data['categories']
# Category names from visualize.py: 0=arith, 1=combo, 2=modular, 3=geom, 4=seq
cat_names_correct = ['arithmetic', 'combinatorics', 'modular', 'geometry', 'sequences']
N = 200
n_layers = 36

results = {}

print(f"{'L':>3} | {'ProcR2':>7} | {'ProcNN':>7} | {'RawNN':>6} | {'MeanOff%':>8} | "
      f"{'WCcos':>6} | {'BCcos':>6} | {'CatRecon%':>9} | {'TopDir_cos':>10} | {'DirChg':>6}")
print("-" * 110)

prev_top_dir = None

for layer in range(n_layers):
    zh = data[f'zh_L{layer}']
    en = data[f'en_L{layer}']
    diff = zh - en

    # 1. Procrustes R² and NN
    zh_c = zh - zh.mean(0)
    en_c = en - en.mean(0)
    combined = np.vstack([zh_c, en_c])
    pca = PCA(n_components=min(100, N), random_state=42)
    pca.fit(combined)
    zh_p = pca.transform(zh_c)
    en_p = pca.transform(en_c)
    R, _ = orthogonal_procrustes(zh_p, en_p)
    zh_rot = zh_p @ R
    ss_res = np.sum((zh_rot - en_p)**2)
    ss_tot = np.sum((en_p - en_p.mean(0))**2)
    proc_r2 = 1 - ss_res / ss_tot

    nbrs_proc = NearestNeighbors(n_neighbors=1).fit(en_p)
    _, idx_proc = nbrs_proc.kneighbors(zh_rot)
    proc_nn = np.mean(idx_proc.flatten() == np.arange(N))

    # Raw NN
    nbrs_raw = NearestNeighbors(n_neighbors=1).fit(en)
    _, idx_raw = nbrs_raw.kneighbors(zh)
    raw_nn = np.mean(idx_raw.flatten() == np.arange(N))

    # 2. Mean offset fraction
    mean_off = diff.mean(0)
    mean_frac = N * np.sum(mean_off**2) / np.sum(diff**2) if np.sum(diff**2) > 0 else 0

    # 3. Top-1 difference direction
    _, s_diff, Vt_diff = svd(diff, full_matrices=False)
    top_dir = Vt_diff[0]

    if prev_top_dir is not None:
        top_dir_cos = abs(np.dot(top_dir, prev_top_dir))
    else:
        top_dir_cos = 1.0
    prev_top_dir = top_dir

    # 4. Within-category vs between-category displacement cosine
    norms_d = np.linalg.norm(diff, axis=1, keepdims=True)
    diff_n = diff / (norms_d + 1e-10)
    cos_mat = diff_n @ diff_n.T

    within, between = [], []
    for i in range(N):
        for j in range(i+1, N):
            if categories[i] == categories[j]:
                within.append(cos_mat[i, j])
            else:
                between.append(cos_mat[i, j])
    wc_cos = np.mean(within)
    bc_cos = np.mean(between)

    # 5. Category-mean reconstruction
    cat_means = np.array([diff[categories == c].mean(0) for c in range(5)])
    diff_recon = cat_means[categories]
    residual = diff - diff_recon
    cat_recon = (1 - np.sum(residual**2) / np.sum(diff**2)) * 100 if np.sum(diff**2) > 0 else 0

    # 6. Direction change from previous layer
    if layer > 0:
        prev_diff = data[f'zh_L{layer-1}'] - data[f'en_L{layer-1}']
        prev_n = prev_diff / (np.linalg.norm(prev_diff, axis=1, keepdims=True) + 1e-10)
        curr_n = diff / (norms_d + 1e-10)
        dir_change = np.mean(np.sum(prev_n * curr_n, axis=1))
    else:
        dir_change = 1.0

    print(f"{layer:>3} | {proc_r2:>7.3f} | {proc_nn:>7.3f} | {raw_nn:>6.3f} | "
          f"{mean_frac*100:>7.1f}% | {wc_cos:>6.3f} | {bc_cos:>6.3f} | "
          f"{cat_recon:>8.1f}% | {top_dir_cos:>10.3f} | {dir_change:>6.3f}")

    results[layer] = {
        'proc_r2': float(proc_r2),
        'proc_nn': float(proc_nn),
        'raw_nn': float(raw_nn),
        'mean_offset_pct': float(mean_frac * 100),
        'within_cat_cos': float(wc_cos),
        'between_cat_cos': float(bc_cos),
        'cat_recon_pct': float(cat_recon),
        'top_dir_cos_prev': float(top_dir_cos),
        'dir_change_cos': float(dir_change),
    }

# Save
with open('output/full_depth_profile.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary: find transition layer
proc_r2s = [results[l]['proc_r2'] for l in range(n_layers)]
mean_offs = [results[l]['mean_offset_pct'] for l in range(n_layers)]
dir_cos = [results[l]['top_dir_cos_prev'] for l in range(n_layers)]

# Largest R² jump
r2_diffs = [proc_r2s[l] - proc_r2s[l-1] for l in range(1, n_layers)]
max_r2_jump = np.argmax(r2_diffs) + 1
print(f"\nLargest Procrustes R² jump: L{max_r2_jump-1}→L{max_r2_jump} "
      f"(+{r2_diffs[max_r2_jump-1]:.3f})")

# Largest mean offset jump
mo_diffs = [mean_offs[l] - mean_offs[l-1] for l in range(1, n_layers)]
max_mo_jump = np.argmax(mo_diffs) + 1
print(f"Largest mean offset jump: L{max_mo_jump-1}→L{max_mo_jump} "
      f"(+{mo_diffs[max_mo_jump-1]:.1f}%)")

# Largest direction break
dir_breaks = [1 - results[l]['dir_change_cos'] for l in range(1, n_layers)]
max_dir_break = np.argmax(dir_breaks) + 1
print(f"Largest direction break: L{max_dir_break-1}→L{max_dir_break} "
      f"(cos={results[max_dir_break]['dir_change_cos']:.3f})")

print(f"\nResults saved to output/full_depth_profile.json")
