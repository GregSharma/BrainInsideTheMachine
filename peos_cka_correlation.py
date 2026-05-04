"""
p_EOS × CKA CORRELATION — Does the model's own progress signal align with
cross-lingual convergence?

For each problem:
1. Plot p_EOS (zh and en) alongside per-token cosine similarity
2. Compute correlation between p_EOS and cosine at matched τ
3. Check if p_EOS spikes coincide with cosine spikes

If they correlate: the model KNOWS when it's at a subway station.
"""

import numpy as np
import json
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import uniform_filter1d
import torch.nn.functional as F

# ---------- LOAD ----------
print("Loading data...")
peos_data = np.load('output/gen_trajectories_peos.npz')
with open('output/gen_trajectories_peos_meta.json') as f:
    peos_meta = json.load(f)

# Also load original trajectories for comparison
traj = np.load('output/gen_trajectories.npz')

GEN_LANGS = ['zh', 'en', 'es', 'ja']

# Build problem list
gen_complete = []
for p in range(22):
    keys_peos = [f"h32_prob{p}_{l}" for l in GEN_LANGS]
    keys_traj = [f"prob{p}_{l}" for l in GEN_LANGS]
    if all(k in peos_data for k in keys_peos) and all(k in traj for k in keys_traj):
        min_steps = min(peos_data[k].shape[0] for k in keys_peos)
        if min_steps >= 10:
            gen_complete.append(p)

print(f"Complete problems: {len(gen_complete)}")


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


# ---------- PER-PROBLEM: p_EOS vs COSINE ----------
print("\n" + "=" * 90)
print("p_EOS × COSINE CORRELATION — Per problem")
print("=" * 90)

all_correlations = []
all_peos_at_cosine_peaks = []
all_peos_at_cosine_valleys = []

for p in gen_complete:
    # Get p_EOS for zh and en
    peos_zh = peos_data[f"peos_prob{p}_zh"]
    peos_en = peos_data[f"peos_prob{p}_en"]
    h_zh = traj[f"prob{p}_zh"]
    h_en = traj[f"prob{p}_en"]

    n_zh = h_zh.shape[0]
    n_en = h_en.shape[0]

    # Mean-center h32
    h_zh_c = h_zh - h_zh.mean(axis=0)
    h_en_c = h_en - h_en.mean(axis=0)

    # Align to common τ grid (use 200 points)
    n_grid = 200
    tau_grid = np.linspace(0, 1, n_grid)

    # Interpolate all signals to common grid
    def interp_to_grid(signal, n_orig):
        orig_tau = np.linspace(0, 1, n_orig)
        return np.interp(tau_grid, orig_tau, signal)

    def interp_vec_to_grid(vecs, n_orig):
        indices = np.clip((tau_grid * (n_orig - 1)).astype(int), 0, n_orig - 1)
        return vecs[indices]

    peos_zh_grid = interp_to_grid(peos_zh, len(peos_zh))
    peos_en_grid = interp_to_grid(peos_en, len(peos_en))
    h_zh_grid = interp_vec_to_grid(h_zh_c, n_zh)
    h_en_grid = interp_vec_to_grid(h_en_c, n_en)

    # Compute cosine at each grid point
    cosine_grid = np.array([cosine_sim(h_zh_grid[i], h_en_grid[i]) for i in range(n_grid)])

    # Average p_EOS across languages
    peos_avg = (peos_zh_grid + peos_en_grid) / 2

    # Compute progress = 1 - prod(1 - p_EOS)
    log_surv_zh = np.cumsum(np.log(1 - np.clip(peos_zh_grid, 0, 1 - 1e-10)))
    log_surv_en = np.cumsum(np.log(1 - np.clip(peos_en_grid, 0, 1 - 1e-10)))
    progress_zh = 1 - np.exp(log_surv_zh)
    progress_en = 1 - np.exp(log_surv_en)
    progress_avg = (progress_zh + progress_en) / 2

    # Correlations
    # Pearson between cosine and p_EOS
    if np.std(cosine_grid) > 1e-6 and np.std(peos_avg) > 1e-6:
        r_peos, p_peos = pearsonr(cosine_grid, peos_avg)
        rho_peos, p_rho = spearmanr(cosine_grid, peos_avg)
    else:
        r_peos, p_peos = 0, 1
        rho_peos, p_rho = 0, 1

    # Pearson between cosine and progress
    if np.std(progress_avg) > 1e-6:
        r_prog, p_prog = pearsonr(cosine_grid, progress_avg)
    else:
        r_prog, p_prog = 0, 1

    cat = peos_meta[f"prob{p}_zh"]['category']

    all_correlations.append({
        'problem': p,
        'category': cat,
        'pearson_peos': float(r_peos),
        'p_value_peos': float(p_peos),
        'spearman_peos': float(rho_peos),
        'pearson_progress': float(r_prog),
        'peos_zh_max': float(np.max(peos_zh)),
        'peos_en_max': float(np.max(peos_en)),
        'cosine_max': float(np.max(cosine_grid)),
    })

    # What's p_EOS at cosine peaks vs valleys?
    cosine_smooth = uniform_filter1d(cosine_grid, size=10)
    top_20pct = np.percentile(cosine_grid, 80)
    bot_20pct = np.percentile(cosine_grid, 20)

    peos_at_high_cos = peos_avg[cosine_grid >= top_20pct]
    peos_at_low_cos = peos_avg[cosine_grid <= bot_20pct]

    if len(peos_at_high_cos) > 0 and len(peos_at_low_cos) > 0:
        all_peos_at_cosine_peaks.extend(peos_at_high_cos.tolist())
        all_peos_at_cosine_valleys.extend(peos_at_low_cos.tolist())

    sig = "***" if p_peos < 0.001 else "**" if p_peos < 0.01 else "*" if p_peos < 0.05 else ""
    print(f"  P{p:>2} ({cat:>13}): r(cos,pEOS)={r_peos:>6.3f}{sig:>3}  "
          f"r(cos,prog)={r_prog:>6.3f}  "
          f"pEOS_max: zh={np.max(peos_zh):.3f} en={np.max(peos_en):.3f}")


# ---------- AGGREGATE STATISTICS ----------
print("\n" + "=" * 90)
print("AGGREGATE CORRELATION STATISTICS")
print("=" * 90)

r_vals = [c['pearson_peos'] for c in all_correlations]
rho_vals = [c['spearman_peos'] for c in all_correlations]
r_prog_vals = [c['pearson_progress'] for c in all_correlations]

print(f"\n  Pearson r(cosine, p_EOS):")
print(f"    Mean:   {np.mean(r_vals):.4f}")
print(f"    Median: {np.median(r_vals):.4f}")
print(f"    Std:    {np.std(r_vals):.4f}")
print(f"    Range:  [{min(r_vals):.4f}, {max(r_vals):.4f}]")
n_positive = sum(1 for r in r_vals if r > 0)
n_sig = sum(1 for c in all_correlations if c['p_value_peos'] < 0.05)
print(f"    Positive: {n_positive}/{len(r_vals)}")
print(f"    Significant (p<0.05): {n_sig}/{len(r_vals)}")

print(f"\n  Spearman ρ(cosine, p_EOS):")
print(f"    Mean:   {np.mean(rho_vals):.4f}")
print(f"    Median: {np.median(rho_vals):.4f}")

print(f"\n  Pearson r(cosine, progress CDF):")
print(f"    Mean:   {np.mean(r_prog_vals):.4f}")
print(f"    Median: {np.median(r_prog_vals):.4f}")

# p_EOS at cosine peaks vs valleys
print(f"\n  p_EOS at cosine peaks (top 20%):   mean={np.mean(all_peos_at_cosine_peaks):.4f}")
print(f"  p_EOS at cosine valleys (bot 20%): mean={np.mean(all_peos_at_cosine_valleys):.4f}")
ratio = np.mean(all_peos_at_cosine_peaks) / max(np.mean(all_peos_at_cosine_valleys), 1e-10)
print(f"  Ratio: {ratio:.2f}x")


# ---------- BY CATEGORY ----------
print("\n" + "=" * 90)
print("CORRELATION BY CATEGORY")
print("=" * 90)

for cat in ['arithmetic', 'combinatorics', 'modular', 'geometry', 'sequences']:
    cat_r = [c['pearson_peos'] for c in all_correlations if c['category'] == cat]
    if cat_r:
        print(f"  {cat:>13}: mean r = {np.mean(cat_r):.4f} (n={len(cat_r)})")


# ---------- POPULATION-LEVEL: BINNED p_EOS vs BINNED CKA ----------
print("\n" + "=" * 90)
print("POPULATION-LEVEL: BINNED p_EOS vs BINNED CKA (5% bins)")
print("=" * 90)

def linear_CKA(X, Y):
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    XtX = Xc @ Xc.T
    YtY = Yc @ Yc.T
    hsic = np.trace(XtX @ YtY) / ((n-1)**2)
    var_x = np.trace(XtX @ XtX) / ((n-1)**2)
    var_y = np.trace(YtY @ YtY) / ((n-1)**2)
    if var_x < 1e-12 or var_y < 1e-12:
        return 0.0
    return hsic / np.sqrt(var_x * var_y)

import itertools
pairs = list(itertools.combinations(GEN_LANGS, 2))

binned_cka = []
binned_peos_avg = []
binned_progress_avg = []

fracs = [i / 20 for i in range(1, 21)]  # 5% to 100%

print(f"\n{'τ':>5} | {'CKA':>6} | {'p_EOS avg':>9} | {'Progress':>8} | CKA bar")
print("-" * 60)

for frac in fracs:
    # CKA at this frac
    cka_vals = []
    for la, lb in pairs:
        X, Y = [], []
        for p in gen_complete:
            h_a = traj[f"prob{p}_{la}"]
            h_b = traj[f"prob{p}_{lb}"]
            idx_a = min(int(h_a.shape[0] * frac), h_a.shape[0] - 1)
            idx_b = min(int(h_b.shape[0] * frac), h_b.shape[0] - 1)
            X.append(h_a[idx_a])
            Y.append(h_b[idx_b])
        cka_vals.append(linear_CKA(np.array(X), np.array(Y)))

    mean_cka = np.mean(cka_vals)

    # Average p_EOS across all problems and languages at this frac
    peos_vals = []
    prog_vals = []
    for p in gen_complete:
        for l in GEN_LANGS:
            pe = peos_data[f"peos_prob{p}_{l}"]
            idx = min(int(len(pe) * frac), len(pe) - 1)
            peos_vals.append(pe[idx])

            # Progress
            log_surv = np.cumsum(np.log(1 - np.clip(pe[:idx+1], 0, 1 - 1e-10)))
            prog_vals.append(1 - np.exp(log_surv[-1]))

    mean_peos = np.mean(peos_vals)
    mean_prog = np.mean(prog_vals)

    binned_cka.append(mean_cka)
    binned_peos_avg.append(mean_peos)
    binned_progress_avg.append(mean_prog)

    bar = '█' * int(mean_cka * 30)
    pct = int(frac * 100)
    print(f"  {pct:>3}% | {mean_cka:.4f} | {mean_peos:.6f} | {mean_prog:.4f}  | {bar}")


# Correlation between binned CKA and binned p_EOS
r_binned, p_binned = pearsonr(binned_cka, binned_peos_avg)
r_binned_prog, p_binned_prog = pearsonr(binned_cka, binned_progress_avg)

print(f"\n  Pearson r(CKA, p_EOS) across bins: {r_binned:.4f} (p={p_binned:.6f})")
print(f"  Pearson r(CKA, progress) across bins: {r_binned_prog:.4f} (p={p_binned_prog:.6f})")


# ---------- THE BIG QUESTION ----------
print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)

if np.mean(r_vals) > 0.15 and n_sig >= len(all_correlations) * 0.3:
    print("\n  p_EOS CORRELATES with cross-lingual cosine similarity.")
    print("  The model's own progress signal aligns with cross-lingual convergence.")
    print("  → The model KNOWS when it's at a subway station.")
elif np.mean(r_vals) > 0.05:
    print("\n  WEAK correlation between p_EOS and cross-lingual similarity.")
    print("  Some signal but not conclusive.")
else:
    print("\n  NO correlation between p_EOS and cross-lingual similarity.")
    print("  The model's progress estimate and cross-lingual alignment are independent signals.")


# ---------- SAVE ----------
output = {
    'per_problem': all_correlations,
    'aggregate': {
        'mean_pearson_peos': float(np.mean(r_vals)),
        'median_pearson_peos': float(np.median(r_vals)),
        'mean_spearman_peos': float(np.mean(rho_vals)),
        'mean_pearson_progress': float(np.mean(r_prog_vals)),
        'n_positive': n_positive,
        'n_significant': n_sig,
        'n_total': len(all_correlations),
        'peos_at_cosine_peaks': float(np.mean(all_peos_at_cosine_peaks)),
        'peos_at_cosine_valleys': float(np.mean(all_peos_at_cosine_valleys)),
    },
    'binned': {
        'fracs': [int(f*100) for f in fracs],
        'cka': [float(x) for x in binned_cka],
        'peos_avg': [float(x) for x in binned_peos_avg],
        'progress_avg': [float(x) for x in binned_progress_avg],
        'r_cka_peos': float(r_binned),
        'r_cka_progress': float(r_binned_prog),
    },
}

with open('output/peos_cka_correlation.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: output/peos_cka_correlation.json")
