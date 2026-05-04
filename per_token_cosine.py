"""
PER-TOKEN COSINE SEISMOGRAPH — Full-resolution cross-lingual alignment.

For each problem, compute mean-centered cosine similarity between language pairs
at every generation step, aligned by τ = step/total_steps.

This gives per-token granularity without needing a population for kernel matrices.
Then bin at 2% for CKA comparison.

Output: per-problem cosine curves, overlaid plot, 2%-binned CKA.
"""

import numpy as np
import json
import itertools

# ---------- LOAD ----------
print("Loading trajectories...")
traj = np.load('output/gen_trajectories.npz')
with open('output/gen_trajectories_meta.json') as f:
    gen_meta = json.load(f)

GEN_LANGS = ['zh', 'en', 'es', 'ja']

# Build problem list
gen_complete = []
for p in range(20):
    keys = [f"prob{p}_{l}" for l in GEN_LANGS]
    if all(k in traj for k in keys):
        min_steps = min(traj[k].shape[0] for k in keys)
        if min_steps >= 10:
            gen_complete.append(p)

print(f"Complete problems: {len(gen_complete)}")

# ---------- PER-TOKEN COSINE ----------
def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


# For each problem, for each language pair, compute cosine at every τ
# τ is normalized to [0, 1] based on each trajectory's own length
N_TAU_BINS = 500  # 0.2% resolution

print("\nComputing per-token cosine similarity...")
pairs = list(itertools.combinations(GEN_LANGS, 2))

# Store: per_problem_curves[prob][pair_str] = (tau_points, cosine_values)
per_problem_curves = {}

# Also accumulate into fine bins for averaging
bin_edges = np.linspace(0, 1, N_TAU_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# For each pair, accumulate (tau, cosine) across problems
pair_all_taus = {f"{a}-{b}": [] for a, b in pairs}
pair_all_cosines = {f"{a}-{b}": [] for a, b in pairs}

for p in gen_complete:
    per_problem_curves[p] = {}

    for la, lb in pairs:
        pair_str = f"{la}-{lb}"
        h_a = traj[f"prob{p}_{la}"]  # (n_a, 2048)
        h_b = traj[f"prob{p}_{lb}"]  # (n_b, 2048)

        n_a = h_a.shape[0]
        n_b = h_b.shape[0]

        # Mean-center each trajectory independently
        h_a_c = h_a - h_a.mean(axis=0)
        h_b_c = h_b - h_b.mean(axis=0)

        # Compute τ for each step
        tau_a = np.arange(n_a) / (n_a - 1) if n_a > 1 else np.array([1.0])
        tau_b = np.arange(n_b) / (n_b - 1) if n_b > 1 else np.array([1.0])

        # Interpolate the shorter trajectory to match the longer one's τ grid
        # Use the finer grid (more steps) as the reference
        if n_a >= n_b:
            tau_grid = tau_a
            interp_b = np.array([
                h_b_c[min(int(t * (n_b - 1) + 0.5), n_b - 1)]
                for t in tau_grid
            ])
            ref_a = h_a_c
            ref_b = interp_b
        else:
            tau_grid = tau_b
            interp_a = np.array([
                h_a_c[min(int(t * (n_a - 1) + 0.5), n_a - 1)]
                for t in tau_grid
            ])
            ref_a = interp_a
            ref_b = h_b_c

        # Compute cosine at each τ
        cosines = np.array([cosine_sim(ref_a[i], ref_b[i]) for i in range(len(tau_grid))])

        per_problem_curves[p][pair_str] = {
            'tau': tau_grid.tolist(),
            'cosine': cosines.tolist(),
            'n_points': len(tau_grid),
        }

        pair_all_taus[pair_str].extend(tau_grid.tolist())
        pair_all_cosines[pair_str].extend(cosines.tolist())


# ---------- BIN INTO 2% BINS ----------
print("Binning into 2% resolution...")

bin_edges_2pct = np.linspace(0, 1, 51)  # 2% bins
bin_centers_2pct = (bin_edges_2pct[:-1] + bin_edges_2pct[1:]) / 2

binned_results = {}
for pair_str in pair_all_taus:
    taus = np.array(pair_all_taus[pair_str])
    coss = np.array(pair_all_cosines[pair_str])

    bin_means = []
    bin_stds = []
    bin_counts = []

    for i in range(len(bin_centers_2pct)):
        mask = (taus >= bin_edges_2pct[i]) & (taus < bin_edges_2pct[i+1])
        if mask.sum() > 0:
            bin_means.append(float(np.mean(coss[mask])))
            bin_stds.append(float(np.std(coss[mask])))
            bin_counts.append(int(mask.sum()))
        else:
            bin_means.append(None)
            bin_stds.append(None)
            bin_counts.append(0)

    binned_results[pair_str] = {
        'bin_centers': bin_centers_2pct.tolist(),
        'means': bin_means,
        'stds': bin_stds,
        'counts': bin_counts,
    }

# ---------- AVERAGE ACROSS PAIRS ----------
print("\nAveraging across all language pairs...")

# Average the binned means across pairs
all_pair_means = []
for pair_str in binned_results:
    means = binned_results[pair_str]['means']
    all_pair_means.append(means)

avg_cosine = []
for i in range(len(bin_centers_2pct)):
    vals = [m[i] for m in all_pair_means if m[i] is not None]
    avg_cosine.append(float(np.mean(vals)) if vals else None)


# ---------- CKA AT 2% BINS ----------
print("Computing CKA at 2% bins...")

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

cka_2pct = {}
for i, tau_center in enumerate(bin_centers_2pct):
    frac = tau_center
    cka_vals = []
    for la, lb in pairs:
        X = []
        Y = []
        for p in gen_complete:
            h_a = traj[f"prob{p}_{la}"]
            h_b = traj[f"prob{p}_{lb}"]
            idx_a = min(int(h_a.shape[0] * frac), h_a.shape[0] - 1)
            idx_b = min(int(h_b.shape[0] * frac), h_b.shape[0] - 1)
            X.append(h_a[idx_a])
            Y.append(h_b[idx_b])
        if len(X) >= 5:
            cka = linear_CKA(np.array(X), np.array(Y))
            cka_vals.append(cka)

    if cka_vals:
        cka_2pct[int(round(tau_center * 100))] = {
            'mean_cka': float(np.mean(cka_vals)),
            'std_cka': float(np.std(cka_vals)),
        }


# ---------- PRINT RESULTS ----------
print("\n" + "=" * 80)
print("PER-TOKEN COSINE SEISMOGRAPH — 2% RESOLUTION")
print("=" * 80)
print(f"\n{'τ':>5} | {'Avg Cosine':>10} | {'CKA':>8} | Visual")
print("-" * 65)

for i, tc in enumerate(bin_centers_2pct):
    pct = int(round(tc * 100))
    cos_val = avg_cosine[i]
    cka_val = cka_2pct.get(pct, {}).get('mean_cka', None)

    if cos_val is not None:
        # Scale bar: cosine can be negative
        bar_len = max(0, int((cos_val + 0.2) * 30))  # shift so -0.2 → 0, 0.8 → 30
        bar = '█' * bar_len
        cos_str = f"{cos_val:>10.4f}"
    else:
        bar = ""
        cos_str = f"{'N/A':>10}"

    cka_str = f"{cka_val:.4f}" if cka_val is not None else "N/A"

    if i % 5 == 0:  # Print every 10% for readability
        print(f"  {pct:>3}% | {cos_str} | {cka_str:>8} | {bar}")
    elif cos_val is not None and (cos_val > 0.3 or cos_val < -0.1):
        print(f"  {pct:>3}% | {cos_str} | {cka_str:>8} | {bar}")


# ---------- PER-PROBLEM SUMMARY ----------
print("\n" + "=" * 80)
print("PER-PROBLEM COSINE CURVES (zh-en)")
print("=" * 80)

for p in gen_complete[:5]:  # Show first 5
    curve = per_problem_curves[p].get('zh-en', {})
    if not curve:
        continue
    taus = np.array(curve['tau'])
    cosines = np.array(curve['cosine'])

    cat = gen_meta[f"prob{p}_zh"]['category']
    n_pts = len(taus)

    # Find peaks and valleys
    if len(cosines) > 3:
        # Smooth slightly for peak detection
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(cosines, size=max(3, n_pts // 20))

        # Find local maxima
        peak_idx = []
        for j in range(1, len(smooth) - 1):
            if smooth[j] > smooth[j-1] and smooth[j] > smooth[j+1] and smooth[j] > 0.2:
                peak_idx.append(j)

        peak_taus = [f"{taus[j]:.2f}" for j in peak_idx[:5]]

        print(f"\n  Problem {p} ({cat}, {n_pts} points):")
        print(f"    Mean cosine: {np.mean(cosines):.4f}, std: {np.std(cosines):.4f}")
        print(f"    Range: [{np.min(cosines):.4f}, {np.max(cosines):.4f}]")
        print(f"    Peaks at τ: {peak_taus}")

        # Mini ASCII plot
        n_cols = 60
        indices = np.linspace(0, len(cosines)-1, n_cols).astype(int)
        sampled = cosines[indices]
        row = ""
        for v in sampled:
            if v > 0.3: row += "▓"
            elif v > 0.1: row += "░"
            elif v > -0.1: row += "·"
            else: row += "_"
        print(f"    τ=0%{row}τ=100%")


# ---------- OSCILLATION ANALYSIS ----------
print("\n" + "=" * 80)
print("OSCILLATION ANALYSIS")
print("=" * 80)

# Check if per-problem peaks align or are independent
all_peak_taus = []
for p in gen_complete:
    curve = per_problem_curves[p].get('zh-en', {})
    if not curve:
        continue
    cosines = np.array(curve['cosine'])
    taus = np.array(curve['tau'])
    if len(cosines) < 10:
        continue

    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(cosines, size=max(3, len(cosines) // 20))
    for j in range(1, len(smooth) - 1):
        if smooth[j] > smooth[j-1] and smooth[j] > smooth[j+1] and smooth[j] > 0.2:
            all_peak_taus.append(taus[j])

if all_peak_taus:
    all_peak_taus = np.array(all_peak_taus)
    # Histogram of peak τ positions
    hist, edges = np.histogram(all_peak_taus, bins=20, range=(0, 1))
    print(f"\n  Distribution of peak τ positions across all problems:")
    for i, count in enumerate(hist):
        pct = int((edges[i] + edges[i+1]) / 2 * 100)
        bar = '█' * count
        print(f"    {pct:>3}%: {bar} ({count})")

    print(f"\n  Total peaks found: {len(all_peak_taus)}")
    print(f"  Mean peak τ: {np.mean(all_peak_taus):.3f}")
    print(f"  Std peak τ: {np.std(all_peak_taus):.3f}")

    # Are peaks clustered or uniform?
    from scipy.stats import kstest
    stat, p_val = kstest(all_peak_taus, 'uniform')
    print(f"  KS test vs uniform: stat={stat:.4f}, p={p_val:.6f}")
    if p_val < 0.05:
        print("  → Peaks are NOT uniformly distributed (clustered at specific τ)")
    else:
        print("  → Peaks are uniformly distributed (no preferred τ)")


# ---------- SAVE ----------
output = {
    'per_problem_curves': {str(k): v for k, v in per_problem_curves.items()},
    'binned_2pct': binned_results,
    'avg_cosine_2pct': avg_cosine,
    'cka_2pct': cka_2pct,
    'bin_centers': bin_centers_2pct.tolist(),
    'n_problems': len(gen_complete),
    'problems': gen_complete,
}

with open('output/per_token_cosine_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: output/per_token_cosine_results.json")
