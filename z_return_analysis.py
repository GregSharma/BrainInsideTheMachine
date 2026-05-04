"""
Z RETURN PATH ANALYSIS — How does the model find the lego?

Z is the resting state. The model DEPARTS for narration, RETURNS for reasoning.
L30-32 activate during departure (anti-correlated with cosine).

Question: What do the layer deltas look like during RETURN transitions?
(cosine going from LOW → HIGH = return to Z)

Uses cached layer_deltas.npz — no GPU needed.
"""

import numpy as np
import json
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d

# ---------- LOAD ----------
print("Loading cached data...")
deltas_data = np.load('output/layer_deltas.npz')
with open('output/layer_deltas_meta.json') as f:
    meta = json.load(f)

# Also load original trajectories for cosine
traj = np.load('output/gen_trajectories.npz')

N_LAYERS = 36
GEN_LANGS = ['zh', 'en']

# Build problem list
selected = sorted(set(int(k.split('_')[0].replace('prob',''))
                      for k in meta.keys() if k.startswith('prob')))

# Filter to problems with both zh and en
complete = [p for p in selected
            if f"prob{p}_zh" in meta and f"prob{p}_en" in meta
            and f"deltas_prob{p}_zh" in deltas_data]

print(f"Problems with full data: {len(complete)}")


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


# ---------- COMPUTE PER-TOKEN COSINE + IDENTIFY TRANSITIONS ----------
print("\nComputing cosine curves and identifying transitions...")

all_returns = []   # (problem, tau_start, tau_end, cosine_start, cosine_end)
all_departures = []

return_delta_profiles = []   # (n_layers,) — avg delta during return
departure_delta_profiles = []

for p in complete:
    h_zh = deltas_data[f"h32_prob{p}_zh"]  # (n_steps, 2048)
    h_en = deltas_data[f"h32_prob{p}_en"]
    d_zh = deltas_data[f"deltas_prob{p}_zh"]  # (n_steps-1, 36)
    d_en = deltas_data[f"deltas_prob{p}_en"]

    n_zh = h_zh.shape[0]
    n_en = h_en.shape[0]

    # Mean-center
    h_zh_c = h_zh - h_zh.mean(axis=0)
    h_en_c = h_en - h_en.mean(axis=0)

    # Common grid
    n_grid = min(n_zh, n_en, 200)
    if n_grid < 20:
        continue

    tau_grid = np.linspace(0, 1, n_grid)

    # Cosine at each grid point
    cosines = np.zeros(n_grid)
    for i, tau in enumerate(tau_grid):
        idx_zh = min(int(tau * (n_zh - 1)), n_zh - 1)
        idx_en = min(int(tau * (n_en - 1)), n_en - 1)
        cosines[i] = cosine_sim(h_zh_c[idx_zh], h_en_c[idx_en])

    # Smooth for transition detection
    smooth = uniform_filter1d(cosines, size=max(5, n_grid // 20))

    # Delta norms interpolated to grid, averaged across zh/en
    n_d_zh = d_zh.shape[0]
    n_d_en = d_en.shape[0]
    delta_grid = np.zeros((n_grid, N_LAYERS))
    for l in range(N_LAYERS):
        d_zh_interp = np.interp(tau_grid, np.linspace(0, 1, n_d_zh), d_zh[:, l])
        d_en_interp = np.interp(tau_grid, np.linspace(0, 1, n_d_en), d_en[:, l])
        delta_grid[:, l] = (d_zh_interp + d_en_interp) / 2

    # Identify transitions: derivative of smoothed cosine
    d_cosine = np.diff(smooth)

    # RETURN = sustained positive derivative (cosine increasing = returning to Z)
    # DEPARTURE = sustained negative derivative (cosine decreasing = leaving Z)
    # Look for runs of 3+ consecutive positive/negative derivatives
    window = max(3, n_grid // 30)

    for i in range(window, len(d_cosine) - window):
        # Check if this is middle of a return (positive run)
        run = d_cosine[i-window:i+window]
        if np.mean(run) > 0.005 and smooth[i+window] - smooth[i-window] > 0.1:
            # This is a return transition
            delta_profile = delta_grid[i-window:i+window].mean(axis=0)  # avg delta during return
            return_delta_profiles.append(delta_profile)
            all_returns.append({
                'problem': p,
                'tau': float(tau_grid[i]),
                'cos_before': float(smooth[i-window]),
                'cos_after': float(smooth[i+window]),
                'delta_cos': float(smooth[i+window] - smooth[i-window]),
            })

        elif np.mean(run) < -0.005 and smooth[i-window] - smooth[i+window] > 0.1:
            # This is a departure
            delta_profile = delta_grid[i-window:i+window].mean(axis=0)
            departure_delta_profiles.append(delta_profile)
            all_departures.append({
                'problem': p,
                'tau': float(tau_grid[i]),
                'cos_before': float(smooth[i-window]),
                'cos_after': float(smooth[i+window]),
                'delta_cos': float(smooth[i-window] - smooth[i+window]),
            })


print(f"\nFound {len(all_returns)} return transitions (LOW→HIGH cosine)")
print(f"Found {len(all_departures)} departure transitions (HIGH→LOW cosine)")


# ---------- COMPARE LAYER PROFILES: RETURN vs DEPARTURE ----------
print("\n" + "=" * 90)
print("LAYER DELTA PROFILES: RETURN TO Z vs DEPARTURE FROM Z")
print("=" * 90)

if return_delta_profiles and departure_delta_profiles:
    avg_return = np.mean(return_delta_profiles, axis=0)
    avg_departure = np.mean(departure_delta_profiles, axis=0)

    # Normalize to make comparable
    avg_return_norm = avg_return / avg_return.mean()
    avg_departure_norm = avg_departure / avg_departure.mean()

    ratio = avg_return / (avg_departure + 1e-10)

    print(f"\n{'Layer':>6} | {'Return δ':>9} | {'Depart δ':>9} | {'Ratio R/D':>9} | Visual")
    print("-" * 75)

    for l in range(N_LAYERS):
        r_val = avg_return[l]
        d_val = avg_departure[l]
        r_ratio = ratio[l]

        # Bar: ratio > 1 means more active during return
        if r_ratio > 1.05:
            bar = '→' * min(int((r_ratio - 1) * 20), 15)  # return dominant
            marker = " RETURN"
        elif r_ratio < 0.95:
            bar = '←' * min(int((1 - r_ratio) * 20), 15)  # departure dominant
            marker = " DEPART"
        else:
            bar = '='
            marker = ""

        # Only print layers with notable asymmetry or every 5th
        if abs(r_ratio - 1) > 0.03 or l % 5 == 0 or l >= 28:
            print(f"  L{l:>2}   | {r_val:>9.2f} | {d_val:>9.2f} | {r_ratio:>9.3f} | {bar}{marker}")

    # Which layers show the biggest return/departure asymmetry?
    asymmetry = ratio - 1  # positive = more active during return

    print(f"\n  TOP 5 RETURN-DOMINANT LAYERS (more active when returning to Z):")
    for l in np.argsort(asymmetry)[-5:][::-1]:
        print(f"    L{l}: ratio = {ratio[l]:.3f} (return δ={avg_return[l]:.2f}, depart δ={avg_departure[l]:.2f})")

    print(f"\n  TOP 5 DEPARTURE-DOMINANT LAYERS (more active when leaving Z):")
    for l in np.argsort(asymmetry)[:5]:
        print(f"    L{l}: ratio = {ratio[l]:.3f} (return δ={avg_return[l]:.2f}, depart δ={avg_departure[l]:.2f})")


    # ---------- KEY QUESTION: Do specific layers DRIVE the return? ----------
    print("\n" + "=" * 90)
    print("RETURN MECHANISM: Which layers' deltas PREDICT the magnitude of return?")
    print("=" * 90)

    # For each return transition, correlate layer delta with cosine recovery magnitude
    if len(return_delta_profiles) > 5:
        return_magnitudes = np.array([r['delta_cos'] for r in all_returns])
        return_deltas = np.array(return_delta_profiles)

        print(f"\n{'Layer':>6} | {'r(δ, Δcos)':>10} | {'p-value':>8} | Interpretation")
        print("-" * 65)

        predictive_layers = []
        for l in range(N_LAYERS):
            if np.std(return_deltas[:, l]) > 1e-8:
                r, p = pearsonr(return_deltas[:, l], return_magnitudes)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if abs(r) > 0.15 or l >= 28 or l % 5 == 0:
                    interp = ""
                    if r > 0.2 and p < 0.05:
                        interp = "← DRIVES return"
                        predictive_layers.append((l, r))
                    elif r < -0.2 and p < 0.05:
                        interp = "← RESISTS return"
                        predictive_layers.append((l, r))
                    print(f"  L{l:>2}   | {r:>10.4f} | {p:>8.4f} | {sig} {interp}")

        if predictive_layers:
            print(f"\n  PREDICTIVE LAYERS:")
            for l, r in sorted(predictive_layers, key=lambda x: -abs(x[1])):
                role = "DRIVES" if r > 0 else "RESISTS"
                print(f"    L{l}: r = {r:.4f} ({role} return to Z)")


# ---------- DEPARTURE/RETURN τ DISTRIBUTION ----------
print("\n" + "=" * 90)
print("WHEN DO DEPARTURES AND RETURNS HAPPEN?")
print("=" * 90)

if all_returns:
    return_taus = [r['tau'] for r in all_returns]
    print(f"\n  Returns (→Z): mean τ = {np.mean(return_taus):.3f}, "
          f"std = {np.std(return_taus):.3f}, n = {len(return_taus)}")

if all_departures:
    depart_taus = [d['tau'] for d in all_departures]
    print(f"  Departures (←Z): mean τ = {np.mean(depart_taus):.3f}, "
          f"std = {np.std(depart_taus):.3f}, n = {len(depart_taus)}")

    # Early vs late
    early_returns = sum(1 for t in return_taus if t < 0.5)
    late_returns = sum(1 for t in return_taus if t >= 0.5)
    early_departs = sum(1 for t in depart_taus if t < 0.5)
    late_departs = sum(1 for t in depart_taus if t >= 0.5)
    print(f"\n  Early (τ<0.5): {early_returns} returns, {early_departs} departures")
    print(f"  Late (τ≥0.5):  {late_returns} returns, {late_departs} departures")


# ---------- SAVE ----------
output = {
    'returns': all_returns,
    'departures': all_departures,
    'n_returns': len(all_returns),
    'n_departures': len(all_departures),
    'avg_return_delta': avg_return.tolist() if return_delta_profiles else [],
    'avg_departure_delta': avg_departure.tolist() if departure_delta_profiles else [],
    'ratio_return_over_departure': ratio.tolist() if return_delta_profiles and departure_delta_profiles else [],
}

with open('output/z_return_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: output/z_return_analysis.json")
