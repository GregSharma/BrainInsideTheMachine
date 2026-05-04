"""
CKA PULSE ANALYSIS — Fine-grained sweep + station cocycle + token annotation.

Step 1: CKA at every 5% from 5% to 100% (20 points). Confirms/denies two-peak pulse.
Step 2: Token annotation at CKA peaks (>0.5) vs valleys (<0.3).
Step 3: Station cocycle — Ridge maps at CKA peaks, cross-validated, cocycle error.
"""

import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import itertools

# ---------- LOAD ----------
print("Loading data...")
traj = np.load('output/gen_trajectories.npz')
with open('output/gen_trajectories_meta.json') as f:
    gen_meta = json.load(f)

GEN_LANGS = ['zh', 'en', 'es', 'ja']
N_GEN_PROBLEMS = 20

# Build problem list with all 4 languages present
gen_complete = []
for p in range(N_GEN_PROBLEMS):
    if all(f"prob{p}_{l}" in traj for l in GEN_LANGS):
        # Check all have enough steps
        min_steps = min(traj[f"prob{p}_{l}"].shape[0] for l in GEN_LANGS)
        if min_steps >= 10:
            gen_complete.append(p)

print(f"Complete problems (all 4 langs, >=10 steps): {len(gen_complete)}")

def linear_CKA(X, Y):
    """Linear CKA between two sets of representations."""
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


# ================================================================
# STEP 1: FINE-GRAINED CKA SWEEP (every 5%)
# ================================================================
print("\n" + "=" * 70)
print("STEP 1: FINE-GRAINED CKA SWEEP (5% to 100%)")
print("=" * 70)

fracs = [i / 20 for i in range(1, 21)]  # 0.05, 0.10, ..., 1.00
cka_results = {}

for frac in fracs:
    cka_vals = []
    pair_labels = []
    for li, la in enumerate(GEN_LANGS):
        for lj, lb in enumerate(GEN_LANGS):
            if li >= lj:
                continue
            X = []
            Y = []
            for p in gen_complete:
                h_a = traj[f"prob{p}_{la}"]
                h_b = traj[f"prob{p}_{lb}"]
                idx_a = min(int(h_a.shape[0] * frac), h_a.shape[0] - 1)
                idx_b = min(int(h_b.shape[0] * frac), h_b.shape[0] - 1)
                X.append(h_a[idx_a])
                Y.append(h_b[idx_b])
            cka = linear_CKA(np.array(X), np.array(Y))
            cka_vals.append(cka)
            pair_labels.append(f"{la}-{lb}")

    frac_pct = int(frac * 100)
    mean_cka = float(np.mean(cka_vals))
    cka_results[frac_pct] = {
        'frac': frac,
        'mean_cka': mean_cka,
        'min_cka': float(min(cka_vals)),
        'max_cka': float(max(cka_vals)),
        'std_cka': float(np.std(cka_vals)),
        'per_pair': {label: float(v) for label, v in zip(pair_labels, cka_vals)},
    }
    bar = '█' * int(mean_cka * 40)
    print(f"  τ={frac_pct:>3}%: CKA={mean_cka:.4f} {bar}")

# Identify peaks and valleys
sorted_fracs = sorted(cka_results.keys())
cka_curve = [(f, cka_results[f]['mean_cka']) for f in sorted_fracs]
peaks = [f for f, c in cka_curve if c > 0.5]
valleys = [f for f, c in cka_curve if c < 0.3]

print(f"\n  Peaks (CKA > 0.5): {peaks}")
print(f"  Valleys (CKA < 0.3): {valleys}")


# ================================================================
# STEP 2: TOKEN ANNOTATION AT PEAKS VS VALLEYS
# ================================================================
print("\n" + "=" * 70)
print("STEP 2: TOKEN ANNOTATION AT PEAKS VS VALLEYS")
print("=" * 70)

token_annotations = {'peaks': {}, 'valleys': {}}

for frac_pct in peaks[:5]:  # top 5 peaks
    frac = frac_pct / 100
    tokens_at_frac = {}
    for l in GEN_LANGS:
        tokens_at_frac[l] = []
        for p in gen_complete:
            key = f"prob{p}_{l}"
            m = gen_meta[key]
            n_steps = m['n_steps']
            idx = min(int(n_steps * frac), n_steps - 1)
            # Extract token from text_preview
            text = m.get('text_preview', '')
            # We don't have per-token text, but we can show the approximate region
            char_frac = int(len(text) * frac)
            window = text[max(0, char_frac-30):char_frac+30]
            tokens_at_frac[l].append(f"p{p}: ...{window}...")
    token_annotations['peaks'][frac_pct] = tokens_at_frac
    print(f"\n  τ={frac_pct}% (PEAK, CKA={cka_results[frac_pct]['mean_cka']:.3f}):")
    for l in ['zh', 'en']:
        print(f"    {l}: {tokens_at_frac[l][0][:80]}")

for frac_pct in valleys[:5]:  # top 5 valleys
    frac = frac_pct / 100
    tokens_at_frac = {}
    for l in GEN_LANGS:
        tokens_at_frac[l] = []
        for p in gen_complete:
            key = f"prob{p}_{l}"
            m = gen_meta[key]
            n_steps = m['n_steps']
            idx = min(int(n_steps * frac), n_steps - 1)
            text = m.get('text_preview', '')
            char_frac = int(len(text) * frac)
            window = text[max(0, char_frac-30):char_frac+30]
            tokens_at_frac[l].append(f"p{p}: ...{window}...")
    token_annotations['valleys'][frac_pct] = tokens_at_frac
    print(f"\n  τ={frac_pct}% (VALLEY, CKA={cka_results[frac_pct]['mean_cka']:.3f}):")
    for l in ['zh', 'en']:
        print(f"    {l}: {tokens_at_frac[l][0][:80]}")


# ================================================================
# STEP 3: STATION COCYCLE — Ridge maps at CKA peaks
# ================================================================
print("\n" + "=" * 70)
print("STEP 3: STATION COCYCLE (Ridge maps at CKA peaks vs valleys)")
print("=" * 70)

# Use input-pass PCA basis for consistency
multi = np.load('output/multilingual_activations.npz')
all_input = np.vstack([multi[l] for l in ['zh', 'en', 'es', 'ja']])
pca = PCA(n_components=20)
pca.fit(all_input)
print(f"  PCA-20 fitted on input-pass data (var explained: {sum(pca.explained_variance_ratio_):.3f})")

def get_activations_at_frac(frac, pca_transform=True):
    """Get h32 activations at fraction frac for all problems × languages."""
    acts = {}
    for l in GEN_LANGS:
        vecs = []
        for p in gen_complete:
            h = traj[f"prob{p}_{l}"]
            idx = min(int(h.shape[0] * frac), h.shape[0] - 1)
            vecs.append(h[idx])
        arr = np.array(vecs)
        if pca_transform:
            arr = pca.transform(arr)
        acts[l] = arr
    return acts

def cocycle_at_frac(frac, n_train=15, alpha=1.0):
    """Fit Ridge maps between all language pairs at a given trajectory fraction.
    Cross-validate with n_train/n_test split.
    Return mean R², cocycle error, and per-pair R²."""
    acts = get_activations_at_frac(frac)
    n = len(gen_complete)
    n_test = n - n_train

    # Random but reproducible split
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # Fit all pairwise Ridge maps
    maps = {}
    r2_train = {}
    r2_test = {}

    pairs = list(itertools.combinations(GEN_LANGS, 2))
    for la, lb in pairs:
        X_tr = acts[la][train_idx]
        Y_tr = acts[lb][train_idx]
        X_te = acts[la][test_idx]
        Y_te = acts[lb][test_idx]

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, Y_tr)

        r2_train[(la, lb)] = float(r2_score(Y_tr, ridge.predict(X_tr)))
        r2_test[(la, lb)] = float(r2_score(Y_te, ridge.predict(X_te)))
        maps[(la, lb)] = ridge

        # Also fit reverse
        ridge_rev = Ridge(alpha=alpha)
        ridge_rev.fit(Y_tr, X_tr)
        r2_train[(lb, la)] = float(r2_score(X_tr, ridge_rev.predict(Y_tr)))
        r2_test[(lb, la)] = float(r2_score(X_te, ridge_rev.predict(Y_te)))
        maps[(lb, la)] = ridge_rev

    # Cocycle error: for each triple (a,b,c), compare T_{a→c} vs T_{b→c} ∘ T_{a→b}
    cocycle_errors = []
    for la, lb, lc in itertools.permutations(GEN_LANGS, 3):
        if (la, lb) not in maps or (lb, lc) not in maps or (la, lc) not in maps:
            continue
        # Direct: a→c
        direct = maps[(la, lc)].predict(acts[la][test_idx])
        # Composed: a→b→c
        intermediate = maps[(la, lb)].predict(acts[la][test_idx])
        composed = maps[(lb, lc)].predict(intermediate)
        # Error: relative Frobenius norm
        err = np.linalg.norm(direct - composed) / (np.linalg.norm(direct) + 1e-10)
        cocycle_errors.append(err)

    return {
        'mean_r2_train': float(np.mean(list(r2_train.values()))),
        'mean_r2_test': float(np.mean(list(r2_test.values()))),
        'cocycle_error': float(np.mean(cocycle_errors)),
        'cocycle_std': float(np.std(cocycle_errors)),
        'per_pair_r2_test': {f"{a}->{b}": v for (a, b), v in r2_test.items()},
    }


# Run cocycle at each 5% point
cocycle_results = {}
print(f"\n  {'τ':>5} | {'CKA':>6} | {'R² train':>9} | {'R² test':>8} | {'Cocycle err':>11} | Station?")
print("  " + "-" * 65)

for frac_pct in sorted_fracs:
    frac = frac_pct / 100
    cka_val = cka_results[frac_pct]['mean_cka']
    cocycle = cocycle_at_frac(frac)
    cocycle_results[frac_pct] = cocycle

    is_peak = "★ PEAK" if cka_val > 0.5 else ("  valley" if cka_val < 0.3 else "")
    print(f"  {frac_pct:>4}% | {cka_val:.4f} | {cocycle['mean_r2_train']:.4f}    | {cocycle['mean_r2_test']:.4f}  | {cocycle['cocycle_error']:.4f}      | {is_peak}")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("PULSE ANALYSIS SUMMARY")
print("=" * 70)

peak_r2 = [cocycle_results[f]['mean_r2_test'] for f in sorted_fracs if cka_results[f]['mean_cka'] > 0.5]
valley_r2 = [cocycle_results[f]['mean_r2_test'] for f in sorted_fracs if cka_results[f]['mean_cka'] < 0.3]
mid_r2 = [cocycle_results[f]['mean_r2_test'] for f in sorted_fracs if 0.3 <= cka_results[f]['mean_cka'] <= 0.5]

peak_cocycle = [cocycle_results[f]['cocycle_error'] for f in sorted_fracs if cka_results[f]['mean_cka'] > 0.5]
valley_cocycle = [cocycle_results[f]['cocycle_error'] for f in sorted_fracs if cka_results[f]['mean_cka'] < 0.3]

print(f"\n  CKA Peaks (>{0.5}):   mean R²_test = {np.mean(peak_r2):.4f} (n={len(peak_r2)})")
print(f"  CKA Valleys (<{0.3}): mean R²_test = {np.mean(valley_r2):.4f} (n={len(valley_r2)})" if valley_r2 else "  CKA Valleys: none found")
print(f"  CKA Middle:          mean R²_test = {np.mean(mid_r2):.4f} (n={len(mid_r2)})" if mid_r2 else "")

if peak_cocycle and valley_cocycle:
    print(f"\n  Cocycle error at peaks:   {np.mean(peak_cocycle):.4f}")
    print(f"  Cocycle error at valleys: {np.mean(valley_cocycle):.4f}")
    if np.mean(peak_cocycle) < np.mean(valley_cocycle):
        print("  → COCYCLE HOLDS BETTER AT STATIONS (predicted)")
    else:
        print("  → Cocycle does NOT differ between peaks/valleys")

# Detect oscillation
cka_series = [cka_results[f]['mean_cka'] for f in sorted_fracs]
diffs = np.diff(cka_series)
sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
print(f"\n  CKA curve: {sign_changes} direction changes in {len(cka_series)} points")
if sign_changes >= 4:
    print("  → OSCILLATION CONFIRMED (4+ direction changes)")
elif sign_changes >= 2:
    print("  → Possible oscillation (2-3 direction changes)")
else:
    print("  → Monotonic or single-peak — no oscillation")

# ================================================================
# SAVE
# ================================================================
output = {
    'cka_sweep': cka_results,
    'cocycle_at_stations': cocycle_results,
    'peaks': peaks,
    'valleys': valleys,
    'summary': {
        'n_problems': len(gen_complete),
        'pca_dims': 20,
        'alpha': 1.0,
        'n_train': 15,
        'sign_changes': int(sign_changes),
        'mean_peak_r2': float(np.mean(peak_r2)) if peak_r2 else None,
        'mean_valley_r2': float(np.mean(valley_r2)) if valley_r2 else None,
        'mean_peak_cocycle': float(np.mean(peak_cocycle)) if peak_cocycle else None,
        'mean_valley_cocycle': float(np.mean(valley_cocycle)) if valley_cocycle else None,
    }
}

with open('output/cka_pulse_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: output/cka_pulse_results.json")
