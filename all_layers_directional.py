"""
Full 36-layer directional analysis of the departure engine.
Extends the four directional questions to ALL layers.

Key analyses:
1. Category-dependent asymmetry profile across all 36 layers
2. Layer-by-layer departure magnitude by category
3. The Q1 ratio (ret/dep) as a function of layer AND category
"""

import numpy as np
import json
from collections import defaultdict

# ── Load ──────────────────────────────────────────────────────────
deltas_data = np.load('output/layer_deltas.npz', allow_pickle=True)
with open('output/z_return_analysis.json') as f:
    zra = json.load(f)
with open('output/gen_trajectories_meta.json') as f:
    meta = json.load(f)

prob_categories = {}
for key, val in meta.items():
    if '_zh' in key:
        prob_categories[val['problem_idx']] = val['category']

categories = sorted(set(prob_categories.values()))
print(f"Categories: {categories}")
print(f"Problems per category: {[(c, sum(1 for v in prob_categories.values() if v==c)) for c in categories]}")

# ══════════════════════════════════════════════════════════════════
# FULL LAYER × CATEGORY ASYMMETRY MATRIX
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("LAYER × CATEGORY DEPARTURE ASYMMETRY")
print("=" * 80)

# For each layer and category, compute mean delta during departures vs returns
n_layers = 36
results_matrix = {}  # layer -> category -> {dep_mean, ret_mean, ratio}

for L in range(n_layers):
    results_matrix[L] = {}
    for cat in categories:
        cat_probs = [p for p, c in prob_categories.items() if c == cat]
        dep_deltas = []
        ret_deltas = []

        for prob in cat_probs:
            for lang in ['zh', 'en']:
                key = f'deltas_prob{prob}_{lang}'
                if key not in deltas_data:
                    continue
                d = deltas_data[key]  # (T, 36)
                T = len(d)

                for dep in zra['departures']:
                    if dep['problem'] == prob:
                        t = int(dep['tau'] * T)
                        if 0 <= t < T:
                            dep_deltas.append(d[t, L])

                for ret in zra['returns']:
                    if ret['problem'] == prob:
                        t = int(ret['tau'] * T)
                        if 0 <= t < T:
                            ret_deltas.append(d[t, L])

        if dep_deltas and ret_deltas:
            dep_mean = np.mean(dep_deltas)
            ret_mean = np.mean(ret_deltas)
            ratio = ret_mean / dep_mean if dep_mean > 0 else float('inf')
            results_matrix[L][cat] = {
                'dep_mean': float(dep_mean),
                'ret_mean': float(ret_mean),
                'ratio': float(ratio),
                'n_dep': len(dep_deltas),
                'n_ret': len(ret_deltas),
            }

# Print the matrix
print(f"\n{'Layer':>6} | ", end='')
for cat in categories:
    print(f"{cat:>14s} ", end='')
print(f"| {'spread':>8}")
print("-" * 100)

layer_spreads = {}
for L in range(n_layers):
    ratios = []
    print(f"L{L:02d}    | ", end='')
    for cat in categories:
        if cat in results_matrix[L]:
            r = results_matrix[L][cat]['ratio']
            ratios.append(r)
            # Color-code: low ratio = strong departure engine
            if r < 0.5:
                marker = "**"
            elif r > 1.2:
                marker = "!!"
            else:
                marker = "  "
            print(f"{marker}{r:>10.3f}{marker} ", end='')
        else:
            print(f"{'---':>14s} ", end='')
    spread = max(ratios) - min(ratios) if len(ratios) > 1 else 0
    layer_spreads[L] = spread
    print(f"| {spread:>8.3f}")

# ══════════════════════════════════════════════════════════════════
# WHERE IS THE CATEGORY EFFECT STRONGEST?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CATEGORY SENSITIVITY BY LAYER (spread of ratios across categories)")
print("=" * 80)

sorted_layers = sorted(layer_spreads.items(), key=lambda x: -x[1])
for L, spread in sorted_layers[:10]:
    print(f"  L{L:02d}: spread = {spread:.3f}")
    for cat in categories:
        if cat in results_matrix[L]:
            r = results_matrix[L][cat]['ratio']
            print(f"       {cat:>14s}: {r:.3f}")

# ══════════════════════════════════════════════════════════════════
# ARITHMETIC vs GEOMETRY: THE EXTREMES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ARITHMETIC vs GEOMETRY RATIO BY LAYER (most vs least asymmetric)")
print("=" * 80)

print(f"\n{'Layer':>6} | {'Arithmetic':>12} | {'Geometry':>12} | {'Diff':>8} | {'Arith dep':>10} | {'Arith ret':>10}")
print("-" * 75)
for L in range(n_layers):
    arith = results_matrix[L].get('arithmetic', {})
    geom = results_matrix[L].get('geometry', {})
    if arith and geom:
        ar = arith['ratio']
        gr = geom['ratio']
        print(f"L{L:02d}    | {ar:>12.3f} | {gr:>12.3f} | {gr-ar:>+8.3f} | {arith['dep_mean']:>10.1f} | {arith['ret_mean']:>10.1f}")

# ══════════════════════════════════════════════════════════════════
# RAW DELTA MAGNITUDES BY LAYER AND CATEGORY (departures only)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DEPARTURE DELTA MAGNITUDE BY LAYER (top 10 most variable layers)")
print("=" * 80)

# Compute coefficient of variation of departure deltas across categories per layer
layer_cv = {}
for L in range(n_layers):
    dep_means = [results_matrix[L][cat]['dep_mean'] for cat in categories if cat in results_matrix[L]]
    if dep_means:
        layer_cv[L] = np.std(dep_means) / np.mean(dep_means)

sorted_cv = sorted(layer_cv.items(), key=lambda x: -x[1])
for L, cv in sorted_cv[:10]:
    print(f"\n  L{L:02d} (CV={cv:.3f}):")
    for cat in categories:
        if cat in results_matrix[L]:
            dm = results_matrix[L][cat]['dep_mean']
            print(f"       {cat:>14s}: dep δ = {dm:.1f}")

# ══════════════════════════════════════════════════════════════════
# THE PHASE STORY: Does category effect align with three phases?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE ALIGNMENT")
print("=" * 80)

phase_a = [0, 1]
phase_b = list(range(2, 30))
phase_c = list(range(30, 36))

for phase_name, layers in [("A (embed, L0-1)", phase_a),
                            ("B (fragment, L2-29)", phase_b),
                            ("C (reassembly, L30-35)", phase_c)]:
    print(f"\n  Phase {phase_name}:")
    phase_spreads = [layer_spreads[L] for L in layers]
    print(f"    Mean category spread: {np.mean(phase_spreads):.3f} ± {np.std(phase_spreads):.3f}")
    print(f"    Max spread at: L{layers[np.argmax(phase_spreads)]}")

    # Average ratio per category in this phase
    for cat in categories:
        cat_ratios = [results_matrix[L][cat]['ratio'] for L in layers if cat in results_matrix[L]]
        if cat_ratios:
            print(f"    {cat:>14s}: mean ratio = {np.mean(cat_ratios):.3f}")


# ══════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════
output = {
    'layer_category_matrix': {
        str(L): results_matrix[L] for L in range(n_layers)
    },
    'layer_spreads': {str(L): float(v) for L, v in layer_spreads.items()},
    'layer_cv': {str(L): float(v) for L, v in layer_cv.items()},
}
with open('output/all_layers_directional.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\n\nSaved to output/all_layers_directional.json")
