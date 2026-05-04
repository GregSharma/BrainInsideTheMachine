"""
Four Directional Questions about the Departure Engine
All computed from cached data (no GPU needed).

Q1: Is the departure asymmetry category-dependent?
Q2: What DIRECTION do departures point at L32?
Q3: Does the return path retrace the departure path?
Q4: Is the L35 compass one direction or problem-specific?

Uses: output/layer_deltas.npz (norms + h32 vectors)
      output/z_return_analysis.json (return/departure events)
      output/gen_trajectories_meta.json (category labels)
"""

import numpy as np
import json
from collections import defaultdict
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────
deltas_data = np.load('output/layer_deltas.npz', allow_pickle=True)
with open('output/z_return_analysis.json') as f:
    zra = json.load(f)
with open('output/gen_trajectories_meta.json') as f:
    meta = json.load(f)

# Build problem → category map (problems 0-19, zh/en only used in layer_deltas)
prob_categories = {}
for key, val in meta.items():
    if '_zh' in key:
        prob_idx = val['problem_idx']
        prob_categories[prob_idx] = val['category']

print("Problem categories:", prob_categories)
print(f"Categories present: {sorted(set(prob_categories.values()))}")
print()

# ── Helper: get cosine spike series for a problem ──────────────────
def get_cosine_series(prob_idx):
    """Compute zh-en cosine similarity at L32 for each zh token."""
    h_zh = deltas_data[f'h32_prob{prob_idx}_zh']  # (T_zh, 2048)
    h_en = deltas_data[f'h32_prob{prob_idx}_en']  # (T_en, 2048)
    T = min(len(h_zh), len(h_en))
    cos = np.array([
        np.dot(h_zh[t], h_en[t]) / (np.linalg.norm(h_zh[t]) * np.linalg.norm(h_en[t]) + 1e-10)
        for t in range(T)
    ])
    return cos

# ══════════════════════════════════════════════════════════════════
# Q1: CATEGORY-DEPENDENT ASYMMETRY
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Q1: IS THE DEPARTURE ASYMMETRY CATEGORY-DEPENDENT?")
print("=" * 70)

# Group returns and departures by category
cat_returns = defaultdict(list)
cat_departures = defaultdict(list)

for r in zra['returns']:
    prob = r['problem']
    cat = prob_categories.get(prob, 'unknown')
    cat_returns[cat].append(r)

for d in zra['departures']:
    prob = d['problem']
    cat = prob_categories.get(prob, 'unknown')
    cat_departures[cat].append(d)

# Compute per-category departure/return ratio at L30-34
# We need delta norms at these layers for departure vs return transitions
print("\nPer-category departure engine activity (L30-34):")
print("-" * 60)

cat_ratios = {}
for cat in sorted(set(prob_categories.values())):
    # Get problems in this category
    cat_probs = [p for p, c in prob_categories.items() if c == cat]

    # For each problem, compute avg delta at L30-34 during departure vs return transitions
    dep_deltas_L30_34 = []
    ret_deltas_L30_34 = []

    for prob in cat_probs:
        for lang in ['zh', 'en']:
            key = f'deltas_prob{prob}_{lang}'
            if key not in deltas_data:
                continue
            d = deltas_data[key]  # (T, 36)
            T = len(d)

            # Get departure/return events for this problem
            for dep in zra['departures']:
                if dep['problem'] == prob:
                    t = int(dep['tau'] * T)
                    if 0 <= t < T:
                        dep_deltas_L30_34.append(d[t, 30:35].mean())

            for ret in zra['returns']:
                if ret['problem'] == prob:
                    t = int(ret['tau'] * T)
                    if 0 <= t < T:
                        ret_deltas_L30_34.append(d[t, 30:35].mean())

    if dep_deltas_L30_34 and ret_deltas_L30_34:
        dep_mean = np.mean(dep_deltas_L30_34)
        ret_mean = np.mean(ret_deltas_L30_34)
        ratio = ret_mean / dep_mean if dep_mean > 0 else float('inf')
        cat_ratios[cat] = {
            'dep_mean': float(dep_mean),
            'ret_mean': float(ret_mean),
            'ratio_ret_over_dep': float(ratio),
            'n_departures': len(dep_deltas_L30_34),
            'n_returns': len(ret_deltas_L30_34),
            'n_problems': len(cat_probs)
        }
        print(f"  {cat:15s}: dep={dep_mean:.4f}  ret={ret_mean:.4f}  "
              f"ratio={ratio:.3f}  (n_dep={len(dep_deltas_L30_34)}, n_ret={len(ret_deltas_L30_34)})")

# Are the ratios significantly different across categories?
ratio_vals = [v['ratio_ret_over_dep'] for v in cat_ratios.values()]
print(f"\n  Ratio range: {min(ratio_vals):.3f} — {max(ratio_vals):.3f}")
print(f"  Ratio spread: {max(ratio_vals) - min(ratio_vals):.3f}")

# Also compute: mean |delta_cos| at departures vs returns per category
print("\n  Mean |Δcos| at transitions per category:")
for cat in sorted(cat_ratios.keys()):
    cat_probs = [p for p, c in prob_categories.items() if c == cat]
    dep_dcos = [abs(d['delta_cos']) for d in zra['departures'] if prob_categories.get(d['problem']) == cat]
    ret_dcos = [abs(r['delta_cos']) for r in zra['returns'] if prob_categories.get(r['problem']) == cat]
    if dep_dcos and ret_dcos:
        print(f"    {cat:15s}: dep Δcos={np.mean(dep_dcos):.4f}  ret Δcos={np.mean(ret_dcos):.4f}")


# ══════════════════════════════════════════════════════════════════
# Q2: DEPARTURE DIRECTION PCA AT L32
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q2: WHAT DIRECTION DO DEPARTURES POINT AT L32?")
print("=" * 70)

# Collect departure δ vectors at L32: h32(t+1) - h32(t) at departure events
departure_vectors = []
return_vectors = []

for dep in zra['departures']:
    prob = dep['problem']
    for lang in ['zh', 'en']:
        key = f'h32_prob{prob}_{lang}'
        if key not in deltas_data:
            continue
        h = deltas_data[key]  # (T, 2048)
        T = len(h)
        t = int(dep['tau'] * T)
        if 0 < t < T - 1:
            delta = h[t + 1] - h[t]
            departure_vectors.append(delta)

for ret in zra['returns']:
    prob = ret['problem']
    for lang in ['zh', 'en']:
        key = f'h32_prob{prob}_{lang}'
        if key not in deltas_data:
            continue
        h = deltas_data[key]
        T = len(h)
        t = int(ret['tau'] * T)
        if 0 < t < T - 1:
            delta = h[t + 1] - h[t]
            return_vectors.append(delta)

departure_vectors = np.array(departure_vectors)  # (N_dep, 2048)
return_vectors = np.array(return_vectors)  # (N_ret, 2048)

print(f"\nCollected {len(departure_vectors)} departure vectors, {len(return_vectors)} return vectors")

# PCA on departure vectors
from sklearn.decomposition import PCA

dep_centered = departure_vectors - departure_vectors.mean(axis=0)
pca_dep = PCA(n_components=min(20, len(departure_vectors)))
pca_dep.fit(dep_centered)

print(f"\nDeparture PCA — variance explained by top components:")
cumvar = np.cumsum(pca_dep.explained_variance_ratio_)
for i in range(min(10, len(pca_dep.explained_variance_ratio_))):
    print(f"  PC{i+1}: {pca_dep.explained_variance_ratio_[i]:.4f}  (cumulative: {cumvar[i]:.4f})")

# Check if PC1 is dominant (>50% = consistent departure direction)
if pca_dep.explained_variance_ratio_[0] > 0.3:
    print(f"\n  → PC1 explains {pca_dep.explained_variance_ratio_[0]:.1%} — DOMINANT departure direction exists")
else:
    print(f"\n  → PC1 explains {pca_dep.explained_variance_ratio_[0]:.1%} — no dominant direction, departures are diffuse")

# Compare departure PC1 with the zh-en difference direction from input-pass
# Load multilingual activations to get zh-en difference
try:
    ml = np.load('output/multilingual_activations.npz', allow_pickle=True)
    # Get zh and en at L32
    zh_acts = ml['zh_L32']  # (200, 2048)
    en_acts = ml['en_L32']  # (200, 2048)
    lang_diff = (zh_acts - en_acts).mean(axis=0)  # average zh-en difference
    lang_diff_normed = lang_diff / np.linalg.norm(lang_diff)

    dep_pc1 = pca_dep.components_[0]
    dep_pc1_normed = dep_pc1 / np.linalg.norm(dep_pc1)

    cos_dep_lang = abs(np.dot(dep_pc1_normed, lang_diff_normed))
    print(f"\n  Cosine(departure_PC1, zh-en_difference): {cos_dep_lang:.4f}")
    if cos_dep_lang > 0.5:
        print(f"  → Departure direction ALIGNS with language difference!")
    elif cos_dep_lang > 0.2:
        print(f"  → Partial alignment with language difference")
    else:
        print(f"  → Departure direction is ORTHOGONAL to language difference")
except Exception as e:
    print(f"  Could not compare with language difference: {e}")


# ══════════════════════════════════════════════════════════════════
# Q3: DOES THE RETURN PATH RETRACE THE DEPARTURE PATH?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q3: DOES THE RETURN PATH RETRACE THE DEPARTURE PATH?")
print("=" * 70)

# Average departure direction vs average return direction
avg_dep = departure_vectors.mean(axis=0)
avg_ret = return_vectors.mean(axis=0)

cos_dep_ret = np.dot(avg_dep, avg_ret) / (np.linalg.norm(avg_dep) * np.linalg.norm(avg_ret) + 1e-10)
print(f"\nCosine(avg_departure, avg_return): {cos_dep_ret:.4f}")

if cos_dep_ret < -0.5:
    print("  → ANTI-PARALLEL: model reverses along same path (simple push/pull)")
elif abs(cos_dep_ret) < 0.3:
    print("  → ORTHOGONAL: model takes a DIFFERENT path back (complex attractor geometry)")
elif cos_dep_ret > 0.5:
    print("  → PARALLEL: departures and returns go the SAME direction (?!)")
else:
    print(f"  → Weakly {'anti-' if cos_dep_ret < 0 else ''}correlated")

# Also check: PCA on return vectors
ret_centered = return_vectors - return_vectors.mean(axis=0)
pca_ret = PCA(n_components=min(20, len(return_vectors)))
pca_ret.fit(ret_centered)

print(f"\nReturn PCA — variance explained by top components:")
cumvar_ret = np.cumsum(pca_ret.explained_variance_ratio_)
for i in range(min(5, len(pca_ret.explained_variance_ratio_))):
    print(f"  PC{i+1}: {pca_ret.explained_variance_ratio_[i]:.4f}  (cumulative: {cumvar_ret[i]:.4f})")

# Compare departure PC1 and return PC1
ret_pc1 = pca_ret.components_[0]
cos_pc1s = abs(np.dot(pca_dep.components_[0], ret_pc1)) / (
    np.linalg.norm(pca_dep.components_[0]) * np.linalg.norm(ret_pc1))
print(f"\n  |Cosine(departure_PC1, return_PC1)|: {cos_pc1s:.4f}")

# Full subspace overlap: top-k PCA subspace alignment
for k in [1, 3, 5]:
    dep_basis = pca_dep.components_[:k]  # (k, 2048)
    ret_basis = pca_ret.components_[:k]
    # Subspace overlap via principal angles
    overlap = np.linalg.svd(dep_basis @ ret_basis.T, compute_uv=False)
    mean_overlap = overlap.mean()
    print(f"  Top-{k} subspace overlap (mean singular value): {mean_overlap:.4f}")

# Per-problem paired analysis: for problems with both departure and return,
# compare the actual vectors
print("\n  Per-problem departure-return cosine:")
prob_dep_vecs = defaultdict(list)
prob_ret_vecs = defaultdict(list)

for dep in zra['departures']:
    prob = dep['problem']
    for lang in ['zh', 'en']:
        key = f'h32_prob{prob}_{lang}'
        if key not in deltas_data:
            continue
        h = deltas_data[key]
        T = len(h)
        t = int(dep['tau'] * T)
        if 0 < t < T - 1:
            prob_dep_vecs[prob].append(h[t + 1] - h[t])

for ret in zra['returns']:
    prob = ret['problem']
    for lang in ['zh', 'en']:
        key = f'h32_prob{prob}_{lang}'
        if key not in deltas_data:
            continue
        h = deltas_data[key]
        T = len(h)
        t = int(ret['tau'] * T)
        if 0 < t < T - 1:
            prob_ret_vecs[prob].append(h[t + 1] - h[t])

paired_cosines = []
for prob in sorted(set(prob_dep_vecs.keys()) & set(prob_ret_vecs.keys())):
    avg_d = np.mean(prob_dep_vecs[prob], axis=0)
    avg_r = np.mean(prob_ret_vecs[prob], axis=0)
    cos = np.dot(avg_d, avg_r) / (np.linalg.norm(avg_d) * np.linalg.norm(avg_r) + 1e-10)
    paired_cosines.append(cos)
    cat = prob_categories.get(prob, '?')
    print(f"    Problem {prob:2d} ({cat:12s}): cos = {cos:+.4f}")

print(f"\n  Mean paired cosine: {np.mean(paired_cosines):+.4f} ± {np.std(paired_cosines):.4f}")
print(f"  Median: {np.median(paired_cosines):+.4f}")


# ══════════════════════════════════════════════════════════════════
# Q4: IS THE L35 COMPASS ONE DIRECTION OR PROBLEM-SPECIFIC?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q4: IS THE L35 COMPASS ONE DIRECTION OR PROBLEM-SPECIFIC?")
print("=" * 70)

# We don't have L35 vectors, only L32. But L35 magnitude correlates with returns.
# Let's use L32 return vectors (already collected) and check if they point consistently.
# The "compass" question is: do returns have a fixed direction?

# We already have PCA on return vectors from Q3
print(f"\nReturn vector PCA (L32 proxy for L35 compass):")
print(f"  PC1 explains: {pca_ret.explained_variance_ratio_[0]:.1%}")
print(f"  PC1+PC2: {sum(pca_ret.explained_variance_ratio_[:2]):.1%}")
print(f"  PC1-PC5: {sum(pca_ret.explained_variance_ratio_[:5]):.1%}")

if pca_ret.explained_variance_ratio_[0] > 0.3:
    print(f"  → FIXED compass direction exists at L32")
else:
    print(f"  → Return directions are DIFFUSE — problem-specific correction, not fixed compass")

# Per-problem return direction consistency: how aligned are return vectors within a problem?
print(f"\n  Intra-problem return vector consistency:")
for prob in sorted(prob_ret_vecs.keys()):
    vecs = np.array(prob_ret_vecs[prob])
    if len(vecs) < 2:
        continue
    # Pairwise cosines within this problem
    cosines = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            c = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-10)
            cosines.append(c)
    cat = prob_categories.get(prob, '?')
    print(f"    Problem {prob:2d} ({cat:12s}): mean intra-cos = {np.mean(cosines):+.4f} (n={len(vecs)})")

# Cross-problem return direction: project all return vectors onto return PC1
# and check sign consistency
ret_projections = return_vectors @ pca_ret.components_[0]
frac_positive = (ret_projections > 0).mean()
print(f"\n  Fraction of returns with positive PC1 projection: {frac_positive:.3f}")
print(f"  (0.5 = random, >0.8 = consistent compass)")

# Compare return PC1 with language difference
try:
    ret_pc1_normed = pca_ret.components_[0] / np.linalg.norm(pca_ret.components_[0])
    cos_ret_lang = abs(np.dot(ret_pc1_normed, lang_diff_normed))
    print(f"\n  |Cosine(return_PC1, zh-en_difference)|: {cos_ret_lang:.4f}")
    if cos_ret_lang > 0.5:
        print(f"  → Return compass ALIGNS with language axis — returning to Z IS returning to language-neutral")
    elif cos_ret_lang > 0.2:
        print(f"  → Partial alignment — compass has language component")
    else:
        print(f"  → Return compass ORTHOGONAL to language axis — Z-return ≠ language-neutralization")
except:
    pass


# ══════════════════════════════════════════════════════════════════
# SYNTHESIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SYNTHESIS: WHAT THE FOUR QUESTIONS TELL US")
print("=" * 70)

results = {
    'Q1_category_asymmetry': cat_ratios,
    'Q2_departure_pca': {
        'n_vectors': len(departure_vectors),
        'pc1_variance': float(pca_dep.explained_variance_ratio_[0]),
        'pc1_pc2_variance': float(sum(pca_dep.explained_variance_ratio_[:2])),
        'top5_variance': float(sum(pca_dep.explained_variance_ratio_[:5])),
        'cos_dep_pc1_lang_diff': float(cos_dep_lang) if 'cos_dep_lang' in dir() else None,
    },
    'Q3_return_retraces': {
        'cos_avg_dep_ret': float(cos_dep_ret),
        'cos_pc1_dep_ret': float(cos_pc1s),
        'mean_paired_cosine': float(np.mean(paired_cosines)),
        'std_paired_cosine': float(np.std(paired_cosines)),
        'median_paired_cosine': float(np.median(paired_cosines)),
    },
    'Q4_compass': {
        'return_pc1_variance': float(pca_ret.explained_variance_ratio_[0]),
        'frac_positive_pc1': float(frac_positive),
        'cos_return_pc1_lang_diff': float(cos_ret_lang) if 'cos_ret_lang' in dir() else None,
    }
}

with open('output/four_directional_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to output/four_directional_results.json")
