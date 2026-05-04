"""
CRITICAL CONTROL: Is the departure rail departure-specific, or just
"how this model moves"?

Compare PCA on:
1. Departure-filtered deltas (594 vectors → 91.6% in PC1)
2. Return-filtered deltas (424 vectors)
3. ALL deltas (every consecutive step, thousands of vectors)
4. NEUTRAL deltas (steps that are neither departure nor return)

If ALL deltas also show 91% in PC1, the rail is generic model dynamics.
If ALL deltas show much less, the rail is departure-specific.

Also: compare the PC1 directions — are they the same axis or different?
"""

import numpy as np
import json
from sklearn.decomposition import PCA

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

# Build set of departure/return timesteps per problem
dep_steps = {}  # (prob, lang) -> set of timesteps
ret_steps = {}

prob_list = sorted(set(d['problem'] for d in zra['departures']))

for prob in prob_list:
    for lang in ['zh', 'en']:
        key = f'h32_prob{prob}_{lang}'
        if key not in deltas_data:
            continue
        h = deltas_data[key]
        T = len(h)

        dep_steps[(prob, lang)] = set()
        ret_steps[(prob, lang)] = set()

        for dep in zra['departures']:
            if dep['problem'] == prob:
                t = int(dep['tau'] * T)
                if 0 < t < T - 1:
                    dep_steps[(prob, lang)].add(t)

        for ret in zra['returns']:
            if ret['problem'] == prob:
                t = int(ret['tau'] * T)
                if 0 < t < T - 1:
                    ret_steps[(prob, lang)].add(t)

# ── Collect ALL delta vectors ────────────────────────────────────
print("Collecting delta vectors...")

all_deltas = []
departure_deltas = []
return_deltas = []
neutral_deltas = []

for prob in prob_list:
    for lang in ['zh', 'en']:
        key = f'h32_prob{prob}_{lang}'
        if key not in deltas_data:
            continue
        h = deltas_data[key]  # (T, 2048)
        T = len(h)

        ds = dep_steps.get((prob, lang), set())
        rs = ret_steps.get((prob, lang), set())

        for t in range(T - 1):
            delta = h[t + 1] - h[t]
            all_deltas.append(delta)

            if t in ds:
                departure_deltas.append(delta)
            elif t in rs:
                return_deltas.append(delta)
            else:
                neutral_deltas.append(delta)

all_deltas = np.array(all_deltas)
departure_deltas = np.array(departure_deltas)
return_deltas = np.array(return_deltas)
neutral_deltas = np.array(neutral_deltas)

print(f"ALL deltas:       {all_deltas.shape[0]:>6} vectors")
print(f"Departure deltas: {departure_deltas.shape[0]:>6} vectors")
print(f"Return deltas:    {return_deltas.shape[0]:>6} vectors")
print(f"Neutral deltas:   {neutral_deltas.shape[0]:>6} vectors")

# ── PCA on each set ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("PCA COMPARISON: THE CRITICAL CONTROL")
print("=" * 70)

results = {}
pca_objects = {}

for name, vectors in [("ALL", all_deltas),
                       ("DEPARTURE", departure_deltas),
                       ("RETURN", return_deltas),
                       ("NEUTRAL", neutral_deltas)]:
    n_comp = min(20, len(vectors) - 1)
    pca = PCA(n_components=n_comp)
    centered = vectors - vectors.mean(axis=0)
    pca.fit(centered)
    pca_objects[name] = pca

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"\n{name} ({len(vectors)} vectors):")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]:.1%})")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"  PC3: {pca.explained_variance_ratio_[2]:.4f}")
    print(f"  PC1-3: {cumvar[2]:.4f} ({cumvar[2]:.1%})")
    print(f"  PC1-5: {cumvar[4]:.4f} ({cumvar[4]:.1%})")
    print(f"  PC1-10: {cumvar[9]:.4f} ({cumvar[9]:.1%})")

    results[name] = {
        'n_vectors': len(vectors),
        'pc1_var': float(pca.explained_variance_ratio_[0]),
        'pc2_var': float(pca.explained_variance_ratio_[1]),
        'pc3_var': float(pca.explained_variance_ratio_[2]),
        'top5_var': float(cumvar[4]),
        'top10_var': float(cumvar[9]),
    }

# ── Compare PC1 directions across sets ───────────────────────────
print("\n" + "=" * 70)
print("PC1 DIRECTION COMPARISON")
print("=" * 70)

names = ["ALL", "DEPARTURE", "RETURN", "NEUTRAL"]
print(f"\n{'':>12}", end='')
for n in names:
    print(f"  {n:>10}", end='')
print()

for n1 in names:
    print(f"{n1:>12}", end='')
    for n2 in names:
        pc1_1 = pca_objects[n1].components_[0]
        pc1_2 = pca_objects[n2].components_[0]
        cos = abs(np.dot(pc1_1, pc1_2) / (np.linalg.norm(pc1_1) * np.linalg.norm(pc1_2)))
        print(f"  {cos:>10.4f}", end='')
    print()

# Also check: top 3 PCs subspace overlap
print(f"\nTop-3 subspace overlap:")
for n1 in names:
    print(f"{n1:>12}", end='')
    for n2 in names:
        b1 = pca_objects[n1].components_[:3]
        b2 = pca_objects[n2].components_[:3]
        svs = np.linalg.svd(b1 @ b2.T, compute_uv=False)
        print(f"  {svs.mean():>10.4f}", end='')
    print()

# ── The verdict ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

all_pc1 = results['ALL']['pc1_var']
dep_pc1 = results['DEPARTURE']['pc1_var']
neut_pc1 = results['NEUTRAL']['pc1_var']
ratio = dep_pc1 / all_pc1

print(f"\nALL deltas PC1:       {all_pc1:.4f} ({all_pc1:.1%})")
print(f"DEPARTURE deltas PC1: {dep_pc1:.4f} ({dep_pc1:.1%})")
print(f"NEUTRAL deltas PC1:   {neut_pc1:.4f} ({neut_pc1:.1%})")
print(f"RETURN deltas PC1:    {results['RETURN']['pc1_var']:.4f} ({results['RETURN']['pc1_var']:.1%})")
print(f"\nRatio departure/all: {ratio:.2f}x")

if all_pc1 > 0.8:
    print("\n⚠️  ALL deltas also dominated by PC1 — the rail may be GENERIC model dynamics")
    print("   The 91.6% departure PC1 may NOT be departure-specific.")

    # But check: is it the SAME direction?
    cos_all_dep = abs(np.dot(pca_objects['ALL'].components_[0],
                             pca_objects['DEPARTURE'].components_[0]))
    print(f"\n   BUT: cos(ALL_PC1, DEPARTURE_PC1) = {cos_all_dep:.4f}")
    if cos_all_dep > 0.95:
        print("   Same axis → the rail IS generic. Departures just ride the same rail as everything else.")
    else:
        print("   Different axes → even though ALL has high PC1, departures use a DIFFERENT direction.")

elif all_pc1 < 0.3:
    print("\n✓ ALL deltas are diffuse — the rail IS departure-specific!")
    print(f"  Departures concentrate {ratio:.0f}x more variance into PC1 than generic model movement.")
else:
    print(f"\n◐ Intermediate case. ALL PC1 = {all_pc1:.1%}, departures = {dep_pc1:.1%}")
    print(f"  The rail exists in general dynamics but departures amplify it {ratio:.1f}x")

# ── Bonus: per-dimension analysis ────────────────────────────────
print("\n" + "=" * 70)
print("DIMENSION ANALYSIS: Are dims 318/1874/1819 special in ALL deltas too?")
print("=" * 70)

for name in ['ALL', 'DEPARTURE', 'NEUTRAL']:
    pc1 = pca_objects[name].components_[0]
    top3_dims = np.argsort(np.abs(pc1))[::-1][:5]
    print(f"\n{name} PC1 top 5 dimensions:")
    for d in top3_dims:
        print(f"  Dim {d:4d}: weight = {pc1[d]:+.4f} (|w| = {abs(pc1[d]):.4f})")

# ── Magnitude comparison ────────────────────────────────────────
print("\n" + "=" * 70)
print("MAGNITUDE: How big are departure deltas vs neutral?")
print("=" * 70)

dep_norms = np.linalg.norm(departure_deltas, axis=1)
ret_norms = np.linalg.norm(return_deltas, axis=1)
neut_norms = np.linalg.norm(neutral_deltas, axis=1)
all_norms = np.linalg.norm(all_deltas, axis=1)

print(f"ALL:       mean ||δ|| = {all_norms.mean():.1f} ± {all_norms.std():.1f}")
print(f"DEPARTURE: mean ||δ|| = {dep_norms.mean():.1f} ± {dep_norms.std():.1f}")
print(f"RETURN:    mean ||δ|| = {ret_norms.mean():.1f} ± {ret_norms.std():.1f}")
print(f"NEUTRAL:   mean ||δ|| = {neut_norms.mean():.1f} ± {neut_norms.std():.1f}")
print(f"\nDeparture/Neutral magnitude ratio: {dep_norms.mean()/neut_norms.mean():.2f}x")

# Save
results['direction_comparison'] = {}
for n1 in names:
    for n2 in names:
        cos = abs(np.dot(pca_objects[n1].components_[0], pca_objects[n2].components_[0]))
        results['direction_comparison'][f'{n1}_vs_{n2}'] = float(cos)

results['magnitude'] = {
    'all_mean': float(all_norms.mean()),
    'departure_mean': float(dep_norms.mean()),
    'return_mean': float(ret_norms.mean()),
    'neutral_mean': float(neut_norms.mean()),
}

with open('output/rail_control_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: output/rail_control_results.json")
