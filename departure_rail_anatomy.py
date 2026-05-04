"""
Anatomy of the Departure Rail.

The departure engine pushes along a SINGLE direction in R^2048 (91.6% variance).
What IS that direction? What does it mean? How does it relate to everything else?

1. Identify the rail (departure PC1)
2. Project ALL generation tokens onto it — the rollercoaster
3. Decompose: how much is language, how much is category, how much is unique?
4. Visualize the ride for every problem
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from collections import defaultdict

# ── Load everything ───────────────────────────────────────────────
deltas_data = np.load('output/layer_deltas.npz', allow_pickle=True)
with open('output/z_return_analysis.json') as f:
    zra = json.load(f)
with open('output/gen_trajectories_meta.json') as f:
    meta = json.load(f)
all_layers = np.load('output/all_layers.npz', allow_pickle=True)

prob_categories = {}
for key, val in meta.items():
    if '_zh' in key:
        prob_categories[val['problem_idx']] = val['category']

categories = sorted(set(prob_categories.values()))
cat_colors = {
    'arithmetic': '#e74c3c',
    'combinatorics': '#3498db',
    'geometry': '#2ecc71',
    'modular': '#9b59b6',
    'sequences': '#f39c12',
}

# ── Step 1: Extract the rail ─────────────────────────────────────
print("=" * 70)
print("STEP 1: EXTRACT THE DEPARTURE RAIL")
print("=" * 70)

# Collect ALL δ vectors at L32 during departure events
departure_vectors = []
dep_meta = []  # track which problem/lang each comes from

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
            delta = h[t + 1] - h[t]
            departure_vectors.append(delta)
            dep_meta.append({'prob': prob, 'lang': lang, 'tau': dep['tau'],
                           'cat': prob_categories.get(prob, '?')})

departure_vectors = np.array(departure_vectors)
print(f"Departure vectors: {departure_vectors.shape}")

# PCA
pca = PCA(n_components=10)
dep_centered = departure_vectors - departure_vectors.mean(axis=0)
pca.fit(dep_centered)
rail = pca.components_[0]  # THE RAIL — the dominant departure direction
rail_normed = rail / np.linalg.norm(rail)

print(f"Rail variance: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Rail norm: {np.linalg.norm(rail):.2f}")

# ── Step 2: Project ALL tokens onto the rail ─────────────────────
print("\n" + "=" * 70)
print("STEP 2: THE ROLLERCOASTER — every token projected onto the rail")
print("=" * 70)

fig, axes = plt.subplots(5, 4, figsize=(24, 20))
fig.suptitle('The Departure Rail: Every Problem\'s Ride\n'
             f'(PC1 = {pca.explained_variance_ratio_[0]:.1%} of departure variance)',
             fontsize=16, fontweight='bold')

# Also collect stats
all_projections = {}
prob_list = sorted(set(d['problem'] for d in zra['departures']))

for idx, prob in enumerate(prob_list):
    ax = axes[idx // 4, idx % 4]
    cat = prob_categories.get(prob, '?')
    color = cat_colors.get(cat, 'gray')

    for lang, ls in [('zh', '-'), ('en', '--')]:
        key = f'h32_prob{prob}_{lang}'
        if key not in deltas_data:
            continue
        h = deltas_data[key]  # (T, 2048)
        # Project each hidden state onto the rail
        proj = h @ rail_normed
        tau = np.linspace(0, 1, len(proj))
        ax.plot(tau, proj, ls, color=color, alpha=0.8, linewidth=1.5,
                label=f'{lang}')
        all_projections[f'{prob}_{lang}'] = proj

    # Mark departure and return events
    for dep in zra['departures']:
        if dep['problem'] == prob:
            ax.axvline(dep['tau'], color='red', alpha=0.3, linewidth=0.5)
    for ret in zra['returns']:
        if ret['problem'] == prob:
            ax.axvline(ret['tau'], color='blue', alpha=0.3, linewidth=0.5)

    ax.set_title(f"P{prob} ({cat})", fontsize=10, color=color, fontweight='bold')
    ax.set_xlabel('τ')
    if idx % 4 == 0:
        ax.set_ylabel('Rail projection')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('output/departure_rail_rollercoaster.png', dpi=150)
plt.close()
print("Saved: output/departure_rail_rollercoaster.png")

# ── Step 3: Rail projection statistics ───────────────────────────
print("\n" + "=" * 70)
print("STEP 3: RAIL STATISTICS")
print("=" * 70)

# Compute: range of projection along rail per problem/lang
print(f"\n{'Prob':>5} {'Cat':>14} {'Lang':>4} | {'Mean':>8} {'Std':>8} {'Range':>8} | {'Min':>8} {'Max':>8}")
print("-" * 75)

cat_ranges = defaultdict(list)
for prob in prob_list:
    cat = prob_categories.get(prob, '?')
    for lang in ['zh', 'en']:
        key = f'{prob}_{lang}'
        if key in all_projections:
            p = all_projections[key]
            rng = p.max() - p.min()
            cat_ranges[cat].append(rng)
            print(f"{prob:>5} {cat:>14} {lang:>4} | {p.mean():>8.1f} {p.std():>8.1f} {rng:>8.1f} | {p.min():>8.1f} {p.max():>8.1f}")

print(f"\n{'Category':>14} | {'Mean Range':>10} {'Std Range':>10}")
print("-" * 40)
for cat in categories:
    if cat in cat_ranges:
        r = cat_ranges[cat]
        print(f"{cat:>14} | {np.mean(r):>10.1f} {np.std(r):>10.1f}")

# ── Step 4: Decompose the rail ───────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: WHAT IS THE RAIL MADE OF?")
print("=" * 70)

# 4a: Language component
zh_L32 = all_layers['zh_L32']  # (200, 2048)
en_L32 = all_layers['en_L32']  # (200, 2048)
lang_diff = (zh_L32 - en_L32).mean(axis=0)
lang_diff_normed = lang_diff / np.linalg.norm(lang_diff)

cos_rail_lang = np.dot(rail_normed, lang_diff_normed)
print(f"\nCosine(rail, zh-en language axis): {cos_rail_lang:+.4f}")
print(f"  |cos| = {abs(cos_rail_lang):.4f}")

# 4b: Category components
cats_arr = all_layers['categories']  # (200,)
cat_id_to_name = {0: 'arithmetic', 1: 'combinatorics', 2: 'modular', 3: 'geometry', 4: 'sequences'}

# Grand mean
grand_mean = np.concatenate([zh_L32, en_L32]).mean(axis=0)

print(f"\nCategory centroids projected onto rail:")
for lang_name, acts in [('zh', zh_L32), ('en', en_L32)]:
    print(f"  {lang_name}:")
    for cat_id in sorted(np.unique(cats_arr)):
        mask = cats_arr == cat_id
        centroid = acts[mask].mean(axis=0)
        proj = np.dot(centroid - grand_mean, rail_normed)
        cat_name = cat_id_to_name.get(cat_id, f'cat{cat_id}')
        print(f"    {cat_name:>14}: projection = {proj:+.2f}")

# 4c: What are the top-loading dimensions?
top_dims = np.argsort(np.abs(rail))[::-1][:20]
print(f"\nTop 20 dimensions of the rail (by |weight|):")
for i, dim in enumerate(top_dims):
    print(f"  Dim {dim:4d}: weight = {rail[dim]:+.4f}  |w| = {abs(rail[dim]):.4f}")

# 4d: Rail vs global PCA of input-pass activations
all_acts = np.concatenate([zh_L32, en_L32])  # (400, 2048)
pca_global = PCA(n_components=20)
pca_global.fit(all_acts - all_acts.mean(axis=0))

print(f"\nRail alignment with input-pass PCA components:")
for i in range(10):
    cos_i = abs(np.dot(rail_normed, pca_global.components_[i]))
    print(f"  PC{i+1}: |cos| = {cos_i:.4f}")

# ── Step 5: Overlay zh and en projections to show synchrony ──────
print("\n" + "=" * 70)
print("STEP 5: ZH-EN SYNCHRONY ON THE RAIL")
print("=" * 70)

fig, axes = plt.subplots(4, 5, figsize=(25, 16))
fig.suptitle('Zh-En Rail Synchrony: Are both languages riding the same rollercoaster?',
             fontsize=14, fontweight='bold')

for idx, prob in enumerate(prob_list):
    ax = axes[idx // 5, idx % 5]
    cat = prob_categories.get(prob, '?')

    key_zh = f'{prob}_zh'
    key_en = f'{prob}_en'
    if key_zh in all_projections and key_en in all_projections:
        p_zh = all_projections[key_zh]
        p_en = all_projections[key_en]

        # Normalize to [0,1] τ
        tau_zh = np.linspace(0, 1, len(p_zh))
        tau_en = np.linspace(0, 1, len(p_en))

        ax.plot(tau_zh, p_zh, '-', color='red', alpha=0.8, linewidth=1.5, label='zh')
        ax.plot(tau_en, p_en, '--', color='blue', alpha=0.8, linewidth=1.5, label='en')

        # Correlation between interpolated projections
        from scipy.interpolate import interp1d
        common_tau = np.linspace(0, 1, 100)
        f_zh = interp1d(tau_zh, p_zh, kind='linear', fill_value='extrapolate')
        f_en = interp1d(tau_en, p_en, kind='linear', fill_value='extrapolate')
        corr = np.corrcoef(f_zh(common_tau), f_en(common_tau))[0, 1]

        ax.set_title(f"P{prob} ({cat}) r={corr:.2f}", fontsize=9,
                    color=cat_colors.get(cat, 'gray'), fontweight='bold')
    else:
        ax.set_title(f"P{prob} (no data)")

    if idx % 5 == 0:
        ax.set_ylabel('Rail proj')
    if idx >= 15:
        ax.set_xlabel('τ')
    if idx == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('output/departure_rail_synchrony.png', dpi=150)
plt.close()
print("Saved: output/departure_rail_synchrony.png")

# Compute all synchrony correlations
print(f"\n{'Prob':>5} {'Cat':>14} {'r(zh,en)':>10}")
print("-" * 35)
sync_corrs = []
sync_by_cat = defaultdict(list)
for prob in prob_list:
    cat = prob_categories.get(prob, '?')
    key_zh = f'{prob}_zh'
    key_en = f'{prob}_en'
    if key_zh in all_projections and key_en in all_projections:
        from scipy.interpolate import interp1d
        p_zh = all_projections[key_zh]
        p_en = all_projections[key_en]
        common_tau = np.linspace(0, 1, 100)
        f_zh = interp1d(np.linspace(0, 1, len(p_zh)), p_zh, fill_value='extrapolate')
        f_en = interp1d(np.linspace(0, 1, len(p_en)), p_en, fill_value='extrapolate')
        corr = np.corrcoef(f_zh(common_tau), f_en(common_tau))[0, 1]
        sync_corrs.append(corr)
        sync_by_cat[cat].append(corr)
        print(f"{prob:>5} {cat:>14} {corr:>+10.4f}")

print(f"\nOverall zh-en rail synchrony: r = {np.mean(sync_corrs):+.4f} ± {np.std(sync_corrs):.4f}")
for cat in categories:
    if cat in sync_by_cat:
        print(f"  {cat:>14}: r = {np.mean(sync_by_cat[cat]):+.4f}")

# ── Step 6: The combined figure ──────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: HERO FIGURE — Category × Rail")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

# Panel A: All problems on rail, colored by category
ax1 = fig.add_subplot(gs[0, :])
for prob in prob_list:
    cat = prob_categories.get(prob, '?')
    color = cat_colors.get(cat, 'gray')
    for lang, alpha in [('zh', 0.9), ('en', 0.5)]:
        key = f'{prob}_{lang}'
        if key in all_projections:
            p = all_projections[key]
            tau = np.linspace(0, 1, len(p))
            ax1.plot(tau, p, color=color, alpha=alpha, linewidth=0.8)

# Legend
for cat, color in cat_colors.items():
    ax1.plot([], [], color=color, linewidth=2, label=cat)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlabel('τ (generation progress)', fontsize=12)
ax1.set_ylabel('Projection onto Departure Rail', fontsize=12)
ax1.set_title(f'All 40 Trajectories on the Departure Rail (PC1 = {pca.explained_variance_ratio_[0]:.1%} of variance)',
              fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3)

# Panel B: Category mean trajectories
ax2 = fig.add_subplot(gs[1, 0])
for cat in categories:
    cat_probs = [p for p, c in prob_categories.items() if c == cat]
    cat_trajs = []
    for prob in cat_probs:
        for lang in ['zh', 'en']:
            key = f'{prob}_{lang}'
            if key in all_projections:
                p = all_projections[key]
                from scipy.interpolate import interp1d
                f = interp1d(np.linspace(0, 1, len(p)), p, fill_value='extrapolate')
                cat_trajs.append(f(np.linspace(0, 1, 100)))
    if cat_trajs:
        cat_trajs = np.array(cat_trajs)
        mean_traj = cat_trajs.mean(axis=0)
        std_traj = cat_trajs.std(axis=0)
        tau100 = np.linspace(0, 1, 100)
        ax2.plot(tau100, mean_traj, color=cat_colors[cat], linewidth=2, label=cat)
        ax2.fill_between(tau100, mean_traj - std_traj, mean_traj + std_traj,
                        color=cat_colors[cat], alpha=0.15)
ax2.legend(fontsize=9)
ax2.set_xlabel('τ')
ax2.set_ylabel('Mean Rail Projection')
ax2.set_title('Category Mean ± Std on Rail', fontsize=12, fontweight='bold')

# Panel C: Range of ride by category (bar chart)
ax3 = fig.add_subplot(gs[1, 1])
cat_range_means = []
cat_range_stds = []
for cat in categories:
    r = cat_ranges.get(cat, [0])
    cat_range_means.append(np.mean(r))
    cat_range_stds.append(np.std(r))

bars = ax3.bar(range(len(categories)), cat_range_means, yerr=cat_range_stds,
               color=[cat_colors[c] for c in categories], alpha=0.8, capsize=5)
ax3.set_xticks(range(len(categories)))
ax3.set_xticklabels(categories, rotation=30, ha='right')
ax3.set_ylabel('Rail Excursion Range')
ax3.set_title('How Far Each Category Rides the Rail', fontsize=12, fontweight='bold')

plt.savefig('output/departure_rail_hero.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/departure_rail_hero.png")

# ── Save results ─────────────────────────────────────────────────
results = {
    'rail_variance_explained': float(pca.explained_variance_ratio_[0]),
    'rail_top10_variance': [float(v) for v in pca.explained_variance_ratio_],
    'cos_rail_language_axis': float(cos_rail_lang),
    'category_ranges': {cat: {'mean': float(np.mean(r)), 'std': float(np.std(r))}
                       for cat, r in cat_ranges.items()},
    'zh_en_synchrony': {
        'overall_mean': float(np.mean(sync_corrs)),
        'overall_std': float(np.std(sync_corrs)),
        'by_category': {cat: float(np.mean(v)) for cat, v in sync_by_cat.items()},
    },
    'top_20_rail_dims': [{'dim': int(d), 'weight': float(rail[d])} for d in top_dims],
    'rail_vs_global_pca': [float(abs(np.dot(rail_normed, pca_global.components_[i])))
                           for i in range(10)],
}
with open('output/departure_rail_anatomy.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: output/departure_rail_anatomy.json")

print("\n" + "=" * 70)
print("DONE. Three figures + full anatomy.")
print("=" * 70)
