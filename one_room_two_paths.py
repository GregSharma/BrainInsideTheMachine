"""
One Room, Two Paths — Learn the canonical geometry from generation-time data.

Instead of applying input-pass ridge maps to generation trajectories (wrong —
different distribution), learn the alignment from the generation data itself.

Approach:
1. Extract "answer states" — final h32 from each generation trajectory.
   These MUST align if Z exists: same problem, same answer, different language.
2. Fit ridge maps on answer states from TRAINING problems (15 of 20).
3. Test on HELD-OUT problems (5 of 20) — does alignment generalize?
4. If yes: apply the generation-fit map to full trajectories.
5. Joint diffusion map on mapped trajectories — do they blob together?

Also: Learn a KERNEL (CKA / centered kernel alignment) that discovers the
metric space where problems cluster by identity, not language. This is
Greg's insight — let the data tell us the right metric.
"""

import numpy as np
import json
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- LOAD DATA ----------
print("Loading generation trajectories...")
traj = np.load('output/gen_trajectories.npz')
with open('output/gen_trajectories_meta.json') as f:
    meta = json.load(f)

# Organize by problem and language
langs = ['zh', 'en', 'es', 'ja']
prob_indices = sorted(set(m['problem_idx'] for m in meta.values()))
print(f"Problems: {prob_indices}")
print(f"Languages: {langs}")

# ---------- EXTRACT ANSWER STATES ----------
# Use the LAST h32 state from each trajectory as the "answer state"
# Also extract "mid-reasoning" state (step at 50% of trajectory)
answer_states = {}  # {(prob_idx, lang): vector}
mid_states = {}
early_states = {}  # 20% through

for key in sorted(traj.keys()):
    m = meta[key]
    h32 = traj[key]
    prob_idx = m['problem_idx']
    lang = m['language']
    n = h32.shape[0]

    if n < 5:
        continue

    answer_states[(prob_idx, lang)] = h32[-1]
    mid_states[(prob_idx, lang)] = h32[n // 2]
    early_states[(prob_idx, lang)] = h32[max(1, n // 5)]

# Problems that have all 4 languages
complete_probs = [p for p in prob_indices
                  if all((p, l) in answer_states for l in langs)]
print(f"\nProblems with all 4 languages: {len(complete_probs)}")
print(f"  {complete_probs}")

# ---------- TRAIN/TEST SPLIT ----------
np.random.seed(42)
perm = np.random.permutation(complete_probs)
n_train = 15
train_probs = sorted(perm[:n_train])
test_probs = sorted(perm[n_train:])
print(f"\nTrain problems ({n_train}): {train_probs}")
print(f"Test problems ({len(test_probs)}): {test_probs}")

# ---------- EXPERIMENT 1: ANSWER STATE ALIGNMENT ----------
print("\n" + "=" * 70)
print("EXPERIMENT 1: Answer State Alignment")
print("=" * 70)
print("Fit ridge maps on ANSWER STATES from training problems.")
print("Test on held-out problems.\n")

results = {}

for state_name, states in [("answer", answer_states), ("mid", mid_states), ("early", early_states)]:
    print(f"\n--- {state_name.upper()} states ---")

    # Fit pairwise ridge maps on training data
    pair_results = {}
    for li, lang_i in enumerate(langs):
        for lj, lang_j in enumerate(langs):
            if li >= lj:
                continue

            # Training data
            X_train = np.array([states[(p, lang_i)] for p in train_probs])
            Y_train = np.array([states[(p, lang_j)] for p in train_probs])

            # Test data
            X_test = np.array([states[(p, lang_i)] for p in test_probs])
            Y_test = np.array([states[(p, lang_j)] for p in test_probs])

            # Fit ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, Y_train)

            # Train R²
            r2_train = r2_score(Y_train, ridge.predict(X_train))
            # Test R²
            r2_test = r2_score(Y_test, ridge.predict(X_test))

            pair_key = f"{lang_i}→{lang_j}"
            pair_results[pair_key] = {
                'r2_train': float(r2_train),
                'r2_test': float(r2_test),
            }
            print(f"  {pair_key}: train R²={r2_train:.4f}, test R²={r2_test:.4f}")

    results[f"{state_name}_pairwise"] = pair_results

    # Summary
    test_r2s = [v['r2_test'] for v in pair_results.values()]
    print(f"\n  Mean test R² ({state_name}): {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")
    print(f"  Min test R²: {min(test_r2s):.4f}, Max: {max(test_r2s):.4f}")


# ---------- EXPERIMENT 2: COCYCLE ON GENERATION ENDPOINTS ----------
print("\n" + "=" * 70)
print("EXPERIMENT 2: Cocycle Consistency on Generation Endpoints")
print("=" * 70)
print("Does ω_{ij} ∘ ω_{jk} ≈ ω_{ik} for generation-time answer states?")
print("Using HELD-OUT test problems only.\n")

# Fit all pairwise maps on training data
ridge_maps = {}
for li, lang_i in enumerate(langs):
    for lj, lang_j in enumerate(langs):
        if li == lj:
            continue
        X = np.array([answer_states[(p, lang_i)] for p in train_probs])
        Y = np.array([answer_states[(p, lang_j)] for p in train_probs])
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, Y)
        ridge_maps[(lang_i, lang_j)] = ridge

# Cocycle test on held-out problems
import itertools
cocycle_errors = []
for i, j, k in itertools.permutations(langs, 3):
    # Direct: i → k
    X_ik = np.array([answer_states[(p, i)] for p in test_probs])
    Y_ik = ridge_maps[(i, k)].predict(X_ik)

    # Composed: i → j → k
    Y_ij = ridge_maps[(i, j)].predict(X_ik)
    Y_ijk = ridge_maps[(j, k)].predict(Y_ij)

    # Cocycle error
    actual_k = np.array([answer_states[(p, k)] for p in test_probs])
    direct_err = np.linalg.norm(Y_ik - actual_k) / np.linalg.norm(actual_k)
    composed_err = np.linalg.norm(Y_ijk - actual_k) / np.linalg.norm(actual_k)
    cocycle_err = np.linalg.norm(Y_ik - Y_ijk) / np.linalg.norm(actual_k)

    cocycle_errors.append({
        'triple': f"{i}→{j}→{k}",
        'direct_err': float(direct_err),
        'composed_err': float(composed_err),
        'cocycle_err': float(cocycle_err),
    })

cocycle_errors.sort(key=lambda x: x['cocycle_err'])
print(f"Cocycle errors (on held-out test problems):")
for ce in cocycle_errors[:5]:
    print(f"  {ce['triple']}: cocycle={ce['cocycle_err']:.4f}, "
          f"direct={ce['direct_err']:.4f}, composed={ce['composed_err']:.4f}")
print(f"  ...")
for ce in cocycle_errors[-3:]:
    print(f"  {ce['triple']}: cocycle={ce['cocycle_err']:.4f}")

mean_cocycle = np.mean([ce['cocycle_err'] for ce in cocycle_errors])
print(f"\n  Mean cocycle error: {mean_cocycle:.4f}")
results['cocycle_generation'] = {
    'mean_error': float(mean_cocycle),
    'all_triples': cocycle_errors,
}


# ---------- EXPERIMENT 3: TRAJECTORY ALIGNMENT ----------
print("\n" + "=" * 70)
print("EXPERIMENT 3: Full Trajectory Alignment (with generation-fit maps)")
print("=" * 70)
print("Apply generation-fit ridge maps to full trajectories.")
print("Compare: do trajectories overlap when language is removed?\n")

# Pick the zh→en map fit on generation endpoints
ridge_zh_en_gen = ridge_maps[('zh', 'en')]

# For each test problem, map zh trajectory into en-space and measure alignment
for prob_idx in test_probs:
    zh_key = f"prob{prob_idx}_zh"
    en_key = f"prob{prob_idx}_en"
    if zh_key not in traj or en_key not in traj:
        continue

    h_zh = traj[zh_key]
    h_en = traj[en_key]
    n_zh, n_en = h_zh.shape[0], h_en.shape[0]

    if n_zh < 10 or n_en < 10:
        continue

    # Map zh into en-space using generation-fit ridge
    h_zh_mapped = ridge_zh_en_gen.predict(h_zh)

    # NN distances
    D_cross = cdist(h_zh_mapped, h_en, 'euclidean')
    nn_zh_to_en = D_cross.min(axis=1).mean()
    nn_en_to_zh = D_cross.min(axis=0).mean()
    within_en = np.mean(pdist(h_en, 'euclidean'))
    within_zh = np.mean(pdist(h_zh_mapped, 'euclidean'))

    nn_ratio = nn_zh_to_en / within_en

    cat_name = meta[zh_key]['category']
    print(f"\n  prob{prob_idx} ({cat_name}): {n_zh} zh, {n_en} en")
    print(f"    NN zh→en: {nn_zh_to_en:.1f}, within_en: {within_en:.1f}, "
          f"ratio: {nn_ratio:.3f}")

    results[f"traj_prob{prob_idx}"] = {
        'category': cat_name,
        'n_zh': n_zh, 'n_en': n_en,
        'nn_zh_to_en': float(nn_zh_to_en),
        'nn_en_to_zh': float(nn_en_to_zh),
        'within_en': float(within_en),
        'nn_ratio': float(nn_ratio),
    }


# ---------- EXPERIMENT 4: JOINT DIFFUSION MAP WITH GENERATION-FIT MAPS ----------
print("\n" + "=" * 70)
print("EXPERIMENT 4: Joint Diffusion Map (all 4 languages, generation-fit)")
print("=" * 70)

def diffusion_map(X, n_components=3, epsilon=None, alpha=1.0):
    n = X.shape[0]
    D2 = squareform(pdist(X, 'sqeuclidean'))
    if epsilon is None:
        epsilon = np.median(pdist(X, 'euclidean')) ** 2
    K = np.exp(-D2 / epsilon)
    q = K.sum(axis=1)
    if alpha > 0:
        q_alpha = q ** alpha
        K = K / np.outer(q_alpha, q_alpha)
    row_sums = K.sum(axis=1)
    T = K / row_sums[:, np.newaxis]
    D_sqrt = np.sqrt(row_sums)
    D_sqrt_inv = 1.0 / D_sqrt
    T_sym = (T * D_sqrt[np.newaxis, :]) * D_sqrt_inv[:, np.newaxis]
    T_sym = 0.5 * (T_sym + T_sym.T)
    n_eig = min(n_components + 1, n - 1)
    eigenvalues, eigenvectors = eigh(T_sym, subset_by_index=[n - n_eig, n - 1])
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[1:n_components+1]
    eigenvectors = eigenvectors[:, 1:n_components+1]
    coords = eigenvectors * D_sqrt_inv[:, np.newaxis]
    coords = coords * eigenvalues[np.newaxis, :]
    return coords, eigenvalues, epsilon


# For each test problem: map all languages into en-space, joint diffusion map
for prob_idx in test_probs:
    keys = {l: f"prob{prob_idx}_{l}" for l in langs}
    if not all(k in traj for k in keys.values()):
        continue

    trajs = {l: traj[keys[l]] for l in langs}
    n_steps = {l: trajs[l].shape[0] for l in langs}

    if any(n < 10 for n in n_steps.values()):
        continue

    cat_name = meta[keys['en']]['category']
    print(f"\n  prob{prob_idx} ({cat_name}): " +
          ", ".join(f"{l}={n_steps[l]}" for l in langs))

    # Map all languages into en-space
    mapped = {}
    mapped['en'] = trajs['en']  # en stays as-is
    for l in langs:
        if l == 'en':
            continue
        mapped[l] = ridge_maps[(l, 'en')].predict(trajs[l])

    # Stack all trajectories
    all_points = []
    labels = []  # which language each point belongs to
    boundaries = [0]
    for l in langs:
        all_points.append(mapped[l])
        labels.extend([l] * mapped[l].shape[0])
        boundaries.append(boundaries[-1] + mapped[l].shape[0])

    all_points = np.vstack(all_points)
    n_total = all_points.shape[0]
    print(f"    Total points: {n_total}")

    # Joint diffusion map
    coords, evals, eps = diffusion_map(all_points, n_components=3)
    print(f"    Eigenvalues: {evals}")

    # Measure: do same-problem different-language points cluster?
    # Centroid per language
    centroids = {}
    spreads = {}
    for i, l in enumerate(langs):
        pts = coords[boundaries[i]:boundaries[i+1]]
        centroids[l] = pts.mean(axis=0)
        spreads[l] = np.mean(np.linalg.norm(pts - centroids[l], axis=1))

    # Between-language centroid distances
    between_dists = []
    for li, la in enumerate(langs):
        for lj, lb in enumerate(langs):
            if li >= lj:
                continue
            d = np.linalg.norm(centroids[la] - centroids[lb])
            between_dists.append(d)

    mean_between = np.mean(between_dists)
    mean_spread = np.mean(list(spreads.values()))
    separation = mean_between / mean_spread

    print(f"    Mean between-lang centroid dist: {mean_between:.6f}")
    print(f"    Mean within-lang spread: {mean_spread:.6f}")
    print(f"    Separation ratio: {separation:.3f}")
    print(f"    ({'OVERLAPPING' if separation < 0.5 else 'SEPARATE' if separation > 1.5 else 'PARTIAL'})")

    # NN analysis: for each zh point, is nearest neighbor from same problem
    # (any language) or from a different time in same trajectory?
    from scipy.spatial.distance import cdist as cdist2
    for src_lang in ['zh', 'ja']:
        src_pts = coords[boundaries[langs.index(src_lang)]:boundaries[langs.index(src_lang)+1]]
        en_pts = coords[boundaries[langs.index('en')]:boundaries[langs.index('en')+1]]
        D_cross = cdist2(src_pts, en_pts, 'euclidean')
        nn_dists = D_cross.min(axis=1)
        # Compare to within-trajectory step distances
        src_steps = np.linalg.norm(np.diff(src_pts, axis=0), axis=1)
        ratio = nn_dists.mean() / src_steps.mean() if src_steps.mean() > 0 else float('inf')
        print(f"    NN {src_lang}→en in diffusion: mean={nn_dists.mean():.6f}, "
              f"step={src_steps.mean():.6f}, ratio={ratio:.2f}")

    # Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'zh': 'red', 'en': 'blue', 'es': 'green', 'ja': 'purple'}
    for i, l in enumerate(langs):
        pts = coords[boundaries[i]:boundaries[i+1]]
        t_color = np.linspace(0.3, 1.0, pts.shape[0])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=[colors[l]] * pts.shape[0], s=8, alpha=0.5, label=l)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
               color=colors[l], alpha=0.2, linewidth=0.5)
        # Mark start and end
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                  color=colors[l], s=80, marker='*')
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                  color=colors[l], s=80, marker='s')

    ax.set_title(f"prob{prob_idx} ({cat_name}) — 4-Language Joint Diffusion\n"
                f"(all mapped to en-space via generation-fit ridge)\n"
                f"separation={separation:.3f}")
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂'); ax.set_zlabel('φ₃')
    ax.legend()

    fig.tight_layout()
    fig.savefig(f'output/diffusion_corrected_prob{prob_idx}.png', dpi=150)
    plt.close(fig)
    print(f"    Saved: output/diffusion_corrected_prob{prob_idx}.png")

    results[f"joint_diffusion_prob{prob_idx}"] = {
        'category': cat_name,
        'n_total': n_total,
        'eigenvalues': evals.tolist(),
        'separation': float(separation),
        'centroids': {l: centroids[l].tolist() for l in langs},
        'spreads': {l: float(spreads[l]) for l in langs},
    }


# ---------- EXPERIMENT 5: CONTROL — Random problem labels ----------
print("\n" + "=" * 70)
print("EXPERIMENT 5: Scrambled Control")
print("=" * 70)
print("Fit ridge maps on RANDOM problem pairings (zh problem i → en problem j).")
print("If this also works, the alignment is trivial.\n")

# Scramble: pair zh problem i with en problem j (randomly)
np.random.seed(99)
scrambled_train = list(train_probs)
np.random.shuffle(scrambled_train)

X_real = np.array([answer_states[(p, 'zh')] for p in train_probs])
Y_real = np.array([answer_states[(p, 'en')] for p in train_probs])
X_test_real = np.array([answer_states[(p, 'zh')] for p in test_probs])
Y_test_real = np.array([answer_states[(p, 'en')] for p in test_probs])

# Real pairing
ridge_real = Ridge(alpha=1.0)
ridge_real.fit(X_real, Y_real)
r2_real_train = r2_score(Y_real, ridge_real.predict(X_real))
r2_real_test = r2_score(Y_test_real, ridge_real.predict(X_test_real))

# Scrambled pairing
Y_scrambled = np.array([answer_states[(p, 'en')] for p in scrambled_train])
ridge_scram = Ridge(alpha=1.0)
ridge_scram.fit(X_real, Y_scrambled)
r2_scram_train = r2_score(Y_scrambled, ridge_scram.predict(X_real))
# Test on real pairings — should fail
r2_scram_test = r2_score(Y_test_real, ridge_scram.predict(X_test_real))

print(f"  Real pairing:     train R²={r2_real_train:.4f}, test R²={r2_real_test:.4f}")
print(f"  Scrambled pairing: train R²={r2_scram_train:.4f}, test R²={r2_scram_test:.4f}")
print(f"\n  Gap: {r2_real_test - r2_scram_test:.4f}")
print(f"  {'ALIGNMENT IS REAL' if r2_real_test > r2_scram_test + 0.1 else 'WARNING: gap too small'}")

results['scrambled_control'] = {
    'real_train': float(r2_real_train),
    'real_test': float(r2_real_test),
    'scrambled_train': float(r2_scram_train),
    'scrambled_test': float(r2_scram_test),
    'gap': float(r2_real_test - r2_scram_test),
}


# ---------- SAVE ----------
with open('output/diffusion_corrected_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: output/diffusion_corrected_results.json")

# ---------- FINAL SUMMARY ----------
print("\n" + "=" * 90)
print("FINAL SUMMARY: ONE ROOM, TWO PATHS")
print("=" * 90)

# Answer state alignment
for state_name in ['answer', 'mid', 'early']:
    key = f"{state_name}_pairwise"
    if key in results:
        test_r2s = [v['r2_test'] for v in results[key].values()]
        print(f"\n  {state_name.upper()} state alignment (mean test R²): "
              f"{np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")

print(f"\n  Cocycle error (generation endpoints): {results['cocycle_generation']['mean_error']:.4f}")
print(f"  Scrambled control gap: {results['scrambled_control']['gap']:.4f}")

print(f"\n  Joint diffusion separation ratios:")
for key, val in sorted(results.items()):
    if key.startswith('joint_diffusion'):
        print(f"    {key}: {val['separation']:.3f} ({val['category']})")

print("\n  INTERPRETATION:")
print("  If answer R² > 0.5 and cocycle < 0.1: Z exists during generation")
print("  If separation < 0.5: One Tape (same path)")
print("  If separation 0.5-1.5: One Room (same space, different paths)")
print("  If separation > 1.5: Two Rooms (different spaces entirely)")
