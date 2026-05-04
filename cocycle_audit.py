"""
COCYCLE AUDIT — The overdue reckoning.

Two questions answered in one script:

1. INPUT-PASS COCYCLE AUDIT: Does the R²=0.77, 2.86% cocycle error survive
   dimensionality reduction? 200 samples in 2048 dims is 10x overparameterized.
   Test with PCA(5, 10, 20, 50) + higher regularization. If cocycle still works
   in PCA(20), the signal is real and lives in ~20 dimensions. If it breaks,
   the input-pass cocycle was also an artifact.

2. GENERATION-TIME LOW-DIM Z: The 0% NN retrieval was in R^2048. Maybe Z is
   low-dimensional and the problem-identity signal is buried under 2000+ dims
   of language-specific noise. Project generation endpoints onto PCA(5,10,20,50)
   of the INPUT-PASS Z basis, then re-run NN retrieval. If NN improves in low
   dims, Z exists during generation but only in a small subspace.

Both tests use the SAME PCA basis (from input-pass multilingual activations)
so results are directly comparable.
"""

import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
import itertools

# ---------- LOAD DATA ----------
print("Loading data...")
multi = np.load('output/multilingual_activations.npz')
traj = np.load('output/gen_trajectories.npz')
with open('output/gen_trajectories_meta.json') as f:
    gen_meta = json.load(f)

INPUT_LANGS = ['zh', 'en', 'es', 'ar', 'ja', 'ko', 'sw']
GEN_LANGS = ['zh', 'en', 'es', 'ja']
N_PROBLEMS = 200

# Input-pass activations: (200, 2048) per language
input_acts = {l: multi[l] for l in INPUT_LANGS}

# Generation endpoints
gen_endpoints = {}
for key in traj.keys():
    m = gen_meta[key]
    h = traj[key]
    if h.shape[0] >= 5:
        gen_endpoints[(m['problem_idx'], m['language'])] = h[-1]

gen_probs = sorted(set(p for p, l in gen_endpoints.keys()))
gen_complete = [p for p in gen_probs if all((p, l) in gen_endpoints for l in GEN_LANGS)]
print(f"Input-pass: {N_PROBLEMS} problems × {len(INPUT_LANGS)} languages")
print(f"Generation: {len(gen_complete)} complete problems × {len(GEN_LANGS)} languages")

results = {}

# ================================================================
# PART 1: INPUT-PASS COCYCLE AUDIT
# ================================================================
print("\n" + "=" * 70)
print("PART 1: INPUT-PASS COCYCLE AUDIT")
print("=" * 70)

# Build PCA basis from ALL input-pass activations (all languages pooled)
all_input = np.vstack([input_acts[l] for l in INPUT_LANGS])  # (1400, 2048)
print(f"PCA basis from {all_input.shape[0]} vectors")

pca_full = PCA(n_components=50)
pca_full.fit(all_input)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
print(f"Variance explained: PC5={cumvar[4]:.3f}, PC10={cumvar[9]:.3f}, "
      f"PC20={cumvar[19]:.3f}, PC50={cumvar[49]:.3f}")

for n_pca in [5, 10, 20, 50, 2048]:
    print(f"\n--- PCA = {n_pca} ---")

    if n_pca < 2048:
        pca = PCA(n_components=n_pca)
        pca.fit(all_input)
        proj = {l: pca.transform(input_acts[l]) for l in INPUT_LANGS}
        var_exp = pca.explained_variance_ratio_.sum()
    else:
        proj = {l: input_acts[l] for l in INPUT_LANGS}
        var_exp = 1.0

    # Cross-validated pairwise R² (150 train / 50 test, 10 folds)
    np.random.seed(42)
    n_folds = 10
    fold_r2s = {f"{a}→{b}": [] for a in INPUT_LANGS for b in INPUT_LANGS if a < b}

    for fold in range(n_folds):
        perm = np.random.permutation(N_PROBLEMS)
        train_idx = perm[:150]
        test_idx = perm[150:]

        for li, la in enumerate(INPUT_LANGS):
            for lj, lb in enumerate(INPUT_LANGS):
                if la >= lb:
                    continue
                key = f"{la}→{lb}"
                X_tr = proj[la][train_idx]
                Y_tr = proj[lb][train_idx]
                X_te = proj[la][test_idx]
                Y_te = proj[lb][test_idx]

                alpha = max(1.0, n_pca / 10)  # scale regularization with dims
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr, Y_tr)
                r2 = r2_score(Y_te, ridge.predict(X_te))
                fold_r2s[key].append(r2)

    # Summary
    all_test_r2 = []
    for key, vals in sorted(fold_r2s.items()):
        mean_r2 = np.mean(vals)
        all_test_r2.append(mean_r2)

    mean_all = np.mean(all_test_r2)
    print(f"  Var explained: {var_exp:.3f}")
    print(f"  Mean CV test R²: {mean_all:.4f} (min={min(all_test_r2):.4f}, max={max(all_test_r2):.4f})")

    # Cocycle test on held-out fold
    perm = np.random.permutation(N_PROBLEMS)
    train_idx = perm[:150]
    test_idx = perm[150:]

    # Fit all pairwise maps
    maps = {}
    for la in INPUT_LANGS:
        for lb in INPUT_LANGS:
            if la == lb:
                continue
            alpha = max(1.0, n_pca / 10)
            ridge = Ridge(alpha=alpha)
            ridge.fit(proj[la][train_idx], proj[lb][train_idx])
            maps[(la, lb)] = ridge

    # Cocycle errors on test set
    cocycle_errs = []
    for a, b, c in itertools.permutations(INPUT_LANGS, 3):
        X_a = proj[a][test_idx]
        actual_c = proj[c][test_idx]

        direct = maps[(a, c)].predict(X_a)
        composed = maps[(b, c)].predict(maps[(a, b)].predict(X_a))

        err = np.linalg.norm(direct - composed) / np.linalg.norm(actual_c)
        cocycle_errs.append(err)

    print(f"  Cocycle error (test): mean={np.mean(cocycle_errs):.4f}, "
          f"median={np.median(cocycle_errs):.4f}")

    # Scrambled control
    np.random.seed(99)
    scramble_perm = np.random.permutation(150)
    scram_maps = {}
    for la in ['zh', 'en']:
        for lb in ['zh', 'en']:
            if la == lb:
                continue
            Y_scram = proj[lb][train_idx][scramble_perm]
            ridge = Ridge(alpha=alpha)
            ridge.fit(proj[la][train_idx], Y_scram)
            scram_maps[(la, lb)] = ridge

    # Test scrambled: zh→en direct vs zh→en on test
    if ('zh', 'en') in scram_maps:
        r2_scram = r2_score(proj['en'][test_idx],
                           scram_maps[('zh', 'en')].predict(proj['zh'][test_idx]))
        r2_real = r2_score(proj['en'][test_idx],
                          maps[('zh', 'en')].predict(proj['zh'][test_idx]))
        print(f"  zh→en: real test R²={r2_real:.4f}, scrambled test R²={r2_scram:.4f}, "
              f"gap={r2_real - r2_scram:.4f}")

    results[f"input_pca{n_pca}"] = {
        'n_pca': n_pca,
        'var_explained': float(var_exp),
        'mean_cv_r2': float(mean_all),
        'min_cv_r2': float(min(all_test_r2)),
        'max_cv_r2': float(max(all_test_r2)),
        'cocycle_mean': float(np.mean(cocycle_errs)),
        'cocycle_median': float(np.median(cocycle_errs)),
    }


# ================================================================
# PART 2: GENERATION-TIME NN IN LOW DIMENSIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 2: GENERATION-TIME NN IN LOW DIMENSIONS")
print("=" * 70)
print("Does problem identity emerge when we project to low-dim Z?\n")

# Use the PCA basis from input-pass (the Z-subspace)
# Project generation endpoints onto this basis
# Then check NN retrieval

for n_pca in [5, 10, 20, 50, 100, 2048]:
    if n_pca > min(len(gen_complete) * len(GEN_LANGS), 2048):
        continue

    if n_pca < 2048:
        pca = PCA(n_components=n_pca)
        pca.fit(all_input)  # basis from input-pass
    else:
        pca = None

    # Project generation endpoints
    vecs = []
    probs = []
    langs_list = []
    for p in gen_complete:
        for l in GEN_LANGS:
            v = gen_endpoints[(p, l)]
            if pca is not None:
                v = pca.transform(v.reshape(1, -1)).flatten()
            vecs.append(v)
            probs.append(p)
            langs_list.append(l)
    vecs = np.array(vecs)

    # Center per language
    for l in GEN_LANGS:
        mask = [i for i, ll in enumerate(langs_list) if ll == l]
        vecs[mask] -= vecs[mask].mean(axis=0)

    # NN retrieval
    D = cdist(vecs, vecs, 'euclidean')
    np.fill_diagonal(D, np.inf)

    same_prob = 0
    same_lang = 0
    # Also: top-3 NN
    top3_same_prob = 0
    for i in range(len(vecs)):
        nn = np.argmin(D[i])
        if probs[nn] == probs[i]:
            same_prob += 1
        if langs_list[nn] == langs_list[i]:
            same_lang += 1
        # top-3
        top3 = np.argsort(D[i])[:3]
        if any(probs[j] == probs[i] for j in top3):
            top3_same_prob += 1

    n_total = len(vecs)
    chance = (len(GEN_LANGS) - 1) / (n_total - 1)  # chance of same-problem NN

    print(f"  PCA={n_pca:>4}: NN same-prob={same_prob}/{n_total} ({same_prob/n_total:.3f}), "
          f"same-lang={same_lang}/{n_total} ({same_lang/n_total:.3f}), "
          f"top3={top3_same_prob}/{n_total} ({top3_same_prob/n_total:.3f}), "
          f"chance={chance:.3f}")

    results[f"gen_nn_pca{n_pca}"] = {
        'n_pca': n_pca,
        'nn_same_prob': same_prob,
        'nn_same_lang': same_lang,
        'top3_same_prob': top3_same_prob,
        'n_total': n_total,
        'nn_same_prob_frac': same_prob / n_total,
        'chance': chance,
    }


# ================================================================
# PART 3: GENERATION MID-POINT NN IN LOW DIMENSIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 3: MID-GENERATION NN IN LOW DIMENSIONS")
print("=" * 70)
print("The midpoint had 20% NN accuracy in R^2048. Does it improve in low dims?\n")

for frac_name, frac in [('mid (50%)', 0.5)]:
    for n_pca in [5, 10, 20, 50, 2048]:
        if n_pca < 2048:
            pca = PCA(n_components=n_pca)
            pca.fit(all_input)
        else:
            pca = None

        vecs = []
        probs = []
        langs_list = []
        for p in gen_complete:
            for l in GEN_LANGS:
                key = f"prob{p}_{l}"
                h = traj[key]
                n = h.shape[0]
                idx = max(0, int(n * frac))
                v = h[idx]
                if pca is not None:
                    v = pca.transform(v.reshape(1, -1)).flatten()
                vecs.append(v)
                probs.append(p)
                langs_list.append(l)
        vecs = np.array(vecs)

        # Center per language
        for l in GEN_LANGS:
            mask = [i for i, ll in enumerate(langs_list) if ll == l]
            vecs[mask] -= vecs[mask].mean(axis=0)

        D = cdist(vecs, vecs, 'euclidean')
        np.fill_diagonal(D, np.inf)

        same_prob = sum(1 for i in range(len(vecs)) if probs[np.argmin(D[i])] == probs[i])
        n_total = len(vecs)

        print(f"  {frac_name} PCA={n_pca:>4}: NN same-prob={same_prob}/{n_total} "
              f"({same_prob/n_total:.3f})")


# ================================================================
# PART 4: SAMPLE SIZE CURVE (input-pass)
# ================================================================
print("\n" + "=" * 70)
print("PART 4: SAMPLE SIZE CURVE")
print("=" * 70)
print("At what N does the input-pass cocycle break?\n")

for n_train in [15, 30, 50, 100, 150, 180]:
    n_test = min(50, N_PROBLEMS - n_train)
    if n_test < 10:
        continue

    np.random.seed(42)
    perm = np.random.permutation(N_PROBLEMS)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:n_train + n_test]

    # PCA=20, alpha=2.0
    pca = PCA(n_components=20)
    pca.fit(all_input)
    proj = {l: pca.transform(input_acts[l]) for l in INPUT_LANGS}

    r2s = []
    for la in INPUT_LANGS:
        for lb in INPUT_LANGS:
            if la >= lb:
                continue
            ridge = Ridge(alpha=2.0)
            ridge.fit(proj[la][train_idx], proj[lb][train_idx])
            r2 = r2_score(proj[lb][test_idx], ridge.predict(proj[la][test_idx]))
            r2s.append(r2)

    print(f"  N_train={n_train:>3}, N_test={n_test:>2}: "
          f"mean R²={np.mean(r2s):.4f} (min={min(r2s):.4f}, max={max(r2s):.4f})")

    results[f"sample_size_{n_train}"] = {
        'n_train': n_train,
        'n_test': n_test,
        'mean_r2': float(np.mean(r2s)),
        'min_r2': float(min(r2s)),
    }


# ================================================================
# PART 5: CKA ON GENERATION TRAJECTORIES AT MULTIPLE POSITIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 5: CKA ALONG TRAJECTORY (generation-time)")
print("=" * 70)

def linear_CKA(X, Y):
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    XtX = Xc @ Xc.T
    YtY = Yc @ Yc.T
    hsic = np.trace(XtX @ YtY) / ((n-1)**2)
    var_x = np.trace(XtX @ XtX) / ((n-1)**2)
    var_y = np.trace(YtY @ YtY) / ((n-1)**2)
    return hsic / np.sqrt(var_x * var_y)

for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
    cka_vals = []
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

    frac_label = f"{int(frac*100)}%"
    print(f"  τ={frac_label:>4}: mean CKA={np.mean(cka_vals):.4f} "
          f"(min={min(cka_vals):.4f}, max={max(cka_vals):.4f})")

    results[f"cka_frac_{frac}"] = {
        'frac': frac,
        'mean_cka': float(np.mean(cka_vals)),
        'min_cka': float(min(cka_vals)),
        'max_cka': float(max(cka_vals)),
    }


# ================================================================
# SAVE
# ================================================================
with open('output/cocycle_audit_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: output/cocycle_audit_results.json")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 90)
print("COCYCLE AUDIT — FINAL SUMMARY")
print("=" * 90)

print("\n1. INPUT-PASS COCYCLE BY DIMENSIONALITY:")
for n_pca in [5, 10, 20, 50, 2048]:
    key = f"input_pca{n_pca}"
    if key in results:
        r = results[key]
        print(f"   PCA={n_pca:>4}: test R²={r['mean_cv_r2']:.4f}, "
              f"cocycle={r['cocycle_mean']:.4f}, var={r['var_explained']:.3f}")

print("\n2. GENERATION-TIME NN BY DIMENSIONALITY (endpoints, centered):")
for n_pca in [5, 10, 20, 50, 100, 2048]:
    key = f"gen_nn_pca{n_pca}"
    if key in results:
        r = results[key]
        print(f"   PCA={n_pca:>4}: same-prob={r['nn_same_prob_frac']:.3f}, "
              f"chance={r['chance']:.3f}")

print("\n3. SAMPLE SIZE CURVE (PCA=20):")
for n in [15, 30, 50, 100, 150, 180]:
    key = f"sample_size_{n}"
    if key in results:
        print(f"   N={n:>3}: R²={results[key]['mean_r2']:.4f}")

print("\n4. CKA ALONG TRAJECTORY:")
for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
    key = f"cka_frac_{frac}"
    if key in results:
        print(f"   τ={int(frac*100):>3}%: CKA={results[key]['mean_cka']:.4f}")
