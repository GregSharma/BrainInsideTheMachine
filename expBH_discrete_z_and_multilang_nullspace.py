"""Exp BH: Discrete Z + multi-language null-space.

Two promising directions from BG:
  1. Multi-language null-space: null of ALL pairwise diffs (not just EN-ZH)
  2. Discrete Z: quantize kernel reps, test if discrete codes transfer cross-lingually

The hypothesis: continuous representations are language-specific in their
coordinate system, but the DISCRETE structure (which cluster, which region)
might be universal. Like different languages having different number systems
but the same arithmetic.

Tests:
  1. Multi-language null-space retrieval (null of 7-lang diffs)
  2. K-means quantization → discrete Z codes
  3. Cross-lingual discrete code prediction: train EN codes → test ZH codes
  4. "Problem fingerprint": concatenate discrete codes across layers
  5. Retrieval by discrete code matching
"""

import json
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

OUT = Path("output")
np.random.seed(42)

print("=" * 60)
print("  Exp BH: Discrete Z + multi-language null-space")
print("=" * 60)

# ── Load data ────────────────────────────────────────────────────────

lasttok = np.load(OUT / "all_layers_lasttok.npz")
multi = np.load(OUT / "multilingual_all_layers.npz")
categories = lasttok["categories"]
N_PROB = 200

ALL_LANGS = [l for l in ["en", "zh", "es", "ar", "ja", "ko", "sw"] if f"{l}_L32" in multi]

def get_acts(lang, layer):
    key = f"{lang}_L{layer}"
    return multi[key] if key in multi else lasttok[key]

# Build kernel
stacked_32 = np.stack([get_acts(l, 32) for l in ALL_LANGS], axis=0)
prob_means = stacked_32.mean(axis=0, keepdims=True)
deviations = (stacked_32 - prob_means).reshape(-1, 2048)
_, _, Vt = np.linalg.svd(deviations, full_matrices=False)
lang_basis_10 = Vt[:10]

pooled = stacked_32.reshape(-1, 2048)
pooled_clean = pooled - (pooled @ lang_basis_10.T) @ lang_basis_10
pca = PCA(n_components=50)
pca.fit(pooled_clean)
kernel_basis = pca.components_

def to_kernel(acts):
    clean = acts - (acts @ lang_basis_10.T) @ lang_basis_10
    return clean @ kernel_basis.T


# ── 1. Multi-language null-space ─────────────────────────────────────

print("\n[1/4] Multi-language null-space retrieval")

# Compute pairwise diffs for ALL language pairs
all_diffs = []
for i, la in enumerate(ALL_LANGS):
    for j, lb in enumerate(ALL_LANGS):
        if i >= j:
            continue
        d = get_acts(la, 32) - get_acts(lb, 32)
        all_diffs.append(d)

all_diffs = np.vstack(all_diffs)  # (21*200, 2048) for 7 langs
print(f"  Pairwise diffs: {all_diffs.shape}")

# SVD on all diffs → language subspace
U_d, S_d, Vt_d = np.linalg.svd(all_diffs, full_matrices=False)

# Null space: directions with SMALL singular values = where ALL langs agree
for n_shared in [10, 20, 30, 50, 100]:
    shared_proj = Vt_d[-n_shared:]  # Bottom n_shared singular vectors

    # Retrieval test
    en_proj = get_acts("en", 32) @ shared_proj.T
    en_n = en_proj / (np.linalg.norm(en_proj, axis=1, keepdims=True) + 1e-8)

    print(f"\n  Shared-{n_shared}D null-space:")
    for lang in ALL_LANGS:
        if lang == "en":
            continue
        lang_proj = get_acts(lang, 32) @ shared_proj.T
        lang_n = lang_proj / (np.linalg.norm(lang_proj, axis=1, keepdims=True) + 1e-8)
        sim = lang_n @ en_n.T
        ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(N_PROB)])
        t1 = (ranks == 0).mean()
        t5 = (ranks < 5).mean()
        mr = ranks.mean()
        print(f"    {lang}→EN: Top-1={t1:.3f} Top-5={t5:.3f} Mean rank={mr:.1f}")


# ── 2. Discrete Z: K-means quantization ─────────────────────────────

print("\n[2/4] Discrete Z via K-means quantization")

# Cluster kernel representations at L32 using ALL languages
all_kernel_32 = np.vstack([to_kernel(get_acts(l, 32)) for l in ALL_LANGS])
print(f"  Clustering {all_kernel_32.shape[0]} representations into K clusters...")

discrete_results = {}
for K in [10, 20, 50, 100]:
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    km.fit(all_kernel_32)

    # Assign codes per language
    lang_codes = {}
    for lang in ALL_LANGS:
        k32 = to_kernel(get_acts(lang, 32))
        lang_codes[lang] = km.predict(k32)

    # Cross-lingual code agreement (ARI between each pair)
    aris = []
    for i, la in enumerate(ALL_LANGS):
        for j, lb in enumerate(ALL_LANGS):
            if i >= j:
                continue
            ari = adjusted_rand_score(lang_codes[la], lang_codes[lb])
            aris.append(ari)

    # Same-problem code match rate
    match_rates = []
    for i, la in enumerate(ALL_LANGS):
        for j, lb in enumerate(ALL_LANGS):
            if i >= j:
                continue
            matches = (lang_codes[la] == lang_codes[lb]).mean()
            match_rates.append(matches)

    mean_ari = np.mean(aris)
    mean_match = np.mean(match_rates)

    # Category purity: do clusters align with math categories?
    en_codes = lang_codes["en"]
    cluster_purity = []
    for c in range(K):
        mask = en_codes == c
        if mask.sum() < 2:
            continue
        cats = categories[mask]
        most_common = np.bincount(cats).max()
        cluster_purity.append(most_common / mask.sum())
    purity = np.mean(cluster_purity) if cluster_purity else 0

    discrete_results[K] = {
        "mean_ari": float(mean_ari),
        "mean_match": float(mean_match),
        "cluster_purity": float(purity),
    }
    print(f"  K={K:>3d}: ARI={mean_ari:.4f}  match_rate={mean_match:.3f}  purity={purity:.3f}")


# ── 3. Cross-lingual discrete code prediction ───────────────────────

print("\n[3/4] Cross-lingual discrete code prediction: kernel(L_early) → code(L32)")

# Use K=50 codes
K_CODE = 50
km50 = KMeans(n_clusters=K_CODE, random_state=42, n_init=10)
km50.fit(all_kernel_32)

code_pred_results = {}
idx = np.random.permutation(N_PROB)
train_prob, test_prob = idx[:160], idx[160:]

for source_layer in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
    # Train on EN+ZH kernel at source layer → EN codes at L32
    X_tr, Y_tr = [], []
    for lang in ["en", "zh"]:
        s = to_kernel(get_acts(lang, source_layer))
        codes = km50.predict(to_kernel(get_acts(lang, 32)))
        X_tr.append(s[train_prob])
        Y_tr.append(codes[train_prob])
    X_tr = np.vstack(X_tr)
    Y_tr = np.concatenate(Y_tr)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, solver='lbfgs')
    try:
        clf.fit(X_tr, Y_tr)
    except Exception:
        continue

    layer_acc = {}
    for lang in ALL_LANGS:
        s = to_kernel(get_acts(lang, source_layer))
        true_codes = km50.predict(to_kernel(get_acts(lang, 32)))
        pred_codes = clf.predict(s[test_prob])
        acc = accuracy_score(true_codes[test_prob], pred_codes)
        layer_acc[lang] = float(acc)

    code_pred_results[source_layer] = layer_acc
    en_a = layer_acc.get("en", 0)
    zh_a = layer_acc.get("zh", 0)
    es_a = layer_acc.get("es", 0)
    ja_a = layer_acc.get("ja", 0)
    print(f"  L{source_layer:>2d}: EN={en_a:.3f} ZH={zh_a:.3f} ES={es_a:.3f} JA={ja_a:.3f}")


# ── 4. Multi-layer discrete fingerprint ──────────────────────────────

print("\n[4/4] Multi-layer discrete fingerprint: code sequence across layers")

# For each problem, compute discrete code at layers [8, 16, 24, 32]
# This "code trajectory" is the discrete fingerprint
FPRINT_LAYERS = [8, 16, 24, 32]

# Build per-layer clusterers using all languages
layer_kms = {}
for L in FPRINT_LAYERS:
    all_k = np.vstack([to_kernel(get_acts(l, L)) for l in ALL_LANGS])
    km = KMeans(n_clusters=20, random_state=42, n_init=10)
    km.fit(all_k)
    layer_kms[L] = km

# Compute fingerprints per problem per language
fingerprints = {}
for lang in ALL_LANGS:
    fp = []
    for L in FPRINT_LAYERS:
        k = to_kernel(get_acts(lang, L))
        codes = layer_kms[L].predict(k)
        fp.append(codes)
    fingerprints[lang] = np.stack(fp, axis=1)  # (200, 4) — 4 codes per problem

# Cross-lingual fingerprint matching
print("  Fingerprint matching (exact code sequence match):")
for lang in ALL_LANGS:
    if lang == "en":
        continue
    en_fp = fingerprints["en"]
    lang_fp = fingerprints[lang]

    # Exact match rate (all 4 codes identical)
    exact = np.all(en_fp == lang_fp, axis=1).mean()

    # Partial match (at least 3/4 codes match)
    partial = (np.sum(en_fp == lang_fp, axis=1) >= 3).mean()

    # Fingerprint retrieval: for each lang problem, find EN problem with most matching codes
    matches = np.zeros(N_PROB, dtype=int)
    for i in range(N_PROB):
        overlaps = (en_fp == lang_fp[i]).sum(axis=1)
        matches[i] = np.argmax(overlaps)
    retrieval_acc = (matches == np.arange(N_PROB)).mean()

    print(f"    {lang}: exact={exact:.3f}  partial(3/4)={partial:.3f}  retrieval={retrieval_acc:.3f}")


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  DISCRETE Z CODES (K-means on kernel, all 7 languages):")
print(f"  {'K':>5s} {'ARI':>8s} {'Match%':>8s} {'Purity':>8s}")
for K, v in sorted(discrete_results.items()):
    print(f"  {K:>5d} {v['mean_ari']:>8.4f} {v['mean_match']:>8.3f} {v['cluster_purity']:>8.3f}")

print(f"\n  CODE PREDICTION: kernel(L_early) → code(L32), K=50:")
for L in sorted(code_pred_results.keys()):
    r = code_pred_results[L]
    parts = [f"{l}={r.get(l,0):.3f}" for l in ALL_LANGS]
    print(f"    L{L:>2d}: {', '.join(parts)}")

# ── Save ─────────────────────────────────────────────────────────────

output = {
    "experiment": "BH",
    "title": "Discrete Z + multi-language null-space",
    "discrete_results": {str(k): v for k, v in discrete_results.items()},
    "code_prediction": {str(k): v for k, v in code_pred_results.items()},
}

with open(OUT / "expBH_discrete_z.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBH_discrete_z.json")
