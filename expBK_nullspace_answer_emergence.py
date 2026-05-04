"""Exp BK: Answer emergence in null-space across layers.

BJ showed encoder R²=0.902 (linear!) and answer codes ARI=1.000.
This experiment tracks HOW the answer-relevant information emerges
layer by layer in the null-space.

Tests:
  1. Layer-by-layer answer code accuracy: at which layer can we first
     predict the final answer code? (crystallization point)
  2. Per-category answer prediction: does accuracy vary by math type?
  3. Null-space trajectory smoothness: is the path through null-space
     smooth (gradual answer formation) or discontinuous (sudden jump)?
  4. Cross-lingual answer consistency: for each problem, do all 7
     languages converge to the same answer code at the same layer?
  5. The 6 failures revisited: are they near category boundaries?
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.decomposition import PCA
from pathlib import Path

OUT = Path("output")
np.random.seed(42)

print("=" * 60)
print("  Exp BK: Answer emergence in null-space")
print("=" * 60)

lasttok = np.load(OUT / "all_layers_lasttok.npz")
multi = np.load(OUT / "multilingual_all_layers.npz")
categories = lasttok["categories"]
N_PROB = 200

ALL_LANGS = [l for l in ["en", "zh", "es", "ar", "ja", "ko", "sw"] if f"{l}_L32" in multi]

def get_acts(lang, layer):
    key = f"{lang}_L{layer}"
    return multi[key] if key in multi else lasttok[key]

def build_nullspace(layer, n_dims):
    all_diffs = []
    for i, la in enumerate(ALL_LANGS):
        for j, lb in enumerate(ALL_LANGS):
            if i >= j: continue
            d = get_acts(la, layer) - get_acts(lb, layer)
            all_diffs.append(d)
    all_diffs = np.vstack(all_diffs)
    _, S, Vt = np.linalg.svd(all_diffs, full_matrices=False)
    return Vt[-n_dims:]

# Build reference: null-space codes at L32 (K=20 for cleaner clusters)
null_32 = build_nullspace(32, 20)
all_null_32 = np.vstack([get_acts(l, 32) @ null_32.T for l in ALL_LANGS])
km = KMeans(n_clusters=20, random_state=42, n_init=10)
km.fit(all_null_32)

# Reference codes per language at L32
ref_codes = {}
for lang in ALL_LANGS:
    ref_codes[lang] = km.predict(get_acts(lang, 32) @ null_32.T)

idx = np.random.permutation(N_PROB)
train_prob, test_prob = idx[:160], idx[160:]

# ── 1. Layer-by-layer answer code crystallization ────────────────────

print("\n[1/5] Answer code crystallization by layer")

crystallization = {}
for L in range(0, 36, 2):
    null_L = build_nullspace(L, 20)

    # Project all langs at this layer, assign to L32 cluster centers
    # (using L32 cluster centers applied to L-projected data)
    layer_codes = {}
    for lang in ALL_LANGS:
        proj = get_acts(lang, L) @ null_L.T
        # Assign to nearest L32 centroid (may not be perfect since different basis)
        layer_codes[lang] = km.predict(proj)

    # Match rate: what fraction of problems get the same code as at L32?
    match_rates = {}
    for lang in ALL_LANGS:
        matches = (layer_codes[lang] == ref_codes[lang]).mean()
        match_rates[lang] = float(matches)

    avg_match = np.mean(list(match_rates.values()))
    crystallization[L] = {"avg_match": float(avg_match), "per_lang": match_rates}
    print(f"  L{L:>2d}: avg code match to L32 = {avg_match:.3f}")


# ── 2. Per-category answer prediction ───────────────────────────────

print("\n[2/5] Per-category answer prediction at L32")

cat_answer = {}
for cat in range(5):
    cat_mask = categories == cat
    cat_probs = np.where(cat_mask)[0]
    if len(cat_probs) < 10:
        continue

    # Split within category
    np.random.shuffle(cat_probs)
    n_train = int(len(cat_probs) * 0.8)
    c_train, c_test = cat_probs[:n_train], cat_probs[n_train:]

    if len(c_test) < 3:
        continue

    # Train on EN+ZH within this category
    X_tr, Y_tr = [], []
    for lang in ["en", "zh"]:
        proj = get_acts(lang, 32) @ null_32.T
        X_tr.append(proj[c_train])
        Y_tr.append(ref_codes[lang][c_train])
    X_tr, Y_tr = np.vstack(X_tr), np.concatenate(Y_tr)

    try:
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf.fit(X_tr, Y_tr)

        accs = {}
        for lang in ALL_LANGS:
            proj = get_acts(lang, 32) @ null_32.T
            pred = clf.predict(proj[c_test])
            accs[lang] = float(accuracy_score(ref_codes[lang][c_test], pred))

        cat_answer[int(cat)] = accs
        avg = np.mean(list(accs.values()))
        print(f"  Cat {cat}: avg acc={avg:.3f} (N_test={len(c_test)})")
    except Exception as e:
        print(f"  Cat {cat}: FAILED ({e})")


# ── 3. Null-space trajectory smoothness ──────────────────────────────

print("\n[3/5] Null-space trajectory smoothness (layer-to-layer cosine)")

smoothness = {}
for lang in ["en", "zh", "es"]:
    layer_projs = {}
    for L in range(0, 36, 2):
        null_L = build_nullspace(L, 20)
        layer_projs[L] = get_acts(lang, L) @ null_L.T  # (200, 20)

    # Cosine between consecutive layers (averaged across problems)
    layers = sorted(layer_projs.keys())
    cos_trace = []
    for i in range(len(layers) - 1):
        L1, L2 = layers[i], layers[i+1]
        p1, p2 = layer_projs[L1], layer_projs[L2]
        cos_vals = []
        for prob in range(N_PROB):
            a, b = p1[prob], p2[prob]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-8 and nb > 1e-8:
                cos_vals.append(np.dot(a, b) / (na * nb))
        cos_trace.append(float(np.mean(cos_vals)))

    smoothness[lang] = cos_trace
    print(f"  {lang}: mean layer-to-layer cos = {np.mean(cos_trace):.4f} "
          f"(min={min(cos_trace):.4f}, max={max(cos_trace):.4f})")


# ── 4. Cross-lingual convergence timing ──────────────────────────────

print("\n[4/5] Cross-lingual convergence: at which layer do all 7 langs agree?")

convergence = {}
for L in range(0, 36, 2):
    null_L = build_nullspace(L, 20)

    # For each problem: do all 7 languages get the same code?
    codes_per_prob = np.zeros((N_PROB, len(ALL_LANGS)), dtype=int)
    for j, lang in enumerate(ALL_LANGS):
        proj = get_acts(lang, L) @ null_L.T
        codes_per_prob[:, j] = km.predict(proj)

    # Fraction of problems where all 7 langs agree
    all_agree = np.all(codes_per_prob == codes_per_prob[:, :1], axis=1).mean()
    # Fraction where at least 6/7 agree
    mode_counts = np.array([np.bincount(row, minlength=20).max() for row in codes_per_prob])
    six_agree = (mode_counts >= 6).mean()

    convergence[L] = {"all_7_agree": float(all_agree), "at_least_6": float(six_agree)}
    print(f"  L{L:>2d}: all 7 agree={all_agree:.3f}  ≥6 agree={six_agree:.3f}")


# ── 5. The 6 failures: near category boundaries? ────────────────────

print("\n[5/5] Analyzing 6 failed problems in detail")

FAILED = [65, 84, 107, 129, 144, 191]

# Their codes at L32
for lang in ["en", "zh", "es"]:
    codes = ref_codes[lang]
    failed_codes = codes[FAILED]
    print(f"  {lang} codes for failed problems: {list(failed_codes)}")

# Are they in the same cluster?
en_codes = ref_codes["en"]
failed_cluster = en_codes[FAILED]
unique_clusters = np.unique(failed_cluster)
print(f"  Unique clusters among failures: {list(unique_clusters)}")

# What other problems share these clusters?
for c in unique_clusters:
    in_cluster = np.where(en_codes == c)[0]
    n_failed = sum(1 for p in in_cluster if p in FAILED)
    print(f"  Cluster {c}: {len(in_cluster)} problems, {n_failed} are failures")

# Distance from failed to nearest non-failed problem
en_null = get_acts("en", 32) @ null_32.T
for fp in FAILED:
    dists = np.linalg.norm(en_null - en_null[fp], axis=1)
    dists[fp] = np.inf  # exclude self
    # Distance to nearest non-failed
    non_failed_dists = dists.copy()
    for f in FAILED:
        non_failed_dists[f] = np.inf
    nearest_nf = np.argmin(non_failed_dists)
    # Distance to nearest failed
    failed_dists = np.full(N_PROB, np.inf)
    for f in FAILED:
        if f != fp:
            failed_dists[f] = dists[f]
    nearest_f = np.argmin(failed_dists)
    print(f"  Prob {fp}: nearest non-failed={nearest_nf} (d={non_failed_dists[nearest_nf]:.3f}), "
          f"nearest failed={nearest_f} (d={failed_dists[nearest_f]:.3f})")


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

# Find crystallization layer (first layer where match > 0.9)
crystal_layer = None
for L in sorted(crystallization.keys()):
    if crystallization[L]["avg_match"] > 0.9:
        crystal_layer = L
        break

print(f"\n  CRYSTALLIZATION: answer code stabilizes at L{crystal_layer} (match>0.9)")
print(f"  CONVERGENCE: all 7 langs agree at L32 = {convergence.get(32, {}).get('all_7_agree', 0):.3f}")
print(f"  SMOOTHNESS: mean layer-to-layer cos = {np.mean(smoothness.get('en', [0])):.4f}")

output = {
    "experiment": "BK",
    "title": "Answer emergence in null-space",
    "crystallization": {str(k): v for k, v in crystallization.items()},
    "per_category": cat_answer,
    "smoothness": smoothness,
    "convergence": {str(k): v for k, v in convergence.items()},
    "crystal_layer": crystal_layer,
}

with open(OUT / "expBK_nullspace_emergence.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBK_nullspace_emergence.json")
