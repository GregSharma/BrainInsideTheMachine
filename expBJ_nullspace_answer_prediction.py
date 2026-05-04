"""Exp BJ: Can the null-space predict answers? + push encoder past 0.9.

BI showed:
  - Encoder R²=0.872 in null-space, identical across 7 languages
  - 3D suffices for 97% retrieval
  - 6 cat-1 failures

This experiment:
  1. Answer prediction: can we extract the numerical answer from null-space reps?
     Train on EN, test on ZH and other languages.
  2. Nonlinear encoder: MLP in null-space, push past 0.872
  3. The 6 failures: what makes them special? Analyze their null-space geometry.
  4. Answer-relevant dimensions: which null-space dims carry answer info?
"""

import json
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

OUT = Path("output")
np.random.seed(42)

print("=" * 60)
print("  Exp BJ: Null-space answer prediction + encoder push")
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

null_20 = build_nullspace(32, 20)

def to_null(acts, proj=null_20):
    return acts @ proj.T

idx = np.random.permutation(N_PROB)
train_prob, test_prob = idx[:160], idx[160:]

# ── 1. Create answer labels ─────────────────────────────────────────

print("\n[1/4] Creating answer labels from problem structure...")

# Use category as coarse label (already have it)
# Create finer label: cluster EN null-space reps at L32 into 40 buckets
from sklearn.cluster import KMeans

en_null_32 = to_null(get_acts("en", 32))
km40 = KMeans(n_clusters=40, random_state=42, n_init=10)
km40.fit(en_null_32)
answer_codes = km40.predict(en_null_32)

# Check cross-lingual consistency of these codes
from sklearn.metrics import adjusted_rand_score
for lang in ["zh", "es", "ja"]:
    lang_null = to_null(get_acts(lang, 32))
    lang_codes = km40.predict(lang_null)
    ari = adjusted_rand_score(answer_codes, lang_codes)
    print(f"  EN-{lang} code ARI (K=40): {ari:.4f}")


# ── 2. Answer code prediction from earlier layers ────────────────────

print("\n[2/4] Answer code prediction: null(L_source) → answer_code")

answer_pred = {}
for source_layer in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
    null_source = build_nullspace(source_layer, 20)

    # Train on EN+ZH
    X_tr, Y_tr = [], []
    for lang in ["en", "zh"]:
        s = to_null(get_acts(lang, source_layer), null_source)
        codes = km40.predict(to_null(get_acts(lang, 32)))
        X_tr.append(s[train_prob])
        Y_tr.append(codes[train_prob])
    X_tr, Y_tr = np.vstack(X_tr), np.concatenate(Y_tr)

    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=42)
    try:
        clf.fit(X_tr, Y_tr)
    except:
        continue

    layer_acc = {}
    for lang in ALL_LANGS:
        s = to_null(get_acts(lang, source_layer), null_source)
        true_codes = km40.predict(to_null(get_acts(lang, 32)))
        pred = clf.predict(s[test_prob])
        acc = accuracy_score(true_codes[test_prob], pred)
        layer_acc[lang] = float(acc)

    answer_pred[source_layer] = layer_acc
    en_a = layer_acc.get("en", 0)
    zh_a = layer_acc.get("zh", 0)
    es_a = layer_acc.get("es", 0)
    ja_a = layer_acc.get("ja", 0)
    print(f"  L{source_layer:>2d}: EN={en_a:.3f} ZH={zh_a:.3f} ES={es_a:.3f} JA={ja_a:.3f}")


# ── 3. Nonlinear encoder: push past R²=0.872 ────────────────────────

print("\n[3/4] Nonlinear encoder: MLP in null-space")

encoder_push = {}
for source_layer in [4, 8, 12, 16, 24, 28]:
    null_source = build_nullspace(source_layer, 20)
    null_target = null_20

    X_tr, Y_tr = [], []
    for lang in ALL_LANGS:
        s = to_null(get_acts(lang, source_layer), null_source)
        t = to_null(get_acts(lang, 32), null_target)
        X_tr.append(s[train_prob])
        Y_tr.append(t[train_prob])
    X_tr, Y_tr = np.vstack(X_tr), np.vstack(Y_tr)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    # Ridge baseline (all-language training)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, Y_tr)

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                       max_iter=2000, early_stopping=True,
                       validation_fraction=0.1, random_state=42,
                       learning_rate_init=0.001)
    mlp.fit(X_tr_s, Y_tr)

    # Deeper MLP
    mlp_deep = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu',
                            max_iter=2000, early_stopping=True,
                            validation_fraction=0.1, random_state=42,
                            learning_rate_init=0.0005)
    mlp_deep.fit(X_tr_s, Y_tr)

    # Test on each language (held-out problems)
    ridge_r2s, mlp_r2s, deep_r2s = [], [], []
    for lang in ALL_LANGS:
        s = to_null(get_acts(lang, source_layer), null_source)
        t = to_null(get_acts(lang, 32), null_target)
        X_te_s = scaler.transform(s[test_prob])

        r2_ridge = r2_score(t[test_prob], ridge.predict(X_te_s), multioutput='uniform_average')
        r2_mlp = r2_score(t[test_prob], mlp.predict(X_te_s), multioutput='uniform_average')
        r2_deep = r2_score(t[test_prob], mlp_deep.predict(X_te_s), multioutput='uniform_average')
        ridge_r2s.append(r2_ridge)
        mlp_r2s.append(r2_mlp)
        deep_r2s.append(r2_deep)

    encoder_push[source_layer] = {
        "ridge_mean": float(np.mean(ridge_r2s)),
        "mlp_mean": float(np.mean(mlp_r2s)),
        "deep_mean": float(np.mean(deep_r2s)),
    }
    print(f"  L{source_layer:>2d}→L32: Ridge={np.mean(ridge_r2s):.4f}  "
          f"MLP={np.mean(mlp_r2s):.4f}  Deep={np.mean(deep_r2s):.4f}")


# ── 4. Analyze 6 failed problems ────────────────────────────────────

print("\n[4/4] Analyzing 6 failed problems")

FAILED = [65, 84, 107, 129, 144, 191]
PASSED = [i for i in range(N_PROB) if i not in FAILED]

# Null-space geometry: are they clustered together?
en_null = to_null(get_acts("en", 32))
failed_null = en_null[FAILED]
passed_null = en_null[PASSED]

# Mean pairwise cosine within failed vs within a random sample of 6
from itertools import combinations
def mean_pairwise_cos(vecs):
    cos_vals = []
    for i, j in combinations(range(len(vecs)), 2):
        a, b = vecs[i], vecs[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            cos_vals.append(np.dot(a, b) / (na * nb))
    return np.mean(cos_vals) if cos_vals else 0

failed_cos = mean_pairwise_cos(failed_null)

# Random baseline: 100 random sets of 6
random_cos = []
for _ in range(100):
    sample = np.random.choice(PASSED, 6, replace=False)
    random_cos.append(mean_pairwise_cos(en_null[sample]))
random_cos = np.array(random_cos)

print(f"  Failed problems mean pairwise cos: {failed_cos:.4f}")
print(f"  Random 6-problem baseline: {random_cos.mean():.4f} ± {random_cos.std():.4f}")
print(f"  Failed vs random: {'MORE clustered' if failed_cos > random_cos.mean() + 2*random_cos.std() else 'NOT significantly different'}")

# Null-space norms
failed_norms = np.linalg.norm(failed_null, axis=1)
passed_norms = np.linalg.norm(passed_null, axis=1)
print(f"  Failed null norms: {failed_norms.mean():.2f} ± {failed_norms.std():.2f}")
print(f"  Passed null norms: {passed_norms.mean():.2f} ± {passed_norms.std():.2f}")

# Cross-lingual consistency for failed vs passed
failed_cross_cos = []
passed_cross_cos = []
for p in FAILED:
    for lang in ["zh", "es", "ja"]:
        a = to_null(get_acts("en", 32))[p]
        b = to_null(get_acts(lang, 32))[p]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            failed_cross_cos.append(np.dot(a, b) / (na * nb))

for p in PASSED[:50]:
    for lang in ["zh", "es", "ja"]:
        a = to_null(get_acts("en", 32))[p]
        b = to_null(get_acts(lang, 32))[p]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            passed_cross_cos.append(np.dot(a, b) / (na * nb))

print(f"  Failed cross-lingual cos: {np.mean(failed_cross_cos):.4f}")
print(f"  Passed cross-lingual cos: {np.mean(passed_cross_cos):.4f}")


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  ANSWER CODE PREDICTION (null-space, K=40 codes):")
for L in sorted(answer_pred.keys()):
    r = answer_pred[L]
    parts = [f"{l}={r.get(l,0):.3f}" for l in ["en", "zh", "es", "ja"]]
    print(f"    L{L:>2d}: {', '.join(parts)}")

print(f"\n  ENCODER PUSH (all-7-lang training, mean R² across languages):")
for L in sorted(encoder_push.keys()):
    v = encoder_push[L]
    print(f"    L{L:>2d}: Ridge={v['ridge_mean']:.4f}  MLP={v['mlp_mean']:.4f}  "
          f"Deep={v['deep_mean']:.4f}")

# Save
output = {
    "experiment": "BJ",
    "title": "Null-space answer prediction + encoder push",
    "answer_pred": {str(k): v for k, v in answer_pred.items()},
    "encoder_push": {str(k): v for k, v in encoder_push.items()},
    "failed_analysis": {
        "problems": FAILED,
        "failed_pairwise_cos": float(failed_cos),
        "random_baseline_cos": float(random_cos.mean()),
        "failed_norms": [float(x) for x in failed_norms],
        "passed_norms_mean": float(passed_norms.mean()),
        "failed_cross_lingual_cos": float(np.mean(failed_cross_cos)),
        "passed_cross_lingual_cos": float(np.mean(passed_cross_cos)),
    },
}

with open(OUT / "expBJ_nullspace_answer.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBJ_nullspace_answer.json")
