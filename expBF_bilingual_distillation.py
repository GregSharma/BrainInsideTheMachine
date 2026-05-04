"""Exp BF: Bilingual distillation — train on BOTH languages, test on held-out.

BD/BE showed that encoder probes trained on EN-only fail catastrophically
on ZH. But category transfer = 1.000. The computation PATH is language-
specific even though the ENDPOINT is aligned.

Fix: train on EN + ZH simultaneously. The probe is forced to find features
that work for both. Test on held-out problems (in-distribution) AND on
held-out languages (es, ja, ar, ko, sw — true cross-lingual transfer).

Also tests:
  1. Bilingual encoder: [EN_L8, ZH_L8] → kernel(L32)
  2. Answer-class prediction: kernel(L) → answer_bucket (discrete distillation)
  3. Bilingual + cocycle alignment: align to common frame, then train
  4. Can the probe extract MORE from kernel than from raw space?
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
print("  Exp BF: Bilingual distillation")
print("=" * 60)

# ── 1. Load data ────────────────────────────────────────────────────

print("\n[1/5] Loading data and building kernel...")

lasttok = np.load(OUT / "all_layers_lasttok.npz")
multi = np.load(OUT / "multilingual_all_layers.npz")
categories = lasttok["categories"]
N_PROB = 200
K_MATH = 50
LANGS_TRAIN = ["en", "zh"]
LANGS_TEST = ["es", "ar", "ja", "ko", "sw"]

# Build kernel at L32 (same as BE)
all_langs = [l for l in LANGS_TRAIN + LANGS_TEST if f"{l}_L32" in multi]
stacked = np.stack([multi[f"{l}_L32"] for l in all_langs], axis=0)
prob_means = stacked.mean(axis=0, keepdims=True)
deviations = (stacked - prob_means).reshape(-1, 2048)
_, _, Vt = np.linalg.svd(deviations, full_matrices=False)
lang_basis = Vt[:10]

pooled = stacked.reshape(-1, 2048)
pooled_clean = pooled - (pooled @ lang_basis.T) @ lang_basis
pca = PCA(n_components=K_MATH)
pca.fit(pooled_clean)
kernel_basis = pca.components_

def to_kernel(acts):
    clean = acts - (acts @ lang_basis.T) @ lang_basis
    return clean @ kernel_basis.T

print(f"  Kernel: {K_MATH}D from {len(all_langs)} languages")
print(f"  Train langs: {LANGS_TRAIN}, Test langs: {[l for l in LANGS_TEST if f'{l}_L32' in multi]}")

# Train/test split on PROBLEMS (not languages)
idx = np.random.permutation(N_PROB)
train_prob, test_prob = idx[:160], idx[160:]

# ── 2. Bilingual encoder probe ──────────────────────────────────────

print("\n[2/5] Bilingual encoder: train on EN+ZH, test on held-out problems AND languages")

def get_acts(lang, layer):
    if f"{lang}_L{layer}" in multi:
        return multi[f"{lang}_L{layer}"]
    return lasttok[f"{lang}_L{layer}"]

results_bilingual = {}
for source_layer in [0, 4, 8, 12, 16, 20, 24, 28]:
    # Training data: EN + ZH, train problems
    X_tr_parts, Y_tr_parts = [], []
    for lang in LANGS_TRAIN:
        raw = get_acts(lang, source_layer)
        target = to_kernel(get_acts(lang, 32))
        X_tr_parts.append(raw[train_prob])
        Y_tr_parts.append(target[train_prob])
    X_tr = np.vstack(X_tr_parts)  # (320, 2048)
    Y_tr = np.vstack(Y_tr_parts)  # (320, K_MATH)

    # Ridge
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr_s, Y_tr)

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu',
                       max_iter=1000, early_stopping=True,
                       validation_fraction=0.15, random_state=42,
                       learning_rate_init=0.0005)
    mlp.fit(X_tr_s, Y_tr)

    layer_results = {}

    # Test on each language (held-out problems)
    for lang in LANGS_TRAIN + LANGS_TEST:
        if f"{lang}_L{source_layer}" not in multi and f"{lang}_L{source_layer}" not in lasttok:
            continue
        raw_te = get_acts(lang, source_layer)[test_prob]
        target_te = to_kernel(get_acts(lang, 32))[test_prob]

        X_te_s = scaler.transform(raw_te)
        r2_ridge = r2_score(target_te, ridge.predict(X_te_s), multioutput='uniform_average')
        r2_mlp = r2_score(target_te, mlp.predict(X_te_s), multioutput='uniform_average')

        layer_results[lang] = {"ridge": float(r2_ridge), "mlp": float(r2_mlp)}

    results_bilingual[source_layer] = layer_results

    # Print compact summary
    en_r2 = layer_results.get("en", {}).get("ridge", float("nan"))
    zh_r2 = layer_results.get("zh", {}).get("ridge", float("nan"))
    es_r2 = layer_results.get("es", {}).get("ridge", float("nan"))
    ja_r2 = layer_results.get("ja", {}).get("ridge", float("nan"))
    print(f"  L{source_layer:>2d}: Ridge EN={en_r2:+.4f} ZH={zh_r2:+.4f} "
          f"ES={es_r2:+.4f} JA={ja_r2:+.4f}")

    en_m = layer_results.get("en", {}).get("mlp", float("nan"))
    zh_m = layer_results.get("zh", {}).get("mlp", float("nan"))
    es_m = layer_results.get("es", {}).get("mlp", float("nan"))
    ja_m = layer_results.get("ja", {}).get("mlp", float("nan"))
    print(f"        MLP  EN={en_m:+.4f} ZH={zh_m:+.4f} "
          f"ES={es_m:+.4f} JA={ja_m:+.4f}")


# ── 3. Answer-class prediction (discrete distillation) ──────────────

print("\n[3/5] Answer-class prediction: kernel → answer bucket")

# Create answer labels by clustering kernel representations at L32
# Use KMeans on the EN kernel reps to create "answer types"
from sklearn.cluster import KMeans

en_kernel_32 = to_kernel(lasttok["en_L32"])  # (200, K_MATH)
zh_kernel_32 = to_kernel(lasttok["zh_L32"])

# Cluster into 20 answer types
N_CLUSTERS = 20
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
km.fit(en_kernel_32)
answer_labels_en = km.predict(en_kernel_32)
answer_labels_zh = km.predict(zh_kernel_32)

# How well do EN and ZH cluster assignments match?
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(answer_labels_en, answer_labels_zh)
print(f"  EN-ZH cluster agreement (ARI): {ari:.4f}")

# Now: can we predict these cluster labels from earlier layers?
answer_results = {}
for source_layer in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
    # Train on EN+ZH
    X_tr_parts, Y_tr_parts = [], []
    for lang, labels in [("en", answer_labels_en), ("zh", answer_labels_zh)]:
        raw = get_acts(lang, source_layer)
        X_tr_parts.append(to_kernel(raw)[train_prob])
        Y_tr_parts.append(labels[train_prob])
    X_tr = np.vstack(X_tr_parts)
    Y_tr = np.concatenate(Y_tr_parts)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X_tr, Y_tr)

    layer_answer = {}
    for lang, labels in [("en", answer_labels_en), ("zh", answer_labels_zh)]:
        raw = get_acts(lang, source_layer)
        X_te = to_kernel(raw)[test_prob]
        acc = accuracy_score(labels[test_prob], clf.predict(X_te))
        layer_answer[lang] = float(acc)

    # Also test on held-out languages
    for lang in LANGS_TEST:
        if f"{lang}_L{source_layer}" not in multi:
            continue
        raw = multi[f"{lang}_L{source_layer}"]
        X_te = to_kernel(raw)[test_prob]
        # Use EN labels as ground truth (cluster assignment should be consistent)
        pred = clf.predict(X_te)
        # Compare to EN labels
        acc = accuracy_score(answer_labels_en[test_prob], pred)
        layer_answer[lang] = float(acc)

    answer_results[source_layer] = layer_answer
    en_a = layer_answer.get("en", 0)
    zh_a = layer_answer.get("zh", 0)
    es_a = layer_answer.get("es", 0)
    ja_a = layer_answer.get("ja", 0)
    print(f"  L{source_layer:>2d}: EN={en_a:.3f} ZH={zh_a:.3f} ES={es_a:.3f} JA={ja_a:.3f}")


# ── 4. Simple kernel-space cosine similarity as "probe" ──────────────

print("\n[4/5] Can we use kernel cosine directly for problem matching?")

# For each test problem in ZH: find the nearest EN problem in kernel space at L32
# This is a retrieval test: does kernel proximity = same problem?
en_k32 = to_kernel(lasttok["en_L32"])
zh_k32 = to_kernel(lasttok["zh_L32"])

# Normalize
en_k32_n = en_k32 / (np.linalg.norm(en_k32, axis=1, keepdims=True) + 1e-8)
zh_k32_n = zh_k32 / (np.linalg.norm(zh_k32, axis=1, keepdims=True) + 1e-8)

sim_matrix = zh_k32_n @ en_k32_n.T  # (200, 200)

# For each ZH problem, what rank is the correct EN problem?
ranks = []
for i in range(N_PROB):
    sorted_idx = np.argsort(-sim_matrix[i])
    rank = np.where(sorted_idx == i)[0][0]
    ranks.append(rank)
ranks = np.array(ranks)

top1 = (ranks == 0).mean()
top5 = (ranks < 5).mean()
top10 = (ranks < 10).mean()
mean_rank = ranks.mean()

print(f"  ZH→EN retrieval at L32 (kernel cosine):")
print(f"    Top-1: {top1:.3f}  Top-5: {top5:.3f}  Top-10: {top10:.3f}  Mean rank: {mean_rank:.1f}")

# Same at different layers
retrieval_results = {"L32": {"top1": float(top1), "top5": float(top5), "mean_rank": float(mean_rank)}}
for L in [0, 4, 8, 16, 24]:
    en_kL = to_kernel(get_acts("en", L))
    zh_kL = to_kernel(get_acts("zh", L))
    en_n = en_kL / (np.linalg.norm(en_kL, axis=1, keepdims=True) + 1e-8)
    zh_n = zh_kL / (np.linalg.norm(zh_kL, axis=1, keepdims=True) + 1e-8)
    sim = zh_n @ en_n.T
    r = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(N_PROB)])
    t1, t5, mr = (r == 0).mean(), (r < 5).mean(), r.mean()
    retrieval_results[f"L{L}"] = {"top1": float(t1), "top5": float(t5), "mean_rank": float(mr)}
    print(f"  L{L:>2d}: Top-1={t1:.3f}  Top-5={t5:.3f}  Mean rank={mr:.1f}")

# Also in RAW space (no kernel) for comparison
print("\n  Comparison: raw space (no kernel) at L32:")
en_raw = lasttok["en_L32"]
zh_raw = lasttok["zh_L32"]
en_rn = en_raw / (np.linalg.norm(en_raw, axis=1, keepdims=True) + 1e-8)
zh_rn = zh_raw / (np.linalg.norm(zh_raw, axis=1, keepdims=True) + 1e-8)
sim_raw = zh_rn @ en_rn.T
r_raw = np.array([np.where(np.argsort(-sim_raw[i]) == i)[0][0] for i in range(N_PROB)])
print(f"  Raw L32: Top-1={float((r_raw==0).mean()):.3f}  Top-5={float((r_raw<5).mean()):.3f}  "
      f"Mean rank={r_raw.mean():.1f}")
retrieval_results["L32_raw"] = {"top1": float((r_raw==0).mean()), "top5": float((r_raw<5).mean()), "mean_rank": float(r_raw.mean())}


# ── 5. Multilingual retrieval — the real test ────────────────────────

print("\n[5/5] 7-language retrieval: given any lang, find same problem in EN")

multi_retrieval = {}
for lang in all_langs:
    if lang == "en":
        continue
    acts = multi[f"{lang}_L32"] if f"{lang}_L32" in multi else lasttok[f"{lang}_L32"]
    k = to_kernel(acts)
    kn = k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-8)
    sim = kn @ en_k32_n.T
    r = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(N_PROB)])
    t1, t5, mr = (r == 0).mean(), (r < 5).mean(), r.mean()
    multi_retrieval[lang] = {"top1": float(t1), "top5": float(t5), "mean_rank": float(mr)}
    print(f"  {lang}→EN: Top-1={t1:.3f}  Top-5={t5:.3f}  Mean rank={mr:.1f}")


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  BILINGUAL ENCODER (train EN+ZH, test held-out problems):")
print(f"  Ridge R² by source layer:")
for L in sorted(results_bilingual.keys()):
    r = results_bilingual[L]
    parts = [f"{lang}={r[lang]['ridge']:+.4f}" for lang in ["en", "zh"] + LANGS_TEST if lang in r]
    print(f"    L{L:>2d}: {', '.join(parts)}")

print(f"\n  ANSWER-CLASS PREDICTION (kernel → 20-cluster labels):")
for L in sorted(answer_results.keys()):
    r = answer_results[L]
    parts = [f"{lang}={r[lang]:.3f}" for lang in ["en", "zh"] + LANGS_TEST if lang in r]
    print(f"    L{L:>2d}: {', '.join(parts)}")

print(f"\n  RETRIEVAL (kernel cosine, ZH→EN):")
for k, v in sorted(retrieval_results.items(), key=lambda x: x[0]):
    print(f"    {k}: Top-1={v['top1']:.3f}  Top-5={v['top5']:.3f}  Mean rank={v['mean_rank']:.1f}")

print(f"\n  7-LANGUAGE RETRIEVAL (kernel cosine, X→EN at L32):")
for lang, v in sorted(multi_retrieval.items()):
    print(f"    {lang}: Top-1={v['top1']:.3f}  Top-5={v['top5']:.3f}  Mean rank={v['mean_rank']:.1f}")

# ── Save ─────────────────────────────────────────────────────────────

output = {
    "experiment": "BF",
    "title": "Bilingual distillation",
    "bilingual_encoder": {str(k): v for k, v in results_bilingual.items()},
    "answer_class": {str(k): v for k, v in answer_results.items()},
    "retrieval": retrieval_results,
    "multilingual_retrieval": multi_retrieval,
    "cluster_ari": float(ari),
}

with open(OUT / "expBF_bilingual_distillation.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBF_bilingual_distillation.json")
