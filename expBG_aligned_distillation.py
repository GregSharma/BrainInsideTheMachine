"""Exp BG: Aligned distillation — mean-centered + CCA + all-language training.

BF showed bilingual encoder works (ZH R²=0.87) but doesn't generalize to
new languages. Three hypotheses for why:
  1. Language-specific means shift the representations
  2. The kernel basis itself is biased toward training languages
  3. The computation path is genuinely language-specific

Tests:
  1. Mean-centered probe: subtract per-language mean before probing
  2. CCA alignment: find maximally correlated subspace across languages
  3. All-7-language training with leave-one-language-out test
  4. Contrastive kernel: learn projection that maximizes same-problem
     cross-lingual similarity (Siamese-style triplet loss)
"""

import json
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from pathlib import Path

OUT = Path("output")
np.random.seed(42)

print("=" * 60)
print("  Exp BG: Aligned distillation")
print("=" * 60)

# ── 1. Load data ────────────────────────────────────────────────────

print("\n[1/5] Loading...")

lasttok = np.load(OUT / "all_layers_lasttok.npz")
multi = np.load(OUT / "multilingual_all_layers.npz")
categories = lasttok["categories"]
N_PROB = 200

ALL_LANGS = [l for l in ["en", "zh", "es", "ar", "ja", "ko", "sw"] if f"{l}_L32" in multi]
print(f"  Languages: {ALL_LANGS}")

def get_acts(lang, layer):
    key = f"{lang}_L{layer}"
    return multi[key] if key in multi else lasttok[key]

# Build kernel basis from all 7 languages
stacked_32 = np.stack([get_acts(l, 32) for l in ALL_LANGS], axis=0)
prob_means = stacked_32.mean(axis=0, keepdims=True)
deviations = (stacked_32 - prob_means).reshape(-1, 2048)
_, _, Vt = np.linalg.svd(deviations, full_matrices=False)
lang_basis = Vt[:10]
K = 50

pooled = stacked_32.reshape(-1, 2048)
pooled_clean = pooled - (pooled @ lang_basis.T) @ lang_basis
pca = PCA(n_components=K)
pca.fit(pooled_clean)
kernel_basis = pca.components_

def to_kernel(acts):
    clean = acts - (acts @ lang_basis.T) @ lang_basis
    return clean @ kernel_basis.T

idx = np.random.permutation(N_PROB)
train_prob, test_prob = idx[:160], idx[160:]


# ── 2. Mean-centered probe ──────────────────────────────────────────

print("\n[2/5] Mean-centered probe: remove language mean, then probe")

mc_results = {}
for source_layer in [0, 8, 16, 24, 28]:
    # Compute per-language mean at source and target layer
    lang_means_source = {}
    lang_means_target = {}
    for lang in ALL_LANGS:
        s = to_kernel(get_acts(lang, source_layer))
        t = to_kernel(get_acts(lang, 32))
        lang_means_source[lang] = s.mean(axis=0)
        lang_means_target[lang] = t.mean(axis=0)

    # Train on EN+ZH (mean-centered)
    X_tr, Y_tr = [], []
    for lang in ["en", "zh"]:
        s = to_kernel(get_acts(lang, source_layer)) - lang_means_source[lang]
        t = to_kernel(get_acts(lang, 32)) - lang_means_target[lang]
        X_tr.append(s[train_prob])
        Y_tr.append(t[train_prob])
    X_tr, Y_tr = np.vstack(X_tr), np.vstack(Y_tr)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr)

    layer_r = {}
    for lang in ALL_LANGS:
        s = to_kernel(get_acts(lang, source_layer)) - lang_means_source[lang]
        t = to_kernel(get_acts(lang, 32)) - lang_means_target[lang]
        pred = ridge.predict(s[test_prob])
        r2 = r2_score(t[test_prob], pred, multioutput='uniform_average')
        layer_r[lang] = float(r2)

    mc_results[source_layer] = layer_r
    en_r = layer_r["en"]
    zh_r = layer_r["zh"]
    es_r = layer_r.get("es", float("nan"))
    ja_r = layer_r.get("ja", float("nan"))
    print(f"  L{source_layer:>2d}: EN={en_r:+.4f} ZH={zh_r:+.4f} ES={es_r:+.4f} JA={ja_r:+.4f}")


# ── 3. All-language training ─────────────────────────────────────────

print("\n[3/5] All-language training, leave-one-out test")

loo_results = {}
for source_layer in [0, 8, 16, 24, 28]:
    for held_out in ALL_LANGS:
        train_langs = [l for l in ALL_LANGS if l != held_out]

        X_tr, Y_tr = [], []
        for lang in train_langs:
            s = to_kernel(get_acts(lang, source_layer))
            t = to_kernel(get_acts(lang, 32))
            X_tr.append(s[train_prob])
            Y_tr.append(t[train_prob])
        X_tr, Y_tr = np.vstack(X_tr), np.vstack(Y_tr)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, Y_tr)

        s_te = to_kernel(get_acts(held_out, source_layer))
        t_te = to_kernel(get_acts(held_out, 32))
        pred = ridge.predict(s_te[test_prob])
        r2 = r2_score(t_te[test_prob], pred, multioutput='uniform_average')

        if source_layer not in loo_results:
            loo_results[source_layer] = {}
        loo_results[source_layer][held_out] = float(r2)

    parts = [f"{l}={loo_results[source_layer][l]:+.3f}" for l in ALL_LANGS]
    print(f"  L{source_layer:>2d}: {', '.join(parts)}")


# ── 4. CCA alignment then probe ─────────────────────────────────────

print("\n[4/5] CCA: find maximally correlated subspace between EN and ZH, test on others")

cca_results = {}
for source_layer in [8, 16, 24, 28]:
    en_k = to_kernel(get_acts("en", source_layer))
    zh_k = to_kernel(get_acts("zh", source_layer))

    n_cca = min(20, K)
    cca = CCA(n_components=n_cca, max_iter=1000)
    cca.fit(en_k[train_prob], zh_k[train_prob])

    # Transform all languages into CCA space
    en_target = to_kernel(get_acts("en", 32))

    # Train probe: CCA(source) → kernel(L32)
    en_cca = cca.transform(en_k)
    X_tr = en_cca[0][train_prob] if isinstance(en_cca, tuple) else en_cca[train_prob]
    Y_tr = en_target[train_prob]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr)

    layer_r = {}
    for lang in ALL_LANGS:
        lang_k = to_kernel(get_acts(lang, source_layer))
        # Transform using the X-side of CCA (treats all langs as X)
        lang_cca = cca.transform(lang_k)
        lc = lang_cca[0] if isinstance(lang_cca, tuple) else lang_cca
        t_te = to_kernel(get_acts(lang, 32))
        pred = ridge.predict(lc[test_prob])
        r2 = r2_score(t_te[test_prob], pred, multioutput='uniform_average')
        layer_r[lang] = float(r2)

    cca_results[source_layer] = layer_r
    parts = [f"{l}={layer_r[l]:+.3f}" for l in ALL_LANGS]
    print(f"  L{source_layer:>2d}: {', '.join(parts)}")


# ── 5. Contrastive retrieval: learned metric ─────────────────────────

print("\n[5/5] Contrastive retrieval: train projection to maximize same-problem similarity")

# Simple approach: learn W such that ||W@en_i - W@zh_i||² is small
# and ||W@en_i - W@en_j||² is large (for i≠j)
# This is equivalent to learning a projection that clusters same-problem
# representations across languages

# Use a simple approach: fit W to minimize ||W@en - W@zh|| for same problems
# This is just W = argmin ||W(en - zh)||² subject to ||W|| = 1
# = PCA on (en - zh), then take NULL space = directions where en ≈ zh

en32 = to_kernel(get_acts("en", 32))[train_prob]
zh32 = to_kernel(get_acts("zh", 32))[train_prob]

# Directions where en ≈ zh: null space of (en - zh)
diffs = en32 - zh32
U_d, S_d, Vt_d = np.linalg.svd(diffs, full_matrices=False)

# Low singular values = directions where en-zh diff is small = shared
# Take bottom 30 directions (shared subspace)
n_shared = 30
shared_proj = Vt_d[-n_shared:]  # (30, K)

# Project and test retrieval
en_all = to_kernel(get_acts("en", 32)) @ shared_proj.T
en_n = en_all / (np.linalg.norm(en_all, axis=1, keepdims=True) + 1e-8)

contrastive_retrieval = {}
for lang in ALL_LANGS:
    if lang == "en":
        continue
    lang_all = to_kernel(get_acts(lang, 32)) @ shared_proj.T
    lang_n = lang_all / (np.linalg.norm(lang_all, axis=1, keepdims=True) + 1e-8)

    sim = lang_n @ en_n.T
    ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(N_PROB)])
    t1 = float((ranks == 0).mean())
    t5 = float((ranks < 5).mean())
    mr = float(ranks.mean())
    contrastive_retrieval[lang] = {"top1": t1, "top5": t5, "mean_rank": mr}
    print(f"  {lang}→EN (shared-30D): Top-1={t1:.3f} Top-5={t5:.3f} Mean rank={mr:.1f}")

# Compare to full kernel retrieval from BF
print("\n  Comparison to full kernel (BF):")
en_full = to_kernel(get_acts("en", 32))
en_fn = en_full / (np.linalg.norm(en_full, axis=1, keepdims=True) + 1e-8)
for lang in ["zh", "es", "ja"]:
    lang_full = to_kernel(get_acts(lang, 32))
    lang_fn = lang_full / (np.linalg.norm(lang_full, axis=1, keepdims=True) + 1e-8)
    sim = lang_fn @ en_fn.T
    ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(N_PROB)])
    print(f"  {lang}→EN (full-{K}D): Top-1={(ranks==0).mean():.3f} "
          f"Top-5={(ranks<5).mean():.3f} Mean rank={ranks.mean():.1f}")

# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  MEAN-CENTERED PROBE (train EN+ZH centered, test all):")
for L in sorted(mc_results.keys()):
    parts = [f"{l}={mc_results[L][l]:+.3f}" for l in ALL_LANGS]
    print(f"    L{L:>2d}: {', '.join(parts)}")

print(f"\n  LEAVE-ONE-OUT (train 6 langs, test held-out):")
for L in sorted(loo_results.keys()):
    parts = [f"{l}={loo_results[L][l]:+.3f}" for l in ALL_LANGS]
    print(f"    L{L:>2d}: {', '.join(parts)}")

print(f"\n  CCA PROBE (CCA on EN+ZH, probe all):")
for L in sorted(cca_results.keys()):
    parts = [f"{l}={cca_results[L][l]:+.3f}" for l in ALL_LANGS]
    print(f"    L{L:>2d}: {', '.join(parts)}")

print(f"\n  CONTRASTIVE RETRIEVAL (shared null-space, 30D):")
for lang, v in sorted(contrastive_retrieval.items()):
    print(f"    {lang}→EN: Top-1={v['top1']:.3f} Top-5={v['top5']:.3f} Mean rank={v['mean_rank']:.1f}")

# ── Save ─────────────────────────────────────────────────────────────

output = {
    "experiment": "BG",
    "title": "Aligned distillation — mean-centered, CCA, all-language, contrastive",
    "mean_centered": {str(k): v for k, v in mc_results.items()},
    "leave_one_out": {str(k): v for k, v in loo_results.items()},
    "cca": {str(k): v for k, v in cca_results.items()},
    "contrastive_retrieval": contrastive_retrieval,
}

with open(OUT / "expBG_aligned_distillation.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBG_aligned_distillation.json")
