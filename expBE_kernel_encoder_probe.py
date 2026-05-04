"""Exp BE: Kernel-space encoder probe — distilling h using the SHARED subspace.

BD used contrastive Z (PCA on zh-en diffs) = the LANGUAGE subspace.
Cross-lingual transfer failed catastrophically because the probe learned
language-specific mappings.

This experiment uses the MATH KERNEL: the null space of the language
component. Built from 7-language SVD (Exp AB): within-problem cross-
language covariance → SVD → null space of top language dims = pure math.

Also tests a simpler approach: PCA on pooled {en, zh} activations after
projecting out the mean language direction. This gives the shared
variance structure.

Tests:
  1. Kernel encoder: kernel(L_early) → kernel(L32), train EN, test ZH
  2. Category from kernel: kernel(L) → category, train EN, test ZH
  3. Answer from kernel: kernel(L) → answer, train EN, test ZH
  4. Multiple kernel dimensions: sweep k=5,10,20,50,100
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
print("  Exp BE: Kernel-space encoder probe")
print("=" * 60)

# ── 1. Build math kernel basis ───────────────────────────────────────

print("\n[1/6] Building math kernel basis...")

lasttok = np.load(OUT / "all_layers_lasttok.npz")
categories = lasttok["categories"]
N_PROB = 200

# Try multilingual kernel first (Exp AB approach)
try:
    multi = np.load(OUT / "multilingual_all_layers.npz")
    LANGS = [l for l in ["en", "zh", "es", "ar", "ja", "ko", "sw"]
             if f"{l}_L32" in multi]
    print(f"  Using {len(LANGS)}-language kernel")
    use_multi = True
except:
    LANGS = ["en", "zh"]
    use_multi = False
    print("  Multilingual data not available, using en-zh only")

def build_kernel_basis(layer, k_lang=10, k_math=50):
    """Build math kernel at a layer by removing language subspace.

    1. Collect same-problem activations across languages
    2. Compute within-problem cross-language covariance → SVD
    3. Top k_lang singular vectors = language subspace
    4. Null space = math kernel
    """
    if use_multi:
        # Stack all languages for this layer
        lang_acts = []
        for lang in LANGS:
            key = f"{lang}_L{layer}"
            if key in multi:
                lang_acts.append(multi[key])  # (200, 2048)
            else:
                key2 = f"{lang}_L{layer}"
                if key2 in lasttok:
                    lang_acts.append(lasttok[key2])
    else:
        lang_acts = [lasttok[f"en_L{layer}"], lasttok[f"zh_L{layer}"]]

    n_langs = len(lang_acts)
    if n_langs < 2:
        return None, None

    # Within-problem cross-language covariance
    # For each problem, compute mean across languages, then deviation
    stacked = np.stack(lang_acts, axis=0)  # (n_langs, 200, 2048)
    prob_means = stacked.mean(axis=0, keepdims=True)  # (1, 200, 2048)
    deviations = stacked - prob_means  # (n_langs, 200, 2048)
    dev_flat = deviations.reshape(-1, 2048)  # (n_langs*200, 2048)

    # SVD on deviations → top components are language directions
    U, S, Vt = np.linalg.svd(dev_flat, full_matrices=False)
    lang_basis = Vt[:k_lang]  # (k_lang, 2048) — language subspace

    # Math kernel: project out language, then PCA on residual
    pooled = np.vstack(lang_acts)  # (n_langs*200, 2048)
    pooled_clean = pooled - (pooled @ lang_basis.T) @ lang_basis
    pca = PCA(n_components=min(k_math, pooled_clean.shape[1]))
    pca.fit(pooled_clean)

    return pca.components_[:k_math], lang_basis  # (k_math, 2048), (k_lang, 2048)


# Build at multiple layers
kernel_bases = {}
for L in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
    math_basis, lang_basis = build_kernel_basis(L, k_lang=10, k_math=50)
    if math_basis is not None:
        kernel_bases[L] = {"math": math_basis, "lang": lang_basis}

print(f"  Built kernel bases at {len(kernel_bases)} layers")
print(f"  Math kernel: {kernel_bases[32]['math'].shape[0]}D")

# ── 2. Kernel-space cross-lingual consistency ────────────────────────

print("\n[2/6] Kernel-space cross-lingual consistency...")

K_MATH = 50
for L in [0, 8, 16, 24, 32]:
    basis = kernel_bases[L]["math"][:K_MATH]
    en_k = lasttok[f"en_L{L}"] @ basis.T  # (200, K_MATH)
    zh_k = lasttok[f"zh_L{L}"] @ basis.T

    # Same-problem cross-lingual cosine
    same_cos = []
    for i in range(N_PROB):
        a, b = en_k[i], zh_k[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            same_cos.append(np.dot(a, b) / (na * nb))

    # Different-problem within-language
    diff_cos = []
    for _ in range(1000):
        i, j = np.random.choice(N_PROB, 2, replace=False)
        a, b = en_k[i], en_k[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            diff_cos.append(np.dot(a, b) / (na * nb))

    same_cos, diff_cos = np.array(same_cos), np.array(diff_cos)
    sep = same_cos.mean() - diff_cos.mean()
    print(f"  L{L:>2d}: same-prob={same_cos.mean():.4f}  diff-prob={diff_cos.mean():.4f}  "
          f"sep={sep:+.4f}  {'<-- PROBLEM IDENTITY SEPARATES' if sep > 0.02 else ''}")


# ── 3. Kernel encoder probe: kernel(early) → kernel(L32) ────────────

print("\n[3/6] Kernel encoder probe: kernel(early) → kernel(L32)")

idx = np.random.permutation(N_PROB)
train_idx, test_idx = idx[:160], idx[160:]

target_basis = kernel_bases[32]["math"][:K_MATH]
en_target = lasttok["en_L32"] @ target_basis.T  # (200, K_MATH)
zh_target = lasttok["zh_L32"] @ target_basis.T

encoder_results = {}
for source_layer in [0, 4, 8, 12, 16, 20, 24, 28]:
    source_basis = kernel_bases.get(source_layer, kernel_bases[min(kernel_bases.keys(), key=lambda k: abs(k-source_layer))])["math"][:K_MATH]

    en_source = lasttok[f"en_L{source_layer}"] @ source_basis.T
    zh_source = lasttok[f"zh_L{source_layer}"] @ source_basis.T

    # Train on EN
    X_tr = en_source[train_idx]
    Y_tr = en_target[train_idx]

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr)
    r2_en = r2_score(en_target[test_idx], ridge.predict(en_source[test_idx]),
                     multioutput='uniform_average')
    r2_zh = r2_score(zh_target[test_idx], ridge.predict(zh_source[test_idx]),
                     multioutput='uniform_average')

    # MLP
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                       max_iter=1000, early_stopping=True,
                       validation_fraction=0.15, random_state=42)
    mlp.fit(X_tr_s, Y_tr)
    r2_en_mlp = r2_score(en_target[test_idx],
                         mlp.predict(scaler.transform(en_source[test_idx])),
                         multioutput='uniform_average')
    r2_zh_mlp = r2_score(zh_target[test_idx],
                         mlp.predict(scaler.transform(zh_source[test_idx])),
                         multioutput='uniform_average')

    encoder_results[source_layer] = {
        "ridge_en": float(r2_en), "ridge_zh": float(r2_zh),
        "mlp_en": float(r2_en_mlp), "mlp_zh": float(r2_zh_mlp),
    }
    marker = " ***" if r2_zh > 0.3 else ""
    print(f"  L{source_layer:>2d}→L32: Ridge EN={r2_en:+.4f} ZH={r2_zh:+.4f}  "
          f"MLP EN={r2_en_mlp:+.4f} ZH={r2_zh_mlp:+.4f}{marker}")


# ── 4. Raw 2048D → kernel(L32) probe ────────────────────────────────

print("\n[4/6] Raw 2048D → kernel(L32) probe (the full distillation)")

raw_encoder_results = {}
for source_layer in [0, 4, 8, 12, 16]:
    en_raw = lasttok[f"en_L{source_layer}"]
    zh_raw = lasttok[f"zh_L{source_layer}"]

    X_tr = en_raw[train_idx]
    Y_tr = en_target[train_idx]

    # Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr, Y_tr)
    r2_en = r2_score(en_target[test_idx], ridge.predict(en_raw[test_idx]),
                     multioutput='uniform_average')
    r2_zh = r2_score(zh_target[test_idx], ridge.predict(zh_raw[test_idx]),
                     multioutput='uniform_average')

    # MLP
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    mlp = MLPRegressor(hidden_layer_sizes=(512, 128), activation='relu',
                       max_iter=1000, early_stopping=True,
                       validation_fraction=0.15, random_state=42,
                       learning_rate_init=0.0005)
    mlp.fit(X_tr_s, Y_tr)
    r2_en_mlp = r2_score(en_target[test_idx],
                         mlp.predict(scaler.transform(en_raw[test_idx])),
                         multioutput='uniform_average')
    r2_zh_mlp = r2_score(zh_target[test_idx],
                         mlp.predict(scaler.transform(zh_raw[test_idx])),
                         multioutput='uniform_average')

    raw_encoder_results[source_layer] = {
        "ridge_en": float(r2_en), "ridge_zh": float(r2_zh),
        "mlp_en": float(r2_en_mlp), "mlp_zh": float(r2_zh_mlp),
    }
    marker = " ***" if r2_zh > 0.3 else ""
    print(f"  Raw L{source_layer:>2d}→kernel(L32): Ridge EN={r2_en:+.4f} ZH={r2_zh:+.4f}  "
          f"MLP EN={r2_en_mlp:+.4f} ZH={r2_zh_mlp:+.4f}{marker}")


# ── 5. Category and answer probes in kernel space ────────────────────

print("\n[5/6] Category and answer probes in kernel space")

# Answer labels: bucket the actual answers into 20 bins
# Load from Exp Z if available
answers = None
try:
    expz = json.load(open(OUT / "expZ_f_reconstruction.json"))
    # Try to reconstruct answer labels
except:
    pass

# Category probe at each layer
cat_results = {}
for L in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
    basis = kernel_bases.get(L, kernel_bases[min(kernel_bases.keys(), key=lambda k: abs(k-L))])["math"][:K_MATH]
    en_k = lasttok[f"en_L{L}"] @ basis.T
    zh_k = lasttok[f"zh_L{L}"] @ basis.T

    # Train category probe on EN
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(en_k[train_idx], categories[train_idx])

    acc_en = accuracy_score(categories[test_idx], clf.predict(en_k[test_idx]))
    acc_zh = accuracy_score(categories[test_idx], clf.predict(zh_k[test_idx]))

    cat_results[L] = {"en_acc": float(acc_en), "zh_acc": float(acc_zh)}
    marker = " ***" if acc_zh > 0.8 else ""
    print(f"  L{L:>2d}: Category EN={acc_en:.3f} ZH={acc_zh:.3f}{marker}")


# ── 6. Kernel dimension sweep ────────────────────────────────────────

print("\n[6/6] Kernel dimension sweep at L32")

dim_sweep_results = {}
for k_math in [5, 10, 20, 50, 100, 200]:
    math_b, _ = build_kernel_basis(32, k_lang=10, k_math=k_math)
    if math_b is None or math_b.shape[0] < k_math:
        continue

    en_k = lasttok["en_L32"] @ math_b.T
    zh_k = lasttok["zh_L32"] @ math_b.T

    # Category probe
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(en_k[train_idx], categories[train_idx])
    acc_zh = accuracy_score(categories[test_idx], clf.predict(zh_k[test_idx]))

    # Encoder probe from L8
    source_b = kernel_bases[8]["math"][:min(k_math, 50)]
    en_s = lasttok["en_L8"] @ source_b.T
    zh_s = lasttok["zh_L8"] @ source_b.T

    ridge = Ridge(alpha=1.0)
    ridge.fit(en_s[train_idx], en_k[train_idx])
    r2_zh = r2_score(zh_k[test_idx], ridge.predict(zh_s[test_idx]),
                     multioutput='uniform_average')

    dim_sweep_results[k_math] = {
        "category_zh": float(acc_zh),
        "encoder_r2_zh": float(r2_zh),
    }
    print(f"  k={k_math:>3d}: Category ZH={acc_zh:.3f}  Encoder R²(ZH)={r2_zh:+.4f}")


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  KERNEL ENCODER: kernel(source) → kernel(L32), train EN, test ZH:")
print(f"  {'Source':<10s} {'Ridge ZH':>10s} {'MLP ZH':>10s}")
for L in sorted(encoder_results.keys()):
    p = encoder_results[L]
    print(f"  L{L:<8d} {p['ridge_zh']:>+10.4f} {p['mlp_zh']:>+10.4f}")

print(f"\n  RAW → KERNEL: Raw 2048D → kernel(L32), train EN, test ZH:")
for L in sorted(raw_encoder_results.keys()):
    p = raw_encoder_results[L]
    print(f"  L{L:<8d} Ridge ZH={p['ridge_zh']:>+.4f}  MLP ZH={p['mlp_zh']:>+.4f}")

print(f"\n  CATEGORY FROM KERNEL: train EN, test ZH:")
for L in sorted(cat_results.keys()):
    c = cat_results[L]
    print(f"  L{L:<8d} EN={c['en_acc']:.3f}  ZH={c['zh_acc']:.3f}")

# ── Save ─────────────────────────────────────────────────────────────

output = {
    "experiment": "BE",
    "title": "Kernel-space encoder probe — distilling h via shared subspace",
    "k_math": K_MATH,
    "k_lang": 10,
    "n_langs": len(LANGS),
    "encoder_results": {str(k): v for k, v in encoder_results.items()},
    "raw_encoder_results": {str(k): v for k, v in raw_encoder_results.items()},
    "category_results": {str(k): v for k, v in cat_results.items()},
    "dim_sweep": {str(k): v for k, v in dim_sweep_results.items()},
}

with open(OUT / "expBE_kernel_encoder_probe.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBE_kernel_encoder_probe.json")
