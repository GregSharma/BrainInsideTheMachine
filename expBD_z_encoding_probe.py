"""Exp BD: Z-encoding probe — distilling the encoder h.

The question: can a small model map from initial problem representation
to the language-agnostic Z-encoding? If yes, we've extracted h from h-f-h'.

Tests:
  1. Per-problem cross-lingual Z consistency: cos(Z_en, Z_zh) per problem per layer
     vs cos(Z_en(prob_i), Z_en(prob_j)) within-language different-problem
     → If same-problem cross-lingual >> different-problem within-language,
       Z encodes PROBLEM IDENTITY not LANGUAGE.

  2. Early→Late Z probe: can we predict Z(L32) from Z(L0)? Z(L4)? Z(L8)?
     Train on EN, test on ZH. If cross-lingual transfer works, the mapping
     is language-agnostic — we've distilled the encoder.

  3. Layer-by-layer Z fingerprint: at which layer does the problem's Z-identity
     crystallize? Measure within-problem cross-lingual cos at each layer.

  4. Raw embedding → Z probe: can we go from the model's raw layer-0 embedding
     directly to Z(L32)? This is the ultimate encoder distillation.

Data: all_layers_lasttok.npz (200 problems × 36 layers × 2048D × 2 langs)
      multilingual_all_layers.npz (200 problems × 36 layers × 2048D × 7 langs)
"""

import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

OUT = Path("output")
np.random.seed(42)

print("=" * 60)
print("  Exp BD: Z-encoding probe — distilling the encoder h")
print("=" * 60)

# ── 1. Load data and build Z-basis ───────────────────────────────────

print("\n[1/5] Loading activation data...")

lasttok = np.load(OUT / "all_layers_lasttok.npz")
categories = lasttok["categories"]  # (200,)
N_PROB = 200
Z_DIM = 20

# Build Z-basis at L32 from zh-en diffs (same as BC)
en_L32, zh_L32 = lasttok["en_L32"], lasttok["zh_L32"]
diffs = zh_L32 - en_L32
pca_z = PCA(n_components=Z_DIM)
pca_z.fit(diffs)
Z_basis = pca_z.components_  # (20, 2048)

# Load all layers for both languages
def get_layer(lang, layer):
    return lasttok[f"{lang}_L{layer}"]  # (200, 2048)

def to_z(activations):
    """Project 2048D activations into 20D Z-space."""
    return activations @ Z_basis.T

print(f"  Z-basis: 20D, explains {pca_z.explained_variance_ratio_.sum()*100:.1f}% of diff variance")
print(f"  Problems: {N_PROB}, Categories: {len(np.unique(categories))}")

# ── 2. Cross-lingual Z consistency per problem ──────────────────────

print("\n[2/5] Cross-lingual Z consistency (same-problem vs different-problem)...")

layer_consistency = {}
for L in range(36):
    en_z = to_z(get_layer("en", L))  # (200, 20)
    zh_z = to_z(get_layer("zh", L))  # (200, 20)

    # Same-problem cross-lingual: cos(en_z[i], zh_z[i])
    same_cos = []
    for i in range(N_PROB):
        a, b = en_z[i], zh_z[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            same_cos.append(np.dot(a, b) / (na * nb))
    same_cos = np.array(same_cos)

    # Different-problem within-language: cos(en_z[i], en_z[j]) for i≠j
    # Sample 1000 pairs for speed
    diff_cos = []
    for _ in range(1000):
        i, j = np.random.choice(N_PROB, 2, replace=False)
        a, b = en_z[i], en_z[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            diff_cos.append(np.dot(a, b) / (na * nb))
    diff_cos = np.array(diff_cos)

    layer_consistency[L] = {
        "same_problem_cross_lingual": float(same_cos.mean()),
        "same_std": float(same_cos.std()),
        "diff_problem_within_lang": float(diff_cos.mean()),
        "diff_std": float(diff_cos.std()),
        "separation": float(same_cos.mean() - diff_cos.mean()),
    }

    if L % 5 == 0 or L == 35:
        print(f"  L{L:>2d}: same-prob cross-lingual={same_cos.mean():+.4f}  "
              f"diff-prob within-lang={diff_cos.mean():+.4f}  "
              f"separation={same_cos.mean() - diff_cos.mean():+.4f}")

# Find crystallization layer (max separation)
best_layer = max(layer_consistency, key=lambda L: layer_consistency[L]["separation"])
print(f"\n  Peak separation at L{best_layer}: {layer_consistency[best_layer]['separation']:.4f}")

# ── 3. Early→Late Z probe: predict Z(L32) from Z(early) ────────────

print("\n[3/5] Early→Late Z probe: Z(L_early) → Z(L32)")

# Train/test split: 160 train, 40 test
idx = np.random.permutation(N_PROB)
train_idx, test_idx = idx[:160], idx[160:]

target_layer = 32
en_z_target = to_z(get_layer("en", target_layer))  # (200, 20)
zh_z_target = to_z(get_layer("zh", target_layer))

probe_results = {}
for source_layer in [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28]:
    en_z_source = to_z(get_layer("en", source_layer))
    zh_z_source = to_z(get_layer("zh", source_layer))

    # Train on EN
    X_tr = en_z_source[train_idx]
    Y_tr = en_z_target[train_idx]

    # Test on EN (in-language)
    X_te_en = en_z_source[test_idx]
    Y_te_en = en_z_target[test_idx]

    # Test on ZH (cross-lingual transfer!)
    X_te_zh = zh_z_source[test_idx]
    Y_te_zh = zh_z_target[test_idx]

    # Ridge probe
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr)

    r2_en = r2_score(Y_te_en, ridge.predict(X_te_en), multioutput='uniform_average')
    r2_zh = r2_score(Y_te_zh, ridge.predict(X_te_zh), multioutput='uniform_average')

    # MLP probe
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                       max_iter=500, early_stopping=True,
                       validation_fraction=0.15, random_state=42)
    mlp.fit(X_tr_s, Y_tr)

    r2_en_mlp = r2_score(Y_te_en, mlp.predict(scaler.transform(X_te_en)),
                         multioutput='uniform_average')
    r2_zh_mlp = r2_score(Y_te_zh, mlp.predict(scaler.transform(X_te_zh)),
                         multioutput='uniform_average')

    probe_results[source_layer] = {
        "ridge_en": float(r2_en), "ridge_zh": float(r2_zh),
        "mlp_en": float(r2_en_mlp), "mlp_zh": float(r2_zh_mlp),
        "cross_lingual_transfer": float(r2_zh),  # THE key metric
    }
    print(f"  L{source_layer:>2d}→L32: Ridge EN={r2_en:+.4f} ZH={r2_zh:+.4f}  "
          f"MLP EN={r2_en_mlp:+.4f} ZH={r2_zh_mlp:+.4f}")

# ── 4. Raw 2048D → Z(L32) probe (full embedding, not Z-projected) ──

print("\n[4/5] Raw embedding → Z(L32) probe (2048D input)")

raw_probe_results = {}
for source_layer in [0, 4, 8, 12, 16]:
    en_raw = get_layer("en", source_layer)  # (200, 2048)
    zh_raw = get_layer("zh", source_layer)

    X_tr = en_raw[train_idx]
    Y_tr = en_z_target[train_idx]

    # Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr, Y_tr)
    r2_en = r2_score(en_z_target[test_idx], ridge.predict(en_raw[test_idx]),
                     multioutput='uniform_average')
    r2_zh = r2_score(zh_z_target[test_idx], ridge.predict(zh_raw[test_idx]),
                     multioutput='uniform_average')

    # MLP
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    mlp = MLPRegressor(hidden_layer_sizes=(256, 64), activation='relu',
                       max_iter=500, early_stopping=True,
                       validation_fraction=0.15, random_state=42)
    mlp.fit(X_tr_s, Y_tr)
    r2_en_mlp = r2_score(en_z_target[test_idx],
                         mlp.predict(scaler.transform(en_raw[test_idx])),
                         multioutput='uniform_average')
    r2_zh_mlp = r2_score(zh_z_target[test_idx],
                         mlp.predict(scaler.transform(zh_raw[test_idx])),
                         multioutput='uniform_average')

    raw_probe_results[source_layer] = {
        "ridge_en": float(r2_en), "ridge_zh": float(r2_zh),
        "mlp_en": float(r2_en_mlp), "mlp_zh": float(r2_zh_mlp),
    }
    print(f"  Raw L{source_layer:>2d}→Z(L32): Ridge EN={r2_en:+.4f} ZH={r2_zh:+.4f}  "
          f"MLP EN={r2_en_mlp:+.4f} ZH={r2_zh_mlp:+.4f}")


# ── 5. 7-language Z consistency (multilingual) ──────────────────────

print("\n[5/5] 7-language Z consistency at L32...")

try:
    multi = np.load(OUT / "multilingual_all_layers.npz")
    LANGS = ["en", "zh", "es", "ar", "ja", "ko", "sw"]
    multi_z = {}
    for lang in LANGS:
        key = f"{lang}_L32"
        if key in multi:
            multi_z[lang] = to_z(multi[key])

    if len(multi_z) >= 2:
        # For each problem, compute mean pairwise cos across all language pairs
        n_langs = len(multi_z)
        lang_list = list(multi_z.keys())
        per_prob_consistency = []

        for i in range(N_PROB):
            cos_pairs = []
            for a_idx in range(n_langs):
                for b_idx in range(a_idx + 1, n_langs):
                    va = multi_z[lang_list[a_idx]][i]
                    vb = multi_z[lang_list[b_idx]][i]
                    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                    if na > 1e-8 and nb > 1e-8:
                        cos_pairs.append(np.dot(va, vb) / (na * nb))
            if cos_pairs:
                per_prob_consistency.append(np.mean(cos_pairs))

        per_prob_consistency = np.array(per_prob_consistency)
        print(f"  Mean 7-lang pairwise cos at L32: {per_prob_consistency.mean():.4f} ± {per_prob_consistency.std():.4f}")
        print(f"  Min: {per_prob_consistency.min():.4f}, Max: {per_prob_consistency.max():.4f}")
        print(f"  Problems with cos > 0.8: {(per_prob_consistency > 0.8).sum()}/{len(per_prob_consistency)}")

        multi_results = {
            "mean_consistency": float(per_prob_consistency.mean()),
            "std": float(per_prob_consistency.std()),
            "min": float(per_prob_consistency.min()),
            "max": float(per_prob_consistency.max()),
            "above_0.8": int((per_prob_consistency > 0.8).sum()),
            "per_problem": [float(x) for x in per_prob_consistency],
        }
    else:
        multi_results = {"error": "insufficient languages in multilingual data"}
        print("  Insufficient multilingual data")
except Exception as e:
    multi_results = {"error": str(e)}
    print(f"  Multilingual data not available: {e}")


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  CROSS-LINGUAL Z CONSISTENCY (same problem, different language):")
print(f"  {'Layer':<8s} {'Same-prob cross-lingual':>25s} {'Diff-prob within-lang':>25s} {'Separation':>12s}")
for L in [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]:
    c = layer_consistency[L]
    print(f"  L{L:<6d} {c['same_problem_cross_lingual']:>+25.4f} {c['diff_problem_within_lang']:>+25.4f} {c['separation']:>+12.4f}")

print(f"\n  ENCODER PROBE: Z(source) → Z(L32), trained on EN, tested on ZH:")
print(f"  {'Source':<10s} {'Ridge ZH':>10s} {'MLP ZH':>10s}")
for L in sorted(probe_results.keys()):
    p = probe_results[L]
    print(f"  L{L:<8d} {p['ridge_zh']:>+10.4f} {p['mlp_zh']:>+10.4f}")

print(f"\n  RAW EMBEDDING PROBE: Raw(source) → Z(L32), trained on EN, tested on ZH:")
for L in sorted(raw_probe_results.keys()):
    p = raw_probe_results[L]
    print(f"  L{L:<8d} Ridge ZH={p['ridge_zh']:>+.4f}  MLP ZH={p['mlp_zh']:>+.4f}")

# ── Save ─────────────────────────────────────────────────────────────

output = {
    "experiment": "BD",
    "title": "Z-encoding probe — distilling the encoder h",
    "z_dim": Z_DIM,
    "n_problems": N_PROB,
    "layer_consistency": layer_consistency,
    "probe_results": {str(k): v for k, v in probe_results.items()},
    "raw_probe_results": {str(k): v for k, v in raw_probe_results.items()},
    "multilingual_consistency": multi_results,
    "crystallization_layer": best_layer,
}

with open(OUT / "expBD_z_encoding_probe.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBD_z_encoding_probe.json")
