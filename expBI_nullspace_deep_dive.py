"""Exp BI: Null-space deep dive — follow-up to BH's 97% retrieval breakthrough.

BH showed the multi-language null-space at L32 gives 97% Top-1 retrieval
across all 7 languages. This experiment investigates:

  1. Which 6 problems fail? Are they the same across languages?
  2. Layer sweep: does this work at L8, L16, L24? Where does it crystallize?
  3. Null-space as encoder basis: does the continuous encoder probe
     work when we use the null-space instead of the kernel?
  4. Null-space dimensionality: what's the minimum dims needed for 95%+?
  5. Leave-language-out: build null-space from 6 languages, test on 7th
"""

import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

OUT = Path("output")
np.random.seed(42)

print("=" * 60)
print("  Exp BI: Null-space deep dive")
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
    """Build multi-language null-space at given layer."""
    all_diffs = []
    for i, la in enumerate(ALL_LANGS):
        for j, lb in enumerate(ALL_LANGS):
            if i >= j: continue
            d = get_acts(la, layer) - get_acts(lb, layer)
            all_diffs.append(d)
    all_diffs = np.vstack(all_diffs)
    _, S, Vt = np.linalg.svd(all_diffs, full_matrices=False)
    return Vt[-n_dims:], S  # bottom n_dims = null space

def retrieval_test(layer, null_proj, ref_lang="en"):
    """Test retrieval: for each lang, find same problem in ref_lang."""
    ref = get_acts(ref_lang, layer) @ null_proj.T
    ref_n = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    results = {}
    for lang in ALL_LANGS:
        if lang == ref_lang: continue
        acts = get_acts(lang, layer) @ null_proj.T
        acts_n = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
        sim = acts_n @ ref_n.T
        ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(N_PROB)])
        results[lang] = {
            "top1": float((ranks == 0).mean()),
            "top5": float((ranks < 5).mean()),
            "mean_rank": float(ranks.mean()),
            "failed_problems": [int(i) for i in np.where(ranks > 0)[0]],
        }
    return results


# ── 1. Which problems fail? ──────────────────────────────────────────

print("\n[1/5] Identifying failed problems at L32 (10D null-space)...")

null_proj_32, svals_32 = build_nullspace(32, 10)
ret_32 = retrieval_test(32, null_proj_32)

# Collect failed problems across all languages
all_failed = set()
per_lang_failed = {}
for lang, r in ret_32.items():
    failed = set(r["failed_problems"])
    all_failed |= failed
    per_lang_failed[lang] = failed

print(f"  Total unique failed problems: {len(all_failed)}")
for prob_id in sorted(all_failed):
    langs_failing = [l for l, f in per_lang_failed.items() if prob_id in f]
    cat = categories[prob_id]
    print(f"  Problem {prob_id}: category={cat}, fails in: {langs_failing}")

# Check if failures are consistent across languages
n_consistent = sum(1 for p in all_failed
                   if all(p in per_lang_failed.get(l, set()) for l in ALL_LANGS if l != "en"))
print(f"  Failures consistent across ALL languages: {n_consistent}/{len(all_failed)}")


# ── 2. Layer sweep ──────────────────────────────────────────────────

print("\n[2/5] Layer sweep: null-space retrieval at every 4th layer")

layer_sweep = {}
for L in range(0, 36, 4):
    try:
        null_proj, svals = build_nullspace(L, 10)
        ret = retrieval_test(L, null_proj)
        # Average Top-1 across all non-en languages
        avg_top1 = np.mean([r["top1"] for r in ret.values()])
        avg_top5 = np.mean([r["top5"] for r in ret.values()])
        avg_mr = np.mean([r["mean_rank"] for r in ret.values()])
        layer_sweep[L] = {
            "avg_top1": float(avg_top1), "avg_top5": float(avg_top5),
            "avg_mean_rank": float(avg_mr),
            "per_lang": {l: {"top1": r["top1"], "mean_rank": r["mean_rank"]}
                        for l, r in ret.items()},
        }
        print(f"  L{L:>2d}: avg Top-1={avg_top1:.3f}  Top-5={avg_top5:.3f}  "
              f"Mean rank={avg_mr:.1f}")
    except Exception as e:
        print(f"  L{L:>2d}: FAILED ({e})")


# ── 3. Minimum dimensionality ────────────────────────────────────────

print("\n[3/5] Minimum dimensionality for 95%+ retrieval at L32")

dim_sweep = {}
for n_dims in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]:
    null_proj, _ = build_nullspace(32, n_dims)
    ret = retrieval_test(32, null_proj)
    avg_top1 = np.mean([r["top1"] for r in ret.values()])
    dim_sweep[n_dims] = float(avg_top1)
    marker = " ***" if avg_top1 >= 0.95 else ""
    print(f"  {n_dims:>3d}D: avg Top-1={avg_top1:.3f}{marker}")


# ── 4. Null-space as encoder basis ───────────────────────────────────

print("\n[4/5] Null-space encoder probe: null(L_early) → null(L32)")

idx = np.random.permutation(N_PROB)
train_prob, test_prob = idx[:160], idx[160:]

encoder_results = {}
null_target, _ = build_nullspace(32, 20)

for source_layer in [0, 4, 8, 12, 16, 20, 24, 28]:
    null_source, _ = build_nullspace(source_layer, 20)

    # Project into null-space
    en_s = get_acts("en", source_layer) @ null_source.T
    zh_s = get_acts("zh", source_layer) @ null_source.T
    en_t = get_acts("en", 32) @ null_target.T
    zh_t = get_acts("zh", 32) @ null_target.T

    # Train on EN+ZH
    X_tr = np.vstack([en_s[train_prob], zh_s[train_prob]])
    Y_tr = np.vstack([en_t[train_prob], zh_t[train_prob]])

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr)

    layer_r = {}
    for lang in ALL_LANGS:
        s = get_acts(lang, source_layer) @ null_source.T
        t = get_acts(lang, 32) @ null_target.T
        pred = ridge.predict(s[test_prob])
        r2 = r2_score(t[test_prob], pred, multioutput='uniform_average')
        layer_r[lang] = float(r2)

    encoder_results[source_layer] = layer_r
    en_r = layer_r["en"]
    zh_r = layer_r["zh"]
    es_r = layer_r.get("es", 0)
    ja_r = layer_r.get("ja", 0)
    print(f"  L{source_layer:>2d}→L32: EN={en_r:+.4f} ZH={zh_r:+.4f} "
          f"ES={es_r:+.4f} JA={ja_r:+.4f}")


# ── 5. Leave-language-out null-space ─────────────────────────────────

print("\n[5/5] Leave-language-out: build null-space from 6 langs, test on 7th")

loo_retrieval = {}
for held_out in ALL_LANGS:
    train_langs = [l for l in ALL_LANGS if l != held_out]
    # Build null-space from 6 languages only
    all_diffs = []
    for i, la in enumerate(train_langs):
        for j, lb in enumerate(train_langs):
            if i >= j: continue
            d = get_acts(la, 32) - get_acts(lb, 32)
            all_diffs.append(d)
    all_diffs = np.vstack(all_diffs)
    _, S, Vt = np.linalg.svd(all_diffs, full_matrices=False)
    null_proj = Vt[-10:]

    # Test retrieval for held-out language → EN
    if held_out == "en":
        # Test en against zh
        ref_lang = "zh"
    else:
        ref_lang = "en"

    ref = get_acts(ref_lang, 32) @ null_proj.T
    ref_n = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    ho_acts = get_acts(held_out, 32) @ null_proj.T
    ho_n = ho_acts / (np.linalg.norm(ho_acts, axis=1, keepdims=True) + 1e-8)

    sim = ho_n @ ref_n.T
    ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(N_PROB)])
    t1 = (ranks == 0).mean()
    t5 = (ranks < 5).mean()

    loo_retrieval[held_out] = {"top1": float(t1), "top5": float(t5),
                                "mean_rank": float(ranks.mean())}
    print(f"  Hold out {held_out}: Top-1={t1:.3f} Top-5={t5:.3f} "
          f"Mean rank={ranks.mean():.1f}")


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  FAILED PROBLEMS: {len(all_failed)} unique out of 200")
print(f"  LAYER SWEEP (avg Top-1 across 6 non-en languages):")
for L in sorted(layer_sweep.keys()):
    v = layer_sweep[L]
    print(f"    L{L:>2d}: {v['avg_top1']:.3f}")

print(f"\n  DIMENSIONALITY (Top-1 at L32):")
for d in sorted(dim_sweep.keys()):
    print(f"    {d:>3d}D: {dim_sweep[d]:.3f}")

print(f"\n  ENCODER PROBE (null-space, train EN+ZH):")
for L in sorted(encoder_results.keys()):
    r = encoder_results[L]
    es = r.get("es", 0)
    print(f"    L{L:>2d}: EN={r['en']:+.3f} ZH={r['zh']:+.3f} ES={es:+.3f}")

print(f"\n  LEAVE-LANGUAGE-OUT:")
for lang, v in sorted(loo_retrieval.items()):
    print(f"    {lang}: Top-1={v['top1']:.3f}")

# Save
output = {
    "experiment": "BI",
    "title": "Null-space deep dive",
    "failed_problems": sorted(list(all_failed)),
    "failed_categories": {int(p): int(categories[p]) for p in all_failed},
    "layer_sweep": {str(k): v for k, v in layer_sweep.items()},
    "dim_sweep": {str(k): v for k, v in dim_sweep.items()},
    "encoder_results": {str(k): v for k, v in encoder_results.items()},
    "loo_retrieval": loo_retrieval,
}

with open(OUT / "expBI_nullspace_deep_dive.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBI_nullspace_deep_dive.json")
