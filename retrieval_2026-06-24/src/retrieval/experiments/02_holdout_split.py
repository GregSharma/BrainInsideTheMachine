#!/usr/bin/env python3
"""Stratified IS/OOS holdout on the aggregation sweep.

Splits 20 positives / 20 negatives into 10 IS + 10 OOS each (seed-fixed).
Selects best (chunk_feat, cue_feat, scorer, layer) cell on IS.
Reports OOS AUC for that cell.

Also reports OOS for two preregistered configs that involve zero selection:
  - last_token_cos: (h, h, cos_last, L27)
  - cross_cos_mean_plateau: average of cross_cos_mean across L27,28,29,32,34,35

If best-IS-OOS gap is small (< 0.05): the 0.990 is real.
If gap is large (> 0.10): selection inflation is dominant.
"""
import sys, json, time, torch
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/greg/Desktop/Projects/BrainInsideTheMachine")

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieval.features import HiddenState, QProj, KProj, VProj, SequenceInnovation
from retrieval.scorers import (CosineLast, BilinearLast, CrossMean, CrossMax,
                                CrossSoftMax, CrossCosineMean, CrossCosineMax, CrossTopK)
from retrieval.pipeline import encode_bare, run_grid, auc
from retrieval.datasets import load_uncertainty, CUES


def split_stratified(pos, neg, seed=12061):
    """5/5 IS+OOS per class on 10 items, 10/10 on 20. Same idea as before."""
    rng = np.random.RandomState(seed)
    n_pos, n_neg = len(pos), len(neg)
    assert n_pos == n_neg, "expecting matched-pair dataset"
    half = n_pos // 2
    pos_perm = rng.permutation(n_pos)
    neg_perm = rng.permutation(n_neg)
    is_pos_idx = pos_perm[:half].tolist()
    oos_pos_idx = pos_perm[half:].tolist()
    is_neg_idx = neg_perm[:half].tolist()
    oos_neg_idx = neg_perm[half:].tolist()
    return is_pos_idx, oos_pos_idx, is_neg_idx, oos_neg_idx


def main():
    t0 = time.time()
    print("loading model...", flush=True)
    MODEL = "Qwen/Qwen2.5-3B"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    pos, neg = load_uncertainty()
    cue_text = CUES["uncertainty"]

    # split
    is_pos_idx, oos_pos_idx, is_neg_idx, oos_neg_idx = split_stratified(pos, neg, seed=12061)
    print(f"split: IS = {len(is_pos_idx)}+{len(is_neg_idx)}, OOS = {len(oos_pos_idx)}+{len(oos_neg_idx)}", flush=True)

    all_texts = pos + neg
    labels = [1]*len(pos) + [0]*len(neg)
    # cache_idx mapping: pos[i] -> i, neg[j] -> len(pos)+j
    is_idx = is_pos_idx + [len(pos) + i for i in is_neg_idx]
    oos_idx = oos_pos_idx + [len(pos) + i for i in oos_neg_idx]

    print(f"encoding {len(all_texts)} chunks + 1 cue...", flush=True)
    caches = [encode_bare(t, tok, model) for t in all_texts]
    cue_cache = encode_bare(cue_text, tok, model)

    h = HiddenState()
    seq_inn = SequenceInnovation(k=4)
    chunk_features = [h, QProj(h), KProj(h), VProj(h), seq_inn, QProj(seq_inn), KProj(seq_inn)]
    cue_features = chunk_features
    scorers = [CosineLast(), BilinearLast(), CrossMean(), CrossMax(),
               CrossSoftMax(), CrossCosineMean(), CrossCosineMax(),
               CrossTopK(k=5), CrossTopK(k=20)]
    layers = [2, 18, 20, 23, 27, 28, 29, 32, 34, 35]

    # Run full grid; we'll project IS/OOS post-hoc
    print(f"running full grid on all 40 chunks...", flush=True)

    # Use run_grid but we need per-chunk scores, not just AUC.
    # Run on full set, get AUC for all 40, then re-run with subset selection.
    # Simpler: do the extraction once, score once per chunk, compute IS-AUC and OOS-AUC.

    from retrieval.pipeline import _extract_to_device
    from tqdm import tqdm

    # Pre-extract
    all_features = {f.name: f for f in chunk_features + cue_features}
    feature_list = list(all_features.values())
    device = "cuda"

    extractions = {}
    all_caches = caches + [cue_cache]
    cue_idx = len(caches)
    print(f"extracting {len(layers) * len(feature_list) * len(all_caches)} feature tensors...", flush=True)
    pbar = tqdm(total=len(layers)*len(feature_list)*len(all_caches), desc="extract")
    for L in layers:
        for f in feature_list:
            for ci, c in enumerate(all_caches):
                extractions[(ci, f.name, L)] = _extract_to_device(f, c, L, model, device)
                pbar.update(1)
    pbar.close()

    # Score each cell, splitting into IS and OOS
    print(f"scoring {len(layers)*len(chunk_features)*len(cue_features)*len(scorers)} cells...", flush=True)
    is_results, oos_results = {}, {}
    pbar = tqdm(total=len(layers)*len(chunk_features)*len(cue_features)*len(scorers), desc="score")
    for L in layers:
        for cf in chunk_features:
            for qf in cue_features:
                qv = extractions[(cue_idx, qf.name, L)]
                for scorer in scorers:
                    all_scores = [scorer(extractions[(i, cf.name, L)], qv, d_h=qv.shape[-1])
                                  for i in range(len(caches))]
                    is_pos_sc = [all_scores[i] for i in is_pos_idx]
                    is_neg_sc = [all_scores[len(pos)+i] for i in is_neg_idx]
                    oos_pos_sc = [all_scores[i] for i in oos_pos_idx]
                    oos_neg_sc = [all_scores[len(pos)+i] for i in oos_neg_idx]
                    key = f"{cf.name}|{qf.name}|{scorer.name}|L{L}"
                    is_results[key] = round(auc(is_pos_sc, is_neg_sc), 4)
                    oos_results[key] = round(auc(oos_pos_sc, oos_neg_sc), 4)
                    pbar.update(1)
    pbar.close()

    # Best on IS
    best_is_key = max(is_results, key=is_results.get)
    print(f"\nBest on IS: {best_is_key}", flush=True)
    print(f"  IS AUC: {is_results[best_is_key]}", flush=True)
    print(f"  OOS AUC for same cell: {oos_results[best_is_key]}", flush=True)
    print(f"  Selection gap (IS - OOS): {is_results[best_is_key] - oos_results[best_is_key]:+.3f}", flush=True)

    # Top 10 by IS, their OOS
    print(f"\nTop 10 IS cells, with OOS:", flush=True)
    top10_is = sorted(is_results.items(), key=lambda x: -x[1])[:10]
    for k, v in top10_is:
        oos = oos_results[k]
        gap = v - oos
        flag = "***" if gap > 0.10 else ("" if gap > -0.05 else "??")
        print(f"  IS={v:.3f}  OOS={oos:.3f}  gap={gap:+.3f}  {flag}  {k}", flush=True)

    # Best on OOS (lower bound on what's achievable)
    best_oos_key = max(oos_results, key=oos_results.get)
    print(f"\nBest on OOS: {best_oos_key}", flush=True)
    print(f"  OOS AUC: {oos_results[best_oos_key]}", flush=True)
    print(f"  IS AUC for same cell: {is_results[best_oos_key]}", flush=True)

    # Preregistered (no selection): cross_cos_mean on Q(h)|Q(h) at L27, our "ship spec" candidate
    prereg_keys = [
        "Q(h)|Q(h)|cross_cos_mean|L27",
        "h|h|cos_last|L27",  # vanilla baseline
        "K(h)|K(h)|cross_mean|L29",
    ]
    print(f"\nPreregistered cells (no selection):", flush=True)
    for k in prereg_keys:
        if k in oos_results:
            print(f"  IS={is_results[k]:.3f}  OOS={oos_results[k]:.3f}  {k}", flush=True)
        else:
            print(f"  MISSING: {k}", flush=True)

    # Plateau (CORRECT: average raw scores across layers, then ONE AUC).
    # Bug fixed: previous version averaged AUCs across layers, which is meaningless
    # because AUC is rank-based and doesn't compose under arithmetic mean.
    plateau_layers = [L for L in [27, 28, 29, 32, 34, 35] if L in layers]
    qq_scorer = CrossCosineMean()
    n_caches = len(caches)
    per_chunk_scores = [0.0] * n_caches
    for L in plateau_layers:
        qv = extractions[(cue_idx, "Q(h)", L)]
        for i in range(n_caches):
            cv = extractions[(i, "Q(h)", L)]
            per_chunk_scores[i] += qq_scorer(cv, qv, d_h=cv.shape[-1])
    per_chunk_scores = [s / len(plateau_layers) for s in per_chunk_scores]
    is_pos_sc = [per_chunk_scores[i] for i in is_pos_idx]
    is_neg_sc = [per_chunk_scores[len(pos)+i] for i in is_neg_idx]
    oos_pos_sc = [per_chunk_scores[i] for i in oos_pos_idx]
    oos_neg_sc = [per_chunk_scores[len(pos)+i] for i in oos_neg_idx]
    full_pos = [s for s, y in zip(per_chunk_scores, labels) if y == 1]
    full_neg = [s for s, y in zip(per_chunk_scores, labels) if y == 0]
    print(f"\nPlateau (CORRECT: mean scores -> one AUC, Q(h)|Q(h)|cross_cos_mean over L{plateau_layers}):", flush=True)
    print(f"  IS={auc(is_pos_sc, is_neg_sc):.3f}  OOS={auc(oos_pos_sc, oos_neg_sc):.3f}  FULL={auc(full_pos, full_neg):.3f}", flush=True)

    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
    out = {"model": MODEL, "dataset": "uncertainty", "cue": cue_text,
           "split_seed": 12061, "is_size": len(is_pos_idx)+len(is_neg_idx),
           "oos_size": len(oos_pos_idx)+len(oos_neg_idx),
           "is_results": is_results, "oos_results": oos_results,
           "best_is_key": best_is_key, "best_oos_key": best_oos_key}
    with open("output/exp_holdout_split.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved to output/exp_holdout_split.json", flush=True)


if __name__ == "__main__":
    main()
