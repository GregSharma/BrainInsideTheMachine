"""Glue: encoding, AUC, grid running. Computationally tight."""
import math
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base import EncodingCache, Feature, Scorer


def encode_bare(text: str, tok, model, device: str = "cuda") -> EncodingCache:
    """One forward pass, full hidden states. Result stays on CPU as fp32."""
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = torch.stack([h[0].cpu().float() for h in out.hidden_states])  # (L+1, T, d)
    L = model.config.num_hidden_layers
    return EncodingCache(hidden_states=hs, n_layers=L, n_tokens=hs.shape[1], d=hs.shape[2])


def auc(pos_scores, neg_scores) -> float:
    """O(N log N) AUC via sort — not the O(N^2) double loop."""
    import numpy as np
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    # Rank-based AUC with tie handling
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # For each negative, count positives ranked higher (with 0.5 for ties)
    ranks = np.empty_like(scores)
    # average rank within ties
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        ranks[i:j] = (i + j - 1) / 2.0 + 1  # 1-indexed average rank
        i = j
    # Sum of ranks of positives
    pos_rank_sum = ranks[y == 1].sum()
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _extract_to_device(feature: Feature, cache: EncodingCache, layer: int, model, device: str) -> torch.Tensor:
    t = feature.extract(cache, layer, model)
    return t.to(device, dtype=torch.float32, non_blocking=True)


def plateau_auc(
    caches: list[EncodingCache],
    labels: list[int],
    cue_cache: EncodingCache,
    chunk_feature: Feature,
    cue_feature: Feature,
    scorer: Scorer,
    layers: list[int],
    model,
    device: str = "cuda",
) -> float:
    """Correct plateau: average raw SCORES across layers per (chunk,cue), then
    one AUC. NOT an average of per-layer AUCs (that operation is meaningless
    because AUC is rank-based).

    For each chunk:
      mean_score_i = mean_{L in layers}  scorer(feat(chunk_i, L), feat(cue, L))
    AUC over mean_score_i vs labels.
    """
    n = len(caches)
    per_chunk_scores = [0.0] * n
    for L in layers:
        qv = cue_feature.extract(cue_cache, L, model).to(device, dtype=torch.float32)
        for i, c in enumerate(caches):
            cv = chunk_feature.extract(c, L, model).to(device, dtype=torch.float32)
            per_chunk_scores[i] += scorer(cv, qv, d_h=cv.shape[-1])
    per_chunk_scores = [s / len(layers) for s in per_chunk_scores]
    pos = [s for s, y in zip(per_chunk_scores, labels) if y == 1]
    neg = [s for s, y in zip(per_chunk_scores, labels) if y == 0]
    return auc(pos, neg)


def plateau_auc_from_cache(
    extractions: dict,                       # (cache_idx, feat_name, layer) -> tensor on device
    labels: list[int],
    n_caches: int,
    cue_idx: int,
    chunk_feature_name: str,
    cue_feature_name: str,
    scorer: Scorer,
    layers: list[int],
) -> float:
    """Same as plateau_auc but using a precomputed extraction cache.

    Use this when you've already extracted features for a full grid sweep
    (e.g. from run_grid's first phase) — no redundant forward passes.
    """
    per_chunk_scores = [0.0] * n_caches
    for L in layers:
        qv = extractions[(cue_idx, cue_feature_name, L)]
        for i in range(n_caches):
            cv = extractions[(i, chunk_feature_name, L)]
            per_chunk_scores[i] += scorer(cv, qv, d_h=cv.shape[-1])
    per_chunk_scores = [s / len(layers) for s in per_chunk_scores]
    pos = [s for s, y in zip(per_chunk_scores, labels) if y == 1]
    neg = [s for s, y in zip(per_chunk_scores, labels) if y == 0]
    return auc(pos, neg)


def run_grid(
    caches: list[EncodingCache],
    labels: list[int],
    cue_cache: EncodingCache,
    chunk_features: list[Feature],
    cue_features: list[Feature],
    scorers: list[Scorer],
    layers: list[int],
    model,
    device: str = "cuda",
) -> dict[str, float]:
    """Run every (chunk_feat, cue_feat, scorer, layer) combination.

    Tight version:
      1. Pre-extract every (feature, layer) for every cache ONCE. Cache on GPU.
      2. For each (cf, qf, layer): dim-check once, then loop scorers.
      3. tqdm progress bar at every level.
      4. Hard asserts — no silent drops.
    """
    # Union of features to extract (avoid double-extraction when cf == qf)
    all_features = {}
    for f in chunk_features + cue_features:
        all_features[f.name] = f
    feature_list = list(all_features.values())

    expected = len(layers) * len(chunk_features) * len(cue_features) * len(scorers)
    print(f"  features to extract: {len(feature_list)} unique", flush=True)
    print(f"  layers: {len(layers)} -> {layers}", flush=True)
    print(f"  caches: {len(caches)} chunks + 1 cue", flush=True)
    print(f"  scorers: {len(scorers)}", flush=True)
    print(f"  expected cells: {expected}", flush=True)

    # ---- Phase 1: extract everything ----
    # extractions[(cache_idx, feat_name, layer)] -> tensor on GPU
    # cache_idx: 0..len(caches)-1 for chunks, len(caches) for cue
    extractions = {}
    all_caches = caches + [cue_cache]
    cue_idx = len(caches)

    n_extract = len(layers) * len(feature_list) * len(all_caches)
    pbar = tqdm(total=n_extract, desc="extract", file=sys.stdout, mininterval=0.5)
    for layer in layers:
        for f in feature_list:
            for ci, c in enumerate(all_caches):
                t = _extract_to_device(f, c, layer, model, device)
                assert t.ndim == 2, f"{f.name} L{layer} cache{ci} ndim={t.ndim}"
                extractions[(ci, f.name, layer)] = t
                pbar.update(1)
    pbar.close()

    # ---- Phase 2: score ----
    results = {}
    n_cells = len(layers) * len(chunk_features) * len(cue_features) * len(scorers)
    pbar = tqdm(total=n_cells, desc="score", file=sys.stdout, mininterval=0.5)
    for layer in layers:
        for cf in chunk_features:
            cv_test = extractions[(0, cf.name, layer)]
            for qf in cue_features:
                qv = extractions[(cue_idx, qf.name, layer)]
                assert cv_test.shape[-1] == qv.shape[-1], (
                    f"DIM MISMATCH L{layer} chunk={cf.name}({cv_test.shape[-1]}) "
                    f"vs cue={qf.name}({qv.shape[-1]})"
                )
                for scorer in scorers:
                    scores_pos, scores_neg = [], []
                    for ci in range(len(caches)):
                        cv = extractions[(ci, cf.name, layer)]
                        s = scorer(cv, qv, d_h=cv.shape[-1])
                        if labels[ci] == 1:
                            scores_pos.append(s)
                        else:
                            scores_neg.append(s)
                    a = auc(scores_pos, scores_neg)
                    key = f"{cf.name}|{qf.name}|{scorer.name}|L{layer}"
                    results[key] = round(a, 4)
                    pbar.update(1)
    pbar.close()
    assert len(results) == expected, f"GRID INCOMPLETE: {len(results)}/{expected}"
    return results
