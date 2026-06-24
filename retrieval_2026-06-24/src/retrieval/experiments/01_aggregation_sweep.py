#!/usr/bin/env python3
"""Aggregation sweep: same Q.K bilinear, every aggregation, every layer.

Tests the hypothesis that last-token-only was leaving signal on the table.
"""
import sys, json, time, torch
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/greg/Desktop/Projects/BrainInsideTheMachine")

from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieval.base import EncodingCache
from retrieval.features import HiddenState, QProj, KProj, VProj, SequenceInnovation, Delta
from retrieval.scorers import (CosineLast, BilinearLast, CrossMean, CrossMax,
                                CrossSoftMax, CrossCosineMean, CrossCosineMax, CrossTopK)
from retrieval.pipeline import encode_bare, run_grid
from retrieval.datasets import load_uncertainty, CUES


def main():
    t0 = time.time()
    print("loading model...")
    MODEL = "Qwen/Qwen2.5-3B"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    pos, neg = load_uncertainty()
    all_texts = pos + neg
    labels = [1]*len(pos) + [0]*len(neg)
    cue_text = CUES["uncertainty"]

    print(f"encoding {len(all_texts)} chunks + 1 cue...")
    caches = [encode_bare(t, tok, model) for t in all_texts]
    cue_cache = encode_bare(cue_text, tok, model)

    h = HiddenState()
    seq_inn = SequenceInnovation(k=4)

    chunk_features = [
        h,
        QProj(h),
        KProj(h),
        VProj(h),
        seq_inn,
        QProj(seq_inn),
        KProj(seq_inn),
    ]
    cue_features = chunk_features  # symmetric

    scorers = [
        CosineLast(),
        BilinearLast(),
        CrossMean(),
        CrossMax(),
        CrossSoftMax(),
        CrossCosineMean(),
        CrossCosineMax(),
        CrossTopK(k=5),
        CrossTopK(k=20),
    ]

    # Focus on the layers that the prior grid said mattered: late band + a few earlier
    layers = [2, 18, 20, 23, 27, 28, 29, 32, 34, 35]

    print(f"grid: {len(chunk_features)}^2 features x {len(scorers)} scorers x {len(layers)} layers")
    print(f"     = {len(chunk_features)**2 * len(scorers) * len(layers)} cells")

    results = run_grid(caches, labels, cue_cache, chunk_features, cue_features, scorers, layers, model)

    sorted_r = sorted(results.items(), key=lambda x: -x[1])
    print(f"\nTOP 25:")
    for k, v in sorted_r[:25]:
        cf, qf, sc, L = k.split("|")
        bar = "#" * int(v * 20)
        print(f"  {v:.3f}  chunk={cf:18s} cue={qf:18s} {sc:15s} {L:4s}  {bar}")

    print(f"\nBOTTOM 10:")
    for k, v in sorted_r[-10:]:
        cf, qf, sc, L = k.split("|")
        print(f"  {v:.3f}  chunk={cf:18s} cue={qf:18s} {sc:15s} {L:4s}")

    # Aggregation marginal: best AUC per scorer (across all feature pairs and layers)
    print(f"\nBEST PER SCORER:")
    scorer_names = sorted(set(k.split("|")[2] for k in results))
    for sn in scorer_names:
        sub = {k: v for k, v in results.items() if k.split("|")[2] == sn}
        if sub:
            best_k = max(sub, key=sub.get)
            parts = best_k.split("|")
            print(f"  {sn:18s}: {sub[best_k]:.3f}  chunk={parts[0]:18s} cue={parts[1]:18s} {parts[3]}")

    print(f"\nBEST PER LAYER:")
    for L in layers:
        sub = {k: v for k, v in results.items() if k.endswith(f"|L{L}")}
        if sub:
            best_k = max(sub, key=sub.get)
            parts = best_k.split("|")
            bar = "#" * int(sub[best_k] * 20)
            print(f"  L{L:2d}: {sub[best_k]:.3f}  {parts[0]}|{parts[1]}|{parts[2]}  {bar}")

    print(f"\ntotal {time.time()-t0:.0f}s, {len(results)} cells")

    out = {"model": MODEL, "dataset": "uncertainty", "cue": cue_text,
           "n_cells": len(results), "results": results, "top25": sorted_r[:25]}
    with open("output/exp_aggregation_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved to output/exp_aggregation_sweep.json")


if __name__ == "__main__":
    main()
