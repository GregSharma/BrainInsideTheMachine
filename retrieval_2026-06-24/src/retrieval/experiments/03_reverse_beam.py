#!/usr/bin/env python3
"""Reverse beam search retrieval on the uncertainty dataset.

For each chunk, find the top-K most likely single-token prefixes.
Take the weighted mean of their hidden states at layer L.
Use that vector as the chunk's representation.

For the cue, do the same.
Score by cosine.

Hypothesis (Greg's, June 14): the latent K that the chunk's queries are
addressing came BEFORE the chunk. The reverse-prefix vector should
capture the "situation" the chunk lives in, and matching across that
gives a structural relevance signal that forward-direction matching misses.

Computational budget:
  20 chunks + 20 negs + 1 cue = 41 texts
  Per text: |V| = 152k forward passes at length avg=130 tokens
  Batched at 32: 152k/32 = 4750 batches
  Each batch ~150ms on RTX 4070 -> ~12 min per text
  Total: 41 * 12 = 8 hours. NOT FEASIBLE on 4070.

  Solution: subsample vocabulary. Use top-N most-frequent tokens
  (or candidate_vocab argument). With N=2048: 32x speedup -> ~15 min total.
"""
import sys, json, time, torch
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/greg/Desktop/Projects/BrainInsideTheMachine")

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieval.reverse_beam import reverse_prefix_feature
from retrieval.pipeline import encode_bare, auc
from retrieval.datasets import load_uncertainty, CUES
import torch.nn.functional as F


def get_common_token_ids(tok, n: int = 2048):
    """Heuristic candidate vocab: the first n tokens of the tokenizer
    (which tend to be the most common in Qwen's BPE)."""
    # Skip special tokens at the very start
    return torch.arange(start=10, end=10 + n)


def main():
    t0 = time.time()
    print("loading model...", flush=True)
    MODEL = "Qwen/Qwen2.5-3B"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    pos, neg = load_uncertainty()
    cue_text = CUES["uncertainty"]

    all_texts = pos + neg
    labels = [1]*len(pos) + [0]*len(neg)

    LAYER = 27
    TOP_K = 32
    CAND_N = 2048
    BATCH = 32

    candidate_vocab = get_common_token_ids(tok, n=CAND_N)
    print(f"candidate vocab size: {CAND_N}", flush=True)
    print(f"top_k prefixes per chunk: {TOP_K}", flush=True)
    print(f"layer for representation: {LAYER}", flush=True)

    print(f"\ncomputing reverse-prefix feature for {len(all_texts)} chunks + cue...", flush=True)
    chunk_features = []
    for i, text in enumerate(tqdm(all_texts, desc="chunks")):
        feat = reverse_prefix_feature(
            text, tok, model, layer=LAYER,
            top_k=TOP_K, extend_steps=0,
            candidate_vocab=candidate_vocab,
            batch_size=BATCH,
        )
        chunk_features.append(feat)

    print("computing cue reverse-prefix feature...", flush=True)
    cue_feat = reverse_prefix_feature(
        cue_text, tok, model, layer=LAYER,
        top_k=TOP_K, extend_steps=0,
        candidate_vocab=candidate_vocab,
        batch_size=BATCH,
    )

    # Score by cosine
    print("scoring...", flush=True)
    scores = []
    for cf in chunk_features:
        s = F.cosine_similarity(cf.unsqueeze(0), cue_feat.unsqueeze(0), dim=-1).item()
        scores.append(s)
    pos_sc = [s for s, y in zip(scores, labels) if y == 1]
    neg_sc = [s for s, y in zip(scores, labels) if y == 0]
    auc_val = auc(pos_sc, neg_sc)

    print(f"\nReverse-prefix cosine AUC (full set): {auc_val:.3f}", flush=True)

    # Compare to vanilla last-token h cosine at the same layer
    print("\nbaseline: forward h cosine at L27...", flush=True)
    forward_scores = []
    for text in tqdm(all_texts, desc="forward"):
        c = encode_bare(text, tok, model)
        h = c.hidden_states[LAYER][-1].float()
        forward_scores.append(h)
    cue_c = encode_bare(cue_text, tok, model)
    cue_h = cue_c.hidden_states[LAYER][-1].float()
    fwd_sc = [F.cosine_similarity(h.unsqueeze(0), cue_h.unsqueeze(0), dim=-1).item() for h in forward_scores]
    fwd_pos = [s for s, y in zip(fwd_sc, labels) if y == 1]
    fwd_neg = [s for s, y in zip(fwd_sc, labels) if y == 0]
    print(f"Forward h cosine AUC: {auc(fwd_pos, fwd_neg):.3f}", flush=True)

    # Save
    out = {
        "model": MODEL, "dataset": "uncertainty", "cue": cue_text,
        "layer": LAYER, "top_k_prefixes": TOP_K, "candidate_vocab_n": CAND_N,
        "auc_reverse_prefix": auc_val,
        "auc_forward_baseline": auc(fwd_pos, fwd_neg),
        "scores": scores,
        "forward_scores": fwd_sc,
        "labels": labels,
    }
    with open("output/exp_reverse_beam.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s, saved to output/exp_reverse_beam.json", flush=True)


if __name__ == "__main__":
    main()
