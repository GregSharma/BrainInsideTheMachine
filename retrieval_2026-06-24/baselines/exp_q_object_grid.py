#!/usr/bin/env python3
"""Exhaustive object-grid sweep on the uncertainty dataset.

For every (chunk_object, cue_object, layer) triple, compute AUC.
Objects: h, delta, eps_k2, pred_k2, Q(h), K(h), V(h), Q(delta), K(delta), V(delta), Q(eps), K(eps), V(eps).
Comparison: cosine only. Positions: last token only.
No judgment, no pruning. Dump everything.
"""

import json, time, sys, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/greg/Desktop/Projects/BrainInsideTheMachine")
from uncertainty_dataset import GENUINE_UNCERTAINTY, PERFORMED_CONFIDENCE

MODEL = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
K_HIST = 2  # innovation history depth

def auc(pos_scores, neg_scores):
    c = sum(1 for p in pos_scores for n in neg_scores if p > n)
    t = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    total = len(pos_scores) * len(neg_scores)
    return (c + t) / total if total > 0 else 0.5

def encode_bare(texts, tok, model):
    """Encode texts bare, return hidden_states: list of (L+1, d) tensors (last-token)."""
    all_hs = []
    for text in texts:
        ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs = torch.stack([h[0, -1, :] for h in out.hidden_states])  # (L+1, d)
        all_hs.append(hs.cpu().float())
    return all_hs

def compute_deltas(hs):
    return hs[1:] - hs[:-1]  # (L, d)

def compute_innovations(deltas, k=2):
    L, d = deltas.shape
    eps = torch.zeros_like(deltas)
    pred = torch.zeros_like(deltas)
    for l in range(L):
        hist_start = max(0, l - k)
        hist = deltas[hist_start:l]
        if hist.shape[0] == 0:
            eps[l] = deltas[l]
            continue
        Q_h, _ = torch.linalg.qr(hist.T)
        proj = Q_h @ (Q_h.T @ deltas[l])
        pred[l] = proj
        eps[l] = deltas[l] - proj
    return eps, pred

def main():
    t0 = time.time()
    print("loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEVICE)
    model.eval()
    n_layers = model.config.num_hidden_layers  # 36

    # Encode
    all_texts = GENUINE_UNCERTAINTY + PERFORMED_CONFIDENCE
    labels = [1]*len(GENUINE_UNCERTAINTY) + [0]*len(PERFORMED_CONFIDENCE)
    print(f"encoding {len(all_texts)} texts...")
    all_hs = encode_bare(all_texts, tok, model)

    # Precompute
    print("computing deltas, innovations...")
    all_deltas = [compute_deltas(hs) for hs in all_hs]
    all_eps, all_pred = [], []
    for d_i in all_deltas:
        e, p = compute_innovations(d_i, k=K_HIST)
        all_eps.append(e)
        all_pred.append(p)

    # Cue
    cue_text = "genuinely uncertain exploring possibilities conditional reasoning diagnostic hedging"
    cue_hs = encode_bare([cue_text], tok, model)[0]
    cue_deltas = compute_deltas(cue_hs)
    cue_eps, cue_pred = compute_innovations(cue_deltas, k=K_HIST)

    # Base object sources: name -> (list_of_per_sample_tensors, cue_tensor)
    # h: (L+1, d), delta/eps/pred: (L, d)
    base_sources = {
        "h":       (all_hs, cue_hs),
        "delta":   (all_deltas, cue_deltas),
        "eps_k2":  (all_eps, cue_eps),
        "pred_k2": (all_pred, cue_pred),
    }

    results = {}
    total_cells = 0

    for l in range(n_layers):
        chunk_vecs = {}
        cue_vecs = {}

        # Extract base vectors at this layer
        for name, (data_all, data_cue) in base_sources.items():
            if name == "h":
                idx = l  # h[0]=embed, h[l]=output of block l
                chunk_vecs[name] = torch.stack([d[idx] for d in data_all])
                cue_vecs[name] = data_cue[idx].unsqueeze(0)
            else:
                if l >= data_cue.shape[0]:
                    continue
                chunk_vecs[name] = torch.stack([d[l] for d in data_all])
                cue_vecs[name] = data_cue[l].unsqueeze(0)

        # QKV projections of h, delta, eps_k2
        # Qwen2.5-3B uses GQA: Q=16 heads x 128, K/V=2 heads x 128
        # Repeat K/V heads to match Q dim so all cross-comparisons work
        layer_mod = model.model.layers[l]
        n_heads = model.config.num_attention_heads        # 16
        n_kv = model.config.num_key_value_heads           # 2
        rep = n_heads // n_kv                             # 8
        for base in ["h", "delta", "eps_k2"]:
            if base not in chunk_vecs:
                continue
            cv = chunk_vecs[base].to(DEVICE).half()
            qv = cue_vecs[base].to(DEVICE).half()
            with torch.no_grad():
                # Q is already full-dim (2048)
                chunk_vecs[f"Q({base})"] = layer_mod.self_attn.q_proj(cv).cpu().float()
                cue_vecs[f"Q({base})"] = layer_mod.self_attn.q_proj(qv).cpu().float()
                # K/V: project then repeat heads to match Q dim
                for proj_name, proj_fn in [("K", layer_mod.self_attn.k_proj),
                                            ("V", layer_mod.self_attn.v_proj)]:
                    c_out = proj_fn(cv).cpu().float()   # (N, n_kv*head_dim=256)
                    q_out = proj_fn(qv).cpu().float()   # (1, 256)
                    # Reshape to (N, n_kv, head_dim), repeat, flatten to (N, n_heads*head_dim=2048)
                    head_dim = c_out.shape[1] // n_kv
                    c_exp = c_out.view(-1, n_kv, head_dim).repeat(1, rep, 1).reshape(-1, n_heads * head_dim)
                    q_exp = q_out.view(-1, n_kv, head_dim).repeat(1, rep, 1).reshape(-1, n_heads * head_dim)
                    chunk_vecs[f"{proj_name}({base})"] = c_exp
                    cue_vecs[f"{proj_name}({base})"] = q_exp

        # Compute AUC for every (chunk_obj, cue_obj) pair — ALL must match dims
        for c_name, cv in chunk_vecs.items():
            for q_name, qv in cue_vecs.items():
                assert cv.shape[1] == qv.shape[1], f"DIM MISMATCH: {c_name}={cv.shape[1]} vs {q_name}={qv.shape[1]}"
                scores = F.cosine_similarity(cv.float(), qv.float().expand_as(cv), dim=1).tolist()
                pos = [s for s, y in zip(scores, labels) if y == 1]
                neg = [s for s, y in zip(scores, labels) if y == 0]
                a = auc(pos, neg)
                results[f"{c_name}|{q_name}|L{l}"] = round(a, 4)
                total_cells += 1

        if l % 6 == 0:
            print(f"  L{l} done ({total_cells} cells)")

    # ASSERT: no silent drops
    expected = 13 * 13 * n_layers  # all base+QKV objects, all cross pairs, all layers
    assert total_cells == expected, f"GRID INCOMPLETE: {total_cells}/{expected} cells. Silent drop detected."
    sorted_results = sorted(results.items(), key=lambda x: -x[1])

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_cells} cells (expected {expected} — {'OK' if total_cells == expected else 'INCOMPLETE'})")
    print(f"\nTOP 30:")
    for key, val in sorted_results[:30]:
        c_obj, q_obj, layer = key.split("|")
        bar = "#" * int(val * 20)
        print(f"  {val:.3f}  chunk={c_obj:14s}  cue={q_obj:14s}  {layer:4s}  {bar}")

    print(f"\nBOTTOM 10:")
    for key, val in sorted_results[-10:]:
        c_obj, q_obj, layer = key.split("|")
        print(f"  {val:.3f}  chunk={c_obj:14s}  cue={q_obj:14s}  {layer:4s}")

    # Marginals
    print(f"\nBEST AUC PER CHUNK OBJECT:")
    for c_name in sorted(set(k.split("|")[0] for k in results)):
        subset = {k: v for k, v in results.items() if k.startswith(c_name + "|")}
        if subset:
            best_k = max(subset, key=subset.get)
            parts = best_k.split("|")
            print(f"  {c_name:14s}: {subset[best_k]:.3f} @ cue={parts[1]:14s} {parts[2]}")

    print(f"\nBEST AUC PER CUE OBJECT:")
    for q_name in sorted(set(k.split("|")[1] for k in results)):
        subset = {k: v for k, v in results.items() if k.split("|")[1] == q_name}
        if subset:
            best_k = max(subset, key=subset.get)
            parts = best_k.split("|")
            print(f"  {q_name:14s}: {subset[best_k]:.3f} @ chunk={parts[0]:14s} {parts[2]}")

    print(f"\nBEST AUC PER LAYER:")
    for l in range(n_layers):
        subset = {k: v for k, v in results.items() if k.endswith(f"|L{l}")}
        if subset:
            best_k = max(subset, key=subset.get)
            bar = "#" * int(subset[best_k] * 20)
            print(f"  L{l:2d}: {subset[best_k]:.3f} {bar}")

    elapsed = time.time() - t0
    print(f"\ntotal {elapsed:.0f}s")

    out = {
        "model": MODEL, "dataset": "uncertainty", "n_pos": 20, "n_neg": 20,
        "k_hist": K_HIST, "cue": cue_text, "n_cells": total_cells,
        "all_results": results, "top30": sorted_results[:30],
    }
    with open("output/exp_q_object_grid.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved to output/exp_q_object_grid.json")

if __name__ == "__main__":
    main()
