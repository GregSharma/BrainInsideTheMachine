"""
Exp Y: Von Neumann Fixed-Point Iteration
=========================================
Hypothesis: Transformer reasoning can run in its own substrate.
If the autoregressive loop is just "rendering":
  - Encode problem → extract h at layer 9 (last token, full prefill)
  - Run h through layers 9-25 REPEATEDLY: same weights, no new tokens, no KV cache
  - After N iterations, project through lm_head → does the answer surface?
  - Track: cos(h_N, h_{N-1}) — convergence; norm; answer rank

If answer surfaces: computation is substrate-independent.
If never surfaces: autoregressive loop is load-bearing.

Key technical note: Qwen2 layer forward requires position_embeddings=(cos,sin).
Layer output shape is (1, d_model) when seq_len=1.
"""

import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
LAYERS_START = 9
LAYERS_END = 26  # iterate layers 9..25 inclusive
ITER_COUNTS = [1, 2, 3, 5, 10, 20, 50]

PROBLEMS = [
    {"prompt": "Calculate 664 + 124.", "answer": "788"},
    {"prompt": "Calculate 769 + 291.", "answer": "1060"},
    {"prompt": "Find the value of C(9, 5).", "answer": "126"},
    {"prompt": "Find the value of C(9, 4).", "answer": "126"},
    {"prompt": "What is the remainder when 1014 is divided by 17?", "answer": "11"},
    {"prompt": "What is the remainder when 1154 is divided by 5?", "answer": "4"},
    {"prompt": "A rectangle has length 34 and width 35. Find its area.", "answer": "1190"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60"},
    {"prompt": "An arithmetic sequence has first term 9 and common difference 5. Find sum of first 24 terms.", "answer": "1596"},
    {"prompt": "An arithmetic sequence has first term 7 and common difference 6. Find sum of first 11 terms.", "answer": "407"},
]


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    return tok, model


def prefill_and_extract(model, tok, prompt):
    """Prefill the full prompt. Return hidden state at LAYERS_START (last token),
    and the position id of that token (for RoPE)."""
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    # hidden_states[i+1] = output of layer i
    # We want h ENTERING layer LAYERS_START = output of layer LAYERS_START-1
    h = out.hidden_states[LAYERS_START][:, -1:, :].clone()  # (1, 1, d)
    return h, seq_len - 1  # h and position id


def run_one_pass(model, h, pos_id):
    """Run h (1, 1, d) through layers LAYERS_START..LAYERS_END-1.
    Returns new h with same shape."""
    layers = model.model.layers
    # Recompute RoPE for this position each pass (position is fixed — the "decode position")
    pos_ids = torch.tensor([[pos_id]], device=DEVICE)
    cos, sin = model.model.rotary_emb(h, pos_ids)
    pos_embeds = (cos, sin)

    h_cur = h  # (1, 1, d)
    for layer_idx in range(LAYERS_START, LAYERS_END):
        out = layers[layer_idx](h_cur, position_embeddings=pos_embeds)
        h_new = out[0]
        # Layer returns (1, d) when seq=1 — unsqueeze back
        if h_new.dim() == 2:
            h_new = h_new.unsqueeze(1)
        h_cur = h_new
    return h_cur  # (1, 1, d)


def project_to_vocab(model, h):
    """h: (1, 1, d) or (1, d). Returns logits (vocab_size,)."""
    if h.dim() == 3:
        h = h.squeeze(1)
    h_normed = model.model.norm(h)
    logits = model.lm_head(h_normed)
    return logits.squeeze(0)  # (vocab_size,)


def top_tokens(logits, tok, k=10):
    vals, idxs = torch.topk(logits.float(), k)
    tokens = [tok.decode([idx.item()]) for idx in idxs]
    return list(zip(tokens, vals.tolist()))


def answer_rank(logits, tok, answer):
    """Rank of answer token (0-indexed). Returns -1 if not found."""
    sorted_idxs = torch.argsort(logits.float(), descending=True)
    for rank, idx in enumerate(sorted_idxs[:500]):
        t = tok.decode([idx.item()]).strip()
        if t == answer:
            return rank
    return -1


def cosine_sim(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def run_experiment():
    tok, model = load_model()

    results = {
        "experiment": "Y: Von Neumann Fixed-Point Iteration",
        "model": MODEL_NAME,
        "layers_iterated": f"{LAYERS_START}-{LAYERS_END-1}",
        "iter_counts": ITER_COUNTS,
        "problems": [],
        "summary": {},
    }

    t0 = time.time()

    for prob in PROBLEMS:
        prompt = prob["prompt"]
        answer = prob["answer"]
        print(f"\n--- {prompt[:55]} (ans={answer}) ---")

        h0, pos_id = prefill_and_extract(model, tok, prompt)
        print(f"    pos_id={pos_id}, h0 norm={h0.float().norm().item():.1f}")

        prob_result = {
            "prompt": prompt,
            "answer": answer,
            "h0_norm": h0.float().norm().item(),
            "pos_id": pos_id,
            "iterations": [],
        }

        h_prev = h0.clone()
        h_cur = h0.clone()
        cumulative = 0

        for n_iters in ITER_COUNTS:
            delta = n_iters - cumulative
            with torch.no_grad():
                for _ in range(delta):
                    h_cur = run_one_pass(model, h_cur, pos_id)
            cumulative = n_iters

            with torch.no_grad():
                logits = project_to_vocab(model, h_cur)

            top = top_tokens(logits, tok, k=10)
            rank = answer_rank(logits, tok, answer)
            cos_prev = cosine_sim(h_prev, h_cur)
            cur_norm = h_cur.float().norm().item()

            print(
                f"  iter={n_iters:3d} | cos={cos_prev:.5f} | "
                f"norm={cur_norm:6.1f} | rank={rank:5d} | "
                f"top3={[t[0].strip() for t in top[:3]]}"
            )

            prob_result["iterations"].append({
                "n_iters": n_iters,
                "cos_to_prev": cos_prev,
                "norm": cur_norm,
                "answer_rank": rank,
                "answer_in_top50": rank >= 0 and rank < 50,
                "top10_tokens": [(t.strip(), round(v, 3)) for t, v in top],
            })

            h_prev = h_cur.clone()

        results["problems"].append(prob_result)

    # Summary
    all_best_ranks = []
    convergence_at = []
    for p in results["problems"]:
        ranks = [it["answer_rank"] for it in p["iterations"]]
        valid = [r for r in ranks if r >= 0]
        all_best_ranks.append(min(valid) if valid else -1)
        conv = next((it["n_iters"] for it in p["iterations"] if it["cos_to_prev"] > 0.9999), None)
        convergence_at.append(conv)

    valid_ranks = [r for r in all_best_ranks if r >= 0]
    results["summary"] = {
        "n_problems": len(PROBLEMS),
        "answer_surfaces_top50_any_iter": sum(
            any(it["answer_in_top50"] for it in p["iterations"])
            for p in results["problems"]
        ),
        "answer_surfaces_top10_any_iter": sum(
            any(it["answer_rank"] >= 0 and it["answer_rank"] < 10 for it in p["iterations"])
            for p in results["problems"]
        ),
        "mean_best_rank": float(np.mean(valid_ranks)) if valid_ranks else -1,
        "median_best_rank": float(np.median(valid_ranks)) if valid_ranks else -1,
        "best_ranks_per_problem": all_best_ranks,
        "convergence_iter_per_problem": convergence_at,
        "n_converged": sum(c is not None for c in convergence_at),
    }

    results["runtime_seconds"] = time.time() - t0

    out_path = "output/expY_neumann.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== SUMMARY ===")
    s = results["summary"]
    print(f"Answer in top-50 (any iter): {s['answer_surfaces_top50_any_iter']}/{len(PROBLEMS)}")
    print(f"Answer in top-10 (any iter): {s['answer_surfaces_top10_any_iter']}/{len(PROBLEMS)}")
    print(f"Mean best rank: {s['mean_best_rank']:.1f}")
    print(f"Converged (cos>0.9999): {s['n_converged']}/{len(PROBLEMS)}")
    print(f"Runtime: {results['runtime_seconds']:.1f}s")
    print(f"Saved → {out_path}")

    return results


if __name__ == "__main__":
    run_experiment()
