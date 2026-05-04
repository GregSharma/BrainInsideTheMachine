#!/usr/bin/env python3
"""
Exp: Readout Anatomy — Combined Per-Head Entropy + Accumulated Readout Rank
==========================================================================
Three measurements from one forward pass:

1. PER-HEAD ENTROPY: Which attention heads at L27-L35 implement the
   readout operator? Previous finding: head-averaged entropy separates
   glue vs content at L32-L35 (p<0.0001). Is it 2-3 heads or all 16?

2. ACCUMULATED READOUT RANK: Over T generation steps, the read head
   reads from context via attention-weighted value sums. The matrix of
   accumulated readouts [v_1, ..., v_T] has an effective rank that
   equals the predictive state dimension. Low rank = KV cache
   compressible. High rank = each step reads new directions.

3. SOFTMAX VISION B (cross-lingual): For matched en/zh problem pairs,
   compare cos(attn^en, attn^zh) vs cos(h^en, h^zh) at each layer.
   If softmax nonlinearity makes operator ≠ state, attention cosine
   should differ from hidden-state cosine.

Model: Qwen2.5-3B (16 heads, 2 KV heads GQA, head_dim=128, 36 layers)
Test set: 20 math problems × 2 languages × 128 gen tokens
"""

import json, time, gc, sys, argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

from expC2c_crossmodel_readhead import (
    generate_problems, get_test_subset, TEMPLATES, SEED, NumpyEncoder,
)

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"

MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
N_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 128
D_MODEL = 2048
GQA_RATIO = N_HEADS // N_KV_HEADS  # 8

MAX_NEW = [128]  # mutable so we can override for dry runs

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)


def build_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{CHAT_SYSTEM}\n\nProblem: {problem_text}\n\nSolution:"


def run_one_problem(model, tokenizer, prompt_text, device):
    """Generate with attention + hidden state capture at every step.

    Returns:
        per_head_entropy: (T, L, H) — Shannon entropy per head per layer per step
        readout_vectors:  (T, L, D) — attention-weighted value sum (full d_model)
        hidden_states:    (T, L, D) — last-token hidden state at each layer
        attn_over_prompt: (T, L, H, P) — attention weights over static prompt positions
        tokens: list of decoded tokens
        surprisals: list of surprisal values
        prompt_len: int
    """
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]

    per_head_entropy = []
    readout_vectors = []
    hidden_states_list = []
    attn_prompt_list = []
    token_ids = []
    surprisals = []
    tokens_decoded = []

    past_kv = None
    cur_input = ids
    prompt_V = None  # Captured once from KV cache after first step

    with torch.inference_mode():
        for step in range(MAX_NEW[0]):
            out = model(
                cur_input,
                past_key_values=past_kv,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=True,
            )
            past_kv = out.past_key_values

            # ── Hidden states at each layer (last token) ──
            # out.hidden_states: tuple of (n_layers+1) tensors, each (1, seq, d)
            # We want layers 0..35 (skip the embedding = index 0, use 1..36)
            step_h = []
            for li in range(N_LAYERS):
                h = out.hidden_states[li + 1][0, -1].float().cpu().numpy()
                step_h.append(h)
            hidden_states_list.append(np.stack(step_h))  # (L, D)

            # ── Attention: per-head entropy over PROMPT positions ──
            # out.attentions: tuple of n_layers, each (1, n_heads, q_len, k_len)
            step_entropy = []
            step_attn_prompt = []
            step_readout = []

            for li in range(N_LAYERS):
                a_full = out.attentions[li][0, :, -1, :]  # (H, k_len)
                # Slice to prompt positions only (skip BOS = pos 0)
                a_prompt = a_full[:, 1:prompt_len].float()  # (H, P-1)
                # Renormalize
                a_prompt = a_prompt / (a_prompt.sum(dim=-1, keepdim=True) + 1e-12)

                # Per-head entropy
                eps = 1e-12
                ent = -(a_prompt * torch.log(a_prompt + eps)).sum(dim=-1)  # (H,)
                step_entropy.append(ent.cpu().numpy())

                # Attention over prompt for cross-lingual comparison
                step_attn_prompt.append(a_prompt.cpu().numpy())

                # ── Readout vector: attn-weighted V sum ──
                # V from KV cache: past_kv.layers[li].values shape (batch, n_kv_heads, seq, head_dim)
                V = past_kv.layers[li].values[0, :, 1:prompt_len, :]  # (n_kv, P-1, head_dim)

                # GQA: each KV head serves GQA_RATIO attention heads
                # a_prompt is (H, P-1), V is (n_kv, P-1, head_dim)
                # Expand V to match attention heads
                V_expanded = V.unsqueeze(1).expand(
                    -1, GQA_RATIO, -1, -1
                ).reshape(N_HEADS, prompt_len - 1, HEAD_DIM)  # (H, P-1, head_dim)

                # Weighted sum: (H, P-1) @ (H, P-1, head_dim) -> (H, head_dim)
                readout = torch.einsum("hp,hpd->hd", a_prompt.to(device), V_expanded.float())
                # Concatenate heads -> (D,)
                readout_flat = readout.reshape(-1).cpu().numpy()  # (H * head_dim,) = (D,)
                step_readout.append(readout_flat)

            per_head_entropy.append(np.stack(step_entropy))     # (L, H)
            attn_prompt_list.append(np.stack(step_attn_prompt))  # (L, H, P-1)
            readout_vectors.append(np.stack(step_readout))       # (L, D)

            # ── Next token ──
            logits = out.logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            next_id = int(logits.argmax().item())
            surp = -float(torch.log(probs[next_id] + 1e-20).item())

            token_ids.append(next_id)
            surprisals.append(surp)
            tokens_decoded.append(tokenizer.decode([next_id]))

            if next_id == tokenizer.eos_token_id:
                break
            cur_input = torch.tensor([[next_id]], device=device)

    return {
        "per_head_entropy": np.stack(per_head_entropy),     # (T, L, H)
        "readout_vectors": np.stack(readout_vectors),        # (T, L, D)
        "hidden_states": np.stack(hidden_states_list),       # (T, L, D)
        "attn_over_prompt": np.stack(attn_prompt_list),      # (T, L, H, P-1)
        "tokens": tokens_decoded,
        "surprisals": surprisals,
        "prompt_len": prompt_len,
        "T": len(token_ids),
    }


def effective_rank(matrix, threshold=0.90):
    """Number of singular values needed to explain `threshold` of total variance."""
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    var = S ** 2
    total = var.sum()
    if total < 1e-12:
        return 0
    cumvar = np.cumsum(var) / total
    return int(np.searchsorted(cumvar, threshold)) + 1


def analyze_results(results_en, results_zh):
    """Compute all three measurements from paired en/zh results."""
    analysis = {
        "per_head_entropy": {},
        "accumulated_rank": {},
        "softmax_vision_b": {},
    }

    n_problems = len(results_en)

    # ═══ 1. PER-HEAD ENTROPY: glue vs content separation per head ═══
    # Use surprisal-based labels (same as original anatomy)
    print("\n" + "=" * 70)
    print("MEASUREMENT 1: PER-HEAD ENTROPY (glue vs content)")
    print("=" * 70)

    for li in range(N_LAYERS):
        head_deltas = []
        for r in results_en + results_zh:
            T = r["T"]
            if T < 10:
                continue
            ent = r["per_head_entropy"][:T, li, :]  # (T, H)
            surp = np.array(r["surprisals"][:T])

            low_thresh = np.percentile(surp, 40)
            high_thresh = np.percentile(surp, 60)
            glue_mask = surp < low_thresh
            content_mask = surp > high_thresh

            if glue_mask.sum() < 3 or content_mask.sum() < 3:
                continue

            glue_ent = ent[glue_mask].mean(axis=0)    # (H,)
            content_ent = ent[content_mask].mean(axis=0)  # (H,)
            head_deltas.append(glue_ent - content_ent)  # positive = glue higher entropy

        if not head_deltas:
            continue
        deltas = np.stack(head_deltas)  # (n_problems, H)
        mean_delta = deltas.mean(axis=0)  # (H,)

        if li >= 27:  # Only print late layers
            sig = np.abs(mean_delta) / (deltas.std(axis=0) / np.sqrt(len(deltas)) + 1e-10)
            print(f"  L{li:2d}: ", end="")
            for h in range(N_HEADS):
                marker = "**" if sig[h] > 2.0 and mean_delta[h] > 0 else "  "
                print(f"H{h:02d}={mean_delta[h]:+.3f}{marker} ", end="")
            print()

        analysis["per_head_entropy"][f"L{li}"] = {
            "mean_delta_per_head": mean_delta.tolist(),
            "std_per_head": deltas.std(axis=0).tolist(),
            "n_problems": len(deltas),
        }

    # ═══ 2. ACCUMULATED READOUT RANK ═══
    print("\n" + "=" * 70)
    print("MEASUREMENT 2: ACCUMULATED READOUT RANK (predictive state dimension)")
    print("=" * 70)
    print(f"  Effective rank at 90% variance explained")
    print(f"  {'Layer':>6} | {'Mean':>6} | {'Std':>6} | {'Min':>4} | {'Max':>4} | Bar")
    print("-" * 60)

    for li in range(N_LAYERS):
        ranks = []
        for r in results_en + results_zh:
            T = r["T"]
            if T < 10:
                continue
            readout = r["readout_vectors"][:T, li, :]  # (T, D)
            # Center
            readout_c = readout - readout.mean(axis=0)
            rank = effective_rank(readout_c, threshold=0.90)
            ranks.append(rank)

        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        bar = "#" * int(mean_rank / 2)
        print(f"  L{li:>3} | {mean_rank:>6.1f} | {std_rank:>6.1f} | {min(ranks):>4d} | {max(ranks):>4d} | {bar}")

        analysis["accumulated_rank"][f"L{li}"] = {
            "mean": float(mean_rank),
            "std": float(std_rank),
            "min": int(min(ranks)),
            "max": int(max(ranks)),
            "n_problems": len(ranks),
        }

    # ═══ 3. SOFTMAX VISION B: cos(attn^en, attn^zh) vs cos(h^en, h^zh) ═══
    print("\n" + "=" * 70)
    print("MEASUREMENT 3: SOFTMAX VISION B (operator vs state similarity)")
    print("=" * 70)
    print(f"  {'Layer':>6} | {'cos(h)':>8} | {'cos(attn)':>10} | {'Delta':>7} | Interpretation")
    print("-" * 70)

    for li in range(N_LAYERS):
        h_cos_list = []
        a_cos_list = []

        for i in range(min(len(results_en), len(results_zh))):
            re, rz = results_en[i], results_zh[i]
            T = min(re["T"], rz["T"])
            if T < 5:
                continue

            # Hidden state cosine (mean over generation steps)
            for t in range(min(T, 20)):  # first 20 steps for efficiency
                he = re["hidden_states"][t, li]
                hz = rz["hidden_states"][t, li]
                cos_h = np.dot(he, hz) / (np.linalg.norm(he) * np.linalg.norm(hz) + 1e-10)
                h_cos_list.append(cos_h)

            # Attention cosine (head-concatenated attention over prompt)
            # Different prompt lengths — use min
            P = min(re["attn_over_prompt"].shape[3], rz["attn_over_prompt"].shape[3])
            for t in range(min(T, 20)):
                ae = re["attn_over_prompt"][t, li, :, :P].flatten()
                az = rz["attn_over_prompt"][t, li, :, :P].flatten()
                cos_a = np.dot(ae, az) / (np.linalg.norm(ae) * np.linalg.norm(az) + 1e-10)
                a_cos_list.append(cos_a)

        if not h_cos_list:
            continue

        mean_h = np.mean(h_cos_list)
        mean_a = np.mean(a_cos_list)
        delta = mean_a - mean_h
        interp = "attn > state" if delta > 0.01 else "attn < state" if delta < -0.01 else "≈ equal"
        print(f"  L{li:>3} | {mean_h:>8.4f} | {mean_a:>10.4f} | {delta:>+7.4f} | {interp}")

        analysis["softmax_vision_b"][f"L{li}"] = {
            "cos_hidden_mean": float(mean_h),
            "cos_attn_mean": float(mean_a),
            "delta": float(delta),
            "n_pairs": len(h_cos_list),
        }

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="3 problems, 64 tokens")
    args = parser.parse_args()

    device = "cuda"
    n_problems = 3 if args.dry else 20
    max_new_override = 64 if args.dry else 128

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Loaded. {N_LAYERS} layers, {N_HEADS} heads, d={D_MODEL}")

    # Generate test problems
    all_problems = generate_problems()
    test_problems = get_test_subset(all_problems)[:n_problems]
    print(f"  {len(test_problems)} test problems")

    MAX_NEW[0] = max_new_override

    results_en = []
    results_zh = []
    t0 = time.time()

    for pi, prob in enumerate(test_problems):
        for lang in ["en", "zh"]:
            prompt_text = build_prompt(tokenizer, prob[lang])
            print(f"  P{pi:2d}/{lang}: {prob[lang][:50]}...", end=" ", flush=True)

            t1 = time.time()
            result = run_one_problem(model, tokenizer, prompt_text, device)
            elapsed = time.time() - t1
            print(f"T={result['T']}, {elapsed:.1f}s")

            # Store lightweight version (drop large arrays for JSON, keep for analysis)
            result["problem_idx"] = pi
            result["lang"] = lang
            result["category"] = prob.get("category", "?")

            if lang == "en":
                results_en.append(result)
            else:
                results_zh.append(result)

    total_time = time.time() - t0
    print(f"\nTotal extraction: {total_time:.0f}s")

    # ── Analysis ──
    analysis = analyze_results(results_en, results_zh)

    # ── Save (drop large numpy arrays, keep metrics) ──
    output = {
        "experiment": "Readout Anatomy (per-head entropy + accumulated rank + softmax Vision B)",
        "model": MODEL_NAME,
        "n_problems": len(test_problems),
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "max_new": MAX_NEW[0],
        "elapsed_s": total_time,
        "analysis": analysis,
    }

    outpath = OUTPUT_DIR / "exp_readout_anatomy_3b.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
