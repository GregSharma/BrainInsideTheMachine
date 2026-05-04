#!/usr/bin/env python3
"""
Exp C3: Constructive Compression Test — 6D Readout Sufficiency

Tests whether the ~6D readout image at L33 (N=20 readout anatomy finding)
is causally sufficient for math generation.

Design:
  Phase 1 (basis): Problems 0-9 (EN+ZH). Capture self_attn post-o_proj output
    at L33 during generation. SVD → top-k basis + mean.
  Phase 2 (test):  Problems 10-19 (EN+ZH). Generate with compression hook that
    projects L33 attention output onto k-D affine subspace.
    Sweep k = {1, 2, 3, 4, 5, 6, 8, 12, 20} + baseline (no hook).

Intervention: ONLY modifies the last token's attention output during generation.
Prompt processing and context-building are untouched.

Prediction: k=6 should preserve most accuracy (matching the measured
effective rank at 90% variance). k<=3 should degrade significantly.
"""

import json, time, re, argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from expC2c_crossmodel_readhead import (
    generate_problems, get_test_subset, NumpyEncoder,
)

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"

MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_NEW = 128

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 12, 20, 50]
TARGET_LAYER = 33


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


def check_answer(text, correct):
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)


# ═════════════════════════════════════════════════════════════════
# HOOKS
# ═════════════════════════════════════════════════════════════════

class CaptureHook:
    """Captures self_attn output (post-o_proj) for last token at each gen step."""

    def __init__(self):
        self.captured = []
        self.active = False

    def __call__(self, module, input, output):
        if not self.active:
            return
        attn_out = output[0]  # (batch, seq, d_model)
        if attn_out.shape[1] == 1:  # generation step, not prompt
            self.captured.append(attn_out[0, 0].float().cpu().numpy())


class CompressionHook:
    """Projects self_attn output onto k-D affine subspace during generation.

    Intervention: attn_out → mean + project(attn_out - mean, top-k basis)
    Only fires when seq_len=1 (generation). Prompt pass-through is untouched.
    """

    def __init__(self, mean_vec, basis_vecs, k, device):
        self.mean = torch.tensor(mean_vec, dtype=torch.float32, device=device)
        self.basis = torch.tensor(
            basis_vecs[:k], dtype=torch.float32, device=device
        )  # (k, d_model)
        self.active = False

    def __call__(self, module, input, output):
        if not self.active:
            return output
        attn_out = output[0]  # (batch, seq, d_model)
        if attn_out.shape[1] != 1:
            return output  # prompt — pass through
        x = attn_out[0, 0].float()  # (d_model,)
        centered = x - self.mean
        coeffs = centered @ self.basis.T  # (k,)
        projected = self.mean + coeffs @ self.basis  # (d_model,)
        new_out = projected.to(attn_out.dtype).unsqueeze(0).unsqueeze(0)
        return (new_out,) + output[1:]


# ═════════════════════════════════════════════════════════════════
# GENERATION
# ═════════════════════════════════════════════════════════════════

def generate(model, tokenizer, prompt_text, device, max_new=MAX_NEW):
    """Token-by-token generation with KV cache. Hooks on model layers fire."""
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    generated_ids = []
    past_kv = None
    cur_input = ids

    with torch.inference_mode():
        for _ in range(max_new):
            out = model(cur_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax().item())
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            cur_input = torch.tensor([[next_id]], device=device)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="3+3 problems, 64 tokens")
    parser.add_argument(
        "--layer", type=int, default=TARGET_LAYER,
        help=f"Target layer (default {TARGET_LAYER})",
    )
    args = parser.parse_args()

    device = "cuda"
    n_basis = 2 if args.dry else 10
    n_test = 1 if args.dry else 10
    max_new = 64 if args.dry else MAX_NEW
    target_layer = args.layer

    print(f"{'='*60}")
    print(f"Exp C3: Constructive Compression Test")
    print(f"{'='*60}")
    print(f"Model:     {MODEL_NAME}")
    print(f"Layer:     L{target_layer}")
    print(f"k-values:  {K_VALUES}")
    print(f"Basis/Test: {n_basis}/{n_test} problems")
    print(f"Max tokens: {max_new}")
    print()

    # ── Load model ──
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # ── Problems (stratified split: half of each category → basis, half → test) ──
    all_problems = generate_problems()
    all_test = get_test_subset(all_problems)  # 4 per category × 5 = 20
    # Group by category, then split each category
    from collections import OrderedDict
    by_cat = OrderedDict()
    for p in all_test:
        cat = p.get("category", "?")
        by_cat.setdefault(cat, []).append(p)
    basis_problems, test_problems = [], []
    for cat, probs in by_cat.items():
        half = max(1, len(probs) // 2)
        basis_problems.extend(probs[:half])
        test_problems.extend(probs[half:])
    if args.dry:
        basis_problems = basis_problems[:3]
        test_problems = test_problems[:3]
    print(f"  {len(basis_problems)} basis + {len(test_problems)} test problems")
    cats_basis = set(p.get("category", "?") for p in basis_problems)
    cats_test = set(p.get("category", "?") for p in test_problems)
    print(f"  Basis categories: {sorted(cats_basis)}")
    print(f"  Test categories:  {sorted(cats_test)}")
    n_basis = len(basis_problems)
    n_test = len(test_problems)
    print()

    attn_module = model.model.layers[target_layer].self_attn

    # ═════════════════════════════════════════════════════════════
    # PHASE 1: BASIS EXTRACTION
    # ═════════════════════════════════════════════════════════════
    print(f"{'='*60}")
    print(f"PHASE 1: Extract SVD basis at L{target_layer}")
    print(f"{'='*60}")

    capture = CaptureHook()
    handle = attn_module.register_forward_hook(capture)

    all_captures = []
    t0 = time.time()

    for pi, prob in enumerate(basis_problems):
        for lang in ["en", "zh"]:
            prompt_text = build_prompt(tokenizer, prob[lang])
            capture.captured = []
            capture.active = True
            generate(model, tokenizer, prompt_text, device, max_new)
            capture.active = False
            all_captures.extend(capture.captured)
            print(f"  Basis P{pi}/{lang}: {len(capture.captured)} steps")

    handle.remove()

    all_readouts = np.stack(all_captures)  # (N_total, d_model)
    print(f"  Readout matrix: {all_readouts.shape}")

    mean_vec = all_readouts.mean(axis=0)
    centered = all_readouts - mean_vec
    _, S, Vh = np.linalg.svd(centered, full_matrices=False)
    basis_vecs = Vh  # rows are right singular vectors, descending order

    cumvar = np.cumsum(S**2) / (S**2).sum()
    # Find 90% rank
    rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f"\n  Cross-problem effective rank at 90% variance: {rank_90}")
    print(f"  Variance explained:")
    for k in K_VALUES + [rank_90]:
        if k <= len(cumvar):
            print(f"    k={k:3d}: {cumvar[k - 1] * 100:5.1f}%")

    basis_time = time.time() - t0
    print(f"  Basis extraction: {basis_time:.0f}s\n")

    # ═════════════════════════════════════════════════════════════
    # PHASE 2: COMPRESSION TEST
    # ═════════════════════════════════════════════════════════════
    print(f"{'='*60}")
    print(f"PHASE 2: Compression test at L{target_layer}")
    print(f"{'='*60}")

    results = {}

    # ── Baseline (no hook) ──
    print(f"\n--- Baseline (no compression) ---")
    bl = []
    for pi, prob in enumerate(test_problems):
        for lang in ["en", "zh"]:
            prompt_text = build_prompt(tokenizer, prob[lang])
            text = generate(model, tokenizer, prompt_text, device, max_new)
            correct = check_answer(text, prob["answer"])
            bl.append({
                "problem_idx": pi,
                "lang": lang,
                "category": prob.get("category", "?"),
                "correct": correct,
                "text": text[:300],
            })
            mark = "\u2713" if correct else "\u2717"
            cat = prob.get("category", "?")[:4]
            print(f"  T{pi}/{lang}({cat}): {mark}  {text[:55]}...")

    bl_correct = sum(r["correct"] for r in bl)
    results["baseline"] = {
        "accuracy": bl_correct / len(bl),
        "correct": bl_correct,
        "total": len(bl),
        "per_problem": bl,
    }
    print(f"  Baseline: {bl_correct}/{len(bl)}\n")

    # ── Compression sweep ──
    for k in K_VALUES:
        print(f"--- k={k} ---")
        compressor = CompressionHook(mean_vec, basis_vecs, k, device)
        handle = attn_module.register_forward_hook(compressor)

        kr = []
        for pi, prob in enumerate(test_problems):
            for lang in ["en", "zh"]:
                prompt_text = build_prompt(tokenizer, prob[lang])
                compressor.active = True
                text = generate(model, tokenizer, prompt_text, device, max_new)
                compressor.active = False
                correct = check_answer(text, prob["answer"])
                kr.append({
                    "problem_idx": n_basis + pi,
                    "lang": lang,
                    "correct": correct,
                    "text": text[:300],
                })
                mark = "\u2713" if correct else "\u2717"
                print(f"  P{n_basis + pi}/{lang}: {mark}  {text[:60]}...")

        handle.remove()
        k_correct = sum(r["correct"] for r in kr)
        results[f"k={k}"] = {
            "accuracy": k_correct / len(kr),
            "correct": k_correct,
            "total": len(kr),
            "per_problem": kr,
        }
        print(f"  k={k}: {k_correct}/{len(kr)}\n")

    # ═════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SUMMARY — L{target_layer} compression")
    print(f"{'='*60}")
    print(f"  {'Condition':>12} | {'Score':>8} | {'Accuracy':>8} | {'VarExpl':>8}")
    print(f"  {'-'*50}")
    for label in ["baseline"] + [f"k={k}" for k in K_VALUES]:
        r = results[label]
        if label == "baseline":
            ve = "  ---"
        else:
            kv = int(label.split("=")[1])
            ve = f"{cumvar[kv - 1] * 100:5.1f}%" if kv <= len(cumvar) else "  ---"
        print(
            f"  {label:>12} | {r['correct']:>3}/{r['total']:<4} | "
            f"{r['accuracy']:>7.1%} | {ve:>8}"
        )

    # ── Save ──
    output = {
        "experiment": "C3: Constructive Compression Test",
        "model": MODEL_NAME,
        "target_layer": target_layer,
        "k_values": K_VALUES,
        "n_basis_problems": n_basis,
        "n_test_problems": n_test,
        "max_new": max_new,
        "basis_readout_shape": list(all_readouts.shape),
        "singular_values_top50": S[:50].tolist(),
        "cross_problem_rank_90pct": rank_90,
        "elapsed_s": total_time,
        "basis_variance_explained": {
            str(k): float(cumvar[k - 1])
            for k in sorted(set(K_VALUES + [rank_90]))
            if k <= len(cumvar)
        },
        "results": results,
    }
    outpath = OUTPUT_DIR / f"expC3_compression_L{target_layer}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
