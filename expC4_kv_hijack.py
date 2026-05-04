#!/usr/bin/env python3
"""
Exp C4: Cross-Lingual KV Hijack

Tests whether the reading mechanism is cross-lingually shared at each layer.

Design:
  For each problem P and each target layer L:
  1. Run P in English → save KV cache (K_en, V_en) at layer L
  2. Run P in Chinese → save KV cache (K_zh, V_zh) at layer L
  3. Start generation from EN prompt, but at layer L replace EN context KV
     with ZH context KV → generate with EN query reading ZH workspace
  4. Start generation from ZH prompt, but at layer L replace ZH context KV
     with EN context KV → generate with ZH query reading EN workspace
  5. Grade accuracy

Prediction from Vision B (softmax flip at L31-32):
  - L32+: attention operator strips language → hijack should WORK
    (EN query can read ZH context successfully)
  - L27-: attention operator is language-specific → hijack should FAIL
  - L31: crossover → marginal

The hijack replaces context-token KV only. The generated-token KV grows
normally from the model's own computation.
"""

import json, time, re, argparse, copy
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

# Single-layer sweep + multi-layer ranges
SINGLE_LAYERS = [10, 18, 24, 27, 29, 31, 32, 33, 35]
RANGE_CONDITIONS = {
    "all_layers": list(range(36)),
    "late_L27_L35": list(range(27, 36)),
    "mid_L18_L26": list(range(18, 27)),
    "early_L0_L17": list(range(18)),
}


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


def encode_prompt(model, tokenizer, prompt_text, device):
    """Run prompt through model, return KV cache and prompt length."""
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    return out.past_key_values, ids.shape[1], out.logits


def generate_from_cache(model, tokenizer, past_kv, first_logits, device,
                        max_new=MAX_NEW):
    """Continue generation from an existing KV cache."""
    # Pick first token from the logits
    next_id = int(first_logits[0, -1].argmax().item())
    if next_id == tokenizer.eos_token_id:
        return tokenizer.decode([next_id], skip_special_tokens=True)

    generated_ids = [next_id]
    cur_input = torch.tensor([[next_id]], device=device)

    with torch.inference_mode():
        for _ in range(max_new - 1):
            out = model(cur_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax().item())
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            cur_input = torch.tensor([[next_id]], device=device)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def hijack_kv_at_layer(kv_query, kv_donor, layer_idx, prompt_len_query,
                       prompt_len_donor):
    """Replace context-token KV at one layer in kv_query with kv_donor's.

    Only replaces the prompt (context) portion of the cache.
    Any previously generated tokens in kv_query are kept.

    Returns a new DynamicCache with the hijacked layer.
    API: kv.layers[li].keys / .values (shape: batch, n_kv_heads, seq, head_dim)
    """
    from transformers.cache_utils import DynamicCache

    new_cache = DynamicCache()

    for li in range(len(kv_query)):
        if li == layer_idx:
            # Replace context KV with donor's context KV
            donor_k = kv_donor.layers[li].keys[:, :, :prompt_len_donor, :]
            donor_v = kv_donor.layers[li].values[:, :, :prompt_len_donor, :]

            # If query had generated tokens beyond prompt, keep them
            q_seq_len = kv_query.layers[li].keys.shape[2]
            if q_seq_len > prompt_len_query:
                gen_k = kv_query.layers[li].keys[:, :, prompt_len_query:, :]
                gen_v = kv_query.layers[li].values[:, :, prompt_len_query:, :]
                new_k = torch.cat([donor_k, gen_k], dim=2)
                new_v = torch.cat([donor_v, gen_v], dim=2)
            else:
                new_k = donor_k
                new_v = donor_v

            new_cache.update(new_k, new_v, li)
        else:
            # Keep original
            new_cache.update(
                kv_query.layers[li].keys.clone(),
                kv_query.layers[li].values.clone(),
                li,
            )

    return new_cache


def hijack_kv_at_layers(kv_query, kv_donor, layer_indices, prompt_len_query,
                        prompt_len_donor):
    """Replace context-token KV at MULTIPLE layers."""
    from transformers.cache_utils import DynamicCache

    hijack_set = set(layer_indices)
    new_cache = DynamicCache()

    for li in range(len(kv_query)):
        if li in hijack_set:
            donor_k = kv_donor.layers[li].keys[:, :, :prompt_len_donor, :]
            donor_v = kv_donor.layers[li].values[:, :, :prompt_len_donor, :]
            new_cache.update(donor_k, donor_v, li)
        else:
            new_cache.update(
                kv_query.layers[li].keys.clone(),
                kv_query.layers[li].values.clone(),
                li,
            )

    return new_cache


def generate_with_hijack(model, tokenizer, prompt_text_query,
                         kv_donor, prompt_len_donor, hijack_layers,
                         device, max_new=MAX_NEW):
    """Generate from query prompt but with donor's KV at hijack_layers.

    hijack_layers: int (single layer) or list of ints (multiple layers).
    """
    if isinstance(hijack_layers, int):
        hijack_layers = [hijack_layers]

    ids = tokenizer(prompt_text_query, return_tensors="pt").input_ids.to(device)
    prompt_len_q = ids.shape[1]

    with torch.inference_mode():
        out = model(ids, use_cache=True)
    query_kv = out.past_key_values

    hijacked_kv = hijack_kv_at_layers(
        query_kv, kv_donor, hijack_layers, prompt_len_q, prompt_len_donor
    )

    return generate_from_cache(
        model, tokenizer, hijacked_kv, out.logits, device, max_new
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="3 problems, 64 tokens")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: preset sweep)")
    args = parser.parse_args()

    device = "cuda"
    n_problems = 3 if args.dry else 10
    max_new = 64 if args.dry else MAX_NEW
    layers = [int(x) for x in args.layers.split(",")] if args.layers else SINGLE_LAYERS

    print(f"{'='*60}")
    print(f"Exp C4: Cross-Lingual KV Hijack")
    print(f"{'='*60}")
    print(f"Model:  {MODEL_NAME}")
    print(f"Layers: {layers}")
    print(f"N:      {n_problems} problems, {max_new} tokens")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Problems (stratified)
    all_problems = generate_problems()
    all_test = get_test_subset(all_problems)[:n_problems]
    print(f"  {len(all_test)} test problems\n")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Baselines + cache collection
    # ═══════════════════════════════════════════════════════════
    print(f"{'='*60}")
    print("PHASE 1: Baselines + KV cache collection")
    print(f"{'='*60}")

    problem_data = []
    t0 = time.time()

    for pi, prob in enumerate(all_test):
        entry = {"idx": pi, "answer": prob["answer"],
                 "category": prob.get("category", "?")}

        for lang in ["en", "zh"]:
            prompt_text = build_prompt(tokenizer, prob[lang])
            kv, plen, logits = encode_prompt(model, tokenizer, prompt_text, device)
            text = generate_from_cache(model, tokenizer, kv, logits, device, max_new)
            correct = check_answer(text, prob["answer"])

            entry[f"{lang}_prompt"] = prompt_text
            entry[f"{lang}_kv"] = kv
            entry[f"{lang}_plen"] = plen
            entry[f"{lang}_logits"] = logits
            entry[f"{lang}_baseline_text"] = text[:300]
            entry[f"{lang}_baseline_correct"] = correct

            mark = "\u2713" if correct else "\u2717"
            cat = prob.get("category", "?")[:4]
            print(f"  P{pi}/{lang}({cat}): {mark}  {text[:55]}...")

        problem_data.append(entry)

    bl_en = sum(e["en_baseline_correct"] for e in problem_data)
    bl_zh = sum(e["zh_baseline_correct"] for e in problem_data)
    n = len(problem_data)
    print(f"\n  Baselines: EN={bl_en}/{n}, ZH={bl_zh}/{n}")
    print(f"  Cache collection: {time.time() - t0:.0f}s\n")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: KV Hijack at each layer
    # ═══════════════════════════════════════════════════════════
    print(f"{'='*60}")
    print("PHASE 2: Cross-lingual KV hijack")
    print(f"{'='*60}")

    results = {
        "baseline_en": bl_en,
        "baseline_zh": bl_zh,
        "baseline_total": bl_en + bl_zh,
        "n_problems": n,
        "layers": {},
    }

    for li in layers:
        print(f"\n--- Layer {li} ---")
        layer_results = {"en_query_zh_kv": [], "zh_query_en_kv": []}

        for pi, entry in enumerate(problem_data):
            cat = entry["category"][:4]

            # EN query + ZH KV at this layer
            text_en_zh = generate_with_hijack(
                model, tokenizer,
                entry["en_prompt"], entry["zh_kv"], entry["zh_plen"],
                li, device, max_new,
            )
            correct_en_zh = check_answer(text_en_zh, entry["answer"])

            # ZH query + EN KV at this layer
            text_zh_en = generate_with_hijack(
                model, tokenizer,
                entry["zh_prompt"], entry["en_kv"], entry["en_plen"],
                li, device, max_new,
            )
            correct_zh_en = check_answer(text_zh_en, entry["answer"])

            mark1 = "\u2713" if correct_en_zh else "\u2717"
            mark2 = "\u2713" if correct_zh_en else "\u2717"
            print(f"  P{pi}({cat}): EN+ZH_kv={mark1}  ZH+EN_kv={mark2}")

            layer_results["en_query_zh_kv"].append({
                "problem_idx": pi, "category": entry["category"],
                "correct": correct_en_zh, "text": text_en_zh[:300],
            })
            layer_results["zh_query_en_kv"].append({
                "problem_idx": pi, "category": entry["category"],
                "correct": correct_zh_en, "text": text_zh_en[:300],
            })

        en_zh_correct = sum(r["correct"] for r in layer_results["en_query_zh_kv"])
        zh_en_correct = sum(r["correct"] for r in layer_results["zh_query_en_kv"])
        layer_results["en_query_zh_kv_total"] = en_zh_correct
        layer_results["zh_query_en_kv_total"] = zh_en_correct
        results["layers"][str(li)] = layer_results

        print(f"  L{li}: EN+ZH_kv={en_zh_correct}/{n}, ZH+EN_kv={zh_en_correct}/{n}")

    # ── Multi-layer range conditions ──
    results["ranges"] = {}
    print(f"\n{'='*60}")
    print("PHASE 3: Multi-layer range hijack")
    print(f"{'='*60}")

    for name, layer_list in RANGE_CONDITIONS.items():
        print(f"\n--- {name} ({len(layer_list)} layers) ---")
        range_results = {"en_query_zh_kv": [], "zh_query_en_kv": []}

        for pi, entry in enumerate(problem_data):
            cat = entry["category"][:4]

            text_en_zh = generate_with_hijack(
                model, tokenizer,
                entry["en_prompt"], entry["zh_kv"], entry["zh_plen"],
                layer_list, device, max_new,
            )
            correct_en_zh = check_answer(text_en_zh, entry["answer"])

            text_zh_en = generate_with_hijack(
                model, tokenizer,
                entry["zh_prompt"], entry["en_kv"], entry["en_plen"],
                layer_list, device, max_new,
            )
            correct_zh_en = check_answer(text_zh_en, entry["answer"])

            mark1 = "\u2713" if correct_en_zh else "\u2717"
            mark2 = "\u2713" if correct_zh_en else "\u2717"
            print(f"  P{pi}({cat}): EN+ZH_kv={mark1}  ZH+EN_kv={mark2}")

            range_results["en_query_zh_kv"].append({
                "problem_idx": pi, "category": entry["category"],
                "correct": correct_en_zh, "text": text_en_zh[:300],
            })
            range_results["zh_query_en_kv"].append({
                "problem_idx": pi, "category": entry["category"],
                "correct": correct_zh_en, "text": text_zh_en[:300],
            })

        en_zh_correct = sum(r["correct"] for r in range_results["en_query_zh_kv"])
        zh_en_correct = sum(r["correct"] for r in range_results["zh_query_en_kv"])
        range_results["en_query_zh_kv_total"] = en_zh_correct
        range_results["zh_query_en_kv_total"] = zh_en_correct
        results["ranges"][name] = range_results
        print(f"  {name}: EN+ZH_kv={en_zh_correct}/{n}, ZH+EN_kv={zh_en_correct}/{n}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print("SUMMARY — Cross-Lingual KV Hijack")
    print(f"{'='*60}")
    print(f"  Baselines: EN={bl_en}/{n}, ZH={bl_zh}/{n}")
    print(f"  {'Layer':>6} | {'EN+ZH_kv':>9} | {'ZH+EN_kv':>9} | Signal")
    print(f"  {'-'*50}")
    for li in layers:
        lr = results["layers"][str(li)]
        ez = lr["en_query_zh_kv_total"]
        ze = lr["zh_query_en_kv_total"]
        # Compare to baselines
        signal = "WORKS" if (ez + ze) >= (bl_en + bl_zh) * 0.8 else "FAILS"
        print(f"  L{li:>4} | {ez:>5}/{n:<3} | {ze:>5}/{n:<3} | {signal}")

    print(f"\n  {'Range':>16} | {'EN+ZH_kv':>9} | {'ZH+EN_kv':>9}")
    print(f"  {'-'*50}")
    for name in RANGE_CONDITIONS:
        rr = results["ranges"][name]
        ez = rr["en_query_zh_kv_total"]
        ze = rr["zh_query_en_kv_total"]
        print(f"  {name:>16} | {ez:>5}/{n:<3} | {ze:>5}/{n:<3}")

    # Save
    output = {
        "experiment": "C4: Cross-Lingual KV Hijack",
        "model": MODEL_NAME,
        "n_layers": N_LAYERS,
        "hijack_layers": layers,
        "n_problems": n,
        "max_new": max_new,
        "elapsed_s": total_time,
        "results": results,
    }

    # Strip non-serializable data
    for entry in problem_data:
        for key in list(entry.keys()):
            if key.endswith("_kv") or key.endswith("_logits"):
                del entry[key]
    output["problem_data"] = problem_data

    outpath = OUTPUT_DIR / "expC4_kv_hijack.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
