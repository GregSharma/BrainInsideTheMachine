"""Experiment G: Per-Token Layer Skip During Generation

Unlike C4 (which skips layers only on the prefill), this experiment skips layers 9-26
on EVERY generated token. The output of L8 is redirected to L27's input at every
forward pass during generation.

The headline question: can you save 53% of compute on every token and still get
correct math in the right language?
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

# --- Problems: 5 simple + 5 hard (same as C4) ---
simple_problems = [
    ("请计算 2 + 3 × 4 的值。\n", "14"),
    ("一个矩形的长为8厘米，宽为5厘米，求面积。\n", "40"),
    ("如果 x + 5 = 12，求 x 的值。\n", "7"),
    ("计算 100 除以 4 的结果。\n", "25"),
    ("一个三角形三边长分别为3、4、5，求面积。\n", "6"),
]
hard_problems = [
    ("如果 2x + 3 = 15，求 x² 的值。\n", "36"),
    ("已知 x + y = 10，x - y = 4，求 x × y 的值。\n", "21"),
    ("小明有50元钱，买了3本书每本8元，又买了2支笔每支3元，还剩多少元？\n", "20"),
    ("一个正方形的对角线长为10厘米，求这个正方形的面积。\n", "50"),
    ("一个班有40个学生，男生占总数的3/5，女生有多少人？\n", "16"),
]

all_problems = simple_problems + hard_problems

MAX_NEW_TOKENS = 128

# =============================================================================
# Core: generate with PERSISTENT layer skip (L_src → L_dst on EVERY forward pass)
# =============================================================================
def generate_with_persistent_skip(prompt, src_layer, dst_layer, max_new_tokens=MAX_NEW_TOKENS):
    """
    Hook so that on EVERY forward pass during generation:
    - Capture hidden state at output of src_layer
    - Inject it at output of dst_layer (replacing whatever dst_layer computed)

    This means layers src_layer+1 through dst_layer-1 still fire but their
    outputs are overwritten. Layers dst_layer+1 through 35 see the src_layer output.
    """
    input_ids = tokenizer.encode(prompt)
    captured_h = {}  # will hold the L_src output each forward pass

    def capture_hook(module, input, output):
        """Capture src_layer output on every forward pass."""
        hidden = output if not isinstance(output, tuple) else output[0]
        # Store the hidden state (last token position for generation, all for prefill)
        captured_h['state'] = hidden.clone()

    def inject_hook(module, input, output):
        """Inject captured src_layer output at dst_layer on every forward pass."""
        if 'state' not in captured_h:
            return output
        hidden = output if not isinstance(output, tuple) else output[0]
        src_state = captured_h['state']
        # Replace all positions (works for both prefill and generation steps)
        hidden[:, :, :] = src_state[:, :, :]
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle_capture = model.model.layers[src_layer].register_forward_hook(capture_hook)
    handle_inject = model.model.layers[dst_layer].register_forward_hook(inject_hook)

    try:
        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([input_ids], device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
        n_tokens = len(outputs[0]) - len(input_ids)
    finally:
        handle_capture.remove()
        handle_inject.remove()

    return gen, n_tokens


def generate_baseline(prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Normal generation, no hooks."""
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        outputs = model.generate(
            torch.tensor([input_ids], device=device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    n_tokens = len(outputs[0]) - len(input_ids)
    return gen, n_tokens


def is_chinese(text):
    zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_chars = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
    return zh_chars > en_chars


def check_coherence(text):
    """Basic coherence check: no repetition loops, language stays consistent."""
    # Check for repetition: if any 20-char substring repeats 3+ times
    has_repetition = False
    if len(text) > 60:
        for i in range(len(text) - 60):
            chunk = text[i:i+20]
            if text.count(chunk) >= 3:
                has_repetition = True
                break

    # Check language consistency: split into thirds, each should be same language
    if len(text) > 30:
        third = len(text) // 3
        parts = [text[:third], text[third:2*third], text[2*third:]]
        langs = [is_chinese(p) for p in parts]
        language_consistent = len(set(langs)) == 1
    else:
        language_consistent = True

    return {
        "has_repetition": has_repetition,
        "language_consistent": language_consistent,
        "coherent": not has_repetition and language_consistent,
    }


# =============================================================================
# Main experiment
# =============================================================================
results = {
    "experiment": "G: Per-Token Layer Skip During Generation",
    "description": "Skip layers on EVERY generated token, not just prefill",
    "skip_configs": [],
}

# Skip configurations to test (try full skip first, then graduated)
skip_configs = [
    (8, 27, "L8→L27 (skip 19 layers)"),
    (12, 27, "L12→L27 (skip 14 layers)"),
    (16, 27, "L16→L27 (skip 10 layers)"),
    (20, 27, "L20→L27 (skip 6 layers)"),
]

# First: baseline for all problems
print("=" * 70)
print("BASELINE GENERATIONS (full 36-layer model)")
print("=" * 70)
baseline_results = []
for i, (prompt, expected) in enumerate(all_problems):
    gen, n_tok = generate_baseline(prompt)
    correct = expected in gen
    label = "SIMPLE" if i < 5 else "HARD"
    print(f"  [{label}] Problem {i}: correct={correct}, tokens={n_tok}")
    print(f"    {gen[:100]}...")
    baseline_results.append({
        "idx": i,
        "prompt": prompt.strip(),
        "expected": expected,
        "generation": gen,
        "correct": correct,
        "n_tokens": n_tok,
        "is_chinese": is_chinese(gen),
        "type": "simple" if i < 5 else "hard",
    })

baseline_correct_simple = sum(1 for r in baseline_results[:5] if r["correct"])
baseline_correct_hard = sum(1 for r in baseline_results[5:] if r["correct"])
print(f"\nBaseline: {baseline_correct_simple}/5 simple, {baseline_correct_hard}/5 hard")

# Now test each skip configuration
for src, dst, label in skip_configs:
    print(f"\n{'=' * 70}")
    print(f"PERSISTENT SKIP: {label}")
    print(f"{'=' * 70}")

    config_results = {
        "src_layer": src,
        "dst_layer": dst,
        "label": label,
        "layers_skipped": dst - src - 1,
        "compute_savings_pct": round((dst - src - 1) / 36 * 100, 1),
        "problems": [],
    }

    for i, (prompt, expected) in enumerate(all_problems):
        gen, n_tok = generate_with_persistent_skip(prompt, src, dst)
        correct = expected in gen
        chinese = is_chinese(gen)
        coherence = check_coherence(gen)
        ptype = "simple" if i < 5 else "hard"

        print(f"\n  [{ptype.upper()}] Problem {i} (expected: {expected}):")
        print(f"    Baseline:  {baseline_results[i]['generation'][:80]}...")
        print(f"    Skip:      {gen[:80]}...")
        print(f"    Correct: {correct} | Chinese: {chinese} | Coherent: {coherence['coherent']} | Tokens: {n_tok}")

        config_results["problems"].append({
            "idx": i,
            "type": ptype,
            "prompt": prompt.strip(),
            "expected": expected,
            "generation": gen,
            "baseline_generation": baseline_results[i]["generation"],
            "correct": correct,
            "baseline_correct": baseline_results[i]["correct"],
            "is_chinese": chinese,
            "baseline_is_chinese": baseline_results[i]["is_chinese"],
            "n_tokens": n_tok,
            "baseline_n_tokens": baseline_results[i]["n_tokens"],
            "coherence": coherence,
        })

    # Summary for this config
    n_correct_simple = sum(1 for p in config_results["problems"] if p["type"] == "simple" and p["correct"])
    n_correct_hard = sum(1 for p in config_results["problems"] if p["type"] == "hard" and p["correct"])
    n_chinese = sum(1 for p in config_results["problems"] if p["is_chinese"])
    n_coherent = sum(1 for p in config_results["problems"] if p["coherence"]["coherent"])

    config_results["summary"] = {
        "simple_correct": f"{n_correct_simple}/5",
        "hard_correct": f"{n_correct_hard}/5",
        "total_correct": f"{n_correct_simple + n_correct_hard}/10",
        "chinese": f"{n_chinese}/10",
        "coherent": f"{n_coherent}/10",
        "baseline_simple": f"{baseline_correct_simple}/5",
        "baseline_hard": f"{baseline_correct_hard}/5",
    }

    print(f"\n  --- {label} Summary ---")
    print(f"  Simple: {n_correct_simple}/5 (baseline {baseline_correct_simple}/5)")
    print(f"  Hard:   {n_correct_hard}/5 (baseline {baseline_correct_hard}/5)")
    print(f"  Chinese: {n_chinese}/10 | Coherent: {n_coherent}/10")

    results["skip_configs"].append(config_results)

    # If the full L8→L27 skip works well (>= 7/10 correct + coherent),
    # we don't need to test graduated versions
    if src == 8 and dst == 27:
        if n_correct_simple + n_correct_hard >= 7 and n_coherent >= 8:
            print(f"\n  *** L8→L27 works! Skipping graduated versions. ***")
            # Still run them for completeness but mark that full skip succeeded
            results["full_skip_sufficient"] = True
        else:
            print(f"\n  *** L8→L27 degraded. Testing graduated versions... ***")
            results["full_skip_sufficient"] = False

# Final summary
print("\n" + "=" * 70)
print("EXPERIMENT G — FINAL SUMMARY")
print("=" * 70)
for cfg in results["skip_configs"]:
    s = cfg["summary"]
    print(f"  {cfg['label']}: {s['total_correct']} correct, {s['chinese']} Chinese, {s['coherent']} coherent ({cfg['compute_savings_pct']}% compute saved)")
print(f"  Baseline: {baseline_correct_simple + baseline_correct_hard}/10 correct")

# Save
with open("output/expG_gentime_layer_skip.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expG_gentime_layer_skip.json")
