"""Experiment C4 Validation: Raw Layer Skip on HARDER Problems
Test whether the 19-layer skip (L8→L27) works on multi-step problems.
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

# Load L8 hidden states (we'll extract fresh ones for new prompts)
data = np.load("output/all_layers_lasttok.npz")

# Harder problems — multi-step reasoning required
hard_problems_zh = [
    # Multi-step algebra
    "如果 2x + 3 = 15，求 x² 的值。\n",
    # System of equations
    "已知 x + y = 10，x - y = 4，求 x × y 的值。\n",
    # Word problem with reasoning
    "小明有50元钱，买了3本书每本8元，又买了2支笔每支3元，还剩多少元？\n",
    # Geometry with multiple steps
    "一个正方形的对角线长为10厘米，求这个正方形的面积。\n",
    # Fraction/ratio problem
    "一个班有40个学生，男生占总数的3/5，女生有多少人？\n",
    # Percentage problem
    "一件商品原价200元，先打八折，再打九折，最终价格是多少？\n",
    # Sequence/pattern
    "数列 2, 6, 18, 54, ...，求第6项的值。\n",
    # Combinatorics-lite
    "从5个人中选2个人组成一组，一共有多少种选法？\n",
    # Multi-step arithmetic
    "计算 (15 × 4 - 8) ÷ (3 + 4) 的值。\n",
    # Speed/distance problem
    "一辆车以60公里/小时的速度行驶了2.5小时，行驶了多少公里？\n",
]

# Expected answers for verification
expected_answers = [
    "36",           # 2x+3=15 → x=6 → x²=36
    "21",           # x=7,y=3 → xy=21
    "20",           # 50-3×8-2×3=50-24-6=20
    "50",           # d=10 → s=10/√2 → area=s²=50
    "16",           # 40×(1-3/5)=40×2/5=16
    "144",          # 200×0.8×0.9=144
    "486",          # 2×3^5=2×243=486
    "10",           # C(5,2)=10
    "约7.43或52/7", # (60-8)/7=52/7≈7.43
    "150",          # 60×2.5=150
]

inject_layers = [27, 28]

def generate_with_injection(prompt, injection_vector, inject_layer, max_new_tokens=128):
    input_ids = tokenizer.encode(prompt)
    injected = [False]

    def hook_fn(module, input, output):
        if not injected[0]:
            hidden = output if not isinstance(output, tuple) else output[0]
            vec = torch.tensor(injection_vector, dtype=hidden.dtype, device=hidden.device)
            hidden[0, -1, :] = vec
            injected[0] = True
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return output

    handle = model.model.layers[inject_layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([input_ids], device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    finally:
        handle.remove()

def extract_L8_hidden(prompt):
    """Extract L8 hidden state for the last token of a prompt."""
    input_ids = tokenizer.encode(prompt)
    captured = {}

    def hook_fn(module, input, output):
        if not captured:
            hidden = output if not isinstance(output, tuple) else output[0]
            captured["h"] = hidden[0, -1, :].detach().cpu().float().numpy()

    handle = model.model.layers[8].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(torch.tensor([input_ids], device=device))
    finally:
        handle.remove()

    return captured["h"]

results = {"problems": [], "summary": {}}

# First: get baseline generations (no injection)
print("=== BASELINE GENERATIONS (no skip) ===")
baselines = []
for i, prompt in enumerate(hard_problems_zh):
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        outputs = model.generate(
            torch.tensor([input_ids], device=device),
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    baselines.append(gen)
    print(f"  Problem {i}: {gen[:100]}...")

# Now: extract L8 and test skip
print("\n=== EXTRACTING L8 HIDDEN STATES ===")
h_L8_list = []
for i, prompt in enumerate(hard_problems_zh):
    h = extract_L8_hidden(prompt)
    h_L8_list.append(h)
    print(f"  Problem {i}: norm={np.linalg.norm(h):.1f}")

# Test at each injection layer
for inject_layer in inject_layers:
    print(f"\n{'='*60}")
    print(f"RAW LAYER SKIP: L8 → L{inject_layer}")
    print(f"{'='*60}")

    skip_gens = []
    for i, prompt in enumerate(hard_problems_zh):
        gen = generate_with_injection(prompt, h_L8_list[i], inject_layer, max_new_tokens=128)
        skip_gens.append(gen)

        # Check if answer is in the generation
        answer_present = expected_answers[i] in gen
        print(f"\n  Problem {i} (expected: {expected_answers[i]}):")
        print(f"    Baseline: {baselines[i][:80]}...")
        print(f"    Skip:     {gen[:80]}...")
        print(f"    Answer present: {'YES' if answer_present else 'NO'}")

        results["problems"].append({
            "problem_idx": i,
            "prompt": prompt.strip(),
            "expected_answer": expected_answers[i],
            "inject_layer": inject_layer,
            "baseline": baselines[i],
            "skip_generation": gen,
            "answer_in_baseline": expected_answers[i] in baselines[i],
            "answer_in_skip": answer_present,
            "skip_is_chinese": sum(1 for c in gen if '\u4e00' <= c <= '\u9fff') > sum(1 for c in gen if ('a' <= c <= 'z') or ('A' <= c <= 'Z')),
        })

    # Summary for this injection layer
    n_correct_baseline = sum(1 for i, b in enumerate(baselines) if expected_answers[i] in b)
    n_correct_skip = sum(1 for p in results["problems"]
                        if p["inject_layer"] == inject_layer and p["answer_in_skip"])
    n_chinese_skip = sum(1 for p in results["problems"]
                        if p["inject_layer"] == inject_layer and p["skip_is_chinese"])

    print(f"\n  --- L{inject_layer} Summary ---")
    print(f"  Baseline correct: {n_correct_baseline}/{len(hard_problems_zh)}")
    print(f"  Skip correct: {n_correct_skip}/{len(hard_problems_zh)}")
    print(f"  Skip in Chinese: {n_chinese_skip}/{len(hard_problems_zh)}")

    results["summary"][f"L{inject_layer}"] = {
        "baseline_correct": n_correct_baseline,
        "skip_correct": n_correct_skip,
        "skip_chinese": n_chinese_skip,
        "total": len(hard_problems_zh),
    }

# Save
with open("output/expC4_hard_validation.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expC4_hard_validation.json")
