"""Experiment G2: Full-Stack Shallow Skip

Every token — prompt AND generation — takes the same path:
  L0-L8 → skip L9-L26 → L27-L35

No token ever sees L9-L26. KV cache only exists at layers 0-8 and 27-35.
This eliminates the KV cache manifold mismatch that killed Experiment G.

The hook redirects L8 output to L27 input on EVERY forward pass.
Layers 9-26 still execute but their outputs are overwritten.
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


def generate_fullstack_shallow(prompt, src_layer=8, dst_layer=27, max_new_tokens=MAX_NEW_TOKENS):
    """
    Every forward pass (prompt + generation): capture L8 output, inject at L27.
    Both hooks fire on EVERY call — no 'fired once' flags.
    """
    input_ids = tokenizer.encode(prompt)
    captured = {}

    def capture_hook(module, input, output):
        hidden = output if not isinstance(output, tuple) else output[0]
        captured['h'] = hidden.clone()

    def inject_hook(module, input, output):
        if 'h' not in captured:
            return output
        hidden = output if not isinstance(output, tuple) else output[0]
        src = captured['h']
        # Replace ALL positions (works for both prefill and single-token generation)
        hidden[:, :src.shape[1], :] = src[:, :src.shape[1], :]
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    h_cap = model.model.layers[src_layer].register_forward_hook(capture_hook)
    h_inj = model.model.layers[dst_layer].register_forward_hook(inject_hook)

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
        h_cap.remove()
        h_inj.remove()

    return gen, n_tokens


def generate_baseline(prompt, max_new_tokens=MAX_NEW_TOKENS):
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
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
    return zh > en


def check_coherence(text):
    has_rep = False
    if len(text) > 60:
        for i in range(len(text) - 60):
            if text.count(text[i:i+20]) >= 3:
                has_rep = True
                break
    if len(text) > 30:
        third = len(text) // 3
        langs = [is_chinese(text[j*third:(j+1)*third]) for j in range(3)]
        lang_con = len(set(langs)) == 1
    else:
        lang_con = True
    return {"has_repetition": has_rep, "language_consistent": lang_con,
            "coherent": not has_rep and lang_con}


# ---- Baselines ----
print("=" * 70)
print("BASELINES")
print("=" * 70)
baseline_results = []
for i, (prompt, expected) in enumerate(all_problems):
    gen, n_tok = generate_baseline(prompt)
    correct = expected in gen
    label = "SIMPLE" if i < 5 else "HARD"
    print(f"  [{label}] {i}: correct={correct}, tokens={n_tok}")
    print(f"    {gen[:100]}...")
    baseline_results.append({
        "idx": i, "prompt": prompt.strip(), "expected": expected,
        "generation": gen, "correct": correct, "n_tokens": n_tok,
        "is_chinese": is_chinese(gen), "type": "simple" if i < 5 else "hard",
    })

b_simple = sum(1 for r in baseline_results[:5] if r["correct"])
b_hard = sum(1 for r in baseline_results[5:] if r["correct"])
print(f"\nBaseline: {b_simple}/5 simple, {b_hard}/5 hard = {b_simple+b_hard}/10")

# ---- Full-stack shallow skip configs ----
configs = [
    (8, 27, "L8→L27 (skip 18, 50% compute)"),
    (12, 27, "L12→L27 (skip 14, 39% compute)"),
    (16, 27, "L16→L27 (skip 10, 28% compute)"),
    (20, 27, "L20→L27 (skip 6, 17% compute)"),
]

results = {
    "experiment": "G2: Full-Stack Shallow Skip (every token same path)",
    "baselines": baseline_results,
    "configs": [],
}

for src, dst, label in configs:
    print(f"\n{'='*70}")
    print(f"FULL-STACK SHALLOW: {label}")
    print(f"{'='*70}")

    cfg = {"src": src, "dst": dst, "label": label,
           "layers_skipped": dst - src - 1,
           "compute_savings_pct": round((dst - src - 1) / 36 * 100, 1),
           "problems": []}

    for i, (prompt, expected) in enumerate(all_problems):
        gen, n_tok = generate_fullstack_shallow(prompt, src, dst)
        correct = expected in gen
        chinese = is_chinese(gen)
        coh = check_coherence(gen)
        ptype = "simple" if i < 5 else "hard"

        print(f"\n  [{ptype.upper()}] {i} (expected: {expected}):")
        print(f"    Baseline: {baseline_results[i]['generation'][:80]}...")
        print(f"    G2:       {gen[:80]}...")
        print(f"    Correct: {correct} | Chinese: {chinese} | Coherent: {coh['coherent']} | Tokens: {n_tok}")

        cfg["problems"].append({
            "idx": i, "type": ptype, "prompt": prompt.strip(), "expected": expected,
            "generation": gen, "baseline_generation": baseline_results[i]["generation"],
            "correct": correct, "baseline_correct": baseline_results[i]["correct"],
            "is_chinese": chinese, "n_tokens": n_tok, "coherence": coh,
        })

    cs = sum(1 for p in cfg["problems"] if p["type"] == "simple" and p["correct"])
    ch = sum(1 for p in cfg["problems"] if p["type"] == "hard" and p["correct"])
    nc = sum(1 for p in cfg["problems"] if p["is_chinese"])
    co = sum(1 for p in cfg["problems"] if p["coherence"]["coherent"])
    cfg["summary"] = {
        "simple_correct": f"{cs}/5", "hard_correct": f"{ch}/5",
        "total_correct": f"{cs+ch}/10", "chinese": f"{nc}/10", "coherent": f"{co}/10",
    }
    print(f"\n  --- {label} ---")
    print(f"  Simple: {cs}/5 | Hard: {ch}/5 | Chinese: {nc}/10 | Coherent: {co}/10")
    results["configs"].append(cfg)

# Summary
print(f"\n{'='*70}")
print("G2 FINAL SUMMARY")
print("="*70)
for c in results["configs"]:
    s = c["summary"]
    print(f"  {c['label']}: {s['total_correct']} correct, {s['chinese']} Chinese, {s['coherent']} coherent")
print(f"  Baseline: {b_simple+b_hard}/10 correct")

with open("output/expG2_fullstack_shallow.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expG2_fullstack_shallow.json")
