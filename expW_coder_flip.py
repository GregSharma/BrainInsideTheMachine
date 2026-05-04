"""Experiment W: Language Direction Flip on Qwen2.5-Coder-3B

The prediction: if the flip effect is training-determined (English=verbose in training data),
then a code-heavy model might NOT show it. If architectural, it will.

Preliminary observation: Coder-3B STILL shows zh=direct, en=verbose for math.
So the entanglement persists. But does the flip intervention still work?

Same pipeline as P3/R3: fit lang direction on 200 math problems, flip at L9-L26, test N=20.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re

MODEL_NAME = 'Qwen/Qwen2.5-Coder-3B'
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

n_layers = model.config.num_hidden_layers
d = model.config.hidden_size
N_TRAIN = 200
BATCH_SIZE = 16
STRIP_LAYERS = list(range(9, 27))
MAX_TOKENS = 128
SCALE = -0.5

print(f"Model: {MODEL_NAME} ({n_layers} layers, d={d})")
t0 = time.time()


def generate_pca_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh, en = f"计算 {a} + {b} 的值。", f"Calculate {a} + {b}."
        else:
            zh, en = f"计算 {a} × {b} 的值。", f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                          "en": f"Find the value of C({n_val}, {k_val})."})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？",
                          "en": f"What is the remainder when {a} is divided by {b}?"})
    for _ in range(per_cat):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                          "en": f"A rectangle has length {w} and width {h}. Find its area."})
    for _ in range(per_cat):
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。",
                          "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_terms} terms."})
    return problems


def compute_answer(text):
    m = re.search(r'(?:Calculate|计算) (\d+) \+ (\d+)', text)
    if m: return str(int(m.group(1)) + int(m.group(2)))
    m = re.search(r'(?:Calculate|计算) (\d+) [×x] (\d+)', text)
    if m: return str(int(m.group(1)) * int(m.group(2)))
    m = re.search(r'C\((\d+),?\s*(\d+)\)', text)
    if m:
        from math import comb
        return str(comb(int(m.group(1)), int(m.group(2))))
    m = re.search(r'(?:remainder when|除以)\s*(\d+)\s*(?:is divided by|除以)\s*(\d+)', text)
    if not m:
        m = re.search(r'(\d+)\s*除以\s*(\d+)', text)
    if m: return str(int(m.group(1)) % int(m.group(2)))
    m = re.search(r'(?:length|长为)\s*(\d+).*?(?:width|宽为)\s*(\d+)', text)
    if m: return str(int(m.group(1)) * int(m.group(2)))
    m = re.search(r'(?:first term|首项为)\s*(\d+).*?(?:common difference|公差为)\s*(\d+).*?(?:first|前)\s*(\d+)', text)
    if m:
        a1, dd, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return str(n * a1 + n * (n - 1) // 2 * dd)
    return None


# =============================================================================
# Fit language direction (batched prefill, same as V3)
# =============================================================================
print("Fitting language directions...")
problems = generate_pca_problems(N_TRAIN, seed=42)

def extract_mlp_deltas(prompts):
    layer_data = {li: [] for li in STRIP_LAYERS}
    captures = {}
    handles = []
    for li in STRIP_LAYERS:
        def make_hook(idx):
            def hook(module, inp, out):
                captures[idx] = out.detach().float()
            return hook
        handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(li)))
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        captures.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            model(**inputs)
        for li in STRIP_LAYERS:
            for j in range(captures[li].shape[0]):
                layer_data[li].append(captures[li][j, last_idx[j]].cpu().numpy())
    for h in handles:
        h.remove()
    for li in STRIP_LAYERS:
        layer_data[li] = np.stack(layer_data[li])
    return layer_data

data_zh = extract_mlp_deltas([p["zh"] for p in problems])
data_en = extract_mlp_deltas([p["en"] for p in problems])

lang_dirs = {}
for li in STRIP_LAYERS:
    diff = data_zh[li].mean(0) - data_en[li].mean(0)
    norm = np.linalg.norm(diff)
    lang_dirs[li] = diff / (norm + 1e-10)
    if li in [10, 18, 26]:
        # Cohen's d
        proj_zh = data_zh[li] @ lang_dirs[li]
        proj_en = data_en[li] @ lang_dirs[li]
        cd = (proj_zh.mean() - proj_en.mean()) / np.sqrt((proj_zh.std()**2 + proj_en.std()**2) / 2 + 1e-10)
        print(f"  L{li}: norm={norm:.1f}, Cohen's d={cd:.1f}")

print(f"Direction fitting done in {time.time()-t0:.1f}s")


# =============================================================================
# Standard N=20 test set
# =============================================================================
per_cat = N_TRAIN // 5
test_problems = []
for lang in ["en", "zh"]:
    for i in range(4):
        for cat_start in range(0, N_TRAIN, per_cat):
            prob = problems[cat_start + i]
            prompt = prob[lang]
            answer = compute_answer(prompt)
            if answer:
                test_problems.append({"prompt": prompt, "answer": answer, "lang": lang})

seen = set()
unique_test = []
for p in test_problems:
    if p["prompt"] not in seen:
        seen.add(p["prompt"])
        unique_test.append(p)
test_problems = unique_test[:20]
print(f"Test set: {len(test_problems)} problems")


# =============================================================================
# Run: baseline vs lang flip
# =============================================================================
def run_condition(prompt, answer, flip_layers, scale):
    handles = []
    for li in flip_layers:
        direction = torch.tensor(lang_dirs[li], dtype=torch.bfloat16, device=device)
        def make_hook(d_vec, s):
            def hook(module, inp, out):
                proj = (out.float() @ d_vec.float()).unsqueeze(-1) * d_vec.float()
                return out + (s - 1.0) * proj.to(out.dtype)
            return hook
        handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(direction, scale)))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    for h in handles:
        h.remove()

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    found = answer in gen_text
    return found, gen_text[:200]


conditions = {
    "baseline": ([], 1.0),
    "flip_-0.5": (STRIP_LAYERS, -0.5),
    "flip_-1.0": (STRIP_LAYERS, -1.0),
}

print("\n" + "=" * 70)
print(f"CODER-3B LANGUAGE FLIP TEST (N={len(test_problems)}, {MAX_TOKENS} tokens)")
print("=" * 70)

results = {}
for cname, (layers, scale) in conditions.items():
    score = 0
    en_score = 0
    zh_score = 0
    details = []
    for p in test_problems:
        if layers:
            f, txt = run_condition(p["prompt"], p["answer"], layers, scale)
        else:
            inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f = p["answer"] in gen_text
            txt = gen_text[:200]
        score += f
        if p["lang"] == "en":
            en_score += f
        else:
            zh_score += f
        details.append({"prompt": p["prompt"][:50], "answer": p["answer"], "lang": p["lang"], "found": f, "output": txt})

    results[cname] = {"score": score, "en": en_score, "zh": zh_score, "details": details}
    print(f"  {cname:<15}: {score}/20 (EN={en_score}, ZH={zh_score})")


# =============================================================================
# Compare to base model
# =============================================================================
print(f"\n{'='*70}")
print("COMPARISON: Coder-3B vs Base-3B")
print(f"{'='*70}")
print(f"  Base-3B (from V3):   baseline=5/20, flip-0.5=13/20  (Δ=+8)")
print(f"  Coder-3B:            baseline={results['baseline']['score']}/20, "
      f"flip-0.5={results['flip_-0.5']['score']}/20  "
      f"(Δ={results['flip_-0.5']['score'] - results['baseline']['score']:+d})")

output = {
    "experiment": "W: Coder-3B Language Flip",
    "model": MODEL_NAME,
    "n_layers": n_layers, "d": d,
    "scale": SCALE,
    "max_tokens": MAX_TOKENS,
    "n_test": len(test_problems),
    "results": {k: {"score": v["score"], "en": v["en"], "zh": v["zh"]}
                for k, v in results.items()},
    "comparison_to_base": {
        "base_3b_baseline": 5, "base_3b_flip": 13,
        "coder_3b_baseline": results["baseline"]["score"],
        "coder_3b_flip": results["flip_-0.5"]["score"],
    },
    "runtime_seconds": time.time() - t0,
}
with open("output/expW_coder_flip.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to output/expW_coder_flip.json")
print(f"Total runtime: {time.time()-t0:.1f}s")
