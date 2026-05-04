"""Experiment V3: Phase Block TC0 Flip at N=20

V2 showed a hint: cooperative block (L18-L21) = +1, adversarial (L9-L17) = -1.
But N=10 is noise. Run the standard N=20 test set (first 4 per category × 2 langs).

5 conditions × 20 problems × 128 tokens = 100 generations. ~3 min.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re

MODEL_NAME = 'Qwen/Qwen2.5-3B'
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

d = model.config.hidden_size
N_TRAIN = 200
BATCH_SIZE = 16
ANALYSIS_LAYERS = list(range(9, 27))
MAX_TOKENS = 128
SCALE = -0.5
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
# Fit TC0
# =============================================================================
print("Fitting TC0 directions...")
problems = generate_pca_problems(N_TRAIN, seed=42)
per_cat = N_TRAIN // 5
direct_idx = list(range(per_cat, 2*per_cat)) + list(range(3*per_cat, 4*per_cat))
verbose_idx = list(range(0, per_cat)) + list(range(4*per_cat, 5*per_cat))

def extract_mlp_deltas(prompts):
    layer_data = {li: [] for li in ANALYSIS_LAYERS}
    captures = {}
    handles = []
    for li in ANALYSIS_LAYERS:
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
        for li in ANALYSIS_LAYERS:
            for j in range(captures[li].shape[0]):
                layer_data[li].append(captures[li][j, last_idx[j]].cpu().numpy())
    for h in handles:
        h.remove()
    for li in ANALYSIS_LAYERS:
        layer_data[li] = np.stack(layer_data[li])
    return layer_data

data_d = extract_mlp_deltas([problems[i]["en"] for i in direct_idx])
data_v = extract_mlp_deltas([problems[i]["en"] for i in verbose_idx])

tc0_dirs = {}
for li in ANALYSIS_LAYERS:
    diff = data_v[li].mean(0) - data_d[li].mean(0)
    tc0_dirs[li] = diff / (np.linalg.norm(diff) + 1e-10)

# Also fit lang dir for comparison
data_zh = extract_mlp_deltas([p["zh"] for p in problems])
data_en = extract_mlp_deltas([p["en"] for p in problems])
lang_dirs = {}
for li in ANALYSIS_LAYERS:
    diff = data_zh[li].mean(0) - data_en[li].mean(0)
    lang_dirs[li] = diff / (np.linalg.norm(diff) + 1e-10)

print(f"Directions fitted in {time.time()-t0:.1f}s")


# =============================================================================
# Standard N=20 test set
# =============================================================================
test_problems = []
for lang in ["en", "zh"]:
    for i in range(4):  # first 4 per category
        for cat_start in range(0, N_TRAIN, per_cat):
            prob = problems[cat_start + i]
            prompt = prob[lang]
            answer = compute_answer(prompt)
            if answer:
                test_problems.append({"prompt": prompt, "answer": answer, "lang": lang})

# Deduplicate (shouldn't have dupes but just in case)
seen = set()
unique_test = []
for p in test_problems:
    if p["prompt"] not in seen:
        seen.add(p["prompt"])
        unique_test.append(p)
test_problems = unique_test[:20]  # cap at 20

print(f"Test set: {len(test_problems)} problems")
for p in test_problems[:5]:
    print(f"  [{p['lang']}] {p['prompt'][:50]}... → {p['answer']}")


# =============================================================================
# Run conditions
# =============================================================================
phase_blocks = {
    "baseline": [],
    "adversarial_L9-L17": list(range(9, 18)),
    "cooperative_L18-L21": list(range(18, 22)),
    "ramp_L22-L26": list(range(22, 27)),
    "all_TC0_L9-L26": list(range(9, 27)),
    "all_LANG_L9-L26": list(range(9, 27)),  # language dir for comparison
}


def run_condition(prompt, answer, flip_layers, use_lang=False):
    handles = []
    dirs = lang_dirs if use_lang else tc0_dirs
    for li in flip_layers:
        direction = torch.tensor(dirs[li], dtype=torch.bfloat16, device=device)
        def make_hook(d_vec):
            def hook(module, inp, out):
                proj = (out.float() @ d_vec.float()).unsqueeze(-1) * d_vec.float()
                return out + (SCALE - 1.0) * proj.to(out.dtype)
            return hook
        handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(direction)))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    for h in handles:
        h.remove()

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    found = answer in gen_text
    return found


print("\n" + "=" * 70)
print(f"PHASE BLOCK TEST (N={len(test_problems)}, {MAX_TOKENS} tokens)")
print("=" * 70)

results = {}
for cname, layers in phase_blocks.items():
    use_lang = (cname == "all_LANG_L9-L26")
    score = 0
    for p in test_problems:
        if layers:
            f = run_condition(p["prompt"], p["answer"], layers, use_lang=use_lang)
        else:
            inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f = p["answer"] in gen_text
        score += f

    results[cname] = score
    n_l = len(layers) if layers else 0
    delta = score - results.get("baseline", score)
    print(f"  {cname:<25}: {score}/{len(test_problems)}  (Δ={delta:+d}, {n_l} layers)")

# =============================================================================
# Summary
# =============================================================================
base = results["baseline"]
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Baseline:             {base}/20")
print(f"  Adversarial (L9-17):  {results['adversarial_L9-L17']}/20  (Δ={results['adversarial_L9-L17']-base:+d})")
print(f"  Cooperative (L18-21): {results['cooperative_L18-L21']}/20  (Δ={results['cooperative_L18-L21']-base:+d})")
print(f"  Ramp (L22-26):        {results['ramp_L22-L26']}/20  (Δ={results['ramp_L22-L26']-base:+d})")
print(f"  All TC0 (L9-26):      {results['all_TC0_L9-L26']}/20  (Δ={results['all_TC0_L9-L26']-base:+d})")
print(f"  All LANG (L9-26):     {results['all_LANG_L9-L26']}/20  (Δ={results['all_LANG_L9-L26']-base:+d})")

# Additivity
sum_deltas = (results['adversarial_L9-L17'] - base) + (results['cooperative_L18-L21'] - base) + (results['ramp_L22-L26'] - base) + base
print(f"\n  Sum of block Δs:      {sum_deltas}/20")
print(f"  Superadditive?        {'YES' if results['all_TC0_L9-L26'] > sum_deltas else 'NO'}")

output = {
    "experiment": "V3: Phase Block TC0 Flip (N=20)",
    "model": MODEL_NAME, "scale": SCALE, "max_tokens": MAX_TOKENS,
    "n_test": len(test_problems),
    "results": results,
    "additivity": {
        "sum_blocks": sum_deltas, "all": results["all_TC0_L9-L26"],
        "superadditive": results["all_TC0_L9-L26"] > sum_deltas,
    },
    "runtime_seconds": time.time() - t0,
}
with open("output/expV3_phase_block_n20.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to output/expV3_phase_block_n20.json")
print(f"Total runtime: {time.time()-t0:.1f}s")
