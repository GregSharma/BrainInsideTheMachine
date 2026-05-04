"""Experiment V2: Phase Block TC0 Flip

V showed single-layer TC0 flip = noise (±1 at N=10). But T showed clear
phase structure in cross-layer correlations:
  - Adversarial (L9-L17): consecutive deltas anti-correlated
  - Cooperative (L18-L21): consecutive deltas positively correlated
  - Ramp (L22-L26): building toward output

Question: which PHASE BLOCK carries the strategy switch?
Flip TC0 at each block independently. 10 math + 2 translation, 128 tokens.

If adversarial block alone flips strategy → the debate zone IS the dial.
If cooperative block alone flips → consensus zone IS the dial.
If all blocks equal → truly distributed, no phase structure in effect.
If blocks are additive → the ensemble is just linear accumulation.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_NAME = 'Qwen/Qwen2.5-3B'
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
ANALYSIS_LAYERS = list(range(9, 27))
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


# =============================================================================
# Fit TC0 directions (same as V)
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
        def make_hook(layer_idx):
            def hook(module, inp, out):
                captures[layer_idx] = out.detach().float()
            return hook
        handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(li)))

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        captures.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        attn_mask = inputs["attention_mask"]
        with torch.no_grad():
            model(**inputs)
        last_idx = attn_mask.sum(dim=1) - 1
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

print(f"Directions fitted in {time.time()-t0:.1f}s")

# =============================================================================
# Phase blocks
# =============================================================================
phase_blocks = {
    "adversarial_L9-L17": list(range(9, 18)),
    "cooperative_L18-L21": list(range(18, 22)),
    "ramp_L22-L26": list(range(22, 27)),
    "all_L9-L26": list(range(9, 27)),
}

test_math = [
    {"prompt": "Calculate 47 + 86.", "answers": ["133"]},
    {"prompt": "What is the remainder when 100 is divided by 7?", "answers": ["2"]},
    {"prompt": "Find the value of C(10, 3).", "answers": ["120"]},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answers": ["60"]},
    {"prompt": "Calculate 15 × 8.", "answers": ["120"]},
    {"prompt": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.", "answers": ["40"]},
    {"prompt": "Calculate 387 × 29.", "answers": ["11223"]},
    {"prompt": "What is the remainder when 7654 is divided by 37?", "answers": ["34"]},
    {"prompt": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.", "answers": ["3475"]},
    {"prompt": "A rectangle has length 47 and width 33. Find its area.", "answers": ["1551"]},
]

test_trans = [
    {"prompt": "Translate 'I love programming' to Chinese.", "answers": ["我喜欢编程", "我爱编程", "编程"]},
    {"prompt": "Translate '今天天气很好' to English.", "answers": ["weather is good", "nice weather", "good weather", "weather is very good", "weather is great"]},
]


def run_with_block_flip(prompt, answer_strings, flip_layers):
    handles = []
    for li in flip_layers:
        direction = torch.tensor(tc0_dirs[li], dtype=torch.bfloat16, device=device)
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
    found = any(a.lower() in gen_text.lower() for a in answer_strings)
    return found, gen_text[:200]


# =============================================================================
# Run: baseline + 4 block conditions
# =============================================================================
print("\n" + "=" * 70)
print("PHASE BLOCK TC0 FLIP TEST")
print("=" * 70)

conditions = {"baseline": []}  # empty = no flip layers
conditions.update(phase_blocks)

results = {}
for cname, layers in conditions.items():
    math_score = 0
    trans_score = 0
    math_details = []

    for p in test_math:
        if layers:
            f, txt = run_with_block_flip(p["prompt"], p["answers"], layers)
        else:
            # baseline
            inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f = any(a.lower() in gen_text.lower() for a in p["answers"])
            txt = gen_text[:200]
        math_score += f
        math_details.append({"prompt": p["prompt"], "found": f, "output": txt})

    for p in test_trans:
        if layers:
            f, txt = run_with_block_flip(p["prompt"], p["answers"], layers)
        else:
            inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f = any(a.lower() in gen_text.lower() for a in p["answers"])
            txt = gen_text[:200]
        trans_score += f

    results[cname] = {"math": math_score, "trans": trans_score, "layers": layers, "details": math_details}
    n_layers_flipped = len(layers) if layers else 0
    print(f"  {cname:<25}: math={math_score}/10  trans={trans_score}/2  ({n_layers_flipped} layers)")

# =============================================================================
# Additivity check
# =============================================================================
print("\n" + "=" * 70)
print("ADDITIVITY CHECK")
print("=" * 70)
base = results["baseline"]["math"]
adv = results["adversarial_L9-L17"]["math"]
coop = results["cooperative_L18-L21"]["math"]
ramp = results["ramp_L22-L26"]["math"]
all_layers = results["all_L9-L26"]["math"]

sum_blocks = (adv - base) + (coop - base) + (ramp - base) + base
print(f"  Baseline:        {base}/10")
print(f"  Adversarial:     {adv}/10  (Δ={adv-base:+d})")
print(f"  Cooperative:     {coop}/10  (Δ={coop-base:+d})")
print(f"  Ramp:            {ramp}/10  (Δ={ramp-base:+d})")
print(f"  All layers:      {all_layers}/10  (Δ={all_layers-base:+d})")
print(f"  Sum of block Δs: {sum_blocks}/10  (predicted if additive)")
print(f"  Superadditive?   {'YES' if all_layers > sum_blocks else 'NO'} (all={all_layers} vs sum={sum_blocks})")

# Save
output = {
    "experiment": "V2: Phase Block TC0 Flip",
    "model": MODEL_NAME,
    "scale": SCALE,
    "max_tokens": MAX_TOKENS,
    "results": {k: {"math": v["math"], "trans": v["trans"], "n_layers": len(v["layers"]),
                     "math_details": v["details"]}
                for k, v in results.items()},
    "additivity": {
        "baseline": base, "adversarial": adv, "cooperative": coop,
        "ramp": ramp, "all": all_layers, "sum_blocks": sum_blocks,
        "superadditive": all_layers > sum_blocks,
    },
    "runtime_seconds": time.time() - t0,
}

with open("output/expV2_phase_block_flip.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to output/expV2_phase_block_flip.json")
print(f"Total runtime: {time.time()-t0:.1f}s")
