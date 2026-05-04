"""Experiment V: Layer-by-Layer TC0 Flip

Which SINGLE layer's TC0 flip contributes most to the math speedup?

From Exp T: L9-L17 are adversarial (consecutive deltas anti-correlated),
L18-L21 are cooperative (positively correlated), L22-L26 ramp.

Prediction: TC0 flip helps most at L9-L17 (adversarial = model debating
verbose vs direct) and does nothing at L18-L21 (cooperative = decision made).

Design: For each layer L in L9-L26, flip TC0 at ONLY that layer.
10 math problems (EN), 128 tokens, score = found/10.
Also: 2 translation canaries per layer to confirm no breakage.

Uses TC0_raw directions from Exp U (category-proxy: combos+areas vs arithmetic+sequences).
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
# Step 1: Fit TC0 directions (same as Exp U — category proxy, prefill only)
# =============================================================================
print("=" * 70)
print("STEP 1: Fitting TC0 directions (category proxy, batched prefill)")
print("=" * 70)

problems = generate_pca_problems(N_TRAIN, seed=42)
per_cat = N_TRAIN // 5

# Direct = combos + areas, Verbose = arithmetic + sequences
direct_idx = list(range(per_cat, 2*per_cat)) + list(range(3*per_cat, 4*per_cat))
verbose_idx = list(range(0, per_cat)) + list(range(4*per_cat, 5*per_cat))


def extract_mlp_deltas(prompts, label=""):
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
            delta = captures[li]
            for j in range(delta.shape[0]):
                layer_data[li].append(delta[j, last_idx[j]].cpu().numpy())

    for h in handles:
        h.remove()

    for li in ANALYSIS_LAYERS:
        layer_data[li] = np.stack(layer_data[li])
    print(f"  {label}: {len(prompts)} samples")
    return layer_data


data_direct = extract_mlp_deltas([problems[i]["en"] for i in direct_idx], "direct_en")
data_verbose = extract_mlp_deltas([problems[i]["en"] for i in verbose_idx], "verbose_en")

tc0_dirs = {}
for li in ANALYSIS_LAYERS:
    diff = data_verbose[li].mean(0) - data_direct[li].mean(0)
    norm = np.linalg.norm(diff)
    tc0_dirs[li] = diff / (norm + 1e-10)
    print(f"  L{li}: TC0 norm={norm:.1f}")

print(f"Direction fitting done in {time.time()-t0:.1f}s")


# =============================================================================
# Step 2: Single-layer flip test (18 layers × 10 math + 2 translation × 128 tok)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Single-layer TC0 flip (18 runs × 12 problems × 128 tokens)")
print("=" * 70)

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

SCALE = -0.5


def generate_with_single_layer_flip(prompt, answer_strings, flip_layer):
    """Flip TC0 at exactly one layer during generation."""
    direction = torch.tensor(tc0_dirs[flip_layer], dtype=torch.bfloat16, device=device)

    def hook(module, inp, out):
        proj = (out.float() @ direction.float()).unsqueeze(-1) * direction.float()
        return out + (SCALE - 1.0) * proj.to(out.dtype)

    handle = model.model.layers[flip_layer].mlp.register_forward_hook(hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    found = any(a.lower() in gen_text.lower() for a in answer_strings)
    return found, gen_text[:200]


def generate_baseline(prompt, answer_strings):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    found = any(a.lower() in gen_text.lower() for a in answer_strings)
    return found, gen_text[:200]


# Baseline first
print("\n--- Baseline (no flip) ---")
base_math = 0
base_trans = 0
baseline_details = []
for p in test_math:
    f, txt = generate_baseline(p["prompt"], p["answers"])
    base_math += f
    baseline_details.append({"prompt": p["prompt"], "found": f})
for p in test_trans:
    f, txt = generate_baseline(p["prompt"], p["answers"])
    base_trans += f
print(f"  Math: {base_math}/10  Translation: {base_trans}/2")

# All-layer flip (L9-L26)
print("\n--- All-layer TC0 flip (L9-L26) ---")
all_handles = []
for li in ANALYSIS_LAYERS:
    direction = torch.tensor(tc0_dirs[li], dtype=torch.bfloat16, device=device)
    def make_hook(d_vec):
        def hook(module, inp, out):
            proj = (out.float() @ d_vec.float()).unsqueeze(-1) * d_vec.float()
            return out + (SCALE - 1.0) * proj.to(out.dtype)
        return hook
    all_handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(direction)))

all_math = 0
all_trans = 0
for p in test_math:
    inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    all_math += any(a.lower() in gen_text.lower() for a in p["answers"])
for p in test_trans:
    inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    all_trans += any(a.lower() in gen_text.lower() for a in p["answers"])
for h in all_handles:
    h.remove()
print(f"  Math: {all_math}/10  Translation: {all_trans}/2")

# Single-layer sweeps
print("\n--- Single-layer TC0 flip sweep ---")
results_per_layer = {}

for li in ANALYSIS_LAYERS:
    math_score = 0
    trans_score = 0

    for p in test_math:
        f, _ = generate_with_single_layer_flip(p["prompt"], p["answers"], li)
        math_score += f

    for p in test_trans:
        f, _ = generate_with_single_layer_flip(p["prompt"], p["answers"], li)
        trans_score += f

    delta_math = math_score - base_math
    results_per_layer[li] = {
        "math_score": math_score,
        "math_delta": delta_math,
        "trans_score": trans_score,
    }
    marker = " <<<" if delta_math > 0 else (" !!!" if delta_math < -1 else "")
    print(f"  L{li}: math={math_score}/10 (Δ={delta_math:+d})  trans={trans_score}/2{marker}")


# =============================================================================
# Summary with T's phase map overlay
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: TC0 flip effect vs T's phase transition map")
print("=" * 70)

# Cross-layer correlation signs from T (math)
phase_map = {
    9: "adversarial",   # L9->L10 cos=-0.068
    10: "adversarial",  # L10->L11 cos=-0.012
    11: "adversarial",  # L11->L12 cos=-0.043
    12: "adversarial",  # L12->L13 cos=-0.047
    13: "adversarial",  # L13->L14 cos=-0.122
    14: "adversarial",  # L14->L15 cos=-0.082
    15: "adversarial",  # L15->L16 cos=-0.069
    16: "adversarial",  # L16->L17 cos=-0.062
    17: "transition",   # L17->L18 cos=+0.052
    18: "cooperative",  # L18->L19 cos=+0.172
    19: "cooperative",  # L19->L20 cos=+0.214
    20: "cooperative",  # L20->L21 cos=+0.274
    21: "reset",        # L21->L22 cos=+0.054
    22: "ramp",         # L22->L23 cos=+0.030
    23: "ramp",         # L23->L24 cos=+0.077
    24: "ramp",         # L24->L25 cos=+0.085
    25: "ramp",         # L25->L26 cos=+0.163
    26: "ramp",
}

print(f"\n{'Layer':<8} {'Phase':<14} {'Math':<8} {'Δ':<6} {'Trans':<8}")
print("-" * 46)

adversarial_deltas = []
cooperative_deltas = []
ramp_deltas = []

for li in ANALYSIS_LAYERS:
    r = results_per_layer[li]
    phase = phase_map[li]
    print(f"  L{li:<4} {phase:<14} {r['math_score']}/10   {r['math_delta']:+d}     {r['trans_score']}/2")

    if phase == "adversarial":
        adversarial_deltas.append(r["math_delta"])
    elif phase == "cooperative":
        cooperative_deltas.append(r["math_delta"])
    elif phase in ("ramp", "reset"):
        ramp_deltas.append(r["math_delta"])

print(f"\nPhase averages (math Δ):")
print(f"  Adversarial (L9-L16):  avg Δ = {np.mean(adversarial_deltas):+.2f}")
print(f"  Cooperative (L18-L20): avg Δ = {np.mean(cooperative_deltas):+.2f}")
print(f"  Ramp (L21-L26):        avg Δ = {np.mean(ramp_deltas):+.2f}")
print(f"\n  Baseline: {base_math}/10  All-layer: {all_math}/10")

# Save
output = {
    "experiment": "V: Layer-by-Layer TC0 Flip",
    "model": MODEL_NAME,
    "scale": SCALE,
    "max_tokens": MAX_TOKENS,
    "n_math": 10,
    "n_trans": 2,
    "baseline": {"math": base_math, "trans": base_trans},
    "all_layer_flip": {"math": all_math, "trans": all_trans},
    "per_layer": {str(li): {**r, "phase": phase_map[li]} for li, r in results_per_layer.items()},
    "phase_averages": {
        "adversarial_L9_L16": {"avg_delta": float(np.mean(adversarial_deltas)), "layers": "L9-L16"},
        "cooperative_L18_L20": {"avg_delta": float(np.mean(cooperative_deltas)), "layers": "L18-L20"},
        "ramp_L21_L26": {"avg_delta": float(np.mean(ramp_deltas)), "layers": "L21-L26"},
    },
    "runtime_seconds": time.time() - t0,
}

with open("output/expV_layer_by_layer_flip.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to output/expV_layer_by_layer_flip.json")
print(f"Total runtime: {time.time()-t0:.1f}s")
