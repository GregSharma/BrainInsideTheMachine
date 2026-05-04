"""Experiment U: TC0 — Language-Deconfounded Verbosity Direction

PHASE 1 (this script): Fit directions, compute cosines, test on MATH ONLY (5 problems × 4 conditions × 128 tokens).
PHASE 2 (separate script if TC0 works): Domain transfer on translation/code/factual.

The question: is TC0 (within-language verbose vs direct) a different direction from
the zh/en language direction? If cos(TC0, lang) << 1, we have a genuinely new dial.
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

n_layers = model.config.num_hidden_layers
d = model.config.hidden_size
N_TRAIN = 200
BATCH_SIZE = 16
ANALYSIS_LAYERS = list(range(9, 27))
MAX_TEST_TOKENS = 128  # keep short for phase 1

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
# Step 1: Use CATEGORY as proxy for verbose vs direct (NO generation needed)
# =============================================================================
# From R4 we know: combinations → often direct (FAT=16), arithmetic/sequences → verbose (FAT=191-298)
# Category layout in generate_pca_problems (seed=42): 40 each of add/mult, combo, remainder, area, sequence
print("=" * 70)
print("STEP 1: Category-based verbose/direct split (zero generation cost)")
print("=" * 70)

problems = generate_pca_problems(N_TRAIN, seed=42)
per_cat = N_TRAIN // 5  # 40

# Direct categories: combinations (idx 40-79) + areas (idx 120-159) — typically short answers
# Verbose categories: arithmetic (idx 0-39) + sequences (idx 160-199) — typically long CoT
direct_idx = list(range(per_cat, 2*per_cat)) + list(range(3*per_cat, 4*per_cat))  # combos + areas
verbose_idx = list(range(0, per_cat)) + list(range(4*per_cat, 5*per_cat))  # add/mult + sequences

print(f"  Direct (combos + areas): {len(direct_idx)} problems")
print(f"  Verbose (add/mult + sequences): {len(verbose_idx)} problems")
print(f"  (Remainders excluded as ambiguous)")
print(f"Step 1 done in {time.time()-t0:.2f}s")


# =============================================================================
# Step 2: Extract MLP deltas + inputs (batched prefill, fast)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Extracting MLP deltas (batched prefill)")
print("=" * 70)

def extract_mlp_deltas_and_inputs(prompts, label=""):
    layer_data = {li: {"deltas": [], "inputs": []} for li in ANALYSIS_LAYERS}
    captures = {}
    handles = []

    for li in ANALYSIS_LAYERS:
        layer = model.model.layers[li]
        def make_mlp_hook(layer_idx):
            def hook(module, inp, out):
                captures.setdefault(layer_idx, {})["delta"] = out.detach().float()
            return hook
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))

        def make_ln_hook(layer_idx):
            def hook(module, inp):
                captures.setdefault(layer_idx, {})["input"] = inp[0].detach().float()
            return hook
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(make_ln_hook(li)))

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        captures.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        attn_mask = inputs["attention_mask"]
        with torch.no_grad():
            model(**inputs)
        last_idx = attn_mask.sum(dim=1) - 1

        for li in ANALYSIS_LAYERS:
            delta = captures[li]["delta"]
            inp = captures[li]["input"]
            for j in range(delta.shape[0]):
                layer_data[li]["deltas"].append(delta[j, last_idx[j]].cpu().numpy())
                layer_data[li]["inputs"].append(inp[j, last_idx[j]].cpu().numpy())

    for h in handles:
        h.remove()

    for li in ANALYSIS_LAYERS:
        layer_data[li]["deltas"] = np.stack(layer_data[li]["deltas"])
        layer_data[li]["inputs"] = np.stack(layer_data[li]["inputs"])
    print(f"  {label}: {len(prompts)} samples × {len(ANALYSIS_LAYERS)} layers")
    return layer_data


# EN direct, EN verbose, all ZH, all EN (for language dir)
data_direct_en = extract_mlp_deltas_and_inputs([problems[i]["en"] for i in direct_idx], "direct_en")
data_verbose_en = extract_mlp_deltas_and_inputs([problems[i]["en"] for i in verbose_idx], "verbose_en")
data_direct_zh = extract_mlp_deltas_and_inputs([problems[i]["zh"] for i in direct_idx], "direct_zh")
data_verbose_zh = extract_mlp_deltas_and_inputs([problems[i]["zh"] for i in verbose_idx], "verbose_zh")
data_all_zh = extract_mlp_deltas_and_inputs([p["zh"] for p in problems], "all_zh")
data_all_en = extract_mlp_deltas_and_inputs([p["en"] for p in problems], "all_en")

print(f"Extraction done in {time.time()-t0:.1f}s")


# =============================================================================
# Step 3: Compute TC0, TC0_innov, lang_dir, and all cosines
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Direction fitting + cosine analysis")
print("=" * 70)

tc0_raw_dirs = {}
tc0_innov_dirs = {}
lang_dirs = {}
tc0_zh_dirs = {}

def get_innovations(layer_data, li):
    delta = layer_data[li]["deltas"]
    inp = layer_data[li]["inputs"]
    inp_norm_sq = np.sum(inp ** 2, axis=1, keepdims=True) + 1e-10
    proj_coeff = np.sum(delta * inp, axis=1, keepdims=True) / inp_norm_sq
    return delta - proj_coeff * inp

for li in ANALYSIS_LAYERS:
    # TC0 raw (EN): verbose - direct
    diff = data_verbose_en[li]["deltas"].mean(0) - data_direct_en[li]["deltas"].mean(0)
    norm = np.linalg.norm(diff)
    tc0_raw_dirs[li] = diff / (norm + 1e-10)

    # TC0 innovation
    innov_v = get_innovations(data_verbose_en, li)
    innov_d = get_innovations(data_direct_en, li)
    diff_i = innov_v.mean(0) - innov_d.mean(0)
    norm_i = np.linalg.norm(diff_i)
    tc0_innov_dirs[li] = diff_i / (norm_i + 1e-10)

    # Language direction (zh - en)
    lang_diff = data_all_zh[li]["deltas"].mean(0) - data_all_en[li]["deltas"].mean(0)
    lang_norm = np.linalg.norm(lang_diff)
    lang_dirs[li] = lang_diff / (lang_norm + 1e-10)

    # TC0 within ZH (cross-validation)
    zh_diff = data_verbose_zh[li]["deltas"].mean(0) - data_direct_zh[li]["deltas"].mean(0)
    zh_norm = np.linalg.norm(zh_diff)
    tc0_zh_dirs[li] = zh_diff / (zh_norm + 1e-10)

    # Cosines
    c_tc0_lang = float(np.dot(tc0_raw_dirs[li], lang_dirs[li]))
    c_tc0i_lang = float(np.dot(tc0_innov_dirs[li], lang_dirs[li]))
    c_tc0_tc0i = float(np.dot(tc0_raw_dirs[li], tc0_innov_dirs[li]))
    c_en_zh = float(np.dot(tc0_raw_dirs[li], tc0_zh_dirs[li]))

    # Cohen's d for TC0 raw
    proj_v = data_verbose_en[li]["deltas"] @ tc0_raw_dirs[li]
    proj_d = data_direct_en[li]["deltas"] @ tc0_raw_dirs[li]
    cd = float((proj_v.mean() - proj_d.mean()) / np.sqrt((proj_v.std()**2 + proj_d.std()**2) / 2 + 1e-10))

    print(f"  L{li}: cos(TC0,lang)={c_tc0_lang:+.3f}  cos(TC0,TC0_i)={c_tc0_tc0i:.3f}  "
          f"cos(TC0_en,TC0_zh)={c_en_zh:+.3f}  d={cd:.1f}  "
          f"norms: tc0={norm:.1f} innov={norm_i:.1f} lang={lang_norm:.1f}")


# =============================================================================
# Step 4: MINIMAL generation test — 5 math problems × 4 conditions × 128 tokens
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Math-only flip test (5 problems × 4 conditions × 128 tokens)")
print("=" * 70)

test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133"},
    {"prompt": "What is the remainder when 100 is divided by 7?", "answer": "2"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60"},
    {"prompt": "Calculate 15 × 8.", "answer": "120"},
]

# Also 2 translation canaries — does TC0 break translation?
canary_problems = [
    {"prompt": "Translate 'I love programming' to Chinese.", "answers": ["我喜欢编程", "我爱编程", "编程"]},
    {"prompt": "Translate '今天天气很好' to English.", "answers": ["weather is good", "nice weather", "good weather", "weather is very good", "weather is great"]},
]


def generate_with_flip(prompt, answer_strings, dirs_dict, scale):
    handles = []
    if dirs_dict is not None:
        for li in ANALYSIS_LAYERS:
            direction = torch.tensor(dirs_dict[li], dtype=torch.bfloat16, device=device)
            def make_hook(d_vec, s):
                def hook(module, inp, out):
                    proj = (out.float() @ d_vec.float()).unsqueeze(-1) * d_vec.float()
                    return out + (s - 1.0) * proj.to(out.dtype)
                return hook
            handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(direction, scale)))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_TEST_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    for h in handles:
        h.remove()

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    found = any(a.lower() in gen_text.lower() for a in answer_strings)
    return {"found": found, "output": gen_text[:200], "n_tokens": len(gen_ids)}


conditions = {
    "baseline": (None, 1.0),
    "lang_flip_-0.5": (lang_dirs, -0.5),
    "tc0_raw_-0.5": (tc0_raw_dirs, -0.5),
    "tc0_innov_-0.5": (tc0_innov_dirs, -0.5),
}

# Math test
print("\n--- Math (5 problems) ---")
math_results = {}
for cname, (dirs, scale) in conditions.items():
    found = 0
    items = []
    for p in test_problems:
        r = generate_with_flip(p["prompt"], [p["answer"]], dirs, scale)
        found += r["found"]
        items.append({"prompt": p["prompt"], **r})
    math_results[cname] = {"found": found, "total": 5, "items": items}
    print(f"  {cname:<20}: {found}/5")

# Translation canary
print("\n--- Translation canaries (2 problems) ---")
trans_results = {}
for cname, (dirs, scale) in conditions.items():
    found = 0
    items = []
    for p in canary_problems:
        r = generate_with_flip(p["prompt"], p["answers"], dirs, scale)
        found += r["found"]
        items.append({"prompt": p["prompt"], **r})
    trans_results[cname] = {"found": found, "total": 2, "items": items}
    print(f"  {cname:<20}: {found}/2")

# =============================================================================
# Save
# =============================================================================
# Average cosines
avg_cos = {}
for name, d1, d2 in [("tc0_lang", tc0_raw_dirs, lang_dirs),
                       ("tc0i_lang", tc0_innov_dirs, lang_dirs),
                       ("tc0_tc0i", tc0_raw_dirs, tc0_innov_dirs),
                       ("tc0en_tc0zh", tc0_raw_dirs, tc0_zh_dirs)]:
    avg_cos[name] = float(np.mean([np.dot(d1[li], d2[li]) for li in ANALYSIS_LAYERS]))

output = {
    "experiment": "U: TC0 Verbosity Direction (Phase 1)",
    "model": MODEL_NAME,
    "method": "Category proxy: combos+areas=direct, arithmetic+sequences=verbose. EN only for TC0.",
    "n_direct": len(direct_idx),
    "n_verbose": len(verbose_idx),
    "avg_cosines": avg_cos,
    "per_layer_cosines": {
        str(li): {
            "cos_tc0_lang": float(np.dot(tc0_raw_dirs[li], lang_dirs[li])),
            "cos_tc0innov_lang": float(np.dot(tc0_innov_dirs[li], lang_dirs[li])),
            "cos_tc0_tc0innov": float(np.dot(tc0_raw_dirs[li], tc0_innov_dirs[li])),
            "cos_tc0en_tc0zh": float(np.dot(tc0_raw_dirs[li], tc0_zh_dirs[li])),
        } for li in ANALYSIS_LAYERS
    },
    "math_results": math_results,
    "translation_canary": trans_results,
    "runtime_seconds": time.time() - t0,
}

with open("output/expU_tc0_verbosity.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Avg cos(TC0_raw, lang_dir) = {avg_cos['tc0_lang']:.3f}")
print(f"Avg cos(TC0_innov, lang_dir) = {avg_cos['tc0i_lang']:.3f}")
print(f"Avg cos(TC0_raw, TC0_innov) = {avg_cos['tc0_tc0i']:.3f}")
print(f"Avg cos(TC0_en, TC0_zh) = {avg_cos['tc0en_tc0zh']:.3f}")
print(f"\nMath: baseline={math_results['baseline']['found']}/5  "
      f"lang_flip={math_results['lang_flip_-0.5']['found']}/5  "
      f"tc0_raw={math_results['tc0_raw_-0.5']['found']}/5  "
      f"tc0_innov={math_results['tc0_innov_-0.5']['found']}/5")
print(f"Translation: baseline={trans_results['baseline']['found']}/2  "
      f"lang_flip={trans_results['lang_flip_-0.5']['found']}/2  "
      f"tc0_raw={trans_results['tc0_raw_-0.5']['found']}/2  "
      f"tc0_innov={trans_results['tc0_innov_-0.5']['found']}/2")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("Saved to output/expU_tc0_verbosity.json")
