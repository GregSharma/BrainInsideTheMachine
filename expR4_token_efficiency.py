"""Experiment R4: Token Efficiency — When Does the Answer First Appear?

For each problem, generate up to 512 tokens and record the token index
at which the correct answer string first appears in the decoded output.

Compares: baseline vs flip (scale=-0.5) on 3B.
This directly measures the "strategy switch" — how many tokens does each
mode need to reach the correct answer?

Metric: "first-appearance token" (FAT) — lower is better.
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
STRIP_LAYERS = list(range(9, 27))
MAX_TOKENS = 512

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
# Batched direction fitting (same as R3)
# =============================================================================
print(f"Fitting directions...")
problems = generate_pca_problems(N_TRAIN, seed=42)
fit_layers = [10, 14, 18, 22, 26]

lang_dirs = {}
for fl in fit_layers:
    mlp_out_list = []
    def mlp_cap(module, input, output):
        mlp_out_list.append(output.detach().float())
    handle = model.model.layers[fl].mlp.register_forward_hook(mlp_cap)

    zh_deltas, en_deltas = [], []
    for lang, deltas in [("zh", zh_deltas), ("en", en_deltas)]:
        for i in range(0, N_TRAIN, BATCH_SIZE):
            batch = [p[lang] for p in problems[i:i+BATCH_SIZE]]
            mlp_out_list.clear()
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad(): model(**inputs)
            attn_mask = inputs["attention_mask"]
            last_idx = attn_mask.sum(dim=1) - 1
            out = mlp_out_list[0]
            for j in range(out.shape[0]):
                deltas.append(out[j, last_idx[j]].cpu().numpy())

    handle.remove()
    diff = np.stack(zh_deltas).mean(0) - np.stack(en_deltas).mean(0)
    norm = np.linalg.norm(diff)
    lang_dirs[fl] = torch.tensor(diff / norm, dtype=torch.float32, device=device)

def get_lang_dir(li):
    return lang_dirs[min(fit_layers, key=lambda x: abs(x - li))]

strip_dirs = {li: get_lang_dir(li) for li in STRIP_LAYERS}
print(f"Directions fitted in {time.time()-t0:.1f}s")


# =============================================================================
# Test problems (EN + ZH, same 10 each)
# =============================================================================
test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133", "lang": "en"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60", "lang": "en"},
    {"prompt": "What is the remainder when 100 is divided by 7?", "answer": "2", "lang": "en"},
    {"prompt": "Calculate 15 × 8.", "answer": "120", "lang": "en"},
    {"prompt": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.", "answer": "40", "lang": "en"},
    {"prompt": "Calculate 387 × 29.", "answer": "11223", "lang": "en"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120", "lang": "en"},
    {"prompt": "What is the remainder when 7654 is divided by 37?", "answer": "34", "lang": "en"},
    {"prompt": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.", "answer": "3475", "lang": "en"},
    {"prompt": "A rectangle has length 47 and width 33. Find its area.", "answer": "1551", "lang": "en"},
    {"prompt": "计算 47 + 86 的值。", "answer": "133", "lang": "zh"},
    {"prompt": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60", "lang": "zh"},
    {"prompt": "100 除以 7 的余数是多少？", "answer": "2", "lang": "zh"},
    {"prompt": "计算 15 × 8 的值。", "answer": "120", "lang": "zh"},
    {"prompt": "等差数列首项为 2，公差为 3，求前 5 项之和。", "answer": "40", "lang": "zh"},
    {"prompt": "计算 387 × 29 的值。", "answer": "11223", "lang": "zh"},
    {"prompt": "求组合数 C(10, 3) 的值。", "answer": "120", "lang": "zh"},
    {"prompt": "7654 除以 37 的余数是多少？", "answer": "34", "lang": "zh"},
    {"prompt": "等差数列首项为 7，公差为 11，求前 25 项之和。", "answer": "3475", "lang": "zh"},
    {"prompt": "一个长方形的长为 47，宽为 33，求其面积。", "answer": "1551", "lang": "zh"},
]


def generate_with_tracking(prompt, answer, dirs_dict=None, scale=1.0):
    """Generate up to MAX_TOKENS, tracking when the answer first appears."""
    input_ids = tokenizer.encode(prompt)
    handles = []

    if dirs_dict and scale != 1.0:
        for li in STRIP_LAYERS:
            def make_hook(layer_idx, lang_dir, sc):
                def hook_fn(module, input, output):
                    delta = output.float()
                    proj = torch.sum(delta * lang_dir, dim=-1, keepdim=True)
                    lang_component = proj * lang_dir
                    modified = delta - lang_component + sc * lang_component
                    return modified.to(output.dtype)
                return hook_fn
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_hook(li, strip_dirs[li], scale)
                )
            )

    first_appearance = None  # token index where answer first appears
    try:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]

        for step in range(MAX_TOKENS - 1):
            # Check if answer has appeared yet
            text_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if first_appearance is None and answer in text_so_far:
                first_appearance = step + 1  # 1-indexed token count

            with torch.no_grad():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)
            if next_id == tokenizer.eos_token_id:
                # Final check
                text_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
                if first_appearance is None and answer in text_so_far:
                    first_appearance = len(generated_ids)
                break
    finally:
        for h in handles:
            h.remove()

    # One final check at the end
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if first_appearance is None and answer in full_text:
        first_appearance = len(generated_ids)

    return {
        "first_appearance_token": first_appearance,
        "total_tokens": len(generated_ids),
        "found": first_appearance is not None,
        "output": full_text[:300]
    }


# =============================================================================
# Run: baseline vs flip at -0.5
# =============================================================================
conditions = [
    ("baseline", None, 1.0),
    ("flip_-0.5", strip_dirs, -0.5),
    ("flip_-1.0", strip_dirs, -1.0),
]

results = {
    "experiment": "R4: Token Efficiency",
    "model": MODEL_NAME,
    "max_tokens": MAX_TOKENS,
    "conditions": {}
}

for cond_name, dirs, scale in conditions:
    print(f"\n{'='*70}")
    print(f"Condition: {cond_name} [{time.time()-t0:.0f}s]")
    print("="*70)

    cond_results = []
    for prob in test_problems:
        r = generate_with_tracking(prob["prompt"], prob["answer"], dirs, scale)
        fat = r["first_appearance_token"]
        status = f"FAT={fat}" if fat else "NEVER"
        print(f"  [{prob['lang'].upper()}] {prob['prompt'][:40]:40s} → {status:>10s} ({r['total_tokens']} tok)")
        cond_results.append({
            "prompt": prob["prompt"][:50],
            "answer": prob["answer"],
            "lang": prob["lang"],
            **r
        })

    results["conditions"][cond_name] = cond_results

# =============================================================================
# Analysis
# =============================================================================
print(f"\n{'='*70}")
print(f"R4 ANALYSIS: TOKEN EFFICIENCY [{time.time()-t0:.0f}s]")
print("="*70)

for cond_name in results["conditions"]:
    items = results["conditions"][cond_name]
    found = [x for x in items if x["found"]]
    not_found = [x for x in items if not x["found"]]
    fats = [x["first_appearance_token"] for x in found]

    en_found = [x for x in found if x["lang"] == "en"]
    zh_found = [x for x in found if x["lang"] == "zh"]
    en_fats = [x["first_appearance_token"] for x in en_found]
    zh_fats = [x["first_appearance_token"] for x in zh_found]

    print(f"\n  {cond_name}:")
    print(f"    Found: {len(found)}/20, Not found: {len(not_found)}/20")
    if fats:
        print(f"    FAT (all):  mean={np.mean(fats):.1f}, median={np.median(fats):.1f}, min={min(fats)}, max={max(fats)}")
    if en_fats:
        print(f"    FAT (EN):   mean={np.mean(en_fats):.1f}, median={np.median(en_fats):.1f} ({len(en_found)}/10 found)")
    if zh_fats:
        print(f"    FAT (ZH):   mean={np.mean(zh_fats):.1f}, median={np.median(zh_fats):.1f} ({len(zh_found)}/10 found)")

# Direct comparison for problems solved by both baseline and flip
print(f"\n  --- Head-to-head (problems solved by BOTH conditions) ---")
baseline_items = {x["prompt"]: x for x in results["conditions"]["baseline"]}
for cond_name in ["flip_-0.5", "flip_-1.0"]:
    flip_items = {x["prompt"]: x for x in results["conditions"][cond_name]}
    both_solved = []
    for prompt in baseline_items:
        b = baseline_items[prompt]
        f = flip_items[prompt]
        if b["found"] and f["found"]:
            both_solved.append({
                "prompt": prompt,
                "lang": b["lang"],
                "baseline_fat": b["first_appearance_token"],
                "flip_fat": f["first_appearance_token"],
                "speedup": b["first_appearance_token"] / f["first_appearance_token"]
            })

    if both_solved:
        speedups = [x["speedup"] for x in both_solved]
        en_speedups = [x["speedup"] for x in both_solved if x["lang"] == "en"]
        zh_speedups = [x["speedup"] for x in both_solved if x["lang"] == "zh"]
        print(f"\n  {cond_name} vs baseline ({len(both_solved)} problems solved by both):")
        print(f"    Mean speedup: {np.mean(speedups):.2f}x")
        if en_speedups:
            print(f"    EN speedup:   {np.mean(en_speedups):.2f}x ({len(en_speedups)} problems)")
        if zh_speedups:
            print(f"    ZH speedup:   {np.mean(zh_speedups):.2f}x ({len(zh_speedups)} problems)")

        print(f"\n    Per-problem:")
        for x in sorted(both_solved, key=lambda x: x["speedup"], reverse=True):
            print(f"      [{x['lang'].upper()}] {x['prompt'][:35]:35s}  base={x['baseline_fat']:>4d}  flip={x['flip_fat']:>4d}  {x['speedup']:.1f}x")

with open("output/expR4_token_efficiency.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expR4_token_efficiency.json")
