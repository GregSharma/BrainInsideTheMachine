"""Experiment P3: Language Direction Flip Scale Sweep

P2 showed that flipping the causal language direction in MLP deltas at L9-L26
IMPROVED math accuracy from 7/20 to 11/20. This was unexpected — the language
component is partially anti-correlated with mathematical efficiency.

Sweep: what scale of flip maximizes math accuracy?
Also: is the improvement from DIRECTION change or MAGNITUDE change?

Test:
- flip scale from -2.0 to +3.0 (negative = flip and amplify, positive = amplify same direction)
- Random direction control: project onto a random unit vector instead
- Per-language breakdown

10 problems × EN only (since EN baseline is weakest at 2/10). 128 tokens.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
STRIP_LAYERS = list(range(9, 27))
N_TRAIN = 200
d = model.config.hidden_size


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


# Compute causal language directions (same as P2)
print("Computing causal language directions...")
problems = generate_pca_problems(N_TRAIN, seed=42)
fit_layers = [10, 14, 18, 22, 26]
lang_dirs = {}

for fl in fit_layers:
    mlp_out = {}
    def mlp_cap(module, input, output):
        mlp_out['d'] = output.detach()[:, -1, :]
    handle = model.model.layers[fl].mlp.register_forward_hook(mlp_cap)

    zh_deltas = np.zeros((N_TRAIN, d), dtype=np.float32)
    en_deltas = np.zeros((N_TRAIN, d), dtype=np.float32)
    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
        zh_deltas[i] = mlp_out['d'].cpu().float().numpy()
        mlp_out.clear()
    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
        en_deltas[i] = mlp_out['d'].cpu().float().numpy()
        mlp_out.clear()
    handle.remove()

    diff = zh_deltas.mean(axis=0) - en_deltas.mean(axis=0)
    lang_dirs[fl] = torch.tensor(diff / np.linalg.norm(diff), dtype=torch.float32, device=device)

def get_lang_dir(li):
    return lang_dirs[min(fit_layers, key=lambda x: abs(x - li))]

strip_dirs = {li: get_lang_dir(li) for li in STRIP_LAYERS}

# Random direction control
torch.manual_seed(42)
random_dir = torch.randn(d, device=device, dtype=torch.float32)
random_dir = random_dir / random_dir.norm()
random_dirs = {li: random_dir for li in STRIP_LAYERS}

# Test problems
test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60"},
    {"prompt": "What is the remainder when 100 is divided by 7?", "answer": "2"},
    {"prompt": "Calculate 15 × 8.", "answer": "120"},
    {"prompt": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.", "answer": "40"},
    {"prompt": "Calculate 387 × 29.", "answer": "11223"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120"},
    {"prompt": "What is the remainder when 7654 is divided by 37?", "answer": "34"},
    {"prompt": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.", "answer": "3475"},
    {"prompt": "A rectangle has length 47 and width 33. Find its area.", "answer": "1551"},
]


def run_generation(prompt, dirs_dict=None, scale=1.0):
    """scale=1.0 means no change. scale=-1.0 means flip. scale=0.0 means zero out."""
    input_ids = tokenizer.encode(prompt)
    handles = []

    if dirs_dict and scale != 1.0:
        for li in STRIP_LAYERS:
            def make_hook(layer_idx, lang_dir, sc):
                def hook_fn(module, input, output):
                    delta = output.float()
                    proj = torch.sum(delta * lang_dir, dim=-1, keepdim=True)
                    lang_component = proj * lang_dir
                    # Replace: remove original, add scaled version
                    modified = delta - lang_component + sc * lang_component
                    return modified.to(output.dtype)
                return hook_fn
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_hook(li, dirs_dict[li], scale)
                )
            )

    try:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]
        for _ in range(MAX_NEW_TOKENS - 1):
            with torch.no_grad():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)
            if next_id == tokenizer.eos_token_id:
                break
    finally:
        for h in handles:
            h.remove()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# =============================================================================
# Sweep: ENGLISH ONLY
# =============================================================================
print(f"\n{'='*70}")
print("P3: LANGUAGE DIRECTION FLIP SCALE SWEEP (EN)")
print("=" * 70)

# scale < 0 means flip, scale > 1 means amplify same direction
# scale = -1 is a pure flip (P2's "flip" mode)
SCALES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
results = {"experiment": "P3: Flip Scale Sweep", "sweeps": []}

for sc in SCALES:
    n_correct = 0
    texts = []
    for prob in test_problems:
        text = run_generation(prob["prompt"], dirs_dict=strip_dirs, scale=sc)
        correct = prob["answer"] in text
        n_correct += correct
        texts.append(text[:80])
    print(f"  scale={sc:+5.1f}: {n_correct}/10 correct")
    results["sweeps"].append({"scale": sc, "correct": n_correct, "texts": texts})

# Random direction control at the best-performing scale
print(f"\n  --- Random direction controls ---")
for sc in [-1.0, 0.0]:
    n_correct = 0
    for prob in test_problems:
        text = run_generation(prob["prompt"], dirs_dict=random_dirs, scale=sc)
        correct = prob["answer"] in text
        n_correct += correct
    print(f"  random_dir scale={sc:+5.1f}: {n_correct}/10 correct")
    results["sweeps"].append({"scale": sc, "correct": n_correct, "random": True})

# =============================================================================
# Now do the SAME SWEEP for CHINESE
# =============================================================================
print(f"\n{'─'*70}")
print("P3: CHINESE SWEEP")
print("─" * 70)

zh_test = [
    {"prompt": "计算 47 + 86 的值。", "answer": "133"},
    {"prompt": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60"},
    {"prompt": "100 除以 7 的余数是多少？", "answer": "2"},
    {"prompt": "计算 15 × 8 的值。", "answer": "120"},
    {"prompt": "等差数列首项为 2，公差为 3，求前 5 项之和。", "answer": "40"},
    {"prompt": "计算 387 × 29 的值。", "answer": "11223"},
    {"prompt": "求组合数 C(10, 3) 的值。", "answer": "120"},
    {"prompt": "7654 除以 37 的余数是多少？", "answer": "34"},
    {"prompt": "等差数列首项为 7，公差为 11，求前 25 项之和。", "answer": "3475"},
    {"prompt": "一个长方形的长为 47，宽为 33，求其面积。", "answer": "1551"},
]

zh_results = []
for sc in SCALES:
    n_correct = 0
    for prob in zh_test:
        text = run_generation(prob["prompt"], dirs_dict=strip_dirs, scale=sc)
        correct = prob["answer"] in text
        n_correct += correct
    print(f"  scale={sc:+5.1f}: {n_correct}/10 correct")
    zh_results.append({"scale": sc, "correct": n_correct})

results["zh_sweeps"] = zh_results

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print("P3 SUMMARY")
print("=" * 70)

print(f"  {'Scale':>7s} {'EN':>5s} {'ZH':>5s} {'Total':>6s}")
print(f"  {'─'*7} {'─'*5} {'─'*5} {'─'*6}")
for i, sc in enumerate(SCALES):
    en_c = results["sweeps"][i]["correct"]
    zh_c = zh_results[i]["correct"] if i < len(zh_results) else "?"
    total = en_c + (zh_c if isinstance(zh_c, int) else 0)
    print(f"  {sc:+7.1f} {en_c:>5d} {zh_c:>5} {total:>6d}")

# Find best scale
best_en = max(results["sweeps"][:len(SCALES)], key=lambda x: x["correct"])
best_zh = max(zh_results, key=lambda x: x["correct"])
print(f"\n  Best EN scale: {best_en['scale']:+.1f} ({best_en['correct']}/10)")
print(f"  Best ZH scale: {best_zh['scale']:+.1f} ({best_zh['correct']}/10)")

with open("output/expP3_flip_sweep.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expP3_flip_sweep.json")
