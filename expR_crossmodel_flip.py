"""Experiment R: Cross-Model Replication of Language Direction Flip

Does the P3 finding (flipping causal language direction in MLP deltas improves math)
generalize to Qwen2.5-1.5B (28 layers, d=1536)?

The 3B finding:
- L9-L26 (layers 9-26 of 36) = 25%-72% depth
- Flipping at scale=-1.5 improved math from 2/10 EN to 7/10 EN (3.5x)
- Overall: 7/20 → 13/20

For 1.5B (28 layers):
- Equivalent depth range: L7-L20 (25%-71% depth)
- We'll also test L5-L22 (broader) and L9-L18 (narrower) to find the sweet spot

Test: same 10 EN + 10 ZH problems as P3.
Scales: -2.0, -1.5, -1.0, -0.5, 0.0, 1.0 (baseline)
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'Qwen/Qwen2.5-1.5B'
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

n_layers = model.config.num_hidden_layers  # 28
d = model.config.hidden_size  # 1536

MAX_NEW_TOKENS = 128
N_TRAIN = 200

# Layer ranges to test (proportional to 3B's L9-L26 out of 36)
LAYER_RANGES = {
    "proportional": list(range(7, 21)),   # L7-L20: 25%-71% of 28
    "broad": list(range(5, 23)),          # L5-L22: wider sweep
    "narrow": list(range(9, 19)),         # L9-L18: tighter core
}

print(f"Model: {MODEL_NAME}")
print(f"Layers: {n_layers}, Hidden: {d}")
print(f"Layer ranges: {', '.join(f'{k}: L{v[0]}-L{v[-1]}' for k, v in LAYER_RANGES.items())}")


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
# Phase 1: Compute causal language directions per layer
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 1: Computing causal language directions")
print("=" * 70)

problems = generate_pca_problems(N_TRAIN, seed=42)

# Sample layers for direction fitting (spread across model depth)
all_target_layers = sorted(set().union(*LAYER_RANGES.values()))
fit_layers = sorted(set([all_target_layers[0], all_target_layers[len(all_target_layers)//4],
                         all_target_layers[len(all_target_layers)//2],
                         all_target_layers[3*len(all_target_layers)//4],
                         all_target_layers[-1]]))
print(f"Fitting directions at layers: {fit_layers}")

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
    norm = np.linalg.norm(diff)
    cohen_d = norm / np.sqrt(0.5 * (zh_deltas.var(axis=0).sum() + en_deltas.var(axis=0).sum()))
    lang_dirs[fl] = torch.tensor(diff / norm, dtype=torch.float32, device=device)
    print(f"  Layer {fl}: ||diff||={norm:.1f}, Cohen's d={cohen_d:.2f}")

def get_lang_dir(li):
    return lang_dirs[min(fit_layers, key=lambda x: abs(x - li))]

# Random direction control
torch.manual_seed(42)
random_dir = torch.randn(d, device=device, dtype=torch.float32)
random_dir = random_dir / random_dir.norm()


# =============================================================================
# Test problems (identical to P3)
# =============================================================================
en_test = [
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


def run_generation(prompt, strip_layers, dirs_dict=None, scale=1.0):
    input_ids = tokenizer.encode(prompt)
    handles = []

    if dirs_dict and scale != 1.0:
        for li in strip_layers:
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
                    make_hook(li, get_lang_dir(li), scale)
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


def eval_test(test_problems, strip_layers, dirs_dict, scale, label=""):
    n_correct = 0
    texts = []
    for prob in test_problems:
        text = run_generation(prob["prompt"], strip_layers, dirs_dict, scale)
        correct = prob["answer"] in text
        n_correct += correct
        texts.append({"prompt": prob["prompt"][:40], "answer": prob["answer"],
                       "correct": correct, "output": text[:120]})
    return n_correct, texts


# =============================================================================
# Phase 2: Baseline (no intervention)
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 2: BASELINE (no intervention)")
print("=" * 70)

en_baseline, en_base_texts = eval_test(en_test, [], None, 1.0, "EN baseline")
zh_baseline, zh_base_texts = eval_test(zh_test, [], None, 1.0, "ZH baseline")
print(f"  EN baseline: {en_baseline}/10")
print(f"  ZH baseline: {zh_baseline}/10")
print(f"  Total baseline: {en_baseline + zh_baseline}/20")

results = {
    "experiment": "R: Cross-Model Flip Replication",
    "model": MODEL_NAME,
    "n_layers": n_layers,
    "hidden_size": d,
    "baseline": {"en": en_baseline, "zh": zh_baseline, "total": en_baseline + zh_baseline},
    "baseline_texts": {"en": en_base_texts, "zh": zh_base_texts},
    "sweeps": {}
}

# =============================================================================
# Phase 3: Scale sweep per layer range
# =============================================================================
SCALES = [-2.0, -1.5, -1.0, -0.5, 0.0]

for range_name, strip_layers in LAYER_RANGES.items():
    print(f"\n{'─' * 70}")
    print(f"SWEEP: {range_name} (L{strip_layers[0]}-L{strip_layers[-1]})")
    print("─" * 70)

    dirs_dict = {li: get_lang_dir(li) for li in strip_layers}
    sweep_results = []

    for sc in SCALES:
        en_c, en_texts = eval_test(en_test, strip_layers, dirs_dict, sc)
        zh_c, zh_texts = eval_test(zh_test, strip_layers, dirs_dict, sc)
        total = en_c + zh_c
        print(f"  scale={sc:+5.1f}: EN={en_c}/10, ZH={zh_c}/10, Total={total}/20")
        sweep_results.append({
            "scale": sc, "en": en_c, "zh": zh_c, "total": total,
            "en_texts": en_texts, "zh_texts": zh_texts
        })

    results["sweeps"][range_name] = {
        "layers": f"L{strip_layers[0]}-L{strip_layers[-1]}",
        "n_layers_modified": len(strip_layers),
        "results": sweep_results
    }

# =============================================================================
# Phase 4: Random direction control at best scale
# =============================================================================
print(f"\n{'─' * 70}")
print("RANDOM DIRECTION CONTROL (proportional range)")
print("─" * 70)

prop_layers = LAYER_RANGES["proportional"]
rand_dirs = {li: random_dir for li in prop_layers}
for sc in [-1.5, -1.0]:
    en_c, _ = eval_test(en_test, prop_layers, rand_dirs, sc)
    zh_c, _ = eval_test(zh_test, prop_layers, rand_dirs, sc)
    print(f"  random scale={sc:+5.1f}: EN={en_c}/10, ZH={zh_c}/10, Total={en_c+zh_c}/20")
    results.setdefault("random_controls", []).append({
        "scale": sc, "en": en_c, "zh": zh_c, "total": en_c + zh_c
    })

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'=' * 70}")
print("EXPERIMENT R: CROSS-MODEL REPLICATION SUMMARY")
print(f"{'=' * 70}")
print(f"Model: {MODEL_NAME} ({n_layers} layers, d={d})")
print(f"Baseline: EN={en_baseline}/10, ZH={zh_baseline}/10, Total={en_baseline+zh_baseline}/20")
print()

for range_name, sweep_data in results["sweeps"].items():
    best = max(sweep_data["results"], key=lambda x: x["total"])
    print(f"  {range_name} ({sweep_data['layers']}): best scale={best['scale']:+.1f} → {best['total']}/20")

# Compare with 3B
print(f"\n  3B reference: baseline 7/20, best flip 13/20 (scale=-1.5, L9-L26)")
best_overall = max(
    [(r["total"], r["scale"], rn) for rn, sd in results["sweeps"].items() for r in sd["results"]],
    key=lambda x: x[0]
)
print(f"  1.5B best: {best_overall[0]}/20 (scale={best_overall[1]:+.1f}, {best_overall[2]})")

if best_overall[0] > en_baseline + zh_baseline:
    improvement = (best_overall[0] - (en_baseline + zh_baseline)) / (en_baseline + zh_baseline) * 100
    print(f"  → REPLICATES: {improvement:.0f}% improvement over baseline")
else:
    print(f"  → DOES NOT REPLICATE: no improvement over baseline")

with open("output/expR_crossmodel_flip.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expR_crossmodel_flip.json")
