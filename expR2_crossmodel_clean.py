"""Experiment R2: Clean Cross-Model Replication — FAST batched version

Tests language-direction flip on Qwen2.5-1.5B with:
- Batched direction fitting (32 at a time instead of 1-by-1)
- Token budget comparison (128 vs 256)
- 5 random direction controls for statistical rigor
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_NAME = 'Qwen/Qwen2.5-1.5B'
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
BATCH_SIZE = 32
STRIP_LAYERS = list(range(7, 21))

print(f"Model: {MODEL_NAME} ({n_layers} layers, d={d})")
print(f"Intervention: L{STRIP_LAYERS[0]}-L{STRIP_LAYERS[-1]}")
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
# BATCHED direction fitting
# =============================================================================
print(f"\n{'='*70}")
print("PHASE 1: Batched direction fitting")
print("="*70)

problems = generate_pca_problems(N_TRAIN, seed=42)
fit_layers = [7, 10, 14, 18, 20]

lang_dirs = {}
for fl in fit_layers:
    mlp_out_list = []

    def mlp_cap(module, input, output):
        # Capture last-token MLP delta for each item in batch
        # output shape: (batch, seq, d) — we want last non-pad token
        mlp_out_list.append(output.detach().float())

    handle = model.model.layers[fl].mlp.register_forward_hook(mlp_cap)

    # Collect ZH deltas in batches
    zh_deltas = []
    zh_texts = [p["zh"] for p in problems]
    for i in range(0, N_TRAIN, BATCH_SIZE):
        batch = zh_texts[i:i+BATCH_SIZE]
        mlp_out_list.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            model(**inputs)
        # Extract last real token per sequence (not padding)
        attn_mask = inputs["attention_mask"]  # (B, seq)
        last_idx = attn_mask.sum(dim=1) - 1  # (B,)
        out = mlp_out_list[0]  # (B, seq, d)
        for j in range(out.shape[0]):
            zh_deltas.append(out[j, last_idx[j]].cpu().numpy())

    # Collect EN deltas in batches
    en_deltas = []
    en_texts = [p["en"] for p in problems]
    for i in range(0, N_TRAIN, BATCH_SIZE):
        batch = en_texts[i:i+BATCH_SIZE]
        mlp_out_list.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            model(**inputs)
        attn_mask = inputs["attention_mask"]
        last_idx = attn_mask.sum(dim=1) - 1
        out = mlp_out_list[0]
        for j in range(out.shape[0]):
            en_deltas.append(out[j, last_idx[j]].cpu().numpy())

    handle.remove()

    zh_arr = np.stack(zh_deltas)
    en_arr = np.stack(en_deltas)
    diff = zh_arr.mean(axis=0) - en_arr.mean(axis=0)
    norm = np.linalg.norm(diff)
    lang_dirs[fl] = torch.tensor(diff / norm, dtype=torch.float32, device=device)
    print(f"  Layer {fl}: ||diff||={norm:.1f} [{time.time()-t0:.1f}s]")

def get_lang_dir(li):
    return lang_dirs[min(fit_layers, key=lambda x: abs(x - li))]

strip_dirs = {li: get_lang_dir(li) for li in STRIP_LAYERS}

# Random directions
torch.manual_seed(42)
random_dirs_list = []
for seed in range(5):
    torch.manual_seed(seed + 100)
    rd = torch.randn(d, device=device, dtype=torch.float32)
    random_dirs_list.append(rd / rd.norm())

print(f"Direction fitting done in {time.time()-t0:.1f}s")

# =============================================================================
# Test problems
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


def run_generation(prompt, dirs_dict=None, scale=1.0, max_tokens=256):
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
            ld = dirs_dict[li] if li in dirs_dict else get_lang_dir(li)
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_hook(li, ld, scale)
                )
            )

    try:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]
        for _ in range(max_tokens - 1):
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


def eval_test(test_problems, dirs_dict, scale, max_tokens=256):
    n_correct = 0
    details = []
    for prob in test_problems:
        text = run_generation(prob["prompt"], dirs_dict, scale, max_tokens)
        correct = prob["answer"] in text
        n_correct += correct
        details.append({"correct": correct, "output_len": len(text),
                         "output": text[:200], "answer": prob["answer"]})
    return n_correct, details


# =============================================================================
# Run sweep
# =============================================================================
results = {
    "experiment": "R2: Clean Cross-Model Replication (batched)",
    "model": MODEL_NAME,
    "n_layers": n_layers, "hidden_size": d,
    "strip_layers": f"L{STRIP_LAYERS[0]}-L{STRIP_LAYERS[-1]}",
    "conditions": []
}

TOKEN_BUDGETS = [128, 256]
SCALES = [-2.0, -1.5, -1.0, -0.5, 0.0]

for max_tok in TOKEN_BUDGETS:
    print(f"\n{'='*70}")
    print(f"TOKEN BUDGET: {max_tok}  [{time.time()-t0:.0f}s elapsed]")
    print("="*70)

    en_c, en_det = eval_test(en_test, None, 1.0, max_tok)
    zh_c, zh_det = eval_test(zh_test, None, 1.0, max_tok)
    en_avg = np.mean([x["output_len"] for x in en_det])
    zh_avg = np.mean([x["output_len"] for x in zh_det])
    print(f"  Baseline: EN={en_c}/10, ZH={zh_c}/10 (avg len: EN={en_avg:.0f}, ZH={zh_avg:.0f})")

    cond = {
        "max_tokens": max_tok,
        "baseline": {"en": en_c, "zh": zh_c, "total": en_c + zh_c,
                      "en_avg_len": en_avg, "zh_avg_len": zh_avg},
        "flip_sweep": [], "random_controls": []
    }

    for sc in SCALES:
        en_c2, en_d2 = eval_test(en_test, strip_dirs, sc, max_tok)
        zh_c2, zh_d2 = eval_test(zh_test, strip_dirs, sc, max_tok)
        en_avg2 = np.mean([x["output_len"] for x in en_d2])
        print(f"  Flip {sc:+5.1f}: EN={en_c2}/10, ZH={zh_c2}/10, Tot={en_c2+zh_c2}/20 (EN len={en_avg2:.0f}) [{time.time()-t0:.0f}s]")
        cond["flip_sweep"].append({
            "scale": sc, "en": en_c2, "zh": zh_c2, "total": en_c2 + zh_c2,
            "en_avg_len": en_avg2,
            "en_texts": en_d2, "zh_texts": zh_d2
        })

    # Random controls — 5 random dirs at scale -1.5 and -1.0
    print(f"  Random controls (5 dirs):")
    for sc in [-1.5, -1.0]:
        scores = []
        for rd in random_dirs_list:
            rd_dict = {li: rd for li in STRIP_LAYERS}
            en_r, _ = eval_test(en_test, rd_dict, sc, max_tok)
            zh_r, _ = eval_test(zh_test, rd_dict, sc, max_tok)
            scores.append(en_r + zh_r)
        avg_r = np.mean(scores)
        std_r = np.std(scores)
        print(f"    rand {sc:+5.1f}: {avg_r:.1f}±{std_r:.1f} ({scores}) [{time.time()-t0:.0f}s]")
        cond["random_controls"].append({"scale": sc, "scores": scores, "mean": float(avg_r), "std": float(std_r)})

    results["conditions"].append(cond)

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print(f"R2 SUMMARY [{time.time()-t0:.0f}s total]")
print("="*70)

for cond in results["conditions"]:
    tok = cond["max_tokens"]
    bl = cond["baseline"]["total"]
    best_flip = max(cond["flip_sweep"], key=lambda x: x["total"])
    best_rand = max(cond["random_controls"], key=lambda x: x["mean"])
    print(f"\n  {tok} tokens:")
    print(f"    Baseline:    {bl}/20 (EN avg len: {cond['baseline']['en_avg_len']:.0f})")
    print(f"    Best flip:   {best_flip['total']}/20 (scale={best_flip['scale']:+.1f}, EN len={best_flip['en_avg_len']:.0f})")
    print(f"    Best random: {best_rand['mean']:.1f}/20 ± {best_rand['std']:.1f}")
    if best_flip["total"] > best_rand["mean"] + best_rand["std"]:
        print(f"    → LANGUAGE-SPECIFIC effect")
    elif best_flip["total"] > bl:
        print(f"    → GENERAL PERTURBATION (not language-specific)")
    else:
        print(f"    → NO EFFECT")

print(f"\n  3B ref: baseline=7/20, flip -1.5=13/20, random=2/10 EN")

with open("output/expR2_crossmodel_clean.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expR2_crossmodel_clean.json")
