"""Experiment R3: 3B Token Budget Confound Test

R2 showed that on 1.5B, the language-direction flip is an EFFICIENCY intervention:
baseline catches up at 256 tokens. Does the same apply to 3B?

Original P3: 128 tokens, baseline 7/20, flip -1.5 = 13/20 (86% improvement)
Question: at 256 and 512 tokens, does baseline catch up?

If yes: flip is efficiency (shortens path to answer)
If no: flip genuinely improves mathematical reasoning

Batched direction fitting from R2 approach.
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

n_layers = model.config.num_hidden_layers  # 36
d = model.config.hidden_size  # 2048
N_TRAIN = 200
BATCH_SIZE = 16  # smaller batch for 3B (more VRAM)
STRIP_LAYERS = list(range(9, 27))

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
# Batched direction fitting
# =============================================================================
print(f"\nBatched direction fitting...")
problems = generate_pca_problems(N_TRAIN, seed=42)
fit_layers = [10, 14, 18, 22, 26]

lang_dirs = {}
for fl in fit_layers:
    mlp_out_list = []
    def mlp_cap(module, input, output):
        mlp_out_list.append(output.detach().float())
    handle = model.model.layers[fl].mlp.register_forward_hook(mlp_cap)

    zh_deltas = []
    for i in range(0, N_TRAIN, BATCH_SIZE):
        batch = [p["zh"] for p in problems[i:i+BATCH_SIZE]]
        mlp_out_list.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad(): model(**inputs)
        attn_mask = inputs["attention_mask"]
        last_idx = attn_mask.sum(dim=1) - 1
        out = mlp_out_list[0]
        for j in range(out.shape[0]):
            zh_deltas.append(out[j, last_idx[j]].cpu().numpy())

    en_deltas = []
    for i in range(0, N_TRAIN, BATCH_SIZE):
        batch = [p["en"] for p in problems[i:i+BATCH_SIZE]]
        mlp_out_list.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad(): model(**inputs)
        attn_mask = inputs["attention_mask"]
        last_idx = attn_mask.sum(dim=1) - 1
        out = mlp_out_list[0]
        for j in range(out.shape[0]):
            en_deltas.append(out[j, last_idx[j]].cpu().numpy())

    handle.remove()
    diff = np.stack(zh_deltas).mean(0) - np.stack(en_deltas).mean(0)
    norm = np.linalg.norm(diff)
    lang_dirs[fl] = torch.tensor(diff / norm, dtype=torch.float32, device=device)
    print(f"  Layer {fl}: ||diff||={norm:.1f}")

def get_lang_dir(li):
    return lang_dirs[min(fit_layers, key=lambda x: abs(x - li))]

strip_dirs = {li: get_lang_dir(li) for li in STRIP_LAYERS}
print(f"Direction fitting: {time.time()-t0:.1f}s")

# =============================================================================
# Test problems (same as P3)
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


def run_generation(prompt, dirs_dict=None, scale=1.0, max_tokens=128):
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


def eval_test(test_problems, dirs_dict, scale, max_tokens):
    n_correct = 0
    details = []
    for prob in test_problems:
        text = run_generation(prob["prompt"], dirs_dict, scale, max_tokens)
        correct = prob["answer"] in text
        n_correct += correct
        details.append({"correct": correct, "output_len": len(text),
                         "output": text[:250], "answer": prob["answer"]})
    return n_correct, details


# =============================================================================
# Main sweep: token budget × scale
# =============================================================================
TOKEN_BUDGETS = [128, 256, 512]
SCALES = [-2.0, -1.5, -1.0, -0.5, 0.0]  # 1.0 is baseline (no intervention)

results = {
    "experiment": "R3: 3B Token Budget Confound Test",
    "model": MODEL_NAME,
    "conditions": []
}

for max_tok in TOKEN_BUDGETS:
    print(f"\n{'='*70}")
    print(f"TOKEN BUDGET: {max_tok}  [{time.time()-t0:.0f}s]")
    print("="*70)

    # Baseline
    en_c, en_det = eval_test(en_test, None, 1.0, max_tok)
    zh_c, zh_det = eval_test(zh_test, None, 1.0, max_tok)
    en_avg = np.mean([x["output_len"] for x in en_det])
    zh_avg = np.mean([x["output_len"] for x in zh_det])
    print(f"  Baseline: EN={en_c}/10, ZH={zh_c}/10, Tot={en_c+zh_c}/20 (EN len={en_avg:.0f}, ZH len={zh_avg:.0f})")

    cond = {
        "max_tokens": max_tok,
        "baseline": {"en": en_c, "zh": zh_c, "total": en_c + zh_c,
                      "en_avg_len": float(en_avg), "zh_avg_len": float(zh_avg),
                      "en_details": en_det, "zh_details": zh_det},
        "flip_sweep": []
    }

    # Flip sweep
    for sc in SCALES:
        en_c2, en_d2 = eval_test(en_test, strip_dirs, sc, max_tok)
        zh_c2, zh_d2 = eval_test(zh_test, strip_dirs, sc, max_tok)
        en_avg2 = np.mean([x["output_len"] for x in en_d2])
        zh_avg2 = np.mean([x["output_len"] for x in zh_d2])
        print(f"  Flip {sc:+5.1f}: EN={en_c2}/10, ZH={zh_c2}/10, Tot={en_c2+zh_c2}/20 (EN len={en_avg2:.0f}) [{time.time()-t0:.0f}s]")
        cond["flip_sweep"].append({
            "scale": sc, "en": en_c2, "zh": zh_c2, "total": en_c2 + zh_c2,
            "en_avg_len": float(en_avg2), "zh_avg_len": float(zh_avg2),
            "en_details": en_d2, "zh_details": zh_d2
        })

    results["conditions"].append(cond)

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print(f"R3 SUMMARY: 3B TOKEN BUDGET CONFOUND TEST [{time.time()-t0:.0f}s]")
print("="*70)

print(f"\n  {'Tokens':>7s} {'Baseline':>10s} {'Best Flip':>10s} {'Scale':>7s} {'Delta':>7s}")
print(f"  {'─'*7} {'─'*10} {'─'*10} {'─'*7} {'─'*7}")

for cond in results["conditions"]:
    tok = cond["max_tokens"]
    bl = cond["baseline"]["total"]
    best = max(cond["flip_sweep"], key=lambda x: x["total"])
    delta = best["total"] - bl
    pct = (delta / bl * 100) if bl > 0 else 0
    print(f"  {tok:>7d} {bl:>10d}/20 {best['total']:>10d}/20 {best['scale']:>+7.1f} {delta:>+5d} ({pct:+.0f}%)")

# The key question
b128 = results["conditions"][0]["baseline"]["total"]
f128 = max(results["conditions"][0]["flip_sweep"], key=lambda x: x["total"])["total"]
b256 = results["conditions"][1]["baseline"]["total"]
b512 = results["conditions"][2]["baseline"]["total"]

print(f"\n  128tok: baseline={b128}, flip={f128} → {(f128-b128)/b128*100:.0f}% improvement")
print(f"  256tok: baseline={b256} (vs 128tok baseline {b128})")
print(f"  512tok: baseline={b512}")

if b256 >= f128:
    print(f"\n  → CONFOUND CONFIRMED: 256tok baseline ({b256}) ≥ 128tok flip ({f128})")
    print(f"    The flip is an EFFICIENCY intervention, not an accuracy intervention.")
elif b512 >= f128:
    print(f"\n  → PARTIAL CONFOUND: 512tok baseline ({b512}) ≥ 128tok flip ({f128})")
    print(f"    More tokens needed, but baseline eventually catches up.")
else:
    print(f"\n  → NO CONFOUND: Even 512tok baseline ({b512}) < 128tok flip ({f128})")
    print(f"    The flip genuinely improves mathematical reasoning on 3B!")

with open("output/expR3_3b_token_budget.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expR3_3b_token_budget.json")
