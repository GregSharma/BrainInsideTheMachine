"""
Exp AR Part 2: 7B-only (4-bit quantized)
Separate process to avoid OOM from residual 1.5B memory.
"""

import json, sys, re, time
import random as pyrandom
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
OUTPUT_DIR = Path("output")
MAX_NEW_TOKENS = 128
BATCH_SIZE = 8  # smaller for 7B


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


# ── Test set ───────────────────────────────────────────────────────────
all_problems = generate_pca_problems(200, seed=42)
per_cat = 40
TEST_INDICES = []
for cat in range(5):
    for i in range(4):
        TEST_INDICES.append(cat * per_cat + i)

PROBLEMS = []
for idx in TEST_INDICES:
    p = all_problems[idx]
    answer = compute_answer(p["en"]) or compute_answer(p["zh"])
    PROBLEMS.append({"zh": p["zh"], "en": p["en"], "answer": answer, "idx": idx})

print(f"Test set: {len(PROBLEMS)} problems")

# ── Load 7B 4-bit ──────────────────────────────────────────────────────
t0 = time.time()
device = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading {MODEL_NAME} (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16,
    device_map=device, trust_remote_code=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"  Layers: {n_layers}, d_model: {d_model}")

# Analysis layers: middle 50%
start_layer = n_layers // 4
end_layer = 3 * n_layers // 4
analysis_layers = list(range(start_layer, end_layer + 1))
print(f"  Analysis layers: L{start_layer}-L{end_layer}")

results = {
    "model": MODEL_NAME,
    "n_layers": n_layers,
    "d_model": d_model,
    "analysis_layers": [start_layer, end_layer],
    "quantized": False,
}

# ── Part 1: Baselines ──────────────────────────────────────────────────
print("\n--- Baselines ---")
for lang in ["zh", "en"]:
    correct = 0
    details = []
    for pi, prob in enumerate(PROBLEMS):
        prompt = prob[lang]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        is_correct = prob["answer"] in gen
        if is_correct:
            correct += 1
        details.append({"correct": is_correct, "gen": gen[:200]})
    results[f"baseline_{lang}"] = {"correct": correct, "total": 20, "details": details}
    print(f"  {lang.upper()}: {correct}/20")

# ── Part 2: Logit Averaging ────────────────────────────────────────────
print("\n--- Logit Averaging ---")
correct = 0
details = []
for pi, prob in enumerate(PROBLEMS):
    zh_ids = tokenizer(prob["zh"], return_tensors="pt").to(device)["input_ids"]
    en_ids = tokenizer(prob["en"], return_tensors="pt").to(device)["input_ids"]
    generated_ids = []

    for step in range(MAX_NEW_TOKENS):
        gen_tensor = torch.tensor(generated_ids, device=device).unsqueeze(0) if generated_ids else None
        if gen_tensor is not None:
            zh_input = torch.cat([zh_ids, gen_tensor], dim=1)
            en_input = torch.cat([en_ids, gen_tensor], dim=1)
        else:
            zh_input = zh_ids
            en_input = en_ids

        with torch.no_grad():
            out_zh = model(input_ids=zh_input)
            out_en = model(input_ids=en_input)

        avg_logits = (out_zh.logits[0, -1, :].float() + out_en.logits[0, -1, :].float()) / 2
        next_token = avg_logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token)

    gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
    is_correct = prob["answer"] in gen
    if is_correct:
        correct += 1
    details.append({"correct": is_correct, "gen": gen[:200], "n_steps": len(generated_ids)})
    mark = "Y" if is_correct else "N"
    print(f"  P{pi}: {mark} — {gen[:50]}...")

results["logit_avg"] = {"correct": correct, "total": 20, "details": details}
print(f"  LOGIT AVG: {correct}/20")

# ── Part 3: Language Flip ──────────────────────────────────────────────
print("\n--- Fitting language direction ---")
train_problems = [all_problems[i] for i in range(len(all_problems)) if i not in TEST_INDICES]

layer_data = {li: {"zh": [], "en": []} for li in analysis_layers}
captures = {}
handles = []

for li in analysis_layers:
    def make_hook(idx):
        def hook(module, inp, out):
            captures[idx] = out.detach().float()
        return hook
    handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(li)))

for lang in ["zh", "en"]:
    prompts = [p[lang] for p in train_problems[:100]]
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        captures.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            model(**inputs)
        for li in analysis_layers:
            for j in range(captures[li].shape[0]):
                layer_data[li][lang].append(captures[li][j, last_idx[j]].cpu().numpy())

for h in handles:
    h.remove()

lang_dirs = {}
for li in analysis_layers:
    zh_mean = np.mean(layer_data[li]["zh"], axis=0)
    en_mean = np.mean(layer_data[li]["en"], axis=0)
    diff = zh_mean - en_mean
    lang_dirs[li] = diff / (np.linalg.norm(diff) + 1e-10)

print("--- Language Flip ---")
for scale in [-0.5, -1.0, -1.5]:
    correct = 0
    details = []

    for pi, prob in enumerate(PROBLEMS):
        hooks = []
        for li in analysis_layers:
            direction = torch.tensor(lang_dirs[li], device=device, dtype=torch.float32)

            def make_flip_hook(d, s):
                def hook(module, inp, out):
                    o = out.float()
                    proj = (o @ d).unsqueeze(-1) * d
                    return (o - 2 * s * proj).to(out.dtype)
                return hook
            hooks.append(model.model.layers[li].mlp.register_forward_hook(
                make_flip_hook(direction, scale)
            ))

        inputs = tokenizer(prob["en"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        for h in hooks:
            h.remove()

        is_correct = prob["answer"] in gen
        if is_correct:
            correct += 1
        details.append({"correct": is_correct, "gen": gen[:200]})

    results[f"flip_scale_{scale}"] = {"correct": correct, "total": 20, "details": details}
    print(f"  FLIP scale={scale}: {correct}/20")

# ── Save ────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
results["elapsed_s"] = elapsed

with open(OUTPUT_DIR / "expAR_3b_instruct_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"3B-INSTRUCT SUMMARY ({elapsed:.0f}s)")
print(f"{'='*70}")
print(f"  baseline_zh : {results['baseline_zh']['correct']}/20")
print(f"  baseline_en : {results['baseline_en']['correct']}/20")
print(f"  logit_avg   : {results['logit_avg']['correct']}/20")
for scale in [-0.5, -1.0, -1.5]:
    k = f"flip_scale_{scale}"
    print(f"  flip({scale:+.1f})  : {results[k]['correct']}/20")
print("Saved to output/expAR_3b_instruct_results.json")
