"""
Extract generation-time L32 trajectories for 20 problems × 4 languages.
Uses KV cache for speed (~10x faster than no-cache approach).
Saves full h32 vectors at every generation step.

Output: output/gen_trajectories.npz
  Keys: prob{idx}_{lang} → (n_steps, 2048)
  Plus metadata in output/gen_trajectories_meta.json
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random as pyrandom
import time

# ---------- LOAD MODEL ----------
print("Loading Qwen2.5-3B...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    torch_dtype=torch.float16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model.eval()

N_LAYERS = model.config.num_hidden_layers  # 36
D = model.config.hidden_size  # 2048
REASONING_LAYER = 31  # 0-indexed → layer 32

# ---------- PROBLEMS ----------
def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh = f"计算 {a} + {b} 的值。"
            en = f"Calculate {a} + {b}."
            es = f"Calcula {a} + {b}."
            ja = f"{a} + {b} を計算してください。"
        else:
            zh = f"计算 {a} × {b} 的值。"
            en = f"Calculate {a} × {b}."
            es = f"Calcula {a} × {b}."
            ja = f"{a} × {b} を計算してください。"
        problems.append({"zh": zh, "en": en, "es": es, "ja": ja, "category": 0})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        es = f"Encuentra el valor de C({n_val}, {k_val})."
        ja = f"C({n_val}, {k_val}) の値を求めてください。"
        problems.append({"zh": zh, "en": en, "es": es, "ja": ja, "category": 1})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        es = f"¿Cuál es el residuo cuando {a} se divide por {b}?"
        ja = f"{a} を {b} で割った余りはいくつですか？"
        problems.append({"zh": zh, "en": en, "es": es, "ja": ja, "category": 2})
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        es = f"Un rectángulo tiene largo {w} y ancho {h}. Encuentra su área."
        ja = f"長方形の長さが{w}、幅が{h}のとき、面積を求めてください。"
        problems.append({"zh": zh, "en": en, "es": es, "ja": ja, "category": 3})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        es = f"Una sucesión aritmética tiene primer término {a1} y diferencia común {d}. Encuentra la suma de los primeros {n_terms} términos."
        ja = f"初項{a1}、公差{d}の等差数列の最初の{n_terms}項の和を求めてください。"
        problems.append({"zh": zh, "en": en, "es": es, "ja": ja, "category": 4})
    rng.shuffle(problems)
    return problems

problems = generate_problems(200, seed=42)
CAT_NAMES = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]

# Select 20 problems: 4 per category
selected = []
cat_count = {i: 0 for i in range(5)}
for i, p in enumerate(problems):
    c = p['category']
    if cat_count[c] < 4:
        selected.append(i)
        cat_count[c] += 1
    if len(selected) == 20:
        break

langs = ['zh', 'en', 'es', 'ja']
print(f"Selected {len(selected)} problems across {len(set(problems[i]['category'] for i in selected))} categories")
print(f"Languages: {langs}")
print(f"Total extractions: {len(selected) * len(langs)}")


# ---------- EXTRACT WITH KV CACHE ----------
def extract_trajectory_cached(model, tokenizer, prompt, max_new_tokens=256):
    """Extract h32 at every generation step using KV cache for speed."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]

    hook_data = {}
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hook_data[layer_idx] = h[:, -1, :].detach().cpu().float().numpy()
        return hook_fn

    hook_handle = model.model.layers[REASONING_LAYER].register_forward_hook(
        make_hook(REASONING_LAYER)
    )

    all_h32 = []
    tokens = []
    past_key_values = None

    with torch.no_grad():
        # First pass: process entire prompt
        outputs = model(inputs.input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        all_h32.append(hook_data[REASONING_LAYER].flatten().copy())

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            hook_handle.remove()
            return np.array(all_h32), tokens, ""

        # Subsequent passes: one token at a time with cache
        for step in range(1, max_new_tokens):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            all_h32.append(hook_data[REASONING_LAYER].flatten().copy())

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

    hook_handle.remove()

    # Decode generated text
    gen_ids = torch.tensor([tokens], device="cuda")
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    all_h32 = np.array(all_h32)  # (n_steps, 2048)
    return all_h32, tokens, text


# ---------- MAIN EXTRACTION ----------
trajectories = {}
metadata = {}
total = len(selected) * len(langs)
done = 0
t_start = time.time()

for prob_idx in selected:
    prob = problems[prob_idx]
    cat_name = CAT_NAMES[prob['category']]

    for lang in langs:
        key = f"prob{prob_idx}_{lang}"
        done += 1
        print(f"[{done}/{total}] {key} ({cat_name})...", end=" ", flush=True)

        t0 = time.time()
        h32, tokens, text = extract_trajectory_cached(model, tokenizer, prob[lang])
        dt = time.time() - t0

        trajectories[key] = h32
        metadata[key] = {
            'problem_idx': prob_idx,
            'category': cat_name,
            'category_id': prob['category'],
            'language': lang,
            'prompt': prob[lang],
            'n_steps': h32.shape[0],
            'n_tokens': len(tokens),
            'text_preview': text[:200],
            'time_seconds': round(dt, 1),
        }

        elapsed = time.time() - t_start
        rate = done / elapsed
        remaining = (total - done) / rate
        print(f"{h32.shape[0]} steps, {dt:.1f}s (ETA: {remaining/60:.1f}min)")

# Save
print(f"\nSaving {len(trajectories)} trajectories...")
np.savez_compressed('output/gen_trajectories.npz', **trajectories)

with open('output/gen_trajectories_meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: output/gen_trajectories.npz, output/gen_trajectories_meta.json")
print(f"Total time: {(time.time() - t_start)/60:.1f} minutes")

# Quick summary
print(f"\n{'='*60}")
print(f"EXTRACTION SUMMARY")
print(f"{'='*60}")
for lang in langs:
    steps = [metadata[k]['n_steps'] for k in metadata if metadata[k]['language'] == lang]
    print(f"  {lang}: {len(steps)} problems, mean steps={np.mean(steps):.0f}, "
          f"min={min(steps)}, max={max(steps)}")
