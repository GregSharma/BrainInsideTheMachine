"""
LAYER DELTA EXTRACTION — Let the architecture speak.

For each generation step, extract h_L(t) at ALL 36 layers.
Compute δ_L(t) = ||h_L(t+1) - h_L(t)|| at each layer at each step.
Correlate with per-token cosine spikes (zh-en).

No hypotheses. No prescribed layers. 36 correlations.
If L30-32 lights up → input-pass Z = generation-time Z.
If something else lights up → Z lives where we didn't look.

Output: output/layer_deltas.npz
  Keys: deltas_prob{idx}_{lang} → (n_steps-1, 36)  [delta norms per layer per step]
        h_all_prob{idx}_{lang} → (n_steps, 36, 2048) would be too big
  So we save just the deltas and the cosine correlations.
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
    dtype=torch.float16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model.eval()

N_LAYERS = model.config.num_hidden_layers  # 36
D = model.config.hidden_size  # 2048
print(f"Model: {N_LAYERS} layers, {D} dims")

# ---------- PROBLEMS (same selection as before) ----------
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

selected = []
cat_count = {i: 0 for i in range(5)}
for i, p in enumerate(problems):
    c = p['category']
    if cat_count[c] < 4:
        selected.append(i)
        cat_count[c] += 1
    if len(selected) == 20:
        break

langs = ['zh', 'en']  # Just zh/en for correlation with cosine spikes
print(f"Selected {len(selected)} problems, {len(langs)} languages = {len(selected)*len(langs)} extractions")


# ---------- EXTRACT ALL-LAYER DELTAS ----------
def extract_all_layer_deltas(model, tokenizer, prompt, max_new_tokens=256):
    """Extract hidden states at ALL 36 layers at every generation step.
    Return delta norms: ||h_L(t+1) - h_L(t)|| for each layer L and step t."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Hook ALL layers
    hook_data = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hook_data[layer_idx] = h[:, -1, :].detach().cpu().float().numpy().flatten()
        return hook_fn

    for layer_idx in range(N_LAYERS):
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    all_states = []  # List of dicts: {layer_idx: h_vector}
    tokens = []
    past_key_values = None

    with torch.no_grad():
        # First pass
        outputs = model(inputs.input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        state = {l: hook_data[l].copy() for l in range(N_LAYERS)}
        all_states.append(state)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            for h in handles:
                h.remove()
            return np.zeros((0, N_LAYERS)), tokens

        # Subsequent passes
        for step in range(1, max_new_tokens):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            state = {l: hook_data[l].copy() for l in range(N_LAYERS)}
            all_states.append(state)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

    for h in handles:
        h.remove()

    # Compute deltas: ||h_L(t+1) - h_L(t)|| for each layer
    n_steps = len(all_states)
    deltas = np.zeros((n_steps - 1, N_LAYERS))

    for t in range(n_steps - 1):
        for l in range(N_LAYERS):
            deltas[t, l] = np.linalg.norm(all_states[t+1][l] - all_states[t][l])

    # Also save full h32 for cosine computation
    h32 = np.array([all_states[t][31] for t in range(n_steps)])

    return deltas, h32, tokens


# ---------- MAIN EXTRACTION ----------
all_deltas = {}
all_h32 = {}
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
        deltas, h32, tokens = extract_all_layer_deltas(model, tokenizer, prob[lang])
        dt = time.time() - t0

        all_deltas[key] = deltas
        all_h32[key] = h32

        text = tokenizer.decode(tokens, skip_special_tokens=True)[:200]

        metadata[key] = {
            'problem_idx': prob_idx,
            'category': cat_name,
            'category_id': prob['category'],
            'language': lang,
            'prompt': prob[lang],
            'n_steps': h32.shape[0],
            'n_tokens': len(tokens),
            'text_preview': text,
            'time_seconds': round(dt, 1),
        }

        elapsed = time.time() - t_start
        rate = done / elapsed
        remaining = (total - done) / rate
        print(f"{h32.shape[0]} steps, {dt:.1f}s (ETA: {remaining/60:.1f}min)")


# ---------- SAVE DELTAS ----------
print(f"\nSaving deltas...")
save_dict = {}
for key in all_deltas:
    save_dict[f"deltas_{key}"] = all_deltas[key]
    save_dict[f"h32_{key}"] = all_h32[key]

np.savez_compressed('output/layer_deltas.npz', **save_dict)
with open('output/layer_deltas_meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: output/layer_deltas.npz, output/layer_deltas_meta.json")


# ---------- CORRELATE DELTAS WITH COSINE SPIKES ----------
print("\n" + "=" * 90)
print("LAYER DELTA × COSINE SPIKE CORRELATION — 36 layers, no hypotheses")
print("=" * 90)

from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import uniform_filter1d

def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)

# For each problem, compute per-token cosine (zh-en) at L32
# Then correlate with delta norms at each layer
layer_correlations = np.zeros((N_LAYERS, len(selected)))
layer_correlations_p = np.zeros((N_LAYERS, len(selected)))

for pi, prob_idx in enumerate(selected):
    key_zh = f"prob{prob_idx}_zh"
    key_en = f"prob{prob_idx}_en"

    if key_zh not in all_h32 or key_en not in all_h32:
        continue

    h32_zh = all_h32[key_zh]
    h32_en = all_h32[key_en]
    deltas_zh = all_deltas[key_zh]
    deltas_en = all_deltas[key_en]

    n_zh = h32_zh.shape[0]
    n_en = h32_en.shape[0]

    # Mean-center
    h32_zh_c = h32_zh - h32_zh.mean(axis=0)
    h32_en_c = h32_en - h32_en.mean(axis=0)

    # Align to common grid (use shorter of the two delta arrays)
    # Deltas have n_steps-1 points
    n_d_zh = deltas_zh.shape[0]
    n_d_en = deltas_en.shape[0]
    n_grid = min(n_d_zh, n_d_en, 200)

    if n_grid < 10:
        continue

    # Interpolate h32 to delta grid (shift by 0.5 since deltas are between steps)
    tau_grid = np.linspace(0, 1, n_grid)

    # Cosine at each grid point
    cosines = np.zeros(n_grid)
    for i, tau in enumerate(tau_grid):
        idx_zh = min(int(tau * (n_zh - 1)), n_zh - 1)
        idx_en = min(int(tau * (n_en - 1)), n_en - 1)
        cosines[i] = cosine_sim(h32_zh_c[idx_zh], h32_en_c[idx_en])

    # Delta norms at each layer, interpolated to grid
    for l in range(N_LAYERS):
        # Average zh and en deltas at this layer
        delta_zh_l = deltas_zh[:, l]
        delta_en_l = deltas_en[:, l]

        # Interpolate to grid
        d_zh_grid = np.interp(tau_grid, np.linspace(0, 1, n_d_zh), delta_zh_l)
        d_en_grid = np.interp(tau_grid, np.linspace(0, 1, n_d_en), delta_en_l)
        delta_avg = (d_zh_grid + d_en_grid) / 2

        if np.std(delta_avg) > 1e-8 and np.std(cosines) > 1e-8:
            r, p = pearsonr(cosines, delta_avg)
            layer_correlations[l, pi] = r
            layer_correlations_p[l, pi] = p
        else:
            layer_correlations[l, pi] = 0
            layer_correlations_p[l, pi] = 1


# ---------- RESULTS ----------
print(f"\n{'Layer':>6} | {'Mean r':>7} | {'Median r':>8} | {'Sig (p<.05)':>11} | Visual")
print("-" * 70)

mean_r_per_layer = np.mean(layer_correlations, axis=1)
median_r_per_layer = np.median(layer_correlations, axis=1)
n_sig_per_layer = np.sum(layer_correlations_p < 0.05, axis=1)

for l in range(N_LAYERS):
    mr = mean_r_per_layer[l]
    mdr = median_r_per_layer[l]
    ns = n_sig_per_layer[l]

    # Bar chart scaled to [-0.3, 0.3]
    bar_len = int((mr + 0.3) * 25)
    bar_len = max(0, min(bar_len, 25))
    bar = ' ' * 8 + '│'  # zero line at position 8
    if mr >= 0:
        bar = ' ' * 8 + '│' + '█' * int(mr * 25)
    else:
        neg_len = int(abs(mr) * 25)
        bar = ' ' * (8 - neg_len) + '▓' * neg_len + '│'

    marker = ""
    if abs(mr) > 0.1:
        marker = " ★" if mr > 0 else " ▼"

    print(f"  L{l:>2}   | {mr:>7.4f} | {mdr:>8.4f} | {ns:>4}/{len(selected):>2}      | {bar}{marker}")


# Highlight top and bottom layers
sorted_layers = np.argsort(mean_r_per_layer)

print(f"\n  TOP 5 LAYERS (highest positive correlation with cosine spikes):")
for l in sorted_layers[-5:][::-1]:
    print(f"    L{l}: mean r = {mean_r_per_layer[l]:.4f}, "
          f"sig = {n_sig_per_layer[l]}/{len(selected)}")

print(f"\n  BOTTOM 5 LAYERS (most negative / anti-correlated):")
for l in sorted_layers[:5]:
    print(f"    L{l}: mean r = {mean_r_per_layer[l]:.4f}, "
          f"sig = {n_sig_per_layer[l]}/{len(selected)}")

# Check if reassembly layers (L30-32) are in the top
reassembly = [30, 31, 32]
reassembly_ranks = [N_LAYERS - 1 - np.where(sorted_layers[::-1] == l)[0][0] for l in reassembly if l in sorted_layers]

print(f"\n  REASSEMBLY LAYERS (L30-32) PREDICTION CHECK:")
for l in reassembly:
    rank = N_LAYERS - 1 - np.where(sorted_layers[::-1] == l)[0][0]
    print(f"    L{l}: rank {rank+1}/{N_LAYERS}, mean r = {mean_r_per_layer[l]:.4f}")

if mean_r_per_layer[sorted_layers[-1]] > 0.05:
    top_l = sorted_layers[-1]
    if top_l in range(29, 33):
        print(f"\n  ★ TOP LAYER IS IN REASSEMBLY ZONE (L{top_l})")
        print(f"  → INPUT-PASS Z AND GENERATION-TIME Z ARE THE SAME MECHANISM")
    else:
        print(f"\n  ★ TOP LAYER IS L{top_l} — NOT in reassembly zone")
        print(f"  → Generation-time Z lives at a DIFFERENT depth than input-pass Z")


# ---------- SAVE CORRELATION RESULTS ----------
output = {
    'mean_r_per_layer': mean_r_per_layer.tolist(),
    'median_r_per_layer': median_r_per_layer.tolist(),
    'n_sig_per_layer': n_sig_per_layer.tolist(),
    'per_problem_correlations': layer_correlations.tolist(),
    'top_5_layers': sorted_layers[-5:][::-1].tolist(),
    'bottom_5_layers': sorted_layers[:5].tolist(),
    'n_problems': len(selected),
    'n_layers': N_LAYERS,
}

with open('output/layer_delta_correlations.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: output/layer_delta_correlations.json")
print(f"Total time: {(time.time() - t_start)/60:.1f} minutes")
