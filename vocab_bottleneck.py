"""
Vocabulary Bottleneck Measurement — "342,234 International Flights"

Core hypothesis: the model reasons in Z-space (layer ~32) but is forced to project
onto vocabulary tokens at every generation step. This projection is lossy. The loss
is language-dependent (denser vocabulary coverage → less loss).

Three metrics at each generation step t:

1. η (eta) — Reasoning efficiency:
   η(t) = ||h₃₂(t+1) - h₃₂(t)|| / ||h₀(t+1) - h₃₂(t)||
   How much Z-space progress per unit of total distance traveled.
   η ≈ 1: direct route. η << 1: international flights.

2. Round-trip projection loss:
   Take h₃₅(t) (pre-vocabulary hidden state), project to vocab logits,
   argmax to get token, embed that token back.
   δ(t) = ||embed(argmax(W·h₃₅(t))) - h₃₅(t)|| / ||h₃₅(t)||
   How much information the vocabulary projection destroys.

3. Path efficiency (cumulative):
   E = ||h₃₂(T) - h₃₂(0)|| / Σ||h₃₂(t+1) - h₃₂(t)||
   Straight-line distance / total path length in Z-space.
   E = 1: perfectly direct. E << 1: spiraling.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys
import random as pyrandom

# ---------- LOAD MODEL ----------
print("Loading Qwen2.5-3B...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    torch_dtype=torch.float16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model.eval()

# Get embedding matrix for round-trip measurement
# Weights are tied: embed_tokens == lm_head (verified)
# But there's an RMSNorm between h35 and lm_head: logits = lm_head(norm(h35))
# So the proper round-trip is: norm(h35) → logits → argmax → embed(token_id)
# Compare embed(token_id) to norm(h35), not to h35.
embed_matrix = model.model.embed_tokens.weight.detach()  # (vocab_size, d)
final_norm = model.model.norm  # RMSNorm before lm_head

N_LAYERS = model.config.num_hidden_layers  # 36 for Qwen2.5-3B
D = model.config.hidden_size  # 2048
REASONING_LAYER = 31  # 0-indexed → layer 32
LAST_LAYER = N_LAYERS - 1  # 35

print(f"Model: {N_LAYERS} layers, d={D}")
print(f"Reasoning layer: {REASONING_LAYER} (= L32)")
print(f"Vocab size: {embed_matrix.shape[0]}")

# ---------- PROBLEMS ----------
# Same generation as extract_all_layers.py
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
        else:
            zh = f"计算 {a} × {b} 的值。"
            en = f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "category": 0})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "category": 1})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "category": 2})
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "category": 3})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        problems.append({"zh": zh, "en": en, "category": 4})
    rng.shuffle(problems)
    return problems

problems = generate_problems(200, seed=42)

# Pick 5 problems across categories (one per category)
# After shuffling, we need to find one of each category
cat_examples = {}
for i, p in enumerate(problems):
    c = p['category']
    if c not in cat_examples:
        cat_examples[c] = i
    if len(cat_examples) == 5:
        break
test_indices = sorted(cat_examples.values())
print(f"Test problems: {test_indices} (categories: {[problems[i]['category'] for i in test_indices]})")

langs = ['zh', 'en']  # Start with the contrastive pair. Add more if this works.

CAT_NAMES = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]
results = {}

for prob_idx in test_indices:
    prob = problems[prob_idx]
    cat_name = CAT_NAMES[prob['category']]
    print(f"\n{'='*60}")
    print(f"Problem {prob_idx} ({cat_name})")
    print(f"  zh: {prob['zh']}")
    print(f"  en: {prob['en']}")

    for lang in langs:
        prompt = prob[lang]

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs.input_ids.shape[1]

        # Storage for per-step activations
        step_h0 = []    # layer 0 output (after first transformer block)
        step_h32 = []   # layer 32 output (reasoning state)
        step_h35 = []   # layer 35 output (pre-vocabulary projection)
        step_tokens = [] # the actual tokens generated

        # Hook storage
        hook_data = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                hook_data[layer_idx] = h[:, -1, :].detach().cpu().float().numpy()
            return hook_fn

        # Register hooks on layers 0, 32 (idx 31), and 35 (idx 35)
        hooks = []
        hooks.append(model.model.layers[0].register_forward_hook(make_hook(0)))
        hooks.append(model.model.layers[REASONING_LAYER].register_forward_hook(make_hook(REASONING_LAYER)))
        hooks.append(model.model.layers[LAST_LAYER].register_forward_hook(make_hook(LAST_LAYER)))

        # Generate token by token (no KV cache for simplicity — slow but correct)
        gen_ids = inputs.input_ids.clone()
        max_new_tokens = 256

        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = model(gen_ids)

                step_h0.append(hook_data[0].flatten().copy())
                step_h32.append(hook_data[REASONING_LAYER].flatten().copy())
                step_h35.append(hook_data[LAST_LAYER].flatten().copy())

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                step_tokens.append(next_token.item())
                gen_ids = torch.cat([gen_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        for h in hooks:
            h.remove()

        n_prompt_steps = input_len  # steps that process prompt tokens
        n_gen = len(step_h32) - n_prompt_steps
        total_steps = len(step_h32)

        if n_gen < 2:
            print(f"  {lang}: only {n_gen} generation steps, skipping")
            continue

        # ===== METRIC 1: η (reasoning efficiency) =====
        # For each generation step t:
        #   reasoning_progress = ||h32(t) - h32(t-1)||
        #   total_travel = ||h0(t) - h32(t-1)||
        #   η = reasoning_progress / total_travel
        etas = []
        reasoning_dists = []
        vocab_dists = []

        for t in range(n_prompt_steps, total_steps):
            h32_prev = step_h32[t - 1]
            h32_curr = step_h32[t]
            h0_curr = step_h0[t]

            r_prog = np.linalg.norm(h32_curr - h32_prev)
            v_det = np.linalg.norm(h0_curr - h32_prev)

            eta = r_prog / v_det if v_det > 1e-8 else 0.0
            etas.append(float(eta))
            reasoning_dists.append(float(r_prog))
            vocab_dists.append(float(v_det))

        # ===== METRIC 2: Round-trip vocabulary projection loss =====
        # For each generation step t:
        #   The model computed h35(t), applied RMSNorm, projected to logits, argmax → token_id
        #   The token_id gets embedded back: e = embed_matrix[token_id]
        #   δ(t) = ||e - norm(h35(t))|| / ||norm(h35(t))||
        #   This is the ACTUAL information loss from the vocabulary bottleneck.
        #   We compare in the normalized space because that's where the projection happens.
        roundtrip_losses = []
        for t in range(n_prompt_steps, total_steps):
            h35 = step_h35[t]
            token_id = step_tokens[t - n_prompt_steps]
            token_embed = embed_matrix[token_id].cpu().float().numpy()

            # Apply RMSNorm to h35 to get what the model actually projects
            h35_tensor = torch.tensor(h35, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to("cuda")
            with torch.no_grad():
                h35_normed = final_norm(h35_tensor).squeeze().cpu().float().numpy()

            h35n_norm = np.linalg.norm(h35_normed)
            loss = np.linalg.norm(token_embed - h35_normed) / h35n_norm if h35n_norm > 1e-8 else 0.0
            roundtrip_losses.append(float(loss))

        # ===== METRIC 3: Path efficiency (the "342K flights" number) =====
        # Straight-line distance from first to last reasoning state,
        # divided by total path length
        h32_gen = [step_h32[t] for t in range(n_prompt_steps, total_steps)]
        straight_line = np.linalg.norm(h32_gen[-1] - h32_gen[0])
        path_length = sum(np.linalg.norm(h32_gen[t+1] - h32_gen[t])
                          for t in range(len(h32_gen) - 1))
        path_efficiency = straight_line / path_length if path_length > 1e-8 else 0.0

        # Decode generated text
        gen_text = tokenizer.decode(gen_ids[0, input_len:], skip_special_tokens=True)

        key = f"prob{prob_idx}_{lang}"
        results[key] = {
            'problem_idx': prob_idx,
            'category': cat_name,
            'language': lang,
            'prompt': prompt,
            'n_generated_tokens': n_gen,
            'mean_eta': float(np.mean(etas)),
            'std_eta': float(np.std(etas)),
            'mean_reasoning_dist': float(np.mean(reasoning_dists)),
            'mean_vocab_dist': float(np.mean(vocab_dists)),
            'mean_roundtrip_loss': float(np.mean(roundtrip_losses)),
            'std_roundtrip_loss': float(np.std(roundtrip_losses)),
            'path_efficiency': float(path_efficiency),
            'straight_line_dist': float(straight_line),
            'total_path_length': float(path_length),
            'path_ratio': float(path_length / straight_line) if straight_line > 1e-8 else float('inf'),
            'etas': etas,
            'roundtrip_losses': roundtrip_losses,
            'reasoning_dists': reasoning_dists,
            'generated_text': gen_text[:500],
        }

        pr = path_length / straight_line if straight_line > 1e-8 else float('inf')
        print(f"  {lang}: {n_gen} tokens | η={np.mean(etas):.4f} | "
              f"δ_roundtrip={np.mean(roundtrip_losses):.4f} | "
              f"path_ratio={pr:.1f}x | "
              f"path_eff={path_efficiency:.4f}")

# ===== SUMMARY =====
print("\n" + "=" * 90)
print("VOCABULARY BOTTLENECK SUMMARY")
print("=" * 90)
print(f"{'Problem':>10} {'Cat':>13} | {'Lang':>4} | {'Tok':>4} | "
      f"{'η':>7} | {'δ_RT':>7} | {'Path×':>7} | {'Eff':>7}")
print("-" * 90)
for prob_idx in test_indices:
    for lang in langs:
        key = f"prob{prob_idx}_{lang}"
        r = results[key]
        print(f"  prob{prob_idx:>3} {r['category']:>13} | {lang:>4} | "
              f"{r['n_generated_tokens']:>4} | {r['mean_eta']:>7.4f} | "
              f"{r['mean_roundtrip_loss']:>7.4f} | {r['path_ratio']:>7.1f} | "
              f"{r['path_efficiency']:>7.4f}")
    print()

# Per-language aggregates
print("=== PER-LANGUAGE AGGREGATES ===")
for lang in langs:
    lang_etas = []
    lang_rt = []
    lang_pe = []
    lang_pr = []
    for prob_idx in test_indices:
        key = f"prob{prob_idx}_{lang}"
        lang_etas.extend(results[key]['etas'])
        lang_rt.extend(results[key]['roundtrip_losses'])
        lang_pe.append(results[key]['path_efficiency'])
        lang_pr.append(results[key]['path_ratio'])
    print(f"  {lang}: η={np.mean(lang_etas):.4f} ± {np.std(lang_etas):.4f} | "
          f"δ_RT={np.mean(lang_rt):.4f} ± {np.std(lang_rt):.4f} | "
          f"path_ratio={np.mean(lang_pr):.1f}x | "
          f"path_eff={np.mean(lang_pe):.4f}")

# THE KEY QUESTION: does Chinese travel less than English?
zh_etas = []
en_etas = []
zh_rt = []
en_rt = []
zh_pe = []
en_pe = []
for prob_idx in test_indices:
    zh_etas.extend(results[f"prob{prob_idx}_zh"]['etas'])
    en_etas.extend(results[f"prob{prob_idx}_en"]['etas'])
    zh_rt.extend(results[f"prob{prob_idx}_zh"]['roundtrip_losses'])
    en_rt.extend(results[f"prob{prob_idx}_en"]['roundtrip_losses'])
    zh_pe.append(results[f"prob{prob_idx}_zh"]['path_efficiency'])
    en_pe.append(results[f"prob{prob_idx}_en"]['path_efficiency'])

print(f"\n=== THE BOTTLENECK QUESTION ===")
print(f"Chinese η: {np.mean(zh_etas):.4f}   English η: {np.mean(en_etas):.4f}   "
      f"Ratio zh/en: {np.mean(zh_etas)/np.mean(en_etas):.3f}")
print(f"Chinese δ_RT: {np.mean(zh_rt):.4f}   English δ_RT: {np.mean(en_rt):.4f}   "
      f"Ratio en/zh: {np.mean(en_rt)/np.mean(zh_rt):.3f}")
print(f"Chinese path_eff: {np.mean(zh_pe):.4f}   English path_eff: {np.mean(en_pe):.4f}   "
      f"Ratio zh/en: {np.mean(zh_pe)/np.mean(en_pe):.3f}")

print(f"\nInterpretation:")
if np.mean(zh_etas) > np.mean(en_etas) * 1.05:
    print(f"  Chinese makes more reasoning progress per step (η ratio {np.mean(zh_etas)/np.mean(en_etas):.2f}x)")
    print(f"  → Vocabulary bottleneck is language-dependent. Chinese pays less.")
elif np.mean(en_etas) > np.mean(zh_etas) * 1.05:
    print(f"  English makes more reasoning progress per step (unexpected!)")
    print(f"  → Vocabulary bottleneck hypothesis may be wrong, or the metric needs refinement.")
else:
    print(f"  η is similar across languages (ratio ≈ 1.0)")
    print(f"  → Vocabulary bottleneck may exist but is NOT language-dependent.")
    print(f"  → The spiral must come from something else.")

if np.mean(en_rt) > np.mean(zh_rt) * 1.05:
    print(f"  English round-trip loss is {np.mean(en_rt)/np.mean(zh_rt):.2f}x Chinese")
    print(f"  → English tokens are farther from the pre-projection hidden state.")
    print(f"  → The model loses more information per English token than per Chinese token.")

# Save everything
with open('output/vocab_bottleneck.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to output/vocab_bottleneck.json")
