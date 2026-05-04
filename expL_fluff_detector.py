"""Experiment L: Fluff vs Computation Token Classification.

The model wastes most generated tokens on narrative ("To solve this...") when the
actual computation is tiny ("5+6=11"). The MLP delta at each generated token tells
us which tokens are COMPUTATION and which are FLUFF.

For each generated token:
1. Capture MLP delta at L8: how much is language-maintenance (high language PC content)
   versus reasoning-progress (movement in non-language dimensions)?
2. Capture attention pattern: what does the model attend to during computation vs narration?

Compute: what fraction of tokens are actually doing reasoning work versus
maintaining language scaffolding?
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import random as pyrandom

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
N_PCA = 200

# 5 problems — mix of easy and hard
test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133", "category": "easy_add"},
    {"prompt": "Calculate 123 × 45.", "answer": "5535", "category": "medium_mult"},
    {"prompt": "What is the remainder when 7654 is divided by 37?", "answer": "34", "category": "division"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120", "category": "combo"},
    {"prompt": "An arithmetic sequence has first term 3 and common difference 7. Find the sum of the first 20 terms.",
     "answer": "1390", "category": "sequence"},
]


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
        a1, d = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
                          "en": f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."})
    return problems


# =============================================================================
# Step 1: Fit language PCA at L8 (to classify MLP deltas)
# =============================================================================
print("=" * 70)
print("FITTING LANGUAGE PCA AT L8")
print("=" * 70)

problems = generate_pca_problems(N_PCA, seed=42)
d = model.config.hidden_size

layer_output = {}
def capture_hook(module, input, output):
    h = output[0] if isinstance(output, tuple) else output
    layer_output['h'] = h.detach()[:, -1, :]

handle = model.model.layers[8].register_forward_hook(capture_hook)

zh_hidden = np.zeros((N_PCA, d), dtype=np.float32)
en_hidden = np.zeros((N_PCA, d), dtype=np.float32)

for i, prob in enumerate(problems):
    inputs = tokenizer(prob["zh"], return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    zh_hidden[i] = layer_output['h'].cpu().float().numpy()
    layer_output.clear()

for i, prob in enumerate(problems):
    inputs = tokenizer(prob["en"], return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    en_hidden[i] = layer_output['h'].cpu().float().numpy()
    layer_output.clear()

handle.remove()

zh_norms = np.linalg.norm(zh_hidden, axis=1, keepdims=True)
en_norms = np.linalg.norm(en_hidden, axis=1, keepdims=True)
zh_unit = zh_hidden / zh_norms
en_unit = en_hidden / en_norms
combined = np.vstack([zh_unit, en_unit])

pca = PCA(n_components=10)
pca.fit(combined)
lang_pcs = pca.components_  # (10, d)

print(f"  Top 10 PCs variance: {sum(pca.explained_variance_ratio_):.1%}")
print(f"  PC0 variance: {pca.explained_variance_ratio_[0]:.1%}")

# =============================================================================
# Step 2: For each test problem, generate token-by-token with MLP delta capture
# =============================================================================
print(f"\n{'='*70}")
print("TOKEN-BY-TOKEN GENERATION WITH MLP DELTA CAPTURE")
print("=" * 70)

results = {
    "experiment": "L: Fluff vs Computation Detector",
    "n_lang_pcs": 10,
    "lang_pca_variance": float(sum(pca.explained_variance_ratio_)),
    "problems": [],
}

# We need hooks on L8 MLP specifically
# In Qwen2.5, each layer has: self_attn, mlp, input_layernorm, post_attention_layernorm
# The MLP delta = mlp_output. We capture pre-MLP and post-MLP at L8.

for prob_idx, prob in enumerate(test_problems):
    print(f"\n  Problem {prob_idx}: {prob['prompt'][:60]}")
    print(f"    Answer: {prob['answer']}")

    input_ids = tokenizer.encode(prob["prompt"])
    prompt_len = len(input_ids)

    # --- Prefill ---
    with torch.no_grad():
        outputs = model(
            torch.tensor([input_ids], device=device),
            use_cache=True,
        )
    past_kv = outputs.past_key_values

    first_token_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[first_token_id]], device=device)

    token_data = []
    generated_ids = [first_token_id]

    # For each generation step, capture:
    # 1. MLP delta at L8
    # 2. Attention pattern at L8
    for step in range(MAX_NEW_TOKENS - 1):
        # Set up hooks to capture MLP input and output at L8
        mlp_captures = {}

        def make_mlp_out_hook():
            def hook_fn(module, input, output):
                # MLP output IS the delta (residual contribution)
                if isinstance(output, tuple):
                    mlp_captures['delta'] = output[0][0, -1, :].detach().cpu().float().numpy()
                else:
                    mlp_captures['delta'] = output[0, -1, :].detach().cpu().float().numpy()
            return hook_fn

        h_out = model.model.layers[8].mlp.register_forward_hook(make_mlp_out_hook())

        with torch.no_grad():
            out = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )

        h_out.remove()

        past_kv = out.past_key_values
        next_logits = out.logits[0, -1, :]
        next_id = int(next_logits.argmax())

        # MLP delta (output IS the delta — MLP residual contribution)
        if 'delta' in mlp_captures:
            mlp_delta = mlp_captures['delta']
        else:
            mlp_delta = np.zeros(d, dtype=np.float32)

        mlp_delta_norm = float(np.linalg.norm(mlp_delta))

        # Project MLP delta onto language PCs
        if mlp_delta_norm > 1e-8:
            mlp_unit = mlp_delta / mlp_delta_norm
        else:
            mlp_unit = mlp_delta

        lang_proj_squared = sum(float(mlp_unit @ pc)**2 for pc in lang_pcs)
        # lang_proj_squared is the fraction of MLP delta variance in language subspace
        reasoning_fraction = 1.0 - lang_proj_squared

        # Attention pattern at L8: what does the new token attend to?
        attn_L8 = out.attentions[8][0].cpu().float().numpy()  # (n_heads, 1, total_seq)
        # Average over heads
        mean_attn = attn_L8.mean(axis=0)[0]  # (total_seq,)
        total_seq = len(mean_attn)
        # Split: prompt tokens vs generated tokens
        attn_on_prompt = float(mean_attn[:prompt_len].sum())
        attn_on_generated = float(mean_attn[prompt_len:].sum())

        # Entropy of attention (high = diffuse, low = focused)
        attn_clipped = np.clip(mean_attn, 1e-10, 1.0)
        attn_entropy = float(-np.sum(attn_clipped * np.log(attn_clipped)))

        # Top-1 attention target
        top_attn_pos = int(np.argmax(mean_attn))
        top_attn_val = float(mean_attn[top_attn_pos])
        top_attn_region = "prompt" if top_attn_pos < prompt_len else "generated"

        decoded_token = tokenizer.decode([next_id])

        # Is this token a digit or math operator?
        is_math = any(c.isdigit() for c in decoded_token) or any(c in decoded_token for c in '=+-×÷*/')
        is_punctuation = all(not c.isalnum() for c in decoded_token.strip())

        token_entry = {
            "step": step,
            "token_id": next_id,
            "token_text": decoded_token,
            "is_math": is_math,
            "is_punctuation": is_punctuation,
            "mlp_delta_norm": mlp_delta_norm,
            "lang_fraction": float(lang_proj_squared),
            "reasoning_fraction": float(reasoning_fraction),
            "attn_on_prompt": attn_on_prompt,
            "attn_on_generated": attn_on_generated,
            "attn_entropy": attn_entropy,
            "top_attn_region": top_attn_region,
            "top_attn_val": top_attn_val,
        }
        token_data.append(token_entry)

        generated_ids.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)

        if next_id == tokenizer.eos_token_id:
            break

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"    Generated: {full_text[:100]}...")
    print(f"    Tokens generated: {len(token_data)}")

    # Classify each token
    n_lang_tokens = sum(1 for t in token_data if t["lang_fraction"] > 0.5)
    n_reasoning_tokens = sum(1 for t in token_data if t["reasoning_fraction"] > 0.5)
    n_math_tokens = sum(1 for t in token_data if t["is_math"])
    total_gen = len(token_data)

    avg_lang_frac = np.mean([t["lang_fraction"] for t in token_data]) if token_data else 0
    avg_reasoning_frac = np.mean([t["reasoning_fraction"] for t in token_data]) if token_data else 0
    avg_attn_prompt = np.mean([t["attn_on_prompt"] for t in token_data]) if token_data else 0
    avg_attn_gen = np.mean([t["attn_on_generated"] for t in token_data]) if token_data else 0
    avg_entropy = np.mean([t["attn_entropy"] for t in token_data]) if token_data else 0

    # Compare attention patterns for math vs non-math tokens
    math_tokens = [t for t in token_data if t["is_math"]]
    nonmath_tokens = [t for t in token_data if not t["is_math"] and not t["is_punctuation"]]

    math_attn_prompt = np.mean([t["attn_on_prompt"] for t in math_tokens]) if math_tokens else 0
    math_attn_gen = np.mean([t["attn_on_generated"] for t in math_tokens]) if math_tokens else 0
    math_entropy = np.mean([t["attn_entropy"] for t in math_tokens]) if math_tokens else 0
    math_reasoning = np.mean([t["reasoning_fraction"] for t in math_tokens]) if math_tokens else 0

    nonmath_attn_prompt = np.mean([t["attn_on_prompt"] for t in nonmath_tokens]) if nonmath_tokens else 0
    nonmath_attn_gen = np.mean([t["attn_on_generated"] for t in nonmath_tokens]) if nonmath_tokens else 0
    nonmath_entropy = np.mean([t["attn_entropy"] for t in nonmath_tokens]) if nonmath_tokens else 0
    nonmath_reasoning = np.mean([t["reasoning_fraction"] for t in nonmath_tokens]) if nonmath_tokens else 0

    prob_result = {
        "prompt": prob["prompt"],
        "answer": prob["answer"],
        "category": prob["category"],
        "full_text": full_text,
        "total_tokens": total_gen,
        "math_tokens": n_math_tokens,
        "lang_dominated_tokens": n_lang_tokens,
        "reasoning_dominated_tokens": n_reasoning_tokens,
        "fluff_ratio": (total_gen - n_math_tokens) / max(total_gen, 1),
        "avg_lang_fraction": float(avg_lang_frac),
        "avg_reasoning_fraction": float(avg_reasoning_frac),
        "avg_attn_on_prompt": float(avg_attn_prompt),
        "avg_attn_on_generated": float(avg_attn_gen),
        "avg_attn_entropy": float(avg_entropy),
        "math_vs_nonmath": {
            "math_attn_on_prompt": float(math_attn_prompt),
            "math_attn_on_generated": float(math_attn_gen),
            "math_attn_entropy": float(math_entropy),
            "math_reasoning_fraction": float(math_reasoning),
            "nonmath_attn_on_prompt": float(nonmath_attn_prompt),
            "nonmath_attn_on_generated": float(nonmath_attn_gen),
            "nonmath_attn_entropy": float(nonmath_entropy),
            "nonmath_reasoning_fraction": float(nonmath_reasoning),
        },
        "per_token": token_data,
    }
    results["problems"].append(prob_result)

    print(f"    Fluff ratio: {prob_result['fluff_ratio']:.0%} "
          f"(math tokens: {n_math_tokens}/{total_gen})")
    print(f"    Avg MLP: {avg_lang_frac:.0%} language / {avg_reasoning_frac:.0%} reasoning")
    print(f"    Avg attention: {avg_attn_prompt:.0%} prompt / {avg_attn_gen:.0%} generated")
    if math_tokens and nonmath_tokens:
        print(f"    Math tokens reasoning fraction: {math_reasoning:.0%}")
        print(f"    Non-math tokens reasoning fraction: {nonmath_reasoning:.0%}")
        print(f"    Math attention on prompt: {math_attn_prompt:.0%}")
        print(f"    Non-math attention on prompt: {nonmath_attn_prompt:.0%}")

# =============================================================================
# Overall Summary
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT L — FLUFF vs COMPUTATION SUMMARY")
print("=" * 70)

all_fluff = [p["fluff_ratio"] for p in results["problems"]]
all_lang = [p["avg_lang_fraction"] for p in results["problems"]]
all_reasoning = [p["avg_reasoning_fraction"] for p in results["problems"]]

print(f"  Mean fluff ratio: {np.mean(all_fluff):.0%} "
      f"(range: {min(all_fluff):.0%}–{max(all_fluff):.0%})")
print(f"  Mean language fraction of MLP delta: {np.mean(all_lang):.0%}")
print(f"  Mean reasoning fraction of MLP delta: {np.mean(all_reasoning):.0%}")

# Aggregate math vs nonmath across all problems
all_math_reasoning = []
all_nonmath_reasoning = []
all_math_attn_prompt = []
all_nonmath_attn_prompt = []
for p in results["problems"]:
    for t in p["per_token"]:
        if t["is_math"]:
            all_math_reasoning.append(t["reasoning_fraction"])
            all_math_attn_prompt.append(t["attn_on_prompt"])
        elif not t["is_punctuation"]:
            all_nonmath_reasoning.append(t["reasoning_fraction"])
            all_nonmath_attn_prompt.append(t["attn_on_prompt"])

if all_math_reasoning and all_nonmath_reasoning:
    print(f"\n  COMPUTATION vs NARRATION tokens (aggregated):")
    print(f"    Math tokens (n={len(all_math_reasoning)}):")
    print(f"      Reasoning fraction: {np.mean(all_math_reasoning):.0%}")
    print(f"      Attention on prompt: {np.mean(all_math_attn_prompt):.0%}")
    print(f"    Narration tokens (n={len(all_nonmath_reasoning)}):")
    print(f"      Reasoning fraction: {np.mean(all_nonmath_reasoning):.0%}")
    print(f"      Attention on prompt: {np.mean(all_nonmath_attn_prompt):.0%}")

    delta_reasoning = np.mean(all_math_reasoning) - np.mean(all_nonmath_reasoning)
    delta_attn = np.mean(all_math_attn_prompt) - np.mean(all_nonmath_attn_prompt)
    print(f"\n    Δ reasoning (math - narration): {delta_reasoning:+.0%}")
    print(f"    Δ attention on prompt (math - narration): {delta_attn:+.0%}")

results["summary"] = {
    "mean_fluff_ratio": float(np.mean(all_fluff)),
    "mean_lang_fraction": float(np.mean(all_lang)),
    "mean_reasoning_fraction": float(np.mean(all_reasoning)),
    "n_math_tokens_total": len(all_math_reasoning),
    "n_nonmath_tokens_total": len(all_nonmath_reasoning),
    "math_reasoning_mean": float(np.mean(all_math_reasoning)) if all_math_reasoning else None,
    "nonmath_reasoning_mean": float(np.mean(all_nonmath_reasoning)) if all_nonmath_reasoning else None,
    "math_attn_prompt_mean": float(np.mean(all_math_attn_prompt)) if all_math_attn_prompt else None,
    "nonmath_attn_prompt_mean": float(np.mean(all_nonmath_attn_prompt)) if all_nonmath_attn_prompt else None,
}

with open("output/expL_fluff_detector.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expL_fluff_detector.json")
