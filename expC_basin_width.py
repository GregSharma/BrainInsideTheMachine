"""Experiment C: Basin Width Mapping
Test how much perturbation the attractor basin can absorb.
C1: Norm tolerance
C2: Content tolerance (random vector)
C3: Procrustes-aligned L8 at injection point
C4: Raw layer skip
Injection points: L27 and L28 (per addendum)
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

# Load hidden states
print("Loading hidden states...")
data = np.load("output/all_layers_lasttok.npz")

# Load test problems (first 10 Chinese)
with open("output/expD_qualitative_read.json") as f:
    expD = json.load(f)

# We need the actual prompts. Load from the original data.
problems_file = "output/per_token_cosine_results.json"
try:
    with open(problems_file) as f:
        ptc = json.load(f)
    # Get problem texts from here if available
except:
    pass

# For injection experiments, we need to:
# 1. Run the model forward up to injection layer
# 2. Replace the hidden state
# 3. Continue from injection layer to end
# 4. Generate tokens

# Helper: generate with hidden state injection at a specific layer
def generate_with_injection(prompt_ids, injection_vector, inject_layer, max_new_tokens=128):
    """
    Run model with a hook that replaces the hidden state at inject_layer
    with injection_vector for the last token position.
    """
    input_ids = torch.tensor([prompt_ids], device=device)

    # We need to hook the model to inject our vector
    injected = [False]

    def hook_fn(module, input, output):
        if not injected[0]:
            # Qwen2 decoder layer outputs a plain tensor (batch, seq, dim)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Replace last token position
            vec = torch.tensor(injection_vector, dtype=hidden.dtype, device=hidden.device)
            hidden[0, -1, :] = vec
            injected[0] = True
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return output

    # Register hook on the target layer
    layer_module = model.model.layers[inject_layer]
    handle = layer_module.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        generated = tokenizer.decode(outputs[0][len(prompt_ids):], skip_special_tokens=True)
    finally:
        handle.remove()

    return generated

# Get 10 Chinese problem prompts
# We need tokenized prompts. Let's use a simple math problem set.
print("Preparing test problems...")

# Use the same problems as our extraction - load from the math dataset
import glob
problem_files = sorted(glob.glob("math_problems/*.json"))
if not problem_files:
    # Try alternative location
    problem_files = sorted(glob.glob("data/math_problems/*.json"))

# Fallback: create simple math prompts in Chinese
zh_prompts = [
    "请计算 2 + 3 × 4 的值。\n",
    "一个矩形的长为8厘米，宽为5厘米，求面积。\n",
    "如果 x + 5 = 12，求 x 的值。\n",
    "计算 100 除以 4 的结果。\n",
    "一个三角形三边长分别为3、4、5，求面积。\n",
    "求方程 2x - 6 = 0 的解。\n",
    "计算 3² + 4² 的值。\n",
    "一个圆的半径为7厘米，求周长（π取3.14）。\n",
    "如果 a = 3, b = 4，计算 a² + b² 的值。\n",
    "从1到100的自然数中，偶数有多少个？\n",
]

# Tokenize
prompt_ids_list = [tokenizer.encode(p) for p in zh_prompts]

# Get natural hidden states for norm reference
injection_layers = [27, 28]
natural_norms = {}
for L in injection_layers:
    zh_L = data[f"zh_L{L}"]
    natural_norms[L] = {
        "mean": float(np.linalg.norm(zh_L, axis=1).mean()),
        "std": float(np.linalg.norm(zh_L, axis=1).std()),
    }
    print(f"  L{L} natural norm: {natural_norms[L]['mean']:.1f} ± {natural_norms[L]['std']:.1f}")

h_L8_zh = data["zh_L8"][:10]  # (10, 2048)
h_L8_en = data["en_L8"][:10]

# Compute PC0 at each injection layer
pc0_by_layer = {}
for L in injection_layers:
    zh_L = data[f"zh_L{L}"]
    en_L = data[f"en_L{L}"]
    all_L = np.concatenate([zh_L, en_L], axis=0)
    mean_L = all_L.mean(axis=0)
    centered = all_L - mean_L
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc0_by_layer[L] = {"pc0": Vt[0], "mean": mean_L}

results = {"injection_layers": injection_layers, "natural_norms": natural_norms, "tests": {}}

n_test = 5  # 5 problems per test to keep runtime reasonable

for inject_layer in injection_layers:
    print(f"\n{'='*60}")
    print(f"INJECTION POINT: L{inject_layer}")
    print(f"{'='*60}")

    nat_norm = natural_norms[inject_layer]["mean"]
    zh_L = data[f"zh_L{inject_layer}"]
    en_L = data[f"en_L{inject_layer}"]

    layer_results = {}

    # ===== TEST C1: Norm Tolerance =====
    print(f"\n--- C1: Norm Tolerance ---")
    scale_factors = [1.0, 0.5, 0.33, 0.1, 0.05]
    c1_results = []

    for scale in scale_factors:
        print(f"\n  Scale = {scale}x (norm ≈ {nat_norm * scale:.1f}):")
        generations = []
        for i in range(n_test):
            # Take natural Chinese hidden state, scale it
            h = zh_L[i].copy() * scale
            gen = generate_with_injection(prompt_ids_list[i], h, inject_layer, max_new_tokens=64)
            generations.append(gen)
            print(f"    Problem {i}: {gen[:80]}...")

        c1_results.append({
            "scale": scale,
            "effective_norm": float(nat_norm * scale),
            "generations": generations,
        })

    layer_results["C1_norm_tolerance"] = c1_results

    # ===== TEST C2: Content Tolerance (Random Vector) =====
    print(f"\n--- C2: Random Vector at Natural Norm ---")
    c2_results = []

    for i in range(n_test):
        # Random 2048-dim vector with natural norm
        rand_vec = np.random.randn(2048).astype(np.float32)
        rand_vec = rand_vec / np.linalg.norm(rand_vec) * nat_norm

        # Set PC0 projection to Chinese mean
        pc0_info = pc0_by_layer[inject_layer]
        current_proj = (rand_vec - pc0_info["mean"]) @ pc0_info["pc0"]
        zh_mean_proj = ((zh_L - pc0_info["mean"]) @ pc0_info["pc0"]).mean()
        rand_vec += (zh_mean_proj - current_proj) * pc0_info["pc0"]

        gen = generate_with_injection(prompt_ids_list[i], rand_vec, inject_layer, max_new_tokens=64)
        c2_results.append({
            "norm": float(np.linalg.norm(rand_vec)),
            "pc0_proj": float((rand_vec - pc0_info["mean"]) @ pc0_info["pc0"]),
            "generation": gen,
        })
        print(f"  Problem {i}: {gen[:80]}...")

    layer_results["C2_random_vector"] = c2_results

    # ===== TEST C3: Procrustes-Aligned L8 at Injection Point =====
    print(f"\n--- C3: Procrustes-Aligned L8 → L{inject_layer} ---")

    # Compute Procrustes alignment from L8 to injection layer
    # Using all 200 problems as reference
    src = data["zh_L8"]  # (200, 2048)
    tgt = data[f"zh_L{inject_layer}"]  # (200, 2048)

    # Center
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    # Optimal rotation: R = V @ U^T where M = tgt^T @ src = U S V^T
    M = tgt_c.T @ src_c  # (2048, 2048)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt  # (2048, 2048)

    # Optimal scaling
    scale_proc = np.trace(np.diag(S)) / np.trace(src_c.T @ src_c)

    # Alignment quality
    aligned_src = scale_proc * (src_c @ R.T) + tgt_mean
    residual = tgt - aligned_src
    r2 = 1 - np.sum(residual**2) / np.sum((tgt - tgt.mean(axis=0))**2)
    print(f"  Procrustes R² = {r2:.4f}")
    print(f"  Scale factor = {scale_proc:.4f}")

    c3_results = {"r_squared": float(r2), "scale": float(scale_proc), "generations": []}

    for i in range(n_test):
        # Apply Procrustes to L8 hidden state
        h_aligned = scale_proc * ((h_L8_zh[i] - src_mean) @ R.T) + tgt_mean
        aligned_norm = np.linalg.norm(h_aligned)

        gen = generate_with_injection(prompt_ids_list[i], h_aligned, inject_layer, max_new_tokens=128)
        c3_results["generations"].append({
            "aligned_norm": float(aligned_norm),
            "natural_norm": float(np.linalg.norm(zh_L[i])),
            "generation": gen,
        })
        print(f"  Problem {i} (norm {aligned_norm:.1f} vs natural {np.linalg.norm(zh_L[i]):.1f}):")
        print(f"    {gen[:120]}...")

    layer_results["C3_procrustes"] = c3_results

    # ===== TEST C4: Raw Layer Skip =====
    print(f"\n--- C4: Raw L8 → L{inject_layer} (No Alignment) ---")
    c4_results = []

    for i in range(n_test):
        h_raw = h_L8_zh[i].copy()
        raw_norm = np.linalg.norm(h_raw)

        gen = generate_with_injection(prompt_ids_list[i], h_raw, inject_layer, max_new_tokens=128)
        c4_results.append({
            "raw_norm": float(raw_norm),
            "natural_norm": float(np.linalg.norm(zh_L[i])),
            "norm_ratio": float(raw_norm / np.linalg.norm(zh_L[i])),
            "generation": gen,
        })
        print(f"  Problem {i} (norm ratio {raw_norm / np.linalg.norm(zh_L[i]):.3f}):")
        print(f"    {gen[:120]}...")

    layer_results["C4_raw_skip"] = c4_results

    results["tests"][f"L{inject_layer}"] = layer_results

# Summary
print("\n\n" + "="*60)
print("EXPERIMENT C SUMMARY")
print("="*60)

for inject_layer in injection_layers:
    lr = results["tests"][f"L{inject_layer}"]
    print(f"\n--- L{inject_layer} ---")

    # C1 summary
    print("  C1 Norm Tolerance:")
    for r in lr["C1_norm_tolerance"]:
        gens = r["generations"]
        # Quick coherence check: average length
        avg_len = np.mean([len(g) for g in gens])
        print(f"    Scale {r['scale']:.2f} (norm {r['effective_norm']:.0f}): avg_len={avg_len:.0f}")

    # C3 summary
    print(f"  C3 Procrustes (R²={lr['C3_procrustes']['r_squared']:.4f}):")
    for i, g in enumerate(lr['C3_procrustes']['generations']):
        print(f"    Problem {i}: {g['generation'][:60]}...")

    # C4 summary
    print(f"  C4 Raw Skip:")
    for i, r in enumerate(lr['C4_raw_skip']):
        print(f"    Problem {i} (ratio {r['norm_ratio']:.3f}): {r['generation'][:60]}...")

# Norm calibration (from addendum)
print(f"\n--- Norm Calibration ---")
zh_L8_norms = np.linalg.norm(data["zh_L8"][:10], axis=1)
for L in injection_layers:
    zh_L_norms = np.linalg.norm(data[f"zh_L{L}"][:10], axis=1)
    ratio = zh_L8_norms.mean() / zh_L_norms.mean()
    print(f"  L8/L{L} norm ratio: {ratio:.3f}")
    print(f"  L8 norm ({zh_L8_norms.mean():.1f}) falls between C1 scale factors {ratio:.2f}")

# Save
with open("output/expC_basin_width.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expC_basin_width.json")
