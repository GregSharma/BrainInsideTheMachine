"""Experiment H2b: Patty Loop with Procrustes Alignment

The raw patty loop (H1) showed convergence but no improvement — likely because
L8 output lives in a different subspace than L5 expects as input.

This test: align L8 output to L4 output space via Procrustes before looping.
If the aligned loop improves reasoning, the patty concept is validated.
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

data = np.load('output/all_layers_lasttok.npz')

# Compute Procrustes alignment: L8 → L4 space
# Use all 400 samples (200 zh + 200 en) to learn the rotation
zh_L4 = data['zh_L4']  # (200, 2048)
en_L4 = data['en_L4']
zh_L8 = data['zh_L8']
en_L8 = data['en_L8']

all_L4 = np.concatenate([zh_L4, en_L4], axis=0)  # (400, 2048)
all_L8 = np.concatenate([zh_L8, en_L8], axis=0)

# Center both
mean_L4 = all_L4.mean(axis=0)
mean_L8 = all_L8.mean(axis=0)
centered_L4 = all_L4 - mean_L4
centered_L8 = all_L8 - mean_L8

# Compute norm ratio for scaling
norm_L4 = np.linalg.norm(centered_L4, axis=1).mean()
norm_L8 = np.linalg.norm(centered_L8, axis=1).mean()
scale_factor = norm_L4 / norm_L8

print(f"Norm ratio L4/L8: {scale_factor:.4f}")
print(f"Mean norm L4: {norm_L4:.1f}, L8: {norm_L8:.1f}")

# Orthogonal Procrustes: find R that minimizes ||centered_L4 - centered_L8 @ R||
R, sval = orthogonal_procrustes(centered_L8, centered_L4)
# R is (2048, 2048) orthogonal matrix

# Verify alignment quality
aligned_L8 = centered_L8 @ R
residual = np.linalg.norm(centered_L4 - aligned_L8) / np.linalg.norm(centered_L4)
print(f"Procrustes residual (lower = better alignment): {residual:.4f}")

# Also compute layernorm stats for L4 (alternative to Procrustes)
ln_mean_L4 = all_L4.mean()
ln_std_L4 = all_L4.std()
ln_mean_L8 = all_L8.mean()
ln_std_L8 = all_L8.std()

# PC0 at L8 for analysis
all_L8_full = np.concatenate([zh_L8, en_L8], axis=0)
mean_L8_full = all_L8_full.mean(axis=0)
centered_L8_full = all_L8_full - mean_L8_full
_, _, Vt_L8 = np.linalg.svd(centered_L8_full, full_matrices=False)
pc0_L8 = Vt_L8[0]

# Test problems
simple_problems = [
    ("请计算 2 + 3 × 4 的值。\n", "14", "arithmetic"),
    ("一个矩形的长为8厘米，宽为5厘米，求面积。\n", "40", "geometry"),
    ("如果 x + 5 = 12，求 x 的值。\n", "7", "algebra"),
    ("计算 100 除以 4 的结果。\n", "25", "arithmetic"),
    ("一个三角形三边长分别为3、4、5，求面积。\n", "6", "geometry"),
]
hard_problems = [
    ("如果 2x + 3 = 15，求 x² 的值。\n", "36", "algebra"),
    ("已知 x + y = 10，x - y = 4，求 x × y 的值。\n", "21", "algebra"),
    ("小明有50元钱，买了3本书每本8元，又买了2支笔每支3元，还剩多少元？\n", "20", "arithmetic"),
    ("一个正方形的对角线长为10厘米，求这个正方形的面积。\n", "50", "geometry"),
    ("一个班有40个学生，男生占总数的3/5，女生有多少人？\n", "16", "arithmetic"),
]
all_problems = simple_problems + hard_problems

MAX_NEW_TOKENS = 128
MAX_LOOPS = 4

# Convert R to torch tensor for GPU use
R_torch = torch.tensor(R, dtype=torch.bfloat16, device=device)
mean_L8_torch = torch.tensor(mean_L8, dtype=torch.bfloat16, device=device)
mean_L4_torch = torch.tensor(mean_L4, dtype=torch.bfloat16, device=device)


def run_aligned_patty_loops(prompt, n_loops):
    """
    Run patty loop with Procrustes alignment on the feedback path.
    L8 output → center → rotate via R → uncenter to L4 space → feed to L5.
    """
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    captured = {}

    def hook_L4(module, input, output):
        hidden = output if not isinstance(output, tuple) else output[0]
        captured['h_L4'] = hidden.clone()

    def hook_L8(module, input, output):
        hidden = output if not isinstance(output, tuple) else output[0]
        captured['h_L8'] = hidden.clone()

    handle_L4 = model.model.layers[4].register_forward_hook(hook_L4)
    handle_L8 = model.model.layers[8].register_forward_hook(hook_L8)

    with torch.no_grad():
        model(input_tensor, use_cache=False)

    handle_L4.remove()
    handle_L8.remove()

    h_states = [captured['h_L8'][:, -1:, :].clone()]
    current_h = captured['h_L8'][:, -1:, :].clone()  # (1, 1, 2048)

    if n_loops <= 1:
        return h_states

    for loop in range(1, n_loops):
        # Align current_h (L8 space) → L4 space via Procrustes
        aligned_h = (current_h - mean_L8_torch) @ R_torch + mean_L4_torch

        loop_captured = {}

        def make_inject_hook(state):
            fired = [False]
            def hook_fn(module, input, output):
                if not fired[0]:
                    hidden = output if not isinstance(output, tuple) else output[0]
                    hidden[0, -1:, :] = state[0, :, :]
                    fired[0] = True
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
            return hook_fn

        def make_capture_hook(store):
            fired = [False]
            def hook_fn(module, input, output):
                if not fired[0]:
                    hidden = output if not isinstance(output, tuple) else output[0]
                    store['h'] = hidden[:, -1:, :].clone()
                    fired[0] = True
            return hook_fn

        # Inject the ALIGNED state at L4 (so L5 sees it naturally)
        h_inject = model.model.layers[4].register_forward_hook(make_inject_hook(aligned_h))
        h_capture = model.model.layers[8].register_forward_hook(make_capture_hook(loop_captured))

        with torch.no_grad():
            model(input_tensor, use_cache=False)

        h_inject.remove()
        h_capture.remove()

        current_h = loop_captured['h'].clone()
        h_states.append(current_h.clone())

    return h_states


def generate_with_injection_at_L27(prompt, h_vector_np, max_new_tokens=MAX_NEW_TOKENS):
    """Generate by injecting h_vector at L27."""
    input_ids = tokenizer.encode(prompt)
    injected = [False]

    def hook_fn(module, input, output):
        if not injected[0]:
            hidden = output if not isinstance(output, tuple) else output[0]
            vec = torch.tensor(h_vector_np, dtype=hidden.dtype, device=hidden.device)
            hidden[0, -1, :] = vec
            injected[0] = True
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return output

    handle = model.model.layers[27].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([input_ids], device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
        n_tokens = len(outputs[0]) - len(input_ids)
    finally:
        handle.remove()
    return gen, n_tokens


def is_chinese(text):
    zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_chars = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
    return zh_chars > en_chars


# =============================================================================
# Run aligned patty loop
# =============================================================================
print("=" * 70)
print("EXPERIMENT H2b: PROCRUSTES-ALIGNED PATTY LOOP")
print("=" * 70)

results = {
    "experiment": "H2b: Procrustes-Aligned Patty Loop",
    "procrustes_residual": float(residual),
    "scale_factor": float(scale_factor),
    "norm_L4": float(norm_L4),
    "norm_L8": float(norm_L8),
    "problems": [],
}

for i, (prompt, expected, category) in enumerate(all_problems):
    ptype = "simple" if i < 5 else "hard"
    print(f"\n{'='*60}")
    print(f"Problem {i} [{ptype.upper()}] ({category}): {prompt.strip()[:50]}...")
    print(f"Expected: {expected}")

    h_states = run_aligned_patty_loops(prompt, MAX_LOOPS)

    problem_result = {
        "idx": i,
        "type": ptype,
        "category": category,
        "prompt": prompt.strip(),
        "expected": expected,
        "passes": [],
        "convergence": [],
    }

    for n_pass in range(len(h_states)):
        h = h_states[n_pass]
        h_np = h[0, 0, :].cpu().float().numpy()

        gen, n_tok = generate_with_injection_at_L27(prompt, h_np)
        correct = expected in gen
        chinese = is_chinese(gen)

        # PC0 projection
        pc0_proj = float(np.dot(h_np - mean_L8_full, pc0_L8))

        print(f"  {n_pass+1}-pass: correct={correct} | Chinese={chinese} | PC0={pc0_proj:.2f} | tokens={n_tok}")
        print(f"    {gen[:80]}...")

        problem_result["passes"].append({
            "n_pass": n_pass + 1,
            "generation": gen,
            "correct": correct,
            "is_chinese": chinese,
            "n_tokens": n_tok,
            "h_norm": float(np.linalg.norm(h_np)),
            "pc0_projection": pc0_proj,
        })

    # Convergence
    for j in range(1, len(h_states)):
        h_prev = h_states[j-1][0, 0, :].cpu().float().numpy()
        h_curr = h_states[j][0, 0, :].cpu().float().numpy()
        cos_sim = float(np.dot(h_prev, h_curr) / (np.linalg.norm(h_prev) * np.linalg.norm(h_curr) + 1e-10))
        problem_result["convergence"].append({
            "from_pass": j,
            "to_pass": j + 1,
            "cosine_similarity": cos_sim,
        })
        print(f"  Convergence {j}→{j+1}: cos={cos_sim:.6f}")

    results["problems"].append(problem_result)

# Summary
print(f"\n{'='*70}")
print("H2b SUMMARY — ALIGNED PATTY LOOP")
print("="*70)
for n in range(MAX_LOOPS):
    correct_simple = sum(1 for p in results["problems"][:5]
                        if len(p["passes"]) > n and p["passes"][n]["correct"])
    correct_hard = sum(1 for p in results["problems"][5:]
                      if len(p["passes"]) > n and p["passes"][n]["correct"])
    total = correct_simple + correct_hard

    # Average PC0
    pc0_vals = [p["passes"][n]["pc0_projection"] for p in results["problems"]
                if len(p["passes"]) > n]
    mean_pc0 = np.mean(pc0_vals) if pc0_vals else 0

    print(f"  {n+1}-pass: {correct_simple}/5 simple, {correct_hard}/5 hard = {total}/10 | mean PC0={mean_pc0:.2f}")

for step in range(MAX_LOOPS - 1):
    cos_vals = [p["convergence"][step]["cosine_similarity"]
                for p in results["problems"]
                if len(p["convergence"]) > step]
    if cos_vals:
        print(f"  Convergence {step+1}→{step+2}: mean cos={np.mean(cos_vals):.6f}")

# Save
with open("output/expH2b_patty_aligned.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expH2b_patty_aligned.json")
