"""Experiment H: The Patty Loop

Architecture: Top Bun (L0-L4) → Patty (L5-L8) → Bottom Bun (L27-L35)

The idea: run L5-L8 multiple times on the same hidden state. Each pass refines
reasoning in Z (the language-agnostic space). Then inject into L27 and generate.

Test H1: Does multi-pass improve output quality?
Test H2: What does the loop ADD? (PC0 projection, category alignment)
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

# Load cached activations for PC0 and category analysis
data = np.load('output/all_layers_lasttok.npz')

# Compute PC0 at L8 for projection analysis
zh_L8 = data['zh_L8']  # (200, 2048)
en_L8 = data['en_L8']  # (200, 2048)
all_L8 = np.concatenate([zh_L8, en_L8], axis=0)
mean_L8 = all_L8.mean(axis=0)
centered_L8 = all_L8 - mean_L8
_, _, Vt_L8 = np.linalg.svd(centered_L8, full_matrices=False)
pc0_L8 = Vt_L8[0]

# Also compute PC0 at L5 input space (= L4 output)
zh_L4 = data['zh_L4']
en_L4 = data['en_L4']
all_L4 = np.concatenate([zh_L4, en_L4], axis=0)
mean_L4 = all_L4.mean(axis=0)

# Compute category centroids at L8 (5 categories, 40 problems each)
# First 200 = zh, problems are ordered by category (40 per category)
category_names = ["arithmetic", "geometry", "algebra", "statistics", "combinatorics"]
category_centroids_L8 = {}
for cat_idx, cat_name in enumerate(category_names):
    start = cat_idx * 40
    end = start + 40
    # Use both zh and en for robust centroids
    cat_zh = zh_L8[start:end]
    cat_en = en_L8[start:end]
    cat_all = np.concatenate([cat_zh, cat_en], axis=0)
    category_centroids_L8[cat_name] = cat_all.mean(axis=0)

# Test problems (same as C4)
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
PATTY_LAYERS = list(range(5, 9))  # L5, L6, L7, L8
MAX_LOOPS = 4


# =============================================================================
# H1: Extract multi-pass hidden states
# =============================================================================
def run_patty_loops(prompt, n_loops):
    """
    Run L0-L8 normally to get h_L8 (1-pass).
    Then feed h_L8 back through L5-L8 for additional passes.

    Returns h_L8 after n_loops total passes through the patty.
    Also returns intermediate states for analysis.
    """
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    # Step 1: Run full forward pass through L0-L8, capture h_L4 and h_L8
    captured = {}

    def hook_L4(module, input, output):
        hidden = output if not isinstance(output, tuple) else output[0]
        captured['h_L4'] = hidden.clone()

    def hook_L8(module, input, output):
        hidden = output if not isinstance(output, tuple) else output[0]
        captured['h_L8'] = hidden.clone()

    handle_L4 = model.model.layers[4].register_forward_hook(hook_L4)
    handle_L8 = model.model.layers[8].register_forward_hook(hook_L8)

    # Run just the embedding + first 9 layers (L0-L8)
    # We use a hook on L8 to capture and then stop (but we need KV cache for L0-L4)
    # Actually: just run the full model forward, capture h_L8, ignore the rest
    with torch.no_grad():
        model(input_tensor, use_cache=False)

    handle_L4.remove()
    handle_L8.remove()

    h_L4_original = captured['h_L4']  # (1, seq, 2048)
    h_states = [captured['h_L8'][:, -1:, :].clone()]  # 1-pass h_L8 (last token only)

    if n_loops <= 1:
        return h_states

    # Step 2: For additional loops, feed h_L8 back through L5-L8
    # The tricky part: L5-L8 have attention that needs KV context.
    # We provide the original L0-L4 KV cache context, but the "current token"
    # hidden state is our looped h_L8.
    #
    # Strategy: Run L5-L8 manually. For each layer, we need to:
    # 1. Apply the layer's self-attention (using the layer's own forward)
    # 2. Apply the layer's MLP
    # But the attention needs position embeddings and potentially KV cache.
    #
    # Simpler approach: Hook L5's input to inject our state, run full forward.
    # Layers 0-4 will compute normally (wasted), L5 gets our injected state.

    current_h = captured['h_L8'][:, -1:, :].clone()  # (1, 1, 2048) — last token

    for loop in range(1, n_loops):
        # Inject current_h at L5 input (= replace L4 output for last token position)
        # and capture new L8 output
        loop_captured = {}
        inject_state = current_h.clone()

        def make_inject_hook(state):
            fired = [False]
            def hook_fn(module, input, output):
                if not fired[0]:
                    hidden = output if not isinstance(output, tuple) else output[0]
                    # Replace last token with our looped state
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

        h_inject = model.model.layers[4].register_forward_hook(make_inject_hook(inject_state))
        h_capture = model.model.layers[8].register_forward_hook(make_capture_hook(loop_captured))

        with torch.no_grad():
            model(input_tensor, use_cache=False)

        h_inject.remove()
        h_capture.remove()

        current_h = loop_captured['h'].clone()  # (1, 1, 2048)
        h_states.append(current_h.clone())

    return h_states


def generate_with_injection_at_L27(prompt, h_vector_np, max_new_tokens=MAX_NEW_TOKENS):
    """Generate by injecting h_vector at L27 (same as C4 procedure)."""
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
# TEST H1: Does double/triple/quad-cooking help?
# =============================================================================
print("=" * 70)
print("EXPERIMENT H1: PATTY LOOP — MULTI-PASS REASONING")
print("=" * 70)

results = {
    "experiment": "H: The Patty Loop",
    "H1": {"description": "Multi-pass L5-L8 reasoning, inject at L27", "problems": []},
    "H2": {"description": "What does the loop add? PC0, category analysis", "problems": []},
}

for i, (prompt, expected, category) in enumerate(all_problems):
    ptype = "simple" if i < 5 else "hard"
    print(f"\n{'='*60}")
    print(f"Problem {i} [{ptype.upper()}] ({category}): {prompt.strip()[:50]}...")
    print(f"Expected: {expected}")

    # Run patty with 1-4 loops
    h_states = run_patty_loops(prompt, MAX_LOOPS)

    problem_result = {
        "idx": i,
        "type": ptype,
        "category": category,
        "prompt": prompt.strip(),
        "expected": expected,
        "passes": [],
        "convergence": [],
    }

    # Generate for each pass count
    for n_pass in range(len(h_states)):
        h = h_states[n_pass]  # (1, 1, 2048) tensor
        h_np = h[0, 0, :].cpu().float().numpy()  # (2048,)

        gen, n_tok = generate_with_injection_at_L27(prompt, h_np)
        correct = expected in gen
        chinese = is_chinese(gen)

        print(f"  {n_pass+1}-pass: correct={correct} | Chinese={chinese} | tokens={n_tok}")
        print(f"    {gen[:80]}...")

        problem_result["passes"].append({
            "n_pass": n_pass + 1,
            "generation": gen,
            "correct": correct,
            "is_chinese": chinese,
            "n_tokens": n_tok,
            "h_norm": float(np.linalg.norm(h_np)),
        })

    # Convergence: cosine similarity between successive passes
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

    results["H1"]["problems"].append(problem_result)

# H1 Summary
print(f"\n{'='*70}")
print("H1 SUMMARY")
print("="*70)
for n in range(MAX_LOOPS):
    correct_simple = sum(1 for p in results["H1"]["problems"][:5]
                        if len(p["passes"]) > n and p["passes"][n]["correct"])
    correct_hard = sum(1 for p in results["H1"]["problems"][5:]
                      if len(p["passes"]) > n and p["passes"][n]["correct"])
    total = correct_simple + correct_hard
    print(f"  {n+1}-pass: {correct_simple}/5 simple, {correct_hard}/5 hard = {total}/10")

# Average convergence per step
for step in range(MAX_LOOPS - 1):
    cos_vals = [p["convergence"][step]["cosine_similarity"]
                for p in results["H1"]["problems"]
                if len(p["convergence"]) > step]
    if cos_vals:
        print(f"  Convergence {step+1}→{step+2}: mean cos={np.mean(cos_vals):.6f}, min={np.min(cos_vals):.6f}")


# =============================================================================
# TEST H2: What does the loop ADD?
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT H2: LOOP ANALYSIS — PC0, CATEGORY, LANGUAGE")
print("="*70)

for i, (prompt, expected, category) in enumerate(all_problems):
    h_states = run_patty_loops(prompt, 2)  # Only need 1-pass and 2-pass

    h_1pass = h_states[0][0, 0, :].cpu().float().numpy()
    h_2pass = h_states[1][0, 0, :].cpu().float().numpy()
    diff = h_2pass - h_1pass

    # PC0 projection
    pc0_proj_1pass = float(np.dot(h_1pass - mean_L8, pc0_L8))
    pc0_proj_2pass = float(np.dot(h_2pass - mean_L8, pc0_L8))
    pc0_proj_diff = float(np.dot(diff, pc0_L8))

    # Category alignment: cosine of diff with each category centroid
    cat_alignments = {}
    for cat_name, centroid in category_centroids_L8.items():
        cat_dir = centroid - mean_L8
        cos_with_cat = float(np.dot(diff, cat_dir) / (np.linalg.norm(diff) * np.linalg.norm(cat_dir) + 1e-10))
        cat_alignments[cat_name] = cos_with_cat

    # Language content of the difference vector
    # Project onto top 5 PCs and check if language signal changed
    diff_norm = float(np.linalg.norm(diff))
    h1_norm = float(np.linalg.norm(h_1pass))
    relative_change = diff_norm / h1_norm if h1_norm > 0 else 0

    best_cat = max(cat_alignments, key=cat_alignments.get)

    print(f"\n  Problem {i} [{category}]: {prompt.strip()[:40]}...")
    print(f"    PC0 change: {pc0_proj_1pass:.2f} → {pc0_proj_2pass:.2f} (Δ={pc0_proj_diff:.2f})")
    print(f"    Diff norm: {diff_norm:.2f} ({relative_change*100:.1f}% of h_1pass)")
    print(f"    Best category alignment: {best_cat} ({cat_alignments[best_cat]:.3f})")
    print(f"    Target category: {category} (alignment={cat_alignments[category]:.3f})")

    results["H2"]["problems"].append({
        "idx": i,
        "category": category,
        "prompt": prompt.strip(),
        "pc0_1pass": pc0_proj_1pass,
        "pc0_2pass": pc0_proj_2pass,
        "pc0_delta": pc0_proj_diff,
        "diff_norm": diff_norm,
        "relative_change_pct": relative_change * 100,
        "category_alignments": cat_alignments,
        "best_aligned_category": best_cat,
        "target_category_alignment": cat_alignments[category],
        "language_neutral": abs(pc0_proj_diff) < 5.0,  # threshold: small PC0 change
    })

# H2 Summary
pc0_deltas = [p["pc0_delta"] for p in results["H2"]["problems"]]
rel_changes = [p["relative_change_pct"] for p in results["H2"]["problems"]]
correct_cat = sum(1 for p in results["H2"]["problems"] if p["best_aligned_category"] == p["category"])
lang_neutral = sum(1 for p in results["H2"]["problems"] if p["language_neutral"])

print(f"\n  H2 Summary:")
print(f"    Mean |PC0 delta|: {np.mean(np.abs(pc0_deltas)):.2f} (language change)")
print(f"    Mean relative change: {np.mean(rel_changes):.1f}%")
print(f"    Category alignment correct: {correct_cat}/10")
print(f"    Language neutral (|ΔPC0|<5): {lang_neutral}/10")

results["H2"]["summary"] = {
    "mean_abs_pc0_delta": float(np.mean(np.abs(pc0_deltas))),
    "mean_relative_change_pct": float(np.mean(rel_changes)),
    "category_alignment_correct": correct_cat,
    "language_neutral_count": lang_neutral,
    "total": 10,
}

# Save
with open("output/expH_patty_loop.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expH_patty_loop.json")
