"""
Diffusion Map on L32 Generation-Time Trajectories

The 200x path ratio is measured in R^2048 — the wrong metric space.
This computes the INTRINSIC geometry of the reasoning manifold.

Algorithm:
1. Extract h32 at every generation step (re-run generation, save full vectors)
2. Build Gaussian kernel: K(h^(t), h^(s)) = exp(-||h^(t) - h^(s)||^2 / eps)
3. Normalize to transition matrix (Markov chain on the data)
4. Eigendecompose — top eigenvectors give diffusion coordinates
5. Embed trajectory in 3D diffusion space
6. Compare zh vs en for same problem — "One Tape" test

If zh and en align in diffusion coordinates: same reasoning path, different tokens.
If path ratio drops from 200x to 5x: manifold is smooth, R^2048 was misleading.
If path ratio stays 200x: reasoning is genuinely geometrically expensive.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random as pyrandom
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

print(f"Model: {N_LAYERS} layers, d={D}")

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

# Same 5 problems as vocab_bottleneck.py
cat_examples = {}
for i, p in enumerate(problems):
    c = p['category']
    if c not in cat_examples:
        cat_examples[c] = i
    if len(cat_examples) == 5:
        break
test_indices = sorted(cat_examples.values())
CAT_NAMES = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]
langs = ['zh', 'en']

print(f"Test problems: {test_indices}")


# ---------- EXTRACT TRAJECTORIES ----------
def extract_trajectory(model, tokenizer, prompt, max_new_tokens=256):
    """Extract h32 at every generation step. Returns array (n_gen_steps, 2048)."""
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

    gen_ids = inputs.input_ids.clone()
    all_h32 = []
    tokens = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(gen_ids)
            all_h32.append(hook_data[REASONING_LAYER].flatten().copy())

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            gen_ids = torch.cat([gen_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    hook_handle.remove()

    # Return only generation-phase h32 (skip prompt processing steps)
    all_h32 = np.array(all_h32)  # (total_steps, 2048)
    gen_h32 = all_h32[input_len:]  # only generation steps
    gen_tokens = tokens  # tokens are already generation-only

    text = tokenizer.decode(gen_ids[0, input_len:], skip_special_tokens=True)
    return gen_h32, gen_tokens, text


# ---------- DIFFUSION MAP ----------
def diffusion_map(X, n_components=3, epsilon=None, alpha=1.0):
    """
    Compute diffusion map embedding.

    X: (n_samples, n_features) — the trajectory points
    n_components: number of diffusion coordinates to return
    epsilon: kernel bandwidth. If None, use median pairwise distance.
    alpha: normalization parameter (1.0 = Laplace-Beltrami approximation)

    Returns:
        coords: (n_samples, n_components) — diffusion coordinates
        eigenvalues: top eigenvalues (excluding trivial λ=1)
        epsilon_used: the epsilon that was used
    """
    n = X.shape[0]

    # Pairwise squared distances
    D2 = squareform(pdist(X, 'sqeuclidean'))

    # Choose epsilon if not provided
    if epsilon is None:
        # Median of pairwise distances (not squared)
        pairwise_dists = pdist(X, 'euclidean')
        epsilon = np.median(pairwise_dists) ** 2
        print(f"  Auto epsilon (median²): {epsilon:.2f}")

    # Gaussian kernel
    K = np.exp(-D2 / epsilon)

    # Alpha-normalization (density correction)
    # q(x) = sum_y K(x,y)
    q = K.sum(axis=1)
    # K_alpha(x,y) = K(x,y) / (q(x)^alpha * q(y)^alpha)
    if alpha > 0:
        q_alpha = q ** alpha
        K = K / np.outer(q_alpha, q_alpha)

    # Row-normalize to get Markov transition matrix
    row_sums = K.sum(axis=1)
    T = K / row_sums[:, np.newaxis]

    # Eigendecompose the symmetric matrix D^{-1/2} T D^{1/2}
    # This gives same eigenvalues as T but symmetric for numerical stability
    D_sqrt = np.sqrt(row_sums)
    D_sqrt_inv = 1.0 / D_sqrt
    T_sym = (T * D_sqrt[np.newaxis, :]) * D_sqrt_inv[:, np.newaxis]
    # Make exactly symmetric
    T_sym = 0.5 * (T_sym + T_sym.T)

    # Get top eigenvalues/vectors
    n_eig = min(n_components + 1, n - 1)
    eigenvalues, eigenvectors = eigh(T_sym, subset_by_index=[n - n_eig, n - 1])

    # Sort descending
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip the first (trivial, λ≈1) eigenvector
    eigenvalues = eigenvalues[1:n_components+1]
    eigenvectors = eigenvectors[:, 1:n_components+1]

    # Convert back from symmetric to original basis
    coords = eigenvectors * D_sqrt_inv[:, np.newaxis]

    # Scale by eigenvalue (diffusion distance = Euclidean in scaled coords)
    coords = coords * eigenvalues[np.newaxis, :]

    return coords, eigenvalues, epsilon


def compute_path_metrics(coords):
    """Compute path ratio and consecutive cosines in diffusion coordinates."""
    n = coords.shape[0]
    if n < 3:
        return {'path_ratio': float('nan'), 'mean_consec_cos': float('nan')}

    # Path length and straight-line distance
    steps = np.diff(coords, axis=0)  # (n-1, d)
    step_norms = np.linalg.norm(steps, axis=1)
    path_length = step_norms.sum()
    straight_line = np.linalg.norm(coords[-1] - coords[0])
    path_ratio = path_length / straight_line if straight_line > 1e-12 else float('inf')

    # Consecutive cosines
    cosines = []
    for t in range(len(steps) - 1):
        d1 = steps[t]
        d2 = steps[t + 1]
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 > 1e-12 and n2 > 1e-12:
            cosines.append(float(np.dot(d1, d2) / (n1 * n2)))

    return {
        'path_ratio': float(path_ratio),
        'path_length': float(path_length),
        'straight_line': float(straight_line),
        'mean_consec_cos': float(np.mean(cosines)) if cosines else float('nan'),
        'std_consec_cos': float(np.std(cosines)) if cosines else float('nan'),
        'frac_anticorrelated': float(np.mean([c < 0 for c in cosines])) if cosines else float('nan'),
        'n_steps': n,
    }


# ---------- MAIN: EXTRACT + DIFFUSION MAP + PLOT ----------
all_results = {}
all_trajectories = {}  # save raw for cross-problem analysis

for prob_idx in test_indices:
    prob = problems[prob_idx]
    cat_name = CAT_NAMES[prob['category']]
    print(f"\n{'='*60}")
    print(f"Problem {prob_idx} ({cat_name})")
    print(f"  zh: {prob['zh']}")
    print(f"  en: {prob['en']}")

    traj_data = {}
    for lang in langs:
        print(f"\n  Extracting {lang} trajectory...")
        h32_traj, tokens, text = extract_trajectory(model, tokenizer, prob[lang])
        print(f"    {h32_traj.shape[0]} generation steps, d={h32_traj.shape[1]}")
        print(f"    Text: {text[:100]}...")
        traj_data[lang] = {
            'h32': h32_traj,
            'tokens': tokens,
            'text': text,
        }

        # R^2048 metrics for comparison
        r2048_metrics = compute_path_metrics(h32_traj)
        print(f"    R^2048: path_ratio={r2048_metrics['path_ratio']:.1f}x, "
              f"consec_cos={r2048_metrics['mean_consec_cos']:.3f}, "
              f"frac_anti={r2048_metrics['frac_anticorrelated']:.3f}")

    # Skip problems with too few generation steps
    min_steps = min(traj_data[l]['h32'].shape[0] for l in langs)
    if min_steps < 5:
        print(f"  Skipping — too few steps ({min_steps})")
        continue

    all_trajectories[prob_idx] = traj_data

    # === DIFFUSION MAP: Per-language (each trajectory independently) ===
    fig_per_lang, axes = plt.subplots(1, 2, figsize=(16, 7),
                                       subplot_kw={'projection': '3d'})

    for i, lang in enumerate(langs):
        h32 = traj_data[lang]['h32']
        n_steps = h32.shape[0]

        if n_steps < 10:
            print(f"  {lang}: only {n_steps} steps, skipping diffusion map")
            continue

        print(f"\n  Diffusion map ({lang}, {n_steps} points)...")
        n_comp = min(3, n_steps - 2)
        coords, evals, eps = diffusion_map(h32, n_components=n_comp)
        diff_metrics = compute_path_metrics(coords)

        r2048_metrics = compute_path_metrics(h32)

        print(f"    Eigenvalues: {evals}")
        print(f"    Diffusion: path_ratio={diff_metrics['path_ratio']:.1f}x, "
              f"consec_cos={diff_metrics['mean_consec_cos']:.3f}, "
              f"frac_anti={diff_metrics['frac_anticorrelated']:.3f}")
        print(f"    R^2048:    path_ratio={r2048_metrics['path_ratio']:.1f}x, "
              f"consec_cos={r2048_metrics['mean_consec_cos']:.3f}")
        print(f"    *** Path ratio change: {r2048_metrics['path_ratio']:.1f}x → "
              f"{diff_metrics['path_ratio']:.1f}x "
              f"({diff_metrics['path_ratio']/r2048_metrics['path_ratio']*100:.0f}% of R^2048)")

        all_results[f"prob{prob_idx}_{lang}"] = {
            'problem_idx': prob_idx,
            'category': cat_name,
            'language': lang,
            'n_steps': n_steps,
            'epsilon': float(eps),
            'eigenvalues': evals.tolist(),
            'r2048_path_ratio': r2048_metrics['path_ratio'],
            'r2048_consec_cos': r2048_metrics['mean_consec_cos'],
            'r2048_frac_anti': r2048_metrics['frac_anticorrelated'],
            'diffusion_path_ratio': diff_metrics['path_ratio'],
            'diffusion_consec_cos': diff_metrics['mean_consec_cos'],
            'diffusion_frac_anti': diff_metrics['frac_anticorrelated'],
            'path_ratio_reduction': diff_metrics['path_ratio'] / r2048_metrics['path_ratio'],
        }

        # Plot trajectory in diffusion coordinates
        ax = axes[i]
        if coords.shape[1] >= 3:
            t_color = np.linspace(0, 1, n_steps)
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                    alpha=0.3, color='gray', linewidth=0.5)
            sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=t_color, cmap='viridis', s=15, alpha=0.8)
            ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2],
                      color='red', s=100, marker='*', label='start')
            ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2],
                      color='blue', s=100, marker='s', label='end')
        ax.set_title(f"{lang.upper()} ({cat_name})\n"
                    f"R²⁰⁴⁸: {r2048_metrics['path_ratio']:.0f}x → "
                    f"Diff: {diff_metrics['path_ratio']:.1f}x\n"
                    f"cos: {r2048_metrics['mean_consec_cos']:.3f} → "
                    f"{diff_metrics['mean_consec_cos']:.3f}")
        ax.set_xlabel('φ₁')
        ax.set_ylabel('φ₂')
        ax.set_zlabel('φ₃')
        ax.legend()

    fig_per_lang.suptitle(f"Problem {prob_idx} ({cat_name}) — Diffusion Map of L32 Trajectory",
                          fontsize=14)
    fig_per_lang.tight_layout()
    fig_per_lang.savefig(f'output/diffusion_map_prob{prob_idx}.png', dpi=150)
    plt.close(fig_per_lang)
    print(f"  Saved: output/diffusion_map_prob{prob_idx}.png")

    # === JOINT DIFFUSION MAP: Both languages embedded together ===
    # This is the "One Tape" test — do zh and en follow the same diffusion path?
    h32_zh = traj_data['zh']['h32']
    h32_en = traj_data['en']['h32']
    n_zh, n_en = h32_zh.shape[0], h32_en.shape[0]

    if n_zh >= 10 and n_en >= 10:
        print(f"\n  Joint diffusion map ({n_zh} zh + {n_en} en = {n_zh + n_en} points)...")
        h32_joint = np.vstack([h32_zh, h32_en])
        n_comp = min(3, n_zh + n_en - 2)
        coords_joint, evals_joint, eps_joint = diffusion_map(
            h32_joint, n_components=n_comp
        )
        coords_zh = coords_joint[:n_zh]
        coords_en = coords_joint[n_zh:]

        # Plot joint embedding
        fig_joint = plt.figure(figsize=(12, 10))
        ax = fig_joint.add_subplot(111, projection='3d')

        if coords_joint.shape[1] >= 3:
            # Chinese trajectory
            t_zh = np.linspace(0, 1, n_zh)
            ax.plot(coords_zh[:, 0], coords_zh[:, 1], coords_zh[:, 2],
                    alpha=0.3, color='red', linewidth=0.5)
            ax.scatter(coords_zh[:, 0], coords_zh[:, 1], coords_zh[:, 2],
                      c=t_zh, cmap='Reds', s=12, alpha=0.7, label='Chinese')
            ax.scatter(coords_zh[0, 0], coords_zh[0, 1], coords_zh[0, 2],
                      color='darkred', s=100, marker='*')
            ax.scatter(coords_zh[-1, 0], coords_zh[-1, 1], coords_zh[-1, 2],
                      color='darkred', s=100, marker='s')

            # English trajectory
            t_en = np.linspace(0, 1, n_en)
            ax.plot(coords_en[:, 0], coords_en[:, 1], coords_en[:, 2],
                    alpha=0.3, color='blue', linewidth=0.5)
            ax.scatter(coords_en[:, 0], coords_en[:, 1], coords_en[:, 2],
                      c=t_en, cmap='Blues', s=12, alpha=0.7, label='English')
            ax.scatter(coords_en[0, 0], coords_en[0, 1], coords_en[0, 2],
                      color='darkblue', s=100, marker='*')
            ax.scatter(coords_en[-1, 0], coords_en[-1, 1], coords_en[-1, 2],
                      color='darkblue', s=100, marker='s')

        # Measure alignment: for each zh step (fractionally), find nearest en point
        # by progress fraction and compute distance
        zh_fracs = np.linspace(0, 1, n_zh)
        en_fracs = np.linspace(0, 1, n_en)
        alignment_dists = []
        for t, frac in enumerate(zh_fracs):
            en_idx = np.argmin(np.abs(en_fracs - frac))
            d = np.linalg.norm(coords_zh[t] - coords_en[en_idx])
            alignment_dists.append(d)

        mean_align = np.mean(alignment_dists)
        # Compare to average spread of each trajectory
        zh_spread = np.mean(np.linalg.norm(coords_zh - coords_zh.mean(axis=0), axis=1))
        en_spread = np.mean(np.linalg.norm(coords_en - coords_en.mean(axis=0), axis=1))
        align_ratio = mean_align / ((zh_spread + en_spread) / 2)

        ax.set_title(f"Problem {prob_idx} ({cat_name}) — JOINT Diffusion Map\n"
                    f"Alignment: mean_dist={mean_align:.4f}, "
                    f"spread={zh_spread:.4f}/{en_spread:.4f}, "
                    f"ratio={align_ratio:.3f}\n"
                    f"(ratio < 1 = trajectories overlap; > 1 = separate)")
        ax.set_xlabel('φ₁')
        ax.set_ylabel('φ₂')
        ax.set_zlabel('φ₃')
        ax.legend()

        fig_joint.tight_layout()
        fig_joint.savefig(f'output/diffusion_joint_prob{prob_idx}.png', dpi=150)
        plt.close(fig_joint)
        print(f"  Saved: output/diffusion_joint_prob{prob_idx}.png")

        all_results[f"prob{prob_idx}_joint"] = {
            'problem_idx': prob_idx,
            'category': cat_name,
            'n_zh': n_zh,
            'n_en': n_en,
            'joint_eigenvalues': evals_joint.tolist(),
            'joint_epsilon': float(eps_joint),
            'alignment_mean_dist': float(mean_align),
            'zh_spread': float(zh_spread),
            'en_spread': float(en_spread),
            'alignment_ratio': float(align_ratio),
        }

        print(f"    Joint eigenvalues: {evals_joint}")
        print(f"    Alignment ratio: {align_ratio:.3f} "
              f"({'OVERLAPPING' if align_ratio < 0.5 else 'SEPARATE' if align_ratio > 1.5 else 'PARTIAL'})")


# ---------- EPSILON SWEEP ----------
# For one problem, sweep epsilon to verify stability
print("\n" + "=" * 60)
print("EPSILON SWEEP (checking stability)")
# Pick the problem with most generation steps
best_prob = max(all_trajectories.keys(),
                key=lambda k: all_trajectories[k]['zh']['h32'].shape[0])
h32_sweep = all_trajectories[best_prob]['zh']['h32']
pairwise = pdist(h32_sweep, 'euclidean')
median_dist = np.median(pairwise)

sweep_results = []
for eps_mult in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]:
    eps = (median_dist * eps_mult) ** 2
    coords, evals, _ = diffusion_map(h32_sweep, n_components=3, epsilon=eps)
    metrics = compute_path_metrics(coords)
    sweep_results.append({
        'eps_mult': eps_mult,
        'epsilon': float(eps),
        'path_ratio': metrics['path_ratio'],
        'consec_cos': metrics['mean_consec_cos'],
        'frac_anti': metrics['frac_anticorrelated'],
        'eigenvalues': evals.tolist(),
    })
    print(f"  ε = {eps_mult:.2f}× median²: path_ratio={metrics['path_ratio']:.1f}x, "
          f"cos={metrics['mean_consec_cos']:.3f}, λ={evals}")

all_results['epsilon_sweep'] = sweep_results


# ---------- SAVE ----------
# Save raw trajectories for later analysis
traj_save = {}
for prob_idx, tdata in all_trajectories.items():
    for lang in langs:
        key = f"prob{prob_idx}_{lang}"
        traj_save[key] = tdata[lang]['h32']

np.savez_compressed('output/h32_trajectories.npz', **traj_save)
print(f"\nSaved raw trajectories: output/h32_trajectories.npz")

with open('output/diffusion_map_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Saved results: output/diffusion_map_results.json")


# ---------- SUMMARY ----------
print("\n" + "=" * 90)
print("DIFFUSION MAP SUMMARY")
print("=" * 90)
print(f"{'Problem':>10} {'Cat':>13} | {'Lang':>4} | {'Steps':>5} | "
      f"{'R²⁰⁴⁸ ×':>8} | {'Diff ×':>8} | {'Reduction':>9} | "
      f"{'R cos':>6} | {'D cos':>6}")
print("-" * 90)

for prob_idx in test_indices:
    for lang in langs:
        key = f"prob{prob_idx}_{lang}"
        if key not in all_results:
            continue
        r = all_results[key]
        reduction_pct = r.get('path_ratio_reduction', float('nan'))
        print(f"  prob{prob_idx:>3} {r['category']:>13} | {lang:>4} | "
              f"{r['n_steps']:>5} | {r['r2048_path_ratio']:>8.1f} | "
              f"{r['diffusion_path_ratio']:>8.1f} | {reduction_pct:>8.1%} | "
              f"{r['r2048_consec_cos']:>6.3f} | {r['diffusion_consec_cos']:>6.3f}")
    # Joint alignment
    jkey = f"prob{prob_idx}_joint"
    if jkey in all_results:
        j = all_results[jkey]
        print(f"           {'JOINT':>13} | {'':>4} | "
              f"{'':>5} | {'':>8} | {'':>8} | {'':>9} | "
              f"align_ratio={j['alignment_ratio']:.3f}")
    print()

print("\nKEY QUESTION: Does the path ratio drop dramatically in diffusion coordinates?")
print("  If yes → the 200x is embedding cost; the manifold is smooth.")
print("  If no  → reasoning is genuinely geometrically expensive in intrinsic coordinates.")
