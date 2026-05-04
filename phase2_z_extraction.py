"""Phase 2: Activation extraction + Z subspace projection at L33.

Run with: MPLBACKEND=Agg .venv_wsl/bin/python phase2_z_extraction.py

Uses the structural finding from 2.py: L33's attention kernel top-k SVD
vectors define the candidate Z subspace. FFN is orthogonal to attention
at L33 (0.57x chance), so the attention subspace is clean.

Strategy:
1. Extract Z masks from attention kernels (single-head + multi-head averaged)
2. Run paired Chinese/English math prompts through model
3. Capture hidden states at L32/L33 (encoding only, no generation)
4. Project hidden states onto Z and Z_perp for k=20,50,78
5. Compute distances: same-problem cross-lingual should be CLOSE in Z,
   different-problem same-language should be FAR in Z
6. Compare to random-subspace baseline (k/d fraction)

Critical design decisions:
- NO thinking suffix — extract at last token of PROBLEM to avoid
  confounding language signal at extraction point
- Multi-head averaged Z mask as primary (not just head 0)
- Random-subspace baseline for statistical calibration
- Both mean-pooling and last-token strategies
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_attn_subspace, get_model_dims

# --- Config ---
MODEL_NAME = "Qwen/Qwen2.5-3B"
TARGET_LAYERS = [32, 33]
K_VALUES = [20, 50, 78]
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
N_RANDOM_BASELINES = 50  # Number of random subspaces for baseline

# --- Prompt pairs ---
# Pure math problems, no thinking suffix. We extract hidden states during
# encoding of the problem itself — the language framing IS part of what we
# want Z to filter out.
PAIRS = [
    {
        "zh": "设 f(n) 为 n 的二进制表示中 1 的个数。满足 1 ≤ n ≤ 2025 且 f(n) = 3 的整数 n 有多少个？",
        "en": "Let f(n) be the number of 1s in the binary representation of n. How many integers n with 1 ≤ n ≤ 2025 satisfy f(n) = 3?",
        "category": "combinatorics",
    },
    {
        "zh": "求方程 x² + y² = 2025 的所有非负整数解的个数。",
        "en": "Find the number of all non-negative integer solutions to x² + y² = 2025.",
        "category": "number_theory",
    },
    {
        "zh": "计算 1 + 2 + 3 + ... + 100 的值。",
        "en": "Calculate the value of 1 + 2 + 3 + ... + 100.",
        "category": "arithmetic",
    },
    {
        "zh": "证明对所有正整数 n，n³ - n 能被 6 整除。",
        "en": "Prove that for all positive integers n, n³ - n is divisible by 6.",
        "category": "proof",
    },
    {
        "zh": "一个袋子里有 5 个红球和 3 个蓝球。随机取出 2 个球，取到 2 个红球的概率是多少？",
        "en": "A bag contains 5 red balls and 3 blue balls. If 2 balls are drawn randomly, what is the probability of getting 2 red balls?",
        "category": "probability",
    },
    {
        "zh": "求函数 f(x) = x³ - 3x 在区间 [-2, 2] 上的最大值。",
        "en": "Find the maximum value of f(x) = x³ - 3x on the interval [-2, 2].",
        "category": "calculus",
    },
    {
        "zh": "在一个 4×4 的棋盘上放置 4 个车，使得它们互不攻击，有多少种放法？",
        "en": "In how many ways can 4 rooks be placed on a 4×4 chessboard so that no two attack each other?",
        "category": "combinatorics",
    },
    {
        "zh": "求不定积分 ∫ x·e^x dx 的结果。",
        "en": "Find the indefinite integral ∫ x·e^x dx.",
        "category": "calculus",
    },
    {
        "zh": "已知等比数列 {a_n} 的首项 a_1 = 2，公比 q = 3，求前 5 项之和。",
        "en": "Given a geometric sequence {a_n} with first term a_1 = 2 and common ratio q = 3, find the sum of the first 5 terms.",
        "category": "sequences",
    },
    {
        "zh": "求矩阵 [[1, 2], [3, 4]] 的行列式。",
        "en": "Find the determinant of the matrix [[1, 2], [3, 4]].",
        "category": "linear_algebra",
    },
    {
        "zh": "用辗转相除法求 gcd(252, 198)。",
        "en": "Use the Euclidean algorithm to find gcd(252, 198).",
        "category": "number_theory",
    },
    {
        "zh": "如果 sin(θ) = 3/5 且 θ 在第一象限，求 cos(θ) 的值。",
        "en": "If sin(θ) = 3/5 and θ is in the first quadrant, find the value of cos(θ).",
        "category": "trigonometry",
    },
    {
        "zh": "一个圆的半径为 7，求其面积。",
        "en": "A circle has radius 7. Find its area.",
        "category": "geometry",
    },
    {
        "zh": "求极限 lim(n→∞) (1 + 1/n)^n 的值。",
        "en": "Find the limit lim(n→∞) (1 + 1/n)^n.",
        "category": "calculus",
    },
    {
        "zh": "将分数 7/12 化为小数（保留 4 位小数）。",
        "en": "Convert the fraction 7/12 to a decimal (to 4 decimal places).",
        "category": "arithmetic",
    },
    {
        "zh": "三个骰子同时掷出，点数之和为 10 的概率是多少？",
        "en": "Three dice are thrown simultaneously. What is the probability that the sum of the points is 10?",
        "category": "probability",
    },
    {
        "zh": "求二次方程 x² - 5x + 6 = 0 的两个根。",
        "en": "Find the two roots of the quadratic equation x² - 5x + 6 = 0.",
        "category": "algebra",
    },
    {
        "zh": "将 255 转换为十六进制。",
        "en": "Convert 255 to hexadecimal.",
        "category": "arithmetic",
    },
    {
        "zh": "一个长方体的长、宽、高分别为 3、4、5，求其体积和表面积。",
        "en": "A rectangular box has length 3, width 4, and height 5. Find its volume and surface area.",
        "category": "geometry",
    },
    {
        "zh": "求数列 1, 1, 2, 3, 5, 8, 13, ... 的第 10 项。",
        "en": "Find the 10th term of the sequence 1, 1, 2, 3, 5, 8, 13, ...",
        "category": "sequences",
    },
]


def build_multi_head_z_mask(model, layer_idx: int, h: int, GQA: int,
                            d: int, k: int) -> torch.Tensor:
    """Build Z mask by averaging across all heads' subspaces.

    Stack all heads' top-k Vh matrices (16 × k × d), then take the SVD
    of the stacked matrix to find the best k-dim subspace spanning all heads.
    This is more robust than using a single head.

    Returns: (k, d) tensor of orthonormal row vectors.
    """
    all_vh = []
    for head in range(h):
        vh = get_attn_subspace(model, layer_idx, h, GQA, d, head, k=k)
        all_vh.append(vh)
    stacked = torch.cat(all_vh, dim=0)  # (h*k, d)
    _, S, Vh_combined = torch.linalg.svd(stacked, full_matrices=False)
    return Vh_combined[:k, :]  # (k, d) — best k-dim subspace


def random_subspace_baseline(d: int, k: int, n_trials: int,
                             delta_vecs: list[np.ndarray]) -> dict:
    """Compute baseline: what fraction of ||delta||² lands in a random
    k-dim subspace of R^d?

    Theory: E[||P_Z delta||²] = (k/d) * ||delta||² for random Z.
    We verify empirically and return mean ± std of the ratio.
    """
    ratios = []
    for _ in range(n_trials):
        # Random orthonormal basis for k-dim subspace
        Q, _ = np.linalg.qr(np.random.randn(d, k))
        P_rand = Q @ Q.T  # (d, d) projection

        trial_ratios = []
        for delta in delta_vecs:
            proj_norm_sq = np.linalg.norm(P_rand @ delta) ** 2
            full_norm_sq = np.linalg.norm(delta) ** 2
            if full_norm_sq > 0:
                trial_ratios.append(proj_norm_sq / full_norm_sq)
        ratios.append(np.mean(trial_ratios))

    return {
        "theoretical": k / d,
        "empirical_mean": float(np.mean(ratios)),
        "empirical_std": float(np.std(ratios)),
    }


def main():
    print("=" * 60)
    print("PHASE 2: Activation Extraction + Z Subspace Projection")
    print(f"Model: {MODEL_NAME}")
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"K values: {K_VALUES}")
    print(f"Prompt pairs: {len(PAIRS)}")
    print(f"Random baseline trials: {N_RANDOM_BASELINES}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    dims = get_model_dims(model)
    L, d, h, GQA = dims["L"], dims["d"], dims["h"], dims["GQA"]
    print(f"Model loaded. L={L}, d={d}, h={h}, GQA={GQA}")

    # --- Step 1: Extract Z masks ---
    print("\n--- Step 1: Extract Z masks (attention SVD) ---")
    z_masks = {}  # (layer, k, "head0"|"multi") -> (k, d) tensor

    for layer in TARGET_LAYERS:
        for k in K_VALUES:
            # Head 0 (consistent with 2.py analysis)
            z_head0 = get_attn_subspace(model, layer, h, GQA, d, head_idx=0, k=k)
            z_masks[(layer, k, "head0")] = z_head0

            # Multi-head averaged (more robust)
            z_multi = build_multi_head_z_mask(model, layer, h, GQA, d, k)
            z_masks[(layer, k, "multi")] = z_multi

            print(f"  L{layer}, k={k}: head0={z_head0.shape}, multi={z_multi.shape}")

    # --- Step 2: Register hooks and extract hidden states ---
    print("\n--- Step 2: Extract hidden states ---")
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output[0].detach().cpu().squeeze(0)  # [seq, d]
        return hook

    hooks = []
    for layer_idx in TARGET_LAYERS:
        handle = model.model.layers[layer_idx].register_forward_hook(
            make_hook(f"layer_{layer_idx}")
        )
        hooks.append(handle)

    # Run all prompts — NO thinking suffix, just the math problem
    hidden_states = {}  # (lang, pair_idx, layer) -> [n_tokens, d]
    for i, pair in enumerate(tqdm(PAIRS, desc="Extracting activations")):
        for lang in ["zh", "en"]:
            prompt = pair[lang]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)

            for layer_idx in TARGET_LAYERS:
                key = (lang, i, layer_idx)
                hidden_states[key] = activations[f"layer_{layer_idx}"].float()

    for handle in hooks:
        handle.remove()

    print(f"  Extracted {len(hidden_states)} activation tensors")
    sample_key = ("zh", 0, TARGET_LAYERS[0])
    print(f"  Sample shape: {hidden_states[sample_key].shape}")

    # --- Step 3: Project onto Z and Z_perp, compute distances ---
    print("\n--- Step 3: Projection + Distance Analysis ---")

    all_results = {}
    n = len(PAIRS)

    for layer in TARGET_LAYERS:
        for k in K_VALUES:
            for mask_type in ["head0", "multi"]:
                Vh = z_masks[(layer, k, mask_type)]  # (k, d)
                P_z = Vh.T @ Vh          # (d, d) orthogonal projector onto Z
                P_zperp = torch.eye(d) - P_z

                for pool_name in ["mean", "last"]:
                    z_vecs = {}
                    zp_vecs = {}
                    full_vecs = {}

                    for i in range(n):
                        for lang in ["zh", "en"]:
                            h_state = hidden_states[(lang, i, layer)]  # [seq, d]
                            if pool_name == "mean":
                                pooled = h_state.mean(dim=0)
                            else:
                                pooled = h_state[-1]

                            z_vecs[(lang, i)] = (P_z @ pooled).numpy()
                            zp_vecs[(lang, i)] = (P_zperp @ pooled).numpy()
                            full_vecs[(lang, i)] = pooled.numpy()

                    # --- Distance computations ---
                    # Cross-lingual same-problem (should be SMALL in Z)
                    cross_z, cross_zp, cross_full = [], [], []
                    for i in range(n):
                        cross_z.append(np.linalg.norm(z_vecs[("zh", i)] - z_vecs[("en", i)]))
                        cross_zp.append(np.linalg.norm(zp_vecs[("zh", i)] - zp_vecs[("en", i)]))
                        cross_full.append(np.linalg.norm(full_vecs[("zh", i)] - full_vecs[("en", i)]))

                    # Same-language different-problem (should be LARGE in Z)
                    same_z, same_zp, same_full = [], [], []
                    for lang in ["zh", "en"]:
                        for i in range(n):
                            for j in range(i + 1, n):
                                same_z.append(np.linalg.norm(z_vecs[(lang, i)] - z_vecs[(lang, j)]))
                                same_zp.append(np.linalg.norm(zp_vecs[(lang, i)] - zp_vecs[(lang, j)]))
                                same_full.append(np.linalg.norm(full_vecs[(lang, i)] - full_vecs[(lang, j)]))

                    cross_z = np.array(cross_z)
                    cross_zp = np.array(cross_zp)
                    cross_full = np.array(cross_full)
                    same_z = np.array(same_z)
                    same_zp = np.array(same_zp)
                    same_full = np.array(same_full)

                    ratio_z = cross_z.mean() / same_z.mean()
                    ratio_zp = cross_zp.mean() / same_zp.mean()
                    ratio_full = cross_full.mean() / same_full.mean()

                    # --- Energy fraction: how much of ||delta||² is in Z? ---
                    energy_fracs_cross = []
                    for i in range(n):
                        delta = full_vecs[("zh", i)] - full_vecs[("en", i)]
                        full_sq = np.linalg.norm(delta) ** 2
                        z_sq = np.linalg.norm((P_z.numpy() @ delta)) ** 2
                        if full_sq > 0:
                            energy_fracs_cross.append(z_sq / full_sq)

                    result = {
                        "layer": layer, "k": k, "mask": mask_type, "pooling": pool_name,
                        "cross_z_mean": float(cross_z.mean()),
                        "cross_z_std": float(cross_z.std()),
                        "same_z_mean": float(same_z.mean()),
                        "same_z_std": float(same_z.std()),
                        "ratio_z": float(ratio_z),
                        "cross_zp_mean": float(cross_zp.mean()),
                        "cross_zp_std": float(cross_zp.std()),
                        "same_zp_mean": float(same_zp.mean()),
                        "same_zp_std": float(same_zp.std()),
                        "ratio_zp": float(ratio_zp),
                        "cross_full_mean": float(cross_full.mean()),
                        "same_full_mean": float(same_full.mean()),
                        "ratio_full": float(ratio_full),
                        "energy_frac_z_mean": float(np.mean(energy_fracs_cross)),
                        "energy_frac_z_std": float(np.std(energy_fracs_cross)),
                        "expected_random_frac": k / d,
                    }
                    all_results[(layer, k, mask_type, pool_name)] = result

                    tag = f"L{layer} k={k} {mask_type} {pool_name}"
                    print(f"\n  {tag}:")
                    print(f"    Z:      cross={cross_z.mean():.4f} same={same_z.mean():.4f} ratio={ratio_z:.4f}")
                    print(f"    Z_perp: cross={cross_zp.mean():.4f} same={same_zp.mean():.4f} ratio={ratio_zp:.4f}")
                    print(f"    Full:   cross={cross_full.mean():.4f} same={same_full.mean():.4f} ratio={ratio_full:.4f}")
                    print(f"    Energy in Z: {np.mean(energy_fracs_cross):.4f} (random baseline: {k/d:.4f})")

                    if ratio_z < ratio_zp:
                        print(f"    >> Z separates better (ratio_z < ratio_zp). GOOD.")
                    else:
                        print(f"    >> Z_perp separates better. Hypothesis NOT supported here.")

    # --- Step 4: Random subspace baseline ---
    print("\n--- Step 4: Random Subspace Baseline ---")
    # Compute for the cross-lingual deltas at L33, mean pooling
    for k in K_VALUES:
        deltas = []
        for i in range(n):
            h_zh = hidden_states[("zh", i, 33)].mean(dim=0).numpy()
            h_en = hidden_states[("en", i, 33)].mean(dim=0).numpy()
            deltas.append(h_zh - h_en)
        baseline = random_subspace_baseline(d, k, N_RANDOM_BASELINES, deltas)
        print(f"  k={k}: theoretical={baseline['theoretical']:.4f}, "
              f"empirical={baseline['empirical_mean']:.4f} ± {baseline['empirical_std']:.4f}")

    # --- Step 5: Generate plots ---
    print("\n--- Step 5: Generate plots ---")

    for layer in TARGET_LAYERS:
        fig, axes = plt.subplots(2, len(K_VALUES), figsize=(6 * len(K_VALUES), 10))

        for col, k in enumerate(K_VALUES):
            for row, mask_type in enumerate(["head0", "multi"]):
                ax = axes[row, col]
                r = all_results.get((layer, k, mask_type, "mean"))
                if r is None:
                    continue

                bars = ax.bar(
                    ["Z\ncross-ling", "Z\nsame-lang", "Z⊥\ncross-ling", "Z⊥\nsame-lang"],
                    [r["cross_z_mean"], r["same_z_mean"],
                     r["cross_zp_mean"], r["same_zp_mean"]],
                    color=["#2196F3", "#64B5F6", "#FF5722", "#FF8A65"],
                    yerr=[r["cross_z_std"], r["same_z_std"],
                          r["cross_zp_std"], r["same_zp_std"]],
                    capsize=5,
                )
                ax.set_title(f"L{layer}, k={k}, {mask_type}")
                ax.set_ylabel("Mean L2 distance")

        plt.suptitle(f"Layer {layer}: Z vs Z⊥ distance structure", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"phase2_L{layer}_distances.png", dpi=150)
        print(f"  Saved: phase2_L{layer}_distances.png")

    # --- Energy fraction plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mask_type in zip(axes, ["head0", "multi"]):
        for layer in TARGET_LAYERS:
            fracs = [all_results[(layer, k, mask_type, "mean")]["energy_frac_z_mean"]
                     for k in K_VALUES]
            baselines = [k / d for k in K_VALUES]
            ax.plot(K_VALUES, fracs, 'o-', label=f"L{layer}")
            ax.plot(K_VALUES, baselines, 's--', alpha=0.5, label=f"random (k/d)")
        ax.set_xlabel("k (subspace dimension)")
        ax.set_ylabel("Fraction of cross-lingual δ² in Z")
        ax.set_title(f"Energy concentration ({mask_type})")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase2_energy_concentration.png", dpi=150)
    print(f"  Saved: phase2_energy_concentration.png")

    # --- Step 6: Save results ---
    save_results = {}
    for key, val in all_results.items():
        save_key = f"L{key[0]}_k{key[1]}_{key[2]}_{key[3]}"
        save_results[save_key] = val

    with open(OUTPUT_DIR / "phase2_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to output/phase2_results.json")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"\n{'Layer':>5} {'k':>3} {'Mask':>6} {'Pool':>5} | {'R_Z':>7} {'R_Zp':>7} {'R_Full':>7} | {'E_Z':>6} {'E_rand':>6} | {'Z<Zp?':>6}")
    print("-" * 80)
    for (layer, k, mask, pool), r in sorted(all_results.items()):
        z_wins = "YES" if r["ratio_z"] < r["ratio_zp"] else "no"
        print(f"{layer:>5} {k:>3} {mask:>6} {pool:>5} | "
              f"{r['ratio_z']:>7.4f} {r['ratio_zp']:>7.4f} {r['ratio_full']:>7.4f} | "
              f"{r['energy_frac_z_mean']:>6.4f} {r['expected_random_frac']:>6.4f} | "
              f"{z_wins:>6}")

    print("\nKey:")
    print("  R_Z / R_Zp / R_Full = cross-lingual/same-lang distance ratio (LOWER = better separation)")
    print("  E_Z = fraction of cross-lingual delta energy in Z")
    print("  E_rand = expected fraction for random k-dim subspace (k/d)")
    print("  Z<Zp? = does Z separate better than Z_perp? (YES = hypothesis supported)")
    print("\n  If E_Z >> E_rand: Z captures MORE of the language difference than expected")
    print("  If E_Z << E_rand: Z captures LESS — language info lives OUTSIDE Z (good for reasoning!)")


if __name__ == "__main__":
    main()
