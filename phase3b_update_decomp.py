"""Experiment B: Residual Update Decomposition (standalone).

Extracts hidden states at all 36 layers for 20 zh/en math pairs,
decomposes each layer's residual update into Z vs Z⊥ components
using L32 k=50 multi-head attention SVD basis.

Fast: ~40 forward passes, no generation. ~5 min on RTX 4070 Super.

Run with: MPLBACKEND=Agg .venv_wsl/bin/python phase3b_update_decomp.py
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B"
BASIS_LAYER = 32
BASIS_K = 50
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Prompt pairs (same as Phase 2/3 for comparability)
# ---------------------------------------------------------------------------
PAIRS = [
    {"zh": "设 f(n) 为 n 的二进制表示中 1 的个数。满足 1 ≤ n ≤ 2025 且 f(n) = 3 的整数 n 有多少个？",
     "en": "Let f(n) be the number of 1s in the binary representation of n. How many integers n with 1 ≤ n ≤ 2025 satisfy f(n) = 3?"},
    {"zh": "求方程 x² + y² = 2025 的所有非负整数解的个数。",
     "en": "Find the number of all non-negative integer solutions to x² + y² = 2025."},
    {"zh": "计算 1 + 2 + 3 + ... + 100 的值。",
     "en": "Calculate the value of 1 + 2 + 3 + ... + 100."},
    {"zh": "一个袋子里有 5 个红球和 3 个蓝球。随机取出 2 个球，取到 2 个红球的概率是多少？",
     "en": "A bag contains 5 red balls and 3 blue balls. If 2 balls are drawn randomly, what is the probability of getting 2 red balls?"},
    {"zh": "求函数 f(x) = x³ - 3x 在区间 [-2, 2] 上的最大值。",
     "en": "Find the maximum value of f(x) = x³ - 3x on the interval [-2, 2]."},
    {"zh": "在一个 4×4 的棋盘上放置 4 个车，使得它们互不攻击，有多少种放法？",
     "en": "In how many ways can 4 rooks be placed on a 4×4 chessboard so that no two attack each other?"},
    {"zh": "已知等比数列 {a_n} 的首项 a_1 = 2，公比 q = 3，求前 5 项之和。",
     "en": "Given a geometric sequence {a_n} with first term a_1 = 2 and common ratio q = 3, find the sum of the first 5 terms."},
    {"zh": "求矩阵 [[1, 2], [3, 4]] 的行列式。",
     "en": "Find the determinant of the matrix [[1, 2], [3, 4]]."},
    {"zh": "用辗转相除法求 gcd(252, 198)。",
     "en": "Use the Euclidean algorithm to find gcd(252, 198)."},
    {"zh": "如果 sin(θ) = 3/5 且 θ 在第一象限，求 cos(θ) 的值。",
     "en": "If sin(θ) = 3/5 and θ is in the first quadrant, find the value of cos(θ)."},
    {"zh": "一个圆的半径为 7，求其面积。",
     "en": "A circle has radius 7. Find its area."},
    {"zh": "求极限 lim(n→∞) (1 + 1/n)^n 的值。",
     "en": "Find the limit lim(n→∞) (1 + 1/n)^n."},
    {"zh": "三个骰子同时掷出，点数之和为 10 的概率是多少？",
     "en": "Three dice are thrown simultaneously. What is the probability that the sum of the points is 10?"},
    {"zh": "求二次方程 x² - 5x + 6 = 0 的两个根。",
     "en": "Find the two roots of the quadratic equation x² - 5x + 6 = 0."},
    {"zh": "一个长方体的长、宽、高分别为 3、4、5，求其体积和表面积。",
     "en": "A rectangular box has length 3, width 4, and height 5. Find its volume and surface area."},
    {"zh": "求数列 1, 1, 2, 3, 5, 8, 13, ... 的第 10 项。",
     "en": "Find the 10th term of the sequence 1, 1, 2, 3, 5, 8, 13, ..."},
    {"zh": "将 255 转换为十六进制。",
     "en": "Convert 255 to hexadecimal."},
    {"zh": "求不定积分 ∫ x·e^x dx 的结果。",
     "en": "Find the indefinite integral ∫ x·e^x dx."},
    {"zh": "100 除以 7 的余数是多少？",
     "en": "What is the remainder when 100 is divided by 7?"},
    {"zh": "从 1 到 100 的整数中，有多少个是 3 的倍数？",
     "en": "Among the integers from 1 to 100, how many are multiples of 3?"},
]


def build_multi_head_z_mask(model, layer_idx, h, GQA, d, k):
    """Top-k SVD of stacked multi-head attention kernels. Returns (k, d)."""
    all_vh = []
    for head in range(h):
        vh = get_attn_subspace(model, layer_idx, h, GQA, d, head, k=k)
        all_vh.append(vh)
    stacked = torch.cat(all_vh, dim=0)  # (h*k, d)
    _, S, Vh_combined = torch.linalg.svd(stacked, full_matrices=False)
    return Vh_combined[:k, :]  # (k, d)


def main():
    print("=" * 70)
    print("EXPERIMENT B: Residual Update Decomposition")
    print(f"Model: {MODEL_NAME}")
    print(f"Z basis: L{BASIS_LAYER} k={BASIS_K} multi-head")
    print(f"Prompts: {len(PAIRS)} pairs")
    print("=" * 70)

    # Load model + tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    dims = get_model_dims(model)
    L, d, h, GQA = dims["L"], dims["d"], dims["h"], dims["GQA"]
    print(f"  L={L} d={d} h={h} GQA={GQA}")

    # Build Z projectors
    print(f"\nBuilding Z mask at L{BASIS_LAYER} k={BASIS_K}...")
    Vh = build_multi_head_z_mask(model, BASIS_LAYER, h, GQA, d, BASIS_K)
    P_Z = Vh.T @ Vh          # (d, d)
    P_Zp = torch.eye(d) - P_Z  # (d, d)
    print(f"  Z mask shape: {Vh.shape}, projector: {P_Z.shape}")

    # Register hooks on all layers
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output[0].detach().cpu().squeeze(0).float()  # (seq, d)
        return hook

    hooks = []
    for li in range(L):
        handle = model.model.layers[li].register_forward_hook(make_hook(f"L{li}"))
        hooks.append(handle)

    # Extract all-layer mean-pooled hidden states
    N = len(PAIRS)
    all_states = {}  # (lang, pair_idx, layer) -> (d,)

    for lang in ["zh", "en"]:
        key = lang
        for i, pair in enumerate(tqdm(PAIRS, desc=f"{lang} forward passes")):
            inputs = tokenizer(pair[lang], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            for li in range(L):
                all_states[(lang, i, li)] = activations[f"L{li}"].mean(dim=0)  # (d,)

    # Remove hooks
    for handle in hooks:
        handle.remove()

    print(f"\nExtracted {len(all_states)} state vectors ({L} layers × {N} pairs × 2 langs)")

    # Compute update decomposition
    print("\nComputing residual update decomposition...")

    update_z_norms = {"zh": np.zeros(L - 1), "en": np.zeros(L - 1)}
    update_zp_norms = {"zh": np.zeros(L - 1), "en": np.zeros(L - 1)}
    update_ratios = {"zh": np.zeros(L - 1), "en": np.zeros(L - 1)}

    # Also track per-prompt for error bars
    per_prompt_ratios = {"zh": np.zeros((N, L - 1)), "en": np.zeros((N, L - 1))}

    for lang in ["zh", "en"]:
        for li in range(L - 1):
            z_norms = []
            zp_norms = []
            for i in range(N):
                h_curr = all_states[(lang, i, li)]
                h_next = all_states[(lang, i, li + 1)]
                delta = h_next - h_curr

                dz = P_Z @ delta
                dzp = P_Zp @ delta
                zn = torch.norm(dz).item()
                zpn = torch.norm(dzp).item()
                z_norms.append(zn)
                zp_norms.append(zpn)
                per_prompt_ratios[lang][i, li] = zn / zpn if zpn > 0 else 0.0

            update_z_norms[lang][li] = np.mean(z_norms)
            update_zp_norms[lang][li] = np.mean(zp_norms)
            update_ratios[lang][li] = np.mean(z_norms) / np.mean(zp_norms) if np.mean(zp_norms) > 0 else 0.0

    # Chance level: if delta is random, R = sqrt(k / (d-k))
    chance_R = np.sqrt(BASIS_K / (d - BASIS_K))

    # ---------------------------------------------------------------------------
    # Plot 1: R(k) vs layer (main result)
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.plot(range(L - 1), update_ratios["zh"], "o-", label="Chinese", markersize=4)
    ax.plot(range(L - 1), update_ratios["en"], "s-", label="English", markersize=4)
    ax.axhline(chance_R, color="gray", linestyle="--", alpha=0.5,
               label=f"Chance R={chance_R:.4f}")
    ax.axvline(BASIS_LAYER, color="red", linestyle=":", alpha=0.4, label=f"L{BASIS_LAYER}")
    ax.axvline(33, color="orange", linestyle=":", alpha=0.4, label="L33")
    ax.set_xlabel("Layer transition k → k+1")
    ax.set_ylabel("R(k) = ||Δh_Z|| / ||Δh_Z⊥||")
    ax.set_title("Update decomposition: reasoning vs language per layer")
    ax.legend(fontsize=8)

    # ---------------------------------------------------------------------------
    # Plot 2: Absolute norms
    # ---------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(range(L - 1), update_z_norms["zh"], "o-", label="zh Z-norm", markersize=3)
    ax.plot(range(L - 1), update_z_norms["en"], "s-", label="en Z-norm", markersize=3)
    ax.plot(range(L - 1), update_zp_norms["zh"], "o--", label="zh Z⊥-norm",
            markersize=3, alpha=0.6)
    ax.plot(range(L - 1), update_zp_norms["en"], "s--", label="en Z⊥-norm",
            markersize=3, alpha=0.6)
    ax.axvline(BASIS_LAYER, color="red", linestyle=":", alpha=0.4)
    ax.axvline(33, color="orange", linestyle=":", alpha=0.4)
    ax.set_xlabel("Layer transition k → k+1")
    ax.set_ylabel("||Δh|| (mean across prompts)")
    ax.set_title("Absolute update norms in Z vs Z⊥")
    ax.legend(fontsize=8)

    # ---------------------------------------------------------------------------
    # Plot 3: Cross-lingual asymmetry
    # ---------------------------------------------------------------------------
    ax = axes[1, 0]
    diff = update_ratios["zh"] - update_ratios["en"]
    colors = ["steelblue" if d >= 0 else "coral" for d in diff]
    ax.bar(range(L - 1), diff, width=0.8, alpha=0.7, color=colors)
    ax.axhline(0, color="gray", linestyle="-")
    ax.axvline(BASIS_LAYER, color="red", linestyle=":", alpha=0.4)
    ax.axvline(33, color="orange", linestyle=":", alpha=0.4)
    ax.set_xlabel("Layer transition k → k+1")
    ax.set_ylabel("R_zh(k) - R_en(k)")
    ax.set_title("Cross-lingual asymmetry (blue=zh more Z-dominant)")

    # ---------------------------------------------------------------------------
    # Plot 4: Per-prompt variability at key layers
    # ---------------------------------------------------------------------------
    ax = axes[1, 1]
    key_layers = [0, 5, 10, 15, 20, 25, 30, 31, 32, 33, 34]
    key_layers = [l for l in key_layers if l < L - 1]
    positions = np.arange(len(key_layers))
    width = 0.35

    zh_means = [per_prompt_ratios["zh"][:, l].mean() for l in key_layers]
    zh_stds = [per_prompt_ratios["zh"][:, l].std() for l in key_layers]
    en_means = [per_prompt_ratios["en"][:, l].mean() for l in key_layers]
    en_stds = [per_prompt_ratios["en"][:, l].std() for l in key_layers]

    ax.bar(positions - width/2, zh_means, width, yerr=zh_stds,
           label="Chinese", alpha=0.7, capsize=3)
    ax.bar(positions + width/2, en_means, width, yerr=en_stds,
           label="English", alpha=0.7, capsize=3)
    ax.axhline(chance_R, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"L{l}" for l in key_layers], rotation=45)
    ax.set_ylabel("R(k) per prompt (mean ± std)")
    ax.set_title("Per-prompt variability at key layers")
    ax.legend(fontsize=8)

    plt.suptitle(f"Experiment B: Residual Update Decomposition\n"
                 f"Z basis: L{BASIS_LAYER} k={BASIS_K} multi-head | "
                 f"chance R = {chance_R:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expB_update_decomposition.png", dpi=150)
    print(f"\nSaved: {OUTPUT_DIR / 'expB_update_decomposition.png'}")

    # ---------------------------------------------------------------------------
    # Save data
    # ---------------------------------------------------------------------------
    exp_b_data = {
        "basis_layer": BASIS_LAYER,
        "basis_k": BASIS_K,
        "chance_R": float(chance_R),
        "z_norms_zh": update_z_norms["zh"].tolist(),
        "z_norms_en": update_z_norms["en"].tolist(),
        "zp_norms_zh": update_zp_norms["zh"].tolist(),
        "zp_norms_en": update_zp_norms["en"].tolist(),
        "ratios_zh": update_ratios["zh"].tolist(),
        "ratios_en": update_ratios["en"].tolist(),
    }
    json_path = OUTPUT_DIR / "expB_update_decomposition.json"
    with open(json_path, "w") as f:
        json.dump(exp_b_data, f, indent=2)
    print(f"Saved: {json_path}")

    # ---------------------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("EXPERIMENT B: SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Basis: L{BASIS_LAYER} k={BASIS_K} multi-head Z mask")
    print(f"  Chance R (random delta): {chance_R:.4f}")
    print(f"  R(k) = ||Δh_Z|| / ||Δh_Z⊥|| averaged over {N} prompts\n")

    print(f"  {'Layer':>5} {'R_zh':>8} {'R_en':>8} {'Diff':>8} {'vs chance':>10}")
    print("  " + "-" * 42)
    for li in range(L - 1):
        diff_val = update_ratios["zh"][li] - update_ratios["en"][li]
        avg_r = (update_ratios["zh"][li] + update_ratios["en"][li]) / 2
        vs = f"{avg_r / chance_R:.2f}x" if chance_R > 0 else "?"
        marker = ""
        if li == BASIS_LAYER:
            marker = " <<< Z basis"
        elif li == 33:
            marker = " <<< bottleneck"
        print(f"  {li:>2}→{li+1:<2} {update_ratios['zh'][li]:>8.4f} "
              f"{update_ratios['en'][li]:>8.4f} {diff_val:>+8.4f} {vs:>10}{marker}")

    # Key findings
    peak_zh = int(np.argmax(update_ratios["zh"]))
    peak_en = int(np.argmax(update_ratios["en"]))
    trough_zh = int(np.argmin(update_ratios["zh"]))
    trough_en = int(np.argmin(update_ratios["en"]))

    print(f"\n  Peak R:   zh at L{peak_zh}→{peak_zh+1} ({update_ratios['zh'][peak_zh]:.4f}), "
          f"en at L{peak_en}→{peak_en+1} ({update_ratios['en'][peak_en]:.4f})")
    print(f"  Trough R: zh at L{trough_zh}→{trough_zh+1} ({update_ratios['zh'][trough_zh]:.4f}), "
          f"en at L{trough_en}→{trough_en+1} ({update_ratios['en'][trough_en]:.4f})")

    ever_above_chance_zh = any(r > chance_R * 1.5 for r in update_ratios["zh"])
    ever_above_chance_en = any(r > chance_R * 1.5 for r in update_ratios["en"])
    print(f"\n  Any layer with R > 1.5× chance?  "
          f"zh={'YES' if ever_above_chance_zh else 'NO'}  "
          f"en={'YES' if ever_above_chance_en else 'NO'}")

    ever_z_dominant = any(r > 1.0 for r in update_ratios["zh"]) or \
                      any(r > 1.0 for r in update_ratios["en"])
    if ever_z_dominant:
        print("  → Some layers have Z-dominated updates: reasoning is localized!")
    else:
        print("  → No layer has R > 1: Z is emergent from mixed computation across all layers.")

    # Biggest cross-lingual gap
    max_diff_idx = int(np.argmax(np.abs(diff)))
    print(f"\n  Largest zh/en divergence: L{max_diff_idx}→{max_diff_idx+1} "
          f"(diff={diff[max_diff_idx]:+.4f})")

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
