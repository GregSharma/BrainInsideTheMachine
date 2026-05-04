"""Visualization: The figures that make this real.

Run with: MPLBACKEND=Agg .venv/bin/python visualize.py

Produces:
  1. t-SNE at L32: 200 problems × 2 languages, color=language, marker=category
  2. Layer animation: L8 → L16 → L24 → L32 → L34 showing zh/en convergence
  3. Bridge R² comparison: contrastive Z vs random baselines
  4. Layer sweep gradient: significance + energy across depth
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import orthogonal_procrustes
import random as pyrandom

from utils import get_model_dims

MODEL_NAME = "Qwen/Qwen2.5-3B"
K = 20
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42
VIZ_LAYERS = [8, 16, 24, 32, 34]
CAT_NAMES = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]
CAT_MARKERS = ["o", "s", "^", "D", "v"]  # circle, square, triangle, diamond, inverted triangle


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


def build_contrastive_z(zh_means, en_means, k, var_threshold=0.90):
    N, d = zh_means.shape
    diffs = zh_means - en_means
    diffs_centered = diffs - diffs.mean(axis=0)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    var_explained = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)
    n_lang = max(1, int(np.searchsorted(cumvar, var_threshold) + 1))
    n_lang = min(n_lang, len(S))
    lang_dirs = Vt[:n_lang]
    proj_out = np.eye(d, dtype=np.float32) - lang_dirs.T @ lang_dirs
    all_means = np.concatenate([zh_means, en_means], axis=0)
    projected = all_means @ proj_out.T
    projected_centered = projected - projected.mean(axis=0)
    _, _, Vt_proj = np.linalg.svd(projected_centered, full_matrices=False)
    actual_k = min(k, Vt_proj.shape[0])
    return Vt_proj[:actual_k].astype(np.float32), n_lang, lang_dirs.astype(np.float32)


def extract_activations(model, tokenizer, problems, layer_idx):
    """Extract mean-pooled hidden states at given layer for all problems."""
    d = model.config.hidden_size
    N = len(problems)
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h_out.detach().cpu().squeeze(0)
        return hook

    hook_handle = model.model.layers[layer_idx].register_forward_hook(make_hook("target"))

    zh_means = np.zeros((N, d), dtype=np.float32)
    en_means = np.zeros((N, d), dtype=np.float32)

    for i, prob in enumerate(tqdm(problems, desc=f"L{layer_idx} zh", leave=False)):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        zh_means[i] = activations["target"].float().numpy().mean(axis=0)

    for i, prob in enumerate(tqdm(problems, desc=f"L{layer_idx} en", leave=False)):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        en_means[i] = activations["target"].float().numpy().mean(axis=0)

    hook_handle.remove()
    return zh_means, en_means


def plot_tsne_at_layer(zh_proj, en_proj, categories, layer_idx, ax, title=None):
    """t-SNE of zh and en projections into contrastive Z, on a given axes."""
    N = zh_proj.shape[0]
    combined = np.concatenate([zh_proj, en_proj], axis=0)  # [2N, k]

    perplexity = min(30, combined.shape[0] // 2 - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
                max_iter=1000, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(combined)

    zh_coords = coords[:N]
    en_coords = coords[N:]

    # Plot — zh in red tones, en in blue tones
    for cat_id in range(5):
        mask = categories == cat_id
        marker = CAT_MARKERS[cat_id]

        # Chinese points
        ax.scatter(zh_coords[mask, 0], zh_coords[mask, 1],
                   c="#e74c3c", marker=marker, s=40, alpha=0.7,
                   edgecolors="darkred", linewidths=0.3)
        # English points
        ax.scatter(en_coords[mask, 0], en_coords[mask, 1],
                   c="#3498db", marker=marker, s=40, alpha=0.7,
                   edgecolors="darkblue", linewidths=0.3)

    # Draw thin lines connecting zh-en pairs
    for i in range(N):
        ax.plot([zh_coords[i, 0], en_coords[i, 0]],
                [zh_coords[i, 1], en_coords[i, 1]],
                color="gray", alpha=0.15, linewidth=0.5)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def make_legend(fig):
    """Create shared legend for language and category."""
    lang_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markeredgecolor="darkred", markersize=10, label="Chinese (zh)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markeredgecolor="darkblue", markersize=10, label="English (en)"),
    ]
    cat_handles = [
        Line2D([0], [0], marker=CAT_MARKERS[i], color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label=CAT_NAMES[i])
        for i in range(5)
    ]
    pair_handle = [
        Line2D([0], [0], color="gray", alpha=0.4, linewidth=1, label="zh↔en pair")
    ]
    all_handles = lang_handles + cat_handles + pair_handle
    fig.legend(handles=all_handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, -0.02))


def apply_procrustes(zh_proj, en_proj):
    """Rotate en_proj onto zh_proj via Procrustes. Returns rotated en."""
    # Center both
    zh_c = zh_proj - zh_proj.mean(axis=0)
    en_c = en_proj - en_proj.mean(axis=0)
    R, scale = orthogonal_procrustes(en_c, zh_c)
    en_rotated = en_c @ R + zh_proj.mean(axis=0)
    return en_rotated


def fig1_tsne_hero(zh_means_L32, en_means_L32, categories):
    """THE hero figure: t-SNE at L32 in contrastive Z, AFTER Procrustes rotation."""
    print("\n--- Figure 1: t-SNE Hero (L32) ---")

    z_basis, n_lang, _ = build_contrastive_z(zh_means_L32, en_means_L32, K)
    zh_proj = zh_means_L32 @ z_basis.T
    en_proj_raw = en_means_L32 @ z_basis.T
    en_proj = apply_procrustes(zh_proj, en_proj_raw)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_tsne_at_layer(zh_proj, en_proj, categories, 32, ax,
                       title="Language-Invariant Math Reasoning at Layer 32\n"
                             f"(Contrastive Z, k={K}, after Procrustes rotation, Qwen2.5-3B)")

    # Add R² annotation
    ax.text(0.02, 0.98, f"Bridge R² = 0.976\n100th percentile vs random",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="goldenrod", alpha=0.9))

    make_legend(fig)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(OUTPUT_DIR / "fig1_tsne_hero.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'fig1_tsne_hero.png'}")
    plt.close(fig)


def fig2_layer_evolution(all_layer_data, categories):
    """5-panel layer progression: L8 → L16 → L24 → L32 → L34."""
    print("\n--- Figure 2: Layer Evolution ---")

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for idx, layer_idx in enumerate(VIZ_LAYERS):
        zh_means, en_means = all_layer_data[layer_idx]
        z_basis, n_lang, _ = build_contrastive_z(zh_means, en_means, K)
        zh_proj = zh_means @ z_basis.T
        en_proj_raw = en_means @ z_basis.T
        en_proj = apply_procrustes(zh_proj, en_proj_raw)

        plot_tsne_at_layer(zh_proj, en_proj, categories, layer_idx, axes[idx],
                           title=f"Layer {layer_idx}")

    fig.suptitle("Cross-Lingual Convergence Across Depth\n"
                 "Watch Chinese (red) and English (blue) merge into shared reasoning space",
                 fontsize=14, fontweight="bold", y=1.04)
    make_legend(fig)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    fig.savefig(OUTPUT_DIR / "fig2_layer_evolution.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'fig2_layer_evolution.png'}")
    plt.close(fig)


def fig3_bridge_and_sweep():
    """Bridge R² comparison + layer sweep gradient from saved JSON."""
    print("\n--- Figure 3: Bridge + Layer Sweep ---")

    with open(OUTPUT_DIR / "phase6_unified.json") as f:
        data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel A: Bridge R² ---
    bridge = data["bridge"]
    bars = ax1.bar(["Contrastive Z", "Random Mean"],
                   [bridge["r2_test"], bridge["random_mean"]],
                   color=["#e74c3c", "#95a5a6"], edgecolor="black", linewidth=0.8)
    ax1.errorbar(1, bridge["random_mean"], yerr=bridge["random_std"],
                 fmt="none", color="black", capsize=5, linewidth=2)
    ax1.set_ylabel("R² (held-out)", fontsize=12)
    ax1.set_title("Bridge: zh↔en Rotation\nin Contrastive Z", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.976, color="#e74c3c", linestyle="--", alpha=0.3)
    ax1.text(0, bridge["r2_test"] + 0.02, f'{bridge["r2_test"]:.3f}',
             ha="center", fontsize=13, fontweight="bold", color="#c0392b")
    ax1.text(1, bridge["random_mean"] + bridge["random_std"] + 0.03,
             f'{bridge["random_mean"]:.3f}±{bridge["random_std"]:.3f}',
             ha="center", fontsize=10, color="#7f8c8d")

    # --- Panel B: Layer sweep significance ---
    sweep = data["layer_sweep"]
    layers = sorted(int(k) for k in sweep.keys())
    percentiles = [sweep[str(l)]["percentile"] for l in layers]
    energy_ratios = [sweep[str(l)]["energy_ratio"] for l in layers]

    color_map = ["#95a5a6" if p < 95 else "#f39c12" if p < 100 else "#e74c3c"
                 for p in percentiles]
    ax2.bar(range(len(layers)), percentiles, color=color_map, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f"L{l}" for l in layers], rotation=45)
    ax2.set_ylabel("Percentile vs Random", fontsize=12)
    ax2.set_title("Within-Category Probe\nSignificance by Layer", fontsize=12, fontweight="bold")
    ax2.axhline(y=95, color="orange", linestyle="--", alpha=0.5, label="p=0.05")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)

    # --- Panel C: Energy concentration ---
    ax3.plot(layers, energy_ratios, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax3.fill_between(layers, 1, energy_ratios, alpha=0.15, color="#e74c3c")
    ax3.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax3.set_xlabel("Layer", fontsize=12)
    ax3.set_ylabel("Energy Ratio (×chance)", fontsize=12)
    ax3.set_title("Language Energy Concentration\nAcross Depth", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)

    # Add n_lang_dirs annotation
    for i, l in enumerate(layers):
        n_dirs = sweep[str(l)]["n_lang_dirs"]
        ax3.annotate(f"{n_dirs}d", (l, energy_ratios[i]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=8, ha="center", color="#c0392b")

    fig.suptitle("Contrastive Z: Bridge Alignment + Depth Gradient (Qwen2.5-3B)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_bridge_sweep.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'fig3_bridge_sweep.png'}")
    plt.close(fig)


def fig4_pair_distances(zh_proj, en_proj, categories):
    """Show that same-problem zh/en distance << different-problem distance in Z."""
    print("\n--- Figure 4: Pair Distance Analysis ---")

    N = zh_proj.shape[0]

    # Same-problem, cross-lingual distances
    same_prob_dists = np.linalg.norm(zh_proj - en_proj, axis=1)

    # Different-problem, same-language distances (sample)
    rng = np.random.default_rng(SEED)
    diff_prob_dists = []
    for _ in range(N * 5):
        i, j = rng.integers(0, N, size=2)
        if i != j:
            diff_prob_dists.append(np.linalg.norm(zh_proj[i] - zh_proj[j]))
    diff_prob_dists = np.array(diff_prob_dists)

    # Different-problem, cross-lingual distances
    cross_dists = []
    for _ in range(N * 5):
        i, j = rng.integers(0, N, size=2)
        if i != j:
            cross_dists.append(np.linalg.norm(zh_proj[i] - en_proj[j]))
    cross_dists = np.array(cross_dists)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram comparison
    ax1.hist(same_prob_dists, bins=30, alpha=0.7, color="#2ecc71", label="Same problem, zh↔en",
             density=True, edgecolor="black", linewidth=0.5)
    ax1.hist(diff_prob_dists, bins=30, alpha=0.5, color="#e74c3c", label="Diff problem, same lang",
             density=True, edgecolor="black", linewidth=0.5)
    ax1.hist(cross_dists, bins=30, alpha=0.3, color="#3498db", label="Diff problem, cross-lang",
             density=True, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Euclidean Distance in Z", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Distance Distributions in Contrastive Z (L32)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)

    # Ratio
    ratio = np.median(diff_prob_dists) / np.median(same_prob_dists)
    ax1.text(0.98, 0.98, f"Median ratio: {ratio:.1f}×\n(diff/same problem)",
             transform=ax1.transAxes, fontsize=11, verticalalignment="top",
             horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="goldenrod"))

    # Per-category pair distances
    cat_same_dists = []
    cat_labels = []
    for cat_id in range(5):
        mask = categories == cat_id
        dists = np.linalg.norm(zh_proj[mask] - en_proj[mask], axis=1)
        cat_same_dists.append(dists)
        cat_labels.append(CAT_NAMES[cat_id])

    bp = ax2.boxplot(cat_same_dists, labels=cat_labels, patch_artist=True)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("zh↔en Distance in Z", fontsize=12)
    ax2.set_title("Cross-Lingual Distance by Category", fontsize=12, fontweight="bold")
    ax2.tick_params(axis='x', rotation=30)

    fig.suptitle("Z Encodes Problem Identity, Not Language",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_pair_distances.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'fig4_pair_distances.png'}")
    plt.close(fig)


def main():
    print("=" * 70)
    print("VISUALIZATION SUITE — Brain Inside The Machine")
    print("=" * 70)

    problems = generate_problems(200, seed=SEED)
    categories = np.array([p["category"] for p in problems])

    # Try to load cached activations first
    cache_path = OUTPUT_DIR / "viz_activations.npz"
    if cache_path.exists():
        print(f"\nLoading cached activations from {cache_path}...")
        data = np.load(cache_path)
        all_layer_data = {}
        for layer_idx in VIZ_LAYERS:
            all_layer_data[layer_idx] = (data[f"zh_L{layer_idx}"], data[f"en_L{layer_idx}"])
        categories = data["categories"]
        print("  Loaded!")
    else:
        # Load model and extract
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, device_map="auto",
        )
        model.eval()

        print(f"\nExtracting activations at layers {VIZ_LAYERS}...")
        all_layer_data = {}
        for layer_idx in VIZ_LAYERS:
            print(f"\n  Layer {layer_idx}...")
            zh_means, en_means = extract_activations(model, tokenizer, problems, layer_idx)
            all_layer_data[layer_idx] = (zh_means, en_means)

        np.savez(cache_path,
                 **{f"zh_L{l}": all_layer_data[l][0] for l in VIZ_LAYERS},
                 **{f"en_L{l}": all_layer_data[l][1] for l in VIZ_LAYERS},
                 categories=categories)
        print(f"\nActivations saved to {cache_path}")

    # Generate all figures
    zh_L32, en_L32 = all_layer_data[32]

    # Figure 1: Hero t-SNE
    fig1_tsne_hero(zh_L32, en_L32, categories)

    # Figure 2: Layer evolution
    fig2_layer_evolution(all_layer_data, categories)

    # Figure 3: Bridge + sweep (from saved JSON)
    fig3_bridge_and_sweep()

    # Figure 4: Pair distances (after Procrustes rotation)
    z_basis, _, _ = build_contrastive_z(zh_L32, en_L32, K)
    zh_proj = zh_L32 @ z_basis.T
    en_proj_raw = en_L32 @ z_basis.T
    en_proj = apply_procrustes(zh_proj, en_proj_raw)
    fig4_pair_distances(zh_proj, en_proj, categories)

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED")
    print("=" * 70)
    print(f"\n  fig1_tsne_hero.png     — THE hero figure")
    print(f"  fig2_layer_evolution.png — Layer convergence animation")
    print(f"  fig3_bridge_sweep.png  — Bridge R² + layer gradient")
    print(f"  fig4_pair_distances.png — Distance analysis")


if __name__ == "__main__":
    main()
