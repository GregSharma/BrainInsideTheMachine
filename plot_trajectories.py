#!/usr/bin/env python3
"""
Plot per-token cosine trajectories and norms across layers.
Time axis normalized to [0, 1] so ZH and EN align regardless of token count.
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

matplotlib.rcParams['figure.dpi'] = 140
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'sans-serif'

data = json.load(open("output/trajectories_Qwen_Qwen2.5-3B_all.json"))
results = data["results"]
n_layers = data["n_layers"]

# Color scheme
DOMAIN_COLORS = {
    "math": "#9C27B0",
    "commonsense": "#4CAF50",
    "code": "#2196F3",
    "logic": "#FF9800",
}

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Per-token cosine heatmaps — one per problem
#   x-axis: normalized time [0, 1]
#   y-axis: layer (0 to n_layers-1)
#   color: cosine similarity between ZH and EN
# ══════════════════════════════════════════════════════════════════════

n_problems = len(results)
n_cols = 4
n_rows = (n_problems + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.2 * n_rows), squeeze=False)

for idx, r in enumerate(results):
    ax = axes[idx // n_cols][idx % n_cols]
    domain = r["domain"]
    pi = r["problem_idx"]

    # Build cosine matrix [n_layers, max_tokens]
    all_layers = sorted(int(k) for k in r["layers"].keys())
    cos_data = r["layers"]

    # Find max cosine length across layers for this problem
    max_t = max(len(cos_data[str(li)]["cosines"]) for li in all_layers)
    if max_t == 0:
        ax.set_title(f"P{pi} [{domain}] NO DATA", fontsize=9)
        continue

    # Build heatmap matrix
    heatmap = np.full((len(all_layers), max_t), np.nan)
    for yi, li in enumerate(all_layers):
        cos = cos_data[str(li)]["cosines"]
        heatmap[yi, :len(cos)] = cos

    # Normalized time axis
    t_norm = np.linspace(0, 1, max_t)

    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=1.0,
                   extent=[0, 1, len(all_layers) - 0.5, -0.5],
                   interpolation="nearest")

    # Layer ticks (show every 6th)
    tick_positions = list(range(0, len(all_layers), 6))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"L{all_layers[i]}" for i in tick_positions], fontsize=7)

    title_color = DOMAIN_COLORS.get(domain, "black")
    short_prob = r["problem_en"][:35] + "..."
    ax.set_title(f"P{pi} [{domain}] {short_prob}", fontsize=8, color=title_color, fontweight="bold")
    ax.set_xlabel("t (normalized)", fontsize=7)
    if idx % n_cols == 0:
        ax.set_ylabel("Layer", fontsize=8)

# Remove empty axes
for idx in range(n_problems, n_rows * n_cols):
    axes[idx // n_cols][idx % n_cols].set_visible(False)

fig.suptitle("ZH↔EN Cosine Through Layers × Tokens (Qwen2.5-3B)", fontsize=14, fontweight="bold", y=1.01)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(ScalarMappable(norm=Normalize(-0.1, 1.0), cmap="RdYlGn"), cax=cbar_ax, label="cosine(ZH, EN)")

plt.tight_layout(rect=[0, 0, 0.91, 0.98])
plt.savefig("output/fig_cosine_heatmaps.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: output/fig_cosine_heatmaps.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Cosine traces at key layers — all problems overlaid by domain
#   x-axis: normalized time [0, 1]
#   y-axis: cosine similarity
#   One subplot per layer (L0, L9, L18, L27, L35)
# ══════════════════════════════════════════════════════════════════════

key_layers = [0, 9, 18, 27, 35]
key_layers = [l for l in key_layers if l < n_layers]

fig, axes = plt.subplots(1, len(key_layers), figsize=(4 * len(key_layers), 4), sharey=True)
if len(key_layers) == 1:
    axes = [axes]

for ax, li in zip(axes, key_layers):
    for r in results:
        domain = r["domain"]
        cos = r["layers"].get(str(li), {}).get("cosines", [])
        if len(cos) < 2:
            continue
        t_norm = np.linspace(0, 1, len(cos))
        ax.plot(t_norm, cos, color=DOMAIN_COLORS.get(domain, "gray"),
                alpha=0.5, linewidth=1.2)

    ax.set_title(f"Layer {li}", fontsize=12, fontweight="bold")
    ax.set_xlabel("t (normalized)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("cosine(ZH, EN)")

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=DOMAIN_COLORS[d], linewidth=2, label=d)
                   for d in sorted(DOMAIN_COLORS.keys()) if any(r["domain"] == d for r in results)]
axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=9)

fig.suptitle("Cross-Lingual Cosine Trajectories by Layer (Qwen2.5-3B)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/fig_cosine_traces.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: output/fig_cosine_traces.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Norm trajectories — ZH vs EN side by side at key layers
#   Shows the "speed" of computation
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, len(key_layers), figsize=(4 * len(key_layers), 7), sharey="row")

for col, li in enumerate(key_layers):
    ax_zh = axes[0][col]
    ax_en = axes[1][col]

    for r in results:
        domain = r["domain"]
        layer_data = r["layers"].get(str(li), {})
        zh_norms = layer_data.get("zh_norms", [])
        en_norms = layer_data.get("en_norms", [])

        if len(zh_norms) > 1:
            t_zh = np.linspace(0, 1, len(zh_norms))
            ax_zh.plot(t_zh, zh_norms, color=DOMAIN_COLORS.get(domain, "gray"),
                       alpha=0.5, linewidth=1)

        if len(en_norms) > 1:
            t_en = np.linspace(0, 1, len(en_norms))
            ax_en.plot(t_en, en_norms, color=DOMAIN_COLORS.get(domain, "gray"),
                       alpha=0.5, linewidth=1)

    ax_zh.set_title(f"L{li}", fontsize=11, fontweight="bold")
    ax_en.set_xlabel("t (normalized)")
    ax_zh.grid(True, alpha=0.2)
    ax_en.grid(True, alpha=0.2)
    for ax in [ax_zh, ax_en]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

axes[0][0].set_ylabel("ZH ||h||")
axes[1][0].set_ylabel("EN ||h||")

fig.suptitle("Hidden State Norm Trajectories: ZH (top) vs EN (bottom)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/fig_norm_trajectories.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: output/fig_norm_trajectories.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Average cosine by layer — one curve per domain
#   The "convergence profile" but at generation time, not prefill
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))

domains = sorted(set(r["domain"] for r in results))
all_layers_sorted = sorted(int(k) for k in results[0]["layers"].keys())

for domain in domains:
    domain_results = [r for r in results if r["domain"] == domain]
    layer_means = []
    layer_stds = []

    for li in all_layers_sorted:
        all_cos = []
        for r in domain_results:
            cos = r["layers"].get(str(li), {}).get("cosines", [])
            if cos:
                all_cos.append(np.mean(cos))
        layer_means.append(np.mean(all_cos) if all_cos else 0)
        layer_stds.append(np.std(all_cos) if len(all_cos) > 1 else 0)

    layer_means = np.array(layer_means)
    layer_stds = np.array(layer_stds)

    ax.plot(all_layers_sorted, layer_means, color=DOMAIN_COLORS[domain],
            linewidth=2.5, label=domain, marker="o", markersize=3)
    ax.fill_between(all_layers_sorted, layer_means - layer_stds, layer_means + layer_stds,
                    color=DOMAIN_COLORS[domain], alpha=0.15)

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Mean cos(ZH, EN) across tokens", fontsize=12)
ax.set_title("Generation-Time Cross-Lingual Cosine by Layer (Qwen2.5-3B)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, n_layers - 1)

plt.tight_layout()
plt.savefig("output/fig_convergence_by_domain.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: output/fig_convergence_by_domain.png")

print("\nAll figures saved.")
