#!/usr/bin/env python3
"""
Deep trajectory plots — extract the juice from per-token hidden state data.
Assumes output/trajectories_Qwen_Qwen2.5-3B_all.json exists.
"""
import json
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

matplotlib.rcParams['figure.dpi'] = 140
matplotlib.rcParams['font.size'] = 10

data = json.load(open("output/trajectories_Qwen_Qwen2.5-3B_all.json"))
results = data["results"]
n_layers = data["n_layers"]

DC = {"math": "#9C27B0", "commonsense": "#4CAF50", "code": "#2196F3", "logic": "#FF9800"}


# ══════════════════════════════════════════════════════════════════════
# FIG 5: ZH norm vs EN norm — overlaid on SAME normalized time axis
#   If f is shared, norms should track each other regardless of language.
#   Pick 3 representative problems, one per domain, show L18 (convergence zone).
# ══════════════════════════════════════════════════════════════════════

# Pick one representative per domain (longest generation for most data)
reps = {}
for r in results:
    d = r["domain"]
    if d not in reps or min(r["zh_ntok"], r["en_ntok"]) > min(reps[d]["zh_ntok"], reps[d]["en_ntok"]):
        reps[d] = r

domains = sorted(reps.keys())
fig, axes = plt.subplots(2, len(domains), figsize=(4.5 * len(domains), 6), sharey="row")

for col, domain in enumerate(domains):
    r = reps[domain]
    li = "18"  # convergence zone
    zh_norms = np.array(r["layers"][li]["zh_norms"])
    en_norms = np.array(r["layers"][li]["en_norms"])
    cos = np.array(r["layers"][li]["cosines"])

    t_zh = np.linspace(0, 1, len(zh_norms))
    t_en = np.linspace(0, 1, len(en_norms))
    t_cos = np.linspace(0, 1, len(cos))

    # Top row: norms overlaid
    ax = axes[0][col]
    ax.plot(t_zh, zh_norms, color="red", alpha=0.7, linewidth=1, label=f"ZH ({len(zh_norms)}tok)")
    ax.plot(t_en, en_norms, color="blue", alpha=0.7, linewidth=1, label=f"EN ({len(en_norms)}tok)")
    ax.set_title(f"{domain} (L18)\nP{r['problem_idx']}", fontsize=10, fontweight="bold",
                 color=DC[domain])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom row: cosine
    ax2 = axes[1][col]
    ax2.plot(t_cos, cos, color=DC[domain], linewidth=1.2)
    ax2.fill_between(t_cos, cos, alpha=0.2, color=DC[domain])
    ax2.set_ylim(-0.1, 1.0)
    ax2.set_xlabel("t (normalized)")
    ax2.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

axes[0][0].set_ylabel("||h|| (L18)")
axes[1][0].set_ylabel("cos(ZH, EN)")

fig.suptitle("Norm Tracking + Cosine: ZH vs EN at L18 (Convergence Zone)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("output/fig_norm_vs_cosine_L18.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig_norm_vs_cosine_L18.png")


# ══════════════════════════════════════════════════════════════════════
# FIG 6: Cosine VELOCITY — d/dt of cosine trajectory
#   Where is convergence accelerating vs decelerating?
#   Derivative of cosine w.r.t. normalized time, at key layers.
# ══════════════════════════════════════════════════════════════════════

key_layers = [0, 9, 18, 27, 35]
key_layers = [l for l in key_layers if l < n_layers]

fig, axes = plt.subplots(1, len(key_layers), figsize=(4 * len(key_layers), 4), sharey=True)

for ax, li in zip(axes, key_layers):
    for r in results:
        cos = np.array(r["layers"].get(str(li), {}).get("cosines", []))
        if len(cos) < 5:
            continue
        # Smooth with window=5 before differencing
        kernel = np.ones(5) / 5
        cos_smooth = np.convolve(cos, kernel, mode="valid")
        velocity = np.diff(cos_smooth)
        t_norm = np.linspace(0, 1, len(velocity))
        domain = r["domain"]
        ax.plot(t_norm, velocity, color=DC.get(domain, "gray"), alpha=0.4, linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"L{li}", fontsize=12, fontweight="bold")
    ax.set_xlabel("t (normalized)")
    ax.set_ylim(-0.15, 0.15)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("d(cosine)/dt")
fig.suptitle("Cosine Velocity: Where Is Cross-Lingual Convergence Accelerating?", fontsize=13, fontweight="bold")

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=DC[d], linewidth=2, label=d) for d in sorted(DC.keys())]
axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("output/fig_cosine_velocity.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig_cosine_velocity.png")


# ══════════════════════════════════════════════════════════════════════
# FIG 7: Layer-to-layer cosine JUMP — how much does cosine change
#   between consecutive layers at each token position?
#   This is the "where does convergence happen in the stack" question.
#   For each problem, compute cos[layer+1] - cos[layer] at each token.
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 5))
if len(domains) == 1:
    axes = [axes]

for ax, domain in zip(axes, domains):
    domain_results = [r for r in results if r["domain"] == domain]

    # Average the layer jump across problems in this domain
    all_layers_sorted = sorted(int(k) for k in domain_results[0]["layers"].keys())
    jump_matrix = []  # [n_layer_pairs, n_tokens_avg]

    for r in domain_results:
        min_tok = min(len(r["layers"][str(li)]["cosines"]) for li in all_layers_sorted)
        if min_tok < 2:
            continue
        cos_by_layer = np.array([r["layers"][str(li)]["cosines"][:min_tok] for li in all_layers_sorted])
        # layer jump = cos[l+1, t] - cos[l, t]
        jumps = np.diff(cos_by_layer, axis=0)  # [n_layers-1, min_tok]
        jump_matrix.append(jumps)

    if not jump_matrix:
        ax.set_title(f"{domain}: no data")
        continue

    # Average across problems (different token counts — use the min)
    min_t_all = min(j.shape[1] for j in jump_matrix)
    avg_jumps = np.mean([j[:, :min_t_all] for j in jump_matrix], axis=0)

    # Heatmap: x=normalized token, y=layer pair (L0→L1, L1→L2, ...)
    t_norm = np.linspace(0, 1, min_t_all)
    im = ax.imshow(avg_jumps, aspect="auto", cmap="RdBu_r", vmin=-0.05, vmax=0.05,
                   extent=[0, 1, len(all_layers_sorted) - 1.5, -0.5],
                   interpolation="bilinear")

    tick_pos = list(range(0, len(all_layers_sorted) - 1, 6))
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f"L{all_layers_sorted[i]}→{all_layers_sorted[i+1]}" for i in tick_pos], fontsize=7)
    ax.set_xlabel("t (normalized)")
    ax.set_title(f"{domain}", fontsize=12, fontweight="bold", color=DC[domain])

axes[0].set_ylabel("Layer transition")
fig.suptitle("Layer-to-Layer Cosine Jump: Where Does Convergence Happen?",
             fontsize=13, fontweight="bold", y=1.02)

# Colorbar
cbar = fig.colorbar(ScalarMappable(norm=Normalize(-0.05, 0.05), cmap="RdBu_r"),
                    ax=axes, shrink=0.8, label="Δcos (blue=convergence, red=divergence)")

plt.tight_layout()
plt.savefig("output/fig_layer_jumps.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig_layer_jumps.png")


# ══════════════════════════════════════════════════════════════════════
# FIG 8: Norm ratio ZH/EN through layers — are they computing at
#   different "energies"? Ratio > 1 = ZH working harder.
#   One curve per problem, faceted by domain.
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, len(domains), figsize=(4.5 * len(domains), 4.5), sharey=True)

for ax, domain in zip(axes, domains):
    domain_results = [r for r in results if r["domain"] == domain]
    all_layers_sorted = sorted(int(k) for k in domain_results[0]["layers"].keys())

    for r in domain_results:
        ratios = []
        for li in all_layers_sorted:
            zh_n = r["layers"][str(li)]["zh_norms"]
            en_n = r["layers"][str(li)]["en_norms"]
            if zh_n and en_n:
                # Mean norm ratio across tokens
                mean_zh = np.mean(zh_n)
                mean_en = np.mean(en_n)
                ratios.append(mean_zh / mean_en if mean_en > 0 else 1.0)
            else:
                ratios.append(1.0)

        ax.plot(all_layers_sorted, ratios, color=DC[domain], alpha=0.5, linewidth=1.2,
                marker=".", markersize=2)

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(domain, fontsize=12, fontweight="bold", color=DC[domain])
    ax.set_xlabel("Layer")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0.5, 2.0)

axes[0].set_ylabel("||h_ZH|| / ||h_EN|| (mean over tokens)")
fig.suptitle("Norm Ratio ZH/EN Through Layers: Which Language Works Harder?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("output/fig_norm_ratio.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig_norm_ratio.png")


# ══════════════════════════════════════════════════════════════════════
# FIG 9: Cosine VOLATILITY — rolling std of cosine trajectory.
#   High volatility = unstable convergence. Low = locked in.
#   Like realized vol on a price series.
# ══════════════════════════════════════════════════════════════════════

window = 10

fig, axes = plt.subplots(1, len(key_layers), figsize=(4 * len(key_layers), 4), sharey=True)

for ax, li in zip(axes, key_layers):
    for r in results:
        cos = np.array(r["layers"].get(str(li), {}).get("cosines", []))
        if len(cos) < window + 2:
            continue
        # Rolling std
        vol = np.array([np.std(cos[max(0, i - window):i + 1]) for i in range(len(cos))])
        t_norm = np.linspace(0, 1, len(vol))
        domain = r["domain"]
        ax.plot(t_norm, vol, color=DC.get(domain, "gray"), alpha=0.4, linewidth=0.8)

    ax.set_title(f"L{li}", fontsize=12, fontweight="bold")
    ax.set_xlabel("t (normalized)")
    ax.set_ylim(0, 0.35)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(f"Rolling σ(cosine), w={window}")
fig.suptitle("Cosine Volatility: Where Is Cross-Lingual Alignment Unstable?",
             fontsize=13, fontweight="bold")

legend_elements = [Line2D([0], [0], color=DC[d], linewidth=2, label=d) for d in sorted(DC.keys())]
axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("output/fig_cosine_volatility.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig_cosine_volatility.png")


# ══════════════════════════════════════════════════════════════════════
# FIG 10: Single-problem deep dive — the Dijkstra problem
#   Full layer x token heatmap + norm overlay + cosine trace.
#   The "chart" a trader would stare at.
# ══════════════════════════════════════════════════════════════════════

# Find Dijkstra
dijk = [r for r in results if "dijkstra" in r.get("problem_en", "").lower()]
if not dijk:
    dijk = [r for r in results if r["domain"] == "code"]
target = dijk[0] if dijk else results[-1]

all_layers_sorted = sorted(int(k) for k in target["layers"].keys())
max_t = max(len(target["layers"][str(li)]["cosines"]) for li in all_layers_sorted)

# Build full heatmap
heatmap = np.full((len(all_layers_sorted), max_t), np.nan)
for yi, li in enumerate(all_layers_sorted):
    cos = target["layers"][str(li)]["cosines"]
    heatmap[yi, :len(cos)] = cos

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

# Top: heatmap
ax1 = fig.add_subplot(gs[0])
t_norm = np.linspace(0, 1, max_t)
im = ax1.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=1.0,
                extent=[0, 1, len(all_layers_sorted) - 0.5, -0.5],
                interpolation="bilinear")
ax1.set_ylabel("Layer")
tick_pos = list(range(0, len(all_layers_sorted), 4))
ax1.set_yticks(tick_pos)
ax1.set_yticklabels([f"L{all_layers_sorted[i]}" for i in tick_pos])
fig.colorbar(im, ax=ax1, shrink=0.6, label="cos(ZH, EN)")
ax1.set_title(f"Deep Dive: P{target['problem_idx']} [{target['domain']}] — {target['problem_en'][:60]}...",
              fontsize=12, fontweight="bold")

# Middle: cosine at L18 (convergence zone)
ax2 = fig.add_subplot(gs[1], sharex=ax1)
cos18 = target["layers"]["18"]["cosines"]
t18 = np.linspace(0, 1, len(cos18))
ax2.plot(t18, cos18, color="#2196F3", linewidth=1.2)
ax2.fill_between(t18, cos18, alpha=0.15, color="#2196F3")
ax2.set_ylabel("cos @ L18")
ax2.set_ylim(-0.1, 1.0)
ax2.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax2.grid(True, alpha=0.15)

# Bottom: ZH and EN norms at L18
ax3 = fig.add_subplot(gs[2], sharex=ax1)
zh_n = target["layers"]["18"]["zh_norms"]
en_n = target["layers"]["18"]["en_norms"]
t_zh = np.linspace(0, 1, len(zh_n))
t_en = np.linspace(0, 1, len(en_n))
ax3.plot(t_zh, zh_n, color="red", alpha=0.7, linewidth=1, label="ZH")
ax3.plot(t_en, en_n, color="blue", alpha=0.7, linewidth=1, label="EN")
ax3.set_ylabel("||h|| @ L18")
ax3.set_xlabel("t (normalized)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.15)

plt.savefig("output/fig_deep_dive.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: fig_deep_dive.png")


print("\nAll v2 figures saved.")
