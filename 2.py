# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Subspace Geometry Experiments
#
# Three experiments probing whether transformer depth is doing real geometric work
# or spinning its wheels in redundant subspaces.
#
# Builds on 1.py's model + effective rank analysis. Reuses the model but does NOT
# recompute effective ranks (those are already in the CSV).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from utils import get_attn_subspace, get_model_dims, subspace_similarity

plt.style.use("seaborn-v0_8-whitegrid")

# %%
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    torch_dtype=torch.float16,
    device_map="auto",
)
dims = get_model_dims(model)
L, d, d_ff, h, GQA = dims["L"], dims["d"], dims["d_ff"], dims["h"], dims["GQA"]
d_head = dims["d_head"]

print(f"L={L}, d={d}, d_ff={d_ff}, h={h}, GQA={GQA}, d_head={d_head}")

# %% [markdown]
# ## Experiment 1: Subspace Overlap Across Layers
#
# For a fixed head, extract the top-k right singular vectors of each layer's
# attention kernel. Compute pairwise Grassmann similarity. If layers query the
# SAME subspace, the matrix is uniformly high and depth is redundant. If
# block-diagonal, clusters of layers share subspaces with sharp phase transitions.

# %%
HEAD = 0
K = 20

subspaces = []
for l in tqdm(range(L), desc="Extracting subspaces"):
    subspaces.append(get_attn_subspace(model, l, h, GQA, d, HEAD, k=K))

# %%
sim_matrix = np.zeros((L, L))
for i in tqdm(range(L), desc="Pairwise similarity"):
    for j in range(i, L):
        s = subspace_similarity(subspaces[i], subspaces[j])
        sim_matrix[i, j] = s
        sim_matrix[j, i] = s

# %%
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
fig.colorbar(im, ax=ax, label="Subspace similarity")
ax.set_xlabel("Layer")
ax.set_ylabel("Layer")
ax.set_title(f"Attention subspace overlap (head {HEAD}, top-{K})")
plt.tight_layout()

# Print block structure diagnostics
print(f"\nDiagonal (self-similarity): all {sim_matrix[np.eye(L, dtype=bool)].mean():.4f} (sanity check)")
off_diag = sim_matrix[~np.eye(L, dtype=bool)]
print(f"Off-diagonal: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}, "
      f"min={off_diag.min():.4f}, max={off_diag.max():.4f}")

# Adjacent vs distant layer similarity
adjacent_sims = np.array([sim_matrix[i, i + 1] for i in range(L - 1)])
distant_sims = np.array([sim_matrix[i, j] for i in range(L) for j in range(L) if abs(i - j) > L // 3])
print(f"Adjacent layers: mean={adjacent_sims.mean():.4f}, std={adjacent_sims.std():.4f}")
print(f"Distant layers (>{L // 3} apart): mean={distant_sims.mean():.4f}, std={distant_sims.std():.4f}")

# %% [markdown]
# ## Experiment 2: Does the Bottleneck Layer Converge to a Shared Subspace?
#
# Layer 33 showed a low effective rank in experiment 1.py. If it's a convergence
# point, similarity to it should INCREASE as you approach from both directions.

# %%
BOTTLENECK_LAYER = 33
# Clamp to valid range in case model has fewer layers
BOTTLENECK_LAYER = min(BOTTLENECK_LAYER, L - 1)

ref_subspace = subspaces[BOTTLENECK_LAYER]  # reuse already-extracted subspaces
sims_to_ref = np.array([
    subspace_similarity(ref_subspace, subspaces[l]) for l in range(L)
])

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(L), sims_to_ref, marker=".", markersize=4)
ax.axvline(BOTTLENECK_LAYER, color="r", linestyle="--", alpha=0.5, label=f"Layer {BOTTLENECK_LAYER}")
ax.set_xlabel("Layer")
ax.set_ylabel(f"Similarity to layer {BOTTLENECK_LAYER}")
ax.set_title(f"Does layer {BOTTLENECK_LAYER} bottleneck converge to a shared subspace? (head {HEAD}, top-{K})")
ax.legend()
plt.tight_layout()

# Convergence test: does similarity increase approaching the bottleneck?
before = sims_to_ref[:BOTTLENECK_LAYER]
after = sims_to_ref[BOTTLENECK_LAYER + 1:]
if len(before) > 1:
    before_trend = np.corrcoef(range(len(before)), before)[0, 1]
    print(f"Pre-bottleneck trend (correlation with layer index): {before_trend:+.4f}")
    print(f"  (positive = similarity increases approaching bottleneck)")
if len(after) > 1:
    after_trend = np.corrcoef(range(len(after)), after)[0, 1]
    print(f"Post-bottleneck trend: {after_trend:+.4f}")
    print(f"  (negative = similarity increases approaching bottleneck from late layers)")
print(f"Mean similarity: before={before.mean():.4f}, after={after.mean():.4f}")

# %% [markdown]
# ## Experiment 3: FFN-Attention Alignment
#
# Project W_gate onto the attention subspace and measure what fraction of FFN
# energy lives there. If FFN operates in the orthogonal complement, the two
# systems are cleanly separated. If aligned, they're coupled.

# %%
def ffn_attention_alignment(model, layer_idx, head_idx, k, subspace=None):
    """Fraction of W_gate's energy lying in the attention subspace.

    If subspace is provided, reuses it instead of recomputing.
    """
    if subspace is None:
        Vh = get_attn_subspace(model, layer_idx, h, GQA, d, head_idx, k=k)
    else:
        Vh = subspace
    P = Vh.T @ Vh  # projection onto attention subspace, (d, d)

    layer = model.model.layers[layer_idx]
    W_gate = layer.mlp.gate_proj.weight.data.float().cpu()  # (d_ff, d)

    projected = W_gate @ P  # (d_ff, d)
    energy_in = (projected ** 2).sum().item()
    energy_total = (W_gate ** 2).sum().item()

    return energy_in / energy_total


# %%
chance_level = K / d
alignments = np.array([
    ffn_attention_alignment(model, l, HEAD, K, subspace=subspaces[l])
    for l in tqdm(range(L), desc="FFN-Attention alignment")
])

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(L), alignments, marker=".", markersize=4)
ax.axhline(chance_level, color="r", linestyle="--", label=f"Chance level ({chance_level:.4f})")
ax.set_xlabel("Layer")
ax.set_ylabel("Fraction of FFN energy in attention subspace")
ax.set_title(f"FFN-Attention alignment (head {HEAD}, top-{K})")
ax.legend()
plt.tight_layout()

above_chance = (alignments > chance_level).sum()
below_chance = (alignments <= chance_level).sum()
mean_ratio = alignments.mean() / chance_level
print(f"Chance level: {chance_level:.4f}")
print(f"Mean alignment: {alignments.mean():.4f} ({mean_ratio:.2f}x chance)")
print(f"Layers above chance: {above_chance}/{L}, below: {below_chance}/{L}")
print(f"Min: {alignments.min():.4f} (layer {alignments.argmin()}), "
      f"Max: {alignments.max():.4f} (layer {alignments.argmax()})")

# %% [markdown]
# ## Multi-Head Analysis
#
# The above used head 0. Run across multiple heads to check robustness.

# %%
# Subspace overlap: average across a sample of heads
N_HEADS_SAMPLE = min(4, h)
head_sample = np.linspace(0, h - 1, N_HEADS_SAMPLE, dtype=int)

avg_sim_matrix = np.zeros((L, L))
per_head_sims_to_ref = np.zeros((len(head_sample), L))
per_head_alignments = np.zeros((len(head_sample), L))

for hi_idx, hi in enumerate(tqdm(head_sample, desc="Multi-head analysis")):
    subs = [get_attn_subspace(model, l, h, GQA, d, hi, k=K) for l in range(L)]

    # Pairwise similarity
    sm = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            s = subspace_similarity(subs[i], subs[j])
            sm[i, j] = s
            sm[j, i] = s
    avg_sim_matrix += sm

    # Similarity to bottleneck
    ref = subs[BOTTLENECK_LAYER]
    per_head_sims_to_ref[hi_idx] = [subspace_similarity(ref, subs[l]) for l in range(L)]

    # FFN alignment
    per_head_alignments[hi_idx] = [
        ffn_attention_alignment(model, l, hi, K, subspace=subs[l]) for l in range(L)
    ]

avg_sim_matrix /= len(head_sample)

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Avg similarity matrix
im = axes[0].imshow(avg_sim_matrix, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
fig.colorbar(im, ax=axes[0], label="Similarity")
axes[0].set_title(f"Avg subspace overlap ({len(head_sample)} heads, top-{K})")
axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Layer")

# Avg similarity to bottleneck
mean_sims = per_head_sims_to_ref.mean(axis=0)
std_sims = per_head_sims_to_ref.std(axis=0)
axes[1].plot(mean_sims, marker=".", markersize=3)
axes[1].fill_between(range(L), mean_sims - std_sims, mean_sims + std_sims, alpha=0.2)
axes[1].axvline(BOTTLENECK_LAYER, color="r", linestyle="--", alpha=0.5)
axes[1].set_title(f"Similarity to layer {BOTTLENECK_LAYER} (mean ± std)")
axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Similarity")

# Avg FFN alignment
mean_align = per_head_alignments.mean(axis=0)
std_align = per_head_alignments.std(axis=0)
axes[2].plot(mean_align, marker=".", markersize=3)
axes[2].fill_between(range(L), mean_align - std_align, mean_align + std_align, alpha=0.2)
axes[2].axhline(chance_level, color="r", linestyle="--", label=f"Chance ({chance_level:.4f})")
axes[2].set_title("FFN-Attention alignment (mean ± std)")
axes[2].set_xlabel("Layer"); axes[2].set_ylabel("Fraction in attn subspace")
axes[2].legend()

plt.suptitle(f"Multi-head summary (heads {list(head_sample)})", fontsize=14)
plt.tight_layout()

# %% [markdown]
# ## Export Results

# %%
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
model_name_fmtd = model.config.name_or_path.replace("/", "_")

# Save raw data
np.save(OUTPUT_DIR / f"{model_name_fmtd}_subspace_sim_matrix_head{HEAD}_k{K}.npy", sim_matrix)
np.save(OUTPUT_DIR / f"{model_name_fmtd}_avg_sim_matrix_k{K}.npy", avg_sim_matrix)
np.save(OUTPUT_DIR / f"{model_name_fmtd}_sims_to_layer{BOTTLENECK_LAYER}_head{HEAD}_k{K}.npy", sims_to_ref)
np.save(OUTPUT_DIR / f"{model_name_fmtd}_ffn_attn_alignment_head{HEAD}_k{K}.npy", alignments)
np.save(OUTPUT_DIR / f"{model_name_fmtd}_multi_head_sims_to_ref_k{K}.npy", per_head_sims_to_ref)
np.save(OUTPUT_DIR / f"{model_name_fmtd}_multi_head_alignments_k{K}.npy", per_head_alignments)

print("Saved all numpy arrays to output/")

# %%
# Load effective rank data from 1.py
eff_rank_path = OUTPUT_DIR / f"{model_name_fmtd}_Layer_Weight_Eff_Ranks.csv"
df_eff = pd.read_csv(eff_rank_path) if eff_rank_path.exists() else None
if df_eff is not None:
    print(f"Loaded effective ranks from {eff_rank_path}: {len(df_eff)} rows")
else:
    print(f"WARNING: {eff_rank_path} not found — effective rank sections will be skipped in report")


# %% [markdown]
# ## Generate Markdown Report

# %%
def generate_markdown_report():
    lines = []
    w = lines.append

    w(f"# Subspace Geometry Analysis: {model.config.name_or_path}")
    w("")
    w("## Model Configuration")
    w("")
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w(f"| Layers (L) | {L} |")
    w(f"| Hidden size (d) | {d} |")
    w(f"| FFN intermediate (d_ff) | {d_ff} |")
    w(f"| Attention heads (h) | {h} |")
    w(f"| GQA heads | {GQA} |")
    w(f"| Head dim (d_head) | {d_head} |")
    w(f"| Subspace rank (k) | {K} |")
    w("")

    # --- Experiment 1 ---
    w("## Experiment 1: Subspace Overlap Across Layers")
    w("")
    w(f"For head {HEAD}, extracted top-{K} right singular vectors of each layer's "
      f"attention kernel (W_Q_h^T @ W_K_h) and computed pairwise Grassmann similarity.")
    w("")
    w("**Interpretation:**")
    w("- Uniformly high similarity → depth is redundant (layers query the same subspace)")
    w("- Block-diagonal structure → layers cluster into phases with sharp transitions")
    w("- Low off-diagonal → each layer queries a genuinely different subspace")
    w("")

    w("### Summary Statistics")
    w("")
    off_diag = sim_matrix[~np.eye(L, dtype=bool)]
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Off-diagonal mean | {off_diag.mean():.4f} |")
    w(f"| Off-diagonal std | {off_diag.std():.4f} |")
    w(f"| Off-diagonal min | {off_diag.min():.4f} |")
    w(f"| Off-diagonal max | {off_diag.max():.4f} |")

    adjacent = np.array([sim_matrix[i, i + 1] for i in range(L - 1)])
    far = np.array([sim_matrix[i, j] for i in range(L) for j in range(L) if abs(i - j) > L // 3])
    w(f"| Adjacent layer mean | {adjacent.mean():.4f} |")
    w(f"| Adjacent layer std | {adjacent.std():.4f} |")
    w(f"| Distant layer (>{L // 3} apart) mean | {far.mean():.4f} |")
    w(f"| Distant layer (>{L // 3} apart) std | {far.std():.4f} |")
    w(f"| Adjacent vs distant gap | {adjacent.mean() - far.mean():+.4f} |")
    w("")

    w("### Full Similarity Matrix (Layer × Layer)")
    w("")
    sm_df = pd.DataFrame(sim_matrix, columns=[f"L{j}" for j in range(L)],
                         index=[f"L{i}" for i in range(L)])
    w(sm_df.round(3).to_markdown())
    w("")

    # Cluster detection: find largest drops in adjacent similarity
    w("### Phase Boundary Detection (largest drops in adjacent-layer similarity)")
    w("")
    diffs = np.diff(adjacent)
    drop_order = np.argsort(diffs)
    w("| Rank | Boundary (layers) | Adjacent sim | Change from prev |")
    w("|------|-------------------|-------------|-----------------|")
    for rank, idx in enumerate(drop_order[:10]):
        w(f"| {rank} | {idx+1}→{idx+2} | {adjacent[idx+1]:.4f} | {diffs[idx]:+.4f} |")
    w("")

    # Band analysis: average similarity within distance bands
    w("### Similarity by Layer Distance")
    w("")
    w("| Distance | Mean similarity | Std | N pairs |")
    w("|----------|----------------|-----|---------|")
    for dist in [1, 2, 3, 5, 10, 15, 20, L // 2]:
        if dist >= L:
            continue
        pairs = [(i, i + dist) for i in range(L - dist)]
        vals = np.array([sim_matrix[i, j] for i, j in pairs])
        w(f"| {dist} | {vals.mean():.4f} | {vals.std():.4f} | {len(vals)} |")
    w("")

    # --- Experiment 2 ---
    w(f"## Experiment 2: Convergence to Layer {BOTTLENECK_LAYER} Subspace")
    w("")
    w(f"Similarity between each layer's attention subspace and layer {BOTTLENECK_LAYER}'s "
      f"(head {HEAD}, top-{K}). If similarity increases approaching layer {BOTTLENECK_LAYER} "
      f"from both directions, the network converges toward a specific subspace.")
    w("")

    before = sims_to_ref[:BOTTLENECK_LAYER]
    after = sims_to_ref[BOTTLENECK_LAYER + 1:]

    w("### Summary")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Self-similarity (layer {BOTTLENECK_LAYER}) | {sims_to_ref[BOTTLENECK_LAYER]:.4f} |")
    w(f"| Mean similarity (all layers) | {sims_to_ref.mean():.4f} |")
    if len(before) > 0:
        w(f"| Mean similarity (before bottleneck) | {before.mean():.4f} |")
    if len(after) > 0:
        w(f"| Mean similarity (after bottleneck) | {after.mean():.4f} |")
    if len(before) > 1:
        corr = np.corrcoef(range(len(before)), before)[0, 1]
        w(f"| Pre-bottleneck trend (Pearson r) | {corr:+.4f} |")
    if len(after) > 1:
        corr = np.corrcoef(range(len(after)), after)[0, 1]
        w(f"| Post-bottleneck trend (Pearson r) | {corr:+.4f} |")
    w("")

    w("### Per-Layer Similarity to Bottleneck")
    w("")
    w("| Layer | Similarity |")
    w("|-------|-----------|")
    for l in range(L):
        marker = " ← bottleneck" if l == BOTTLENECK_LAYER else ""
        w(f"| {l} | {sims_to_ref[l]:.4f}{marker} |")
    w("")

    # --- Experiment 3 ---
    w("## Experiment 3: FFN-Attention Alignment")
    w("")
    w(f"Fraction of W_gate's Frobenius energy lying in the top-{K} attention subspace. "
      f"Chance level = k/d = {K}/{d} = {chance_level:.4f}.")
    w("")
    w("**Interpretation:**")
    w(f"- Near chance ({chance_level:.4f}): FFN and attention are randomly oriented")
    w("- Significantly below chance: FFN actively avoids the attention subspace (clean separation)")
    w("- Significantly above chance: FFN and attention are coupled in the same subspace")
    w("")

    w("### Summary")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Chance level (k/d) | {chance_level:.4f} |")
    w(f"| Mean alignment | {alignments.mean():.4f} |")
    w(f"| Mean / chance | {alignments.mean() / chance_level:.2f}x |")
    w(f"| Std | {alignments.std():.4f} |")
    w(f"| Min | {alignments.min():.4f} (layer {int(alignments.argmin())}) |")
    w(f"| Max | {alignments.max():.4f} (layer {int(alignments.argmax())}) |")
    w(f"| Layers above chance | {int((alignments > chance_level).sum())}/{L} |")
    w(f"| Layers below chance | {int((alignments <= chance_level).sum())}/{L} |")
    w("")

    w("### Per-Layer Alignment")
    w("")
    w("| Layer | Alignment | vs Chance |")
    w("|-------|-----------|-----------|")
    for l in range(L):
        ratio = alignments[l] / chance_level
        w(f"| {l} | {alignments[l]:.4f} | {ratio:.2f}x |")
    w("")

    # --- Multi-head ---
    w("## Multi-Head Robustness Check")
    w("")
    w(f"Averaged experiments across heads {list(head_sample)} to check if single-head "
      f"results are representative.")
    w("")

    w("### Averaged Subspace Overlap Matrix Statistics")
    w("")
    avg_off = avg_sim_matrix[~np.eye(L, dtype=bool)]
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Off-diagonal mean | {avg_off.mean():.4f} |")
    w(f"| Off-diagonal std | {avg_off.std():.4f} |")
    avg_adj = np.array([avg_sim_matrix[i, i + 1] for i in range(L - 1)])
    avg_far = np.array([avg_sim_matrix[i, j] for i in range(L) for j in range(L) if abs(i - j) > L // 3])
    w(f"| Adjacent mean | {avg_adj.mean():.4f} |")
    w(f"| Distant mean | {avg_far.mean():.4f} |")
    w("")

    w("### Averaged Similarity Matrix (Layer × Layer)")
    w("")
    avg_df = pd.DataFrame(avg_sim_matrix, columns=[f"L{j}" for j in range(L)],
                          index=[f"L{i}" for i in range(L)])
    w(avg_df.round(3).to_markdown())
    w("")

    w(f"### Per-Head Similarity to Layer {BOTTLENECK_LAYER}")
    w("")
    w("| Layer | " + " | ".join(f"Head {hi}" for hi in head_sample) + " | Mean | Std |")
    w("|-------" + "|------" * len(head_sample) + "|------|-----|")
    for l in range(L):
        vals = per_head_sims_to_ref[:, l]
        row = f"| {l} "
        for v in vals:
            row += f"| {v:.4f} "
        row += f"| {vals.mean():.4f} | {vals.std():.4f} |"
        w(row)
    w("")

    w("### Per-Head FFN-Attention Alignment")
    w("")
    w("| Layer | " + " | ".join(f"Head {hi}" for hi in head_sample) + " | Mean | Std |")
    w("|-------" + "|------" * len(head_sample) + "|------|-----|")
    for l in range(L):
        vals = per_head_alignments[:, l]
        row = f"| {l} "
        for v in vals:
            row += f"| {v:.4f} "
        row += f"| {vals.mean():.4f} | {vals.std():.4f} |"
        w(row)
    w("")

    # --- Cross-reference with effective ranks from 1.py ---
    if df_eff is not None:
        w("## Cross-Reference: Effective Rank vs Subspace Geometry")
        w("")
        w("Comparing the effective rank data from 1.py with the subspace experiments above.")
        w("")

        # Per-layer mean effective rank alongside subspace metrics
        layer_means = df_eff.groupby("layer_idx")["eff_rank"].mean()
        w("| Layer | Mean Eff Rank | Adj Sim | Sim to Bottleneck | FFN-Attn Align |")
        w("|-------|--------------|---------|-------------------|----------------|")
        for l in range(L):
            adj_s = sim_matrix[l, l + 1] if l < L - 1 else float("nan")
            w(f"| {l} | {layer_means.get(l, float('nan')):.1f} | {adj_s:.4f} | "
              f"{sims_to_ref[l]:.4f} | {alignments[l]:.4f} |")
        w("")

        # Correlation analysis
        w("### Correlation Between Metrics")
        w("")
        common_layers = sorted(set(range(L)) & set(layer_means.index))
        if len(common_layers) > 2:
            er = np.array([layer_means[l] for l in common_layers])
            sr = sims_to_ref[common_layers]
            al = alignments[common_layers]

            r_er_sim = np.corrcoef(er, sr)[0, 1]
            r_er_align = np.corrcoef(er, al)[0, 1]
            r_sim_align = np.corrcoef(sr, al)[0, 1]

            w("| Pair | Pearson r |")
            w("|------|-----------|")
            w(f"| Effective rank vs Bottleneck similarity | {r_er_sim:+.4f} |")
            w(f"| Effective rank vs FFN-Attn alignment | {r_er_align:+.4f} |")
            w(f"| Bottleneck similarity vs FFN-Attn alignment | {r_sim_align:+.4f} |")
            w("")

    # --- Key findings ---
    w("## Key Findings Summary")
    w("")

    # Experiment 1 verdict
    if adjacent.mean() - far.mean() > 0.05:
        w("**Experiment 1 (Subspace Overlap):** Adjacent layers share significantly more "
          f"subspace ({adjacent.mean():.3f}) than distant layers ({far.mean():.3f}). "
          "This suggests genuine phase structure — nearby layers refine similar subspaces, "
          "but distant layers target different geometry.")
    elif off_diag.mean() > 0.7:
        w("**Experiment 1 (Subspace Overlap):** High uniform similarity across all layers "
          f"(mean {off_diag.mean():.3f}). Depth may be partially redundant — layers "
          "are querying similar subspaces.")
    else:
        w(f"**Experiment 1 (Subspace Overlap):** Moderate similarity (mean {off_diag.mean():.3f}). "
          "Layers share some subspace but also target distinct directions.")
    w("")

    # Experiment 2 verdict
    if len(before) > 1 and len(after) > 1:
        pre_corr = np.corrcoef(range(len(before)), before)[0, 1]
        post_corr = np.corrcoef(range(len(after)), after)[0, 1]
        if pre_corr > 0.3 and post_corr < -0.3:
            w(f"**Experiment 2 (Bottleneck Convergence):** Layer {BOTTLENECK_LAYER} IS a convergence "
              f"point. Similarity increases approaching it from both sides (pre r={pre_corr:+.3f}, "
              f"post r={post_corr:+.3f}).")
        elif pre_corr > 0.3:
            w(f"**Experiment 2 (Bottleneck Convergence):** Partial convergence — similarity increases "
              f"approaching layer {BOTTLENECK_LAYER} from early layers (r={pre_corr:+.3f}) but not "
              f"from late layers (r={post_corr:+.3f}).")
        else:
            w(f"**Experiment 2 (Bottleneck Convergence):** No clear convergence pattern "
              f"(pre r={pre_corr:+.3f}, post r={post_corr:+.3f}). The low rank at layer "
              f"{BOTTLENECK_LAYER} may be a local anomaly rather than a convergence point.")
    w("")

    # Experiment 3 verdict
    mean_ratio_val = alignments.mean() / chance_level
    if mean_ratio_val > 1.5:
        w(f"**Experiment 3 (FFN-Attention Alignment):** FFN is COUPLED with attention "
          f"({mean_ratio_val:.1f}x chance). They operate in overlapping subspaces, "
          "suggesting FFN transforms representations that attention has already selected.")
    elif mean_ratio_val < 0.7:
        w(f"**Experiment 3 (FFN-Attention Alignment):** FFN AVOIDS the attention subspace "
          f"({mean_ratio_val:.1f}x chance). Clean separation — attention routes in one "
          "subspace, FFN transforms in the orthogonal complement.")
    else:
        w(f"**Experiment 3 (FFN-Attention Alignment):** FFN alignment is near chance "
          f"({mean_ratio_val:.1f}x). No strong coupling or avoidance — the two systems "
          "are approximately independent.")
    w("")

    return "\n".join(lines)


md_content = generate_markdown_report()
md_path = OUTPUT_DIR / "two_output.md"
md_path.write_text(md_content)
print(f"Markdown report written to: {md_path}")

# %%
