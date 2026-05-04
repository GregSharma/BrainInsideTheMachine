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
# # Synthetic Z Extraction POC — Phases 2-4
#
# **Spec:** `FriMarch7-Z_POC_Spec_v2.md`  
# **Training:** `z_poc.py` (Config B, imbalanced random + one-hot)  
# **Model:** 6-layer MLP, width 128, GELU, no skip connections  
# **Function:** fixed permutation → cumsum → tanh(0.5·) → reverse → L2 normalize  
# **Languages:** 7 (zh 60%, en 20%, es 8%, ar 5%, ja 4%, ko 2%, sw 1%)  
#
# ### Training Results (all pass < 0.01)
# | Lang | Freq | Test MSE |
# |------|------|----------|
# | Chinese | 60% | 0.000029 |
# | English | 20% | 0.000082 |
# | Spanish | 8% | 0.000166 |
# | Arabic | 5% | 0.000335 |
# | Japanese | 4% | 0.000360 |
# | Korean | 2% | 0.000646 |
# | Swahili | 1% | 0.001286 |
#
# MSE degrades monotonically with training frequency — the **Qwen gradient** reproduced synthetically.

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE
import warnings
import os
from typing import Any

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.facecolor'] = 'white'

# ---------------------------------------------------------------------------
# Output directory for all figures
# ---------------------------------------------------------------------------
FIG_DIR: str = 'output/z_poc_figures'
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name: str) -> None:
    """Save current figure to FIG_DIR and print confirmation."""
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Load trained model + activations produced by z_poc.py
# ---------------------------------------------------------------------------
data: dict[str, Any] = torch.load(
    'output/z_poc_trained.pt', map_location='cpu', weights_only=False
)

K: int          = data['K']           # number of languages
D: int          = data['D']           # input/output dimension
W: int          = data['W']           # hidden width
H: int          = data['H']           # number of hidden layers
N_TEST: int     = data['N_TEST']      # test problems per language
LANG_NAMES: list[str]   = data['lang_names']
LANG_FREQS: list[float] = data['lang_freqs']
rotations: list[torch.Tensor] = data['rotations']
test_meta: dict  = data['test_meta']
test_inputs: torch.Tensor  = data['test_inputs']
test_targets: torch.Tensor = data['test_targets']
# act[ell][h] -> (N_TEST, W) activations for language ell at layer h
act: dict[int, dict[int, torch.Tensor]] = data['act_by_lang_layer']

# ---------------------------------------------------------------------------
# Reconstruct model (no dropout — matches training config)
# ---------------------------------------------------------------------------
class ReasoningMLP(nn.Module):
    """Plain MLP matching the architecture in z_poc.py (no skip connections)."""
    def __init__(self, d_in: int, d_out: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(d_in, width))
        layers.append(nn.GELU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.GELU())
        layers.append(nn.Linear(width, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = ReasoningMLP(D + K, D, W, H)
model.load_state_dict(data['model_state'])
model.eval()

# Per-language colors (consistent across all plots)
LANG_COLORS: list[str] = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf',
]

print(f"Loaded: {K} languages, {len(act[0])} layers, {N_TEST} test problems")
print(f"Activation shape per language per layer: {act[0][0].shape}")
print(f"Figures will be saved to: {FIG_DIR}/")

# %% [markdown]
# ---
# ## t-SNE Layer Evolution
#
# Visualize activations at each hidden layer, colored by language.  
# **Prediction:** Early layers show language separation, later layers show convergence.

# %%
# t-SNE at every hidden layer (skip output layer 6 which is 10-dim)
layers_to_plot = [h for h in range(H)]  # 0..5

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('t-SNE of Hidden Activations by Layer — Colored by Language', fontsize=14, fontweight='bold')

for idx, h in enumerate(layers_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    # Stack all languages
    all_act = torch.cat([act[ell][h] for ell in range(K)], dim=0).numpy()
    lang_labels = np.concatenate([np.full(N_TEST, ell) for ell in range(K)])
    
    tsne = TSNE(n_components=2, perplexity=25, random_state=42)
    emb = tsne.fit_transform(all_act)
    
    for ell in range(K):
        mask = lang_labels == ell
        ax.scatter(emb[mask, 0], emb[mask, 1], c=LANG_COLORS[ell], 
                   label=f"{LANG_NAMES[ell]} ({LANG_FREQS[ell]*100:.0f}%)",
                   s=8, alpha=0.6)
    ax.set_title(f'Layer {h}', fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

axes[0, 0].legend(fontsize=7, loc='upper left', markerscale=2)
plt.tight_layout()
savefig('z_poc_tsne_layers.png')
plt.show()


# %% [markdown]
# ---
# ## Phase 2a — Activation Patching (assert failure)
#
# **Goal:** Show that naive cross-lingual patching BREAKS the model.  
# Replace language ℓ's activations at layer h with language ℓ' activations, continue forward pass.  
# **Expected:** MSE_patch >> MSE_baseline at every layer.

# %%
def forward_from_layer(model: ReasoningMLP, activations: torch.Tensor, start_layer_idx: int) -> torch.Tensor:
    """Run forward pass from a given layer using pre-extracted activations.

    Args:
        model: The trained MLP.
        activations: Tensor of shape (N, W) — activations to inject.
        start_layer_idx: Which GELU layer output to inject at (0-indexed).

    Returns:
        Model output tensor of shape (N, D).

    Each hidden block is Linear+GELU (2 modules). Output is just Linear (1 module).
    Layer i's GELU output feeds into module index 2*(i+1) (next Linear).
    """
    h = activations
    start_module: int = 2 * (start_layer_idx + 1)  # skip past this layer's Linear+GELU
    for module in list(model.net)[start_module:]:
        h = module(h)
    return h

# Baseline: each language through its own pathway
baseline_mse: dict[int, float] = {}
lang_ids: list[int] = test_meta['lang_ids']
lang_masks: dict[int, list[int]] = {}
for ell in range(K):
    lang_masks[ell] = [i for i, l in enumerate(lang_ids) if l == ell]

with torch.no_grad():
    test_pred = model(test_inputs)
    for ell in range(K):
        m = lang_masks[ell]
        baseline_mse[ell] = nn.functional.mse_loss(test_pred[m], test_targets[m]).item()

# Patching: inject Chinese activations into English pathway at each layer
# (and vice versa, and Chinese->Swahili)
patch_pairs: list[tuple[int, int, str]] = [
    (0, 1, 'zh→en'), (1, 0, 'en→zh'), (0, 6, 'zh→sw'), (0, 5, 'zh→ko'),
]

results: dict[str, dict[str, Any]] = {}
for src, tgt, label in patch_pairs:
    results[label] = {'layers': [], 'patch_mse': [], 'baseline': baseline_mse[tgt]}
    for h in range(H):
        with torch.no_grad():
            # Take source language activations at layer h
            src_act = act[src][h]  # (200, W)
            # Forward through remaining layers (which expect target language coords)
            pred = forward_from_layer(model, src_act, h)
            # Compare to target language's expected output
            tgt_targets = test_targets[lang_masks[tgt]]
            mse = nn.functional.mse_loss(pred, tgt_targets).item()
        results[label]['layers'].append(h)
        results[label]['patch_mse'].append(mse)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
for label, r in results.items():
    ax.plot(r['layers'], r['patch_mse'], 'o-', label=f"{label} (patch MSE)", linewidth=2)
    ax.axhline(r['baseline'], linestyle='--', alpha=0.3, color='gray')

ax.set_xlabel('Injection Layer', fontsize=12)
ax.set_ylabel('MSE (log scale)', fontsize=12)
ax.set_yscale('log')
ax.set_title('Phase 2a: Activation Patching — Naive Cross-Lingual Injection', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xticks(range(H))
ax.grid(True, alpha=0.3)

# Annotate baseline
ax.annotate('baseline (no patch)', xy=(H-1, list(baseline_mse.values())[0]), fontsize=8, color='gray')
plt.tight_layout()
savefig('z_poc_phase2a_patching.png')
plt.show()

print("\nPhase 2a Summary:")
for label, r in results.items():
    ratios = [p / r['baseline'] for p in r['patch_mse']]
    print(f"  {label}: patch/baseline ratio = {[f'{x:.1f}x' for x in ratios]}")


# %% [markdown]
# ---
# ## Phase 2b — SVD Subspace Removal
#
# Compute cross-lingual difference matrix, SVD it, project activations onto the orthogonal complement.  
# Measure nearest-neighbor accuracy.  
#
# ## Phase 2c — Random Subspace Control
#
# **Expected:** SVD-identified "language subspace" is no better than random — the **pyrite**.

# %%
def cross_lingual_nn(A1: torch.Tensor, A2: torch.Tensor) -> float:
    """Fraction of points in A1 whose nearest neighbor in A2 is the same problem.

    Args:
        A1: Tensor of shape (M, d) — activations for language 1.
        A2: Tensor of shape (M, d) — activations for language 2, paired by row.

    Returns:
        Float in [0, 1] — fraction of correct nearest-neighbor matches.
    """
    D_mat = torch.cdist(A1.float(), A2.float())  # (M, M)
    nn_idx = D_mat.argmin(dim=1)
    correct = (nn_idx == torch.arange(len(A1))).float().mean().item()
    return correct

# Reference language = Chinese (0)
ref: int = 0
nn_raw: dict[int, dict[int, float]] = {}       # raw NN accuracy per layer per language
nn_svd: dict[int, dict[int, float]] = {}       # after SVD subspace removal
nn_random: dict[int, dict[int, float]] = {}    # after random subspace removal (mean over trials)

k_remove: int = 5  # number of SVD components to remove
n_random_trials: int = 50

for h in range(H):
    nn_raw[h] = {}
    nn_svd[h] = {}
    nn_random[h] = {}
    
    A_ref = act[ref][h]  # (200, W)
    
    for ell in range(1, K):  # compare each language to Chinese
        A_ell = act[ell][h]
        
        # Raw NN
        nn_raw[h][ell] = cross_lingual_nn(A_ref, A_ell)
        
        # SVD subspace removal: find top-k directions of cross-lingual difference,
        # project both sets of activations onto the orthogonal complement
        Delta = (A_ref - A_ell).numpy()  # (M, W)
        U, S, Vt = np.linalg.svd(Delta, full_matrices=False)
        V_k = Vt[:k_remove].T  # (W, k)
        P_perp = torch.eye(W) - torch.from_numpy(V_k @ V_k.T).float()
        A_ref_proj = A_ref @ P_perp
        A_ell_proj = A_ell @ P_perp
        nn_svd[h][ell] = cross_lingual_nn(A_ref_proj, A_ell_proj)
        
        # Random subspace removal: same procedure but with random orthonormal basis
        rand_scores: list[float] = []
        for _ in range(n_random_trials):
            Q, _ = np.linalg.qr(np.random.randn(W, k_remove))
            P_rand = torch.eye(W) - torch.from_numpy(Q @ Q.T).float()
            score = cross_lingual_nn(A_ref @ P_rand, A_ell @ P_rand)
            rand_scores.append(score)
        nn_random[h][ell] = np.mean(rand_scores)

# Plot: NN accuracy per layer, SVD vs Random vs Raw
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Phase 2b/c: SVD vs Random Subspace Removal (k={k_remove}) — NN Accuracy vs Chinese', 
             fontsize=13, fontweight='bold')

for plot_idx, (metric, title) in enumerate([
    (nn_raw, 'Raw (no removal)'),
    (nn_svd, f'SVD top-{k_remove} removed'),
    (nn_random, f'Random {k_remove}-d removed (mean of {n_random_trials})')
]):
    ax = axes[plot_idx]
    for ell in range(1, K):
        vals = [metric[h][ell] for h in range(H)]
        ax.plot(range(H), vals, 'o-', color=LANG_COLORS[ell], 
                label=LANG_NAMES[ell], linewidth=2)
    ax.set_xlabel('Layer'); ax.set_ylabel('NN Accuracy')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xticks(range(H))
    ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
savefig('z_poc_phase2bc_svd_vs_random.png')
plt.show()

# Summary: is SVD better than random?
print("\nPhase 2b/c Summary — SVD vs Random NN accuracy (layer 3, mid-network):")
h_mid: int = 3
for ell in range(1, K):
    svd_val = nn_svd[h_mid][ell]
    rand_val = nn_random[h_mid][ell]
    raw_val = nn_raw[h_mid][ell]
    verdict = "SVD ≈ Random (PYRITE)" if abs(svd_val - rand_val) < 0.1 else "SVD > Random (SVD works)"
    print(f"  zh-{LANG_NAMES[ell][:2]}: raw={raw_val:.3f} svd={svd_val:.3f} rand={rand_val:.3f}  → {verdict}")

# %% [markdown]
# ---
# ## Phase 3 — Procrustes Alignment
#
# At each layer, Procrustes-align all languages to Chinese.  
# Compare NN accuracy: **Procrustes-aligned vs raw**.  
#
# **Key question:** Does Procrustes reveal that the representations ARE related by rotation?

# %%
nn_procrustes: dict[int, dict[int, float]] = {}  # Procrustes-aligned NN accuracy
procrustes_residuals: dict[int, dict[int, float]] = {}  # alignment residual per layer per language

for h in range(H):
    nn_procrustes[h] = {}
    procrustes_residuals[h] = {}
    A_ref = act[ref][h].numpy()
    
    for ell in range(1, K):
        A_ell = act[ell][h].numpy()
        
        # Procrustes: find R such that A_ell @ R ≈ A_ref
        R, scale = orthogonal_procrustes(A_ell, A_ref)
        A_aligned = torch.from_numpy(A_ell @ R).float()
        A_ref_t = torch.from_numpy(A_ref).float()
        
        nn_procrustes[h][ell] = cross_lingual_nn(A_aligned, A_ref_t)
        procrustes_residuals[h][ell] = np.mean(np.linalg.norm(A_ell @ R - A_ref, axis=1))

# Plot: Procrustes vs Raw NN accuracy
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Raw NN
ax = axes[0]
for ell in range(1, K):
    vals = [nn_raw[h][ell] for h in range(H)]
    ax.plot(range(H), vals, 'o-', color=LANG_COLORS[ell], label=LANG_NAMES[ell], linewidth=2)
ax.set_title('Raw NN Accuracy', fontweight='bold')
ax.set_xlabel('Layer'); ax.set_ylabel('NN Accuracy')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.05); ax.set_xticks(range(H))

# Panel 2: Procrustes NN
ax = axes[1]
for ell in range(1, K):
    vals = [nn_procrustes[h][ell] for h in range(H)]
    ax.plot(range(H), vals, 'o-', color=LANG_COLORS[ell], label=LANG_NAMES[ell], linewidth=2)
ax.set_title('Procrustes-Aligned NN Accuracy', fontweight='bold')
ax.set_xlabel('Layer'); ax.set_ylabel('NN Accuracy')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.05); ax.set_xticks(range(H))

# Panel 3: Procrustes residual
ax = axes[2]
for ell in range(1, K):
    vals = [procrustes_residuals[h][ell] for h in range(H)]
    ax.plot(range(H), vals, 'o-', color=LANG_COLORS[ell], label=LANG_NAMES[ell], linewidth=2)
ax.set_title('Procrustes Residual (lower = better fit)', fontweight='bold')
ax.set_xlabel('Layer'); ax.set_ylabel('Mean ||A·R - A_ref||')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xticks(range(H))

fig.suptitle('Phase 3: Procrustes Geometric Check', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig('z_poc_phase3_procrustes.png')
plt.show()

# Find best layer
print("\nProcrustes NN accuracy by layer (averaged across languages):")
for h in range(H):
    avg = np.mean([nn_procrustes[h][ell] for ell in range(1, K)])
    raw_avg = np.mean([nn_raw[h][ell] for ell in range(1, K)])
    boost = avg - raw_avg
    print(f"  Layer {h}: raw={raw_avg:.3f}  procrustes={avg:.3f}  boost={boost:+.3f}")

# %% [markdown]
# ---
# ## Phase 3 — Procrustes NN vs Training Frequency
#
# **The money plot.** Does Procrustes alignment quality correlate with training frequency?  
# If Chinese-English alignment is best and Chinese-Swahili is worst → frequency IS the mechanism.

# %%
# Pick the best Procrustes layer
best_h: int = max(range(H), key=lambda h: np.mean([nn_procrustes[h][ell] for ell in range(1, K)]))
print(f"Best Procrustes layer: {best_h}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: NN accuracy vs frequency
ax = axes[0]
freqs: list[float] = [LANG_FREQS[ell] for ell in range(1, K)]
nn_vals: list[float] = [nn_procrustes[best_h][ell] for ell in range(1, K)]
raw_vals: list[float] = [nn_raw[best_h][ell] for ell in range(1, K)]

for i, ell in enumerate(range(1, K)):
    ax.scatter(freqs[i], nn_vals[i], c=LANG_COLORS[ell], s=100, zorder=5, edgecolors='black')
    ax.scatter(freqs[i], raw_vals[i], c=LANG_COLORS[ell], s=100, zorder=5, marker='x')
    ax.annotate(LANG_NAMES[ell][:2], (freqs[i], nn_vals[i]), textcoords='offset points',
                xytext=(8, 5), fontsize=9)

ax.set_xlabel('Training Frequency', fontsize=12)
ax.set_ylabel('NN Accuracy vs Chinese', fontsize=12)
ax.set_title(f'Layer {best_h}: NN Accuracy vs Training Frequency\n(dots=Procrustes, x=raw)', fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Panel 2: Procrustes residual vs frequency  
ax = axes[1]
resid_vals: list[float] = [procrustes_residuals[best_h][ell] for ell in range(1, K)]
for i, ell in enumerate(range(1, K)):
    ax.scatter(freqs[i], resid_vals[i], c=LANG_COLORS[ell], s=100, zorder=5, edgecolors='black')
    ax.annotate(LANG_NAMES[ell][:2], (freqs[i], resid_vals[i]), textcoords='offset points',
                xytext=(8, 5), fontsize=9)

ax.set_xlabel('Training Frequency', fontsize=12)
ax.set_ylabel('Procrustes Residual', fontsize=12)
ax.set_title(f'Layer {best_h}: Alignment Residual vs Training Frequency', fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig('z_poc_frequency_gradient.png')
plt.show()

# %% [markdown]
# ---
# ## Phase 3b — Ride the Chinese Highway
#
# Procrustes-align at layer A, continue forward through original model.  
# Compare output to Chinese-coordinate correct answer.  
# **Control:** random rotation instead of Procrustes.

# %%
# Use the best Procrustes layer as injection point
layer_A: int = best_h

# Chinese baseline MSE
with torch.no_grad():
    zh_pred = forward_from_layer(model, act[0][layer_A], layer_A)
    zh_targets = test_targets[lang_masks[0]]
    zh_baseline: float = nn.functional.mse_loss(zh_pred, zh_targets).item()

highway_results: list[dict[str, Any]] = []
for ell in range(1, K):
    A_ell = act[ell][layer_A].numpy()
    A_ref = act[0][layer_A].numpy()
    
    # Procrustes align to Chinese
    R, _ = orthogonal_procrustes(A_ell, A_ref)
    aligned = torch.from_numpy(A_ell @ R).float()
    
    # Random rotation control
    Q_rand, _ = np.linalg.qr(np.random.randn(W, W))
    random_rot = torch.from_numpy(A_ell @ Q_rand).float()
    
    with torch.no_grad():
        # Procrustes highway
        pred_proc = forward_from_layer(model, aligned, layer_A)
        mse_proc: float = nn.functional.mse_loss(pred_proc, zh_targets).item()
        
        # Random highway
        pred_rand = forward_from_layer(model, random_rot, layer_A)
        mse_rand: float = nn.functional.mse_loss(pred_rand, zh_targets).item()
        
        # Naive (no alignment)
        pred_naive = forward_from_layer(model, act[ell][layer_A], layer_A)
        mse_naive: float = nn.functional.mse_loss(pred_naive, zh_targets).item()
    
    highway_results.append({
        'lang': LANG_NAMES[ell], 'freq': LANG_FREQS[ell],
        'procrustes': mse_proc, 'random': mse_rand, 'naive': mse_naive,
        'baseline': zh_baseline
    })

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
langs = [r['lang'] for r in highway_results]
x = np.arange(len(langs))
w = 0.22

ax.bar(x - w, [r['naive'] for r in highway_results], w, label='Naive (no align)', color='#d62728', alpha=0.8)
ax.bar(x, [r['procrustes'] for r in highway_results], w, label='Procrustes → zh highway', color='#2ca02c', alpha=0.8)
ax.bar(x + w, [r['random'] for r in highway_results], w, label='Random rot → zh highway', color='#7f7f7f', alpha=0.8)
ax.axhline(zh_baseline, color='blue', linestyle='--', alpha=0.5, label=f'Chinese baseline ({zh_baseline:.5f})')

ax.set_xlabel('Source Language', fontsize=12)
ax.set_ylabel('MSE (log scale)', fontsize=12)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n({r['freq']*100:.0f}%)" for l, r in zip(langs, highway_results)])
ax.set_title(f'Phase 3b: Ride the Chinese Highway (inject at layer {layer_A})', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
savefig('z_poc_phase3b_highway.png')
plt.show()

print(f"\nChinese baseline MSE: {zh_baseline:.6f}")
print(f"\nPhase 3b Summary (layer {layer_A}):")
for r in highway_results:
    ratio = r['procrustes'] / zh_baseline
    verdict = 'PASS' if ratio < 2.0 else 'WEAK' if ratio < 10 else 'FAIL'
    print(f"  {r['lang']:>8s}: proc={r['procrustes']:.5f} ({ratio:.1f}x baseline)  "
          f"random={r['random']:.5f}  naive={r['naive']:.5f}  [{verdict}]")

# %% [markdown]
# ---
# ## Phase 4 — Pack-a-Punch: Train Reasoning Extractor
#
# Train a small MLP to map Procrustes-aligned layer A activations → Procrustes-aligned layer B activations.  
# - **Train languages:** zh, en, es, ar, ja (1-5)  
# - **Test languages:** ko, sw (6-7) — **never seen by extractor**  
#
# ### Controls
# 1. **No Procrustes** — raw activations  
# 2. **Scrambled** — break problem correspondence  
# 3. **Random subspace** — random rotation instead of Procrustes

# %%
# Layer A = best Procrustes layer, Layer B = last hidden layer
layer_B = H - 1  # layer 5
print(f"Layer A (inflection) = {layer_A}, Layer B (reasoning complete) = {layer_B}")

# Procrustes-align at BOTH endpoints
aligned_A = {}  # aligned_A[ell] = (200, W)
aligned_B = {}
raw_A = {}      # unaligned
raw_B = {}

for ell in range(K):
    raw_A[ell] = act[ell][layer_A]
    raw_B[ell] = act[ell][layer_B]
    
    if ell == 0:  # Chinese is reference
        aligned_A[ell] = act[ell][layer_A]
        aligned_B[ell] = act[ell][layer_B]
    else:
        # Align at layer A
        R_A, _ = orthogonal_procrustes(act[ell][layer_A].numpy(), act[0][layer_A].numpy())
        aligned_A[ell] = torch.from_numpy(act[ell][layer_A].numpy() @ R_A).float()
        
        # Align at layer B
        R_B, _ = orthogonal_procrustes(act[ell][layer_B].numpy(), act[0][layer_B].numpy())
        aligned_B[ell] = torch.from_numpy(act[ell][layer_B].numpy() @ R_B).float()

# Split: train on languages 0-4, test on 5-6 (Korean, Swahili)
train_langs = list(range(5))   # zh, en, es, ar, ja
test_langs = [5, 6]            # ko, sw

def build_dataset(act_in, act_out, langs, scramble=False):
    """Build (X, Y) dataset from activation dicts."""
    Xs, Ys = [], []
    for ell in langs:
        Xs.append(act_in[ell])
        y = act_out[ell]
        if scramble:
            perm = torch.randperm(len(y))
            y = y[perm]
        Ys.append(y)
    return torch.cat(Xs), torch.cat(Ys)

def train_extractor(X_train, Y_train, width=32, epochs=500, lr=1e-3):
    """Train a small MLP extractor."""
    d_in, d_out = X_train.shape[1], Y_train.shape[1]
    ext = nn.Sequential(
        nn.Linear(d_in, width), nn.GELU(),
        nn.Linear(width, width), nn.GELU(),
        nn.Linear(width, d_out)
    )
    opt = torch.optim.Adam(ext.parameters(), lr=lr)
    for ep in range(epochs):
        pred = ext(X_train)
        loss = nn.functional.mse_loss(pred, Y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    ext.eval()
    return ext

def compute_r2(ext, X, Y):
    """R² score."""
    with torch.no_grad():
        pred = ext(X)
    ss_res = ((pred - Y) ** 2).sum().item()
    ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().item()
    return 1 - ss_res / ss_tot

print("Training extractors (this takes ~10s)...")

# 1. Procrustes-aligned extractor
X_tr_proc, Y_tr_proc = build_dataset(aligned_A, aligned_B, train_langs)
ext_proc = train_extractor(X_tr_proc, Y_tr_proc)

# 2. Raw (no Procrustes) extractor
X_tr_raw, Y_tr_raw = build_dataset(raw_A, raw_B, train_langs)
ext_raw = train_extractor(X_tr_raw, Y_tr_raw)

# 3. Scrambled control
X_tr_scr, Y_tr_scr = build_dataset(aligned_A, aligned_B, train_langs, scramble=True)
ext_scr = train_extractor(X_tr_scr, Y_tr_scr)

# 4. Random rotation control
random_A = {}
random_B = {}
for ell in range(K):
    Q_A, _ = np.linalg.qr(np.random.randn(W, W))
    Q_B, _ = np.linalg.qr(np.random.randn(W, W))
    random_A[ell] = torch.from_numpy(act[ell][layer_A].numpy() @ Q_A).float()
    random_B[ell] = torch.from_numpy(act[ell][layer_B].numpy() @ Q_B).float()
X_tr_rand, Y_tr_rand = build_dataset(random_A, random_B, train_langs)
ext_rand = train_extractor(X_tr_rand, Y_tr_rand)

print("Done training. Computing R² scores...")

# %%
# Compute per-language R² for all methods
methods = {
    'Procrustes': (ext_proc, aligned_A, aligned_B),
    'Raw (no align)': (ext_raw, raw_A, raw_B),
    'Scrambled': (ext_scr, aligned_A, aligned_B),  # same input, but trained on scrambled
    'Random rot': (ext_rand, random_A, random_B),
}

r2_results = {}  # r2_results[method][ell] = R²
for method_name, (ext, a_in, a_out) in methods.items():
    r2_results[method_name] = {}
    for ell in range(K):
        r2 = compute_r2(ext, a_in[ell], a_out[ell])
        r2_results[method_name][ell] = r2

# THE DECISION MATRIX
print("=" * 80)
print("PHASE 4 — DECISION MATRIX")
print("=" * 80)
print(f"{'Language':<12} {'Freq':>6} {'Procrustes':>12} {'Raw':>12} {'Scrambled':>12} {'Random':>12}  {'Set':>6}")
print("-" * 80)
for ell in range(K):
    split = 'TRAIN' if ell in train_langs else 'TEST'
    print(f"{LANG_NAMES[ell]:<12} {LANG_FREQS[ell]*100:>5.0f}% "
          f"{r2_results['Procrustes'][ell]:>12.4f} "
          f"{r2_results['Raw (no align)'][ell]:>12.4f} "
          f"{r2_results['Scrambled'][ell]:>12.4f} "
          f"{r2_results['Random rot'][ell]:>12.4f}  "
          f"{split:>6}")

# Aggregate
print("-" * 80)
for method in methods:
    train_r2 = np.mean([r2_results[method][ell] for ell in train_langs])
    test_r2 = np.mean([r2_results[method][ell] for ell in test_langs])
    print(f"  {method}: train_avg={train_r2:.4f}  test_avg={test_r2:.4f}")

# Verdict
proc_test = np.mean([r2_results['Procrustes'][ell] for ell in test_langs])
raw_test = np.mean([r2_results['Raw (no align)'][ell] for ell in test_langs])
scr_test = np.mean([r2_results['Scrambled'][ell] for ell in test_langs])

print("\n" + "=" * 80)
if proc_test > raw_test + 0.1 and proc_test > scr_test + 0.1:
    print("VERDICT: GOLD — Procrustes separates pyrite from gold.")
    print("Reasoning is extractable once coordinates are aligned. The factorization is real.")
elif proc_test > scr_test + 0.1 and abs(proc_test - raw_test) < 0.1:
    print("VERDICT: Pyrite was never the problem. Raw extraction works.")
    print("The model learned to canonicalize on its own.")
elif proc_test > raw_test + 0.1 and abs(raw_test - scr_test) < 0.05:
    print("VERDICT: Strong coordinate entanglement. Procrustes is essential.")
else:
    print("VERDICT: Unclear. Need further investigation.")
print("=" * 80)

# %%
# Visualization: R² per language per method
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Bar chart of R² by method and language
ax = axes[0]
x = np.arange(K)
w = 0.18
for i, (method, color) in enumerate([
    ('Procrustes', '#2ca02c'), ('Raw (no align)', '#d62728'),
    ('Scrambled', '#7f7f7f'), ('Random rot', '#9467bd')
]):
    vals = [r2_results[method][ell] for ell in range(K)]
    bars = ax.bar(x + i*w - 1.5*w, vals, w, label=method, color=color, alpha=0.85)

# Mark train/test split
ax.axvline(4.5, color='black', linestyle=':', alpha=0.5)
ax.text(2, ax.get_ylim()[1]*0.95, 'TRAIN', ha='center', fontsize=10, fontstyle='italic')
ax.text(5.5, ax.get_ylim()[1]*0.95, 'TEST', ha='center', fontsize=10, fontstyle='italic', color='red')

ax.set_xticks(x)
ax.set_xticklabels([f"{LANG_NAMES[ell]}\n({LANG_FREQS[ell]*100:.0f}%)" for ell in range(K)], fontsize=8)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('Phase 4: Extractor R² by Language and Method', fontweight='bold')
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: R² vs training frequency (Procrustes only)
ax = axes[1]
for ell in range(K):
    marker = 's' if ell in test_langs else 'o'
    edge = 'red' if ell in test_langs else 'black'
    ax.scatter(LANG_FREQS[ell], r2_results['Procrustes'][ell], 
               c=LANG_COLORS[ell], s=120, marker=marker, edgecolors=edge, linewidth=2, zorder=5)
    ax.annotate(LANG_NAMES[ell], (LANG_FREQS[ell], r2_results['Procrustes'][ell]),
                textcoords='offset points', xytext=(10, 5), fontsize=9)

ax.set_xlabel('Training Frequency', fontsize=12)
ax.set_ylabel('Procrustes Extractor R²', fontsize=12)
ax.set_title('R² vs Training Frequency\n(squares = held-out test languages)', fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig('z_poc_phase4_extractor.png')
plt.show()

# %% [markdown]
# ---
# ## t-SNE: Before and After Procrustes
#
# Side-by-side: raw activations vs Procrustes-aligned activations at the best layer.  
# After alignment, all languages should overlap if Z is real.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for panel, (title, act_dict) in enumerate([
    (f'Raw Activations (Layer {best_h})', {ell: act[ell][best_h] for ell in range(K)}),
    (f'Procrustes-Aligned (Layer {best_h})', aligned_A)
]):
    ax = axes[panel]
    all_act = torch.cat([act_dict[ell] for ell in range(K)], dim=0).numpy()
    lang_labels = np.concatenate([np.full(N_TEST, ell) for ell in range(K)])
    
    tsne = TSNE(n_components=2, perplexity=25, random_state=42)
    emb = tsne.fit_transform(all_act)
    
    for ell in range(K):
        mask = lang_labels == ell
        marker = 's' if ell in test_langs else 'o'
        ax.scatter(emb[mask, 0], emb[mask, 1], c=LANG_COLORS[ell],
                   label=f"{LANG_NAMES[ell]} ({LANG_FREQS[ell]*100:.0f}%)",
                   s=12, alpha=0.6, marker=marker)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc='upper left', markerscale=2)

fig.suptitle('Procrustes Alignment: Before vs After', fontsize=14, fontweight='bold')
plt.tight_layout()
savefig('z_poc_tsne_procrustes_comparison.png')
plt.show()

# %% [markdown]
# ---
# ## Summary
#
# ### Phase 2 (Naive Extraction)
# - **2a (Patching):** Cross-lingual activation patching breaks the model — activations in wrong coordinates.
# - **2b/c (SVD vs Random):** SVD subspace removal ≈ random subspace removal → **pyrite confirmed.**
#
# ### Phase 3 (Procrustes)
# - Procrustes alignment boosts NN accuracy across all layers.
# - Alignment quality correlates with training frequency (the Qwen gradient).
# - **3b (Highway):** Procrustes-aligned activations routed through Chinese pathway produce reasonable outputs.
#
# ### Phase 4 (Extractor)
# - Procrustes-aligned extractor achieves high R² on train languages.
# - **Held-out languages (Korean, Swahili)** — the key test of generalization.
# - Scrambled and random controls confirm the extractor learns reasoning, not statistics.
