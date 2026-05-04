"""
2.py experiments executed via safetensors (memory-safe, no full model load).
Loads only the weights needed per layer: Q, K projections for attention subspace,
gate_proj for FFN-attention alignment.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open
from tqdm.auto import tqdm

plt.style.use('seaborn-v0_8-whitegrid')

# Model dims (Qwen2.5-3B, from 1.py output)
L, d, d_ff, h, GQA = 36, 2048, 11008, 16, 2
d_head = d // h  # 128
K = 20  # top-k subspace dimensions

# Safetensor paths
SNAP = Path("/home/greg/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b")
FILES = [SNAP / "model-00001-of-00002.safetensors", SNAP / "model-00002-of-00002.safetensors"]

# Pre-open both files (memory-mapped, doesn't load tensors)
handles = [safe_open(str(f), framework="pt") for f in FILES]

def get_tensor(name):
    """Get a tensor by name from whichever file has it."""
    for h in handles:
        if name in h.keys():
            return h.get_tensor(name)
    raise KeyError(f"Tensor {name} not found")


def get_attn_subspace_direct(layer_idx, head_idx, k=K):
    """Get top-k right singular vectors of W_Q_h^T @ W_K_h."""
    n_kv_groups = h // GQA
    kv_head_idx = head_idx // n_kv_groups

    W_Q = get_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight").float()
    W_K = get_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.weight").float()

    W_Q_h = W_Q[head_idx * d_head : (head_idx + 1) * d_head, :]
    W_K_h = W_K[kv_head_idx * d_head : (kv_head_idx + 1) * d_head, :]

    kernel = W_Q_h.T @ W_K_h  # (d, d)
    _, S, Vh = torch.linalg.svd(kernel)
    return Vh[:k, :]  # (k, d)


def subspace_similarity(V1, V2):
    """Grassmann similarity: mean squared cosine of principal angles."""
    M = V1 @ V2.T
    svals = torch.linalg.svdvals(M)
    return (svals ** 2).mean().item()


OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ===== Experiment 1: Subspace Overlap Across Layers =====
HEAD = 0
print(f"\n=== Experiment 1: Subspace Overlap (head {HEAD}, top-{K}) ===")

subspaces = []
for l in tqdm(range(L), desc="Extracting subspaces"):
    subspaces.append(get_attn_subspace_direct(l, HEAD, K))

print("Computing pairwise similarity...")
sim_matrix = np.zeros((L, L))
for i in range(L):
    for j in range(i, L):
        s = subspace_similarity(subspaces[i], subspaces[j])
        sim_matrix[i, j] = s
        sim_matrix[j, i] = s

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
fig.colorbar(im, ax=ax, label="Subspace similarity")
ax.set_xlabel("Layer"); ax.set_ylabel("Layer")
ax.set_title(f"Attention subspace overlap (head {HEAD}, top-{K})")
plt.tight_layout(); plt.savefig(OUT / "exp1_subspace_overlap.png", dpi=150); plt.close()

off_diag = sim_matrix[~np.eye(L, dtype=bool)]
adjacent = np.array([sim_matrix[i, i+1] for i in range(L-1)])
distant = np.array([sim_matrix[i, j] for i in range(L) for j in range(L) if abs(i-j) > L//3])
print(f"Off-diagonal: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
print(f"Adjacent: mean={adjacent.mean():.4f}, std={adjacent.std():.4f}")
print(f"Distant (>{L//3} apart): mean={distant.mean():.4f}")
print(f"Gap (adjacent - distant): {adjacent.mean() - distant.mean():+.4f}")

# ===== Experiment 2: Bottleneck Convergence =====
BN = 33
print(f"\n=== Experiment 2: Convergence to layer {BN} ===")

ref = subspaces[BN]
sims_to_ref = np.array([subspace_similarity(ref, subspaces[l]) for l in range(L)])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(L), sims_to_ref, marker=".", markersize=5)
ax.axvline(BN, color="r", linestyle="--", alpha=0.5, label=f"Layer {BN}")
ax.set_xlabel("Layer"); ax.set_ylabel(f"Similarity to layer {BN}")
ax.set_title(f"Bottleneck convergence (head {HEAD}, top-{K})")
ax.legend(); plt.tight_layout()
plt.savefig(OUT / "exp2_bottleneck_convergence.png", dpi=150); plt.close()

before = sims_to_ref[:BN]
after = sims_to_ref[BN+1:]
pre_r = np.corrcoef(range(len(before)), before)[0, 1]
post_r = np.corrcoef(range(len(after)), after)[0, 1]
print(f"Pre-bottleneck trend (Pearson r): {pre_r:+.4f}")
print(f"Post-bottleneck trend: {post_r:+.4f}")
print(f"Mean sim: before={before.mean():.4f}, after={after.mean():.4f}")
print(f"Layer-by-layer to L{BN}:")
for l in range(L):
    print(f"  L{l:2d}: {sims_to_ref[l]:.4f}")

# ===== Experiment 3: FFN-Attention Alignment =====
print(f"\n=== Experiment 3: FFN-Attention Alignment ===")
chance = K / d

alignments = []
for l in tqdm(range(L), desc="FFN-Attn alignment"):
    Vh = subspaces[l]
    P = Vh.T @ Vh  # projection matrix onto attention subspace
    W_gate = get_tensor(f"model.layers.{l}.mlp.gate_proj.weight").float()
    proj = W_gate @ P
    frac = ((proj ** 2).sum() / (W_gate ** 2).sum()).item()
    alignments.append(frac)
alignments = np.array(alignments)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(L), alignments, marker=".", markersize=5)
ax.axhline(chance, color="r", linestyle="--", label=f"Chance ({chance:.4f})")
ax.set_xlabel("Layer"); ax.set_ylabel("Fraction of FFN energy in attention subspace")
ax.set_title(f"FFN-Attention alignment (head {HEAD}, top-{K})")
ax.legend(); plt.tight_layout()
plt.savefig(OUT / "exp3_ffn_attention_alignment.png", dpi=150); plt.close()

above = (alignments > chance).sum()
ratio = alignments.mean() / chance
print(f"Chance: {chance:.4f}")
print(f"Mean alignment: {alignments.mean():.4f} ({ratio:.2f}x chance)")
print(f"Above chance: {above}/{L}")
print(f"Min: {alignments.min():.4f} (L{alignments.argmin()}), Max: {alignments.max():.4f} (L{alignments.argmax()})")
print(f"Layer-by-layer:")
for l in range(L):
    tag = " ***" if alignments[l] > 2 * chance else ""
    print(f"  L{l:2d}: {alignments[l]:.4f}{tag}")

# ===== Multi-Head Robustness =====
print(f"\n=== Multi-Head Robustness (4 heads) ===")
hsample = [0, 5, 10, 15]
avg_sm = np.zeros((L, L))
ph_sims = np.zeros((len(hsample), L))
ph_align = np.zeros((len(hsample), L))

for hi_idx, hi in enumerate(tqdm(hsample, desc="Multi-head")):
    subs = [get_attn_subspace_direct(l, hi, K) for l in range(L)]
    sm = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            s = subspace_similarity(subs[i], subs[j])
            sm[i, j] = s; sm[j, i] = s
    avg_sm += sm
    r = subs[BN]
    ph_sims[hi_idx] = [subspace_similarity(r, subs[l]) for l in range(L)]
    for l in range(L):
        P = subs[l].T @ subs[l]
        Wg = get_tensor(f"model.layers.{l}.mlp.gate_proj.weight").float()
        p = Wg @ P
        ph_align[hi_idx, l] = ((p**2).sum() / (Wg**2).sum()).item()
avg_sm /= len(hsample)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
im = axes[0].imshow(avg_sm, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
fig.colorbar(im, ax=axes[0], label="Similarity")
axes[0].set_title(f"Avg overlap (heads {hsample})"); axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Layer")
ms = ph_sims.mean(0); ss = ph_sims.std(0)
axes[1].plot(ms, marker=".", markersize=3)
axes[1].fill_between(range(L), ms-ss, ms+ss, alpha=0.2)
axes[1].axvline(BN, color="r", linestyle="--", alpha=0.5)
axes[1].set_title(f"Sim to L{BN} (mean +/- std)"); axes[1].set_xlabel("Layer")
ma = ph_align.mean(0); sa = ph_align.std(0)
axes[2].plot(ma, marker=".", markersize=3)
axes[2].fill_between(range(L), ma-sa, ma+sa, alpha=0.2)
axes[2].axhline(chance, color="r", linestyle="--")
axes[2].set_title("FFN-Attn alignment (multi-head)"); axes[2].set_xlabel("Layer")
plt.suptitle(f"Multi-head summary (heads {hsample})", fontsize=14)
plt.tight_layout(); plt.savefig(OUT / "multi_head_summary.png", dpi=150); plt.close()

# Save all data
mn = "Qwen_Qwen2.5-3B"
np.save(OUT / f"{mn}_subspace_sim_matrix_head{HEAD}_k{K}.npy", sim_matrix)
np.save(OUT / f"{mn}_avg_sim_matrix_k{K}.npy", avg_sm)
np.save(OUT / f"{mn}_sims_to_layer{BN}_head{HEAD}_k{K}.npy", sims_to_ref)
np.save(OUT / f"{mn}_ffn_attn_alignment_head{HEAD}_k{K}.npy", alignments)
np.save(OUT / f"{mn}_multi_head_sims_to_ref_k{K}.npy", ph_sims)
np.save(OUT / f"{mn}_multi_head_alignments_k{K}.npy", ph_align)

print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"Exp 1 — Subspace Overlap:")
print(f"  Adjacent similarity: {adjacent.mean():.4f}")
print(f"  Distant similarity:  {distant.mean():.4f}")
print(f"  Gap:                 {adjacent.mean()-distant.mean():+.4f}")
print(f"Exp 2 — Bottleneck Convergence to L{BN}:")
print(f"  Pre-bottleneck trend (r):  {pre_r:+.4f}")
print(f"  Post-bottleneck trend (r): {post_r:+.4f}")
print(f"  Mean sim before: {before.mean():.4f}, after: {after.mean():.4f}")
print(f"Exp 3 — FFN-Attention Alignment:")
print(f"  {ratio:.2f}x chance, {above}/{L} layers above chance")
print(f"  Chance={chance:.4f}, Mean={alignments.mean():.4f}")
print(f"Multi-head avg off-diag: {avg_sm[~np.eye(L,dtype=bool)].mean():.4f}")
print(f"\nAll plots + data saved to output/")
