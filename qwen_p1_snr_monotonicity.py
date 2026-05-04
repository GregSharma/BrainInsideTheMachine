"""
Qwen Prediction P1: Cross-lingual NN accuracy increases monotonically with depth.

Uses existing cached activations (zh, en at layers 8, 16, 24, 32, 34).
200 matched math problems. Qwen2.5-3B.

This is the FIRST Qwen validation of Theorem 2 (SNR monotonicity).
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes

# Load cached activations
data = np.load("output/viz_activations.npz", allow_pickle=True)
categories = data["categories"]

LAYERS = [8, 16, 24, 32, 34]
N = 200  # matched problems

print("="*70)
print("QWEN PREDICTION P1: SNR MONOTONICITY TEST")
print("="*70)
print(f"Model: Qwen2.5-3B, {N} matched zh-en math problems")
print(f"Layers: {LAYERS}")
print()

# ============================================================
# TEST 1: Cross-lingual NN accuracy per layer
# ============================================================
print("TEST 1: Cross-lingual NN accuracy (zh→en)")
print("  For each zh problem, is the nearest en neighbor the SAME problem?")
print()

for L in LAYERS:
    zh = data[f"zh_L{L}"]  # (200, 2048)
    en = data[f"en_L{L}"]  # (200, 2048)

    # Raw NN: nearest en neighbor for each zh sample
    dists = cdist(zh, en, metric="cosine")
    nn_raw = np.mean(np.argmin(dists, axis=1) == np.arange(N))

    # Procrustes-aligned NN
    R, _ = orthogonal_procrustes(zh, en)
    zh_aligned = zh @ R
    dists_proc = cdist(zh_aligned, en, metric="cosine")
    nn_proc = np.mean(np.argmin(dists_proc, axis=1) == np.arange(N))

    print(f"  Layer {L:2d}: Raw NN = {nn_raw:.3f}  Procrustes NN = {nn_proc:.3f}")

# ============================================================
# TEST 2: SNR (D_same / D_diff) per layer
# ============================================================
print()
print("TEST 2: SNR = D_same / D_diff")
print("  D_same = mean ||zh(x) - en(x)||² (same problem)")
print("  D_diff = mean ||zh(x) - zh(x')||² (different problems)")
print()

np.random.seed(42)
for L in LAYERS:
    zh = data[f"zh_L{L}"]
    en = data[f"en_L{L}"]

    # D_same: same problem, different language
    d_same = np.mean(np.sum((zh - en)**2, axis=1))

    # D_diff: different problems, same language (Chinese)
    n_pairs = 2000
    idx1 = np.random.choice(N, n_pairs)
    idx2 = np.random.choice(N, n_pairs)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    d_diff = np.mean(np.sum((zh[idx1] - zh[idx2])**2, axis=1))

    snr = d_same / d_diff
    print(f"  Layer {L:2d}: D_same={d_same:.1f}  D_diff={d_diff:.1f}  SNR={snr:.4f}")

# ============================================================
# TEST 3: Per-layer deltas (h_{k+1} - h_k) and cross-lingual analysis
# ============================================================
print()
print("TEST 3: Delta analysis (h_{k+1} - h_k)")
print("  Delta norms and cross-lingual disagreement between consecutive layers")
print()

for i in range(len(LAYERS) - 1):
    L1, L2 = LAYERS[i], LAYERS[i+1]
    zh1 = data[f"zh_L{L1}"]
    zh2 = data[f"zh_L{L2}"]
    en1 = data[f"en_L{L1}"]
    en2 = data[f"en_L{L2}"]

    # Deltas
    delta_zh = zh2 - zh1
    delta_en = en2 - en1

    # Mean delta norm
    norm_zh = np.linalg.norm(delta_zh, axis=1).mean()
    norm_en = np.linalg.norm(delta_en, axis=1).mean()

    # Cross-lingual delta disagreement
    delta_diff = delta_zh - delta_en
    norm_diff = np.linalg.norm(delta_diff, axis=1).mean()

    # Ratio: disagreement / mean contribution
    ratio = norm_diff / ((norm_zh + norm_en) / 2)

    print(f"  L{L1}→L{L2}: ||Δ_zh||={norm_zh:.1f}  ||Δ_en||={norm_en:.1f}  ||Δ_zh-Δ_en||={norm_diff:.1f}  ratio={ratio:.3f}")

# ============================================================
# TEST 4: Procrustes R² per layer
# ============================================================
print()
print("TEST 4: Procrustes R² per layer (zh→en alignment quality)")
print()

for L in LAYERS:
    zh = data[f"zh_L{L}"]
    en = data[f"en_L{L}"]

    # Center
    zh_c = zh - zh.mean(axis=0)
    en_c = en - en.mean(axis=0)

    # Procrustes
    R, _ = orthogonal_procrustes(zh_c, en_c)
    zh_rot = zh_c @ R

    # R²
    ss_res = np.sum((zh_rot - en_c)**2)
    ss_tot = np.sum(en_c**2)
    r2 = 1 - ss_res / ss_tot

    # Affine (per-dimension scaling on top of Procrustes)
    # s_j = Σ(en_j * zh_rot_j) / Σ(zh_rot_j²)
    scales = np.sum(en_c * zh_rot, axis=0) / (np.sum(zh_rot**2, axis=0) + 1e-12)
    zh_affine = zh_rot * scales
    ss_res_aff = np.sum((zh_affine - en_c)**2)
    r2_aff = 1 - ss_res_aff / ss_tot

    print(f"  Layer {L:2d}: Procrustes R² = {r2:.4f}  Affine R² = {r2_aff:.4f}  Delta = {r2_aff - r2:+.4f}")

# ============================================================
# TEST 5: Gram matrix of deltas (orthogonality check)
# ============================================================
print()
print("TEST 5: Orthogonality of layer contributions (zh)")
print("  Cosine similarity between delta vectors at consecutive layer pairs")
print()

# We only have 4 deltas (between 5 layers)
deltas_zh = []
for i in range(len(LAYERS) - 1):
    L1, L2 = LAYERS[i], LAYERS[i+1]
    delta = data[f"zh_L{L2}"] - data[f"zh_L{L1}"]
    deltas_zh.append(delta)

n_deltas = len(deltas_zh)
G = np.zeros((n_deltas, n_deltas))
labels = [f"L{LAYERS[i]}→{LAYERS[i+1]}" for i in range(n_deltas)]

for j in range(n_deltas):
    for k in range(n_deltas):
        fj = deltas_zh[j].flatten()
        fk = deltas_zh[k].flatten()
        cos = np.dot(fj, fk) / (np.linalg.norm(fj) * np.linalg.norm(fk) + 1e-12)
        G[j, k] = cos

print("  Gram matrix:")
print("             ", "  ".join(f"{l:>10s}" for l in labels))
for j in range(n_deltas):
    row = "  ".join(f"{G[j,k]:10.4f}" for k in range(n_deltas))
    print(f"  {labels[j]:>10s}  {row}")

off_diag = [abs(G[j,k]) for j in range(n_deltas) for k in range(j+1, n_deltas)]
print(f"\n  Off-diagonal |cos|: mean={np.mean(off_diag):.4f}, max={np.max(off_diag):.4f}")

print()
print("="*70)
print("SUMMARY")
print("="*70)
