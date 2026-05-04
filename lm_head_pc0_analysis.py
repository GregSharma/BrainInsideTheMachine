"""LM Head PC0 Analysis — THE CAPSTONE EXPERIMENT.

Question: Is the LM head (lm_head.weight, 151936 × 2048) disproportionately
aligned with PC0? If yes, this explains why swapping PC0 on the final residual
stream controls language output despite the other 65% also carrying language.

Loads ONLY lm_head.weight and final norm from safetensors directly (no full model load).
"""

import numpy as np
import torch
import json
from pathlib import Path
from sklearn.decomposition import PCA
from safetensors import safe_open
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import glob as globmod

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")

print("=" * 70)
print("LM HEAD PC0 ANALYSIS")
print("=" * 70)

# --- Step 1: Compute PC0 at L35 from cached activations ---
print("\n[1] Computing PC0 at L35 from cached activations...")
data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")
zh_L35 = data["zh_L35"]  # (200, 2048)
en_L35 = data["en_L35"]  # (200, 2048)

combined = np.vstack([zh_L35, en_L35])  # (400, 2048)
pca = PCA(n_components=10)
pca.fit(combined)
pc0 = pca.components_[0]  # (2048,)
pc0_var = pca.explained_variance_ratio_[0]

zh_proj = zh_L35 @ pc0
en_proj = en_L35 @ pc0
if zh_proj.mean() > en_proj.mean():
    pc0 = -pc0
    zh_proj = -zh_proj
    en_proj = -en_proj

print(f"  PC0 variance explained: {pc0_var:.3f} ({pc0_var*100:.1f}%)")
print(f"  zh proj mean: {zh_proj.mean():.3f}, en proj mean: {en_proj.mean():.3f}")
print(f"  Gap: {en_proj.mean() - zh_proj.mean():.3f}")

# Cross-layer cosines
pc0_vectors = {}
for layer in [8, 20, 26, 28, 32, 34, 35]:
    zh_l = data[f"zh_L{layer}"]
    en_l = data[f"en_L{layer}"]
    comb = np.vstack([zh_l, en_l])
    p = PCA(n_components=1)
    p.fit(comb)
    v = p.components_[0]
    if (zh_l @ v).mean() > (en_l @ v).mean():
        v = -v
    pc0_vectors[layer] = v

print(f"  PC0 cross-layer cosines with L35:")
for layer in [8, 20, 26, 28, 32, 34]:
    cos = np.dot(pc0_vectors[layer], pc0_vectors[35])
    print(f"    L{layer} vs L35: {cos:.4f}")

del data  # free memory

# --- Step 2: Load ONLY lm_head.weight from safetensors ---
print("\n[2] Loading lm_head.weight from safetensors (NOT full model)...")

# Find the cached model files
from huggingface_hub import snapshot_download
cache_dir = snapshot_download(MODEL_NAME, allow_patterns=["*.safetensors", "model.safetensors.index.json"])

# Find which shard has lm_head.weight
index_path = Path(cache_dir) / "model.safetensors.index.json"
if index_path.exists():
    with open(index_path) as f:
        index = json.load(f)
    # Qwen2.5-3B ties lm_head to embed_tokens — no separate lm_head.weight
    lm_head_key = "lm_head.weight" if "lm_head.weight" in index["weight_map"] else "model.embed_tokens.weight"
    lm_head_file = index["weight_map"][lm_head_key]
    norm_file = index["weight_map"].get("model.norm.weight", lm_head_file)
    print(f"  {lm_head_key} in: {lm_head_file}")
    print(f"  model.norm.weight in: {norm_file}")
    print(f"  NOTE: Tied embeddings — lm_head IS embed_tokens" if lm_head_key != "lm_head.weight" else "")
else:
    safetensor_files = list(Path(cache_dir).glob("*.safetensors"))
    lm_head_file = safetensor_files[0].name
    norm_file = lm_head_file
    lm_head_key = "model.embed_tokens.weight"

# Load via torch framework (handles bfloat16), then convert to numpy float32
with safe_open(Path(cache_dir) / lm_head_file, framework="pt") as f:
    lm_head_weight = f.get_tensor(lm_head_key).float().numpy()  # (151936, 2048)
print(f"  {lm_head_key} shape: {lm_head_weight.shape}, dtype: {lm_head_weight.dtype}")

with safe_open(Path(cache_dir) / norm_file, framework="pt") as f:
    norm_weight = f.get_tensor("model.norm.weight").float().numpy()  # (2048,)
print(f"  model.norm.weight shape: {norm_weight.shape}")

# Load tokenizer
print("  Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Step 3: Compute dot product of each vocab token with PC0 ---
print("\n[3] Computing vocab-PC0 alignment...")

pc0_scores = lm_head_weight @ pc0.astype(np.float32)  # (151936,)

print(f"  Score range: [{pc0_scores.min():.3f}, {pc0_scores.max():.3f}]")
print(f"  Score mean: {pc0_scores.mean():.3f}, std: {pc0_scores.std():.3f}")

# Also with RMSNorm scaling
pc0_normed = pc0 * norm_weight.astype(np.float64)
pc0_normed = (pc0_normed / np.linalg.norm(pc0_normed)).astype(np.float32)
pc0_scores_normed = lm_head_weight @ pc0_normed
print(f"  After RMSNorm scaling:")
print(f"    Score range: [{pc0_scores_normed.min():.3f}, {pc0_scores_normed.max():.3f}]")
print(f"    Score mean: {pc0_scores_normed.mean():.3f}, std: {pc0_scores_normed.std():.3f}")

# --- Step 4: Sort and display top/bottom tokens ---
print("\n[4] Top 50 tokens (MOST ENGLISH-aligned, highest PC0 score):")
sorted_idx = np.argsort(pc0_scores)
top_50 = sorted_idx[-50:][::-1]
for rank, idx in enumerate(top_50):
    token_str = tokenizer.decode([idx])
    token_bytes = tokenizer.convert_ids_to_tokens([idx])[0]
    print(f"  {rank+1:3d}. id={idx:6d} score={pc0_scores[idx]:+.4f}  '{token_str}'  ({token_bytes})")

print(f"\n[5] Bottom 50 tokens (MOST CHINESE-aligned, lowest PC0 score):")
bottom_50 = sorted_idx[:50]
for rank, idx in enumerate(bottom_50):
    token_str = tokenizer.decode([idx])
    token_bytes = tokenizer.convert_ids_to_tokens([idx])[0]
    print(f"  {rank+1:3d}. id={idx:6d} score={pc0_scores[idx]:+.4f}  '{token_str}'  ({token_bytes})")

# --- Step 5: Variance analysis ---
print("\n[6] Variance analysis: how much of lm_head is along PC0?")

total_var = np.sum(lm_head_weight.astype(np.float64) ** 2)
pc0_var_lm = np.sum(pc0_scores.astype(np.float64) ** 2)

print(f"  Total ||W||²_F: {total_var:.1f}")
print(f"  PC0 component: {pc0_var_lm:.1f}")
print(f"  Fraction along PC0: {pc0_var_lm / total_var:.6f} ({pc0_var_lm / total_var * 100:.4f}%)")

# Compare with random directions
rng = np.random.RandomState(42)
rand_fracs = []
for _ in range(100):
    rand_dir = rng.randn(2048).astype(np.float32)
    rand_dir /= np.linalg.norm(rand_dir)
    rand_scores = lm_head_weight @ rand_dir
    rand_var = np.sum(rand_scores.astype(np.float64) ** 2)
    rand_fracs.append(rand_var / total_var)

print(f"  Random direction fraction: {np.mean(rand_fracs):.6f} ± {np.std(rand_fracs):.6f}")
print(f"  PC0 / random ratio: {(pc0_var_lm / total_var) / np.mean(rand_fracs):.2f}x")

# --- Step 6: PCA of lm_head itself ---
print("\n[7] PCA of lm_head.weight — is PC0 aligned with its top directions?")
# Compute covariance efficiently: (2048, 2048)
lm_mean = lm_head_weight.mean(axis=0, keepdims=True)
lm_centered = lm_head_weight - lm_mean
# Use float64 for eigendecomposition
cov = (lm_centered.astype(np.float64).T @ lm_centered.astype(np.float64)) / lm_head_weight.shape[0]
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx_sorted = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx_sorted]
eigenvectors = eigenvectors[:, idx_sorted]

total_eig = eigenvalues.sum()
for i in range(10):
    cos_with_pc0 = abs(np.dot(eigenvectors[:, i], pc0))
    print(f"  LM_PC{i}: var_ratio={eigenvalues[i]/total_eig:.4f}, |cos(residual_PC0)|={cos_with_pc0:.4f}")

# --- Step 7: Chinese vs English token distribution ---
print("\n[8] Chinese vs English token distribution along PC0:")

chinese_ids = []
english_ids = []
other_ids = []

for token_id in range(lm_head_weight.shape[0]):
    token_str = tokenizer.decode([token_id])
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in token_str)
    has_ascii_alpha = any('a' <= c.lower() <= 'z' for c in token_str)
    if has_cjk:
        chinese_ids.append(token_id)
    elif has_ascii_alpha:
        english_ids.append(token_id)
    else:
        other_ids.append(token_id)

print(f"  Chinese tokens: {len(chinese_ids)}")
print(f"  English tokens: {len(english_ids)}")
print(f"  Other tokens: {len(other_ids)}")

zh_scores = pc0_scores[chinese_ids]
en_scores = pc0_scores[english_ids]
other_scores = pc0_scores[other_ids]

print(f"  Chinese PC0 score: mean={zh_scores.mean():.4f}, std={zh_scores.std():.4f}")
print(f"  English PC0 score: mean={en_scores.mean():.4f}, std={en_scores.std():.4f}")
print(f"  Other PC0 score:   mean={other_scores.mean():.4f}, std={other_scores.std():.4f}")
print(f"  Gap (en - zh): {en_scores.mean() - zh_scores.mean():.4f}")

pooled_std = np.sqrt((zh_scores.std()**2 + en_scores.std()**2) / 2)
cohens_d = (en_scores.mean() - zh_scores.mean()) / pooled_std
print(f"  Cohen's d: {cohens_d:.3f}")

en_median = np.median(en_scores)
zh_above_en_median = (zh_scores > en_median).mean()
print(f"  Fraction of zh tokens above en median: {zh_above_en_median:.3f}")

# --- Step 8: Per-PC analysis ---
print("\n[9] Per-PC breakdown: how much does lm_head read each PC?")
pcs = pca.components_[:10].astype(np.float32)
for i in range(10):
    pc_scores_i = lm_head_weight @ pcs[i]
    pc_var_i = float(np.sum(pc_scores_i.astype(np.float64) ** 2) / total_var)
    zh_s = pc_scores_i[chinese_ids]
    en_s = pc_scores_i[english_ids]
    gap = float(en_s.mean() - zh_s.mean())
    print(f"  PC{i}: var_frac={pc_var_i:.4f}, zh/en_gap={gap:+.4f}, "
          f"zh_mean={zh_s.mean():+.4f}, en_mean={en_s.mean():+.4f}")

# --- Step 9: The killer question — how many PCs needed to explain zh/en gap? ---
print("\n[10] Cumulative zh/en separation by number of PCs projected through lm_head:")
pcs_all = pca.components_[:10].astype(np.float32)
for n_pcs in [1, 2, 3, 5, 10]:
    # Project lm_head through top n PCs of residual stream
    proj = lm_head_weight @ pcs_all[:n_pcs].T  # (151936, n_pcs)
    # Sum absolute zh/en gaps across PCs
    zh_proj_scores = proj[chinese_ids]
    en_proj_scores = proj[english_ids]
    # Euclidean distance between centroids in n-PC space
    zh_centroid = zh_proj_scores.mean(axis=0)
    en_centroid = en_proj_scores.mean(axis=0)
    centroid_dist = np.linalg.norm(en_centroid - zh_centroid)
    print(f"  Top {n_pcs:2d} PCs: centroid distance = {centroid_dist:.4f}")

# --- Save results ---
results = {
    "pc0_variance_ratio_L35": float(pc0_var),
    "pc0_zh_proj_mean": float(zh_proj.mean()),
    "pc0_en_proj_mean": float(en_proj.mean()),
    "lm_head_pc0_variance_fraction": float(pc0_var_lm / total_var),
    "random_direction_variance_fraction": float(np.mean(rand_fracs)),
    "pc0_over_random_ratio": float((pc0_var_lm / total_var) / np.mean(rand_fracs)),
    "chinese_token_count": len(chinese_ids),
    "english_token_count": len(english_ids),
    "chinese_pc0_mean": float(zh_scores.mean()),
    "english_pc0_mean": float(en_scores.mean()),
    "cohens_d": float(cohens_d),
    "zh_above_en_median_frac": float(zh_above_en_median),
    "top_50_english": [
        {"id": int(idx), "score": float(pc0_scores[idx]),
         "token": tokenizer.decode([idx])}
        for idx in top_50
    ],
    "bottom_50_chinese": [
        {"id": int(idx), "score": float(pc0_scores[idx]),
         "token": tokenizer.decode([idx])}
        for idx in bottom_50
    ],
    "lm_pca_pc0_alignment": [
        {"pc": i, "var_ratio": float(eigenvalues[i]/total_eig),
         "cos_with_residual_pc0": float(abs(np.dot(eigenvectors[:, i], pc0)))}
        for i in range(10)
    ],
    "per_pc_lm_head_read": [
        {"pc": i,
         "var_frac": float(np.sum((lm_head_weight @ pcs[i]).astype(np.float64)**2) / total_var),
         "zh_en_gap": float((lm_head_weight @ pcs[i])[english_ids].mean() -
                           (lm_head_weight @ pcs[i])[chinese_ids].mean())}
        for i in range(10)
    ]
}

with open(OUTPUT_DIR / "lm_head_pc0_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR / 'lm_head_pc0_analysis.json'}")
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
