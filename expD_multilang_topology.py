"""Experiment D: 7-Language Attractor Topology
Map the attractor landscape for 7 languages.
How many basins? Do they cluster? Can cross-language steering work?
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# Load multilingual activations at L32
print("Loading multilingual activations...")
data = np.load("output/multilingual_activations.npz")
langs = sorted(data.files)  # ['ar', 'en', 'es', 'ja', 'ko', 'sw', 'zh']
print(f"Languages: {langs}")

# Compute centroids
centroids = {}
for lang in langs:
    centroids[lang] = data[lang].mean(axis=0)  # (2048,)
    print(f"  {lang}: mean norm = {np.linalg.norm(data[lang], axis=1).mean():.1f}")

centroid_matrix = np.array([centroids[l] for l in langs])  # (7, 2048)

# 1. Pairwise cosine distances
print("\n=== PAIRWISE COSINE DISTANCES ===")
# Cosine similarity matrix
norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
normed = centroid_matrix / norms
cos_sim = normed @ normed.T
cos_dist = 1 - cos_sim

print(f"\n{'':>4}", end="")
for l in langs:
    print(f"{l:>8}", end="")
print()
for i, l1 in enumerate(langs):
    print(f"{l1:>4}", end="")
    for j, l2 in enumerate(langs):
        print(f"{cos_dist[i,j]:>8.4f}", end="")
    print()

# 2. Euclidean distances between centroids
print("\n=== EUCLIDEAN DISTANCES ===")
euc_dists = squareform(pdist(centroid_matrix, metric='euclidean'))
print(f"\n{'':>4}", end="")
for l in langs:
    print(f"{l:>8}", end="")
print()
for i, l1 in enumerate(langs):
    print(f"{l1:>4}", end="")
    for j, l2 in enumerate(langs):
        print(f"{euc_dists[i,j]:>8.1f}", end="")
    print()

# 3. PCA on centroids — effective dimensionality
print("\n=== PCA ON CENTROIDS ===")
centroid_centered = centroid_matrix - centroid_matrix.mean(axis=0)
U, S, Vt = np.linalg.svd(centroid_centered, full_matrices=False)
var_explained = S**2 / (S**2).sum()
cumvar = np.cumsum(var_explained)

for i in range(min(7, len(S))):
    print(f"  PC{i}: variance={var_explained[i]:.4f}, cumulative={cumvar[i]:.4f}")

# How many PCs for 95%?
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"\nEffective dimensionality (95% variance): {n_95} PCs")

# 4. Hierarchical clustering
print("\n=== HIERARCHICAL CLUSTERING ===")
condensed = pdist(centroid_matrix, metric='cosine')
Z = linkage(condensed, method='ward')

# Print dendrogram-like structure
print("Linkage (Ward, cosine):")
for i, row in enumerate(Z):
    c1 = langs[int(row[0])] if row[0] < 7 else f"cluster_{int(row[0])-7}"
    c2 = langs[int(row[1])] if row[1] < 7 else f"cluster_{int(row[1])-7}"
    print(f"  Step {i}: merge {c1} + {c2}, distance={row[2]:.4f}, size={int(row[3])}")

# 5. Language group analysis
cjk = ['zh', 'ja', 'ko']
european = ['en', 'es']
other = ['ar', 'sw']

cjk_dists = [cos_dist[langs.index(l1), langs.index(l2)]
             for l1 in cjk for l2 in cjk if l1 < l2]
eur_dists = [cos_dist[langs.index(l1), langs.index(l2)]
             for l1 in european for l2 in european if l1 < l2]
cross_dists = [cos_dist[langs.index(l1), langs.index(l2)]
               for l1 in cjk for l2 in european]

print(f"\nWithin-CJK mean cosine distance: {np.mean(cjk_dists):.4f}")
print(f"Within-European mean cosine distance: {np.mean(eur_dists):.4f}")
print(f"Cross-group mean cosine distance: {np.mean(cross_dists):.4f}")

# 6. PC0 analysis at L32 — is there a single language axis or multiple?
all_data = np.concatenate([data[l] for l in langs], axis=0)  # (1400, 2048)
all_mean = all_data.mean(axis=0)
centered_all = all_data - all_mean
_, S_all, Vt_all = np.linalg.svd(centered_all, full_matrices=False)
pc0_all = Vt_all[0]
pc1_all = Vt_all[1]

print("\n=== LANGUAGE PROJECTIONS ONTO PC0/PC1 ===")
for lang in langs:
    proj0 = ((data[lang] - all_mean) @ pc0_all).mean()
    proj1 = ((data[lang] - all_mean) @ pc1_all).mean()
    print(f"  {lang}: PC0={proj0:>8.2f}, PC1={proj1:>8.2f}")

# 7. Steering tests (need GPU)
print("\n=== STEERING TESTS ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

# Steering pairs: use centroid difference vectors at L32 as steering
# But we need to inject at a layer. Use L28 (strong attractor).
# Compute centroid differences as steering vectors

# First, get L28 data too (we have L32 multilingual, but we inject at L28)
# We'll project the L32 centroid differences onto the L28 space
# Actually, simpler: just use the L32 centroid differences as they are,
# applied at L32 to the residual stream

inject_layer = 28  # Where we inject

# For steering, compute direction from source to target language at L32
steering_pairs = [
    ("zh", "en", "请计算 2 + 3 × 4 的值。\n"),
    ("zh", "sw", "请计算 2 + 3 × 4 的值。\n"),
    ("en", "ja", "Calculate the value of 2 + 3 × 4.\n"),
    ("sw", "zh", "Hesabu thamani ya 2 + 3 × 4.\n"),
    ("zh", "es", "请计算 2 + 3 × 4 的值。\n"),
    ("en", "ko", "Calculate the value of 2 + 3 × 4.\n"),
]

# We need L28 centroids. Compute from our 2-language data.
# Actually, let's use L32 centroids and inject there instead.
inject_layer = 32

def generate_with_steering(prompt, steer_vector, inject_layer, max_new_tokens=64):
    input_ids = tokenizer.encode(prompt)
    injected = [False]

    def hook_fn(module, input, output):
        if not injected[0]:
            hidden = output if not isinstance(output, tuple) else output[0]
            vec = torch.tensor(steer_vector, dtype=hidden.dtype, device=hidden.device)
            hidden[0, -1, :] += vec
            injected[0] = True
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return output

    handle = model.model.layers[inject_layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([input_ids], device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    finally:
        handle.remove()

results = {
    "cosine_distance_matrix": {langs[i]: {langs[j]: float(cos_dist[i,j])
                                           for j in range(7)} for i in range(7)},
    "euclidean_distance_matrix": {langs[i]: {langs[j]: float(euc_dists[i,j])
                                              for j in range(7)} for i in range(7)},
    "pca_variance_explained": [float(v) for v in var_explained],
    "effective_dimensionality_95pct": int(n_95),
    "pc_projections": {},
    "clustering": [],
    "steering_results": [],
}

for lang in langs:
    proj0 = float(((data[lang] - all_mean) @ pc0_all).mean())
    proj1 = float(((data[lang] - all_mean) @ pc1_all).mean())
    results["pc_projections"][lang] = {"PC0": proj0, "PC1": proj1}

for src, tgt, prompt in steering_pairs:
    steer = centroids[tgt] - centroids[src]
    steer_norm = np.linalg.norm(steer)

    # Generate with steering
    gen = generate_with_steering(prompt, steer, inject_layer)

    # Detect language of output
    n_zh = sum(1 for c in gen if '\u4e00' <= c <= '\u9fff')
    n_ja = sum(1 for c in gen if '\u3040' <= c <= '\u30ff' or '\u31f0' <= c <= '\u31ff')
    n_ko = sum(1 for c in gen if '\uac00' <= c <= '\ud7af')
    n_ar = sum(1 for c in gen if '\u0600' <= c <= '\u06ff')
    n_latin = sum(1 for c in gen if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))

    # Simple language detection
    char_counts = {"zh": n_zh, "ja": n_ja, "ko": n_ko, "ar": n_ar, "latin": n_latin}
    detected = max(char_counts, key=char_counts.get)
    if detected == "latin":
        # Could be en, es, sw — check for specific markers
        if any(w in gen.lower() for w in ["thamani", "hesabu", "jibu"]):
            detected = "sw"
        elif any(w in gen.lower() for w in ["valor", "calcular", "resultado"]):
            detected = "es"
        else:
            detected = "en"

    print(f"\n  {src}→{tgt} (steer norm={steer_norm:.1f}):")
    print(f"    Detected: {detected}")
    print(f"    Output: {gen[:120]}...")

    results["steering_results"].append({
        "source": src,
        "target": tgt,
        "prompt": prompt,
        "steer_norm": float(steer_norm),
        "detected_language": detected,
        "target_hit": detected == tgt,
        "generation": gen,
    })

# Summary
print("\n\n=== SUMMARY ===")
hits = sum(1 for r in results["steering_results"] if r["target_hit"])
total = len(results["steering_results"])
print(f"Steering accuracy: {hits}/{total}")
print(f"Effective dimensionality: {n_95} PCs for 95% variance")
print(f"CJK within-group distance: {np.mean(cjk_dists):.4f}")
print(f"European within-group distance: {np.mean(eur_dists):.4f}")
print(f"Cross-group distance: {np.mean(cross_dists):.4f}")

# Save
with open("output/expD_multilang_topology.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expD_multilang_topology.json")
