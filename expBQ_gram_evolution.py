"""
Exp BQ: Gram Matrix Evolution — The Von Neumann Question

Question: What is PRESERVED despite chaotic 77°/layer coordinate rotation?
Answer hypothesis: The pairwise metric structure (Gram matrix).

Method:
  For each layer L (0-35), stack all 7 languages × 200 problems = 1400 vectors.
  Compute G^(L)_ij = cos(h_i, h_j) for all 1400×1400 pairs.
  Measure:
    1. G^(L) vs G^(L+1) Pearson correlation (upper triangle) — metric stability
    2. Frobenius norm of ΔG = G^(L+1) - G^(L) — metric perturbation magnitude
    3. Phase breakdown: which layers MODIFY the metric vs PRESERVE it
    4. Within-language vs cross-language metric evolution
    5. Rank of ΔG — is the metric perturbation low-rank?

If G is nearly constant: layers are approximately conformal (angle-preserving).
If G changes at specific layers: those layers ARE the reasoning.

Uses multilingual_all_layers.npz cache. No model load needed.
"""

import numpy as np
import json
import time
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "output/multilingual_all_layers.npz"
OUT_PATH = "output/expBQ_gram_evolution.json"

LANGS = ["ar", "en", "es", "ja", "ko", "sw", "zh"]
N_LAYERS = 36
N_PROBLEMS = 200

# Phase definitions
PHASES = {
    "early": list(range(0, 9)),       # L0-L8
    "adversarial": list(range(9, 18)), # L9-L17
    "cooperative": list(range(18, 27)),# L18-L26
    "late": list(range(27, 36)),       # L27-L35
}


def load_layer(data, layer):
    """Stack all 7 languages at a given layer → (1400, 2048)."""
    arrays = []
    for lang in LANGS:
        key = f"{lang}_L{layer}"
        arrays.append(data[key])
    return np.vstack(arrays)  # (1400, 2048)


def upper_triangle(G):
    """Extract upper triangle (no diagonal) as flat vector."""
    idx = np.triu_indices(G.shape[0], k=1)
    return G[idx]


def gram_correlation(G1, G2):
    """Pearson correlation between upper triangles of two Gram matrices."""
    u1 = upper_triangle(G1)
    u2 = upper_triangle(G2)
    return float(np.corrcoef(u1, u2)[0, 1])


def gram_frobenius_delta(G1, G2):
    """Frobenius norm of ΔG, normalized by Frobenius norm of G1."""
    dG = G2 - G1
    return float(np.linalg.norm(dG, 'fro') / np.linalg.norm(G1, 'fro'))


def delta_rank(G1, G2, threshold=0.99):
    """Effective rank of ΔG (number of singular values capturing threshold of energy)."""
    dG = G2 - G1
    U, s, Vt = np.linalg.svd(dG, full_matrices=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    rank_99 = int(np.searchsorted(energy, threshold) + 1)
    return rank_99, s[:20].tolist()  # rank + top 20 singular values


def within_cross_language_gram(G):
    """Split Gram matrix into within-language and cross-language blocks.
    Returns mean cosine for within-lang pairs and cross-lang pairs."""
    n = N_PROBLEMS  # 200 per language
    within_vals = []
    cross_vals = []
    for i in range(len(LANGS)):
        for j in range(i, len(LANGS)):
            block = G[i*n:(i+1)*n, j*n:(j+1)*n]
            if i == j:
                # Within-language: upper triangle only (exclude diagonal)
                idx = np.triu_indices(n, k=1)
                within_vals.extend(block[idx].tolist())
            else:
                # Cross-language: all pairs
                cross_vals.extend(block.flatten().tolist())
    return float(np.mean(within_vals)), float(np.mean(cross_vals))


def main():
    t0 = time.time()
    print("Loading multilingual cache...")
    data = np.load(DATA_PATH, allow_pickle=True)
    categories = data["categories"]

    print("Computing Gram matrices for all 36 layers...")
    grams = []
    for L in range(N_LAYERS):
        H = load_layer(data, L)  # (1400, 2048)
        G = cosine_similarity(H)  # (1400, 1400)
        grams.append(G)
        if L % 6 == 0:
            print(f"  L{L}: G computed, shape {G.shape}")

    # === 1. Layer-to-layer metric stability ===
    print("\nComputing layer-to-layer Gram correlations...")
    layer_results = []
    for L in range(N_LAYERS - 1):
        corr = gram_correlation(grams[L], grams[L+1])
        frob_delta = gram_frobenius_delta(grams[L], grams[L+1])
        rank99, top_sv = delta_rank(grams[L], grams[L+1])
        within_mean, cross_mean = within_cross_language_gram(grams[L])

        layer_results.append({
            "layer_pair": f"L{L}→L{L+1}",
            "gram_correlation": round(corr, 6),
            "frobenius_delta_norm": round(frob_delta, 6),
            "delta_rank_99": rank99,
            "top_5_singular_values": [round(s, 4) for s in top_sv[:5]],
            "within_lang_mean_cos": round(within_mean, 4),
            "cross_lang_mean_cos": round(cross_mean, 4),
        })
        if L % 6 == 0:
            print(f"  L{L}→L{L+1}: corr={corr:.6f}, Δ_frob={frob_delta:.6f}, rank99={rank99}")

    # === 2. Long-range stability: G^(0) vs G^(L) ===
    print("\nComputing long-range stability (L0 vs each layer)...")
    long_range = []
    for L in range(1, N_LAYERS):
        corr = gram_correlation(grams[0], grams[L])
        frob_delta = gram_frobenius_delta(grams[0], grams[L])
        long_range.append({
            "layer": L,
            "corr_vs_L0": round(corr, 6),
            "frob_delta_vs_L0": round(frob_delta, 6),
        })
        if L % 6 == 0:
            print(f"  L0→L{L}: corr={corr:.6f}, Δ_frob={frob_delta:.6f}")

    # === 3. Phase-aggregated metrics ===
    print("\nPhase-aggregated metric perturbation...")
    phase_stats = {}
    for phase_name, layers in PHASES.items():
        corrs = []
        frobs = []
        for L in layers:
            if L < N_LAYERS - 1:
                corrs.append(layer_results[L]["gram_correlation"])
                frobs.append(layer_results[L]["frobenius_delta_norm"])
        phase_stats[phase_name] = {
            "mean_gram_correlation": round(float(np.mean(corrs)), 6) if corrs else None,
            "mean_frobenius_delta": round(float(np.mean(frobs)), 6) if frobs else None,
            "min_gram_correlation": round(float(np.min(corrs)), 6) if corrs else None,
            "max_frobenius_delta": round(float(np.max(frobs)), 6) if frobs else None,
        }
        print(f"  {phase_name}: mean_corr={phase_stats[phase_name]['mean_gram_correlation']}, "
              f"mean_frob_Δ={phase_stats[phase_name]['mean_frobenius_delta']}")

    # === 4. Within vs cross language metric per layer ===
    print("\nWithin vs cross language cosine evolution...")
    lang_metric_evolution = []
    for L in range(N_LAYERS):
        within_mean, cross_mean = within_cross_language_gram(grams[L])
        lang_metric_evolution.append({
            "layer": L,
            "within_lang_mean_cos": round(within_mean, 4),
            "cross_lang_mean_cos": round(cross_mean, 4),
            "gap": round(within_mean - cross_mean, 4),
        })
        if L % 6 == 0:
            print(f"  L{L}: within={within_mean:.4f}, cross={cross_mean:.4f}, gap={within_mean-cross_mean:.4f}")

    # === 5. Category-aware: same-category vs different-category metric ===
    print("\nCategory-aware metric analysis at key layers...")
    category_metric = {}
    for L in [0, 8, 12, 17, 18, 26, 32, 35]:
        G = grams[L]
        # For English only (rows 200-399), compute same-cat vs diff-cat cosine
        en_block = G[N_PROBLEMS:2*N_PROBLEMS, N_PROBLEMS:2*N_PROBLEMS]  # EN-EN block
        same_cat = []
        diff_cat = []
        for i in range(N_PROBLEMS):
            for j in range(i+1, N_PROBLEMS):
                if categories[i] == categories[j]:
                    same_cat.append(en_block[i, j])
                else:
                    diff_cat.append(en_block[i, j])
        category_metric[f"L{L}"] = {
            "same_category_mean_cos": round(float(np.mean(same_cat)), 4),
            "diff_category_mean_cos": round(float(np.mean(diff_cat)), 4),
            "category_separation": round(float(np.mean(same_cat)) - float(np.mean(diff_cat)), 4),
        }
        print(f"  L{L}: same_cat={np.mean(same_cat):.4f}, diff_cat={np.mean(diff_cat):.4f}, "
              f"sep={np.mean(same_cat)-np.mean(diff_cat):.4f}")

    # === 6. Gram matrix eigenvalue spectrum at key layers ===
    print("\nGram matrix eigenspectrum at key layers...")
    eigenspectra = {}
    for L in [0, 8, 17, 26, 35]:
        eigenvals = np.linalg.eigvalsh(grams[L])[::-1]  # descending
        total_energy = np.sum(eigenvals**2)
        cumulative = np.cumsum(eigenvals**2) / total_energy
        rank_50 = int(np.searchsorted(cumulative, 0.50) + 1)
        rank_90 = int(np.searchsorted(cumulative, 0.90) + 1)
        rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)
        eigenspectra[f"L{L}"] = {
            "top_5_eigenvalues": [round(float(e), 2) for e in eigenvals[:5]],
            "rank_50": rank_50,
            "rank_90": rank_90,
            "rank_99": rank_99,
        }
        print(f"  L{L}: rank_50={rank_50}, rank_90={rank_90}, rank_99={rank_99}")

    elapsed = time.time() - t0

    # === Compile results ===
    results = {
        "experiment": "BQ",
        "name": "Gram Matrix Evolution — The Von Neumann Question",
        "method": "Pairwise cosine Gram matrix G^(L) for all 1400 vectors at each of 36 layers. "
                  "Pearson correlation between upper triangles of adjacent G^(L), Frobenius norm of ΔG, "
                  "effective rank of ΔG, within/cross language decomposition, category-aware analysis.",
        "data": f"{len(LANGS)} languages × {N_PROBLEMS} problems × {N_LAYERS} layers",
        "layer_to_layer": layer_results,
        "long_range_vs_L0": long_range,
        "phase_aggregated": phase_stats,
        "language_metric_evolution": lang_metric_evolution,
        "category_metric": category_metric,
        "eigenspectra": eigenspectra,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH} in {elapsed:.1f}s")

    # === Summary print ===
    print("\n" + "="*70)
    print("SUMMARY: Gram Matrix Evolution")
    print("="*70)
    corrs = [r["gram_correlation"] for r in layer_results]
    frobs = [r["frobenius_delta_norm"] for r in layer_results]
    print(f"Mean adjacent Gram correlation: {np.mean(corrs):.6f}")
    print(f"Range: {np.min(corrs):.6f} — {np.max(corrs):.6f}")
    print(f"Mean Frobenius delta: {np.mean(frobs):.6f}")
    print(f"\nLowest correlation transitions (most metric change = most reasoning):")
    sorted_idx = np.argsort(corrs)
    for i in sorted_idx[:5]:
        r = layer_results[i]
        print(f"  {r['layer_pair']}: corr={r['gram_correlation']:.6f}, "
              f"Δ_frob={r['frobenius_delta_norm']:.6f}, rank99={r['delta_rank_99']}")
    print(f"\nHighest correlation transitions (most conformal = least reasoning):")
    for i in sorted_idx[-5:]:
        r = layer_results[i]
        print(f"  {r['layer_pair']}: corr={r['gram_correlation']:.6f}, "
              f"Δ_frob={r['frobenius_delta_norm']:.6f}, rank99={r['delta_rank_99']}")

    # Long-range
    print(f"\nLong-range: L0 vs L35 corr = {long_range[-1]['corr_vs_L0']:.6f}")
    print(f"Long-range: L0 vs L17 corr = {long_range[16]['corr_vs_L0']:.6f}")
    print(f"Long-range: L0 vs L26 corr = {long_range[25]['corr_vs_L0']:.6f}")


if __name__ == "__main__":
    main()
