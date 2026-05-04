"""
Exp BQ2-7B: Kernel Dynamics on Qwen2.5-7B — Lyapunov Exponents and ΔG Spectral Analysis

Cross-model replication of BQ2 (originally on Qwen2.5-3B, 36 layers).
Qwen2.5-7B: 28 layers, d=3584. Same architecture family.

Phases normalized by depth fraction:
  3B: early(0-8), adversarial(9-17), cooperative(18-26), late(27-35) — 9 layers each
  7B: early(0-6), adversarial(7-13), cooperative(14-20), late(21-27) — 7 layers each

Key question: does the 4-phase Lyapunov funnel replicate when normalized by depth?
"""

import numpy as np
import json
import time
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "output/multilingual_all_layers_qwen2_5_7b.npz"
OUT_PATH = "output/expBQ2_7b_kernel_dynamics.json"

LANGS = ["ar", "en", "es", "ja", "ko", "sw", "zh"]
N_LAYERS = 28
N_PROBLEMS = 200
N_TOTAL = len(LANGS) * N_PROBLEMS  # 1400

# Phases: 4 equal bands of 7 layers, normalized by depth fraction
PHASES = {
    "early": list(range(0, 7)),       # 0-6   (3B: 0-8)
    "adversarial": list(range(7, 14)),  # 7-13  (3B: 9-17)
    "cooperative": list(range(14, 21)), # 14-20 (3B: 18-26)
    "late": list(range(21, 28)),        # 21-27 (3B: 27-35)
}

# 3B reference values for comparison
REF_3B = {
    "phase_positive_modes": {"early": 16, "adversarial": 1, "cooperative": 3, "late": 15},
    "rank_50_all_layers": 1,
    "rank_90_peak": 9,
    "gram_correlation": 0.974,
    "mode_0_hourglass": "1044→920(L9)→1182(L26)→885(L35)",
}


def load_layer(data, layer):
    """Stack all 7 languages at a given layer → (1400, 3584)."""
    arrays = []
    for lang in LANGS:
        arrays.append(data[f"{lang}_L{layer}"])
    return np.vstack(arrays)


def make_label_vectors(categories):
    """Build label indicator vectors for projecting eigenvectors."""
    n = N_PROBLEMS
    lang_ind = np.zeros((N_TOTAL, len(LANGS)))
    for i, lang in enumerate(LANGS):
        lang_ind[i*n:(i+1)*n, i] = 1.0

    n_cats = int(categories.max()) + 1
    cat_ind = np.zeros((N_TOTAL, n_cats))
    for i in range(len(LANGS)):
        for j in range(n):
            cat_ind[i*n + j, categories[j]] = 1.0

    return lang_ind, cat_ind


def eigvec_alignment(v, indicators):
    """How much of eigenvector v is explained by a set of indicator vectors."""
    Q, _ = np.linalg.qr(indicators)
    proj = Q @ (Q.T @ v)
    ss_proj = np.sum(proj**2)
    ss_total = np.sum(v**2)
    return float(ss_proj / ss_total) if ss_total > 0 else 0.0


def main():
    t0 = time.time()
    print("=" * 70)
    print("BQ2-7B: Kernel Dynamics on Qwen2.5-7B (28 layers, d=3584)")
    print("=" * 70)
    print(f"\nLoading 7B multilingual cache from {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)
    categories = data["categories"]
    lang_ind, cat_ind = make_label_vectors(categories)

    # =========================================================
    # 1. Compute G at every layer + full eigendecomposition
    # =========================================================
    print(f"\nComputing Gram matrices and eigendecompositions for all {N_LAYERS} layers...")
    grams = []
    eigenvalues_all = []
    eigenvectors_top = []

    TOP_K = 20

    for L in range(N_LAYERS):
        H = load_layer(data, L)
        G = cosine_similarity(H)
        grams.append(G)

        eigenvals, eigvecs = np.linalg.eigh(G)
        eigenvals = eigenvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        eigenvalues_all.append(eigenvals)
        eigenvectors_top.append(eigvecs[:, :TOP_K])

        if L % 4 == 0:
            print(f"  L{L}: top eigenvalues = {eigenvals[:5].round(1)}, "
                  f"sum = {eigenvals.sum():.1f}")

    eigenvalues_all = np.array(eigenvalues_all)  # (28, 1400)

    # =========================================================
    # 2. Eigenvalue trajectories through depth
    # =========================================================
    print("\nTracking eigenvalue trajectories...")
    eigenvalue_trajectories = {}
    for k in range(TOP_K):
        eigenvalue_trajectories[f"mode_{k}"] = {
            "values": [float(eigenvalues_all[L, k]) for L in range(N_LAYERS)],
        }

    total_energy = np.array([eigenvalues_all[L].sum() for L in range(N_LAYERS)])
    effective_ranks = {}
    for threshold in [0.50, 0.90, 0.95, 0.99]:
        ranks = []
        for L in range(N_LAYERS):
            cumsum = np.cumsum(eigenvalues_all[L]) / eigenvalues_all[L].sum()
            rank = int(np.searchsorted(cumsum, threshold) + 1)
            ranks.append(rank)
        effective_ranks[f"rank_{int(threshold*100)}"] = ranks
    print(f"  Effective rank_50 range: {min(effective_ranks['rank_50'])} - {max(effective_ranks['rank_50'])}")
    print(f"  Effective rank_90 range: {min(effective_ranks['rank_90'])} - {max(effective_ranks['rank_90'])}")
    print(f"  Effective rank_99 range: {min(effective_ranks['rank_99'])} - {max(effective_ranks['rank_99'])}")

    # =========================================================
    # 3. Lyapunov exponents of eigenvalue modes
    # =========================================================
    print("\nComputing Lyapunov exponents...")
    lyapunov = {}
    for k in range(TOP_K):
        log_ratios = []
        for L in range(N_LAYERS - 1):
            v_curr = abs(eigenvalues_all[L, k])
            v_next = abs(eigenvalues_all[L + 1, k])
            if v_curr > 1e-6:
                log_ratios.append(np.log(v_next / v_curr))
            else:
                log_ratios.append(0.0)
        lyap_exp = float(np.mean(log_ratios))
        lyapunov[f"mode_{k}"] = {
            "exponent": round(lyap_exp, 6),
            "per_layer": [round(r, 6) for r in log_ratios],
        }
    print(f"  Mode 0 Lyapunov exponent: {lyapunov['mode_0']['exponent']:.6f}")
    print(f"  Mode 1 Lyapunov exponent: {lyapunov['mode_1']['exponent']:.6f}")
    print(f"  Mode 5 Lyapunov exponent: {lyapunov['mode_5']['exponent']:.6f}")
    print(f"  Mode 10 Lyapunov exponent: {lyapunov['mode_10']['exponent']:.6f}")

    # =========================================================
    # 4. ΔG spectral analysis
    # =========================================================
    print("\nComputing ΔG spectral analysis at each layer transition...")
    delta_g_analysis = []
    for L in range(N_LAYERS - 1):
        dG = grams[L+1] - grams[L]

        dG_vals, dG_vecs = np.linalg.eigh(dG)
        dG_vals = dG_vals[::-1]
        dG_vecs = dG_vecs[:, ::-1]

        top_lang_alignment = []
        top_cat_alignment = []
        for k in range(min(5, len(dG_vals))):
            v = dG_vecs[:, k]
            la = eigvec_alignment(v, lang_ind)
            ca = eigvec_alignment(v, cat_ind)
            top_lang_alignment.append(round(la, 4))
            top_cat_alignment.append(round(ca, 4))

        abs_vals = np.abs(dG_vals)
        sorted_abs = np.sort(abs_vals)[::-1]
        cumulative = np.cumsum(sorted_abs**2) / np.sum(sorted_abs**2)
        delta_rank_90 = int(np.searchsorted(cumulative, 0.90) + 1)
        delta_rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)

        frob = float(np.linalg.norm(dG, 'fro'))
        spectral_gap = float(abs(dG_vals[0]) / abs(dG_vals[1])) if abs(dG_vals[1]) > 1e-10 else float('inf')

        delta_g_analysis.append({
            "layer_pair": f"L{L}→L{L+1}",
            "top_5_eigenvalues": [round(float(v), 4) for v in dG_vals[:5]],
            "bottom_3_eigenvalues": [round(float(v), 4) for v in dG_vals[-3:]],
            "frobenius_norm": round(frob, 4),
            "delta_rank_90": delta_rank_90,
            "delta_rank_99": delta_rank_99,
            "spectral_gap": round(spectral_gap, 4),
            "top5_language_alignment": top_lang_alignment,
            "top5_category_alignment": top_cat_alignment,
        })

        if L % 4 == 0:
            print(f"  L{L}→L{L+1}: top_eigval={dG_vals[0]:.4f}, "
                  f"rank_90={delta_rank_90}, rank_99={delta_rank_99}, "
                  f"lang_align={top_lang_alignment[0]:.3f}, cat_align={top_cat_alignment[0]:.3f}")

    # =========================================================
    # 5. Eigenvector alignment at G — what structure does G encode?
    # =========================================================
    print("\nTop eigenvector alignment with language/category at each layer...")
    g_eigvec_alignment = []
    for L in range(N_LAYERS):
        top_vecs = eigenvectors_top[L]
        lang_aligns = []
        cat_aligns = []
        for k in range(min(10, TOP_K)):
            v = top_vecs[:, k]
            lang_aligns.append(round(eigvec_alignment(v, lang_ind), 4))
            cat_aligns.append(round(eigvec_alignment(v, cat_ind), 4))
        g_eigvec_alignment.append({
            "layer": L,
            "top10_language_alignment": lang_aligns,
            "top10_category_alignment": cat_aligns,
        })
        if L % 4 == 0:
            print(f"  L{L}: mode0_lang={lang_aligns[0]:.3f} cat={cat_aligns[0]:.3f}, "
                  f"mode1_lang={lang_aligns[1]:.3f} cat={cat_aligns[1]:.3f}")

    # =========================================================
    # 6. Phase-resolved Lyapunov spectrum
    # =========================================================
    print("\nPhase-resolved Lyapunov spectrum...")
    phase_lyapunov = {}
    for phase_name, layers in PHASES.items():
        phase_exps = []
        for k in range(TOP_K):
            per_layer = lyapunov[f"mode_{k}"]["per_layer"]
            phase_vals = [per_layer[L] for L in layers if L < N_LAYERS - 1]
            phase_exps.append(round(float(np.mean(phase_vals)), 6) if phase_vals else 0.0)
        phase_lyapunov[phase_name] = phase_exps
        n_positive = sum(1 for e in phase_exps if e > 0)
        n_negative = sum(1 for e in phase_exps if e < 0)
        print(f"  {phase_name}: {n_positive} positive, {n_negative} negative modes (of {TOP_K})")

    # =========================================================
    # 7. Gram correlation (consecutive layer pairs)
    # =========================================================
    print("\nGram correlation between consecutive layers...")
    gram_correlations = []
    for L in range(N_LAYERS - 1):
        G_curr = grams[L].flatten()
        G_next = grams[L+1].flatten()
        corr = float(np.corrcoef(G_curr, G_next)[0, 1])
        gram_correlations.append(round(corr, 6))
    mean_corr = float(np.mean(gram_correlations))
    print(f"  Mean Gram correlation: {mean_corr:.4f} (3B ref: {REF_3B['gram_correlation']})")
    print(f"  Min: {min(gram_correlations):.4f}, Max: {max(gram_correlations):.4f}")

    elapsed = time.time() - t0

    # =========================================================
    # Compile results
    # =========================================================
    results = {
        "experiment": "BQ2-7B",
        "name": "Kernel Dynamics on Qwen2.5-7B — Cross-Model Replication",
        "model": "Qwen2.5-7B (28 layers, d=3584)",
        "reference": "Qwen2.5-3B (36 layers, d=2048)",
        "method": f"Full eigendecomposition of G^(l) at all {N_LAYERS} layers. "
                  "Eigenvalue trajectory tracking. Lyapunov exponents from eigenvalue "
                  "growth rates. ΔG eigendecomposition with language/category alignment. "
                  "Phases normalized by depth fraction (7 layers each vs 9 on 3B).",
        "eigenvalue_trajectories": eigenvalue_trajectories,
        "effective_ranks": effective_ranks,
        "lyapunov_exponents": {k: v["exponent"] for k, v in lyapunov.items()},
        "lyapunov_per_layer": {k: v["per_layer"] for k, v in lyapunov.items()},
        "phase_lyapunov_spectrum": phase_lyapunov,
        "delta_g_analysis": delta_g_analysis,
        "g_eigenvector_alignment": g_eigvec_alignment,
        "gram_correlations": gram_correlations,
        "mean_gram_correlation": mean_corr,
        "reference_3b": REF_3B,
        "elapsed_seconds": round(elapsed, 1),
    }

    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    results = to_native(results)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH} in {elapsed:.1f}s")

    # =========================================================
    # Summary with 3B comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY: BQ2-7B Kernel Dynamics (with 3B reference)")
    print("=" * 70)

    print(f"\n--- Effective Rank of G through depth ---")
    for thr in ["rank_50", "rank_90", "rank_95", "rank_99"]:
        vals = effective_ranks[thr]
        print(f"  {thr}: min={min(vals)} (L{vals.index(min(vals))}), "
              f"max={max(vals)} (L{vals.index(max(vals))})")
    r50 = effective_ranks["rank_50"]
    r50_all_one = all(v == 1 for v in r50)
    print(f"  rank_50 = 1 at ALL layers: {'YES' if r50_all_one else 'NO'} "
          f"(3B ref: rank_50=1 at ALL layers)")

    print(f"\n--- Mode 0 Hourglass ---")
    mode0 = eigenvalue_trajectories["mode_0"]["values"]
    print(f"  7B: L0={mode0[0]:.0f} → L{mode0.index(min(mode0))}={min(mode0):.0f} (min) "
          f"→ L{mode0.index(max(mode0))}={max(mode0):.0f} (max) → L{N_LAYERS-1}={mode0[-1]:.0f}")
    print(f"  3B ref: {REF_3B['mode_0_hourglass']}")

    print(f"\n--- Gram Correlation ---")
    print(f"  7B mean: {mean_corr:.4f} (3B ref: {REF_3B['gram_correlation']})")

    print(f"\n--- Top Lyapunov Exponents ---")
    for k in range(10):
        exp = lyapunov[f"mode_{k}"]["exponent"]
        label = "GROWING" if exp > 0.01 else "SHRINKING" if exp < -0.01 else "STABLE"
        print(f"  Mode {k}: λ = {exp:+.6f} ({label})")

    print(f"\n--- Phase Lyapunov Profile (THE KEY TEST) ---")
    print(f"  {'Phase':<15s} {'7B pos/20':>10s}  {'3B ref pos/20':>14s}  {'Match?':>7s}")
    for phase_name in ["early", "adversarial", "cooperative", "late"]:
        exps = phase_lyapunov[phase_name]
        n_pos = sum(1 for e in exps if e > 0)
        ref_pos = REF_3B["phase_positive_modes"][phase_name]
        match = "YES" if abs(n_pos - ref_pos) <= 3 else "NO"
        print(f"  {phase_name:<15s} {n_pos:>10d}  {ref_pos:>14d}  {match:>7s}")

    print(f"\n--- ΔG Top Eigenvector Character ---")
    print(f"  Layer pair : top_eigval | lang_align | cat_align | interpretation")
    for entry in delta_g_analysis:
        lang = entry["top5_language_alignment"][0]
        cat = entry["top5_category_alignment"][0]
        val = entry["top_5_eigenvalues"][0]
        if lang > 0.3:
            interp = "LANGUAGE"
        elif cat > 0.3:
            interp = "CATEGORY"
        elif lang > 0.1 or cat > 0.1:
            interp = "mixed"
        else:
            interp = "other"
        if abs(val) > 5:
            print(f"  {entry['layer_pair']:>10s}: {val:>8.2f} | {lang:.3f}      | {cat:.3f}     | {interp}")

    # G eigenvector alignment at key layers (normalized depth fractions)
    key_layers = [0, 5, 9, 13, 14, 20, 24, 27]  # ~same fractions as 3B's [0,8,12,17,18,26,32,35]
    print(f"\n--- G Top Eigenvector Alignment at Key Layers ---")
    print(f"  (Key layers chosen at ~same depth fractions as 3B's [0,8,12,17,18,26,32,35])")
    for L in key_layers:
        if L < N_LAYERS:
            entry = g_eigvec_alignment[L]
            l0 = entry["top10_language_alignment"][0]
            c0 = entry["top10_category_alignment"][0]
            l1 = entry["top10_language_alignment"][1]
            c1 = entry["top10_category_alignment"][1]
            frac = L / (N_LAYERS - 1)
            print(f"  L{L} ({frac:.0%}): mode0 lang={l0:.3f} cat={c0:.3f} | "
                  f"mode1 lang={l1:.3f} cat={c1:.3f}")

    print(f"\n--- VERDICT ---")
    phases_match = True
    for phase_name in ["early", "adversarial", "cooperative", "late"]:
        exps = phase_lyapunov[phase_name]
        n_pos = sum(1 for e in exps if e > 0)
        ref_pos = REF_3B["phase_positive_modes"][phase_name]
        if abs(n_pos - ref_pos) > 5:
            phases_match = False
    if r50_all_one and phases_match:
        print("  FUNNEL REPLICATES: rank_50=1 + 4-phase Lyapunov signature matches 3B")
    elif r50_all_one:
        print("  PARTIAL: rank_50=1 replicates, but phase signature differs")
    else:
        print("  DIFFERENT: rank or phase signature does NOT match 3B")


if __name__ == "__main__":
    main()
