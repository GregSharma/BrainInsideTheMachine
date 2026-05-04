"""
Exp BQ2: Kernel Dynamics — Lyapunov Exponents and ΔG Spectral Analysis

Extends BQ with the dynamical systems analysis of Gram matrix evolution.

Method:
  1. Eigendecompose G^(ℓ) at ALL 36 layers. Track eigenvalue trajectories through depth.
  2. Compute Lyapunov exponents: growth/decay rates of each eigenvalue mode.
  3. Eigendecompose ΔG^(ℓ) at each layer. Align top eigenvectors of ΔG with
     category/language structure to determine WHAT each layer changed.
  4. Track effective rank of G through depth — the reasoning bottleneck.
  5. Separate ΔG into within-math and language components.

Uses multilingual_all_layers.npz cache. No model load needed.
"""

import numpy as np
import json
import time
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "output/multilingual_all_layers.npz"
OUT_PATH = "output/expBQ2_kernel_dynamics.json"

LANGS = ["ar", "en", "es", "ja", "ko", "sw", "zh"]
N_LAYERS = 36
N_PROBLEMS = 200
N_TOTAL = len(LANGS) * N_PROBLEMS  # 1400

PHASES = {
    "early": list(range(0, 9)),
    "adversarial": list(range(9, 18)),
    "cooperative": list(range(18, 27)),
    "late": list(range(27, 36)),
}


def load_layer(data, layer):
    """Stack all 7 languages at a given layer → (1400, 2048)."""
    arrays = []
    for lang in LANGS:
        arrays.append(data[f"{lang}_L{layer}"])
    return np.vstack(arrays)


def make_label_vectors(categories):
    """Build label indicator vectors for projecting eigenvectors.
    Returns:
      lang_indicators: (1400, 7) — one-hot language
      cat_indicators: (1400, 5) — one-hot category
    """
    n = N_PROBLEMS
    # Language indicators
    lang_ind = np.zeros((N_TOTAL, len(LANGS)))
    for i, lang in enumerate(LANGS):
        lang_ind[i*n:(i+1)*n, i] = 1.0

    # Category indicators (same for all languages)
    n_cats = int(categories.max()) + 1
    cat_ind = np.zeros((N_TOTAL, n_cats))
    for i in range(len(LANGS)):
        for j in range(n):
            cat_ind[i*n + j, categories[j]] = 1.0

    return lang_ind, cat_ind


def eigvec_alignment(v, indicators):
    """How much of eigenvector v is explained by a set of indicator vectors.
    Returns: R² (fraction of variance of v explained by indicators)."""
    # Project v onto indicator subspace
    Q, _ = np.linalg.qr(indicators)
    proj = Q @ (Q.T @ v)
    ss_proj = np.sum(proj**2)
    ss_total = np.sum(v**2)
    return float(ss_proj / ss_total) if ss_total > 0 else 0.0


def main():
    t0 = time.time()
    print("Loading multilingual cache...")
    data = np.load(DATA_PATH, allow_pickle=True)
    categories = data["categories"]
    lang_ind, cat_ind = make_label_vectors(categories)

    # =========================================================
    # 1. Compute G at every layer + full eigendecomposition
    # =========================================================
    print("Computing Gram matrices and eigendecompositions for all 36 layers...")
    grams = []
    eigenvalues_all = []  # (36, 1400) — sorted descending
    eigenvectors_top = []  # store top-k eigenvectors for alignment analysis

    TOP_K = 20  # track top 20 eigenvalues/vectors

    for L in range(N_LAYERS):
        H = load_layer(data, L)
        G = cosine_similarity(H)
        grams.append(G)

        # Full eigendecomposition (symmetric → eigvalsh is fast)
        eigenvals, eigvecs = np.linalg.eigh(G)
        # eigh returns ascending; reverse for descending
        eigenvals = eigenvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        eigenvalues_all.append(eigenvals)
        eigenvectors_top.append(eigvecs[:, :TOP_K])

        if L % 6 == 0:
            print(f"  L{L}: top eigenvalues = {eigenvals[:5].round(1)}, "
                  f"sum = {eigenvals.sum():.1f}")

    eigenvalues_all = np.array(eigenvalues_all)  # (36, 1400)

    # =========================================================
    # 2. Eigenvalue trajectories through depth
    # =========================================================
    print("\nTracking eigenvalue trajectories...")
    # Track top-k eigenvalues through depth
    eigenvalue_trajectories = {}
    for k in range(TOP_K):
        eigenvalue_trajectories[f"mode_{k}"] = {
            "values": [float(eigenvalues_all[L, k]) for L in range(N_LAYERS)],
        }

    # Cumulative energy in top-k modes
    total_energy = np.array([eigenvalues_all[L].sum() for L in range(N_LAYERS)])
    effective_ranks = {}
    for threshold in [0.50, 0.90, 0.95, 0.99]:
        ranks = []
        for L in range(N_LAYERS):
            cumsum = np.cumsum(eigenvalues_all[L]) / eigenvalues_all[L].sum()
            rank = int(np.searchsorted(cumsum, threshold) + 1)
            ranks.append(rank)
        effective_ranks[f"rank_{int(threshold*100)}"] = ranks
    print(f"  Effective rank_90 range: {min(effective_ranks['rank_90'])} - {max(effective_ranks['rank_90'])}")
    print(f"  Effective rank_99 range: {min(effective_ranks['rank_99'])} - {max(effective_ranks['rank_99'])}")

    # =========================================================
    # 3. Lyapunov exponents of eigenvalue modes
    # =========================================================
    print("\nComputing Lyapunov exponents...")
    # For each eigenvalue mode k, Lyapunov exponent = mean(log(λ_k^(L+1) / λ_k^(L))) over L
    # We need to handle negative eigenvalues carefully — use absolute values
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

        # Eigendecompose ΔG
        dG_vals, dG_vecs = np.linalg.eigh(dG)
        dG_vals = dG_vals[::-1]
        dG_vecs = dG_vecs[:, ::-1]

        # Top eigenvectors of ΔG: align with language and category structure
        top_lang_alignment = []
        top_cat_alignment = []
        for k in range(min(5, len(dG_vals))):
            v = dG_vecs[:, k]
            la = eigvec_alignment(v, lang_ind)
            ca = eigvec_alignment(v, cat_ind)
            top_lang_alignment.append(round(la, 4))
            top_cat_alignment.append(round(ca, 4))

        # Effective rank of ΔG
        abs_vals = np.abs(dG_vals)
        sorted_abs = np.sort(abs_vals)[::-1]
        cumulative = np.cumsum(sorted_abs**2) / np.sum(sorted_abs**2)
        delta_rank_90 = int(np.searchsorted(cumulative, 0.90) + 1)
        delta_rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)

        # Frobenius norm
        frob = float(np.linalg.norm(dG, 'fro'))

        # Spectral gap: ratio of top eigenvalue to second
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

        if L % 6 == 0:
            print(f"  L{L}→L{L+1}: top_eigval={dG_vals[0]:.4f}, "
                  f"rank_90={delta_rank_90}, rank_99={delta_rank_99}, "
                  f"lang_align={top_lang_alignment[0]:.3f}, cat_align={top_cat_alignment[0]:.3f}")

    # =========================================================
    # 5. Eigenvector alignment at G (not ΔG) — what structure does G encode?
    # =========================================================
    print("\nTop eigenvector alignment with language/category at each layer...")
    g_eigvec_alignment = []
    for L in range(N_LAYERS):
        top_vecs = eigenvectors_top[L]  # (1400, TOP_K)
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
        if L % 6 == 0:
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

    elapsed = time.time() - t0

    # =========================================================
    # Compile results
    # =========================================================
    results = {
        "experiment": "BQ2",
        "name": "Kernel Dynamics — Lyapunov Exponents and ΔG Spectral Analysis",
        "method": "Full eigendecomposition of G^(ℓ) at all 36 layers. Eigenvalue trajectory tracking. "
                  "Lyapunov exponents from eigenvalue growth rates. ΔG eigendecomposition with "
                  "language/category alignment of top eigenvectors.",
        "eigenvalue_trajectories": eigenvalue_trajectories,
        "effective_ranks": effective_ranks,
        "lyapunov_exponents": {k: v["exponent"] for k, v in lyapunov.items()},
        "lyapunov_per_layer": {k: v["per_layer"] for k, v in lyapunov.items()},
        "phase_lyapunov_spectrum": phase_lyapunov,
        "delta_g_analysis": delta_g_analysis,
        "g_eigenvector_alignment": g_eigvec_alignment,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Convert all numpy types to Python natives for JSON serialization
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
    # Summary
    # =========================================================
    print("\n" + "="*70)
    print("SUMMARY: Kernel Dynamics")
    print("="*70)

    print(f"\n--- Effective Rank of G through depth ---")
    for thr in ["rank_50", "rank_90", "rank_95", "rank_99"]:
        vals = effective_ranks[thr]
        print(f"  {thr}: min={min(vals)} (L{vals.index(min(vals))}), "
              f"max={max(vals)} (L{vals.index(max(vals))})")

    print(f"\n--- Top Lyapunov Exponents ---")
    for k in range(10):
        exp = lyapunov[f"mode_{k}"]["exponent"]
        label = "GROWING" if exp > 0.01 else "SHRINKING" if exp < -0.01 else "STABLE"
        print(f"  Mode {k}: λ = {exp:+.6f} ({label})")

    print(f"\n--- Phase Lyapunov Profile ---")
    for phase_name in ["early", "adversarial", "cooperative", "late"]:
        exps = phase_lyapunov[phase_name]
        mean_exp = np.mean(exps)
        n_pos = sum(1 for e in exps if e > 0)
        print(f"  {phase_name}: mean={mean_exp:+.6f}, positive={n_pos}/{TOP_K}")

    print(f"\n--- ΔG Top Eigenvector Character (what each layer changed) ---")
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
        if abs(val) > 5:  # only print notable transitions
            print(f"  {entry['layer_pair']:>10s}: {val:>8.2f} | {lang:.3f}      | {cat:.3f}     | {interp}")

    # G eigenvector alignment summary
    print(f"\n--- G Top Eigenvector Character (what G encodes at each layer) ---")
    for L in [0, 8, 12, 17, 18, 26, 32, 35]:
        entry = g_eigvec_alignment[L]
        l0 = entry["top10_language_alignment"][0]
        c0 = entry["top10_category_alignment"][0]
        l1 = entry["top10_language_alignment"][1]
        c1 = entry["top10_category_alignment"][1]
        print(f"  L{L}: mode0 lang={l0:.3f} cat={c0:.3f} | mode1 lang={l1:.3f} cat={c1:.3f}")


if __name__ == "__main__":
    main()
