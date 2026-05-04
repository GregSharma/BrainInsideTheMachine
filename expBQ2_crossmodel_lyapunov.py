"""
Exp BQ2-XM: Cross-Model Lyapunov Analysis
==========================================
Unified Gram matrix / Lyapunov analysis across ALL models:
  - Qwen2.5-3B  (36 layers, d=2048)  — baseline
  - Qwen2.5-7B  (28 layers, d=3584)  — confirmed
  - Qwen2.5-14B (48 layers, d=5120)  — NEW: same family, 3rd scaling point
  - Qwen3-8B    (36 layers, d=4096)  — NEW: different generation

Computes per-model:
  - Full Gram eigendecomposition at each layer
  - Eigenvalue trajectories (top 20 modes)
  - Lyapunov exponents (global and phase-resolved)
  - ΔG spectral analysis with lang/cat alignment
  - Gram correlation between consecutive layers
  - Effective ranks (50/90/95/99)

Outputs a single JSON with all models for direct cross-model comparison.
All layer indices are also reported as depth fractions for normalization.
"""

import numpy as np
import json
import time
from sklearn.metrics.pairwise import cosine_similarity

LANGS = ["ar", "en", "es", "ja", "ko", "sw", "zh"]
N_PROBLEMS = 200
N_TOTAL = len(LANGS) * N_PROBLEMS  # 1400
TOP_K = 20

# Model configs: name, npz path, n_layers
MODELS = {
    "qwen2_5_3b": {
        "label": "Qwen2.5-3B",
        "path": "output/multilingual_all_layers.npz",
        "n_layers": 36,
        "d": 2048,
        "family": "qwen2",
    },
    "qwen2_5_7b": {
        "label": "Qwen2.5-7B",
        "path": "output/multilingual_all_layers_qwen2_5_7b.npz",
        "n_layers": 28,
        "d": 3584,
        "family": "qwen2",
    },
    "qwen2_5_14b": {
        "label": "Qwen2.5-14B",
        "path": "output/multilingual_all_layers_qwen2_5_14b.npz",
        "n_layers": 48,
        "d": 5120,
        "family": "qwen2",
    },
    "qwen3_8b": {
        "label": "Qwen3-8B",
        "path": "output/multilingual_all_layers_qwen3_8b.npz",
        "n_layers": 36,
        "d": 4096,
        "family": "qwen3",
    },
}

OUT_PATH = "output/expBQ2_crossmodel_lyapunov.json"


def load_layer(data, layer):
    """Stack all 7 languages at a given layer → (1400, d)."""
    return np.vstack([data[f"{lang}_L{layer}"] for lang in LANGS])


def make_label_vectors(categories):
    """Build language and category indicator vectors."""
    n = N_PROBLEMS
    lang_ind = np.zeros((N_TOTAL, len(LANGS)))
    for i in range(len(LANGS)):
        lang_ind[i*n:(i+1)*n, i] = 1.0

    n_cats = int(categories.max()) + 1
    cat_ind = np.zeros((N_TOTAL, n_cats))
    for i in range(len(LANGS)):
        for j in range(n):
            cat_ind[i*n + j, categories[j]] = 1.0

    return lang_ind, cat_ind


def eigvec_alignment(v, indicators):
    """Fraction of eigenvector variance explained by indicator subspace."""
    Q, _ = np.linalg.qr(indicators)
    proj = Q @ (Q.T @ v)
    return float(np.sum(proj**2) / np.sum(v**2)) if np.sum(v**2) > 0 else 0.0


def analyze_model(model_key, config):
    """Full Gram/Lyapunov analysis for one model. Returns results dict."""
    import os
    if not os.path.exists(config["path"]):
        print(f"  SKIPPING {model_key}: {config['path']} not found")
        return None

    t0 = time.time()
    n_layers = config["n_layers"]
    label = config["label"]

    print(f"\n{'='*60}")
    print(f"Analyzing {label} ({n_layers} layers, d={config['d']})")
    print(f"{'='*60}")

    data = np.load(config["path"], allow_pickle=True)
    categories = data["categories"]
    lang_ind, cat_ind = make_label_vectors(categories)

    # 1. Gram matrices + eigendecomposition
    print("  Computing Gram matrices...")
    grams = []
    eigenvalues_all = []
    eigenvectors_top = []

    for L in range(n_layers):
        H = load_layer(data, L)
        G = cosine_similarity(H)
        grams.append(G)

        eigenvals, eigvecs = np.linalg.eigh(G)
        eigenvals = eigenvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        eigenvalues_all.append(eigenvals)
        eigenvectors_top.append(eigvecs[:, :TOP_K])

    eigenvalues_all = np.array(eigenvalues_all)

    # 2. Effective ranks
    print("  Computing effective ranks...")
    effective_ranks = {}
    for threshold in [0.50, 0.90, 0.95, 0.99]:
        ranks = []
        for L in range(n_layers):
            cumsum = np.cumsum(eigenvalues_all[L]) / eigenvalues_all[L].sum()
            rank = int(np.searchsorted(cumsum, threshold) + 1)
            ranks.append(rank)
        effective_ranks[f"rank_{int(threshold*100)}"] = ranks

    r50 = effective_ranks["rank_50"]
    r90 = effective_ranks["rank_90"]
    print(f"    rank_50: all={set(r50)}, constant={len(set(r50))==1}")
    print(f"    rank_90: min={min(r90)} (L{r90.index(min(r90))}), "
          f"max={max(r90)} (L{r90.index(max(r90))}), peak={max(r90)}")

    # 3. Mode-0 trajectory (hourglass)
    mode0 = [float(eigenvalues_all[L, 0]) for L in range(n_layers)]
    mode0_min_idx = int(np.argmin(mode0))
    mode0_max_idx = int(np.argmax(mode0))

    # 4. Lyapunov exponents
    print("  Computing Lyapunov exponents...")
    lyapunov_per_layer = {}
    lyapunov_global = {}
    for k in range(TOP_K):
        log_ratios = []
        for L in range(n_layers - 1):
            v_curr = abs(eigenvalues_all[L, k])
            v_next = abs(eigenvalues_all[L + 1, k])
            if v_curr > 1e-6:
                log_ratios.append(float(np.log(v_next / v_curr)))
            else:
                log_ratios.append(0.0)
        lyapunov_per_layer[f"mode_{k}"] = log_ratios
        lyapunov_global[f"mode_{k}"] = float(np.mean(log_ratios))

    # 5. Phase-resolved Lyapunov (4 equal bands by depth fraction)
    print("  Computing phase Lyapunov...")
    band_size = n_layers // 4
    remainder = n_layers % 4
    phase_boundaries = []
    start = 0
    for i in range(4):
        end = start + band_size + (1 if i < remainder else 0)
        phase_boundaries.append((start, end))
        start = end

    phase_names = ["early", "adversarial", "cooperative", "late"]
    phase_lyapunov = {}
    phase_positive_modes = {}
    for pi, (pstart, pend) in enumerate(phase_boundaries):
        pname = phase_names[pi]
        phase_exps = []
        for k in range(TOP_K):
            per_layer = lyapunov_per_layer[f"mode_{k}"]
            # Layer L's log_ratio is for L→L+1 transition
            phase_vals = [per_layer[L] for L in range(pstart, min(pend, n_layers - 1))]
            phase_exps.append(float(np.mean(phase_vals)) if phase_vals else 0.0)
        phase_lyapunov[pname] = phase_exps
        n_pos = sum(1 for e in phase_exps if e > 0)
        phase_positive_modes[pname] = n_pos
        print(f"    {pname} (L{pstart}-L{pend-1}): {n_pos}/{TOP_K} positive modes")

    # 6. ΔG analysis
    print("  Computing ΔG spectral analysis...")
    delta_g_data = []
    for L in range(n_layers - 1):
        dG = grams[L+1] - grams[L]
        dG_vals, dG_vecs = np.linalg.eigh(dG)
        dG_vals = dG_vals[::-1]
        dG_vecs = dG_vecs[:, ::-1]

        # Lang/cat alignment of top ΔG eigenvector
        top_lang = eigvec_alignment(dG_vecs[:, 0], lang_ind)
        top_cat = eigvec_alignment(dG_vecs[:, 0], cat_ind)

        frob = float(np.linalg.norm(dG, 'fro'))

        delta_g_data.append({
            "layer_pair": f"L{L}->L{L+1}",
            "depth_frac": round(L / (n_layers - 1), 4),
            "frobenius": round(frob, 2),
            "top_eigenvalue": round(float(dG_vals[0]), 4),
            "lang_align": round(top_lang, 4),
            "cat_align": round(top_cat, 4),
        })

    # 7. Gram correlations
    print("  Computing Gram correlations...")
    gram_corrs = []
    for L in range(n_layers - 1):
        corr = float(np.corrcoef(grams[L].flatten(), grams[L+1].flatten())[0, 1])
        gram_corrs.append(round(corr, 6))
    mean_gram_corr = float(np.mean(gram_corrs))

    # 8. G eigenvector alignment at each layer
    g_alignments = []
    for L in range(n_layers):
        top_vecs = eigenvectors_top[L]
        mode0_lang = eigvec_alignment(top_vecs[:, 0], lang_ind)
        mode0_cat = eigvec_alignment(top_vecs[:, 0], cat_ind)
        mode1_lang = eigvec_alignment(top_vecs[:, 1], lang_ind) if TOP_K > 1 else 0
        mode1_cat = eigvec_alignment(top_vecs[:, 1], cat_ind) if TOP_K > 1 else 0
        g_alignments.append({
            "layer": L,
            "depth_frac": round(L / (n_layers - 1), 4),
            "mode0_lang": round(mode0_lang, 4),
            "mode0_cat": round(mode0_cat, 4),
            "mode1_lang": round(mode1_lang, 4),
            "mode1_cat": round(mode1_cat, 4),
        })

    # Output rupture detection (last layer transition)
    last_frob = delta_g_data[-1]["frobenius"]
    second_last_frob = delta_g_data[-2]["frobenius"] if len(delta_g_data) > 1 else 0
    last_gram_corr = gram_corrs[-1]
    output_rupture = {
        "present": last_frob > 2 * second_last_frob,
        "last_transition_frob": last_frob,
        "penultimate_frob": second_last_frob,
        "frob_ratio": round(last_frob / second_last_frob, 2) if second_last_frob > 0 else float('inf'),
        "last_gram_corr": last_gram_corr,
        "last_dG_lang_align": delta_g_data[-1]["lang_align"],
    }

    # 9. Sliding-window phase detection (window_size=9, matching 3B phase width)
    # This is more accurate than equal bands for models with different layer counts.
    # The compress zone is where the window shows <=2/20 positive modes.
    print("  Computing sliding-window phase profile...")
    WINDOW = 9
    sliding_window_profile = []
    for start in range(max(0, n_layers - WINDOW)):
        end = min(start + WINDOW, n_layers - 1)  # last valid transition index
        n_pos = 0
        for k in range(TOP_K):
            per_layer = lyapunov_per_layer[f"mode_{k}"]
            window_vals = [per_layer[L] for L in range(start, end)]
            if window_vals and np.mean(window_vals) > 0:
                n_pos += 1
        sliding_window_profile.append({
            "start": start,
            "end": start + WINDOW - 1,
            "depth_frac_start": round(start / (n_layers - 1), 3),
            "depth_frac_end": round((start + WINDOW - 1) / (n_layers - 1), 3),
            "positive_modes": n_pos,
        })

    # Detect compress zone: contiguous region where positive_modes <= 2
    compress_windows = [w for w in sliding_window_profile if w["positive_modes"] <= 2]
    if compress_windows:
        compress_start = compress_windows[0]["start"]
        compress_end = compress_windows[-1]["end"]
        compress_min = min(w["positive_modes"] for w in compress_windows)
        compress_depth_start = compress_windows[0]["depth_frac_start"]
        compress_depth_end = compress_windows[-1]["depth_frac_end"]
    else:
        compress_start = compress_end = compress_min = -1
        compress_depth_start = compress_depth_end = -1

    sliding_window_result = {
        "window_size": WINDOW,
        "profile": sliding_window_profile,
        "compress_zone": {
            "layer_start": compress_start,
            "layer_end": compress_end,
            "depth_frac_start": compress_depth_start,
            "depth_frac_end": compress_depth_end,
            "min_positive_modes": compress_min,
            "n_windows_at_min": sum(1 for w in compress_windows if w["positive_modes"] == compress_min),
        },
    }
    print(f"    Compress zone: L{compress_start}-L{compress_end} "
          f"({compress_depth_start:.0%}-{compress_depth_end:.0%} depth), "
          f"min={compress_min}/20")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    return {
        "label": label,
        "n_layers": n_layers,
        "d": config["d"],
        "family": config["family"],
        "effective_ranks": effective_ranks,
        "rank_50_all_one": all(v == 1 for v in r50),
        "rank_90_peak": max(r90),
        "rank_90_peak_layer": r90.index(max(r90)),
        "mode_0_trajectory": mode0,
        "mode_0_min": {"layer": mode0_min_idx, "value": round(mode0[mode0_min_idx], 1)},
        "mode_0_max": {"layer": mode0_max_idx, "value": round(mode0[mode0_max_idx], 1)},
        "lyapunov_global": {k: round(v, 6) for k, v in lyapunov_global.items()},
        "lyapunov_per_layer": lyapunov_per_layer,
        "phase_boundaries": {phase_names[i]: list(range(phase_boundaries[i][0], phase_boundaries[i][1]))
                             for i in range(4)},
        "phase_positive_modes": phase_positive_modes,
        "phase_lyapunov": phase_lyapunov,
        "delta_g": delta_g_data,
        "gram_correlations": gram_corrs,
        "mean_gram_correlation": round(mean_gram_corr, 4),
        "g_eigenvector_alignment": g_alignments,
        "output_rupture": output_rupture,
        "sliding_window": sliding_window_result,
        "elapsed_seconds": round(elapsed, 1),
    }


def cross_model_summary(results):
    """Print formatted cross-model comparison."""
    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON")
    print("=" * 80)

    models = sorted(results.keys())

    # Table 1: Rank invariants
    print("\n--- Rank Invariants ---")
    print(f"  {'Model':<20s} {'rank_50=1?':>10s} {'rank_90 peak':>12s} {'rank_90 @ L':>11s}")
    for m in models:
        r = results[m]
        print(f"  {r['label']:<20s} {'YES' if r['rank_50_all_one'] else 'NO':>10s} "
              f"{r['rank_90_peak']:>12d} {r['rank_90_peak_layer']:>11d} "
              f"({r['rank_90_peak_layer']/(r['n_layers']-1):.0%} depth)")

    # Table 2: Phase Lyapunov
    print("\n--- Phase Positive Modes (of 20) ---")
    print(f"  {'Model':<20s} {'Early':>6s} {'Advers':>7s} {'Coop':>6s} {'Late':>6s}")
    for m in models:
        r = results[m]
        pp = r["phase_positive_modes"]
        print(f"  {r['label']:<20s} {pp['early']:>6d} {pp['adversarial']:>7d} "
              f"{pp['cooperative']:>6d} {pp['late']:>6d}")

    # Table 3: Gram correlation + output rupture
    print("\n--- Gram Dynamics ---")
    print(f"  {'Model':<20s} {'Mean corr':>10s} {'Rupture?':>9s} {'Frob ratio':>11s} "
          f"{'Last corr':>10s} {'Lang align':>11s}")
    for m in models:
        r = results[m]
        rup = r["output_rupture"]
        print(f"  {r['label']:<20s} {r['mean_gram_correlation']:>10.4f} "
              f"{'YES' if rup['present'] else 'no':>9s} {rup['frob_ratio']:>11.1f}x "
              f"{rup['last_gram_corr']:>10.4f} {rup['last_dG_lang_align']:>11.4f}")

    # Table 4: Mode-0 hourglass
    print("\n--- Mode-0 Hourglass ---")
    for m in models:
        r = results[m]
        m0 = r["mode_0_trajectory"]
        print(f"  {r['label']}: L0={m0[0]:.0f} → min=L{r['mode_0_min']['layer']}"
              f"({r['mode_0_min']['value']:.0f}) → max=L{r['mode_0_max']['layer']}"
              f"({r['mode_0_max']['value']:.0f}) → L{r['n_layers']-1}={m0[-1]:.0f}")

    # Table 5: Sliding-window compress zone
    print("\n--- Sliding-Window Compress Zone (window=9, <=2/20 positive) ---")
    print(f"  {'Model':<20s} {'Zone':>12s} {'Depth':>14s} {'Min modes':>10s}")
    for m in models:
        r = results[m]
        if "sliding_window" in r:
            cz = r["sliding_window"]["compress_zone"]
            print(f"  {r['label']:<20s} L{cz['layer_start']}-L{cz['layer_end']:>3d}"
                  f" {cz['depth_frac_start']:.0%}-{cz['depth_frac_end']:.0%}   "
                  f"        {cz['min_positive_modes']}/20")

    # Key questions
    print("\n--- KEY QUESTIONS ---")

    # Q1: Is 1/20 compress universal? (using sliding window, not equal bands)
    compress_modes_sw = {}
    for m in models:
        r = results[m]
        if "sliding_window" in r:
            compress_modes_sw[m] = r["sliding_window"]["compress_zone"]["min_positive_modes"]
        else:
            compress_modes_sw[m] = results[m]["phase_positive_modes"]["adversarial"]
    all_one = all(v == 1 for v in compress_modes_sw.values())
    print(f"  Q1: 1/20 compress invariant (sliding window)? {compress_modes_sw}")
    if all_one:
        print(f"      → YES: exactly 1/20 on ALL {len(models)} models. UNIVERSAL CONSTANT.")
    else:
        print(f"      → NO: varies across models.")

    # Q2: rank_90 scaling
    rank90s = {results[m]["label"]: results[m]["rank_90_peak"] for m in models}
    print(f"  Q2: rank_90 scaling: {rank90s}")

    # Q3: Does Qwen3-8B replicate the funnel?
    if "qwen3_8b" in results:
        q3 = results["qwen3_8b"]
        q25_3b = results.get("qwen2_5_3b")
        if q25_3b:
            same_rank50 = q3["rank_50_all_one"] == q25_3b["rank_50_all_one"]
            compress_close = abs(q3["phase_positive_modes"]["adversarial"] -
                                q25_3b["phase_positive_modes"]["adversarial"]) <= 2
            print(f"  Q3: Qwen3-8B funnel replicates? rank_50={'MATCH' if same_rank50 else 'DIFF'}, "
                  f"compress={'MATCH' if compress_close else 'DIFF'}")
            if same_rank50 and compress_close:
                print(f"      → FUNNEL IS ARCHITECTURE-INVARIANT (not training-specific)")
            else:
                print(f"      → TRAINING MATTERS: different generation = different funnel")


def to_native(obj):
    """Convert numpy types to native Python for JSON serialization."""
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


def main():
    t_total = time.time()

    all_results = {}
    for model_key, config in MODELS.items():
        result = analyze_model(model_key, config)
        if result is not None:
            all_results[model_key] = result

    if not all_results:
        print("No models found. Check npz file paths.")
        return

    # Cross-model comparison
    cross_model_summary(all_results)

    # Save
    output = {
        "experiment": "BQ2-XM",
        "name": "Cross-Model Lyapunov Analysis",
        "models_analyzed": list(all_results.keys()),
        "results": all_results,
        "total_elapsed": round(time.time() - t_total, 1),
    }
    output = to_native(output)

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")
    print(f"Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
