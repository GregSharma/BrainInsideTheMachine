"""
Exp C1: Computational Gram Matrix — representational vs computational rank

The centered Gram G^repr has rank_90 ≈ 20 across all layers.
BS proved SVD truncation at k=500 (99.9% variance) → 0/20 accuracy.

Hypothesis: The MLP activation pattern a_p = SiLU(gate_proj(h_p)) * up_proj(h_p) ∈ ℝ^{d_ff}
has MUCH higher effective rank than the hidden state h_p ∈ ℝ^d, because SiLU is
non-polynomial analytic → generic rank explosion from r=20 latent to min(d_ff, N).

This experiment:
1. Loads cached hidden states (1400 problems × 36 layers × 2048-D)
2. Loads model weights (gate_proj, up_proj per layer)
3. Computes MLP activations: a_p = SiLU(W_gate @ h_p) * (W_up @ h_p)
4. Builds centered Gram matrices G^comp and G^repr at each layer
5. Compares rank_50, rank_90, rank_95, rank_99 trajectories
6. Computes Lyapunov spectrum of G^comp through depth
7. Measures the rank gap = rank_90(G^comp) - rank_90(G^repr)

If rank_90(G^comp) ≈ 200: the model describes in 20D, processes in 200D.
If rank_90(G^comp) ≈ 20: the rank gap theory needs rethinking.
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

MODEL_NAME = "Qwen/Qwen2.5-3B"
CACHE_PATH = Path("output/multilingual_all_layers.npz")
OUTPUT_DIR = Path("output")

LANGS = ["ar", "en", "es", "ja", "ko", "sw", "zh"]
N_LAYERS = 36

def effective_rank(eigenvalues, threshold):
    """Minimum k such that sum(top-k) / sum(all) >= threshold."""
    total = eigenvalues.sum()
    if total < 1e-12:
        return len(eigenvalues)
    cumulative = np.cumsum(eigenvalues) / total
    k = np.searchsorted(cumulative, threshold) + 1
    return min(k, len(eigenvalues))


def centered_gram(H):
    """Compute centered Gram matrix from (N, d) matrix. Returns G and eigenvalues."""
    mu = H.mean(axis=0, keepdims=True)
    H_centered = H - mu
    G = H_centered @ H_centered.T  # (N, N)
    eigvals = np.linalg.eigvalsh(G)[::-1]  # descending
    eigvals = np.maximum(eigvals, 0)  # clip numerical negatives
    return G, eigvals


def lyapunov_exponents(eigvals_by_layer, window=9):
    """Compute sliding-window Lyapunov exponents from eigenvalue trajectories."""
    n_layers = len(eigvals_by_layer)
    n_modes = min(len(e) for e in eigvals_by_layer)
    n_modes = min(n_modes, 20)  # track top 20 modes

    # Build eigenvalue matrix (layers × modes)
    eig_matrix = np.zeros((n_layers, n_modes))
    for l in range(n_layers):
        eig_matrix[l, :] = eigvals_by_layer[l][:n_modes]

    # Sliding window Lyapunov
    half = window // 2
    lyap_windows = []
    for center in range(half, n_layers - half):
        start, end = center - half, center + half
        exponents = np.zeros(n_modes)
        for m in range(n_modes):
            if eig_matrix[start, m] > 1e-10 and eig_matrix[end, m] > 1e-10:
                exponents[m] = np.log(eig_matrix[end, m] / eig_matrix[start, m]) / window
            else:
                exponents[m] = 0.0
        positive = int(np.sum(exponents > 0.001))
        lyap_windows.append({
            "center_layer": center,
            "exponents": exponents.tolist(),
            "positive_modes": positive,
            "negative_modes": int(np.sum(exponents < -0.001)),
        })
    return lyap_windows


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP C1: COMPUTATIONAL GRAM MATRIX")
    print("Representational rank vs Computational rank")
    print("=" * 70)

    # --- Load cached hidden states ---
    print(f"\nLoading cached hidden states from {CACHE_PATH}...")
    cache = np.load(CACHE_PATH)

    # Stack all 1400 problems per layer
    # Order: ar(200) + en(200) + es(200) + ja(200) + ko(200) + sw(200) + zh(200) = 1400
    H_all = {}
    for l in range(N_LAYERS):
        arrays = []
        for lang in LANGS:
            key = f"{lang}_L{l}"
            arrays.append(cache[key].astype(np.float32))
        H_all[l] = np.concatenate(arrays, axis=0)  # (1400, 2048)
    del cache

    N, d = H_all[0].shape
    print(f"  N={N} problems, d={d} hidden dim, {N_LAYERS} layers")

    # --- Load model (weights only, for gate_proj and up_proj) ---
    print(f"\nLoading model weights from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        device_map="cpu", trust_remote_code=True
    )
    d_ff = model.config.intermediate_size
    print(f"  d_ff={d_ff} (MLP intermediate dimension)")
    print(f"  Rank explosion theory: r=20 latent → up to min(d_ff={d_ff}, N={N})={min(d_ff, N)}")

    # --- Compute both Grams at every layer ---
    results_by_layer = []
    repr_eigvals_all = []
    comp_eigvals_all = []

    for l in range(N_LAYERS):
        t_layer = time.time()
        H = H_all[l]  # (N, d) = (1400, 2048)

        # --- Representational Gram ---
        G_repr, eig_repr = centered_gram(H)

        # --- Computational Gram: a_p = SiLU(W_gate @ h_p) * (W_up @ h_p) ---
        mlp = model.model.layers[l].mlp
        W_gate = mlp.gate_proj.weight.detach().float().numpy()  # (d_ff, d)
        W_up = mlp.up_proj.weight.detach().float().numpy()      # (d_ff, d)

        # Compute activations for all problems at once
        # gate_out = H @ W_gate.T  → (N, d_ff)
        # up_out = H @ W_up.T     → (N, d_ff)
        # activations = SiLU(gate_out) * up_out  → (N, d_ff)
        gate_out = H @ W_gate.T  # (1400, d_ff)
        up_out = H @ W_up.T      # (1400, d_ff)

        # SiLU = x * sigmoid(x)
        gate_activated = gate_out * (1.0 / (1.0 + np.exp(-np.clip(gate_out, -50, 50))))
        activations = gate_activated * up_out  # (1400, d_ff)

        # Centered Gram of activations
        G_comp, eig_comp = centered_gram(activations)

        # Free memory
        del gate_out, up_out, gate_activated, activations

        # --- Compute ranks ---
        r50_repr = effective_rank(eig_repr, 0.50)
        r90_repr = effective_rank(eig_repr, 0.90)
        r95_repr = effective_rank(eig_repr, 0.95)
        r99_repr = effective_rank(eig_repr, 0.99)

        r50_comp = effective_rank(eig_comp, 0.50)
        r90_comp = effective_rank(eig_comp, 0.90)
        r95_comp = effective_rank(eig_comp, 0.95)
        r99_comp = effective_rank(eig_comp, 0.99)

        # Gram correlation between repr and comp
        upper_repr = G_repr[np.triu_indices(N, k=1)]
        upper_comp = G_comp[np.triu_indices(N, k=1)]
        gram_corr = float(np.corrcoef(upper_repr, upper_comp)[0, 1])

        layer_result = {
            "layer": l,
            "repr": {
                "rank_50": r50_repr, "rank_90": r90_repr,
                "rank_95": r95_repr, "rank_99": r99_repr,
                "top5_eigvals": eig_repr[:5].tolist(),
                "trace": float(eig_repr.sum()),
            },
            "comp": {
                "rank_50": r50_comp, "rank_90": r90_comp,
                "rank_95": r95_comp, "rank_99": r99_comp,
                "top5_eigvals": eig_comp[:5].tolist(),
                "trace": float(eig_comp.sum()),
            },
            "rank_gap_90": r90_comp - r90_repr,
            "rank_gap_50": r50_comp - r50_repr,
            "gram_correlation": gram_corr,
        }
        results_by_layer.append(layer_result)
        repr_eigvals_all.append(eig_repr)
        comp_eigvals_all.append(eig_comp)

        dt = time.time() - t_layer
        print(f"  L{l:2d}: repr r50={r50_repr:3d} r90={r90_repr:3d} | "
              f"comp r50={r50_comp:3d} r90={r90_comp:3d} | "
              f"gap90={r90_comp - r90_repr:+4d} | corr={gram_corr:.3f} | {dt:.1f}s")

    # --- Lyapunov spectra for both ---
    print("\nComputing Lyapunov spectra...")
    lyap_repr = lyapunov_exponents(repr_eigvals_all, window=9)
    lyap_comp = lyapunov_exponents(comp_eigvals_all, window=9)

    # --- Phase comparison ---
    print("\n" + "=" * 70)
    print("LYAPUNOV PHASE COMPARISON (window=9)")
    print(f"{'Center':>6} | {'Repr +modes':>11} | {'Comp +modes':>11} | {'Divergence':>10}")
    print("-" * 50)
    for lr, lc in zip(lyap_repr, lyap_comp):
        center = lr["center_layer"]
        rp = lr["positive_modes"]
        cp = lc["positive_modes"]
        div = "<<<" if abs(rp - cp) >= 5 else ""
        print(f"  L{center:2d}   |    {rp:2d}/20     |    {cp:2d}/20     |  {div}")

    # --- Summary ---
    repr_r90 = [r["repr"]["rank_90"] for r in results_by_layer]
    comp_r90 = [r["comp"]["rank_90"] for r in results_by_layer]
    gap_r90 = [r["rank_gap_90"] for r in results_by_layer]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Representational rank_90: min={min(repr_r90)}, max={max(repr_r90)}, mean={np.mean(repr_r90):.1f}")
    print(f"  Computational rank_90:   min={min(comp_r90)}, max={max(comp_r90)}, mean={np.mean(comp_r90):.1f}")
    print(f"  Rank gap (r90):          min={min(gap_r90)}, max={max(gap_r90)}, mean={np.mean(gap_r90):.1f}")
    print(f"  Gram correlation:        min={min(r['gram_correlation'] for r in results_by_layer):.3f}, "
          f"max={max(r['gram_correlation'] for r in results_by_layer):.3f}")

    if np.mean(comp_r90) > 2 * np.mean(repr_r90):
        print(f"\n  >>> RANK EXPLOSION CONFIRMED: comp/repr ratio = {np.mean(comp_r90)/np.mean(repr_r90):.1f}x")
        print(f"  >>> The model DESCRIBES in ~{int(np.mean(repr_r90))}D, PROCESSES in ~{int(np.mean(comp_r90))}D")
    elif np.mean(comp_r90) < 1.5 * np.mean(repr_r90):
        print(f"\n  >>> RANK EXPLOSION NOT CONFIRMED: comp/repr ratio = {np.mean(comp_r90)/np.mean(repr_r90):.1f}x")
        print(f"  >>> Activations track representations — BS failure is not about rank")
    else:
        print(f"\n  >>> MODERATE RANK GAP: comp/repr ratio = {np.mean(comp_r90)/np.mean(repr_r90):.1f}x")

    # --- Delta-G for cheap proxy S_ℓ ---
    delta_g_frob = []
    for l in range(N_LAYERS - 1):
        G_curr, _ = centered_gram(H_all[l])
        G_next, _ = centered_gram(H_all[l + 1])
        dg = np.linalg.norm(G_next - G_curr, 'fro')
        delta_g_frob.append(float(dg))
    delta_g_frob.append(0.0)  # last layer has no successor

    # Cheap paradox index: S̃_ℓ = r_90(comp) / (r_90(repr) * (||ΔG|| + ε))
    eps = 1e-6
    paradox_index = []
    for l in range(N_LAYERS):
        s = comp_r90[l] / (repr_r90[l] * (delta_g_frob[l] + eps))
        paradox_index.append({
            "layer": l,
            "S_tilde": float(s),
            "r90_comp": comp_r90[l],
            "r90_repr": repr_r90[l],
            "delta_g_frob": delta_g_frob[l],
        })

    # --- Save ---
    output = {
        "experiment": "C1: Computational Gram Matrix",
        "model": MODEL_NAME,
        "N": N,
        "d": d,
        "d_ff": d_ff,
        "n_layers": N_LAYERS,
        "layers": results_by_layer,
        "lyapunov_repr": lyap_repr,
        "lyapunov_comp": lyap_comp,
        "paradox_index": paradox_index,
        "summary": {
            "repr_rank_90": {"min": min(repr_r90), "max": max(repr_r90), "mean": float(np.mean(repr_r90))},
            "comp_rank_90": {"min": min(comp_r90), "max": max(comp_r90), "mean": float(np.mean(comp_r90))},
            "rank_gap_90": {"min": min(gap_r90), "max": max(gap_r90), "mean": float(np.mean(gap_r90))},
            "comp_repr_ratio": float(np.mean(comp_r90) / np.mean(repr_r90)),
        },
        "runtime_seconds": time.time() - t0,
    }

    out_path = OUTPUT_DIR / "expC1_computational_gram.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
