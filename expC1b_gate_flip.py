"""
Exp C1b: Gate Flip Analysis — How many MLP gates change under PCA truncation?

C1 showed comp rank ≈ repr rank (no explosion). BS showed truncation kills accuracy.
Hypothesis: truncation disrupts the CARRIER WAVE (mean μ in 2048-D), which changes
the gate pattern D(h) = diag(σ'(W_gate · h)), breaking the computation.

This experiment:
1. For each problem at each layer, compute the full gate pattern: g(h) = SiLU(W_gate · h)
2. Compute the truncated gate pattern: g(P_k(h-μ)+μ) for k = {5, 20, 50, 100, 200, 500}
3. Measure:
   - Gate sign flips: how many of 11008 gates change sign (on→off or off→on)
   - Gate magnitude change: ||g(h) - g(h_trunc)|| / ||g(h)||
   - Correlation of gate patterns: corr(g(h), g(h_trunc))
   - Active gate overlap: Jaccard(active_gates(h), active_gates(h_trunc))
4. Report per-layer, per-k, averaged over problems

If gates flip massively even at k=500: carrier wave confirmed.
If gates barely change: something else explains BS failure.
"""

import json
import time
import numpy as np
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
K_VALUES = [2, 5, 10, 20, 50, 100, 200, 500]

# Focus on equilibrium layers (L9-L26) plus a few from build/output
PROBE_LAYERS = [0, 4, 8, 9, 12, 16, 20, 24, 26, 30, 34, 35]


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))


def compute_pca_bases(H, max_k=500):
    """Compute centered PCA basis for H (N, d). Returns mean, eigvecs."""
    mu = H.mean(axis=0)  # (d,)
    H_centered = H - mu   # (N, d)
    # Gram approach: N×N is smaller than d×d
    G = H_centered @ H_centered.T  # (N, N)
    eigvals, eigvecs_gram = np.linalg.eigh(G)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs_gram = eigvecs_gram[:, idx]
    # Convert to d-space PCs
    # PC_j = (1/sqrt(lambda_j)) * H_centered.T @ v_j
    k = min(max_k, len(eigvals))
    good = eigvals[:k] > 1e-10
    pcs = np.zeros((k, H.shape[1]))
    for j in range(k):
        if good[j]:
            pcs[j] = H_centered.T @ eigvecs_gram[:, j] / np.sqrt(eigvals[j])
    return mu, pcs, eigvals


def truncate_batch(H, mu, pcs, k):
    """Project all h onto top-k PCs: P_k(H - mu) + mu. Vectorized."""
    Delta = H - mu[None, :]  # (N, d)
    coords = Delta @ pcs[:k].T  # (N, k)
    proj = coords @ pcs[:k]  # (N, d)
    return proj + mu[None, :]  # (N, d)


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP C1b: GATE FLIP ANALYSIS")
    print("How many MLP gates change under PCA truncation?")
    print("=" * 70)

    # Load cached hidden states
    print(f"\nLoading cached hidden states from {CACHE_PATH}...")
    cache = np.load(CACHE_PATH)
    H_all = {}
    for l in range(N_LAYERS):
        arrays = [cache[f"{lang}_L{l}"].astype(np.float32) for lang in LANGS]
        H_all[l] = np.concatenate(arrays, axis=0)
    del cache
    N, d = H_all[0].shape
    print(f"  N={N}, d={d}")

    # Load model weights
    print(f"\nLoading model weights from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype="float32",
        device_map="cpu", trust_remote_code=True
    )
    d_ff = model.config.intermediate_size
    print(f"  d_ff={d_ff}")

    results_by_layer = []

    for l in PROBE_LAYERS:
        t_layer = time.time()
        H = H_all[l]  # (N, d)

        # Get PCA basis
        mu, pcs, eigvals = compute_pca_bases(H, max_k=max(K_VALUES))

        # Get gate weights
        W_gate = model.model.layers[l].mlp.gate_proj.weight.detach().float().numpy()  # (d_ff, d)

        # Full gate patterns for all problems
        gate_full = silu(H @ W_gate.T)  # (N, d_ff)
        gate_full_sign = (gate_full > 0).astype(np.float32)
        gate_full_norms = np.linalg.norm(gate_full, axis=1, keepdims=True)  # (N, 1)
        # Fraction of gates that are "active" (positive) per problem
        active_frac_full = gate_full_sign.mean(axis=1).mean()

        k_results = []
        for k in K_VALUES:
            if k > len(eigvals):
                continue

            # Truncated gate patterns — vectorized
            H_trunc = truncate_batch(H, mu, pcs, k)  # (N, d)
            gate_trunc = silu(H_trunc @ W_gate.T)  # (N, d_ff)

            gate_trunc_sign = (gate_trunc > 0).astype(np.float32)

            # Metrics across all problems
            # 1. Sign flips: fraction of gates that change sign
            sign_flips = (gate_full_sign != gate_trunc_sign).mean(axis=1)  # (N,)
            mean_sign_flips = float(sign_flips.mean())

            # 2. Magnitude change: relative error
            diff_norms = np.linalg.norm(gate_full - gate_trunc, axis=1)  # (N,)
            rel_errors = diff_norms / (gate_full_norms.squeeze() + 1e-10)
            mean_rel_error = float(rel_errors.mean())

            # 3. Correlation of gate patterns
            corrs = []
            for p in range(N):
                if np.std(gate_full[p]) > 1e-10 and np.std(gate_trunc[p]) > 1e-10:
                    corrs.append(float(np.corrcoef(gate_full[p], gate_trunc[p])[0, 1]))
            mean_corr = float(np.mean(corrs)) if corrs else 0.0

            # 4. Active gate Jaccard overlap
            intersection = (gate_full_sign * gate_trunc_sign).sum(axis=1)
            union = ((gate_full_sign + gate_trunc_sign) > 0).sum(axis=1).astype(float)
            jaccard = intersection / (union + 1e-10)
            mean_jaccard = float(jaccard.mean())

            # 5. Variance explained at this k
            var_explained = float(eigvals[:k].sum() / (eigvals.sum() + 1e-10))

            k_result = {
                "k": k,
                "var_explained": var_explained,
                "sign_flip_frac": mean_sign_flips,
                "rel_magnitude_error": mean_rel_error,
                "gate_correlation": mean_corr,
                "jaccard_overlap": mean_jaccard,
                "sign_flip_pct": round(mean_sign_flips * 100, 2),
                "sign_flip_count": round(mean_sign_flips * d_ff),
            }
            k_results.append(k_result)

            print(f"  L{l:2d} k={k:4d}: sign_flip={mean_sign_flips*100:5.1f}% "
                  f"({round(mean_sign_flips * d_ff):5d}/{d_ff}) "
                  f"rel_err={mean_rel_error:.3f} corr={mean_corr:.4f} "
                  f"jaccard={mean_jaccard:.3f} var={var_explained:.4f}")

        layer_result = {
            "layer": l,
            "active_gate_frac": float(active_frac_full),
            "k_results": k_results,
        }
        results_by_layer.append(layer_result)
        dt = time.time() - t_layer
        print(f"  L{l} done in {dt:.1f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: GATE SIGN FLIPS AT k=500 (99.9% variance)")
    print(f"{'L':>3} | {'flips %':>7} | {'count':>6}/{d_ff} | {'corr':>7} | {'jaccard':>7}")
    print("-" * 50)
    for lr in results_by_layer:
        k500 = [kr for kr in lr["k_results"] if kr["k"] == 500]
        if k500:
            kr = k500[0]
            print(f"L{lr['layer']:2d} | {kr['sign_flip_pct']:5.1f}%  | {kr['sign_flip_count']:5d}  | "
                  f"{kr['gate_correlation']:7.4f} | {kr['jaccard_overlap']:7.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY: GATE SIGN FLIPS AT k=20 (Gram rank_90)")
    print(f"{'L':>3} | {'flips %':>7} | {'count':>6}/{d_ff} | {'corr':>7} | {'jaccard':>7}")
    print("-" * 50)
    for lr in results_by_layer:
        k20 = [kr for kr in lr["k_results"] if kr["k"] == 20]
        if k20:
            kr = k20[0]
            print(f"L{lr['layer']:2d} | {kr['sign_flip_pct']:5.1f}%  | {kr['sign_flip_count']:5d}  | "
                  f"{kr['gate_correlation']:7.4f} | {kr['jaccard_overlap']:7.3f}")

    # Save
    output = {
        "experiment": "C1b: Gate Flip Analysis",
        "model": MODEL_NAME,
        "N": N, "d": d, "d_ff": d_ff,
        "k_values": K_VALUES,
        "probe_layers": PROBE_LAYERS,
        "layers": results_by_layer,
        "runtime_seconds": time.time() - t0,
    }
    out_path = OUTPUT_DIR / "expC1b_gate_flip.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
