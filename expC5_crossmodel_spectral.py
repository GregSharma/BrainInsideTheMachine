"""
Exp C5: Cross-model spectral concentration sweep.
Computes σ₁(W_U · OV_ℓ) for all layers across multiple models.
Tests ansatz: ℓ* = argmax_ℓ σ₁(W_U · OV_ℓ) predicts the empirical readout layer.

Method:
  For composed map W_U @ OV_ℓ (shape |V| × d), singular values satisfy:
    σ_i² = eigenvalues of OV_ℓ^T @ (W_U^T @ W_U) @ OV_ℓ
  Precompute G_U = W_U^T @ W_U once (d × d), then per-layer:
    M_ℓ = OV_ℓ^T @ G_U @ OV_ℓ  (d × d)
    eigendecompose M_ℓ → σ_i = sqrt(λ_i)
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open

MODELS = {
    "Qwen2.5-3B": {
        "path": Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots",
        "n_layers": 36,
        "d": 2048,
        "n_q": 16,
        "n_kv": 2,
        "d_h": 128,
        "tied": True,
    },
    "Qwen2.5-7B": {
        "path": Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots",
        "n_layers": 28,
        "d": 3584,
        "n_q": 28,
        "n_kv": 4,
        "d_h": 128,
        "tied": True,  # technically has lm_head but it's tied
    },
    "Qwen2.5-14B": {
        "path": Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots",
        "n_layers": 48,
        "d": 5120,
        "n_q": 40,
        "n_kv": 8,
        "d_h": 128,
        "tied": False,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def get_snapshot_dir(model_path):
    """Get the single snapshot directory."""
    snapshots = list(model_path.glob("*"))
    snapshots = [s for s in snapshots if s.is_dir() and not s.name.startswith(".")]
    assert len(snapshots) == 1, f"Expected 1 snapshot, got {len(snapshots)}: {snapshots}"
    return snapshots[0]


def load_tensor(snapshot_dir, key):
    """Load a specific tensor from sharded safetensors."""
    index_file = snapshot_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    shard = index["weight_map"][key]
    shard_path = snapshot_dir / shard
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def compute_gram_U(snapshot_dir, d):
    """Precompute G_U = W_U^T @ W_U (d × d)."""
    # Use lm_head.weight if available, else fall back to embed_tokens (tied)
    index_file = snapshot_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    if "lm_head.weight" in index["weight_map"]:
        W_U = load_tensor(snapshot_dir, "lm_head.weight")
    else:
        W_U = load_tensor(snapshot_dir, "model.embed_tokens.weight")
    print(f"  W_U shape: {W_U.shape}, dtype: {W_U.dtype}")
    # Compute in float32 for stability
    W_U = W_U.float().to(device)
    G_U = W_U.T @ W_U  # (d, d)
    del W_U
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return G_U


def compute_OV(snapshot_dir, layer_idx, n_q, n_kv, d_h):
    """Compute OV_ℓ = O_proj @ V_expanded for a single layer."""
    V = load_tensor(snapshot_dir, f"model.layers.{layer_idx}.self_attn.v_proj.weight")
    O = load_tensor(snapshot_dir, f"model.layers.{layer_idx}.self_attn.o_proj.weight")
    # V shape: (n_kv * d_h, d), O shape: (d, n_q * d_h)
    # Expand V for GQA: repeat each KV head G = n_q/n_kv times
    G = n_q // n_kv
    # V is (n_kv * d_h, d). Reshape to (n_kv, d_h, d), repeat to (n_q, d_h, d), reshape to (n_q * d_h, d)
    V = V.float()
    V_reshaped = V.view(n_kv, d_h, -1)  # (n_kv, d_h, d)
    V_expanded = V_reshaped.repeat_interleave(G, dim=0)  # (n_q, d_h, d)
    V_expanded = V_expanded.reshape(n_q * d_h, -1)  # (n_q * d_h, d)
    O = O.float()  # (d, n_q * d_h)
    OV = O @ V_expanded  # (d, d)
    return OV.to(device)


def spectral_analysis(G_U, OV):
    """Compute spectral metrics of W_U @ OV from precomputed G_U and OV."""
    # M = OV^T @ G_U @ OV (d × d, symmetric positive semi-definite)
    M = OV.T @ G_U @ OV
    # Eigendecompose (symmetric → use eigh for stability)
    eigenvalues = torch.linalg.eigvalsh(M)  # ascending order
    eigenvalues = eigenvalues.flip(0)  # descending
    # Clamp negative eigenvalues (numerical noise)
    eigenvalues = eigenvalues.clamp(min=0)
    sigmas = eigenvalues.sqrt()
    # Metrics
    sigma_1 = sigmas[0].item()
    sigma_2 = sigmas[1].item() if len(sigmas) > 1 else 1e-10
    ratio = sigma_1 / max(sigma_2, 1e-10)
    total_var = eigenvalues.sum().item()
    top1_var = eigenvalues[0].item() / max(total_var, 1e-10)
    # r@90%
    cumvar = eigenvalues.cumsum(0) / max(total_var, 1e-10)
    r90 = (cumvar < 0.9).sum().item() + 1
    return {
        "sigma_1": sigma_1,
        "sigma_2": sigma_2,
        "ratio": ratio,
        "top1_var_frac": top1_var,
        "r90": r90,
    }


def sweep_model(model_name, config):
    """Full spectral sweep for one model."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"  Layers: {config['n_layers']}, d: {config['d']}, n_q: {config['n_q']}, n_kv: {config['n_kv']}")
    print(f"  Tied: {config['tied']}")
    print(f"{'='*60}")

    snapshot_dir = get_snapshot_dir(config["path"])
    print(f"  Snapshot: {snapshot_dir.name}")

    # Precompute G_U = W_U^T @ W_U
    print("  Computing G_U = W_U^T @ W_U ...")
    t0 = time.time()
    G_U = compute_gram_U(snapshot_dir, config["d"])
    print(f"  G_U computed in {time.time()-t0:.1f}s, shape: {G_U.shape}")

    results = []
    for ell in range(config["n_layers"]):
        t0 = time.time()
        OV = compute_OV(snapshot_dir, ell, config["n_q"], config["n_kv"], config["d_h"])
        metrics = spectral_analysis(G_U, OV)
        elapsed = time.time() - t0
        metrics["layer"] = ell
        results.append(metrics)
        if ell % 5 == 0 or ell == config["n_layers"] - 1:
            print(f"  L{ell:2d}: σ₁={metrics['sigma_1']:.2f}, ratio={metrics['ratio']:.2f}, "
                  f"top1={metrics['top1_var_frac']:.3f}, r90={metrics['r90']}, ({elapsed:.1f}s)")
        del OV
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Find argmax
    best = max(results, key=lambda x: x["sigma_1"])
    best_ratio = max(results, key=lambda x: x["ratio"])
    print(f"\n  PREDICTION: ℓ* = L{best['layer']} (by σ₁={best['sigma_1']:.2f})")
    print(f"  Alt:        ℓ* = L{best_ratio['layer']} (by ratio={best_ratio['ratio']:.2f})")

    del G_U
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return results


if __name__ == "__main__":
    all_results = {}
    for model_name, config in MODELS.items():
        results = sweep_model(model_name, config)
        all_results[model_name] = results

    # Save
    output_path = Path("output/expC5_crossmodel_spectral.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Predicted readout layers (ℓ* = argmax σ₁(W_U · OV_ℓ))")
    print("="*70)
    for model_name, results in all_results.items():
        best = max(results, key=lambda x: x["sigma_1"])
        n_layers = len(results)
        depth_frac = best["layer"] / (n_layers - 1)
        print(f"  {model_name:20s}: ℓ* = L{best['layer']:2d}/{n_layers} "
              f"(depth {depth_frac:.1%}), σ₁={best['sigma_1']:.2f}, "
              f"ratio={best['ratio']:.2f}, top1={best['top1_var_frac']:.3f}")
