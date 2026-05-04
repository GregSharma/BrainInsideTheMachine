"""Exp BL: Null-space MLP energy fraction ρ(ℓ) — GPU version.

For each layer's MLP, measure what fraction of computation lives in the
null-space (reasoning) vs complement (language maintenance).

ρ(ℓ) = ||MLP(Π h)||² / ||MLP(h)||²

Uses torch on GPU for speed.
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from pathlib import Path

OUT = Path("output")
device = "cuda"

print("=" * 60)
print("  Exp BL: Null-space MLP energy fraction ρ(ℓ) [GPU]")
print("=" * 60)

# ── 1. Build null-space projections from saved activations ───────────

print("\n[1/3] Building null-space projections...")

lasttok = np.load(OUT / "all_layers_lasttok.npz")
multi = np.load(OUT / "multilingual_all_layers.npz")
categories = lasttok["categories"]

ALL_LANGS = [l for l in ["en", "zh", "es", "ar", "ja", "ko", "sw"] if f"{l}_L32" in multi]

def get_acts_np(lang, layer):
    key = f"{lang}_L{layer}"
    return multi[key] if key in multi else lasttok[key]

N_NULL = 20  # null-space dimensionality

null_projs = {}  # layer → (2048, 2048) projection matrix as torch tensor
for L in range(36):
    all_diffs = []
    for i, la in enumerate(ALL_LANGS):
        for j, lb in enumerate(ALL_LANGS):
            if i >= j: continue
            d = get_acts_np(la, L) - get_acts_np(lb, L)
            all_diffs.append(d)
    all_diffs = np.vstack(all_diffs)
    _, S, Vt = np.linalg.svd(all_diffs, full_matrices=False)

    # Language subspace = top singular vectors; null = bottom N_NULL
    n_lang = len(S) - N_NULL
    lang_basis = Vt[:n_lang]
    Pi = np.eye(2048, dtype=np.float32) - lang_basis.T @ lang_basis
    null_projs[L] = torch.tensor(Pi, device=device, dtype=torch.float16)

    if L % 6 == 0:
        null_frac = S[-N_NULL:].sum() / S.sum()
        print(f"  L{L:>2d}: null-{N_NULL}D energy fraction of diffs = {null_frac:.4f}")

# Prepare activation tensors
H_per_layer = {}
for L in range(36):
    en_h = get_acts_np("en", L)
    zh_h = get_acts_np("zh", L)
    H_per_layer[L] = torch.tensor(
        np.vstack([en_h, zh_h]), device=device, dtype=torch.float16
    )  # (400, 2048)

print(f"  Projections and activations ready on {device}")

# ── 2. Load model and compute ρ ──────────────────────────────────────

print("\n[2/3] Loading Qwen2.5-3B on GPU and computing ρ(ℓ)...")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    torch_dtype=torch.float16,
    device_map=device,
    attn_implementation="eager",
)
model.eval()

results = {}
with torch.no_grad():
    for L in range(36):
        mlp = model.model.layers[L].mlp
        Pi = null_projs[L]  # (2048, 2048)
        H = H_per_layer[L]  # (400, 2048)

        # Full MLP forward
        full_out = mlp(H)  # (400, 2048)
        full_energy = (full_out ** 2).sum(dim=1).mean().item()

        # Null-space input MLP forward
        H_null = H @ Pi.T  # (400, 2048) — projected into null-space
        null_out = mlp(H_null)  # (400, 2048)
        null_energy = (null_out ** 2).sum(dim=1).mean().item()

        # How much of FULL output lives in null-space?
        full_out_proj = full_out @ Pi.T
        out_null_energy = (full_out_proj ** 2).sum(dim=1).mean().item()

        rho_input = null_energy / (full_energy + 1e-10)
        rho_output = out_null_energy / (full_energy + 1e-10)

        # Effective rank of the projected MLP (linear approximation)
        W_gate = mlp.gate_proj.weight.float()  # (intermediate, 2048)
        W_up = mlp.up_proj.weight.float()
        W_down = mlp.down_proj.weight.float()  # (2048, intermediate)
        Pi_f = Pi.float()

        W_eff = W_down @ W_up @ Pi_f.T  # (2048, 2048)
        sv = torch.linalg.svdvals(W_eff).cpu().numpy()
        cum_energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
        rank_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
        rank_99 = int(np.searchsorted(cum_energy, 0.99)) + 1

        results[L] = {
            "rho_input": float(rho_input),
            "rho_output": float(rho_output),
            "full_energy": float(full_energy),
            "null_energy": float(null_energy),
            "rank_90": rank_90,
            "rank_99": rank_99,
            "top_10_svals": [float(s) for s in sv[:10]],
        }

        print(f"  L{L:>2d}: ρ_in={rho_input:.4f}  ρ_out={rho_output:.4f}  "
              f"rank90={rank_90}  rank99={rank_99}")

# ── 3. Layer delta analysis ──────────────────────────────────────────

print("\n[3/3] Layer delta null-space fraction...")

delta_results = {}
for L in range(1, 36):
    Pi = null_projs[L]
    delta = H_per_layer[L] - H_per_layer[L-1]  # (400, 2048)
    delta_null = delta @ Pi.T

    total_e = (delta ** 2).sum(dim=1).mean().item()
    null_e = (delta_null ** 2).sum(dim=1).mean().item()
    rho_delta = null_e / (total_e + 1e-10)
    delta_results[L] = float(rho_delta)

# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY: ρ(ℓ) — Null-space MLP Energy Fraction")
print("=" * 60)

print(f"\n  {'Layer':<8s} {'ρ_input':>10s} {'ρ_output':>10s} {'ρ_delta':>10s} "
      f"{'rank90':>8s} {'rank99':>8s}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

for L in range(36):
    r = results[L]
    rd = delta_results.get(L, float("nan"))
    print(f"  L{L:<6d} {r['rho_input']:>10.4f} {r['rho_output']:>10.4f} {rd:>10.4f} "
          f"{r['rank_90']:>8d} {r['rank_99']:>8d}")

mid_rho_in = np.mean([results[L]["rho_input"] for L in range(9, 27)])
mid_rho_out = np.mean([results[L]["rho_output"] for L in range(9, 27)])
mid_rank90 = np.mean([results[L]["rank_90"] for L in range(9, 27)])

print(f"\n  Middle layers (L9-L26) averages:")
print(f"    ρ_input  = {mid_rho_in:.4f}")
print(f"    ρ_output = {mid_rho_out:.4f}")
print(f"    rank_90  = {mid_rank90:.1f}")

if mid_rho_in > 0.8:
    verdict = "REASONING DOMINATES — massive compression possible"
elif mid_rho_in > 0.4:
    verdict = "MIXED — moderate compression, factored architecture helps"
else:
    verdict = "LANGUAGE DOMINATES — MLP-level factorization insufficient"
print(f"\n  VERDICT: {verdict}")

# Save
output = {
    "experiment": "BL",
    "title": "Null-space MLP energy fraction rho(ell)",
    "mlp_results": {str(k): v for k, v in results.items()},
    "delta_results": {str(k): v for k, v in delta_results.items()},
    "middle_layer_avg": {
        "rho_input": float(mid_rho_in),
        "rho_output": float(mid_rho_out),
        "rank_90": float(mid_rank90),
    },
    "verdict": verdict,
}

with open(OUT / "expBL_nullspace_mlp_energy.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBL_nullspace_mlp_energy.json")
del model
torch.cuda.empty_cache()
