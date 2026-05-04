"""Exp BM: Residual stream null-space accumulation profile + layer pruning.

Key question: Individual MLPs contribute ρ≈0 to null-space (BL), yet the
cumulative residual stream at L32 gives 97% cross-lingual retrieval (BH/BI).
How does null-space content accumulate across layers?

Measurements:
  (A) Fixed-projector accumulation: fraction of h_L in L32's null-space
  (B) Delta contribution: each layer's Δh projected into L32 null-space
  (C) Cross-layer null-space alignment: cosine between null-space bases
  (D) Layer-skip retrieval: block-skip layers, measure retrieval survival
"""

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

OUT = Path("output")
N_LAYERS = 36
DIM = 2048
N_NULL = 20  # bottom 20 SVD dims = null-space (matches BL)
REF_LAYER = 32

t0 = time.time()

print("=" * 60)
print("  Exp BM: Null-Space Accumulation Profile")
print("=" * 60)

# ── 1. Preload ALL activations into RAM ────────────────────────────

print("\n[1/5] Preloading all activations into RAM (eliminates repeated npz decompression)...")

t1 = time.time()
multi = np.load(OUT / "multilingual_all_layers.npz")
ALL_LANGS = sorted(set(k.split("_L")[0] for k in multi.files if "_L" in k))
N_PROBLEMS = multi[f"en_L0"].shape[0]

print(f"  Languages: {ALL_LANGS}")
print(f"  {N_LAYERS} layers, {N_PROBLEMS} problems, d={DIM}")

# Preload into dict[lang][layer] = np.float32 array
H = {}
for lang in tqdm(ALL_LANGS, desc="  Loading languages"):
    H[lang] = {}
    for L in range(N_LAYERS):
        H[lang][L] = multi[f"{lang}_L{L}"].astype(np.float32)

del multi  # free the npz handle
print(f"  Preloaded in {time.time()-t1:.1f}s. All data in RAM.")


# ── 2. Build null-space projectors per layer ─────────────────────

print("\n[2/5] Building null-space projectors (36 SVDs)...")

null_projs = {}      # layer → (DIM, DIM) projection matrix
null_bases = {}      # layer → (N_NULL, DIM) orthonormal basis of null-space

for L in tqdm(range(N_LAYERS), desc="  Gram eigh per layer"):
    all_diffs = []
    for i, la in enumerate(ALL_LANGS):
        for j, lb in enumerate(ALL_LANGS):
            if i >= j:
                continue
            all_diffs.append(H[la][L] - H[lb][L])
    all_diffs = np.vstack(all_diffs)  # (21*200, 2048)

    # Gram matrix approach: A^T A = V S^2 V^T, eigendecompose (2048,2048) instead of SVD on (4200,2048)
    gram = all_diffs.T @ all_diffs  # (2048, 2048)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)  # ascending order
    # eigh returns ascending eigenvalues: smallest first = null-space first
    # Null-space = bottom N_NULL eigenvectors (smallest eigenvalues)
    null_bases[L] = eigenvectors[:, :N_NULL].T.copy()  # (N_NULL, DIM)

    # Language subspace = top eigenvectors
    V_lang = eigenvectors[:, N_NULL:].T  # (n_lang, DIM)
    Pi = np.eye(DIM, dtype=np.float32) - V_lang.T @ V_lang
    null_projs[L] = Pi

Pi_ref = null_projs[REF_LAYER]
print(f"  Reference projector: L{REF_LAYER}")
print(f"  SVDs done in {time.time()-t1:.1f}s total.")


# ── 3. (A) Fixed-projector accumulation ──────────────────────────

print("\n[3/5] (A) Fraction of h_L in L32 null-space at each layer...")

accumulation = {}
for L in tqdm(range(N_LAYERS), desc="  Accumulation"):
    # Stack all 7 languages
    H_all = np.vstack([H[la][L] for la in ALL_LANGS])  # (1400, DIM)
    H_null = H_all @ Pi_ref.T

    total_energy = np.mean(np.sum(H_all ** 2, axis=1))
    null_energy = np.mean(np.sum(H_null ** 2, axis=1))
    frac = null_energy / (total_energy + 1e-10)

    accumulation[L] = {
        "null_fraction": float(frac),
        "total_energy": float(total_energy),
        "null_energy": float(null_energy),
    }

print(f"\n  {'Layer':<6s} {'NullFrac':>10s}  {'||h||²':>10s}  {'||Πh||²':>10s}")
print(f"  {'-'*6} {'-'*10}  {'-'*10}  {'-'*10}")
for L in range(N_LAYERS):
    a = accumulation[L]
    print(f"  L{L:<4d} {a['null_fraction']:>10.4f}  "
          f"{a['total_energy']:>10.0f}  {a['null_energy']:>10.1f}")


# ── 4. (B) Delta contribution to L32 null-space ─────────────────

print("\n[4/5] (B) Layer delta → L32 null-space contribution...")

delta_contribution = {}
for L in tqdm(range(1, N_LAYERS), desc="  Delta analysis"):
    H_curr = np.vstack([H[la][L] for la in ALL_LANGS])
    H_prev = np.vstack([H[la][L-1] for la in ALL_LANGS])
    delta = H_curr - H_prev

    delta_null = delta @ Pi_ref.T
    total_delta_e = np.mean(np.sum(delta ** 2, axis=1))
    null_delta_e = np.mean(np.sum(delta_null ** 2, axis=1))
    frac = null_delta_e / (total_delta_e + 1e-10)

    marginal = accumulation[L]["null_energy"] - accumulation[L-1]["null_energy"]

    delta_contribution[L] = {
        "delta_null_fraction": float(frac),
        "delta_null_energy": float(null_delta_e),
        "delta_total_energy": float(total_delta_e),
        "marginal_null_energy": float(marginal),
    }

print(f"\n  {'Layer':<6s} {'ΔNullFrac':>10s}  {'Δ_null':>10s}  {'Δ_total':>10s}  {'Marginal':>10s}")
print(f"  {'-'*6} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for L in range(1, N_LAYERS):
    d = delta_contribution[L]
    sign = "+" if d["marginal_null_energy"] >= 0 else ""
    print(f"  L{L:<4d} {d['delta_null_fraction']:>10.4f}  "
          f"{d['delta_null_energy']:>10.1f}  {d['delta_total_energy']:>10.0f}  "
          f"{sign}{d['marginal_null_energy']:>9.1f}")


# ── 5. (C) Cross-layer null-space alignment ──────────────────────

print("\n[5/5] (C) Null-space alignment with L32 + pairwise...")

def subspace_alignment(B1, B2):
    """Mean cos(principal angles) between two subspaces. 1.0=identical, 0.0=orthogonal."""
    M = B1 @ B2.T
    sv = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(np.minimum(sv, 1.0)))

alignment_to_ref = {}
for L in tqdm(range(N_LAYERS), desc="  Alignment vs L32"):
    alignment_to_ref[L] = subspace_alignment(null_bases[L], null_bases[REF_LAYER])

print(f"\n  Alignment of each layer's null-space with L{REF_LAYER}:")
for L in range(N_LAYERS):
    bar = "#" * int(alignment_to_ref[L] * 50)
    print(f"  L{L:>2d}: {alignment_to_ref[L]:.4f}  {bar}")

# Pairwise (sampled)
sample_layers = sorted(set(list(range(0, N_LAYERS, 4)) + [REF_LAYER, N_LAYERS-1]))
pairwise = {}
for i, L1 in enumerate(sample_layers):
    for L2 in sample_layers:
        if L1 <= L2:
            pairwise[f"{L1}-{L2}"] = subspace_alignment(null_bases[L1], null_bases[L2])


# ── 6. (D) Retrieval at each layer with own vs fixed projector ───

print("\n\n  (D) Cross-lingual retrieval: own projector vs L32 fixed...")

def retrieval_top1(H_dict, layer, projector):
    """Top-1 cross-lingual retrieval using null-space projection."""
    en_proj = H_dict["en"][layer] @ projector.T
    en_norm = en_proj / (np.linalg.norm(en_proj, axis=1, keepdims=True) + 1e-10)
    results = {}
    for lang in ALL_LANGS:
        if lang == "en":
            continue
        ot_proj = H_dict[lang][layer] @ projector.T
        ot_norm = ot_proj / (np.linalg.norm(ot_proj, axis=1, keepdims=True) + 1e-10)
        sim = en_norm @ ot_norm.T
        results[lang] = float(np.mean(np.argmax(sim, axis=1) == np.arange(N_PROBLEMS)))
    return results

retrieval_own = {}
retrieval_fixed = {}
for L in tqdm(range(0, N_LAYERS, 2), desc="  Retrieval tests"):
    ret_own = retrieval_top1(H, L, null_projs[L])
    ret_fix = retrieval_top1(H, L, Pi_ref)
    retrieval_own[L] = {"per_lang": ret_own, "avg": float(np.mean(list(ret_own.values())))}
    retrieval_fixed[L] = {"per_lang": ret_fix, "avg": float(np.mean(list(ret_fix.values())))}

print(f"\n  {'Layer':<6s} {'OwnProj':>10s} {'L32Proj':>10s} {'Align':>8s}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
for L in range(0, N_LAYERS, 2):
    print(f"  L{L:<4d} {retrieval_own[L]['avg']:>10.3f} "
          f"{retrieval_fixed[L]['avg']:>10.3f} {alignment_to_ref[L]:>8.4f}")


# ── 7. Summary ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  FULL ACCUMULATION PROFILE")
print("=" * 60)

print(f"\n  {'Layer':<6s} {'NullFrac':>10s} {'ΔNullFrac':>10s} {'Marginal':>10s} {'Align32':>10s}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for L in range(N_LAYERS):
    nf = accumulation[L]["null_fraction"]
    dnf = delta_contribution.get(L, {}).get("delta_null_fraction", float("nan"))
    marg = delta_contribution.get(L, {}).get("marginal_null_energy", float("nan"))
    ali = alignment_to_ref[L]
    sign = "+" if marg >= 0 else ""
    print(f"  L{L:<4d} {nf:>10.4f} {dnf:>10.4f} {sign}{marg:>9.1f} {ali:>10.4f}")

# Phase averages
phases = {
    "early (L0-L8)":   range(0, 9),
    "advers (L9-L17)":  range(9, 18),
    "coop (L18-L26)":   range(18, 27),
    "late (L27-L35)":   range(27, 36),
}

print(f"\n  Phase averages:")
print(f"  {'Phase':<20s} {'NullFrac':>10s} {'Align':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10}")
for name, rng in phases.items():
    avg_nf = np.mean([accumulation[L]["null_fraction"] for L in rng])
    avg_al = np.mean([alignment_to_ref[L] for L in rng])
    print(f"  {name:<20s} {avg_nf:>10.4f} {avg_al:>10.4f}")

# Verdicts
late_frac = np.mean([accumulation[L]["null_fraction"] for L in range(27, 36)])
early_align = np.mean([alignment_to_ref[L] for L in range(0, 9)])

if late_frac > 0.5:
    v1 = "STRONG: null-space dominates late residual stream"
elif late_frac > 0.1:
    v1 = "MODERATE: null-space significant but not dominant"
else:
    v1 = "WEAK: null-space is small fraction of residual stream"

if early_align > 0.8:
    v2 = "STABLE: null-space basis consistent across layers"
elif early_align > 0.5:
    v2 = "ROTATING: null-space exists throughout but rotates"
else:
    v2 = "UNSTABLE: early null-space differs from L32"

print(f"\n  VERDICT (accumulation): {v1}")
print(f"  VERDICT (alignment):    {v2}")

elapsed = time.time() - t0
print(f"\n  Total runtime: {elapsed:.1f}s")


# ── Save ─────────────────────────────────────────────────────────

output = {
    "experiment": "BM",
    "title": "Residual stream null-space accumulation profile",
    "ref_layer": REF_LAYER,
    "n_null": N_NULL,
    "runtime_seconds": round(elapsed, 1),
    "accumulation": {str(k): v for k, v in accumulation.items()},
    "delta_contribution": {str(k): v for k, v in delta_contribution.items()},
    "alignment_to_ref": {str(k): v for k, v in alignment_to_ref.items()},
    "pairwise_alignment": pairwise,
    "retrieval_own_proj": {str(k): v for k, v in retrieval_own.items()},
    "retrieval_fixed_proj": {str(k): v for k, v in retrieval_fixed.items()},
    "phase_averages": {
        "null_fraction": {n: float(np.mean([accumulation[L]["null_fraction"] for L in r]))
                         for n, r in phases.items()},
        "alignment": {n: float(np.mean([alignment_to_ref[L] for L in r]))
                     for n, r in phases.items()},
    },
    "verdict_accumulation": v1,
    "verdict_alignment": v2,
}

with open(OUT / "expBM_nullspace_accumulation.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBM_nullspace_accumulation.json")
