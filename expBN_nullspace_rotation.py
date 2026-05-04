"""Exp BN: Null-space rotation tracking — the moving frame.

BM showed: null-space is layer-specific (alignment ~0.10 between layers).
97% retrieval at every layer, but in DIFFERENT 20D subspaces each time.

This experiment tracks HOW the null-space rotates:
  (A) Adjacent-layer Procrustes: rotation between null_L and null_{L+1}
  (B) Cumulative drift: how far has null-space drifted from L0?
  (C) Rotation velocity profile: does it correlate with adversarial/cooperative phases?
  (D) Rotation predictability: is the rotation smooth (predictable) or chaotic?
"""

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

OUT = Path("output")
N_LAYERS = 36
DIM = 2048
N_NULL = 20

t0 = time.time()

print("=" * 60)
print("  Exp BN: Null-Space Rotation Tracking")
print("=" * 60)

# ── 1. Preload activations ──────────────────────────────────────

print("\n[1/4] Preloading activations...")
t1 = time.time()
multi = np.load(OUT / "multilingual_all_layers.npz")
ALL_LANGS = sorted(set(k.split("_L")[0] for k in multi.files if "_L" in k))
N_PROBLEMS = multi["en_L0"].shape[0]

H = {}
for lang in tqdm(ALL_LANGS, desc="  Loading"):
    H[lang] = {}
    for L in range(N_LAYERS):
        H[lang][L] = multi[f"{lang}_L{L}"].astype(np.float32)
del multi
print(f"  Done in {time.time()-t1:.1f}s")


# ── 2. Build null-space bases ───────────────────────────────────

print("\n[2/4] Building null-space bases (gram eigh)...")

null_bases = {}  # L → (N_NULL, DIM) orthonormal
for L in tqdm(range(N_LAYERS), desc="  Eigh"):
    diffs = []
    for i, la in enumerate(ALL_LANGS):
        for j, lb in enumerate(ALL_LANGS):
            if i >= j:
                continue
            diffs.append(H[la][L] - H[lb][L])
    diffs = np.vstack(diffs)
    gram = diffs.T @ diffs
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    # eigh: ascending. Bottom N_NULL = null-space (smallest eigenvalues)
    null_bases[L] = eigenvectors[:, :N_NULL].T.copy()  # (N_NULL, DIM)

print(f"  Bases computed in {time.time()-t1:.1f}s")


# ── 3. (A) Procrustes rotation between adjacent layers ─────────

print("\n[3/4] Computing rotations...")

def procrustes_analysis(B1, B2):
    """Procrustes rotation from subspace B1 to B2.
    Returns: rotation matrix R (N_NULL x N_NULL), principal angles, mean angle."""
    # B1, B2: (N_NULL, DIM) orthonormal bases
    # Cross-correlation in subspace coordinates
    M = B2 @ B1.T  # (N_NULL, N_NULL)
    U, S, Vt = np.linalg.svd(M)
    # Optimal rotation: R = U @ Vt
    R = U @ Vt
    # Principal angles: arccos of singular values (clamped)
    S_clamped = np.clip(S, -1.0, 1.0)
    angles = np.arccos(S_clamped)  # radians
    # Rotation magnitude: Frobenius angle = sqrt(sum(angles^2))
    frob_angle = np.sqrt(np.sum(angles ** 2))
    return {
        "R": R,
        "singular_values": S,
        "principal_angles_deg": np.degrees(angles),
        "mean_angle_deg": float(np.degrees(np.mean(angles))),
        "max_angle_deg": float(np.degrees(np.max(angles))),
        "frob_angle_deg": float(np.degrees(frob_angle)),
        "mean_cos": float(np.mean(S_clamped)),
    }


# (A) Adjacent-layer rotations
print("\n  (A) Adjacent-layer Procrustes rotations:")
adjacent = {}
for L in tqdm(range(N_LAYERS - 1), desc="  Adjacent Procrustes"):
    p = procrustes_analysis(null_bases[L], null_bases[L + 1])
    adjacent[L] = {k: v for k, v in p.items() if k != "R"}
    # Store R for later
    adjacent[L]["_R"] = p["R"]

print(f"\n  {'L→L+1':<8s} {'MeanAngle':>10s} {'MaxAngle':>10s} {'FrobAngle':>10s} {'MeanCos':>10s}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for L in range(N_LAYERS - 1):
    a = adjacent[L]
    print(f"  L{L}→{L+1:<4d} {a['mean_angle_deg']:>10.2f}° {a['max_angle_deg']:>10.2f}° "
          f"{a['frob_angle_deg']:>10.2f}° {a['mean_cos']:>10.4f}")


# (B) Cumulative drift from L0
print("\n\n  (B) Cumulative drift from L0:")
cumulative = {}
for L in tqdm(range(N_LAYERS), desc="  Cumulative drift"):
    p = procrustes_analysis(null_bases[0], null_bases[L])
    cumulative[L] = {k: v for k, v in p.items() if k != "R"}

print(f"\n  {'Layer':<8s} {'MeanAngle':>10s} {'MaxAngle':>10s} {'FrobAngle':>10s}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for L in range(N_LAYERS):
    c = cumulative[L]
    print(f"  L{L:<6d} {c['mean_angle_deg']:>10.2f}° {c['max_angle_deg']:>10.2f}° "
          f"{c['frob_angle_deg']:>10.2f}°")


# (C) Rotation velocity profile
print("\n\n  (C) Rotation velocity (mean angle per layer step):")
velocities = [adjacent[L]["mean_angle_deg"] for L in range(N_LAYERS - 1)]

phases = {
    "early (L0-L8)":   range(0, 8),
    "advers (L9-L17)":  range(8, 17),
    "coop (L18-L26)":   range(17, 26),
    "late (L27-L35)":   range(26, 35),
}

print(f"\n  {'Phase':<20s} {'MeanVelocity':>14s} {'MinVel':>10s} {'MaxVel':>10s}")
print(f"  {'-'*20} {'-'*14} {'-'*10} {'-'*10}")
for name, rng in phases.items():
    vs = [velocities[i] for i in rng]
    print(f"  {name:<20s} {np.mean(vs):>13.2f}° {np.min(vs):>9.2f}° {np.max(vs):>9.2f}°")


# (D) Rotation predictability: does R_L predict R_{L+1}?
print("\n\n  (D) Rotation predictability (does R_L ≈ R_{L+1}?):")
rotation_consistency = {}
for L in range(N_LAYERS - 2):
    R_curr = adjacent[L]["_R"]
    R_next = adjacent[L + 1]["_R"]
    # Similarity: Frobenius inner product of rotation matrices
    # cos(angle between rotations) = trace(R1^T R2) / N_NULL
    cos_sim = float(np.trace(R_curr.T @ R_next) / N_NULL)
    rotation_consistency[L] = cos_sim

print(f"\n  {'L→L+1→L+2':<12s} {'cos(R_L, R_L+1)':>16s}")
print(f"  {'-'*12} {'-'*16}")
for L in range(N_LAYERS - 2):
    c = rotation_consistency[L]
    bar = "+" * max(0, int((c + 1) * 20))
    print(f"  L{L}→{L+1}→{L+2:<4d} {c:>16.4f}  {bar}")

avg_consistency = np.mean(list(rotation_consistency.values()))
print(f"\n  Average rotation consistency: {avg_consistency:.4f}")
if avg_consistency > 0.5:
    verdict_pred = "SMOOTH: rotations are predictable (fiber bundle structure)"
elif avg_consistency > 0.0:
    verdict_pred = "PARTIALLY SMOOTH: some structure but not fully predictable"
else:
    verdict_pred = "CHAOTIC: each layer rotates independently"


# ── 4. Summary ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  ROTATION SUMMARY")
print("=" * 60)

avg_adjacent = np.mean(velocities)
max_drift = cumulative[N_LAYERS - 1]["frob_angle_deg"]

print(f"\n  Average rotation per layer step: {avg_adjacent:.2f}°")
print(f"  Total drift L0→L35:              {max_drift:.2f}°")
print(f"  Rotation predictability:          {avg_consistency:.4f}")
print(f"\n  VERDICT (predictability): {verdict_pred}")

# Check if phase boundaries show acceleration
adv_start_vel = velocities[8]  # L8→L9
coop_start_vel = velocities[17]  # L17→L18
late_start_vel = velocities[26]  # L26→L27
print(f"\n  Phase boundary velocities:")
print(f"    L8→L9 (adversarial onset):  {adv_start_vel:.2f}°")
print(f"    L17→L18 (cooperative onset): {coop_start_vel:.2f}°")
print(f"    L26→L27 (late onset):        {late_start_vel:.2f}°")

elapsed = time.time() - t0
print(f"\n  Total runtime: {elapsed:.1f}s")


# ── Save ─────────────────────────────────────────────────────────

# Clean up non-serializable _R matrices
for L in adjacent:
    adj_clean = {k: v for k, v in adjacent[L].items() if k != "_R"}
    # Convert numpy arrays to lists
    if "singular_values" in adj_clean:
        adj_clean["singular_values"] = adj_clean["singular_values"].tolist()
    if "principal_angles_deg" in adj_clean:
        adj_clean["principal_angles_deg"] = adj_clean["principal_angles_deg"].tolist()
    adjacent[L] = adj_clean

for L in cumulative:
    if "singular_values" in cumulative[L]:
        cumulative[L]["singular_values"] = cumulative[L]["singular_values"].tolist()
    if "principal_angles_deg" in cumulative[L]:
        cumulative[L]["principal_angles_deg"] = cumulative[L]["principal_angles_deg"].tolist()

output = {
    "experiment": "BN",
    "title": "Null-space rotation tracking — the moving frame",
    "n_null": N_NULL,
    "runtime_seconds": round(elapsed, 1),
    "adjacent_rotations": {str(k): v for k, v in adjacent.items()},
    "cumulative_drift": {str(k): v for k, v in cumulative.items()},
    "rotation_consistency": {str(k): v for k, v in rotation_consistency.items()},
    "phase_velocities": {name: {
        "mean": float(np.mean([velocities[i] for i in rng])),
        "min": float(np.min([velocities[i] for i in rng])),
        "max": float(np.max([velocities[i] for i in rng])),
    } for name, rng in phases.items()},
    "avg_rotation_per_step": float(avg_adjacent),
    "total_drift_deg": float(max_drift),
    "avg_consistency": float(avg_consistency),
    "verdict_predictability": verdict_pred,
}

with open(OUT / "expBN_nullspace_rotation.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBN_nullspace_rotation.json")
