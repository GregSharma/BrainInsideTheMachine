"""Experiment E: Convergence Decomposition
Decompose the Exp E propagation convergence into attention vs MLP contributions.
Uses the attn/MLP deltas data.
"""

import json
import numpy as np

# Load Exp E data
with open("output/expE_propagation_tracking.json") as f:
    expE = json.load(f)

# Load attn/MLP deltas
print("Loading attn/MLP deltas...")
deltas = np.load("output/attn_mlp_deltas.npz")
# Keys should be like attn_delta_L{i}, mlp_delta_L{i} for each layer

# Check available keys
delta_keys = sorted(deltas.files)
print(f"Available keys: {delta_keys[:10]}...")

# Get attn and mlp delta keys
attn_keys = sorted([k for k in delta_keys if k.startswith("attn_delta")])
mlp_keys = sorted([k for k in delta_keys if k.startswith("mlp_delta")])
print(f"Attn delta layers: {len(attn_keys)}, MLP delta layers: {len(mlp_keys)}")

if attn_keys:
    sample = deltas[attn_keys[0]]
    print(f"Sample shape: {sample.shape}")  # Should be (200, 2, 2048) or similar

# Compute PC0 at each layer
# We need hidden states to compute layer-specific PC0
data = np.load("output/all_layers_lasttok.npz")

# Focus on swap@L12 convergence (most data points)
swap_layer = 12

# For each layer L from 13 to 35:
# The flip_fraction changes. How much came from attention vs MLP?
#
# At each layer: h_L = h_{L-1} + attn_delta_L + mlp_delta_L
# The PC0 projection changes at each substep:
# 1. After attention: pc0_proj(h_{L-1} + attn_delta_L)
# 2. After MLP: pc0_proj(h_{L-1} + attn_delta_L + mlp_delta_L)
#
# But we need PC0 at each layer. The Exp E tracking uses the SAME PC0 (computed from
# that layer's natural distribution). So we need layer-specific PC0.

results = {"swap_layer": swap_layer, "per_layer": []}

for L in range(13, 36):
    # Compute PC0 at this layer
    zh_L = data[f"zh_L{L}"]  # (200, 2048)
    en_L = data[f"en_L{L}"]  # (200, 2048)
    all_L = np.concatenate([zh_L, en_L], axis=0)
    mean_L = all_L.mean(axis=0)
    centered = all_L - mean_L
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc0 = Vt[0]

    # Natural PC0 projections
    zh_proj = ((zh_L - mean_L) @ pc0).mean()
    en_proj = ((en_L - mean_L) @ pc0).mean()
    gap = en_proj - zh_proj

    # Get attn and MLP deltas at this layer
    zh_attn_key = f"zh_attn_delta_L{L}"
    en_attn_key = f"en_attn_delta_L{L}"
    zh_mlp_key = f"zh_mlp_delta_L{L}"
    en_mlp_key = f"en_mlp_delta_L{L}"

    if zh_attn_key not in deltas.files:
        print(f"  L{L}: missing delta data, skipping")
        continue

    attn_zh = deltas[zh_attn_key]  # (200, 2048)
    attn_en = deltas[en_attn_key]
    mlp_zh = deltas[zh_mlp_key]
    mlp_en = deltas[en_mlp_key]
    print(f"  L{L}: attn shape={attn_zh.shape}, mlp shape={mlp_zh.shape}")

    # PC0 contribution from attention delta
    attn_pc0_zh = (attn_zh @ pc0).mean()
    attn_pc0_en = (attn_en @ pc0).mean()

    # PC0 contribution from MLP delta
    mlp_pc0_zh = (mlp_zh @ pc0).mean()
    mlp_pc0_en = (mlp_en @ pc0).mean()

    # The SWAP condition would have the same attn/mlp structure but different
    # residual stream. However, we don't have separate deltas for the swapped
    # condition. What we CAN compute is:
    # - How much PC0 does attention ADD at this layer? (attn contribution to convergence)
    # - How much PC0 does MLP ADD at this layer? (MLP contribution to convergence)

    # For natural Chinese: total PC0 change = attn_pc0_zh + mlp_pc0_zh
    # The attention contribution to the zh-en gap closure:
    attn_gap_change = (attn_pc0_en - attn_pc0_zh)  # attention's differential effect
    mlp_gap_change = (mlp_pc0_en - mlp_pc0_zh)    # MLP's differential effect

    total_gap_change = attn_gap_change + mlp_gap_change

    layer_result = {
        "layer": L,
        "zh_en_gap": float(gap),
        "attn_pc0_zh": float(attn_pc0_zh),
        "attn_pc0_en": float(attn_pc0_en),
        "mlp_pc0_zh": float(mlp_pc0_zh),
        "mlp_pc0_en": float(mlp_pc0_en),
        "attn_gap_contribution": float(attn_gap_change),
        "mlp_gap_contribution": float(mlp_gap_change),
        "total_gap_change": float(total_gap_change),
        "attn_fraction": float(attn_gap_change / total_gap_change) if total_gap_change != 0 else None,
        "mlp_fraction": float(mlp_gap_change / total_gap_change) if total_gap_change != 0 else None,
    }
    results["per_layer"].append(layer_result)

# Summary
print("\n" + "="*70)
print("CONVERGENCE DECOMPOSITION: Attention vs MLP")
print("="*70)
print(f"{'Layer':>5} {'Attn ΔPC0':>10} {'MLP ΔPC0':>10} {'Total':>10} {'Attn%':>8} {'MLP%':>8}")
print("-"*55)

total_attn = 0
total_mlp = 0

for r in results["per_layer"]:
    attn_pct = r["attn_fraction"] * 100 if r["attn_fraction"] is not None else 0
    mlp_pct = r["mlp_fraction"] * 100 if r["mlp_fraction"] is not None else 0
    print(f"L{r['layer']:>3}  {r['attn_gap_contribution']:>10.3f} {r['mlp_gap_contribution']:>10.3f} "
          f"{r['total_gap_change']:>10.3f} {attn_pct:>7.1f}% {mlp_pct:>7.1f}%")
    total_attn += r["attn_gap_contribution"]
    total_mlp += r["mlp_gap_contribution"]

total = total_attn + total_mlp
print("-"*55)
if total != 0:
    print(f"{'TOTAL':>5}  {total_attn:>10.3f} {total_mlp:>10.3f} {total:>10.3f} "
          f"{total_attn/total*100:>7.1f}% {total_mlp/total*100:>7.1f}%")
else:
    print(f"{'TOTAL':>5}  {total_attn:>10.3f} {total_mlp:>10.3f} {total:>10.3f}    N/A")

print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
print(f"Cumulative attention contribution to gap: {total_attn:.3f}")
print(f"Cumulative MLP contribution to gap: {total_mlp:.3f}")

if total_attn > 0 and total_mlp > 0:
    print(f"\nBoth attention and MLP build the zh-en gap.")
    print(f"Attention: {total_attn/total*100:.1f}% of total gap creation")
    print(f"MLP: {total_mlp/total*100:.1f}% of total gap creation")
elif total_attn > 0 and total_mlp < 0:
    print(f"\nAttention BUILDS the gap ({total_attn:.3f}), MLP ERODES it ({total_mlp:.3f})")
    print(f"Net: attention contributes {total_attn/(total_attn-total_mlp)*100:.1f}% of the work")
elif total_attn < 0 and total_mlp > 0:
    print(f"\nMLP BUILDS the gap ({total_mlp:.3f}), Attention ERODES it ({total_attn:.3f})")
    print(f"Net: MLP contributes {total_mlp/(total_mlp-total_attn)*100:.1f}% of the work")

# Save
with open("output/expE_convergence_decomposition.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to output/expE_convergence_decomposition.json")
