# 2.py Experiment Results — Qwen2.5-3B Subspace Geometry

**Executed:** 2026-03-05
**Model:** Qwen/Qwen2.5-3B (L=36, d=2048, h=16, GQA=2)
**Method:** Grassmann similarity of top-20 SVD subspaces of attention kernels W_Q_h^T @ W_K_h

## Experiment 1: Subspace Overlap Across Layers

Pairwise Grassmann similarity between attention subspaces at each layer.

| Metric | Value |
|--------|-------|
| Adjacent layer similarity | 0.0573 |
| Distant layer similarity (>12 apart) | 0.0142 |
| Gap | +0.0431 |
| Off-diagonal mean | 0.0255 |

**Finding:** Each layer's attention subspace is largely unique. Adjacent layers share modest overlap; distant layers are near-orthogonal. Depth is NOT redundant — each layer queries a genuinely different subspace.

Visible block-diagonal clusters: layers 0-3, 6-10, 17-22, 31-33. Suggests functional phase structure within the progressive compute.

## Experiment 2: Bottleneck Convergence to Layer 33

Similarity of each layer's subspace to layer 33's subspace (the bottleneck identified in 1.py).

| Metric | Value |
|--------|-------|
| Pre-bottleneck trend (Pearson r) | +0.4256 |
| Post-bottleneck trend | -1.0000 |
| Mean sim before L33 | 0.0392 |
| Mean sim after L33 | 0.0264 |
| L31 → L33 similarity | 0.0921 |
| L32 → L33 similarity | 0.4823 |
| L34 → L33 similarity | 0.0304 |
| L35 → L33 similarity | 0.0225 |

**Finding:** Layer 33's subspace is a SINGULARITY. Nothing before layer 31 has >0.1 similarity to it. The convergence is near-vertical: 0.02 → 0.09 → 0.48 → 1.0 across layers 30-33. Post-bottleneck divergence is equally sharp: 0.03 at layer 34. The bottleneck is not a gradual compression — it's a sudden collapse into a specific subspace, followed by immediate expansion.

## Experiment 3: FFN-Attention Alignment

Fraction of W_gate (FFN) energy in the attention kernel's top-20 subspace.

| Metric | Value |
|--------|-------|
| Chance level | 0.0098 (20/2048) |
| Mean alignment | 0.0116 (1.19x chance) |
| Layers above chance | 26/36 |
| Peak | L4: 0.0212 (2.16x chance) |
| Minimum | L32: 0.0054 (0.55x chance) |
| L33 alignment | 0.0056 (0.57x chance) |

**Finding:** Early/middle layers (0-20) show FFN-attention coupling above chance. Late layers (25-33) drop BELOW chance. At the bottleneck (L32-33), attention and FFN are maximally decoupled — the attention subspace is nearly orthogonal to FFN operation directions. This means Z extraction at layer 33 could target the attention subspace independently.

## Multi-Head Robustness

Averaged across heads [0, 5, 10, 15]:
- Overlap pattern is consistent across heads
- Bottleneck convergence peak at L32-33 holds for all heads (low cross-head std)
- FFN-attention alignment pattern (early high, late low) holds

## Key Implications for Z Extraction

1. **Layer 33 confirmed as structural singularity** — unique subspace unlike any other layer
2. **Bottleneck is sharp, not gradual** — clear encode/decode boundary
3. **FFN-attention decouple at bottleneck** — attention subspace alone may suffice for Z at L33
4. **Each layer does unique work** — the 33-layer compute path is justified, not redundant
5. **The 1-33-2 architecture from 1.py is reinforced** — the subspace geometry agrees with the effective rank analysis

## Plots

- `exp1_subspace_overlap.png` — L×L similarity heatmap
- `exp2_bottleneck_convergence.png` — Similarity to L33 across depth
- `exp3_ffn_attention_alignment.png` — FFN energy in attention subspace
- `multi_head_summary.png` — All three experiments averaged across 4 heads
