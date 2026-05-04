# Post-Phase 2 Insights: What We Learned from the Web Chat Analysis

## Date: 2026-02-21

## Key Conceptual Breakthroughs

### 1. The Encoding/Decoding Asymmetry

The h(x) → f*(Z) → h'(y) decomposition is NOT three equal boxes.

**Encoding (h): Layers 0-31 — THICK, gradual, 30+ layers**
- L0-L14: Noise floor (~0.01 similarity to L33). Still thinking linguistically.
- L15-L19: Convergence starts. Language stripping begins.
- L20-L31: Accelerating compression. Reasoning and encoding happen SIMULTANEOUSLY.
- The model doesn't have a clean "first encode, then reason" phase.
  It's doing both at once, with linguistic clothing getting thinner each layer.

**Z instantiation: Layer 32 — the purest Z**
- 12/12 configs confirm. Maximum language-agnostic compression.
- Not L33 (the bottleneck) but L32 (the approach layer) — more dynamic range.

**Decoding (h'): Layers 33-36 — THIN, sharp, 4 layers**
- L33→L34 drop of 0.452 is the sharpest phase transition in the network.
- Once you have the answer in Z-space, re-wrapping in language is EASY.
- This matches the "thin wrapper" prediction from the original hypothesis.

### 2. L32 as the Rosetta Stone

Critical insight: You DON'T need to find Z independently at every layer.

- Find Z where it's cleanest (L32)
- Use L32's Z basis to decompose EVERY other layer's activations
- At layer 5, activations are tangled — but projecting onto L32's basis tells you:
  how much of what's here will eventually become reasoning content vs linguistic scaffolding

### 3. The Residual Update Decomposition (NEW — not in gameplan)

For each layer transition k → k+1:
  Δh_k = h_{k+1} - h_k

Project onto Z and Z⊥ (using L32's basis):
  ||Δh_k^Z||  = magnitude of update within reasoning dims
  ||Δh_k^Z⊥|| = magnitude of update within linguistic dims

The RATIO tells you what each layer is doing:
- ||Δh_k^Z⊥|| >> ||Δh_k^Z|| → layer is mostly stripping language (encoding)
- ||Δh_k^Z|| >> ||Δh_k^Z⊥|| → layer is mostly computing reasoning steps
- Comparable → layer is doing both simultaneously

**Prediction:** Early layers Z⊥-dominated (encoding), middle mixed (thinking in language
while compressing), L33-36 Z⊥-dominated again (decoding). Pure Z layers cluster at L32.

**Most profound case:** If the ratio is NEVER strongly Z-dominated, then the model never
has a "pure reasoning" phase. Z is an emergent property of 30 layers of mixed computation,
not a discrete computational phase. This would mean encoding and reasoning are the SAME process.

### 4. Z Rotation Across Layers (testable)

Open question: Does Z rotate as information flows forward?

- If same dimension indices survive ARD at L10, L20, L25, L32 → Z is a FIXED subspace
  (reasoning allocated to specific dimensions at initialization)
- If different indices → Z ROTATES, and L32's mask underestimates early-layer Z-content
- Either answer is informative. Fixed = pre-allocated. Rotating = active reorganization.

Test: Run ARD-MMD at L10, L20, L25, L32 independently. Compare surviving indices.

## Updated Confidence Tiers

| Tier | Before Phase 2 | After Phase 2 | What changed |
|------|----------------|---------------|--------------|
| ~90%: Something interesting at a specific layer | ~90% | **~98% (confirmed)** | L32 found, 12/12 configs |
| ~80%: Linear bridge works | ~80% | **~75%** | No behavioral asymmetry at 3B weakens bridge narrative |
| ~65%: Z is low-dim, clean | ~65% | **~65%** | 83% config success encouraging but lengthscale shape unknown |
| ~30%: Universal across models | ~30% | **~30%** | No cross-model comparison run |
| NEW: Language actively excluded from Z | n/a | **~85%** | Energy fraction 54% of random |

## What Phase 3 (Patching) Must Answer

1. Does swapping Z-activations at L32 between zh/en change output?
2. Does the change go in the predicted direction? (English gets "Chinese-style" reasoning)
3. Is the effect size large enough to rule out noise?

Phase 3 is the difference between "interesting structural observation" and "paper."

## New Experiment: Layer-by-Layer Update Decomposition

Add ~20 lines to Phase 3 script:
- Extract activations at every layer for zh/en pairs
- Compute Δh_k at each layer
- Project onto L32's Z basis
- Plot ||Δh_k^Z|| / ||Δh_k^Z⊥|| ratio across all 36 layers
- Identify pure-reasoning vs pure-encoding vs mixed layers

## Paper Framing (if Phase 3 works)

"Structural identification of language-agnostic reasoning subspaces without multilingual data"

Key differentiation from NeurIPS 2505.15257:
- They need paired multilingual activations → data-driven SVD
- We need only the weight matrices → structural SVD
- We can predict WHERE Z lives before seeing any data
- Our approach is model-introspective, not dataset-dependent
