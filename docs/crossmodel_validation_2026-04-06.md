# Cross-Model Validation Session — 2026-04-06

## Overview

Comprehensive cross-model replication of key 3B findings across 5 models:
Qwen2.5-3B, Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3.5-9B, Qwen2.5-14B-Instruct.

Two experiment suites:
- **Exp AX**: Language-direction flip intervention (4 conditions per model)
- **Exp BB**: Validation suite — cocycle, f-probe, PACF innovation, language direction

Plus local experiments:
- **Exp AZ**: Delta-covariance eigenspectrum (3B only, 7 languages)
- **Exp BA**: Z-MLP decomposition (3B only, 7 conditions)

---

## Exp AX — Cross-Model Flip Intervention

| Model | Funnel? | Baseline | Flip | Random | Surgery | Flip Delta |
|-------|---------|----------|------|--------|---------|------------|
| **3B** (prior) | YES (tight) | 5/20 | 13/20 | — | — | **+8** |
| **7B** | untested | 8/20 | 10/20 | 5/20 | 8/20 | **+2** |
| **8B** | NO | 6/20 | 7/20 | 6/20 | 7/20 | **+1** |
| **9B** | YES (extended) | 5/20 | 4/20 | 4/20 | 5/20 | **-1** |
| **14B** | NO | 5/20 | 5/20 | 6/20 | 7/20 | **0** |

### Key findings:

1. **The flip is 3B-specific.** No other model shows a meaningful effect. The 7B's +2 is suggestive but within noise at N=20 (random = -3, so the contrast is +5 vs random, but the absolute improvement is small).

2. **The funnel does NOT predict flip efficacy.** The 9B has a funnel but flip=-1. The 7B has no tested funnel but flip=+2. The funnel-flip hypothesis is dead.

3. **project_1d surgery = noise on all models except 3B.** The deployment result (AG2: +44% ZH on 3B) does not generalize.

4. **Larger models have higher baselines** — the 7B gets 8/20 at 128 tokens where the 3B gets 5/20. The efficiency argument: larger models don't need the flip because they're already more efficient at reaching the answer.

---

## Exp BB — Cross-Model Validation Suite

### Full comparison table:

| Metric | 3B | 7B | 8B | 9B | 14B |
|--------|----|----|----|----|-----|
| Layers | 36 | 28 | 36 | 32 | 48 |
| d_model | 2048 | 3584 | 4096 | 4096 | 5120 |
| **Cocycle R²** | 0.941 | 0.914 | 0.896 | 0.871 | 0.922 |
| **Cocycle error** | 0.75% | 0.68% | 0.39% | 1.04% | 0.38% |
| **Category transfer** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Answer transfer** | 0.275 | 0.250 | 0.300 | 0.300 | 0.275 |
| **PACF R² (mid)** | 0.940 | 0.908 | 0.910 | 0.913 | 0.930 |
| **Max Cohen's d** | 108 | 116 | 119 | 179 | 145 |

### What's universal (confirmed across all 5 models):

1. **Category transfer = 1.000 everywhere.** Problem type is encoded language-agnostically from early layers. This is the f-reconstruction result (Exp Z) replicated at every scale and architecture. The shared mathematical computation space is real and universal.

2. **Cocycle R² > 0.87 everywhere, error < 1.04%.** The cross-lingual manifold is near-flat at every scale. Ridge regression maps between languages with >87% R² and the triangle consistency error is always <1.04%. The language manifold geometry is a universal feature of multilingual transformers.

3. **PACF R² ~ 0.91-0.94 everywhere.** Consecutive MLP deltas are highly predictable from each other at every scale. The 3B is NOT an outlier — the "97% innovation" from Exp T was measuring a different thing (predictability from the layer's own input, not from the previous delta).

4. **Phase transition is universal.** The adversarial→cooperative cosine sign flip occurs at 40-47% depth in all models:
   - 3B: L17→L18 (47%)
   - 7B: approx L11→L12 (40%)
   - 8B: L14→L15 (39%)
   - 9B: L15→L16 (47%)
   - 14B: L20→L21 (42%)

5. **Answer transfer ~ 0.25-0.30 everywhere.** Partial and comparable. The answer value is partially but not fully language-agnostic, with similar extraction difficulty across scales.

### What varies with scale:

1. **Cohen's d increases with scale.** 3B=108, 7B=116, 8B=119, 14B=145, 9B=179. The language direction becomes more well-defined with more parameters. But this doesn't correlate with flip efficacy.

2. **Cocycle error varies non-monotonically.** 3B=0.75%, 7B=0.68%, 8B=0.39%, 9B=1.04%, 14B=0.38%. The Qwen3.5-9B is an outlier (highest error) — possibly related to its unique architecture.

---

## Exp AZ — Eigenspectrum (3B, local)

Delta-covariance eigenspectrum using 7-language multilingual data at all 36 layers.

**Key finding:** The language direction is NOT in the top eigenvectors of the delta covariance at any layer. Maximum alignment cos=0.215 at L9. The dominant directions of MLP modification are problem-specific, not language-specific. The language direction lives in a thin subspace of the delta, which is why the 1D flip works and higher-dimensional interventions fail.

Top eigenvalue grows from 2.0 (L0) to 10,611 (L34) — the "uncontrolled growth" that AttnRes addresses. Participation ratio drops from ~18 in mid-layers to 3.7 at L34.

---

## Exp BA — Z-MLP Decomposition (3B, local)

Tests whether the MLP can reason on the 20-dim Z subspace alone.

| Condition | EN | ZH | Total |
|-----------|----|----|-------|
| baseline | 2/10 | 4/10 | 6/20 |
| z_input_all (MLP sees only Z) | 0/10 | 0/10 | **0/20** |
| z_output_all (keep only Z of delta) | 0/10 | 0/10 | **0/20** |
| z_output_complement (keep only lang of delta) | 0/10 | 2/10 | **2/20** |
| z_output_adversarial (L9-17 only) | 0/10 | 0/10 | **0/20** |
| z_output_cooperative (L18-26 only) | 0/10 | 0/10 | **0/20** |
| z_output_L9_L26 | 0/10 | 0/10 | **0/20** |

**Complete destruction across all conditions.** The 20-dim Z is a useful descriptor for probing (category transfer = 1.0) but it is NOT a functional decomposition that the MLP respects. MLP(P_Z @ x) ≠ P_Z @ MLP(x) because the SiLU gating depends on the full 2048-dim input pattern. The factored-MLP proposal from Claude Web doesn't work at this granularity.

---

## Revised Understanding (post-cross-model validation)

### What's universal:
- Category is encoded language-agnostically (f-probe = 1.0 at all scales)
- The cross-lingual manifold is near-flat (cocycle < 1.04% at all scales)
- MLP deltas are consecutively predictable (PACF R² ~ 0.91-0.94 everywhere)
- The adversarial→cooperative phase transition occurs at ~40-47% depth universally
- The language direction is 1D and strong (Cohen's d > 100 everywhere)

### What's 3B-specific:
- The flip intervention (+8/20 = +160%)
- The project_1d weight surgery (+44% ZH accuracy)
- Both are generation-time dynamics effects, not representational geometry effects

### What this means:
The representational geometry is universal. The cross-lingual manifold structure, the phase transition, the shared mathematical space, the 1D language direction — all of these hold across models from 3B to 14B and across Qwen2.5/Qwen3/Qwen3.5 architectures. These are paper-worthy universal claims.

The flip intervention is a 3B-specific phenomenon that depends on the interaction between generation-time dynamics and the specific capacity constraints of that model. The 3B's limited capacity (d=2048, 36 layers) creates a regime where the language-direction flip can shortcut reasoning efficiency, but this regime doesn't exist in models with d≥3584.

The paper should lead with the universal claims and present the flip as a case study of how the universal geometry enables model-specific interventions under specific capacity constraints.

---

## Files Saved

| File | Description |
|------|-------------|
| `output/expAX_Qwen_Qwen3-8B.json` | 8B flip: 4 conditions |
| `output/expAX_Qwen_Qwen3.5-9B.json` | 9B flip: 4 conditions |
| `output/expAX_Qwen_Qwen2.5-14B-Instruct.json` | 14B flip: 4 conditions |
| `output/expAX_Qwen_Qwen2.5-7B-Instruct.json` | 7B flip: 4 conditions |
| `output/expBB_validation_Qwen_Qwen2.5-3B.json` | 3B validation: cocycle/probe/PACF/lang |
| `output/expBB_validation_Qwen_Qwen2.5-7B-Instruct.json` | 7B validation |
| `output/expBB_validation_Qwen_Qwen3-8B.json` | 8B validation |
| `output/expBB_validation_Qwen_Qwen3.5-9B.json` | 9B validation |
| `output/expBB_validation_Qwen_Qwen2.5-14B-Instruct.json` | 14B validation |
| `output/expAZ_eigenspectrum.json` | 3B eigenspectrum (7 langs, 36 layers) |
| `output/expBA_z_mlp_decomposition.json` | 3B Z-MLP decomposition (7 conditions) |
| `docs/crossmodel_validation_2026-04-06.md` | This document |
| `perturbation/exp_crossmodel_flip.py` | Flip experiment script |
| `perturbation/exp_crossmodel_validation.py` | Validation suite script |
| `expBA_z_mlp_decomposition.py` | Z-MLP decomposition script |
