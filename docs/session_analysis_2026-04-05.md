# Session Analysis — 2026-04-05

## Context

Picking up from the 4-model trajectory replication session (earlier today). Greg had a separate conversation with Claude Webb (not VEGA, not fully in the loop on all experiments) that organically evolved from Kepler's laws, metrics, and Christoffel symbols into a differential geometry framing proposal. This document captures (a) what Webb proposed, (b) VEGA's assessment after full inventory of all 55+ experiments and project data, and (c) the priority-ordered next steps.

---

## 1. Where We Left Off (4-Model Trajectory Results)

Three snapshots from today's earlier session (session `b9e8d9f7`):

### Trajectory Replication Results (Colab A100, 4 models)

| Model | d_model | Layers | Funnel? | Spread at tightest | Notes |
|-------|---------|--------|---------|-------------------|-------|
| Qwen2.5-3B | 2048 | 36 | YES, tight | 0.015 at L9 | Tight funnel at L9, second pinch at L12 |
| Qwen3.5-9B | 4096 | 32 | YES, extended | 0.02-0.04 L0-L14 | Peaks at 35% depth then DECAYS — unique pattern |
| Qwen3-8B | 4096 | 36 | NO | 0.09-0.14 | Spread stays high throughout |
| Qwen2.5-14B-Instruct | 5120 | 48 | NO | 0.08-0.16 | Spread stays high throughout |

**Headline finding:** The funnel is NOT universal. It's a capacity-times-training interaction. Smaller models (3B) are forced to compress cross-lingual representations through a bottleneck. Larger models (8B, 14B) have enough capacity to maintain separate language tracks. The 9B is architecturally different (Qwen3.5 family) and shows an extended funnel with early peak followed by decay — the most interesting and unexplained pattern.

### Five Threads Presented to Greg (from prior session)
1. 9B decay investigation — why does alignment peak at 35% depth then degrade?
2. MOAMS transplant on 9B — test if optimal transplant window differs
3. Code L35 crash paradox — strip scaffolding tokens, measure code-only cosine
4. Phase-aligned analysis — segment by computation phase instead of normalized time
5. Paper figures — the three core plots that tell the story

---

## 2. Webb's Proposal (Claude Webb, separate conversation)

### 2a. AttnRes Paper (Kimi Team, arXiv 2603.15031, March 2026)

"Attention Residuals" replaces fixed unit-weight residual connections (x_{l+1} = x_l + f_l(x_l)) with softmax attention over preceding layer outputs. Results: 1.25x compute advantage, biggest gains on reasoning (GPQA-Diamond +7.5, Math +3.6). Block AttnRes partitions layers into ~8 blocks for practicality.

**Webb's connection:** Our three-phase structure (align, adversarial debate, cooperative assembly) explains WHY AttnRes works for reasoning. Late layers (reassembly phase) need information from pre-shatter representations (early layers) and category-specific processing (middle layers). Standard residuals dilute these signals. AttnRes lets late layers attend directly back to pre-shatter representations.

**dim 318 connection:** dim 318 carries 70% of the mean direction, the mean direction explains 90% of total variance, and the reasoning content lives in the remaining 10%. This IS the "uncontrolled hidden-state growth" AttnRes addresses. Our Z subspace (20-dim reasoning space) gets drowned by dim 318's accumulated signal.

**Ziming Liu's blog response:** AttnRes works as regularization — constraining toward structured, low-dimensional solutions. Our Z subspace IS that structured solution. His regularization view and our geometric view are two perspectives on the same phenomenon.

### 2b. Metric Tensor Formalization

Core idea: the spread plot IS a metric measurement. Each layer defines a local Riemannian geometry on activation space. The funnel is metric contraction — the metric in the cross-lingual direction shrinks at those layers.

**Proposed computation:** At each layer l, compute the effective metric via the Jacobian J_l = dh_{l+1}/dh_l. Induced metric: g(l) = J_l^T J_l. Eigenvalues tell which directions are compressed (small) vs stretched (large).

**Simpler proxy:** Covariance matrix of layer-to-layer deltas (h_{l+1} - h_l) across all problems and languages. Eigenspectrum shows which directions each layer is actively modifying. Compare eigenspaces to Z subspace PCA directions.

**Predictions:** Funnel layers in 3B should show eigenvalue collapse in cross-lingual direction. 8B/14B should show uniform eigenvalues (no selective compression). 9B should show early collapse followed by re-expansion (explaining the decay pattern).

### 2c. Cocycle Curvature Interpretation

Cocycle error (2.86%) measures whether parallel transport around a language triangle returns to origin. Low error = near-zero curvature. Christoffel symbols computable from the metric. Per-layer, per-language-pair metric would give the Riemann curvature tensor — a stronger statement than "the cocycle holds."

### 2d. Paper Framing

Differential geometry of the residual stream: metric tensors, curvature, parallel transport. Spread plot becomes "sectional curvature profile." Funnel becomes "metric contraction in the cross-lingual fiber bundle." Cocycle becomes "flatness of the language connection." AttnRes becomes engineering validation.

---

## 3. VEGA's Assessment (After Full Project Inventory)

### 3a. What the Inventory Shows

**Data on disk:**
- 80+ output JSON files (experiments A through AQ, plus phases 0-6, controls, sweeps)
- 23 numpy archives with raw hidden states (total ~1.5 GB)
- 4 trajectory captures (3B/8B/9B/14B), but only 3B has raw vectors; 8B/9B/14B have only cosines and norms
- Full 7-language data for 3B: `multilingual_all_layers.npz` — 7 langs x 200 problems x 36 layers x 2048 dims (241 MB)
- Full 2-language data for 3B: `all_layers.npz`, `all_layers_lasttok.npz` — 200 x 36 x 2048
- MLP delta data: `attn_mlp_deltas.npz`, `layer_deltas.npz`
- Generation trajectories: `gen_trajectories.npz`, `gen_trajectories_peos.npz`
- dim 318 analysis: confirmed as 70% of mean direction, scale factor not content signal

**Experiment coverage:** 55+ experiments spanning:
- Mechanism discovery (Exps 1-7, A-F): MLP deltas carry language, PC0 swap, steering
- Subspace extraction (Exps G-Q): Z subspace, kernel, independence, cross-lingual transfer
- Causal validation (Exps R-W): flip sweep, token efficiency, domain transfer, TC0 killed, Coder dissociation
- Dynamics (Exps X-Z, AA-AQ): Z-iteration, von Neumann, f-reconstruction, kernel, generation interventions, trajectory dynamics, crystallization, probes
- Cross-model (perturbation/): MOAMS transplant, trajectory capture on 4 models

### 3b. On the Metric Proposal — Honest Assessment

**Where Webb is right:**
- AttnRes citation is load-bearing for the paper. Our three-phase story provides THE mechanistic explanation for their architectural improvement. This should be cited.
- dim 318 = their "uncontrolled growth." Concrete, quantitative connection.
- The eigenspectrum computation on 3B is cheap (we have all the data locally) and could add one clean figure to the paper.
- Ziming Liu's regularization interpretation + our geometric interpretation = two views of the same thing. Worth noting.

**Where the metric proposal oversells:**
- We already know the operative mechanism is 1D (Exp AE: flip beats Procrustes at every dimensionality, "the swish doesn't exist"). The metric formalization would confirm "one eigenvalue collapses" — restating in Riemannian language what the spread plot already shows.
- Relabeling 2.86% cocycle error as "near-zero Riemann curvature" adds jargon, not insight, unless the tensor decomposition reveals additional structure beyond the scalar we already have.
- The cross-model eigenspectrum comparison (the actually interesting computation) can't be done: 8B/9B/14B trajectory data has only cosines and norms, not raw hidden state vectors. We'd need to recapture on Colab.
- "Nobody else has this framing" — true, but the framing we already have (1D flip as symmetry, training-determined entanglement, efficiency not accuracy) is also novel and more empirically grounded.

**Bottom line:** The metric is notation-upgrade for existing results on 3B, and discovery-grade only if we recapture raw hidden states for larger models and the eigenspectra diverge between funnel/no-funnel models. It's paper decoration, not paper foundation.

### 3c. What's Actually Staring Us in the Face

**Gap 1: The funnel-flip correlation is completely untested.**

We have funnel data for 4 models and flip intervention data for only 2 (3B and 1.5B). The most obvious untested prediction of the project:

| Model | Has funnel? | Flip tested? | Flip prediction |
|-------|------------|-------------|-----------------|
| Qwen2.5-1.5B | Presumed yes (untested) | YES: +43% at 128 tok | Confirmed |
| Qwen2.5-3B | YES (tight, L9) | YES: +160% at 128 tok | Confirmed |
| Qwen3.5-9B | YES (extended, L0-L14) | NO | Should work |
| Qwen3-8B | NO | NO | Should NOT work |
| Qwen2.5-14B | NO | NO | Should NOT work |

One Colab session testing the flip on 8B, 9B, and 14B either unifies the entire story or breaks it. Either outcome is publication-grade.

**Gap 2: project_1d scales to larger models?**

Exp AG2 showed project_1d weight surgery gives +44% ZH accuracy at zero inference cost on 3B. If the funnel is a capacity bottleneck, then larger models that DON'T funnel might benefit EVEN MORE from the 1D projection — they have more capacity wasted on maintaining separate language tracks. Or they might not benefit at all because their internal structure is different. We don't know because we've never tested it.

**Gap 3: The project_1d finding IS a deployment result.**

55 experiments of science, and the engineering punchline is: one line of weight modification permanently improves cross-lingual performance. The paper isn't just science if this scales — it's an engineering contribution. This is arguably more impactful than the metric formalization.

### 3d. The Unified Narrative (What We Can Already Write)

The 55+ experiments tell a single coherent story:

1. **The model computes in a shared mathematical space** (Exp Z: category probe transfers perfectly EN→ZH from L4 onward; Exp AB: 7-language SVD extracts kernel; Exp N: MLP deltas are cross-lingually transferable)

2. **Language is a 1D symmetry axis** (Exp AE: flip = reflection, beats Procrustes at all dimensionalities; Exp P2: mean-difference direction, not PCA, is causal; Exp V3: distributed ensemble, all 18 layers needed)

3. **The flip is efficiency, not accuracy** (Exp P3/R/R2/R3: baseline catches up given enough tokens; the flip shortens the path to the answer, it doesn't unlock new capabilities)

4. **Entanglement is training-determined, not architectural** (Exp W: Coder-3B has the direction but flip=0 because no Chinese math competence exists; Exp X: EN math survives language stripping, ZH math breaks — asymmetric entanglement favoring the dominant training language)

5. **Capacity determines compression** (trajectory replication: 3B forced to funnel, 14B has room to maintain separate tracks)

6. **Each layer is an independent specialist** (Exp T: R²=0.03 for math MLP deltas, 97% fresh innovation; phase transition from adversarial to cooperative at L17→L18)

7. **The autoregressive loop is load-bearing** (Exp Y: layer iteration without tokens diverges; Exp K2b: KV cache itself is expendable — it's the token→embed→attend cycle that re-anchors computation in the reasoning manifold)

8. **The mechanism is surgically deployable** (Exp AG2: project_1d in weights = +44% ZH, no language damage, zero inference cost)

---

## 4. Priority-Ordered Next Steps

### Priority 1: Flip Intervention on 8B/9B/14B (Colab A100, ~30 min)

Tests the funnel-flip correlation. Adapt the flip intervention code from our existing experiments for larger models. This is the one experiment that writes or redirects the paper.

**What we need:** The language direction (mean-difference between ZH and EN hidden states at each layer) for each model, then the flip intervention during generation. The exp_trajectory_capture.py script has most of the infrastructure; we need to add the flip intervention mode.

**Prediction table:**
- 9B (has funnel): flip should improve ZH math accuracy
- 8B (no funnel): flip should have zero or negligible effect
- 14B (no funnel): flip should have zero or negligible effect

### Priority 2: project_1d Weight Surgery on 8B (same Colab session)

Tests whether the deployment result scales. If project_1d helps a model WITHOUT a funnel, the mechanism is different from what we think and that's important to know. If it only helps funneling models, it confirms the capacity story.

### Priority 3: Eigenspectrum of Delta-Covariance on 3B (local, ~5 min)

Cheap computation using existing `all_layers.npz` and `multilingual_all_layers.npz`. Compute covariance of (h_{l+1} - h_l) at each layer across 200 problems x 7 languages. Extract eigenspectrum. Check whether funnel layers (L9-L12) specifically compress the language direction or compress everything uniformly.

This adds one clean figure to the paper and connects to Webb's metric story without requiring the full Riemannian formalism.

### Priority 4: AttnRes Citation and Paper Framing (writing, no compute)

Position three-phase structure as mechanistic explanation for AttnRes. Connect dim 318 to "uncontrolled growth." Note Ziming Liu's regularization interpretation as complementary view. Frame as: "Our Z subspace is the structured solution that AttnRes's regularization effect promotes."

### Priority 5: Full Eigenspectrum Recapture on Colab (if thread 3 is interesting)

If the 3B eigenspectrum shows clear language-direction collapse at funnel layers, go back to Colab and recapture raw hidden states for 8B/9B/14B at key layers. Then compute cross-model eigenspectrum comparison. This is the metric story done properly — but only worth it if the 3B diagnostic is compelling.

---

## 5. Data Inventory (Quick Reference)

### Raw Hidden States (local, 3B only)
- `output/all_layers.npz` — 200 problems x 36 layers x 2048 dims, 2 langs (100 MB)
- `output/all_layers_lasttok.npz` — same but last-token only (69 MB)
- `output/multilingual_all_layers.npz` — 200 x 36 x 2048, 7 langs (241 MB)
- `output/attn_mlp_deltas.npz` — attention and MLP delta decomposition (139 MB)
- `output/layer_deltas.npz` — layer-to-layer deltas (39 MB)
- `output/gen_trajectories.npz` — generation-time trajectories (69 MB)

### Trajectory Captures (cosines + norms only for 8B/9B/14B)
- `output/trajectories_Qwen_Qwen2.5-3B_all.json` — 16 problems, 36 layers, per-token (6.7 MB)
- `output/trajectories_Qwen_Qwen3-8B_all.json` — 16 problems, 36 layers (7.2 MB)
- `output/trajectories_Qwen_Qwen3.5-9B_all.json` — 16 problems, 32 layers (7.0 MB)
- `output/trajectories_Qwen_Qwen2.5-14B-Instruct_all.json` — 16 problems, 48 layers (8.9 MB)

### Key Experiment Results
- `output/expAE_procrustes_rotation.json` — Procrustes catastrophic, flip_1d optimal
- `output/expAF_trajectory_dynamics.json` — Lyapunov/Z/gauge all confirmed
- `output/expAG2_kernel_sweep.json` — project_1d = +44% ZH, zero inference cost
- `output/expV3_phase_block_n20.json` — TC0=0 effect, lang=+8, distributed ensemble
- `output/expT_pacf_innovation.json` — R²=0.03, 97% innovation, phase transition
- `output/expW_coder3b.json` (if exists) or equivalent — Coder-3B dissociation
- `output/dim318_deep_dive.json` — dim 318 is scale factor (70% of mean, 90% of variance)

### Perturbation Scripts
- `perturbation/exp_trajectory_capture.py` — trajectory recorder (used for all 4 models)
- `perturbation/exp_moams_x.py` — cross-domain transplant with --fast mode
- `perturbation/exp_moams_transplant.py` — original MOAMS transplant

### Plots (4-model comparison)
- `output/fig_convergence_4models.png` — 4-panel convergence profiles
- `output/fig_spread_4models.png` — cross-domain spread comparison
- Earlier 3B-only: cosine heatmaps, velocity, volatility, norm ratio, layer jumps, deep dive

---

## 6. Experiment Naming Convention (for continuity)

If we proceed with the priorities above:
- **Exp AX**: Cross-model flip intervention (8B/9B/14B)
- **Exp AY**: Cross-model project_1d weight surgery
- **Exp AZ**: Delta-covariance eigenspectrum (3B, local)

These follow the existing alphabetical sequence (last was AQ, then the trajectory work was unnumbered).
