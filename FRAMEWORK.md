# Unified Theoretical Framework: Cross-Lingual Reasoning Subspace (Z)

Status as of 2026-03-07. Honest accounting. No inflation.

---

## PROVEN

### Theorem 1 (Bilingual Gradient Equilibrium — Prop 1)

**Source:** `toy_theorem_derivation.md`, verified in `verify_toy_theorem.py`

Single shared linear map W trained on bilingual data (languages ℓ₁, ℓ₂ with
frequencies α₁, α₂). MSE loss. The optimal W has SVD W = UΣV^T where:

    σᵢ* ∝ √ρᵢ

    ρᵢ = (α₁z₁ᵢ + α₂z₂ᵢ)² / (α₁z₁ᵢ² + α₂z₂ᵢ²)

Directions where languages agree (ρᵢ ≈ 1) get amplified.
Directions where they disagree (ρᵢ ≈ 0) get killed.
Z = span of high-ρ directions. Z⊥ = span of low-ρ directions.

**Corollary (Gauge Symmetry Breaking):** Monolingual training has a continuous
family of equivalent solutions (any rotation of W works). Bilingual training
breaks this symmetry — V is determined by the agreement structure ρᵢ.

**Verified numerically:** `output/toy_theorem_verification.json`


### Theorem 2 (SNR Monotonicity — Prop 2)

**Source:** Derived in Claude Web session 2026-03-07, validated in
`z_check_orthogonality.py`

**Setup.** Residual network, L layers, shared weights. Two languages on same
problem. Define:

    h_L^ℓ = h_0^ℓ + Σ_{k=0}^{L-1} f_k(h_k^ℓ)

    D_same(L) = E_x[||h_L^1(x) - h_L^2(x)||²]   (same problem, diff language)
    D_diff(L) = E_{x,x'}[||h_L^ℓ(x) - h_L^ℓ(x')||²]   (diff problem, same lang)
    SNR(L) = D_same(L) / D_diff(L)

**Assumption.** Layer contributions f_k are approximately orthogonal:
⟨f_j, f_k⟩ ≈ 0 for j ≠ k.

**Statement.** SNR(L) is strictly decreasing (alignment improves) at layer L
if and only if:

    E[||δ_L||²] / σ_L² < r_L = D_same(L) / D_diff(L)

where δ_L = f_L(h_L^1) - f_L(h_L^2) and σ_L² = E_{x,x'}[||f_L(x) - f_L(x')||²].

**Proof.** Direct algebra on r_{L+1} < r_L. See Claude Web transcript.

**Assumption validated (2026-03-07, z_check_orthogonality.py):**
- Gram matrix off-diagonal |cos| mean: 0.046, max: 0.171
- Cross-lingual Pythagorean cross-terms: 9.6% of ||Δ_L||²
- SNR empirically monotonic: 1.663 → 1.525 → 1.452 → 1.373 → 1.364 → 1.325
- Every layer satisfies the condition. No exceptions.

**Sharp observation:** Per-layer disagreement ratio ||δ_k||/||f_k|| ≈ 1.4-1.6,
CONSTANT across depth. Individual layers are NOT getting more language-agnostic.
Convergence is purely accumulation (law of large numbers), not per-layer
improvement. This is a cleaner statement than "each layer refines alignment."


---

## CONJECTURES (testable but unproven)

### Conjecture 3 (Z Doubly Suppresses δ_k)

Extends Theorem 1 per-layer. Claims each layer's W_k develops its own Z/Z⊥
decomposition, and δ_k² is doubly suppressed: small projections in Z (languages
agree) + small singular values in Z⊥ (Theorem 1 kills them).

**Gap:** Theorem 1 is for a single isolated linear layer. In a deep network,
the gradient at layer k depends on all subsequent layers through backprop. The
optimal W_k is NOT the same as the isolated bilingual equilibrium. The per-layer
application is heuristic.

**Status:** Plausible. Not proven. Would require showing that backprop coupling
doesn't substantially change the per-layer equilibrium.


### Conjecture 4 (Frequency Controls Convergence Depth)

Lower training frequency α_ℓ → deeper alignment depth L*(ℓ).

**Motivation:** Lower α means W_k is less optimized for that language, so δ_k
is larger per layer, requiring more layers of accumulation.

**Gap:** The "proof sketch" from Claude Web conflates singular VALUES (what
Theorem 1 gives) with singular VECTORS (what you need for the argument about
how different languages project onto W_k). The sketch says "singular vectors
are less aligned to language ℓ" but Theorem 1 says nothing about vectors
changing with α.

**Status:** Consistent with synthetic data (only Chinese converges in 6 layers).
Testable on Qwen (per-language alignment depth vs resource level).


### Conjecture 5 (Difficulty Controls Alignment Depth)

Harder problems → deeper alignment depth L*(x).

**Motivation:** Hard problems have more "active" layers (large ||f_k||), and each
active layer contributes nonzero δ_k.

**Gap:** Hard problems also produce larger σ_k (stronger reasoning signal per
layer). If σ_k grows faster than δ_k, the SNR could improve FASTER for hard
problems, and the conclusion reverses. The sign depends on magnitudes nobody
has computed.

**Status:** Interesting conjecture. Could go either way. Testable on Qwen
(per-problem alignment depth vs difficulty/generation length).


---

## PARKED IDEAS (correct but need future data)

### GARCH on Generation-Step Delta Norms

**Idea (Greg, 2026-03-07):**
Fix a layer l. Track ||δ_l^(t)||² across generation steps t = 1, ..., T where
T is the number of tokens generated. This IS a time series. The index is
generation step, not layer. The stochasticity is real — even at temperature 0,
the sequence is complex and unknown a priori.

GARCH(1,1) on y_t = ||δ_l^(t)||² captures: if the model just produced a large
reasoning update, is the next step likely to also be large? That's volatility
clustering = reasoning momentum. Periods of high activity (mid-proof) should be
autocorrelated. Periods of low activity (narrating) should also be autocorrelated.

**Feynman-Kac interpretation:** Latent state = model's reasoning progress.
Observable = y_t. Filtering problem: given observed {y_t}, what's the posterior
over the latent reasoning state? The regime change from high-variance to
low-variance IS the "aha moment."

**NOT a category error.** The earlier criticism (Claude Code and Claude Web both
initially flagged it) confused this with layers-as-time (which IS a category
error). Generation-steps-as-time at a fixed layer is legitimate.

**Requires:** Generation-time activation extraction from Qwen. Save h_l^(t)
at every generation step t for every layer l. Side quest 12.

**Prediction:** Reasoning-active generation steps show volatility clustering
in per-layer delta norms, detectable by GARCH(1,1) with significant β.


### Johansen Cointegration for dim(Z)

**Idea:** Treat residual stream positions for two languages as two "price series"
evolving through depth. If cointegrated, the cointegrating rank gives the number
of shared stochastic trends. Rank = d - dim(Z⊥) estimates dim(Z).

**This transfers** because it's about the rank of a cross-sectional relationship,
not about stochastic dynamics. Applicable to existing Qwen data.


### VECM Leadership

Error-correction coefficients α₁, α₂ in VECM tell which language adjusts faster
to restore equilibrium. Expect |α_sw| >> |α_zh| (Swahili adjusts to Chinese,
not vice versa). The ratio |α_sw|/|α_zh| is a dynamic measure of wrapper
thickness, complementary to the static condition number κ.


---

## TESTABLE PREDICTIONS (no theorem needed)

### P1. Cross-lingual NN monotonically increases with depth in Qwen
Supported by: Theorem 2, synthetic residual model data.
Test: Compute per-layer NN accuracy on existing Qwen activations.

### P2. Delta t-SNEs cluster by problem before raw t-SNEs do
Supported by: Theorem 2 implies δ_k is more language-agnostic than h_k.
Test: 5 lines on existing Qwen activations. Nobody has done this.

### P3. Per-language alignment depth correlates with training frequency
Supported by: Conjecture 4. Consistent with synthetic data.
Test: Compute L*(ℓ) per language on Qwen.

### P4. Per-problem alignment depth correlates with difficulty
Supported by: Conjecture 5 (sign uncertain).
Test: Compute L*(x) per problem, sort by category or generation length.

### P5. Affine delta increases with depth for low-resource language pairs
Supported by: Synthetic residual model (Swahili ProcR² → -0.218 at L5).
Test: Compute per-layer affine vs Procrustes R² per language pair on Qwen.

### P6. Condition number κ(T_k) at layer k predicts Procrustes vs affine gap
Supported by: Synthetic data. Low κ → Procrustes sufficient. High κ → need affine.
Test: SVD of Qwen bridge matrix (may already exist in project files).


---

## SYNTHETIC EXPERIMENTS (complete)

| Config | Description | Result |
|--------|-------------|--------|
| B (one-hot, orthogonal) | GOLD standard | All 7 languages converge |
| B-blind orthogonal | Remove language flag, keep orthogonal transforms | Impossible — proves language ID necessary |
| B-blind invertible, no centroids | Non-orthogonal transforms, no centroids | Fails — single-sample covariance detection impossible |
| B-blind centroids, no residual | Add centroids as geometric language signal | Chinese converges. Mid-network Procrustes peak at L4. Affine delta +0.174 |
| B-blind centroids, residual | Add skip connections (1-4-1 architecture) | Chinese converges. Monotonic ProcNN increase (0.787→0.980). Affine delta +0.146 |

**Orthogonality validation on residual model:**
- Gram matrix off-diagonal |cos|: mean 0.046, max 0.171
- Cross-lingual Pythagorean: 9.6% cross-terms
- SNR strictly monotonic through all 6 layers

**Key files:**
- `z_poc_blind_sweep.py` — main experiment script
- `z_check_orthogonality.py` — orthogonality validation
- `output/z_poc_blind_sweep_eps0.10.json` — results
- `output/z_poc_blind_sweep_eps0.10_trained.pt` — trained model + data


---

## NOTATION REFERENCE

| Symbol | Meaning |
|--------|---------|
| ρᵢ | Per-direction agreement ratio (Theorem 1) |
| σᵢ* | Optimal singular value of W (∝ √ρᵢ) |
| Z | Span of high-ρ directions (shared reasoning subspace) |
| Z⊥ | Span of low-ρ directions (language-specific subspace) |
| f_k(h_k) | Layer k's contribution to the residual stream |
| δ_k | Cross-lingual disagreement at layer k: f_k(h_k^1) - f_k(h_k^2) |
| σ_k² | Between-problem variance of layer k's contribution |
| D_same(L) | Expected cross-lingual distance at layer L (same problem) |
| D_diff(L) | Expected between-problem distance at layer L (same language) |
| SNR(L) | D_same/D_diff — decreasing = better alignment |
| κ(T_k) | Condition number of cross-lingual transform at layer k |
| L*(ℓ) | Alignment depth for language ℓ (layer where NN > threshold) |
| L*(x) | Alignment depth for problem x |
