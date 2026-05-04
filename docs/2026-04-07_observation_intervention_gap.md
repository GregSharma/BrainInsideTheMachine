# The Observation-Intervention Gap: Why Gram Rank ≠ Computational Rank

**Date:** April 7, 2026
**Authors:** Greg Sharma, VEGA (Claude Opus 4.6), Claude Web (Sonnet 4.5), GPT 5.4
**Status:** Active theoretical framework — experiments C1/C2 proposed, not yet run

---

## 0. The Problem in One Paragraph

We showed the centered Gram matrix of hidden states across 1,400 problems has rank_90 ≈ 20 at every layer, across 4 models (Qwen2.5-3B/7B/14B, Qwen3-8B). We used the Gram's eigenvalue dynamics to predict which layers are skippable (BQ3 pruning: 5/36 layers removed, accuracy preserved). Then we tried the natural next step: if 20 dimensions capture 90% of the inter-problem variance, can we project hidden states onto those 20 dimensions during inference? **No.** SVD truncation at k=500 (99.9% variance retained) produces 0/20 accuracy. At every k, every layer range, both prefill and generation. The Gram matrix predicts layer-level importance perfectly but is completely blind to within-layer compressibility. This document explains why, and proposes two experiments to measure what the Gram was missing.

---

## 1. What BS/BS2 Killed

### Experiment BS: SVD-Truncated Inference (23 conditions)
- Script: `expBS_svd_truncation.py`
- Output: `output/expBS_svd_truncation.json`
- Method: At equilibrium layers (L9-L26), replace h with P_k(h - μ) + μ where P_k projects onto top-k PCs of the centered Gram.
- **Result: ALL 0/20 at every k ∈ {2, 5, 10, 20, 50, 100, 200, 500}.**
- Failure gradient: k≤50 → gibberish. k≥200 → fluent English, 0 math, ZH→EN shift.

### Experiment BS2: Generation-Only Truncation (8 conditions)
- Script: `expBS2_genonly_truncation.py`
- Output: `output/expBS2_genonly_truncation.json`
- Method: Separate prefill and generation phases. Test truncation in each independently.
- **Result: ALL 0/20. Including prefill-only truncation (corrupt prefill + clean generation).**

### Combined with K2b (KV Expendability)
- K2b: Scrambling ALL 36 layers' KV cache position indices = zero effect on accuracy.
- The model bootstraps generation entirely from the last-token residual stream h_35.
- BS/BS2: That residual stream needs all 2048 dimensions. Truncation at any point breaks the cascade.

### The Interpretive Chain
1. K2b: KV cache is vestigial for these problems. Everything flows through h.
2. BS: PCA truncation at any k kills accuracy. Low-rank Gram ≠ low-rank computation.
3. BS2: Separating prefill/gen doesn't help. Corrupted h_35 bootstraps into wrong basin.
4. **Conclusion:** Full 2048 dimensions are load-bearing. The centered Gram rank_90 = 20 is about inter-problem relational geometry, not per-problem computational dimensionality.

---

## 2. The Theoretical Explanation

### 2.1 Two Kinds of Dimensionality

| Object | Dimension | What it measures |
|--------|-----------|-----------------|
| rank(G_ℓ^repr) at 90% energy | ~20 | How many axes of variation distinguish problems from each other |
| Computational rank of F_ℓ | ≤ 2048 | How many dimensions F_ℓ's weight matrices read/write |
| Eckart-Young optimal k for G | 20 | Best rank-k approximation to pairwise structure |
| Minimum d preserving F's output | ≥ 500, likely ≈ 2048 | Best rank-k approximation to the computation |

Eckart-Young guarantees ‖G - G_k‖_F is minimized at k = 20. But ‖F(P_k h) - F(h)‖ is a completely different objective. PCA truncation optimizes the former. BS tested the latter. They don't agree.

### 2.2 The V ⊕ V⊥ Decomposition (GPT 5.4's Framework)

The fundamental mathematical structure is a splitting:

```
ℝ^d = V_ℓ ⊕ V_ℓ^⊥
```

where:
- **V_ℓ** (dim ~20): carries almost all between-problem variance. This is what the Gram sees.
- **V_ℓ^⊥** (dim ~2028): the "carrier subspace." Near-zero inter-problem variance. Still interrogated by W_up through the nonlinear gates.

Layer deletion stays in the pre-image (the full d-dimensional space) — it just skips one additive perturbation to the residual stream. That's why BQ3 works.

Dimension truncation projects OUT of the pre-image into a k-dimensional affine subspace. Every weight matrix in the next layer receives inputs missing d-k coordinates. That's why BS fails.

**Layer deletion is robust because the residual stream is a sum.**
**Dimension truncation is catastrophic because the weight matrices are a product.**

### 2.3 Why Low Variance ≠ Low Relevance (The Tail Sensitivity Argument)

Let h = μ + δ, P = P_k, Q = I - P. The truncation error is e = Qδ.

```
F_{ℓ+1}(μ + δ) - F_{ℓ+1}(μ + Pδ) = J_{ℓ+1}(μ + Pδ) · e + O(‖e‖²)
```

"Projection is safe" requires not that ‖e‖² is small (it is — 0.1% of variance), but that:

```
‖J_{ℓ+1}(h) · Q‖ · ‖e‖ < decision margin
```

The tail is low-variance but **high-leverage**. The paradox is only apparent: low variance does not imply low dynamical relevance.

### 2.4 The SiLU Gate-Control Channel

The MLP Jacobian at h:

```
J_MLP(h) = W_down · D(h) · W_up,    D(h) = diag(σ'(W_up · h))
```

Projection changes not just the input linearly, but the diagonal gate matrix: D(h) → D(Ph).

The error includes:
```
W_down · (D(h) - D(Ph)) · W_up · Ph
```

not just:
```
W_down · D(h) · W_up · (h - Ph)
```

For SiLU, σ' and σ'' vary substantially near zero. Small tail perturbations alter many diagonal entries of D(h) if many preactivations lie near the transition region. The tail is a **gate-control channel**: tiny geometric energy, order-one computational change.

Even in the linear case (F(h) = Ah + b), catastrophe is possible if ‖AQ‖ is non-negligible relative to the task margin. Nonlinearity is not required for the gap. But SiLU makes the gap **generic and worse** by turning the tail into a gate controller.

---

## 3. The Jacobian Visibility Decomposition

### 3.1 Block Structure

Decompose the Jacobian J_ℓ = ∂F_ℓ/∂h|_{μ_ℓ} with respect to V ⊕ V⊥:

```
J = | A  B |
    | C  K |
```

where:
- A = Π_V J Π_V (visible → visible)
- B = Π_V J Π_{V⊥} (invisible → visible)
- C = Π_{V⊥} J Π_V (visible → invisible)
- K = Π_{V⊥} J Π_{V⊥} (invisible → invisible)

### 3.2 Why the Gram Is Blind to K

If Σ_ℓ ≈ Π_V Σ_ℓ Π_V (covariance concentrated in V), then:

```
Σ_{ℓ+1} ≈ J Σ_ℓ Jᵀ ≈ | AΣA'   AΣC' |
                         | CΣA'   CΣC' |
```

**K does not appear at all.** The Gram evolution sees J restricted to V completely. It sees K not at all. This is the exact mathematical reason Gram-based layer importance can predict deletability (via A, C) while being blind to compressibility (which depends on K).

### 3.3 The Coexistence Regime (E2 + E3)

**For deletability (E2):** The layer's action on the data-varying subspace is small: A ≈ I_V, C small → ΔG small → layer skippable.

**For truncation catastrophe (E3):** K has many singular values of order one (σ_i(K) ≈ 1 for many i). V⊥ is not dead space — it's a preserved carrier. K is well-coupled to the next layer's readout/gating: sr(D_ℓ W_{up,ℓ} Π_{V⊥}) is large.

**The deepest point:** K does not need to be expansive to be essential. If K ≈ I_{V⊥} on a huge subspace, skipping the layer changes little (E2 holds). But if downstream MLPs read those coordinates, projecting them away destroys the carrier (E3 holds). The invisible block can be "boringly transport-like" and still load-bearing.

---

## 4. The Computational Gram: What the Gram Was Missing

### 4.1 Definition

At each MLP, the model computes:

```
a_p^{(ℓ)} = SiLU(W_up · h_p^{(ℓ)}) ∈ ℝ^{4d}     (= ℝ^{8192} for 3B)
```

This is the **activation pattern** — which neurons fire and how strongly, for problem p at layer ℓ. Two problems identical in the representational Gram could have completely different activation patterns.

The **computational Gram matrix:**

```
G_ℓ^{comp} = Ã_ℓ Ã_ℓᵀ ∈ ℝ^{N×N}
```

where Ã_ℓ is centered: ã_p = a_p - (1/N)Σa_p.

### 4.2 Why rank(G^comp) ≫ rank(G^repr) Is Generically Expected

Since rank(H̃) = r, we can write h_p = μ + V x_p where V ∈ ℝ^{d×r}, x_p ∈ ℝ^r.

Then:
```
a_p = σ(W_up h_p) = σ(b + M x_p)
```
where b = W_up μ ∈ ℝ^{4d}, M = W_up V ∈ ℝ^{4d×r}.

Each activation is: a_{ip} = σ(b_i + m_i' x_p). This is a **one-hidden-layer nonlinear feature lift** of an r-dimensional latent variable.

**The polynomial vs non-polynomial dichotomy (GPT 5.4):**

If σ were polynomial of degree q, rank would be bounded by C(r+q, q). But SiLU is non-polynomial analytic. Its Taylor expansion has infinitely many nonzero orders. After composing with affine forms, you generate arbitrarily high-degree monomials. **Generically:**

```
rank(A) = min(4d, N) = min(8192, 1400) = 1400
```

even when the representational latent dimension is only r = 20.

**The conjecture r_90(G^comp) ≫ r_90(G^repr) is not just plausible — it is the generic expectation** unless preactivations are in a degenerate regime.

### 4.3 What the Gap Measures

```
rank_90(G^comp) - rank_90(G^repr) = dimensionality the model USES but the Gram HIDES
```

If rank_90(G^comp) ≈ 200 while rank_90(G^repr) ≈ 20: the model uses 20 dimensions to **describe** problems and 200 dimensions to **process** them. The processing dimensions are invisible to any Gram analysis on hidden states because they exist inside the MLP for one matrix multiplication before being projected back to ℝ^d.

If rank_90(G^comp) ≈ 20: BS failed for a different reason and the activation patterns track the representations faithfully.

---

## 5. The Paradox Index: Unifying Deletability and Compressibility

### 5.1 No Single Scalar From G^repr Alone Can Work (Proof)

Two layers can have identical G^repr_ℓ and identical ΔG^repr_ℓ but different invisible blocks K, hence opposite compressibility behavior. Any scalar built only from representational Grams is provably blind.

### 5.2 The Right Object Is Two Numbers

```
a_ℓ = ‖Π_V (J_ℓᵀ J_ℓ - I) Π_V‖_{F,Σ_ℓ}
```
Measures how much the layer changes geometry on V (the problem-varying subspace).

```
b_ℓ = sr(D_ℓ W_{up,ℓ} Π_{V⊥})
```
Measures how many invisible directions are read by the MLP near the operating point.

- Small a_ℓ → layer is deletable.
- Large b_ℓ → Gram-rank truncation is destructive.

### 5.3 The Substrate-to-Geometry Ratio (If You Insist on One Scalar)

```
S_ℓ = sr(D_ℓ W_{up,ℓ} Π_{V⊥}) / (‖Π_V (J_ℓᵀ J_ℓ - I) Π_V‖_{F,Σ} + ε)
```

- **Large S_ℓ**: layer looks skippable by Gram but is internally dimension-rigid. The paradox regime.
- **Small S_ℓ**: either visibly important or the invisible carrier is weak.

S_ℓ is not an importance scalar. It's a **paradox index**: detects the E2+E3 coexistence regime.

### 5.4 Cheap Proxy (No Jacobian Required)

```
S̃_ℓ = r_τ(G_ℓ^comp) / (r_τ(G_ℓ^repr) · (‖G_{ℓ+1}^repr - G_ℓ^repr‖_F + ε))
```

High computational rank, low representational motion. Cruder but operationally appealing.

---

## 6. Proposed Experiments

### Experiment C1: Computational Gram Matrix (PRIORITY)

**Objective:** Measure rank_90(G^comp) at every layer and compare to rank_90(G^repr).

**Method:**
1. Forward pass with hooks on `model.layers[ℓ].mlp.up_proj` (or `gate_proj` for gated MLPs).
2. Extract a_p^{(ℓ)} = SiLU(W_up · h_p^{(ℓ)}) ∈ ℝ^{4d} for all 1400 problems, all layers.
3. Center: ã_p = a_p - mean(a).
4. Build G^comp_ℓ = Ã Ãᵀ ∈ ℝ^{1400×1400}.
5. Eigendecompose. Compute rank_50, rank_90 at each layer.
6. Compare rank trajectory to G^repr.
7. Compute Lyapunov spectrum of G^comp through depth.

**What it tells us:**
- If rank_90(G^comp) ≈ 200: the 180-dimension gap is where BS was blind. The model processes in high-dimensional activation space but represents in low-dimensional hidden-state space.
- If rank_90(G^comp) ≈ 20: the rank gap is not in the activations. BS failed because of something else (gate sensitivity, error amplification).
- If the Lyapunov phases diverge between G^comp and G^repr (e.g., G^repr compresses at L9 but G^comp keeps expanding): smoking gun that computation and representation live in different spaces.

**Cost:** One forward pass with hooks on MLP intermediates. ~10 minutes on RayGun.

**Note on Qwen2.5 MLP architecture:** Qwen2.5 uses a gated MLP: the "up_proj" gives the gate, "gate_proj" gives the value, and the activation is SiLU(gate) * value. The correct activation to extract is: `SiLU(gate_proj(h)) * up_proj(h)` — the full intermediate before W_down projects back. Need to verify exact attribute names in the model.

### Experiment C2: Jacobian Visibility Decomposition

**Objective:** Measure the spectral structure of the invisible Jacobian block K.

**Method:**
1. At 3-4 representative layers (one per Lyapunov phase), compute the full Jacobian J_ℓ = ∂F_ℓ/∂h|_{μ_ℓ} via `torch.autograd.functional.jacobian`.
2. Compute V = top-20 PCs of H̃_ℓ (the data subspace).
3. Extract:
   - A = Π_V J Π_V (visible block)
   - K = Π_{V⊥} J Π_{V⊥} (invisible block)
4. SVD of each block.
5. If σ(K) ≈ 1 for many singular values: V⊥ is carrier (transport, no computation).
6. If σ(K) has spread: model actively computes in V⊥.

**Cost:** Full 2048×2048 Jacobian is expensive but feasible for 3-4 layers. ~30 min per layer.

**What it tells us:**
- Whether the invisible subspace is infrastructure (identity-like K) or active computation (structured K).
- Directly measures the quantities in the (a_ℓ, b_ℓ) diagnostic pair.

---

## 7. Previous Session State (BS/BS2 Context)

### State ID: 42870062 (April 7, 2026)

**Key results already established:**
- BS: 23 conditions, all 0/20. Even k=500 on single layer = 0/20.
- BS2: 8 conditions, all 0/20. Separating prefill/gen makes no difference.
- BS2 detailed: gen_only preserves EN scaffolding, kills ZH identity. Prefill-only also 0/20.
- PCA bases cached to `output/pca_centered_bases.npz`.
- Z-embedding thesis fully dead.

**Files:**
- `expBS_svd_truncation.py`, `output/expBS_svd_truncation.json`
- `expBS2_genonly_truncation.py`, `output/expBS2_genonly_truncation.json`
- `output/pca_centered_bases.npz` (cached centered PCA bases)

---

## 8. The Evidence Chain (Full Arc Through BS)

```
Z exists (Exp Z)
  → Z is linear (AB)
  → Z rotates chaotically at 77°/layer (BN)
  → Gram matrix is the right abstraction (BQ) — coordinate-free
  → Lyapunov reveals funnel: build→compress→sustain→expand (BQ2)
  → Gram dynamics predict which layers to SKIP (BQ3) — CAUSAL
  → Diverse tasks same funnel (BR) — ARCHITECTURAL
  → 4-model replication (BQ2-XM) — UNIVERSAL across Qwen family
  → 1/20 compress is invariant
  → Output rupture = tied-embedding artifact

  → BUT: Gram rank does NOT predict which DIMENSIONS to keep (BS/BS2)
  → Low-rank Gram ≠ low-rank computation
  → V ⊕ V⊥ decomposition explains both BQ3 success and BS failure
  → Next: measure the computational Gram (C1) and invisible Jacobian (C2)
```

---

## 9. What Von Neumann Would Care About

The central mathematical insight is a **decoupling of relational geometry from computational substrate:**

> ℝ^d = V_ℓ ⊕ V_ℓ^⊥

V_ℓ is the low-rank problem-differential geometry seen by the representational Gram. V_ℓ^⊥ is a high-dimensional carrier subspace that contributes little covariance but is still interrogated by W_up through the nonlinear gate D_ℓ = diag(σ'(W_up μ_ℓ)).

Layer deletion depends mainly on the visible block A = Π_V J Π_V.
Truncation failure depends on the invisible readout load sr(D_ℓ W_{up} Π_{V⊥}).

The paradox is not a contradiction but a **decoupling**. The Gram matrix was the right tool all along — it just measures one half of a two-part story. C1 measures the other half.

This is the transformer's version of the distinction between the **spectrum of a linear operator** (eigenvalues, which determine stability) and the **range of an operator** (which dimensions it can reach, which determine controllability). In control theory, a system can be stable (all eigenvalues in the unit disk) but uncontrollable (the range doesn't span the state space). Here, the Gram is the stability analysis, and C1/C2 are the controllability analysis.

---

## 10. Paper Implications

The BS failure is not a setback. It's the second act of the story.

**Act 1:** The Gram matrix reveals universal low-rank relational geometry and predicts layer-level pruning. (BQ through BQ2-XM, BR, BQ3.)

**Act 2:** But the Gram is fundamentally blind to within-layer compressibility. SVD truncation fails catastrophically. The V ⊕ V⊥ decomposition explains why: the computation lives in the carrier subspace that the Gram can't see. (BS, BS2, theoretical framework.)

**Act 3 (pending):** The computational Gram and Jacobian visibility decomposition measure the other half. The substrate-to-geometry ratio S_ℓ unifies both phenomena. (C1, C2.)

This is a stronger paper than "we found low-rank geometry and it predicts pruning." It's a paper about the **limits** of representational analysis — when and why looking at hidden states tells you about the computation, and when it doesn't. That's a contribution to methodology, not just empirics.
