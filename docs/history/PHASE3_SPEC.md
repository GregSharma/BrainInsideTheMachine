# Phase 3: Causal Identification of Z

## Model
Qwen2.5-3B (`Qwen/Qwen2.5-3B`), d=2048, L=36, h=16, GQA=2, d_head=128.

## What We Know (from Phases 0–2)

| Finding | Source | Numbers |
|---------|--------|---------|
| L33 attention heads collapse to eff rank ~78 (CV < 0.08) | 1.py | mean 80.4, std 6.0 |
| FFN actively avoids attention subspace at L32–L33 | 2.py | 0.57× chance, std 0.0001 |
| L32↔L33 similarity = 0.482 (2× any other pair) | 2.py | multi-head confirmed |
| L33→L34 sharpest phase boundary in network | 2.py | drop 0.452 |
| Z separates better than Z⊥ for mean-pooling (12/12 configs) | Phase 2 | ratio_Z < ratio_Zp |
| Language energy in Z below random at k=20–50 | Phase 2 | 80–86% of k/d |
| Effect vanishes at k=78 | Phase 2 | E_Z ≈ E_rand |
| Last-token pooling reverses the effect | Phase 2 | ratio_Z > ratio_Zp |
| Best config: L32, multi-head, k=50, mean-pool | Phase 2 | ratio_Z=0.730, ratio_Zp=0.824 |
| No behavioral zh>en asymmetry at 3B | Phase 0 | 12 zh, 13 en |

## What Phase 3 Must Answer

**Primary question:** Is Z causal? Does replacing Z-content at L32 change model behavior in a predictable, directional way?

**Operationally, "done" looks like this table, filled in for N ≥ 15 problems:**

| Problem | Original output (en) | Z-patched output (en←zh) | Z⊥-patched output (en←zh) | Full-patched (all dims) |
|---------|---------------------|--------------------------|---------------------------|-------------------------|
| ... | ... | ... | ... | ... |

With these derived metrics:
1. **Z-patch answer change rate**: fraction of problems where Z-patching changes the final numerical answer
2. **Z⊥-patch answer change rate**: same for Z⊥
3. **Z-patch language preservation rate**: fraction where output language remains English after Z-patch
4. **Z⊥-patch language disruption rate**: fraction where output language shifts after Z⊥-patch
5. **Full-patch comparison**: does full replacement behave more like Z-patch or Z⊥-patch?

**Success (double dissociation):**
- Z-patch changes answers but preserves language
- Z⊥-patch changes/disrupts language but preserves answers

**Partial success:**
- Z-patch changes answers more than Z⊥-patch does (even without clean language effects)

**Null:**
- No difference between Z-patch and Z⊥-patch effects

---

## Experiment A: Activation Patching

### A.1 Setup

Let P = {(p_zh^i, p_en^i)}_{i=1}^N be N paired math problems.

For problem i, define:
- h_zh^i(ℓ) ∈ ℝ^{T_zh × d} — hidden states at layer ℓ from the Chinese prompt
- h_en^i(ℓ) ∈ ℝ^{T_en × d} — hidden states at layer ℓ from the English prompt

where T_zh, T_en are sequence lengths (generally T_zh ≠ T_en).

### A.2 Z Mask Construction

Reuse `build_multi_head_z_mask` from Phase 2. The mask is:

V_Z ∈ ℝ^{k × d} — top-k right singular vectors of the stacked multi-head attention kernel at layer ℓ

Orthogonal projectors:
- P_Z = V_Z^T V_Z ∈ ℝ^{d × d} (onto Z)
- P_{Z⊥} = I_d - P_Z (onto Z-complement)

Use **k = 50** at **layer 32** (best Phase 2 config). Also run k=20 as a secondary check.

### A.3 Pooling for Patch Source

Phase 2 showed mean-pooling works, last-token doesn't. For patching, we need a d-dimensional vector to inject. Define:

z̄_zh^i = (1/T_zh) Σ_t h_zh^i(ℓ)[t,:] ∈ ℝ^d — mean-pooled Chinese hidden state

The Z-component of this:
z̄_zh^i|_Z = P_Z · z̄_zh^i ∈ ℝ^d

### A.4 Patching Protocol

**Z-patch**: During English forward pass at layer ℓ, for each token position t:
```
h_en^i(ℓ)[t,:] ← P_{Z⊥} · h_en^i(ℓ)[t,:] + z̄_zh^i|_Z
```
Keep the English token's Z⊥-content (language). Replace its Z-content with the Chinese mean Z-content.

**Z⊥-patch** (control): Same structure, swap roles:
```
h_en^i(ℓ)[t,:] ← P_Z · h_en^i(ℓ)[t,:] + z̄_zh^i|_{Z⊥}
```
Keep reasoning (Z). Replace language scaffold (Z⊥) with Chinese.

**Full-patch**: Replace everything:
```
h_en^i(ℓ)[t,:] ← z̄_zh^i
```

**No-patch baseline**: Run English prompt unmodified.

### A.5 What to Measure

For each of the 4 conditions × N problems, record:
1. **Raw output** (full generated text, ≤ 100 tokens)
2. **Extracted numerical answer** (regex or manual)
3. **Output language** (classify as en/zh/mixed — can use a simple heuristic: fraction of CJK characters)
4. **Answer correctness** (compare to known answer)

### A.6 Degrees of Freedom (where creativity lives)

**Layer choice.** Primary: L32. Secondary: L33. The spec says to run both and compare. If results differ, that's informative — L32 is the "approach" layer, L33 is the bottleneck. Different patching effects would tell you something about the encode→bottleneck transition.

**How to handle the mean-pooled vector.** The spec above broadcasts the mean Chinese Z-content to all English token positions. An alternative: per-token patching using optimal transport alignment between Chinese and English token sequences. Specifically:

Let C_{ij} = ||h_zh^i(ℓ)[s,:] - h_en^i(ℓ)[t,:]||² be the cost matrix. Solve the OT problem:
```
π* = argmin_{π ∈ Π(a,b)} Σ_{st} π_{st} C_{st}
```
where a = (1/T_zh)·1, b = (1/T_en)·1 are uniform marginals. Then patch token t with the transport-weighted combination:
```
z_source(t) = P_Z · Σ_s (π*_{st} / b_t) · h_zh^i(ℓ)[s,:]
```

This is mathematically cleaner but harder to implement. Either approach is acceptable. If the mean-broadcast works, OT is unnecessary. If mean-broadcast shows no effect, try OT before declaring null.

**The k parameter.** Run k ∈ {20, 50}. Not 78 (Phase 2 showed signal vanishes there).

**Prompt set.** Reuse the 20 pairs from Phase 2, or write new ones. If reusing, the results are directly comparable to Phase 2's distance metrics. If writing new ones, you get independence from Phase 2 data. Either is fine — the Phase 2 prompts are clean and cover 10 categories.

---

## Experiment B: Residual Update Decomposition

This is ~20 lines of code piggybacked on the activation extraction from Experiment A. It answers: **what is each layer doing — stripping language, computing reasoning, or both?**

### B.1 Definition

For each layer transition k → k+1 and each prompt, the residual update is:

Δh_k = h(k+1) - h(k) ∈ ℝ^{T × d}

Mean-pool across tokens:

Δh̄_k = (1/T) Σ_t Δh_k[t,:] ∈ ℝ^d

Project onto Z and Z⊥ using L32's basis (fixed — the "Rosetta Stone"):

||Δh̄_k|_Z|| = ||P_Z · Δh̄_k||
||Δh̄_k|_{Z⊥}|| = ||P_{Z⊥} · Δh̄_k||

### B.2 The Ratio

For each layer k, averaged across all prompts and both languages:

R(k) = ||Δh̄_k|_Z|| / ||Δh̄_k|_{Z⊥}||

### B.3 What to Plot

One plot: R(k) vs layer index k, for k = 0, 1, ..., 34.

**Interpretation:**
- R(k) >> 1 at some layer → that layer is primarily computing in Z (reasoning step)
- R(k) << 1 → that layer is primarily modifying Z⊥ (language processing)
- R(k) ≈ 1 everywhere → reasoning and language processing are always interleaved, never separated

**The most profound outcome:** If R(k) is NEVER strongly Z-dominated, then Z is emergent — it's the cumulative effect of 30+ layers of mixed computation, not a discrete phase. This would mean "the model doesn't have a reasoning step; it has a reasoning gradient."

### B.4 Extension: Cross-Lingual Asymmetry

Compute R(k) separately for Chinese and English prompts. If R_zh(k) ≠ R_en(k) at specific layers, those layers process the two languages differently. This connects back to the wrapper-cost model from Chat_2.

---

## Experiment C: ARD-MMD Z Extraction (the full kernel approach)

This is the original Gameplan.md idea, now informed by structural priors. It's independent of Experiments A and B — it discovers Z from data rather than weights.

### C.1 The Kernel

The ARD-RBF kernel on ℝ^d:

k_ℓ(x, y) = exp(-½ Σ_{j=1}^{d} (x_j - y_j)² / ℓ_j²)

where ℓ = (ℓ_1, ..., ℓ_d) ∈ ℝ_{>0}^d is the lengthscale vector. Each dimension gets its own scale.

### C.2 The MMD

Given two empirical measures μ = (1/n) Σ δ_{x_i} and ν = (1/m) Σ δ_{y_j} (token activations from Chinese and English), the unbiased MMD² estimator is:

MMD²(μ, ν; ℓ) = [1/(n(n-1))] Σ_{i≠i'} k_ℓ(x_i, x_{i'}) + [1/(m(m-1))] Σ_{j≠j'} k_ℓ(y_j, y_{j'}) - [2/(nm)] Σ_{i,j} k_ℓ(x_i, y_j)

This is zero iff the two token clouds are identically distributed in the kernel's feature space. It handles T_zh ≠ T_en by construction.

### C.3 The Optimization

Minimize total MMD across all prompt pairs with L1 sparsity on inverse lengthscales:

min_{ℓ} Σ_{i=1}^{N} MMD²(μ_zh^i, μ_en^i; ℓ) + λ Σ_{j=1}^{d} ℓ_j^{-1}

Parameterize as log_ℓ_j to ensure positivity. Optimize with Adam.

**What the optimization finds:**
- Dimensions where ℓ_j → ∞: the kernel ignores these. They carry language-specific information (different between zh/en).
- Dimensions where ℓ_j stays finite: the kernel uses these. They carry shared information across languages. **These are Z.**

The sparsity penalty λ Σ ℓ_j^{-1} encourages most dimensions to shut off (ℓ → ∞), yielding a compact Z.

### C.4 Structural Prior (what makes this v3, not v2)

Instead of initializing all log_ℓ = 0, use the SVD-based Z mask from Phase 2:

```
V_Z = build_multi_head_z_mask(model, layer=32, ..., k=50)  # (50, 2048)
# Project each dimension onto the Z subspace
z_energy = (V_Z[:, j]**2).sum() for each j  # how much dim j participates in Z
log_ℓ_init[j] = -log(z_energy[j] + ε)  # small ℓ where Z is active
```

This gives the optimizer a warm start. The SVD mask says "these ~50 dimensions are structurally important." The ARD optimization can confirm, refine, or contradict that.

### C.5 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| λ (sparsity) | Grid search: {0.001, 0.01, 0.1} | λ controls Z size. Too small → all dims active. Too large → Z = ∅. |
| lr | 0.01 | Standard for Adam on log-scale parameters |
| steps | 500–1000 | Monitor convergence; stop when MMD plateaus |
| Target layers | {10, 15, 20, 25, 28, 30, 32, 33, 34, 35} | Structural analysis says these matter |
| d | 2048 | (not 4096 — we're on Qwen2.5-3B) |

### C.6 What to Plot

1. **|Z(k)| vs layer** — number of dimensions with ℓ_j < threshold (e.g., median of converged ℓ values) at each layer

2. **Lengthscale spectrum at L32** — histogram of log₁₀(ℓ_j) and sorted curve. Looking for bimodality (clean Z) vs power-law (graded) vs uniform (no Z found).

3. **Overlap between SVD-Z and ARD-Z** — what fraction of ARD's surviving dimensions are also in the SVD mask's top-50? If high → weight-based and data-based approaches agree (strong evidence). If low → the approaches find different subspaces (interesting but complicates the story).

### C.7 Why Do Both SVD-Z and ARD-Z?

SVD-Z (Experiments A–B) derives Z from model weights alone — no data needed. It's structurally motivated and fast.

ARD-Z (Experiment C) derives Z from cross-lingual activation data — no weight analysis needed. It's statistically principled and self-calibrating.

If they agree, that's powerful: the model's static architecture and its dynamic behavior both point to the same subspace. If they disagree, the disagreement itself is informative — it tells you whether Z is a property of the weights or of the computation.

---

## Experiment D: The Bridge

Only run this if Experiments A or C succeed. It answers: **is the mapping between languages within Z a simple linear transform?**

### D.1 Setup

Collect paired Z-projected mean activations at L32:

z_zh^i = P_Z · z̄_zh^i ∈ ℝ^d (only the k nonzero components matter)
z_en^i = P_Z · z̄_en^i ∈ ℝ^d

Stack into matrices:
- Z_zh ∈ ℝ^{N × k} (extracting the k active components)
- Z_en ∈ ℝ^{N × k}

### D.2 Linear Bridge

Solve: W* = argmin_W ||Z_zh - Z_en · W||²_F

Closed form: W* = (Z_en^T Z_en)^{-1} Z_en^T Z_zh

Size: k × k. For k=50: 2,500 parameters. For k=20: 400 parameters. Negligible overfitting risk with N=20 pairs.

### D.3 Metrics

1. **R²** = 1 - ||Z_zh - Z_en · W*||²_F / ||Z_zh - Z̄_zh||²_F
   - R² > 0.9 → languages are near-rotations of each other in Z
   - R² > 0.8 → thin wrappers
   - R² < 0.5 → relationship is nonlinear

2. **Orthogonality error** = ||W*^T W* - I_k||_F / k
   - Small → W* is approximately a rotation (isometric languages in Z)
   - Large → W* is a more general linear map (one language uses Z-dims differently)

3. **SVD of W*** — singular value spectrum
   - Flat spectrum → rotation-like
   - Steep decay → some Z-dims matter more than others for the mapping

4. **Leave-one-out cross-validation R²** — can the bridge predict held-out pairs?

---

## Compute Budget

| Experiment | Estimated Time | GPU Needed | Dependencies |
|------------|---------------|------------|--------------|
| A (Patching) | 2–3 hours | Yes (forward passes + generation) | Z mask from Phase 2 |
| B (Update decomposition) | 30 min | Piggybacks on A's activation extraction | A's hidden states |
| C (ARD-MMD) | 3–5 hours | Yes (gradient through kernel, 500 steps × 10 layers) | Fresh activations |
| D (Bridge) | 15 min | No (just linear algebra) | A or C must succeed |

**Recommended execution order:** A+B together (share activations), then D if A works, then C independently.

C is the most expensive but also the most self-contained — it doesn't need Phases 0–2 at all. If you're short on time, A+B+D is the core contribution. C is the luxury version that provides a second, independent path to Z.

---

## What "Done" Looks Like

### Minimum: answer the causal question
- The patching table (Experiment A) is filled in for ≥15 problems
- R(k) vs layer plot exists (Experiment B)
- You can state: "Z-patching changes answers X% of the time while preserving language Y% of the time, vs Z⊥-patching which changes answers X'% and disrupts language Y'%"

### Strong: double dissociation + data-driven confirmation
- All of minimum, plus:
- ARD lengthscale spectrum exists (Experiment C)
- SVD-Z and ARD-Z overlap is quantified
- Bridge R² exists (Experiment D)

### Gold: the full story
- All of strong, plus:
- Results hold at both L32 and L33
- OT-aligned patching tried (if mean-broadcast was weak)
- R(k) decomposition reveals the encode/reason/decode phases
- Bridge is approximately orthogonal (languages are rotations in Z)
