# Mathematical Observations

**Purpose.** This document records, in formal notation, the mathematical structure underneath the empirical findings of the BrainInsideTheMachine project. It is not an experiment log and not a paper draft. It is the spine — the claims that need to be either derived, verified, or honestly labeled as conjectural before any paper narrative can close. Greg asked for this after we agreed that the compressibility / glue-vs-content framing is not yet coherent and paper writing is deferred.

**Status tags.** Every labeled claim carries one of:
- **[T]** theorem or derivation — proved on paper or in published literature
- **[O]** observation — measured empirically in this project with specific experiment ID
- **[C]** conjecture — stated but not derived or verified
- **[D]** definition — a naming convention, not a claim

Scope: Qwen2.5-3B unless otherwise noted. Extensions to 7B / 14B / Qwen3-8B are flagged.

---

## 1. Setup and notation

**[D1.1] Model.** Let $\mathcal{M}$ be a decoder-only transformer with $L$ layers, hidden dimension $d$, vocabulary $V$. For Qwen2.5-3B: $L=36$, $d=2048$, $|V|=151{,}936$, tied input/output embeddings, RoPE, SwiGLU MLPs, GQA attention.

**[D1.2] Forward pass.** Given a token sequence $x_{1:T}$ with token embeddings $e_{1:T} \in \mathbb{R}^{T \times d}$, define residual states $h^{(\ell)} \in \mathbb{R}^{T \times d}$ for $\ell = 0, 1, \ldots, L$ by
$$
h^{(0)} = e_{1:T}, \qquad h^{(\ell+1)} = h^{(\ell)} + \mathrm{attn}^{(\ell)}(h^{(\ell)}) + \mathrm{mlp}^{(\ell)}(h^{(\ell)} + \mathrm{attn}^{(\ell)}(h^{(\ell)})).
$$
We abuse notation slightly and write $h^{(\ell)}_t \in \mathbb{R}^d$ for the state at position $t$ in layer $\ell$, and $h^{(\ell)}_{\text{last}}$ for $h^{(\ell)}_T$.

**[D1.3] Per-layer deltas.** The attention and MLP contributions at layer $\ell$, position $t$ are
$$
\Delta^{\text{attn},(\ell)}_t := \mathrm{attn}^{(\ell)}(h^{(\ell)})_t, \qquad \Delta^{\text{mlp},(\ell)}_t := \mathrm{mlp}^{(\ell)}(h^{(\ell)} + \mathrm{attn}^{(\ell)}(h^{(\ell)}))_t,
$$
and the total per-layer delta is $\Delta^{(\ell)}_t := \Delta^{\text{attn},(\ell)}_t + \Delta^{\text{mlp},(\ell)}_t = h^{(\ell+1)}_t - h^{(\ell)}_t$.

**[D1.4] Problem ensemble.** Let $\mathcal{P} = \{p_1, \ldots, p_N\}$ be a set of $N$ prompts (in this project, $N=20$ math problems unless otherwise stated). For each problem $p_i$ and each layer $\ell$, define the last-token state $H^{(\ell)}_i := h^{(\ell)}_{\text{last}}(p_i) \in \mathbb{R}^d$. Let $H^{(\ell)} \in \mathbb{R}^{N \times d}$ be the matrix with rows $H^{(\ell)}_i$.

**[D1.5] Centered Gram.** The centered Gram matrix at layer $\ell$ is
$$
G^{(\ell)} := \tilde{H}^{(\ell)} (\tilde{H}^{(\ell)})^\top \in \mathbb{R}^{N \times N}, \qquad \tilde{H}^{(\ell)} := H^{(\ell)} - \frac{1}{N}\sum_i H^{(\ell)}_i.
$$
Define $\mathrm{rank}_{90}(G^{(\ell)}) := \min\{k : \sum_{j=1}^{k} \lambda_j \geq 0.9 \sum_{j=1}^{N} \lambda_j\}$ where $\lambda_j$ are eigenvalues of $G^{(\ell)}$ in decreasing order.

---

## 2. The central empirical objects

**[O2.1] Centered Gram rank trajectory (needs careful statement).** For Qwen2.5-3B, measured on $N=20$ bilingual math problems:
- $\mathrm{rank}_{90}(G^{(0)}) = 8$
- $\mathrm{rank}_{90}(G^{(9)}) = 19$
- $\mathrm{rank}_{90}(G^{(20)}) = 21$
- $\mathrm{rank}_{90}(G^{(35)}) = 10$

The trajectory at $N=20$ math is: build $(L_0 \to L_9)$ → equilibrium $(L_9 \to L_{25})$ → reconcentration $(L_{25} \to L_{35})$. Replicates across Qwen2.5-7B, Qwen2.5-14B, Qwen3-8B at the same $N=20$ setting (expBQ2-XM).

**However**, the centered Gram rank is *language-dependent* and the "funnel" shape is partially a small-N artifact. Measured on $N=200$ diverse reasoning problems per language (exp centered_gram_n200, ran this session):

| Layer | en | zh | es | ar | ja | sw | ko | pooled ($N$=1400) |
|---|---|---|---|---|---|---|---|---|
| L0  | 20 | 9  | 25 | 17 | 12 | 23 | 14 | 9  |
| L9  | 15 | 13 | 21 | 17 | 14 | 23 | 14 | 38 |
| L17 | 12 | 15 | 19 | 21 | 15 | 19 | 17 | 46 |
| L20 | 14 | 17 | 22 | 24 | 16 | 22 | 18 | 54 |
| L25 | 14 | 16 | 20 | 23 | 16 | 21 | 17 | 50 |
| L30 | 16 | 18 | 16 | 11 | 18 | 10 | 18 | 32 |
| L35 | 26 | 23 | 27 | 16 | 22 | 14 | 32 | 19 |

Key observations:
- **Per-language rank_90 at equilibrium is 12-24, roughly constant in N** (N=20 math matched N=200 diverse within factor 1.3 for en, zh).
- **Pooled rank_90 scales approximately linearly with number of languages** (7 langs × ~13 per lang → pooled ~45 at equilibrium).
- **The "funnel" shape (rank up at equilibrium, down at output) is robust at N=20 math but does not generalize cleanly to N=200 diverse per language.** English and Spanish show rank_90 *higher* at L0 than at equilibrium.
- **rank_99 at N=200 per-language is 65-100**; at pooled N=1400 it is 200-350. The tail 10% of variance contains problem-specific information that scales with N.

Source: `output/centered_gram_n200.json`, `output/expBQ2_crossmodel_lyapunov.json`.

**[O2.2] Causal delta-Gram pruning (BQ3).** Rank layers by $\|G^{(\ell+1)} - G^{(\ell)}\|_F$ and delete the five layers with the smallest delta. Math accuracy preserved at baseline levels (20/20 → 20/20 on the $N=20$ test set). Layer deletion is robust; PCA dimension truncation (BS) is catastrophic at any $k \leq 500$. Source: `output/expBQ3_lyapunov_pruning.json`.

**[O2.3] Read-head asymmetry (C2/C2b/C2c).** For each layer $\ell$, decompose $h^{(\ell)}_{\text{last}}$ as $V_\parallel \oplus V_\perp$ where $V_\parallel := \mathrm{span}(H^{(\ell)})$ is the $\mathrm{rank}_{90}$ subspace and $V_\perp$ is its orthogonal complement in $\mathbb{R}^d$. Replacing $V_\perp$ at all $L$ layers **at the last token only** preserves baseline accuracy (and sometimes improves it — Qwen2.5-14B EN: 5→11). Replacing $V_\perp$ **at all tokens** for even a single layer destroys generation (0/20). Replicated across Qwen2.5-3B (tied), Qwen2.5-7B (tied), Qwen2.5-14B (untied), Qwen3-8B (untied). Source: `output/expC2*.json`.

**[O2.4] PACF consecutive delta cosine (phase transition, Exp T).** For math problems, define $c^{(\ell)} := \langle \Delta^{\text{mlp},(\ell)}_{\text{last}}, \Delta^{\text{mlp},(\ell-1)}_{\text{last}} \rangle / (\|\cdot\|\|\cdot\|)$ averaged across problems. Measured:
- $\ell \in [9, 17]$: $c^{(\ell)} \in [-0.12, -0.05]$ (adversarial)
- $\ell \in [18, 21]$: $c^{(\ell)} \in [+0.17, +0.27]$ (cooperative)
- $\ell = 22$: $c^{(22)} \approx +0.05$ (reset)
- $\ell \in [23, 26]$: ramp to cooperative again
Non-math problems are cooperative everywhere ($c^{(\ell)} \in [+0.06, +0.39]$).

**[O2.5] Attention entropy at content vs glue tokens (late layers).** Let $A^{(\ell)}_{t,s}$ denote the head-averaged attention weight from position $t$ to position $s$ at layer $\ell$, restricted to the static prompt (BOS dropped, renormalized). Let $\mathcal{G}, \mathcal{C}$ be the sets of glue and content token positions, labeled by surprisal percentile and tokenizer heuristic. Define token entropy $H^{(\ell)}_t := -\sum_s A^{(\ell)}_{t,s} \log A^{(\ell)}_{t,s}$. Then at layers $\ell \in \{32, 34, 35\}$:
- $\mathbb{E}[H^{(\ell)}_t \mid t \in \mathcal{G}] - \mathbb{E}[H^{(\ell)}_t \mid t \in \mathcal{C}] \in [+0.04, +0.19]$ nats
- $p < 10^{-4}$ (permutation test over problems)
- 37/37 problems show positive delta at $\ell = 35$
The effect is absent or reversed for $\ell \leq 20$ and concentrated in $\ell \in [20, 35]$. Source: `output/exp_attention_anatomy_3b.json`.

**[O2.6] Cocycle universality (5 Qwen models).** For certain operator compositions across layer pairs (exact definition in expBA / crossmodel validation), the composition law is approximately multiplicative: the observed composition matches the product of per-layer operators with $R^2 > 0.87$ and Frobenius error $< 1.04\%$. Holds across Qwen2.5-3B, 7B, 14B, Qwen3-8B, and a fifth Qwen variant. Source: `docs/crossmodel_validation_2026-04-06.md`.

---

## 3. The V⊕V⊥ decomposition and what it does (and does not) predict

**[D3.1] Parallel and perpendicular subspaces.** At layer $\ell$, let $U_\parallel \in \mathbb{R}^{d \times k_\ell}$ be the left singular vectors of $\tilde{H}^{(\ell)}$ corresponding to the top $k_\ell$ singular values where $k_\ell = \mathrm{rank}_{90}(G^{(\ell)})$. Let $P_\parallel := U_\parallel U_\parallel^\top$ and $P_\perp := I - P_\parallel$. For any residual state $h \in \mathbb{R}^d$, write $h = h_\parallel + h_\perp := P_\parallel h + P_\perp h$.

**[T3.2] Jacobian block structure (from docs/2026-04-07_observation_intervention_gap.md).** Let $f^{(\ell)}: \mathbb{R}^d \to \mathbb{R}^d$ be the layer map $h \mapsto h + \Delta^{(\ell)}(h)$. In the basis $(P_\parallel, P_\perp)$, the Jacobian $J^{(\ell)} = \partial f^{(\ell)} / \partial h$ has block structure
$$
J^{(\ell)} = \begin{pmatrix} I_{k_\ell} + A^{(\ell)} & B^{(\ell)} \\ C^{(\ell)} & I_{d-k_\ell} + K^{(\ell)} \end{pmatrix}
$$
where $A$ is the parallel-to-parallel block, $B$ the perpendicular-to-parallel read, $C$ the parallel-to-perpendicular write, and $K$ the perpendicular substrate. **This is an identity of partial derivatives, not a claim about training.** Per-block norms must be measured, not assumed.

**[O3.3] Empirical block norms.** Not measured in this project. Flagged as experiment α.1.

**[C3.4] Asymmetric-computation conjecture.** For a *trained* model on in-distribution prompts, the empirical block norms satisfy $\|B^{(\ell)}\|$ is small relative to $\|K^{(\ell)}\|$ at last-token positions but not at context positions. This is the formal statement of "asymmetric computation" that the read-head result O2.3 implies. It has not been directly measured.

**[C3.5] Why V⊕V⊥ does not predict 20D.** The decomposition in D3.1 defines $k_\ell$ as $\mathrm{rank}_{90}(G^{(\ell)})$, which is itself an empirical quantity measured at $0.9$ variance threshold across an $N=20$ problem ensemble. The number 20 is not derived from training dynamics, architecture, or information-theoretic limits. **It is a free parameter of the measurement.** Specifically:
- Changing $N$ changes $k_\ell$ monotonically up to $N$.
- Changing the variance threshold changes $k_\ell$.
- Changing the problem distribution (math vs diverse reasoning) changes the basis of $U_\parallel$ but not its rank.

The coincidence that C2c-read-head accuracy is preserved when you zero $V_\perp$ at the last token, at the same rank where $\mathrm{rank}_{90} \approx 20$, is **not** a derivation of the 20. It is a demonstration that *whatever the effective read-head dimension is*, it sits inside the $\mathrm{rank}_{90}$ subspace. The effective read-head dimension could be smaller.

**[C3.6] Open question.** What is the minimal $k$ such that projecting the last-token state onto the top-$k$ subspace of $\tilde{H}^{(\ell)}$, at all layers, preserves math accuracy? If this number is $\ll 20$, the read head is strictly lower-rank than the Gram. If it equals 20, they coincide. If it equals $d$, the read head's effective rank is unbounded by $\mathrm{rank}_{90}(G^{(\ell)})$ and the coincidence is spurious. **This experiment has not been run.**

---

## 4. What "delta norm at token class" estimates

**[D4.1] Null hypothesis.** Let $\mathcal{G}, \mathcal{C} \subset \{1, \ldots, T\}$ be disjoint sets of glue and content token positions in a prompt. The test statistic for Experiment 1 is
$$
T^{(\ell)}_{\text{sublayer}} := \mathbb{E}[\|\Delta^{\text{sublayer},(\ell)}_t\| \mid t \in \mathcal{G}] - \mathbb{E}[\|\Delta^{\text{sublayer},(\ell)}_t\| \mid t \in \mathcal{C}]
$$
for $\text{sublayer} \in \{\text{attn}, \text{mlp}\}$ and $\ell \in \{0, \ldots, L-1\}$.

**[D4.2] Permutation null.** Within each problem, randomly reassign glue/content labels to token positions, preserving the marginal count $|\mathcal{G}|, |\mathcal{C}|$. Recompute $T^{(\ell)}_{\text{sublayer}}$. Repeat $R$ times. Report two-sided $p$-value.

**[C4.3] Position confound.** Glue tokens are not uniformly distributed over position: "Therefore" appears near the end of a reasoning chain, "the" appears everywhere. If $\|\Delta^{(\ell)}_t\|$ depends on position $t$ (which it likely does — depth-of-computation grows with context length), the raw test T4.1 confounds token class with position. **Correction:** compute $T^{(\ell)}$ within fixed position-bins (e.g., quartiles), or regress $\|\Delta^{(\ell)}_t\|$ on position and compare residuals.

**[C4.4] Frequency confound.** Glue tokens are high-frequency; content tokens are low-frequency. If MLP delta magnitude correlates with input token surprisal (Exp T showed $R^2 \approx 0.03$ for math, so this is probably small but nonzero), the test leaks frequency into the class contrast. **Correction:** regress out unigram log-probability before testing.

**[C4.5] Prompt vs generation confound.** Exp T / exp_attention_anatomy measured generation tokens. The attention entropy finding O2.5 is also on generated text. Delta norms computed on the static prompt are a *different* measurement and may not match. Experiment 1 should measure deltas on **generated tokens**, in parallel with the entropy labeling, not on prompt tokens.

**[C4.6] Interpretation.** Under Interpretation A (computation-light glue):
$$
T^{(\ell)}_{\text{attn}} < 0 \text{ and } T^{(\ell)}_{\text{mlp}} < 0 \text{ at late layers } \ell \in [20, 35]
$$
with roughly equal ratio across sublayers. Under Interpretation B (differently computed glue):
$$
T^{(\ell)}_{\text{attn}} \approx 0, T^{(\ell)}_{\text{mlp}} \approx 0 \text{ in total norm, but the directions } \Delta^{(\ell)}_t \text{ live in disjoint subspaces across classes.}
$$
Interpretation B is distinguishable from A only if we *also* measure $\cos(\bar{\Delta}^{(\ell)}_\mathcal{G}, \bar{\Delta}^{(\ell)}_\mathcal{C})$ and related subspace metrics. **Experiment 1 must include the subspace test, not just the norm test.**

---

## 5. The read-head rank and the centered-Gram rank — RESOLVED (mostly)

**This section was rewritten after running `centered_gram_n200.py` this session. The three claims in the original draft are all partially wrong. The corrected picture below is supported by the data in `output/centered_gram_n200.json`.**

**[O5.1] Numerical starting point.** At $N=20$ bilingual math problems, centered $\mathrm{rank}_{90}(G^{(\ell)}) \in [8, 21]$ at equilibrium layers. The read-head claim from C2c is that zeroing $V_\perp$ at the last token (using $V_\parallel$ defined by this same ensemble) preserves accuracy. We previously said "the read head is about 20D" and treated the numerical agreement between the two as suggestive.

**[O5.2] What the $N=200$ measurement shows (new this session).** Running centered Gram on the 7-language × 200-diverse-problem cache (expBR data, recomputed centered), *per language*:

- Equilibrium rank_90 is language-dependent: en=12-15, zh=13-17, es=18-22, ar=17-24, ja=14-16, sw=19-23, ko=14-18.
- These values are within ~1.3x of the $N=20$ math measurement for the corresponding languages. **Per-language rank_90 is roughly N-invariant** from $N=20$ to $N=200$.
- Pooling across 7 languages, rank_90 jumps to 42-55 at equilibrium. **Pooled rank scales approximately linearly with number of languages**, not with total $N$.
- rank_99 per language is 65-100, pooled is 200-350. **The tail scales with N**.

**[F5.3] Corrected picture — three facts, one interpretation.**

**Fact 1:** Per-language centered rank_90 is approximately N-invariant in the range $N \in [20, 200]$ and sits at roughly 12-24 depending on language. This rules out the "rank is just N-1" interpretation (Claim 1 in the original draft). If rank were ensemble-determined in the pure sense, it would have gone from 20 to 180 when N went from 20 to 200. It did not.

**Fact 2:** Pooled centered rank_90 scales approximately linearly with the number of languages. Going from 1 language to 7 languages multiplies rank_90 by roughly 3-4x (not 7x, which means there is a shared core subspace across languages). This is *exactly* what Claim 5.5 predicted: the rank is language-local with a shared piece.

**Fact 3:** The "funnel" shape (rank up at equilibrium, down at output) is robust at $N=20$ math but does not generalize cleanly to $N=200$ diverse per language. English and Spanish show rank_90 actually *higher* at L0 than at equilibrium. The funnel is partially a small-N-on-math artifact.

**[C5.4] The correct interpretation.** The centered Gram rank measures the dimensionality of the *within-language problem manifold* at the given ensemble. Within a single language, the problem manifold has a training-determined intrinsic dimensionality (~12-24 at rank_90, depending on language) that is insensitive to whether you sample 20 or 200 problems. The read head at the last token operates in a subspace whose rank is at most this intrinsic dimensionality. **The ~20 we kept citing is an EN+ZH average at N=20 that happens to be near the true per-language intrinsic rank for those languages. It is not a universal constant and it is not a pure measurement artifact.**

**[C5.5] What this changes for the paper.** The headline cannot be "20D reasoning subspace." It must be one of:
- **Option A:** "Per-language problem manifold has training-determined intrinsic rank_90 of 12-24, language-dependent, with shared cross-language core subspace at rank ≈ (pooled - single-language-max) ≈ 20-30."
- **Option B:** Drop the rank_90 number from the headline entirely and lead with the read-head causal result (C2c), which is N-independent.
- **Option C:** Keep the 20D claim but scope it to "last-token readout subspace on Qwen-family math" and state explicitly that it is not a universal architectural constant.

My recommendation: **Option B.** The read-head causal result is the clean finding; the rank_90 number is a noisy proxy that depends on language and ensemble choice. Use rank_90 only as a sanity check, not as a headline.

**[C5.6] Still-open sub-questions.**
1. What is the *minimal* $k$ such that top-$k$ projection of the last-token state preserves math accuracy? The C2c experiment only measures "zero V⊥ where V⊥ is the complement of rank_90." It does not sweep $k$. This is a small experiment that would directly measure the read head's operational dimension, distinct from the ensemble's rank.
2. Is the language-specific intrinsic rank related to the PC0 swap finding (language wrapper)? The shared cross-language core we're inferring here and the null-space retrieval in BH at 97% Top-1 should be the same object. This needs to be verified, not assumed.
3. The funnel shape's small-N sensitivity means the BQ3 causal pruning result (skip 5 layers by delta-Gram) should also be rechecked at $N=200$ per language — the delta-Gram ranking may be unstable.

**Greg: this is the section that most changed when I actually ran the numbers. The original three-way fork was miscast. The real answer is "per-language rank is training-local and roughly stable in N; the 20 was an EN+ZH coincidence with the per-language intrinsic scale." The downstream consequence is that I should stop citing the 20 as a universal constant and lean on the read-head causal result instead.**

---

## 6. The cocycle result, restated carefully

**[D6.1] Cocycle condition.** Let $f^{(\ell \to \ell+1)}$ denote the layer operator from $\ell$ to $\ell+1$. Define the composed operator over a layer range as $f^{(\ell \to \ell+k)} := f^{(\ell+k-1 \to \ell+k)} \circ \cdots \circ f^{(\ell \to \ell+1)}$. The *cocycle condition* in the strict sense is
$$
f^{(\ell \to m)} = f^{(r \to m)} \circ f^{(\ell \to r)} \qquad \forall\, \ell \leq r \leq m.
$$
This is an exact identity for the true forward pass by definition of function composition. It is *not* a theorem; it is the definition of "layer stack."

**[O6.2] What expBA actually measured.** The measurement in crossmodel_validation_2026-04-06.md is not the strict cocycle (which is trivially true). It is an *approximation* of the layer operators by linear maps $\hat{f}^{(\ell \to \ell+1)} \approx f^{(\ell \to \ell+1)}$ fit on an ensemble, and a test of whether the composed linear approximation matches the composed true operator at $R^2 > 0.87$. **This is a statement about how well linear approximations compose, not about the cocycle law itself.**

**[C6.3] Correct interpretation.** The expBA result says: *on this ensemble, at this layer range, the layer operators are close enough to linear that their linear approximations compose multiplicatively with $< 1.04\%$ Frobenius error.* Since $R^2$ is computed against the true nonlinear composition, a high $R^2$ implies the nonlinearity is a small perturbation. **What it does not say:** that there is a Lie-group-like structure, that the layer operators commute, or that the residual stream is "almost linear" in the strong sense.

**[C6.4] Load-bearing-ness.** If the paper uses the cocycle result, it must state the actual claim: "layer operators are well-approximated by linear maps at the resolution of problem-level contrasts, and the linear approximations compose with $< 1\%$ error across 5 Qwen models." Any stronger statement is over-interpretation.

**Greg: this is the second place I've been loose. My earlier session said "cocycle R² > 0.87" as a universality banner without explaining what composition was being tested. It's fixable but needs to be stated correctly in any paper.**

---

## 7. Phase transition — a three-stage working interpretation, not a settled theorem

**Framing note.** After draft v1 of this section, GPT Web pushed back with the following: the three-stage story is coherent and integrative, but epistemic load is *very unequal* across the three phases. The late phase (L27-L35, read-tip assembly) is directly supported by causal evidence (C2c, attention entropy, Gram reconcentration). The middle phase (L18-L26, resolution) is plausible by convergent observation but not causally tested. The early phase (L9-L17, fusion) is the weakest — supported mostly by the PACF sign flip (Exp T) and the answer-transfer collapse (Exp Y/Z), both of which admit purely geometric explanations. The right move is to split observation from interpretation from discriminating tests, and grade each phase's epistemic status. That is what this section now does.

### 7A. Observed phase structure

**[O7A.1] Consecutive MLP delta cosine sign flip (Exp T).** Define $\phi^{(\ell)} := \mathbb{E}_i\left[\frac{\langle \Delta^{\text{mlp},(\ell)}_{\text{last}}(p_i), \Delta^{\text{mlp},(\ell-1)}_{\text{last}}(p_i)\rangle}{\|\Delta^{\text{mlp},(\ell)}_{\text{last}}(p_i)\|\|\Delta^{\text{mlp},(\ell-1)}_{\text{last}}(p_i)\|}\right]$. On math problems, $\phi^{(\ell)}$ is negative for $\ell \in [9, 17]$ (values $-0.12$ to $-0.05$), positive for $\ell \in [18, 21]$ (values $+0.17$ to $+0.27$), resets at $\ell=22$, then ramps cooperatively again through $\ell=26$. On non-math problems, $\phi^{(\ell)} > 0$ throughout.

**[O7A.2] Cross-lingual answer-transfer collapse (Exp Y/Z).** On a cross-lingual answer probe (trained on EN answer representations, tested on ZH), transfer peaks at 0.35 at $\ell=2$ and collapses to 0 for $\ell \in [7, 14]$. Partial recovery follows at later layers. This window overlaps substantially with the adversarial phase from O7A.1.

**[O7A.3] Centered Gram trajectory (per-language, from §2 and §5 correction).** Per language at $N=200$, rank_90 has a layer-dependent profile that varies across languages; the "funnel" shape is clean on $N=20$ math but blurs on $N=200$ diverse.

**[O7A.4] Attention entropy split (O2.5, restated).** Content tokens have significantly lower attention entropy than glue tokens at $\ell \in [32, 35]$, effect 37/37 problems at $\ell=35$. Concentrated in $\ell \in [20, 35]$, absent or reversed earlier.

**These four observations partition the layer stack into three empirically distinct regions, roughly $[9, 17]$, $[18, 26]$, $[27, 35]$. This partition is an observed fact. What follows is interpretation.**

### 7B. Mechanistic interpretation (working hypothesis, epistemically graded)

**[C7B.1] Late phase $[27, 35]$ — read-tip assembly. Epistemic status: strong, causally supported.**
Evidence: Gram reconcentration (rank_90 collapses 21→10 at $N=20$ math), attention entropy split 37/37 (O2.5), C2c last-token read-head result across 4 models (O2.3), BQ3 pruning effect concentrated away from this region (cannot safely prune late layers). This is the phase where the read head is assembled from context. Multiple independent evidence streams converge and at least one (C2c) is a direct causal manipulation.

**[C7B.2] Middle phase $[18, 26]$ — resolution / consensus. Epistemic status: moderate, observational-only.**
Evidence: cooperative $\phi^{(\ell)}$ sign (consecutive deltas agree), partial recovery of cross-lingual answer transfer, phase transition in language-related embedding structure at roughly 40-47% depth across 5 Qwen models (crossmodel validation). Interpretation: layers that previously "disagreed" now point in the same direction, suggesting the model is integrating a conclusion. But there is no direct causal test that this phase is doing integration work rather than, say, simply transporting a signal formed earlier. An alternative interpretation is that L18-L26 is a pure transport region and all the integration happened at L17 itself.

**[C7B.3] Early phase $[9, 17]$ — fusion / entanglement. Epistemic status: weak, multiple non-causal interpretations.**
Evidence: adversarial $\phi^{(\ell)}$ sign, answer-transfer collapse, known location of language-math entanglement from Exp V/V3 (no single layer dominates strategy switch). Interpretation: language identity and math content are maximally fused here. **But:** the adversarial sign is also what you would expect from any smooth trajectory at its high-curvature segment (C7C.2 below). The answer-transfer collapse is consistent with "features have been transformed into a form the probe cannot read" without requiring fusion. The "entanglement" language is borrowed from prior sessions' narrative and has not been operationalized.

### 7C. Discriminating tests

**[C7C.1] Experiment β (adversarial quenching).** Scale $\Delta^{(\ell)}_t$ by $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$ for $\ell \in [9, 17]$, all positions. Measure math accuracy vs $\alpha$. Predictions:
- If L9-L17 is doing causal computational work: monotone accuracy decline as $\alpha \to 0$, plateau at 0 for $\alpha \ll 1$.
- If L9-L17 is a geometric segment with no independent causal load: near-invariance of accuracy across $\alpha$, or only a speed-of-convergence effect that vanishes at longer token budgets.
- If L9-L17 is load-bearing only for cross-lingual problems but not monolingual ones: differential drop across EN-only vs ZH-EN-mixed conditions.

**[C7C.2] Null model: smooth-trajectory-curvature.** A smooth curve $\gamma: [0, L] \to \mathbb{R}^d$ produces consecutive-delta cosines that depend only on local curvature: $\cos(\gamma'(\ell), \gamma'(\ell-1)) \approx 1 - \frac{1}{2}\kappa^2$ near straight segments and becomes negative at points of high angular change. Under this null, the $\phi^{(\ell)}$ sign flip is just the trajectory turning. **Falsification:** Experiment β. If accuracy is invariant under quenching, the adversarial sign is curvature, not computation.

**[C7C.3] Experiment β-prime (boundary localization).** Take $h^{(17)}$ and replace $\Delta^{(\ell)}$ for $\ell \in [18, 26]$ by (i) its per-problem mean, (ii) its across-problem mean, (iii) a frozen template from one reference problem. If the boundary is sharp, (iii) should kill accuracy while (ii) is safe. If it is smooth, results interpolate.

**[C7C.4] Experiment β-prime-prime (transport vs integration).** At $\ell=17$, measure the information the probe needs to reconstruct the final answer. Use linear probe on the residual stream to predict next-token. If probe accuracy is already high at $\ell=17$, the middle phase is transport. If probe accuracy is low at $\ell=17$ and grows monotonically through $\ell=26$, the middle phase is integration.

### 7D. The binding claim, epistemically graded

**The current working interpretation of the layer stack in Qwen2.5-3B on bilingual math is a three-phase mechanistic story: an early entanglement phase $[9, 17]$ where language identity and math content are fused, a middle resolution phase $[18, 26]$ where the fused representation is integrated into a consensus, and a late read-tip assembly phase $[27, 35]$ where the last-token read head is constructed from the consensus. Only the late phase is directly supported by causal evidence (C2c read-head across 4 models, attention entropy 37/37, Gram reconcentration at $N=20$ math). The middle phase is supported by convergent observation but no direct causal test. The early phase rests on observations that admit purely geometric null interpretations and must be treated as a working hypothesis until Experiment β or equivalent distinguishes it from curvature-only explanations.**

This binding claim is the spine of the narrative we are circling. It is strong enough to ground a paper *if and only if* the paper states the epistemic grading honestly and the middle and early phases are framed as proposals with discriminating experiments, not established facts. The temptation to say "L9-L17 is the entanglement zone" as if it were settled is exactly the temptation that produced the premature paper push earlier and must be resisted.

---

## 8. What is derived, what is observed, what is conjectured

| Claim | Status | Source |
|---|---|---|
| V⊕V⊥ Jacobian block structure (identity) | [T] | D3.1, D3.2 |
| $\mathrm{rank}_{90}(G^{(\ell)}) \in [8, 21]$ at $N=20$ math across 4 Qwen | [O] | expBQ2-XM |
| Per-lang centered rank_90 $\in [12, 24]$ at $N=200$, roughly N-invariant | [O] | centered_gram_n200.json (this session) |
| Pooled rank_90 scales linearly with num languages | [O] | centered_gram_n200.json (this session) |
| BQ3 causal pruning (drop 5 layers by delta-Gram) | [O] | expBQ3 |
| Read-head last-only preservation | [O] | expC2c, 4 Qwen |
| $\phi^{(\ell)}$ sign flip at L17/18 on math | [O] | Exp T |
| Answer-transfer probe collapse L7-L14 | [O] | Exp Y/Z |
| Attention entropy split content vs glue, $\ell \in [20, 35]$, 37/37 at $\ell=35$ | [O] | exp_attention_anatomy |
| "20D reasoning subspace" as universal constant | [C → killed] | §5 corrected; it's per-language, 12-24 |
| Read-head rank $=$ Gram rank | [C → refined] | §5 corrected: per-language intrinsic rank upper-bounds read head |
| Cocycle = "linear approximations compose, not strict group law" | [T] | §6, correctly stated |
| Late phase (L27-L35) = read-tip assembly | [C, strong] | §7B.1, causally supported |
| Middle phase (L18-L26) = resolution/consensus | [C, moderate] | §7B.2, observational only |
| Early phase (L9-L17) = fusion/entanglement | [C, weak] | §7B.3, curvature null not ruled out |
| Phase transition is causally real (not just geometric) | [C] | test = Experiment β (§7C.1) |
| Compressibility vision as adaptive compute (A) | [C] | test = Experiment 1 norm contrast |
| Compressibility vision as asymmetric computation (B) | [C] | test = Experiment 1 subspace contrast + α.1 block norms |
| Minimal $k$ for which top-$k$ last-token projection preserves accuracy | [C] | unmeasured — small dedicated experiment needed (§5.6.1) |

---

## 9. What Experiment 1 is, stated mathematically

**Pre-registration.**

**Model:** Qwen2.5-3B, float16, eager attention.
**Prompts:** 20 math problems × 2 languages (EN, ZH), same set as exp_attention_anatomy.
**Generation:** chat template wrap, max 128 new tokens.
**Labels:** per generated token, assign $t \in \mathcal{G}$ (glue) or $t \in \mathcal{C}$ (content) using the *exact same* surprisal-percentile + tokenizer-heuristic labeling as exp_attention_anatomy_3b.json. Drop tokens where labels disagree; report strict-agreement subset and full-set results separately.
**Measurements per generated token, per layer $\ell \in \{0, \ldots, 35\}$:**
1. $\|\Delta^{\text{attn},(\ell)}_t\|_2$
2. $\|\Delta^{\text{mlp},(\ell)}_t\|_2$
3. $\|\Delta^{(\ell)}_t\|_2 = \|h^{(\ell+1)}_t - h^{(\ell)}_t\|_2$
4. Direction: $\hat{\Delta}^{\text{sublayer},(\ell)}_t$ unit vector
5. Per-class mean direction: $\bar{u}^{\text{sublayer},(\ell)}_\mathcal{G}, \bar{u}^{\text{sublayer},(\ell)}_\mathcal{C}$, and their cosine

**Test statistics:**
- Norm contrast (magnitude): $T^{(\ell)}_{\text{sublayer}}$ from D4.1
- Direction contrast (subspace): $1 - |\cos(\bar{u}^{\text{sublayer},(\ell)}_\mathcal{G}, \bar{u}^{\text{sublayer},(\ell)}_\mathcal{C})|$

**Null:** permutation over class labels within problem, $R = 1000$ permutations.

**Corrections:** position-bin (quartiles) and unigram-surprisal regression before testing.

**Predictions:**

| Outcome | Interpretation A (comp-light) | Interpretation B (different comp) |
|---|---|---|
| $T^{(\ell)}_{\text{mlp}} < 0$ at $\ell \in [20, 35]$ | Yes, large | No, near zero |
| $T^{(\ell)}_{\text{attn}} < 0$ at $\ell \in [20, 35]$ | Yes, proportional | No |
| Direction cosine G vs C at $\ell \in [20, 35]$ | $\approx 1$ (same subspace, different magnitude) | $< 0.8$ (disjoint subspaces) |

**Decision rule.** If norm contrast is significantly negative AND direction cosine is high, A is supported. If norm contrast is near zero AND direction cosine is low, B is supported. Any other pattern (both large, both near zero, mixed signs across layers) is a "neither" outcome that requires a third interpretation, which we should think about *before* running the experiment, not after.

---

## 10. Open questions this document does not resolve

1. **Minimal read-head $k$ (C3.6, C5.6.1):** What is the smallest $k$ such that projecting the last-token hidden state onto the top-$k$ subspace of $\tilde{H}^{(\ell)}$ preserves math accuracy at all layers? C2c tested only $k = \mathrm{rank}_{90}$-based partition, not a sweep. A small dedicated script (few hours) would answer this and directly measure the read head's operational rank independent of the ensemble.
2. **Confounds in Experiment 1 (C4.3-C4.5):** Whether position, unigram surprisal, and prompt-vs-generation corrections will survive the signal is unknown until we run. Pre-register the corrections before running (D4.1 extended with C4.3-C4.5).
3. **Language-intrinsic rank and shared core (C5.6.2):** The centered Gram result shows per-language rank 12-24 and pooled rank 42-55. Under a simple "each language adds an orthogonal direction on top of a shared core" model, the shared-core rank is implicit but not directly measured. Running expBH-style null-space retrieval on the pooled $N=1400$ set would test this.
4. **BQ3 pruning stability at N=200 (C5.6.3):** The delta-Gram ranking used by BQ3 was computed on $N=20$ math. Re-rank using $N=200$ diverse; are the same 5 layers safe to drop?
5. **Cocycle measurement clarification (C6.3):** What precisely were the operators being linearized and composed in expBA? Need to re-open the script. Without this, the "cocycle R² > 0.87" statement remains imprecisely sourced even though its content is correct.
6. **Phase causality (C7C.1):** Experiment β has not been run. This is the distinguishing test for §7B.3 (fusion phase) and §7B.2 (resolution phase).
7. **Transport vs integration (C7C.4):** Linear probe at $\ell=17$ vs $\ell=26$ directly distinguishes whether the middle phase is doing work or transporting a signal. Small experiment.
8. **Experiment 1 "neither" outcome:** If delta norms are inconclusive across A and B, pre-register the third-interpretation possibilities *before* running: (a) heterogeneous across token subclasses (some glue tokens cheap, others expensive), (b) heterogeneous across layers (adaptive compute only at specific layers), (c) computation is in attention output-projection norms not residual delta norms (deltas get absorbed into residual). Rule out the ones we can rule out in advance; reserve the ones that need further data.
9. **Read-head discriminability of the operational subspace:** even granting per-language intrinsic rank ~15, what fraction of the rank is needed to distinguish *correct from incorrect* answers (not just problem A from problem B)? A read head that can distinguish problems but not answers is not actually doing the task.
10. **Cross-model phase-structure replication:** $\phi^{(\ell)}$ sign flip has been measured on 3B. The crossmodel validation doc says "phase transition at 40-47% depth" across 5 Qwen — this must be the same phenomenon but I have not verified the specific metric matches. Check before treating the 5-model claim as replicating the 3B phase transition.

---

*Revision log.*
- v1 (this session): initial draft with 10 sections.
- v2 (this session, same turn): §2 (O2.1), §5, §7, §8, §10 updated after running `centered_gram_n200.py` and after GPT Web pushback on §7. The rank-coincidence question is substantially resolved; the phase-transition narrative is now epistemically graded with §7B.1 strong, §7B.2 moderate, §7B.3 weak; the "20D universal" claim is explicitly killed and replaced with "per-language 12-24, language-dependent."

*Greg asked for this so he could sit with it and push back. The v2 version responds to one round of his pushback (via GPT Web) and one round of empirical data (the N=200 measurement). Further rounds welcomed.*
