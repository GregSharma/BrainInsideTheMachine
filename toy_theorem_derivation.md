# Theoretical Foundation: Z Emergence in Bilingual Linear Maps

---

## 1. Setup

- Inputs: $x_{zh}, x_{en} \in \mathbb{R}^d$ (same math problem, different languages)
- Shared linear map: $f(x) = Wx + b$, where $W \in \mathbb{R}^{d \times d}$, $b \in \mathbb{R}^d$
- Target: $y \in \mathbb{R}^d$ (correct answer representation, shared across languages)
- Loss: $L = \alpha_{zh} \|Wx_{zh} + b - y\|^2 + \alpha_{en} \|Wx_{en} + b - y\|^2$
- $\alpha_{zh} + \alpha_{en} = 1$ (training data proportions)

## 2. SVD of W and change of variables

Let $W = U \Sigma V^\top$ with $U, V$ orthogonal, $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_d)$.

Define rotated coordinates:
$$z_{zh} = V^\top x_{zh}, \quad z_{en} = V^\top x_{en}, \quad \tilde{y} = U^\top (y - b)$$

Since $V$ is orthogonal, this is just a rotation — no information lost.

Now $Wx = U\Sigma V^\top x = U\Sigma z$, so:

$$Wx + b - y = U(\Sigma z - \tilde{y})$$

And since $U$ is orthogonal, the norm is preserved:

$$\|Wx + b - y\|^2 = \|\Sigma z - \tilde{y}\|^2 = \sum_{i=1}^d (\sigma_i z_i - \tilde{y}_i)^2$$

**The loss decomposes into $d$ independent scalar problems:**

$$L = \sum_{i=1}^d \left[ \alpha_{zh}(\sigma_i z_{zh,i} - \tilde{y}_i)^2 + \alpha_{en}(\sigma_i z_{en,i} - \tilde{y}_i)^2 \right]$$

## 3. Per-direction optimality

For each direction $i$, take $\partial L / \partial \sigma_i = 0$:

$$\alpha_{zh} \cdot 2(\sigma_i z_{zh,i} - \tilde{y}_i) z_{zh,i} + \alpha_{en} \cdot 2(\sigma_i z_{en,i} - \tilde{y}_i) z_{en,i} = 0$$

$$\sigma_i \left(\alpha_{zh} z_{zh,i}^2 + \alpha_{en} z_{en,i}^2\right) = \tilde{y}_i \left(\alpha_{zh} z_{zh,i} + \alpha_{en} z_{en,i}\right)$$

**Closed form for optimal singular value:**

$$\boxed{\sigma_i^* = \tilde{y}_i \cdot \frac{\alpha_{zh} z_{zh,i} + \alpha_{en} z_{en,i}}{\alpha_{zh} z_{zh,i}^2 + \alpha_{en} z_{en,i}^2}}$$

## 4. Interpretation: where Z comes from

Define the **agreement ratio** for direction $i$:

$$\rho_i = \frac{(\alpha_{zh} z_{zh,i} + \alpha_{en} z_{en,i})^2}{\alpha_{zh} z_{zh,i}^2 + \alpha_{en} z_{en,i}^2}$$

By Cauchy-Schwarz, $0 \leq \rho_i \leq 1$.

- **$\rho_i \approx 1$**: Both languages project similarly onto direction $v_i$. W amplifies this direction. **This is Z.**
- **$\rho_i \approx 0$**: Languages project in opposite directions ($z_{zh,i} \approx -z_{en,i}$). Numerator cancels. $\sigma_i^* \approx 0$. W kills this direction. **This is Z-perp.**
- **$\rho_i$ intermediate**: Partial agreement. W partially amplifies. **Boundary of Z.**

## 5. Proposition 1 (proved)

**Proposition 1** (Bilingual gradient equilibrium). *Under the setup above:*

*(i) $\sigma_i^*$ is given by the closed form above.*
*(ii) $|\sigma_i^*| \propto \sqrt{\rho_i}$: high agreement → amplified; low agreement → killed.*
*(iii) $z_{zh,i} = -z_{en,i}$ implies $\sigma_i^* = 0$.*
*(iv) Monolingual limit ($\alpha_{en} \to 0$): $\sigma_i^* \to \tilde{y}_i / z_{zh,i}$ — no cross-lingual structure in the spectrum.*

**Proof.** Direct computation from $\partial L / \partial \sigma_i = 0$. Cauchy-Schwarz gives $\rho_i \in [0,1]$. (iii) by substitution. (iv) by limit. $\square$

**Definition** (Z-subspace). For a threshold $\tau \in (0,1)$:
$$Z_\tau = \text{span}\{v_i : \rho_i > \tau\}, \qquad Z_\tau^\perp = \text{span}\{v_i : \rho_i \leq \tau\}$$

**Corollary** (Gauge breaking). *In the monolingual case ($\alpha_{en}=0$), $W^*$ has a $d$-parameter family of equivalent solutions (any rotation $W^* \to W^* R$ yields the same loss). Adding $\alpha_{en} > 0$ breaks this degeneracy: the right singular vectors $V$ become determined (up to sign) by the bilingual agreement structure.*

**Verified** (verify_toy_theorem.py, $d=50$, $d_{\text{shared}}=15$, $N=500$): closed-form matches SVD at $r = 1.000$. $\operatorname{corr}(\sigma_{\text{bilingual}}, \text{overlap}) = 0.9995$ vs $\operatorname{corr}(\sigma_{\text{mono}}, \text{overlap}) = 0.9963$.

## 6. Predictions that match empirical data

| Prediction | Evidence |
|---|---|
| dim(Z) correlates with bilingual training overlap | Qwen (bilingual) >> DeepSeek (partial) >> LLaMA (English-only) |
| Z directions have large $\sigma_i^*$ | Contrastive Z captures 4.24x NN over random |
| Z-perp directions have small $\sigma_i^*$ | Language-specific dims (5 at L32) separate cleanly |
| More languages → Z shrinks to true shared core | 7-language NN still 4.24x — Z is robust, not diluted |
| Monolingual model has no Z | LLaMA flat floor in Lyapunov figure |

## 7. Extension: N problem pairs

With $N$ bilingual pairs $(x_{zh}^{(n)}, x_{en}^{(n)}, y^{(n)})$, the per-direction optimal singular value becomes:

$$\sigma_i^* = \frac{\sum_n \tilde{y}_i^{(n)} \left(\alpha_{zh} z_{zh,i}^{(n)} + \alpha_{en} z_{en,i}^{(n)}\right)}{\sum_n \left(\alpha_{zh} (z_{zh,i}^{(n)})^2 + \alpha_{en} (z_{en,i}^{(n)})^2\right)}$$

This is 1D least squares for each direction.

**CORRECTED multi-sample agreement ratio.** The single-pair $\rho_i$ from Section 4 does not naively extend to multiple samples: the sum-of-means formulation vanishes for zero-mean data ($r = 0.24$). The correct multi-sample statistic is the **cross-view correlation**:

$$\rho_i^{(\text{multi})} = \text{corr}(z_{zh,i}^{(1:N)}, \, z_{en,i}^{(1:N)})^2$$

**Verified**: $\operatorname{corr}(\rho^{(N)}, \sigma) = 0.999$; $\operatorname{corr}(\rho^{(N)}, \text{true overlap}) = 0.999$.

## 8. Proposition 2: contrastive extraction (proved)

**Problem**: In the unsupervised regime (no target $y$), naive bilingual PCA fails when language-specific variance exceeds shared variance — the realistic case (syntax > semantics in raw activation magnitude).

**Proposition 2** (Contrastive Z recovery). *Let $\Delta = X_{zh} - X_{en} \in \mathbb{R}^{N \times d}$ be the matrix of bilingual differences. Let $\Delta = U_\Delta S_\Delta V_\Delta^\top$ be its SVD. Define:*
- *Language subspace: $\mathcal{L} = \text{span}\{v_{\Delta,1}, \ldots, v_{\Delta,k}\}$ where $k$ is chosen by the singular value gap of $\Delta$.*
- *Contrastive projection: $P_\perp = I - V_{\Delta,:k} V_{\Delta,:k}^\top$ (project out language directions).*
- *Contrastive Z: top eigenvectors of $(P_\perp X_{zh})^\top (P_\perp X_{zh}) + (P_\perp X_{en})^\top (P_\perp X_{en})$.*

*Then $\mathcal{L}$ captures the language-specific directions, and the contrastive Z recovers the shared subspace.*

**Why this works** (connection to Proposition 1):
- The difference $x_{zh} - x_{en}$ has zero projection onto any direction $v_i$ where $z_{zh,i} = z_{en,i}$ (shared directions cancel in the difference).
- Directions where $z_{zh,i} \neq z_{en,i}$ survive in the difference, with magnitude $\propto |z_{zh,i} - z_{en,i}|$.
- The SVD of $\Delta$ therefore finds language-specific directions in order of discriminability.
- The singular value gap in $\Delta$ marks the boundary between language-specific and shared + noise directions.

**Verified**: Auto-detected 20 language directions (true: 20) via SV gap cliff (0.699 → 0.072). Contrastive Z captures 14.99/15 shared dims. NN = 500x chance vs naive bilingual 4x.

**Implication for the paper:** Our contrastive extraction method is not ad hoc — it is the natural procedure for recovering Z when language-specific variance dominates. The SV gap in the difference matrix is a principled diagnostic for choosing how many directions to remove.

## 9. Relationship to CCA

- **Standard CCA** maximizes correlation between projected views $u^\top x_{zh}, v^\top x_{en}$ *without* reference to a target $y$.
- **Our construction** is supervised: we have a target $y$ and the agreement ratio $\rho_i$ measures how well both views cooperate toward predicting $y$.
- This is closer to **"least-squares CCA" / multi-view regression via CCA** but not identical.
- **Novel packaging:** The specific punchline — directions with high per-direction agreement get large $\sigma_i$ and define Z — does not appear as a stated theorem in prior work (verified via Kimi/Perplexity search).
- **Gaussian connection:** In the Gaussian case, Z is the minimal linear subspace whose projections retain all mutual information between $x_{zh}$ and $x_{en}$. The number of nonzero canonical correlations equals $\dim(Z)$.

## 10. Multi-layer extension (sketch, not proved)

For deep linear networks $f(x) = W_L \cdots W_1 x$:

- Deep linear network theory shows gradient flow performs **spectral filtering**: directions with stronger signal-to-noise get amplified across layers in a staged manner.
- Define $Z_k$ = span of right singular vectors of the effective map from layer $k$ activations to target, filtered by high bilingual agreement $\rho_i$.
- **Prediction:** If a direction lies in $Z_k$ and is amplified at layer $k$, its image is favored at layer $k+1$, yielding an inductive bias toward propagating shared directions forward.
- This predicts **gradual Z emergence** across depth — matching the Lyapunov integration figure (Qwen rises and sustains, LLaMA never forms, InternLM2 rises then falls).

---

## 11. Open problem: partition-invariant subspace

**Status: CONJECTURE. Proof has a gap. Formula not yet verified numerically.**

### Setup

Languages tokenize differently. Posit a latent signal $s(t) \in \mathbb{R}^d$, $t \in [0,1]$, covariance operator $K$ with Mercer expansion $K(t,t') = \sum_m \lambda_m \phi_m(t)\phi_m(t')^\top$. Each language $\ell$ averages $s$ over its token intervals:

$$h_j^{(\ell)} = \frac{1}{\Delta\tau_j^\ell} \int_{\tau_{j-1}^\ell}^{\tau_j^\ell} s(t)\,dt + \epsilon_j^{(\ell)}, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

Discretize: $P_\ell \in \mathbb{R}^{T_\ell \times M}$ averages grid points per token. $x_\ell = P_\ell s + \epsilon_\ell$.

### The CCA structure

For scalar case ($d=1$), define:
- $A = P_{zh} V \in \mathbb{R}^{T_{zh} \times M}$, $B = P_{en} V \in \mathbb{R}^{T_{en} \times M}$ (partition-averaged eigenfunctions)
- $\Sigma_{zh} = A \Lambda A^\top + \sigma^2 I$, $\Sigma_{en} = B \Lambda B^\top + \sigma^2 I$
- $\Sigma_{zh,en} = A \Lambda B^\top$

### The survival factor

**Definition.** $\psi_m^\ell = \|P_\ell v_m\|^2$ — how much of eigenfunction $m$ survives averaging by partition $\ell$. For a constant function: $\psi_1^\ell = 1$. For a function oscillating faster than the token width: $\psi_m^\ell \approx 0$.

For the Fourier basis on a uniform partition with $T$ tokens: $\psi_m \approx \operatorname{sinc}^2(m/T)$.

### Conjecture (Proposition 3)

*If the partition-averaged eigenfunctions are approximately orthogonal (both $A^\top A \approx \operatorname{diag}(\psi_m^{zh})$ and $B^\top B \approx \operatorname{diag}(\psi_m^{en})$), AND the cross-partition Gram matrix $A^\top B$ is approximately diagonal, then the squared canonical correlations satisfy:*

$$\rho_m^2 \approx \frac{\lambda_m^2 \, \psi_m^{zh} \, \psi_m^{en}}{(\lambda_m \psi_m^{zh} + \sigma^2)(\lambda_m \psi_m^{en} + \sigma^2)}$$

*The number of canonical correlations exceeding threshold $\tau$ is:*

$$\dim(Z_\tau) = \left|\left\{m : \lambda_m > \frac{\sigma^2 \tau}{(1-\tau) \sqrt{\psi_m^{zh} \psi_m^{en}}}\right\}\right|$$

*Symmetric case: eigenfunction $m$ enters $Z$ iff effective SNR $\lambda_m \psi_m / \sigma^2 > \tau/(1-\tau)$.*

### Proof gap

The within-partition orthogonality ($A^\top A$ diagonal) is reasonable when partitions are finer than eigenfunction oscillation. **But $A^\top A$ diagonal does NOT imply $A^\top B$ diagonal.** The cross-partition Gram matrix $A^\top B$ couples eigenfunction $m$ as seen by Chinese tokens with eigenfunction $m'$ as seen by English tokens. If $A^\top B$ has significant off-diagonal entries, the CCA doesn't decompose per-eigenfunction, and the formula breaks.

**What would close the gap:** Either (a) prove $A^\top B$ is approximately diagonal under stated partition conditions, (b) bound the off-diagonal contribution and show it's small, or (c) numerically verify the formula against exact CCA on synthetic data and identify the regime where it holds.

### What the conjecture predicts (if true)

$\dim(Z)$ controlled by a three-way race:
1. **Spectral decay of $K$**: how fast $\lambda_m$ drops (how complex the shared signal is)
2. **Partition filtering**: $\psi_m^\ell$ kills frequencies above $T_\ell/2$
3. **Noise floor $\sigma^2$**: language-specific variance sets the detection threshold

Bottleneck partition sets ceiling: $\dim(Z) \lesssim \min(T_{zh}, T_{en})/2$. But for smooth signals (fast spectral decay), $\dim(Z) \ll \min(T_{zh}, T_{en})$.

### Testable predictions

| Prediction | Experiment |
|---|---|
| Mean-pooled activations agree in Z | Already confirmed (R² = 0.976) |
| $\dim(Z)$ bounded by min token count | Check if shorter-tokenized languages yield lower NN |
| SV gap of difference matrix predicts $n_{\text{remove}}$ | Compute SVD of $(h_{zh} - h_{en})$ at L32, check gap location |
| Similar tokenization → higher per-token agreement | Japanese/Chinese NN > Chinese/Arabic NN without alignment |
| Smooth signal → $\dim(Z) \ll T$ | Our k-sweep: category structure at $k=2$, fine-grained saturates at $k \approx 50 \ll T$ |

### Why attention matters

Self-attention re-weights token representations, partially undoing language-specific partitioning — interpolating toward the shared $s(t)$. This predicts attention heads in middle layers learn to *de-tokenize*: reconstruct something closer to $s(t)$ from the language-specific chunking. The Lyapunov integration figure (Z strength across depth) measures how well this de-tokenization succeeds.

---

## 12. Open problem: the non-trivial existence theorem

**Status: NOT PROVED. The version we attempted (Theorem 1) was circular — assumed shared content V of dim r, concluded rank(Z) >= r. Tautology.**

The real theorem would be one of:

**Version A (hard lower bound):** Don't assume V exists. Assume only that both observers achieve low loss simultaneously on shared parameters. Prove a shared subspace MUST exist and bound its dimension in terms of the losses.

**Version B (upper bound):** Prove $\operatorname{rank}(Z) \leq \dim(\text{shared content}) + \text{small}$. Z doesn't hallucinate structure. Combined with a lower bound, this pins $\dim(Z)$.

**Version C:** This IS Proposition 3 — the partition structure controls $\dim(Z)$ through spectral decay. Produces new predictions. Most promising path.

---

## 13. The abstract framework (aspirational, not proved)

**Definitions only. No theorem claimed.**

Given a latent state $S_k \in \mathbb{R}^d$ at stage $k$, observers $\{O_i\}$ each with loss $L_i$:

- **Sensitivity matrix**: $F_{i,k} = \mathbb{E}[\nabla_{S_k} L_i \cdot \nabla_{S_k} L_i^\top]$ (empirical Fisher of observer $i$)
- **Shared eigenspace**: $Z_k = \bigcap_i \operatorname{colspan}(F_{i,k})$
- **Shared Fisher geometry**: $F_{i,k}|_{Z_k}$ determines cross-observer transfer

The claim — that any multi-observer shared-parameter system converges to this factorization, with $\dim(Z)$ determined by mutual information between observers — is a research program, not a theorem. Proposition 1 is one verified instance (linear, supervised, bilingual). Proposition 3, if proved, would be a second (continuous signal, partition-dependent, CCA-based).

## 14. Von Neumann methodology note

Per research into von Neumann's workflow (Ulam, Goldstine accounts):
- He used toy models as **scaffolding**, not endpoints.
- His bar for publishing: be explicit about limitations, situate in a bigger program, provide at least one nontrivial generalization beyond the toy case.
- He'd push us to find the "right abstract language" — likely representation theory or information geometry — where Z becomes a natural structural property.
- BUT he published partial frameworks (The Computer and the Brain was itself unfinished).
- **Our bar:** Toy theorem (done) + N-pair extension (done) + multi-layer sketch + frame as opening of a general program. That's publishable — IF we don't overclaim what's proved.

## 15. Limitations

- **$U, V$ are learned, not fixed.** The SVD decomposition treats $\Sigma$ as the free parameter with $U, V$ fixed. In unconstrained training, $U, V$ also move. The toy theorem captures the equilibrium structure but not the dynamics.
- **Linear only.** Element-wise nonlinearities (ReLU, GELU) preserve subspace membership (don't rotate Z into Z-perp), so the qualitative picture survives, but the clean scalar decomposition doesn't. No formal argument yet.
- **No attention.** Attention is a content-dependent routing mechanism that can dynamically select which directions to amplify. The toy theorem doesn't capture this.
- **Information-theoretic bound.** A clean $\dim(Z) \geq I(x_{zh}; x_{en}) / \log d$ does NOT follow from standard results. Requires Gaussian or other distributional assumptions. Don't claim this.

## 16. Speculative Connection: String Kernels, ARD, and SSMs

**Status: NOTE — not developed, may not use. Recorded for future reference.**

String kernels (GPML §4.4.1) decompose similarity between sequences by counting weighted substring occurrences: $k(x, x') = \sum_{s \in A^*} w_s \, \phi_s(x) \, \phi_s(x')$. The weight vector $w_s$ controls which sequential features matter — functionally identical to what ARD length scales do in continuous kernel spaces: selectively amplifying or suppressing dimensions.

The same operation appears in structured state-space models (S4/H3/Hungry Hungry Hippos): the diagonal state matrix selects which frequencies persist across sequence positions, "clamping" onto signal and letting noise decay. All three mechanisms are instances of **selective projection of sequential input onto a lower-dimensional feature space where similarity is computed**.

**Relevance to Z:** The cross-lingual bridge Z is itself a selective projection — certain directions carry semantic content (high weight), language-specific directions get suppressed (low weight). The k-sweep (§k-sweep results) showed this weighting is smooth (no sharp elbow), consistent with ARD-style continuous relevance weighting rather than a hard rank cutoff. If Z's structure can be characterized as an implicit kernel over token sequences, the string kernel formalism might provide a complementary theoretical lens — particularly for understanding *which* sequential features (substrings of reasoning tokens) the model preserves across languages.

**Open question:** Is there a formal connection between the SVD-based Z extraction and a kernel defined over the token sequences that produced the activations? The substring feature space $\phi_s(x)$ is explicit; Z's feature space is learned. Whether the learned projection converges to something kernel-like under bilingual training is unknown.

## 17. TODO

- [x] Proposition 1 — proved, verified r=1.000
- [x] Multi-sample rho correction — cross-view correlation
- [x] Proposition 2 — contrastive extraction, verified 500x chance
- [ ] **Proposition 3 numerical verification** — synthetic signal + two partitions + CCA
- [ ] **Close the A^TB diagonality gap** — or characterize when it fails
- [ ] SV gap diagnostic on real Qwen L32 data
- [ ] Version A existence theorem (the hard one)
- [ ] Nonlinear extension: formal argument for ReLU subspace preservation
- [ ] LaTeX for paper
- [ ] Decide: Section 3 (theory before experiments) vs appendix
