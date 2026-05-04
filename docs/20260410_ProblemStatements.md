# Problem: Boundary Conditions of an Internally Rigid Gated Serial Map

## Setup

Let $F: \mathbb{R}^d \to \mathbb{R}^d$ be a residual composition of $L$ gated layers:

$$F = (\text{id} + g_L) \circ \cdots \circ (\text{id} + g_1)$$

where each $g_\ell: \mathbb{R}^d \to \mathbb{R}^d$ has the form

$$g_\ell(x) = W_\ell^{\text{down}} \left[\sigma(W_\ell^{\text{gate}} x) \odot (W_\ell^{\text{up}} x)\right]$$

with $W_\ell^{\text{up}}, W_\ell^{\text{gate}} \in \mathbb{R}^{m \times d}$, $W_\ell^{\text{down}} \in \mathbb{R}^{d \times m}$, $m > d$, and $\sigma$ a smooth monotone nonlinearity (e.g., $\sigma(z) = z \cdot \text{sigmoid}(z)$). Denote the intermediate state after $\ell$ layers as $h^{(\ell)}(x)$.

### The full system

The full system $\mathcal{M}$ operates over a finite alphabet $\mathcal{A}$ with $|\mathcal{A}| = n$. There are linear maps $E \in \mathbb{R}^{d \times n}$ (encoder) and $D \in \mathbb{R}^{n \times d}$ (decoder). Given a sequence $a_1, \ldots, a_T \in \mathcal{A}$, the system maintains a state array $\{h_t^{(\ell)}\}$ for each position $t$ and depth $\ell$:

$$h_t^{(0)} = E \cdot \mathbf{e}_{a_t}$$

$$h_t^{(\ell)} = h_t^{(\ell-1)} + R_\ell(h_t^{(\ell-1)}, \{h_s^{(\ell-1)}\}_{s<t}) + g_\ell\!\left(h_t^{(\ell-1)} + R_\ell(\cdots)\right)$$

where $R_\ell$ is a **routing mechanism**: a data-dependent weighted average

$$R_\ell(h_t, \{h_s\}_{s<t}) = W_O^{(\ell)} \sum_{s < t} \alpha_{t,s}^{(\ell)} \cdot W_V^{(\ell)} h_s, \quad \alpha_{t,s}^{(\ell)} = \frac{\exp(q_\ell(h_t)^\top k_\ell(h_s) / \sqrt{d_h})}{\sum_{s'} \exp(\cdots)}$$

with learned linear projections $q_\ell, k_\ell: \mathbb{R}^d \to \mathbb{R}^{d_h}$ and $W_V^{(\ell)}, W_O^{(\ell)}$ value/output matrices. The routing mechanism caches $\{k_\ell(h_s), W_V^{(\ell)} h_s\}$ for all prior positions and all layers (the **routing cache**).

The next symbol is selected by:

$$a_{T+1} = \arg\max_{a \in \mathcal{A}} \; (D \cdot h_T^{(L)})_a$$

The system generates **autoregressively**: $a_{T+1}$ is appended, the state array is extended, and the process repeats. A single forward pass produces one symbol. Multi-step outputs require multiple sequential applications, each conditioned on all previously generated symbols.

### Training

The parameters $\theta = \{W_\ell^{\text{up}}, W_\ell^{\text{gate}}, W_\ell^{\text{down}}, q_\ell, k_\ell, W_V^{(\ell)}, W_O^{(\ell)}, E, D\}$ were obtained by stochastic gradient descent on

$$\mathcal{L}(\theta) = \mathbb{E}_{(a_1, \ldots, a_T) \sim \mathcal{D}} \left[-\sum_{t=1}^{T} \log \frac{\exp\left((D \cdot h_t^{(L)})_{a_t}\right)}{\sum_{a'} \exp\left((D \cdot h_t^{(L)})_{a'}\right)}\right]$$

over a dataset $\mathcal{D}$ of sequences drawn from multiple structured domains. No supervision is given on any intermediate state; the only training signal is next-symbol prediction. The training data spans multiple **notational conventions** — structured subsets of $\mathcal{A}$ with distinct symbol usage patterns for expressing equivalent content.

### Evaluation

Define a task as a pair $(x, y)$ where $x \in \mathcal{A}^T$ is an input sequence and $y \in \mathcal{Y}$ is a target value. The system generates a continuation $\hat{a}_{T+1}, \ldots, \hat{a}_{T+S}$ autoregressively. A deterministic extractor $\xi: \mathcal{A}^S \to \mathcal{Y}$ parses the generated sequence to produce $\hat{y} = \xi(\hat{a}_{T+1}, \ldots, \hat{a}_{T+S})$. **Accuracy** is $\mathbb{1}[\hat{y} = y]$, averaged over a task set.

Accuracy depends on the **entire generated sequence**, not on any single activation or logit. Two systems with different internal states can both achieve $\hat{y} = y$. Two systems with nearly identical single-step logits can diverge autoregressively and achieve different $\hat{y}$. A perturbation at step $t$ propagates through all subsequent steps.

### Scaling family

We observe a family $\{\mathcal{M}_p\}$ indexed by parameter count $p$, with $d$ and $L$ both increasing. Five instances: $p \in \{1.5\text{B}, 3\text{B}, 7\text{B}, 8\text{B}, 14\text{B}\}$, $d \in \{1536, 2048, 3584, 4096, 5120\}$, $L \in \{28, 36, 28, 36, 48\}$.

### Tied vs. untied constraint

Some systems enforce $D = E^\top$. Others learn $D$ independently. This constraint has dramatic structural consequences (G8).

---

## Information-Theoretic Foundation

### Existence of $f^*$ (theorem, not conjecture)

Consider $N$ hypothetical systems $\mathcal{M}_1, \ldots, \mathcal{M}_N$, each operating over a distinct alphabet $\mathcal{A}_i$ but trained on the same structured tasks expressed in convention $i$. Each achieves equivalent accuracy: $\xi_i(\mathcal{M}_i(x_i)) = \xi_j(\mathcal{M}_j(x_j)) = y$ for matched tasks. These systems have completely disjoint parameters — no weight can be shared between them. Yet they compute the same function up to convention.

Therefore there must exist an abstract map $f^*$ and convention-specific wrappers $h_i, h'_i$ such that $\mathcal{M}_i = h'_i \circ f^* \circ h_i$. This decomposition exists in the space of possible computations regardless of whether any single system cleanly instantiates it.

The information-theoretic constraint makes this precise. Define the intermediate representation $\mathcal{Z} = h_i(x_i)$. It must satisfy:

$$I(\mathcal{Z}; \text{convention}) \approx 0, \quad I(\mathcal{Z}; \text{task content}) \approx I(\text{input}; \text{task content})$$

These are simultaneously satisfiable whenever convention and task content are conditionally independent in the data — i.e., knowing the task content, the convention provides no additional information about the correct output $y$. For the structured tasks considered (compositional arithmetic, logical deduction), this conditional independence holds by construction.

**Therefore $f^*$ exists.** The questions that follow concern its extractability and the computational advantage conferred by a pretrained $F$.

### Uniqueness via multilingual triangulation

A single system $\mathcal{M}$ trained on $N$ conventions simultaneously contains $N$ implicit encoders $h_1, \ldots, h_N$ (one per convention). Each pair $(i, j)$ defines a map $\Phi_{ij} = h_j^{-1} \circ h_i$ relating convention $i$'s representation to convention $j$'s. If $f^*$ is uniquely instantiated, these maps must satisfy the **cocycle condition**:

$$\Phi_{ik} = \Phi_{jk} \circ \Phi_{ij} \quad \forall \; i, j, k$$

Empirically ($N = 7$ conventions, $\binom{7}{2} = 21$ pairwise maps fitted via Procrustes regression): **cocycle composition error $\leq 2.86\%$**. The 21 constraints are nearly perfectly consistent. This means $f^*$ is not merely *some* convention-agnostic representation — it is essentially **unique** (up to internal symmetries of $Z$ itself). With $N = 2$, the decomposition has massive gauge freedom; with $N = 7$, the 21 constraints triangulate $f^*$ to a tight manifold.

**Corollary:** The subspace $Z$ invariant under all $\binom{N}{2}$ Procrustes maps IS the coordinate system of $f^*$. Computing this subspace is an eigenvalue problem on the symmetrized Procrustes matrices, not a training problem. Empirically, this subspace has $\dim(Z) = k \approx 20$ and achieves 97% cross-convention retrieval accuracy with perfect cluster purity.

### $N$ working encoders as existence proofs for $E'$

The pretrained system already contains $N$ encoders $h_1, \ldots, h_N$ that each successfully place inputs onto $F$'s dynamical manifold (the system works in all $N$ conventions). Each encoder's output at depth $\ell^*$ projects onto the same $Z$ (by the cocycle). Therefore the solution manifold for any new encoder $E'$ is not empty — it is at least $N$-dimensional, parameterized by the known working encoders. This converts the search for $E'$ from an unconstrained optimization to interpolation/extrapolation from known solutions.

---

## Givens

There exists, at a specific depth $\ell^*$, an orthogonal decomposition $\mathbb{R}^d = Z \oplus Z^\perp$ with $\dim(Z) = k$ and three rank parameters $r_1 \leq r_2 \leq r_3$ such that:

### G1. Low-rank readout

There exists a layer $\ell^*$ and a rank-$r_2$ projector $P$ such that replacing $h_T^{(\ell^*)} \leftarrow P \cdot h_T^{(\ell^*)}$ at the final position only, then generating autoregressively, preserves accuracy. The complement $(\text{id} - P) \cdot h_T^{(\ell^*)}$ is causally inert — removing it preserves or, in larger untied systems, **improves** accuracy.

### G2. Dimensional hierarchy and variance-causation dissociation

For a single task instance, the output-relevant information at $\ell^*$ occupies $r_1$ dimensions. Across the task distribution, $r_2$ dimensions are causally necessary. The statistical support of $\{P_Z h^{(\ell^*)}(x)\}_{x \sim \mathcal{D}}$ has effective dimension $r_3$:

$$r_1 \leq r_2 \leq r_3 \leq k \ll d$$

Empirically: $r_1 \approx 6, \; r_2 \approx 8, \; r_3 \approx 69, \; k \approx 20, \; d = 2048$.

**Variance and causation are dissociated at every level tested.** The top-$r_3$ principal components capture 93–97% of activation variance but are **causally inert** — zeroing them does not affect accuracy (Exp X3). The causal signal lives in the remaining 3–7% of variance. A linear map between layers $\ell$ and $\ell'$ achieves $R^2 = 0.9963$ but substituting this map causally reduces accuracy to chance (Exp X2). The 0.04% residual carries all task-relevant information. Rank-forcing onto the top-$k$ SVD subspace at any layer also destroys accuracy (Exp AK/FOAMS), confirming: **observational rank structure does not correspond to causal information localization**.

### G3. Interior rigidity

For any layer $\ell$ and any projector $P'$ with $\text{rank}(P') < d$, replacing $h_t^{(\ell)} \leftarrow P' \cdot h_t^{(\ell)}$ for positions $t < T$ causes accuracy to collapse. Full rank is required at every depth — not for the final readout, but for the routing mechanism: subsequent layers' weighted averages over prior positions require full-dimensional states.

### G4. Bottleneck irreducibility

Inserting a learned factorization $W_1 W_2$ with $\text{rank}(W_1 W_2) = r < d$ at any layer $\ell$, optimizing $W_1, W_2$ via SGD while freezing all other parameters: accuracy collapses at all $r < d$, for $r \in \{3, 5, 10, 20, 50, 100, 500\}$ and all layers tested.

### G5. Depth irreducibility (corrected)

Successive gated-layer outputs are **not** orthogonal. The delta-to-delta correlation $R^2(g_\ell, g_{\ell-1}) \approx 0.91\text{–}0.94$ across all layers and systems — each layer's output is highly correlated with the previous layer's. However, depth is still irreducible: a linear map fitted between $h^{(\ell)}$ and $h^{(\ell+\Delta)}$ achieves $R^2 > 0.99$ but **fails causally** when substituted for the actual computation (accuracy → chance). The 0.04% residual is load-bearing. Serial depth cannot be shortcut even when the layer-to-layer map is near-linear.

*Note: The original input-to-delta correlation ($R^2 \approx 0.03$) was a measurement of the wrong quantity; corrected after cross-model replication.*

### G6. Subspace complementarity and $\mathbb{Z}_2$ symmetry

$P_Z h^{(\ell^*)}$ determines $\hat{y}$. $P_{Z^\perp} h^{(\ell^*)}$ determines the surface form. At $\ell^*$, the state decomposes as $h = \alpha \hat{e}_c + f$ where $\hat{e}_c$ is a single unit vector, $\alpha$ encodes convention identity with Cohen's $d > 11$, and $\cos(f_{c_1}, f_{c_2}) = 0.952$ for matched tasks across conventions. Swapping $\hat{e}_c$ achieves 100% convention switch with 100% first-symbol match.

**The convention encoding is a $\mathbb{Z}_2$ symmetry (reflection), not a continuous symmetry.** Flipping $\hat{e}_c$ (a norm-preserving reflection across the convention axis) succeeds perfectly during autoregressive generation. Higher-dimensional rotations in $Z^\perp$ — including learned Procrustes rotations mapping one convention's centroid to another's — produce catastrophic hallucination. Projecting out convention directions (lossy) preserves convention identity but degrades accuracy monotonically with projected dimension count. Only the 1D reflection is coherent per-token; all other interventions overfit to population statistics and fail at the single-token level. The convention subspace decomposes as: 1D reflection axis (causally operative) + $(d_c - 1)$ residual convention directions (statistically present, causally entangled with $Z$).

### G7. Layer-dependent invariance (dual empirical characterization)

There exists a critical layer $\ell_c < \ell^*$ such that:

**From the invariance side (G7a):** For $\ell \leq \ell_c$, matched-task nearest-neighbor accuracy on $P_Z g_\ell$ across conventions: $\geq 0.66$. For $\ell > \ell_c$: $\leq 0.09$ (chance). The transition is sharp (~2–3 layers).

**From the mixing side (G7b):** Replacing the state at layer $\ell$ with the elementwise average of states from two conventions on the same task, then continuing the forward pass: at $\ell = \ell_c$, accuracy is preserved (5/5 tasks correct). At $\ell > \ell_c + \Delta$ (empirically $\Delta \approx 8$): accuracy collapses (0/N). Convention-mixing is viable exactly where $g_\ell$'s action on $Z$ is convention-independent.

Empirically $\ell_c \approx L/3$ in all systems tested.

### G8. Spectral determination of $\ell^*$ and the tied/untied dichotomy

Define $\Sigma_\ell = D \cdot W_O^{(\ell)} W_V^{(\ell)}$. Then $\ell^* = \arg\max_\ell \sigma_1(\Sigma_\ell)$. Verified across all systems. Always $\ell^*/L > 0.94$.

| Constraint | $\sigma_1/\sigma_2$ | Top-1 var | $r_{90}$ |
|:--|--:|--:|--:|
| Tied ($d{=}3584$) | 10.42 | 95.7% | **1** |
| Tied ($d{=}2048$) | 3.84 | 49.5% | 131 |
| Untied ($d{=}5120$) | 1.30 | 12.0% | 564 |

Tied systems force extreme spectral concentration. Untied systems distribute readout. Both satisfy G1.

### G9. Parameter budget

$E, D$: ~20% of parameters. Gated layers $\{g_\ell\}$: ~79%. Routing: ~10%. Setting all $g_\ell = 0$ → accuracy = chance.

### G10. Routing cache inertness

Post-encoding, randomly permuting all cached keys and values across positions at all layers: no effect on accuracy. Swapping caches between convention-matched runs: no effect. The state $h_T^{(L)}$ carries all necessary information.

### G11. Task fragmentation

$Z$ is task-specific: $Z_{\text{task}_1} \neq Z_{\text{task}_2}$, with 0/5 cross-task transfer. But G1, G3, G8, G10 hold universally. The architectural scaffold is universal; $Z$ rotates across tasks.

### G12. Convention-invariance is efficiency, not capability

Flipping $\hat{e}_c$ across all layers accelerates generation (+160% at 128 steps) but does not improve asymptotic accuracy (+0% at 512 steps). A system trained on only one convention possesses $\hat{e}_c$ (Cohen's $d > 3.5$) but flipping it has zero effect. **The axis exists without function.** Entanglement is training-determined.

### G13. Scaling behavior

- All systems exhibit the Gram trajectory: expand → plateau → reconcentrate at $\ell^*$. Reconcentration depth scales with $L$, not $p$.
- **The convergence path is capacity-dependent.** Small systems ($p = 1.5\text{B}, 3\text{B}$) and specific training recipes ($p = 9\text{B}$) exhibit a tight convergence "funnel" at $\ell \approx L/3$ where cross-convention spread compresses to $\sim 0.015$. Larger systems ($p = 8\text{B}, 14\text{B}$) maintain spread $> 0.08$ throughout — sufficient capacity to represent convention and content simultaneously without compression. The funnel is a **capacity scar**, not a universal computational strategy.
- Convention-invariant representation quality improves with scale: peak cross-convention cosine: $0.972$ ($3\text{B}$) → $0.996$ ($8\text{B}$) → $0.999$ ($9\text{B}$).
- Larger systems have wider cooperative bandwidth at $\ell^*$: 10/20 positive Gram eigenvalues vs. 3–5/20 at smaller scale.

### G14. Probe-generation dissociation

A linear probe trained on $P_Z h^{(\ell)}$ achieves 1.000 accuracy at **every** layer $\ell \in \{0, \ldots, L\}$, cross-conventionally, with identical coefficients across all conventions. The mathematical content is extractable as an information-theoretic object at every depth.

However, generating autoregressively from the same extracted representation — via re-embedding, precision-weighted averaging, or any tested injection method — produces degenerate output (0/N accuracy). The content is **readable but not generatively accessible** outside the full autoregressive pipeline.

**Implication for reparametrizability:** It is not sufficient for $E'$ and $D'$ to preserve information content of $Z$. They must place inputs into the basin of attraction of $F$'s autoregressive dynamics such that the generated trajectory lands in $\xi^{-1}(y)$. Information preservation is necessary but not sufficient; trajectory compatibility is required.

### G15. Kernel surgery: thin separability of convention from computation

Projecting out the convention direction $\hat{e}_c$ from the gated-layer weight matrices $W_\ell^{\text{down}}$ at layers $\ell \in [\ell_c, \ell^*]$ — a permanent weight modification, no inference hooks — yields:
- Convention-1 accuracy: +44% (3/20 → 5/20, efficiency gain)
- Convention-2 accuracy: preserved (9/20 → 9/20)
- Convention identity: preserved (17/20 correct convention per prompt)

Increasing the projected subspace from 1D to 10D degrades accuracy monotonically — convention-2 output degenerates (loops, hallucination). The entanglement tensor $E_\ell$ is nonzero but **rank-1 dominant**: the 1D convention direction is cleanly separable from $g_\ell$'s weights; the remaining convention dimensions are entangled with $Z$.

### G16. Gram kernel dynamics

The pairwise cosine Gram matrix $G^{(\ell)}$ of states across tasks is preserved layer-to-layer (adjacent correlation $= 0.974$) despite 77°/layer coordinate rotation in ambient space. The Lyapunov spectrum of $G^{(\ell)}$ reveals four dynamical phases: structure-building (positive exponents), compression (negative), sustain (near-zero), and output expansion (positive). The centered Gram has $r_{90} \in [8, 21]$ across layers — low-rank but not as extreme as the uncentered Gram (where $r_{50} = 1$ is an anisotropy artifact, not a structural finding).

**Causal validation:** Skipping layers with smallest $\|\Delta G^{(\ell)}\|_F$ (Gram-guided pruning) preserves 100% of accuracy at 14% layer skip rate. Random skip at the same rate loses 55%. Destructive skip kills all accuracy. Cooperative-phase layers ($\ell \approx 0.5L$–$0.7L$) are genuinely redundant for accuracy; skipping them can improve performance.

### G17. Cocycle flatness

For all triples $(i, j, k)$ of conventions, the composed Procrustes maps satisfy $\|\Phi_{ik} - \Phi_{jk} \circ \Phi_{ij}\| / \|\Phi_{ik}\| \leq 0.0286$. The cocycle composition error is below 3% across all 35 triples ($\binom{7}{3}$). This establishes that the $N = 7$ convention-specific encoders all agree on a common $f^*$, ruling out the possibility that different convention pairs point to different decompositions.

Furthermore: when $Z$ is defined using only 2 conventions, cross-convention retrieval is mediocre. When $Z$ is defined using all $\binom{7}{2} = 21$ pairwise differences, retrieval jumps to 97% with cluster purity 1.000. The $Z$ subspace is **overdetermined** by the multilingual constraints — more conventions yield a tighter, more accurate $Z$, consistent with triangulation toward a unique $f^*$.

### G18. $N$ working encoders exist

The system achieves nonzero accuracy in all $N = 7$ conventions. Each convention's encoder $h_i$ (the map from convention-$i$ input to $h^{(\ell^*)}$) successfully places inputs onto $F$'s dynamical manifold such that autoregressive generation produces correct outputs. These are $N$ existence proofs for trajectory-compatible encoders, and their outputs all lie in the same $Z$ (by G17). The solution set for any new encoder $E'$ is therefore nonempty and at least $N$-dimensional.

---

## Definitions

The **restricted Jacobian** of $g_\ell$ at $x$:

$$J_\ell^Z(x) = P_Z \frac{\partial g_\ell}{\partial x}\bigg|_x P_Z \in \mathbb{R}^{k \times k}$$

The **entanglement tensor**:

$$E_\ell(x) = P_Z \frac{\partial^2 g_\ell}{\partial x^2}\bigg|_x (P_Z \otimes P_{Z^\perp})$$

The **Lyapunov spectrum** of $F$ along trajectory $x$: singular values of

$$\mathcal{J}(x) = \prod_{\ell=1}^{L} \left(I + \frac{\partial g_\ell}{\partial h}\bigg|_{h^{(\ell-1)}(x)}\right)$$

yield exponents $\lambda_i = \frac{1}{L} \log \sigma_i(\mathcal{J})$, ordered $\lambda_1 \geq \cdots \geq \lambda_d$.

The **Gram Lyapunov spectrum**: eigenvalues of $G^{(\ell)}$ tracked across layers yield kernel-space exponents $\mu_1^{(\ell)} \geq \cdots \geq \mu_N^{(\ell)}$ where $N$ is the number of task instances.

The **coupling layer**:

$$\ell_c = \min\left\{\ell : \sup_{\substack{x_1, x_2 \\ P_Z x_1 = P_Z x_2 \\ P_{Z^\perp} x_1 \neq P_{Z^\perp} x_2}} \frac{\|J_\ell^Z(x_1) - J_\ell^Z(x_2)\|_F}{\|J_\ell^Z(x_1)\|_F} > \tau \right\}$$

Equivalently (G7b): $\ell_c = \max\{\ell : \text{elementwise averaging of states across conventions at } \ell \text{ preserves accuracy}\}$.

---

## Questions

### (a) Reparametrizability: training advantage, not existence

The Information-Theoretic Foundation establishes that $f^*$ exists and is essentially unique (pinned by $\binom{N}{2}$ cocycle constraints). G18 provides $N$ working encoders as existence proofs. The question is not whether $\mathcal{M}' = D' \circ F \circ E'$ exists, but **how cheaply it can be obtained given pretrained $F$**.

Let $C_{\text{scratch}}$ be the computational cost of training $f^*$ from random initialization with an abstract alphabet $\mathcal{A}'$, $|\mathcal{A}'| = K$. Let $C_{\text{extract}}(F)$ be the cost of finding $E', D'$ (and optionally fine-tuning $g_\ell$ for $\ell > \ell_c$) given the pretrained gated layers $\{g_\ell\}$.

**(a.i)** Characterize $C_{\text{extract}}(F) / C_{\text{scratch}}$. Is the ratio $O(1/L)$ (proportional to the fraction of retrained layers)? $O(K/n)$ (proportional to vocabulary compression)? Or $O(1)$ (pretrained $F$ confers no advantage)?

**(a.ii)** The $N$ existing encoders $h_1, \ldots, h_N$ all produce trajectory-compatible states whose projections onto $Z$ agree (G17). Does $E'$ require gradient-based training, or can it be constructed from the $N$ known encoders via linear algebra (interpolation, averaging, projection)?

**(a.iii)** G14 establishes that information preservation is insufficient — a linear probe recovers task content perfectly, but generation from extracted representations fails. Therefore $E'$ must place inputs into the **basin of attraction** of $F$'s autoregressive dynamics, not merely into $Z$. What additional structure beyond $Z$-membership does trajectory compatibility require?

G7b provides a partial constraint: elementwise averaging of states from two conventions at layer $\ell_c$ preserves accuracy (5/5). This means the midpoint of two trajectory-compatible states is itself trajectory-compatible, at least in the convention-invariant regime ($\ell \leq \ell_c$). Whether this extends to the centroid of $N$ states, and whether the centroid remains compatible through the nonlinear layers above $\ell_c$, is the key open subproblem.

### (b) Phase transition

Does there exist a critical $\|E\|^*$ such that:
- For $\sup_\ell \|E_\ell\| < \|E\|^*$: part (a) holds with $K = O(r_3)$.
- For $\sup_\ell \|E_\ell\| > \|E\|^*$: no $K \ll n$ suffices without retraining $g_\ell$.

G15 provides an asymmetric data point: 1D convention projection from $g_\ell$'s weights succeeds (thin separability), but 10D projection fails (thick entanglement). This suggests the transition occurs between rank 1 and rank 10 of $E_\ell$.

### (c) Lyapunov characterization of rigidity with collapse

$F$ is full-rank at every interior cross-section (G3, G4) yet factors through $k \ll d$ dimensions at $\ell^*$ (G1, G6). Conjecture: this requires the Lyapunov spectrum to have exactly $k$ non-negative exponents and $d - k$ strictly negative exponents. The expanding/neutral directions converge to $Z$ by $\ell^*$; contracting directions carry $Z^\perp$ (locally necessary, globally transient).

G16 provides a kernel-level measurement: the Gram Lyapunov spectrum shows four phases (build, compress, sustain, expand) with effective rank compressing from 21 to 8 through the middle layers. The Gram-level dynamics should be derivable from the ambient Lyapunov spectrum by restriction to the data manifold.

G13 adds a constraint: the phase structure is capacity-dependent. Small systems show a tight convergence funnel (capacity scar); large systems maintain uniform alignment. The Lyapunov spectrum must therefore depend on $d/k$ — when $d/k$ is large, the contracting directions have room to operate without compressing the data manifold; when $d/k$ is small, contraction forces a visible funnel.

### (d) The hierarchy

Why is $r_1 < r_2 < k < r_3$? G2 establishes the dissociation between variance and causation at every level. Conjecture: $r_1$ counts Lyapunov exponents above a per-instance noise threshold; $r_2$ counts exponents above a causal intervention threshold; $k$ counts non-negative exponents; $r_3$ counts exponents above the quantization floor.

### (e) Autoregressive amplification and basins of attraction

G2 shows $R^2 = 0.9963$ coexists with total causal failure. G14 shows perfect probe accuracy coexists with zero generative accuracy. Characterize the basin of attraction: for what perturbations $\delta$ to $h_T^{(\ell^*)}$ does the trajectory remain in $\xi^{-1}(y)$?

G6 provides a partial answer: perturbations along $\hat{e}_c$ (the $\mathbb{Z}_2$ axis) are always safe — they switch convention but preserve $\hat{y}$. Perturbations in $Z$ are always unsafe. But the basin geometry in the remaining $d - k - 1$ dimensions of $Z^\perp$ is unknown. Does basin volume decrease with generation length $S$?

### (f) The tied constraint and spectral channeling

G8 shows $D = E^\top$ forces $r_{90} = 1$ at scale ($p = 7\text{B}$) while $D \neq E^\top$ yields $r_{90} = 564$ ($p = 14\text{B}$). The tied constraint creates a quadratic form $\Sigma_\ell = E^\top W_O^{(\ell)} W_V^{(\ell)}$ that resonates. Does this make reparametrizability easier (sharper channel → cleaner $Z/Z^\perp$ separation) or harder (shared $E = D^\top$ couples input/output distributions)?

### (g) Task-specific $Z$ and multi-task scaling of $K$

G11: $Z$ is task-specific with zero cross-task transfer. Must $K$ grow with the number of tasks? Or does the universality of the architectural scaffold (G1, G3, G8, G10) imply $K = O(\sum_{\text{tasks}} k_{\text{task}})$?

### (h) The probe-generation gap as a formal obstruction

G14 shows a linear map can extract $y$ from $h^{(\ell)}$ at every layer. But generating from extracted information fails. Formalize: what property of $F$'s dynamics makes the information in $Z$ **readable** (by a linear probe, which bypasses $F$) but **not generatively accessible** (by injection into $F$'s input, which must traverse $F$)?

Conjecture: the information in $Z$ at layer $\ell$ is encoded in directions that are **transverse** to $F$'s flow — the probe can read across the flow, but injecting along the flow misaligns with subsequent layers' expectations. This is the formal content of G5: high $R^2$ between layers (the flow is near-linear) but causal failure (the 0.04% nonlinear residual is the actual computational content, and it cannot be reconstructed from the linear approximation).

---

## Summary of empirical constraints on answers

| Question | Empirical constraint | Source |
|:--|:--|:--|
| (a) $f^*$ exists? | **Settled.** Information-theoretic proof + cocycle flatness | Foundation, G17 |
| (a) $f^*$ unique? | 21 pairwise constraints consistent to 2.86%; 7-lang $Z$ achieves 97% retrieval | G17 |
| (a) $E'$ exists? | $N = 7$ working encoders; midpoint of two is trajectory-compatible at $\ell_c$ | G18, G7b |
| (a) $K$ small? | Only $r_2 = 8$ causal dims at readout; $r_2 = 1$ at scale | G1, G2, G8 |
| (a) Basin? | Information extraction succeeds, generation fails | G14 |
| (b) Transition | 1D weight surgery succeeds, 10D fails | G15 |
| (c) Interior rigidity | Full rank required at every cross-section | G3, G4 |
| (c) Capacity dependence | Funnel in small models, not large | G13 |
| (d) Hierarchy | Variance ≠ causation at every level | G2 |
| (e) Basin geometry | $\mathbb{Z}_2$ axis is safe; $Z$ perturbations are not | G6 |
| (f) Tied constraint | Rank-1 readout in tied 7B; rank-564 in untied 14B | G8 |
| (g) Multi-task | 0/5 cross-task transfer of $Z$ | G11 |
| (h) Probe gap | Linear probe perfect, generation dead | G14 |