This is a rich theoretical framework. Let me work through each question using the givens and definitions provided.

---

## (a) Reparametrizability

**(a.i)** The ratio is **$O(1/L)$ modulated by entanglement**.

The key decomposition: layers $\ell \leq \ell_c \approx L/3$ are convention-invariant (G7a), so $E'$ only needs to place inputs into a regime where these early layers act universally. Layers $\ell_c < \ell \leq \ell^*$ have rank-1 entanglement (G15) — fixable by 1D kernel surgery. Layers $\ell > \ell^*$ are readout. So $C_{\text{extract}}$ scales with retraining $E', D'$ (thin adapter layers) while $\{g_\ell\}$ stays frozen, giving roughly $O(K \cdot d / (n \cdot d \cdot L)) \sim O(K/(nL))$. The pretrained $F$ confers massive advantage because the 79% of parameters in $\{g_\ell\}$ (G9) encode $f^*$ and need not be retrained.

However, this is not purely $O(1/L)$ because of the basin-of-attraction constraint (G14). The cost has an additional term from ensuring trajectory compatibility, not just information preservation.

**(a.ii)** $E'$ can be **partially constructed via linear algebra, but requires gradient refinement**.

The argument: By G17, all $N$ encoders project onto the same $Z$ at $\ell^*$. By G7b, the midpoint of two convention-matched states at $\ell_c$ is trajectory-compatible. So the centroid $\bar{h}(x) = \frac{1}{N}\sum_i h_i(x_i)$ (for matched tasks) at $\ell_c$ should be trajectory-compatible, giving an initial $E'$ via:

$$E' = \arg\min_{W} \sum_{x} \|W \cdot \mathbf{e}_{a'(x)} - \bar{h}^{(\ell_c)}(x)\|^2$$

This is a least-squares problem. But G14 says information preservation ≠ generative access, so this initializer likely needs gradient-based fine-tuning to enter the basin of attraction. The linear-algebraic construction provides an initialization within the solution manifold's convex hull; SGD finds the nearest trajectory-compatible point.

**(a.iii)** Trajectory compatibility requires **$Z^\perp$ coherence across layers**, not just $Z$-membership.

G3 says full rank is required at interior positions. G6 says $Z^\perp$ carries convention/surface form. G14's probe-generation gap means the state must satisfy layer-to-layer consistency conditions that a probe bypasses. Formally: let $\mathcal{B}_\ell \subset \mathbb{R}^d$ be the basin of attraction at layer $\ell$. Then:

$$\mathcal{B}_\ell = \{h : (I + g_{\ell+1})(h) \in \mathcal{B}_{\ell+1}\}$$

recursively from $\mathcal{B}_{\ell^*} = \{h : P_Z h \in Z_{\text{correct}}\}$ (which is a $Z^\perp$-fiber). But each $g_\ell$ is nonlinear, so preimages of $\mathcal{B}_{\ell+1}$ under $(I + g_{\ell+1})$ are complicated. The $Z^\perp$ component must be consistent with the routing mechanism's expectations (even though G10 says the cache is inert at the final position, earlier layers built the state assuming specific $Z^\perp$ structure).

The additional structure is: $P_{Z^\perp} E'(a)$ must lie on the **stable manifold** of $F$'s dynamics restricted to $Z^\perp$. Since the $Z^\perp$ directions have negative Lyapunov exponents (question (c)), perturbations decay — but they must start within the contracting basin, not outside it.

---

## (b) Phase transition

**Yes, a critical $\|E\|^*$ exists, and it lies at rank 1 of $E_\ell$.**

G15 directly: 1D projection succeeds, 10D fails. The entanglement tensor $E_\ell$ is rank-1 dominant. Define:

$$\|E\|^* = \sigma_1(E_\ell) \cdot \epsilon$$

where $\epsilon$ is the tolerance set by the basin width in $Z$. For $\text{rank}(E_\ell) = 1$: the convention-content coupling is a single bilinear term $E_\ell \approx u \otimes v \otimes w$ with $u \in Z, v \in Z, w \in Z^\perp$. This can be surgically removed (G15). For $\text{rank}(E_\ell) > 1$: the coupling is distributed, and removing it destroys $Z$-structure.

The phase transition occurs at:

$$\|E\|^* = \sigma_2(E_\ell)$$

i.e., the second singular value of the entanglement tensor. When $\sigma_2 \ll \sigma_1$ (as observed), clean separation is possible. When $\sigma_2 \sim \sigma_1$, it is not. This predicts the transition is between rank 1 and rank 2 of effective entanglement — the empirical observation of failure at 10D is consistent since the 10D projection removes directions that carry entangled $Z$-information.

For the reparametrizability implication: when $\sup_\ell \|E_\ell\| < \sigma_2$, one can set $K = O(r_2)$ (just the causal dimensions). When above, $K$ must grow to accommodate entangled dimensions, or $g_\ell$ must be retrained.

---

## (c) Lyapunov characterization

**The conjecture is almost correct but needs refinement for the four-phase structure.**

The Lyapunov spectrum of $\mathcal{J}(x)$ decomposes as:

- $k$ exponents with $\lambda_i \geq 0$: these are the $Z$-directions. They must be non-negative because $P_Z h^{(\ell^*)}$ determines $\hat{y}$ (G1) — if they were negative, the signal would decay and readout would fail.
- $d - k$ exponents with $\lambda_i < 0$: these are the $Z^\perp$-directions. They must be negative because G1 says $Z^\perp$ is causally inert at readout — information placed there is transient.

G3 (interior rigidity) is compatible because full rank ≠ all positive exponents. The map $(I + g_\ell)$ has Jacobian $(I + J_\ell)$; for $\|J_\ell\| < 1$ (residual regime), all singular values of $(I + J_\ell)$ exceed $1 - \|J_\ell\| > 0$, so no rank is lost at any single layer. But the *product* $\prod_\ell (I + J_\ell)$ can have singular values that grow (positive exponents) or decay (negative exponents) while each factor is individually full-rank.

**Capacity dependence (G13):** The ratio $d/k$ controls the spectrum's structure. When $d/k$ is large (large models), the $d-k$ contracting directions have ample room — the contraction is gentle ($\lambda_i$ slightly negative), and the data manifold isn't squeezed (no funnel). When $d/k$ is small (small models), contraction must be aggressive to compress $d - k$ directions into transience by $\ell^*$, creating a visible funnel at $\ell_c$.

Formally: the mean contracting exponent satisfies $\bar{\lambda}_{-} \sim -\frac{1}{L}\log(d/k)$. For $d/k$ large, $|\bar{\lambda}_{-}|$ is small and contraction is diffuse. For $d/k$ small, $|\bar{\lambda}_{-}|$ is large and contraction concentrates at $\ell_c$.

**Gram-to-ambient derivation:** The Gram Lyapunov exponents $\mu_i^{(\ell)}$ are the restriction of the ambient exponents to the data manifold. If the data manifold has intrinsic dimension $r_3$, then $\mu_i$ for $i \leq r_3$ track the ambient $\lambda_i$ projected onto data directions. The four Gram phases (build, compress, sustain, expand) map to: building $Z$ structure (positive $\lambda_i$ in $Z$-directions dominate), compressing convention variation (negative $\lambda_i$ in convention-specific $Z^\perp$ directions), sustaining (near-equilibrium), and readout expansion at $\ell^*$ (the spectral peak $\sigma_1(\Sigma_{\ell^*})$ from G8).

---

## (d) The hierarchy $r_1 < r_2 < k < r_3$

The four ranks correspond to four thresholds on the Lyapunov spectrum:

- **$r_1 \approx 6$**: Number of exponents $\lambda_i > \lambda_{\text{instance}}$, where $\lambda_{\text{instance}}$ is the threshold above which a direction carries enough signal to determine $\hat{y}$ for a *single* input. This is task-instance-specific because different inputs activate different subsets of $Z$.

- **$r_2 \approx 8$**: Number of exponents $\lambda_i > \lambda_{\text{causal}}$, where $\lambda_{\text{causal}}$ is the threshold below which causal interventions (zeroing, projecting) don't affect the output distribution over the *task distribution*. $r_2 > r_1$ because some directions are individually sub-threshold but collectively necessary across inputs.

- **$k \approx 20$**: Number of non-negative exponents ($\lambda_i \geq 0$). This is the convention-invariant subspace dimension. $k > r_2$ because some $Z$-directions carry information that is convention-invariant and extractable by probes (G14) but not causally active at readout — they encode task structure that is read out indirectly through nonlinear interactions.

- **$r_3 \approx 69$**: Number of exponents $\lambda_i > \lambda_{\text{floor}}$, where $\lambda_{\text{floor}} \sim -\frac{1}{L}\log(\text{numerical precision})$. These directions haven't fully decayed to noise by $\ell^*$. $r_3 \gg k$ because many $Z^\perp$ directions decay slowly (slightly negative exponents) and retain statistical variance without causal relevance — exactly the variance-causation dissociation of G2.

**Why dissociation is inevitable:** The training objective (next-symbol prediction) optimizes the *composition* $D \circ F \circ E$, not individual directions. Gradient descent finds solutions where high-variance directions serve the loss landscape's curvature (enabling stable optimization), not causal readout. The top-$r_3$ PCA directions are optimized for gradient flow during training; the causal directions are optimized for functional correctness. These are generically different objectives, so their solutions occupy different subspaces.

---

## (e) Autoregressive amplification and basins

The basin of attraction $\mathcal{B}(y, S) \subset \mathbb{R}^d$ at $\ell^*$ for target $y$ with generation length $S$ is:

$$\mathcal{B}(y, S) = \{h : \xi(\text{AR}(h, S)) = y\}$$

where $\text{AR}(h, S)$ denotes $S$-step autoregressive generation starting from $h$.

**Structure from the givens:**

Along $\hat{e}_c$ ($\mathbb{Z}_2$ axis, G6): the basin is the entire real line — any perturbation $\delta \cdot \hat{e}_c$ preserves $\hat{y}$ (just switches convention). So $\mathcal{B}$ is an infinite cylinder along $\hat{e}_c$.

In $Z$ (causal subspace): the basin is thin. G2 says $r_2 = 8$ dimensions are causally necessary, and G14 says even information-preserving perturbations fail generatively. The basin width in $Z$ is bounded by the nonlinear residual — the 0.04% that separates $R^2 = 0.9963$ from causal sufficiency (G5).

In $Z^\perp \setminus \text{span}(\hat{e}_c)$: unknown from the givens, but G3 (interior rigidity) implies these directions are needed for routing. Since G10 says the cache is inert, the $Z^\perp$ information at $\ell^*$ serves the *generating* position's own trajectory, not other positions'. The basin width here is set by the contracting Lyapunov exponents — perturbations in these directions decay, so the basin is wider than in $Z$.

**Basin volume vs. generation length:** Yes, $\text{vol}(\mathcal{B}(y, S))$ decreases with $S$.

Each generation step $t$ adds a constraint: the $t$-th symbol must be correct. At step $t$, the perturbation $\delta_0$ at $\ell^*$ has been amplified along expanding directions and contracted along contracting ones. The relevant quantity is the expanding Lyapunov exponents through the autoregressive loop (which includes re-encoding and re-processing). If $\lambda_{\max}^{\text{AR}}$ is the top exponent of the autoregressive map, then:

$$\text{width}(\mathcal{B}, S) \sim \epsilon_{\text{symbol}} \cdot e^{-\lambda_{\max}^{\text{AR}} \cdot S}$$

where $\epsilon_{\text{symbol}}$ is the per-step tolerance (distance to decision boundary in logit space). This exponential shrinkage explains why G14's probe-to-generation gap is so severe: a probe needs only single-step correctness, while generation needs $S$-step trajectory stability.

G12 provides a consistency check: convention flipping accelerates generation (+160% at 128 steps) without improving asymptotic accuracy. This means the basin along $\hat{e}_c$ doesn't shrink with $S$ (it's an exact symmetry), but the effective basin in other directions does shrink, and the convention-flip places the state in a region with faster convergence (fewer steps to reach the correct output), not a wider basin.

---

## (f) Tied constraint and spectral channeling

**The tied constraint makes reparametrizability easier in low dimensions and harder in high dimensions.**

With $D = E^\top$: $\Sigma_\ell = E^\top W_O^{(\ell)} W_V^{(\ell)}$ is a quadratic form in $E$. The extreme spectral concentration ($\sigma_1/\sigma_2 = 10.42$, $r_{90} = 1$ at $d = 3584$) means:

1. **$Z$ is essentially 1-dimensional** at readout. This is maximally easy to target — $E'$ needs only to align a single direction with $\Sigma_{\ell^*}$'s top eigenvector.
2. **But $E = D^\top$ couples input and output.** Changing $E'$ automatically changes $D' = (E')^\top$, constraining the decoder. The new $D'$ must simultaneously be a good readout *and* $(D')^\top$ must be a good encoder. This is a rank-1 feasibility condition: $E'$ must satisfy $(E')^\top \cdot v_1 \propto$ correct output logits, where $v_1$ is $\Sigma_{\ell^*}$'s top right singular vector.

For $K \ll n$: the tied constraint is **easier** because the rank-1 channeling means $E'$ only needs to get one direction right, and the coupling $D = E^\top$ automatically ensures decoder alignment.

For $K \sim n$ or for multi-task (G11): the tied constraint is **harder** because different tasks require different $Z$ subspaces, and the single shared $E = D^\top$ must serve all of them through a rank-1 channel.

The untied system ($r_{90} = 564$) distributes information across 564 dimensions, requiring $E'$ to align a high-dimensional subspace — harder per-task but more flexible across tasks. The reparametrizability cost scales as $O(r_{90} \cdot K)$, so tied systems with $r_{90} = 1$ are cheapest per-task.

---

## (g) Multi-task scaling of $K$

**$K = O(\sum_{\text{tasks}} k_{\text{task}})$ with sharing, but not $O(\max k_{\text{task}})$.**

G11 says $Z_{\text{task}_1} \neq Z_{\text{task}_2}$ with zero cross-task transfer. But G1, G3, G8, G10 hold universally — the architectural scaffold doesn't change. This means:

- The scaffold $F$ is a universal dynamical system.
- Each task activates a different $k$-dimensional subspace of $F$'s state space.
- $E'$ must map inputs to the correct task-specific subspace.

If tasks have disjoint $Z$'s: $K \geq \sum k_{\text{task}}$ (worst case).
If tasks share partial structure: $K \geq \dim(\text{span}(\bigcup Z_{\text{task}}))$.

The scaffold universality (G1, G3, G8) implies $K$ need not grow with $L$ or $d$ — only with the total dimension of task-relevant subspaces. Since $k_{\text{task}} \approx 20$ and $d = 2048$, even 100 tasks would require $K \leq 2000 \ll n$ if their $Z$'s are fully disjoint. In practice, partial overlap is likely, so $K = O(\sum k_{\text{task}} - \text{overlap})$.

---

## (h) The probe-generation gap

**Formal statement:** Let $\phi_\ell: \mathbb{R}^d \to \mathcal{Y}$ be a linear probe at layer $\ell$, and let $\text{AR}_\ell: \mathbb{R}^d \to \mathcal{A}^S$ be autoregressive generation from a state injected at layer $\ell$. Then:

$$\sup_\ell \text{acc}(\phi_\ell) = 1.0, \quad \sup_\ell \text{acc}(\xi \circ \text{AR}_\ell \circ \text{inject}_\ell) = 0.0$$

The obstruction is **transversality of information encoding relative to the dynamical flow.**

At layer $\ell$, the task content $y$ is encoded in direction $v_\ell \in Z$. The flow $(I + g_{\ell+1})$ maps $h^{(\ell)}$ to $h^{(\ell+1)}$. The Jacobian $(I + J_{\ell+1})$ rotates the state, and the information about $y$ at layer $\ell+1$ is in direction $v_{\ell+1} = (I + J_{\ell+1}) v_\ell + \delta_\ell$, where $\delta_\ell$ is the **nonlinear correction** (the 0.04% residual from G5).

A probe reads $v_\ell$ directly — it doesn't traverse $F$. But injection at layer $\ell$ must traverse layers $\ell+1, \ldots, L$, and each layer expects the state to contain not just $v_\ell$ (the information) but also the specific $Z^\perp$ scaffolding that enables correct nonlinear correction $\delta_\ell$ at each subsequent layer.

**Formally:** Define the **flow-compatible submanifold** $\mathcal{M}_\ell \subset \mathbb{R}^d$ as the image of the data distribution under layers $1, \ldots, \ell$:

$$\mathcal{M}_\ell = \{h^{(\ell)}(x) : x \in \text{supp}(\mathcal{D})\}$$

The probe extracts information from $\mathcal{M}_\ell$ by projection onto $Z$. But injection creates a state $\tilde{h} \notin \mathcal{M}_\ell$ (because the injected state lacks the correct $Z^\perp$ component). The subsequent flow $(I + g_{\ell+1})(\tilde{h})$ deviates from $(I + g_{\ell+1})(h^{(\ell)})$ by:

$$\|(I + g_{\ell+1})(\tilde{h}) - (I + g_{\ell+1})(h)\| \geq \|E_{\ell+1}\| \cdot \|P_Z \tilde{h}\| \cdot \|P_{Z^\perp}(\tilde{h} - h)\|$$

where $E_{\ell+1}$ is the entanglement tensor. Even if $P_Z \tilde{h} = P_Z h$ (information preserved), the $Z^\perp$ mismatch couples into $Z$ through $E_{\ell+1}$, and this error compounds over $L - \ell$ layers.

This is exactly the content of G5: the layer-to-layer map is 99.96% linear, but the 0.04% nonlinear residual carries the computation, and it depends on the full state (both $Z$ and $Z^\perp$). The probe bypasses this dependency; generation cannot.

**The gap is therefore a geometric property of $F$'s foliation:** $\mathcal{M}_\ell$ is a thin submanifold of $\mathbb{R}^d$, $Z$-projection from $\mathcal{M}_\ell$ is bijective (probes work), but the $Z$-fiber through any point of $\mathcal{M}_\ell$ (the set of states with the same $Z$-projection) intersects $\mathcal{M}_\ell$ in exactly one point. Injection picks the wrong point in the fiber.