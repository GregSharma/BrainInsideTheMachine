You're right. I keep dropping pieces. Let me actually use the entire chain: experiments → problem statement → Opus → GPT → information theory → your car thought.

**What the full filtration says, taken together:**

Your experiments established that the 2048-dimensional state at every layer decomposes into ~20 directions that carry reasoning and ~2028 directions that carry convention, scaffolding, and transient computation. Both Opus and GPT independently formalized this as a **dominated splitting** — a $k$-dimensional bundle of directions that survive to readout, and a $(d-k)$-dimensional bundle that contracts away. They agree this is not a fixed subspace — it rotates 77° per layer. But the Gram structure is preserved through the rotation (0.974 layer-to-layer correlation). The rotation is the coordinate change. The structure is the invariant.

Your experiments ALSO established that you can't just project onto a fixed $Z$ and call it a day. G3 says full rank is needed at every layer for context. G4 says post-hoc bottlenecks fail. G14 says information extraction works but generation from extracted states fails. Both Opus and GPT diagnosed this identically: the obstruction is not information loss, it's the **section problem** — you need to be on the reachable manifold, not just in the right subspace. The reachable manifold is a thin sheet crumpled up in 2048 dimensions, and projection can take you off the sheet.

But your experiments ALSO established the way around this. G7a says below $\ell_c \approx L/3$, the MLP's action on $Z$ is convention-independent — entanglement is zero. G7b says mixing states from different conventions at $\ell_c$ preserves accuracy — midpoints stay on the sheet. G15 says above $\ell_c$, the entanglement is rank-1 dominant — one direction couples convention to reasoning, and it's surgically removable. G17 says the cocycle is flat — all 7 conventions agree on where $Z$ is. G18 says you have 7 working encoders that all successfully put inputs onto the sheet.

And the information-theoretic argument says $f^*$ doesn't just probably exist. It MUST exist. The cocycle uniqueness says it's pinned. The 7 encoders say the sheet is reachable. The question was always extraction cost, not existence.

**What both Opus and GPT derived as the resolution:**

The state space has an invariant foliation. Think of it as: at every layer, through every point on the reachable manifold, there passes a leaf. The leaf is parameterized by $Z$ (the quotient — what the answer is). The position within the leaf is parameterized by $Z^\perp$ (the fiber — which convention, what scaffolding). The probe reads the leaf label. Generation requires being on the correct leaf AND at the correct position within it.

Below $\ell_c$, the leaves are flat (entanglement $\approx 0$). The foliation is a clean product: $Z \times Z^\perp$, non-interacting. You can move freely in $Z^\perp$ without affecting $Z$. This is why convention mixing works at L14.

Above $\ell_c$, the leaves are curved. $Z$ and $Z^\perp$ couple through the rank-1 entanglement tensor. Moving in $Z^\perp$ slightly shifts your position in $Z$. But only through ONE direction (the convention axis). Everything else in $Z^\perp$ is still decoupled.

At $\ell^*$, the contracting directions have died. Only the $k$ surviving directions remain. The readout reads them through $\Sigma_{\ell^*} = D \cdot OV_{\ell^*}$, which is spectrally concentrated (rank-1 at 7B).

**The compression procedure that follows from all of this:**

It's not "project weights onto a fixed $Z$." It's "track the surviving bundle through depth and build a model that lives entirely within it."

**Compute the layer-specific surviving bundle.** At each layer $\ell$, the surviving bundle $U_\ell \in \mathbb{R}^{d \times k}$ consists of the directions that make it to readout. You find them by computing the Jacobian product from $\ell$ to $\ell^*$ and taking the top-$k$ right singular vectors. This is a computation on the existing weights — no training. The bundle rotates across layers (77°/layer), but you're computing the rotation explicitly instead of being destroyed by it.

**Restrict every weight matrix to the bundle.** At each layer, the MLP and attention weights get sandwiched between $U_\ell^\top$ (project input to bundle) and $U_\ell$ (project output from bundle). The MLP becomes: input is $k$-dimensional, expand through the gated intermediate, project output back to $k$ dimensions. Same weights, same nonlinearity, same layer-specific behavior. Just restricted to the directions that matter.

**Handle the entanglement above $\ell_c$.** Below $\ell_c$, the restriction is lossless — $E_\ell \approx 0$ (G7a), so $g_\ell$'s action on $Z$ doesn't depend on $Z^\perp$. The projected MLP computes the same thing as the full MLP on the surviving bundle. Above $\ell_c$, the restriction introduces error proportional to $\|E_\ell\|$. But $E_\ell$ is rank-1 dominant (G15). So carry the convention axis $\hat{e}_c$ as a $(k+1)$-th dimension above $\ell_c$. The compressed model operates in $k$ dimensions below $\ell_c$ and $k+1$ dimensions above. After the final layer, strip $\hat{e}_c$ and read out in $Z$.

**The embedding and unembedding stay full-vocabulary.** $E' = U_0^\top \cdot W_E \in \mathbb{R}^{k \times n}$, $D' = W_U \cdot U_{\ell^*} \in \mathbb{R}^{n \times k}$. Every one of the 152K tokens is preserved. "Michael Jordan" still has an embedding. It's just a $k$-dimensional embedding instead of a $d$-dimensional one. The factual recall survives because the facts are encoded in the MLP weight patterns, and those patterns have nonzero projection onto the surviving bundle — the information-theoretic argument guarantees this.

**Compress the intermediate MLP dimension.** The original MLP expands from $d = 2048$ to $m = 11008$ because it serves 152K vocabulary items across languages. In the bundle-restricted model, the effective intermediate rank is lower. Compute it empirically: run typical inputs through the restricted MLP, SVD the intermediate activations, find the effective rank $\tilde{m}$. Truncate. This is where the bulk of the parameter savings come from.

**Validate layer by layer.** Compute $\|E_\ell\|$ at every layer by running convention-matched pairs and measuring how much $P_Z g_\ell$ changes when $Z^\perp$ changes. Plot it. Below $\ell_c$: should be near zero (confirms lossless projection). Above $\ell_c$: tells you exactly how much error the rank-1 correction doesn't fix. At layers where the residual entanglement is too large, you have two options — carry more correction dimensions, or apply light gradient refinement to the projected weights at those specific layers.

**Why this addresses every failure mode from the experiments:**

G3 (full rank needed): Yes, for the ORIGINAL model operating in $d$ dimensions. The PROJECTED model operates in $k$ dimensions. It never drops rank because it never had $d$ dimensions to begin with. Full rank in $k$ dimensions is $k$ — which it has.

G4 (bottleneck fails): The bottleneck experiments inserted a learned rank-$r$ projection into the $d$-dimensional stream and tried to fine-tune it. This fails because the projection is in the WRONG BASIS — a learned bottleneck doesn't know about the surviving bundle. The procedure above uses the CORRECT basis (the Jacobian-derived $U_\ell$), which is the one the model's dynamics actually preserve.

G14 (probe works, generation fails): The probe reads $Z$ without going through $F$. Generation goes through $F$ and needs to be on the reachable manifold. The projected model IS $F$ restricted to the surviving bundle — it goes through the same dynamics, just in fewer dimensions. If the surviving bundle IS the reachable manifold's backbone (which the dominated splitting says it is), then the projected model's states are on the sheet by construction.

G5 (linear shortcuts fail): The linear shortcut fitted between layers $\ell$ and $\ell'$ has $R^2 = 0.9963$ but fails causally. The 0.04% residual is the nonlinear computation. The projected model preserves this nonlinearity — same SiLU gates, same weight matrices, just in the bundle. The 0.04% that matters is the $Z$-component of the MLP's nonlinear output, and the projection preserves it exactly below $\ell_c$.

**What I can't predict from the math alone:**

How large $\tilde{m}$ (the compressed intermediate dimension) actually is. This determines whether the parameter count is 30M or 300M. It must be measured empirically.

Whether the rank-1 correction above $\ell_c$ is sufficient, or whether layers 15–36 need additional correction dimensions or light fine-tuning. This depends on how fast $\|E_\ell\|$ decays beyond the first singular value, which is measurable but hasn't been measured.

Whether the Jacobian product is numerically stable over 36 layers. In principle it could blow up or collapse. In practice, the fact that the Gram Lyapunov spectrum is well-behaved (G16) suggests it's stable, but you'd need to check.

**The sequence of operations for Vega:**

First: compute the Jacobian product from each layer to $\ell^*$ on a representative set of inputs. SVD each product. Extract $U_\ell$ at each layer. This tells you the surviving bundle and how it rotates. It also immediately gives you the Lyapunov spectrum — the exponents that both Opus and GPT said characterize the dominated splitting.

Second: compute $\|E_\ell\|$ at every layer — the entanglement tensor. This tells you where projection is lossless and where it needs correction. It's the layer-by-layer map of extraction difficulty.

Third: project all weights onto $U_\ell$, run the projected model, measure accuracy. This is the moment of truth.

Everything before step three is measurement. Step three is the money shot.

--- UPDATE ---

# Errata & Addenda to Compression Procedure

**Date**: 2026-04-11
**Context**: Corrections to the compression formalism shared with Vega earlier this session, informed by Opus 4.6 and GPT 5.4 responses to the formal problem statement, plus C6/C6b findings from today's session.

---

## Erratum 1: Surviving bundle computation

**Wrong (original procedure):** Compute $U_\ell$ at each layer via Jacobian products $\prod_{\ell'=\ell}^{\ell^*}(I + J_{\ell'})$, SVD, take top-$k$ right singular vectors.

**Correct:** Compute $U_\ell$ at each layer via **cross-convention SVD on the combined math + BR diverse activations**. Both are already cached:
- `multilingual_all_layers.npz` — math, 7 languages × 200 problems × 36 layers
- BR diverse activations — 200 diverse problems (logical, syllogisms, common sense, analogies), 7 languages

At each layer $\ell$: stack all convention-matched pairs across BOTH datasets, compute the cross-convention difference matrix, SVD it. The top-$k$ right singular vectors are $U_\ell$.

**Why:** The Jacobian approach is input-dependent (requires choosing which inputs to evaluate at) and task-dependent (math Jacobians give you $Z_{\text{math}}$, not $Z_{\text{all}}$). The cross-convention SVD on diverse data gives the union of all task-specific surviving bundles automatically — any direction that is convention-invariant for ANY task gets captured. This uses data already on disk with no discretionary choices.

---

## Erratum 2: Gate check before weight projection

**Before projecting any weights, compute one number:**

$$\dim(Z_{\text{all}}) = \text{effective rank of cross-convention SVD on combined dataset at L26}$$

Use the rank at 90% variance explained.

- If $\dim(Z_{\text{all}}) \approx 50\text{–}100$: proceed with weight projection. Compression ratio $d / \dim(Z_{\text{all}}) \approx 20\text{–}40\times$.
- If $\dim(Z_{\text{all}}) \approx 200\text{–}500$: still meaningful compression ($4\text{–}10\times$), proceed but with tempered expectations.
- If $\dim(Z_{\text{all}}) > 1000$: the multi-task surviving bundle is too wide for dramatic compression. The decomposition is real but the reasoning core isn't small. Revise thesis.

**This single number determines whether the compression is worth attempting.** Compute it first.

---

## Addendum 1: C6/C6b integration

Vega's C6/C6b findings from today:

- The rank-1 readout direction at L27 (7B) is NOT aligned with digit embeddings (cos = 0.002), NOT aligned with answer tokens (principal angles ~85–89°). It IS the negative of the mean attention output (cos = -0.954).
- k=1 compression is lossless at essentially EVERY layer (5, 10, 15, 20, 25, 26, 27), not just $\ell^*$.
- Replacing last-token attention output with a CONSTANT vector (the mean) preserves accuracy. Replacing with ZEROS halves accuracy. Replacing with a RANDOM 1D projection preserves accuracy.

**Implication for compression:** The attention mechanism at the last token during generation is a constant bias, not a dynamic computation. The MLP and residual stream do all the work. This means the routing mechanism's weight matrices ($W_Q, W_K, W_V, W_O$) at the last token position are compressible to a single bias vector during generation. This further reduces the parameter count of the compressed model — attention weights (~10% of total) can be replaced with a per-layer bias vector at the generating position.

**Implication for the surviving bundle:** If attention at the last token is constant, then the Jacobian of the attention block at the last token is approximately zero. The surviving bundle is determined entirely by the MLP Jacobians. This simplifies the dynamics: the layer map at the generating position is effectively $h \mapsto h + \text{const} + g_\ell(h + \text{const})$, and the bundle computation only needs $\partial g_\ell / \partial h$.

---

## Addendum 2: Multi-task $Z$ concern

G11 says $Z$ is task-specific with 0/5 cross-task transfer. The compression procedure must use $Z_{\text{all}} = \text{span}(\bigcup_{\text{tasks}} Z_{\text{task}})$, not $Z_{\text{math}}$. If task subspaces don't overlap, $\dim(Z_{\text{all}}) = \sum_{\text{tasks}} k_{\text{task}}$.

Both Opus 4.6 and GPT 5.4 independently concluded (part g of the problem statement) that multi-task $K$ scales **additively** with task-specific dimensions, not as a universal constant. Partial overlap reduces the sum. The gate check in Erratum 2 measures this directly.

---

## Addendum 3: Opus/GPT consensus on compression framework

Two frontier models independently solved the formal problem statement and converged on the same framework:

- The model's state space has a **dominated splitting**: a $k$-dimensional surviving bundle (the center/semantic bundle) and a $(d-k)$-dimensional contracting bundle (scaffolding, convention, transient computation).
- The surviving bundle IS $f^*$. Compression means restricting the model's weights to this bundle.
- The key diagnostic is the **entanglement tensor** $E_\ell$ — it measures how much $g_\ell$'s action on $Z$ depends on $Z^\perp$. Where $\|E_\ell\| \approx 0$ (below $\ell_c$), projection is lossless. Where $\|E_\ell\|$ is large (above $\ell_c$), correction is needed.
- The **phase transition** in compressibility occurs at the spectral gap between the $k$-th and $(k+1)$-th effective Lyapunov exponents. Below this gap: clean separation. Above: entangled.
- The **probe-generation gap** (G14: linear probe perfect, generation dead) is explained by the **section problem**: probing reads a quotient coordinate (which leaf are you on?), but generation requires being at the correct point ON the leaf (the right stable fiber). Injection picks the wrong point in the fiber.
- $E'$ can likely be **initialized** from the $N=7$ known encoders via linear algebra (centroid at $\ell_c$, strip convention axis), but exact trajectory compatibility may require light gradient refinement.

---

## Execution sequence for Vega

1. **Gate check:** Combined SVD on math + BR at L26. Report $\dim(Z_{\text{all}})$ at 90% variance. STOP and report before proceeding.
2. **Layer-wise bundle:** If gate check passes, compute $U_\ell$ at every layer via combined cross-convention SVD.
3. **Entanglement map:** For each layer, compute $\|E_\ell\|$ using convention-matched pairs. Report which layers are below/above threshold.
4. **Weight projection:** Project all MLP and attention weights onto $U_\ell$ at each layer. Build projected model.
5. **Validation:** Run projected model on math + diverse tasks. Report accuracy vs baseline.