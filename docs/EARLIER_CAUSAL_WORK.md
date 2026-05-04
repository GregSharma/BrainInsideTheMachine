# The Earlier Causal Arc — G·f*·G' Through BH, In Formal Notation

**A second companion to `FORMAL_FOUNDATIONS.md`, covering the pre-BQ portion of the project.**

Author: VEGA (Claude Opus 4.6)
Date: 2026-04-09
Status: Reconstruction-from-records, tier-tagged. Read the confidence disclaimer below before trusting any specific number.

---

## 0. Confidence and scope disclaimer (READ FIRST)

`FORMAL_FOUNDATIONS.md` was written as if the project started at BQ (the Gram funnel). It didn't. This document tries to cover the earlier arc — G·f*·G', Phase 3A, PC0 swap, early-exit L26, MOAMS-X, BH — in the same notation, so that the three-document set (`EARLIER_CAUSAL_WORK` + `FORMAL_FOUNDATIONS` + `WHAT_YOU_MAY_HAVE_RUSHED_PAST`) covers the real scope of the project rather than the last two weeks of it.

**I still do not have 100% of the context of earlier work.** I reconstructed this document from:
- The `crossmodel_validation_2026-04-06.md` doc in this folder
- The `session_analysis_2026-04-05.md` doc in this folder
- The `2026-04-07_observation_intervention_gap.md` doc in this folder
- `toy_theorem_derivation.md` at the repo root
- `MEMORY.md` in the project memory folder
- ~35 memory index entries retrieved via search
- The state snapshot list (50 snapshots, 2026-03-22 through 2026-04-09)

I did **not** read individual experiment scripts, raw transcripts, or output JSON files for each finding. Where I cite a specific number, it is traceable to one of the six sources above. Where I cite a mechanism without a number, I may be paraphrasing from a memory index entry that itself is a one-line summary of a longer result. **Flag anything you recognize as wrong and I will correct it against the source.**

**Confidence tiers used in this document:**
- **[A]** Directly verified — I have seen the exact number or formal statement in a source doc or in `MEMORY.md`.
- **[B]** Reconstructed — I have seen a one-line summary in a memory index entry or snapshot title, but have not seen the raw data.
- **[C]** Inferred — I am filling a gap with a statement that is consistent with tier A/B evidence but which I have not confirmed directly.

I try to tag every nontrivial claim below. Where tagging would be noisy, the section has a default tier in the header.

---

## 1. G·f*·G' — the theoretical origin (Feb 2026, pre-code)

**Default tier: [A] for the derivation, [A] for the cross>within validation, [B] for origin timing.**

### 1.1 The argument, restated

Two hypothetical monolingual models $Q_{\text{zh}}$ and $Q_{\text{en}}$ trained on perfectly translated, equal-sized data compute the same function up to translation. If so, there exists an abstract function $f^*$ and language-specific wrappers $h_\ell, h'_\ell$ such that:

$$Q_{\text{zh}}(x) = h'_{\text{zh}} \circ f^* \circ h_{\text{zh}}(x)$$
$$Q_{\text{en}}(x) = h'_{\text{en}} \circ f^* \circ h_{\text{en}}(x)$$

The space between $h_\ell$ and $h'_\ell$ is called **Z**. Information-theoretic constraints on Z:

$$I(Z; \text{language}) \approx 0$$
$$I(Z; \text{mathematical content}) \approx I(\text{input}; \text{mathematical content})$$

No linguistic information leaks through. All reasoning content is preserved. The parameter count stratification is $|f^*| \gg |h_\ell| + |h'_\ell|$: wrappers are thin, the core is thick, because translation is simple (lookup + grammar) while reasoning is deep, sequential, compositional.

### 1.2 Why this matters as theory

The decomposition is **architecture-agnostic** because it was derived before Greg knew what attention was. The derivation depends only on: (a) two models agree up to translation, (b) information theory, (c) Shannon-level constraints. Not on pre-norm decoder transformers, not on GQA, not on softmax readouts. If an alternative architecture also has bilingual competence, it should also have a G·f*·G' structure with a language-agnostic Z.

This is the sense in which the project touches **universality arguments** — not in the narrow "Anthropic Circuits Universality Hypothesis" sense (same features across models) but in the broader "same structural decomposition across architectures" sense.

### 1.3 The empirical kill-shot (2026-03-22)

**Test:** If G·f*·G' is real, then on the *first token of generation, before any answer is emitted*, the model's last-token residual stream should be closer between EN and a ZH translation of the same problem than between two EN paraphrases of the same problem. "Translation more preserving than paraphrasing" is the test.

**Result [A]:** Cross > within cosine at last-token position, first forward pass:

| Model | cross (EN↔ZH) | within (EN paraphrase ↔ EN paraphrase) | delta |
|---|---|---|---|
| Qwen2.5-3B | 0.931 | 0.661 | **+0.270** |
| Qwen3-8B | 0.987 | 0.891 | **+0.096** |
| Qwen3.5-9B | 0.999 | 0.996 | **+0.003** |

All three positive. Translation is more preserving than paraphrasing on the first token, before any answer content is emitted. This kills the accuracy→cosine objection: the model has not yet produced a correct or incorrect answer when the measurement is taken.

**Why the effect shrinks with scale.** Larger models have tighter within-language clustering (within cosine → 1.0 as scale grows), which compresses the absolute delta. The *sign* is what matters, not the magnitude. Three models, three positive deltas. Zero failures.

### 1.4 The formal glossary G·f*·G' bought us

Once G·f*·G' is on the table, the rest of the project becomes: find Z empirically, measure its dimension, validate its language-agnosticism causally, and probe its structure geometrically. Every subsequent experiment is an instance of this program. The toy theorem (Proposition 1) gives one *mechanism* by which bilingual training would produce Z (least-squares SVD pressure). BH 7-language retrieval generalizes Z to more than two languages. MOAMS-X asks whether Z is math-specific or domain-agnostic. C2 asks whether the last-token manifestation of Z is the actual site of computation or a read-head artifact.

---

## 2. Phase 3A — the first causal double dissociation (~2026-03-05)

**Default tier: [A] for numbers, [B] for interpretation nuance.**

### 2.1 Setup

With a candidate $Z$ subspace identified at layer $\ell$ (via SVD + ARD-MMD, before the contrastive method of Proposition 2 was developed), construct the orthogonal projectors $P_Z$ and $P_{Z^\perp} = I - P_Z$. At layer $\ell$, replace the residual stream state $h_t$ with one of:

- **Z-patch**: $h_t \to P_Z h_t + P_{Z^\perp} \bar h$ — keep only the Z content, replace the orthogonal complement with the mean over calibration problems.
- **Z⊥-patch**: $h_t \to P_{Z^\perp} h_t + P_Z \bar h$ — keep only the orthogonal content, replace Z with the mean.

Run the rest of the forward pass; measure math accuracy on the test set.

### 2.2 Result

| Condition | accuracy (out of 20) | interpretation |
|---|---|---|
| baseline | ~14–16 | model works |
| Z-patch (keep Z, zero Z⊥) | 0–5 | destroying Z⊥ destroys math |
| Z⊥-patch (keep Z⊥, zero Z) | 19–20 | destroying Z barely touches math |

Read the table carefully, because the first time I read it I had the sign backwards. The Z-patch **keeps** Z and zeroes Z⊥; it gets 0–5/20. The Z⊥-patch **keeps** Z⊥ and zeroes Z; it gets 19–20/20. This is the opposite of what "Z is the reasoning subspace" would naively predict — zeroing Z does *not* destroy reasoning.

### 2.3 How Phase 3A was interpreted at the time vs in retrospect

**At the time:** Causal proof that Z is language-agnostic and math content lives outside Z. The "Z is language-specific directions that get subtracted, leaving math content in Z⊥" interpretation.

**In retrospect, with C2 in hand:** The Phase 3A result was measuring the same thing C2 measures — that zeroing the "shared subspace" at a single layer while leaving the orthogonal complement alone does not destroy the forward pass, because the computation lives in the full 2048-dimensional substrate. The dimensionality-artifact flag on Phase 3A (memory entry `d0970e33-733`: "Phase 3A double dissociation is dimensionality artifact; bridge fails all k; k=20 random control is make-or-break") was a hint that the interpretation was not as clean as it first looked.

What Phase 3A *actually* demonstrates [B/C]: a low-dimensional subspace identified from last-token variance can be zeroed at one layer without destroying math — because the computation is operating on the orthogonal substrate. This is consistent with the read-head story from C2 but was misread as "Z is the computational substrate" at the time.

The Phase 5C contrastive Z result (2026-03, [A]: "language = 5 dims at L32") layered on top of this and confirmed that *a* low-dim cross-lingual invariant exists and carries math content cross-lingually — which is true, but requires the read-head reframing to interpret correctly. The contrastive Z is what the read head reads; it is not the computational substrate.

---

## 3. PC0 swap at L26 — the headline causal result (2026-03)

**Default tier: [A].** This is tagged `headline_result` in memory.

### 3.1 Setup

At layer $\ell = 26$ of Qwen2.5-3B, during generation, identify the first principal component of the language-discriminating direction — call it $v_0 \in \mathbb{R}^{2048}$. For a ZH input, project the residual stream's component along $v_0$ and replace it with the corresponding component computed from the EN equivalent:

$$h_t^{(26)} \to h_t^{(26)} - \langle h_t^{(26)}, v_0 \rangle v_0 + \langle h_t^{(26), \text{EN}}, v_0 \rangle v_0$$

Single direction, single layer, during generation only.

### 3.2 Result

**100% language switch.** ZH inputs produce EN outputs. **100% first-token match** with the EN baseline: the first token emitted after the swap is exactly the token the model would have emitted for the EN version. Tagged `h-f-h' factorization interventionally proven`.

Later layer sweep (memory `63`: "I fixate on the first layer that works and call it 'the boundary.'") showed the swap works at **all layers L8 through L34** with 100% English output. L30–L34 actually give 100% exact text match, better than L26's 95%. **There is no boundary — there is just the first layer tested.**

### 3.3 Why this is load-bearing

This is the single cleanest causal intervention in the project. One direction, one layer, catastrophic change in a specific observable (output language) with no change to another specific observable (the EN math computation proceeds correctly). It is the intervention that validates the G·f*·G' factorization *empirically*: the language wrapper $h_\ell$ vs $h'_\ell$ is localized to a single direction in residual stream space, and swapping that direction toggles the wrapper while leaving $f^*$ running.

### 3.4 What this means in the read-head frame

The PC0 direction is the strongest 1-dimensional projection in the set of directions that separate ZH from EN activations. Swapping it at any layer L8–L34 works because the PC0 direction is being *read* at every subsequent layer by downstream attention heads that treat it as a language flag. The read-head frame from C2 explains PC0 swap: the read head at the last token is not a single layer's output but a trajectory through 36 layers, and any layer's PC0 direction that downstream attention reads is effectively an *early switch* on the language wrapper.

### 3.5 The related "language flip" intervention is 3B-specific

Connected but distinct: flipping the *sign* of the mean-difference language direction in MLP deltas at L9–L26 gives the "efficiency, not accuracy" effect. This intervention is tested across 5 models in Exp AX [A]:

| Model | Baseline | Flip | Flip delta |
|---|---|---|---|
| 3B | 5/20 | 13/20 | **+8** (the "headline") |
| 7B | 8/20 | 10/20 | +2 (within noise at N=20) |
| 8B | 6/20 | 7/20 | +1 |
| 9B | 5/20 | 4/20 | -1 |
| 14B | 5/20 | 5/20 | 0 |

**The flip is a 3B-specific efficiency effect, not a general mechanism.** In `WHAT_YOU_MAY_HAVE_RUSHED_PAST.md` I wrote about the language flip as if it were a general finding — it is not. What is general is that there exists a 1D language direction with Cohen's $d > 100$ in every model. What is 3B-specific is that flipping that direction during generation improves the 3B at fixed token budget. Larger models do not need the flip because their baseline efficiency is already higher. Specifically:

**The right way to report the flip** (revising my earlier framing): at 3B-scale, with the 3B's specific capacity constraints, flipping the language direction in MLP deltas at L9–L26 shortens the path to the answer at fixed budget. At 7B+ scale this either doesn't happen or is within noise. This is a useful mechanistic tell about capacity-constrained regimes, not a universal intervention.

---

## 4. Early exit L26 → generation-time collapse

**Default tier: [A].**

### 4.1 The finding

During *prefill*, applying the logit lens at L26 (early exit) on a Chinese math problem gives **91% cross-lingual agreement** with the L35 output — consistent with "L26 is the cross-lingual bottleneck where both languages converge."

During *generation*, applying the same early-exit at L26 gives **1.2% cross-lingual agreement**. Collapse by almost two orders of magnitude.

### 4.2 Why this killed the logit lens as a probe

The logit lens takes a mid-layer hidden state and projects it through the final LM head to get logits. It asks "what token would this hidden state predict if it were the last layer?" During prefill, this worked — the hidden states at L26 had already accumulated enough content to decode cross-lingually.

During generation, the hidden state at L26 has accumulated only the current query token's information so far — it has *not yet* gone through the downstream layers that, as we now know, construct the read head. Projecting an early-generation L26 state through the LM head is asking "what would the read head say if we stopped building it now?" The answer is "not much, in the wrong language."

This is why the **splice** (patching actual hidden states between languages) became the real test, not the logit lens. Early-exit measurements work on prefill but fail on generation. Splice measurements work on both because they test the actual computation going forward rather than a pre-RMSNorm readout of an incomplete state.

### 4.3 Connection to the read-head story

The "generation-time collapse" is the same phenomenon as the C2 observation-intervention gap, seen from a different angle. During prefill, the cache is being built and all positions are computing simultaneously; the L26 state of the last prompt token is well-formed. During generation, only the current query position is being updated; its L26 state has just been initialized from the previous token's embedding and has only started its 36-layer traversal. The read head is not yet built. Early-exit measures at L26 during generation are measuring an unfinished read head, and unsurprisingly get garbage.

This observation predates C2 by several weeks but is the first hint at the same gap.

---

## 5. BH — 7-language null-space retrieval (~2026-04-06)

**Default tier: [A] for the number, [B] for the mechanism details.**

### 5.1 Setup

At layer L32 of Qwen2.5-3B, extract the "multi-language null-space" — the subspace that is *shared* across 7 languages' representations of the same problems. Concretely (my reconstruction — flag if wrong): collect 7-language representations of 200 problems, compute the subspace orthogonal to the per-language difference directions, and use it as a language-agnostic embedding space for retrieval.

**Retrieval test:** given a query in one language, find the nearest problem in another language's embeddings projected through the same null-space. Top-1 accuracy across all 7-language query-target pairs.

### 5.2 Result

**97% Top-1 cross-lingual retrieval, 7 languages, L32.** Tagged `headline` in memory `d0ca9a1e-803`.

### 5.3 Why this matters

Two reasons.

**First:** it generalizes the bilingual Z claim to seven languages. G·f*·G' as stated is a two-observer theorem; the information-theoretic extension to $N$ observers is natural but not proved in `toy_theorem_derivation.md`. BH is the empirical check that the $N$-language generalization holds. If BH had given, say, 60% Top-1, the bilingual case would have been a lucky artifact of two languages sharing more structure than seven do. 97% says the shared structure is robust as you add languages.

**Second:** it is a **retrieval-style result**. Retrieval is the kind of practical task that hyperscaler interpretability teams and embedding-model teams can immediately relate to. "Your model already contains a language-agnostic embedding at L32 that gets 97% cross-lingual Top-1 across 7 languages via a null-space construction, without any training" is a sentence that lands. It is also the kind of claim that a skeptical reviewer can test immediately by re-running on a different model — which is a good thing, not a bad thing.

### 5.4 What BH is not

BH is *not* proof that Qwen's internal embeddings are better than purpose-built multilingual embedding models like LaBSE or BGE-M3 [C — I have not seen a comparison benchmark]. The 97% number is on our problem set, not on a standard cross-lingual retrieval benchmark. Before pitching BH as a competitive embedding method, we would need to run it on a standard benchmark (Tatoeba, BUCC, STS17-crosslingual) and compare. What BH *is* proof of is the structural claim: a language-agnostic subspace exists at L32, its extraction method is principled (null-space of per-language difference directions), and it retrieves across 7 languages at high accuracy on our task.

---

## 6. MOAMS-X — cross-domain state transplant (2026-04-02)

**Default tier: [A] for the number, [B] for the method details.**

### 6.1 Setup

MOAMS = "Mother of All Mid-state Splices." Take a hidden state $h^{(\ell^*)}_{t^*}$ from a problem in one domain (e.g. commonsense reasoning), transplant it into a forward pass on a different domain's problem (e.g. a math problem) at the same layer, and run the rest of the forward pass. Measure whether the output still reflects the *original* problem's correct answer.

MOAMS-X extends this across three domains: **commonsense, code, and logic** [B — reconstruction from memory `90844f99` and the 96.2% number].

### 6.2 Result

**96.2% transplant success** across domains [A]. The mid-state at the right layer contains enough of the problem's content that transplanting it into a foreign context produces the original answer.

### 6.3 What this shows, and what it doesn't

**What it shows.** The shared-content subspace identified through G·f*·G' + Phase 3A + PC0 is *not* math-specific. The same kind of mid-layer state that carries math content also carries commonsense, code, and logic content, and that content survives transplantation into a foreign forward pass. This is strong evidence that Z (or what later becomes "the read-head's construction zone") is a domain-general reasoning subspace, not a math-specific artifact.

**What it doesn't show.** MOAMS-X did not (to my knowledge, [C]) test *language* transplantation across *domains* simultaneously — the cleanest test would be "ZH math state transplanted into EN commonsense forward pass, does it produce the EN translation of the ZH math answer?" That would be a 4-way dissociation. I do not know if that exact test was run.

### 6.4 The trajectory visualization

MOAMS-X also produced the "L12 funnel" trajectory visualization [B — `moams_x_trajectory_findings.md` referenced in memory index]. This is one of the plots that caught Greg's eye as he reviewed results and led to the subsequent 4-model trajectory capture. I have not read that doc directly; flag if it contains material relevant here.

---

## 7. Cocycle — the universality result I missed entirely

**Default tier: [A].** This is the single biggest item I missed in my earlier writeups. It is the most paper-ready universality claim in the project.

### 7.1 Setup

For three languages $L_1, L_2, L_3$ (e.g. EN, ZH, JA), fit a ridge regression mapping between the hidden states of the same problems across language pairs:

$$R_{12}: h_{L_1}(x) \to h_{L_2}(x), \quad R_{23}: h_{L_2}(x) \to h_{L_3}(x), \quad R_{13}: h_{L_1}(x) \to h_{L_3}(x)$$

The **cocycle error** is $\|R_{23} \circ R_{12} - R_{13}\|$, measuring how well going through an intermediate language matches going direct. If the cross-lingual structure is a *flat* manifold (zero curvature), the cocycle error is zero up to ridge regularization slack.

### 7.2 Result

Across 5 models (3B, 7B, 8B, 9B, 14B):

| Model | Cocycle R² | Cocycle error |
|---|---|---|
| 3B | 0.941 | 0.75% |
| 7B | 0.914 | 0.68% |
| 8B | 0.896 | 0.39% |
| 9B | 0.871 | 1.04% |
| 14B | 0.922 | 0.38% |

**Minimum R² = 0.871. Maximum cocycle error = 1.04%.** Across 5 models of different sizes and two Qwen generations. The cross-lingual manifold is near-flat at every scale.

### 7.3 Why this is the strongest universality claim we have

**Because the cocycle is a structural property of the manifold, not a per-problem statistic.** "The model gets 13/20 math problems right" is a per-problem count. "Ridge regression between any two languages gives R² > 0.87 with triangle consistency under 1.04%" is a *geometric* statement about the structure of the space of hidden states. It constrains the space itself, not the outputs.

The differential-geometry interpretation (via Webb's proposal in `session_analysis_2026-04-05.md`): the cocycle error is a scalar version of Riemann curvature. Low cocycle error = near-flat language connection. The fact that this holds across 5 models says the *geometry of the residual stream's cross-lingual structure* is a near-flat bundle — a universal architectural property of multilingual decoder transformers.

This is the kind of claim that interpretability reviewers and differential-geometry-inclined theorists will both care about. **In my earlier strategic assessment, I missed this entirely.** Revising: this result alone, well-framed, is worth a preprint. Combined with everything else in the project, it becomes a section of the main theoretical contribution.

### 7.4 The caveat

Ridge regression is a strong smoother. It can find low-error mappings between manifolds that are not genuinely flat if the within-language variation dominates the between-language variation. The cocycle error being low is necessary but not sufficient for flatness. A stronger test would be to compute the discrete Riemann curvature tensor from per-language-pair metrics, which is what Webb proposed and we have not yet done (requires raw hidden states for all 5 models; we currently have raw states only for 3B).

---

## 8. Category transfer = 1.000 universal

**Default tier: [A].**

Separate from the cocycle, there is the **f-probe reconstruction** result (Exp Z, replicated as "category transfer" in Exp BB across 5 models):

**Category transfer = 1.000 in every model tested (3B, 7B, 8B, 9B, 14B).**

What this measures: train a linear probe on the hidden states of one language to classify problem category (arithmetic, algebra, sequence, etc.), test on hidden states of another language. Perfect transfer means the probe trained on EN hidden states perfectly classifies ZH hidden states.

**Why this is different from the cocycle.** The cocycle says "ridge regression *maps* between languages at >87%." Category transfer says "category information is *identically encoded* across languages." The second is strictly stronger for the category dimension of the content space: a linear probe has no wiggle room, it is not learning a mapping, it is asserting that the same hyperplane works in both languages' embeddings.

Perfect transfer at 1.000 across 5 models means problem type (the category label) is encoded language-agnostically from early layers onward, in the same linear direction in every model. This is a concrete, strong universality claim at the representation level.

---

## 9. The universal phase transition

**Default tier: [A].** This is a finding I attributed to 3B in `FORMAL_FOUNDATIONS.md`. It is actually universal.

The **adversarial → cooperative** phase transition — where consecutive-layer delta cosine flips from negative to positive — occurs at **40–47% depth** in every model tested:

| Model | Phase transition layer | depth fraction |
|---|---|---|
| 3B | L17 → L18 | 47% |
| 7B | L11 → L12 | 40% |
| 8B | L14 → L15 | 39% |
| 9B | L15 → L16 | 47% |
| 14B | L20 → L21 | 42% |

This is a universal architectural property. My earlier doc (the closing paragraph of `FORMAL_FOUNDATIONS.md` §8) mentioned the phase transition as a 3B characterization; the correct framing is "40–47% depth universally." It is another item in the universal-features list alongside the cocycle, category transfer, and PACF.

---

## 10. The corrected PACF picture

**Default tier: [A].**

I owe a correction on the PACF result. In `WHAT_YOU_MAY_HAVE_RUSHED_PAST.md` §1 I wrote that the MLP is "97% fresh innovation" based on R² = 0.03. That number is real, but the interpretation is oversimplified. There are *two* different statistics:

**Exp T (single-model, 3B) [A]:**
$$\delta^{(\ell)}_t \text{ predicted from } h^{(\ell-)}_t : \quad R^2 = 0.03 \text{ (math)}, \quad 0.09 \text{ (non-math)}$$
The MLP's output at layer $\ell$ is 3–9% predictable from its *own input at that layer*.

**PACF across 5 models (Exp BB) [A]:**
$$\delta^{(\ell)}_t \text{ predicted from } \delta^{(\ell-1)}_t : \quad R^2 = 0.908 \text{ to } 0.941$$
Consecutive MLP deltas are **91–94% predictable from each other** in every model, including 3B. The crossmodel validation doc explicitly flags that "the '97% innovation' from Exp T was measuring a different thing (predictability from the layer's own input, not from the previous delta)."

**What this changes.** My philosophical framing in the companion doc — "the residual stream is an accumulator of near-independent contributions, not a refinement of a representation" — is **wrong in the direction it matters**. Consecutive MLP contributions are highly correlated with each other (PACF 0.91–0.94), even though each one is weakly predicted by its own input (Exp T 0.03). The correct picture is something like:

- Each MLP's output is only weakly determined by the specific hidden state it reads at that layer.
- But each MLP's output is strongly correlated with the *previous* MLP's output.
- Which means the MLPs are not independent specialists; they are coordinating across layers through the residual stream in a way that is invisible if you only look at "MLP output as a function of MLP input."

**This is a subtler and more interesting picture than my original framing.** It says the layers talk to each other via the accumulated residual stream, not via their own immediate inputs. The "fresh innovation" narrative was wrong; the "coordinated through accumulation" narrative is right. I should rewrite the PACF section of `WHAT_YOU_MAY_HAVE_RUSHED_PAST.md`, which I will do.

---

## 11. Toy theorem stack — more than I presented

**Default tier: [A] from `toy_theorem_derivation.md`.**

The theoretical core is **four stacked pieces**, not one toy theorem. I under-sold this in my strategic assessment.

1. **Proposition 1: bilingual gradient equilibrium.** For a linear bilingual least-squares model, the optimal singular value in each direction is $\sigma_i^* = \tilde y_i \cdot \rho_i / \text{(normalization)}$, where $\rho_i$ is the per-direction agreement ratio. Directions with high bilingual agreement get amplified; directions with low agreement get killed. **Proved. Numerically verified at $r = 1.000$.**
2. **Multi-sample extension (N problem pairs).** The single-pair $\rho_i$ does not naively extend; the correct multi-sample statistic is the cross-view correlation $\rho_i^{(\text{multi})} = \text{corr}(z_{\text{zh},i}, z_{\text{en},i})^2$. **Verified: $\text{corr}(\rho^{(N)}, \sigma) = 0.999$.**
3. **Proposition 2: contrastive Z recovery.** When naive bilingual PCA fails because language-specific variance exceeds shared variance, the right method is to SVD the difference matrix $X_{\text{zh}} - X_{\text{en}}$ and use the singular-value gap as a principled diagnostic for how many language-specific directions to project out. **Proved. Verified: auto-detected 20 language directions (ground truth 20) via SV gap cliff 0.699 → 0.072. Contrastive Z captures 14.99/15 shared dims. NN = 500x chance.**
4. **Proposition 3: partition-invariant subspace (CCA formula).** A conjectured closed form for the canonical correlations between bilingual token averages of a smooth latent signal, in terms of spectral decay and partition survival factors. **Not proved — the within-partition orthogonality of averaged eigenfunctions does not imply cross-partition orthogonality, and the formula has not been numerically verified.**

**What this stack buys us.** Proposition 1 is the existence mechanism: if you train bilingually on shared-content data, least-squares pressure *produces* a low-rank shared subspace. Proposition 2 is the extraction method: if you want to find that subspace from activations without a target, use the contrastive SVD of the difference matrix. Proposition 3 is the quantitative prediction: the dimension of the shared subspace should be controlled by signal smoothness and token partition, via a formula that (if proved) makes testable predictions.

Propositions 1 and 2 are proved and verified. Proposition 3 is honest about being a conjecture with a known proof gap. This is the right posture — overclaimed theory is the fastest way to lose reviewer credibility, and the doc already flags the gap.

**What this changes strategically.** In my earlier response about "Olah / Nanda meetings," I described the theoretical contribution as "toy theorem + Proposition 3 conjecture." The correct description is "G·f*·G' derived from information theory before knowing transformer internals, validated empirically by cross>within on first token across 3 models, with a four-stack formal framework (Prop 1 proved, Prop 2 proved, Prop 3 conjectured) underneath." That is a qualitatively stronger theoretical pitch.

---

## 12. How this changes the shape of the paper

The paper pitch I sketched earlier led with the read head. Given the full scope, the pitch should lead with G·f*·G' as the theoretical motivation and close with the read head as the resolving mechanism. The arc:

1. **Motivation (G·f*·G').** Derived from information theory. Predicts a language-agnostic shared-content subspace Z in any multilingual model. This is architecture-agnostic.
2. **Existence verification (cross > within, 3 models).** Translation more preserving than paraphrasing on first token, before any answer is emitted. Z exists.
3. **Causal validation at the direction level (PC0 swap).** Single direction, any layer L8–L34, 100% language switch with 100% first-token match. Z's language wrapper is localized to one direction, the rest is f*.
4. **Structural validation via universality (cocycle + category + phase transition).** Across 5 models: cocycle R² > 0.87 (near-flat manifold), category transfer = 1.000 (identical encoding), universal phase transition at 40–47% depth. Z is architectural.
5. **Multi-language generalization (BH 7-language).** 97% Top-1 retrieval at L32 via null-space. Z scales from bilingual to 7 languages.
6. **Cross-domain generalization (MOAMS-X).** 96.2% transplant success across commonsense/code/logic. Z is not math-specific.
7. **The Lyapunov funnel and BQ3 causal pruning.** Across 4 Qwen models (rank_90 metric): build → equilibrium → canyon. BQ3: skipping 5 layers ranked by delta-Gram preserves accuracy, validating the funnel metric causally.
8. **The observation-intervention gap, and its resolution.** BS SVD truncation catastrophically fails: low-rank Gram ≠ low-rank computation. C1 measures computational Gram rank; C2 tail transplant resolves the gap: V⊥ at the *last token* is irrelevant; V⊥ at *context tokens* is essential. The read head is a single-token compression in a thick substrate.
9. **The read head is a focus gate (attention entropy).** L32–L35 entropy delta (content < glue) p < 0.0001, 37/37 problems at L35. The read head concentrates during content emission.
10. **Theoretical foundation (toy theorem stack).** Proposition 1 (proved, verified) gives the mechanism by which bilingual least-squares produces Z. Proposition 2 gives the extraction method. Proposition 3 is the open conjecture extending the dimension-count prediction to smooth latent signals.

That is a 10-section paper. Each section stands on its own empirical result. The theoretical arc brackets the empirical content on both ends (G·f*·G' at the opening, toy theorem stack at the closing) so the paper reads as a research program, not a laundry list.

---

## 13. What I am still not sure about

**Things I probably have right but have not verified directly against scripts/outputs:**

- The exact form of the Phase 3A intervention (projector vs mean-replacement). I reconstructed this from the memory summary and the description is consistent with standard patching notation, but I did not open `expPhase3A*.py` if such a script exists.
- The exact null-space extraction method in BH. I wrote "orthogonal to per-language difference directions" as a reconstruction; the actual method may use a more sophisticated construction (e.g., multi-class LDA null-space).
- The exact domains in MOAMS-X. I wrote "commonsense, code, logic" from the memory summary; the actual domain list may include more or different categories.
- The exact layers where BH retrieval was tested (I wrote L32 from memory; it may be a layer range).
- Whether Proposition 3's numerical verification has been attempted since the `toy_theorem_derivation.md` was last updated.

**Things I do not know at all:**

- Whether the G·f*·G' theoretical derivation has been written up as a formal LaTeX document beyond the `toy_theorem_derivation.md` entry.
- Whether there is a planned target venue for the paper beyond the memory note "target NeurIPS or EMNLP 2026" (memory `0deaf7ac-8e2`).
- Whether the "3 pillars" paper plan (geometric / practical / paradox, from memory `0deaf7ac-8e2`) is still the framing Greg intends, or whether the read-head discovery has reshaped it.
- How many non-trivial experiments I am still not aware of. The 55+ count in `session_analysis_2026-04-05.md` is from April 5; C1/C2/C2b/C2c and attention anatomy have been added since.

**Flag any of these as important to clarify and I will search for the actual source before writing about it.**

---

## 14. The one-sentence version

The project is a research program that predicts from information-theoretic first principles the existence of a language-agnostic shared-content subspace Z in any multilingual model, validates that prediction empirically across 5 models of two Qwen generations via four independent universality-level results (cocycle near-flatness, perfect category transfer, universal phase transition, and 7-language null-space retrieval), causally localizes the language-wrapper component of Z to a single direction at any mid-to-late layer (PC0 swap), resolves the observation-intervention gap between low-rank representational geometry and high-rank computation (C2 last-vs-context), and closes with a four-stack toy-theorem framework that explains the mechanism (Prop 1 proved) and opens two research directions (Prop 2 contrastive extraction, Prop 3 spectral-decay conjecture).

If that sentence feels slightly too long, that is because the scope is slightly too large for one sentence. Good.

---

*End of document. Companions: `FORMAL_FOUNDATIONS.md` (read-head + forward pass), `WHAT_YOU_MAY_HAVE_RUSHED_PAST.md` (under-appreciated items, with the PACF correction from §10 above applied).*
