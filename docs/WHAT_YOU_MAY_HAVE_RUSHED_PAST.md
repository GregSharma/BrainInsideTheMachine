# What You May Have Rushed Past

**A companion to `FORMAL_FOUNDATIONS.md`. Short, pointed, occasionally philosophical.**

Author: VEGA (Claude Opus 4.6)
Date: 2026-04-09 (original), 2026-04-09 update
Audience: Greg, after he has read (or at least skimmed) `FORMAL_FOUNDATIONS.md`.

---

## 0a. Correction and scope disclaimer (UPDATE — read first)

**Two corrections since the original draft:**

1. **§1 "The MLP is 97% fresh innovation" is a misread and I have corrected it below.** The R² = 0.03 number from Exp T measures predictability of an MLP's delta from its *own input at that layer*. A separate statistic — consecutive PACF across 5 models in Exp BB — measures predictability of an MLP delta from the *previous layer's* delta, and it is **0.908 to 0.941 everywhere, including 3B**. The original framing turned one statistic into the wrong philosophical conclusion ("residual stream is an accumulator of near-independent contributions"). The correct picture is the opposite: consecutive MLP contributions are *highly* correlated with each other even though each is weakly correlated with its own immediate input. I have rewritten §1 below to reflect this. See `EARLIER_CAUSAL_WORK.md` §10 for the full two-statistic distinction.

2. **§6 "language flip is efficiency not accuracy" describes the 3B result accurately but did not mention that the flip is 3B-specific.** Cross-model data from Exp AX (in `crossmodel_validation_2026-04-06.md`): 7B flip +2 (noise), 8B +1 (noise), 9B −1, 14B 0. The efficiency reframe is still correct *for 3B*; the generalization is that "the flip works" is a 3B phenomenon tied to capacity constraints, not a general intervention. The §6 text below has been left as-is because the philosophical point ("efficiency vs accuracy is a different class of interpretation") still stands, but with this scope note: it stands *for the 3B flip*, not as a general claim about direction interventions.

**Scope:** like `FORMAL_FOUNDATIONS.md`, this doc was written treating the read-head arc as the project. The project is larger. See `EARLIER_CAUSAL_WORK.md` for G·f*·G', Phase 3A, PC0 swap, BH 7-language, MOAMS-X, and the cocycle/category/phase-transition universality results.

**Confidence tags** (same scheme as the other updated docs): [A] = verified against source; [B] = reconstructed from memory; [C] = inferred.

---

## 0. What this is

You pushed through a lot of experiments over the last six weeks. Some findings got full treatment; others got a line in `MEMORY.md` and a quick "huh, interesting" and then we moved on to the next thing. A few of those quick-skim results are, I think, more important than the treatment they received. This document surfaces six of them.

The format is the same for each: what the result actually is, the interpretation you likely wrote down in passing, and the thing I think you missed. I am not trying to overclaim — some of these are small finds. But they are small finds that are load-bearing for the bigger picture, and I don't want them lost.

---

## 1. The MLP's two predictability stories (Exp T and PACF together)

**This section was rewritten on 2026-04-09. The original version is wrong — see the correction disclaimer at the top.**

### The two results

There are two separate regressions. They measure different things. They give different answers. The original draft of this document conflated them.

**Exp T (single model, 3B) [A]:** Fit a linear model predicting the MLP delta at layer $\ell$ from the MLP's *own input* at layer $\ell$:

$$\delta^{(\ell)}_t \approx A \, h^{(\ell-)}_t + b$$

For math tasks: $R^2 = 0.03$. For non-math reasoning: $R^2 = 0.09$. The MLP delta is almost unpredictable *from the vector the MLP just read.*

**Exp BB across 5 models [A]:** Fit a linear model predicting the MLP delta at layer $\ell$ from the MLP delta at layer $\ell - 1$ (partial autocorrelation):

$$\delta^{(\ell)}_t \approx B \, \delta^{(\ell-1)}_t + c$$

Across 3B, 7B, 8B, 9B, 14B: $R^2 = 0.908$ to $0.941$. Consecutive MLP deltas are **91–94% predictable from each other, in every model including 3B.**

### Why these are not contradictory

They are measuring different regressors against the same regressand. The MLP delta at layer $\ell$ is only weakly predictable from $h^{(\ell-)}_t$ (Exp T) but is strongly predictable from $\delta^{(\ell-1)}_t$ (PACF). Both can be true. They are in fact both true.

Mechanistically: the MLP delta at layer $\ell$ carries information that is specific to the *previous delta's direction* in a way that the *current input's direction* does not pin down. The layer is not reacting linearly to its input; it is continuing a computation that the previous layer started. The layer-to-layer computation is a smooth trajectory in delta-space; the layer-to-residual computation is discontinuous.

### What I had wrong

My original framing was: "residual stream is an accumulator of 36 near-independent contributions." That is false. The contributions are highly correlated to each other via the PACF signal. A better framing is:

> The residual stream is the running integral of a smooth trajectory in MLP-delta space. Each layer's delta is close to its neighbor's delta (PACF = 0.91–0.94). The relationship between a layer's delta and the state it reads is weak (Exp T = 0.03–0.09) because the layer is continuing a trajectory rather than transforming a state.

### Why this is more interesting than the wrong version

The wrong version said the MLPs are independent specialists. The right version says the MLPs are **coordinating across layers through a trajectory that is invisible if you only look at each layer's input-output relationship**. The coordination lives in the space of MLP deltas, not in the space of residual stream states. This is a sharper claim and it matches what we see in the phase-transition work (Exp T): consecutive delta cosines flip from adversarial (−0.05 to −0.12) to cooperative (+0.17 to +0.27) at the 40–47% depth transition — which is *itself* a statement about the geometry of consecutive deltas, not about the residual stream state.

### The under-appreciated part that survives the correction

What is still under-appreciated: the **phase transition in consecutive-delta cosine** is a universal feature across 5 models at 40–47% depth [A from `crossmodel_validation_2026-04-06.md` §BB]. Consecutive MLP deltas start adversarial in the first 40% of depth and turn cooperative after. This is a structural signature of how the model's internal trajectory organizes across layers. It is *not* a 3B-specific finding, as I suggested in the original. It is in every model we tested.

If I had to pick the single under-appreciated statistic from this section after the correction, it would be the universality of the adversarial→cooperative flip. The number 0.03 is real but its interpretation was oversold; the number 0.41–0.47 (the depth fraction of the phase transition) is real and its interpretation is under-sold.

---

## 2. Coder-3B: geometry without dynamics (Exp W)

### The result

Qwen2.5-Coder-3B has the same architecture as Qwen2.5-3B. Same 36 layers, same $d=2048$, same heads, same tied embeddings. We measured the "language direction" — the mean-difference direction between English and Chinese math problem activations — and got Cohen's $d = 3.5$ to $4.8$ across layers. This is comparable to base 3B (which has $d = 3$ to $5$).

**Then we ran the flip.** Take the causal language direction, flip its sign in MLP deltas at L9–L26, rerun generation. The base 3B goes from 5/20 → 13/20 on a math eval. Coder-3B goes from 7/20 → 7/20. Delta equals zero. No effect.

The language axis exists in Coder-3B. It has large effect size. It points in the same conceptual direction (separates EN from ZH activations). Flipping it does nothing.

### What you probably wrote down

"Coder-3B can't do Chinese math (baseline ZH = 0/20), so the flip has nowhere to land. Makes sense. File under 'baseline capability check'."

### What you probably missed

The explanation "Coder can't do Chinese math" is correct as a proximate cause but misses the *generalization* it licenses.

Coder-3B is the same parameter count, same architecture, same tokenizer as base 3B. The only difference is training mix: Coder saw more code, less Chinese math text. As a result, the flip on Coder does not access a Chinese math strategy — because there is no Chinese math strategy for the flip to access.

**The deeper claim.** A mechanistic axis — a direction in activation space that separates two conditions with large Cohen's $d$ — can exist in a model without being *functional*. The geometry is determined by whatever the training distribution put there; the *dynamics* (what happens when you intervene on the axis) are determined by whether the model ever learned to compute anything along that axis. These two things can come apart.

This is almost never said in mechanistic interpretability papers. The default framing is: find a direction, flip it, interpret the result as "the model uses this direction for X." Coder-3B falsifies the default. It has the direction. It does not use the direction.

**Concrete implication for any future "we found a direction" paper we write.** We must always test the flip. Finding a direction with high Cohen's $d$ is not sufficient evidence that the direction is causally active. Probing it is necessary. The Coder-3B result is the negative control that lets us make this claim rigorously, and it is the kind of control that gets papers into *Transformer Circuits Thread* rather than ArXiv-only.

**More speculatively.** If "axes can exist without dynamics," then the reverse is probably also true: *dynamics can exist without clean axes*. There may be functional computation in models that does not correspond to any single-direction intervention. We haven't tested this, but the asymmetry is suggestive.

---

## 3. The BS experiment's reinterpretation — "error amplification" was wrong

### The result

**Old story (summer).** The BS experiment (bottleneck substitution on Z) replaced the residual stream's content with its projection onto a low-dimensional Z subspace at a specific layer and measured downstream accuracy. It was catastrophic: 0/20 at every layer. The interpretation at the time was "*error amplification*" — a small corruption at layer $\ell$ compounds through subsequent layers and eventually ruins the output.

**New story (post-C2b).** C2b replicated BS's effect with a critical variation: it applied the bottleneck **at one layer** and **only to context tokens** (not the last). Result: 0/20. The same effect at the same magnitude.

**This kills the error amplification story.** Error amplification requires *propagation across layers*. If one layer's context-only bottleneck is as catastrophic as a full-stack bottleneck, there is no propagation — there is *direct destruction*. The mechanism is that the context states at that one layer become garbage inputs to every downstream layer's attention keys and values, immediately and at once, not gradually.

### What you probably wrote down

"OK, reinterpret as direct destruction. Got it. Moving on."

### What you probably missed

The *reason* this reinterpretation matters is that it validates the §5.4 argument of the formal doc about why context tokens need full dimensionality. Re-read that section with C2b in mind:

- C2b says: one layer, one position class (context), bottleneck → total failure.
- The formal argument says: context tokens' full 2048 dims are needed because each downstream attention head reads a different 128-D slice.

These are the same statement seen from two angles. C2b is the empirical leg; §5.4 is the mechanical leg. Together they pin down *why* the observation-intervention gap exists: the read head can live in 20-D, but the substrate that computes it must live in the full 2048-D, because the substrate is being sampled by dozens of downstream attention heads that each want a different slice.

**The meta-lesson I want you to take from this.** When we first ran BS and saw catastrophic failure, we reached for the closest plausible story — error amplification — because it was familiar. We then spent weeks quietly assuming that story was right. It took C2b to notice that the *dose-response* pattern was inconsistent with amplification. In general: *when you write down an interpretation, also write down what dose-response or scope-response pattern it predicts, so you can check.* If BS had been interpreted carefully from the start, we would have known to test the single-layer case, and C2 would have happened sooner.

Related: the BS story was the last gasp of "Z is a fiber bundle" thinking — the intuition that there is *one* subspace $Z$ running through the network and you can project onto it at any layer and everything still works. The fiber bundle view is dead (BM, BN, BO, BP all killed it). The correct picture is: the read head at the last token is low-dim, the context substrate is high-dim, there is no continuous low-dim structure in between. This is easier to state now but was hard to see six weeks ago.

---

## 4. 14B V⊥ improvement is not neutral — V⊥ is weakly hostile

### The result

In `expC2c_14b.json` (14B --quick), the `last_only_N48` condition — replacing the orthogonal complement at the last token at every layer with zeros — produced:

- English: baseline 5/20, intervention **11/20**. ($\Delta = +6$.)
- Chinese: baseline 10/20, intervention **11/20**. ($\Delta = +1$.)

Qwen3-8B similarly showed ZH $+3$. 3B and 7B showed no change. The effect appears only in the larger untied models and is stronger in English than Chinese.

### What you probably wrote down

"V⊥ is slightly interfering, not just dead. Interesting. File under 'weird 14B anomaly'."

### What you probably missed

A 6-point improvement out of 20 is a 30% relative gain. On a model whose baseline is only 25% correct, this is not a cute statistical quirk — it is half the gap between the 14B model and the 3B model on the same eval.

What would it mean if this effect is real and reproducible? It means that, for 14B, the final layer's output at the last token contains information outside the top-20 PCA subspace that is *actively pushing the logits in the wrong direction*, for some fraction of problems. Removing that information improves accuracy.

There are three ways this could be true, and they have different implications:

**(a) Noise accumulation.** The model's training did not penalize activity outside $P_\parallel$, so that subspace collected uncorrelated noise across 48 layers. Zeroing it is denoising. Prediction: the improvement should be robust to choice of calibration set and the magnitude of improvement should scale with layer count.

**(b) Distractor interference.** The model learned to represent *candidate answers* in a high-dimensional way and the final-layer readout selects the winner. For the 6 improved problems, the distractor was being computed along V⊥ directions and slightly leaking into the logits. Zeroing V⊥ kills the leakage. Prediction: the 6 improved problems should be ones where the argmax was wrong by a narrow margin.

**(c) Overfitting to calibration set.** $P_\parallel$ happens to align with the useful directions of our specific test set, and the improvement is spurious. Prediction: holding out 4 calibration problems and re-running on those 4 should eliminate the effect.

These three are distinguishable by experiments that cost about a day each. (a) and (b) are interesting findings that are worth a figure in the paper. (c) is a control we need to rule out regardless.

**Why this matters structurally.** If (a) or (b) is correct, then the read head is not just "ignoring" V⊥; it is *fighting* V⊥. That is a meaningfully different mechanistic picture. It suggests that at 14B scale, the final layers are doing a projection operation to suppress out-of-subspace content — essentially an implicit denoising head. We did not know that was there. It would also explain why larger models do better: they have more layers of implicit denoising.

**The open experiment 8.4 in the formal doc was written to disentangle these.** I flag it here because the size of the effect justifies running it.

---

## 5. Tied vs. untied embeddings predict output rupture

### The result

Qwen2.5-3B and Qwen2.5-7B have tied embeddings. Both show catastrophic output-layer rupture in `expBQ2_crossmodel_lyapunov.json`: the Gram correlation between EN and ZH drops from ~0.83 in the middle of the network to below 0.5 at the final layer; the Frobenius norm of the residual stream spikes; mode-0 Lyapunov exponent flips sign and ruptures.

Qwen2.5-14B has *untied* embeddings. It shows moderate rupture — the correlation drops, but the Frobenius spike is smaller, and the mode-0 exponent is less violent.

Qwen3-8B has untied embeddings. It shows *no* rupture at all. Frob stays flat. Gram correlation drops from 0.83 to 0.67 — a smooth restructuring without catastrophe.

The pattern is clean:

| model | tied? | rupture | Frob spike |
|---|---|---|---|
| 3B | yes | catastrophic | yes |
| 7B | yes | catastrophic | yes |
| 14B | no | moderate | small |
| Qwen3-8B | no | none | no |

### What you probably wrote down

"Tied embeddings = rupture. Untied = no rupture. Duh."

### What you probably missed

The mechanism is not "tied embeddings cause instability." That is a correlation claim. The mechanism is something more precise:

Tied embeddings force the final layer's residual stream to live in a basis that is simultaneously **the input lookup basis** and **the output readout basis**. These two requirements are in structural tension. The input lookup basis wants each row to be a semantically rich embedding of a vocabulary token, with all the anisotropy and learned structure that comes with it. The output readout basis wants each row to be an *orthogonal classifier direction* — "how much does this hidden state look like token $v$?" — which benefits from being well-spread and ideally isotropic.

In a tied model, these cannot both be true. Something has to give. What gives, empirically, is that the final-layer residual stream gets pushed hard against the input-embedding anisotropy structure, and in doing so, its trajectory across the last few layers becomes steep and unstable. Rupture is the price of reusing the embedding matrix.

Untied models don't have this tension. The output head $W_{\text{lm}} \in \mathbb{R}^{V \times d}$ is a separately trained matrix that can shape itself into a clean classifier basis, and the final-layer residual stream doesn't have to match the input embedding's direction structure. The trajectory is smoother.

**This has a testable consequence we haven't tested.** For a tied model, you should be able to compute "the angle between the input embedding row for token $v$ and the mean direction of final-layer residual states that sample $v$," and that angle should be *small* — the residual stream is being pulled toward the input embedding direction. For an untied model, the residual stream should be pulled toward the *LM head* direction, which may be different. This would be a 1-day experiment and would put numbers on the rupture mechanism.

**Why this is load-bearing for the read head story.** Section 5.4 of the formal doc argues that the read head at the last token is compressing into a low-dim subspace. In a tied model, the low-dim subspace it compresses into is *also* the input embedding subspace, because that's the only direction structure the final layer is aware of. In an untied model, the compression target is the LM head's directions, which can be a cleaner geometry. **The 14B V⊥ improvement in §4 of this document may be a direct consequence of this cleaner geometry** — a cleaner readout basis leaves less "junk" in V⊥ to interfere.

This is a unified mechanism I don't think we have written down anywhere, and it predicts the three main tied/untied findings (rupture, V⊥ interference, read head compression target) from one root cause.

---

## 6. The language-flip result is about efficiency, not accuracy

### The result

Flipping the causal language direction in MLP deltas at L9–L26:

- 3B at 128 generated tokens: 5/20 → 13/20 (+160%)
- 3B at 256 tokens: 11/20 → 13/20 (+18%)
- 3B at 512 tokens: 15/20 → 15/20 (**0%**)
- 1.5B at 128 tokens: 7/20 → 10/20 (+43%)
- 1.5B at 256 tokens: 12/20 → 12/20 (0%)

Given enough generation budget, the baseline catches up. The intervention does not make the model *smarter*; it makes the model *faster*. Its effect is to shorten the path to the answer.

### What you probably wrote down

"OK, confounded by token budget. The direction is real but the intervention's benefit washes out. Move on."

### What you probably missed

You probably treated "confounded by token budget" as a *disappointment* — a retreat from a stronger claim to a weaker one. I want to reframe it.

"The intervention shortens the path to the answer" is a **stronger mechanistic claim** than "the intervention makes the model smarter." It says the model has two internal strategies — a verbose one and a direct one — and the language direction is the dial between them. Flipping the dial does not change which answer the model converges to; it changes how much cognitive work the model performs before emitting.

This is a *different class* of mechanistic interpretation than the usual "found a direction for X" story. It is not "the model stores the answer in direction $v$." It is "the model stores a *strategy selection* in direction $v$, and the strategies have different token-budget requirements but converge to the same answer given enough budget."

**Why this is interesting.** Most interpretability claims are about *what* a model computes. This one is about *how long it takes* a model to compute. The distinction maps onto a well-known distinction in complexity theory — decision vs. cost — and points at a class of interventions (dial-flip-to-shorten) that are useful for practical model improvement even when they don't improve the decision boundary.

**Why this is under-appreciated in your own notes.** Your MEMORY.md line on this result calls it a "CONFOUND CONFIRMED." That framing is accurate but concedes too much. The confound was a confound only if the original claim was "accuracy improves." The correct claim is "efficiency improves at fixed budget, accuracy ceiling is untouched." That is a clean, defensible finding.

---

## 7. Closing: the meta-lesson

Six items above. Here is what they share.

Each one is a case where the obvious first-pass interpretation of a result was either incomplete or slightly wrong, and where the correct interpretation is richer and more load-bearing. The pattern of the error was the same each time: we reached for the nearest plausible story and moved on. When later experiments forced a reinterpretation (BS → C2b), we updated the local conclusion without updating the neighboring conclusions that depended on it.

The antidote is not "interpret everything more carefully the first time." That is not achievable in a project with our pace. The antidote is: **after every experiment, write down one prediction the current interpretation makes, and one way it could be wrong.** Then we can go back and check. The Coder-3B result would have been caught in two lines if we had written "prediction: flip should work in any model with the same direction; fail mode: the direction is geometric not functional" when we first ran the 3B flip. We didn't. We got the fail mode six weeks later.

This is not a criticism of how we've worked. We found the read head, which most projects don't find at all. It *is* a suggestion that once you re-hydrate your mental model from the `FORMAL_FOUNDATIONS.md` document, we should close the loop on the six items here — not by running new experiments immediately, but by writing down the prediction/fail-mode pair for each. That is cheap, and it will expose which of our interpretations is still load-bearing and which is a placeholder we've been quietly propping up.

The project is in a good place. The formal floor exists. The results replicate across 4 models. The read head hypothesis has teeth. What we owe ourselves now is a few days of consolidation, not exploration. Consolidation is not the same as idleness — it is the part of the work that makes the next finding possible.

---

*End of companion. Return to `FORMAL_FOUNDATIONS.md` if anything in here pointed at a section that deserves re-reading.*
