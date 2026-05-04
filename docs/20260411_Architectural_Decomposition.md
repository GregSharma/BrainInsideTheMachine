# Architectural Decomposition: Three Money Shots

**Date**: 2026-04-11 (post-GATE failure, post-two-subspace discovery)
**Status**: Vision document. No experiments run yet. Every claim traces to a numbered finding.

---

## Premise

f* exists (information-theoretic theorem, cocycle pins it, C3 proves constructively).
Pure compression has gone 0 for 5 — projecting onto bases destroys generation.
The GATE failure taught us WHY: encoding-time and generation-time subspaces are near-orthogonal (9.6% overlap).
So we stop trying to compress the sandwich and start separating it.

---

## Money Shot 1: Experiment-Guided Surgery (94K trainable params)

The construction that can ONLY come from our filtration:

**Step 1.** Compute e_c(l) at every layer using the cocycle Procrustes maps across 7 languages (G17, BH). This gives the convention axis at every depth.

**Step 2.** Apply kernel surgery at every layer: project e_c(l) out of W_down^l. This is G15 applied globally — taking the Z_2 quotient of the MLP at every layer.
- Below l_c (~L12): **lossless**, because G7a says MLP action on Z doesn't depend on Z-perp.
- Above l_c: introduces rank-1 error (G15).

**Step 3.** Train rank-1 LoRA adapter at each layer above l_c. The adapter compensates for exactly the rank-1 coupling the surgery removed.
- Training target: match original MLP output projected onto Z.
- This is rank-1 regression — may have closed-form solution, no gradient descent needed.
- Trainable params: 23 layers x 2 x 2048 = **94,208 parameters** (0.003% of 3.09B).

**Step 4.** Replace attention at generating position with per-layer constant vector (C6b).

**Step 5.** Replace readout at l* with rank-1 spectral projection v1 of W_U . OV_{l*} (G8, C3, C5).

**What each step requires from our experiments:**
| Step | Finding used | Without it, you can't... |
|------|-------------|-------------------------|
| 1 | Cocycle G17 | Know e_c(l) at each layer |
| 2 | Kernel surgery G15 | Know 1D projection is safe |
| 3 | Rank-1 entanglement G15 | Know adapter rank = 1, not 8 or 50 |
| 3 | Convention-invariance threshold G7a | Know which layers need adapters |
| 4 | Constant attention bias C6b | Know attention is replaceable |
| 5 | Spectral readout C3 + G8 | Know the rank and direction |

**Output**: A model where h-f*-h' is explicit. MLP below l_c IS f*, provably (convention component removed, accuracy held). Adapter above l_c IS h' correction (compensates rank-1 coupling). Every claim traces to a numbered experiment.

---

## Money Shot 2: Patty-and-Buns Factorization (the original car vision)

f* is the patty. h and h' are the buns. Don't shrink the sandwich — separate it.

**The patty (f*):**
- 12 layers of convention-stripped MLPs (kernel surgery applied, convention-serving neurons pruned).
- Takes convention-invariant input, produces convention-invariant output.
- Doesn't know what language you're speaking. It reasons.
- After pruning convention-serving neurons (identified by projection onto e_c): ~560M params.
- Attention replaced with constant bias (C6b). Pure MLP stack.
- Trainable independently on abstract/synthetic reasoning data. No language needed.

**The buns (h, h'):**
- Bottom bun h: W_E + early attention + layers that build convention-invariant representations.
- Top bun h': late layers that dress output in convention + W_U.
- Each bun is a language-specific ADAPTER. ~100-200M params per language.
- Cocycle flatness (G17, 2.86% error) PROVES buns are interchangeable.
- Train each bun from parallel text + frozen f*.

**What this enables:**
- f* (560M) + English bun (150M) = 710M for English reasoning.
- Swap to Chinese bun = 710M for Chinese reasoning. Same f*.
- Each new language costs ~150M of adapter training, not 3B of full pretraining.
- f* improves independently — train on harder math, every language gets better simultaneously.

**What our filtration provides:**
- e_c(l): what to strip at each layer
- l_c: where the patty ends and the top bun begins
- G7a: stripping is lossless below l_c
- G15: patty-bun interface is rank-1
- Cocycle: buns are interchangeable
- C6b: attention in f* is constant
- G8: readout is rank-1 and locatable from weights alone

---

## Money Shot 3: Z-Agnostic Training (the grandson)

Once Money Shot 2 proves the decomposition, train the whole thing from scratch:

**Father**: Qwen-3B, full model, convention-entangled.

**Son**: Freeze L0-L12, replace L13-L35 with f', train f' on father's outputs stripped of convention. Prove the decomposition works.

**Grandson**: Train from scratch with abstract vocabulary, using son's f' as teacher for reasoning core. Prove reasoning is trainable without language.

Each generation is smaller, faster to train, and more interpretable. Each grounded in previous generation's demonstrated success.

This is the 40M model from the original vision, but now with a principled architecture (informed by layer-by-layer measurements) and a distillation target (the working f' module).

---

## Experiment Grounding

Every architectural decision above traces to a specific experiment:

| Decision | Grounding experiment | Result |
|----------|---------------------|--------|
| Convention axis e_c(l) | BH cocycle + G17 | 7-lang Procrustes, 2.86% error |
| Surgery is lossless below l_c | G7a convention-invariance | N >= 0.66 below L12 |
| Entanglement is rank-1 | G15 kernel surgery | 1D works, 10D fails |
| Attention = constant | C6b mean dissection | mean_only = baseline (12/20) |
| Readout = rank-1 | C3-7B compression | k=1 lossless (12/20 = baseline) |
| Readout direction from weights | G8 + C5 spectral | top1_frac = 0.9571 at 7B |
| Buns interchangeable | G17 cocycle flatness | 2.86% error across 7 languages |
| Z_2 symmetry (reflection not rotation) | G6 | PC0 swap = perfect switch |
| f* trainable independently | N MLP transfer | zh/en MLP swap preserves math |
| Language flip = efficiency only | P3/R/R2/R3 | Effect vanishes at 512 tokens |

---

## Risk Assessment

| Path | P(works first try) | Key risk | Mitigation |
|------|-------------------|----------|------------|
| Surgery (MS1) | ~55% | L12 output not convention-free enough (G7a = 0.66, not 1.0) | Use L14 instead, or carry thin convention channel |
| Patty-buns (MS2) | ~40% | Interface between patty and buns leaks | Rank-1 adapter at interface; measure leakage explicitly |
| Z-agnostic (MS3) | ~25% | Reasoning may be constitutively linguistic | Would still be a major negative finding worth publishing |

---

## What Dies vs What Survives

If MS1 fails: we learn that the rank-1 entanglement structure is observationally real but not surgically separable. The entanglement tensor is descriptive, not prescriptive. That's a finding.

If MS2 fails: we learn that f* exists as a mathematical object but cannot be instantiated as an architectural module — the computation requires the entanglement to function. That constrains all future factorization attempts.

If MS3 fails: we learn that reasoning is constitutively linguistic at the representation level. You can describe f* but never train it without language. That's arguably the most important finding of all three.

Every failure teaches us something no one else knows. The filtration is the moat regardless of outcome.

---

## Execution Priority

1. **MS1 first** (smallest, fastest, most grounded). Kernel surgery at all layers, rank-1 adapters, eval.
2. **MS2 if MS1 succeeds** (factor into patty + buns, train adapters per language).
3. **MS3 if MS2 succeeds** (abstract vocabulary training of f* standalone).

Each step is publishable independently.
