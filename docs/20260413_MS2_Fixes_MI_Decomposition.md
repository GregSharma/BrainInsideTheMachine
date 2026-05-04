# Handoff: MS2 Surgery Fixes + Mutual Information Decomposition

**From:** Claude (web session, April 13 2026)
**To:** Vega (Claude Code)
**Context:** Greg's 7-language centroid SVD surgery experiment (expMS2) produced EN surgery = 19/20 (vs 13/20 baseline) but ZH surgery = 10/20 (vs 17/20 baseline). The EN improvement is partially confounded. Two fixes needed before interpreting results. Separately, a new analytical framework is proposed for future work.

---

## Part 1: Immediate Fixes for MS2

### Fix 1: Centroid Bias Audit

**The problem.** The centroid SVD computes the mean activation across all 7 languages per problem, then SVDs the deviations from that mean. The top singular vectors of the deviation matrix define the "convention subspace" that gets projected out during surgery (L13–L35).

If the centroid is closer to English than to other languages — which is likely given Qwen2.5-3B's training distribution — then projecting out deviations-from-centroid is equivalent to "push everything toward English." English loses the least signal because it's already near the centroid. Chinese, Arabic, Swahili lose the most because they're farthest.

**Evidence from the surgery outputs.** ZH surgery outputs show English fragments leaking into mid-Chinese text:
- Problem 1: `"3x + What is the value of x?"`
- Problem 5: `"347 + What is the sum of 347 and 658?"`
- Problem 8: `"1728 ÷ What is the quotient?"`
- Problem 10: `"c² = 5² + What is the value of c?"`
- Problem 16: `"确定小于2 What is the sum of all prime numbers less than 20?"`

The surgery strips the Chinese convention signal, and the model defaults to English. EN surgery doesn't show this problem because defaulting to English *is* English.

**The audit.** For the 200-problem × 7-language cache at each layer:
1. Compute the centroid (mean across 7 languages) per problem
2. Compute `cos(mean_lang_activation, centroid)` for each language
3. Report the per-language distance from centroid

**Expected finding:** English is closest to the centroid. If confirmed, the EN 19/20 result is confounded — it's partially "English-default surgery" not "language-agnostic surgery."

**The fix (if confirmed).** Two options:
- **(a) Leave-one-out centroid:** For each language being evaluated, compute the centroid from the OTHER 6 languages only. Surgery on English uses a centroid from {zh, ar, es, ja, ko, sw}. This removes the self-reinforcing bias.
- **(b) Per-language-pair surgery:** Instead of centroid SVD, use the bilateral mean-difference direction between the target language and each other language. Project out the mean of all 6 bilateral difference vectors. More conservative, controls bias better.

### Fix 2: Grading

**The problem.** Problem 14 across ALL languages asks "Is 97 prime? Answer yes or no." The model correctly identifies 97 as prime in every single language, every condition. But it never starts with the literal word "yes" / "是" / "نعم" / "sí" / "はい" — it says "97 is a prime number" and then explains. The grader marks it wrong everywhere.

**The fix.** The answer-matching regex needs to accept the equivalent of "X is prime" or "X is a prime number" as matching `answer=yes`. This is at least 1 free point per language per condition (baseline and surgery). Likely also affects problem 1 EN baseline (marked XX despite correct output "x = 5") and similar cases.

**Impact.** The real baseline and surgery scores are both higher than reported. The *delta* (surgery minus baseline) is what matters for the surgery story, and the delta is probably less affected, but it needs to be recomputed with corrected grading.

Also check: some problems marked [XX] in baseline have the correct answer in the output text (EN problem 1: output says "x = 5" but marked XX; EN problem 6: output says "613" but marked XX). The grader may be checking for exact format rather than answer presence.

---

## Part 2: Baseline Analysis (No Fixes Needed, Already Informative)

From the 7-language baselines (before surgery), patterns visible without any surgery:

| Language | Score | Notes |
|----------|-------|-------|
| EN | 13/20 | Verbose, repetition loops, arithmetic errors |
| ZH | 17/20 | Best performer, concise |
| AR | 13/20 | Some loops, some degenerate outputs |
| ES | 16/20 | Strong, clean formatting |
| JA | ~10-12/20 | Many repetition loops, arithmetic failures |
| KO | TBD | |
| SW | TBD | |

Key observations:
- **Algebra is "patty" (language-agnostic).** Problem 2 (2x^2-8=0) correct across EN, ZH, AR, ES, JA. Same procedure, same answer, different tokens.
- **Arithmetic is at the patty-bun interface.** Problem 5 (347+658): EN gets 905 (wrong), ZH gets 1005 (correct), ES gets 1005 (correct). Carrying algorithm fails differently per language.
- **Output control is pure "bun."** Repetition loops (emoji-delimited fake conversations) appear in JA, AR, ZH but rarely in EN, ES. The model's stopping behavior is language-specific.

---

## Part 3: Mutual Information Decomposition (Future Work)

### Motivation

All current cross-lingual alignment metrics (NN accuracy, CKA, R^2, Procrustes) are coordinate-dependent. They change depending on the basis, kernel, or regression model chosen. This has been a recurring source of confusion (Procrustes failure, NN accuracy sensitive to rotation, etc.).

Mutual information is **coordinate-free**. I(X_zh; X_en) measures the shared information between Chinese and English activations regardless of any rotation, scaling, or nonlinear transformation. It doesn't care about the "convention" — the language-specific coordinate system that's been causing grief since the Procrustes failure.

### The Decomposition

At each layer l, the residual stream update is:

```
h^(l) = h^(l-1) + attn_out^(l) + mlp_out^(l)
```

For each component, across matched zh/en problem pairs, compute cross-lingual MI:

**Curve 1: I(attn_zh^(l); attn_en^(l) | problem)** — How much shared information does attention carry?
- Prior finding ("attention is language-blind," C6b "attention is constant bias") predicts this should be HIGH at every layer.

**Curve 2: I(mlp_zh^(l); mlp_en^(l) | problem)** — How much shared information does MLP carry?
- Prior MLP delta finding predicts: HIGH at L5-L8 (NN=0.66, shared computation), CRASH at L13-L15, LOW through L35.

**Curve 3: I(residual_zh^(l); residual_en^(l) | problem)** — Total shared information.
- This is the envelope. Its derivative d/dl tells you whether each layer builds or destroys cross-lingual alignment.

**Curve 4 (the new thing): Interaction term.**

```
Interaction(l) = I(residual) - I(attn) - I(mlp)
```

This is NOT zero in general. It captures the synergy or redundancy between attention and MLP at each layer:
- **Positive** = synergy. The components do more together than separately. Possibly: MLP reads attention's cross-lingual alignment and amplifies it.
- **Negative** = redundancy. The components duplicate information. Possibly: MLP re-encodes what attention already placed.

Every sign change in this curve is a hypothesis about how the two components interact at that layer.

### Practical Computation

For approximately Gaussian activations (which residual streams tend to be after normalization), MI has a closed-form expression:

```
I(X_zh; X_en) = -1/2 log det(I - CCA_matrix^2)
```

where CCA_matrix is the canonical correlation matrix between the two sets of activations. This is computable from cross-covariance matrices — the same objects already being built for PCA in the existing caches.

**Steps:**
1. For each layer, collect attn_out, mlp_out, and residual for all 200 problems x {zh, en}
2. Compute cross-covariance matrices between zh and en versions of each component
3. SVD each cross-covariance to get canonical correlations
4. Apply the log-det formula
5. Plot all three MI curves + interaction term across depth

This is ~50 lines on top of existing cached activations.

### What This Buys

1. **A single number per layer per component** that's invariant to the basis problem. No more "did I pick the right PCs."

2. **Information bottleneck interpretation.** The layer where I(residual_zh; residual_en) peaks relative to I(residual; input) IS the information bottleneck layer. Should be L26. If confirmed, the model implements the IB principle as architecture.

3. **Layer-by-layer hypotheses.** Every peak, trough, and sign change in the interaction curve is a testable prediction about how attention and MLP cooperate or compete at that specific depth.

4. **Bridge to the formalism.** The Z_repr vs Z_comp distinction (encoding-time vs generation-time subspaces are 9.6% aligned) can be restated as: I(Z_repr; Z_comp) is low. The formalism conflated two objects with low mutual information. The MI framework prevents this mistake by measuring the actual shared information rather than assuming subspace overlap.

### Connection to Existing Findings

| Prior Finding | MI Restatement |
|---|---|
| "Attention is language-blind" | I(attn_zh; attn_en \| problem) ~ maximal at all layers |
| "MLP reads language, writes language into all dims" | I(mlp_zh; mlp_en \| problem) drops sharply at L13 |
| "Attention builds, MLP erodes" | d/dl I(residual) has positive contributions from attn, negative from MLP |
| "PC0 swap works" | The language-specific MI is concentrated in ~1 direction |
| "Encoding != generation subspaces" | I(Z_repr; Z_comp) << I(Z_repr; Z_repr) |
| Cocycle composition holds | Conditional MI I(X_zh; X_sw \| X_en) ~ 0 (en mediates fully) |

### Priority

This is not urgent — Fix 1 and Fix 2 come first. But the MI framework should inform how we interpret the corrected surgery results and may replace the NN-accuracy / CKA metrics in the final paper figures. The advantage is that it tells a clean, coordinate-free story that doesn't require explaining why Procrustes failed or why NN accuracy depends on the basis.

---

## Summary of Action Items

| Priority | Item | Effort |
|----------|------|--------|
| **P0** | Fix 2: Patch grader to handle "97 is prime" and similar format issues | 15 min |
| **P0** | Fix 1: Audit centroid-to-language distances, check EN bias | 30 min |
| **P1** | If bias confirmed: re-run surgery with leave-one-out centroid | 2-3 hrs (reuses existing cache) |
| **P1** | Re-score all conditions with fixed grader | 15 min |
| **P2** | MI decomposition: compute canonical correlations from cached activations | 1-2 hrs |
| **P2** | Plot MI curves (attn, mlp, residual, interaction) across depth | 30 min |
