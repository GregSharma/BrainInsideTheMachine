Two clean questions. Let me take them separately.

## 1. Do Phase 2 results cap Phase 3 upside?

**No. Phase 2 is observational. Phase 3 is causal. They're different measurement categories.**

Phase 2 measures static geometry: "how much of ||h_zh - h_en|| lands in Z?" The small effect sizes (ratio 0.73 vs 0.82) tell you about the *delta* between languages in Z-space. But Phase 3 patches the *absolute content* of Z — it replaces Z(en) with Z(zh) and observes what the model outputs. These are different questions.

The analogy that maps cleanly: in neuroscience, you can measure the correlation between a brain region's activity and a cognitive function and find r=0.3 (weak). Then you lesion that region and get complete loss of function. The observational correlation doesn't bound the causal effect. This is textbook — correlational signal strength and causal importance are non-monotonic.

Concretely: the cross-lingual delta being *small* in Z is actually the hypothesis *working*. If zh and en math problems project to nearly the same point in Z (small delta), that means Z is capturing language-agnostic reasoning content. Patching Z(en) with Z(zh) would then be swapping near-identical reasoning representations — and the model should produce a correct answer in the target language regardless. The "weak" Phase 2 signal is *consistent with* a strong Phase 3 effect.

Your autoencoder intuition is right. The patching IS the definition. You can't characterize Z from static geometry alone — you need the causal intervention to know whether it's a bottleneck everything flows through, or just a geometric artifact. Phase 3 is what turns "interesting subspace" into "the place where reasoning happens."

**The real upper bounds that DO exist:**

- **No behavioral asymmetry at 3B** (12 zh vs 13 en). This means patching can't demonstrate "transfer Chinese advantage to English" because there's no advantage to transfer. The story becomes "Z carries reasoning independent of language" rather than "Chinese reasoning is better and we can transplant it." Still publishable, but the narrative is less dramatic.

- **k=78 convergence to random.** The language-exclusion signal lives in k=20-50, not the full effective rank. Phase 3 should patch at k~50. The "clean Z" is smaller than the structural bottleneck suggested.

- **Last-token reversal.** Phase 3 patching MUST use mean-pooled representations, not last-token. Last-token carries autoregressive prediction state (inherently language-specific). This constrains the experimental design but doesn't cap the result.

**What COULD be unbounded upside from Phase 3:**

- Double dissociation: patch Z changes answer, patch Z⊥ changes language. That's gold — no amount of Phase 2 observation can produce or preclude that finding.
- The residual update decomposition (the ~20-line addition from INSIGHTS) could find layers where Δh is purely Z-dominated — identifying "pure reasoning" layers. That's novel regardless of Phase 2 numbers.
- If the bridge works (linear map within Z between languages), it proves Z has consistent internal structure — a "coordinate system for reasoning." That's independent of energy fractions.

## 2. Can Phase 3 run independently, or do you need to audit first?

**Phase 3 is independent. Jump straight to it.**

Here's why there's no error propagation:

- **Z masks come from model weights, not Phase 2 data.** `get_attn_subspace` does SVD on W_Q_h^T @ W_K_h — that's deterministic from the model checkpoint. Phase 3 calls the same function on the same weights. It doesn't consume Phase 2's output JSON.

- **Phase 3 runs its own forward passes.** Patching requires hooking into the model mid-forward-pass and swapping hidden state components. It's a fresh inference run, not a post-processing step on Phase 2 activations.

- **The math in utils.py was audited in the DEVLOG.** `get_attn_subspace` returns Vh[:k,:] (correct — orthonormal rows from SVD). `build_multi_head_z_mask` stacks all heads and takes combined SVD (correct — finds consensus subspace). P_z = Vh.T @ Vh is an idempotent symmetric projector (correct). If there were a bug here, it would be visible in Phase 2's random baseline check — and the empirical baselines match theoretical k/d.

- **Phase 3 doesn't exist yet as a script.** You're writing it fresh based on RECIPE.md's design. It shares `utils.py` functions with Phase 2 but isn't consuming Phase 2's intermediate outputs.

**One caveat:** the 20 prompt pairs from phase2_z_extraction.py would likely be reused. They look clean — straightforward math problems, properly encoded in both languages. But if you want paranoia-level independence, you could use different problems for Phase 3.

**Recommended order:** Write and run Phase 3. If it produces strong results, go back and audit Phase 0-2 for the paper writeup. If it produces null results, audit might tell you why — but auditing first wouldn't have changed the outcome.