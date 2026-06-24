# Z HUNT v3: STRUCTURAL-INFORMED KERNEL APPROACH

## What changed from v2

v2 (ARD-MMD) was designed blind — the author had no knowledge of the
static weight analysis already completed in 1.py and 2.py. v3 integrates
the structural findings to make the activation experiments faster, more
targeted, and more likely to find signal.

### Key structural findings (from 1.py/2.py on Qwen2.5-3B):

1. **Layer 33 is the bottleneck.** All 16 attention heads collapse to
   eff rank ~78 (mean 80.4, std 6.0). This is the lowest of all 36 layers
   and uniquely uniform across heads. Layer 34-35 snap back to ~112-118.

2. **1-33-2 architecture, not equal thirds.** Layer 0 is anomalous (massive
   rank drops 0→1). Layers 1-33 are progressive compute. Layers 34-35 are
   decode. The model spends 92% of depth on "middle" computation.

3. **Attention narrows, FFN broadens.** Attention effective rank decreases
   with depth (monotone down). FFN effective rank increases (monotone up).
   They're doing opposite things at every layer.

4. **FFN and attention are COUPLED.** 2.py experiment 3 (pending execution)
   measures FFN energy in the attention subspace. Code review suggests
   most layers are ABOVE chance — they operate in overlapping subspaces.

5. **W_V is constant.** Effective rank ~250 ± 5 across all 36 layers
   (CV 0.02). The "messages" being routed have fixed format. Only the
   routing decisions change with depth.

6. **GQA groups diverge functionally.** Heads 0-7 (KV head 0) vs 8-15
   (KV head 1) show systematically different effective ranks in many layers.

### What this means for Z extraction:

- **Layer 33 is the prime candidate for Z.** It's where the representation
  is most compressed, most uniform across heads, and furthest from both
  input text and output vocabulary.
- **Don't run all 36 layers equally.** Focus compute on layers 10-35.
  Layers 0-2 are encoding noise. Layer 33 and neighbors are the target.
- **The Z mask should respect the attention/FFN coupling.** You can't
  assume Z lives only in the attention subspace. FFN transforms in
  overlapping subspace.
- **W_V constancy suggests the "content" has fixed dimensionality.**
  Z probably has consistent size across layers, even if it's most
  compressed at layer 33.

### Model decision: Qwen2.5-3B, not Qwen3-8B

v2 targeted Qwen3-8B. v3 targets **Qwen2.5-3B** because:
- All structural analysis was done on 2.5-3B. Layer 33 bottleneck,
  GQA divergence, etc. are confirmed for this model.
- 3B fits easily in 12GB VRAM at 4-bit with full activation extraction.
- The behavioral tests (Chinese >> English) were on Qwen3-Vision-8B.
  We need to verify the behavioral asymmetry exists in 2.5-3B too
  (PHASE 0 below). If it doesn't, we pivot to 3-8B.
- Dimensions: d=2048 (not 4096). Kernel matrices are 4x smaller.
  Optimization is 4x faster. This is a significant speedup.

---

## PHASE 0: BEHAVIORAL VERIFICATION (30 min)

**v3 addition.** Before running expensive activation experiments, verify
that Qwen2.5-3B actually shows the Chinese >> English asymmetry on math.

- [ ] Load Qwen2.5-3B (already done from 1.py setup)
- [ ] Run 5 of the harder Putnam/math prompts in both Chinese and English:
  - Popcount f(n)=3 in [1,2025]
  - Putnam A2 (sin inequalities)
  - Integer solutions to x² + y² = 2025
  - Euler's number digit count
  - A combinatorics problem
- [ ] Score: does Chinese consistently outperform English?

**IF YES:** Continue with Qwen2.5-3B.
**IF NO:** The asymmetry may be specific to Qwen3-8B or the Vision variant.
Options: (a) try Qwen3-8B anyway, (b) use 2.5-3B for structural Z
extraction even without behavioral asymmetry (Z might still exist but
the behavioral gap may be too small to observe at 3B).

---

## PHASE 1: RUN 2.py (1 hour)

**v3 addition.** 2.py was written but NEVER EXECUTED. It contains three
experiments that directly inform the activation strategy:

- [ ] Run 2.py / 2.ipynb end-to-end
- [ ] Review Experiment 1 (subspace overlap): Does the overlap matrix show
  block-diagonal structure? Are the "phases" visible?
- [ ] Review Experiment 2 (bottleneck convergence): Does similarity to
  layer 33 increase from both directions?
- [ ] Review Experiment 3 (FFN-attention alignment): Are they coupled or
  independent? This determines whether Z extraction should target
  attention alone or the full hidden state.
- [ ] Update this gameplan with findings before proceeding

---

## PHASE 2: ACTIVATION EXTRACTION (1-2 hours)

Same approach as v2 but with d=2048 and targeted layer selection.

- [ ] Hooks on all 36 layers (same code as v2, but d=2048 not 4096)
- [ ] Hardcode 20-30 paired Chinese/English math prompts
- [ ] Run all prompts through model, store full sequence activations
- [ ] Each activation: [n_tokens, 2048] — 4x smaller than v2's 4096

**Storage estimate:** 20 pairs × 2 langs × 36 layers × ~50 tokens × 2048 × 4 bytes ≈ 1.2 GB

---

## PHASE 3: TARGETED ARD-MMD OPTIMIZATION (2-3 hours)

Same mathematical core as v2, but with structural priors.

### Change 1: Focus on layers 10, 15, 20, 25, 28, 30, 32, 33, 34, 35

Don't optimize all 36 layers. The structural analysis shows:
- Layers 0-2: encoding transition, noisy, skip
- Layers 3-9: still settling in, low priority
- Layers 10-30: progressive compute, sample every 5
- Layers 28-35: the critical zone around the bottleneck, sample every 1-2

This cuts optimization from 36 layers to ~10, saving 3x compute.

### Change 2: Initialize lengthscales using effective rank data

v2 initialized all lengthscales equal. v3 uses the structural analysis:
```python
# Dimensions where attention is low-rank → likely Z candidates
# Use the attention kernel's top singular vectors to seed the prior
V_layer33 = get_attn_subspace(model, 33, h, GQA, d, head_idx=0, k=80)
# These 80 dimensions are the attention's "query space" at the bottleneck
# Initialize log_ell smaller for these dimensions (stronger prior for Z)
```

This gives the optimizer a head start — it already knows roughly where
the low-dimensional structure is.

### Change 3: d=2048 instead of 4096

Kernel matrices are [n, m] but the lengthscale vector is [2048].
4x fewer parameters to optimize. Faster convergence expected.

---

## PHASE 4: READ THE PLOTS (same as v2)

Same interpretive framework. The |Z_k| vs layer plot now has a strong
prediction from the structural analysis:

**Structural prediction:** |Z| should be SMALLEST at layer 33 (where
the attention kernel has the lowest effective rank). If |Z_33| ≈ 80-100
dimensions, that matches the attention kernel's effective rank. If |Z_33|
is very different from ~80, the attention rank and the activation-level
Z are measuring different things.

**The lengthscale spectrum at layer 33 should be the cleanest** — the
structural analysis shows all heads converge to similar rank there,
which suggests the representation is maximally organized.

---

## PHASE 5: VALIDATION (same as v2)

Same three tests:
1. Same problem, different languages → close in Z
2. Different problems, same language → far in Z
3. Z predicts answer quality better than Z⊥

---

## PHASE 6: ACTIVATION PATCHING (same as v2, but targeted)

Patch at layer 33 specifically. The structural analysis says this is
where the representation is most compressed and most interpretable.

If layer 33 patching doesn't work, try layer 32 and layer 34 as
immediate neighbors of the bottleneck.

---

## PHASE 7: BRIDGE (same as v2)

Linear bridge within Z at layer 33. Size: [|Z|, |Z|] ≈ [80, 80] = 6,400
parameters. Tiny. Almost no overfitting risk even with 20 prompt pairs.

---

## EMERGENCY SHORTCUTS (updated)

**If behavioral verification fails (Phase 0):**
Skip to activation extraction anyway. Z might exist even if the behavioral
gap is small at 3B. Or switch to Qwen3-8B.

**If 2.py reveals FFN-attention are independent (Phase 1):**
Good news — Z extraction can target attention subspace alone. Use the
attention kernel's top-k SVD vectors as a fixed Z mask instead of
optimizing ARD lengthscales. Saves Phase 3 entirely.

**If 2.py reveals subspaces are all redundant (Phase 1):**
The depth IS partially wasteful. Consider averaging activations across
blocks of 5 layers for a smoother, less noisy Z extraction.

---

## TIME BUDGET

| Phase | Time | What |
|-------|------|------|
| 0 | 30 min | Behavioral verification |
| 1 | 1 hr | Run 2.py + review |
| 2 | 1-2 hr | Activation extraction |
| 3 | 2-3 hr | ARD-MMD optimization |
| 4 | 30 min | Read plots |
| 5 | 1 hr | Validation |
| 6 | 1 hr | Patching |
| 7 | 30 min | Bridge |
| Buffer | 1-2 hr | Debug, iterate |
| **Total** | **~10-12 hr** | |

Down from v2's 24 hours because:
- Smaller model (2048 vs 4096)
- Fewer layers to optimize (10 vs 36)
- Structural priors for initialization
- No time spent rediscovering structure that 1.py already found

---

*This is the plan. Phase 0 and Phase 1 come first because they
determine whether the rest of the plan needs revision.*
