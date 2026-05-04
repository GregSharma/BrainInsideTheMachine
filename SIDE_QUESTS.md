# Z PROJECT: OUTSTANDING SIDE QUESTS
## Captured March 5, 2026 ~11:00 AM

---

## IMMEDIATE (before writing paper)

### 1. InternLM2-Math at L28 (not L31)
- DONE: R² jumped from 0.52 to 0.754 at L28
- Four-model gradient confirmed: LLaMA(0) → DeepSeek(0.35) → InternLM2(0.75) → Qwen(0.94)

### 2. Label the outlier cluster in DeepSeek t-SNE
- One cluster in fig7 shows DeepSeek separating from Qwen
- Which math category is it? Arithmetic? Sequences?
- Tells you exactly where DeepSeek's math-specific training diverges

### 3. Normalized-depth NN degradation figure
- All models on one plot: NN accuracy vs normalized depth (0 to 1)
- Qwen-3B (stays high), Qwen-1.5B (slow decay), DeepSeek (fast decay), LLaMA (immediate floor)
- InternLM2-Math (INCREASES - the integration signature)
- Area under curve = single-number bilingual integration metric
- This is THE Lyapunov figure

---

## THEORETICAL (formalize during/after paper)

### 4. Bilingual gauge symmetry theorem
- Monolingual model: gauge-invariant, any basis works, no forced Z
- Bilingual model: shared weights W_k must advance both languages
- Forces factorization into Z (reasoning, language-invariant) and Z-perp (language-specific)
- Z = nullspace of (dL_zh/dW_k - dL_en/dW_k) -- where gradients agree
- |Z| correlates with mutual information between bilingual training objectives
- LLaMA fails because bilingual pressure never broke the gauge symmetry

### 5. Proof by contradiction (from the walk)
- If Z does NOT exist -> shared W_k cannot synchronize bilingual outputs
  -> Lyapunov exponents diverge across languages
  -> no last-layer alignment
  -> but we observe R^2 = 0.976. Contradiction.
- If Z exists -> same W_k advances both languages in Z
  -> Lyapunov exponents match within Z
  -> last-layer alignment follows necessarily

### 6. von Neumann / brain connection
- Brain: ~10W, ~100Hz, low precision -> MUST use distributed overlapping codes
- Transformer: finite parameters -> MUST use shared representations across languages
- Z is the hippocampal index of the transformer -- content-addressable, invariant across input paths
- Same optimization problem: maximize reasoning across multiple input modalities under resource constraints
- Reference: von Neumann "The Computer and the Brain" (1958)

---

## APPLICATIONS (prototype after paper)

### 7. Z-preserving training objective
- Auxiliary loss: at each layer k, penalize corruption of Z-directions
- L_aux = sum_k ||P_Z(h_{k+1}) - P_Z(h_k + f_k(h_k))||^2
- Teaches MLPs to route around Z, do language work in Z-perp only
- Target: make 1.5B retain Z as well as 3B does
- Middle layers are where the bleed happens, not early or late

### 8. Z-aware quantization
- Quantize Z-directions at FP16/FP32 (reasoning-critical, 20 dims)
- Quantize Z-perp at INT4/INT2 (language and noise, 2028 dims)
- Better than uniform quantization because bits allocated where they matter
- 3B quantized this way might match FP16 reasoning at half memory
- LITERATURE: FASC (2026) is closest -- Fisher-aligned subspace compression
  DiaQ (2026) preserves direction of activation vectors
  No one has done explicit functional-subspace mixed precision yet

### 9. Speculative Z-decoding (model splicing)
- Run 1.5B through middle layers (cheap reasoning channel)
- Splice Z-projection into 3B's late layers (expensive but better decoder)
- R^2 = 0.94 means the 3B's decoder should accept 1.5B's Z-output
- CORRECTION: the expensive part is middle layers (preserving Z through depth)
  so splice should be: 3B's MIDDLE layers + 1.5B's early/late layers
- 30-min prototype: project 1.5B L27 Z into 3B L28, generate, check coherence
- LITERATURE: Lyanna (2025) does hidden-state chain speculative decoding
  BRIDGE (2025) projects across modalities. No one has done small->large LLM projection.

### 10. Z-filtered training data
- Score each training example by Z-update vs Z-perp-update magnitude
- High Z-update = reasoning-rich example (hard, multiple paths, Lyapunov-unstable)
- High Z-perp-update = pure language pattern-matching
- Train more on Z-hard examples -> stronger Z-preservation
- This is the Lyapunov training data filter

---

## EXPERIMENTAL (for v2 paper or follow-up)

### 11. Attention vs MLP decomposition
- Hook INSIDE each layer: after attention block, after MLP block separately
- Measure NN accuracy at each sub-step
- Does Z degrade at the attention step or MLP step?
- Hypothesis: MLP is the bottleneck (attention copies, MLP overwrites)
- Proves whether bigger MLPs = better Z preservation

### 12. Proof geometry / trajectory analysis
- Token-by-token Z-trajectories for multi-step problems (not mean-pooled)
- Plot in top 3 PCA dimensions of Z
- Do proof steps = directional changes in Z-space?
- If yes: auditable reasoning, alignment-relevant, Anthropic cares
- Hardest experiment but most profound

### 13. Non-template diverse math problems
- All 200 current problems are from 5 templates
- Test with natural-language problems: "A farmer has three times as many..."
- Does contrastive Z still work when templates aren't rigid?
- Would strengthen the claim significantly

### 14. More languages (beyond 7)
- Current: zh, en, es, ar, ja, ko, sw
- Add: hi (Hindi), ru (Russian), de (German), fr (French), pt (Portuguese)
- More points on the distance gradient
- Test: does distance correlate with training data volume per language?

### 15. Larger Qwen models (7B, 14B, 72B)
- Does Z dimensionality change with scale?
- Does the NN degradation curve flatten (better channel)?
- At what scale does the 1.5B's bleed disappear entirely?
- Might need API access or cloud GPU for 72B

### 16. Second model family with deep bilingual (non-Chinese lab?)
- All current bilingual models are Chinese labs
- Need: European/US lab with genuinely bilingual model
- If such a model exists and shares Z -> truly universal
- If only Chinese labs show it -> might be training culture, not math

---

## NESTED OU CASCADE (separate paper, math finance)

### 17. The rough vol paper
- Cascaded OU as Markovian lift of rough volatility
- Sequential Riccati CFs, exact MC, Hermite density expansion
- Convergence rates for cascade kernel -> fractional kernel
- Different from Abi Jaber (parallel sum) -- cascade is sequential (bidiagonal)
- Full handoff in nested_ou_handoff.md
- Quick start: 100 lines Python, variogram of 5-stage cascade, check H < 0.5

---

## META / CAREER

### 18. Paper structure
- Title: "Language Lives in Five Dimensions" or similar
- Narrative: SVD wrong -> patching wrong tool -> circularity caught -> contrastive works
- Figures: hero t-SNE, layer evolution, distance distributions, multilingual heatmap,
  cross-scale alignment, DeepSeek partial alignment, InternLM2 result, NN degradation curves
- Theory section: gauge symmetry argument
- Venues: ICML 2026, NeurIPS 2026, EMNLP

### 19. Training data auditing tool
- Distance gradient in Z = forensic map of training data composition
- Rank languages by mathematical training exposure from Z-distances alone
- Publishable as standalone tool/method paper
- Applicable to any multilingual model

### 20. The LinkedIn/Twitter thread
- Lead with Figure 1 (hero t-SNE)
- One sentence: "Language lives in 5 dimensions. Math lives in 20."
- 8-10 tweets max
- Tag: Anthropic interp team, Neel Nanda, NeurIPS 2505.15257 authors
- DO NOT POST until paper is on arxiv (priority protection)
