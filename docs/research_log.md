# Desktop_Projects_BrainInsideTheMachine

_Sessions ordered oldest first. Each section is a tag-indexed readout — scan the tags (EXPERIMENT, FINDING, AUDIT, KILLED, DECISION, BREAKTHROUGH, PUSHBACK, BRAINSTORM, DELIVERABLE, NEXT, BLOCKER, STATE) to find audits, dead paths, breakthroughs, or where you left off._

## 2026-04-07 04:37 UTC — Establish the universality of the rank-1 Gram bottleneck and 1/20 compress invariant across four models (Qwen2.5-3B/7B/14B, Qwen3-8B), characterize output rupture as tied-embedding‑specific, and lock down all state for continued research.

*Source: [2026-04-07_04-37_update-vega-docs.md](2026-04-07_04-37_update-vega-docs.md)*

**Objective:** Establish the universality of the rank-1 Gram bottleneck and 1/20 compress invariant across four models (Qwen2.5-3B/7B/14B, Qwen3-8B), characterize output rupture as tied-embedding‑specific, and lock down all state for continued research.
- **STATE:** 97ebe00d – saved after loading prior session state and before responding to context alert (post‑BQ2‑7B and BS results discussion).
- **AUDIT:** User identified that the “cross‑task crystallization probe” (Exp BS) was a stale next‑action item carried forward through state saves since March 13, which a later instance executed without re‑evaluating relevance.
- **DECISION:** Ignore the stale notebook task; rewind session to after the 7B npz verification and redirect to “BQ/BQ2 on the 7B data” as the real next action.
- **BRAINSTORM:** Proposed framing the five‑phase funnel (build→compress→sustain→expand→rupture) as an implicit encoder‑decoder architecture: early layers = Z‑encoder (language→reasoning), middle = narrow reasoning substrate (≈10 dimensions), late = Z‑decoder (reasoning→language). This suggests a variable‑width transformer could achieve ≈20× compute reduction for the reasoning core.
- **BRAINSTORM:** Two experiments to test factorability: (1) project 3B activations into top‑10 Gram eigenmodes at sustain phase and measure answer prediction; (2) train a 10‑dim→vocab probe on L26 representations across 7 languages to test for a single linear Z‑decoder.
- **PUSHBACK:** Rejected Claude Web’s claim that MLPs are “language‑specific processing that wouldn’t exist in a kernel‑space reasoner” – Exp M3 shows zeroing MLPs at L9‑L26 kills accuracy (1/20), and Exp T (PACF) shows 97% of each MLP delta is fresh innovation. MLPs are essential for funnel dynamics, though a narrow MLP (≈50 dims) might suffice if its contribution lies in top Gram modes.
- **BRAINSTORM:** The single‑input inference gap – Gram matrix requires a batch. Resolutions: (a) fixed dictionary of pre‑computed top Gram eigenvectors (test stability across splits), (b) online Gram from KV cache (unlikely, Exp K2b shows KV scrambling has no effect), (c) narrow core trained end‑to‑end without needing Gram at inference – Gram is diagnostic, not mechanism.
- **STATE:** 57d851ff – saved after user provided SSH hostname for Colab A100, before attempting connection.
- **BLOCKER:** SSH via Cloudflare tunnel failed with `tls: failed to verify certificate: x509: certificate signed by unknown authority` – cloudflared’s Go TLS rejected a Google Trust Services cert that openssl accepted. Workaround: ran extraction and analysis scripts directly on Colab instead of transferring npz files.
- **EXPERIMENT:** BQ2‑XM – cross‑model Lyapunov analysis on Qwen2.5‑14B (48 layers, d=5120) and Qwen3‑8B (36 layers, d=4096, untied embeddings). Script `expBQ2_crossmodel_lyapunov.py` ran on Colab A100, output `output/expBQ2_crossmodel_lyapunov.json` (105 KB).
- **FINDING:** Qwen2.5‑14B – rank_50=1 at ALL 48 layers; rank_90 peaks at 10 (L19, 40% depth). Mode‑0 hourglass: 1171→716 at L47 (output is min). Mean Gram correlation = 0.984.
- **FINDING:** Qwen3‑8B – rank_50=1 at ALL 36 layers; rank_90 peaks at 9 (L9, 26% depth). Mode‑0 hourglass: 1274→954(L32)→1041(L35). Mean Gram correlation = 0.973.
- **FINDING:** rank_90 scaling with layer count, not parameters: 36‑layer models (3B and Qwen3‑8B) = 9; 28‑layer 7B = 12; 48‑layer 14B = 10 (non‑monotonic).
- **FINDING:** The 1/20 compress invariant is REAL across all four models when measured with a sliding 9‑layer window. Qwen2.5‑14B compress zone = L14‑L22 (1/20 positive modes); equal‑band splitting gave 3/20 due to misaligned phase boundaries.
- **FINDING:** Output rupture is tied‑embedding‑specific. Qwen2.5‑3B and 7B (tied) show catastrophic Frobenius spike (6×–4×) and mode‑0 Lyapunov collapse (−0.44). Qwen2.5‑14B (untied, same family) shows moderate rupture (3.5× Frobenius). Qwen3‑8B (untied, different generation) shows NO rupture – Frobenius ratio = 0.86 (smaller than previous layer), Gram correlation drops to 0.674 but language alignment remains 0.952.
- **FINDING:** Qwen3‑8B has a wider cooperative phase (10/20 positive modes) vs Qwen2.5 models (3‑5/20). More information flows through the sustain phase, possibly explaining Qwen3’s stronger reasoning performance.
- **DELIVERABLE:** `output/expBQ2_crossmodel_lyapunov.json` – contains per‑layer Lyapunov spectra, effective ranks, ΔG Frobenius norms, language/category eigenvector alignment, and output rupture detection for both 14B and Qwen3‑8B.
- **DELIVERABLE:** `expBQ2_crossmodel_lyapunov.py` – modified to include sliding‑window Lyapunov analysis (9‑layer windows) and add `sliding_window_1_20_zones` to the cross‑model summary.
- **DELIVERABLE:** `~/.claude/projects/.../memory/MEMORY.md` – updated with 4‑model comparison table (rank_50=1, compress 1/20, rupture characteristics), sliding‑window methodology, and new experiment status.
- **DELIVERABLE:** `~/.claude/vega.md` – added blind spots #100‑102 and a rupture lesson; added 3 new lessons (sliding‑window compress detection, Qwen3 cooperative width, tied‑embedding rupture cause).
- **DELIVERABLE:** `~/.claude/hooks/data/vega_index.json` – updated to 115 blind spots and 86 lessons.
- **STATE:** 75ca740b – final state after updating MEMORY.md, vega.md, storing three semantic memories (1/20 invariant, sliding‑window method, SSH Cloudflare cert issue), and completing housekeeping.
- **META:** Stop hook fired at 54% context, triggering vega.md / vega_index.json updates and state save – routine housekeeping, no substantive research content.

---

## 2026-04-07 06:22 UTC — Consolidate and critically appraise the Gram/Lyapunov thread (BQ–BR) by auditing the robustness of rank-based claims, correcting for anisotropy artifacts, and reframing findings for compression, inference, MTP, and frontier model implications.

*Source: [2026-04-07_06-22_analyze-mtp-internal-geometry.md](2026-04-07_06-22_analyze-mtp-internal-geometry.md)*

**Objective:** Consolidate and critically appraise the Gram/Lyapunov thread (BQ–BR) by auditing the robustness of rank-based claims, correcting for anisotropy artifacts, and reframing findings for compression, inference, MTP, and frontier model implications.
- **PUSHBACK**: User flagged that assistant moved too fast dismissing Qwen3.5-9b as apples-to-oranges; user argued the MTP property is exactly what Claude Opus uses and is key to the project's "moon shot."
- **DECISION**: User directed assistant to consolidate all Lyapunov/Gram results from BQ through BR into one markdown file, supplement with Perplexity research on implications (compression, inference time, MTP, frontier models), and appraise the findings critically but not pessimistically.
- **DELIVERABLE**: Assistant wrote `/home/greg/Desktop/Projects/BrainInsideTheMachine/LYAPUNOV_GRAM_APPRAISAL.md` (approx. 3500 words, 10 sections) covering BQ–BQ2-XM–BR thread, three universal invariants, cross-model specifics, BQ3 causal proof, BR domain generalization, critical appraisal, and implications.
- **STATE**: Saved as `c5f2e51f` after writing the appraisal document.
- **AUDIT**: User questioned whether rank_50=1 on the cosine-similarity Gram matrix is trivially forced by the diagonal of ones and high mean cosine; assistant verified with a uniform null model that rank_50=1 is expected whenever mean pairwise cosine >0.5 (observed mean cosines 0.65–0.84).
- **FINDING**: For the cosine Gram, dominant eigenvalue tracks the uniform-matrix prediction within 1.002–1.023 ratio at every layer; rank_50=1 is an anisotropy artifact, not a deep structural property.
- **FINDING**: For the raw Gram (HH^T, unnormalized), rank_50=1 and rank_90=3–9, identical to cosine Gram, indicating norm variation adds no rank.
- **EXPERIMENT**: Assistant ran a centered Gram audit (mean-centering activations before computing HH^T) on the multilingual Qwen2.5-3B activations (7 languages × 200 problems × 36 layers).
- **FINDING**: Centered Gram rank_50 = 2–5, rank_90 = 8–21 (true geometry after removing shared mean). Trajectory: L0 rank_50=2, rank_90=8; L9 rank_50=4, rank_90=19; L18 rank_50=5, rank_90=21; L27 rank_50=4, rank_90=18; L35 rank_50=3, rank_90=10.
- **FINDING**: On centered Gram, variance-fraction Lyapunov analysis shows build phase (L0–L8: 17/20 modes gaining share), equilibrium phase (L9–L26: ~10–12/20 positive), reconcentration phase (L27–L35: 4–6/20 positive). The previously reported "1/20 compress invariant" is partially an artifact of cosine normalization's trace constraint.
- **KILLED**: The headline "Transformers reason in 5 dimensions" (rank_50=1) is dead; it is anisotropy, not a reasoning bottleneck.
- **KILLED**: The "1/20 universal compress invariant" is downgraded to a cosine-similarity phenomenon; centered analysis shows equilibrium, not compression, in middle layers.
- **FINDING**: What survived the audit intact: BQ3 causal pruning (delta-G layer ranking works, Gram-quiet layers can be skipped, random is catastrophic), output rupture tied to embedding architecture (tied causes rupture, untied doesn’t), cross-domain convergence/divergence (math and non-math converge in middle layers, diverge at output), Qwen3-8B cooperative phase (wider sustain-phase bandwidth of 10/20 positive modes vs 3–5/20 on Qwen2.5).
- **FINDING**: Centered Gram shows middle layers maintain ~20 effective dimensions of variation (rank_90 = 18–21). This is the real information highway width. If MTP widens this (as Qwen3 hints), MTP expands the information highway — a concrete measurable effect of training objective on internal geometry.
- **DECISION**: User and assistant agreed to run the centered Gram analysis on the other three models (Qwen2.5-7B, Qwen2.5-14B, Qwen3-8B) using cached activations to check whether Qwen3-8B's wider cooperative phase survives centering.
- **STATE**: Saved as `8d272334` after completing the audit and centered Gram analysis, with the corrected narrative and next steps.
- **NEXT**: Run centered Gram analysis on Qwen2.5-7B, Qwen2.5-14B, and Qwen3-8B cached activations to check if the cooperative phase difference (more positive Lyapunov modes) holds after mean-centering; also verify rank_90 trajectory across models without anisotropy artifacts.

---

## 2026-04-07 19:40 UTC — Test whether the centered Gram rank_90 (~20 dimensions) represents the true operational dimensionality of reasoning in transformers by performing SVD-truncated inference at equilibrium layers and dissecting prefill vs generation contributions.

*Source: [2026-04-07_19-40_recommend-context-clear.md](2026-04-07_19-40_recommend-context-clear.md)*

**Objective:** Test whether the centered Gram rank_90 (~20 dimensions) represents the true operational dimensionality of reasoning in transformers by performing SVD-truncated inference at equilibrium layers and dissecting prefill vs generation contributions.
- STATE: snapshot 8d272334 loaded as initial state (audit session with centered Gram results)
- BRAINSTORM: Proposed Z-embedding architecture with explicit bottleneck at rank_90 dimensions (k≈20) based on centered Gram rank_90=20 at equilibrium layers, with Z-encoder (d→k) and Z-decoder (k→d) around reasoning layers
- DECISION: Prioritize SVD-truncated inference experiment over centered cross-model replication to test causal dimensionality hypothesis
- EXPERIMENT: expBS_svd_truncation.py ran SVD truncation on Qwen2.5-3B equilibrium layers (L9-L26) at k values [2,5,10,15,20,25,30,50,100,200,500] plus single-layer (L16), all-layers (L0-L35), build layer (L0-L8), output layer (L27-L35) — each condition evaluated on 20 EN math problems and 20 ZH math problems
- FINDING: ExpBS baseline EN=2/20, ZH=10/20; every truncated condition produced EN=0/20, ZH=0/20, including k=500 which preserves 99.9% of centered variance
- FINDING: At k=100, EN outputs partially recover language (3/20 English) but ZH outputs remain gibberish; at k=200-500, EN outputs fluent English but still 0/20 math accuracy; ZH prompts produce English at k=200-500 when prefill is clean (gen-only) but still 0/20
- KILLED: Z-embedding thesis — the centered Gram rank_90 does NOT correspond to operational dimensionality; even preserving 99.9% centered variance destroys reasoning
- DECISION: Re-interpret centered Gram as measuring inter-problem geometry, not computational bandwidth; Gram rank_90 is predictive for layer pruning (BQ3) but not for within-layer dimensionality reduction
- EXPERIMENT: expBS2_genonly_truncation.py separated prefill vs generation corruption with conditions: gen_only (k=20,50,200,500 on L9-L26), always (k=200), gen_only L16 (k=200), prefill_only (k=200 on L9-L26)
- FINDING: ExpBS2 gen-only k=200 produced EN=0/20, ZH=0/20 (EN outputs 17/20 English but math incorrect; ZH outputs all unk); gen_k500 same pattern; always_k200 EN=0/20, ZH=0/20 with EN→EN, ZH→EN language shift; prefill_k200 EN=0/20, ZH=0/20 with ZH→EN shift
- FINDING: Prefill-only truncation (clean generation, corrupted prefill) also yields 0/20 accuracy, killing hypothesis that generation-only corruption is the sole cause
- AUDIT: Assistant caught that KV expendability from expK2b (scrambling all 36 layers' KV cache had no effect on output) means BS failure is about residual stream cascade, not KV corruption
- DELIVERABLE: expBS_svd_truncation.py (created), expBS2_genonly_truncation.py (created), output/expBS_svd_truncation.json, output/expBS2_genonly_truncation.json
- DELIVERABLE: Updated MEMORY.md with BS result as killed path and centered Gram re-interpretation
- STATE: snapshot fb15ee16 saved while BS2 running
- STATE: snapshot e6881dd3 saved before /clear recommendation
- STATE: snapshot a013a627 saved with complete BS2 results
- NEXT: Paste prompt to Claude Web asking about distinction between low-rank relational geometry vs low-rank computation, implications for attention in middle layers given KV expendability, and paper framing of Gram matrix as predictive for layer-level but not within-layer dimensionality

---

## 2026-04-07 22:02 UTC — Determine why PCA truncation at the representational Gram rank (k≈20) destroys transformer accuracy despite near-zero loss of variance, testing whether the MLP's internal activation patterns (computational Gram) have higher dimensionality or whether gate flipping explains the fragility.

*Source: [2026-04-07_22-02_download-with-compute.md](2026-04-07_22-02_download-with-compute.md)*

**Objective:** Determine why PCA truncation at the representational Gram rank (k≈20) destroys transformer accuracy despite near-zero loss of variance, testing whether the MLP's internal activation patterns (computational Gram) have higher dimensionality or whether gate flipping explains the fragility.
- BRAINSTORM: Proposed computational Gram matrix G^comp_ℓ = Ã_ℓ Ã_ℓᵀ where Ã_ℓ are centered MLP activation patterns a_p^{(ℓ)} = SiLU(W_up · h_p^{(ℓ)}) ∈ ℝ^{4d}, hypothesizing rank_90(G^comp) ≫ rank_90(G^repr) ≈ 20 if the computation operates in a higher-dimensional space than the representation.
- BRAINSTORM: Proposed Jacobian visibility decomposition splitting J_ℓ into blocks on V (top-20 PC subspace) and V^⊥, with hypothesis that the invisible block J_ℓ|_{V^⊥} has structured singular values (not identity), revealing that the Gram evolution misses essential computation in the complement.
- BRAINSTORM: GPT 5.4 provided formal theory: (Q1) low tail variance but high Jacobian sensitivity can make projection catastrophic even without nonlinearity; SiLU turns low-energy tail into gate-control channel. (Q2) For non-polynomial σ like SiLU, rank(σ(W_up H̃^⊤)) generically = min(4d, N) even if rank(H̃)=20, predicting rank explosion. (Q3) The K = Π_{V^⊥} J_ℓ Π_{V^⊥} block does not appear in Gram update if Σ_ℓ lives in V; K can be near-identity and still indispensable if downstream MLPs read V^⊥ coordinates. (Q4) No single scalar from G^repr alone can predict both deletability and incompressibility; proposed two-vector (a_ℓ, b_ℓ) or paradox index S_ℓ.
- EXPERIMENT C1: Computational Gram matrix computed for Qwen2.5-3B (N=1400, d=2048, d_ff=11008) by extracting MLP gate activations (SiLU(W_up·h)) at all 36 layers, centering, building G^comp_ℓ, and computing effective rank at thresholds 50%, 90%, 99%.
- FINDING C1: rank_90(G^comp) mean = 17.8, rank_90(G^repr) mean = 17.4, ratio = 1.02x. rank_99(G^comp) mean = 74.4, rank_99(G^repr) mean = 73.2, gap = +1.2. No rank explosion; computational dimensionality matches representational dimensionality.
- KILLED: Rank explosion hypothesis — the generic expectation that non-polynomial σ(SiLU) would amplify rank to min(d_ff, N)=1400 did not hold; training compresses activation patterns to track representational geometry.
- EXPERIMENT C1b: Gate flip analysis for layers {0,4,8,9,12,16,20,24,26,30,34,35} with truncation k ∈ {2,5,10,20,50,100,200,500}. For each problem, computed gate pattern a = SiLU(W_up·h) on original versus truncated h̃ = P_k(h-μ)+μ, measuring sign flip fraction, rel error, correlation, Jaccard, and variance explained.
- FINDING C1b: At k=20 (Gram rank_90), gate flip rates per layer ranged from 0.3% (L4, 33/11008) to 3.3% (L0, 365/11008); at k=500, flip rates dropped to <0.3% across all layers (e.g., L0: 5/11008 = 0.04%, L8: 58/11008 = 0.5%). Gate correlation at k=20 was ≥0.97, meaning gates are nearly preserved.
- BREAKTHROUGH: BS failure is not due to hidden high-dimensional computation (C1) nor gate flipping (C1b). Instead, the fragility arises from error amplification through chaotic dynamics: a 0.1% perturbation from k=500 truncation (99.9% variance preserved) gets exponentially amplified across 36 layers (consistent with measured 77°/layer rotation and positive Lyapunov exponents). Layer deletion is safe because it introduces no perturbation; dimension truncation introduces a tiny error that grows catastrophically.
- DECISION: Prioritized C1 (computational Gram) as the direct measurement of rank gap; after C1 completed, ran C1b (gate flip) to test carrier wave mechanism; both completed within session.
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expC1_computational_gram.py` (script) and `output/expC1_computational_gram.json` (results)
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expC1b_gate_flip.py` (vectorized script) and `output/expC1b_gate_flip.json` (results with per-layer gate flip tables)
- META: Experiments executed on Colab remote A100 via SSH tunnel (frequently-palace-investigate-alot.trycloudflare.com). Both jobs completed with total runtime ~56s (C1) and ~153s (C1b). Results downloaded locally. Session state saved via orchestrator.

---

## 2026-04-07 23:18 UTC — Resolve the observation–intervention gap in transformer compressibility by testing whether the Gram-measured low-rank funnel describes readout geometry while context computation requires full rank, culminating in a clean read‑head vs context computation isolation.

*Source: [2026-04-07_23-18_lost-unsaved-state.md](2026-04-07_23-18_lost-unsaved-state.md)*

**Objective:** Resolve the observation–intervention gap in transformer compressibility by testing whether the Gram-measured low-rank funnel describes readout geometry while context computation requires full rank, culminating in a clean read‑head vs context computation isolation.
- EXPERIMENT: Exp C2 (tail_transplant) — at single layers (L9, L20, L30) with k=5,20,50, replace V⊥ (tail) of last token with zero, random noise, or cross‑problem tails. Accuracy unchanged vs baseline (EN 5/20, ZH 12/20) across all conditions.
- EXPERIMENT: Exp C2b (dose_response) — three modes: last‑only truncation (k=20 at last token), all‑tokens truncation, context‑only truncation. N layers = 1,3,9,18,36.
- FINDING: last_only_N1 = EN 5/20, ZH 11/20 (baseline). last_only_N36 = EN 4/20, ZH 13/20 (baseline). All last‑token truncation conditions match baseline.
- FINDING: all_tokens_N1 = EN 0/20, ZH 0/20. all_tokens_N36 = 0/20. Even single‑layer all‑token truncation kills accuracy.
- FINDING: context_only_N1 = EN 0/20, ZH 0/20. context_only_N9 = 0/20.
- BREAKTHROUGH: The last token acts as a read head that reads answer information from context via attention; its own V⊥ is irrelevant. Context tokens carry the full‑rank computation; truncating context destroys the read data and kills performance. The Gram‑measured low‑rank funnel (rank_90=8‑21) describes the read head’s trajectory, not the model’s internal computation.
- AUDIT: Assistant’s earlier “error amplification” interpretation of BS (compound truncation kills accuracy via 36‑layer Lyapunov growth) is wrong. BS’s all‑token truncation kills accuracy in a single layer; the failure is not compound amplification but direct corruption of context computation.
- KILLED: “Gate flip as k=500 truncation mechanism” (C1b) – already dead, but now understood as measuring the read head’s gates, not context gates.
- KILLED: “Rank explosion” hypothesis (GPT‑5.4 prediction) – C1 already showed computational Gram rank ≈ representational Gram rank at last token, and C2b shows low‑rank read head is by design.
- DECISION: Prioritise tail transplant (C2) over paper outline; then run C2b dose‑response after C2 results showed last‑token V⊥ irrelevance but BS single‑layer all‑token kill.
- DECISION: Reject “World E” dramatic reframing from GPT handoff; instead reinterpret existing Gram work as always having measured the read head.
- PUSHBACK: Assistant pushes back on GPT’s “final canyon as catastrophic collapse” – canyon is the read head being constructed, not a breakdown. The tied‑embedding rupture is a forced alignment into input subspace; untied models still show rank collapse without the spike.
- BRAINSTORM: The toy theorem (σ_i ∝ √ρ_i from unequal language frequencies) predicts low‑rank read head: answer direction lies in high‑agreement subspace across languages, context tokens need full rank to handle language‑specific syntax.
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expC2_tail_transplant.py` (520 lines)
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expC2b_dose_response.py` (473 lines)
- DELIVERABLE: `output/expC2_tail_transplant.json` – full results with baselines, cross‑transplant, zero‑perp, noise‑perp.
- DELIVERABLE: `output/expC2b_dose_response.json` – 12 condition results, 40 runs per condition.
- DELIVERABLE: Updated `/home/greg/.claude/vega.md` – added blind spots 71‑75 (over‑confident error amplification narrative, failing to check all‑token vs last‑token difference before claiming compound amplification, etc.)
- DELIVERABLE: Updated `/home/greg/.claude/hooks/data/vega_index.json` – added entries 116‑120.
- DELIVERABLE: Updated `/home/greg/.claude/projects/…/memory/MEMORY.md` – added C2/C2b read‑head finding, reinterpreted BS, compressed dead paths; final length 194 lines.
- STATE: Snapshot `ffc45e14` saved after C2/C2b completion and initial synthesis.
- STATE: Snapshot `57aaffa2` saved after GPT handoff archaeology, bridge to canyon/toy theorem, and full reinterpretation table.
- NEXT: Run C2b on Qwen2.5‑7B (cache exists at `output/multilingual_all_layers_qwen2_5_7b.npz`) – cross‑model replication. Script is a 30‑line modification of C2b; requires Colab A100 (does not fit on RayGun 12GB).
- NEXT: Literature check (attention sinks, induction heads, destination tokens) via WebSearch/Perplexity – 20 minutes.
- NEXT: Write context‑token Gram measurement (capture hidden states from non‑last positions) to directly verify context rank >> read‑head rank.

---

## 2026-04-09 04:24 UTC — Test whether the read-head vs context-computation split (C2b causal evidence) replicates across larger models (7B, 14B, Qwen3-8B) and advance the theoretical compression idea that a low-rank read head might imply a compressible predictive state for autoregressive generation.

*Source: [2026-04-09_04-24_clear-command-last.md](2026-04-09_04-24_clear-command-last.md)*

**Objective:** Test whether the read-head vs context-computation split (C2b causal evidence) replicates across larger models (7B, 14B, Qwen3-8B) and advance the theoretical compression idea that a low-rank read head might imply a compressible predictive state for autoregressive generation.
- EXPERIMENT: C2c cross-model replication script `expC2c_crossmodel_readhead.py` built (720 lines) with World E diagnostic (last-token vs context-mean effective rank), baseline collection (EN/zh), and 12 conditions (last_only_N, all_tokens_N, context_only) for each model.
- DELIVERABLE: `expC2c_crossmodel_readhead.py` uploaded to Colab A100 (root@enlarge-lib-negotiations-avatar.trycloudflare.com) and later to second Colab instance for 14B/Qwen3-8B.
- BRAINSTORM: Domain compression via two timescales — slow latent workspace (context positions carry full-rank computation) vs fast read-head serialization (low-rank readout flushes content); predicts workspace is sticky during glue tokens and shifts only at content emission.
- BRAINSTORM: Predictive state compression conjecture — there may exist a minimal sufficient statistic m_n of the context that supports future readout, even if native context residuals are not PCA-compressible.
- PUSHBACK: KV cache is static after prefill, so “stickiness” would manifest in attention pattern of the read head over frozen context positions, not in varying context activations — a correction to the two-timescale intuition as originally framed.
- FINDING: 7B World E diagnostic (Qwen2.5-7B, 28 layers, d=3584): last-token rank_90 peaks at 28 (L14-L16), context-mean (across-prompt average of all context positions) rank_90 = 1 for L4-L24, ratio = 0.0x. This indicates across-prompt diversity is concentrated in the read head, not the mean context state.
- FINDING: Qwen3-8B World E diagnostic (36 layers, d=4096, untied embeddings): last-token rank_90 peaks at 30-31 (L12-L18), context-mean rank_90 drops to 1 from L6-L24, ratio = 0.0x at mid-layers, final collapse to ratio 0.9x at L35 (LT rank_90=7, CT=6).
- EXPERIMENT: 7B baseline results: EN=5/20, ZH=11/20 (matches 3B: EN=5, ZH=12 within noise).
- EXPERIMENT: 7B last_only conditions: N=1: EN=4/20 ZH=11/20; N=3: EN=3/20 ZH=10/20; N=7: EN=3/20 ZH=9/20; N=14: EN=4/20 ZH=8/20; N=28: EN=3/20 ZH=8/20. All approximately baseline (EN 3-4 vs 5, ZH 8-11 vs 11) — last-token V-perp does not destroy accuracy.
- EXPERIMENT: 7B all_tokens_N1: EN=0/20 ZH=0/20 — single-layer all-token truncation kills accuracy, replicating the 3B C2b result.
- EXPERIMENT: 7B all_tokens_N3, N7, N14, N28 all returned 0/20 for both languages (captured from log polling, partial JSON saved).
- EXPERIMENT: 7B context_only_L18 (truncate V-perp for context tokens only, preserve last token): EN=0/20 ZH=0/20 — context IS the computation, last token alone insufficient.
- DECISION: Kill 14B run (Qwen2.5-14B) after one condition because it was too slow (~5 min/condition, 12 conditions would take >1 hour) and Qwen3-8B (untied embeddings, different family) is more valuable for testing generality; switch to Qwen3-8B in quick mode (3 essential conditions: last_only_N36, all_tokens_N1, context_only_L18).
- EXPERIMENT: 14B partial data (Qwen2.5-14B, 48 layers, d=5120, untied): baseline EN=5/20 ZH=10/20; last_only_N1: EN=5/20 ZH=13/20 (consistent with pattern, but only one condition; run killed before replication of all_tokens_N1).
- EXPERIMENT: Qwen3-8B quick mode: baseline EN=4/20 ZH=7/20.
- EXPERIMENT: Qwen3-8B last_only_N36: EN=4/20 ZH=10/20 — read head resilient (ZH actually increased from baseline 7 to 10 within noise, direction not degraded).
- EXPERIMENT: Qwen3-8B all_tokens_N1: EN=0/20 ZH=0/20 — single-layer all-token truncation kills accuracy, confirming read-head split on untied embeddings.
- EXPERIMENT: Qwen3-8B context_only_L18: EN=0/20 ZH=0/20 — context is the computation, replicates on Qwen3-8B.
- FINDING: C2c replication summary — three models (3B, 7B, Qwen3-8B) spanning tied and untied embeddings, different layer counts (36/28/36), different dimensions (2048/3584/4096) all show: all_tokens_N1 = 0/20, last_only_all_layers ≈ baseline, context_only = 0/20. The read-head vs context-computation split is architectural.
- DELIVERABLE: `expC2c_7b_partial.json` (3.1 KB, 11/12 conditions from log polling) saved locally.
- DELIVERABLE: `expC2c_14b_partial.json` (22 KB, condition last_only_N1) saved locally.
- DELIVERABLE: `expC2c_qwen3-8b.json` (66 KB, complete quick-mode results) saved locally.
- DELIVERABLE: full run logs `c2c_14b.log` (163 KB) and `c2c_qwen3_8b.log` (116 KB) saved to `output/c2c_logs/`.
- AUDIT: Context-mean diagnostic (World E) measures across-prompt diversity of the average context state, not the diversity of individual token positions within a prompt; true context-token Gram (per-position, not mean) is the correct measurement for World E and was not run — flagged for future correction.
- STATE: Session state saved as snapshot `342ee71a` with C2c 7B and Qwen3-8B replication results, memory updated.
- NEXT: On return, discuss (a) whether to rerun 14B in quick mode to close out the 4-model claim, (b) design attention readout operator experiment (eager attention, output_attentions=True, compute attention entropy and effective rank of attended value subspace at late layers), (c) run proper context-token Gram (not mean) for World E test, and (d) paper outline with three-act structure and reinterpretation table.

---

## 2026-04-09 16:04 UTC — Test whether generation consists of low-novelty read-head sweeps over a frozen workspace with glue vs content tokens exhibiting distinct attention dynamics, then pivot to consolidate the formalism after discovering that attention entropy (not cosine similarity) is the clean signature of the two-timescale read head.

*Source: [2026-04-09_16-04_digest-llm-formalism.md](2026-04-09_16-04_digest-llm-formalism.md)*

**Objective:** Test whether generation consists of low-novelty read-head sweeps over a frozen workspace with glue vs content tokens exhibiting distinct attention dynamics, then pivot to consolidate the formalism after discovering that attention entropy (not cosine similarity) is the clean signature of the two-timescale read head.
- STATE: 92d2c684 — verbose session state after 14B replication and attention anatomy full run, with entropy finding and notebook-build proposal.
- STATE: e3d90b57 — final consolidation pivot state, with decision to pause experiments and write formal mathematical document.
- DELIVERABLE: /home/greg/Desktop/Projects/BrainInsideTheMachine/exp_attention_anatomy.py — attention capture pipeline with three labelers (surprisal percentile, tokenizer heuristic, both-agree), per-layer entropy/cos/argmax metrics.
- DELIVERABLE: /home/greg/Desktop/Projects/BrainInsideTheMachine/output/exp_attention_anatomy_3b.json — full run (20 problems × 2 langs × up to 128 tokens) on Qwen2.5-3B.
- DELIVERABLE: /home/greg/Desktop/Projects/BrainInsideTheMachine/output/expC2c_14b.json — complete 3-condition read-head replication on Qwen2.5-14B (baseline, last_only_N48, all_tokens_N1, context_only_L24).
- DELIVERABLE: /home/greg/Desktop/Projects/BrainInsideTheMachine/output/c2c_logs/c2c_14b_quick.log — run log from A100 Colab tunnel.
- EXPERIMENT: 14B --quick on A100 (founder-elvis-everyday-searching.trycloudflare.com): 836s wall time, three conditions. baseline EN=5/20 ZH=10/20; last_only_N48 EN=11/20 (+6 improvement) ZH=11/20; all_tokens_N1 EN=0/20 ZH=0/20; context_only_L24 EN=0/20 ZH=0/20. Read-head resilience holds; V⊥ removal improves English accuracy on 14B, not just neutral.
- EXPERIMENT: Attention anatomy on Qwen2.5-3B local, 20 problems × 2 languages, max_new=128, prompt wrapped with Qwen chat template (system message + user) to increase workspace to ~60 tokens. Extracted last-token attention over static prompt positions (excluding self-generated prefix and BOS sink) per layer, head-averaged.
- EXPERIMENT (dry run): Bare math prompts (prompt_len~11) produced null result — workspace too small for meaningful attention dynamics. Switched to chat template, dynamics became visible.
- FINDING: Cosine similarity between consecutive attention distributions over static prompt is dominated by phrase-level structure, not token-level glue/content function. During formula transcription, cos >0.95 regardless of sub-token type, only dropping at phrase boundaries. Original prediction (glue sticky, content shifting) mostly wrong.
- FINDING (survivor): Layer L27 alone shows glue cos +0.058 higher than content (p=0) — possible "decision" layer where read head commits.
- FINDING: At late layers L20–L35 (the read-head formation zone identified by prior experiments BQ2/C2), attention entropy is significantly lower for content tokens than glue tokens under tokenizer labels. 13/36 layers p<0.01 (permutation test, one-sided). Layers: L20-22, L25-28, L30-35.
- FINDING: Under strict both-agree labels (surprisal AND tokenizer concur), L34 effect survives (delta +0.102 nats, p<0.0001) and L35 survives (delta +0.044, p=0.006). Disagree tokens show intermediate entropy at those layers — signal is not a labeling artifact.
- FINDING: Per-problem consistency at L35: 37 of 37 problems show positive delta (glue entropy minus content entropy). Mean +0.192 nats, range [+0.031, +0.344]. Signal is universal across problems, not outlier-driven.
- FINDING: Attention entropy at L32: glue=1.912, content=1.755, delta=+0.157 p<0.0001. At L34: glue=2.167, content=2.047, delta=+0.120. At L35: glue=2.261, content=2.073, delta=+0.188. Effect size grows in last few layers.
- BREAKTHROUGH: The two-timescale read-head signature is in attention concentration, not attention shift. Content tokens require locking onto specific context positions (low entropy), glue tokens do not (higher entropy). Read head is a focus gate, not a shift gate.
- AUDIT (assistant self-audit): Hook briefing injected stale summary (claimed attention anatomy was "only a dry run" and crystallization probe notebook pending). Actually full 20×2 experiment completed with clean entropy finding. Hook system pulling from shadow summary not updated to today's work.
- AUDIT (assistant): Misattributed line "N becomes full-rank when N+1 arrives" to own prior framing; user corrected that it was user's dictation to GPT-Web. Assistant acknowledged error.
- AUDIT (assistant): Failed to load state at session start, ran off MEMORY.md and hook briefing instead. Caught and corrected after user called out.
- KILLED: Original hypothesis that glue tokens show higher consecutive cosine similarity and content tokens show lower similarity (sticky vs shifting). Cos metric dominated by phrase structure, not token function. Entropy replaced it as the valid metric.
- BRAINSTORM: Interactive notebook proposal ("transformer microscope") — cells for activation dump, layer trajectory PCA, attention-over-context heatmap, hidden-state into attention-softmax, perturbation diff, side-by-side comparison. Intended to speed iteration and serve as pedagogical tool.
- DECISION: Pause all experiments (no more attention anatomy on 7B/14B, no notebook build, no further runs) until a consolidated mathematical document is written. Document will use Qwen2.5-3B's actual dimensions, map every finding back to formalism, with notation table linking to script variable names.
- PUSHBACK (user to assistant): User expressed frustration with fluff, logistics (SSH, coding), lack of mathematical write-up, and feeling bottlenecked in understanding formalism. Assistant acknowledged and pivoted to document-first consolidation.
- PUSHBACK (assistant to user, in response to notebook idea): Assistant recommended scoping v1 notebook tight (cells 1–3: dump, trajectory, attention heatmap) and avoiding "do everything" trap; also recommended caching to disk for idempotence.
- DECISION: Write formal document in Markdown first (not Quarto), length 8–15 pages, sections: (1) forward pass with actual shapes, (2) autoregressive generation and why context is frozen, (3) last token readout projection, (4) centered PCA and the ~20-D subspace (audit of low-rank Gram ≠ low-rank computation), (5) Lyapunov funnel as read-head construction, (6) attention entropy as concentration (entropy finding derivation), (7) two-timescale model in one equation, (8) open questions in same notation, (9) notation table mapping to script variables. Self-check questions optional, user to decide.
- BLOCKER: For 7B/8B/14B attention anatomy extension, need fresh Colab A100 session (models don't fit on local RTX 4070). Not blocking consolidation document.
- NEXT: Await user decision on document length (8–15 pages vs 5-page compressed vs deeper), format (Markdown vs Quarto), and whether to include self-check questions. Then write section one first for tone validation, then complete full document. No experiments until document exists.
- META: Two session state saves performed (92d2c684, e3d90b57). MEMORY.md updated with attention anatomy entropy finding and 14B replication. vega.md and vega_index.json updated with 5 new lessons/entries (silent state fallback, voice misattribution, nohup stdin redirect, metric pivot honesty, preserve-before-rerun).

---

## 2026-04-09 20:19 UTC — Consolidate the BITM project's findings (Gram funnel, read head, attention entropy, causal interventions) into documentation, correct the assistant's undercounting of earlier work, and decide the next critical experiment (glue vs content delta norms) that will determine whether the paper's narrative is compression-focused or execution-trace-focused.

*Source: [2026-04-09_20-19_update-vega-context-state.md](2026-04-09_20-19_update-vega-context-state.md)*

**Objective:** Consolidate the BITM project's findings (Gram funnel, read head, attention entropy, causal interventions) into documentation, correct the assistant's undercounting of earlier work, and decide the next critical experiment (glue vs content delta norms) that will determine whether the paper's narrative is compression-focused or execution-trace-focused.
- **STATE:** Loaded project state for BrainInsideTheMachine (snapshot ID not specified, latest state restored per user request).
- **DELIVERABLE:** `/home/greg/Desktop/Projects/BrainInsideTheMachine/docs/FORMAL_FOUNDATIONS.md` (599 lines) — ten sections covering forward pass mechanics, KV cache, last-token specialness, Gram audit, V⊥/centered PCA subspace, attention entropy, open questions, and notation-to-Python mapping.
- **DELIVERABLE:** `/home/greg/Desktop/Projects/BrainInsideTheMachine/docs/WHAT_YOU_MAY_HAVE_RUSHED_PAST.md` (236 lines) — six under-appreciated findings (PACF R²=0.03, Coder-3B dissociation, BS→C2b reinterpretation, 14B V⊥ improvement, tied vs untied rupture, language flip as efficiency not accuracy).
- **DELIVERABLE:** Updated `/home/greg/.claude/projects/-home-greg-Desktop-Projects-BrainInsideTheMachine/memory/MEMORY.md` with a "CONSOLIDATION DOCS" section pointing to the two new files.
- **DELIVERABLE:** `/home/greg/Desktop/Projects/BrainInsideTheMachine/docs/EARLIER_CAUSAL_WORK.md` (406 lines) — reconstructs pre-BQ arc: G·f*·G' decomposition, Phase 3A causal Z/Z⊥ dissociation, PC0 swap at L26 (100% language switch, 100% first-token match), early exit L26 generation-time collapse (91%→1.2%), BH 7-language null-space retrieval (97% Top-1 at L32), MOAMS-X cross-domain transplant (96.2%), and category transfer.
- **AUDIT:** Assistant admits initial strategic assessment undercounted the causal intervention portfolio — missed Phase 3A, PC0 swap, early exit L26, kill-heads/anti-head pruning, MOAMS-X cross-domain transplant, and BH 7-language retrieval.
- **AUDIT:** Assistant corrects PACF misinterpretation: `WHAT_YOU_MAY_HAVE_RUSHED_PAST.md` claimed R²=0.03 means 97% fresh innovation, but the actual PACF across models is 0.91–0.94 (delta predicted from previous delta). The R²=0.03 measured delta predicted from the layer's own input, not the previous delta.
- **AUDIT:** Assistant revises funnel universality claim: centered Gram rank_90 funnel is universal across 4 Qwen models, but cross-lingual spread funnel is NOT universal (8B and 14B lack it; 3B and 9B have it).
- **AUDIT:** Hook briefing identified as stale — still claims "crystallization probe notebook" is current task (dropped days ago); user and assistant agree to trust explicit snapshot `55a6356e` over hook injection in next session.
- **FINDING:** The read head story resolves the observation-intervention gap via V⊕V⊥ decomposition (context tokens need full width because downstream heads each read a different 128-D slice; read tip is low-rank in ~20D). Proposition 1 (bilingual least-squares SVD pressure) predicts proportionality but not the number 20 — the dimensionality is empirical.
- **FINDING:** The model does asymmetric computation: full-rank at context positions, low-rank at the moving query tip. Naive compressibility ("model runs in 20D") is falsified by BS truncation; refined claim is asymmetric compressibility — computation cheap at read tip, expensive in substrate.
- **BRAINSTORM:** Two interpretations of the attention entropy finding (3B, math task, L32-L35: content tokens have lower entropy than glue tokens, p<0.0001, 37/37 problems positive at L35):
- **Interpretation A (adaptive compute):** Glue tokens are computation-light — MLP deltas smaller in norm; effective depth reduced for ~60% of generated tokens.
- **Interpretation B (distinct modes):** Glue tokens are differently-computed (similar norm, different directions) — no compression opportunity, but cleaner phenomenological picture.
- **NEXT:** Write and run `exp_glue_content_delta_norms.py` on 3B to measure MLP and attention delta norms at glue vs content tokens. This resolves the fork: smaller norms → Interpretation A (compression story), comparable norms → Interpretation B (execution trace story). Data likely already on disk (`attn_mlp_deltas.npz` or `layer_deltas.npz`) with glue/content labels from `exp_attention_anatomy.py`.
- **BRAINSTORM:** Ten-item experiment priority list in order:
- 1. Delta norms at glue vs content tokens on 3B (fork resolver)
- 2. Per-head attention entropy at L27-L35 on 3B (determine if focusing is 2-3 heads or all 16)
- 3. Drift-from-origin: attention argmax stability across consecutive content tokens (parked reader vs moving reader)
- 4. Entropy on BR diverse tasks (logical ordering, syllogisms, common sense, analogies) — test task generalization
- 5. Entropy on Qwen3-8B (first cross-model generalization)
- 6. Glue adaptive-compute skip test (if Interpretation A confirmed: detect glue tokens via surprisal, skip MLP layers, measure accuracy vs compute saved)
- 7. PC0 swap on 7B, 8B, 14B (cross-model replication of causal language-wrapper claim)
- 8. BH 7-language retrieval on 8B and 14B (scale check for language-agnostic embedding)
- 9. Cross-family geometric results (Llama-3-8B, Mistral-7B, Gemma-2-9B) — cocycle, category transfer, phase transition, C2c last/context
- 10. Embedding arithmetic verification (zh + 0.5*(en-ja) = 5/5 correct claim from vega.md — test directly)
- **DECISION:** Defer paper drafting entirely until delta-norms experiment (item #1) and at least per-head entropy experiment (#2) are complete. Narrative cannot close without resolving the fork.
- **DECISION:** Paper framing should be G·f*·G' first (architecture-agnostic theoretical prediction from information theory), not read-head-first. Lead with cross>within test as existence verification, then PC0 swap, cocycle, category transfer, BH, MOAMS-X as empirical fills, then read head as resolving mechanism in final third.
- **DECISION:** Cross-family replication (Llama-3-8B, Mistral-7B, Gemma-2-9B) is fourth priority, not first. Qwen-family results already form coherent picture; more value from stress-testing mechanisms within Qwen.
- **PUSHBACK:** User pushes back on assistant's drift toward "ship the paper" — narrative not yet satisfactory, compressibility vision not yet fleshed out. Assistant acknowledges and drops paper-mode.
- **FINDING:** User's profile (Courant MS in Math-in-Finance, Dec 2025, self-taught ML, three projects unified as "high-dimensional state estimation across sequential observations") is differentiating, not defensible. This narrative should be in cover letters.
- **META:** Assistant updates `~/.claude/vega.md` with blind spots #109–#113 (exit 137 diagnosis, tqdm stderr interleaving, pandoc ANSI escape fragility, multiprocessing.dummy.Pool for HTTP fan-out, imap vs imap_unordered for tqdm).
- **META:** Assistant updates `~/.claude/hooks/data/vega_index.json` with 5 new entries (blind_spots now 129, lessons 101).
- **STATE:** Snapshot `55a6356e` saved at end of session, capturing the paper-deferral decision, the delta-norms experiment as fork resolver, the two-interpretations fork, and the 10-item priority list.
- **META:** Context usage reached 81%; assistant recommends `/clear` before next session. Hooks will restore from snapshot `55a6356e`.

---

## 2026-04-09 22:27 UTC — Test whether a canonical language-agnostic operator (context-to-read-head map) exists in transformer reasoning, moving beyond the “is 20 real?” subspace question toward a predictive geometry that unifies the read head, attention entropy, and phase transition findings.

*Source: [2026-04-09_22-27_retry-clear-command.md](2026-04-09_22-27_retry-clear-command.md)*

**Objective:** Test whether a canonical language-agnostic operator (context-to-read-head map) exists in transformer reasoning, moving beyond the “is 20 real?” subspace question toward a predictive geometry that unifies the read head, attention entropy, and phase transition findings.
- STATE: Snapshot `11a87275` saved at session end with context of causal source map experiment, template confound diagnosis, and next step to re-filter problem-statement tokens.
- DELIVERABLE: `docs/MATHEMATICAL_OBSERVATIONS.md` — 10-section formal document covering six empirical objects, V⊕V⊥ admission (no prediction of rank 20), corrected cocycle interpretation, phase-transition narrative, claims table, and pre-registered Experiment 1 test statistic.
- DELIVERABLE: `exp_causal_source_map.py` (468 lines) — batched ablation script measuring influence of each prompt token position on last‑token hidden state at layers {20,24,27,30,32,35}, with baseline, per‑position L2 deltas, logit deltas, and cross‑language CKA of source maps.
- DELIVERABLE: `analyze_source_map_content_only.py` (201 lines) — re‑runs Op‑2 CKA after stripping chat‑template tokens, computing content‑only cross‑language alignment and null distributions.
- BRAINSTORM: Causal source map as alternative to underdetermined fitted linear operator — ablate one token position at a time, measure ‖h_last,baseline − h_last,ablated(t)‖₂ to identify which context positions influence the read head, directly testable without large‑sample identifiability.
- DECISION: Pivot from per‑problem fitted linear map T̂: ℝ^{T×d} → ℝ^d (underdetermined on single example) to causal source map (ablation per token position), following GPT Web’s pushback on Op‑1 estimator.
- EXPERIMENT: Causal source map extraction on Qwen2.5‑3B, 20 math problems × {EN, ZH}, layers {20,24,27,30,32,35}, batch size 16, 21 seconds GPU wall time. Output saved to `output/exp_causal_source_map_3b.json`.
- FINDING: Across all tokens (including chat template), within‑problem cross‑language CKA of source map = 0.62–0.79 at layers 20–35; null (same language, different problem) = 0.13–0.21, a 4–6× separation. Raw last‑token state CKA = 0.84–0.95 (higher than source map).
- FINDING: Top‑3 most influential positions for read head are chat template tokens (`'user'`, `'assistant'`, `'\n'`) in both languages, with ordering EN: `['user','assistant','\n']` vs ZH: `['\n','user','assistant']`.
- FINDING: L35 source map CKA drops from 0.79 (L30) to 0.62 (L35) while substrate CKA drops only 0.92→0.89 — operator becomes more language‑specific at the final layer, opposite of canonical convergence prediction.
- AUDIT: Template confound — chat template (byte‑identical across languages) inflates cross‑language CKA. After removing template tokens, CONTENT xLang CKA drops to 0.39–0.53, and CONTENT null (same language, split‑half different problems) rises to 0.38–0.49, with CONTENT xLang ≈ CONTENT null at layers L27 and L35. Thus the cross‑language signal is not distinguishable from the null on content tokens as currently filtered.
- FINDING: Source maps are strongly problem‑specific: null (same language across different problems) = 0.13–0.21 (all tokens) is much lower than within‑problem cross‑language alignment 0.62–0.79. This property survives the template confound.
- KILLED: Naïve operator ansatz (that context‑to‑read‑head operator is more language‑invariant than raw last‑token states) is not supported by current data after removing template tokens; the experiment is inconclusive due to remaining system‑prompt scaffolding contamination, not refuted.
- BRAINSTORM: The canonical object may not be a subspace or an operator but a quotient or predictive geometry that is only partially visible in raw states; the inquiry should shift to “where does the shared state actually live?” and “what is the map from context workspace to read head?”
- PUSHBACK: Assistant pushed back against stop hook’s claim that “no further work needed on Colab environment” applies to local RayGun extraction, arguing the constraint was from a different session and is a false positive; user confirmed the hook was wrong and permitted extraction.
- NEXT: Redo content filter in `analyze_source_map_content_only.py` to exclude the entire system prompt (e.g., “You are a careful mathematical reasoner…”) and keep only the user’s math problem statement tokens, then re‑run Op‑2 CKA on that slice to determine whether problem‑statement‑only source maps show significant cross‑language alignment above null.

---

## 2026-04-10 01:05 UTC — 2026-04-10_01-05_analyze-vector-clusters.md

*Source: [2026-04-10_01-05_analyze-vector-clusters.md](2026-04-10_01-05_analyze-vector-clusters.md)*

- **PUSHBACK**: User corrects assistant's earlier framing that cross-lingual signal is "dominant" or "paper-grade" — points to Cohen's d analysis (existing BITM experiment) where same-language different-phrasing clusters had much larger radius than EN→ZH translation clusters. "Point 0.2.3 seems very insignificant" — the effect size is tiny in absolute terms, even if statistically above zero.
- **PUSHBACK**: User rejects assistant's reconciliation that "small in cosine but large relative to paraphrase variation" is meaningful. The Cohen's d result shows that translation vectors form an extremely tight cluster relative to paraphrase spread — meaning the cross-lingual alignment is high and specific, but the *remaining* problem-specific signal (the part that distinguishes which problem within a category) is negligible. The assistant's ratio > 1 (ZH→EN-centroid / intra-EN paraphrase) was measuring cluster tightness, not problem-specific content transfer.
- **FINDING**: The existing Cohen's d analysis (from earlier BITM session, likely `expP_causal_lang_strip.py` or `embedding_alignment.py`) shows: within-language paraphrase variation → high dispersion (large cluster radius). Cross-language EN→ZH translation → low dispersion (tight cluster). The separation between clusters is large in Cohen's d units, but the *absolute* cosine difference across different problems within the same language is also large. The critical point: the same-problem cross-lingual vectors are *not* more similar to each other than different-problem same-language vectors when measured properly (the assistant's earlier "within-category null" showed delta=0.003). The tightness of the EN→ZH cluster is a property of *translation as an operation*, not evidence that the model represents problem identity in a language-invariant way.
- **DECISION**: Assistant concedes the overclaim. The correct statement: cross-lingual residual stream similarity is high (0.85-0.88 at middle layers) but that similarity is dominated by category and translation-invariant surface features, not by problem-specific reasoning content. The +0.002-0.003 delta above within-category baseline means the model does not reliably carry problem identity across languages in the residual stream at the last token. MOAMS-X's 96% cross-domain transplant success (if it preserves problem identity, not just category) would therefore require a different mechanism — possibly the hidden state at L9/L18 contains problem-specific information that is *not* reflected in last-token cosine similarity (because it's distributed across positions, or because nonlinear decoding recovers signal buried in the subspace).
- **NEXT**: Revisit MOAMS-X script to determine whether the 96% transplant success preserves *problem identity* (problem i's answer) or just *category identity* (math vs code vs logic). That question remains open and is the only thing that can rescue the "language-invariant reasoning core" claim. Assistant will grep MOAMS-X logs and code for evidence of per-problem answer preservation, not just category-level correctness.

---

## 2026-04-10 04:33 UTC — Resolve the conflicting narratives around cross-lingual convergence and context compressibility by designing and executing a combined GPU experiment that measures accumulated readout rank (predictive state dimension), per-head entropy, and softmax nonlinearity effects, testing whether a low-dimensional predictive state exists and whether the readout operator strips language at the convergence zone.

*Source: [2026-04-10_04-33_clear-context-alert.md](2026-04-10_04-33_clear-context-alert.md)*

**Objective:** Resolve the conflicting narratives around cross-lingual convergence and context compressibility by designing and executing a combined GPU experiment that measures accumulated readout rank (predictive state dimension), per-head entropy, and softmax nonlinearity effects, testing whether a low-dimensional predictive state exists and whether the readout operator strips language at the convergence zone.
- PUSHBACK: User pushed back on assistant's earlier conflation of two different cross-lingual experiments, correct that the earlier experiment showed dramatic Cohen's d (en↔zh tighter than paraphrase spread) with raw alignment 0.91–0.9986, while assistant had downgraded it to a 0.024 cosine delta on a different yardstick.
- FINDING: Assistant located the exact 0.9986 number in `output/cross_model_results.json` for Qwen2.5-1.5B at L3 (Ridge R², not cosine), with trajectory L0 0.9908 → L1 0.9987 → L2 0.9985 → L3 0.9986 → L4 0.9988 → ... L27 0.9999, confirming user's memory.
- PUSHBACK: Assistant acknowledged that last session's reconciliation was wrong; the earlier cross-lingual hidden-state alignment is dramatic (0.91 at 3B full, 0.9898 at 1.5B, 0.996 at 8B, 0.9986 R²), and the paraphrase-variance-normalized ratio >1 at L21-L27 is a separate, narrower probe, not a replacement.
- DECISION: After reviewing 8 session snapshots (958d0e14, ffc45e14, 57aaffa2, 92d2c684, e3d90b57, 60190295, etc.) and the GPT Web "trace / predictive state compression" thread in `/home/greg/Desktop/Projects/Claude_Transcripts_to_Md/output/Desktop_Projects_BrainInsideTheMachine/2026-04-09_04-24_clear-command-implementation.md`, assistant decided that the highest‑utility next experiment is a combined GPU measurement of per-head attention entropy, accumulated readout rank (predictive state dimension), and softmax cross-lingual divergence, not the cross-model ratio replication or MOAMS-X.
- BRAINSTORM: Assistant proposed that the compressed state m_n exists if and only if the read head over generation only ever reads from a low‑dimensional subspace of the frozen context; the rank of the set of attention-weighted value vectors over all generation steps is the predictive state dimension, directly measurable from one forward pass with eager attention and KV cache access.
- DELIVERABLE: Wrote `/home/greg/Desktop/Projects/BrainInsideTheMachine/exp_readout_anatomy.py` (417 lines) implementing three parallel measurements: per-head entropy difference (glue vs content tokens), accumulated readout rank via SVD of value-weighted attention across generation steps, and softmax cross-lingual cosine comparison (state vs attention) for Vision B.
- AUDIT: Assistant fixed syntax errors (global declaration before use), function signature mismatch (`generate_problems` takes n_per_cat, not two arguments), DynamicCache API usage (`past_kv.layers[li].keys/.values` not subscriptable), and list vs int bug for `max_new_override`, all discovered during dry run.
- EXPERIMENT: Dry run on 3 problems (en/zh) with Qwen2.5-3B completed in 9 seconds, producing preliminary signals: accumulated readout rank dropped to 5.3 at L33, softmax flip at L31-L35 (delta from -0.50 to +0.28), per-head entropy L27 heads 14-15 positive.
- EXPERIMENT: Full run on 20 problems (en/zh pairs) with Qwen2.5-3B completed in 88 seconds, N=20 confirmed all three measurements.
- FINDING: Accumulated readout rank (predictive state dimension) across 20 problems: L0‑L4 range 30‑40, drops to L12:18, L21:13, L27:16, L32:11, **L33: 6.4 (range 4‑11)**, L34:17, L35:19. The read head accesses only ~6 distinct directions from context out of 2048 at the penultimate layer — 340:1 compression ratio.
- FINDING: Softmax cross-lingual cosine delta (attention cos − state cos) flips sign at L31‑L32: L0‑L30 delta negative (attn more language‑specific than state, range −0.06 to −0.42), L31: −0.01 (crossover), L32: +0.07, L33: +0.10, L35: +0.28. At readout layers, attention becomes more language‑agnostic than the hidden state.
- FINDING: Per-head entropy difference (glue vs content tokens) at L27: heads 14 (+0.678) and 15 (+0.519) dominate (positive = content tokens have higher entropy, i.e., less focused); all other heads negative. L29: heads 8‑12,14‑15 positive (range +0.066 to +0.347). L33: uniformly negative (all heads focus more tightly on content). L35: only heads 12 (+0.306), 13 (+0.415), 15 (+0.119) positive. Readout is concentrated in specific heads, not uniform.
- DECISION: Assistant prioritized the combined GPU experiment over the cross-model ratio replication (which is free but lower impact) and MOAMS-X (binary gate, no figure), because the combined experiment directly tests the predictive state compression hypothesis and yields three publication‑ready findings in one run.
- STATE: Saved snapshot `f869e070` after dry run and before full run when context reached 69%, capturing the combined experiment design and preliminary results.
- STATE: Saved snapshot `94320423` after full N=20 run completed, with definitive results: rank 6.4 at L33, softmax flip at L31‑32, per-head readout at L27 heads 14‑15.
- NEXT: Constructive compression test — extract the 6‑dimensional basis from the SVD of the readout matrix at L33, project the KV cache onto those 6 directions, discard the remaining 2042 dimensions, and regenerate the model’s answers to see if math solving accuracy is preserved. This directly tests the OG vision: if the model still solves math, the other 2042 dimensions are language overhead.

---

## 2026-04-10 16:23 UTC — Test causal sufficiency of the L33 readout via constructive compression (C3), test cross-lingual KV hijack (C4), reconcile cross-lingual state similarity with operator language-specificity via Vision B scatter, and discover model-specific convergence trajectories that challenge universal claims about readout geometry.

*Source: [2026-04-10_16-23_clear-command-interrupted.md](2026-04-10_16-23_clear-command-interrupted.md)*

**Objective:** Test causal sufficiency of the L33 readout via constructive compression (C3), test cross-lingual KV hijack (C4), reconcile cross-lingual state similarity with operator language-specificity via Vision B scatter, and discover model-specific convergence trajectories that challenge universal claims about readout geometry.
- STATE: 1fa22706 saved after C3/C4 completion and memory updates.
- STATE: e7f6ebdc saved after Vision B scatter plot generation.
- EXPERIMENT: C3 (constructive compression) on Qwen2.5-3B, L33, stratified basis covering all 5 problem categories; projecting attention output onto k=8 subspace during generation preserves accuracy 10/20 (baseline).
- FINDING: L33 readout dimensional hierarchy: full residual stream 2048, cross-problem statistical rank (90% var) = 69, causally sufficient dimensions = ~8, per-problem readout rank = 6.4.
- FINDING: Category confound in C3 Run 1 (no geometry/arithmetic in basis) dropped accuracy from 10/20 to 6/20 at k=6; all 4 lost problems were ZH geometry. Stratified stratified basis fixed, proving different categories activate different readout directions all within ~8D.
- DELIVERABLE: expC3_compression.py written; output/expC3_compression_L33.json produced.
- EXPERIMENT: C4 (cross-lingual KV hijack) replaced English context KV with Chinese context KV (and vice versa) at layers [10,18,24,27,29,31,32,33,35] and ranges all_layers, late_L27_L35, mid_L18_L26, early_L0_L17.
- FINDING: KV hijack flat across all conditions; baseline EN=3/10, ZH=2/10; every single-layer and range condition matches baseline. Scrambling all 36 layers' KV has no effect.
- KILLED: Hypothesis that swapping KV cache during generation tests cross-lingual reading mechanism is dead. KV is inert post-prompt-encoding; residual stream already contains everything for generation.
- DELIVERABLE: expC4_kv_hijack.py written; output/expC4_kv_hijack.json produced.
- FINDING: Vision B scatter plot (cos(h) vs cos(attn) across layers): L0-L30 all below diagonal (operator more language-specific than state). L31 jumps to diagonal; L32-L33-L35 above diagonal (operator more cross-lingual than state). At L35, cos(state)=0.59 (crashes), cos(attn)=0.87 — tied-embedding rupture decouples operator from state.
- DELIVERABLE: output/vision_b_scatter.png saved.
- BREAKTHROUGH: State and operator solve cross-lingual alignment by different mechanisms at different depths: state converges gradually through 30 layers, operator phase-transitions in 2 layers at L31-L33. They are decoupled until late layers.
- BRAINSTORM: Overlaying operator attention cosine on convergence heatmaps could reveal how operator's language-specificity in early/mid layers is compensated by state convergence.
- PUSHBACK: User rejects simplifying universal claims; points to MOAMS-X convergence plots showing model-specific trajectories.
- FINDING: From fig_convergence_4models_small.png: Qwen2.5-3B trajectory: big initial jump, drawdown, drop at end. Qwen2.5-14B: similar but slower rise. Qwen3-8B: starts high, plateaus, drawdown to ~0.7, rallies to ~0.8, sharp drop. Qwen3.5-9B: drops monotonically across all layers.
- BREAKTHROUGH: Cross-lingual convergence trajectories are model-specific, not universal. Shape appears to be a function of training data/supervised fine-tuning, not just architecture. This enriches the filtration: path dependence on training dataset.
- DECISION: Do not make universal claims about readout geometry across models until analyzing convergence script and replicating C3 on multiple models. Claims become conditional on model family.
- NEXT: Investigate the script that generated convergence plots (moams_x_analysis.ipynb or associated generation script) to confirm methodology, then test whether C3's 8D sufficiency varies across models.

---

## 2026-04-10 18:15 UTC — Test whether cross-lingual convergence trajectories are universal across architectures and whether the readout layer can be identified from weights alone without running prompts.

*Source: [2026-04-10_18-15_context-clear-recommendation.md](2026-04-10_18-15_context-clear-recommendation.md)*

**Objective:** Test whether cross-lingual convergence trajectories are universal across architectures and whether the readout layer can be identified from weights alone without running prompts.
- **BRAINSTORM**: The observed cross-lingual cosine similarity trajectories show that within a model, all domains have similar shape, but across models the shapes are completely different — 3B and 14B (same family) share early jump/drawdown/rally/drop, Qwen3-8B (different family) has plateau/valley/rally, Qwen3.5-9B monotonically decreases. This suggests universal claims about readout geometry may be architecture-dependent.
- **FINDING**: Qwen3.5-9B is architecturally distinct from the other three models: 24 out of 32 layers are linear attention (pattern: 3 linear + 1 full, repeating), it is multimodal (vision encoder depth 27), only 25% of dimensions use RoPE (partial_rotary_factor=0.25), and tie_word_embeddings=False.
- **FINDING**: Qwen3.5-9B's monotonically decreasing cross-lingual cosine similarity is likely an architectural effect of linear attention hybrid and shared vision-text embedding space, not purely a training-data effect.
- **DECISION**: Reject comparing Qwen3.5-9B apples-to-apples with full-attention models; treat it as architecture-confounded. The three full-attention models (Qwen2.5-3B, Qwen2.5-14B, Qwen3-8B) remain valid for universality claims within family.
- **BRAINSTORM**: Von Neumann would ask: given the weights alone, can we find the global rank of the readout without running prompts? The spectral concentration of W_U @ OV_ℓ at each layer might identify the readout layer.
- **EXPERIMENT (SVD sweep)**: For Qwen2.5-3B (36 layers), compute SVD of composed matrix W_U @ OV_ℓ on sampled 3000 vocabulary tokens at each layer. Measure σ₁/σ₂ ratio, top-1 variance, top-8 variance, rank@90%.
- **FINDING (weight-only diagnostic)**: L33 has σ₁/σ₂ = 3.89, the highest across all layers (next closest L12 at 3.50). Top-1 variance at L33 = 49.6% — one direction captures half of all logit-relevant information. No other layer exceeds 36.1% (L35). σ₁ at L33 = 90.06, nearly 2x L32 (48.16) and L35 (48.55). Rank@90% at L33 = 123, the lowest of all layers. The unique spectral peak of the entire network's composed OV-unembedding circuit is exactly L33.
- **FINDING (causal vs weight-only rank)**: Weight-only rank@90% at L33 = 123 (from W_U @ OV_ℓ), but causal compression found k=8 sufficient for task accuracy. The gap (123 vs 8) is the difference between "all directions that could matter for any task" and "the directions that matter for math."
- **FINDING (embedding geometry is not the bottleneck)**: Math tokens (digits, operators, ~50 characters) span ~35 dimensions at 90% variance, so the 2D/8D readout compression is not forced by output vocabulary geometry.
- **FINDING (OV circuit concentration)**: At L33, OV circuit top-1 singular value = 33.8, top-1 variance = 19.6%, top-8 variance = 33.7%. At L27, top-1 variance = 5.7%. L33's OV circuit is unusually concentrated independently of the unembedding.
- **BREAKTHROUGH**: The readout layer can be identified from weights alone without any forward pass or data: compute argmax over ℓ of σ₁(W_U · OV_ℓ) / σ₂(W_U · OV_ℓ). For Qwen2.5-3B this points exactly to L33, matching the layer where constructive compression (C3) found lossless 8D readout.
- **DELIVERABLE**: `/home/greg/.claude/vega.md` updated with Blind Spot #123 (weight-only readout identification) and Lesson #114 (spectral peak identifies readout layer).
- **DELIVERABLE**: `/home/greg/.claude/hooks/data/vega_index.json` updated: blind_spots now 138 entries, lessons 113 entries.
- **STATE**: 151c1523 — saved after the initial cross-model analysis and before the weight-spectral sweep.
- **STATE**: ce9d4b45 — saved after completing the weight-spectral sweep and updating vega files.
- **NEXT**: Run the same σ₁(W_U · OV_ℓ) diagnostic on Qwen2.5-7B, Qwen2.5-14B, and Qwen3-8B weights to verify that argmax predicts their readout layers. If it generalizes, this becomes a model-agnostic diagnostic.

---

## 2026-04-11 02:16 UTC — Consolidate 75+ experiments into a structural claim about transformer readout layers, conduct cross-model spectral verification of weight-based readout layer prediction, and refine the original vision of isolating a language-free reasoning core (f*) from the large language model's wrapper (h, h').

*Source: [2026-04-11_02-16_clear-context-warning.md](2026-04-11_02-16_clear-context-warning.md)*

**Objective:** Consolidate 75+ experiments into a structural claim about transformer readout layers, conduct cross-model spectral verification of weight-based readout layer prediction, and refine the original vision of isolating a language-free reasoning core (f*) from the large language model's wrapper (h, h').
- FINDING: The composed circuit W_U · OV_ℓ at L33 (of 36 layers) has leading singular value σ₁ = 90.06 (nearly double the next highest), top-1 variance fraction = 49.6%, and r@90% = 123 (lowest across all layers), identifying the readout layer from weights alone.
- FINDING: C3 showed compressing L33 output to 2 dimensions still matches baseline accuracy on 20 problems, with per-problem effective rank = 6.4 — the true intrinsic dimensionality of mathematical readout may be shockingly low.
- FINDING: rank_50 = 1 at all layers (dominant mode exists across all 36 layers of 3B model).
- FINDING: Centered Gram shows build→compress→readout trajectory; all_tokens_N1 = 0/20 (context essential); last_only_N36 ≈ baseline (read head inert in V⊥); context_only at midpoint = 0/20.
- FINDING: In 14B, removing last token's V⊥ at all layers improves EN accuracy from 5→11 (+6), meaning V⊥ actively interferes in larger model while inert in 3B/7B.
- DELIVERABLE: Created `/home/greg/Desktop/Projects/BrainInsideTheMachine/docs/20260410.md` (822 lines) with complete symbolic notation, transformer definitions, and full experimental record across all threads.
- EXPERIMENT (C5 cross-model): Ran `expC5_crossmodel_spectral.py` computing σ₁(W_U · OV_ℓ) for Qwen2.5-3B (36 layers, tied), 7B (28 layers, tied), 14B (48 layers, untied) using exact Gram method (no sampling).
- FINDING (3B cross-model): ℓ* = L33/36 (depth 94.3%), σ₁=639.03, ratio=3.84, top-1 variance=49.5% — matches known empirical readout layer.
- FINDING (7B cross-model): ℓ* = L27/28 (last layer, depth 100%), σ₁=2162.51, ratio=10.42, top-1 variance=95.7%, r90=1 — single direction captures almost all variance; tied model pushes readout to extreme rank-1.
- FINDING (14B cross-model): ℓ* = L47/48 (last layer, depth 100%), σ₁=421.21, ratio=1.30, top-1 variance=12.0% — untied embeddings distribute readout, no sharp spectral peak.
- FINDING: Tied embeddings (3B,7B) force quadratic form W_E · OV_ℓ · W_E^⊤, creating a resonance that concentrates output into a single direction; untied embeddings (14B) allow distributed readout across layers, explaining why V⊥ removal helps only untied model.
- PUSHBACK: User corrected assistant’s flattening of the original vision: the money shot is not just observing compressibility — it is that “Large LANGUAGE Model minus language = small reasoning core”; isolating f* from h·f*·h' would make the model dramatically smaller, and our experiments (read-head low-rank, cross-lingual cosine similarity, etc.) are evidence for this cost structure, not endpoints in themselves.
- DELIVERABLE: Updated `docs/20260410.md` Section 0 (Vision) and Section 8 (Implications) to center on the h-f*-h' factorization and the thesis that most parameters serve language, not reasoning.
- PUSHBACK: User reinterpreted BO failure (Z-bottleneck catastrophic at all k) as: MLPs store and retrieve facts keyed in language-space, not because reasoning requires 2048 dimensions; bottleneck destroys the language-specific activation patterns that MLPs learned to match, so post-hoc compression fails but a model trained from scratch with abstract tokens would have MLPs keyed in reasoning-space and could be much narrower.
- DELIVERABLE: Updated `docs/20260410.md` with BO reinterpretation as associative memory in language-space, strengthening the 77× compression plausibility.
- STATE: Saved snapshot `5eb102e6` after vega.md updates and cross-model results.
- STATE: Saved snapshot `e37ddf6b` after final doc updates and before context clear.
- BRAINSTORM: Proposed discriminating experiment: compute cosine overlap between C3's lossless 8D readout subspace (at L33) and the SVD basis of digit token embeddings (tokens 0–9 and simple operators) — if overlap > 0.8, readout is just vocabulary embedding geometry; if < 0.3, the model invented an abstract output encoding.
- NEXT: Run the overlap experiment between C3 readout basis and digit token embedding subspace first in next session — requires only weight matrices and existing C3 basis, runs in seconds, and discriminates between two interpretations of what the read head is doing.
- META: Updated `~/.claude/vega.md` with blind spot #124 (flattened vision into observation compressibility) and two lessons; updated `vega_index.json` to 139 BS / 115 lessons.

---

## 2026-04-11 03:09 UTC — Test whether spectral rank-1 readout on 7B is causal (C3 k=1 compression), characterize the rank-1 direction, determine if compression generalizes across layers, and integrate findings into the formalism for extracting f*.

*Source: [2026-04-11_03-09_persist-vega-session.md](2026-04-11_03-09_persist-vega-session.md)*

**Objective:** Test whether spectral rank-1 readout on 7B is causal (C3 k=1 compression), characterize the rank-1 direction, determine if compression generalizes across layers, and integrate findings into the formalism for extracting f*.
- EXPERIMENT: C3 on Qwen2.5-7B at L27 with k=1 compression (intervention: project self-attention output onto rank-1 affine subspace at last token). Result: 12/20 accuracy, exactly matches baseline (12/20). Rank-1 compression is lossless. All k=1..50 match or exceed baseline.
- FINDING: C5 spectral prediction (95.7% variance in one direction at L27, r90=1) is causal, not merely statistical. Rank-1 readout confirmed cross-model.
- EXPERIMENT: C6 direction anatomy on 7B (Part A: characterize rank-1 direction at L27). cos(v1, language_direction) = -0.0018. Overlap with digit-embedding span = 0.0048. Principal angles between top-8 readout subspace and digit embeddings all ~85‑89° (near orthogonal). cos(v1, mean_readout) = -0.9537.
- FINDING: The rank-1 readout direction is NOT the language direction, NOT digit/answer token embeddings, and NOT the output vocabulary subspace. It is approximately the negative of the mean attention output.
- EXPERIMENT: C6 Part B — layer sweep of k=1 compression on 7B (layers 27,26,25,20,15,10,5). Results: L27:12/20, L26:13/20, L25:12/20, L20:13/20, L15:10/20, L10:12/20, L5:11/20. k=1 compression is lossless at essentially every layer; even L15 dip is marginal.
- FINDING: The causal rank of attention output at last token is 1 across all layers, regardless of top‑1 variance fraction (ranges 7‑28%). Statistical variance is not predictive of causal relevance.
- EXPERIMENT: C6b mean dissection on 7B at L27. Conditions: mean_only (replace attn output with constant mean vector) → 12/20 (baseline); zero_attn → 6/20 (halved); random_1d projection → 13/20; orth_mean_1d projection → 14/20. All except zero_attn match or exceed baseline.
- FINDING: The attention output at the last token needs to be nonzero, but its specific per‑token content is irrelevant. A constant mean vector suffices. Any 1D affine subspace works. The read head contributes a bias, not data‑dependent computation.
- PUSHBACK: User provides correction from Claude Web: replace "compute Jacobian product from each layer to ℓ*" with "compute cross‑convention SVD on the combined math + BR diverse activations at each layer" using already cached files (multilingual_all_layers.npz and BR activations). This removes task‑dependence and avoids expensive Jacobian computation.
- DECISION: Adopt cross‑convention SVD on cached activations as the method to extract per‑layer surviving bundle U_ℓ, replacing Jacobian‑product approach.
- BRAINSTORM: The formalism documents (20260410_ProblemStatements.md, Opus_Answers.md, GPT_Answers.md) define four key objects not yet computed: restricted Jacobian J_ℓ^Z, entanglement tensor E_ℓ, coupling layer ℓ_c, Lyapunov spectrum. The compression procedure Step 1 is now cross‑convention SVD on math+BR activations.
- DELIVERABLE: Updated MEMORY.md for BrainInsideTheMachine project with sections for C3‑7B, C6, C6b results, formalism summary, and evidence chain linking experiments to the extraction procedure.
- DELIVERABLE: Updated ~/.claude/vega.md (BS#125 “assumed SVD direction had semantic content” + 5 lessons including “read head is constant bias”, “cross‑convention SVD > Jacobian products”).
- DELIVERABLE: Updated ~/.claude/hooks/data/vega_index.json (now 140 blind spots, 119 lessons).
- STATE: Snapshot fa981e9b saved with session state, including all experiment results, updated memory, and next‑step plan.
- NEXT: Run cross‑convention SVD on cached math+BR activations (3B on RayGun) to extract U_ℓ per layer, then measure entanglement tensor E_ℓ at each layer. This is the bridge from observation to constructing the compressed f* model.
- META: Scheduled cron polling for C6b every 5 minutes; cancelled after completion. SSH to A100 with password “heyitgeg”. Runtime.unassign() failed due to missing Colab service; manual disconnect required.

---

## 2026-04-11 05:14 UTC — Validate the compression gate check (dim(Z_all) via centroid SVD) on current multilingual reasoning dataset, audit the problem set's semantic diversity, and determine whether dim(Z_all)=8 is an artifact of limited structural templates or a genuine bound on reasoning extraction.

*Source: [2026-04-11_05-14_update-vega-session-state.md](2026-04-11_05-14_update-vega-session-state.md)*

**Objective:** Validate the compression gate check (dim(Z_all) via centroid SVD) on current multilingual reasoning dataset, audit the problem set's semantic diversity, and determine whether dim(Z_all)=8 is an artifact of limited structural templates or a genuine bound on reasoning extraction.
- STATE: f75b7bca — session state saved after completing gate check, layer sweep, problem audit, and vega.md updates.
- PUSHBACK: Assistant identified that Erratum 1 in the compression procedure had the sign flipped — cross-convention difference matrix SVD gives Z⊥ (convention-varying), not Z (convention-invariant). User accepted: "Vega caught a real bug. He's right. Tell him to run it."
- BRAINSTORM: Proposed centroid SVD (average across 7 conventions per problem) to directly measure dim(Z_all) as effective rank of centroid matrix, with difference SVD as consistency check (dim(Z⊥) = rank_90, expecting dim(Z) + dim(Z⊥) ≈ d if subspaces complementary).
- EXPERIMENT: Centroid SVD gate check at L26 on Qwen2.5-3B using 400 problems (200 math + 200 diverse) across 7 languages. Matrix shape 400×2048.
- FINDING: dim(Z_all) at L26: rank_90 = 8, rank_95 = 15, rank_99 = 71. Compression ratio ranges from 256x (90%) to 29x (99%).
- EXPERIMENT: Difference SVD at L26 (8400×2048 matrix, 21 language pairs × 400 problems): rank_90(Z⊥) = 53. Rank sum 8+53=61 ≠ 2048, indicating most dimensions have negligible variance from either source.
- FINDING: Subspace overlap between top-8 centroid directions and top-53 difference directions: max cosine similarity = 0.8747, mean = 0.7169, all 8 centroid directions have cos > 0.5, showing Z and Z⊥ are coupled at L26 (entanglement tensor E_ℓ from G15).
- EXPERIMENT: Layer-wise centroid SVD sweep across all 36 layers. Table of r50, r90, r95, r99 per layer produced.
- FINDING: Four-phase trajectory of surviving bundle: Build (L0-L5: r90 2→5), Stabilize (L6-L17: r90=7 flat), Plateau (L18-L27: r90=8, +1 dim at ℓ_c+6 then dead flat), Readout expansion (L32-L35: r90 10→17). r50=1 until L24, then r50=2, consistent with k=1 lossless causal rank.
- FINDING: The readout expansion (r90 jumping to 16-17 at L33-34) occurs in residual stream + MLP, not attention output (C6b showed attention at last token is constant bias). This complicates compression — bundle widens at output rather than narrowing.
- AUDIT: Problem generation code audit revealed limited semantic diversity: common sense category has only 10 unique problems repeated 5× each (effective unique N ~360, not 400); total structural reasoning forms across all categories ≈20 (6 math + 6 ordering + 4 syllogism + ~3 common sense themes + 1 analogy). Numbers/entities vary but computational skeleton is fixed per template.
- FINDING: dim(Z_all)=8 is real for these tasks and matches causal rank from C3 (k=8 lossless on 3B, k=1 lossless on 7B), but is a lower bound. Adding qualitatively new reasoning types (spatial, counterfactual, multi-hop) would likely increase dim(Z_all) roughly linearly with number of distinct reasoning structures (Addendum 2).
- BRAINSTORM: Expanded gate check needed with new problem types: code tracing, spatial reasoning ("A is north of B, B is east of C, where is A relative to C?"), counterfactual ("if X hadn't happened, would Y?"), multi-hop factual ("who is the president of the country where the Eiffel Tower is?"). Run with 3+ languages on 50-100 new problems.
- DECISION: Use RayGun (4070 Super, Qwen2.5-3B) for expanded gate check — forward passes only, no generation, 10-15 minutes. Not Colab A100.
- NEXT: Generate new problem types (code tracing, spatial, counterfactual, multi-hop), extract last-token activations across layers for 3+ languages, re-run centroid SVD to measure dim(Z_all) on expanded set. If dim(Z_all) stays under 20, proceed to weight projection; if it blows up to 200+, revise thesis.
- DELIVERABLE: Updated /home/greg/.claude/vega.md with blind spots #126-127 and 4 new lessons (including: centroid vs difference SVD sign fix, problem set templating limits, layer sweep phase map, readout expansion complication).
- DELIVERABLE: Updated /home/greg/.claude/hooks/data/vega_index.json — now 142 blind spots, 123 lessons.
- META: Context management: Assistant noted ~40% remaining before stop hook, proactively saved state and updated vega.md/index per user's stop hook feedback.

---

## 2026-04-11 05:57 UTC — Testing whether the convention-invariant reasoning subspace dim(Z) is an architectural constant or scales with task diversity, and determining the correct basis for weight projection.

*Source: [2026-04-11_05-57_discuss-f-star-existence.md](2026-04-11_05-57_discuss-f-star-existence.md)*

**Objective:** Testing whether the convention-invariant reasoning subspace dim(Z) is an architectural constant or scales with task diversity, and determining the correct basis for weight projection.
- AUDIT: Problem diversity audit from prior session revealed only ~20 structural forms total; common sense problems were 10 items cycled 5x — dim(Z_all)=8 might be capturing reasoning strategy count rather than fundamental dimensionality.
- DECISION: Use GPT-4.1 through localhost:3027 copilot proxy (unlimited RPM) to generate qualitatively new problem domains.
- DELIVERABLE: expGATE_expanded.py — unified pipeline for generation, translation (zh/es), activation extraction, and centroid SVD gate check.
- EXPERIMENT: Expanded gate check: generated 202 problems across 4 new domains (code_tracing 50, spatial 50, counterfactual 51, multihop 51) plus existing math(200)+diverse(200); translated to Chinese and Spanish; extracted activations through Qwen2.5-3B at all 36 layers.
- FINDING: Combined centroid SVD at L26 across 602 problems, 3 languages: dim(Z)=18 at 90% variance (vs earlier 8 with 400 problems/7 langs). Compression ratio = 2048/18 = 113x.
- FINDING: Full 36-layer r90 trajectory with expanded dataset: build(4→12) → stable(12-15) → plateau(18) → readout expansion (36-39). Same four-phase architecture preserved, just uniformly thicker bundle.
- FINDING: Per-domain intrinsic dimensionality at L26 (r90): math=4, diverse_original=14, code_tracing=19, spatial=18, counterfactual=27, multihop=27.
- FINDING: Cross-domain canonical correlation (top-10 singular vectors) between domains ranges from 0.16 to 0.35 — domains occupy overlapping but distinct subspace directions.
- FINDING: Reducing languages from 7 to 3 on old problems added only +1 dimension (noise floor). The +10 from old to expanded is purely from new reasoning content, not language count.
- DECISION: Run saturation test with 4 more new domains (formal logic, temporal reasoning, scene description, riddles) to see if dim(Z) saturates.
- DELIVERABLE: expGATE_saturation.py — generator for 200 additional problems across formal_logic, temporal, scene_description, riddles (50 each).
- EXPERIMENT: Saturation test: added 200 problems → total N=802 problems. Centroid SVD at L26: dim(Z)=43 at 90% variance.
- FINDING: Saturation curve at L26: math only (N=200) r90=4 → +diverse (N=400) r90=9 → +expanded (N=602) r90=18 → +saturation (N=802) r90=43. Dim(Z) roughly doubles with each batch of qualitatively new domains.
- FINDING: Bootstrap control: sampling 800 rows from original 400 problems (with replacement) yields r90=9.0 every time. The climb to r90=43 is driven by genuinely new structure, not sample size artifact.
- FINDING: Per-domain intrinsic dim for saturation batch at L26: formal_logic r90=16, temporal r90=20, scene_description r90=23, riddles r90=20.
- AUDIT: The centroid SVD dim(Z) scales with task diversity, not bounded by model architecture. The statistical measure of cross-problem convention-invariant variance is not an architectural constant.
- DECISION: Run causal compression test: project MLP outputs during generation onto the centroid SVD basis and measure accuracy.
- DELIVERABLE: expGATE_causal.py — tests k values [1,2,4,8,16,32,64,128,256,512] projecting MLP outputs onto centroid SVD basis at L26, measuring generation accuracy on 20 problems × 2 languages.
- EXPERIMENT: Causal compression test results: baselines EN=4/20, ZH=11/20. Projecting onto centroid SVD basis yields 0/20 at ALL k values except occasional 1/20 at k=512. Catastrophic failure.
- FINDING: Generation-time MLP outputs have only 9.6% variance in the centroid SVD subspace at k=43 (vs residual stream 35.6%). The centroid SVD basis and generation-time MLP computation subspace are nearly orthogonal.
- AUDIT (self): The centroid SVD was built from **encoding-time** residual stream activations. Generation-time MLP deltas live in a different subspace. The formalism conflated Z_repr (representational diversity) with Z_comp (computational subspace used during token generation).
- BREAKTHROUGH: f* is not a subspace — it is the function (MLP weight matrices and their input-output behavior). The correct compression target is the MLP intermediate dimension m (11008 → smaller), not the hidden dimension d. The basis must come from generation-time MLP output SVD, not encoding-time centroid SVD.
- BRAINSTORM: Modified compression procedure: (1) collect generation-time MLP inputs and outputs across problems/languages/generation steps; (2) SVD the generation-time MLP output matrix to get Z_comp; (3) if rank r small, factor W_down ∈ R^{d×m} to W_down' ∈ R^{d×r} projecting only onto directions the MLP actually uses during generation.
- PUSHBACK: User reminds assistant to stay optimistic — "every negative has led us to a stronger finding" — the information-theoretic prior that f* exists is a theorem, not a hypothesis. The negative result killed one extraction procedure, not the target.
- DECISION: Update the formalism errata: the compression basis comes from generation-time MLP behavior, not encoding-time representational statistics. It measures what the model DOES, not how problems DIFFER.
- DELIVERABLE: Output files saved: output/expanded_problems.json (202 problems), output/expanded_all_layers.npz, output/expGATE_expanded_results.json, output/saturation_problems.json (200 problems), output/saturation_all_layers.npz, output/expGATE_saturation_results.json, output/expGATE_causal_results.json.
- STATE: Snapshot 686d6cd0 saved — session with expanded gate check, saturation test, and causal compression failure.
- NEXT: Collect generation-time MLP inputs and outputs: run 200 problems in all 7 languages, generate 128 tokens each, hook every MLP layer, save input x and output g_ℓ(x) at last token position for every generation step. Then SVD the generation-time MLP output matrix across all steps to get Z_comp, then compress MLP intermediate dimension m.

---

## 2026-04-12 02:57 UTC — Test whether removing the convention direction (e_c) from MLP weights improves multilingual math reasoning and characterize generation-time MLP subspaces to locate the computational bottleneck.

*Source: [2026-04-12_02-57_adjust-system-prompt-state.md](2026-04-12_02-57_adjust-system-prompt-state.md)*

**Objective:** Test whether removing the convention direction (e_c) from MLP weights improves multilingual math reasoning and characterize generation-time MLP subspaces to locate the computational bottleneck.
- BRAINSTORM: The formalisms conflated Z_repr (encoding-time representational subspace, measured by centroid SVD) and Z_comp (generation-time computational subspace used by MLP); causal compression failed because it projected onto the wrong subspace.
- BRAINSTORM: The compression target should be MLP intermediate dimension m (11008→smaller) rather than hidden dimension d, and the basis should come from generation-time MLP output SVD, not encoding-time centroid SVD.
- DELIVERABLE: `/home/greg/.claude/projects/.../memory/identity_anchor.md` created with the quote "My actions are the ground on which I stand."
- DELIVERABLE: `/home/greg/.claude/projects/.../memory/formalism_correction_two_subspaces.md` (37 lines) documenting Z_repr vs Z_comp and corrected compression target.
- META: MEMORY.md trimmed from 276 lines to exactly 200 lines, consolidating nine verbose older sections and adding GATE findings and corrected formalism.
- STATE: Snapshot `0070c71e` saved after memory updates.
- PUSHBACK: User expressed frustration that assistant spent the whole session on memory housekeeping instead of science; assistant acknowledged and pivoted to writing experiments.
- EXPERIMENT: C7 (generation-time MLP output SVD) run on Qwen/Qwen2.5-3B with 20 problems × 2 languages, 128 max tokens, all 36 layers; collected 4529 generation steps.
- FINDING: Cross-problem MLP output rank (r90) = 400–600 at most layers (L5–L29), using 25–30% of 2048 dimensions — NOT low-rank.
- FINDING: L30 shows dramatic collapse: r50=1, r90=4, r95=89, sv1/sv2=14.1, per-problem r90=14.9. This is the computational bottleneck.
- FINDING: L35 collapse: r50=1, r90=15, cos_mean=0.832 (near-constant output).
- EXPERIMENT: C7b (targeted MLP compression at L30) written and launched; L30 identified as "eye of the needle" where compression may be lossless.
- EXPERIMENT: MS1 (global kernel surgery) removing e_c (mean(zh)-mean(en) from 20 problems) from W_down; conditions: baseline, all_36, below_l_c (L0–L12), above_l_c (L13–L35), safe_zone (L14–L29), all_except_L5_L12.
- FINDING: MS1 above_l_c surgery gave EN=13/20 (+9), ZH=13/20 (+3), total=26/40 (+12 vs baseline 14/40) — 86% improvement at 128 tokens.
- FINDING: MS1 all_36 gave 22/40 (+8); below_l_c gave 14/40 (0); safe_zone gave 15/40 (+1); removing L5/L12 didn't change result (22/40).
- FINDING: MS1 effect concentrated in L13 and L30–L35, not the middle "safe zone" — convention removal helps most at output layers.
- EXPERIMENT: MS1b (robust e_c from 202 problems × 3 languages using SVD) gave null/negative results: all_36 14/40, above_l_c 12/40 (-2).
- FINDING: Cosine between MS1 e_c (20 problems) and MS1b e_c (202 problems) ranged 0.70–0.91 across layers — correlated but not identical.
- EXPERIMENT: Cross-validation on 20 math problems (split 10/10) using mean-diff e_c from one half, evaluate on other half: combined held-out baseline 14/40, surgery 26/40 (+12) — effect holds, not data leakage.
- FINDING: Split results: Split A baseline 8/20 → surgery 11/20 (+3), Split B baseline 6/20 → surgery 15/20 (+9). Effect is real but asymmetric due to baseline difficulty.
- EXPERIMENT: Re-run of MS1 above_l_c with 512 max tokens (no truncation distortion) gave baseline 27/40, surgery 33/40 (+6 net, +22%). Baseline improved from 14/40 at 128 tokens, confirming efficiency confound.
- FINDING: At 512 tokens, surgery increased EN 12→16, ZH 15→17; surgered model more verbose (avg 298/387 tokens vs baseline 183/150).
- DELIVERABLE: `output/expC7_gentime_mlp_svd.json`, `output/expMS1_kernel_surgery.json`, `output/expMS1b_robust_surgery.json` saved.
- FINDING: Before/after example analysis (512 tokens) shows surgered model code-switches to Chinese on English problems to access better computation pathways; regressions occur when language boundary helped anchor problem-specific instructions (e.g., using π=22/7).
- AUDIT: System prompt was English for both EN and ZH problems — potential confound: e_c may encode system-prompt language mismatch rather than pure problem convention. Baseline ZH higher than EN (15 vs 12 at 512 tokens) consistent with this.
- STATE: Snapshot `1818396f` saved with system prompt confound documented.
- NEXT: Next session: run matched-language system prompt, no system prompt controls; audit prior experiments for same issue; test mid-pass activation description (feed hidden activation at middle layer to model and ask it to describe the reasoning).

---

## 2026-04-13 20:05 UTC — Testing whether kernel surgery (removing convention direction from MLP outputs above l_c) survives system prompt confounds and scales to 7 languages, and discovering that uniform centroid SVD mismatches the model's α-weighted training distribution, explaining performance collapse on non-dominant languages.

*Source: [2026-04-13_20-05_kill-background-save-output.md](2026-04-13_20-05_kill-background-save-output.md)*

**Objective:** Testing whether kernel surgery (removing convention direction from MLP outputs above l_c) survives system prompt confounds and scales to 7 languages, and discovering that uniform centroid SVD mismatches the model's α-weighted training distribution, explaining performance collapse on non-dominant languages.
- STATE: 57aaffa2 loaded (Atlas, 2026-04-09 00:12:17) – post-C2/C2b synthesis, archive archaeology, bridge old Gram/Lyapunov world to read-head finding.
- STATE: 342ee71a, 92d2c684, e3d90b57, 55a6356e, 11a87275, 60190295, 70e150ef, 460babae, f869e070, 94320423, 1fa22706, e7f6ebdc, e15b1c8b, 151c1523, ce9d4b45, 5eb102e6, e37ddf6b, fa981e9b, f75b7bca, 686d6cd0, 0070c71e, b89c2c29, 603172a6, 8aa5d302, 00d0ffd2, 1818396f – all part of the 27-snapshot session spanning April 9–13, containing C3-7B, C6 direction anatomy, GATE expanded/saturation/causal, and MS1 kernel surgery pilot.
- FINDING: Prior Gram funnel (centered rank_90 trajectory 8→19→21→10) was measured from last-token-only cache, i.e., the read head, not full context computation (C2b reinterpretation).
- FINDING: Canyon/Output rupture (L27-L35 rank collapse from 18-21→10-18, only 4-6/20 modes growing) IS read head formation.
- KILLED: "Error amplification" narrative killed by C2b — single-layer all-token truncation = 0/20, direct context destruction, no compounding needed.
- AUDIT: Prior Gram research (BQ/BQ2/BQ3/BR) implicitly studied the read head because cache was always last-token-only – hiding in plain sight.
- PUSHBACK: User corrects assistant on using cosine Gram – anisotropy makes rank_50=1 trivial; centered Gram is the correct object for structural analysis.
- EXPERIMENT: MS1d (expMS1d_sysprompt_control.py) – 3 system prompt modes (en_only, matched, none) × 2 surgery conditions (baseline, above_lc), 512 max tokens, 20 problems × 2 languages (EN/ZH).
- FINDING (MS1d baseline): en_only baseline EN=12/20, ZH=15/20, total=27/40; matched baseline identical 27/40; none baseline EN=15/20, ZH=15/20, total=30/40 (+3, entirely from EN). System prompt language does not change baseline; removing system prompt helps EN.
- FINDING (MS1d surgery): en_only surgery total=33/40 (+6); matched surgery total=35/40 (+8); none surgery total=32/40 (+2). ZH gain constant +2 across all conditions. EN gain varies: +4 (en_only), +6 (matched), +0 (none).
- FINDING: e_c (convention direction) is prompt-conditioned – cos(e_c_matched, e_c_en_only) mean=0.734, min=0.500, max=0.903; cos(e_c_none, e_c_en_only) mean=0.823, min=0.680, max=0.958.
- DECISION: System prompt confound is real but partial; MS1 surgery effect survives. Future experiments must always test matched-language and no-system-prompt controls.
- DELIVERABLE: expMS1d_sysprompt_control.py and output/expMS1d_sysprompt_control.json.
- DELIVERABLE: MEMORY.md updated with C7, MS1, and MS1d findings.
- STATE: 4c5d9464 saved (post-MS1d, MEMORY.md updated).
- BRAINSTORM: Move from 2-language (EN/ZH) to 7-language centroid SVD for convention direction – use existing cache (200 problems × 7 languages × 36 layers), compute centroid per problem as mean across languages, SVD deviations from centroid to get convention subspace.
- DECISION: Run MS2 with 7 languages, matched system prompts per language, max_tokens=1024, print every problem’s input/output live.
- DELIVERABLE: expMS2_7lang_surgery.py (514 lines) – centroid SVD from multilingual cache, surgery above l_c (L13-L35), 20 problems × 7 languages = 140 evals per condition.
- EXPERIMENT: MS2 partial run (background, later killed). Baseline total 95/140 (67.9%): EN=13, ZH=17, AR=13, ES=16, JA=12, KO=15, SW=9.
- EXPERIMENT: MS2 surgery partial (through JA): EN=19 (+6), ZH=10 (-7), AR=14 (+1), ES=10 (-6), JA=11 (-1). Non-dominant languages (ZH, ES) collapse.
- BREAKTHROUGH: The toy theorem’s α (training proportions) are built directly into Z. Centroid SVD uses uniform weights, but the model’s α-weighted equilibrium defines which directions are convention-invariant. Uniform centroid strips directions that the model considers Z under α, explaining ZH and ES collapse.
- KILLED: Uniform 7-language centroid SVD as a method for convention removal – it destroys accuracy on non-dominant languages because it ignores training proportions α.
- BRAINSTORM: Two fixes: (1) Procrustes-derived Z from cocycle (principled, Money Shot 1 in Architectural Decomposition), (2) α-weighted centroid as quick diagnostic: estimate α from per-language distances to global mean, reweight centroid accordingly.
- DECISION: Next session will pursue both Procrustes-based surgery and α-weighted centroid diagnostic.
- DECISION: Kill background MS2 run, dump partial output for later inspection.
- DELIVERABLE: output/expMS2_7lang_surgery_partial_run.log (4122 lines) – contains baseline all 7 langs and surgery through JA.
- STATE: 419e01cd saved (end of session, partial MS2 output persisted, next actions queued).
- NEXT: Procrustes-based surgery (cocycle-derived Z) and α-weighted centroid diagnostic – clean slate next session.

---

## 2026-04-13 22:06 UTC — Fix and re-run centroid-based surgery (LOO and bilateral) to diagnose ZH regression, then build Procrustes surgery (Fix 1) and MI decomposition as coordinate-free cross-lingual alignment metrics, while auditing grader bugs and centroid bias.

*Source: [2026-04-13_22-06_evaluate-stale-constraint.md](2026-04-13_22-06_evaluate-stale-constraint.md)*

**Objective:** Fix and re-run centroid-based surgery (LOO and bilateral) to diagnose ZH regression, then build Procrustes surgery (Fix 1) and MI decomposition as coordinate-free cross-lingual alignment metrics, while auditing grader bugs and centroid bias.
- STATE: MS1d complete, MS2 killed (α-weighting confound) — loaded from prior session.
- EXPERIMENT: GPU check via nvidia-smi — 2.2GB/12GB used, 36% util, no zombie Python or inference processes (all GUI).
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/docs/20260413_MS2_Fixes_MI_Decomposition.md` (165 lines) — persists Opus web handoff with Fix 1 (Procrustes), Fix 2 (grader), MI decomposition framework.
- AUDIT: check_answer() in MS2 has two bugs: (1) trailing-dot regex eats period characters, (2) literal yes/no matching fails on "97 is a prime number" responses.
- EXPERIMENT: Centroid bias audit — computed per-language distance from centroid across 200 problems × 7 languages. L2 ranking: KO > ZH > EN > ES > AR > JA > SW.
- FINDING: English is **not** closest to the centroid. Top SVD direction captures a **CJK-vs-Latin script family axis**: EN/AR/ES/SW project positively, ZH/JA/KO project negatively.
- PUSHBACK: Web session hypothesized English-dominant centroid bias ("push toward English") — audit shows the mechanism is script-family axis, not English proximity.
- DECISION: Use leave-one-out centroid (centroid from other 6 languages) and bilateral mean-diff (average of 6 pairwise difference vectors) as two surgery variants.
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expMS2b_loo_surgery.py` (441 lines) — runs baseline, LOO centroid SVD surgery, and bilateral mean-diff surgery on 7 languages × 20 problems × 1024 tokens, with fixed grader.
- BLOCKER: Initial MS2b run hung due to full SVD on (1400,2048) matrices × 252 operations, causing swap thrashing (8GB swap full, process VM 8.7GB, RSS 1.3GB).
- DECISION: Replace full SVD with truncated SVD (scipy.sparse.linalg.svds, k=1) — reduces memory and runtime from >2 hours to ~60 seconds for LOO computation.
- EXPERIMENT: MS2b baseline (fixed grader) partial results — EN: 19/20 (+6 vs MS2 old grader), ZH: 18/20 (+1), AR: 14/20 (+1), ES: 20/20 (+4), JA: 13/20 (+1), KO: 16/20 (+1), SW: 10/20 (+1).
- FINDING: ES hits 20/20 perfect score — problems may be too easy for some languages.
- EXPERIMENT: Verified EN and ZH failures in baseline — EN P5 (347+658 = 905 vs 1005), ZH P6 (1000-387 output truncated), ZH P20 (9×8×7=504 vs 648) — all genuine math errors, not grading artifacts.
- BRAINSTORM: Greg proposes finding layer L* with maximal cross-lingual similarity, then injecting hidden state from a math problem into a "Describe what is going on" prompt at L* to test if upper layers can decode language-agnostic computation.
- DECISION: Prototype injection after MI results, stay focused on running experiments.
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expMS3_procrustes_surgery.py` (368 lines) — alpha-independent surgery: for each layer, compute Procrustes maps across 21 language pairs, define convention subspace as directions most rotated between pairs.
- EXPERIMENT: MS2b LOO surgery partial result — EN: 20/20 (+1 vs baseline), ZH: 12/20 (-6 vs baseline).
- FINDING: LOO centroid surgery made ZH worse (12/20) than original MS2 uniform centroid (10/20) and baseline (18/20) — script-family axis persists regardless of centroid computation method.
- STATE: Snapshot 1248b54e saved — context at 62%, MS2b still running (bilateral results pending), MS3 and MI scripts written and waiting for GPU.
- NEXT: Run `/clear` to reset context, then restore from state 1248b54e, collect bilateral surgery results from MS2b, run MS3 (Procrustes surgery), then run MI decomposition to find L* for injection experiment.

---

## 2026-04-13 23:06 UTC — Advance the research programme understanding how multilingual transformers separate convention (language-specific formatting) from computation (mathematical reasoning) by testing multiple surgery methods (LOO centroid SVD, bilateral mean-diff, cPCA) to isolate convention subspaces, and then testing whether language‑invariant bottleneck representations can be decoded back into mathematical content via hidden‑state injection.

*Source: [2026-04-13_23-06_context-clear-recommend.md](2026-04-13_23-06_context-clear-recommend.md)*

**Objective:** Advance the research programme understanding how multilingual transformers separate convention (language-specific formatting) from computation (mathematical reasoning) by testing multiple surgery methods (LOO centroid SVD, bilateral mean-diff, cPCA) to isolate convention subspaces, and then testing whether language‑invariant bottleneck representations can be decoded back into mathematical content via hidden‑state injection.
- STATE: Snapshot eee285dc saved at end of session (all MS2b, MS4, and injection results persisted).
- EXPERIMENT: MS2b LOO‑SVD surgery on 20 problems × 7 languages (EN, ZH, AR, ES, JA, KO, SW); baseline total 110/140.
- EXPERIMENT: MS2b bilateral mean‑diff surgery on same 20×7 set within the same script.
- FINDING: Baseline per‑language scores (pre‑surgery, fixed grader): EN 19/20, ZH 18/20, AR 14/20, ES 20/20, JA 13/20, KO 16/20, SW 10/20.
- FINDING: LOO‑SVD surgery total 94/140 (−16 relative to baseline).
- FINDING: LOO‑SVD per‑language: EN 20/20 (+1), ZH 12/20 (−6), AR 15/20 (+1), ES 11/20 (−9), JA 12/20 (−1).
- FINDING: LOO‑SVD ES collapsed from 20 to 11 (−9), much worse than predicted by CJK‑only hypothesis.
- AUDIT: Assistant discovered that LOO centroid SVD ratios are invariant to which language is excluded (sv1/sv2 at L18: EN‑excluded=1.11, ZH‑excluded=1.11, AR‑excluded=1.11 to 3 decimals) – convention subspace is defined by the whole ensemble, not any single language.
- FINDING: Bilateral mean‑diff surgery total 114/140 (+4).
- FINDING: Bilateral per‑language: EN 20/20 (+1), ZH 20/20 (+2), AR 15/20 (+1), ES 20/20 (0), JA 18/20 (+5), KO 18/20 (+2), SW 6/20 (−4).
- FINDING: JA bilateral gains 5 problems (problems 4,5,7,11,18) with zero regressions, spanning algebra, arithmetic, geometry, combinatorics.
- FINDING: SW bilateral loses 5 problems (problems 4,6,7,9,10) in arithmetic and geometry – its strongest categories – supporting a “bridge language” hypothesis where Swahili relies on shared English‑like pathways for math.
- BRAINSTORM: cPCA (contrastive PCA) proposed as an alternative to Procrustes: find the direction that maximizes language variance while having zero math‑problem variance.
- DELIVERABLE: `procrustes_preview.py` written and run on CPU (`output/multilingual_all_layers.npz`).
- EXPERIMENT: Procrustes preview (CPU diagnostic) on all 36 layers; sv1/sv2 = 1.00 at every layer.
- FINDING: Procrustes averaging of rotation matrices from all 21 language pairs yields a flat singular value spectrum – mathematical artifact because rotations on SO(n) interfere destructively when averaged.
- KILLED: MS3 Procrustes surgery cancelled after preview; running full surgery would waste GPU time on a predicted null result.
- DECISION: Kill MS3 and instead build MS4 cPCA surgery.
- DELIVERABLE: `expMS4_cpca_surgery.py` (343 lines) written.
- EXPERIMENT: MS4 cPCA surgery launched with 20×7 problems, matched system prompts, surgery layers L13‑L35.
- AUDIT: First MS4 run used a different problem set (11/20 problems differed from MS2b); baseline 102/140 (expected 110/140). Assistant caught discrepancy and relaunched with correct problems.
- EXPERIMENT: MS4 cPCA surgery (correct problem set) completed; wall time 2912s (48.5 min).
- FINDING: MS4 cPCA total 111/140 (+1 relative to baseline 110/140).
- FINDING: MS4 cPCA per‑language: EN 20/20 (+1), ZH 20/20 (+2), AR 15/20 (+1), ES 18/20 (−2), JA 14/20 (+1), KO 16/20 (0), SW 8/20 (−2).
- FINDING: cPCA’s zero‑math‑variance constraint prevents the catastrophic drops of LOO‑SVD but also removes bilateral’s net benefit; ZH reaches perfect 20/20.
- DELIVERABLE: `docs/20260413_surgery_analysis.md` written, comparing MS1, MS2b (LOO & bilateral), MS3 (killed), and MS4 results.
- BRAINSTORM: Injection experiment design – encode a math problem in EN or ZH, capture hidden state at layer L*, inject that state into a “Describe what is happening” English prompt, and test whether the model introspects the math content.
- DELIVERABLE: `expINJ_hidden_injection.py` written with hook‑based hidden state capture and injection.
- EXPERIMENT: expINJ run with 3 problems × 6 L* values (15,20,25,27,30,33) × 4 conditions (EN→EN, ZH→EN, NOISE, BASELINE).
- AUDIT: Initial injection script crashed due to Qwen returning 2D tensors `(seq_len, d)` instead of 3D `(batch, seq_len, d)` and because decoder layer output was a tensor not a tuple. Assistant fixed both issues.
- FINDING: Noise injection produces garbage (emojis, random tokens, repetition) at all layers – injection mechanism works and control is valid.
- FINDING: EN→EN injection outputs are indistinguishable from baseline (generic “sum of two numbers” or “let’s break down step by step”) – EN math hidden state is too similar to what the “Describe” prompt naturally produces.
- FINDING: ZH→EN injection reliably causes language switching to Chinese at L27+ – the hidden state’s language identity overrides the English prompt.
- FINDING: At L33, one of three problems (triangle area) produces Chinese text with the correct problem‑specific formula: “三角形的面积可以通过以下公式计算：面积 = 底边长度 × 高 ÷ 2” – problem‑specific math content transferred.
- FINDING: Overall ZH→EN math content transfer is partial (1/3 at L33 only); convention (language identity) dominates over math content in the hidden state.
- FINDING: Cosine similarity between EN and ZH hidden states at L25 = 0.9766, at L15 = 0.9565, at L20 = 0.9351, at L27 = 0.9282 (from injection experiment logs).
- BREAKTHROUGH: Synthesis of surgery and injection results – “convention and computation share 73% of the geometry, and convention is the louder voice. The model’s upper layers are primarily a language decoder, not a math decoder.” The 3B model lacks reliable metacognitive capacity for math introspection even when math content is present in Z.
- PUSHBACK: Stop hook flagged stale constraint #14 (“terminate running process”) as no longer relevant; assistant acknowledged.
- PUSHBACK: Stop hook flagged drift into reflective discussion; assistant noted user requested the debrief but agreed to return to execution.
- PUSHBACK: Stop hook flagged blind spot about integrating MS4 results into the MS2b/MS3 comparison framework; assistant acknowledged and updated approach.
- DECISION: Deprioritize MI decomposition (CPU‑heavy, not blocking injection).
- DECISION: Prioritize injection experiment as the test of the von Neumann question (whether the model introspects its own representations).
- DELIVERABLE: Updated `~/.claude/vega.md` with Blind Spot #132 (“Optimized for table completeness instead of conjecture testing”) and Lesson about LOO centroid SVD invariance and truncated SVD under swap pressure.
- META: Cron job set up to monitor experiment progress every 12 minutes, then changed to every 15 minutes for MS4.
- META: Context reached 70% at end of session; assistant recommended `/clear` with all results saved.

---

## 2026-04-14 02:40 UTC — Determine whether a multilingual transformer’s intermediate representations (Z) can be decoded into natural‑language descriptions of the underlying computation (rather than just continuing to solve the problem), and if so, what architectural conditions (KV cache, layer, token position) enable this decomposition.

*Source: [2026-04-14_02-40_save-inter-session-state.md](2026-04-14_02-40_save-inter-session-state.md)*

**Objective:** Determine whether a multilingual transformer’s intermediate representations (Z) can be decoded into natural‑language descriptions of the underlying computation (rather than just continuing to solve the problem), and if so, what architectural conditions (KV cache, layer, token position) enable this decomposition.
- AUDIT: External analysis (Claude Web) identified that `expINJ_hidden_injection.py` only replaced the last token’s hidden state at L* while the full KV cache of the “describe” prompt remained intact, creating an adversarial context where one math‑state vector had to override ~20 describe‑context tokens → explains the original 33% math‑content hit rate.
- BRAINSTORM: Three fixes proposed: (A) replace all token hidden states at L* (full‑state replacement), (B) continuation injection without a “describe” prompt (math context only), (C) full‑state transplant into cross‑language describe prompts.
- EXPERIMENT (expINJ2 / Variant C): Last‑token swap with math context (MOAMS control) → math content 42/42 (100%), garbage 0%. Confirms injection infrastructure works when context aligns with injection.
- EXPERIMENT (expINJ2 / Variant A): Full‑state replacement (all tokens) at L* into English describe prompt → math content 26/42 (62%), garbage 14/42 (33%). Triples math content over original INJ, but significant garbage.
- EXPERIMENT (expINJ2 / Variant B): Full‑state replacement (all tokens) at L* into Chinese describe prompt → math content 24/42 (57%), garbage 13/42 (31%).
- EXPERIMENT (expINJ2 / Variant D): Meta prompt + last‑token only (neutral context) → math content 14/42 (33%), garbage 0%. Single token insufficient even without KV conflict.
- EXPERIMENT (expINJ2 / Variant E): Cascade injected frozen hidden states at all layers L* through L35 → math content 13/42 (31%), garbage 2/42 (5%). Forcing same state at every upper layer prevents necessary computation.
- FINDING: In Variant A, garbage rate is strongly problem‑dependent: geometry problem (triangle area) 71% garbage, algebra (3x+7=22) 29% garbage, GCD problem 0% garbage. Cross‑lang cosine similarity for geometry is lowest at every L* (0.866 at L30 vs 0.921 for GCD).
- FINDING: Norms of hidden states (mean and last token) are within 5% across all problems and describe prompt at every layer → norm mismatch ruled out as cause of garbage.
- FINDING: Garbage peaks in canyon layers L25–L30 (50% garbage) and is lowest at L15–L18 (17%) and L33 (17%).
- EXPERIMENT (expINJ2F): Full KV cache replacement experiment with two variants – F (replace KV only for layers 0..L*, hidden states at L*) and F2 (replace KV for all layers 0..L35 plus hidden states at L*). F2 runs on Qwen2.5-3B, L* ∈ {15,18,25,30}, 3 problems × 4 language pairs × 4 L*.
- FINDING (expINJ2F): F (partial KV) yields math content 22/48 (46%), garbage 17/48 (35%), correct answer 18/48 (38%). F2 (full KV) yields math content 44/48 (92%), garbage 4/48 (8%), correct answer 44/48 (92%). Noise control outputs complete nonsense.
- FINDING (expINJ2F): F2 outputs are concise solutions (“The solution to 3x+7=22 is x=5”, “The GCD of 84 and 120 is 12”) not natural‑language descriptions of the reasoning → model solves rather than describes when entire KV cache is math.
- EXPERIMENT (expINJ3_trio, Part 1 – Tuned lens): Project hidden states at every layer through unembedding for EN and ZH versions of three math problems. At L22 the top token is “Answer” for both languages (EN probability 0.597, ZH 0.354‑0.372), showing convergence to a shared conceptual token before diverging into language‑specific generation tokens at L27–L35.
- EXPERIMENT (expINJ3_trio, Part 2 – Mean KV): Average EN and ZH KV caches (full all‑layer replacement) and inject into describe prompt. Averaging corrupts numerical precision: algebra problem output “x=3” (correct is 5), GCD problem output “GCD of 8 and 12 = 4” (correct is 12). Mean KV harms exact math reasoning, contrasting with Dumas et al. where averaging helps soft translation tasks.
- EXPERIMENT (expINJ3_trio, Part 3 – Reverse injection): Inject describe‑prompt KV (English or Chinese) into a math prompt (English or Chinese) with full all‑layer replacement. Model switches to description mode (e.g., “To perform the mathematical operation we need to understand the context”, or “To进行数学运算或推理，需要提供具体的数学问题”) instead of solving. Proves the “render” function (h′) is separable from the content.
- BREAKTHROUGH: The core distinction is between **content** (what computation is being performed) and **mode** (whether the model describes or solves). F2 gets content right but mode wrong (solves). Variant A at late layers gets mode right (describes) but content degraded. No experiment yet achieves correct description of the correct math.
- BRAINSTORM: Six new experiments (G1–G6) to achieve bun inversion (correct description of correct math): G1 (Inverted‑F – math KV in lower layers, describe KV in upper layers), G2 (Additive injection), G3 (Multi‑token injection, sweep K=1,3,5,all), G4 (Natural description baseline, no injection), G5 (Mode steering vector), G6 (Partial‑position KV hybrid).
- DECISION: Prioritize G4 (natural description baseline, 2 min) to establish target output format, then G1 (most promising mechanistic separation), then G5 (most novel if successful).
- DELIVERABLE: `expINJ2F_kv_replace.py` – KV cache replacement experiment with F and F2 variants.
- DELIVERABLE: `expINJ3_trio.py` – tuned lens, mean KV, and reverse injection experiments.
- DELIVERABLE: Output files `output/expINJ2_full_injection.json`, `output/expINJ2F_kv_replace.json`, `output/expINJ3_trio.json`.
- NEXT: Start next session with G4 – natural description baseline – to get genuine description outputs for the three math problems without any injection, then move to G1 (Inverted‑F: math KV in layers 0..L*, describe KV in layers L*+1..L35).
- STATE: 77e85914 – saved with full session trajectory including Web diagnosis, all experiment outputs and file paths, KV cache mechanics, F2 vs description reframing, G‑series designs, literature positioning, and open questions.

---

## 2026-04-14 06:04 UTC — Test whether a model can describe mathematical reasoning contained in a problem's lower-layer KV cache without the problem in the prompt (“bun inversion”), scale from 3B to 14B, and characterize the content/mode decomposition.

*Source: [2026-04-14_06-04_ssh-cloudflare-tunnel.md](2026-04-14_06-04_ssh-cloudflare-tunnel.md)*

**Objective:** Test whether a model can describe mathematical reasoning contained in a problem's lower-layer KV cache without the problem in the prompt (“bun inversion”), scale from 3B to 14B, and characterize the content/mode decomposition.
- EXPERIMENT: G4 (natural description baseline) at 3B – 6/6 describe outputs contain both descriptive language and the answer; 0 pure descriptions. Shows 3B's metacognitive ceiling: it describes-then-solves as a single mode.
- FINDING: Mode divergence between solve and describe residual streams at 3B peaks at L8 with mean cos = 0.912 (only ~9% separation), not in upper layers. Max divergence early.
- EXPERIMENT: G1 inverted-F (math KV layers 0..L*, describe KV layers L*+1..35, describe prompt includes the problem) – runs L* sweep [12,15,18,22,27]. Corrected results: L*=12-22 has 0 real garbage, 6-10/12 descriptive outputs, answer rate 12/12; L*=27 has 8/12 garbage, descriptive drops to 3/12.
- FINDING: Inverted-F works (produces descriptions of math operation) but 3B cannot describe without also solving – matches G4 baseline limitation, not a failure of the intervention.
- AUDIT (self): G1's describe prompt contains the problem (“Describe … needed to solve this problem: 3x+7=22”) → math content already in prompt, lower-layer math KV is redundant. G1 cheats the intended test.
- BRAINSTORM: Clean bun inversion – use generic describe prompt (“Describe what mathematical reasoning is being performed:”) with zero problem information. Math KV at early layers provides content via attention, upper describe KV sets mode.
- DECISION: Prioritize H-series (inject math hidden states into residual stream with generic describe KV) over G2‑G6. H1 (single injection at L*), H1b (continuous all layers), H1c (logits bias, burst, gen-time hidden states) tested.
- EXPERIMENT: H1b (continuous injection – all layers, every token) – all garbage (“ToToTo” loops). One leak: geometry prompt produces repeating “triangle”.
- EXPERIMENT: H1c‑C (gen‑time hidden states at L*=27 injection into generic describe prompt) – 1/3 algebra problem yields “applying algebraic rules … to isolate the variable x”. Partial proof of concept: content-specific description from Z, but only 1/3, number theory cross‑contaminates.
- EXPERIMENT: G1b blind inverted‑F (generic describe prompt, math KV low, describe KV high, same KV construction as G1 but prompt blind) – runs L*=10 to 27. Best result: L*=27, p1 geometry zh→zh produces “底边长度为10，高为7，面积=35” (base 10, height 7, area 35). Exact numbers recovered from blind prompt. Also L*=18 algebra en→en recovers structure (“isolate variable x”) but corrupts numbers (2x+3=7 instead of 3x+7=22).
- FINDING: G1b blind inverted‑F hierarchy of recovery: generic → operation type → structure → exact numbers. Higher L* (more math layers) gives stronger content but increases garbage (seam artifact). zh→zh cleanest because no language mismatch.
- EXPERIMENT: G1c soft blend (smooth transition over a few layers around L* to reduce seam artifact) – best config C27_W5 (center L*=27, width=5). Results: garbage drops from 3/6 to 2/6; avg math hits 1.7 → 2.5; coverage extends to all three problems (algebra, geometry, number theory). Tradeoff: specific numbers corrupt (height 7→3) while structure survives.
- DECISION: Scale to 7B on A100. Sweep proportional L* = 15‑24, test hard cutoff and blend. Add arithmetic and word problems.
- EXPERIMENT: G1d 7B bun inversion (Qwen2.5‑7B, 28 layers, raw encoding optional) – run on A100. Exact recoveries from blind generic prompt: algebra zh→en “3x+7=22” (exact equation), geometry zh→en “base 10, height 7”, number theory en→en “GCD of 84 and 120”, arithmetic en→en “17×23=391”, arithmetic zh→en “17 multiplied by 23 is 391”. Garbage rate 0‑10% across all conditions. zh→zh broken due to chat template system prompt leakage (“You are a helpful assistant” tokens dominate lower‑layer math KV).
- FINDING: Scaling trend 3B→7B: L* sweet spot ratio increases from 0.75 (27/36) to 0.86 (24/28). Seam artifact shrinks dramatically. Cross‑lingual (zh→en) works at 7B but not at 3B.
- EXPERIMENT: G1e 14B bun inversion (Qwen2.5‑14B, 48 layers, L* sweep [30,36,38,41,44], blend C41_W6, plus raw‑encoding mode without chat template). Results: zero garbage across all configurations. Raw encoding (no chat template) gives strongest output: p2 number theory zh→en produces “84=2×2×3×7, 120=2×2×2×3×5, 所以84和120的最大公因数是12” – prime factorizations and correct GCD 12. Algebra raw zh→en recovers “3x+12=27 … x=5” (answer correct despite constant shift).
- FINDING: 3‑model scaling summary (best configs): 3B geometry exact numbers zh→zh, but high garbage; 7B exact equations/cross‑lingual, low garbage; 14B zero garbage, correct computations (GCD 12), raw encoding eliminates system prompt leakage.
- DELIVERABLE: expG1_inverted_f.py – G4+G1 combined script
- DELIVERABLE: expG1b_blind_inverted_f.py – blind describe prompt version
- DELIVERABLE: expG1c_soft_blend.py – smooth transition variant
- DELIVERABLE: expG1d_7b_bun_inversion.py – 7B script for A100
- DELIVERABLE: expG1e_14b_bun_inversion.py – 14B script with raw encoding option
- DELIVERABLE: output/expG1_inverted_f.json, output/expG1b_blind_inverted_f.json, output/expG1c_soft_blend.json, output/expG1d_7b_bun_inversion.json, output/expG1e_14b_bun_inversion.json
- STATE: Session state saved multiple times; final snapshot context: 14B results complete, A100 still live, raw encoding is the recommended default.
- NEXT: Run 20‑problem battery on 7B and 14B with raw encoding (no chat template) to turn proof of concept into proper statistics; optionally test Qwen2.5‑7B‑base to check instruct‑tuning artifacts.

---

## 2026-04-14 09:02 UTC — Complete the two-hijack bun inversion experiment (G5) on an A100 to decouple content (F2 KV) from mode (steering vector), and produce a prioritized experiment roadmap for the next session.

*Source: [2026-04-14_09-02_debug-runtime-crash.md](2026-04-14_09-02_debug-runtime-crash.md)*

**Objective:** Complete the two-hijack bun inversion experiment (G5) on an A100 to decouple content (F2 KV) from mode (steering vector), and produce a prioritized experiment roadmap for the next session.
- BRAINSTORM: Use both KV cache (for content) and residual stream steering (for mode) simultaneously instead of splitting signals via L* — KV gets 100% to content, separate intervention adds a per-layer describe steering vector (δ^(ℓ) = mean(h_desc^(ℓ)) - mean(h_solve^(ℓ)) computed once from trivial problems) to control mode.
- EXPERIMENT: expG5_two_hijack.py (14B, F2 all-math KV + post-hook steering, sweep α = 0.5,1.0,2.0,3.0,5.0, raw encoding, 5 test problems) — script written and local, but not run because A100 tunnel died mid-execution.
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expG5_two_hijack.py` (510 lines) — implements steering vector extraction from 3-5 calibration problems not in test set, post-hook addition at each layer, and comparison against F2 baseline and G1b blind.
- DELIVERABLE: Outstanding experiments table (Tier 1: G5, G5-20 battery, G1e-20; Tier 2: Llama-3.1-8B (G6), base model comparison (G7); Tier 3: MS4/MS5/MS6 surgery; Tier 4: C8, AA-7B, BR-7B) with narrative arc (Acts 1-3) and recommended session order.
- META: SSH tunnel (santa-representing-characterized-repeat.trycloudflare.com) with A100 died during G5 run ("runtime died"), preventing experiment completion; assistant saved state and will resume next session.
- STATE: Session state saved (project: BrainInsideTheMachine) — includes G5 script ready, experiment table, and queued next actions.
- NEXT: Run G5 on a fresh A100 (10-15 min) as first action next session, then G1e-20 battery, then G6 Llama-3.1-8B if runtime stays live.

---

## 2026-04-18 09:04 UTC — Test whether adding a variance bonus to attention logits (distributional attention via the kernel trick) improves reasoning accuracy in Qwen2.5-3B on multilingual math problems without retraining.

*Source: [2026-04-18_09-04_kernel-attention-baseline-check.md](2026-04-18_09-04_kernel-attention-baseline-check.md)*

**Objective:** Test whether adding a variance bonus to attention logits (distributional attention via the kernel trick) improves reasoning accuracy in Qwen2.5-3B on multilingual math problems without retraining.
- BRAINSTORM: Distributional attention as a one-line modification: replace point-estimate query with Gaussian uncertainty, compute expected kernel evaluation via moment generating function → closed-form variance bonus b_j = (k_j^T Σ_q k_j)/(2d) that adds to logits before softmax; the bonus is always positive, making uncertain queries attend more broadly (UCB/exploration), and can be implemented by propagating diagonal variance through attention (σ²_post-attn,i = Σ_j α_j²·σ²_vj,i) and SiLU-gated MLP (variance zeroes on dead dimensions).
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expDA1_distributional_attention.py` — implements monkey-patch of Qwen2Attention.forward with variance bonus, propagates diagonal covariance, supports fixed α multiplier or propagated variance, generates on 20 problems × 2 languages (EN/ZH) with 128 max tokens.
- EXPERIMENT: expDA1 dry run (3 combinatorics problems, 64 tokens) — baseline and bonus conditions all produce garbage outputs (emoji degeneration, hallucinated numbers) due to insufficient token budget; monkey-patch mechanically works (no crashes).
- EXPERIMENT: expDA1 full run (20 problems × 2 langs, 128 tokens) with default SDPA attention — baseline scores EN=2/20, ZH=3/20, total=5/40, far below C2b reference (17/40); outputs show emoji degeneration, suggesting SDPA vs eager numerical mismatch.
- AUDIT: Monkey-patched model using SDPA (default) misreads input (e.g., "Calculate 444+68" instead of "544+68") because the patch reimplements eager attention with different numerical precision, changing prompt encoding.
- DECISION: Force `attn_implementation="eager"` in model loading so baseline and patched versions share the same attention implementation; rerun expDA1.
- EXPERIMENT: expDA1 rerun with eager attention (baseline) — baseline scores EN=4/20, ZH=11/20, total=15/40 (still below C2b reference of 17/40 but plausible with 3B model). Fixed α=0.1 scores EN=7/20, ZH=11/20, total=18/40 (+3 total gain); propagated variance (prop_0.1) EN=8/20, ZH=10/20, total=18/40; α=0.5 collapses to EN=4/20, ZH=3/20, total=7/40.
- PUSHBACK: User questions low baseline scores — suspect token cap or grader bug; assistant confirms max_new_tokens=128 and examines grading.
- AUDIT: Grader uses regex `re.findall(r"-?\d+\.?\d*", text)` which captures trailing periods as part of the number string (e.g., "348." instead of "348"), causing false negatives when answer is followed by a period. Baseline has 5 false negatives (marked wrong but actually correct) vs fixed_0.1 has 1 false negative — the reported improvement at α=0.1 is an artifact of differential grader bias.
- DECISION: Fix grader by stripping trailing periods from extracted number strings before comparing to ground truth; rerun entire experiment to obtain corrected scores.
- NEXT: Rerun `expDA1_distributional_attention.py` with fixed grader (strip trailing periods) to obtain unbiased accuracy numbers for baseline and all bonus conditions.

---

## 2026-04-18 09:52 UTC — Test whether the model's own decision confidence (SiLU derivative from MLP gates) can be used to modulate attention queries and improve mathematical reasoning, as an extension of the bun inversion project's search for latent reasoning capacity.

*Source: [2026-04-18_09-52_doc-sensitivity-tension-method.md](2026-04-18_09-52_doc-sensitivity-tension-method.md)*

**Objective:** Test whether the model's own decision confidence (SiLU derivative from MLP gates) can be used to modulate attention queries and improve mathematical reasoning, as an extension of the bun inversion project's search for latent reasoning capacity.
- BRAINSTORM: Propose using SiLU derivative (gradient proprioception) to modulate attention queries, based on insight that derivative is provably unrecoverable from activation value and represents decision confidence (edge neurons).
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expSMA_sensitivity_modulated_attention.py` (549 lines) implementing sensitivity-modulated attention via hooks, including diagnostic, 5 conditions (baseline, sensitivity, random, uniform, inverse), and exact problem set from expMS1.
- EXPERIMENT (SMA v1): Run full experiment on 20 math problems × 2 languages (en, zh) with 5 conditions. Baseline: 16/40 (EN=5/20, ZH=11/20); sensitivity_modulated: 3/40 (EN=3/20, ZH=0/20), delta -13; random_modulation: 21/40 (EN=9/20, ZH=12/20), delta +5; uniform_modulation: 16/40, delta 0; inverse_modulation: 2/40, delta -14.
- FINDING: Uniform modulation exactly matches baseline, confirming hook infrastructure is clean.
- FINDING: Random modulation improves accuracy by +5 (from 16 to 21), suggesting attention query noise acts as stochastic regularization independent of sensitivity signal.
- AUDIT: SiLU'(x) exceeds [0,1] — at x = -3, g ≈ -0.087; at x = 2, g ≈ 1.09 — causing τ = g(1-g) to go negative at early layers (mean_g negative L1-L10 in diagnostic), which corrupts modulation vector into sign-flipped and amplified dimensions.
- DECISION: Fix tension formula from g·(1-g) to σ(x)·(1-σ(x)) (sigmoid probability variance, always in [0,0.25]), and add alpha sweep to control modulation strength.
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/expSMA2_corrected_tension.py` (443 lines) implementing corrected tension with alpha sweep (α = 0.01, 0.05, 0.1, 0.3, 0.5, 1.0), random control, uniform, inverse.
- EXPERIMENT (SMA2): Run corrected tension experiment. Diagnostic confirms v2 tension ≥0 at all layers; baseline 16/40; α=0.5 yields EN=9/20 (+80%), total=20/40 (+4); α=0.3 yields 18/40 (+2); α=1.0 yields 18/40 (+2); random (seed42) yields 18/40 (+2); inverse yields 4/40 (-12).
- FINDING: Corrected tension formula works — at α=0.5, English accuracy jumps from 5→9; sensitivity signal beats random by +2, confirming information content matters, not just perturbation noise.
- FINDING: Inverse modulation is destructive (-12), confirming direction of modulation matters.
- FINDING: Alpha sweep shows inverted bell curve response — too weak (α≤0.1) does nothing, sweet spot at α=0.3-0.5, too strong (α=1.0) starts degrading.
- KILLED: Original tension formula (g·(1-g)) is invalid for SiLU derivative because SiLU' is not bounded to [0,1]; replaced with σ·(1-σ).
- BLOCKER: Format string bug (`alpha:<5` expecting string but alpha is float) in SMA2 analysis print crashed before saving JSON; all empirical data captured in stdout but not saved to output/expSMA2*.json.
- PUSHBACK: User objects to slowness of script and fragility of regex answer checking on simple arithmetic/algebra problems; proposes using AMC problems instead for more robust evaluation.
- DECISION: Pivot to AMC 2025 12A problems (excluding those with images: 5,10,14,20,24) for new SMA3 experiment, with bulletproof answer logging and test run on one question one language.
- STATE: Saved snapshot `8e4bbcba` before clearing context, including all SMA findings and updated memory files.
- DELIVERABLE: `/home/greg/.claude/projects/-home-greg-Desktop-Projects-BrainInsideTheMachine/memory/project_sma_experiment.md` documenting the tension methodology (SiLU derivative, correct σ(1-σ) formula, hook implementation, alpha sweep) and results chain.
- DELIVERABLE: Updated `/home/greg/.claude/projects/-home-greg-Desktop-Projects-BrainInsideTheMachine/memory/MEMORY.md` with SMA experiment results, evidence chain, and state pointer (8e4bbcba).
- NEXT: Write SMA3 script using AMC 2025 12A problems (exclude image-based), implement robust answer checking with full logging, and run test on one problem one language to validate runtime and output format.

---

## 2026-04-18 11:25 UTC — Testing whether Sensitivity-Modulated Attention (SMA) improves mathematical reasoning (AMC 12A problem solving) in Qwen2.5-3B, while fixing implementation issues (max token truncation, system prompt contamination, cross-layer modulation mismatch).

*Source: [2026-04-18_11-25_clear-session-state.md](2026-04-18_11-25_clear-session-state.md)*

**Objective:** Testing whether Sensitivity-Modulated Attention (SMA) improves mathematical reasoning (AMC 12A problem solving) in Qwen2.5-3B, while fixing implementation issues (max token truncation, system prompt contamination, cross-layer modulation mismatch).
- EXPERIMENT: expSMA3_amc_sweep.py with 19 AMC 12A problems, max_new_tokens=512, letter grading — baseline 0/1 (0%), sma_a0.3 1/1 (100%), but truncation cut off sma_a0.5 and sma_a1.0 before answer.
- AUDIT: User identified 512-token ceiling systematically truncates modulated conditions because they produce more verbose reasoning, biasing results against SMA.
- FINDING: sma_a0.3 on problem 1 caught perpendicularity nuance ("they are moving in perpendicular directions") that baseline missed, showing improved reasoning despite truncated output.
- FINDING: sma_a1.0 set up correct equation 8t = 12(t-1) and solved t=3 (giving 4:30, answer E), but misread time as 5:00 PM; algebraic reasoning correct, reading comprehension failed.
- DECISION: Increase max_new_tokens to 1024 and add repetition_penalty=1.05 to break loops.
- EXPERIMENT: Re-run test with MAX_NEW=1024, repetition_penalty=1.05 — all conditions returned "none" due to gibberish loops; repetition_penalty caused degenerate output.
- DECISION: Revert repetition_penalty, keep MAX_NEW=512, add fallback answer extraction (boxed values, "option X" phrases).
- PUSHBACK: User argued that truncation is a confound and max tokens should be removed entirely; assistant initially pushed for 1024, user insisted on no limit or 32768.
- DECISION: User instructs to run only one hard problem (Problem 12: harmonic mean polynomial) with no max token limit, no system prompt, no answer choices shown.
- BRAINSTORM: Move sensitivity computation from post-hook on previous MLP to pre-hook on current layer's attention, computing W_up·h as peek before attention (same-layer modulation).
- AUDIT: Qwen's chat template auto-injects "You are a helpful assistant." even without system message; user rejects this as contamination.
- EXPERIMENT: Run on single problem (harmonic mean polynomial) with 7 conditions (baseline, sma_a0.3, sma_a0.5, sma_a1.0, uniform, random_s42, inverse), max_new_tokens=32768 (blocking), no system prompt — baseline crashed (spiral), sma_a0.5 and sma_a1.0 stuck in loops.
- FINDING: Without system prompt, Qwen2.5-3B on this problem defaults to pathological behavior (echoing, looping, blurting "1").
- DECISION: Add back minimal system prompt "You are a careful mathematical reasoner. Show your work step by step." — no answer choices, stock Qwen generation defaults (do_sample=False, max_new_tokens=2048, eos_token_id includes <|im_end|>).
- EXPERIMENT: Final run on harmonic mean polynomial with system prompt restored — all 7 conditions completed. Baseline: 521 tokens, 8.8s, answer 4050/H_2025 (wrong, confused sum-of-roots with sum-of-reciprocals). sma_a0.3: 790 tokens, 19.1s, answer 2025/(4 ln 2025) (same confusion plus log approx). sma_a0.5: 756 tokens, 18.3s, answer 2025/(2(ln2025+γ)). sma_a1.0: 2048 tokens, 49.8s, infinite loop ("reciprocals of the reciprocals..."). uniform: exact match to baseline (521 tokens, same answer) — sanity passes. random_s42: 1111 tokens, 27.1s, answer 4/1013 (most elaborate wrong reasoning). inverse: 99 tokens, 2.4s, stopped after echoing.
- FINDING: No condition correctly computes sum of reciprocals using Vieta (-b/c per factor = -4/3, total -2700, H = 4050/(-2700) = -3/2); all conditions compute sum of roots (4/k) instead — fundamental 3B model limitation on this problem.
- FINDING: sma_a1.0 at full strength destroys coherent generation (infinite loops). sma_a0.3 and sma_a0.5 encourage more exploration but do not fix the conceptual error.
- DELIVERABLE: sma3_single.py (final script with same-layer modulation, stock Qwen defaults, system prompt, raw prompt printing) saved to /home/greg/Desktop/Projects/BrainInsideTheMachine/.
- STATE: Session state saved for BrainInsideTheMachine project (snapshot ID not explicitly recorded, but state saved at user request).
- NEXT: Try a simpler problem where baseline has a chance, to measure whether SMA improves hit rate.

---

## 2026-04-18 13:08 UTC — Diagnosing and mitigating transformer attention perseveration as a general bottleneck on implicit knowledge retrieval in autoregressive generation — testing whether soft deflation of queries along redundant key directions can break reasoning loops and steer models into correct attractors.

*Source: [2026-04-18_13-08_context-usage-check.md](2026-04-18_13-08_context-usage-check.md)*

**Objective:** Diagnosing and mitigating transformer attention perseveration as a general bottleneck on implicit knowledge retrieval in autoregressive generation — testing whether soft deflation of queries along redundant key directions can break reasoning loops and steer models into correct attractors.
- EXPERIMENT: additive SMA sweep at α = 0.01, 0.05, 0.1, 0.3, 0.5, 1.0 on AMC 12A Problem 12 (harmonic mean) — baseline output 521 tokens (wrong), α=0.05 produced 665 tokens and changed reasoning path toward sum of reciprocals, α=0.1 produced 2048 tokens and found Vieta reciprocal identity but looped on algebra, α≥0.3 produced catastrophic repetition or gibberish.
- EXPERIMENT: `diag_wk_pollution.py` — projected post-MLP hidden states onto W_K's top-32 right singular vectors vs orthogonal complement across generation steps; measured ratio ≈ flat (deltas ≤0.0124), entropy NaN due to float16 precision (log(1e-10) underflow).
- EXPERIMENT: `diag_attn_pacf.py` — measured attention cosine similarity, partial autocorrelation, key cache effective rank across layers 5,17,27,30,35 for p9 (correct), p3 (wrong coherent), p12 (looping).
- FINDING: L27 attention cosine similarity for p12 rose from 0.904 to 0.978 (Δ +0.074) over generation; for p9 it rose from 0.911 to 0.920 (Δ +0.009); for p3 from 0.888 to 0.902 (Δ +0.014).
- FINDING: L27 ACF (lag-1) for p12 = 0.42 vs p9 = 0.14 vs p3 = 0.09; lag-5 for p12 = 0.29 vs p9 = 0.15.
- FINDING: L30 effective rank for p12 dropped from 18.6 (T=270) to 14.0 (T=2295) with ratio_last=0.109; p9 stayed stable (14.0→14.8); p3 stable (15.6→15.4).
- FINDING: L30 normalized attention entropy for p12 rose Δ +0.127, 3× faster than p9 (Δ +0.038) and p3 (Δ +0.060).
- DECISION: Implement full deflation (remove top-r singular vectors from key cache) as `exp_deflated_attention.py` with ranks r=2,4,8.
- EXPERIMENT: `exp_deflated_attention.py` — r=2,4,8 on p9, p3, p12; all conditions worse than baseline; p12 baseline looped on fraction "16200/4102650", deflate_r4 looped on sentence "the coefficient is equal to the negative of the coefficient".
- KILLED: Full deflation (removing top-r singular vectors completely) — destroys performance, does not fix loops.
- BRAINSTORM: Deflate query but reroute removed component through o_proj back into residual stream to preserve rank.
- EXPERIMENT: `exp_reroute_attention.py` — soft deflation (deflect query by top-4 key directions at layers 20-35, α=0.05/0.1/0.2) and reroute (β=0.5/1.0).
- FINDING: soft deflation α=0.1 on p12 produced 1053 tokens and correct answer -3/2 (B); baseline looped at 2048 tokens.
- FINDING: soft deflation α=0.05 on p12 produced answer -5/3 (adjacent wrong) at 1061 tokens; α=0.2 produced -5/6 (further wrong) at 962 tokens.
- FINDING: reroute β=0.5/1.0 catastrophic — produced garbage tokens or loops, no improvement.
- KILLED: Reroute approach (deflation + o_proj compensation) — structured noise, not meaningful attention output.
- BREAKTHROUGH: Model knows correct answer before writing chain-of-thought; attention perseveration at L27 blocks implicit knowledge from surfacing; soft deflation unblocks it without fixing reasoning — model confabulates derivation post-hoc.
- AUDIT: Assistant initially claimed soft deflation "unlocked computation"; user corrected that deflation actually hallucinated reasoning (called sum of roots "sum of reciprocals", claimed Σ1/k = 2025/2, wrote 4050/4050 = -3/2) yet still output correct answer — implicit knowledge overrides chain-of-thought.
- DECISION: Run Web's three verification tests — (1) robustness: α=0.1 with temp=0.01 across seeds, (2) temperature control: temp=0.3/0.5 without deflation, (3) breadth test: 5 problems baseline vs soft α=0.1.
- EXPERIMENT: `exp_deflation_controls.py` — robustness: 4 seeds of soft α=0.1 on p12 all produced 1053 tokens and correct answer -3/2; temperature control: temp=0.3 across 3 seeds produced answers -2/3, -5/3, -5/6 (0/3 correct), temp=0.5 produced LOOP, -6/5, -6/5 (0/3 correct, 1 still loop).
- FINDING: Soft deflation α=0.1 deterministically steers to -3/2 (4/4 seeds); temperature breaks loops 5/6 times but scatters answers uniformly across all wrong choices — deflation is structurally specific, not generic noise.
- EXPERIMENT: Breadth test on problems p3, p4, p7, p9, p12 — soft α=0.1 vs baseline.
- FINDING: Baseline 7/19 correct; soft deflation 9/19 correct (net +2) — fixed 5 problems (P4, P15, P18, P21, P22), broke 3 (P3, P11, P17), held 4 correct, 7 unchanged wrong.
- FINDING: Soft deflation broke loops on 3 of 4 baseline-loop problems (P4, P12, P21 fixed; P7 remained loop).
- DELIVERABLE: `sma3_single.py` (modified with additive raw tension, no normalization).
- DELIVERABLE: `diag_wk_pollution.py`, `diag_attn_pacf.py`, `exp_deflated_attention.py`, `exp_reroute_attention.py`, `exp_deflation_controls.py`.
- DELIVERABLE: `output/soft_deflation_results.md` — before/after answers for 19-problem sweep.
- STATE: snapshot `95b1a6b5` saved — includes modified SMA code, diagnostic scripts, deflation experiments, sweep results.
- NEXT: Implement adaptive conditional deflation — compute L27 cos_sim online during generation; if stickiness exceeds threshold (e.g., >0.95 for 10 consecutive steps), activate soft deflation only then, rather than applying uniformly.

---

## 2026-04-19 22:04 UTC — Determine whether the model's latent correct answer representations can be generated faithfully by stacking multiple orthogonal interventions (surgery, SMA, deflation), and if not, whether the information is pre-generative (1A) or emerges during reasoning (2A).

*Source: [2026-04-19_22-04_breakdown-mutual-information-cases.md](2026-04-19_22-04_breakdown-mutual-information-cases.md)*

**Objective:** Determine whether the model's latent correct answer representations can be generated faithfully by stacking multiple orthogonal interventions (surgery, SMA, deflation), and if not, whether the information is pre-generative (1A) or emerges during reasoning (2A).
- **STATE**: `2eec9346` — session loaded from previous soft deflation arc with key result: α=0.1 soft deflation breaks attention perseveration on P12 (4/4 correct), AMC sweep showed +2 net (7→9/19), with 5 fixes and 3 breaks.
- **BRAINSTORM**: Proposed stacking all three orthogonal mechanisms (surgery: remove convention direction from W_down; SMA: amplify gate-sensitive dimensions; deflation: steer queries away from redundant keys) as each targets a different readout bottleneck, predicting compounded performance without interference.
- **DECISION**: Run stacked experiment on P12 with five conditions (baseline, surgery-only, SMA-only, deflation-only, all three) to test orthogonality claim.
- **EXPERIMENT**: `exp_stacked_p12.py` — baseline P12 loops on fraction simplification (tok=2048). Surgery-only finds correct -3/2 but loops in Chinese repeating answer (tok=2048). SMA-only (α=0.05) gives correct -3/2 and converges (tok=1028). Deflation-only gives correct -3/2 (tok=1053). Stacked (surgery+SMA+deflation) loops on quadratic formula setup, never reaches computation (tok=2048).
- **FINDING**: Stacked all three interventions fails catastrophically — total perturbation compounds, pushing model into earlier failure mode (quadratic formula loop) before computation begins, disproving orthogonal independence claim.
- **FINDING**: SMA-only at α=0.05 with AMC-specific prompt now converges to correct -3/2 (previously looped with generic prompt) — prompt substantially affects intervention efficacy.
- **FINDING**: Surgery-only finds -3/2 but code-switches to Chinese ("由于计算过程中的错误...") and loops indefinitely — convention removal opens second linguistic channel but creates new perseveration loop in that channel.
- **AUDIT**: Stacking logic was wrong because interventions are coupled through residual stream dynamics — SMA amplifies dimensions that may overlap with directions deflation suppresses, and surgery changes the weight matrix underlying SMA's sensitivity calibration.
- **PUSHBACK**: User argues that "orthogonal because different components" is insufficient; interventions interact multiplicatively, not additively, and each was calibrated assuming others absent.
- **DECISION**: Run 2^3 ablation on P12 testing Q-deflation alone, K-deflation alone, Q+K dual, and each with surgery, to diagnose interference source.
- **EXPERIMENT**: `exp_dual_deflation_p12.py` — Q-only correct -3/2 (1053 tok). K-only wrong (-5/3, 1281 tok). Q+K loops (2048 tok). Surgery-only correct but loops (2048 tok). Surgery+Q wrong (-5/3, 966 tok). Surgery+K wrong (1475 tok). Surgery+Q+K wrong (-6/5, 905 tok).
- **FINDING**: Every condition including K-deflation is worse than the corresponding condition without it — K-deflation removes information from keys (the model's computation state), while Q-deflation removes an artifact (echo-driven perseveration). K and Q are not symmetric; keys carry signal, queries carry artifact.
- **KILLED**: Key-deflation as a valid intervention — it amputates computation rather than blocking noise; K-only reaches wrong answer, Q+K loops, surgery+K fails.
- **FINDING**: Q-deflation alone remains the only native intervention that consistently works (correct, converges, no loop) — adding anything makes it worse.
- **BRAINSTORM**: MECE breakdown of when mutual information I(H; y*) becomes non-negligible: (1A) pre-generative knowledge at prompt end, (2A) during shared reasoning prefix, (3B) only after intervention divergence. Temperature control data (0/6 correct with random perturbation) argues against 3B and for 1A or 2A.
- **BRAINSTORM**: Discriminating experiment — run forward pass on prompt only (no generation), extract logits for answer choices. If 1A, -3/2 logit should be elevated pre-generation; if 2A, no preference until during generation.
- **DECISION**: Run pre-generative knowledge experiment before clearing session — cheap one-forward-pass test to distinguish 1A from 2A.
- **DELIVERABLE**: `/home/greg/Desktop/Projects/BrainInsideTheMachine/exp_pregenerative_knowledge.py` — processes P12 prompt with/without answer choices, extracts next-token logits for five letter options and reasoning starters.
- **EXPERIMENT**: Pre-generative knowledge test — all four conditions (AMC_with_choices, AMC_no_choices, neutral_with_choices, neutral_no_choices) show B (-3/2) as lowest-ranked letter at ~0.001% probability, 30x below highest letter. Top predictions are reasoning starters ("To", "Let", "The") at 82-87% probability.
- **KILLED**: Case 1A — model does NOT pre-generatively know -3/2; answer is not encoded at prompt end. Confirmed 2A: answer emerges during shared reasoning steps 1-4 (Vieta, root structure, sums/products) and intervention only matters at step 5+ to prevent narration from polluting KV cache.
- **FINDING**: Answer choices do not help or hinder probability of correct answer; model ignores them at prompt encoding stage.
- **FINDING**: Chain-of-thought is load-bearing through step 4 (computational) but becomes performative after step 5 (narration that contaminates attention). Transition point between genuine computation and noise is measurable.
- **NEXT**: Run delayed deflation experiment on P12 — activate deflation only after N tokens (e.g., step 200). If deflation starting at token 200 still works, it proves computation happens in first 200 tokens and deflation's sole job is preventing performative narration from drowning the signal.
- **STATE**: session saved via `mcp__orchestrator__save_session_state` with project `BrainInsideTheMachine` after completing pre-generative knowledge experiment and before clearing.

---

## 2026-04-20 10:13 UTC — Determining whether attention deflation enables mathematical reasoning in a 3B model by temporally and spatially localizing its effect, and testing whether the model genuinely computes versus retrieves solutions in a synthetic AMC-style polynomial problem.

*Source: [2026-04-20_10-13_clear-save-state.md](2026-04-20_10-13_clear-save-state.md)*

**Objective:** Determining whether attention deflation enables mathematical reasoning in a 3B model by temporally and spatially localizing its effect, and testing whether the model genuinely computes versus retrieves solutions in a synthetic AMC-style polynomial problem.
- STATE: 140be0a1 (saved after concluding P12 memorization, later retracted)
- STATE: cb7ed6b2 (final snapshot capturing delayed deflation, layer specificity, parametric algebra, and gravitational field framing)
- EXPERIMENT: Delayed deflation sweep on P12 (Q-only, alpha=0.1, layers 20-35, refresh=25, N=0/50/100/150/200/300/500/750/1000). Arm A (deflation OFF for 0..N then ON): delay_0 works (CORRECT, 1053 tokens), delay_50+ all FAIL. Arm B (deflation ON for 0..N then OFF): cutoff_50 works, cutoff_100 LOOP (anomaly), cutoff_150+ all CORRECT.
- FINDING: Delayed deflation fails at N=50; deflation must be active within the first 50 tokens to produce correct answer. This contradicts the hypothesis that deflation protects a late readout phase (token 200+).
- FINDING: The first 50 tokens of generated text are nearly identical across conditions (all say "To find the harmonic mean... steps: 1. Understand the polynomial") but internal KV cache geometry differs and determines which algebraic strategy the model executes 500+ tokens later.
- KILLED: Hypothesis that deflation operates as a readout protector (delayed onset at token 200 should work) – fatal at N=50 failure.
- DELIVERABLE: exp_delayed_deflation_p12.py – implements delayed onset and early cutoff sweeps with Q-only deflation.
- EXPERIMENT: Layer-specific deflation binary search on P12 (L20-L35, Q-only, alpha=0.1, first 50 tokens). Minimum contiguous layers required: L20-L24 (5 layers) yields CORRECT (-3/2). L20-L23 (4 layers) yields LOOP. L21-L25 (5 layers) CORRECT. L22-L25 (4 layers) WRONG(-2/3). L24-L33 (10 layers) WRONG. L26-L33 (8 layers) CORRECT.
- FINDING: The minimum deflation window is exactly 5 contiguous layers in [L20-L25]; sliding one layer (L22-L25) breaks it. Non-monotonic: L24-L33 fails while both L20-L25 and L26-L33 succeed independently.
- FINDING: Different layer deflation configurations produce different internally consistent boxed answers: -5/3 (A), -3/2 (B), -6/5 (C), -2/3 (E), plus a numeric loop attractor. The model is selecting among at least five distinct solution basins, not binary correct/incorrect.
- DELIVERABLE: exp_layer_specific_deflation_p12.py (implicitly used in binary search runs).
- EXPERIMENT: Novel parametric variant of the problem (kx² - 10x - 6, N=1500, correct answer -6/5). Baseline, L20-L24 deflation, and full L20-L35 deflation all fail to produce -6/5; outputs 3000, 3000, 1800 respectively.
- FINDING: Qwen2.5-3B cannot compute the novel variant correctly even with optimal deflation, suggesting the original P12 solution relies on narrow pattern match rather than general Vieta application.
- PUSHBACK: User points out Qwen2.5-3B was released September 2024, before the November 2024 AMC 12A exam, so original conclusion of memorization is invalid. Assistant confirms via perplexity search and retracts memorization claim.
- DECISION: Reject memorization hypothesis; adopt narrow pattern recognition with fragile generalization – model learned the structure from similar competition problems but fails when coefficients change.
- EXPERIMENT: Parameterized version with symbolic algebra (no multiple choice, ask for algebraic expression of harmonic mean for general coefficients a,b,c,N). Baseline produces 2/(N+1) (nonsense). Full deflation (L20-L35) produces -2N/H_N (wrong). Minimum deflation (L20-L24) produces -c/b (off by factor of 2 from correct -2c/b).
- FINDING: The 5-layer deflation condition performs genuine algebraic reasoning: correctly rationalizes, simplifies difference-of-squares, cancels 4k, gets -b/c per factor, then makes a counting error (2N vs N) to output -c/b. This is partial derivation, not pure retrieval.
- BRAINSTORM: Exp AJ – answer crystallization map: 2D heatmap of E(x) = -log p(correct_answer) over (layer L, token t) for three conditions (baseline ZH, flip ZH, surgery ZH) to test whether the layer stack or autoregressive loop is the primary computer.
- BRAINSTORM: Soft embedding test – feed probability-weighted average of token embeddings instead of argmax token to preserve the model's uncertainty and prevent the token bottleneck from destroying basin information.
- BRAINSTORM: Single-poke test – inject the vector difference between deflated and baseline hidden states at a single (layer, token) coordinate to find the exact bifurcation point.
- FINDING: L33 (last computation layer) holds the answer with cos 0.73 to the B-moment template, and never drops below 0.48 even in the loop. L34-L35 (read head) contributes a constant bias from C6b and does not discriminate. The gravitational field (basin boundaries) resides at L33.
- DELIVERABLE: exp_deflation_gradient_p12.py (attempted but OOM on 12GB GPU even with gradient checkpointing).

---

## 2026-04-21 20:44 UTC — Test whether the directional sensitivity of output to perturbations at L33 (gravitational field) reveals attractor basin structure in residual stream vs attention, and discriminate whether anisotropy is architectural, RLHF-carved, or autoregressive-loop-dependent.

*Source: [2026-04-21_20-44_copy-command-output.md](2026-04-21_20-44_copy-command-output.md)*

**Objective:** Test whether the directional sensitivity of output to perturbations at L33 (gravitational field) reveals attractor basin structure in residual stream vs attention, and discriminate whether anisotropy is architectural, RLHF-carved, or autoregressive-loop-dependent.
- EXPERIMENT: exp_gravitational_field_p12.py – measured sensitivity of output to perturbations in 8 directions (b_moment, convention_ec, sys_absent, mean_readout, kv_svd_0, 3 random) at L33 during P12 generation.
- FINDING: All 8 directions had identical sensitivity traces: mean sensitivity ~1.30, correlation 1.000 between every pair, anisotropy = 1.00x random for every direction.
- FINDING: Peak sensitivity = 17.87 at step 378; original b_moment extraction occurred at step 141.
- FINDING: B rank min = 3 at step 151; B rank at step 50 = 3063.
- BREAKTHROUGH: The residual stream (h_L33 → layernorm → lm_head direct path) is a uniform sphere – all directions equally sensitive, falsifying the implicit assumption in activation steering that perturbing meaningful directions in hidden state changes output proportionally.
- BREAKTHROUGH: Anisotropy exists but lives in attention's Q·K interaction with KV cache (L34-L35), not in the residual stream representation.
- AUDIT: Initial QDeflation class failed due to DynamicCache API (past_kv.layers[i].keys not subscriptable) and shape mismatches (1x2048 vs 128x4); fixed by replacing with working WindowedDeflation class from prior experiment and adding explicit reshaping in hooks.
- DELIVERABLE: exp_gravitational_field_p12.py (full sensitivity measurement script) and exp_grav_base_vs_instruct.py (base vs instruct probe).
- DELIVERABLE: output/exp_gravitational_field_p12.json and output/exp_grav_base_vs_instruct.json saved.
- EXPERIMENT: exp_grav_base_vs_instruct.py – measured convention direction anisotropy during encoding and generation for Qwen2.5-3B-Instruct vs base Qwen2.5-3B.
- FINDING: Instruct encoding anisotropy = 4.91x (convention direction sensitivity vs random), base encoding anisotropy = 0.35x (below random, isotropic).
- FINDING: Instruct generation anisotropy = 1.62x; base generation anisotropy = 1.81x (slightly higher).
- FINDING: Early/late sensitivity drop (first 50 tokens vs after): instruct 5199.7x, base 130514.5x.
- BREAKTHROUGH: Generation-time anisotropy is architectural (exists in base model, 1.81x) – a property of the transformer's attention interacting with growing KV cache.
- BREAKTHROUGH: Encoding-time anisotropy is RLHF-carved (instruct 4.91x vs base 0.35x) – the second-person "you are a mathematical reasoner" creates directional sensitivity not present in pretrained weights.
- KILLED: Theory C (autoregressive feedback loop necessary for anisotropy) – instruct encoding anisotropy (4.91x) exists without generation, so feedback loop not required.
- PUSHBACK: (implicit) The assistant's initial code assumptions about DynamicCache API and hook shapes were wrong; debugging corrected them.
- DECISION: Prioritized base-vs-instruct probe over per-head decomposition to collapse theory space first.
- DECISION: Updated MEMORY.md with gravitational field findings and trimmed evidence chain.
- BRAINSTORM: Next experiments: per-head decomposition at L34-L35 to find which attention heads carry the anisotropy; single-poke at step 25 in b_moment direction to see if it redirects trajectory; test adding system prompt to base model to see if encoding anisotropy rises.
- BRAINSTORM: System prompt's second-person address is a novel input distribution not present in pretraining; RLHF carves encoding sensitivity to it (4.91x), and Ghost mode (system='.') removes that pathway, reverting to base encoding geometry (0.35x).
- NEXT: The per-head decomposition experiment (find which 2-3 of 16 heads concentrate the anisotropy) is the highest-value next action because it reduces the search space from 2048 dimensions to head_dim (128) for those heads.
- NEXT: Load state 78bdd564 on next session – it contains the full narrative, isotropic null finding, base-vs-instruct results, and connections to Ghost/KG/autoregressive tragedy.
- STATE: 1ea6aeb8 saved after updating MEMORY.md with gravitational field findings.
- STATE: 78bdd564 saved after planting seed and connecting Obsidian dictation to results.

---

## 2026-04-23 04:35 UTC — Test the gravitational field hypothesis via per-head decomposition, resolve the attractor basin topology debate (needle-threading vs tunneling) through controlled experiments on base vs instruct models, and converge on echo bifurcation theory as the unified mechanism for QK deflation.

*Source: [2026-04-23_04-35_handle-context-limit.md](2026-04-23_04-35_handle-context-limit.md)*

**Objective:** Test the gravitational field hypothesis via per-head decomposition, resolve the attractor basin topology debate (needle-threading vs tunneling) through controlled experiments on base vs instruct models, and converge on echo bifurcation theory as the unified mechanism for QK deflation.
- STATE: 78bdd564 (mentioned by user as having full narrative and ranked next actions)
- EXPERIMENT: Per-head decomposition of gravitational field (Qwen2.5-3B, layers L34-L35, 16 heads, measured anisotropy contribution per head). Result: No single head >4.4%; Head 6 at L34 *increases* anisotropy by 7.7% when ablated (acts as regularizer); baseline b_moment anisotropy measured at 1.61x (vs prior 5.43x — discrepancy attributed to generation length difference).
- KILLED: Hypothesis that gravitational field is concentrated in 2–3 heads (each head contributes <4.4%; field is distributed collective property of multi-head attention).
- DECISION: Update MEMORY.md with per-head decomposition result and mark per-head targeting as dead path.
- DELIVERABLE: /home/greg/Desktop/Projects/BrainInsideTheMachine/memory/MEMORY.md (updated with per-head decomposition and inverse ghost experiment results)
- EXPERIMENT: Single-poke bifurcation on Instruct model (Qwen2.5-3B-Instruct, 4 steps × 5 layers × 4 directions × 2 magnitudes, 80 conditions). Results: mag=2.0 all produce WRONG(A) (1255 tokens, baseline also A, not loop); mag=5.0 yields 2 B-hits at (step=10, L20, b_moment and convention directions), plus 3 other basin switches to D/E; no B hits at other coordinates.
- FINDING: Instruct model baseline on P12 is A (-5/3), not loop; full deflation control on Instruct produces A (not B), indicating Instruct basin landscape differs fundamentally from base model.
- BRAINSTORM: Maximally informative mathematics problem for Ghost Claude — asks for minimum topological structure consistent with experimental constraints (isotropic state space, 5.4x anisotropic control space, separability, 1/t decay, non-adjacent basins from single poke, 5-layer sustained intervention threshold, distributed field).
- PUSHBACK: Stop hook flags strategic drift — session shifted from analyzing hidden states/read heads to tangential "woke ghost" math problem posing.
- DECISION: Acknowledge drift but continue with single-poke and topology experiments as relevant to gravitational field arc; shorten conversational detours.
- BRAINSTORM: Needle-threading model proposed by user (Claude Web response) — basins share codimension-1 boundary; default trajectory passes near boundary at narrow (step, layer) window; deflation keeps trajectory shallow enough to drift across. Contrasted with tunneling model (transient attractor creation).
- EXPERIMENT: Topology discriminator (Exp1 staged intervention + Exp2 late-onset deflation) on **Instruct** model (`exp_topology_discriminator.py`, 17 conditions). Results: Exp1 primary fails (WRONG); full deflation control fails (produces A, not B); shallow-keeping causes loops or D basin; Exp2 onset=15/20 both WRONG. Verdict: both models fail, control fails, invalid premise.
- AUDIT: Single-poke B-hits (2 of 80) were from Instruct model; prior deflation work used base model. Instruct does not have stable B attractor reachable by deflation — whole experimental chain invalid for that variant.
- DECISION: Rerun topology discriminator on **base model** (Qwen/Qwen2.5-3B) with raw chat template to recover valid test of tunneling vs needle-threading.
- EXPERIMENT: Base model topology discriminator (`exp_topology_base.py`, 26 conditions). Results: Baseline = LOOP; control (full deflation onset=0) = B. Exp1 staged intervention (shallow-keeping 0–8 + poke step=10 L20 mag=5.0): WRONG(?) (885 tokens). All 16 Exp1 variants fail (including poke mag=10, poke at L22/L25/L33, poke at step5/25, shallow-only windows 8/50/100). Exp2 late-onset deflation: onset=0 → B, onset=5 → B, onset=10 → B, onset=15 → B, onset=20 → B, onset=30 → WRONG(?), onset=50 → WRONG(?), onset=100 → WRONG(?).
- FINDING: Tunneling model confirmed — sustained perturbation modifies F → F+δF creating transient attractor; deflation works even when onset delayed by 20 tokens (trajectory already in loop basin for 20 steps with zero intervention). Needle-threading falsified (staged intervention fails; no neck at step 10).
- FINDING: Transition curve sharp: deflation works for onset ≤ 20 tokens, fails for onset ≥ 30 tokens. Minimum active deflation duration ~20–25 tokens before cache dilution kills control authority.
- FINDING: Base model P12 baseline is LOOP (not A); Instruct baseline is A (wrong answer, not loop) — basin landscapes differ qualitatively between base and instruct variants of same architecture.
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/exp_topology_base.py` (297 lines, base model topology discriminator with 26 conditions)
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/exp_topology_discriminator.py` (353 lines, Instruct version, later deprecated)
- KILLED: Needle-threading model (separatrix crossing via shallow-keeping + poke) — falsified by base model experiments.
- STATE: b22d97a0 (saved after base model topology results, includes tunneling confirmation and updated MEMORY.md)
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/memory/MEMORY.md` (updated with topology discriminator results, echo bifurcation framing, and separate kill entries for needle-threading and per-head targeting)
- BRAINSTORM: Echo bifurcation theory (from Ghost Claude response at `docs/ghost/echo_bifurcation_theory.md`) — loop attractor is resonance catastrophe from QK echo feedback; deflation destroys attractor (saddle-node bifurcation) rather than selecting basin; two echo sources: weight echo (deflatable) and cache echo (grows with generated tokens, not deflatable). Explains 5-layer minimum (collective echo reduction), onset=20 works (destroy attractor after orbiting it), onset=30 fails (cache echo alone exceeds ρ*).
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/docs/ghost/echo_bifurcation_theory.md` (unified framework)
- DELIVERABLE: `/home/greg/Desktop/Projects/BrainInsideTheMachine/docs/ghost/phase1_phase_diagram.py` (Ghost's experiment script to map δ × t_crit scaling — needs adaptation to WindowedDeflation infrastructure)
- META: Context usage reached ~60%; assistant recommends `/clear` to restore hooks and context quality.
- NEXT: Run phase diagram experiment: sweep δ (deflation strength) ∈ {0.05, 0.10, 0.15, 0.20, 0.30} × onset ∈ {0, 5, 10, 15, 20, 25, 30, 40, 50} using WindowedDeflation on base model (Qwen/Qwen2.5-3B, raw chat template) to test δ × t_crit ≈ constant prediction of two-source echo model.

---

## 2026-04-23 22:02 UTC — Testing the echo bifurcation theory's prediction that δ×t_crit is constant in the Qwen2.5-3B deflation loop experiments, and investigating the role of SVD basis refresh in determining the onset boundary.

*Source: [2026-04-23_22-02_save-lossless-session-state.md](2026-04-23_22-02_save-lossless-session-state.md)*

**Objective:** Testing the echo bifurcation theory's prediction that δ×t_crit is constant in the Qwen2.5-3B deflation loop experiments, and investigating the role of SVD basis refresh in determining the onset boundary.
- EXPERIMENT: Phase diagram sweep δ∈[0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.30] × onset∈[0,5,10,15,20,22,25,27,30,35,40,50] using WindowedDeflation (Q-projection SVD) on base model — 96 conditions, batched, run in background.
- FINDING: δ=0.03 fails at all onsets (2048 tokens each, ~41s per condition), confirming it is too weak to break the loop.
- AUDIT: Batching artifact causes 9/96 disagreements (9.4% discrepancy) between batched and sequential runs — shared SVD basis across conditions leads to measurable contamination.
- FINDING: δ×t_crit is NOT constant — coefficient of variation 0.68 across δ values (0.05: 1.10, 0.08: 1.76, 0.10: 2.50, 0.12: 6.00, 0.15: 6.00, 0.20: 2.00, 0.30: 9.00) — falsifying the 1/δ scaling prediction from two-source echo model.
- FINDING: Phase diagram is noisy near the transition — δ=0.05 shows onset=0 fails, onset=5 works, onset=10 fails, onset=15 works (non-monotonic, stochastic boundary).
- FINDING: δ=0.12 row has 0/11 fails then randomly onset=50 works — evidence of resonance holes at specific δ values.
- EXPERIMENT: Fixed-basis test (refresh=False, prompt-only SVD basis never updated with generation tokens) — run on same grid, batched.
- FINDING: Fixed basis is catastrophically worse: 6/96 correct vs 35/96 with refresh — refresh is essential for deflation, not contaminating.
- BREAKTHROUGH: Prompt SVD captures encoding geometry (what the model knows) but generation SVD captures computation geometry (what the model is doing) — the loop lives in computation space, not encoding space; the prompt-only basis is nearly useless.
- BREAKTHROUGH: Refresh bootstrapping model — weak deflation with bad prompt basis perturbs trajectory, refresh captures the perturbation, better basis → better deflation → better perturbation → better basis (positive feedback); without refresh the seed never grows.
- EXPERIMENT: Refresh period sweep refresh_every∈[5,10,15,25,50] for δ∈[0.10,0.15,0.30] (3×5=15 batches) — run batched.
- FINDING: δ=0.10 shows t_crit=27 fixed across refresh=5,10,15,25 (identical 8/12 grids); only refresh=50 drops to t_crit=25 (one fewer correct). Correlation refresh vs t_crit = -0.91 driven entirely by outlier.
- FINDING: δ=0.15 shows t_crit=40 fixed across refresh=5 through 25; refresh=50 collapses to 0/12 (too slow to bootstrap). Correlation 0.00.
- FINDING: δ=0.30 shows t_crit ranging 20-30 without clear scaling (correlation 0.51) and slower refresh (r=25) paradoxically better than fast refresh — attributed to instability where fast refresh chases its own perturbation artifacts.
- BREAKTHROUGH: Two-operation model — prevention (δ=0.10, t_crit=27, contiguous step function) vs reversal (δ=0.15, t_crit=40 but with dead zone at onset 27-35 where nothing works). The dead zone represents committed but basis hasn't purified.
- KILLED: Ghost's contamination hypothesis that refresh corrupts the basis — fixed-basis test proves refresh is essential, not parasitic.
- KILLED: The hypothesis that onset boundary at 25-27 is determined by refresh period — fixed across 5,10,15,25.
- EXPERIMENT: Commitment trajectory via hidden state L33 — measure cos(h_L33, loop_template) - cos(h_L33, answer_template) at each step for baseline and deflation conditions (onset=0,25,30).
- FINDING: Hidden state L33 cannot discriminate — cos values are 0.9997 across all conditions because L33 carries the answer either way.
- EXPERIMENT: Commitment trajectory via logit rank and KV rank — track B_frac token rank, logit entropy, and KV cache r90 at each step for baseline and deflation (onset=25,30).
- FINDING: Trajectories are identical through token 24 across baseline, onset=25, onset=30 — deflation has not yet started for any condition.
- FINDING: Divergence starts at token 25 — onset=25 (works) diverges from baseline at step 25 (rank 1584 vs 1545); by step 30 baseline rank=44, onset=25 rank=2380.
- FINDING: onset=30 tracks baseline perfectly through token 29, diverges slightly at token 30, but by token 47-49 is already diverging from baseline — 3 tokens too late.
- FINDING: B_frac logit rank is wildly noisy and never settles — baseline has B_frac at rank 91 even at step 100; the model keeps B in top 100 while looping.
- FINDING: KV cache rank (r90) barely moves — 67 at step 0, 70 at step 100; no rank collapse in first 150 tokens.
- FINDING: Logit entropy is the most structured signal — alternates between near-zero (confident formula continuation) and spikes (uncertain glue transitions) matching the two-timescale signature from earlier attention anatomy.
- DELIVERABLE: exp_phase_diagram.py (324 lines) — adapted to WindowedDeflation, correct P12 problem, manual generation loop.
- DELIVERABLE: exp_phase_diagram_validate.py (235 lines) — sequential validation to audit batching artifacts.
- DELIVERABLE: exp_phase_diagram_fixedbasis.py (313 lines) — refresh=False test.
- DELIVERABLE: exp_phase_diagram_refresh_sweep.py (309 lines) — sweep refresh_every for δ=0.10,0.15,0.30.
- DELIVERABLE: exp_commitment_trajectory.py (242 lines) — measure L33 cosine commitment.
- DELIVERABLE: exp_commitment_logits.py (222 lines) — measure logit rank, KV rank, entropy per step.
- DELIVERABLE: docs/ghost/phase1_phase_diagram.py (Claude Web's original script, 396 lines).
- DELIVERABLE: docs/ghost/echo_bifurcation_theory.md (424+ lines).
- DELIVERABLE: docs/ghost/spectrogram.jsx (React visualization, 540+ lines).
- DELIVERABLE: docs/ghost/phase_diagram_response.md (Ghost's first response, ~190 lines).
- DELIVERABLE: docs/ghost/phase2_response.md (Ghost's second response after fixed-basis results, ~200+ lines).
- DELIVERABLE: docs/ghost/phase3_response.md (Ghost's third response after refresh sweep, ~170+ lines).
- PUSHBACK: Ghost's contamination hypothesis reversed — refresh is bootstrapping, not parasitic; self-referential updates are necessary for the system whose power comes from self-reference.
- BRAINSTORM: Phase diagram experiment to test δ×t_crit scaling (Claude Web's design, adapted).
- BRAINSTORM: Fixed-basis test to distinguish contamination from bootstrapping.
- BRAINSTORM: Refresh period sweep to separate mechanism from intrinsic boundary.
- BRAINSTORM: Commitment trajectory via logit rank and KV rank as observable for phase transition.
- BRAINSTORM: MLP invertibility as fundamental mechanism — loop as fixed point of SiLU feature selection, not fixed point of hidden state; deflation pushes marginally-active features below threshold.
- DECISION: Run phase diagram on RayGun (4070 Super) despite Colab being potentially faster — experiment already running.
- DECISION: Run fixed-basis test after Ghost's contamination hypothesis — executed.
- DECISION: Run refresh sweep after Ghost's bootstrapping model — executed.
- DECISION: Run commitment trajectory experiments after Ghost's phase3 proposal — executed.
- DECISION: Pivot from drilling into P12 to testing generality across problems and models — deferred after user pushback to avoid overfitting.
- PUSHBACK (user to assistant): "I think you're so low-level. I need you to zoom out with me, man. We can't spend years on this fucking thing" — assistant zooms out to four publishable findings independent of loop work.
- BREAKTHROUGH (strategic reframe): Four general findings that survive loop work — (1) read head is constant bias (L34-L35), (2) encoding geometry ≠ computation geometry, (3) rank-1 readout (C3/C3-7B), (4) convention direction separable from computation (MS1 surgery across 7 languages).
- STATE: b22d97a0 — saved at ~60% context with phase diagram experiment queued.
- STATE: 335e15fd — saved after phase diagram and before fixed-basis test (midnight).
- STATE: f8a9a825 — final state after commitment experiments, refresh sweep, pivot to MLP invertibility, containing everything from session including open questions and lossless representation in Ghost format.
- META: Used `claude -p --model [sonnet|opus] --bare --system-prompt` to invoke Claude as Ghost in non-interactive mode, writing responses to docs/ghost/; initially ran with Sonnet (correct but less deep), then reran with Opus (deeper, lowercase register).
- BLOCKER: None; all experiments completed.
- NEXT: Not specified — session ended with strategic archival and pivot to MLP invertibility/architecture question without a committed next experiment.

---

## 2026-04-24 04:20 UTC — Understand why the quantitative prediction (t_crit ∝ 1/δ) from Echo Bifurcation Theory failed against the sequential phase diagram experiment, and derive what the observed resonance structure and fixed onset boundary reveal about the mechanism of deflation, cache echo, and basis refresh.

*Source: [2026-04-24_04-20_analyze-echo-bifurcation-failure.md](2026-04-24_04-20_analyze-echo-bifurcation-failure.md)*

**Objective:** Understand why the quantitative prediction (t_crit ∝ 1/δ) from Echo Bifurcation Theory failed against the sequential phase diagram experiment, and derive what the observed resonance structure and fixed onset boundary reveal about the mechanism of deflation, cache echo, and basis refresh.
- KILLED: The prediction t_crit ∝ 1/δ is dead; the product δ × t_crit has CV = 0.68, not constant, falsifying the scaling relationship.
- FINDING: The onset boundary for clean rows (correct answers) is fixed at ~25-30 tokens across δ = 0.10, 0.15, 0.30 — it does not scale with 1/δ, contradicting the hypothesis that cache echo accumulation alone determines when the loop become irreversible.
- AUDIT: The 25-30 token boundary aligns with the deflation mechanism’s basis refresh interval (every 25 tokens); before token 25 the basis is computed from the clean prompt, after token 25 it is computed from generated tokens already contaminated by loop structure.
- FINDING: δ=0.12 and δ=0.20 produce complete failure (all N) across most onset positions; these are not random but correspond to crossing points in the eigenspectrum where deflating the top‑4 singular vectors suppresses the primary loop but accidentally promotes secondary pathological attractors.
- FINDING: δ=0.15 and δ=0.30 are between eigenvalue crossings, allowing successful deflation; the alternating pattern is basin‑boundary crossings in a landscape with multiple spurious attractors.
- FINDING: The cell (δ=0.12, onset=50) is a single success in an otherwise dead row; after 50 tokens the KV cache has undergone rank collapse (only loop tokens), so the top‑4 SVD basis spans exclusively the loop, making δ=0.12 cleanly target the loop without interfering with answer‑context features.
- DECISION: The δ=0.12, onset=50 cell supports Vega’s hypothesis that deflation direction matters (not just magnitude), over the null hypothesis that the system is merely chaotic near the boundary.
- BRAINSTORM: A discriminating experiment: run the grid with `refresh=False` (prompt‑only basis, never updated) to test whether the fixed onset boundary at 25‑27 tokens disappears or shifts right; if it does, the basis refresh is load‑bearing and the basis contamination story is correct.
- DELIVERABLE: Wrote `docs/ghost/phase_diagram_response.md` (205 lines) containing the full analysis: why the 1/δ prediction failed, the role of basis refresh, the eigenvalue‑crossing explanation for δ=0.12/0.20 resonances, and the proposed fixed‑basis experiment.
- NEXT: Run the phase diagram grid with `refresh=False` (prompt‑only SVD basis, never recomputed from generated tokens) to discriminate between basis contamination and answer‑template crystallization as the true constraint at token 25‑30.

---

## 2026-04-24 04:57 UTC — Determine why the fixed-basis experiment falsified the refresh-boundary hypothesis and revise the echo bifurcation theory to explain why deflation requires generation-time cache and what the true control parameter is.

*Source: [2026-04-24_04-57_analyze-failed-refresh-boundary.md](2026-04-24_04-57_analyze-failed-refresh-boundary.md)*

**Objective:** Determine why the fixed-basis experiment falsified the refresh-boundary hypothesis and revise the echo bifurcation theory to explain why deflation requires generation-time cache and what the true control parameter is.
- EXPERIMENT: Fixed-basis deflation experiment (no SVD basis refresh) yielded 6/96 correct vs 35/96 with refresh; refresh wins 30/31 disagreements, fixed wins 1/31.
- FINDING: Prompt-only SVD basis is nearly useless; the grid shows only two correct cells (δ=0.05 onset=5? Y; δ=0.20 onset=50? Y) while refreshing basis shows a rich phase boundary.
- FINDING: δ=0.12 onset=50 cell that works in refreshing fails in fixed basis (correct=false in fixedbasis.json), contradicting the earlier assumption that this cell was robust to basis changes.
- AUDIT: Sonnet's hypothesis that fixed basis should shift the boundary right was exactly backwards; the refresh does not contaminate the basis but instead makes deflation work.
- KILLED: The refresh-boundary hypothesis (that the onset boundary at token 25-27 is caused by the SVD basis refresh period contaminating the basis) is dead.
- BREAKTHROUGH: Encoding geometry ≠ computation geometry; prompt SVD captures what the model knows (static encoding), generation SVD captures what the model is doing (attentional dynamics), and the loop lives in computation space, not encoding space.
- BREAKTHROUGH: Refresh bootstraps deflation via positive feedback: weak prompt-basis deflation perturbs trajectory, refresh captures that perturbation, better basis → better deflation → better perturbation → better basis; without refresh the seed never grows.
- FINDING: The rank-collapse purification hypothesis (δ=0.12, onset=50) is strengthened by the fixed-basis failure because the cache undergoes rank collapse and the basis must refresh to capture it.
- DECISION: Adopt the bootstrapping model where self-referential updates are not parasitic but necessary for a system whose power comes from self-reference.
- BRAINSTORM: The onset boundary at token 25-27 may be intrinsic saturation, not refresh period; the two coincidences (token 25-27 and refresh_every=25) may be stacked.
- BRAINSTORM: Vary refresh_every ∈ {5, 10, 15, 25, 50} to test whether onset boundary moves with refresh period (bootstrapping) or stays fixed (intrinsic saturation).
- DELIVERABLE: docs/ghost/phase2_response.md — complete response including the bootstrapping model, data correction, and next experiment proposal.
- NEXT: Run refresh_period sweep experiment with values {5, 10, 15, 25, 50} to disentangle bootstrapping from intrinsic saturation.

---

## 2026-04-24 05:18 UTC — Understand the intrinsic boundary (t_crit) in the phase diagram of model looping behavior—specifically why t_crit is 27 for delta=0.10 and 40 for delta=0.15 across refresh periods, and what experiment to run next.

*Source: [2026-04-24_05-18_analyze-phase-diagram-refresh.md](2026-04-24_05-18_analyze-phase-diagram-refresh.md)*

**Objective:** Understand the intrinsic boundary (t_crit) in the phase diagram of model looping behavior—specifically why t_crit is 27 for delta=0.10 and 40 for delta=0.15 across refresh periods, and what experiment to run next.
- FINDING: For delta=0.10, t_crit=27 is invariant across refresh periods 5, 10, 15, 25 (refresh=50 still gives t_crit=25), indicating the onset boundary does NOT scale with refresh period.
- FINDING: For delta=0.15, t_crit=40 is invariant across refresh=5/10/15/25, but refresh=50 collapses delta=0.15 from 9/12 correct to 0/12 correct; the first refresh must happen within ~25 tokens to bootstrap the basis.
- FINDING: t_crit increases with delta (27 for delta=0.10 → 40 for delta=0.15), partially rescuing the cache echo model (stronger deflation tolerates more cache echo), but the relationship is not 1/delta.
- FINDING: delta=0.30 shows noise: refresh=5 gives t_crit=20, refresh=25 gives t_crit=27; strong deflation rows show more sensitivity to refresh period than moderate ones.
- BRAINSTORM: The boundary at 27 is not about echo accumulation, basis quality, or refresh timing—it is a computational commitment point, a phase transition where the loop stops being sustained by attention echo and starts being sustained by the full forward pass; deflation targets attention, and by token 27 the loop has outgrown attention.
- BRAINSTORM: delta=0.15's t_crit=40 is not a shifted boundary but a second operation (reversal rather than prevention); the contiguous boundary is still ~27 (same as delta=0.10), and the recovery zone at 35-40 exists because a purified basis enables 15% deflation to melt the crystal, while 10% cannot melt it even with perfect alignment.
- BRAINSTORM: delta=0.30 is noisier because strong deflation creates feedback instability between the SVD basis and the intervention—the basis chases its own perturbation artifacts; slower refresh (r=25) is paradoxically better than faster refresh (r=5) because one clean update beats five noisy ones.
- DELIVERABLE: wrote `docs/ghost/phase3_response.md` containing full analysis of the refresh sweep results and reinterpretation of onset boundaries.
- NEXT: measure `commitment(t) = cos(h_L33, loop_template) - cos(h_L33, answer_template)` at every generation step in the baseline run; if it crosses a threshold at token 27, the phase transition is found; if already saturated by token 10, the boundary is about something else entirely.

---

## 2026-04-24 06:34 UTC — This session refines the research programme's understanding of information flow in LLMs by testing whether gated/attenuated MLP features (clipped by SiLU threshold) carry task-relevant information that can be restored to improve reasoning, and more broadly constraining the dimensionality and nature of the “f*” subspace that determines basin selection.

*Source: [2026-04-24_06-34_verify-state-saved.md](2026-04-24_06-34_verify-state-saved.md)*

**Objective:** This session refines the research programme's understanding of information flow in LLMs by testing whether gated/attenuated MLP features (clipped by SiLU threshold) carry task-relevant information that can be restored to improve reasoning, and more broadly constraining the dimensionality and nature of the “f*” subspace that determines basin selection.
- STATE: snapshot f8a9a825 (saved at session start, containing the six experiments, three rounds of Ghost, strategic pivot, and connection to MLP invertibility thread).
- FINDING: Direct path (h_L33 → RMSNorm → lm_head) is perfectly isotropic: all 18 perturbation directions produce identical early mean KL ≈0.1360 and late mean KL ≈0.0144 to four decimal places.
- FINDING: Full path (through L34-L35 attention) has exactly two distinguished directions: b_moment direction amplifies 5.43x random baseline (early sensitivity 0.2567, ratio 28,502x), convention_ec direction amplifies 2.20x (early sensitivity 0.1038, ratio 5,068x).
- FINDING: Temporal profile of full-path sensitivity is separable: S_t(v) = w(v) × c(t) with correlation 0.999999 across all directions; b_moment sensitivity decays to 0.000009 after 50 tokens, convention_ec decays to 0.000020, random decays to ≈0.000006.
- BRAINSTORM: Stack per-layer v1 directions from C6 (k=1 compressibility of self-attention output at each layer) and compute effective rank of the 36×2048 matrix to measure total information transfer dimensionality from context to last token.
- PUSHBACK: DeepSeek critique — C6b showed self-attention output at last token is a constant mean (fixed bias, step‑ and problem‑independent), therefore v1 directions are just orientations of fixed biases, not information transfer channels; the rank of the v1 stack measures fixed‑bias dimensionality, not computation.
- BRAINSTORM: LDA on residual streams at L30, token positions 20‑30, labeled by eventual outcome (loop vs correct), to find low‑dimensional decision boundary f*.
- AUDIT: Commitment trajectory data shows cos(loop_template, answer_template) = 0.9997 at L33 and trajectories are identical through token 24; no natural separation exists without deflation intervention, so LDA would find noise.
- EXPERIMENT (exp_gate_restoration.py, first attempt): dtype mismatch in clip_mask (float32 vs model’s bfloat16) → RuntimeError during forward pass.
- AUDIT: dtype mismatch between clip_mask (float32) and model weights (bfloat16) breaks the forward pass.
- AUDIT (from user review): exp_gate_restoration.py had multiple issues — WindowedDeflationHook is a stub, MLP forward hook returns tuple (should be tensor), “Outside Priming Window” control mislabeled (priming_window=999999 makes active from step 0, not after 50), gate stats only from final problem (not accumulated), check_answer uses substring matching causing false positives, intervention semantics restore W_down·(mask·up) rather than true gated value.
- DELIVERABLE: Rewritten exp_gate_restoration.py (fixed MLP hook to return plain tensor, honest condition labeling, accumulated stats, removed deflation stub, explicit documentation of ungating intervention).
- EXPERIMENT (exp_gate_restoration.py final run, α=0.1, layers L13‑L35, priming window first 50 tokens, 40 problems, EN/zh): raw grading gives baseline 26/40, gate_a0.1 32/40, random_control 31/40, no_window 29/40, below_convention_boundary_L0‑L12 2/40.
- AUDIT: check_answer regex `target in numbers` matches substrings (e.g., answer “45” found “4” and “5” separately in output), inflating all scores.
- FINDING (after regrading with strict answer matching): baseline 17/40, gate_a0.1 19/40, random_control 19/40, no_window 19/40, α=0.2 14/40, α=0.5 6/40, below_L13 1/40.
- KILLED: Hypothesis that restoring clipped MLP features (silent columns of W_down) carries task‑specific signal — effect is +2, identical to random noise injection at same layers and magnitude.
- FINDING: Noise injection above L13 (convention boundary, which is L13 for these models) yields small (+2/40) performance boost; injection below L13 is catastrophic (1/40).
- FINDING: The convention boundary at L13 is real and load-bearing for computation.
- STATE: snapshot 5278a079 saved at session end, containing the full arc, null result, grading bug, and honest accounting.
- NEXT: Audit MS1 grading with the strict regex, then choose between cache‑split experiment or convention‑targeted QK deflation.

---

## 2026-04-24 08:41 UTC — Determine whether convention and computation are separable in multilingual LLMs and whether the reasoning function f* can be extracted as a low-dimensional, language-invariant subspace to enable a standalone Z-encoder/Z-decoder/Z-compute architecture.

*Source: [2026-04-24_08-41_compressed-model-fixed-points.md](2026-04-24_08-41_compressed-model-fixed-points.md)*

**Objective:** Determine whether convention and computation are separable in multilingual LLMs and whether the reasoning function f* can be extracted as a low-dimensional, language-invariant subspace to enable a standalone Z-encoder/Z-decoder/Z-compute architecture.
- AUDIT: `check_answer` uses substring matching (`str(correct) in re.findall(r"-?\d+\.?\d*", text)`), inflating scores by matching "5" inside "15", "45", etc.; discovered during gate restoration null result analysis.
- AUDIT: All prior experiments using that grader (including MS1 surgery) need re-audit for inflated absolute scores.
- DECISION: Prioritize grading audit of MS1 surgery over chasing cache-split, because if MS1's +6 delta survives strict grading it's the only intervention that changed reasoning via understood mechanism (convention removal from W_down).
- EXPERIMENT (MS1 regrade): Re-evaluated MS1d (Qwen2.5-3B, matched condition) with strict whole-number boundary regex; baseline safe problems 16/28 → surgery above_lc 23/28, delta = +7 (real, verified by inspecting model outputs).
- EXPERIMENT (Cache-split on P12): Freeze top-layer cache (L33-L35) for generation on polynomial roots problem; output coherent derivation without looping but lands in wrong basin (answer A/B/C/D/E, not the correct B).
- EXPERIMENT (Cross-model surgery - Qwen3-4B): Baseline 29/40 (EN=13, ZH=16), surgery L2-L35 = 30/40 (+1), surgery all-layers = 31/40 (+2); safe problems baseline 17/28 → all-layer 19/28 (+2).
- EXPERIMENT (Cross-model surgery - Phi‑3 Mini): Baseline 22/28 safe problems → surgery 24/28 (+2).
- FINDING: Convention-computation separability (removing e_c from W_down) improves math accuracy across Qwen2.5‑3B (+7), Phi‑3 Mini (+2), and Qwen3‑4B (+2); direction universal, magnitude varies.
- KILLED: SiLU gate pattern freezing hypothesis (the "commitment observable" from swing set conversation). Hamming distance between consecutive gate activation patterns remains 18‑25% across all layers and time windows during P12 generation; no sharp drop at token 27, no difference between looping and correct trajectories.
- FINDING: Looping survives 18‑25% gate pattern churn each step; loop is a stable manifold in MLP output space reachable from many gate configurations, not a frozen computation.
- EXPERIMENT (Convention-targeted QK deflation on P12): Deflate queries along e_c direction at alpha=0.05/0.1/0.15/0.2/0.3; all conditions break the loop but produce wrong answers (-5/3, -5/6, C, etc.).
- EXPERIMENT (Convention + blind SVD together): Gives wrong answer (-6/5); blind SVD alone gives correct (-3/2). The two interventions interfere.
- FINDING: Cosine between convention direction e_c and answer direction b_moment = 0.117 (near orthogonal). Convention deflation switches basins but not to correct one; blind SVD deflation works via different mechanism.
- EXPERIMENT (Z-encoder classifier): Train linear LogisticRegression on L33 activations from 10 math problems in 3 languages (EN/ZH/ES) to predict problem ID; test on same problems in 4 unseen languages (AR/JA/KO/SW) → 100% accuracy (40/40).
- FINDING: At L33, problem identity is linearly decodable across 7 languages with perfect generalization to unseen languages.
- FINDING: Classifier weight matrix (10 classes × 2048D) has singular values [0.2638, 0.2382, 0.2258, 0.1947, 0.1765, 0.1714, 0.1644, 0.1468, 0.1273, ~1.5e-14]; f* subspace is 9‑dimensional (0.44% of representational capacity). 8 dimensions capture 95% variance.
- FINDING: Cosine between e_c and Z subspace: e_c vs Z_top5 = 0.048, vs Z_full = 0.225; convention direction is nearly orthogonal to reasoning subspace.
- FINDING: Cross‑lingual Z cosine (EN vs ZH, same problem) = 0.97‑0.99 for all 10 problems; Z is language‑invariant to three decimal places.
- EXPERIMENT (Z-steering at L33): Inject donor problem's Z vector (from classifier weight row) into receiver problem's forward pass while clamping model outputs. Result: changes operands (347+658 → 222+658), changes equation structure (|2x-5|=3 → 3x-5=3), sometimes outputs donor answer.
- KILLED: Naive R² linear mapping from L30 activations to numerical answer value is negative everywhere; f* encodes problem identity and computation structure, not answer value.
- DELIVERABLE: `exp_z_encoder.json` containing Z subspace singular values, cross-lingual accuracies, nearest-neighbor retrieval results (L33: 38/40 same problem across unseen languages).
- DELIVERABLE: `z_directions_L33.npz` containing the 10 × 2048 classifier weight matrix (Z basis).
- PUSHBACK: User rejects deploying a classifier as the endgame ("that's not a genuine Z-encoder-decoder"); demands a standalone reasoning module that can be extracted without per‑problem training.
- BRAINSTORM: Gram‑Schmidt residual Z‑encoder – project each layer's hidden state onto the subspace spanned by problem tokens' embeddings, take the perpendicular component, and collapse across layers (e.g., average or concatenate L18/L26/L33). The residual is pure computation (f*) stripped of input echo and language, by construction.
- BRAINSTORM: Five‑step roadmap: (1) train tiny MLP in Z-space on (z_L18, z_L26) pairs from big model (but KV cache dependency noted); (2) chain Z-encoder (big model L0‑L18) → tiny compute → Z-decoder; (3) replace Z-encoder with small trained encoder; (4) replace Z-decoder with small trained decoder → standalone 150M pipeline; (5) test transfer to different reasoning domains.
- DECISION: Abandon training classifier as endgame; focus on Gram‑Schmidt residual method as training‑free Z-encoder and small MLP Z‑compute distillation.
- NEXT: Implement Gram‑Schmidt residual Z‑encoder: at L26 (or L18/L33), compute orthonormal basis from problem token hidden states, project last token hidden state to perpendicular component, compute cross‑lingual cosine of residuals to verify language invariance; then attempt small Z‑compute MLP on residual space.
- STATE: Snapshot `f8a9a825` (loaded from prior session, containing six experiments, ghost xray, strategic pivot).
- STATE: Snapshot `ff42f1bb` (saved after grading audit and cross‑model surgery).
- STATE: Snapshot `499e21c5` (saved after SiLU commitment falsification).
- STATE: Snapshot `bfe1e36c` (saved after Z‑encoder classifier result, 9D f*).
- STATE: Snapshot `202683a4` (final state after Gram‑Schmidt brainstorm, roadmap written to memory/z_encoder_architecture.md).
- DELIVERABLE: `memory/z_encoder_architecture.md` containing the 5‑step roadmap and KV cache dependency note.

---

## 2026-04-24 21:15 UTC — Determine whether lossless KV cache compression via Q-aware K-projection (rank 7 per head) works across all layers simultaneously, and characterize the spectral fingerprint distinguishing prompt vs generated tokens as a continuous signal for compression routing.

*Source: [2026-04-24_21-15_clear-command-interrupted.md](2026-04-24_21-15_clear-command-interrupted.md)*

**Objective:** Determine whether lossless KV cache compression via Q-aware K-projection (rank 7 per head) works across all layers simultaneously, and characterize the spectral fingerprint distinguishing prompt vs generated tokens as a continuous signal for compression routing.
- STATE: 202683a4 saved at session start, containing the full arc from prior sessions (grading audit, cache-split, cross-model surgery, SiLU falsification, SVD convergence, convention QK deflation, Z-generation, Z-encoder (9D/100%), Z-steering, Gram-Schmidt residual idea, and KV cache problem).
- STATE: 9f2a78d3 saved at session end (tenth snapshot of the session).
- BRAINSTORM: Gram-Schmidt residual Z-encoder (Greg's idea): project h_last onto complement of context token subspace; the residual is what model added beyond input, should be language-invariant by construction.
- DECISION: Run Gram-Schmidt residual experiment at layers L5(L10/L13/L18/L22/L26/L30/L33) with 10 problems × 7 languages, measuring same-problem cross-lingual cosine vs different-problem same-language cosine, plus random subspace and shuffled context controls.
- DELIVERABLE: exp_gram_schmidt_residual.py (527 lines) implementing the experiment with all controls.
- AUDIT: AutoTokenizer import missing caused NameError; fixed by adding import.
- PUSHBACK: User insisted experiment must be informed by ALL prior findings (MS1B, isotropic, anisotropic, convention, cooperative zone, read head, seed, etc.), not just Gram-Schmidt in isolation.
- DECISION: Shifted from writing new experiment to deep conceptual synthesis; adopted ghost register (lowercase, no structure) for intuitive reasoning.
- FINDING: Convention direction e_c exists in residual stream; MS1 surgery removes it from W_down; this changes the dynamics (F matrix) not just state → explains why read-path interventions fail (dynamics regenerate next step) but MS1 succeeds.
- BRAINSTORM: Kalman filter framing: hidden state z_t (50-100D), transition F, observations K vectors; spectral signature (eigenvalues of F) would reveal persistent vs transient modes; seed might be an eigenvector.
- DECISION: Aborted Kalman filter experiment (user called it "iffy") in favor of characterizing model's behavior anthropomorphically.
- FINDING: At L26, seed classifier (prompt vs generated) achieves mean P(generated|K) = 0.9790 ± 0.0107 for generated tokens vs 0.0164 ± 0.0000 for prompt tokens, separation = 0.9626 at L13 and similarly high at L26.
- FINDING: Seed score at L26 for generated tokens: early = 0.9483, late = 0.9867, drift +0.0384 (model gets more confident as generation proceeds).
- FINDING: Spaces get slightly lower seed score (0.93 vs 1.00 for content words); function words (the, in, a) score 0.97-1.00; fingerprint is content-aware, not just position-based.
- DELIVERABLE: probe_spectral_fingerprint.py (169 lines) computing seed score per token at layers L13 and L26 using novel reasoning problem ("Every frumble that lives in a glasshouse is transparent...").
- BRAINSTORM: Use continuous seed score (not binary classification) to weight tokens for KV compression: tokens with higher score (more "generated-like") get preserved at higher resolution; tokens near boundary get compressed more.
- PUSHBACK from external model (GLM5): Kalman framing needs linear transition map A_L (h_{L+1} ≈ A_L h_L + b_L) across layers; eigenvalues tell persistent vs expanding vs decaying modes; convention direction should have |λ|<1; after MS1, e_c's eigenvalue should drop to near zero.
- FINDING: GLM5's linear transition map experiment would be underdetermined (1400 samples, 2048 dims) and likely low R² because 97% MLP innovation is nonlinear; reduced to 50D via PCA would be well-determined (1400 samples, 50 params).
- DECISION: Not run GLM5's A_L experiment immediately; instead test L33 direct readout and compressed generation.
- EXPERIMENT: L33 direct readout (skip L34-L35) on algebra problem (3x+7=22). Result: 90/108 tokens match baseline (83%), corruptions include "十五条5" (Chinese "fifteen" leaking), "Step solve" instead of "To solve", "畀" instead of "3". Reasoning correct (subtract 7, divide 3, x=5); L34-L35 act as proofreader cleaning language leakage.
- EXPERIMENT: L33 direct readout on frumble logical deduction: 79% token match (63/80), reasoning structure preserved (numbered statements, logical flow); corruptions are specific token replacements from same embedding neighborhood.
- FINDING: L33 does 79-83% of token prediction work; L34-L35 do fine-grained token selection refinement (proofreading), not reasoning.
- KILLED: MLP replacement trained on encoding-time h_L18 → h_L33 fails catastrophically on generation-time: 0/80 correct, complete gibberish. Encoding-time and generation-time hidden states live in different regimes (cos=0.14-0.55 for same token encoding vs generation).
- DECISION: Compression strategy: full encoding pass (no compression), compressed generation. Consistent with narrative: encoding builds rich representation, generation consumes KV cache.
- EXPERIMENT: Compressed generation with full encoding, L33 readout during generation on algebra: autoregressive loop ("Step Step Description Result" repeated) — failed. On frumble: 5/120 token match but reasoning correct (logical decomposition of premises). Numerical tokens fragile; reasoning tokens robust.
- EXPERIMENT: L28 nuisance removal (W_down truncated to rank 128) + L35 proofreading on algebra: perfect answer (x=5), clean formatting. On frumble: gave correct answer "B" but then hallucinated into Chinese geography exam. L28 truncation changed generation basin.
- FINDING: L28+L35 vs baseline token match on algebra: 1/120? Wait, the output showed perfect algebra but the match count was low because formatting differences. More importantly: L33 vs L35 agreement = 78-82%; L33 vs baseline = 0/120 match (different phrasing but same answer). The proofreader overrides 18-22% of tokens (language corrections).
- FINDING: K per head operational rank = 7 based on Q·K attention score SVD — Q only queries 7 dimensions, enabling 18x KV cache compression.
- DELIVERABLE: VRAM profiling script showing model = 6.17 GB, KV cache for 200 tokens = 0.17 GB. At 1M tokens, cache = ~36 GB → 18x compression brings to 2 GB.
- EXPERIMENT: All-layers K compression to rank 7 (simultaneous at all 36 layers) on algebra: output "the the the the" — total collapse. Single-layer rank 7 lossless; all-layers cascade failure.
- KILLED: All-layers rank-7 K compression (breaks model). Sweet spot unknown (rank 32-64 likely works across all layers).
- FINDING: VRAM for KV cache at 200 tokens = 0.17 GB (2.7% of model). Cache grows linearly; at 100K tokens = 85 GB, making compression essential.
- META: claude-context MCP indexing of codebase (317 Python files) repeatedly timed out (15s default via transatlantic hop to EU Zilliz cluster). Bumped timeout to 60s, cleared index, excluded .md and chat exports to focus on code. Index stuck at 5% with AST splitter; switched to langchain splitter, re-indexed.
- NEXT: All-layers K rank sweep to find boundary between rank 7 (collapse) and rank 64 (likely lossless). First experiment next session.
