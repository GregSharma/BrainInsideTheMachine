# The Brain Inside The Machine

**Can we isolate a language-agnostic reasoning subspace inside a transformer?**

prosodic.org/showcases — interactive knowledge graph of all concepts and sessions

---

A transformer prompted in Chinese solves math problems it fails in English — same weights, same temperature. If the model's output is invariant under input language but the reasoning is correct, that invariance doesn't automatically mean a language-agnostic reasoning core exists *inside* the model. It could be a surface-level phenomenon. Whether the internal computation factorizes as $Q_\lambda(x) = h'_\lambda(f^*(h_\lambda(x)))$ — thin language wrappers around a shared reasoning core — is a separate, harder question. This repo is 193 experiments trying to answer it.

Side project, started February 2026 with no prior mechanistic interpretability experience.

**Primary model:** Qwen2.5-3B (36L, d=2048) | **Cross-model:** Qwen2.5-7B, 14B, Qwen3-8B, Phi-3 Mini | **Hardware:** RTX 4070 Super 12GB, occasional Colab A100

---

## Findings

**Transformers are encoder-decoders in disguise.** The prompt does the heavy computational work — building a rich, cross-lingual representation in full dimensionality. Generation rides the attractor. The last token at each layer is a read head that contributes a constant bias — replaceable with a mean vector without accuracy loss (`expC6b_mean_dissection.py`). Context tokens carry the computation (`expC2b_dose_response.py`: all-tokens N=1 → 0/20, last-only N=36 → baseline). This unifies the rank-1 causal readout (`expC3_7b_compression.py`), the MLP innovation structure (97% of each MLP delta is fresh, not propagated — `expT_pacf_innovation.py`), and the fact that nothing is reconstructed in the canyon (L27-L32 rank collapse, `expBQ2_crossmodel_lyapunov.py`).

**Language convention is thin and removable.** Convention lives in the bottom singular vectors of $W_{down}$ above layer 12 (`expC7c_convention_anatomy.py`). Projecting it out improves multilingual math accuracy by +6 to +8 points across 200+ problems and 3 languages (`expMS1_kernel_surgery.py`, `expMS1b_robust_surgery.py`). The effect replicates on Phi-3 (Microsoft) and Qwen3-4B (`exp_crossmodel_surgery.py`) — it's not model-specific. Convention and reasoning are nearly orthogonal: cos(convention direction, 9D reasoning subspace) = 0.048 (`exp_z_encoder.py`).

**The model knows answers it can't say.** There are two distinct mechanisms for this. (1) The correct answer is in the residual stream but the read head can't express it because the KV cache is swamped with echo tokens — soft Q-deflation at alpha=0.1 deterministically unlocks it (`exp_deflated_attention.py`). (2) The correct answer is expressible at L33 (79-83% token match with the full model), but L34-L35 convention scrubbing is needed to clean up a multilingual draft. L33 thinks; L34-L35 proofread (inline L33-direct readout experiments, `probe_decisive3.py`).

**Generation tokens carry a seed signal.** A binary classifier on K-vectors at L13 separates prompt tokens from generated tokens with 96% accuracy (`probe_spectral_fingerprint.py`). The seed projection tracks originality: echo tokens (repeating the premise) score +2.19, novel tokens (original reasoning) score +3.95, prompt tokens score -4.97 — three distinct regimes (`probe_echo_vs_novel.py`). This suggests a natural compression lever — echo tokens are redundant in the KV cache.

---

## Research trajectory

Each row is a belief update. Top to bottom tracks how understanding evolved over the project.

| Phase | Prior belief | What actually happened |
|-------|-----------------|----------------------|
| Apr 7 | Cosine Gram rank_50=1 means transformers reason in ~5 dimensions. | **Refined.** rank_50=1 is trivially expected from anisotropy. Centered rank_90=8-21 is real — 100-200x null discrepancy. |
| Apr 7 | SVD truncation to rank_90 should compress inference. | **Killed.** 0/20 at ALL k, even k=500. Observation ≠ intervention. |
| Apr 7 | Gram dynamics might be specific to one model. | **Replicated.** 4 models, 2 labs. Output rupture tied to embedding tying. |
| Apr 9 | The last token accumulates computation across layers. | **Supported.** Last token = read head. Constant mean vector = lossless. Context tokens do the work. |
| Apr 10 | Read head operates in a compressible subspace. | **Supported.** k=8 lossless at L33 (3B). k=1 lossless at L27 (7B). Statistical rank 92, causal rank 1. |
| Apr 10 | Swapping KV cache between languages should break reasoning. | **Null.** No effect at any layer. KV cache inert post-encoding. |
| Apr 11 | The rank-1 readout direction encodes something meaningful. | **Refined.** v1 ≈ negative mean. cos(v1, language)=-0.002. Any nonzero direction works. Attention = constant bias. |
| Apr 12 | Language convention is entangled with reasoning. | **Supported.** Convention lives in W_down, surgically removable. +6 to +8 accuracy. Cross-model. |
| Apr 14 | A blind prompt can read math content from lower-layer KV. | **Scales.** 3B: structure only. 7B: exact equations. 14B: correct answers. Zero garbage. |
| Apr 18 | Soft deflation of queries can break loop attractors. | **Supported.** alpha=0.1 deterministically steers to correct answer (4/4 seeds, identical output). Temperature scatters randomly. |
| Apr 20 | Deflation protects a late readout phase. | **Killed.** First 50 tokens necessary and sufficient. Trajectory primer, not readout protector. 5 layers x 50 tokens = 0.66%. |
| Apr 23 | Loop onset scales with deflation strength. | **Killed.** t_crit intrinsic at ~27 tokens. Fixed SVD basis catastrophically worse. Encoding ≠ computation geometry. |
| Apr 25 | L33 activations encode problem identity across languages. | **Supported.** 9D subspace, 100% accuracy on 4 unseen languages. cos(convention, f*) = 0.048. Orthogonal. |
| Apr 25 | Z-bottleneck, MLP factorization, fiber bundle, gate freezing, needle-threading. | **All killed.** Null-space layer-specific. Rotation chaotic (77 deg/layer). Gates churn 18-25%/step. |

---

## Timeline

| Dates | Phase | Summary |
|-------|-------|---------|
| Feb–Mar | Original hypothesis & behavioral testing | Information-theoretic framing; Chinese vs English reasoning chains on Qwen3-Vision-8B; polyglot experiments. Work done in a separate directory (LM Studio install on Windows); migrated to this repo in April. |
| Apr 7–11 | Gram geometry & read-head discovery | Centered rank_90=8-21; killed "5D reasoning"; read head = constant bias; 4-model replication |
| Apr 11–14 | Convention surgery & bun inversion | 9D language-invariant f*; MS1 surgery +6-8 accuracy; 14B blind inversion zero garbage |
| Apr 14–24 | Deflation & echo bifurcation | Trajectory primer (50 tokens); phase diagram; per-head entropy signature |
| Apr 24–25 | Z-encoder & dead-end consolidation | 100% cross-lingual classifier; 15+ hypotheses killed and documented |

---

## Experiment prefixes

Experiments are named `exp<PREFIX><NUMBER>_<description>.py`. Prefixes are roughly chronological and indicate research arcs:

| Prefix | Arc | What it tests |
|--------|-----|---------------|
| A–F | Cross-lingual activation geometry | MLP deltas, attention weights, propagation, basin width |
| G | Layer skip & generation-time effects | Block deletion, generation-time layer-specific interventions |
| H | Patty loop & attention dynamics | Loop attractor characterization, attention pattern analysis |
| I–L | Format, language, destruction tests | Format vs language disentangling, fluff detection, destruction sweeps |
| M | Computation heads | Per-head contribution to reasoning, MLP decomposition |
| N–P | Cross-lingual MLP surgery | MLP swap, ridge shortcut, language stripping |
| Q–R | Non-math & cross-model flip | Domain transfer of flip interventions, cross-model validation |
| S–Z | Domain transfer through Z-reconstruction | Transfer experiments, PACF innovation, Neumann iteration, Z-space |
| BQ/BR | Gram matrix & kernel dynamics | Gram evolution, Lyapunov spectrum, diverse-problem replication |
| BS | SVD truncation | Causal compression via truncation (killed) |
| C | Causal compression & read-head | Tail transplant, dose-response, constructive compression, convention anatomy |
| MS | Kernel surgery | Convention removal from W_down, system prompt controls, multilingual surgery |
| G1–G5 | Bun inversion | KV cache surgery to separate content from mode, blind description |
| INJ | Hidden state injection | Cross-lingual activation transplant, KV replacement |
| GATE | Gate expansion & saturation | Encoding-time subspace dimensionality and causal transfer |
| SMA | Sensitivity-modulated attention | Gate-derived attention reweighting |
| DA1 | Distributional attention | Variance-bonus exploration mechanism |

53 additional standalone experiments (`exp_attention_anatomy.py`, `exp_z_encoder.py`, `exp_silu_commitment.py`, etc.) don't follow the prefix convention.

---

## Highlighted experiments

These are selected results, not an exhaustive index. Full results are in `output/`.

| Script | Question | Result |
|--------|----------|--------|
| `exp_z_encoder.py` | Linear probe for problem identity across unseen languages? | 100% on 4 unseen languages, 9 dimensions, L33 |
| `expC2b_dose_response.py` | Last token: read head or reasoner? | all-tokens N=1 → 0/20; last-only N=36 → baseline |
| `expC6b_mean_dissection.py` | What does last-token attention contribute? | Constant bias. mean_only = baseline. |
| `expC3_7b_compression.py` | Readout rank at L27 on 7B? | k=1 lossless. Statistical rank 92, causal rank 1. |
| `expMS1_kernel_surgery.py` | Remove convention from W_down? | +6 to +8 accuracy, 200+ problems, 3 languages |
| `exp_crossmodel_surgery.py` | Convention surgery on Phi-3, Qwen3-4B? | Both positive (+2 to +7) |
| `expBQ2_crossmodel_lyapunov.py` | Gram dynamics across 4 models? | rank_50=1 all layers, all models |
| `expG1e_14b_bun_inversion.py` | Blind prompt reads math from KV cache? | 14B: zero garbage, correct answers |
| `exp_attention_anatomy.py` | Read-head focus: content vs glue tokens? | p < 0.0001 at L32-35, 37/37 problems |
| `expBS_svd_truncation.py` | Compress inference to Gram rank_90? | 0/20 at all k. Observation ≠ intervention. |

*Grading caveat: early experiments used a substring-matching grader that inflated absolute accuracy. Directional claims (treatment vs control) are unaffected — both arms use the same grader. Strict regrading confirmed all directional results hold or strengthen.*

---

## How to navigate

- **Experiments**: `exp*.py` in root. Results in `output/<name>.json`. Activation caches in `output/*.npz`.
- **`docs/`**: Writeups generated during the research with Claude Code, consolidated using external models (DeepSeek, GPT) for cross-auditing and Perplexity for literature grounding — as close to an independent audit as one person with LLMs can get.
- **[docs/research_log.md](docs/research_log.md)**: Session-by-session log of what was done, what was found, what was killed. Think of it as a timesheet — clocking in and out of research sessions with findings, breakthroughs, and decisions tagged for scanning.
- **[MATHEMATICAL_SPEC.md](MATHEMATICAL_SPEC.md)**: Formal mathematical encoding of experiments — hypotheses and methods as equations, auto-generated from the Python scripts. Equations are faithful to source; verify quantitative claims against `output/`.
- **[docs/original_hypothesis.md](docs/original_hypothesis.md)**: The February 2026 information-theoretic framework. If the model's computation is invariant under language, does a language-agnostic core $f^*$ necessarily exist inside it? Two questions: (1) does the factorization hold abstractly, and (2) does the model actually instantiate it? This document argues for (1); the experiments test (2).

---

## Future work

1. **KV cache compression kernel** — a learned or spectral kernel that compresses the KV cache at inference time, targeting the echo/redundancy structure identified above
2. **Seed-token compression** — skip echo tokens in the KV cache (tokens with low prediction-mismatch score), keep only high-seed integration checkpoints
3. **Lightweight MLP layer bridge** — MLP deltas have cos ~0.94 between adjacent layers; a tiny learned MLP could skip layers if magnitude calibration is solved
4. Cross-model family transfer of the 9D subspace (Llama, Gemma, Mistral)
5. Gram-Schmidt residual Z-encoder — training-free language-invariant extraction (script written, not run)
6. Constructive compression at L33 using the readout basis end-to-end

---

## Abandoned directions

Explored and deliberately deprioritized based on experimental evidence:

- **Z-bottleneck inference** (exp BO) — Catastrophic at all k and all layers. Must be trained, not retrofitted.
- **MLP factorization** (exp BL/BP) — Attention 48%, MLP 52% of rotation. No single component to isolate.
- **Fixed null-space** (exp BM) — Layer-specific (own-proj=0.97, cross-proj=0.01). No universal null-space.
- **Gate freezing** (exp SiLU commitment) — Gates churn 18-25%/step even during loops. Loop ≠ frozen gates.
- **Needle-threading** — Falsified by topology discriminator. Late-onset works; separatrix does not.
- **Encoding-time basis for generation** (exp GATE causal) — 9.6% overlap, 0/20 at all k.

---

## Reproduction

```bash
pip install torch transformers scipy numpy tqdm
python expMS1_kernel_surgery.py        # convention surgery example
# Results: output/expMS1_kernel_surgery.json
```

Qwen2.5-3B fits in 6.2GB VRAM. 14B experiments require A100 or equivalent.

---

## Model specifications

| Model | Layers | d | Embeddings | Role |
|-------|--------|------|-----------|------|
| Qwen2.5-3B | 36 | 2048 | tied | Primary |
| Qwen2.5-7B | 28 | 3584 | tied | Replication |
| Qwen2.5-14B | 48 | 5120 | untied | Replication (A100) |
| Qwen3-8B | 36 | 4096 | untied | Cross-architecture |
| Phi-3 Mini | 32 | 3072 | untied | Cross-lab (Microsoft) |

---

*Gregory Sharma*
