# Z HUNT: The Recipe

## What we're looking for

A **language-agnostic reasoning subspace Z** inside Qwen2.5-3B's hidden states at Layer 33. The hypothesis: when the model reasons about math, it encodes the problem into a compressed subspace Z that is structurally independent of the input language. Chinese 加法 and English addition activate the same Z dimensions; the language-specific information lives in Z⊥.

## Why we believe it's there (structural evidence from 1.py + 2.py)

| Finding | Source | Implication |
|---------|--------|-------------|
| L33 attention heads collapse to eff rank ~78 (mean 80.4, std 6.0) | 1.py | L33 operates in a ~78-dim subspace — compressed bottleneck |
| All 16 heads converge to same rank at L33 (CV < 0.08) | 1.py | Heads agree on what matters at this depth — maximally organized |
| FFN alignment at L33: 0.57× chance (actively avoids attention) | 2.py | FFN and attention operate in ORTHOGONAL subspaces at bottleneck |
| FFN alignment std = 0.0001 across heads at L33 | 2.py | Not a fluke — FFN systematically avoids attention's subspace |
| L32↔L33 similarity = 0.482 (2× any other pair) | 2.py | L32 is the approach layer — convergence happens over L32-L33 |
| L33→L34 snap: biggest phase boundary in entire network | 2.py | L34 breaks out of the compressed representation — decode begins |
| 1-33-2 architecture: encode (L0), compute (L1-33), decode (L34-35) | 1.py+2.py | 92% of depth is "middle" computation |
| Attention rank ↓ monotone, FFN rank ↑ monotone | 1.py | They do opposite things at every layer — complementary roles |
| W_V constant ~250 across all layers (CV = 0.02) | 1.py | The messages have fixed format; only routing changes |

## What distinguishes us from the NeurIPS 2025 paper (arxiv 2505.15257)

They find language-specific directions via **data-driven SVD** (mean activations per language → SVD to find M_s). They ablate M_s at inference and find middle layers work best.

Our approach is **structurally-motivated**:

| Aspect | NeurIPS 2505.15257 | Our approach |
|--------|---------------------|--------------|
| Z construction | Data-driven: language centroids → SVD | Weight-based: attention kernel SVD at bottleneck |
| Why this layer? | Empirical sweep (try all layers) | Structural prediction: L33 has lowest eff rank |
| Why Z is clean | Not addressed | FFN orthogonality: 0.57× chance at L33 |
| Data required | Parallel multilingual corpus | NONE — Z derived from weights alone |
| Theoretical backing | Ablation → improved reasoning | Information bottleneck + Grassmann geometry |
| Architecture insight | None (black box optimization) | 1-33-2 structure, encode-compute-decode |

**Key novelty:** We can predict WHERE the reasoning subspace lives without any multilingual data, using static weight properties alone. The NeurIPS paper has to run activation extraction + SVD to discover it empirically.

## The experiment sequence

### Phase 0: Behavioral Verification (script: `phase0_behavioral.py`)
**Question:** Does Qwen2.5-3B actually show Chinese >> English on math?
**Run:** `MPLBACKEND=Agg .venv_wsl/bin/python phase0_behavioral.py`
**Time:** ~10 min (15 problems × 2 langs × ~20s per generation)

**Contingencies:**
- **Chinese > English by ≥3 problems:** PROCEED. Strong behavioral evidence.
- **Chinese = English (±1):** PROCEED ANYWAY. The 3B base model may not show behavioral asymmetry even if Z exists. The structural evidence is strong enough. Flag this as a limitation in any writeup.
- **English > Chinese:** INVESTIGATE prompt formatting. The few-shot exemplars may be biased. Try without few-shot. If still English-dominant, this is genuinely surprising — check if model is actually Qwen2.5-3B (not a renamed English model).

### Phase 2: Activation Extraction + Z Projection (script: `phase2_z_extraction.py`)
**Question:** Does the L33 attention SVD subspace separate language from reasoning?
**Run:** `MPLBACKEND=Agg .venv_wsl/bin/python phase2_z_extraction.py`
**Time:** ~20 min (20 problems × 2 langs, no generation, just forward pass)

**What to look for:**
1. **ratio_Z < ratio_Zp** across most configurations → Z captures reasoning, Z⊥ captures language
2. **energy_frac_Z < k/d** for cross-lingual deltas → language difference lives OUTSIDE Z (good!)
3. **energy_frac_Z > k/d** for same-language different-problem deltas → reasoning variation IS in Z
4. **Multi-head mask ≥ head0 mask** → multi-head averaging is more robust

**Contingencies:**
- **Clear separation (ratio_Z << ratio_Zp for all k):** SVD mask works! Skip ARD-MMD entirely. Go to Phase 3 (patching).
- **Partial separation (some k work, others don't):** Note which k works best. The k≈78 (matching eff rank) should be strongest. If k=20 works but k=78 doesn't, the reasoning subspace is even MORE compressed than the eff rank suggests.
- **No separation (ratio_Z ≈ ratio_Zp everywhere):** Three sub-contingencies:
  - a) Try the NeurIPS 2505.15257 approach: activation-based SVD (data-driven) instead of weight-based SVD. This is a direct comparison.
  - b) Try L32 instead of L33 — the "approach" layer may have cleaner separation.
  - c) Fall back to ARD-MMD on ~10 layers (Phase 3 in Gameplan_v3).

### Phase 3: Activation Patching (if Phase 2 succeeds)
**Question:** Does patching Z at L33 transfer reasoning between languages?
**Design:**
1. Run Chinese math problem → extract h_zh at L33
2. Run English version of same problem → extract h_en at L33
3. Project h_zh onto Z and Z⊥: h_zh_Z and h_zh_Zp
4. Patch: replace h_en_Z with h_zh_Z (swap reasoning, keep language)
5. Continue forward pass → does the model now answer the English problem using Chinese reasoning?
6. **Control:** swap Z⊥ instead (should break language, not reasoning)

**Success criteria:** Patching Z changes the ANSWER without changing the output LANGUAGE. Patching Z⊥ changes the LANGUAGE without changing the ANSWER.

### Phase 3b: Bridge (tiny model, if patching works)
**Question:** Can we learn a linear map within Z that translates between languages?
**Design:** Linear regression from Z(Chinese, problem_i) to Z(English, problem_i) for all 20 problems.
**Size:** [78, 78] = 6,084 parameters. Zero overfitting risk with 20 pairs.
**Success criteria:** Bridge predicts held-out Z vectors (leave-one-out cross-validation).

## Math verification

### Grassmann similarity metric (from utils.py)
- **Definition:** `sim(V1, V2) = mean(σ²)` where σ = svdvals(V1 @ V2.T)
- **Equivalent to:** `(1/k) ||V1 @ V2.T||_F²` — mean squared cosine of principal angles
- **Range:** [0, 1]. 1 = identical subspaces, 0 = orthogonal
- **Relationship to chordal distance:** `d_ch² = k(1 - sim)`. Our sim IS linearly related to the squared Grassmann chordal distance.
- **Caveat:** Not a proper metric (fails triangle inequality). But we only use it for pairwise comparisons, so this is fine.

### Projection math (from phase2)
- **P_z = Vh[:k,:].T @ Vh[:k,:]** — orthogonal projector because Vh has orthonormal rows (SVD guarantee)
- **||P_z(h1 - h2)|| = ||Z_mask(h1 - h2)||** — distance in R^d via projector equals distance in R^k via coordinates
- **Random baseline:** For random k-dim subspace Z, E[||P_Z δ||²] = (k/d)||δ||². For k=78, d=2048: 3.8%.

### FFN alignment metric (from 2.py)
- Project W_gate's top-k SVD vectors onto attention subspace
- Measure energy ratio: `||P_attn V_gate||²_F / ||V_gate||²_F`
- At L33: 0.0056 (chance = k/d = 0.0098). Ratio = 0.57× — actively BELOW chance
- This means FFN's gate directions are systematically avoiding the attention kernel's subspace

### Information-theoretic framing
- L33 as **information bottleneck**: layers 1-33 progressively compress the input into a minimal sufficient statistic for the task
- IB theory predicts: `I(Z_L; X)` decreases with depth, `I(Z_L; Y)` increases until convergence
- Our observation: effective rank drops (= compression) while semantic function is preserved
- The FFN orthogonality at L33 suggests SEPARATE channels: attention carries the bottleneck representation (Z), FFN carries the complement (language, surface form, etc.)
- Formal claim: at L33, the residual stream decomposes as h = P_Z(h) + P_Z⊥(h), where P_Z(h) ≈ minimal sufficient statistic for the math answer, and P_Z⊥(h) ≈ language-specific encoding

### Cross-disciplinary connections
- **Neuroscience:** Analogous to the "language of thought" hypothesis — Fodor (1975) proposed that cognition operates in an amodal representation independent of natural language. Our Z is the transformer equivalent.
- **Cognitive science:** Bilingual speakers show language-independent math representations in fMRI (Spelke & Tsivkin, 2001). The intraparietal sulcus activates identically for math in either language.
- **Compression theory:** Rate-distortion theory predicts that optimal compression produces representations that strip task-irrelevant information (language) while preserving task-relevant information (math structure).
- **Linear representation hypothesis:** Recent work (Park et al., 2023) shows that concepts in LLMs are linearly encoded. Our Z is the specific linear subspace for "reasoning-relevant" concepts at the bottleneck layer.

## What would make this publishable

### Minimal viable paper (estimated 45-55% chance):
1. Structural analysis confirming L33 bottleneck + FFN orthogonality (DONE)
2. Z subspace separates language from reasoning in activation space (Phase 2)
3. Comparison with NeurIPS 2505.15257 approach on same model/data

### Strong paper (estimated 25-35% chance):
All of minimal, plus:
4. Patching experiments showing causal role of Z
5. Generalization to at least one other model (e.g., Qwen3-8B, LLaMA-3-8B)
6. Theoretical connection to information bottleneck formalism

### Transformative paper (estimated 5-10% chance):
All of strong, plus:
7. Universal bottleneck detection algorithm (predict which layer for any model)
8. Bridge that enables cross-lingual transfer at inference time
9. Evidence that Z captures not just math but ALL reasoning

## Files on this machine

| File | Status | Purpose |
|------|--------|---------|
| `1.py` | DONE (run) | Effective rank analysis across all layers |
| `2.py` | DONE (run) | Subspace overlap, FFN alignment, convergence |
| `utils.py` | DONE | Shared functions: eff_rank, subspace sim, etc. |
| `phase0_behavioral.py` | READY TO RUN | Chinese vs English math scoring |
| `phase2_z_extraction.py` | READY TO RUN | Activation extraction + Z projection |
| `Gameplan_v3.md` | Reference | Full gameplan with time estimates |
| `Gameplan.md` | Reference | Original ARD-MMD approach (backup) |
| `output/one_output.md` | Results | 1.py analysis report |
| `output/two_output.md` | Results | 2.py analysis report |
| `output/*.npy` | Data | Saved numpy arrays from 2.py |
| `.venv_wsl/` | Environment | Python 3.12, torch 2.6.0+cu124, transformers |

## Run commands

```bash
# Activate environment
source .venv_wsl/bin/activate

# Phase 0
MPLBACKEND=Agg python phase0_behavioral.py

# Phase 2
MPLBACKEND=Agg python phase2_z_extraction.py
```

## Decision tree for tomorrow

```
START
  ↓
Run Phase 0
  ↓
Chinese >> English? ─── NO ──→ Flag limitation, proceed anyway
  ↓ YES                        (structural evidence > behavioral)
  ↓
Run Phase 2
  ↓
ratio_Z < ratio_Zp? ─── NO ──→ Try: (a) data-driven SVD
  ↓ YES                              (b) L32 instead of L33
  ↓                                  (c) ARD-MMD backup
  ↓
SVD mask works!
  ↓
Design patching experiment (Phase 3)
  ↓
Patching transfers reasoning? ─── NO ──→ Z is correlational, not causal
  ↓ YES                                   Still publishable as structural finding
  ↓
GOLD: Causal Z subspace identified
  ↓
Generalize to second model
```
