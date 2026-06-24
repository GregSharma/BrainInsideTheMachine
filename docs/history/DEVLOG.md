# BrainInsideTheMachine — DEVLOG

## Session: 2026-02-21 (VEGA Audit)

### Context Recovery

Read all project files via SSH from the Perploxity machine:

- Chat_0.md (genesis conversation, 1974 lines) — origin of hypothesis + 1.py/2.py instructions
- Chat_1-5.md (5599 lines) — behavioral testing + extended hypothesis. Superset of Chat_1.md
- Chat_2.md (3621 lines) — theoretical framework, ARD-MMD, Gameplan derivation
- Gameplan.md — 24hr sprint plan, unaware of 1.py/2.py structural work
- 1.py, 2.py, utils.py — source code
- 1.ipynb (12/19 cells executed), 2.ipynb (0/23 cells executed)
- output/ — CSV + one_output.md from 1.py only

Updated VEGA_CONTEXT.md with full project state including Chat_0 origin story.

### Code Audit: 1.py

**Status: RUN SUCCESSFULLY. Output verified.**

**Mathematical correctness:**

- Effective rank computation (utils.py): `r_eff = exp(H(σ_hat))` where `σ_hat = σ/Σσ`. CORRECT. Filters zero singular values before log. Uses `torch.linalg.svdvals` (numerically stable).
- Attention kernel: `W_Q_h^T @ W_K_h` → (d, d). CORRECT. This is the bilinear form defining attention: `score(x_i, x_j) = x_i^T (W_Q^T W_K) x_j`. Note: this is the FULL kernel projected back to d-space, not the d_head×d_head product. The rank is bounded by d_head=128.
- GQA handling: `kv_head_idx = head_idx // n_kv_groups` where `n_kv_groups = h // GQA = 16 // 2 = 8`. So heads 0-7 share KV head 0, heads 8-15 share KV head 1. CORRECT.
- Eigendecomposition: `torch.linalg.eigvals(attn_kernel)` on the (2048, 2048) kernel. CORRECT — this is a square matrix, eigendecomposition is valid. Complex eigenvalues expected for non-symmetric matrix.
- Asymmetry metric: `||A - A^T|| / ||A||`. CORRECT measure of departure from symmetry.
- Phase analysis splits into equal thirds (0-11, 12-23, 24-35). Not optimal — 1-33-2 would be more meaningful — but adequate for initial survey.

**Potential issues:**

1. `attn_kernel_eff_ranks_for_layer` uses SVD-based effective rank on `W_Q_h^T @ W_K_h`, not eigenvalue-based. For the single-head analysis above, eigendecomposition was used. These give DIFFERENT numbers for asymmetric matrices. The heatmap uses SVD-based rank (through `svdvals`), the single analysis uses eigenvalue magnitudes. Both are valid but measure slightly different things. The heatmap's SVD-based approach is arguably more standard.
2. The ThreadPoolExecutor for SVD computation: SVD is CPU-bound and Python's GIL means threads don't truly parallelize PyTorch CPU operations. This LOOKS parallel but is likely serialized. Not a correctness issue, just a performance one. The `BATCH` approach of loading weights to CPU in batches is the real optimization.
3. The markdown report generator uses variables from the notebook's global scope (`attn_kernel`, `eigvals_mag`, etc.) rather than recomputing. This means the report is consistent with the notebook state but couldn't be regenerated standalone.

**Output quality:**

- `one_output.md`: 20KB comprehensive markdown. All tables well-formatted. Includes model config, attention kernel analysis, heatmap data, W_down SVD, effective rank statistics, phase analysis.
- `Qwen_Qwen2.5-3B_Layer_Weight_Eff_Ranks.csv`: 252 rows (36 layers × 7 weight types). Verified structure correct.

### Code Audit: 2.py

**Status: NEVER RUN. Zero cell outputs in 2.ipynb. No output files.**

**Mathematical correctness (code review only):**

- `get_attn_subspace`: Returns top-k right singular vectors (Vh[:k, :]) of W_Q_h^T @ W_K_h. CORRECT — these span the dominant k-dimensional subspace of the attention kernel's image.
- `subspace_similarity`: Grassmann similarity via principal angles. Computes M = V1 @ V2.T, takes SVD, returns mean(σ²). CORRECT — the singular values of M are the cosines of the principal angles between the two k-dimensional subspaces. Mean squared cosine gives a scalar similarity measure. 1.0 = identical subspaces, chance level = k²/(d·min(k,d)) for random subspaces.
- `ffn_attention_alignment`: Projects W_gate rows onto attention subspace via P = Vh^T @ Vh (projection matrix), measures energy ratio. CORRECT — P is idempotent, projects onto the k-dimensional subspace, and the energy ratio measures what fraction of FFN computation lies in the same directions attention is querying.
- Multi-head analysis: Averages similarity matrices across 4 sampled heads. CORRECT.

**Potential issues:**

1. The `get_attn_subspace` function signature is `(model, layer_idx, h, GQA, d, head_idx, k=20)` — note `h, GQA, d` are positional args passed before `head_idx`. This is correct but fragile — any caller needs to pass them in the right order.
2. The bottleneck layer (33) is hardcoded. This is fine for Qwen2.5-3B but would need updating for different models.
3. Experiment 2's convergence test uses Pearson correlation — this assumes linear trend. A monotone non-linear convergence would score lower. Not wrong, but a Spearman rank correlation would be more robust.
4. The multi-head analysis samples `np.linspace(0, h-1, 4)` = heads [0, 5, 10, 15]. This spans both GQA groups (0-7 and 8-15), which is good. But 4 is a small sample for 16 heads.

**What 2.py would tell us if run:**

- Experiment 1: Whether the 33 layers query genuinely different subspaces or redundantly overlap → directly tests the "wasted depth" hypothesis
- Experiment 2: Whether layer 33 is a convergence point (similarity increases approaching it from both sides) or just an isolated anomaly
- Experiment 3: Whether FFN and attention operate in complementary or overlapping subspaces → affects whether Z can be found independently in attention alone
- Cross-reference: Combines effective rank with subspace geometry for a unified per-layer profile

### Code Audit: utils.py

**4 functions, all mathematically sound:**

- `effective_rank(W)`: Entropy-based. Handles zero singular values correctly (filters before log).
- `get_attn_subspace(model, layer_idx, h, GQA, d, head_idx, k)`: Clean GQA handling. Returns (k, d) orthonormal rows.
- `subspace_similarity(V1, V2)`: Grassmann metric via SVD of inner product. Standard approach.
- `get_model_dims(model)`: Config extraction. Correct for Qwen architecture.

### Run Status Summary


| File           | Status                       | Output              |
| -------------- | ---------------------------- | ------------------- |
| 1.py / 1.ipynb | RUN (12/19 cells)            | CSV + one_output.md |
| 2.py / 2.ipynb | NOT RUN (0/23 cells)         | None                |
| utils.py       | Library, no direct execution | N/A                 |


**Why 2.py wasn't run:** The conversation in Chat_0 ended with Claude assigning the three experiments (which became 2.py). Greg likely wrote the code but ran out of time or moved to Chat_1-5/Chat_2 for theoretical development before executing. The code itself appears correct and ready to run.

**Cannot run 2.py remotely:** WSL on wunderwaffe doesn't have torch/transformers installed. The Jupyter kernel used `.venv` which was likely a Windows-side Python environment. Running 2.py requires either:

- Greg to run it manually via Jupyter on Windows
- Setting up the Python environment in WSL (installing torch, transformers, etc.)
- Running it on a different machine with Qwen2.5-3B downloaded

### Key Findings from Audit

1. **1.py is solid.** Code correct, GQA handled properly, output verified. The attention kernel analysis (eigendecomposition) and the heatmap (SVD-based rank) use slightly different rank measures but both are valid.
2. **2.py is ready but unexecuted.** Code looks correct on review. The three experiments are the natural next step from 1.py's findings. Running this is the highest priority — it directly tests whether the depth is doing real work (subspace overlap) and whether the layer 33 bottleneck is a convergence point or an anomaly.
3. **The phase analysis in 1.py uses equal thirds (0-11, 12-23, 24-35).** The data itself shows the actual structure is 1-33-2 (one encoding layer, 33 compute layers, 2 decode layers). A revised analysis with these natural boundaries would be more informative.
4. **The Gameplan targets Qwen3-8B but all structural work was done on Qwen2.5-3B.** This mismatch needs resolution. Either: (a) run activation experiments on 2.5-3B (already loaded, structural analysis done), or (b) redo structural analysis on 3-8B and then run activation experiments there.

### Decisions


| Decision                                           | Rationale                                                          |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| Do NOT attempt to install torch in WSL             | Fragile, CUDA may not be available, risk breaking Greg's setup     |
| Priority: run 2.py when Greg is available          | This is the single highest-value next step                         |
| Revise Gameplan to incorporate structural findings | The Gameplan was written blind; 1.py/2.py data should inform it    |
| Target Qwen2.5-3B for activation experiments       | Already analyzed structurally, fits in VRAM, avoids model mismatch |


### Files Modified


| File            | Change                                                           |
| --------------- | ---------------------------------------------------------------- |
| VEGA_CONTEXT.md | Major rewrite with Chat_0 origin story, chronological source map |
| DEVLOG.md       | Created (this file)                                              |


### Revisit

- Run 2.py (Greg must do this via Jupyter, or set up WSL env)
- Redo phase analysis with 1-33-2 boundaries instead of equal thirds
- Decide Qwen2.5-3B vs Qwen3-8B for activation experiments
- Write revised Gameplan incorporating structural findings

---

## Session: 2026-02-21 (Continued — Gameplan Revision + Evaluation)

### Accomplished

- Created Gameplan_v3.md — full revision incorporating 1.py/2.py structural findings
- Key changes from v2: target Qwen2.5-3B (not 3-8B), add Phase 0 (behavioral verification), add Phase 1 (run 2.py), reduce layer count from 36 to ~10, use structural priors for initialization
- Estimated time reduced from 24hr to 10-12hr
- Confirmed 2.py was NEVER RUN (0/23 cells with output)
- Cannot run 2.py remotely (WSL lacks torch/transformers)

### Project Evaluation

#### Strengths

1. **The hypothesis is well-formed and testable.** h'(f(h(X))) is concrete enough to be falsified. The ARD-MMD approach is a specific, implementable method for testing it.
2. **Strong structural foundation.** 1.py provides genuine data about the model's internal organization. The layer 33 bottleneck, opposing attention/FFN trends, and W_V constancy are real empirical findings that constrain theories.
3. **The methodology is novel.** Applying ARD-MMD to multilingual activation extraction is genuinely new. Most cross-lingual interpretability work uses CKA or probing classifiers. The kernel approach handles the token count mismatch naturally.
4. **The researcher's background is unusually suited.** Courant math/finance training means the mathematical tools (SVD, kernel methods, information theory) are native, not learned secondhand.
5. **Rapid iteration capability.** 1.py was written and run in 30 minutes. The code quality is good — proper assertions, GQA handling, defensive checks.

#### Risks

1. **Model mismatch.** The behavioral asymmetry (Chinese >> English) was observed on Qwen3-Vision-8B. All structural work was on Qwen2.5-3B. These are different models. Phase 0 in v3 addresses this but it's a real risk.
2. **Z might not separate cleanly.** The FFN-attention coupling (from 2.py code review) suggests the representation space is not neatly partitioned. Z might be distributed across interleaved attention and FFN subspaces in a way that ARD-MMD can detect but that doesn't yield a clean mask for patching.
3. **The behavioral gap might be too small at 3B.** Smaller models often show less language-dependent behavior because they have less capacity for language-specific specialization. The Chinese advantage might be a big-model phenomenon.
4. **ARD optimization might not converge.** With d=2048 and ~50 tokens per prompt, the kernel matrices are 50×50 which is tractable, but 2048 lengthscale parameters with only 20 prompt pairs means the problem is heavily overparameterized. Regularization is critical.
5. **The layer 33 bottleneck might be an artifact of the effective rank metric.** Effective rank based on entropy of normalized singular values can be sensitive to the spectral shape. A different rank metric might not show the same bottleneck.

#### Confidence Assessment


| Outcome                                                                        | Probability | Why                                                                                                            |
| ------------------------------------------------------------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------- |
| Find language-dependent vs independent dimensions at some layer                | 60-70%      | Strong prior from multilingual BERT literature. ARD-MMD is well-suited.                                        |
| Confirm layer 33 as Z compression point in activations                         | 40-50%      | Static weight structure is suggestive but not proof. Activations could tell a different story.                 |
| Successfully patch Z-only to improve English reasoning                         | 15-25%      | Patching is hard. Sequence length mismatch, attention pattern disruption, token-level alignment issues.        |
| Publishable finding (novel, reproducible, interpretable)                       | 30-40%      | Even partial results (Z exists but doesn't fully explain the gap) are publishable given the novel methodology. |
| Transformative insight (changes how people think about multilingual reasoning) | 10-15%      | Would require clean separation + successful patching + replication across models.                              |


#### Strategic Recommendation

The project should proceed. The expected value is positive even at 30-40% chance of publishable results, because:

- The downside (a few days of work) is small
- The upside (genuine contribution to mechanistic interpretability) is significant
- The structural analysis already done (1.py) is valuable independent of whether Z pans out
- The methodology (ARD-MMD for cross-lingual activation analysis) is novel regardless of the outcome

The immediate next steps are:

1. Run 2.py (Greg must do this via Jupyter)
2. Run Phase 0 behavioral verification
3. Then Phase 2-3 (activation extraction + ARD-MMD)

### Files Modified


| File           | Change                                                   |
| -------------- | -------------------------------------------------------- |
| Gameplan_v3.md | NEW — revised gameplan incorporating structural findings |
| DEVLOG.md      | Appended evaluation section                              |


### Decisions


| Decision                              | Rationale                                                                                           |
| ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Target Qwen2.5-3B for v3              | Structural analysis already done, fits VRAM, 4x faster kernels                                      |
| Add Phase 0 (behavioral verification) | Must confirm the Chinese>>English gap exists at 3B before spending 10+ hours on activation analysis |
| Reduce to 10 target layers            | Structural analysis shows most information is in layers 10-35, with layer 33 as focal point         |
| Keep original Gameplan.md intact      | v3 is a new file (Gameplan_v3.md), preserving v2 for reference                                      |


---

## Session: 2026-02-21 (VEGA — 2.py Execution + Analysis)

### Accomplished

- Installed uv, created WSL venv (.venv_wsl), installed PyTorch+CUDA+transformers on wunderwaffe
- **Ran 2.py end-to-end successfully** — all 3 experiments + multi-head robustness + cross-reference + markdown report
- All output saved: 7 .npy files + two_output.md in output/
- Full analysis of results below

### 2.py Results Summary

#### Experiment 1: Subspace Overlap

- Off-diagonal mean similarity = 0.025 (chance ~0.01). Layers are mostly independent — depth is NOT redundant.
- **L32↔L33 = 0.482** — strongest pair by 2x. Multi-head avg = 0.386. Not a single-head artifact.
- Layers 31-33 form tight cluster. Layers 22-27 form weaker mid-depth bloc.
- Sharpest phase boundary: L33→L34 (drop of 0.452). Confirms 1-33-2 architecture.

#### Experiment 2: Bottleneck Convergence

- Pre-bottleneck Pearson r = +0.426: similarity to L33 increases approaching from early layers
- Ramp pattern: noise floor through L14, gradual L15-19, accelerating L20-22, sharp L31-32
- L34-L35 snap away from L33 — convergence is one-directional
- **Layer 33 IS a convergence point, not an anomaly.** Robust across all 4 sampled heads.

#### Experiment 3: FFN-Attention Alignment (THE SURPRISE)

- Mean alignment = 1.19x chance overall. Barely above random.
- **But at the bottleneck: L32 = 0.55x chance, L33 = 0.57x chance.**
- FFN ACTIVELY AVOIDS the attention subspace at L32-L33. Zero variance across heads (0.0056 ± 0.0001).
- Early layers show mild coupling (~1.2-2.2x chance), but by L25+ FFN systematically drops below chance.
- Correlation: bottleneck similarity vs FFN alignment = r = -0.414. The two trends are linked.

#### Cross-Reference

- Effective rank does NOT predict bottleneck similarity (r = -0.023). Different structural features.
- The organizing principle: attention converges to a specific subspace while FFN moves to its orthogonal complement.

### Key Implications for Z Extraction

1. **FFN-attention independence at L33 = green light for attention-only Z mask.** The Gameplan_v3 "emergency shortcut" is now the primary strategy: use L33's attention kernel SVD vectors as the Z candidate, not full ARD-MMD.
2. **L33's top-k attention subspace IS the candidate Z.** The network spends 32 layers rotating toward it. k ≈ 78 (matching effective rank from 1.py).
3. **Decode layers (L34-35) read Z in a different basis.** The sharp break means L33 is a genuine information bottleneck, not just smoothly continued.

### Revised Strategy

- Phase 3 (ARD-MMD) may be unnecessary. Instead: extract L33 attention subspace (top-78 SVD), project activations onto it, test if Chinese/English math prompts separate in Z vs Z⊥.
- If this works, it cuts ~3 hours from the gameplan and provides a cleaner, more interpretable result.
- ARD-MMD becomes the backup: only needed if the SVD-based mask doesn't separate languages cleanly.

### Environment Setup (for future runs)

- WSL venv: `/mnt/c/Users/grego/Desktop/LM_Studio/BrainInsideTheMachine/.venv_wsl/`
- PyTorch 2.6.0+cu124, transformers 5.2.0, Python 3.12.3
- GPU: RTX 4070 SUPER (12GB VRAM), CUDA 13.1 via WSL
- Model cached at `/home/grego/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/`
- Run with: `MPLBACKEND=Agg .venv_wsl/bin/python 2.py`

### Files Modified


| File                   | Change                                      |
| ---------------------- | ------------------------------------------- |
| output/two_output.md   | NEW — full 2.py results report              |
| output/*.npy (7 files) | NEW — raw numpy arrays from all experiments |
| .venv_wsl/             | NEW — WSL Python venv with torch+CUDA       |
| DEVLOG.md              | Appended this analysis                      |


### Decisions


| Decision                                     | Rationale                                                                               |
| -------------------------------------------- | --------------------------------------------------------------------------------------- |
| Use L33 attention SVD as primary Z candidate | FFN-attention independence at bottleneck means attention subspace alone is clean target |
| ARD-MMD becomes backup, not primary          | SVD-based mask is faster, more interpretable, and structurally motivated                |
| Keep .venv_wsl for future WSL runs           | Torch+CUDA confirmed working, no need for Greg to run via Jupyter anymore               |


### Revisit

- Phase 0: behavioral verification (Chinese>>English on Qwen2.5-3B) — still needed before activation work
- Test SVD-based Z mask on actual activations (project hidden states at L33 onto top-78 attention subspace)
- If SVD mask doesn't separate: fall back to ARD-MMD
- Consider running experiments on L32 as well (similarity 0.482 to L33 — might be even more interesting as the "approaching" layer)

## Session: 2026-02-21 (overnight prep)

### Accomplished

- Deep literature review via Perplexity (7 searches)
- Found NeurIPS 2025 closest competitor paper (arxiv 2505.15257): SVD-based language ablation
- Verified our differentiation: weight-based Z construction vs their data-driven approach
- Math verification: Grassmann similarity correct, projection math correct, random baseline k/d confirmed
- Fixed phase0_behavioral.py: added few-shot prompting for base model, better answer extraction
- Fixed phase2_z_extraction.py: removed confounding thinking suffix, added multi-head mask, added random-subspace baseline, added energy fraction metric
- Wrote RECIPE.md: consolidated experiment plan with contingencies, math verification, cross-disciplinary connections, decision tree
- Information bottleneck theory research: L33 = minimal sufficient statistic for reasoning
- FFN-attention orthogonality literature: our finding (FFN actively avoids attention at bottleneck) appears novel

### Issues


| Issue                                        | Status    | Notes                                                                                    |
| -------------------------------------------- | --------- | ---------------------------------------------------------------------------------------- |
| Qwen2.5-3B may not show behavioral asymmetry | KNOWN GAP | Perplexity search confirms no published ZH>EN gap for base model. Phase 0 will test.     |
| NeurIPS 2025 paper exists                    | KNOWN     | Different construction method (data-driven vs weight-based). We should cite and compare. |
| Base model answer extraction fragile         | MITIGATED | Added few-shot format + better regex                                                     |


### Revisit

- Run Phase 0 when Greg wakes
- Run Phase 2 after Phase 0
- If Phase 2 fails: try NeurIPS approach (activation-based SVD) as comparison
- Cross-check: does energy_frac_Z differ for same-lang-diff-problem vs cross-lingual?

### Files Modified


| File                   | Change                                                                             |
| ---------------------- | ---------------------------------------------------------------------------------- |
| phase0_behavioral.py   | Added few-shot prompting, better answer extraction, per-problem detail output      |
| phase2_z_extraction.py | Removed thinking suffix, added multi-head Z mask, random baseline, energy fraction |
| RECIPE.md              | NEW: consolidated recipe with contingencies, math, theory, decision tree           |


### Decisions


| Decision                                   | Rationale                                                                                                        |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Remove thinking suffix from phase2 prompts | Confounds language signal at extraction point — we want to measure the problem encoding, not the reasoning chain |
| Add multi-head averaged Z mask             | Single head (head0) is fragile; multi-head SVD captures the consensus subspace                                   |
| Add random-subspace baseline               | Without baseline, can't distinguish "Z captures language" from "any 78-dim subspace captures some language"      |
| Add energy fraction metric                 | Directly measures whether cross-lingual delta concentrates in Z (bad for hypothesis) or avoids Z (good)          |
| Cite NeurIPS 2505.15257                    | Closest work, but different method. Our weight-based approach is novel.                                          |


