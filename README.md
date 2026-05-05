# The Brain Inside The Machine
> **Status (May 2026):** This document describes the original hypothesis (February 2026). Since then, 170+ causal intervention experiments across 4 model families. Results in [`docs/`](docs/) and [`output/`](output/). Interactive knowledge graph: [prosodic.org/showcases](https://prosodic.org/showcases).

---


## Can we isolate a language-agnostic reasoning subspace inside a transformer?

---

## 1. The Observation

Qwen3-Vision-8B, prompted with Putnam 2025 A2 in Chinese, produces the correct answer: $a = 1/\pi$, $b = 4/\pi^2$. Claude itself got this wrong. The same model, same weights, same temperature — prompted in English — fails.

This is not an isolated result. Across increasingly hard problems (polynomial interpolation, popcount combinatorics, Putnam-level analysis), Chinese reasoning chains are structurally tighter: fewer tokens, fewer logical detours, higher accuracy. The performance gap isn't stylistic. It's computational.

**The only variable is the input tokens.** The weights are identical. The architecture is identical. Something inside the model reasons better when the input is Chinese.

---

## 2. The Universality Argument

Consider two hypothetical monolingual models trained on perfectly translated, equal-volume data:

$$Q_{zh}: V_{zh}^* \to V_{zh}^*, \quad Q_{en}: V_{en}^* \to V_{en}^*$$

By assumption, both achieve equivalent reasoning quality:

$$\text{translate}_{zh \to en}(Q_{zh}(t_{zh}(X))) \approx Q_{en}(t_{en}(X))$$

But $Q_{zh}$ and $Q_{en}$ have completely different weights. Different attention matrices, different FFN parameters — you cannot swap a single layer between them. They are internally incompatible. Yet they compute the same function (up to translation).

**Therefore** there must exist an abstract function $f^*$ and language-specific wrappers $h_i, h'_i$ such that:

$$Q_{zh}(x) = h'_{zh}(f^*(h_{zh}(x)))$$
$$Q_{en}(x) = h'_{en}(f^*(h_{en}(x)))$$

Where:
- $h_i$: language-specific encoder (strips language, exposes meaning)
- $f^*$: language-agnostic reasoning core
- $h'_i$: language-specific decoder (re-wraps meaning in language)

The decomposition exists *in the space of possible computations*, regardless of whether any single model cleanly instantiates it.

---

## 3. The Stronger Claim

The decomposition is not $f^*$ with equal-sized components. We claim:

$$\text{params}(f^*) \gg \text{params}(h_i) + \text{params}(h'_i)$$

The wrappers are **thin**. The core is **thick**. Why? Translation is a shallow function (roughly lookup + local grammar — there's a reason word2vec translation matrices work at all). Reasoning is deep, sequential, compositional. In any honest decomposition, the computational bulk must live in $f^*$.

---

## 4. The Information-Theoretic Framing

The space $\mathcal{Z}$ between $h$ and $h'$ must satisfy two constraints simultaneously:

$$I(\mathcal{Z}; \text{language}) \approx 0 \quad \text{(no linguistic information leaks through)}$$

$$I(\mathcal{Z}; \text{math\_content}) \approx I(\text{input}; \text{math\_content}) \quad \text{(all reasoning information preserved)}$$

**The bet:** these two objectives are compatible — linguistic form and mathematical content are sufficiently disentangled in the data that you can kill one without killing the other. "2+2=4" and "二加二等于四" have near-zero mutual information between their surface forms conditional on the mathematical content.

---

## 5. The Chinese Density Argument

Chinese is the optimal source language for extracting $\mathcal{Z}$ because:

$$d(\text{Chinese tokens}, \mathcal{Z}) < d(\text{English tokens}, \mathcal{Z})$$

Chinese characters are higher-entropy per token. "求满足" packs into 3 tokens what English takes ~6 to say. Chinese is already partially compressed toward $\mathcal{Z}$. The encoder $h_{zh}$ has less work to do than $h_{en}$.

**This predicts something testable:** CKA similarity between Chinese and English activations should be higher at earlier layers for Chinese — it reaches the shared representation sooner.

---

## 6. The Latent Logical Space

If $\mathcal{Z}$ is real — a clean, low-dimensional subspace where reasoning lives stripped of language — then the operations *in* $\mathcal{Z}$ should be almost embarrassingly simple:

- "A implies B" is a projection
- "A and B" is an intersection
- "Substitute x into f" is a matrix-vector multiply
- "This equals that" is two vectors having distance zero

The dream: transformers perform these operations *in disguise*, buried under $d$-dimensional activations that are 80% linguistic clothing and 20% actual logical content. Strip the clothing, and what remains is linear algebra on a small space. Near-tautological matrix operations composed sequentially.

**The testable prediction:** Take a problem whose logical proof has $K$ steps. Extract the $\mathcal{Z}$-space trajectory across layers. If $\mathcal{Z}$ is real and the operations are near-tautological, that trajectory should have approximately $K$ dominant directional changes — $K$ "turns" corresponding to $K$ proof steps.

---

## 7. Behavioral Evidence

### 7.1 Single-Language Runs (Popcount: $f(n) = |\{1\text{-bits in } n\}|$, count $n \leq 2025$ with $f(n)=3$)

| Language | Math Training Data | Answer | Tokens | Found $C(11,3)$ shortcut? |
|----------|-------------------|--------|--------|--------------------------|
| Chinese (run 1) | Massive | **165** | 774 | Yes ($1792 < 2025$) |
| Spanish | Sparse | **165** | 827 | Yes ($1792 < 2025$) |
| Chinese (run 2) | Massive | 495 | 1712 | No (summed $\sum C(k,3)$) |
| Swahili | Near-zero | 495 | ~1000+ | No (same sum trap) |
| Turkish | Near-zero | Context limit | 32768 | No (correct setup, ran out) |
| English | Large | Spiral | Limit | No |

### 7.2 Polyglot Experiments (same problem, 8 translations simultaneously)

| Prompt Set | Answer | Tokens | Shortcut? | Notes |
|------------|--------|--------|-----------|-------|
| **Top 8** (zh/en/es/fr/de/ja/ko/ru) | **165** | 696 | Yes | Clean, found 1792 |
| **Bottom 8** (sw/yo/tl/mn/ne/am/lo/km) | **165** | 523 | No | Correct answer, flawed verification |

### 7.3 What The Evidence Shows

1. **The shortcut lives in $\mathcal{Z}$.** It's accessible from Chinese, Spanish, and polyglot prompts — languages with vastly different math training corpora.

2. **Spanish finding the shortcut is the crown jewel.** There is near-zero chance Alibaba's training data contains this specific popcount trick in Spanish math competition format. Spanish found it because it reached $\mathcal{Z}$.

3. **Both reasoning and failure modes are language-agnostic.** The "sum all $C(k,3)$" trap appeared in Chinese (run 2) and Swahili. The elegant shortcut appeared in Chinese (run 1), Spanish, and the top-8 polyglot. Both computational options exist in $\mathcal{Z}$; the language wrapper determines probability of reaching each path.

4. **Wrapper cost is real.** The bottom-8 polyglot got the right answer in fewer tokens (523 vs 696) but made a surface verification error (claimed 2025 has 3 ones — it has 7). The logical skeleton survived; the linguistic checking capacity didn't. The wrapper degradation damaged surface checking but not the reasoning core. **This is exactly what the decomposition predicts.**

5. **Multilingual redundancy stabilizes $\mathcal{Z}$ access.** Single low-resource languages fail (Swahili solo: wrong). Multiple low-resource languages together succeed (bottom-8: correct). Redundancy compensates for thick individual wrappers.

6. **Wrapper cost model:**

$$P(\text{finding shortcut}) \approx g(\text{capacity\_remaining\_after\_wrapper})$$
$$\text{capacity\_remaining} = \text{total\_capacity} - \text{wrapper\_cost}(\ell)$$
$$\text{wrapper\_cost}: \text{Chinese} < \text{Spanish} \approx \text{English} \ll \text{Turkish} < \text{Swahili}$$

---

## 8. Structural Evidence: The Spectral Autopsy

### 8.1 Weight SVD Analysis (Qwen2.5-3B, L=36, d=2048)

Every weight matrix in the model decomposed via SVD. Entropy-based effective rank: $r_{\text{eff}} = \exp(H(\hat{\sigma}))$ where $\hat{\sigma}_i = \sigma_i / \sum_j \sigma_j$.

**Key findings:**

| Finding | Data | Interpretation |
|---------|------|----------------|
| **Layer 33 bottleneck** | All 16 heads collapse to eff rank ~80 (mean 80.4, std 6.0) | Prime candidate for $\mathcal{Z}$ compression point |
| **Attention is extremely low-rank** | 50% energy in 8/2048 components, 99% in 82/2048 | The model queries a 4%-dimensional subspace |
| **1-33-2 architecture** | Layer 0→1: rank drops of -768 (W_down), -767 (W_up), -545 (W_gate). Layers 34-35 snap back to 112-118. | 1 encode layer, 33 compute layers, 2 decode layers. 92% of depth is "middle." |
| **Attention narrows, FFN broadens** | Attention eff rank: monotone DOWN. FFN eff rank: monotone UP. | Opposing trends — attention compresses while FFN expands. |
| **$W_V$ is constant** | Eff rank ~250 ± 5, CV = 0.02 across all 36 layers | Messages have fixed format. Only routing changes. |
| **GQA groups diverge** | Heads 0-7 (KV head 0) vs 8-15 (KV head 1) systematically different | Functional specialization within GQA groups |
| **Layer 33: maximum uniformity** | All 16 heads converge to 76-100 range (vs layer 20: range 57-110) | The representation is maximally organized at the bottleneck |

### 8.2 What The Structure Tells Us About $\mathcal{Z}$

The model spends 92% of its depth on "middle" computation (layers 1-33). If a clean latent space $\mathcal{Z}$ existed and the model could efficiently encode into it, most of this middle should collapse — the computation would be trivially simple in $\mathcal{Z}$. The fact that it doesn't collapse means the encoding is spread inefficiently across the full network. **The model hasn't found the clean latent space. We're looking for it from the outside.**

Layer 33 is where the representation is most compressed and most uniform. If $|\mathcal{Z}_{33}| \approx 80$ dimensions (matching the attention kernel effective rank), that means the model's reasoning at the bottleneck lives in a subspace that is ~4% of the ambient dimension. The other 96% is linguistic clothing.

### 8.3 Attention Kernel Eigenstructure (Layer 18, Head 0, representative)

$$W_Q^T W_K \in \mathbb{R}^{d \times d}$$

| Metric | Value |
|--------|-------|
| Asymmetry $\|A - A^T\|/\|A\|$ | 1.35 |
| Complex eigenvalues | 116/2048 (5.7%) |
| Effective rank (entropy) | 80.1 |
| 99% spectral energy | 82/2048 components |

Asymmetry of 1.35 means directed relationships, not similarity. 116 complex eigenvalues indicate rotational/oscillatory modes. The attention kernel is not performing nearest-neighbor lookup — it's implementing a structured, asymmetric, partially rotational operator on a small subspace.

---

## 9. Finding $\mathcal{Z}$: The ARD-MMD Kernel Method

### 9.1 Why Not CKA?

CKA (Centered Kernel Alignment) produces a single number per layer and requires mean-pooling (throwing away sequence structure, assuming uniform density per token). Chinese and English tokenize to different lengths. CKA can't handle this.

### 9.2 The ARD-MMD Approach

Treat Chinese and English activations as **empirical measures** on $\mathbb{R}^d$. Compare distributions, not vectors. Handle token count mismatch by construction.

**ARD-RBF kernel:**

$$k_\ell(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{1}{2} \sum_{j=1}^{d} \frac{(x_j - y_j)^2}{\ell_j^2}\right)$$

where $\boldsymbol{\ell} \in \mathbb{R}^d$ is a per-dimension lengthscale vector.

**MMD² (Maximum Mean Discrepancy):**

$$\text{MMD}^2(P_{zh}, P_{en}; \boldsymbol{\ell}) = \mathbb{E}[k(x,x')] + \mathbb{E}[k(y,y')] - 2\mathbb{E}[k(x,y)]$$

where $x, x' \sim P_{zh}$ (Chinese activations) and $y, y' \sim P_{en}$ (English activations).

**The optimization:** for each layer $k$, find $\boldsymbol{\ell}$ minimizing MMD across all prompt pairs, with L1 sparsity on $1/\ell$:

$$\min_{\boldsymbol{\ell}} \sum_{i=1}^{N} \text{MMD}^2(P_{zh}^{(i,k)}, P_{en}^{(i,k)}; \boldsymbol{\ell}) + \lambda \sum_{j=1}^{d} \ell_j^{-1}$$

**Interpretation:**
- Dimensions where $\ell_j \to \infty$: **ignored** (language-specific, activations differ here)
- Dimensions where $\ell_j$ stays finite: **$\mathcal{Z}$** (cross-lingually shared, activations agree here)

The number of finite-$\ell$ dimensions at each layer = $|\mathcal{Z}_k|$.

### 9.3 Key Diagnostic Plots

**Plot 1: $|\mathcal{Z}_k|$ vs layer index.** Where does $\mathcal{Z}$ live?

- If $|\mathcal{Z}|$ is small at middle layers and large at early/late → inverted-U, model compresses to compact shared representation in the middle
- If $|\mathcal{Z}|$ is roughly constant → stable thread, distributed not localized
- If $|\mathcal{Z}|$ grows monotonically → model progressively strips language going deeper
- If $|\mathcal{Z}| \approx 0$ everywhere → hypothesis in serious trouble

**Plot 2: Lengthscale spectrum at the most interesting layer.**

- If **bimodal** (cluster of small $\ell$ + cluster of large $\ell$) → clean separation, $\mathcal{Z}$ is well-defined. **Best case.**
- If **power-law** (smooth decay) → graded separation, still useful but "clean subspace" dream is weaker
- If **uniform** (all $\ell$ equal) → ARD found no structure. Rethink everything.

**Structural prediction from spectral autopsy:** $|\mathcal{Z}_{33}| \approx 80$, matching the attention kernel effective rank at the bottleneck.

---

## 10. Validation Protocol

### 10.1 Does $\mathcal{Z}$ contain reasoning?

Finding shared dimensions isn't enough. They could be shared but irrelevant (both languages encode punctuation the same way).

**Test 1:** Same problem, different languages should be **close** in $\mathcal{Z}$:
$$d_\mathcal{Z}(\text{same problem, diff lang}) \ll d_{\mathcal{Z}^\perp}(\text{same problem, diff lang})$$

**Test 2:** Different problems, same language should be **far** in $\mathcal{Z}$:
$$d_\mathcal{Z}(\text{same problem, diff lang}) \ll d_\mathcal{Z}(\text{diff problem, same lang})$$

This means $\mathcal{Z}$ encodes **problem identity**, not language identity.

**Test 3:** Train classifier on $\mathcal{Z}$-projected activations to predict answer correctness. Compare to classifier on $\mathcal{Z}^\perp$-projected activations. Prediction: $\mathcal{Z}$-classifier is better.

### 10.2 Activation Patching

Surgical: swap **only the $\mathcal{Z}$ dimensions** at layer 33.

$$\text{patched}[:, :, \mathcal{Z}_{\text{mask}}] = \text{chinese\_acts}[:, :, \mathcal{Z}_{\text{mask}}]$$

- If $\mathcal{Z}$-only patching works but full patching breaks → $\mathcal{Z}$ is real and $\mathcal{Z}^\perp$ is language-essential
- If both work → $\mathcal{Z}$ identification wasn't necessary
- If neither works → sequence length mismatch issue (try last-token patching)

### 10.3 The Bridge ("Son Mimicking Father")

Train a tiny linear probe $g$ mapping English activations → Chinese activations at layer $k^*$:

$$\mathcal{L} = \frac{1}{N} \sum_i \| g(\bar{X}_{en,i}^{k^*}) - \bar{X}_{zh,i}^{k^*} \|^2$$

If $g = W$ (a single matrix) achieves high $R^2$, then the two languages are related by a rotation — the thinnest possible wrapper.

**SVD of the bridge** $W = U\Sigma V^T$:
- High-$\sigma$ directions: where languages **agree** → this **is** $\mathcal{Z}$
- Low-$\sigma$ directions: where languages **diverge** → language-specific surface structure

Bridge within $\mathcal{Z}$ only: $W \in \mathbb{R}^{|\mathcal{Z}| \times |\mathcal{Z}|}$. If $|\mathcal{Z}| = 80$, that's 6,400 parameters — 0.00008% of the model. Almost no overfitting risk.

**The $R^2$ number:**
- $R^2 > 0.9$ within $\mathcal{Z}$ → the two languages are rotations of each other in $\mathcal{Z}$-space
- $R^2 > 0.8$ → thin wrappers, strong result
- $R^2 < 0.5$ → even within $\mathcal{Z}$, the relationship is nonlinear

**If $W$ is approximately orthogonal** ($\|W^T W - I\| / |\mathcal{Z}|$ small), then languages are **isometric copies** in $\mathcal{Z}$. The bridge is a pure rotation. This is the strongest possible evidence for the decomposition.

---

## 11. The End State

At inference:

$$\text{English input} \xrightarrow{\text{layers } 1..k^*} X_{en}^{k^*} \xrightarrow{g} \tilde{X} \xrightarrow{\text{layers } k^*\!+1..L} \text{output}$$

Added: ~6,400 parameters (0.00008% of model). The model is frozen. English now reasons at Chinese quality because we found where language ends and reasoning begins, and built a bridge across.

---

## 12. Implications If This Works

**Layer 1 — Engineering.** Multilingual reasoning parity for every bilingual model. Every lab with a model that reasons better in one language wants this bridge.

**Layer 2 — Interpretability.** If $\mathcal{Z}$ is small, project onto it and *read off the reasoning directly*. The operations in $\mathcal{Z}$-space **are** the proof steps. The model's internal reasoning becomes auditable not by reverse-engineering circuits, but by finding the subspace where reasoning was always simple.

**Layer 3 — Compression.** If 8B parameters are really a 500M reasoning core in a 7.5B language costume, that's the most extreme compression target in deep learning. Operate directly in $\mathcal{Z}$, dress up output at the final layer. Order-of-magnitude inference cost reduction.

**Layer 4 — Universality.** Is $\mathcal{Z}$ the same across models? If Qwen's $\mathcal{Z}$ and LLaMA's $\mathcal{Z}$ and Claude's $\mathcal{Z}$ are rotations of each other, then reasoning has a *canonical geometry*. One of the most important findings in the science of intelligence.

**Layer 5 — Philosophy.** A universal $\mathcal{Z}$ says: the structure of valid reasoning is a geometric fact about high-dimensional spaces, and any system that learns to reason — human, transformer, alien — will find approximately the same subspace. Mathematical Platonism with empirical evidence.

---

## 13. Known Risks

- $\mathcal{Z}$ is messy. The probe $g$ needs a deep MLP, not a matrix. The operations in $\mathcal{Z}$ are just as inscrutable as the full activations.
- $|\mathcal{Z}|$ is 3000 out of 4096. It's not really a subspace at all.
- The lengthscale spectrum is uniform — ARD finds no structure.
- The behavioral gap is training data bias, not a reflection of internal decomposition.
- The autoregressive nature means activations at layer $k^*$ carry forward-looking next-token planning information, not just "meaning."

Phase 1-3 results will tell us. But if the linear probe works and the singular value spectrum of $W$ drops off sharply — say, 200 dimensions capture 95% of cross-lingual variance — then we're holding something real.

---

## 14. Project Structure

| File | Description | Status |
|------|-------------|--------|
| `1.py` / `1.ipynb` | Weight effective rank analysis (Qwen2.5-3B). Entropy-based SVD of all 7 weight types across 36 layers. Full 576-kernel heatmap. | **Complete** |
| `2.py` / `2.ipynb` | Subspace geometry: Grassmann overlap, bottleneck convergence, FFN-attention alignment. | Code complete, **not yet run** |
| `utils.py` | `effective_rank`, `get_attn_subspace`, `subspace_similarity`, `get_model_dims` | **Complete** |
| `Gameplan.md` | 24-hour sprint plan v2: ARD-MMD on Qwen3-8B | Written |
| `Gameplan_v3.md` | Structural-informed plan integrating 1.py/2.py findings, targeting Qwen2.5-3B | Written |
| `VEGA_CONTEXT.md` | Chronological source map and synthesis of all prior conversations | Written |
| `Chat_0.md` | Genesis: hypothesis formation, spectral autopsy assignment, 1.py/2.py results | Reference |
| `Chat_1-5.md` | Behavioral testing: Qwen3-VL-8B, Chinese vs English across problem types | Reference |
| `Chat_2.md` | Theoretical framework: universality argument, ARD-MMD, behavioral experiments | Reference |
| `output/` | Effective rank CSVs, markdown reports | Generated |

---

## 15. Model Specifications

**Analysis target:** Qwen2.5-3B

| Parameter | Value |
|-----------|-------|
| Layers ($L$) | 36 |
| Hidden size ($d$) | 2048 |
| FFN intermediate ($d_{ff}$) | 11008 |
| Attention heads ($h$) | 16 |
| GQA heads | 2 |
| Vocab size ($V$) | 151936 |
| Head dim ($d_{head}$) | 128 |

**Behavioral testing:** Qwen3-Vision-8B and Qwen3-8B via LM Studio (RTX 4070 Super, temp 0.4, top_k 30, top_p 0.7, min_p 0.1)

---

## 16. The Three Numbers That Matter

| Number | What It Measures | Where It Comes From |
|--------|-----------------|-------------------|
| $\|\mathcal{Z}\|$ at best layer | How many dimensions reasoning needs | ARD-MMD lengthscale spectrum |
| $R^2$ of bridge within $\mathcal{Z}$ | How linear the cross-lingual mapping is | Linear probe regression |
| Orthogonality error of bridge | Is it a rotation? | $\|W^T W - I\| / \|\mathcal{Z}\|$ |

---

*Greg Sharma, M.S. Mathematics in Finance, Courant Institute (NYU), 2025*
