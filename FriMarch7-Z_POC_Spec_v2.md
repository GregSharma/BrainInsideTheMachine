# Synthetic Z Extraction: Specification (v2)

---

## 1. The Reasoning Function

Let $d = 10$. Define $f: \mathbb{R}^d \to \mathbb{R}^d$ as the following composition.

**Step 1 — Sort by magnitude.** Given $x = (x_1, \ldots, x_d)$, let $\pi$ be the permutation such that $|x_{\pi(1)}| \leq |x_{\pi(2)}| \leq \cdots \leq |x_{\pi(d)}|$. Define:

$$s_i = x_{\pi(i)}, \qquad s = P_\pi x$$

where $P_\pi$ is the permutation matrix induced by $\pi$. Note: $s$ retains the signs of $x$; only the ordering is by absolute value. This step requires pairwise comparison of all elements — it is inherently nonlinear and non-local.

**Step 2 — Cumulative sum.** Define:

$$c_i = \sum_{j=1}^{i} s_j, \qquad c = Ls$$

where $L$ is the $d \times d$ lower-triangular matrix of ones. This is linear but sequential — each $c_i$ depends on all preceding elements. It encodes accumulation, the core operation in any multi-step deduction.

**Step 3 — Nonlinearity.** Apply a compressed hyperbolic tangent:

$$n_i = \tanh(\tfrac{1}{2} c_i)$$

This bounds the representation and introduces saturation. Without it, cumulative sums grow unboundedly with $d$, making the function trivially invertible from magnitudes alone.

**Step 4 — Reverse.** Define:

$$r_i = n_{d+1-i}, \qquad r = J n$$

where $J$ is the $d \times d$ exchange matrix (ones on the anti-diagonal). This breaks the monotone structure of the cumulative sum, preventing the network from learning a simple "always increasing" heuristic.

**Step 5 — Normalize.** Project onto the unit sphere:

$$f(x) = \frac{r}{\|r\|_2}$$

This removes scale information, forcing the network to encode the answer purely in direction. The full composition:

$$\boxed{f = \text{normalize} \circ J \circ \tanh(\tfrac{1}{2}(\cdot)) \circ L \circ P_\pi}$$

**Why this function.** It requires: (i) global comparison (sorting), (ii) sequential accumulation (cumsum), (iii) nonlinear compression (tanh), (iv) structural rearrangement (reversal), and (v) scale-invariant encoding (normalization). A network cannot compute $f$ with a single linear layer. It must learn internal representations that track sorted order and running totals — analogous to multi-step mathematical reasoning where each deduction builds on the previous.

---

## 2. Language Wrappers

Let $\ell \in \{1, \ldots, K\}$ index $K = 7$ "languages." For each language, draw a fixed random orthogonal matrix:

$$R_\ell \in O(d), \qquad R_\ell R_\ell^\top = I$$

The **observed input-output pair** for language $\ell$ on problem $x$ is:

$$\tilde{x}_\ell = R_\ell\, x, \qquad \tilde{y}_\ell = R_\ell\, f(x)$$

The network never sees $x$ or $f(x)$ directly. It sees only the rotated versions. The rotation $R_\ell$ is drawn once and fixed for the duration of training and testing.

**Why rotations.** This is the simplest transformation that (a) changes all coordinates simultaneously, (b) preserves norms and angles, and (c) is exactly what Procrustes can undo. If Procrustes fails even here, it will certainly fail on the real model where per-language transforms may be nonlinear. This is the *easiest possible* version of the problem. We are not testing whether Z exists in a hard case. We are testing whether our extraction method works in the easiest possible case. If it fails here, we save ourselves from wasting time on Qwen.

---

## 3. The One-Hot Question

### Config B (with one-hot)

Concatenate a one-hot vector $e_\ell \in \mathbb{R}^K$ to the input:

$$\text{input}_\ell = [\tilde{x}_\ell \;;\; e_\ell] \in \mathbb{R}^{d + K}$$

**The risk this creates.** The model receives an explicit language tag. A sufficiently powerful network could learn to perfectly undo each rotation in the first layer: "if one-hot says language 3, multiply by $R_3^{-1}$." If this happens, all languages are in canonical coordinates by layer 2. Activations are identical across languages. Procrustes is unnecessary. Raw extraction works trivially. The experiment proves nothing.

**Why imbalanced training saves us.** With 60% language 1, the model gets a strong gradient signal for learning $R_1^{-1}$. With 1% language 7, the gradient signal for $R_7^{-1}$ is 60x weaker. The model will learn a partial, approximate inverse — enough to reduce loss on Swahili but not enough to fully canonicalize its coordinates. This produces the messy partial entanglement we're looking for: high-resource languages cleanly undone, low-resource languages still partially rotated. That IS the Qwen gradient, reproduced synthetically.

**Critical check after training.** Compute cross-lingual activation distance at layer 1 for each language pair (Chinese vs English, Chinese vs Swahili, etc.). If Chinese-English distance is near zero but Chinese-Swahili is large, the model partially undid rotations proportional to data frequency. This is the expected and desired outcome. If ALL distances are near zero, the model is too powerful — it learned perfect inverses for all languages even with 1% data. In that case, either shrink the network or switch to Config B-blind.

### Config B-blind (no one-hot — additional variant)

Same as Config B but remove the one-hot entirely:

$$\text{input}_\ell = \tilde{x}_\ell \in \mathbb{R}^{d}$$

Now the model has NO explicit language tag. It must infer which rotation was applied purely from the statistical properties of the input. This is harder and closer to the real transformer, which also has no explicit language tag — Qwen infers language from token co-occurrence patterns in the embeddings, not from a flag.

**Why this matters.** In Config B (with one-hot), the model has a shortcut: read the tag, look up the rotation. The entanglement between rotation-undoing and reasoning is a choice the model makes for efficiency, not something forced by the architecture. In Config B-blind, the model genuinely has to figure out the coordinate system from the data itself. The rotation-undoing MUST be entangled with the early processing of the input because there's no other way to do it. This produces a more realistic version of the pyrite problem.

**Recommendation.** Run both. Config B first (faster to debug, clearer signal). Config B-blind second (more realistic, stronger result if it works). If Procrustes helps in both, the result is robust. If Procrustes only helps in B-blind, then the one-hot was making Config B too easy.

---

## 4. Training Data

### Frequency distribution

Define language frequencies $\alpha = (\alpha_1, \ldots, \alpha_K)$ with $\sum \alpha_\ell = 1$:

| Language | $\ell$ | $\alpha_\ell$ | Rationale |
|----------|--------|---------------|-----------|
| Chinese  | 1      | 0.60          | Dominant training language (Qwen = Alibaba, Chinese-first) |
| English  | 2      | 0.20          | Strong secondary (universal in web data) |
| Spanish  | 3      | 0.08          | Medium-resource (found popcount shortcut in real Qwen) |
| Arabic   | 4      | 0.05          | Medium-resource, different script family |
| Japanese | 5      | 0.04          | Low-resource but shares characters with Chinese |
| Korean   | 6      | 0.02          | Low-resource, held out for testing |
| Swahili  | 7      | 0.01          | Near-zero resource, held out for testing |

Total training set: $N = 10{,}000$ samples. Language $\ell$ contributes $\lfloor \alpha_\ell N \rfloor$ samples. Each sample has a fresh random $x \sim \mathcal{N}(0, I_d)$.

**Why these specific frequencies matter.** This is not arbitrary. The gradient budget for each language is proportional to its frequency. At 60%, Chinese gets 60% of all gradient updates. Swahili at 1% gets 1/60th the gradient signal. The model has no incentive to develop shared representations between Chinese and Swahili because Swahili contributes almost nothing to the total loss. Any cross-lingual alignment that emerges is a side effect of weight sharing under capacity constraints, not an explicit optimization target. This is the key prediction of the toy theorem: $\alpha_\ell$ directly controls the singular value spectrum of $W$, and low-$\alpha$ languages contribute negligible cross-lingual structure. If the POC reproduces this — Procrustes R² degrades proportionally to $\alpha_\ell$ — then training frequency IS the mechanism behind the Qwen behavioral gradient.

### Test data (critical design choice)

Generate $M = 200$ raw problems $\{x^{(m)}\}_{m=1}^M$. For **every** language $\ell$, compute the corresponding pair $(\tilde{x}_\ell^{(m)}, \tilde{y}_\ell^{(m)})$ on the **same** raw problems. This gives us ground truth alignment: test sample $m$ in language $\ell$ and test sample $m$ in language $\ell'$ correspond to the **identical** reasoning problem $f(x^{(m)})$.

**Why this is critical.** In the real Qwen experiments, we translated the same math problems into multiple languages to create aligned test sets. Here we have exact alignment by construction — same $x$, different $R_\ell$. This is what enables Procrustes (which requires paired points) and what makes the NN accuracy metric meaningful (we can check if the nearest cross-lingual neighbor is the SAME problem).

### Four training configurations

**Config A — Sequential.** Train on all Chinese samples first (full convergence), then English, then Spanish, ..., then Swahili. Each language block trains until loss plateaus before moving to the next. This maximizes catastrophic forgetting and minimizes cross-lingual gradient interaction.

**Config B — Imbalanced random with one-hot (realistic).** Shuffle all $N$ samples randomly regardless of language. Standard mini-batch SGD. Each batch has $\approx 60\%$ Chinese, $\approx 20\%$ English, etc., in expectation. One-hot language indicator appended. This mimics real pre-training. **This is the primary config. Run this first.**

**Config B-blind — Imbalanced random, no one-hot.** Same as Config B but the model receives only $\tilde{x}_\ell$, no language indicator. The model must infer language from the data. Closer to how real transformers operate (no explicit language tag).

**Config C — Balanced random (control).** Override frequencies: $\alpha_\ell = 1/K$ for all $\ell$. Same total $N$, uniform allocation. Shuffle randomly. This maximizes cross-lingual gradient pressure. If Z doesn't emerge here, it doesn't emerge anywhere.

---

## 5. The Network

$$g_\theta : \mathbb{R}^{d+K} \to \mathbb{R}^d \quad \text{(Config B)} \qquad g_\theta : \mathbb{R}^{d} \to \mathbb{R}^d \quad \text{(Config B-blind)}$$

Architecture: MLP with $H$ hidden layers of width $w$. Activation: GELU (or ReLU). No skip connections.

$$g_\theta(z) = W_H \sigma(W_{H-1} \sigma(\cdots \sigma(W_1 z + b_1) \cdots) + b_{H-1}) + b_H$$

Suggested: $H = 6$, $w = 64$. Small enough that the network is forced to share representations, large enough to learn $f$.

**Why no skip connections.** Real transformers have residual connections ($h_{\ell+1} = h_\ell + f_\ell(h_\ell)$), which means the input embedding rides the residual stream through every layer. This is how the "moon vector" (language identity) survives to the output in the real model — it's written into the stream at layer 0 and never overwritten. Our MLP has no such mechanism. Each layer's output completely replaces the previous state. This means:

1. If the model learns to undo rotations early, that information is available at every subsequent layer (good for us — clean separation).
2. There is no "language watermark" riding through the network. The model can only know the language from the one-hot (Config B) or from the activation statistics (Config B-blind). There's no residual channel carrying language identity alongside reasoning.

**Known gap.** This means the synthetic model is a less faithful replica of the real transformer in one specific way: the moon vector phenomenon can't be reproduced here. The moon surgery experiment (Procrustes + swap language identity vector + ride the Chinese highway) requires residual connections. That experiment must be done on the real model, not the synthetic one. The synthetic POC tests only: can Procrustes alignment separate reasoning from coordinate entanglement? The moon is a separate question.

**Convergence criterion.** Per-language test MSE $< 0.01$ for all $K$ languages. If any language fails, the network hasn't learned $f$ for that language and the experiment is invalid. **Check Swahili especially** — with only 1% of training data (100 samples), the model may not learn $f$ for Swahili at all. If Swahili MSE is bad, that's informative (the model couldn't learn with so little data), but it means Swahili can't be used as a held-out test language for the extractor. In that case, use Japanese (4%) and Korean (2%) as held-out instead.

Denote the hidden activations at layer $h$ on input $z$ as:

$$a_h(z) = \sigma(W_h a_{h-1}(z) + b_h), \qquad a_0(z) = z$$

---

## 6. Phase 2 — Naive Extraction (assert failure)

**Purpose.** We need to show that standard extraction methods fail BEFORE introducing Procrustes. If naive methods work, the experiment is trivially solved and proves nothing about the real model. The pyrite must come up with the gold. We must reproduce the same "Z doesn't seem to exist" frustration that occurred with Qwen.

For each test problem $m$ and each language $\ell$, extract $a_h^\ell(m) = a_h(\text{input}_\ell^{(m)})$ at every hidden layer $h$.

### Test 2a — Activation patching

Fix a layer $h$. For each test problem $m$, replace the activations of language $\ell'$ with those of language $\ell$ at layer $h$, then continue the forward pass through layers $h+1, \ldots, H$:

$$\hat{y} = W_H \sigma(\cdots \sigma(W_{h+1}\, a_h^\ell(m) + b_{h+1}) \cdots) + b_H$$

Measure:

$$\text{MSE}_{\text{patch}}(h) = \frac{1}{M} \sum_{m=1}^{M} \left\| \hat{y}^{(m)} - \tilde{y}_{\ell'}^{(m)} \right\|^2$$

**Why we expect failure.** The weights between layer $h$ and the output were trained expecting activations in language $\ell'$'s coordinate system. Injecting language $\ell$'s activations puts them in the wrong coordinates. The weights try to continue processing as if they're still in $\ell'$-space, producing garbage. This is the same reason patching failed on Qwen — the reasoning content may be the same, but the coordinate packaging is different.

**Success criterion for this phase:** $\text{MSE}_{\text{patch}} \gg \text{MSE}_{\text{baseline}}$ at every layer. Patching breaks things.

**What if it doesn't break?** If patching works cleanly (especially for Chinese → English), the model may have learned to canonicalize coordinates early, making the remaining layers language-agnostic. Check the layer 1 cross-lingual distance (see Section 3 check). If distances are near zero, the model has already undone rotations and the experiment needs redesigning (try Config B-blind or smaller network).

### Test 2b — SVD subspace removal

At layer $h$, compute the cross-lingual difference matrix:

$$\Delta_h = \begin{bmatrix} a_h^1(1) - a_h^2(1) \\ \vdots \\ a_h^1(M) - a_h^2(M) \end{bmatrix} \in \mathbb{R}^{M \times w}$$

SVD: $\Delta_h = U \Sigma V^\top$. Take the top $k$ right singular vectors $V_{:k} \in \mathbb{R}^{w \times k}$ as "language directions." Project activations onto the orthogonal complement:

$$P_{\perp} = I_w - V_{:k} V_{:k}^\top$$

Compute cross-lingual nearest-neighbor accuracy on projected activations:

$$\text{NN}(h, k) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}\!\left[\arg\min_{m'} \left\| P_\perp a_h^1(m) - P_\perp a_h^2(m') \right\| = m\right]$$

### Test 2c — Random subspace control

Repeat Test 2b but replace $V_{:k}$ with a random $k$-dimensional subspace (Haar-distributed orthonormal frame). Run $T = 50$ random trials.

**Success criterion for this phase:** $\text{NN}_{\text{SVD}} \approx \text{NN}_{\text{random}}$. The SVD-identified "language subspace" is no better than a random subspace at isolating reasoning. This is the pyrite — the SVD directions capture coordinate differences, not the language/reasoning boundary. This directly mirrors what happened on Qwen: when we patched Z versus random subspaces, random did equally well (or equally badly), suggesting the identified Z wasn't causally special.

---

## 7. Phase 3 — Procrustes Geometric Check

At each layer $h$, Procrustes-align all languages to language 1 (Chinese):

$$R_h^\ell = \arg\min_{R \in O(w)} \left\| A_h^\ell R - A_h^1 \right\|_F$$

where $A_h^\ell \in \mathbb{R}^{M \times w}$ is the matrix of activations for language $\ell$ at layer $h$. The closed-form solution is:

$$A_h^{\ell\top} A_h^1 = U \Sigma V^\top, \qquad R_h^\ell = U V^\top$$

Compute cross-lingual NN accuracy on Procrustes-aligned activations $\tilde{A}_h^\ell = A_h^\ell R_h^\ell$ versus raw activations $A_h^\ell$.

**Note (Sharma's correction).** This is a *geometric sanity check only*. It measures whether the representations are related by rotation, but it does NOT demonstrate that the original model can use aligned activations. The aligned activations cannot be fed back through the model's remaining layers — the weights expect the unaligned coordinates and will attempt to undo a rotation that has already been undone, producing garbage ("double correction"). Phase 3 answers "is the relationship rotational?" Phase 4 answers "can we exploit that?"

**What to look for in the NN curves.** Plot Procrustes-aligned NN accuracy versus raw NN accuracy at each layer. If Procrustes helps massively at early layers but not late layers, the rotation entanglement is being resolved by the network itself as depth increases. If Procrustes helps at ALL layers, the network never fully resolves the rotations. The layer where Procrustes-aligned NN peaks is a candidate for layer A (the inflection point).

---

## 8. Phase 3b — Rotate and Ride the Chinese Highway

**Purpose.** Before training a separate extractor (Phase 4), test whether the original model can process Procrustes-aligned activations directly. The reasoning: if we align everything to Chinese coordinates, and the weights are 60% optimized for Chinese, then the remaining layers should process the aligned activations well — because from the model's perspective, they look Chinese.

### Procedure

1. Run language $\ell$ (e.g., Swahili) through the network to layer $A$.
2. Procrustes-align: $\tilde{a}_A^\ell = a_A^\ell \cdot R_A^\ell$ (rotate to Chinese coordinates).
3. Continue the forward pass through the original model's remaining layers ($A+1$ through output), feeding $\tilde{a}_A^\ell$ as if it were a Chinese activation.
4. Get output $\hat{y}$. This should be in Chinese coordinates (the model thinks it's processing Chinese).
5. Compare $\hat{y}$ to $\tilde{y}_1^{(m)} = R_1 f(x^{(m)})$ (the Chinese-coordinate correct answer).

### Why this might work (and why it might not)

**Argument for:** The weights between $A$ and output were trained predominantly on Chinese data (60% of gradient). They're optimized for processing Chinese-coordinate activations. Procrustes alignment makes Swahili look Chinese. The weights should handle it.

**Argument against:** The weights are a COMPROMISE across all languages, not purely Chinese-optimal. 40% of the gradient came from other languages and pushed the weights in other directions. The aligned Swahili activations may be close to Chinese but not close enough for the compromised weights to process correctly.

**Also relevant (from the MLP-specific gap):** In this synthetic setup, there's no residual stream, so there's no "moon vector" issue. The model's only source of language identity is the one-hot (Config B) or the input statistics (Config B-blind). By layer $A$, the one-hot signal has been mixed into the activations. When we Procrustes-align, we rotate the entire activation including whatever remains of the one-hot signal. The model may or may not be confused by receiving a Chinese-looking activation when it expected Swahili. In Config B-blind, there is no one-hot, so this concern doesn't apply.

### Success criterion

$\text{MSE}(\hat{y}, \tilde{y}_{\text{Chinese}}) < 2 \times \text{MSE}_{\text{baseline Chinese}}$. The model produces reasonable outputs for Procrustes-aligned Swahili routed through the Chinese pathway. Compare to a control: do the same but with a random rotation instead of Procrustes. If random rotation is equally bad, Procrustes alignment is doing something real.

---

## 9. Phase 4 — Pack-a-Punch

This is the real test. We train a **separate** model to learn the reasoning map between Procrustes-aligned representations.

**Why a separate model.** Phase 3b asks "can the ORIGINAL model handle aligned activations?" If it can, great. But if it can't — because the weights are a compromise, or because of double-correction, or because of residual one-hot contamination — we need an alternative. Phase 4 sidesteps all of those issues by training a fresh model that has never seen rotated coordinates. It learns ONLY the mapping from aligned-comprehension to aligned-reasoning-complete. It has no expectations about coordinate systems because it was trained exclusively on Procrustes-aligned data.

### Step 1: Identify layers

Let $A$ = the "inflection layer" where Procrustes-aligned NN accuracy peaks or plateaus (determined empirically from Phase 3 curves).

Let $B$ = the last hidden layer before output (reasoning complete, pre-decode).

### Step 2: Procrustes-align at BOTH endpoints

At layer $A$: compute $R_A^\ell$ aligning each language to Chinese. Define $\tilde{A}_A^\ell = A_A^\ell R_A^\ell$.

At layer $B$: compute $R_B^\ell$ aligning each language to Chinese. Define $\tilde{A}_B^\ell = A_B^\ell R_B^\ell$.

**Why align at both endpoints.** The extractor learns $\tilde{a}_A \to \tilde{a}_B$. If we only aligned at $A$, the targets $a_B$ would still be in language-specific coordinates, and the extractor would have to learn both the reasoning transformation AND a per-language output rotation. By aligning at $B$ too, we strip the coordinate system from both input and output. The extractor only needs to learn reasoning.

### Step 3: Train the reasoning extractor

Split languages into train and test:

$$\mathcal{T}_{\text{train}} = \{1, 2, 3, 4, 5\} \quad (\text{zh, en, es, ar, ja})$$
$$\mathcal{T}_{\text{test}} = \{6, 7\} \quad (\text{ko, sw — never seen by extractor})$$

**Why held-out LANGUAGES, not held-out PROBLEMS.** We could hold out 20% of problems instead of 2 languages. But that would only test generalization to new problems within known languages. Holding out entire languages tests something stronger: does the extractor learn a language-agnostic reasoning function? If Korean and Swahili — languages the extractor has never seen — produce high R², then the reasoning map is truly language-independent.

Training data for the extractor:

$$\mathcal{D}_{\text{ext}} = \bigcup_{\ell \in \mathcal{T}_{\text{train}}} \left\{ \left( \tilde{a}_A^\ell(m),\; \tilde{a}_B^\ell(m) \right) \right\}_{m=1}^{M}$$

Train a small MLP:

$$\phi_\psi : \mathbb{R}^w \to \mathbb{R}^w, \qquad \psi^* = \arg\min_\psi \sum_{(X, Y) \in \mathcal{D}_{\text{ext}}} \|  \phi_\psi(X) - Y \|^2$$

Architecture: 2 hidden layers, width 32. This is intentionally tiny — if it works, the reasoning map is simple.

### Step 4: Evaluate on held-out languages

$$R^2_{\text{test}} = 1 - \frac{\sum_{\ell \in \mathcal{T}_{\text{test}}} \sum_m \| \phi_{\psi^*}(\tilde{a}_A^\ell(m)) - \tilde{a}_B^\ell(m) \|^2}{\sum_{\ell \in \mathcal{T}_{\text{test}}} \sum_m \| \tilde{a}_B^\ell(m) - \bar{Y} \|^2}$$

**Also compute per-language R².** The per-language breakdown is where the Qwen gradient should appear. In Config B:
- R² for Korean (2% training data): moderate?
- R² for Swahili (1% training data): lower?
- If so, the degradation correlates with training frequency. That IS the Qwen pattern.

### Step 5: Controls

**Control 1 — No Procrustes.** Train extractor on raw $a_A^\ell \to a_B^\ell$ without alignment. If $R^2$ is comparable, Procrustes was unnecessary and the pyrite argument is wrong.

**Control 2 — Scrambled.** Train extractor on Procrustes-aligned $\tilde{a}_A^\ell(m) \to \tilde{a}_B^\ell(\sigma(m))$ where $\sigma$ is a random permutation that breaks problem correspondence. If $R^2$ is comparable, the extractor is memorizing statistical structure, not learning a reasoning map.

**Control 3 — Random subspace.** Replace Procrustes alignment with projection onto a random $k$-dimensional subspace. If $R^2$ is comparable, the improvement isn't specific to rotational alignment.

### Decision matrix

| Result | Interpretation |
|--------|----------------|
| $R^2_{\text{Procrustes}} \gg R^2_{\text{raw}} \gg R^2_{\text{scrambled}}$ | **Gold.** Procrustes separates pyrite from gold. Reasoning is extractable once coordinates are aligned. The factorization is real. |
| $R^2_{\text{Procrustes}} \approx R^2_{\text{raw}} \gg R^2_{\text{scrambled}}$ | Pyrite was never the problem. Reasoning is extractable without alignment. The rotation entanglement hypothesis is wrong. Model learned to canonicalize on its own. |
| $R^2_{\text{Procrustes}} \gg R^2_{\text{raw}} \approx R^2_{\text{scrambled}}$ | Procrustes helps but raw extraction is at floor. Strong evidence for coordinate entanglement. |
| $R^2_{\text{Procrustes}} \approx R^2_{\text{raw}} \approx R^2_{\text{scrambled}}$ | Nothing works. Entanglement too deep for linear methods. Reconsider layer choice or abandon linear decomposition. |

### Step 6: Compare across training configs

Run Phases 2–4 for all four training configurations (A, B, B-blind, C).

**Predictions:**

| Config | Phase 2 (naive) | Phase 3b (highway) | Phase 4 (extractor) |
|--------|-----------------|---------------------|---------------------|
| A (sequential) | Fails | Fails | Fails or weak. No Z formed. |
| B (imbalanced + one-hot) | Fails for low-resource | Partial — works for zh/en, degrades for sw | Works for zh/en, degrades for ko/sw. The Qwen gradient. |
| B-blind (imbalanced, no tag) | Fails | Harder — no tag to spoof | Most realistic. If Procrustes helps HERE, strong result. |
| C (balanced) | Fails | Works for all | Works for all including held-out. Cleanest Z. |

If Config B reproduces the Qwen behavioral gradient — high $R^2$ for high-resource held-out languages, degraded $R^2$ for low-resource — then training data frequency is the mechanism, and the synthetic POC explains the real phenomenon.

---

## 10. Sanity Checks and Empirical Validations

### Pre-training checks

- [ ] Verify $f$ is non-trivial: a single linear layer cannot approximate it. Test by training a linear model $Wx + b$ on the data — should get MSE $\gg 0.01$.
- [ ] Verify rotations are proper orthogonal: $\|R_\ell R_\ell^\top - I\| < 10^{-10}$ for all $\ell$.
- [ ] Verify training data frequencies: count samples per language, confirm they match $\alpha_\ell$.

### Post-training checks

- [ ] **Per-language MSE.** All languages below 0.01? If Swahili is bad, it can't be a test language.
- [ ] **Layer 1 cross-lingual distance.** For all $(\ell, \ell')$ pairs, compute $\frac{1}{M}\sum_m \|a_1^\ell(m) - a_1^{\ell'}(m)\|$. If Chinese-English $\approx 0$ but Chinese-Swahili $\gg 0$: model partially undid rotations proportional to data frequency (GOOD). If all $\approx 0$: model is too powerful, switch to B-blind or smaller network. If all $\gg 0$: model didn't learn to undo rotations at all, even for Chinese (check training convergence).
- [ ] **t-SNE at layers 1, 3, 5 (output).** Does it look like the Qwen layer evolution? Expect: layer 1 has some language separation (especially for low-resource), layer 3 shows convergence, layer 5 (output) shows... what? In the MLP there's no decode phase (no residual stream to carry language back). Output should be mixed. This is a DIFFERENCE from Qwen worth noting.

### Phase 2 checks

- [ ] **Patching breaks for at least some language pairs.** If Chinese → English patching works fine, check whether the model canonicalized those two (layer 1 distance check above).
- [ ] **SVD patch ≈ random patch.** The z-score between SVD subspace MSE and random subspace MSE distribution should be < 2.0. If SVD is significantly better than random, the pyrite/gold distinction doesn't apply — the SVD directions ARE meaningful, and we don't need Procrustes.

### Phase 3 checks

- [ ] **Procrustes NN accuracy > raw NN accuracy at most layers.** If Procrustes doesn't help, the representations aren't related by rotation (surprising given we BUILT them as rotations — check for bugs).
- [ ] **Procrustes NN accuracy varies by language.** Expect Chinese-English alignment to be best, Chinese-Swahili worst. Plot NN accuracy vs training frequency $\alpha_\ell$. If there's a clean correlation, that's a figure for the paper.

### Phase 4 checks

- [ ] **Scrambled R² is at floor** (near 0 or negative). If scrambled control has high R², the extractor is learning language-level statistical structure, not problem-level reasoning.
- [ ] **Per-language R² breakdown.** The money plot. R² for each held-out language separately. Does it correlate with $\alpha_\ell$?
- [ ] **Extractor works on TRAINING languages too.** Compute R² on Chinese and English (seen during extractor training). Should be highest. If held-out R² > training R², something is wrong.

### Cross-config comparisons

- [ ] **Config C R² > Config B R² > Config A R².** If balanced training produces better Z, training frequency is the mechanism.
- [ ] **Config B-blind vs Config B.** If B-blind produces WORSE raw extraction but SIMILAR Procrustes extraction, the one-hot was helping the model canonicalize (expected), but Procrustes compensates for the lack of tag.

---

## 11. Implementation Notes

**Framework.** PyTorch or numpy+sklearn. The model is tiny — sklearn's MLPRegressor works for a quick version, PyTorch for more control over activation extraction.

**Activation extraction.** Need to hook into the forward pass and save intermediate activations at every hidden layer. In PyTorch, use forward hooks. In sklearn, manually compute the forward pass layer by layer.

**Procrustes.** `scipy.linalg.orthogonal_procrustes`. Takes two matrices of paired points, returns the optimal rotation. One line of code. Make sure the input matrices have the same shape and the rows correspond to the same problems.

**t-SNE.** `sklearn.manifold.TSNE`. Use perplexity 20-30 for 200 points × 7 languages. Color by language, shape by... actually there are no "problem categories" in the synthetic setup (all problems are random vectors). Consider adding categories: e.g., problems where $x_1 > 0$ vs $x_1 < 0$, or problems where $\|x\|$ is above/below median. This lets you check whether the t-SNE shows topic clustering analogous to Qwen's math categories.

**Compute time.** With $d = 10$, $w = 64$, $H = 6$, $N = 10{,}000$, $M = 200$: training is seconds. Activation extraction is trivial. Procrustes is instant. The whole experiment is minutes on a CPU. The 4070 is overkill. Save it for Qwen.

---

## 12. Connection to Qwen: What Maps and What Doesn't

| Synthetic POC | Real Qwen | Maps? |
|---------------|-----------|-------|
| Rotation $R_\ell$ | Language-specific embedding geometry | Partially. Real transforms may be nonlinear. Procrustes tests the linear component. |
| One-hot language tag | Token co-occurrence patterns | Loosely. Real model infers language, doesn't get a tag. Config B-blind is closer. |
| MLP, no skip connections | Transformer with residual stream | No. Moon vector cannot be reproduced. Moon surgery must be done on real model. |
| Random $x \sim \mathcal{N}(0, I)$ | Math problems with category structure | No. Synthetic has no problem categories. Real model shows topic clustering. |
| Mean-pooled single vector | Mean-pooled across variable-length token sequence | Roughly. Real model's mean-pool averages across different numbers of tokens per language. |
| 6 layers | 36 layers | Scale difference. The "inflection layer" concept should transfer but the specific layer will differ. |
| Input-pass only | Input-pass only (our existing data) + generation-time (future work) | Matches for existing data. Generation-time analysis is separate. |

**The key thing that DOES map.** The core question: can Procrustes alignment at an intermediate layer separate reasoning from coordinate entanglement? If yes in synthetic, test on Qwen. If no in synthetic, there's no point trying on Qwen.

---

*Sharma, March 2026*
