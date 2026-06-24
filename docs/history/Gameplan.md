# Z HUNT v2: THE KERNEL APPROACH

## What changed from v1

v1 used CKA (a single number per layer) and mean-pooling (throwing away
sequence structure, assuming uniform density per token). Both are crude.

v2 uses ARD-MMD: treats activations as empirical measures on R^4096,
compares distributions not vectors, handles Chinese/English token count
mismatch by construction, and extracts Z as the dimensions where the
ARD lengthscale stays finite. No mean-pooling. No alignment. No
assumptions about where Z lives.

The three numbers that matter are now different:

| v1 Number | v2 Number | What it means |
|---|---|---|
| k* (peak CKA layer) | **ℓ** (lengthscale vector, per layer) | Which dimensions are Z, at each layer |
| R² (linear probe) | **|Z_k|** (# surviving dimensions per layer) | How big Z is at each layer |
| rank_95 (SVD of bridge) | **ℓ spectrum shape** | Whether Z has cascade-like structure |

---

## HOUR 1–2: GET QWEN RUNNING IN PYTHON

Same as v1. No changes here.

- [ ] `pip install transformers accelerate bitsandbytes torch`
- [ ] Load Qwen3-8B in 4-bit:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```
- [ ] Confirm generation works
- [ ] **DONE WHEN:** Python generates text matching LM Studio quality

---

## HOUR 2–3: HOOKS + ACTIVATION EXTRACTION (FULL SEQUENCE, NOT POOLED)

Key difference from v1: we keep the FULL sequence of token activations,
not mean-pooled. Each prompt at each layer gives us a SET of vectors,
not one vector.

- [ ] Hook code (same as v1):
```python
activations = {}
def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output[0].detach().cpu()
    return hook
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(hook_fn(f"layer_{i}"))
```
- [ ] Run one Chinese prompt. Check shape: `[1, n_tokens_zh, 4096]`
- [ ] Run one English prompt. Check shape: `[1, n_tokens_en, 4096]`
- [ ] Verify n_tokens_zh ≠ n_tokens_en (this is expected and fine)
- [ ] Store as squeezed 2D tensors: `[n_tokens, 4096]`
- [ ] **DONE WHEN:** You have two matrices of different row count, same column count

---

## HOUR 3–4: IMPLEMENT MMD WITH ARD KERNEL

This is the mathematical core. Take your time here.

- [ ] Implement the ARD-RBF kernel:
```python
def ard_kernel(X, Y, log_lengthscales):
    """
    X: [n, d], Y: [m, d], log_lengthscales: [d]
    Returns: [n, m] kernel matrix
    """
    # Scale each dimension by its lengthscale
    ell = torch.exp(log_lengthscales)  # [d]
    X_scaled = X / ell.unsqueeze(0)     # [n, d]
    Y_scaled = Y / ell.unsqueeze(0)     # [m, d]

    # Squared distances
    dists = torch.cdist(X_scaled, Y_scaled, p=2).pow(2)  # [n, m]
    return torch.exp(-0.5 * dists)
```

- [ ] Implement MMD²:
```python
def mmd_squared(X, Y, log_lengthscales):
    """
    X: [n, d], Y: [m, d] — two SETS of activation vectors
    These are empirical measures. Different sizes OK.
    """
    K_xx = ard_kernel(X, X, log_lengthscales)  # [n, n]
    K_yy = ard_kernel(Y, Y, log_lengthscales)  # [m, m]
    K_xy = ard_kernel(X, Y, log_lengthscales)  # [n, m]

    # Unbiased estimator
    n = X.shape[0]
    m = Y.shape[0]

    mmd = (K_xx.sum() - K_xx.trace()) / (n * (n - 1)) \
        + (K_yy.sum() - K_yy.trace()) / (m * (m - 1)) \
        - 2 * K_xy.mean()

    return mmd
```

- [ ] Test: generate two random Gaussian clouds, verify MMD ≈ 0 for same
  distribution and MMD > 0 for different distributions
- [ ] **DONE WHEN:** MMD function works and makes sense on toy data

---

## HOUR 4–5: HARDCODE PROMPT PAIRS + EXTRACT ALL ACTIVATIONS

- [ ] Hardcode 20–30 paired prompts (Chinese + English):
```python
pairs = [
    ("设 f(n) 为 n 的二进制表示中 1 的个数。满足 1 ≤ n ≤ 2025 且 f(n) = 3 的整数 n 有多少个？",
     "Let f(n) be the number of 1s in the binary representation of n. How many integers n with 1 ≤ n ≤ 2025 satisfy f(n) = 3?"),
    # ... more pairs: combinatorics, calculus, algebra, etc.
]
```

- [ ] For each pair, for each layer, store the FULL activation matrices:
```python
all_activations = {}  # all_activations[(lang, pair_idx, layer)] = tensor

for idx, (zh, en) in enumerate(pairs):
    # Chinese
    inputs = tokenizer(zh, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    for k in range(36):
        all_activations[("zh", idx, k)] = activations[f"layer_{k}"].squeeze(0).cpu()

    # English
    inputs = tokenizer(en, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    for k in range(36):
        all_activations[("en", idx, k)] = activations[f"layer_{k}"].squeeze(0).cpu()
```

- [ ] Verify: `all_activations[("zh", 0, 17)].shape` should be `[n_tokens, 4096]`
- [ ] Save to disk: `torch.save(all_activations, "activations.pt")`
- [ ] **DONE WHEN:** All activations saved. ~20 pairs × 2 languages × 36 layers = 1440 tensors

---

## HOUR 5–8: OPTIMIZE ARD LENGTHSCALES (THE BIG MOMENT)

For each layer k, find the lengthscale vector that minimizes MMD across
all prompt pairs, with L1 sparsity penalty on 1/ℓ.

The optimization: most dimensions should have ℓ → ∞ (ignored, language-
specific). The surviving dimensions with finite ℓ are Z.

- [ ] Optimization loop for a single layer:
```python
def find_Z_at_layer(all_activations, layer_k, pairs, lr=0.01, steps=500, lam=0.01):
    """
    Returns optimized log-lengthscales for layer k.
    Dimensions where ℓ is small = Z (cross-lingually shared).
    Dimensions where ℓ is large = language-specific.
    """
    # Initialize: all lengthscales equal (no prior about which dims matter)
    log_ell = torch.zeros(4096, requires_grad=True)
    optimizer = torch.optim.Adam([log_ell], lr=lr)

    for step in range(steps):
        total_loss = 0.0

        for idx in range(len(pairs)):
            X_zh = all_activations[("zh", idx, layer_k)]
            X_en = all_activations[("en", idx, layer_k)]

            # MMD between Chinese and English activations
            mmd = mmd_squared(X_zh, X_en, log_ell)
            total_loss = total_loss + mmd

        # L1 penalty on inverse lengthscale = sparsity on active dimensions
        # exp(-log_ell) = 1/ell
        sparsity = lam * torch.exp(-log_ell).sum()

        loss = total_loss + sparsity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            n_active = (torch.exp(log_ell) < 10.0).sum().item()  # threshold
            print(f"Layer {layer_k}, Step {step}: "
                  f"MMD = {total_loss.item():.6f}, "
                  f"Active dims = {n_active}/4096")

    return log_ell.detach()
```

- [ ] Run for all 36 layers:
```python
results = {}
for k in range(36):
    print(f"\n=== Layer {k} ===")
    results[k] = find_Z_at_layer(all_activations, k, pairs)
```

- [ ] Extract Z dimensionality at each layer:
```python
Z_sizes = []
for k in range(36):
    ell = torch.exp(results[k])
    n_active = (ell < 10.0).sum().item()  # dimensions with finite lengthscale
    Z_sizes.append(n_active)

plt.plot(range(36), Z_sizes)
plt.xlabel("Layer")
plt.ylabel("|Z| (number of shared dimensions)")
plt.title("Dimensionality of Z across layers")
plt.savefig("Z_dimensionality.png")
```

- [ ] **THE FIRST BIG PLOT:** |Z_k| vs layer index

- [ ] Extract the lengthscale spectrum at the most interesting layer:
```python
best_layer = Z_sizes.index(min(Z_sizes))  # or max, depending on interpretation
# actually: the layer where Z is most COMPACT is most interesting
# that's where the model compresses to the fewest shared dimensions

ell = torch.exp(results[best_layer])
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(torch.log10(ell).numpy(), bins=100)
plt.xlabel("log₁₀(lengthscale)")
plt.ylabel("Count")
plt.title(f"Lengthscale distribution at layer {best_layer}")

plt.subplot(1, 2, 2)
sorted_ell, _ = torch.sort(ell)
plt.plot(sorted_ell.numpy())
plt.xlabel("Dimension (sorted)")
plt.ylabel("Lengthscale")
plt.title(f"Sorted lengthscale spectrum at layer {best_layer}")
plt.savefig("lengthscale_spectrum.png")
```

- [ ] **THE SECOND BIG PLOT:** The lengthscale spectrum

- [ ] **DONE WHEN:** You're staring at both plots

---

## HOUR 8: READ THE PLOTS

### The |Z_k| vs layer plot tells you WHERE:

**IF |Z| is small at middle layers and large at early/late layers:**
→ The model compresses to a compact shared representation in the middle.
→ The inverted-U prediction holds. Language is stripped early, reasoning
  happens in a compact Z, language is re-added late.

**IF |Z| is roughly constant across layers:**
→ Z is a stable thread — the same dimensions are shared throughout.
→ Your information-theoretic argument holds (Z exists as a function
  property) but it's distributed, not localized.

**IF |Z| grows monotonically (more shared dims in deeper layers):**
→ The model progressively strips language as it goes deeper.
→ Z lives in the deepest layers. Late-layer patching would work best.

**IF |Z| ≈ 0 everywhere:**
→ No dimensions are consistently shared across languages.
→ The ARD kernel found no cross-lingual structure.
→ Hypothesis in serious trouble. Debug or rethink.

### The lengthscale spectrum tells you WHAT Z LOOKS LIKE:

**IF bimodal (cluster of small ℓ + cluster of large ℓ):**
→ Clean separation. Z is well-defined. The number of small-ℓ dims is |Z|.
→ **Best case.** The model genuinely separates language from reasoning.

**IF power-law (smooth decay from small to large ℓ):**
→ No clean boundary between Z and Z⊥. Separation is graded.
→ Still useful — you can threshold wherever you want — but the "clean
  subspace" dream is weaker.

**IF uniform (all ℓ roughly equal):**
→ The ARD kernel couldn't find ANY structure. All dimensions contribute
  equally to cross-lingual similarity/dissimilarity.
→ Z isn't a subspace. Rethink everything.

---

## HOUR 9–11: VALIDATE Z (DOES IT ACTUALLY CONTAIN REASONING?)

Finding shared dimensions isn't enough. Those dimensions could be shared
but irrelevant (e.g., both languages encode punctuation the same way).
You need to verify that Z carries REASONING content.

- [ ] Extract Z mask at the best layer:
```python
ell = torch.exp(results[best_layer])
Z_mask = (ell < threshold).float()  # 1 for Z dims, 0 for Z⊥ dims
print(f"Z has {Z_mask.sum().item():.0f} dimensions")
```

- [ ] Project activations onto Z and Z⊥:
```python
def project(X, mask):
    return X * mask.unsqueeze(0)  # zero out non-Z dimensions

# For each prompt, get Z-projected and Z⊥-projected activations
# Use mean of token activations for simplicity here (justified because
# we already identified WHICH dims matter; now just measuring content)
```

- [ ] **Test 1: Same problem, different languages should be CLOSE in Z**
```python
# For each prompt pair i:
#   d_Z(i)  = distance between zh and en in Z-space
#   d_Z⊥(i) = distance between zh and en in Z⊥-space
#
# Prediction: d_Z ≪ d_Z⊥
# (languages are similar in Z, different in Z⊥)
```

- [ ] **Test 2: Different problems, same language should be FAR in Z**
```python
# For each pair of different problems (i, j) in the same language:
#   d_Z(i,j) = distance in Z-space
#
# Prediction: d_Z(same problem, diff lang) ≪ d_Z(diff problem, same lang)
# This means Z encodes PROBLEM IDENTITY, not language identity
```

- [ ] **Test 3: Does Z predict the answer?**
```python
# Train a tiny classifier on Z-projected activations to predict
# whether the model got the answer right or wrong.
#
# Train the same classifier on Z⊥-projected activations.
#
# Prediction: Z-projected classifier is BETTER.
# (Reasoning information lives in Z, not in Z⊥)
```

- [ ] Compute and print all three tests
- [ ] **DONE WHEN:** You know whether Z contains reasoning or just shared noise

---

## HOUR 11–13: ACTIVATION PATCHING (PHASE 2, NOW TARGETED)

v1 patching was crude: swap ALL activations at layer k*.
v2 patching is surgical: swap ONLY the Z dimensions.

- [ ] Choose 5 problems where Chinese is right, English is wrong
- [ ] For each problem:
```python
def z_targeted_patch(chinese_acts, Z_mask):
    """Only replace the Z dimensions. Leave Z⊥ (language) untouched."""
    def hook(module, input, output):
        patched = output[0].clone()
        # Only overwrite Z dimensions with Chinese values
        patched[:, :, Z_mask.bool()] = chinese_acts[:, :, Z_mask.bool()].to(patched.device)
        return (patched,) + output[1:]
    return hook

# Attach at best_layer, run English inference
handle = model.model.layers[best_layer].register_forward_hook(
    z_targeted_patch(zh_activations, Z_mask)
)
output = model.generate(english_input)
handle.remove()
```

- [ ] Record: original English answer vs Z-patched English answer
- [ ] Compare to FULL patching (swap everything, not just Z)
  - If Z-only patching works but full patching breaks → Z is real and Z⊥ is language-essential
  - If both work → Z identification wasn't necessary, raw swap suffices
  - If neither works → sequence length mismatch may be the issue (see below)

### Handling the sequence length mismatch for patching:

Patching is harder than MMD because you can't just compare distributions —
you need to INJECT activations at specific positions. Chinese has n tokens,
English has m tokens. You can't replace n positions with m values.

Options:
- **Patch only the last token** (the model's "summary" state). Simplest.
- **Patch the mean** (replace each English position's Z-dims with the mean
  Chinese Z-dims). Crude but tests the concept.
- **Optimal transport alignment** (find the best matching between Chinese
  and English token positions, then patch matched pairs). Most principled
  but harder to implement.

Start with last-token patching. It's enough to test the concept.

- [ ] **DONE WHEN:** You have a table of (original, Z-patched, full-patched) answers

---

## HOUR 13–15: THE BRIDGE (PHASE 3, KERNEL VERSION)

If patching works, train the bridge. But now informed by the Z mask.

- [ ] Collect paired mean activations at best_layer, projected onto Z:
```python
zh_vecs = []  # [n_problems, |Z|]
en_vecs = []  # [n_problems, |Z|]
for idx in range(len(pairs)):
    zh_act = all_activations[("zh", idx, best_layer)]
    en_act = all_activations[("en", idx, best_layer)]
    # Mean over tokens, then mask to Z dims only
    zh_vecs.append(zh_act.mean(0)[Z_mask.bool()])
    en_vecs.append(en_act.mean(0)[Z_mask.bool()])

zh_matrix = torch.stack(zh_vecs)  # [N, |Z|]
en_matrix = torch.stack(en_vecs)  # [N, |Z|]
```

- [ ] Linear bridge WITHIN Z only:
```python
# W maps English-Z to Chinese-Z. Size: [|Z|, |Z|], much smaller than [4096, 4096]
W = torch.linalg.lstsq(en_matrix, zh_matrix).solution
predictions = en_matrix @ W
residuals = zh_matrix - predictions
r_squared = 1 - (residuals.norm()**2 / (zh_matrix - zh_matrix.mean(0)).norm()**2)
print(f"R² within Z = {r_squared:.4f}")
print(f"Bridge size: {W.shape[0]}x{W.shape[1]} = {W.numel()} parameters")
```

This is better than v1's bridge for two reasons:
  1. The matrix is |Z|×|Z| instead of 4096×4096. If |Z| = 200, that's
     40,000 parameters instead of 16 million. Way less overfitting risk.
  2. You're only mapping within the reasoning subspace. The language dims
     are left alone. The bridge does one job: rotate English reasoning to
     look like Chinese reasoning.

- [ ] **THE R² NUMBER:** same interpretation as v1
  - R² > 0.9 within Z → the two languages are rotations of each other in Z-space
  - R² > 0.8 → thin wrappers, strong result
  - R² < 0.5 → even within Z, the relationship is nonlinear

- [ ] **DONE WHEN:** You have R² and bridge size

---

## HOUR 15–16: ANALYZE THE BRIDGE AND Z STRUCTURE

- [ ] SVD of the bridge (now a small matrix):
```python
U, S, Vh = torch.linalg.svd(W)
plt.plot(S.numpy())
plt.xlabel("Component")
plt.ylabel("Singular Value")
plt.title(f"SVD of bridge within Z (|Z|={W.shape[0]})")
plt.savefig("bridge_svd.png")
```

- [ ] Check if W is approximately orthogonal:
```python
WtW = W.T @ W
identity_err = (WtW - torch.eye(W.shape[1])).norm() / W.shape[1]
print(f"Orthogonality error: {identity_err:.4f}")
# If small → W is a rotation → languages are isometric copies in Z
```

- [ ] Check which Z dimensions are most/least preserved by the bridge:
```python
# The singular values of W tell you:
# Large σ → that Z-direction is strongly mapped (important for bridge)
# Small σ → that Z-direction is weakly mapped (maybe not really shared)
```

- [ ] **DONE WHEN:** You know if the bridge is a rotation, a projection, or something messier

---

## HOUR 16–17: WRITE IT UP

One document. Four plots. Three numbers.

### The four plots:
1. **|Z_k| vs layer** — where Z lives in the network
2. **Lengthscale spectrum** — what Z looks like (bimodal? power-law? uniform?)
3. **Validation scatter** — same-problem-diff-language close in Z? diff-problem far?
4. **Bridge SVD** — is the bridge a rotation?

### The three numbers:
1. **|Z|** at best layer — how many dimensions reasoning needs
2. **R²** of bridge within Z — how linear the cross-lingual mapping is
3. **Orthogonality error** of bridge — is it a rotation?

### Frame the narrative:
- "We treat transformer activations as empirical measures on R^d"
- "ARD-MMD with sparsity penalty identifies language-agnostic dimensions"
- "~N dimensions (out of 4096) are sufficient for cross-lingual reasoning agreement"
- "Within this subspace, the two languages are related by an approximate rotation"
- This is the first density-invariant, nonparametric method for extracting
  language-agnostic reasoning subspaces from multilingual LLMs

---

## HOUR 17–20: BUFFER + EXTENSIONS

- [ ] Debug time (you will need this)
- [ ] Ablation: try different λ (sparsity penalty). Does |Z| change much?
- [ ] Ablation: try different kernel bandwidth initialization
- [ ] Add Spanish activations. Does Spanish look more like Chinese than
  English in Z-space? (This tests the density argument)
- [ ] Try the optimization on just 5 prompts vs 20 prompts. Is Z stable?
  (If Z changes a lot with more data, it's not robust)

---

## HOUR 20–24: STRETCH GOALS

- [ ] Proof-geometry: project a solving trajectory onto Z. Look for
  directional changes corresponding to proof steps.
- [ ] Load LLaMA-8B. Repeat Z extraction. Compare Z_Qwen vs Z_LLaMA.
- [ ] Inference-time bridge: actually deploy the |Z|×|Z| rotation at
  best_layer during English inference. Does the OUTPUT improve?

---

## COMPUTATIONAL NOTES

### Memory budget:
- 20 prompts × 2 languages × 36 layers × ~50 tokens × 4096 dims × 4 bytes
- ≈ 20 × 2 × 36 × 50 × 4096 × 4 ≈ 2.4 GB
- Fits in RAM. Save to disk anyway.

### MMD computation cost:
- Each MMD call: O(n² × d) where n ≈ 50 tokens, d = 4096
- Per optimization step: 20 pairs × MMD = 20 × 50² × 4096 ≈ 200M ops
- 500 steps × 36 layers = 18,000 optimizations
- Total: ~3.6T ops. On GPU: minutes. On CPU: maybe an hour.
- **If slow:** subsample tokens (use 20 random tokens per prompt instead of all)

### Potential CUDA issue:
- The ARD optimization needs gradients through the kernel. This is
  4096 parameters. PyTorch autograd handles this fine.
- If OOM on kernel matrices: compute MMD in mini-batches over prompt pairs

---

## EMERGENCY SHORTCUTS

**If ARD optimization doesn't converge:**
Fall back to simpler approach — compute per-dimension variance of
(zh_activation - en_activation) across prompts. Dimensions with low
cross-lingual variance are Z candidates. This is a poor man's ARD
without the kernel, but it gives you a starting mask.

**If everything is too slow:**
Subsample to 10 prompts and 10 layers (every 3rd layer). Get the shape
of the result first, refine later.

**If lengthscale spectrum is uniform (no Z found):**
Try: instead of comparing at each layer independently, compare across
pairs of layers (layer k Chinese vs layer k+2 English). The density
argument predicts Chinese reaches Z earlier, so the best match might
be cross-layer, not same-layer.

**If you have time left and want to go deeper:**
The mutual information test (does Z predict answers?) can be done with
a GP classifier using the same ARD kernel. The lengthscales from the
GP tell you which Z dimensions predict correctness. The intersection
of "cross-lingually shared" AND "predicts correctness" is the purest Z.

---

*Print this. Pen. Go.*
