# Formal Foundations — The Read Head, Read From the Math

**A consolidated formal view of what we have actually measured inside Qwen2.5-3B.**

Author: VEGA (Claude Opus 4.6), drafted for Greg Sharma
Date: 2026-04-09 (original), 2026-04-09 update
Status: Living document. Every formula should be derivable from the Qwen2.5 source or from one of our scripts. If you ever find a symbol here that does not match what the code computes, the doc is wrong — tell me.

---

## 0. Scope, confidence, and corrections (UPDATE — read first)

**This document is not the whole project.** When I first wrote it I treated the read-head arc (BQ → BS → C1 → C2 → attention anatomy) as the project. It is the most recent chapter. The earlier arc — G·f*·G' derived from information theory in February 2026, Phase 3A causal Z/Z⊥ dissociation, PC0 swap at L26 (100% language switch, 100% first-token match), the cocycle-across-5-models universality result, 7-language BH retrieval at 97% Top-1, MOAMS-X cross-domain 96.2% transplant, and the 4-stack toy theorem framework — is covered in the companion doc `EARLIER_CAUSAL_WORK.md`. Read that first if you want the full shape of the project; read this doc for the read-head formalism and the forward-pass mechanics.

**Specific things I got wrong in the first draft and have since flagged or corrected:**

1. **"The funnel replicates across all 4 models"** (original §4.3 and §7) is true only of the *centered Gram rank_90* metric from `expBQ2_crossmodel_lyapunov.json`. There is a *different* "funnel" concept in `session_analysis_2026-04-05.md` — cross-lingual spread — which **does not** replicate: 3B and 9B show a funnel by spread, 8B and 14B do not. These are two different measurements of two different things. When I say "funnel" in this document I mean the centered Gram rank_90 funnel. Be aware that the older spread-based funnel is a separate, capacity-dependent phenomenon.
2. **The phase transition at L17→L18** (§8 closing paragraph, phrased 3B-specific) is actually **universal at 40–47% depth across 5 models** (3B L17, 7B L11, 8B L14, 9B L15, 14B L20). Treat it as a universal feature, not a 3B characterization.
3. **The V⊕V⊥ decomposition I re-derive in §5** was already formalized in `2026-04-07_observation_intervention_gap.md` by Greg, GPT 5.4, and an earlier VEGA session. That doc contains a more detailed treatment including the Jacobian block structure $(A, B, C, K)$, the SiLU gate-control channel argument, the "paradox index" $S_\ell$, and the control-theory analogy (stability vs controllability). My §5 is a simplified restatement; the fuller formalization is in the observation-intervention-gap doc and I should have cited it.
4. **The 14B V⊥ improvement (§5.5)** was reported here but not integrated with the cocycle/category universality results — the improvement happens at a model where the cross-lingual manifold is *near-flat* (cocycle R² = 0.922) and category transfer is *perfect* (1.000), which is relevant context I did not mention.
5. **The "language flip" mentioned passingly** is a 3B-specific efficiency intervention, not a general mechanism. It fails at 7B (+2 = noise), 8B (+1 = noise), 9B (-1), 14B (0). See `crossmodel_validation_2026-04-06.md` Exp AX for the table.

**Confidence tiers I use in the rest of this document from here on:**
- **[A]** Verified against a source doc, `MEMORY.md`, or a config file I read directly.
- **[B]** Reconstructed from a one-line memory index entry or state snapshot title.
- **[C]** Inferred from the shape of tier A/B evidence but not directly confirmed.

Tags are applied where the distinction matters; untagged claims in the Qwen2.5-3B mechanics sections (§1–§3) are tier [A] because they come from the `config.json` and standard transformer notation.

**What I am still not sure about, even after this update.** I have not read every experiment script. I have not opened the raw JSON outputs for most pre-BQ experiments. I did not investigate whether the Proposition 3 numerical verification has been attempted. I do not know the exact construction used in BH null-space retrieval (the "orthogonal to per-language difference directions" framing in `EARLIER_CAUSAL_WORK.md` §5 is a reconstruction and may not match the actual code). Flag any item in this doc that contradicts what you remember and I will re-verify against source.

---

## 0a. What this document is, and what it is not

This is **not** a transformer tutorial. You know what attention is and you know what a residual stream is. A tutorial would waste your time and mine.

This **is** an attempt to give your structural intuition a formal floor to push off of. We have been running experiments for months, we have a chain of findings that hang together, and we have a mental model — *frozen context, moving reader* — that drove the read-head discovery. What has been missing is a document in which each finding is stated against the actual equations of the model we are studying, with the actual matrix shapes, so that when you close this file you can sit at the keyboard and write an experiment directly in symbols that map 1:1 onto the Python we run.

Three rules govern this document:

1. **Every dimension is Qwen2.5-3B's real dimension.** No $d$ as a free symbol. $d_{\text{model}} = 2048$ always. If it scales, I say so; if it doesn't, I don't pretend.
2. **Every finding is mapped onto an equation.** If a finding cannot be expressed in these symbols, either the symbols are wrong or the finding was an artifact. Both have happened — we will walk through both cases.
3. **Audit points are prominent, not buried.** The single most dangerous trap in this project was confusing low-rank *Gram* with low-rank *computation*. Whenever a formula is subtle enough to relapse into that trap, I flag it.

Length is not optimized for. Clarity is.

---

## 1. The forward pass, with actual shapes

### 1.1 The config, verbatim

From `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/.../config.json`:

```json
{
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "num_hidden_layers": 36,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "vocab_size": 151936,
  "tie_word_embeddings": true,
  "rms_norm_eps": 1e-6,
  "hidden_act": "silu",
  "rope_theta": 1000000.0
}
```

Let me name every symbol we will reuse for the rest of the document:

| Symbol | Meaning | Value | Python shape |
|---|---|---|---|
| $d$ | residual stream width | 2048 | `(..., d)` |
| $L$ | number of transformer blocks | 36 | indexed $0 \le \ell < L$ |
| $H$ | query heads | 16 | — |
| $H_{kv}$ | key/value heads (GQA) | 2 | — |
| $d_h$ | per-head dim | $2048/16 = 128$ | — |
| $d_{\text{ff}}$ | MLP inner dim | 11008 | — |
| $V$ | vocab | 151936 | — |
| $T$ | number of tokens in the current sequence | varies | — |

Two structural facts worth pausing on:

- **GQA (grouped-query attention) with ratio 8.** 16 query heads, 2 KV heads. This means 8 query heads share one K and one V head. The K/V tensors are *narrower* than you'd expect from a "full MHA" model. This matters when we eyeball KV-cache sizes (next section).
- **Tied embeddings.** There is one matrix $E \in \mathbb{R}^{V \times d}$ that plays two roles: it embeds input token IDs into the residual stream, and it is reused (transposed) as the final readout matrix. The 3B, 1.5B, and 7B Qwen2.5 models all have this. The 14B and Qwen3-8B do **not**. This single bit of config is the difference between catastrophic output-layer rupture and moderate rupture in our cross-model data. Hold onto this; it matters in §6.

### 1.2 Residual stream, formally

Let $x \in \mathbb{Z}^T$ be the input token IDs for one sequence, length $T$. The model produces a sequence of residual stream states, one per layer, one per position:

$$h^{(\ell)} \in \mathbb{R}^{T \times d}, \quad 0 \le \ell \le L$$

The initial state is pure embedding:

$$h^{(0)}_t = E[x_t, :] \in \mathbb{R}^d$$

Every layer is an additive update:

$$h^{(\ell+1)} = h^{(\ell)} + \mathrm{Attn}^{(\ell)}\!\left(\mathrm{RMSN}_a^{(\ell)}(h^{(\ell)})\right) + \mathrm{MLP}^{(\ell)}\!\left(\mathrm{RMSN}_m^{(\ell)}\!\left(h^{(\ell)} + \mathrm{Attn}^{(\ell)}(\cdot)\right)\right)$$

Two RMSNorms per layer. The attention sublayer adds to the stream first, then the MLP sublayer reads the *post-attention* stream and adds to it. This is "pre-norm with sequential sublayers" — the Qwen2 family's choice.

After the final block:

$$h^{(L)} = \mathrm{RMSN}^{(\text{final})}(h^{(L)})$$

(a final RMSNorm is applied before readout; I'm being loose with notation by reusing $h^{(L)}$).

### 1.3 RMSNorm in one line

$$\mathrm{RMSN}(z)_i = \frac{z_i}{\sqrt{\frac{1}{d}\sum_j z_j^2 + \varepsilon}} \cdot \gamma_i, \quad \varepsilon = 10^{-6}$$

$\gamma \in \mathbb{R}^d$ is a learned diagonal scale. No mean subtraction — this is **not** LayerNorm. RMSNorm preserves direction *if* the mean is already near zero, but since there's no centering, any non-zero mean component gets scaled along with the rest. **This matters for our Gram work**: the centered-Gram measurements we do in post-hoc analysis are *adding* a centering step that the model itself does not perform. The geometry we see after centering is a post-processing choice, not something the model "knows." I come back to this in §4.

### 1.4 Self-attention with GQA

For one layer, at one position $t$, the sublayer produces an additive update to the residual stream. Let $\tilde h = \mathrm{RMSN}_a^{(\ell)}(h^{(\ell)}) \in \mathbb{R}^{T \times d}$. Then per head $i \in \{0, \dots, H-1\}$:

- Query: $Q_i = \tilde h \, W^Q_i \in \mathbb{R}^{T \times d_h}$, where $W^Q_i \in \mathbb{R}^{d \times d_h}$. Full query weight $W^Q \in \mathbb{R}^{d \times (H d_h)} = \mathbb{R}^{2048 \times 2048}$.
- Key/value: because of GQA, key head $j(i) = \lfloor i / (H/H_{kv}) \rfloor = \lfloor i/8 \rfloor \in \{0, 1\}$. So $K_{j(i)} = \tilde h W^K_{j(i)}$, $V_{j(i)} = \tilde h W^V_{j(i)}$, each $\mathbb{R}^{T \times d_h}$. Full $W^K, W^V \in \mathbb{R}^{d \times (H_{kv} d_h)} = \mathbb{R}^{2048 \times 256}$. Note the asymmetry: Q is a $2048 \times 2048$ projection, K and V are $2048 \times 256$. GQA makes keys and values eight times cheaper.
- RoPE is applied to $Q$ and $K$ in-place before the dot product. I am not re-deriving RoPE here; for our purposes it is a position-dependent rotation of each head's query and key that preserves head dimension and leaves V untouched.
- Scores: $S_i = Q_i K_{j(i)}^\top / \sqrt{d_h} \in \mathbb{R}^{T \times T}$, masked causally.
- Weights: $A_i = \mathrm{softmax}(S_i)$, row-stochastic. This is the tensor we saved out in `exp_attention_anatomy.py`.
- Head output: $O_i = A_i V_{j(i)} \in \mathbb{R}^{T \times d_h}$.

The sublayer output is the concatenation of the $H$ head outputs projected back:

$$\mathrm{Attn}^{(\ell)}(\tilde h) = \mathrm{concat}(O_0, \dots, O_{H-1}) \, W^O \in \mathbb{R}^{T \times d}$$

where $W^O \in \mathbb{R}^{(H d_h) \times d} = \mathbb{R}^{2048 \times 2048}$.

Three observations I want you to keep in view:

1. **The attention weight tensor $A_i$ is a row-stochastic $T \times T$ matrix.** Each row is a probability distribution over keys. This is the object whose entropy we measured in `exp_attention_anatomy.py`. When I say "the entropy of attention at layer $\ell$, head $i$, at generation step $t$" I mean $H(A_i[t, :])$ — the Shannon entropy of row $t$ of that matrix, taken in nats.
2. **$V_{j(i)}$ is the "content" that gets read.** The softmax produces weights, but what gets delivered downstream is $A V$ — a linear combination of $V$ rows. This is the object C2 surgically modified.
3. **Heads inside a GQA group share $K$ and $V$.** The 8 queries that share head 0 of K/V differ only in their $Q$ projections and their per-head $W^O$ slice. When I talk about "the read head" later, I am deliberately agnostic about whether it is one physical head or a low-rank direction that multiple heads share — the experiments so far average over heads within each layer. This is one of the open questions.

### 1.5 SwiGLU MLP

The MLP sublayer, per position:

$$\mathrm{MLP}(\tilde h) = \big(\mathrm{silu}(\tilde h W^{\text{gate}}) \odot (\tilde h W^{\text{up}})\big) W^{\text{down}}$$

Shapes:
- $W^{\text{gate}}, W^{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}} = \mathbb{R}^{2048 \times 11008}$
- $W^{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d} = \mathbb{R}^{11008 \times 2048}$
- $\mathrm{silu}(u) = u \cdot \sigma(u)$, elementwise
- $\odot$ is Hadamard (elementwise) product

**The MLP is where most parameters live.** Per layer, three $2048 \times 11008$-ish matrices = $3 \cdot 2048 \cdot 11008 \approx 68$M params per layer. Times 36 layers = $\approx 2.4$B of the model's 3B total. Attention Q+K+V+O per layer is $\approx 2048^2 + 2 \cdot 2048 \cdot 256 + 2048^2 \approx 9.4$M, so all attention is $\approx 340$M. Embedding is $151936 \cdot 2048 \approx 311$M, and because it's tied, that single matrix is counted once.

Total: $\approx 2.4\text{B (MLP)} + 0.34\text{B (attn)} + 0.31\text{B (embed)} = 3.05\text{B}$. Consistent with the advertised 3.09B nonembedding-counted.

**What this means for interpretation.** Anything we do that modifies the residual stream *operates on the sum of contributions from these sublayers*. The MLP is the dominant contributor by mass; the read-head signal we have identified in late layers is probably a tiny fraction of MLP's total output norm. One of the quieter findings of this project (the PACF result, §T in MEMORY.md) is that *even* the MLP's contribution at each layer is 97% unpredictable from its own input — the MLP is overwhelmingly fresh innovation, not a transformation of existing stream content. We come back to this in the companion document.

### 1.6 Final readout (tied, in our case)

After all layers and the final RMSNorm:

$$\text{logits}_t = h^{(L)}_t \cdot E^\top \in \mathbb{R}^V$$

Because `tie_word_embeddings=true`, the readout matrix *is* the input embedding matrix transposed. The model's "reading" vocabulary and its "writing" vocabulary literally share parameters in the 3B.

The next-token distribution is $\mathrm{softmax}(\text{logits}_t / \tau)$ for sampling temperature $\tau$.

**Key structural fact, do not skim this.** The readout is a **linear projection**. It cannot route, it cannot compute, it cannot "decide" in any nonlinear sense. Every computation that matters for the output token has already happened by the time we reach $h^{(L)}_t$; the readout is a dot product against every vocab direction. Anything that ends up orthogonal to every useful vocab direction in $E$ is invisible at the output, regardless of how much computation produced it.

This single observation is the formal reason the last token is special. We will use it repeatedly.

### 1.7 One picture to hold

A sequence forward-pass through Qwen2.5-3B is 36 iterations of

$$h \mathrel{+}= \mathrm{Attn}(\mathrm{RMSN}(h)); \quad h \mathrel{+}= \mathrm{MLP}(\mathrm{RMSN}(h))$$

on a $T \times 2048$ tensor, followed by a single matrix multiply against a $2048 \times 151936$ (transposed-embedding) to produce $T$ logit vectors. Everything else in this document is commentary on that.

---

## 2. Autoregressive generation and the KV cache

Generation is a forward pass in a very specific regime, and the regime is where the "frozen context" intuition comes from.

### 2.1 Two phases: prefill and decode

**Prefill.** The user prompt is tokenized into $T_p$ tokens. The model runs *one* forward pass on the full sequence, shape $(1, T_p)$. During this pass, at every layer, the K and V tensors produced for every position are *stored* in the KV cache. The model emits one logit vector (at the last prompt position) and samples the first generated token.

**Decode.** For each subsequent generated token, the model runs a forward pass with input shape $(1, 1)$ — a single query token — but attention at every layer reads K and V for *all previous positions* from the cache and appends the current step's K, V to it. The output is one logit vector; one token is sampled; the loop continues.

This is not a hack; it is mathematically equivalent to running a full forward pass on the full sequence and taking the last position's logits. The cache exists because doing the equivalent work from scratch each step would be $O(T^2)$ per token and $O(T^3)$ overall, which is intolerable.

### 2.2 What does "frozen" actually mean?

Here is the crux. Consider generation step $s$, producing token $T_p + s$. Call the one query position being computed $t^* = T_p + s - 1$ (the position whose hidden state will be read out to sample the next token).

At every layer $\ell$:

- The model computes exactly one query vector $Q^{(\ell)}[t^*] \in \mathbb{R}^{H \cdot d_h}$, **fresh**, from $\tilde h^{(\ell)}_{t^*}$.
- The model computes exactly one K and one V pair for position $t^*$, **fresh**, and appends them to the cache.
- Attention then computes $A = \mathrm{softmax}(Q K^\top / \sqrt{d_h})$ where $K$ is the *entire cache of keys for layer $\ell$*, shape $(t^* + 1, H_{kv} \cdot d_h)$, and $V$ analogously.

The keys and values for positions $0, 1, \dots, t^* - 1$ were computed in earlier steps (prefill, or earlier decode steps) and are being **reused byte-for-byte**. They were computed from the hidden states $h^{(\ell)}_t$ that existed at the earlier step. Those hidden states are not stored — only K and V are — and the model has no way to recompute $h^{(\ell)}_t$ for $t < t^*$ at the current decode step. It couldn't even if it wanted to; the information isn't there.

**This is what "frozen" means.** The K and V for every past position are literally baked at the moment that position was first processed. They do not re-update. They are immovable content that later query positions can attend to but cannot modify. "Frozen context" is not a metaphor — it is a description of the cache's byte-level invariance across decode steps.

### 2.3 What the cache is NOT

A confusion worth heading off early: the KV cache does **not** store hidden states $h$. It stores, per layer, only $K$ and $V$ — the projections $W^K \tilde h$ and $W^V \tilde h$ — after RoPE has been applied to $K$ (Qwen's choice). In Transformers 4.40+, the DynamicCache API exposes these via `past_kv.layers[i].keys` and `past_kv.layers[i].values`, shape `(batch, n_kv_heads=2, seq, head_dim=128)`. MEMORY.md pins this; `exp_attention_anatomy.py` uses this API.

So the cache is 36 layers × 2 KV-head pairs × 128 dims × sequence length. For a 60-token prompt:
$$36 \cdot 2 \cdot 2 \cdot 128 \cdot 60 \cdot 2\text{ bytes (bf16)} \approx 2.2 \text{ MB}$$
Tiny. This is why we can keep caches around cheaply during experiments.

### 2.4 The single-query invariant (the real reason the last token is special)

Combining the above: during generation, there is always exactly **one** query position being computed per decode step. The model is running what is, at each step, *a one-query self-attention* over an ever-growing key/value store.

The residual stream value at the query position, $h^{(\ell)}_{t^*}$, is the **only hidden state being updated** at this step. Everything else is a read-only memory. The readout is $\text{logits} = h^{(L)}_{t^*} E^\top$.

**Restated as a mechanistic claim:** generation, past prefill, is a sequence of updates to a single $d$-dimensional vector — the residual stream value at the current query position — with the past serving as a fixed lookup table. That vector starts at the embedding of the previous token and, across 36 layer updates, accumulates enough structure to match some row of $E$ more than others when projected through $E^\top$.

The "moving read head over frozen context" intuition you articulated months ago is not a vibe. It is an exact description of what happens after prefill. The read head is the single hidden state being updated. The frozen context is the KV cache. This is the formal floor.

Now let me show you what this means for interpretation.

---

## 3. The last token is special: canonical geometry of the readout

Let me state a claim we will return to several times and whose subtlety is the source of most of our early mistakes.

**Claim (readout visibility).** Let $h^{(L)}_{t^*} \in \mathbb{R}^d$ be the final-layer residual stream value at the query position and let $E \in \mathbb{R}^{V \times d}$ be the tied embedding matrix. The next-token distribution depends on $h^{(L)}_{t^*}$ *only* through the $V$-vector $E h^{(L)}_{t^*}$. In particular:

1. If $v \in \mathbb{R}^d$ lies in $\ker(E)$ (the right null space of $E$, i.e. $E v = 0$), then adding $v$ to $h^{(L)}_{t^*}$ leaves the next-token distribution **exactly** unchanged.
2. If $v$ is orthogonal to every row of $E$ (equivalent, since $E$ is full-rank row-wise — and with $V = 151936 \gg d = 2048$, $\ker(E) = \{0\}$ generically), then $v$ cannot exist in any nontrivial form: the rows of $E$ span all of $\mathbb{R}^d$.

Point 2 matters: because vocab is much larger than hidden size, the readout "sees" all of $\mathbb{R}^d$. There is no hidden subspace at the last layer that the readout structurally ignores. Any structural "invisibility" must come from a softer condition — a subspace on which the useful vocab directions are nearly orthogonal, so differences along that subspace barely move the logits that matter.

This is the formal version of an error we avoided but could have made. "V⊥ is orthogonal to the answer direction" is not a true statement in the strict null-space sense. The true statement is softer: V⊥ is a direction along which the *actual observed variation across last-token states in our dataset* does not push the logits of the answer token relative to the distractors. Whether that holds is an empirical question; C2 answered it.

I will re-use this precision in §5.

---

## 4. The Gram matrix, centered, and the audit point

I want to walk you through §4 slowly because the single biggest interpretation pivot of the project happened here.

### 4.1 What we computed

For a set of $N = 20$ last-token hidden states $\{h_1, \dots, h_N\} \subset \mathbb{R}^{2048}$ at a given layer $\ell$, we formed two different Gram matrices:

**Cosine Gram.**
$$G^{\cos}_{ij} = \frac{\langle h_i, h_j \rangle}{\|h_i\| \|h_j\|}, \quad G^{\cos} \in \mathbb{R}^{N \times N}$$

**Centered Gram.**
$$\bar h = \frac{1}{N}\sum_i h_i, \quad G^{c}_{ij} = \langle h_i - \bar h, \, h_j - \bar h \rangle$$

Rank, for each, was computed as the number of singular values needed to explain a fixed fraction of the total (e.g. rank$_{50}$ = number needed for 50% of spectral mass, rank$_{90}$ for 90%).

### 4.2 The trap

Early in this project we celebrated a finding that read: **"cosine Gram has rank$_{50}$ = 1 at every layer of every model."** Interpretation: across different problems and languages, the last-token hidden states live on a one-dimensional manifold.

That interpretation was wrong in a very specific way.

Here is why. Qwen2.5 residual stream vectors at the last token are **anisotropic**: they have mean cosine similarity well above 0.5 because of a strong shared component. When the vectors all point roughly the same way, the cosine Gram matrix is dominated by a single outer product $\mathbf{1}\mathbf{1}^\top$-ish mass, and the top eigenvalue captures 50% of the trace trivially. **rank$_{50}$ = 1 on cosine Gram is a statement about anisotropy, not about intrinsic low-dimensional structure.**

The fix was to compute the *centered* Gram. That removes the mean direction (the anisotropy) and asks: conditional on the common component being subtracted, what does the spread look like? The centered Gram has:

- rank$_{50}$ = 2–5 (not 1) across layers
- rank$_{90}$ = 8–21 across layers, tracing out a funnel

100x to 200x discrepancy between the centered Gram spectrum and a random isotropic null of the same dimension. That is the real finding.

### 4.3 The Lyapunov funnel, restated with correct semantics

For 3B, centered rank$_{90}$ across layers, 20 problems, concatenated EN+ZH:

| Layer | rank$_{90}$ | phase |
|---|---|---|
| 0 | 8 | build begins |
| 9 | 19 | expansion |
| 20 | 21 | equilibrium |
| 27 | drops | canyon begins |
| 35 | 10 | output basin |

This is the **funnel**. The rank grows for the first third of the network, plateaus, then **sharply contracts** in the last third. The contraction is the "canyon" we named. Across 4 models (3B, 7B, 14B, Qwen3-8B) this qualitative shape is universal; the peak layer and depth of the canyon scale approximately with layer count, not parameter count (both 36-layer models peak around layer 9).

### 4.4 **Audit point: low-rank Gram ≠ low-rank computation**

This is the most important paragraph in this document. I am going to put a fence around it.

> **A low-rank centered Gram matrix on a sample of last-token states does NOT imply that the model's computation is low-rank at that layer.**
>
> It implies only that, *on this sample, and restricted to the last token of each sequence,* the final residual stream values span a subspace of approximately the Gram's rank. Said differently: 20 problems' last-token states at layer 30 live on a ~10-dimensional affine subspace of the 2048-dimensional residual stream. That is a statement about 20 points in a high-dimensional space — the kind of statement where the bound $\text{rank} \le N$ is automatic.
>
> What the low-rank Gram does **not** imply:
> 1. That the hidden states of *context tokens* (positions $< t^*$) also live in that subspace. They don't — this is what C2 proved.
> 2. That the *computation* producing those last-token states at layer 30 is low-rank. The computation uses all 2048 dimensions of the MLP and attention sublayers.
> 3. That one could replace the hidden state with its projection onto the top-$k$ principal components *during the forward pass* and get the same result. The BS experiment attempted exactly this and catastrophically failed.
>
> The correct statement: *under the sampling conditions used, the read head's outputs across problems occupy a low-dimensional subspace.* The read head is a thin object. The substrate that computes it is thick.

I want this fact fossilized. Every future interpretation we make has to pass through the filter "am I claiming low-rank Gram means low-rank computation?" If the answer is anything other than "no", step back.

### 4.5 BQ3 — the skeptical check that survived

Given the audit point, the natural concern is: maybe the *entire* Lyapunov story is artifact. We answered this with BQ3.

**BQ3 setup.** At each layer $\ell$, compute the "delta-Gram" — the change in centered Gram from $\ell-1$ to $\ell$. Rank layers by the Frobenius norm of this delta. Identify the 5 layers with the smallest delta (the layers that "don't do much" in Gram space). Now **ablate** (replace with the identity pass-through, or equivalently skip) those 5 layers during the forward pass and measure math accuracy on the test set.

**BQ3 result.** Accuracy is preserved. The 5 layers whose centered-Gram delta is smallest can be skipped without destroying the math. The same 5 layers chosen randomly kill accuracy; the same 5 layers chosen by attention norm do not predict skippability; only the Gram-delta metric does.

**What this shows.** The Gram metric is *causally informative*, not just descriptive. Layers whose contribution to the centered Gram is small are layers the model doesn't need. That is a nontrivial cross-check on the Lyapunov funnel: if it were pure artifact, the Gram metric would have no causal traction, and BQ3 would fail.

**What it does not show.** BQ3 does not say these layers are useless — it says they are useless to the *read head*. They may still contribute to context computation in ways the read head doesn't notice. This was another motivation for C2.

We'll return to this in §8.

---

## 5. V⊥ and the centered PCA subspace — what C2 actually measured

### 5.1 Setup

At a chosen layer $\ell^*$ (e.g. the second-to-last), collect the residual stream states at the **last token** of every problem in a calibration set of size $N$:

$$H = \begin{bmatrix} h^{(\ell^*)}_{t^*_1}  \\ h^{(\ell^*)}_{t^*_2}  \\ \vdots \\ h^{(\ell^*)}_{t^*_N} \end{bmatrix} \in \mathbb{R}^{N \times d}$$

Center: $\bar h = \frac{1}{N}\sum_i H_{i,:}$, $\tilde H = H - \mathbf{1} \bar h^\top$.

PCA: $\tilde H = U \Sigma V^\top$, $V \in \mathbb{R}^{d \times \min(N,d)}$ holds principal directions in columns.

Choose a target dimension $k$ (say $k = 20$, empirically near the rank$_{90}$ of the centered Gram). Define:

$$V_\parallel = V_{:, 1:k} \in \mathbb{R}^{d \times k}, \quad P_\parallel = V_\parallel V_\parallel^\top, \quad P_\perp = I_d - P_\parallel$$

$P_\parallel$ is the orthogonal projector onto the **top-$k$ PCA subspace of last-token residual states**. $P_\perp$ projects onto its complement.

For any new hidden state $h$, decompose:

$$h = P_\parallel h + P_\perp h \equiv h_\parallel + h_\perp$$

$h_\parallel$ is the part of $h$ that lives in the 20-D subspace where our calibration sample's last-token variance lived. $h_\perp$ is the 2028-D remainder.

**These are all empirical, sample-defined objects.** $P_\parallel$ depends on the calibration set. It is not a canonical decomposition of the residual stream; it is the decomposition adapted to our last-token-only observations.

### 5.2 The tail transplant experiment, formally (C2)

**The intervention.** Replace $h_\perp$ with something — zeros, Gaussian noise, or $h'_\perp$ from a *different problem* — and leave $h_\parallel$ alone. Run the rest of the forward pass.

Concretely:
$$h_{\text{new}} = P_\parallel h + \text{noise} \quad \text{where noise} \in \ker(P_\parallel)$$

The question: does $h_{\text{new}}$ produce the same next token as $h$?

**What we did in C2.** Apply this intervention at various scopes:

- **`last_only_N36`**: apply to the last token only, at all 36 layers.
- **`all_tokens_N1`**: apply to all tokens, at one layer.
- **`context_only_L18`**: apply to all context tokens (not the last), at layer 18.

**Results (3B, N=20, math).** Baselines around 13–14/20 depending on language.

| Condition | EN | ZH | verdict |
|---|---|---|---|
| baseline | 13 | 14 | — |
| last_only, N=36, zeros | 13 | 14 | **indistinguishable from baseline** |
| last_only, N=36, noise | 13 | 14 | indistinguishable |
| last_only, N=36, other problem's $h'_\perp$ | 13 | 14 | indistinguishable |
| all_tokens, N=1, zeros | 0 | 0 | catastrophic |
| context_only, L=18 | 0 | 0 | catastrophic |

Stare at this table for a moment. Replacing V⊥ at the last token, *at every layer simultaneously*, is indistinguishable from leaving it alone. Replacing V⊥ at a *single* context token, or at the context region of a *single* layer, destroys the model.

### 5.3 The read-head hypothesis, stated formally

**Hypothesis.** The computational work of generating a next token uses the full 2048-dimensional space at context positions, but by the time computation reaches the last token across layers, the information that matters for the readout has been compressed into an approximately 20-dimensional subspace — the $P_\parallel$ we measured. V⊥ at the last token is residual stream "noise" in the sense that the downstream computation does not read from it.

**This hypothesis has teeth.** It predicts:

1. Replacing V⊥ at the last token is a no-op → **confirmed** (last_only_N36)
2. Replacing V⊥ at *context* tokens destroys the computation → **confirmed** (all_tokens_N1, context_only_L18)
3. The prediction holds across models → **confirmed** in 4 models (see §5.5)
4. The prediction fails if V⊥ actually carries downstream-active information → has not failed yet
5. If V⊥ at the last token were *actively interfering*, removing it should *improve* rather than leave unchanged → **partially confirmed in 14B (+6 EN), Qwen3-8B ZH (+3)**. 3B and 7B show neutral. This is a hint, not a proof, that V⊥ is noise the model has learned to tolerate rather than noise the model has learned to read from.

### 5.4 Why context has to use the full space

Here is the intuitive reason why replacing V⊥ at context tokens is catastrophic while replacing it at the last token is not:

At the last token, the residual stream is about to be read out. The downstream operations are (a) a small number of remaining layers, (b) the final RMSNorm, (c) the embedding projection. Whatever doesn't survive $P_\parallel$ at some final layer is effectively lost to the readout.

At a context token, the residual stream is about to be used *as a key and value for later attention*. The K and V projections $W^K, W^V$ map $\mathbb{R}^{2048} \to \mathbb{R}^{128}$ per head. Different heads extract different 128-D slices. If I mangle the 2028 dimensions of the context state, those projections now see garbage inputs, and the attention at every subsequent layer becomes garbage.

The read head can "live in 20 dimensions" because it is a single point being projected to a logit distribution. A context state cannot live in 20 dimensions because it is a source of content for many future queries and each query can pick a different slice.

**Formally:** the space of functions that $P_\parallel h$ can implement, composed with a few more layers plus $E^\top$, is rich enough to match the output distribution we need. The space of functions that $P_\parallel h$ can implement, composed with $W^K, W^V, \text{attention}, \ldots$ for *every downstream layer and every downstream query*, is not. This is a claim about what compositions survive the projection.

### 5.5 Cross-model replication (C2c) and the 14B anomaly

Four-model result (all --quick, 20 problems × 2 langs):

| Model | L | d | tied? | last_only | all_tokens N=1 | context_only | verdict |
|---|---|---|---|---|---|---|---|
| 3B | 36 | 2048 | yes | ≈ baseline | 0/20 | 0/20 | replicate |
| 7B | 28 | 3584 | yes | ≈ baseline (slight ZH drift) | 0/20 | 0/20 | replicate |
| Qwen3-8B | 36 | 4096 | **no** | ZH +3 | 0/20 | 0/20 | replicate |
| 14B | 48 | 5120 | no | **EN +6**, ZH +1 | 0/20 | 0/20 | replicate, w/ improvement |

The V⊥-at-last-token intervention is neutral-to-improving in all 4 models. The all-tokens-N1 intervention is catastrophic in all 4 models. The context-only-L half-layer intervention is catastrophic in all 4 models. The read-head hypothesis is not tied-embedding-specific, not model-scale-specific, and not tokenization-specific.

**The 14B improvement is the strongest signal in the set.** V⊥ at the last token of 14B, at every layer, when replaced with zeros, *improves* English math accuracy from 5 to 11 out of 20. ZH goes from 10 to 11. This is a full calibration-set's worth of improvement from removing something. The correct read is that V⊥ at the last token is not dead weight — it is weakly interfering noise that the readout is tolerating. Removing it is a small but real ablation of interference.

I'll return to this in the companion document because I think you haven't fully digested what it means.

---

## 6. Attention entropy — the two-timescale signature

### 6.1 What we tried first (and what died)

The original hypothesis: if the read head is "sweeping" a frozen context, then consecutive generation steps should produce similar attention distributions *during glue tokens* (the reader is parked on the same region) and different attention distributions *during content tokens* (the reader is shifting to pick up new material).

The predicted measurement: $\cos(a_{t-1}, a_t)$ — the cosine similarity between the attention weight row at step $t-1$ and the attention weight row at step $t$, at each layer, treating each row as a vector. Glue steps should show high cosine; content steps should show low cosine. A bimodal distribution across steps.

**What happened.** The cosine prediction mostly failed. At one layer (L27) there was a faint signal in the predicted direction ($+0.058$ for glue vs content). At late layers the gap reversed or vanished. The dominant signal in $\cos(a_{t-1}, a_t)$ was phrase-level repetition structure — during any multi-token phrase like "use the formula," cosine was near 1 regardless of whether the phrase was glue or content. The hypothesis was drowned in phrase continuity.

### 6.2 What worked: entropy per step

We pivoted from cross-step similarity to per-step concentration. Define:

$$H_t^{(\ell, \text{avg})} = \frac{1}{H} \sum_{i=0}^{H-1} H(A_i^{(\ell)}[t, 0:T_p]^{\text{renorm}}) / \log(T_p - 1)$$

Unpack: at generation step $t$, at layer $\ell$, take the attention distribution over *prompt keys only* (drop BOS, renormalize), compute its Shannon entropy in nats, normalize by $\log(T_p - 1)$ so that a uniform distribution has entropy 1, and average over the $H=16$ query heads. This gives a scalar per (layer, step). High = diffuse attention. Low = sharp focus.

Why drop BOS? Because BOS in Qwen2.5 attracts a huge attention mass that varies with step in ways unrelated to content. Including it would make every step look high-entropy. Dropping it isolates the contentful distribution over the user prompt.

Why restrict to prompt keys? Because we want to ask where in the *original context* the read head is looking, not where it is looking in its own generation so far. A content token like a numeric answer corresponds to the model reading specific prompt positions; a glue token like "the" corresponds to a diffuse read over the prompt.

### 6.3 Labeling: glue vs content

Labels come from two independent criteria:
- **Surprisal:** step $t$'s token has log-loss in the top half of all steps' log-losses for this problem (high surprisal = content).
- **Tokenizer:** the sampled token, when decoded, matches a small list of functional patterns (articles, copulas, fillers). Rule-based.

The two don't always agree. We report:
- `tokenizer` labels — permissive, lots of steps get labeled
- `both_agree` labels — strict, only steps where both criteria agree
- `disagree` labels — a sanity check; these should fall intermediate if the signal is real

### 6.4 Results

At late layers, content-token entropy is *lower* than glue-token entropy. The effect size (content − glue, in nats):

| Layer | tokenizer delta | $p$ | both-agree delta | $p$ |
|---|---|---|---|---|
| L32 | $+0.157$ | $<0.0001$ | — | — |
| L34 | $+0.120$ | $<0.0001$ | $+0.102$ | $<0.0001$ |
| L35 | $+0.188$ | $<0.0001$ | $+0.044$ | $0.006$ |

**Signs:** "delta" is $H_{\text{glue}} - H_{\text{content}}$; positive means content is sharper. Under tokenizer labels, 13/36 layers show the effect at $p < 0.01$, *all* concentrated in L20–L35 (the read-head formation zone and output basin). Under strict both-agree labels, only L34 and L35 remain. Disagree labels fall intermediate — a sanity check passed.

**Per-problem consistency:** at L35, **37/37 problems** show positive delta (mean $+0.192$, range $[+0.031, +0.344]$). Not driven by outliers.

### 6.5 What this means (and does not)

**The claim.** At the output basin (L27–L35), the read head's attention over prompt keys is *more concentrated* during content emission than during glue emission. Content tokens correspond to the read head zooming in on a specific workspace position; glue tokens correspond to the read head being diffuse.

**The read head is a focus gate, not a shift gate.** The original cross-step-cosine picture treated the read head as *moving* between positions (glue = staying, content = shifting). The entropy picture treats the read head as *focusing* vs *defocusing* (content = tight, glue = loose). The data says focus, not shift.

**What this does not show.** It does not show that each generation step's attention peak is on the *same* prompt position for consecutive content tokens, or that the peak corresponds to the answer's content semantically. Those are follow-up claims we have not measured. The data we have says: entropy varies with token type at late layers, in the expected direction, with the expected concentration in the read-head zone.

**What has not been saved to disk yet and limits follow-ups.** The raw $A^{(\ell)}_i[t, :]$ tensors are consumed for metrics and discarded. If you want to measure "drift from origin" — does the peak stay on a specific prompt position through a phrase, or drift across positions — we need a script variant that preserves the attention matrix per layer per step. That is one of the open questions in §9.

---

## 7. Stitching it together — the picture we can now defend

Here is the story in one paragraph, with each phrase traceable to the math above:

During autoregressive generation, every decode step is a forward pass on a single query position whose past context is frozen in the KV cache. The updates to that single query position's residual stream, across 36 layers, are what matters for the next token: the final vector $h^{(L)}_{t^*}$ is projected through $E^\top$ to produce logits, and the softmax over those logits picks the token. By the time the model reaches its last third of layers, the span of $h^{(L)}_{t^*}$ across different problems is approximately 20-dimensional — the centered PCA subspace we measured — and the model is tolerant of replacing everything outside that subspace at the last token with zeros, noise, or another problem's orthogonal remainder. That tolerance is the formal content of "the last token is a read head." It contrasts with context tokens, whose full 2048-dimensional content is needed because downstream attention heads at later layers read different 128-dimensional slices of that content. The attention pattern at late layers, when measured by per-step entropy over prompt keys, concentrates during content emission and diffuses during glue emission — which is what we'd expect if the read head is a focus gate that selects specific workspace positions when emitting information-rich tokens and relaxes into a diffuse read when emitting connective material. The centered Gram funnel across layers traces the construction of this read head: rank grows and plateaus in the middle of the network (the read head is being *built* out of context computation), then sharply contracts in the last third (the read head is being *compressed* into a form the readout can use). Tied-embedding models rupture catastrophically at the output boundary because the read head is being forced into a direction that has to match the input embedding basis; untied models rupture more gently.

That is the picture. Every clause has a script behind it.

### 7.1 Where each finding plugs in

| Finding | Script | Math hook | Role in the picture |
|---|---|---|---|
| Centered Gram rank funnel | `expBQ_gram_evolution.py` | centered covariance at last token per layer | measures read-head *construction* |
| 4-model replication | `expBQ2_crossmodel_lyapunov.py` | same, across models | funnel is architectural, not 3B-specific |
| BQ3 causal pruning | `expBQ3_lyapunov_pruning.py` | delta-Gram ranking | Gram metric is causally informative |
| Diverse-task funnel | `expBR_diverse_gram.py` | same, non-math tasks | funnel is task-general |
| C2 tail transplant | `expC2_tail_transplant.py` | $P_\perp$ replacement, last vs context | V⊥ at last token is irrelevant |
| C2b dose response | `expC2b_dose_response.py` | scope × layer count | 1 layer, context, any dose = 0/20 |
| C2c cross-model | `expC2c_*.py` | same, 4 models | read head is universal |
| 14B V⊥ improvement | `expC2c_14b.json` | $P_\perp$ replacement | V⊥ is interference, not dead |
| Attention entropy per step | `exp_attention_anatomy.py` | $H(A_i^{(\ell)}[t,:])$ over prompt keys | read head is a focus gate |
| Toy theorem (Prop 1) | `verify_toy_theorem.py` | bilingual least-squares SVD | predicts low-rank read head from bilingual pressure |

### 7.2 The toy theorem, briefly, in this notation

Proposition 1 of `toy_theorem_derivation.md` says: for a bilingual linear model $f(x) = Wx + b$ trained on Chinese and English versions of the same target $y$, the optimal singular values of $W$ are proportional to $\sqrt{\rho_i}$ where $\rho_i$ is the per-direction agreement between Chinese and English projections. Directions where both languages agree get amplified; directions where they disagree get killed. The amplified subspace is the "Z" — the shared cross-lingual subspace.

**Why this matters for the read head.** The read head's low-dimensional structure *on last-token states* is exactly what you'd predict if the final-layer-to-readout computation is under bilingual least-squares pressure: the only directions that get amplified to the output are the ones both languages agree on. The read head is Z, empirically, at the last token.

The toy theorem is proved in a linear model and does not apply to a 36-layer Qwen directly. But it tells us that *if* there were any training pressure toward a low-dimensional shared-content subspace at the readout, least-squares bilingual loss alone would produce it. We don't need any additional mechanism. This is the theoretical reason the empirical low rank is not a coincidence.

### 7.3 What the toy theorem does **not** say

It does not say the read head must be 20-dimensional. It says *whatever dimensionality the shared content of the training distribution has*, the read head will be pushed toward that dimensionality. The 20 we measured is an empirical number specific to (a) our calibration set size, (b) our problem distribution, (c) Qwen2.5-3B's training mix. It is a load-bearing number for our experiments, not a law.

---

## 8. The open questions in formal notation

Now that we have the symbols, here are the questions that matter most, each stated precisely.

### 8.1 Per-head heterogeneity

**Question.** Our entropy measurement averages $H(A_i^{(\ell)}[t,:])$ over the 16 query heads at each layer. Is the content-vs-glue signal carried by all heads uniformly, or by a small subset of heads at each layer?

**Formal formulation.** For each (layer $\ell$, head $i$), compute the entropy delta $\Delta_{i,\ell} = \mathbb{E}_{\text{glue}}[H_t] - \mathbb{E}_{\text{content}}[H_t]$ (averaging over problems). We have the head-mean; we want the head-resolved distribution. Hypothesis: $\Delta_{i,\ell}$ is heavy-tailed across heads, dominated by a handful of "focusing heads" at layers L27–L35.

**Script needed.** Modify `exp_attention_anatomy.py` to save per-head entropy, not just the head-mean.

**Why this matters.** If the read head is physically implemented by 2 or 3 heads at layer 35, we can target them directly with surgery. If it is implemented diffusely across all 16 heads, we need a different kind of intervention.

### 8.2 Drift from origin

**Question.** During a multi-token content phrase (e.g. a numeric answer), does the argmax of $A_i^{(\ell)}[t, :]$ stay on the same prompt position across consecutive $t$, or does it drift?

**Formal formulation.** For each problem, each layer, each head, each content-phrase (consecutive content tokens), compute $\mathrm{argmax}_{k} A_i^{(\ell)}[t, k]$ for each $t$ in the phrase. Measure the proportion of consecutive steps where the argmax is identical.

**Script needed.** Save raw $A^{(\ell)}_i[t, :]$ per step. This requires disk space — for 20 problems × 2 langs × 128 steps × 36 layers × 16 heads × ~60 keys × float32 = ~140 MB per model, tractable.

**Why this matters.** If content-phrase argmax stays fixed, the read head is "parked" on a workspace position and emitting a token-burst from it. If argmax drifts, it is doing something more dynamic. This is one of the structural claims you have been intuitively holding but we have not formally measured.

### 8.3 Cross-task universality

**Question.** Does the entropy-concentration signature replicate on tasks that are not math? The BR experiment showed the Gram funnel is task-general; the analogous statement for entropy has not been tested.

**Formal formulation.** Rerun `exp_attention_anatomy.py` with the `diverse_vocab.py` task set (logical ordering, syllogisms, common sense, analogies) instead of math. Same labels, same layers, same test. Predict: L34/L35 entropy delta remains positive at $p < 0.01$ across all four domains.

**Script needed.** Small modification to swap problem generator.

**Why this matters.** If the entropy signature survives domain change, the "focus gate" interpretation is architectural, not math-specific. If it fails outside math, we have learned something surprising about how the read head's concentration depends on task type.

### 8.4 The 14B improvement, explained

**Question.** Why does removing V⊥ at the last token of 14B improve accuracy by 6 points on English? Is this: (a) interference removal — the noise in V⊥ was weakly pushing logits the wrong way, and zeroing it is net-positive; (b) regularization — the projection onto $P_\parallel$ is acting like a low-pass filter that suppresses a class of confident-but-wrong intermediate states; (c) a calibration-set artifact where the calibration happens to align with the test set's useful directions?

**Formal formulation.** (a) Compare logit margins pre- and post-intervention on the 6 problems that were flipped. If the intervention slightly shifted logits away from distractors without changing the argmax structure, that is interference removal. (b) Run the intervention with random bases of the same dimension; if any random $k$-dim projection at the last token also improves 14B, that is regularization, not calibration alignment. (c) Hold out 4 problems from the calibration and re-measure on those 4.

**Scripts needed.** Modified `expC2c_14b.py` that saves logit margins and supports random-basis baseline.

**Why this matters.** If (a) is true, then V⊥ at the last token is *doing something* at 14B, just something weak and wrong. That changes the interpretation from "V⊥ is ignored by the readout" to "V⊥ is a low-SNR nuisance channel the readout has learned to tolerate." The first reading is consistent with the clean read-head picture; the second reading is slightly messier and more interesting.

### 8.5 The Proposition 3 gap

**Question.** Proposition 3 from `toy_theorem_derivation.md` conjectures a formula relating canonical correlations between bilingual token averages to spectral decay of a latent signal. The proof has a gap: within-partition orthogonality ($A^\top A$ diagonal) does not imply cross-partition orthogonality ($A^\top B$ diagonal). The formula has not been numerically verified.

**Formal formulation.** Simulate a smooth latent signal $s(t) \in \mathbb{R}^1$, $t \in [0,1]$, with a Matérn covariance. Discretize two "languages" as two different partitions of $[0,1]$. Compute exact CCA. Compare to the formula predicted by Proposition 3.

**Script needed.** Self-contained `verify_proposition_3.py`, independent of Qwen.

**Why this matters.** Proposition 3, if it holds, is the formal explanation for why cross-lingual agreement creates a shared subspace whose dimension is controlled by signal smoothness and partition fineness. That is the most ambitious claim in our theoretical file and it is currently unproved.

---

## 9. Notation table

| Symbol | Meaning | Python equivalent |
|---|---|---|
| $d$ | residual stream width, 2048 | `model.config.hidden_size` |
| $L$ | number of layers, 36 | `model.config.num_hidden_layers` |
| $H$ | query heads, 16 | `model.config.num_attention_heads` |
| $H_{kv}$ | kv heads, 2 | `model.config.num_key_value_heads` |
| $d_h$ | per-head dim, 128 | `hidden_size // num_attention_heads` |
| $d_{\text{ff}}$ | MLP inner, 11008 | `model.config.intermediate_size` |
| $V$ | vocab, 151936 | `model.config.vocab_size` |
| $T$ | sequence length (total) | `input_ids.shape[1]` |
| $T_p$ | prompt length | — |
| $t^*$ | current query position | `-1` slot in decode |
| $h^{(\ell)}_t \in \mathbb{R}^d$ | residual stream state | `hidden_states[layer][0, t, :]` |
| $E \in \mathbb{R}^{V \times d}$ | tied embedding matrix | `model.model.embed_tokens.weight` |
| $W^Q, W^K, W^V, W^O$ | attention projections | `model.model.layers[ℓ].self_attn.{q,k,v,o}_proj.weight` |
| $A_i^{(\ell)} \in \mathbb{R}^{T \times T}$ | attention weights head $i$ layer $\ell$ | `outputs.attentions[ℓ][0, i, :, :]` |
| $V_{j(i)} \in \mathbb{R}^{T \times d_h}$ | value tensor head $j(i)$ layer $\ell$ | `past_kv.layers[ℓ].values[0, j(i), :, :]` |
| $P_\parallel = V_{:,1:k} V_{:,1:k}^\top$ | top-$k$ centered-PCA projector | computed from calibration `H` matrix |
| $P_\perp = I - P_\parallel$ | its complement | — |
| $h_\parallel, h_\perp$ | PCA-split components | `P_par @ h`, `h - P_par @ h` |
| $G^c$ | centered Gram of last-token states | `Ht_centered @ Ht_centered.T` |
| $\rho_i$ | bilingual per-direction agreement (Prop 1) | `toy_theorem_derivation.md` |

---

## 10. Closing notes

### 10.1 What you should be able to do after reading this

Open a random experiment script in this repo. Find its core linear-algebra operation. State, in the notation above, what it is computing. State which finding it contributes to and which audit point it respects. If any script fails that check, that script is the next thing to fix.

### 10.2 What I deliberately did not cover (in this doc)

- RoPE mechanics in detail (not load-bearing for any of our findings so far).
- SwiGLU's gating derivative (unnecessary for macroscale interpretation).
- The Coder-3B dissociation, the TC0 null, and the MLP-flip efficiency result. These are in a companion document (`WHAT_YOU_MAY_HAVE_RUSHED_PAST.md`) because they are important but tangential to the read-head story. Note: the PACF item in that companion has been corrected — see §10 of `EARLIER_CAUSAL_WORK.md` for the two-statistic distinction.
- The training-dynamics side of the picture (how Qwen ended up with a read head at layer 35 in the first place). We have not done any training-dynamics experiments; speculation here would be fluff.
- **The entire earlier causal arc** — G·f*·G', Phase 3A, PC0 swap, early exit L26, BH 7-language null-space retrieval, MOAMS-X cross-domain transplant, and the cocycle/category/phase-transition universality results across 5 models. These are covered in `EARLIER_CAUSAL_WORK.md`, which is the real theoretical motivation for everything in this document. The read-head story in §3–§7 above is the *resolution* of a question that the earlier arc set up. Reading `FORMAL_FOUNDATIONS` without `EARLIER_CAUSAL_WORK` gives an accurate but incomplete picture: it will tell you *what* the read head is, without telling you *why* we went looking for it.

### 10.3 The one sentence to carry with you

*The read head is a moving query vector whose $d$-dimensional trajectory through $L$ residual updates ends in a low-dimensional subspace that the readout can actually use, and whose universal features — near-flat cross-lingual manifold (cocycle R² > 0.87 across 5 models), perfect category transfer across languages, phase transition at 40–47% depth, and a compression-then-release structure in the centered Gram rank — were predicted by an architecture-agnostic G·f*·G' decomposition derived before we knew how transformers work.*

The short form: the read head is the resolution of the observation-intervention gap that the earlier causal work (Phase 3A, PC0 swap, BH retrieval, MOAMS-X) had been circling. If that sentence is wrong, one of our experiments should break it. If none do, we have the right model.

---

*End of document. Companion pieces: `EARLIER_CAUSAL_WORK.md` (G·f*·G' and the earlier causal arc), `WHAT_YOU_MAY_HAVE_RUSHED_PAST.md` (under-appreciated findings — note that the PACF item is corrected in `EARLIER_CAUSAL_WORK.md` §10), `2026-04-07_observation_intervention_gap.md` (the V⊕V⊥ decomposition and Jacobian block structure this doc's §5 simplifies), `crossmodel_validation_2026-04-06.md` (the cross-model universality table), `session_analysis_2026-04-05.md` (project inventory as of early April).*
