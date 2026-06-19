# Subway sentence: can 9 words ride in fewer than 9 tokens?

**Question (Greg).** Take a short sentence —

> *do not lean on car doors on the subway*

— which tokenizes to exactly **9 tokens (one per word)** under Qwen. Can we
replace those 9 token-embeddings with *fewer* injected vectors and still
recover the exact same information? Verify by asking, word by word: what was
the 1st word? the 2nd? … the 9th? All should hit.

**Setup.** No GPU on this box (4 CPU / 15 GB), so this runs at the smallest
faithful scale: **Qwen2.5-0.5B-Instruct**, fp32, CPU. Same `inputs_embeds`
injection mechanism as `expAN_token_compression.py`; only the model size
differs from the repo's mainline 3B runs.

A 0.5B model **cannot** do positional *counting* ("what is the 7th word")
even from clean text (1/9) — that's a capability limit, not a compression
one. But it *can* recite a sentence verbatim. So we verify by injecting the
sentence representation, asking the model to recite, and scoring each of the
9 word positions against the recitation. Script: `exp_subway_token_compression.py`.

## Result

| representation | k (tokens) | compression | word hits |
|---|---|---|---|
| **baseline** (real 9 embeddings) | 9 | 1.0× | **9/9** |
| meanpool (naive) | 1–4 | 9×–2.25× | 0/9 |
| stride (naive) | 1–4 | 9×–2.25× | 1/9 (just "do") |
| svd (naive) | 1–4 | 9×–2.25× | 0–1/9 |
| **optimized soft tokens** | **1** | **9×** | **9/9** |
| optimized soft tokens | 2,3,4 | 4.5×–2.25× | 9/9 |

The literal answer to the question, at the most aggressive setting — **a
single injected vector standing in for all 9 tokens (9× compression)**:

```
model recited: 'do not lean on car doors on the subway'
  1st word -> do      HIT      6th word -> doors   HIT
  2nd word -> not     HIT      7th word -> on      HIT
  3rd word -> lean    HIT      8th word -> the     HIT
  4th word -> on      HIT      9th word -> subway  HIT
  5th word -> car     HIT
                                => 9/9
```

## What this does and doesn't show

- **Naive compression destroys it.** Mean-pooling, striding, or SVD-ing the 9
  embeddings down to k vectors loses the word order / identity almost
  entirely (0–1 of 9). You cannot just average tokens together.
- **The information *fits* in fewer tokens — but you have to encode it.** One
  continuous 896-dim vector, found by ~150 steps of gradient descent against
  the *frozen* model (only the vector learns), carries all 9 words. This is the
  "soft prompt / gist token" result, reproduced at toy scale: a discrete token
  sequence is not the only carrier of its own information.
- **Is the single vector cheating?** Partly. Trained on one recite
  instruction, we then tested the *same* vectors against an **unseen**
  instruction phrasing:
  - k=1 → 1/9 (the single vector overfits the training trigger — it memorizes
    "emit this string" rather than "here is a sentence").
  - **k=2 → 9/9, k=4 → 9/9** under the unseen instruction. Two vectors already
    encode the sentence *content*, not a brittle trigger.

So the honest headline: the sentence's information is recoverable from **~2
injected vectors (4.5× fewer than its 9 tokens)** in a way that survives a
prompt it was never optimized against — and from **1 vector** if you only need
it under the prompt you trained on. Naive dimensionality reduction of the raw
embeddings does not work; an optimized injection does.

## Can we argmax it onto a real token? (snap test, no training)

Follow-up `exp_subway_snap.py`. The optimized vector recites 9/9, but how far
from the vocabulary does it live, and does snapping it back to the nearest real
token preserve anything?

| k | soft-token hits | nearest-token cosine | snapped token(s) | snapped hits |
|---|---|---|---|---|
| 1 | 9/9 | **0.163** | `' Doming'` | 0/9 |
| 2 | 9/9 | 0.151, 0.174 | `' DO'`, `' DO'` | 1/9 |
| 4 | 9/9 | 0.15–0.18 | `'"Not'`,`'istol'`,`' car'`,`' onPostExecute'` | 0/9 |

The working vector's cosine to its **nearest** of 151,936 real tokens is only
**~0.16** — essentially orthogonal to the entire vocabulary. It does not sit
near any token; it sits *between* the lattice points, in the interstitial
volume of embedding space. Argmax (snap to grid) collapses 9/9 → 0/9. The
nearest tokens are junk (`' Doming'`, `' onPostExecute'`).

And a purely **gradient-free discrete search** — pick the real token(s) that
maximize P(sentence | tokens) by forward passes only — also fails: the best
single token (`' subway'`) and best pair (`' subway'+' lean'`) both give 0/9.
No short *readable* prompt carries the sentence.

**Takeaway.** The compression is not expressible on the token grid. A real
token holds ~log2(151936) ≈ 17 bits; the continuous vector exploits the full
~896-dim channel. Snapping to the lattice rounds the payload away. The win
*requires* "non-readable words" — points off the vocabulary lattice — which is
also why this compresses *positions / KV / attention cost*, not storable bytes,
and is tied to these exact weights.

## Can we COMPUTE the carrier instead of training it? (closed-form, no SGD)

`exp_subway_closedform.py`. The dream: if mid-layer representations are
near-additive, the carrier could be built by a formula — sum the token
contributions, optionally bind position multiplicatively (a Vector Symbolic
Architecture / holographic encoding, with RoPE as the bind) — instead of found
by gradient descent. Every training-free single-vector construction, injected
at the input (embedding) layer:

| construction | hits |
|---|---|
| meanpool (mean of 9 embeddings) | 0/9 |
| bag_sum (raw sum) | 0/9 |
| bag_norm (sum, rescaled) | 0/9 |
| rope_bind (Σ RoPE-rotate(Eᵢ, i)) | 1/9 (spurious) |
| rope_bind_n (rescaled) | 0/9 |
| **opt_1 (150-step SGD, reference)** | **9/9** |

**Negative result, stated straight.** At the input boundary, no naive additive
or holographic construction recovers the sentence; SGD is still required. The
"transformer is a VSA you can write to analytically" idea, in its *cheapest*
form (sum embeddings at layer 0), is false here.

Two honest reasons this is a *weak* test of the deeper conjecture, not a
refutation of it:
1. **Wrong space.** The additivity/superposition hypothesis is about *mid-layer*
   residual stream, after attention has mixed positions — not raw input
   embeddings. RoPE binds *inside attention*, not on embeddings, so rotating
   embeddings and injecting at layer 0 is not how the model uses position.
2. **Weak binding keys.** RoPE's low-frequency schedule (θ=1e6) leaves most of
   an 896-dim vector nearly unrotated at positions 0–8, so `rope_bind` ≈
   `bag_sum` — it never actually tested holographic binding with near-orthogonal
   keys.

What it *does* establish: the closed form, if one exists, does **not** live at
the input boundary. The proper next test is to construct/inject in *mid-layer*
residual-stream space (where the repo's kernel/Z work already lives) and to use
real near-orthogonal binding keys, keeping k>1 positions so RoPE's scaffold
survives. Until then: existence of a cheap analytic encoder is **unproven**, and
SGD (or a learned amortized encoder) remains the only thing demonstrated to work.

## Capacity: how many tokens fit in ONE vector? (exp_subway_capacity.py)

Single optimized vector (k=1), free-generation reconstruction, vs target length:

| n tokens | natural English | random tokens |
|---|---|---|
| 4 | 4/4 | 4/4 |
| 8 | 8/8 | 8/8 |
| 12 | 12/12 | 11/12 (loss 1.28) |
| 16 | 16/16 | **0/16** (loss 4.84) |
| 24 | 24/24 | 0/24 |
| 32 | **32/32** | 0/32 |

One 896-dim vector reconstructs **32 tokens of English verbatim** (32×; never
broke within our 49-token pool), while **random/incompressible tokens collapse
at ~12–16**. The natural-vs-random gap is linguistic redundancy: compressible
content packs ~3× further. Min-k on a 32-token natural target: k=1 already gives
32/32, so larger k is redundant at this length.

Honest caveat: the random ceiling here is **optimization-limited**, not proven
to be the information ceiling — at n≥16 the optimizer failed to fit the *training*
loss (4.8–8.4) in 200 steps, so "0/16" means "couldn't be found in budget," not
"provably impossible." The JL/superposition bound for linear decode is ~d/log2(V)
≈ 52 random slots; we're well under it because the readout is nonlinear and the
budget is small. The qualitative law (natural ≫ random, capacity scales) holds.

## Prior art — this is a faithful REPRODUCTION, not a discovery

Novelty check (web search, June 2026): single-vector / soft-prompt context
compression is an established, active subfield. The experiment here — optimize
one continuous input vector by gradient descent so a frozen LLM reconstructs a
sequence — is essentially **"Cramming 1568 Tokens into a Single Vector and Back
Again"** (Kuratov et al., arXiv 2502.13063, Feb 2025), which crams up to **1,568
tokens into one vector** and includes the capacity + data-compressibility
(natural-vs-random) analysis our toy reproduces. Adjacent: GIST (Mu 2023, ~26×),
AutoCompressors (Chevalier 2023), ICAE (Ge, ICLR 2024, ~4×), 500xCompressor
(Cambridge 2024, up to 480× into ~1 token), KV-Distill / Cartridges (2025).

What these papers do *not* settle, and where original work could live: a
**training-free, closed-form** encoder (they all either train an encoder or run
per-instance GD), and a mechanistic *theory of why* the capacity is what it is
(VSA / RoPE-as-binding). Our input-layer closed-form attempt FAILED, so that
remains open, not claimed.

## Is the sentence in any single MID-LAYER vector? No. (exp_subway_midlayer.py)

The deeper conjecture / "free encoder" hope: maybe the sentence is recoverable
from one mid-layer residual the model *already* produces (cost = one forward
pass, no training). Test: capture the last-sentence-token residual at every
layer (causal attention => it has seen the whole sentence), inject it at a
single slot, ask the model to recite. Result, **every layer L=0..22: 0/9.**

| L | real-vector recite | what the model says |
|---|---|---|
| 0–6 | 0/9 | just "subway" (the last token only) |
| 8–10 | 0/9 | "...I gave you, word for word." |
| 12–20 | 0/9 | **"I'm sorry, but I cannot repeat a sentence"** |
| 22 | 0/9 | "I gave you nothing else." |

No-injection control (neutral slot): 0/9. Norm-matched random: 0/9.

**The model knows it was given nothing.** The single mid-layer residual the model
naturally produces at the last position does NOT carry the recitable sentence.
This is the load-bearing negative:

> In normal operation the sentence lives **distributed across all 9 positions'
> KV cache**, not compressed into any one vector. The SGD soft token works
> *because it forces all of it into a single position* — somewhere the model's
> own dynamics never go. The compression is **off-manifold**: that is exactly
> why you cannot harvest it from a natural activation, and why optimization (or
> a trained encoder, à la GIST/ICAE) is mandatory rather than optional.

Consequence for the "cheap activations / free encoder" idea: you cannot grab a
mid-layer vector and call it the compressed carrier. The carrier is a point the
forward pass never produces; reaching it requires pushing off the manifold.

## What is the trained vector, mechanistically? Write-only attention memory.
(exp_subway_lens.py)

Logit-lens the slot position layer by layer for the working SGD vector vs the
failing mean-pool vector. The working vector is **not decodable into the
sentence at any layer** — its top tokens are junk at every depth:

```
L0  dni / dess / dif        L8  ibri/oola      L16 '].'/ / ��이
L4  CGPointMake / rames     L12 '].'/          L24 were / broke / soared
```

It never surfaces "do"/"not"/"lean". So the carrier is **write-only memory for
attention**: recitation is produced at the *suffix* positions, which read the
slot through its keys/values; the slot's own residual->unembedding path is
irrelevant and stays junk. The vector is optimized to be *attended to*, not to
be *decoded*.

This is the same fact seen three ways:
- off the **token lattice** (nearest-token cosine 0.16; snap -> 0/9),
- off the **activation manifold** (harvesting a natural mid-layer vector -> 0/9),
- off the **readout path** (logit-lens junk at every layer).

It is an attention-addressed memory cell — not a token, not a readable state.
(Caveat: residual streams predict the *next* token, so a slot need not lens to
its own content even normally; but "not decodable to the sentence at any depth"
is the strong, pinning observation.)

## Control: is B's negative just "injection doesn't work"? No. (exp_subway_positions.py)

Transplant the model's own layer-L residuals for the last m of 9 sentence
positions into m of 9 slots; recite; sweep m. Recall (/9):

| L | m=1 | m=3 | m=5 | m=7 | m=9 |
|---|---|---|---|---|---|
| 4 | 0 | 1 | 0 | 0 | **6** |
| 8 | 0 | 1 | 1 | 1 | **6** |
| 12 | 0 | 1 | 1 | 0 | 0 |
| 16,20 | 0 | 0 | 0 | 0 | 0 |

Two clean reads: (1) **injection works** — all-9 transplant at shallow layers
recites 6/9 vs 0/9 for one position, so B's single-vector negative is not an
artifact of broken machinery; (2) **you need ~all positions** — any subset
(m≤7) recovers essentially nothing. Both corroborate: the sentence is
distributed across positions, not held in any few.

Honest confound (not hidden): we overwrite only layer L while the slots' layers
< L carry a neutral seed, corrupting lower-layer attention — that ceilings m=9
at ~6/9 and makes deep-layer transplant fail outright. So this is directional
support, not a clean monotone curve. A faithful version would inject the full
per-layer residual stack for each slot.

## Caveats / next

- 0.5B, one sentence — a demonstration, not a sweep. Natural follow-ups:
  multiple sentences, longer text (where 1-vector won't hold), the 3B model
  (where ordinal *counting* works, so we could verify by literally asking
  "what was the 7th word"), and measuring how k scales with sentence length /
  entropy (the real "compression ratio" curve).
- Output: `output/exp_subway_token_compression.json`.
