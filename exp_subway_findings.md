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

## Caveats / next

- 0.5B, one sentence — a demonstration, not a sweep. Natural follow-ups:
  multiple sentences, longer text (where 1-vector won't hold), the 3B model
  (where ordinal *counting* works, so we could verify by literally asking
  "what was the 7th word"), and measuring how k scales with sentence length /
  entropy (the real "compression ratio" curve).
- Output: `output/exp_subway_token_compression.json`.
