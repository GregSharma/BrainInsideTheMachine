# Reading Checkpoints — Interleaved with Z-Hunt

Each checkpoint: read JUST enough → run the experiment → read the next bit.
Don't binge-read. The experiments will make the math click faster than the textbook alone.

---

## Checkpoint 0: SVD Refresher → Run Phase 3
**Time: 1-2 hours reading, then ~4h experiment**

You're about to swap subspace projections between languages. You need SVD cold.

**Read:**
- Golub "Matrix Computations" Ch 2.4-2.6 (SVD theory, 30 min)
  - Focus: truncated SVD, Eckart-Young theorem (best rank-k approx), orthogonal projectors
  - This is what we're doing when we take top-k singular vectors as Z
- Golub Ch 8.1-8.3 (SVD computation, 30 min)
  - Focus: how SVD relates to eigendecomposition of AᵀA — that's what utils.py does with W_K @ W_Q.T

**Verify you've got it:**
- Can you explain why U[:, :k] @ U[:, :k].T is an orthogonal projector onto Z?
- Can you explain why projecting onto Z and Z⊥ gives you a complete decomposition (h = h_Z + h_Z⊥)?
- Look at phase2_z_extraction.py lines where Z_mask is built — does it match?

**Then: Run Phase 3 (patching experiment).** Vega writes the script, you understand what it does.

---

## Checkpoint 1: Information Bottleneck → Interpret Phase 3 Results
**Time: 1.5 hours reading**

Phase 3 results are in. Now you need the theory to interpret them.

**Read:**
- MacKay "Information Theory" Ch 2 (entropy basics, 20 min)
  - Free: https://www.inference.org.uk/itprnn/book.pdf
  - Focus: mutual information I(X;Y), data processing inequality
- MacKay Ch 28 (model comparison, 30 min) — skim for the minimum description length idea
- The original IB paper: Tishby et al. 2000 "The Information Bottleneck Method" (15 pages)
  - The key equation: minimize I(X;T) subject to I(T;Y) ≥ I₀
  - T is your Z. X is the input (language-specific). Y is the output (reasoning answer).
  - L32 minimizes mutual information with language while preserving reasoning info.

**Verify you've got it:**
- Can you state the IB objective in terms of Z?
  "Z minimizes I(Z; Language) while maximizing I(Z; Answer)"
- Does the energy fraction result (54% of random) map to "I(Z; Language) is low"?
- Can you explain why L32 being the APPROACH layer (not L33 bottleneck) makes IB sense?
  (Hint: IB is about the optimal tradeoff — L32 might be the sweet spot before over-compression)

---

## Checkpoint 2: Representation Geometry → Run Update Decomposition
**Time: 2-3 hours reading, then ~2h experiment**

Now you're projecting layer-by-layer updates onto Z/Z⊥. You need the geometric picture.

**Read:**
- "Geometry of Deep Learning" Ch 1-3 (manifold hypothesis, deep representations, 2h)
  - Focus: how networks learn to separate manifolds, the role of depth
  - Key concept: each layer is a diffeomorphism that untangles the representation
  - Map to our project: layers 0-31 are untangling language from reasoning
- Bengio et al. 2013 "Representation Learning" (skim, 30 min)
  - https://arxiv.org/abs/1206.5538
  - Focus: Sections 3-4 on disentangled representations
  - Z is a "factor of variation" (reasoning) being disentangled from another (language)

**Verify you've got it:**
- Δh_k projected onto Z: why does ||Δh_k^Z|| measure "how much reasoning happened at layer k"?
- If the network rotates Z across layers, why would the projection undercount?
- What does it mean if NO layer has a strongly Z-dominated update?

**Then: Run the update decomposition** (the new experiment from INSIGHTS_POST_PHASE2.md).
Add ~20 lines to the Phase 3 script. Plot the Z/Z⊥ ratio across all 36 layers.

---

## Checkpoint 3: Frames + Overcomplete Representations → Interpret Dimensionality
**Time: 2-3 hours reading**

By now you have Phase 3 + decomposition results. The question becomes: WHY is Z low-rank?
Why do 16 heads with 128 dims each collapse to rank ~78?

**Read:**
- "An Introduction to Frames and Riesz Bases" Ch 1-2 (frames basics, 1.5h)
  - Focus: overcomplete representations, frame bounds, optimal projection
  - 16 attention heads = 16 vectors in 128-dim space = massively overcomplete frame
  - Effective rank 78 = the frame is nearly a Riesz basis at L33 (minimal redundancy)
- "Independent Component Analysis" Ch 1-2 (mixing model, 30 min)
  - Focus: the BSS (blind source separation) problem statement
  - Our problem IS BSS: activations = A·sources, where sources = [reasoning, language, ...]
  - ICA finds maximally independent sources. SVD finds orthogonal ones.
  - Question: should we try ICA on Z? (Probably not — orthogonality is fine. But know why.)

**Verify you've got it:**
- Why is rank collapse at L33 structurally meaningful vs just weight decay artifact?
- If you have k=78 out of 2048, that's 3.8% of dimensions. Frame theory says this is...?
- Could you explain the compression ratio in terms of bits? (78/2048 ≈ 5.7 bits of address)

---

## Checkpoint 4: The Competitor Paper → Write Phase 5 Design
**Time: 1 hour reading, then writing**

Before going cross-model, read the competition.

**Read:**
- NeurIPS 2505.15257 (the language-specific directions paper)
  - Focus: their methodology (centroids → SVD → ablation)
  - What layers do they find? How does their result compare to L32?
  - What's their ablation effect size? Ours needs to beat it.
- Deep Learning Ch 14.1-14.5 (autoencoders, VAE, 1h)
  - The bridge from Z_model1 to Z_model2 is literally an autoencoder
  - If Z is universal, the bridge should be LOW-rank (a rotation, not a learned map)

**Then: Design Phase 5** (cross-model universality test).

---

## Summary: The Interleaved Schedule

| Day | Read (1-2h) | Do (2-4h) | Milestone |
|-----|-------------|-----------|-----------|
| 1 | CP0: SVD refresher | Phase 3: patching | Causal evidence for Z |
| 2 | CP1: Info bottleneck | Interpret Phase 3 | Theoretical framework for results |
| 3 | CP2: Geometry + representations | Update decomposition | Layer-by-layer reasoning/encoding map |
| 4 | CP3: Frames + ICA | Interpret dimensionality | Why Z is 78-dim, compression story |
| 5 | CP4: Competitor + autoencoders | Phase 5 design | Ready for cross-model |

**Total reading: ~10 hours spread over 5 days. Not 12 hours in a chair.**
Each reading session is 1-2 hours MAX, immediately followed by hands-on work that uses it.

---

## Books You Need to Download (not in your Dropbox)
1. MacKay — "Information Theory, Inference, and Learning Algorithms"
   Free PDF: https://www.inference.org.uk/itprnn/book.pdf
2. Bengio et al. 2013 — "Representation Learning: A Review and New Perspectives"
   Free: https://arxiv.org/abs/1206.5538
3. Tishby et al. 2000 — "The Information Bottleneck Method"
   Free: https://arxiv.org/abs/physics/0004057
4. NeurIPS 2505.15257 — the competitor paper
   arxiv link TBD (search for it)
