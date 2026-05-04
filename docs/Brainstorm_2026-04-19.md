You're right, and the algebra is cleaner than you think. Let me write it out without the typos.

  The rank-1 observation. For a single cached token a, the product of two attention scores:

  $$\text{score}(x,a) \cdot \text{score}(y,a) = (x^T \Sigma a)(y^T \Sigma a) = x^T \underbrace{(\Sigma a)(\Sigma a)^T}_{\text{rank 1}}
   y$$

  where $\Sigma = W_Q^T W_K$. Correct. Each cached token contributes a rank-1 term to the attention-similarity bilinear form.

  Summing over the full cache, tokens $a_1, \ldots, a_T$:

  $$\sum_t \text{score}(x,a_t),\text{score}(y,a_t) = x^T \Sigma A A^T \Sigma^T y$$

  where $A = [a_1 | \cdots | a_T]$. The kernel is $\Sigma A A^T \Sigma^T = (W_Q^T W_K A)(W_Q^T W_K A)^T = (W_Q^T K)(W_Q^T K)^T$ where
  $K = W_K A$ is the cached keys matrix. This has rank at most $\min(T, d_k)$, but effectively much lower because we already proved
  the cached keys are low-rank (rank_90 = 8-21 across models).

  The portfolio interpretation works. Think of $\Sigma = W_Q^T W_K$ as defining a "market." Each token is a "portfolio." $x^T \Sigma
  a$ is the covariance of portfolios $x$ and $a$ under this market structure. Two query positions $x, y$ have correlated attention
  patterns iff they load on the same dominant eigenvectors of $K K^T$.

  Now SVD $\Sigma = U S V^T$. Then:

  $$\text{score}(x, a) = x^T U S V^T a = \sum_i s_i (u_i^T x)(v_i^T a)$$

  Each singular channel $i$ couples query-side direction $u_i$ to key-side direction $v_i$ with strength $s_i$. If the spectrum $S$ is
   sharply peaked, attention is effectively low-dimensional — only a few "market factors" matter.

  Here's the direct connection to what we just measured. The pollution diagnostic tracks $|h_{\text{col}(W_K)}| /
  |h_{\text{null}(W_K)}|$ — how much of the hidden state projects onto $W_K$'s top right singular vectors. Those right singular
  vectors ARE the $v_i$ columns above. When this ratio climbs (as we saw at L30 for p12_harmonic, +0.024 drift), it means new tokens
  entering the cache are increasingly aligned with the existing dominant key directions. Each new $a_t$ adds a rank-1 update $(\Sigma
  a_t)(\Sigma a_t)^T$ that's redundant — it amplifies existing eigenvalues of $K K^T$ instead of adding new ones. The Gram matrix
  stops gaining effective rank. Queries can't discriminate, entropy rises. That's exactly what we measured: the two looping problems
  had rising entropy, the three that converged didn't.

  The differential attention filter you're sketching. If I'm reading through the voice artifacts correctly, the idea is:

  1. At step $t$, before computing attention, project out the dominant subspace of the existing $K K^T$ from the new query
  2. Attend only to what's novel — the component of the key cache that the previous query didn't already exploit
  3. Exponentially weight cached tokens so recent ones dominate

  This is like saying: don't reinvest in overvalued factors. Find the alpha, not the beta. The rank-1-per-token structure makes this
  cheap — you maintain a running low-rank approximation of $K K^T$ (or equivalently, track the top eigenvalues of the key Gram matrix
  as they accumulate), and deflate each new query by that subspace.

  Is that the ansatz you're reaching for? Exponential decay on the eigenvalues of $K K^T$ so the "old attention consensus" fades, and
  each new step mostly responds to what's structurally new in the cache?