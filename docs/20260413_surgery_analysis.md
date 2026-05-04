# Surgery Method Comparison: Session 2026-04-13

## Executive Summary

Four convention-removal methods tested. **Bilateral mean-diff is the current winner (+4/140)** but the new contrastive PCA method (MS4, running) has the strongest theoretical basis. The Procrustes method (MS3) is mathematically flawed due to rotation-space averaging artifacts.

## Results Table

| Method | Direction Source | Total | Δ | EN | ZH | AR | ES | JA | KO | SW |
|--------|-----------------|-------|---|----|----|----|----|----|----|-----|
| **Baseline** | — | 110/140 | — | 19 | 18 | 14 | 20 | 13 | 16 | 10 |
| **MS1 mean-diff** | EN−ZH pair | 33/40* | +6 | 16→20 | 11→13 | — | — | — | — | — |
| **MS2b LOO-SVD** | Centroid SVD, LOO | 94/140 | **−16** | 20 | 12 | 15 | 11 | 12 | 17 | 7 |
| **MS2b Bilateral** | Avg 6 pairwise diffs | 114/140 | **+4** | 19 | 18 | 17 | 19 | 18 | 17 | 6 |
| **MS3 Procrustes** | Rotation deviation | N/A | N/A | — | — | — | — | — | — | — |
| **MS4 cPCA** | Task-orthogonal | Running | — | — | — | — | — | — | — | — |

*MS1 tested only EN+ZH (40 total), not all 7 languages.

## Key Findings

### 1. LOO-SVD is destructive (−16)

The centroid SVD direction captures the **CJK-vs-Latin script-family axis**, which is the dominant structure in cross-lingual variation. But this axis is entangled with computation:
- ES collapses from 20→11 (−9): the SVD removes Latin-script computation
- ZH drops from 18→12 (−6): CJK computation stripped
- EN is IMMUNE (20/20, +1): the convention direction = "non-English-ness"

Per-problem analysis shows combinatorics most vulnerable (38% regression rate) but all categories affected (20-38%). The damage is not category-specific — it's a broad corruption of non-English reasoning.

### 2. Bilateral mean-diff works (+4)

The bilateral approach averages 6 pairwise mean-diffs per target language. This is more conservative than SVD because:
- It's averaging directions, not finding max-variance
- Each pairwise diff captures convention specific to that pair
- Averaging cancels pair-specific biases, preserving common convention

**JA +5 is the standout**: all 5 improvements span different categories (algebra, arithmetic, geometry, combinatorics). **Zero regressions** — bilateral is purely additive for Japanese. JA has robust independent math training, so convention removal reveals clean computation underneath.

**SW −4 is the cautionary tale**: bilateral strips Swahili's arithmetic and geometry (its STRONGEST categories at baseline). SW likely uses a computational "bridge" through English — its convention direction IS its math pathway. This is evidence that low-resource languages rely on shared representations that surgery damages.

### 3. Procrustes rotation spectrum is flat

sv1/sv2 = 1.00-1.015 at ALL 36 layers. No dominant rotation axis exists.

**Why**: rotation matrices live on SO(n), a curved manifold. Euclidean averaging (R_avg = mean of R_ij) causes destructive interference — off-diagonal elements cancel. The averaged rotation is nearly identity, and its deviation has no structure. Each pair rotates in a different direction; averaging them gives noise.

**Fix**: average in the Lie algebra (log-map R matrices, average the skew-symmetric matrices, exp-map back). Or use joint diagonalization to find common rotation axes across pairs.

**Implication**: MS3 as designed would remove a random direction, producing null results. Killed the run.

### 4. Contrastive PCA finds a clean convention axis

cPCA solves the generalized eigenvalue problem Σ_lang v = λ Σ_math v:
- Σ_lang: between-language covariance (7 centroids)
- Σ_math: between-problem covariance (200 problems, averaged across languages)
- Top eigenvector: max language variance, constrained to nullspace of math variance

Results:
- **λ1/λ2 ≈ 1.5-4.0** (moderate to clean convention axis). Best at L0 (4.0), surgery range mean = 1.95
- **math_var = 0.0000** at every layer (by construction, direction is orthogonal to problem variance)
- **cos(cPCA, ms1) ≈ 0.56** in surgery range (moderately aligned with MS1 mean-diff)
- **cos(cPCA, csvd) ≈ 0.73** in surgery range (substantially aligned with centroid SVD)
- Convention axis WEAKENS at bottleneck (L30+, λ1/λ2 drops to 1.15 at L35)

The 73% alignment with centroid SVD is both promising and concerning: cPCA captures a similar direction but constrained to zero math variance. Whether this constraint prevents the destructive effects of LOO-SVD is what MS4 will answer.

## Theoretical Framework

### Why does aggressiveness matter?

The convention subspace and computation subspace are **not orthogonal**. They share a substantial overlap (~73% cosine similarity). The difference between methods:

1. **Full SVD (LOO)**: removes the direction of MAXIMUM cross-lingual variance. This direction is ~73% convention + ~27% computation. Removing it strips critical computation.

2. **Mean-diff (MS1, bilateral)**: removes a pairwise difference direction. This is gentler — the mean-diff between two specific languages captures mostly convention-specific offset with less computational contamination.

3. **cPCA**: mathematically constrained to zero math variance. Even though it's 73% aligned with the centroid SVD direction, the constraint forces it into the nullspace of Σ_math, avoiding the computational components.

### The bridge-language effect

Languages fall into two categories under surgery:
- **Computationally autonomous** (EN, JA, AR, KO, ES): have robust, independent math reasoning pathways. Convention removal reveals clean computation. Net positive.
- **Bridge-dependent** (SW): relies on shared pathways with higher-resource languages. Convention direction IS computational infrastructure. Surgery damages basic capabilities.

This predicts: convention surgery benefits scale with training data quality for each language.

## Open Questions

1. **MS4 result**: will cPCA surgery beat bilateral (+4)? Zero math variance is a strong constraint, but the direction might still carry causally important constant offsets.
2. **Per-language cPCA**: could we compute per-language cPCA directions (like bilateral does for mean-diffs)? This would allow different surgery for different languages, potentially protecting SW.
3. **INLP approach**: iterative nullspace projection trains a language classifier and projects out its weight vector repeatedly. More aggressive than single-direction removal but targeted.
4. **Injection experiment**: does the model introspect on language-agnostic hidden states? Would bypass the surgery question entirely.

## Files

- `expMS2b_loo_surgery.py` → `output/expMS2b_loo_bilateral_surgery.json` (complete)
- `procrustes_preview_fast.py` (diagnostic, CPU-only)
- `cpca_preview.py` (diagnostic, CPU-only)
- `expMS4_cpca_surgery.py` → running
- `expINJ_hidden_injection.py` → staged, not run
