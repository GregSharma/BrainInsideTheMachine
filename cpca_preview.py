"""Contrastive PCA preview: find convention directions orthogonal to math computation.

Standard SVD finds max-variance direction across languages.
cPCA finds directions where language variance is HIGH but math variance is LOW.

This should isolate pure convention (language formatting/structure) from
computation (problem-solving pathways that happen to differ across languages).

Method:
- Σ_lang: covariance of language-mean activations (7 centroids, rank 6)
- Σ_math: covariance of problem-mean activations (20 problems, rank 19)
- Solve generalized eigenvalue: Σ_lang @ v = λ Σ_math @ v
- Top eigenvectors = directions with highest language/math variance ratio
"""
import numpy as np
from scipy.linalg import eigh
import time

LANGS = ['en', 'zh', 'ar', 'es', 'ja', 'ko', 'sw']
N_LAYERS = 36
D_MODEL = 2048
N_PROBLEMS = 200


def main():
    cache_path = "output/multilingual_all_layers.npz"
    print("Loading cache...", flush=True)
    data = np.load(cache_path, allow_pickle=True)
    t0 = time.time()

    print(f"\n{'L':>3s} | {'cPCA λ1':>9s} | {'λ1/λ2':>7s} | "
          f"{'cos(cPCA,ms1)':>13s} | {'cos(cPCA,csvd)':>14s} | "
          f"{'cos(ms1,csvd)':>13s} | {'lang_var':>9s} | {'math_var':>9s}", flush=True)
    print("-" * 105, flush=True)

    results = []

    for L in range(N_LAYERS):
        lang_acts = {}
        for lang in LANGS:
            lang_acts[lang] = data[f"{lang}_L{L}"].astype(np.float64)

        # All activations: (7 langs × 200 problems, 2048)
        # Each language has 200 problems

        # --- Language centroids (7, 2048) ---
        lang_means = np.stack([lang_acts[l].mean(axis=0) for l in LANGS])  # (7, d)
        grand_mean = lang_means.mean(axis=0)

        # Σ_lang: between-language covariance
        lang_centered = lang_means - grand_mean  # (7, d)
        Sigma_lang = lang_centered.T @ lang_centered / len(LANGS)  # (d, d)

        # --- Problem means averaged across languages (200, 2048) ---
        # For each problem i, average across all 7 languages
        problem_means = np.zeros((200, D_MODEL))
        for pi in range(200):
            for lang in LANGS:
                problem_means[pi] += lang_acts[lang][pi]
            problem_means[pi] /= len(LANGS)

        # Σ_math: between-problem covariance
        prob_grand = problem_means.mean(axis=0)
        prob_centered = problem_means - prob_grand  # (200, d)
        Sigma_math = prob_centered.T @ prob_centered / 200  # (d, d)

        # Add small regularization to Sigma_math to make it invertible
        Sigma_math += np.eye(D_MODEL) * 1e-6

        # --- Generalized eigenvalue: Σ_lang @ v = λ Σ_math @ v ---
        # Use scipy.linalg.eigh for symmetric generalized eigenvalue
        # Returns eigenvalues in ascending order
        try:
            eigenvalues, eigenvectors = eigh(Sigma_lang, Sigma_math,
                                             subset_by_index=[D_MODEL-3, D_MODEL-1])
            # Top 3 eigenvalues (descending)
            idx = np.argsort(-eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            e_c_cpca = eigenvectors[:, 0]  # top eigenvector
            e_c_cpca /= np.linalg.norm(e_c_cpca)
            lambda1 = eigenvalues[0]
            lambda2 = eigenvalues[1] if len(eigenvalues) > 1 else 1e-10
            ratio = lambda1 / lambda2 if abs(lambda2) > 1e-10 else float('inf')
        except Exception as e:
            print(f"L{L:2d} | ERROR: {e}", flush=True)
            results.append({'layer': L, 'error': str(e)})
            continue

        # --- MS1 mean-diff (EN - ZH) ---
        e_c_ms1 = lang_acts['en'].mean(axis=0) - lang_acts['zh'].mean(axis=0)
        e_c_ms1 /= np.linalg.norm(e_c_ms1) + 1e-12

        # --- Centroid SVD ---
        _, _, Vt_c = np.linalg.svd(lang_centered, full_matrices=False)
        e_c_csvd = Vt_c[0]

        # Cosines
        cos_cpca_ms1 = abs(np.dot(e_c_cpca, e_c_ms1))
        cos_cpca_csvd = abs(np.dot(e_c_cpca, e_c_csvd))
        cos_ms1_csvd = abs(np.dot(e_c_ms1, e_c_csvd))

        # Variance along cPCA direction
        lang_var = np.var(lang_centered @ e_c_cpca)
        math_var = np.var(prob_centered @ e_c_cpca)

        print(f"L{L:2d} | {lambda1:9.4f} | {ratio:7.2f} | "
              f"{cos_cpca_ms1:13.4f} | {cos_cpca_csvd:14.4f} | "
              f"{cos_ms1_csvd:13.4f} | {lang_var:9.4f} | {math_var:9.4f}", flush=True)

        results.append({
            'layer': L, 'lambda1': lambda1, 'lambda2': lambda2, 'ratio': ratio,
            'cos_cpca_ms1': cos_cpca_ms1, 'cos_cpca_csvd': cos_cpca_csvd,
            'cos_ms1_csvd': cos_ms1_csvd, 'lang_var': lang_var, 'math_var': math_var,
        })

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s", flush=True)

    # Phase summary
    valid = [r for r in results if 'error' not in r]
    print(f"\n{'=' * 70}")
    print("PHASE SUMMARY")
    print(f"{'=' * 70}")
    for phase, start, end in [
        ("Early (L0-L4)", 0, 5),
        ("Convention onset (L5-L12)", 5, 13),
        ("Surgery range (L13-L29)", 13, 30),
        ("Bottleneck (L30-L35)", 30, 36),
    ]:
        r = [x for x in valid if start <= x['layer'] < end]
        if not r:
            continue
        print(f"\n  {phase}:")
        print(f"    λ1/λ2 mean:         {np.mean([x['ratio'] for x in r]):.3f}")
        print(f"    cos(cPCA, ms1):     {np.mean([x['cos_cpca_ms1'] for x in r]):.4f}")
        print(f"    cos(cPCA, csvd):    {np.mean([x['cos_cpca_csvd'] for x in r]):.4f}")
        print(f"    lang/math var:      {np.mean([x['lang_var'] for x in r]):.4f} / {np.mean([x['math_var'] for x in r]):.4f}")

    # Key finding
    best = max(valid, key=lambda x: x['ratio'])
    print(f"\n  Peak λ1/λ2: L{best['layer']} = {best['ratio']:.3f}")
    print(f"    cos(cPCA, ms1) at peak: {best['cos_cpca_ms1']:.4f}")
    print(f"    cos(cPCA, csvd) at peak: {best['cos_cpca_csvd']:.4f}")

    print(f"\n  INTERPRETATION:")
    surgery_r = [x for x in valid if 13 <= x['layer'] < 36]
    mean_ratio = np.mean([x['ratio'] for x in surgery_r])
    mean_cos = np.mean([x['cos_cpca_ms1'] for x in surgery_r])
    if mean_ratio > 2:
        print(f"  cPCA finds a CLEAN convention axis (λ1/λ2={mean_ratio:.1f})")
        print(f"  High language variance, low math variance in this direction.")
        if mean_cos > 0.5:
            print(f"  ALIGNS with MS1 mean-diff (cos={mean_cos:.3f}) — should work as surgery target")
        else:
            print(f"  DIFFERENT from MS1 (cos={mean_cos:.3f}) — novel convention direction")
    else:
        print(f"  cPCA spectrum is moderate (λ1/λ2={mean_ratio:.1f})")
        print(f"  Convention and computation partially separable but not cleanly.")


if __name__ == "__main__":
    main()
