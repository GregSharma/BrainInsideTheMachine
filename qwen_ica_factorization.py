"""
ICA Factorization Analysis on Qwen2.5-3B Cross-Lingual Activations

Core hypothesis: The model learns a FACTORED representation h = f(problem) + g(language)
where f and g live in complementary subspaces. ICA should recover these independent factors.

Tests:
1. ICA decomposition at each layer — classify components as "language" vs "reasoning"
2. Factorization quality: how cleanly do ICA components separate language from content?
3. Language subspace dimensionality: how many ICs are needed to predict language?
4. Subspace removal: project out language ICs, measure cross-lingual NN on residual
5. Inter-layer correlation structure: shared vs language-specific component correlations
"""

import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
data = np.load('output/viz_activations.npz', allow_pickle=True)
layers = [8, 16, 24, 32, 34]
categories = data['categories']  # 5 problem categories

results = {}

for layer in layers:
    zh = data[f'zh_L{layer}']  # (200, 2048)
    en = data[f'en_L{layer}']  # (200, 2048)
    N = zh.shape[0]

    # Combined matrix: 400 x 2048
    combined = np.vstack([zh, en])
    lang_labels = np.array([0]*N + [1]*N)  # 0=zh, 1=en
    problem_ids = np.concatenate([np.arange(N), np.arange(N)])

    print(f"\n{'='*70}")
    print(f"LAYER {layer}")
    print(f"{'='*70}")

    # --- Step 1: PCA first (ICA needs dimensionality reduction for stability) ---
    n_components = 100  # enough to capture structure, few enough for stable ICA
    pca = PCA(n_components=n_components, random_state=42)
    combined_pca = pca.fit_transform(combined)
    var_explained = pca.explained_variance_ratio_.cumsum()
    print(f"PCA: {n_components} components explain {var_explained[-1]*100:.1f}% variance")
    print(f"  50 components explain {var_explained[49]*100:.1f}%")
    print(f"  20 components explain {var_explained[19]*100:.1f}%")

    # --- Step 2: ICA on PCA-reduced data ---
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, tol=1e-4)
    S = ica.fit_transform(combined_pca)  # (400, n_components) — independent sources

    # --- Step 3: Classify each IC as "language" vs "reasoning" ---
    # For each IC, compute:
    #   - language_score: |mean(zh) - mean(en)| / std (effect size for language prediction)
    #   - problem_score: correlation with problem identity (do matched pairs have similar IC values?)

    language_scores = []
    problem_scores = []

    for ic in range(n_components):
        ic_zh = S[:N, ic]  # this IC's values for Chinese problems
        ic_en = S[N:, ic]  # this IC's values for English problems

        # Language effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(ic_zh) + np.var(ic_en)) / 2)
        if pooled_std > 1e-10:
            lang_d = abs(np.mean(ic_zh) - np.mean(ic_en)) / pooled_std
        else:
            lang_d = 0
        language_scores.append(lang_d)

        # Problem correlation: correlation between zh and en values for matched problems
        if np.std(ic_zh) > 1e-10 and np.std(ic_en) > 1e-10:
            prob_corr = np.corrcoef(ic_zh, ic_en)[0, 1]
        else:
            prob_corr = 0
        problem_scores.append(prob_corr)

    language_scores = np.array(language_scores)
    problem_scores = np.array(problem_scores)

    # Classify: "language" if lang_d > 1.0, "reasoning" if prob_corr > 0.3, "mixed" otherwise
    lang_ics = np.where(language_scores > 1.0)[0]
    reasoning_ics = np.where((problem_scores > 0.3) & (language_scores < 1.0))[0]
    mixed_ics = np.where((language_scores <= 1.0) & (problem_scores <= 0.3))[0]

    print(f"\nIC Classification (n={n_components}):")
    print(f"  Language ICs (Cohen's d > 1.0): {len(lang_ics)}")
    print(f"  Reasoning ICs (corr > 0.3, not lang): {len(reasoning_ics)}")
    print(f"  Mixed/noise ICs: {len(mixed_ics)}")

    # Top 5 language ICs
    top_lang = np.argsort(language_scores)[::-1][:5]
    print(f"\n  Top 5 language ICs (by Cohen's d):")
    for idx in top_lang:
        print(f"    IC{idx}: d={language_scores[idx]:.3f}, prob_corr={problem_scores[idx]:.3f}")

    # Top 5 reasoning ICs
    top_reason = np.argsort(problem_scores)[::-1][:5]
    print(f"\n  Top 5 reasoning ICs (by problem correlation):")
    for idx in top_reason:
        print(f"    IC{idx}: d={language_scores[idx]:.3f}, prob_corr={problem_scores[idx]:.3f}")

    # --- Step 4: Language prediction with cumulative ICs ---
    # Sort ICs by language_score descending, then measure accuracy with top-k
    lang_order = np.argsort(language_scores)[::-1]

    print(f"\n  Language prediction accuracy (logistic regression on top-k ICs):")
    for k in [1, 3, 5, 10, 20]:
        top_k = lang_order[:k]
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(S[:, top_k], lang_labels)
        acc = clf.score(S[:, top_k], lang_labels)
        print(f"    k={k:2d}: accuracy={acc:.3f}")

    # --- Step 5: Remove language subspace, measure cross-lingual NN ---
    # Project out top-k language ICs and reconstruct
    for k_remove in [1, 3, 5, 10]:
        top_k_lang = lang_order[:k_remove]
        S_cleaned = S.copy()
        S_cleaned[:, top_k_lang] = 0  # zero out language ICs

        # Reconstruct in PCA space, then original space
        combined_cleaned = ica.inverse_transform(S_cleaned)
        combined_cleaned = pca.inverse_transform(combined_cleaned)

        zh_clean = combined_cleaned[:N]
        en_clean = combined_cleaned[N:]

        # Cross-lingual NN accuracy
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(en_clean)
        _, indices = nbrs.kneighbors(zh_clean)
        nn_acc = np.mean(indices.flatten() == np.arange(N))

        # Also compute Procrustes R² on cleaned data
        zh_c = zh_clean - zh_clean.mean(0)
        en_c = en_clean - en_clean.mean(0)
        R, _ = orthogonal_procrustes(zh_c, en_c)
        zh_rot = zh_c @ R
        ss_res = np.sum((zh_rot - en_c)**2)
        ss_tot = np.sum((en_c - en_c.mean(0))**2)
        r2 = 1 - ss_res / ss_tot

        print(f"\n  After removing top-{k_remove} language ICs:")
        print(f"    Cross-lingual NN accuracy: {nn_acc:.3f}")
        print(f"    Procrustes R²: {r2:.3f}")

    # --- Step 6: Independence test ---
    # Are language ICs truly independent of reasoning ICs?
    # Compute mutual information proxy: correlation between |language IC| and |reasoning IC|
    if len(lang_ics) > 0 and len(reasoning_ics) > 0:
        cross_corrs = []
        for li in lang_ics[:5]:  # top 5 language
            for ri in reasoning_ics[:5]:  # top 5 reasoning
                cc = abs(np.corrcoef(S[:, li], S[:, ri])[0, 1])
                cross_corrs.append(cc)
        print(f"\n  Independence check (|corr| between language and reasoning ICs):")
        print(f"    Mean: {np.mean(cross_corrs):.4f}")
        print(f"    Max:  {np.max(cross_corrs):.4f}")
        print(f"    (ICA guarantees decorrelation, so these should be ~0)")

    # --- Step 7: Category structure in reasoning ICs ---
    # Do reasoning ICs encode problem category?
    if len(reasoning_ics) >= 5:
        # Use zh data only, look at top reasoning ICs
        reason_features = S[:N, reasoning_ics[:20]] if len(reasoning_ics) >= 20 else S[:N, reasoning_ics]
        from sklearn.neighbors import NearestNeighbors
        nbrs_cat = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(reason_features)
        _, cat_indices = nbrs_cat.kneighbors(reason_features)
        # What fraction of 5-NN share the same category?
        same_cat = np.mean([
            np.mean(categories[cat_indices[i, 1:]] == categories[i])
            for i in range(N)
        ])
        print(f"\n  Category structure in reasoning ICs:")
        print(f"    5-NN same-category rate: {same_cat:.3f} (chance={1/5:.3f}=0.200)")

    # Store results
    results[f'L{layer}'] = {
        'n_language_ics': int(len(lang_ics)),
        'n_reasoning_ics': int(len(reasoning_ics)),
        'n_mixed_ics': int(len(mixed_ics)),
        'top_lang_scores': language_scores[top_lang].tolist(),
        'top_reason_scores': problem_scores[top_reason].tolist(),
        'pca_var_50': float(var_explained[49]),
        'pca_var_100': float(var_explained[-1]),
    }

# --- CROSS-LAYER ANALYSIS ---
print(f"\n{'='*70}")
print("CROSS-LAYER SUMMARY")
print(f"{'='*70}")

print(f"\n{'Layer':>6} | {'Lang ICs':>8} | {'Reason ICs':>10} | {'Mixed':>6} | {'Top Lang d':>10} | {'Top Reason r':>12}")
print("-" * 70)
for layer in layers:
    r = results[f'L{layer}']
    print(f"  L{layer:>3} | {r['n_language_ics']:>8} | {r['n_reasoning_ics']:>10} | {r['n_mixed_ics']:>6} | {r['top_lang_scores'][0]:>10.3f} | {r['top_reason_scores'][0]:>12.3f}")

# Save results
with open('output/ica_factorization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to output/ica_factorization_results.json")
