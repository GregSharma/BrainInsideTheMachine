#!/usr/bin/env python3
"""
Experiment A: Cleaned MLP Delta Cross-Lingual Match
====================================================
THE money shot experiment. Tests whether MLP reasoning is language-agnostic
after removing language PCs.

For each layer:
1. Load zh_mlp_delta and en_mlp_delta (200 problems, 2048 dims)
2. Fit PCA on combined deltas, take top K PCs (K=10)
3. Project out top K PCs -> "cleaned" deltas (language removed)
4. Nearest-neighbor accuracy: does zh[i]'s cleaned delta land closest to en[i]?
5. Cosine similarity between paired cleaned deltas
6. Compare to uncleaned NN and random baseline
7. Category structure survival after cleaning

If cleaned NN > 0.5: MLP reasoning is language-agnostic. Language is additive.
If cleaned NN < 0.2: MLP reasoning is fundamentally language-shaped.
"""

import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Problem categories (from generate_problems)
np.random.seed(42)
CATEGORIES = ['arithmetic', 'combinatorics', 'modular', 'geometry', 'sequences']
N_PER_CAT = 40  # 200 / 5
category_labels = np.array([i for i in range(5) for _ in range(N_PER_CAT)])  # 200 labels

def nearest_neighbor_accuracy(query, gallery):
    """For each query[i], find nearest gallery vector. Accuracy = fraction where argmin == i."""
    # query: (N, D), gallery: (N, D)
    sim = cosine_similarity(query, gallery)  # (N, N)
    predictions = sim.argmax(axis=1)
    correct = (predictions == np.arange(len(query))).mean()
    return correct

def category_nn_accuracy(vecs, labels):
    """NN accuracy for category classification (5-way)."""
    sim = cosine_similarity(vecs, vecs)
    np.fill_diagonal(sim, -np.inf)
    preds = labels[sim.argmax(axis=1)]
    return (preds == labels).mean()

def run_layer(data, layer, n_pcs_list=[5, 10, 20]):
    zh = data[f'zh_mlp_delta_L{layer}'].astype(np.float64)  # (200, 2048)
    en = data[f'en_mlp_delta_L{layer}'].astype(np.float64)

    results = {'layer': layer}

    # --- Uncleaned baseline ---
    nn_uncleaned = nearest_neighbor_accuracy(zh, en)
    cos_uncleaned = np.array([np.dot(zh[i], en[i]) / (np.linalg.norm(zh[i]) * np.linalg.norm(en[i]) + 1e-10)
                               for i in range(200)])
    results['uncleaned_nn'] = float(nn_uncleaned)
    results['uncleaned_cos_mean'] = float(cos_uncleaned.mean())
    results['uncleaned_cos_std'] = float(cos_uncleaned.std())

    # Category NN on uncleaned zh and en separately
    cat_nn_zh_uncleaned = category_nn_accuracy(zh, category_labels)
    cat_nn_en_uncleaned = category_nn_accuracy(en, category_labels)
    results['uncleaned_cat_nn_zh'] = float(cat_nn_zh_uncleaned)
    results['uncleaned_cat_nn_en'] = float(cat_nn_en_uncleaned)

    # --- Random baseline: shuffle pairing ---
    perm = np.random.permutation(200)
    nn_random = nearest_neighbor_accuracy(zh, en[perm])
    results['random_nn'] = float(nn_random)

    # --- PCA cleaning ---
    combined = np.vstack([zh, en])  # (400, 2048)

    for n_pcs in n_pcs_list:
        pca = PCA(n_components=n_pcs)
        pca.fit(combined)

        # Project out top PCs
        zh_proj = pca.transform(zh)  # (200, n_pcs)
        en_proj = pca.transform(en)

        zh_cleaned = zh - pca.inverse_transform(zh_proj)  # (200, 2048)
        en_cleaned = en - pca.inverse_transform(en_proj)

        # NN accuracy on cleaned
        nn_cleaned = nearest_neighbor_accuracy(zh_cleaned, en_cleaned)

        # Cosine similarity on cleaned
        cos_cleaned = np.array([np.dot(zh_cleaned[i], en_cleaned[i]) /
                                (np.linalg.norm(zh_cleaned[i]) * np.linalg.norm(en_cleaned[i]) + 1e-10)
                                for i in range(200)])

        # Category NN on cleaned (combined zh+en)
        cat_nn_zh_cleaned = category_nn_accuracy(zh_cleaned, category_labels)
        cat_nn_en_cleaned = category_nn_accuracy(en_cleaned, category_labels)

        # Cross-lingual category NN: does zh[i]'s cleaned delta land in same category as en[i]'s?
        sim_cross = cosine_similarity(zh_cleaned, en_cleaned)
        cross_preds = category_labels[sim_cross.argmax(axis=1)]
        cat_cross_acc = (cross_preds == category_labels).mean()

        # Language classifier after cleaning: can we still tell zh from en?
        combined_cleaned = np.vstack([zh_cleaned, en_cleaned])
        lang_labels = np.array([0]*200 + [1]*200)
        sim_lang = cosine_similarity(combined_cleaned)
        np.fill_diagonal(sim_lang, -np.inf)
        lang_preds = lang_labels[sim_lang.argmax(axis=1)]
        lang_acc = (lang_preds == lang_labels).mean()

        # Variance explained by removed PCs
        var_explained = pca.explained_variance_ratio_.sum()

        prefix = f'pc{n_pcs}'
        results[f'{prefix}_nn'] = float(nn_cleaned)
        results[f'{prefix}_cos_mean'] = float(cos_cleaned.mean())
        results[f'{prefix}_cos_std'] = float(cos_cleaned.std())
        results[f'{prefix}_cat_nn_zh'] = float(cat_nn_zh_cleaned)
        results[f'{prefix}_cat_nn_en'] = float(cat_nn_en_cleaned)
        results[f'{prefix}_cat_cross_acc'] = float(cat_cross_acc)
        results[f'{prefix}_lang_acc'] = float(lang_acc)
        results[f'{prefix}_var_explained'] = float(var_explained)

        # Also try: fit PCA on zh only, project out from both
        pca_zh = PCA(n_components=n_pcs).fit(zh)
        zh_cleaned_zhpca = zh - pca_zh.inverse_transform(pca_zh.transform(zh))
        en_cleaned_zhpca = en - pca_zh.inverse_transform(pca_zh.transform(en))
        nn_zhpca = nearest_neighbor_accuracy(zh_cleaned_zhpca, en_cleaned_zhpca)
        results[f'{prefix}_nn_zhpca'] = float(nn_zhpca)

    return results


def main():
    print("Loading attn_mlp_deltas.npz...")
    data = np.load('output/attn_mlp_deltas.npz')

    all_results = []

    for layer in range(36):
        print(f"Layer {layer}...", end=' ', flush=True)
        res = run_layer(data, layer)
        all_results.append(res)

        # Print key metrics
        print(f"uncleaned_nn={res['uncleaned_nn']:.3f}  "
              f"pc10_nn={res['pc10_nn']:.3f}  "
              f"pc10_cos={res['pc10_cos_mean']:.3f}  "
              f"pc10_lang={res['pc10_lang_acc']:.3f}  "
              f"pc10_cat_zh={res['pc10_cat_nn_zh']:.3f}  "
              f"pc10_cat_cross={res['pc10_cat_cross_acc']:.3f}")

    # Summary
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Layer':>5} | {'Uncl NN':>8} | {'PC5 NN':>8} | {'PC10 NN':>8} | {'PC20 NN':>8} | "
          f"{'PC10 cos':>8} | {'PC10 lang':>9} | {'Cat zh':>7} | {'Cat cross':>9}")
    print("-"*100)

    for r in all_results:
        print(f"  L{r['layer']:>2}  | {r['uncleaned_nn']:>8.3f} | {r['pc5_nn']:>8.3f} | {r['pc10_nn']:>8.3f} | "
              f"{r['pc20_nn']:>8.3f} | {r['pc10_cos_mean']:>8.3f} | {r['pc10_lang_acc']:>9.3f} | "
              f"{r['pc10_cat_nn_zh']:>7.3f} | {r['pc10_cat_cross_acc']:>9.3f}")

    # Key summary stats
    layers_mid = [r for r in all_results if 12 <= r['layer'] <= 28]
    layers_late = [r for r in all_results if r['layer'] >= 28]

    print(f"\nPeak PC10 NN: L{max(all_results, key=lambda r: r['pc10_nn'])['layer']} = "
          f"{max(r['pc10_nn'] for r in all_results):.3f}")
    print(f"Mean PC10 NN (L12-L28): {np.mean([r['pc10_nn'] for r in layers_mid]):.3f}")
    print(f"Mean PC10 cos (L12-L28): {np.mean([r['pc10_cos_mean'] for r in layers_mid]):.3f}")
    print(f"Mean PC10 lang acc (L12-L28): {np.mean([r['pc10_lang_acc'] for r in layers_mid]):.3f}")

    # Verdict
    peak_nn = max(r['pc10_nn'] for r in all_results)
    if peak_nn > 0.5:
        print(f"\n>>> VERDICT: MLP reasoning is LANGUAGE-AGNOSTIC (peak NN={peak_nn:.3f} > 0.5)")
        print(">>> Language coupling in MLP is ADDITIVE. Separable. Paper's climax confirmed.")
    elif peak_nn > 0.2:
        print(f"\n>>> VERDICT: PARTIAL decomposability (peak NN={peak_nn:.3f})")
        print(">>> Some reasoning structure survives language removal, but not cleanly separable.")
    else:
        print(f"\n>>> VERDICT: MLP reasoning is LANGUAGE-SHAPED (peak NN={peak_nn:.3f} < 0.2)")
        print(">>> Computation is fundamentally different per language.")

    # Save
    output = {
        'experiment': 'A',
        'description': 'Cleaned MLP Delta Cross-Lingual Match',
        'method': 'PCA on combined zh+en MLP deltas, project out top K PCs, measure NN accuracy',
        'n_problems': 200,
        'n_dims': 2048,
        'n_layers': 36,
        'n_pcs_tested': [5, 10, 20],
        'per_layer': all_results,
        'peak_pc10_nn': float(peak_nn),
        'peak_pc10_layer': int(max(all_results, key=lambda r: r['pc10_nn'])['layer']),
    }

    with open('output/expA_mlp_delta_crosslingual.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to output/expA_mlp_delta_crosslingual.json")


if __name__ == '__main__':
    main()
