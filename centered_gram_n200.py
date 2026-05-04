"""Centered Gram rank at N=200 — resolves §5 rank-coincidence question.

Under Claim 5.2 (rank is ensemble-determined, bounded by N-1): centered rank_90 at
N=200 should be much larger than the N=20 value (~20), scaling roughly with N or sqrt(N).

Under Claim 5.3 (rank is training-determined, ensemble-insensitive): centered rank_90
should stay near 20 independent of N.

Under Claim 5.5 (convergent, upper-bound by training): centered rank_90 should increase
with N but asymptote at some training-determined ceiling.

Measurements:
    1. Load diverse_all_layers.npz (N=200 diverse, 7 langs, 36 layers, d=2048).
    2. For EACH language separately, compute centered Gram at each layer.
    3. Report rank_90, rank_95, rank_99 trajectories per language.
    4. Also combine languages (pool 7*200=1400) and report.
    5. Compare to N=20 baseline from multilingual_all_layers.npz if available.
"""
import numpy as np
import json
import sys


def centered_gram_ranks(H, thresholds=(0.5, 0.9, 0.95, 0.99)):
    """H: (N, d). Returns list of effective ranks at each threshold, and eigenvalues."""
    H = H.astype(np.float64)
    H_centered = H - H.mean(axis=0, keepdims=True)
    G = H_centered @ H_centered.T  # (N, N)
    eigs = np.linalg.eigvalsh(G)
    eigs = eigs[::-1]  # descending
    eigs = np.maximum(eigs, 0)
    total = eigs.sum()
    if total <= 0:
        return {f"rank_{int(t*100)}": 0 for t in thresholds}, eigs.tolist()
    cumfrac = np.cumsum(eigs) / total
    ranks = {}
    for t in thresholds:
        k = int(np.searchsorted(cumfrac, t) + 1)
        ranks[f"rank_{int(t*100)}"] = k
    return ranks, eigs.tolist()


def main():
    path = '/home/greg/Desktop/Projects/BrainInsideTheMachine/output/diverse_all_layers.npz'
    print(f"Loading {path}...", flush=True)
    d = np.load(path)
    langs = ['ar', 'en', 'es', 'ja', 'ko', 'sw', 'zh']
    n_layers = 36
    n_problems = 200

    results = {
        'experiment': 'centered_gram_n200',
        'question': '§5 rank coincidence: does centered rank_90 stay at ~20 when N=200?',
        'n_problems_per_lang': n_problems,
        'n_langs': len(langs),
        'per_language': {},
        'pooled_1400': {},
    }

    # Per-language centered Gram ranks
    for lang in langs:
        print(f"\nLanguage: {lang}", flush=True)
        lang_ranks = {'rank_50': [], 'rank_90': [], 'rank_95': [], 'rank_99': []}
        for L in range(n_layers):
            key = f'{lang}_L{L}'
            H = d[key]  # (200, 2048)
            ranks, _ = centered_gram_ranks(H)
            for k, v in ranks.items():
                lang_ranks[k].append(v)
        results['per_language'][lang] = lang_ranks
        r90 = lang_ranks['rank_90']
        r99 = lang_ranks['rank_99']
        print(f"  rank_90 first5={r90[:5]} mid5={r90[15:20]} last5={r90[-5:]}", flush=True)
        print(f"  rank_99 first5={r99[:5]} mid5={r99[15:20]} last5={r99[-5:]}", flush=True)

    # Pooled: stack all 7 languages × 200 problems = 1400 rows per layer
    print("\nPooled across all 7 languages (N=1400)...", flush=True)
    pooled_ranks = {'rank_50': [], 'rank_90': [], 'rank_95': [], 'rank_99': []}
    for L in range(n_layers):
        H_stack = np.concatenate([d[f'{lang}_L{L}'] for lang in langs], axis=0)  # (1400, 2048)
        ranks, _ = centered_gram_ranks(H_stack)
        for k, v in ranks.items():
            pooled_ranks[k].append(v)
    results['pooled_1400'] = pooled_ranks
    print(f"  rank_90: first5={pooled_ranks['rank_90'][:5]} mid5={pooled_ranks['rank_90'][15:20]} last5={pooled_ranks['rank_90'][-5:]}", flush=True)
    print(f"  rank_99: first5={pooled_ranks['rank_99'][:5]} mid5={pooled_ranks['rank_99'][15:20]} last5={pooled_ranks['rank_99'][-5:]}", flush=True)

    # Also check per-category (50 problems) for a small-N reference
    print("\nPer-language, per-category (N=50)...", flush=True)
    cats = d['categories']
    results['per_lang_per_cat_n50'] = {}
    for lang in langs[:2]:  # only en and zh for brevity
        for cat in [5, 6, 7, 8]:
            mask = cats == cat
            key_label = f'{lang}_cat{cat}'
            cat_ranks = {'rank_50': [], 'rank_90': [], 'rank_95': [], 'rank_99': []}
            for L in range(n_layers):
                H = d[f'{lang}_L{L}'][mask]  # (50, 2048)
                ranks, _ = centered_gram_ranks(H)
                for k, v in ranks.items():
                    cat_ranks[k].append(v)
            results['per_lang_per_cat_n50'][key_label] = cat_ranks

    # Headline comparison
    print("\n=== HEADLINE ===", flush=True)
    print("Layer | en(N=200)r90 | en(N=200)r99 | pooled(N=1400)r90 | pooled(N=1400)r99", flush=True)
    for L in [0, 9, 17, 20, 25, 30, 35]:
        en90 = results['per_language']['en']['rank_90'][L]
        en99 = results['per_language']['en']['rank_99'][L]
        p90 = pooled_ranks['rank_90'][L]
        p99 = pooled_ranks['rank_99'][L]
        print(f"L{L:02d}   |   {en90:3d}        |   {en99:3d}        |   {p90:3d}             |   {p99:3d}", flush=True)

    out_path = '/home/greg/Desktop/Projects/BrainInsideTheMachine/output/centered_gram_n200.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
