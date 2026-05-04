"""Systematic PCA rank sweep: how does alignment split between subspace and residual?

For k = 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000:
- Compute z-score in top-k PCA space
- Compute z-score in residual (d-k) space
- Track how alignment distributes
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

OUTPUT_DIR = Path("output")
data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")
N = 200


def z_score_from_vecs(zh_u, en_u, n_perms=300):
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    rng = np.random.RandomState(42)
    scr = [np.mean(np.sum(zh_u * en_u[rng.permutation(len(en_u))], axis=1)) for _ in range(n_perms)]
    scr = np.array(scr)
    return (matched - np.mean(scr)) / np.std(scr)


ranks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

for l in [15, 32, 35]:
    zh = data[f"zh_L{l}"]
    en = data[f"en_L{l}"]

    # Unit normalize
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

    # Full space z
    z_full = z_score_from_vecs(zh_u, en_u)

    # Fit PCA on combined
    combined = np.vstack([zh_u, en_u])
    max_components = min(combined.shape[0], combined.shape[1]) - 1
    pca_full = PCA(n_components=max_components)
    pca_full.fit(combined)

    print(f"\n=== L{l} (full z={z_full:.1f}) ===")
    print(f"{'rank k':>8s} {'z_top_k':>8s} {'z_resid':>8s} {'var_top_k':>10s} {'var_resid':>10s}")

    for k in ranks:
        if k > pca_full.n_components_:
            continue

        # Top-k projection
        V_k = pca_full.components_[:k]  # (k, d)
        zh_proj = zh_u @ V_k.T  # (N, k)
        en_proj = en_u @ V_k.T

        # Residual
        zh_resid = zh_u - zh_proj @ V_k
        en_resid = en_u - en_proj @ V_k

        # Normalize and compute z
        zh_proj_u = zh_proj / np.linalg.norm(zh_proj, axis=1, keepdims=True)
        en_proj_u = en_proj / np.linalg.norm(en_proj, axis=1, keepdims=True)
        z_proj = z_score_from_vecs(zh_proj_u, en_proj_u)

        zh_resid_u = zh_resid / np.linalg.norm(zh_resid, axis=1, keepdims=True)
        en_resid_u = en_resid / np.linalg.norm(en_resid, axis=1, keepdims=True)
        z_resid = z_score_from_vecs(zh_resid_u, en_resid_u)

        var_k = pca_full.explained_variance_ratio_[:k].sum()
        var_resid = 1 - var_k

        print(f"{k:8d} {z_proj:8.1f} {z_resid:8.1f} {var_k:10.3f} {var_resid:10.3f}")

    # Cross-validated version: fit PCA on zh only, project en through it
    print(f"\n  Cross-validated (fit zh, test en):")
    pca_zh = PCA(n_components=min(199, zh_u.shape[1]))
    pca_zh.fit(zh_u)

    for k in [5, 10, 20, 50, 100]:
        V_k = pca_zh.components_[:k]
        zh_proj = zh_u @ V_k.T
        en_proj = en_u @ V_k.T

        zh_proj_u = zh_proj / np.linalg.norm(zh_proj, axis=1, keepdims=True)
        en_proj_u = en_proj / np.linalg.norm(en_proj, axis=1, keepdims=True)
        z_proj = z_score_from_vecs(zh_proj_u, en_proj_u)

        zh_resid = zh_u - zh_proj @ V_k
        en_resid = en_u - en_proj @ V_k
        zh_resid_u = zh_resid / np.linalg.norm(zh_resid, axis=1, keepdims=True)
        en_resid_u = en_resid / np.linalg.norm(en_resid, axis=1, keepdims=True)
        z_resid = z_score_from_vecs(zh_resid_u, en_resid_u)

        var_k = pca_zh.explained_variance_ratio_[:k].sum()
        print(f"    k={k:3d}: z_proj={z_proj:6.1f}, z_resid={z_resid:6.1f}, var={var_k:.3f}")
