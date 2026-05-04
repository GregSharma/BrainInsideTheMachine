"""Cross-model alignment comparison on BOS-free last-token data.

All three models (Qwen-3B, Qwen-1.5B, phi-2) on same 200 problems.
Does monotonic z-score accumulation hold universally?
"""

import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA

OUTPUT_DIR = Path("output")

models = {
    "Qwen-3B": {"file": "all_layers_lasttok.npz", "n_layers": 36, "d": 2048},
    "Qwen-1.5B": {"file": "qwen15b_all_layers_lasttok.npz", "n_layers": 28, "d": 1536},
    "phi-2": {"file": "phi2_all_layers_lasttok.npz", "n_layers": 32, "d": 2560},
}

N = 200
results = {}


def matched_vs_scrambled(zh, en, n_perms=500):
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    rng = np.random.RandomState(42)
    scr = [np.mean(np.sum(zh_u * en_u[rng.permutation(N)], axis=1)) for _ in range(n_perms)]
    scr = np.array(scr)
    return matched, scr.mean(), (matched - scr.mean()) / scr.std()


for model_name, info in models.items():
    print(f"\n{'='*60}")
    print(f"  {model_name} ({info['n_layers']} layers, d={info['d']})")
    print(f"{'='*60}")

    data = np.load(OUTPUT_DIR / info["file"])
    n_layers = info["n_layers"]

    z_scores = []
    gaps = []

    print(f"\n  Layer-by-layer alignment z-scores:")
    for l in range(n_layers):
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]
        matched, scr_mean, z = matched_vs_scrambled(zh, en)
        z_scores.append(float(z))
        gaps.append(float(matched - scr_mean))

        # Compact output
        bar = "█" * int(z / 2)
        print(f"    L{l:2d}: z={z:5.1f} gap={matched - scr_mean:.4f} {bar}")

    results[model_name] = {
        "z_scores": z_scores,
        "gaps": gaps,
        "n_layers": n_layers,
    }

    # Cross-validated PCA test at final layers
    print(f"\n  Cross-validated PCA (fit zh_half1, test half2):")
    for l in [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]

        zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
        en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

        pca = PCA(n_components=20)
        pca.fit(zh_u[:100])

        zh_proj = pca.transform(zh_u[100:])
        en_proj = pca.transform(en_u[100:])

        zh_proj_u = zh_proj / np.linalg.norm(zh_proj, axis=1, keepdims=True)
        en_proj_u = en_proj / np.linalg.norm(en_proj, axis=1, keepdims=True)

        m = np.mean(np.sum(zh_proj_u * en_proj_u, axis=1))
        rng = np.random.RandomState(42)
        s_vals = [np.mean(np.sum(zh_proj_u * en_proj_u[rng.permutation(100)], axis=1)) for _ in range(500)]
        s_vals = np.array(s_vals)
        z_cv = (m - s_vals.mean()) / s_vals.std()

        print(f"    L{l:2d}: z_cv={z_cv:.1f}, var_captured={pca.explained_variance_ratio_.sum():.3f}")

    results[model_name]["pca_cv"] = "see output"

# Cross-model comparison summary
print(f"\n{'='*60}")
print(f"  CROSS-MODEL SUMMARY")
print(f"{'='*60}")

for model_name, info in results.items():
    zs = info["z_scores"]
    print(f"\n  {model_name}:")
    print(f"    z at L0: {zs[0]:.1f}")
    print(f"    z at 25%: {zs[len(zs)//4]:.1f}")
    print(f"    z at 50%: {zs[len(zs)//2]:.1f}")
    print(f"    z at 75%: {zs[3*len(zs)//4]:.1f}")
    print(f"    z at final: {zs[-1]:.1f}")
    print(f"    Max z: {max(zs):.1f} at L{zs.index(max(zs))}")
    print(f"    Monotonic? {all(zs[i+1] >= zs[i] - 2.0 for i in range(len(zs)-1))}")  # Allow small dips

outpath = OUTPUT_DIR / "crossmodel_clean_alignment.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outpath}")
