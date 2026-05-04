"""Experiment 2: R² trajectory across layers.

For each layer L (all 36):
1. Load last-token activations for 200 problems, zh + en
2. Fit PC0 at layer L (PCA on unit-normalized combined)
3. For each zh vector, do PC0 swap toward English mean projection
4. Compute R² between swapped-zh and actual-en at that layer
5. Also: norm of PC0 component relative to total vector norm
6. Also: residual language classification after PC0 removal

Answers: after removing the dominant language direction, how much
language-specific content remains? Where does it converge?
"""

import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

OUTPUT_DIR = Path("output")

def main():
    data = np.load("output/all_layers_lasttok.npz")

    # Detect available layers
    layers = sorted(set(int(k.split("_L")[1]) for k in data.keys() if k.startswith("zh_L")))
    n_problems = data["zh_L0"].shape[0]
    d = data["zh_L0"].shape[1]
    print(f"Layers: {len(layers)}, problems: {n_problems}, d: {d}")

    results = {
        "n_layers": len(layers),
        "n_problems": n_problems,
        "hidden_size": d,
        "per_layer": []
    }

    for L in layers:
        zh = data[f"zh_L{L}"].astype(np.float64)  # (200, 2048)
        en = data[f"en_L{L}"].astype(np.float64)

        # Norms
        zh_norms = np.linalg.norm(zh, axis=1, keepdims=True)
        en_norms = np.linalg.norm(en, axis=1, keepdims=True)

        zh_unit = zh / zh_norms
        en_unit = en / en_norms

        # Fit PCA on combined unit-normalized
        combined = np.vstack([zh_unit, en_unit])
        pca = PCA(n_components=10)
        pca.fit(combined)
        pc0 = pca.components_[0]

        # Cohen's d for PC0
        zh_proj = zh_unit @ pc0
        en_proj = en_unit @ pc0
        cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt(
            (zh_proj.std()**2 + en_proj.std()**2) / 2
        )

        # Mean projections
        zh_mean_proj = zh_proj.mean()
        en_mean_proj = en_proj.mean()

        # === PC0 SWAP: zh → en ===
        # For each zh vector: remove zh's PC0 projection, add en's mean projection, rescale
        zh_pc0_proj = (zh_unit @ pc0).reshape(-1, 1)  # (200, 1)
        zh_swapped_unit = zh_unit - zh_pc0_proj * pc0 + en_mean_proj * pc0  # swap PC0
        zh_swapped = zh_swapped_unit * zh_norms  # rescale to original norms

        # === R² between swapped-zh and actual-en ===
        # Per-problem cosine similarity
        cos_swapped_en = np.array([
            np.dot(zh_swapped[i], en[i]) / (np.linalg.norm(zh_swapped[i]) * np.linalg.norm(en[i]))
            for i in range(n_problems)
        ])

        # R² = 1 - ||swapped_zh - en||² / ||en - mean(en)||²
        # Use centered R² (variance explained)
        en_mean = en.mean(axis=0)
        ss_res = np.sum((zh_swapped - en)**2)
        ss_tot = np.sum((en - en_mean)**2)
        r2_raw = 1.0 - ss_res / ss_tot

        # Also per-problem R² (fraction of variance in en explained by swapped-zh)
        # Using correlation-based R²
        from scipy.stats import pearsonr
        # Flatten and correlate
        r2_flat = pearsonr(zh_swapped.ravel(), en.ravel())[0]**2

        # === Residual language content after PC0 removal ===
        # Remove PC0 from both zh and en, then classify
        zh_residual = zh_unit - (zh_unit @ pc0).reshape(-1, 1) * pc0
        en_residual = en_unit - (en_unit @ pc0).reshape(-1, 1) * pc0

        X_residual = np.vstack([zh_residual, en_residual])
        y_residual = np.array([0]*n_problems + [1]*n_problems)

        # Logistic regression on residual (how much language info remains?)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_residual, y_residual)
        residual_acc = clf.score(X_residual, y_residual)

        # Also try with more PCs removed
        for n_remove in [5, 10, 20]:
            pcs_to_remove = pca.components_[:n_remove]
            zh_res_n = zh_unit.copy()
            en_res_n = en_unit.copy()
            for pc in pcs_to_remove:
                zh_res_n -= (zh_res_n @ pc).reshape(-1, 1) * pc
                en_res_n -= (en_res_n @ pc).reshape(-1, 1) * pc
            X_n = np.vstack([zh_res_n, en_res_n])
            clf_n = LogisticRegression(max_iter=1000, C=1.0)
            clf_n.fit(X_n, y_residual)

        # === PC0 component magnitude ===
        # Absolute PC0 projection (in original scale, not unit-normalized)
        zh_pc0_abs = np.abs(zh @ pc0)  # in raw activation scale
        en_pc0_abs = np.abs(en @ pc0)
        zh_pc0_frac = zh_pc0_abs / zh_norms.squeeze()
        en_pc0_frac = en_pc0_abs / en_norms.squeeze()

        # PC0 gap in raw scale
        pc0_gap_raw = np.abs((zh @ pc0).mean() - (en @ pc0).mean())
        mean_norm = (zh_norms.mean() + en_norms.mean()) / 2
        pc0_gap_frac = pc0_gap_raw / mean_norm

        layer_result = {
            "layer": L,
            "pc0_var_explained": float(pca.explained_variance_ratio_[0]),
            "cohens_d": float(cohens_d),
            "zh_mean_proj": float(zh_mean_proj),
            "en_mean_proj": float(en_mean_proj),
            # R² metrics
            "r2_raw": float(r2_raw),
            "r2_flat_corr": float(r2_flat),
            "mean_cos_swapped_en": float(cos_swapped_en.mean()),
            "std_cos_swapped_en": float(cos_swapped_en.std()),
            "min_cos_swapped_en": float(cos_swapped_en.min()),
            # Residual language content
            "residual_lang_acc_remove_pc0": float(residual_acc),
            "residual_lang_acc_remove_pc0_5": float(clf_n.score(X_n, y_residual)) if n_remove == 20 else None,
            # Norms
            "mean_zh_norm": float(zh_norms.mean()),
            "mean_en_norm": float(en_norms.mean()),
            "pc0_gap_raw": float(pc0_gap_raw),
            "pc0_gap_frac_of_norm": float(pc0_gap_frac),
            "mean_zh_pc0_frac": float(zh_pc0_frac.mean()),
            "mean_en_pc0_frac": float(en_pc0_frac.mean()),
        }

        results["per_layer"].append(layer_result)

        print(f"L{L:2d}: R²={r2_raw:.4f}  cos={cos_swapped_en.mean():.4f}  "
              f"residual_acc={residual_acc:.3f}  pc0_var={pca.explained_variance_ratio_[0]:.3f}  "
              f"d={cohens_d:+.1f}  gap_frac={pc0_gap_frac:.4f}  norm={mean_norm:.1f}")

    # Now redo residual accuracy properly for all removal levels
    print("\n\nRESIDUAL LANGUAGE ACCURACY (how much language info survives PC removal)")
    print(f"{'Layer':>5s}  {'rm PC0':>8s}  {'rm PC0-4':>8s}  {'rm PC0-9':>8s}  {'rm PC0-19':>9s}")
    print("-" * 50)

    for L in layers:
        zh = data[f"zh_L{L}"].astype(np.float64)
        en = data[f"en_L{L}"].astype(np.float64)
        zh_norms = np.linalg.norm(zh, axis=1, keepdims=True)
        en_norms = np.linalg.norm(en, axis=1, keepdims=True)
        zh_unit = zh / zh_norms
        en_unit = en / en_norms

        combined = np.vstack([zh_unit, en_unit])
        pca = PCA(n_components=20)
        pca.fit(combined)

        y = np.array([0]*n_problems + [1]*n_problems)

        accs = []
        for n_remove in [1, 5, 10, 20]:
            pcs = pca.components_[:n_remove]
            zh_r = zh_unit.copy()
            en_r = en_unit.copy()
            for pc in pcs:
                zh_r -= (zh_r @ pc).reshape(-1, 1) * pc
                en_r -= (en_r @ pc).reshape(-1, 1) * pc
            X = np.vstack([zh_r, en_r])
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X, y)
            accs.append(clf.score(X, y))

        print(f"  L{L:<3d}  {accs[0]:>7.3f}  {accs[1]:>7.3f}  {accs[2]:>7.3f}  {accs[3]:>8.3f}")

        # Store in results
        for r in results["per_layer"]:
            if r["layer"] == L:
                r["residual_lang_acc_remove_pc0"] = accs[0]
                r["residual_lang_acc_remove_5"] = accs[1]
                r["residual_lang_acc_remove_10"] = accs[2]
                r["residual_lang_acc_remove_20"] = accs[3]

    # Summary table
    print("\n\n" + "="*100)
    print("EXPERIMENT 2 SUMMARY: R² TRAJECTORY")
    print("="*100)
    print(f"{'Layer':>5s}  {'R²':>7s}  {'cos':>7s}  {'res_acc':>8s}  {'PC0var':>7s}  {'|d|':>6s}  {'gap%':>7s}  {'norm':>7s}")
    print("-"*70)
    for r in results["per_layer"]:
        print(f"  L{r['layer']:<3d}  {r['r2_raw']:>6.4f}  {r['mean_cos_swapped_en']:>6.4f}  "
              f"{r['residual_lang_acc_remove_pc0']:>7.3f}  {r['pc0_var_explained']:>6.3f}  "
              f"{abs(r['cohens_d']):>5.1f}  {r['pc0_gap_frac_of_norm']:>6.4f}  {r['mean_zh_norm']:.1f}")

    # Save
    outpath = OUTPUT_DIR / "exp2_r2_trajectory.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
