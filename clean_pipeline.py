"""Full analysis pipeline on BOS-free last-token data.

Tests what survives from the original findings:
1. Three-phase structure (Procrustes R² across layers)
2. Backbone lifecycle (dim 318 dominance across layers)
3. Cross-lingual alignment (matched vs scrambled, unit-normalized)
4. PCA subspace (does a low-rank Z exist in clean data?)
5. Cocycle consistency (Ridge regression across languages)
"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

OUTPUT_DIR = Path("output")


def orthogonal_procrustes_r2(X, Y):
    """Orthogonal Procrustes R² between X and Y (both N×d)."""
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    R, _ = orthogonal_procrustes(X_c, Y_c)
    Y_pred = X_c @ R
    ss_res = np.sum((Y_c - Y_pred) ** 2)
    ss_tot = np.sum(Y_c ** 2)
    return 1 - ss_res / ss_tot


def matched_vs_scrambled(zh, en, n_perms=1000):
    """Unit-normalized matched vs scrambled z-score."""
    # Unit normalize
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

    matched_cos = np.mean(np.sum(zh_u * en_u, axis=1))

    scrambled = []
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        perm = rng.permutation(len(en_u))
        scrambled.append(np.mean(np.sum(zh_u * en_u[perm], axis=1)))
    scrambled = np.array(scrambled)

    z = (matched_cos - scrambled.mean()) / scrambled.std()
    return matched_cos, scrambled.mean(), z


def main():
    print("Loading BOS-free last-token data...")
    data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")

    n_layers = 36
    d = 2048
    N = 200

    results = {
        "data_source": "all_layers_lasttok.npz (last token, no BOS contamination)",
        "n_problems": N,
        "n_layers": n_layers,
        "hidden_size": d,
    }

    # ========== 1. THREE-PHASE STRUCTURE ==========
    print("\n=== 1. Three-Phase Structure (Procrustes R² across layers) ===")
    procrustes_r2 = []
    for l in range(n_layers):
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]
        r2 = orthogonal_procrustes_r2(zh, en)
        procrustes_r2.append(float(r2))
        print(f"  L{l:2d}: R²={r2:.4f}")

    results["procrustes_r2_by_layer"] = procrustes_r2

    # Find phases
    max_drop = 0
    drop_layer = -1
    for i in range(1, n_layers):
        drop = procrustes_r2[i-1] - procrustes_r2[i]
        if drop > max_drop:
            max_drop = drop
            drop_layer = i

    max_jump = 0
    jump_layer = -1
    for i in range(1, n_layers):
        jump = procrustes_r2[i] - procrustes_r2[i-1]
        if jump > max_jump:
            max_jump = jump
            jump_layer = i

    print(f"\n  Biggest drop: L{drop_layer-1}→L{drop_layer} (Δ={-max_drop:.4f})")
    print(f"  Biggest jump: L{jump_layer-1}→L{jump_layer} (Δ={+max_jump:.4f})")

    results["phase_transitions"] = {
        "biggest_drop": {"layer": drop_layer, "delta": float(-max_drop)},
        "biggest_jump": {"layer": jump_layer, "delta": float(max_jump)},
    }

    # ========== 2. BACKBONE LIFECYCLE ==========
    print("\n=== 2. Backbone Lifecycle (dim 318 dominance) ===")
    dim318_stats = []
    for l in range(n_layers):
        combined = np.vstack([data[f"zh_L{l}"], data[f"en_L{l}"]])
        d318_var = np.var(combined[:, 318])
        total_var = np.sum(np.var(combined, axis=0))
        frac = d318_var / total_var
        mean_norm = np.linalg.norm(combined, axis=1).mean()
        d318_mean = np.mean(np.abs(combined[:, 318]))
        dim318_stats.append({
            "layer": l,
            "var_fraction": float(frac),
            "mean_abs": float(d318_mean),
            "mean_norm": float(mean_norm),
            "ratio": float(d318_mean / mean_norm),
        })
        if frac > 0.01:
            print(f"  L{l:2d}: var_frac={frac:.4f}, |dim318|/norm={d318_mean/mean_norm:.4f}")

    results["dim318_lifecycle"] = dim318_stats

    # Find the NEW dominant dims
    print("\n  Top 5 variance dimensions at key layers:")
    for l in [0, 1, 2, 15, 29, 30, 35]:
        combined = np.vstack([data[f"zh_L{l}"], data[f"en_L{l}"]])
        variances = np.var(combined, axis=0)
        top5 = np.argsort(variances)[::-1][:5]
        total = variances.sum()
        print(f"  L{l:2d}: {[(int(d), f'{variances[d]/total:.3f}') for d in top5]}")

    # ========== 3. CROSS-LINGUAL ALIGNMENT ==========
    print("\n=== 3. Cross-Lingual Alignment (matched vs scrambled, unit-norm) ===")
    alignment_by_layer = []
    for l in range(n_layers):
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]
        matched_cos, scrambled_mean, z = matched_vs_scrambled(zh, en)
        alignment_by_layer.append({
            "layer": l,
            "matched_cos": float(matched_cos),
            "scrambled_mean": float(scrambled_mean),
            "z_score": float(z),
        })
        print(f"  L{l:2d}: matched={matched_cos:.4f}, scrambled={scrambled_mean:.4f}, z={z:.1f}")

    results["alignment_by_layer"] = alignment_by_layer

    # ========== 4. PCA SUBSPACE ==========
    print("\n=== 4. PCA Subspace (does low-rank Z exist in clean data?) ===")
    # Use L32 (pre-final, where gen-time data comes from)
    for l in [15, 32, 34]:
        zh = data[f"zh_L{l}"]
        en = data[f"en_L{l}"]
        combined = np.vstack([zh, en])

        # Unit normalize first
        combined_u = combined / np.linalg.norm(combined, axis=1, keepdims=True)

        pca = PCA(n_components=20)
        pca.fit(combined_u)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        # Project and test alignment
        zh_proj = pca.transform(zh / np.linalg.norm(zh, axis=1, keepdims=True))
        en_proj = pca.transform(en / np.linalg.norm(en, axis=1, keepdims=True))

        # Matched vs scrambled in subspace
        zh_proj_u = zh_proj / np.linalg.norm(zh_proj, axis=1, keepdims=True)
        en_proj_u = en_proj / np.linalg.norm(en_proj, axis=1, keepdims=True)

        matched = np.mean(np.sum(zh_proj_u * en_proj_u, axis=1))
        scrambled_vals = []
        rng = np.random.RandomState(42)
        for _ in range(1000):
            perm = rng.permutation(N)
            scrambled_vals.append(np.mean(np.sum(zh_proj_u * en_proj_u[perm], axis=1)))
        scrambled_vals = np.array(scrambled_vals)
        z = (matched - scrambled_vals.mean()) / scrambled_vals.std()

        print(f"\n  L{l}: cumulative variance in 20 PCs = {cumvar[-1]:.3f}")
        print(f"  L{l}: PC0 var = {pca.explained_variance_ratio_[0]:.3f}")
        print(f"  L{l}: Z-subspace matched vs scrambled: z={z:.1f}")

        # Cross-validated version: fit on zh, test alignment on en
        pca_zh = PCA(n_components=20)
        pca_zh.fit(zh / np.linalg.norm(zh, axis=1, keepdims=True))

        # Split-half
        zh_half1 = zh[:100]
        zh_half2 = zh[100:]
        en_half1 = en[:100]
        en_half2 = en[100:]

        pca_half = PCA(n_components=20)
        pca_half.fit(zh_half1 / np.linalg.norm(zh_half1, axis=1, keepdims=True))

        zh2_proj = pca_half.transform(zh_half2 / np.linalg.norm(zh_half2, axis=1, keepdims=True))
        en2_proj = pca_half.transform(en_half2 / np.linalg.norm(en_half2, axis=1, keepdims=True))

        zh2_u = zh2_proj / np.linalg.norm(zh2_proj, axis=1, keepdims=True)
        en2_u = en2_proj / np.linalg.norm(en2_proj, axis=1, keepdims=True)

        m = np.mean(np.sum(zh2_u * en2_u, axis=1))
        s_vals = []
        rng2 = np.random.RandomState(42)
        for _ in range(1000):
            perm = rng2.permutation(100)
            s_vals.append(np.mean(np.sum(zh2_u * en2_u[perm], axis=1)))
        s_vals = np.array(s_vals)
        z_cv = (m - s_vals.mean()) / s_vals.std()

        print(f"  L{l}: CROSS-VALIDATED (fit zh_half1, test half2): z={z_cv:.1f}")

    results["pca_subspace"] = "see console output"

    # ========== 5. COCYCLE on clean data ==========
    print("\n=== 5. Cocycle Consistency (Ridge, L32 only — now BOS-free) ===")
    # We only have zh and en in this extraction, so we do zh→en, en→zh round-trip
    zh_32 = data["zh_L32"]
    en_32 = data["en_L32"]

    # Unit normalize
    zh_u = zh_32 / np.linalg.norm(zh_32, axis=1, keepdims=True)
    en_u = en_32 / np.linalg.norm(en_32, axis=1, keepdims=True)

    # Split: train on first 150, test on last 50
    zh_train, zh_test = zh_u[:150], zh_u[150:]
    en_train, en_test = en_u[:150], en_u[150:]

    # zh→en
    ridge_ze = Ridge(alpha=1.0).fit(zh_train, en_train)
    en_pred = ridge_ze.predict(zh_test)
    # en→zh
    ridge_ez = Ridge(alpha=1.0).fit(en_train, zh_train)
    zh_pred = ridge_ez.predict(en_test)

    # Round trip: zh→en→zh
    en_pred_full = ridge_ze.predict(zh_test)
    zh_roundtrip = ridge_ez.predict(en_pred_full)

    # Cocycle error
    roundtrip_cos = np.mean(np.sum(zh_test * zh_roundtrip / np.linalg.norm(zh_roundtrip, axis=1, keepdims=True), axis=1))

    # Direct alignment accuracy
    en_pred_u = en_pred / np.linalg.norm(en_pred, axis=1, keepdims=True)
    direct_cos = np.mean(np.sum(en_test * en_pred_u, axis=1))

    print(f"  zh→en direct cosine (test): {direct_cos:.4f}")
    print(f"  zh→en→zh roundtrip cosine: {roundtrip_cos:.4f}")
    print(f"  Cocycle error: {1 - roundtrip_cos:.4f}")

    # Scrambled control
    rng = np.random.RandomState(42)
    perm = rng.permutation(150)
    ridge_scr = Ridge(alpha=1.0).fit(zh_train, en_train[perm])
    en_scr_pred = ridge_scr.predict(zh_test)
    en_scr_u = en_scr_pred / np.linalg.norm(en_scr_pred, axis=1, keepdims=True)
    scr_cos = np.mean(np.sum(en_test * en_scr_u, axis=1))
    print(f"  Scrambled zh→en cosine: {scr_cos:.4f}")
    print(f"  Gap (matched - scrambled): {direct_cos - scr_cos:.4f}")

    results["cocycle_clean"] = {
        "direct_cos": float(direct_cos),
        "roundtrip_cos": float(roundtrip_cos),
        "cocycle_error": float(1 - roundtrip_cos),
        "scrambled_cos": float(scr_cos),
        "gap": float(direct_cos - scr_cos),
    }

    # ========== 6. COMPARISON: old (BOS) vs new (clean) ==========
    print("\n=== 6. Comparison with BOS-contaminated data ===")
    old_data = np.load(OUTPUT_DIR / "all_layers.npz")

    for l in [0, 1, 15, 32, 35]:
        old_zh = old_data[f"zh_L{l}"]
        old_en = old_data[f"en_L{l}"]
        new_zh = data[f"zh_L{l}"]
        new_en = data[f"en_L{l}"]

        old_r2 = orthogonal_procrustes_r2(old_zh, old_en)
        new_r2 = orthogonal_procrustes_r2(new_zh, new_en)

        _, _, old_z = matched_vs_scrambled(old_zh, old_en)
        _, _, new_z = matched_vs_scrambled(new_zh, new_en)

        print(f"  L{l:2d}: Procrustes R²: {old_r2:.4f}→{new_r2:.4f} | z-score: {old_z:.1f}→{new_z:.1f}")

    results["comparison"] = "see console output"

    # Save
    outpath = OUTPUT_DIR / "clean_pipeline_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {outpath}")


if __name__ == "__main__":
    main()
