"""Re-run cross-model alignment at peak-NN layer (L28) using cached activations.
No GPU/torch needed — pure numpy/scipy."""
import json
import numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("output")
SEED = 42
K = 20
N_RANDOM = 50
SECOND_MODEL_NAME = "internlm/internlm2-math-7b"


def generate_problems(n=200, seed=42):
    """Must match cross_model.py exactly for identical category ordering."""
    rng = np.random.RandomState(seed)
    per_cat = n // 5
    problems = []
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        problems.append({"category": 0})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"category": 1})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        problems.append({"category": 2})
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        problems.append({"category": 3})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"category": 4})
    rng.shuffle(problems)
    return problems


def build_contrastive_z(zh_means, en_means, k, var_threshold=0.90):
    N, d = zh_means.shape
    diffs = zh_means - en_means
    diffs_centered = diffs - diffs.mean(axis=0)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    var_explained = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)
    n_lang = max(1, int(np.searchsorted(cumvar, var_threshold) + 1))
    n_lang = min(n_lang, len(S))
    lang_dirs = Vt[:n_lang]
    proj_out = np.eye(d, dtype=np.float32) - lang_dirs.T @ lang_dirs
    all_means = np.concatenate([zh_means, en_means], axis=0)
    projected = all_means @ proj_out.T
    projected_centered = projected - projected.mean(axis=0)
    _, _, Vt_proj = np.linalg.svd(projected_centered, full_matrices=False)
    actual_k = min(k, Vt_proj.shape[0])
    return Vt_proj[:actual_k].astype(np.float32), n_lang, lang_dirs.astype(np.float32)


def generate_random_basis(d, k, rng):
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].T


def apply_procrustes(ref_proj, other_proj):
    ref_c = ref_proj - ref_proj.mean(axis=0)
    other_c = other_proj - other_proj.mean(axis=0)
    R, _ = orthogonal_procrustes(other_c, ref_c)
    return other_c @ R + ref_proj.mean(axis=0)


def within_cat_nn_accuracy(proj_a, proj_b, categories):
    N = len(categories)
    correct = 0
    for i in range(N):
        cat = categories[i]
        cat_mask = categories == cat
        cat_indices = np.where(cat_mask)[0]
        dists = np.linalg.norm(proj_a[cat_mask] - proj_b[i], axis=1)
        nearest_cat_idx = cat_indices[np.argmin(dists)]
        if nearest_cat_idx == i:
            correct += 1
    return correct / N


def main():
    rng = np.random.default_rng(SEED)
    problems = generate_problems(200, seed=SEED)
    categories = np.array([p["category"] for p in problems])
    N = len(problems)

    print("=" * 70)
    print(f"REALIGN: Qwen2.5-3B vs {SECOND_MODEL_NAME} at PEAK-NN layer")
    print("=" * 70)

    # Load Qwen Z (L32)
    print("\n--- Loading Qwen L32 ---")
    qwen_data = np.load(OUTPUT_DIR / "viz_activations.npz")
    qwen_zh = qwen_data["zh_L32"]
    qwen_en = qwen_data["en_L32"]
    qwen_d = qwen_zh.shape[1]
    print(f"  Qwen L32: d={qwen_d}, N={qwen_zh.shape[0]}")

    qwen_z_basis, qwen_n_lang, _ = build_contrastive_z(qwen_zh, qwen_en, K)
    qwen_zh_proj = qwen_zh @ qwen_z_basis.T
    qwen_en_proj_raw = qwen_en @ qwen_z_basis.T
    qwen_en_proj = apply_procrustes(qwen_zh_proj, qwen_en_proj_raw)
    print(f"  Qwen Z: k={K}, language dims={qwen_n_lang}")

    # Load InternLM2 activations
    print(f"\n--- Loading {SECOND_MODEL_NAME} cached activations ---")
    mdata = np.load(OUTPUT_DIR / "internlm2_math_activations.npz")

    # Layer sweep
    sweep_layers = sorted([int(k.split("_L")[1]) for k in mdata.files if k.startswith("zh_L")])
    mistral_d = mdata[f"zh_L{sweep_layers[0]}"].shape[1]
    print(f"  Available layers: {sweep_layers}, d={mistral_d}")

    layer_metrics = {}
    for l in sweep_layers:
        zh_m = mdata[f"zh_L{l}"]
        en_m = mdata[f"en_L{l}"]
        z_basis_l, n_lang_l, _ = build_contrastive_z(zh_m, en_m, K)
        zh_proj = zh_m @ z_basis_l.T
        en_proj_raw = en_m @ z_basis_l.T
        en_proj = apply_procrustes(zh_proj, en_proj_raw)
        nn_acc = within_cat_nn_accuracy(zh_proj, en_proj, categories)

        diffs = zh_m - en_m
        diffs_centered = diffs - diffs.mean(axis=0)
        _, S_diff, _ = np.linalg.svd(diffs_centered, full_matrices=False)
        lang_energy = (S_diff[:n_lang_l] ** 2).sum() / (S_diff ** 2).sum()
        energy_ratio = lang_energy / (n_lang_l / mistral_d)

        layer_metrics[l] = {"nn_acc": nn_acc, "n_lang": n_lang_l, "energy_ratio": energy_ratio}
        print(f"  L{l:2d}: NN={nn_acc:.3f}, lang_dims={n_lang_l}, energy_ratio={energy_ratio:.1f}x")

    # Peak-NN layer (exclude L0 — embedding layer is not meaningful for cross-model)
    candidate_layers = [l for l in sweep_layers if l > 0]
    best_layer = max(candidate_layers, key=lambda l: layer_metrics[l]["nn_acc"])
    last_layer = max(sweep_layers)
    print(f"\n  Peak-NN layer: L{best_layer} (NN={layer_metrics[best_layer]['nn_acc']:.3f})")
    print(f"  Last layer:    L{last_layer} (NN={layer_metrics[last_layer]['nn_acc']:.3f})")

    # Build Z at peak layer
    zh_best = mdata[f"zh_L{best_layer}"]
    en_best = mdata[f"en_L{best_layer}"]
    mistral_z_basis, mistral_n_lang, _ = build_contrastive_z(zh_best, en_best, K)
    mistral_zh_proj = zh_best @ mistral_z_basis.T
    mistral_en_proj_raw = en_best @ mistral_z_basis.T
    mistral_en_proj = apply_procrustes(mistral_zh_proj, mistral_en_proj_raw)
    print(f"  Model-B Z at L{best_layer}: k={K}, language dims={mistral_n_lang}")

    # Cross-model alignment
    print("\n--- Cross-model Procrustes alignment ---")
    train_idx, test_idx = [], []
    for cat in range(5):
        cat_indices = np.where(categories == cat)[0]
        np.random.default_rng(SEED).shuffle(cat_indices)
        half = len(cat_indices) // 2
        train_idx.extend(cat_indices[:half].tolist())
        test_idx.extend(cat_indices[half:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    qwen_train = qwen_zh_proj[train_idx]
    mistral_train = mistral_zh_proj[train_idx]
    qwen_c = qwen_train - qwen_train.mean(axis=0)
    mistral_c = mistral_train - mistral_train.mean(axis=0)
    R_cross, _ = orthogonal_procrustes(qwen_c, mistral_c)

    qwen_test_aligned = (qwen_zh_proj[test_idx] - qwen_train.mean(axis=0)) @ R_cross + mistral_train.mean(axis=0)
    mistral_test = mistral_zh_proj[test_idx]

    ss_res = np.sum((qwen_test_aligned - mistral_test) ** 2)
    ss_tot = np.sum((mistral_test - mistral_test.mean(axis=0)) ** 2)
    r2_cross = 1 - ss_res / ss_tot
    print(f"  Cross-model R² (Qwen→Model-B, held-out): {r2_cross:.4f}")

    # Random baseline
    random_r2s = []
    for _ in range(N_RANDOM):
        perm = rng.permutation(len(train_idx))
        qwen_perm = qwen_zh_proj[train_idx[perm]]
        qc = qwen_perm - qwen_perm.mean(axis=0)
        mc = mistral_c.copy()
        R_rand, _ = orthogonal_procrustes(qc, mc)
        aligned = (qwen_zh_proj[test_idx[perm]] - qwen_perm.mean(axis=0)) @ R_rand + mistral_train.mean(axis=0)
        ss_res_r = np.sum((aligned - mistral_test) ** 2)
        random_r2s.append(1 - ss_res_r / ss_tot)
    rand_mean = np.mean(random_r2s)
    rand_std = np.std(random_r2s)
    print(f"  Random baseline: {rand_mean:.4f} ± {rand_std:.4f}")

    # Cross-model NN
    qwen_all_aligned = (qwen_zh_proj - qwen_train.mean(axis=0)) @ R_cross + mistral_train.mean(axis=0)
    cross_nn = within_cat_nn_accuracy(qwen_all_aligned, mistral_zh_proj, categories)
    print(f"  Cross-model NN matching (Qwen→Model-B): {cross_nn:.3f}")

    # Random NN baseline
    rand_nns = []
    for _ in range(N_RANDOM):
        rand_basis_q = generate_random_basis(qwen_d, K, rng)
        rand_basis_m = generate_random_basis(mistral_d, K, rng)
        q_rand = qwen_zh @ rand_basis_q.T
        m_rand = zh_best @ rand_basis_m.T
        q_aligned = apply_procrustes(m_rand, q_rand)
        rand_nns.append(within_cat_nn_accuracy(q_aligned, m_rand, categories))
    rand_nn_mean = np.mean(rand_nns)
    print(f"  Random NN baseline: {rand_nn_mean:.3f} (chance={1/40:.3f})")

    # Also run at last layer for comparison
    print(f"\n--- Comparison: same analysis at L{last_layer} (last layer) ---")
    zh_last = mdata[f"zh_L{last_layer}"]
    en_last = mdata[f"en_L{last_layer}"]
    last_z_basis, last_n_lang, _ = build_contrastive_z(zh_last, en_last, K)
    last_zh_proj = zh_last @ last_z_basis.T

    qwen_train_l = qwen_zh_proj[train_idx]
    last_train = last_zh_proj[train_idx]
    qc_l = qwen_train_l - qwen_train_l.mean(axis=0)
    lc = last_train - last_train.mean(axis=0)
    R_last, _ = orthogonal_procrustes(qc_l, lc)
    qwen_test_last = (qwen_zh_proj[test_idx] - qwen_train_l.mean(axis=0)) @ R_last + last_train.mean(axis=0)
    last_test = last_zh_proj[test_idx]
    ss_res_l = np.sum((qwen_test_last - last_test) ** 2)
    ss_tot_l = np.sum((last_test - last_test.mean(axis=0)) ** 2)
    r2_last = 1 - ss_res_l / ss_tot_l

    qwen_all_last = (qwen_zh_proj - qwen_train_l.mean(axis=0)) @ R_last + last_train.mean(axis=0)
    nn_last = within_cat_nn_accuracy(qwen_all_last, last_zh_proj, categories)
    print(f"  L{last_layer} R²={r2_last:.4f}, NN={nn_last:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("LAYER COMPARISON")
    print(f"  L{best_layer} (peak-NN): R²={r2_cross:.4f}, cross-NN={cross_nn:.3f}")
    print(f"  L{last_layer} (last):    R²={r2_last:.4f}, cross-NN={nn_last:.3f}")
    improvement = r2_cross - r2_last
    print(f"  R² improvement: {improvement:+.4f}")
    print("=" * 70)

    # Figure
    print("\n--- Generating figure ---")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: t-SNE overlay
    combined = np.concatenate([qwen_all_aligned, mistral_zh_proj], axis=0)
    perp = min(30, combined.shape[0] // 2 - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED,
                max_iter=1000, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(combined)
    ax1.scatter(coords[:N, 0], coords[:N, 1], c="#e74c3c", s=20, alpha=0.5,
                edgecolors="darkred", linewidths=0.2, label="Qwen2.5-3B (L32)")
    ax1.scatter(coords[N:, 0], coords[N:, 1], c="#3498db", s=20, alpha=0.5,
                edgecolors="darkblue", linewidths=0.2, label=f"InternLM2-Math (L{best_layer})")
    for i in range(N):
        ax1.plot([coords[i, 0], coords[N+i, 0]], [coords[i, 1], coords[N+i, 1]],
                 color="gray", alpha=0.05, linewidth=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title(f"Cross-Model t-SNE\n(Qwen L32 aligned to InternLM2 L{best_layer})", fontsize=11, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel B: Layer sweep
    layers_sorted = sorted(layer_metrics.keys())
    nn_accs = [layer_metrics[l]["nn_acc"] for l in layers_sorted]
    energy_ratios = [layer_metrics[l]["energy_ratio"] for l in layers_sorted]

    ax2.plot(layers_sorted, energy_ratios, "o-", color="#e74c3c", linewidth=2, markersize=6, label="Energy ratio")
    ax2.fill_between(layers_sorted, 1, energy_ratios, alpha=0.15, color="#e74c3c")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Energy Ratio (×chance)", fontsize=11, color="#e74c3c")
    ax2.set_title("InternLM2-Math Layer Sweep", fontsize=11, fontweight="bold")
    ax2b = ax2.twinx()
    ax2b.plot(layers_sorted, nn_accs, "s--", color="#3498db", linewidth=1.5, markersize=5, label="NN acc")
    ax2b.set_ylabel("zh↔en NN Accuracy", fontsize=11, color="#3498db")
    ax2b.set_ylim(0, 1)
    ax2b.axvline(x=best_layer, color="#3498db", linestyle=":", alpha=0.5, label=f"Peak L{best_layer}")

    # Panel C: R² comparison (peak vs last vs random)
    bars = ax3.bar(
        [f"L{best_layer}\n(peak-NN)", f"L{last_layer}\n(last)", "Random\nBaseline"],
        [r2_cross, r2_last, rand_mean],
        color=["#27ae60", "#e74c3c", "#95a5a6"],
        edgecolor="black", linewidth=0.8
    )
    ax3.errorbar(2, rand_mean, yerr=rand_std, fmt="none", color="black", capsize=5, linewidth=2)
    ax3.set_ylabel("R² (held-out)", fontsize=11)
    ax3.set_title("Cross-Model Alignment\npeak-NN vs last layer", fontsize=11, fontweight="bold")
    ax3.text(0, r2_cross + 0.02, f"{r2_cross:.3f}", ha="center", fontsize=13,
             fontweight="bold", color="#1e8449")
    ax3.text(1, r2_last + 0.02, f"{r2_last:.3f}", ha="center", fontsize=13,
             fontweight="bold", color="#c0392b")
    ax3.text(2, rand_mean + rand_std + 0.03, f"{rand_mean:.3f}±{rand_std:.3f}",
             ha="center", fontsize=10, color="#7f8c8d")

    fig.suptitle(
        f"Universal Z: Qwen2.5-3B (L32) vs InternLM2-Math-7B\n"
        f"L{best_layer} R²={r2_cross:.3f}, NN={cross_nn:.3f} | L{last_layer} R²={r2_last:.3f}, NN={nn_last:.3f}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_cross_model_L28.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'fig7_cross_model_L28.png'}")
    plt.close(fig)

    # Save results
    results = {
        "peak_layer": int(best_layer),
        "last_layer": int(last_layer),
        "peak_layer_r2": float(r2_cross),
        "peak_layer_nn": float(cross_nn),
        "last_layer_r2": float(r2_last),
        "last_layer_nn": float(nn_last),
        "random_r2_mean": float(rand_mean),
        "random_r2_std": float(rand_std),
        "random_nn_mean": float(rand_nn_mean),
        "r2_improvement": float(improvement),
        "layer_metrics": {
            str(l): {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                     for k, v in m.items()}
            for l, m in layer_metrics.items()
        },
    }
    with open(OUTPUT_DIR / "cross_model_results_L28.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'cross_model_results_L28.json'}")

    print("\n" + "=" * 70)
    print("VERDICT")
    if r2_cross > 0.8 and cross_nn > rand_nn_mean * 3:
        print("  CROSS-LAB UNIVERSALITY CONFIRMED")
    elif r2_cross > 0.5 and cross_nn > rand_nn_mean * 2:
        print("  STRONG PARTIAL ALIGNMENT — shared structure confirmed")
    elif r2_cross > 0.3:
        print("  PARTIAL ALIGNMENT — geometry partially shared")
    else:
        print("  NO CROSS-MODEL ALIGNMENT")
    print(f"  InternLM2-Math at L{best_layer}: R²={r2_cross:.4f}, NN={cross_nn:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
