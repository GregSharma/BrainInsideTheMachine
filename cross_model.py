"""Cross-Model Universal Z: Does Model-B-8B share the same reasoning geometry as Qwen2.5-3B?

Run with: HF_TOKEN=... MPLBACKEND=Agg .venv/bin/python cross_model.py

If contrastive Z from two independently trained models can be aligned via Procrustes,
the 20-dim reasoning geometry is universal — not model-specific.

Steps:
  1. Load Model-B-8B in 4-bit (double quant, CPU offload)
  2. Same 200 template problems, zh and en
  3. Extract at every 4th layer to find LLaMA's bottleneck
  4. Build LLaMA's contrastive Z at best layer
  5. Procrustes align Qwen Z → LLaMA Z (different d! need shared projection)
  6. Cross-model probe transfer: train on Qwen activations, test on LLaMA
"""

import gc
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.linalg import orthogonal_procrustes
import random as pyrandom

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42
K = 20
N_RANDOM = 50

QWEN_NAME = "Qwen/Qwen2.5-3B"
SECOND_MODEL_NAME = "internlm/internlm2-math-7b"

CAT_NAMES = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]


def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh, en = f"计算 {a} + {b} 的值。", f"Calculate {a} + {b}."
        else:
            zh, en = f"计算 {a} × {b} 的值。", f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "category": 0})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                          "en": f"Find the value of C({n_val}, {k_val}).", "category": 1})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？",
                          "en": f"What is the remainder when {a} is divided by {b}?", "category": 2})
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                          "en": f"A rectangle has length {w} and width {h}. Find its area.", "category": 3})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
                          "en": f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms.",
                          "category": 4})
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


def get_layers(model):
    """Get the list of transformer layers regardless of architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen, Mistral, LLaMA, Phi-3
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2, Model-B
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def extract_activations(model, tokenizer, problems, layer_idx):
    d = model.config.hidden_size
    N = len(problems)
    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h_out.detach().cpu().squeeze(0)
        return hook
    layers = get_layers(model)
    hook_handle = layers[layer_idx].register_forward_hook(make_hook("target"))
    zh_means = np.zeros((N, d), dtype=np.float32)
    en_means = np.zeros((N, d), dtype=np.float32)
    for i, prob in enumerate(tqdm(problems, desc=f"L{layer_idx} zh", leave=False)):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        zh_means[i] = activations["target"].float().numpy().mean(axis=0)
    for i, prob in enumerate(tqdm(problems, desc=f"L{layer_idx} en", leave=False)):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        en_means[i] = activations["target"].float().numpy().mean(axis=0)
    hook_handle.remove()
    return zh_means, en_means


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
    print(f"CROSS-MODEL: Qwen2.5-3B vs {SECOND_MODEL_NAME} — Universal Z?")
    print("=" * 70)

    # ====================================================================
    # Load Qwen Z from cache (already computed)
    # ====================================================================
    print("\n--- Loading Qwen activations from cache ---")
    qwen_cache = OUTPUT_DIR / "viz_activations.npz"
    qwen_data = np.load(qwen_cache)
    qwen_zh = qwen_data["zh_L32"]
    qwen_en = qwen_data["en_L32"]
    qwen_d = qwen_zh.shape[1]
    print(f"  Qwen L32: d={qwen_d}, N={qwen_zh.shape[0]}")

    qwen_z_basis, qwen_n_lang, _ = build_contrastive_z(qwen_zh, qwen_en, K)
    qwen_zh_proj = qwen_zh @ qwen_z_basis.T
    qwen_en_proj_raw = qwen_en @ qwen_z_basis.T
    qwen_en_proj = apply_procrustes(qwen_zh_proj, qwen_en_proj_raw)
    print(f"  Qwen Z: k={K}, language dims={qwen_n_lang}")

    # ====================================================================
    # Load second model
    # ====================================================================
    print(f"\n--- Loading {SECOND_MODEL_NAME} ---")
    gc.collect()
    torch.cuda.empty_cache()
    mistral_tokenizer = AutoTokenizer.from_pretrained(SECOND_MODEL_NAME, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    from transformers import AutoConfig
    from transformers import DynamicCache
    # Compat shim: InternLM2 custom code uses legacy DynamicCache API removed in newer transformers
    if not hasattr(DynamicCache, "from_legacy_cache"):
        DynamicCache.from_legacy_cache = classmethod(lambda cls, past: DynamicCache() if past is None else past)
    if not hasattr(DynamicCache, "to_legacy_cache"):
        DynamicCache.to_legacy_cache = lambda self: None
    _cfg = AutoConfig.from_pretrained(SECOND_MODEL_NAME, trust_remote_code=True)
    # Fix InternLM2 rope_scaling compat: custom code chokes on transformers' auto-populated
    # rope_scaling dict. "default" means no scaling, so just clear it.
    if _cfg.rope_scaling and _cfg.rope_scaling.get("rope_type") == "default":
        _cfg.rope_scaling = None
    mistral_model = AutoModelForCausalLM.from_pretrained(
        SECOND_MODEL_NAME,
        config=_cfg,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    mistral_model.eval()
    mistral_L = mistral_model.config.num_hidden_layers
    mistral_d = mistral_model.config.hidden_size
    print(f"  {SECOND_MODEL_NAME}: L={mistral_L}, d={mistral_d}")

    # ====================================================================
    # Layer sweep on Model-B
    # ====================================================================
    sweep_layers = list(range(0, mistral_L, 4)) + [mistral_L - 1]
    sweep_layers = sorted(set(sweep_layers))
    print(f"\n--- Model-B layer sweep: {sweep_layers} ---")

    mistral_cache = OUTPUT_DIR / "internlm2_math_activations.npz"
    if mistral_cache.exists():
        print("  Loading cached Model-B activations...")
        mdata = np.load(mistral_cache)
        mistral_layer_data = {}
        for l in sweep_layers:
            key_zh = f"zh_L{l}"
            key_en = f"en_L{l}"
            if key_zh in mdata:
                mistral_layer_data[l] = (mdata[key_zh], mdata[key_en])
        print(f"  Loaded {len(mistral_layer_data)} layers from cache")
        # Extract any missing layers
        missing = [l for l in sweep_layers if l not in mistral_layer_data]
        if missing:
            print(f"  Extracting missing layers: {missing}")
            for l in missing:
                zh_m, en_m = extract_activations(mistral_model, mistral_tokenizer, problems, l)
                mistral_layer_data[l] = (zh_m, en_m)
            # Resave
            save_dict = {}
            for l in mistral_layer_data:
                save_dict[f"zh_L{l}"] = mistral_layer_data[l][0]
                save_dict[f"en_L{l}"] = mistral_layer_data[l][1]
            np.savez(mistral_cache, **save_dict)
    else:
        mistral_layer_data = {}
        for l in sweep_layers:
            print(f"\n  Layer {l}...")
            zh_m, en_m = extract_activations(mistral_model, mistral_tokenizer, problems, l)
            mistral_layer_data[l] = (zh_m, en_m)
        save_dict = {}
        for l in mistral_layer_data:
            save_dict[f"zh_L{l}"] = mistral_layer_data[l][0]
            save_dict[f"en_L{l}"] = mistral_layer_data[l][1]
        np.savez(mistral_cache, **save_dict)
        print(f"\n  Saved to {mistral_cache}")

    # Free GPU memory — model no longer needed
    del mistral_model, mistral_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ====================================================================
    # Find Model-B's bottleneck: contrastive Z at each layer
    # ====================================================================
    print("\n--- Finding Model-B's bottleneck ---")
    layer_metrics = {}
    for l in sweep_layers:
        zh_m, en_m = mistral_layer_data[l]
        z_basis_l, n_lang_l, _ = build_contrastive_z(zh_m, en_m, K)

        # Within-category NN accuracy (pooled zh+en)
        zh_proj = zh_m @ z_basis_l.T
        en_proj_raw = en_m @ z_basis_l.T
        en_proj = apply_procrustes(zh_proj, en_proj_raw)

        nn_acc = within_cat_nn_accuracy(zh_proj, en_proj, categories)

        # Energy concentration
        diffs = zh_m - en_m
        total_var = np.var(diffs)
        diffs_centered = diffs - diffs.mean(axis=0)
        _, S_diff, _ = np.linalg.svd(diffs_centered, full_matrices=False)
        lang_energy = (S_diff[:n_lang_l] ** 2).sum() / (S_diff ** 2).sum()
        energy_ratio = lang_energy / (n_lang_l / mistral_d)

        layer_metrics[l] = {
            "nn_acc": nn_acc,
            "n_lang": n_lang_l,
            "energy_ratio": energy_ratio,
        }
        print(f"  L{l:2d}: NN={nn_acc:.3f}, lang_dims={n_lang_l}, energy_ratio={energy_ratio:.1f}x")

    # Use peak-NN layer — the layer where bilingual integration is strongest
    best_layer = max(sweep_layers, key=lambda l: layer_metrics[l]["nn_acc"])
    print(f"\n  Using peak-NN layer: L{best_layer} (NN={layer_metrics[best_layer]['nn_acc']:.3f}, "
          f"energy_ratio={layer_metrics[best_layer]['energy_ratio']:.1f}x)")

    # ====================================================================
    # Build Model-B Z at best layer
    # ====================================================================
    zh_best, en_best = mistral_layer_data[best_layer]
    mistral_z_basis, mistral_n_lang, _ = build_contrastive_z(zh_best, en_best, K)
    mistral_zh_proj = zh_best @ mistral_z_basis.T
    mistral_en_proj_raw = en_best @ mistral_z_basis.T
    mistral_en_proj = apply_procrustes(mistral_zh_proj, mistral_en_proj_raw)
    print(f"  Model-B Z: k={K}, language dims={mistral_n_lang}, layer={best_layer}")

    # ====================================================================
    # Cross-model alignment: Qwen Z → Model-B Z
    # ====================================================================
    print("\n--- Cross-model Procrustes alignment ---")

    # Both Z projections are k-dimensional. Align Qwen zh-proj onto Model-B zh-proj.
    # Use zh (Chinese) as anchor since it's the reference in both models.

    # Train/test split
    train_idx, test_idx = [], []
    for cat in range(5):
        cat_indices = np.where(categories == cat)[0]
        np.random.default_rng(SEED).shuffle(cat_indices)
        half = len(cat_indices) // 2
        train_idx.extend(cat_indices[:half].tolist())
        test_idx.extend(cat_indices[half:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # Procrustes: align Qwen Z onto Model-B Z using training set
    qwen_train = qwen_zh_proj[train_idx]
    mistral_train = mistral_zh_proj[train_idx]

    qwen_c = qwen_train - qwen_train.mean(axis=0)
    mistral_c = mistral_train - mistral_train.mean(axis=0)
    R_cross, _ = orthogonal_procrustes(qwen_c, mistral_c)

    # Apply to test set
    qwen_test_aligned = (qwen_zh_proj[test_idx] - qwen_train.mean(axis=0)) @ R_cross + mistral_train.mean(axis=0)
    mistral_test = mistral_zh_proj[test_idx]

    # R² on test set
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

    # Cross-model NN matching
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

    # ====================================================================
    # FIGURES
    # ====================================================================
    print("\n--- Generating figures ---")

    # Figure 7: Cross-model alignment scatter
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: Qwen Z and Model-B Z overlaid via t-SNE
    combined = np.concatenate([qwen_all_aligned, mistral_zh_proj], axis=0)
    perp = min(30, combined.shape[0] // 2 - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED,
                max_iter=1000, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(combined)

    ax1.scatter(coords[:N, 0], coords[:N, 1], c="#e74c3c", s=20, alpha=0.5,
                edgecolors="darkred", linewidths=0.2, label="Qwen2.5-3B")
    ax1.scatter(coords[N:, 0], coords[N:, 1], c="#3498db", s=20, alpha=0.5,
                edgecolors="darkblue", linewidths=0.2, label=SECOND_MODEL_NAME)
    for i in range(N):
        ax1.plot([coords[i, 0], coords[N+i, 0]], [coords[i, 1], coords[N+i, 1]],
                 color="gray", alpha=0.05, linewidth=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title("Cross-Model t-SNE\n(Qwen Z aligned to Model-B Z)", fontsize=11, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel B: Model-B layer sweep
    layers_sorted = sorted(layer_metrics.keys())
    energy_ratios = [layer_metrics[l]["energy_ratio"] for l in layers_sorted]
    nn_accs = [layer_metrics[l]["nn_acc"] for l in layers_sorted]

    ax2.plot(layers_sorted, energy_ratios, "o-", color="#e74c3c", linewidth=2, markersize=6, label="Energy ratio")
    ax2.fill_between(layers_sorted, 1, energy_ratios, alpha=0.15, color="#e74c3c")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Energy Ratio (×chance)", fontsize=11, color="#e74c3c")
    ax2.set_title("Model-B Layer Sweep", fontsize=11, fontweight="bold")

    ax2b = ax2.twinx()
    ax2b.plot(layers_sorted, nn_accs, "s--", color="#3498db", linewidth=1.5, markersize=5, label="NN acc")
    ax2b.set_ylabel("zh↔en NN Accuracy", fontsize=11, color="#3498db")
    ax2b.set_ylim(0, 1)

    # Panel C: Cross-model R² comparison
    bars = ax3.bar(["Cross-Model\n(Qwen→Model-B)", "Random\nPermutation"],
                    [r2_cross, rand_mean],
                    color=["#e74c3c", "#95a5a6"], edgecolor="black", linewidth=0.8)
    ax3.errorbar(1, rand_mean, yerr=rand_std, fmt="none", color="black", capsize=5, linewidth=2)
    ax3.set_ylabel("R² (held-out)", fontsize=11)
    ax3.set_title("Cross-Model Alignment\nin 20-dim Z", fontsize=11, fontweight="bold")
    ax3.text(0, r2_cross + 0.02, f"{r2_cross:.3f}", ha="center", fontsize=13,
             fontweight="bold", color="#c0392b")
    ax3.text(1, rand_mean + rand_std + 0.03, f"{rand_mean:.3f}±{rand_std:.3f}",
             ha="center", fontsize=10, color="#7f8c8d")

    fig.suptitle(f"Universal Z: Qwen2.5-3B (L32) vs {SECOND_MODEL_NAME} (L{best_layer})\n"
                 f"Cross-model NN: {cross_nn:.3f} (random: {rand_nn_mean:.3f}, chance: {1/40:.3f})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_cross_model.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'fig7_cross_model.png'}")
    plt.close(fig)

    # Save results
    results = {
        "modelb_bottleneck_layer": int(best_layer),
        "modelb_d": int(mistral_d),
        "modelb_L": int(mistral_L),
        "cross_model_r2": float(r2_cross),
        "random_r2_mean": float(rand_mean),
        "random_r2_std": float(rand_std),
        "cross_nn": float(cross_nn),
        "random_nn": float(rand_nn_mean),
        "chance_nn": float(1/40),
        "layer_metrics": {
            str(l): {k: (float(v) if np.isfinite(v) else str(v)) for k, v in metrics.items()}
            for l, metrics in layer_metrics.items()
        },
    }
    with open(OUTPUT_DIR / "cross_model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("CROSS-MODEL TEST COMPLETE")
    print(f"  Model-B bottleneck: L{best_layer} (d={mistral_d})")
    print(f"  Cross-model R²: {r2_cross:.4f} (random: {rand_mean:.4f})")
    print(f"  Cross-model NN: {cross_nn:.3f} (random: {rand_nn_mean:.3f}, chance: {1/40:.3f})")
    if r2_cross > 0.5 and cross_nn > rand_nn_mean * 2:
        print("  Verdict: UNIVERSAL Z CONFIRMED")
    elif r2_cross > 0.3:
        print("  Verdict: PARTIAL ALIGNMENT — shared structure exists")
    else:
        print("  Verdict: NO CROSS-MODEL ALIGNMENT")
    print("=" * 70)


if __name__ == "__main__":
    main()
