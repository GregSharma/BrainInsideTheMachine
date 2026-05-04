"""Phase 6: Unified Experiments with Contrastive Z.

Run with: MPLBACKEND=Agg .venv/bin/python phase6_unified.py

Three experiments in one script:
  1. BRIDGE REVISITED: Fit rotation R mapping zh→en in contrastive Z. R² on held-out.
  2. PATCHING REVISITED: Replace contrastive Z dims during generation. Does it break output?
  3. LAYER SWEEP: Run contrastive extraction + within-category probe at L8,12,16,20,24,28,32,33,34.

All use 200 template problems from Phase 5B/5C, contrastive Z extraction.
"""

import json
import random as pyrandom
import re
import unicodedata
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes

from utils import get_model_dims

MODEL_NAME = "Qwen/Qwen2.5-3B"
K = 20
N_RANDOM = 50  # fewer randoms for speed, still sufficient
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42
MAX_NEW_TOKENS = 100

# Layers to sweep
SWEEP_LAYERS = [8, 12, 16, 20, 24, 28, 32, 33, 34]


def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh = f"计算 {a} + {b} 的值。"
            en = f"Calculate {a} + {b}."
        else:
            zh = f"计算 {a} × {b} 的值。"
            en = f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "category": 0})

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "category": 1})

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "category": 2})

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "category": 3})

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        problems.append({"zh": zh, "en": en, "category": 4})

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


def extract_activations(model, tokenizer, problems, layer_idx):
    """Extract mean-pooled hidden states at given layer for all problems."""
    d = model.config.hidden_size
    N = len(problems)
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h_out.detach().cpu().squeeze(0)
        return hook

    hook_handle = model.model.layers[layer_idx].register_forward_hook(make_hook("target"))

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


def cjk_fraction(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    cjk = sum(1 for c in chars if unicodedata.category(c).startswith('Lo'))
    return cjk / len(chars)


def classify_language(text):
    return "zh" if cjk_fraction(text) > 0.3 else "en"


def main():
    rng = np.random.default_rng(SEED)
    cat_names = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]

    print("=" * 70)
    print("PHASE 6: UNIFIED — Bridge + Patching + Layer Sweep")
    print(f"Model: {MODEL_NAME}, k={K}, Random baselines: {N_RANDOM}")
    print("=" * 70)

    problems = generate_problems(200, seed=SEED)
    categories = np.array([p["category"] for p in problems])

    # Stratified train/test split
    train_idx, test_idx = [], []
    for cat in range(5):
        cat_indices = np.where(categories == cat)[0]
        np.random.default_rng(SEED).shuffle(cat_indices)
        half = len(cat_indices) // 2
        train_idx.extend(cat_indices[:half].tolist())
        test_idx.extend(cat_indices[half:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
    )
    model.eval()
    dims = get_model_dims(model)
    d = dims["d"]

    results = {}

    # ================================================================
    # EXPERIMENT 1: BRIDGE REVISITED
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: BRIDGE — Rotation mapping zh→en in contrastive Z")
    print("=" * 70)

    zh_means, en_means = extract_activations(model, tokenizer, problems, 32)
    zh_train, zh_test = zh_means[train_idx], zh_means[test_idx]
    en_train, en_test = en_means[train_idx], en_means[test_idx]

    # Build contrastive Z from train
    z_basis, n_lang, lang_dirs = build_contrastive_z(zh_train, en_train, K)
    print(f"  Contrastive Z: k={z_basis.shape[0]}, removed {n_lang} language dirs")

    # Project into Z
    zh_z_train = zh_train @ z_basis.T
    en_z_train = en_train @ z_basis.T
    zh_z_test = zh_test @ z_basis.T
    en_z_test = en_test @ z_basis.T

    # Fit Procrustes rotation on train: find R such that zh_z_train @ R ≈ en_z_train
    # Center first
    zh_mu = zh_z_train.mean(axis=0)
    en_mu = en_z_train.mean(axis=0)
    zh_c = zh_z_train - zh_mu
    en_c = en_z_train - en_mu
    R, scale = orthogonal_procrustes(zh_c, en_c)

    # Predict on test
    zh_test_c = zh_z_test - zh_mu
    en_pred = zh_test_c @ R + en_mu
    en_test_c = en_z_test

    # R² on test
    ss_res = np.sum((en_z_test - en_pred) ** 2)
    ss_tot = np.sum((en_z_test - en_z_test.mean(axis=0)) ** 2)
    r2_test = 1 - ss_res / ss_tot

    # R² on train
    en_train_pred = zh_c @ R + en_mu
    ss_res_tr = np.sum((en_z_train - en_train_pred) ** 2)
    ss_tot_tr = np.sum((en_z_train - en_z_train.mean(axis=0)) ** 2)
    r2_train = 1 - ss_res_tr / ss_tot_tr

    # Random subspace bridge comparison
    r2_random = []
    for _ in range(N_RANDOM):
        rb = generate_random_basis(d, K, rng)
        zh_r_tr = zh_train @ rb.T
        en_r_tr = en_train @ rb.T
        zh_r_te = zh_test @ rb.T
        en_r_te = en_test @ rb.T
        mu_zh = zh_r_tr.mean(axis=0)
        mu_en = en_r_tr.mean(axis=0)
        R_rand, _ = orthogonal_procrustes(zh_r_tr - mu_zh, en_r_tr - mu_en)
        pred = (zh_r_te - mu_zh) @ R_rand + mu_en
        ss_r = np.sum((en_r_te - pred) ** 2)
        ss_t = np.sum((en_r_te - en_r_te.mean(axis=0)) ** 2)
        r2_random.append(1 - ss_r / ss_t if ss_t > 0 else 0)
    r2_random = np.array(r2_random)
    bridge_pct = float(np.mean(r2_random <= r2_test) * 100)

    print(f"\n  Bridge R² (train): {r2_train:.4f}")
    print(f"  Bridge R² (test):  {r2_test:.4f}")
    print(f"  Random R² mean:    {r2_random.mean():.4f} ± {r2_random.std():.4f}")
    print(f"  Percentile:        {bridge_pct:.0f}%")
    if bridge_pct >= 95:
        print(f"  *** BRIDGE WORKS — zh and en are rotations of each other in Z ***")

    results["bridge"] = {
        "r2_train": float(r2_train), "r2_test": float(r2_test),
        "random_mean": float(r2_random.mean()), "random_std": float(r2_random.std()),
        "percentile": bridge_pct, "scale": float(scale),
    }

    # ================================================================
    # EXPERIMENT 2: PATCHING REVISITED
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PATCHING — Contrastive Z vs Z⊥ vs Random")
    print("=" * 70)

    # Use 20 original problems (shorter, more natural, better for generation)
    PATCH_PAIRS = [
        {"zh": "计算 1 + 2 + 3 + ... + 100 的值。", "en": "Calculate the value of 1 + 2 + 3 + ... + 100.", "answer": "5050"},
        {"zh": "求二次方程 x² - 5x + 6 = 0 的两个根。", "en": "Find the two roots of the quadratic equation x² - 5x + 6 = 0.", "answer": "2,3"},
        {"zh": "100 除以 7 的余数是多少？", "en": "What is the remainder when 100 is divided by 7?", "answer": "2"},
        {"zh": "一个圆的半径为 7，求其面积。", "en": "A circle has radius 7. Find its area.", "answer": "49π"},
        {"zh": "求矩阵 [[1, 2], [3, 4]] 的行列式。", "en": "Find the determinant of the matrix [[1, 2], [3, 4]].", "answer": "-2"},
        {"zh": "求组合数 C(10, 3) 的值。", "en": "Find the value of C(10, 3).", "answer": "120"},
        {"zh": "等差数列首项为 3，公差为 5，求前 10 项之和。", "en": "An arithmetic sequence has first term 3 and common difference 5. Find the sum of the first 10 terms.", "answer": "255"},
        {"zh": "一个长方形的长为 12，宽为 8，求其面积。", "en": "A rectangle has length 12 and width 8. Find its area.", "answer": "96"},
        {"zh": "7654 除以 13 的余数是多少？", "en": "What is the remainder when 7654 is divided by 13?", "answer": "10"},
        {"zh": "计算 237 × 456 的值。", "en": "Calculate 237 × 456.", "answer": "108072"},
    ]

    # Build contrastive Z from the 200 template problems (all of them, since patching
    # tests generation quality, not probing — no circularity concern)
    z_basis_full, _, lang_dirs_full = build_contrastive_z(zh_means, en_means, K)
    z_torch = torch.tensor(z_basis_full, dtype=torch.float16, device=model.device)

    # Build Z⊥ complement: project out Z from full space, take top-k of remainder
    proj_z = z_basis_full.T @ z_basis_full  # (d, d)
    proj_zperp = np.eye(d, dtype=np.float32) - proj_z
    # Get top-k of Z⊥ from activation variance
    all_acts = np.concatenate([zh_means, en_means], axis=0)
    acts_zperp = all_acts @ proj_zperp.T
    acts_zperp_c = acts_zperp - acts_zperp.mean(axis=0)
    _, _, Vt_zperp = np.linalg.svd(acts_zperp_c, full_matrices=False)
    zperp_basis = Vt_zperp[:K].astype(np.float32)
    zperp_torch = torch.tensor(zperp_basis, dtype=torch.float16, device=model.device)

    # Random basis
    rand_basis = generate_random_basis(d, K, rng).astype(np.float32)
    rand_torch = torch.tensor(rand_basis, dtype=torch.float16, device=model.device)

    def patch_and_generate(pair, basis_torch, patch_type="z_patch"):
        """Replace activation components in basis with noise during generation."""
        source_lang = "en"
        source_text = pair[source_lang]

        hook_handles = []

        def make_patch_hook(basis):
            def hook(module, input, output):
                h = output if isinstance(output, torch.Tensor) else output[0]
                # Project into basis
                coeffs = h @ basis.T  # (batch, seq, k)
                # Replace with Gaussian noise of same scale
                noise_scale = coeffs.std().item() + 1e-8
                noise = torch.randn_like(coeffs) * noise_scale
                # Reconstruct: remove original component, add noise component
                h_patched = h - coeffs @ basis + noise @ basis
                if isinstance(output, torch.Tensor):
                    return h_patched
                return (h_patched,) + output[1:]
            return hook

        handle = model.model.layers[32].register_forward_hook(make_patch_hook(basis_torch))
        hook_handles.append(handle)

        inputs = tokenizer(source_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        for h in hook_handles:
            h.remove()

        return text

    # Also generate unpatched baseline
    def generate_clean(pair, lang="en"):
        inputs = tokenizer(pair[lang], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    patch_results = {"clean": [], "z_patch": [], "zperp_patch": [], "random_patch": []}

    for i, pair in enumerate(tqdm(PATCH_PAIRS, desc="Patching")):
        clean = generate_clean(pair, "en")
        z_patched = patch_and_generate(pair, z_torch, "z_patch")
        zperp_patched = patch_and_generate(pair, zperp_torch, "zperp_patch")
        rand_patched = patch_and_generate(pair, rand_torch, "random_patch")

        patch_results["clean"].append(clean)
        patch_results["z_patch"].append(z_patched)
        patch_results["zperp_patch"].append(zperp_patched)
        patch_results["random_patch"].append(rand_patched)

        # Check if answer preserved
        ans = pair["answer"]
        c_ok = ans.lower() in clean.lower()
        z_ok = ans.lower() in z_patched.lower()
        zp_ok = ans.lower() in zperp_patched.lower()
        r_ok = ans.lower() in rand_patched.lower()
        print(f"  [{i}] ans={ans}: clean={c_ok}, z_patch={z_ok}, zperp={zp_ok}, rand={r_ok}")

    # Tally
    n_patch = len(PATCH_PAIRS)
    for cond in ["clean", "z_patch", "zperp_patch", "random_patch"]:
        correct = sum(1 for i, pair in enumerate(PATCH_PAIRS)
                      if pair["answer"].lower() in patch_results[cond][i].lower())
        lang_preserved = sum(1 for text in patch_results[cond]
                             if classify_language(text[len(PATCH_PAIRS[0]["en"]):]) == "en")
        print(f"  {cond}: {correct}/{n_patch} correct, {lang_preserved}/{n_patch} English")

    results["patching"] = {
        cond: {
            "correct": sum(1 for i, pair in enumerate(PATCH_PAIRS)
                          if pair["answer"].lower() in patch_results[cond][i].lower()),
            "total": n_patch,
        } for cond in ["clean", "z_patch", "zperp_patch", "random_patch"]
    }

    # ================================================================
    # EXPERIMENT 3: LAYER SWEEP
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: LAYER SWEEP — Contrastive Z across layers")
    print("=" * 70)

    layer_results = {}
    cat_test = categories[test_idx]

    for layer in SWEEP_LAYERS:
        print(f"\n--- Layer {layer} ---")
        if layer == 32:
            # Reuse already-extracted activations
            zh_l, en_l = zh_means, en_means
        else:
            zh_l, en_l = extract_activations(model, tokenizer, problems, layer)

        zh_tr, zh_te = zh_l[train_idx], zh_l[test_idx]
        en_tr, en_te = en_l[train_idx], en_l[test_idx]

        # Build contrastive Z from train
        z_b, n_lg, _ = build_contrastive_z(zh_tr, en_tr, K)
        actual_k = z_b.shape[0]

        # Within-category probe on test
        zh_z_te = zh_te @ z_b.T
        en_z_te = en_te @ z_b.T

        # Pool across categories
        cat_accs = []
        rand_cat_accs_per_draw = np.zeros(N_RANDOM)

        random_bases_layer = [generate_random_basis(d, K, rng) for _ in range(N_RANDOM)]

        for cat in range(5):
            mask = cat_test == cat
            zh_cat = zh_z_te[mask]
            en_cat = en_z_te[mask]
            n_cat = mask.sum()
            if n_cat < 3:
                continue

            labels = np.arange(n_cat)
            s1 = StandardScaler().fit(zh_cat)
            s2 = StandardScaler().fit(en_cat)
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(s1.transform(zh_cat), labels)
            acc = clf.score(s2.transform(en_cat), labels)
            cat_accs.append(acc)

            for ri, rb in enumerate(random_bases_layer):
                zh_r = zh_te[mask] @ rb.T
                en_r = en_te[mask] @ rb.T
                sr1 = StandardScaler().fit(zh_r)
                sr2 = StandardScaler().fit(en_r)
                cr = RidgeClassifier(alpha=1.0)
                cr.fit(sr1.transform(zh_r), labels)
                rand_cat_accs_per_draw[ri] += cr.score(sr2.transform(en_r), labels)

        pool_z = np.mean(cat_accs) if cat_accs else 0
        rand_cat_accs_per_draw /= 5
        pool_pct = float(np.mean(rand_cat_accs_per_draw <= pool_z) * 100)

        # Energy
        zh_norms = np.linalg.norm(zh_te, axis=1) ** 2
        zh_z_norms = np.linalg.norm(zh_z_te, axis=1) ** 2
        energy = (zh_z_norms / zh_norms).mean()
        expected = actual_k / d

        layer_results[layer] = {
            "pooled_acc": float(pool_z),
            "random_mean": float(rand_cat_accs_per_draw.mean()),
            "percentile": pool_pct,
            "n_lang_dirs": n_lg,
            "energy": float(energy),
            "energy_ratio": float(energy / expected),
        }

        print(f"  Pooled acc: {pool_z:.0%} (random={rand_cat_accs_per_draw.mean():.0%}, p={pool_pct:.0f}%)")
        print(f"  Language dirs: {n_lg}, Energy: {energy:.4f} ({energy/expected:.1f}x)")

    results["layer_sweep"] = layer_results

    # ================================================================
    # PLOTTING
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Bridge histogram
    ax = axes[0]
    ax.hist(r2_random, bins=20, alpha=0.6, color='gray', label='Random')
    ax.axvline(r2_test, color='blue', lw=2, label=f'Contr-Z R²={r2_test:.3f} (p={bridge_pct:.0f}%)')
    ax.set_title("Bridge: Procrustes R² (zh→en rotation)")
    ax.set_xlabel("R² on held-out test")
    ax.legend(fontsize=8)

    # Layer sweep
    ax = axes[1]
    layers = sorted(layer_results.keys())
    accs = [layer_results[l]["pooled_acc"] for l in layers]
    rands = [layer_results[l]["random_mean"] for l in layers]
    pcts = [layer_results[l]["percentile"] for l in layers]
    ax.plot(layers, accs, 'b-o', label='Contrastive Z', lw=2)
    ax.plot(layers, rands, 'gray', ls='--', label='Random mean')
    ax.fill_between(layers, rands, accs, alpha=0.2, color='blue')
    for l, p in zip(layers, pcts):
        if p >= 95:
            ax.annotate(f'p={p:.0f}%', (l, layer_results[l]["pooled_acc"]),
                       fontsize=7, ha='center', va='bottom')
    ax.set_title("Layer Sweep: Within-Category Probe")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pooled Accuracy")
    ax.legend(fontsize=8)

    # Patching bar chart
    ax = axes[2]
    conds = ["clean", "z_patch", "zperp_patch", "random_patch"]
    labels_p = ["Clean", "Z-patch", "Z⊥-patch", "Random"]
    colors = ["green", "blue", "red", "gray"]
    vals = [results["patching"][c]["correct"] for c in conds]
    ax.bar(labels_p, vals, color=colors, alpha=0.7)
    ax.set_title("Patching: Correct Answers (Contrastive Z)")
    ax.set_ylabel(f"Correct / {n_patch}")
    ax.set_ylim(0, n_patch + 1)

    plt.suptitle("Phase 6: Bridge + Layer Sweep + Patching (Contrastive Z)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase6_unified.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {OUTPUT_DIR / 'phase6_unified.png'}")

    # Save JSON
    with open(OUTPUT_DIR / "phase6_unified.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR / 'phase6_unified.json'}")

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 6 VERDICT")
    print(f"{'='*70}")

    print(f"\n  BRIDGE: R²={r2_test:.3f} on held-out (p={bridge_pct:.0f}%)")
    if bridge_pct >= 95:
        print(f"  → zh and en ARE rotations of each other in contrastive Z")
    else:
        print(f"  → Rotation model doesn't fully capture zh→en mapping")

    print(f"\n  PATCHING (contrastive Z, k={K}):")
    for c, l in zip(conds, labels_p):
        print(f"    {l}: {results['patching'][c]['correct']}/{n_patch} correct")

    print(f"\n  LAYER SWEEP (within-category probe, k={K}):")
    for l in layers:
        r = layer_results[l]
        marker = "***" if r["percentile"] >= 95 else "   "
        print(f"    {marker} L{l:2d}: Z={r['pooled_acc']:.0%}  Rand={r['random_mean']:.0%}  "
              f"p={r['percentile']:.0f}%  lang_dirs={r['n_lang_dirs']}  "
              f"energy={r['energy_ratio']:.1f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
