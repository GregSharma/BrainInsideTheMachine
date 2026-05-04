"""Phase 5: Data-Driven Z via Contrastive Directions.

Run with: MPLBACKEND=Agg .venv/bin/python phase5_contrastive.py

Instead of extracting Z from SVD of attention weight kernels (proven wrong —
energy below random), extract Z from activations directly:

1. Compute difference vectors: diff_i = zh_mean_i - en_mean_i (20 pairs)
2. PCA on those 20 diffs → top components = "language directions"
3. The NULL SPACE of that PCA = directions where zh and en are indistinguishable
   = data-driven Z candidates
4. Test data-driven Z with CKA, probe transfer, energy vs 100 random subspaces

Also tests the SVD-Z from Phase 4 side-by-side for comparison.

If data-driven Z is special but SVD-Z isn't → extraction was wrong, hypothesis lives.
If data-driven Z also isn't special → hypothesis is in trouble.
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_attn_subspace, get_model_dims

MODEL_NAME = "Qwen/Qwen2.5-3B"
LAYER = 32
K_VALUES = [20, 50]
N_RANDOM = 100
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42

PAIRS = [
    {"zh": "设 f(n) 为 n 的二进制表示中 1 的个数。满足 1 ≤ n ≤ 2025 且 f(n) = 3 的整数 n 有多少个？",
     "en": "Let f(n) be the number of 1s in the binary representation of n. How many integers n with 1 ≤ n ≤ 2025 satisfy f(n) = 3?",
     "answer": "120", "category": "combinatorics"},
    {"zh": "求方程 x² + y² = 2025 的所有非负整数解的个数。",
     "en": "Find the number of all non-negative integer solutions to x² + y² = 2025.",
     "answer": "4", "category": "number_theory"},
    {"zh": "计算 1 + 2 + 3 + ... + 100 的值。",
     "en": "Calculate the value of 1 + 2 + 3 + ... + 100.",
     "answer": "5050", "category": "arithmetic"},
    {"zh": "一个袋子里有 5 个红球和 3 个蓝球。随机取出 2 个球，取到 2 个红球的概率是多少？",
     "en": "A bag contains 5 red balls and 3 blue balls. If 2 balls are drawn randomly, what is the probability of getting 2 red balls?",
     "answer": "5/14", "category": "probability"},
    {"zh": "求函数 f(x) = x³ - 3x 在区间 [-2, 2] 上的最大值。",
     "en": "Find the maximum value of f(x) = x³ - 3x on the interval [-2, 2].",
     "answer": "2", "category": "calculus"},
    {"zh": "在一个 4×4 的棋盘上放置 4 个车，使得它们互不攻击，有多少种放法？",
     "en": "In how many ways can 4 rooks be placed on a 4×4 chessboard so that no two attack each other?",
     "answer": "24", "category": "combinatorics"},
    {"zh": "已知等比数列 {a_n} 的首项 a_1 = 2，公比 q = 3，求前 5 项之和。",
     "en": "Given a geometric sequence {a_n} with first term a_1 = 2 and common ratio q = 3, find the sum of the first 5 terms.",
     "answer": "242", "category": "sequences"},
    {"zh": "求矩阵 [[1, 2], [3, 4]] 的行列式。",
     "en": "Find the determinant of the matrix [[1, 2], [3, 4]].",
     "answer": "-2", "category": "linear_algebra"},
    {"zh": "用辗转相除法求 gcd(252, 198)。",
     "en": "Use the Euclidean algorithm to find gcd(252, 198).",
     "answer": "18", "category": "number_theory"},
    {"zh": "如果 sin(θ) = 3/5 且 θ 在第一象限，求 cos(θ) 的值。",
     "en": "If sin(θ) = 3/5 and θ is in the first quadrant, find the value of cos(θ).",
     "answer": "4/5", "category": "trigonometry"},
    {"zh": "一个圆的半径为 7，求其面积。",
     "en": "A circle has radius 7. Find its area.",
     "answer": "49π", "category": "geometry"},
    {"zh": "求极限 lim(n→∞) (1 + 1/n)^n 的值。",
     "en": "Find the limit lim(n→∞) (1 + 1/n)^n.",
     "answer": "e", "category": "calculus"},
    {"zh": "三个骰子同时掷出，点数之和为 10 的概率是多少？",
     "en": "Three dice are thrown simultaneously. What is the probability that the sum of the points is 10?",
     "answer": "27/216", "category": "probability"},
    {"zh": "求二次方程 x² - 5x + 6 = 0 的两个根。",
     "en": "Find the two roots of the quadratic equation x² - 5x + 6 = 0.",
     "answer": "2,3", "category": "algebra"},
    {"zh": "一个长方体的长、宽、高分别为 3、4、5，求其体积和表面积。",
     "en": "A rectangular box has length 3, width 4, and height 5. Find its volume and surface area.",
     "answer": "60,94", "category": "geometry"},
    {"zh": "求数列 1, 1, 2, 3, 5, 8, 13, ... 的第 10 项。",
     "en": "Find the 10th term of the sequence 1, 1, 2, 3, 5, 8, 13, ...",
     "answer": "55", "category": "sequences"},
    {"zh": "将 255 转换为十六进制。",
     "en": "Convert 255 to hexadecimal.",
     "answer": "FF", "category": "arithmetic"},
    {"zh": "求不定积分 ∫ x·e^x dx 的结果。",
     "en": "Find the indefinite integral ∫ x·e^x dx.",
     "answer": "(x-1)e^x+C", "category": "calculus"},
    {"zh": "100 除以 7 的余数是多少？",
     "en": "What is the remainder when 100 is divided by 7?",
     "answer": "2", "category": "arithmetic"},
    {"zh": "从 1 到 100 的整数中，有多少个是 3 的倍数？",
     "en": "Among the integers from 1 to 100, how many are multiples of 3?",
     "answer": "33", "category": "counting"},
]


def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    if hsic_xx * hsic_yy == 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def generate_random_basis(d, k, rng):
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].T  # (k, d)


def build_svd_z(model, layer_idx, h, GQA, d, k):
    all_vh = []
    for head in range(h):
        vh = get_attn_subspace(model, layer_idx, h, GQA, d, head, k=k)
        all_vh.append(vh)
    stacked = torch.cat(all_vh, dim=0)
    _, S, Vh_combined = torch.linalg.svd(stacked, full_matrices=False)
    return Vh_combined[:k, :].cpu().numpy()


def build_contrastive_z(zh_means, en_means, k):
    """Extract data-driven Z from activation differences.

    1. diff_i = zh_mean_i - en_mean_i for each paired problem
    2. PCA on diffs → top components are "language directions"
    3. NULL SPACE (bottom components of full-space PCA) = language-invariant Z

    Actually: we want the directions where zh and en AGREE, not differ.
    So we take the SHARED space: PCA on the concatenated [zh; en] means,
    then remove the language directions found from diffs.

    Simpler approach that's more robust with N=20:
    - Compute diff vectors (20 x 2048)
    - SVD of diffs → top singular vectors = language-specific directions
    - Project those OUT → remaining space is language-invariant
    - Take top-k PCA of mean activations projected into null space
    """
    N, d = zh_means.shape

    # Language-specific directions from differences
    diffs = zh_means - en_means  # (20, 2048)
    diffs_centered = diffs - diffs.mean(axis=0)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)

    # How many language directions to remove? Use explained variance.
    # With N=20, we get at most 19 nonzero singular values.
    var_explained = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)
    # Remove directions that explain 90% of language variance
    n_lang = max(1, int(np.searchsorted(cumvar, 0.90) + 1))
    n_lang = min(n_lang, len(S))

    lang_dirs = Vt[:n_lang]  # (n_lang, d) — language-specific directions

    # Project language directions out of activation space
    # P_perp = I - V_lang^T @ V_lang
    proj_out = np.eye(d, dtype=np.float32) - lang_dirs.T @ lang_dirs

    # Project all activations (both languages) into null space
    all_means = np.concatenate([zh_means, en_means], axis=0)  # (40, 2048)
    projected = all_means @ proj_out.T  # (40, d_remaining_effective)

    # PCA on projected activations to find top-k directions of shared variance
    projected_centered = projected - projected.mean(axis=0)
    _, S_proj, Vt_proj = np.linalg.svd(projected_centered, full_matrices=False)

    # Take top-k directions of the projected (language-free) space
    z_basis = Vt_proj[:k]  # (k, d)

    # Orthonormalize (should already be orthogonal from SVD, but ensure)
    z_basis_f32 = z_basis.astype(np.float32)

    return z_basis_f32, n_lang, var_explained[:min(10, len(var_explained))].tolist()


def run_tests(zh_means, en_means, z_basis, random_bases, label):
    """Run CKA, probe transfer, energy tests for a given Z basis."""
    N = zh_means.shape[0]
    k = z_basis.shape[0]
    d = z_basis.shape[1]

    # Project into Z
    zh_z = zh_means @ z_basis.T  # (N, k)
    en_z = en_means @ z_basis.T

    # --- CKA ---
    cka_z = linear_cka(zh_z, en_z)
    cka_random = []
    for rb in random_bases:
        zh_r = zh_means @ rb.T
        en_r = en_means @ rb.T
        cka_random.append(linear_cka(zh_r, en_r))
    cka_random = np.array(cka_random)
    cka_pct = float(np.mean(cka_random <= cka_z) * 100)

    # --- Probe transfer ---
    labels = np.arange(N)
    scaler_zh = StandardScaler().fit(zh_z)
    scaler_en = StandardScaler().fit(en_z)

    clf = RidgeClassifier(alpha=1.0)
    clf.fit(scaler_zh.transform(zh_z), labels)
    zh_en_acc = float(clf.score(scaler_en.transform(en_z), labels))

    clf2 = RidgeClassifier(alpha=1.0)
    clf2.fit(scaler_en.transform(en_z), labels)
    en_zh_acc = float(clf2.score(scaler_zh.transform(zh_z), labels))

    probe_random_zh_en = []
    probe_random_en_zh = []
    for rb in random_bases:
        zh_r = zh_means @ rb.T
        en_r = en_means @ rb.T
        s1 = StandardScaler().fit(zh_r)
        s2 = StandardScaler().fit(en_r)
        c1 = RidgeClassifier(alpha=1.0)
        c1.fit(s1.transform(zh_r), labels)
        probe_random_zh_en.append(c1.score(s2.transform(en_r), labels))
        c2 = RidgeClassifier(alpha=1.0)
        c2.fit(s2.transform(en_r), labels)
        probe_random_en_zh.append(c2.score(s1.transform(zh_r), labels))

    probe_random_zh_en = np.array(probe_random_zh_en)
    probe_random_en_zh = np.array(probe_random_en_zh)
    probe_pct_zh_en = float(np.mean(probe_random_zh_en <= zh_en_acc) * 100)
    probe_pct_en_zh = float(np.mean(probe_random_en_zh <= en_zh_acc) * 100)

    # --- Energy ---
    zh_norms = np.linalg.norm(zh_means, axis=1) ** 2
    en_norms = np.linalg.norm(en_means, axis=1) ** 2
    zh_z_norms = np.linalg.norm(zh_z, axis=1) ** 2
    en_z_norms = np.linalg.norm(en_z, axis=1) ** 2
    zh_energy = (zh_z_norms / zh_norms).mean()
    en_energy = (en_z_norms / en_norms).mean()
    expected_energy = k / d

    energy_random = []
    for rb in random_bases:
        zh_r = zh_means @ rb.T
        en_r = en_means @ rb.T
        zh_r_n = np.linalg.norm(zh_r, axis=1) ** 2
        en_r_n = np.linalg.norm(en_r, axis=1) ** 2
        energy_random.append(((zh_r_n / zh_norms).mean() + (en_r_n / en_norms).mean()) / 2)
    energy_random = np.array(energy_random)
    combined_energy = (zh_energy + en_energy) / 2
    energy_pct = float(np.mean(energy_random <= combined_energy) * 100)

    result = {
        "cka": {"real": float(cka_z), "random_mean": float(cka_random.mean()),
                "random_std": float(cka_random.std()), "percentile": cka_pct,
                "random_dist": cka_random.tolist()},
        "probe": {"zh_en": zh_en_acc, "en_zh": en_zh_acc,
                  "random_zh_en_mean": float(probe_random_zh_en.mean()),
                  "random_zh_en_std": float(probe_random_zh_en.std()),
                  "random_en_zh_mean": float(probe_random_en_zh.mean()),
                  "pct_zh_en": probe_pct_zh_en, "pct_en_zh": probe_pct_en_zh,
                  "random_zh_en_dist": probe_random_zh_en.tolist(),
                  "random_en_zh_dist": probe_random_en_zh.tolist()},
        "energy": {"zh": float(zh_energy), "en": float(en_energy),
                   "expected": float(expected_energy), "combined": float(combined_energy),
                   "random_mean": float(energy_random.mean()),
                   "random_std": float(energy_random.std()), "percentile": energy_pct,
                   "random_dist": energy_random.tolist()},
    }

    print(f"\n  [{label}] CKA: {cka_z:.4f} (p={cka_pct:.0f}%, random={cka_random.mean():.4f})")
    print(f"  [{label}] Probe zh→en: {zh_en_acc:.0%} (p={probe_pct_zh_en:.0f}%, random={probe_random_zh_en.mean():.0%})")
    print(f"  [{label}] Probe en→zh: {en_zh_acc:.0%} (p={probe_pct_en_zh:.0f}%, random={probe_random_en_zh.mean():.0%})")
    print(f"  [{label}] Energy: {combined_energy:.4f} (p={energy_pct:.0f}%, expected={expected_energy:.4f})")

    return result


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("PHASE 5: CONTRASTIVE Z — Data-Driven Language-Invariant Subspace")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    print(f"k values: {K_VALUES}, Random baselines: {N_RANDOM}")
    print("=" * 70)

    # Load model & extract activations
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
    )
    model.eval()
    dims = get_model_dims(model)
    L, d, h, GQA = dims["L"], dims["d"], dims["h"], dims["GQA"]
    print(f"  L={L} d={d} h={h} GQA={GQA}")
    N = len(PAIRS)

    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h_out.detach().cpu().squeeze(0)
        return hook

    hook_handle = model.model.layers[LAYER].register_forward_hook(make_hook("target"))

    zh_means = np.zeros((N, d), dtype=np.float32)
    en_means = np.zeros((N, d), dtype=np.float32)

    for i, pair in enumerate(tqdm(PAIRS, desc="Chinese forward")):
        inputs = tokenizer(pair["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        acts = activations["target"].float().numpy()
        zh_means[i] = acts.mean(axis=0)

    for i, pair in enumerate(tqdm(PAIRS, desc="English forward")):
        inputs = tokenizer(pair["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        acts = activations["target"].float().numpy()
        en_means[i] = acts.mean(axis=0)

    hook_handle.remove()

    # Build SVD-Z for comparison
    print("\n--- Building SVD-Z (Phase 4 baseline) ---")
    svd_z = {}
    for k in K_VALUES:
        svd_z[k] = build_svd_z(model, LAYER, h, GQA, d, k)
        print(f"  SVD-Z k={k}: shape {svd_z[k].shape}")

    del model
    torch.cuda.empty_cache()

    print(f"\n  zh_means: {zh_means.shape}, en_means: {en_means.shape}")

    # Build contrastive Z
    print("\n--- Building Contrastive Z (data-driven) ---")
    contrastive_z = {}
    contrastive_meta = {}
    for k in K_VALUES:
        z_basis, n_lang, lang_varexp = build_contrastive_z(zh_means, en_means, k)
        contrastive_z[k] = z_basis
        contrastive_meta[k] = {"n_lang_removed": n_lang, "lang_var_explained": lang_varexp}
        print(f"  Contrastive-Z k={k}: removed {n_lang} language dirs, shape {z_basis.shape}")
        print(f"    Language variance explained (top 5): {[f'{v:.3f}' for v in lang_varexp[:5]]}")

    # Overlap between SVD-Z and Contrastive-Z
    print("\n--- SVD-Z vs Contrastive-Z overlap ---")
    overlap_results = {}
    for k in K_VALUES:
        # Principal angle between subspaces
        M = svd_z[k] @ contrastive_z[k].T  # (k, k)
        svals = np.linalg.svd(M, compute_uv=False)
        svals = np.clip(svals, 0, 1)
        angles_deg = np.degrees(np.arccos(svals))
        mean_overlap = float(svals.mean())
        overlap_results[k] = {
            "mean_cosine": mean_overlap,
            "min_cosine": float(svals.min()),
            "max_cosine": float(svals.max()),
            "principal_angles_deg": angles_deg.tolist()[:5],
        }
        print(f"  k={k}: mean cosine overlap = {mean_overlap:.4f} "
              f"(1.0 = identical, 0.0 = orthogonal)")
        print(f"    Top 5 principal angles: {[f'{a:.1f}°' for a in angles_deg[:5]]}")

    # Generate random baselines (shared across tests)
    print(f"\n--- Generating {N_RANDOM} random baselines ---")
    random_bases = {}
    for k in K_VALUES:
        random_bases[k] = [generate_random_basis(d, k, rng) for _ in range(N_RANDOM)]

    # Run all tests
    all_results = {}
    for k in K_VALUES:
        print(f"\n{'='*70}")
        print(f"TESTING k={k}")
        print(f"{'='*70}")

        print(f"\n--- SVD-Z (weight-derived) ---")
        svd_res = run_tests(zh_means, en_means, svd_z[k], random_bases[k], f"SVD k={k}")

        print(f"\n--- Contrastive-Z (activation-derived) ---")
        con_res = run_tests(zh_means, en_means, contrastive_z[k], random_bases[k], f"Contrastive k={k}")

        all_results[f"k{k}"] = {
            "svd_z": svd_res,
            "contrastive_z": con_res,
            "overlap": overlap_results[k],
            "contrastive_meta": contrastive_meta[k],
        }

    # --- Plotting ---
    fig, axes = plt.subplots(len(K_VALUES), 3, figsize=(18, 5 * len(K_VALUES)))
    if len(K_VALUES) == 1:
        axes = axes[np.newaxis, :]

    for row, k in enumerate(K_VALUES):
        res = all_results[f"k{k}"]

        # CKA
        ax = axes[row, 0]
        ax.hist(res["svd_z"]["cka"]["random_dist"], bins=20, alpha=0.5, color='gray', label='Random')
        ax.axvline(res["svd_z"]["cka"]["real"], color='red', lw=2, label=f'SVD-Z ({res["svd_z"]["cka"]["percentile"]:.0f}%)')
        ax.axvline(res["contrastive_z"]["cka"]["real"], color='blue', lw=2, ls='--',
                   label=f'Contr-Z ({res["contrastive_z"]["cka"]["percentile"]:.0f}%)')
        ax.set_title(f"k={k}: CKA(zh, en)")
        ax.legend(fontsize=8)

        # Probe zh→en
        ax = axes[row, 1]
        ax.hist(res["svd_z"]["probe"]["random_zh_en_dist"], bins=20, alpha=0.5, color='gray', label='Random')
        ax.axvline(res["svd_z"]["probe"]["zh_en"], color='red', lw=2,
                   label=f'SVD-Z ({res["svd_z"]["probe"]["pct_zh_en"]:.0f}%)')
        ax.axvline(res["contrastive_z"]["probe"]["zh_en"], color='blue', lw=2, ls='--',
                   label=f'Contr-Z ({res["contrastive_z"]["probe"]["pct_zh_en"]:.0f}%)')
        ax.axvline(0.05, color='green', lw=1, ls=':', label='Chance')
        ax.set_title(f"k={k}: Probe zh→en Transfer")
        ax.legend(fontsize=8)

        # Energy
        ax = axes[row, 2]
        ax.hist(res["svd_z"]["energy"]["random_dist"], bins=20, alpha=0.5, color='gray', label='Random')
        ax.axvline(res["svd_z"]["energy"]["combined"], color='red', lw=2,
                   label=f'SVD-Z ({res["svd_z"]["energy"]["percentile"]:.0f}%)')
        ax.axvline(res["contrastive_z"]["energy"]["combined"], color='blue', lw=2, ls='--',
                   label=f'Contr-Z ({res["contrastive_z"]["energy"]["percentile"]:.0f}%)')
        ax.set_title(f"k={k}: Energy Concentration")
        ax.legend(fontsize=8)

    plt.suptitle("Phase 5: SVD-Z vs Contrastive-Z vs Random", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase5_contrastive.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {OUTPUT_DIR / 'phase5_contrastive.png'}")

    # Save JSON (strip distributions for readability)
    save_results = {}
    for kkey, kres in all_results.items():
        save_results[kkey] = {}
        for ztype in ["svd_z", "contrastive_z"]:
            sr = {}
            for test in ["cka", "probe", "energy"]:
                sr[test] = {k2: v2 for k2, v2 in kres[ztype][test].items()
                           if not k2.endswith("_dist")}
            save_results[kkey][ztype] = sr
        save_results[kkey]["overlap"] = kres["overlap"]
        save_results[kkey]["contrastive_meta"] = kres["contrastive_meta"]

    with open(OUTPUT_DIR / "phase5_contrastive.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'phase5_contrastive.json'}")

    # --- VERDICT ---
    print(f"\n{'='*70}")
    print("PHASE 5 VERDICT")
    print(f"{'='*70}")

    for k in K_VALUES:
        res = all_results[f"k{k}"]
        svd = res["svd_z"]
        con = res["contrastive_z"]
        olap = res["overlap"]

        print(f"\n  k={k} (overlap: {olap['mean_cosine']:.3f}):")
        print(f"    {'Test':<12} {'SVD-Z':>20} {'Contrastive-Z':>20}")
        print(f"    {'CKA':<12} {svd['cka']['percentile']:>18.0f}% {con['cka']['percentile']:>18.0f}%")
        print(f"    {'Probe zh→en':<12} {svd['probe']['pct_zh_en']:>18.0f}% {con['probe']['pct_zh_en']:>18.0f}%")
        print(f"    {'Probe en→zh':<12} {svd['probe']['pct_en_zh']:>18.0f}% {con['probe']['pct_en_zh']:>18.0f}%")
        print(f"    {'Energy':<12} {svd['energy']['percentile']:>18.0f}% {con['energy']['percentile']:>18.0f}%")

        con_special = (con["cka"]["percentile"] >= 95 or
                       con["probe"]["pct_zh_en"] >= 95 or
                       con["probe"]["pct_en_zh"] >= 95)
        svd_special = (svd["cka"]["percentile"] >= 95 or
                       svd["probe"]["pct_zh_en"] >= 95 or
                       svd["probe"]["pct_en_zh"] >= 95)

        if con_special and not svd_special:
            print(f"    ==> CONTRASTIVE Z IS SPECIAL, SVD wasn't. Extraction was the problem!")
        elif con_special and svd_special:
            print(f"    ==> BOTH are special. Strong signal for language-invariant structure.")
        elif not con_special and svd_special:
            print(f"    ==> SVD special but contrastive isn't. Surprising — investigate.")
        else:
            print(f"    ==> NEITHER is special at k={k}. Hypothesis in trouble.")

    print("\nDone.")


if __name__ == "__main__":
    main()
