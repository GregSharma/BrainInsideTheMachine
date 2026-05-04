"""Phase 4: Geometric Tests — CKA, Cross-Lingual Probe Transfer, Energy.

Run with: MPLBACKEND=Agg .venv/bin/python phase4_geometric.py

No generation needed — pure forward passes + numpy/sklearn on activations.
Should complete in ~2-3 minutes total.

Three tests:
  1. CKA in Z vs 100 random subspaces (k=20 and k=50)
  2. Cross-lingual probe transfer (20-class problem ID classifier)
  3. Energy concentration (norm fraction in Z vs random)

If Z's CKA percentile > 95% AND probe transfer > random: Z is special,
patching was just the wrong tool. If not: SVD extraction is wrong, pivot
to data-driven Z via ARD-MMD or contrastive directions.
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
from sklearn.model_selection import LeaveOneOut
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_multi_head_z_mask(model, layer_idx, h, GQA, d, k):
    all_vh = []
    for head in range(h):
        vh = get_attn_subspace(model, layer_idx, h, GQA, d, head, k=k)
        all_vh.append(vh)
    stacked = torch.cat(all_vh, dim=0)
    _, S, Vh_combined = torch.linalg.svd(stacked, full_matrices=False)
    return Vh_combined[:k, :]


def generate_random_basis(d, k, rng):
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].T  # (k, d) numpy


def linear_cka(X, Y):
    """Compute linear CKA between X (n, p) and Y (n, q)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    if hsic_xx * hsic_yy == 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("PHASE 4: GEOMETRIC TESTS — CKA + Probe Transfer + Energy")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    print(f"k values: {K_VALUES}, Random baselines: {N_RANDOM}")
    print("=" * 70)

    # Load model
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

    # Extract hidden states at target layer (mean-pooled)
    print("\n--- Extracting hidden states ---")
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h_out.detach().cpu().squeeze(0)
        return hook

    hook_handle = model.model.layers[LAYER].register_forward_hook(make_hook("target"))

    zh_means = np.zeros((N, d), dtype=np.float32)
    en_means = np.zeros((N, d), dtype=np.float32)
    zh_all_tokens = {}  # pair_idx -> (T, d) for per-token analysis
    en_all_tokens = {}

    for i, pair in enumerate(tqdm(PAIRS, desc="Chinese forward")):
        inputs = tokenizer(pair["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        acts = activations["target"].float().numpy()
        zh_means[i] = acts.mean(axis=0)
        zh_all_tokens[i] = acts

    for i, pair in enumerate(tqdm(PAIRS, desc="English forward")):
        inputs = tokenizer(pair["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        acts = activations["target"].float().numpy()
        en_means[i] = acts.mean(axis=0)
        en_all_tokens[i] = acts

    hook_handle.remove()

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    print(f"  zh_means: {zh_means.shape}, en_means: {en_means.shape}")

    # ==================================================================
    # Build Z masks for each k
    # ==================================================================
    print("\n--- Building Z masks ---")
    # Reload model briefly just for weight extraction
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
    )
    model.eval()

    Z_bases = {}  # k -> (k, d) numpy
    for k in K_VALUES:
        Vh = build_multi_head_z_mask(model, LAYER, h, GQA, d, k)
        Z_bases[k] = Vh.float().cpu().numpy()
        print(f"  k={k}: Z shape {Z_bases[k].shape}")

    del model
    torch.cuda.empty_cache()

    results = {}

    for k in K_VALUES:
        print(f"\n{'='*70}")
        print(f"TESTING k={k}")
        print(f"{'='*70}")

        Vh_Z = Z_bases[k]  # (k, d)

        # Project means onto Z
        zh_Z = zh_means @ Vh_Z.T  # (N, k)
        en_Z = en_means @ Vh_Z.T  # (N, k)

        # ==============================================================
        # TEST 1: CKA in Z vs random subspaces
        # ==============================================================
        print(f"\n--- Test 1: CKA (k={k}) ---")

        cka_real = linear_cka(zh_Z, en_Z)
        print(f"  CKA(zh, en) in real Z: {cka_real:.6f}")

        # Full-space CKA for reference
        cka_full = linear_cka(zh_means, en_means)
        print(f"  CKA(zh, en) in full space: {cka_full:.6f}")

        cka_random = []
        for ri in range(N_RANDOM):
            Vh_rand = generate_random_basis(d, k, rng)
            zh_R = zh_means @ Vh_rand.T
            en_R = en_means @ Vh_rand.T
            cka_random.append(linear_cka(zh_R, en_R))

        cka_random = np.array(cka_random)
        percentile = np.mean(cka_random < cka_real) * 100
        print(f"  Random CKA: mean={cka_random.mean():.6f}, std={cka_random.std():.6f}")
        print(f"  Z percentile: {percentile:.1f}%")
        print(f"  Z CKA / Random mean: {cka_real / cka_random.mean():.2f}x")

        if percentile >= 95:
            print(f"  *** Z CKA is in top {100-percentile:.1f}% — Z captures more cross-lingual structure ***")
        else:
            print(f"  Z CKA is NOT in top 5% of random distribution.")

        # ==============================================================
        # TEST 2: Cross-lingual probe transfer
        # ==============================================================
        print(f"\n--- Test 2: Cross-Lingual Probe Transfer (k={k}) ---")

        # Labels = problem ID (0-19)
        labels = np.arange(N)

        # Train on Chinese, test on English (in Z)
        def probe_transfer(X_train, X_test, y_train, y_test):
            """Ridge classifier with LOO on test for robustness."""
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_tr, y_train)
            preds = clf.predict(X_te)
            acc = np.mean(preds == y_test)
            return acc

        # Real Z: zh→en transfer
        acc_zh_to_en_Z = probe_transfer(zh_Z, en_Z, labels, labels)
        # Real Z: en→zh transfer
        acc_en_to_zh_Z = probe_transfer(en_Z, zh_Z, labels, labels)
        # Real Z: same-language (zh→zh LOO)
        loo = LeaveOneOut()
        zh_same = []
        for train_idx, test_idx in loo.split(zh_Z):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(zh_Z[train_idx])
            X_te = scaler.transform(zh_Z[test_idx])
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_tr, labels[train_idx])
            zh_same.append(clf.predict(X_te)[0] == labels[test_idx][0])
        acc_zh_loo = np.mean(zh_same)

        print(f"  Real Z: zh→en transfer = {acc_zh_to_en_Z:.1%}")
        print(f"  Real Z: en→zh transfer = {acc_en_to_zh_Z:.1%}")
        print(f"  Real Z: zh→zh LOO      = {acc_zh_loo:.1%}")
        print(f"  Chance level: {1/N:.1%}")

        # Random subspace baselines
        rand_zh_to_en = []
        rand_en_to_zh = []
        for ri in range(N_RANDOM):
            Vh_rand = generate_random_basis(d, k, rng)
            zh_R = zh_means @ Vh_rand.T
            en_R = en_means @ Vh_rand.T
            rand_zh_to_en.append(probe_transfer(zh_R, en_R, labels, labels))
            rand_en_to_zh.append(probe_transfer(en_R, zh_R, labels, labels))

        rand_zh_to_en = np.array(rand_zh_to_en)
        rand_en_to_zh = np.array(rand_en_to_zh)

        pct_zh_en = np.mean(rand_zh_to_en < acc_zh_to_en_Z) * 100
        pct_en_zh = np.mean(rand_en_to_zh < acc_en_to_zh_Z) * 100

        print(f"\n  Random zh→en: mean={rand_zh_to_en.mean():.1%}, std={rand_zh_to_en.std():.1%}")
        print(f"  Random en→zh: mean={rand_en_to_zh.mean():.1%}, std={rand_en_to_zh.std():.1%}")
        print(f"  Z zh→en percentile: {pct_zh_en:.1f}%")
        print(f"  Z en→zh percentile: {pct_en_zh:.1f}%")

        if pct_zh_en >= 95 or pct_en_zh >= 95:
            print(f"  *** PROBE TRANSFER IS SPECIAL — Z enables cross-lingual generalization ***")
        else:
            print(f"  Probe transfer in Z is NOT significantly better than random subspaces.")

        # ==============================================================
        # TEST 3: Energy concentration
        # ==============================================================
        print(f"\n--- Test 3: Energy Concentration (k={k}) ---")

        zh_norms_total = np.linalg.norm(zh_means, axis=1)  # (N,)
        en_norms_total = np.linalg.norm(en_means, axis=1)
        zh_norms_Z = np.linalg.norm(zh_Z, axis=1)
        en_norms_Z = np.linalg.norm(en_Z, axis=1)

        zh_energy_frac = (zh_norms_Z / zh_norms_total) ** 2
        en_energy_frac = (en_norms_Z / en_norms_total) ** 2

        # Expected energy fraction for random k-dim subspace = k/d
        expected = k / d

        print(f"  Expected random energy fraction: {expected:.4f} ({k}/{d})")
        print(f"  zh energy in Z: mean={zh_energy_frac.mean():.4f}, std={zh_energy_frac.std():.4f}")
        print(f"  en energy in Z: mean={en_energy_frac.mean():.4f}, std={en_energy_frac.std():.4f}")
        print(f"  Z concentration ratio (actual/expected): {zh_energy_frac.mean()/expected:.2f}x")

        # Random energy baselines
        rand_energy = []
        for ri in range(N_RANDOM):
            Vh_rand = generate_random_basis(d, k, rng)
            zh_R = zh_means @ Vh_rand.T
            zh_R_norms = np.linalg.norm(zh_R, axis=1)
            frac = (zh_R_norms / zh_norms_total) ** 2
            rand_energy.append(frac.mean())

        rand_energy = np.array(rand_energy)
        energy_pct = np.mean(rand_energy < zh_energy_frac.mean()) * 100

        print(f"  Random energy: mean={rand_energy.mean():.4f}, std={rand_energy.std():.4f}")
        print(f"  Z energy percentile: {energy_pct:.1f}%")

        # Cross-lingual energy similarity
        energy_corr = np.corrcoef(zh_energy_frac, en_energy_frac)[0, 1]
        print(f"  zh-en energy correlation in Z: {energy_corr:.4f}")

        # Store results
        results[f"k{k}"] = {
            "cka": {
                "real_Z": float(cka_real),
                "full_space": float(cka_full),
                "random_mean": float(cka_random.mean()),
                "random_std": float(cka_random.std()),
                "percentile": float(percentile),
            },
            "probe_transfer": {
                "zh_to_en_Z": float(acc_zh_to_en_Z),
                "en_to_zh_Z": float(acc_en_to_zh_Z),
                "zh_loo_Z": float(acc_zh_loo),
                "random_zh_to_en_mean": float(rand_zh_to_en.mean()),
                "random_zh_to_en_std": float(rand_zh_to_en.std()),
                "random_en_to_zh_mean": float(rand_en_to_zh.mean()),
                "random_en_to_zh_std": float(rand_en_to_zh.std()),
                "percentile_zh_en": float(pct_zh_en),
                "percentile_en_zh": float(pct_en_zh),
                "chance": float(1/N),
            },
            "energy": {
                "zh_mean": float(zh_energy_frac.mean()),
                "en_mean": float(en_energy_frac.mean()),
                "expected_random": float(expected),
                "concentration_ratio": float(zh_energy_frac.mean() / expected),
                "random_mean": float(rand_energy.mean()),
                "random_std": float(rand_energy.std()),
                "percentile": float(energy_pct),
                "zh_en_correlation": float(energy_corr),
            },
        }

    # ==================================================================
    # Plots
    # ==================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 4: Geometric Tests — Is Z Special?", fontsize=14)

    for ki, k in enumerate(K_VALUES):
        r = results[f"k{k}"]

        # CKA histogram
        ax = axes[ki, 0]
        # Regenerate random CKA for plotting
        rng_plot = np.random.default_rng(SEED + 1000 + k)
        cka_rand_plot = []
        for ri in range(N_RANDOM):
            Vh_rand = generate_random_basis(d, k, rng_plot)
            zh_R = zh_means @ Vh_rand.T
            en_R = en_means @ Vh_rand.T
            cka_rand_plot.append(linear_cka(zh_R, en_R))
        ax.hist(cka_rand_plot, bins=25, color="gray", alpha=0.7, edgecolor="black", label="Random")
        ax.axvline(r["cka"]["real_Z"], color="red", linewidth=2, label=f"Real Z = {r['cka']['real_Z']:.4f}")
        ax.set_xlabel("CKA(zh, en)")
        ax.set_ylabel("Count")
        ax.set_title(f"k={k}: CKA — Z at {r['cka']['percentile']:.0f}th percentile")
        ax.legend(fontsize=8)

        # Probe transfer histogram
        ax = axes[ki, 1]
        rng_plot2 = np.random.default_rng(SEED + 2000 + k)
        probe_rand_plot = []
        for ri in range(N_RANDOM):
            Vh_rand = generate_random_basis(d, k, rng_plot2)
            zh_R = zh_means @ Vh_rand.T
            en_R = en_means @ Vh_rand.T
            probe_rand_plot.append(probe_transfer(zh_R, en_R, np.arange(N), np.arange(N)))
        ax.hist(probe_rand_plot, bins=25, color="gray", alpha=0.7, edgecolor="black", label="Random")
        ax.axvline(r["probe_transfer"]["zh_to_en_Z"], color="red", linewidth=2,
                   label=f"Real Z = {r['probe_transfer']['zh_to_en_Z']:.1%}")
        ax.axvline(1/N, color="blue", linestyle="--", alpha=0.5, label=f"Chance = {1/N:.1%}")
        ax.set_xlabel("zh→en Transfer Accuracy")
        ax.set_ylabel("Count")
        ax.set_title(f"k={k}: Probe Transfer — Z at {r['probe_transfer']['percentile_zh_en']:.0f}th pct")
        ax.legend(fontsize=8)

        # Energy histogram
        ax = axes[ki, 2]
        rng_plot3 = np.random.default_rng(SEED + 3000 + k)
        energy_rand_plot = []
        for ri in range(N_RANDOM):
            Vh_rand = generate_random_basis(d, k, rng_plot3)
            zh_R = zh_means @ Vh_rand.T
            zh_R_norms = np.linalg.norm(zh_R, axis=1)
            zh_norms = np.linalg.norm(zh_means, axis=1)
            energy_rand_plot.append(((zh_R_norms / zh_norms) ** 2).mean())
        ax.hist(energy_rand_plot, bins=25, color="gray", alpha=0.7, edgecolor="black", label="Random")
        ax.axvline(r["energy"]["zh_mean"], color="red", linewidth=2,
                   label=f"Real Z = {r['energy']['zh_mean']:.4f}")
        ax.axvline(k/d, color="blue", linestyle="--", alpha=0.5, label=f"Expected = {k/d:.4f}")
        ax.set_xlabel("Energy Fraction in Subspace")
        ax.set_ylabel("Count")
        ax.set_title(f"k={k}: Energy — Z at {r['energy']['percentile']:.0f}th pct")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "phase4_geometric.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved: {out_path}")

    # Save results
    out_json = OUTPUT_DIR / "phase4_geometric.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_json}")

    # ==================================================================
    # Final Verdict
    # ==================================================================
    print(f"\n{'='*70}")
    print("PHASE 4 VERDICT")
    print(f"{'='*70}")

    for k in K_VALUES:
        r = results[f"k{k}"]
        cka_special = r["cka"]["percentile"] >= 95
        probe_special = r["probe_transfer"]["percentile_zh_en"] >= 95 or r["probe_transfer"]["percentile_en_zh"] >= 95
        energy_special = r["energy"]["percentile"] >= 95

        print(f"\n  k={k}:")
        print(f"    CKA:    {'SPECIAL' if cka_special else 'not special'} (p={r['cka']['percentile']:.0f}%)")
        print(f"    Probe:  {'SPECIAL' if probe_special else 'not special'} (zh→en p={r['probe_transfer']['percentile_zh_en']:.0f}%, en→zh p={r['probe_transfer']['percentile_en_zh']:.0f}%)")
        print(f"    Energy: {'SPECIAL' if energy_special else 'not special'} (p={r['energy']['percentile']:.0f}%)")

        if cka_special and probe_special:
            print(f"    ==> Z IS SPECIAL at k={k}. Patching was the wrong tool. Hypothesis alive.")
        elif cka_special or probe_special:
            print(f"    ==> PARTIAL signal at k={k}. Z captures some cross-lingual structure.")
        else:
            print(f"    ==> Z is NOT special at k={k}. SVD extraction may be wrong.")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
