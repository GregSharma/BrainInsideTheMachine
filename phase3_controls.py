"""Phase 3 Controls: Random Subspace Control + Experiment D (Bridge).

Run with: MPLBACKEND=Agg .venv/bin/python phase3_controls.py

Part 1 — Random Subspace Control:
  Generate N_RANDOM random orthonormal 50-dim subspaces, run the same
  patching protocol as Phase 3A, and compare answer-change rates to the
  real Z subspace. If Z is special, random subspaces should NOT show the
  double dissociation (0/20 vs 19/20).

Part 2 — Experiment D (Bridge):
  Collect Z-projected mean activations for all 20 zh/en pairs at L32 k=50.
  Fit W* = (Z_en^T Z_en)^{-1} Z_en^T Z_zh. Report R², orthogonality error,
  SVD spectrum, and leave-one-out cross-validation R².

  If W* is approximately orthogonal, languages are rotations in Z →
  Stiefel manifold structure.
"""

import json
import re
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_attn_subspace, get_model_dims

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B"
LAYER = 32
K = 50
N_RANDOM = 5          # number of random subspace controls
MAX_NEW_TOKENS = 150
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
# Helpers (same as phase3.py)
# ---------------------------------------------------------------------------
def build_multi_head_z_mask(model, layer_idx, h, GQA, d, k):
    """Top-k SVD of stacked multi-head attention kernels. Returns (k, d)."""
    all_vh = []
    for head in range(h):
        vh = get_attn_subspace(model, layer_idx, h, GQA, d, head, k=k)
        all_vh.append(vh)
    stacked = torch.cat(all_vh, dim=0)
    _, S, Vh_combined = torch.linalg.svd(stacked, full_matrices=False)
    return Vh_combined[:k, :]


def cjk_fraction(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    cjk = sum(1 for c in chars if unicodedata.category(c).startswith("Lo"))
    return cjk / len(chars)


def classify_language(text):
    frac = cjk_fraction(text)
    if frac > 0.3:
        return "zh"
    if frac < 0.05:
        return "en"
    return "mixed"


def extract_answer(text):
    first_line = text.strip().split("\n")[0].strip()
    for prefix in ["Answer:", "答案：", "答案:", "The answer is", "= ", "answer is "]:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip()
    match = re.search(r"[-]?\d+(?:[./]\d+)?(?:π|\\pi)?", first_line)
    if match:
        return match.group(0)
    return first_line[:80]


def generate_random_basis(d, k, rng):
    """Generate a random orthonormal (k, d) basis via QR decomposition."""
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return torch.from_numpy(Q[:, :k].T)  # (k, d)


def run_patching(model, tokenizer, pair, layer, P_Z, P_Zp, zh_mean, condition):
    """Run a single patching condition and return result dict."""
    zh_mean_dev = zh_mean.to(P_Z.device).half()
    zh_Z = P_Z @ zh_mean_dev
    zh_Zp = P_Zp @ zh_mean_dev

    if condition == "baseline":
        patch_hook = None
    elif condition == "z_patch":
        zh_Z_h = zh_Z.half()
        P_Zp_h = P_Zp

        def patch_hook(module, input, output, _pzp=P_Zp_h, _zhz=zh_Z_h):
            h = output if isinstance(output, torch.Tensor) else output[0]
            patched = h.clone()
            if patched.dim() == 3:
                for t in range(patched.shape[1]):
                    patched[0, t, :] = _pzp @ patched[0, t, :] + _zhz
            else:
                for t in range(patched.shape[0]):
                    patched[t, :] = _pzp @ patched[t, :] + _zhz
            return patched
    elif condition == "zperp_patch":
        zh_Zp_h = zh_Zp.half()
        P_Z_h = P_Z

        def patch_hook(module, input, output, _pz=P_Z_h, _zhzp=zh_Zp_h):
            h = output if isinstance(output, torch.Tensor) else output[0]
            patched = h.clone()
            if patched.dim() == 3:
                for t in range(patched.shape[1]):
                    patched[0, t, :] = _pz @ patched[0, t, :] + _zhzp
            else:
                for t in range(patched.shape[0]):
                    patched[t, :] = _pz @ patched[t, :] + _zhzp
            return patched

    handle = None
    if patch_hook is not None:
        handle = model.model.layers[layer].register_forward_hook(patch_hook)

    inputs = tokenizer(pair["en"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=1.0,
        )

    if handle is not None:
        handle.remove()

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = gen_ids[0, prompt_len:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "extracted_answer": extract_answer(raw_output),
        "output_language": classify_language(raw_output),
        "raw_output": raw_output[:500],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("PHASE 3 CONTROLS: Random Subspace + Experiment D Bridge")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}, k: {K}")
    print(f"Random controls: {N_RANDOM}, Pairs: {len(PAIRS)}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Build real Z mask + projectors
    # ------------------------------------------------------------------
    print("\n--- Building Z mask (L32 k=50) ---")
    Vh_Z = build_multi_head_z_mask(model, LAYER, h, GQA, d, K)
    P_Z = (Vh_Z.T @ Vh_Z).to(model.device).half()
    P_Zp = (torch.eye(d) - Vh_Z.T @ Vh_Z).to(model.device).half()
    print(f"  Z mask shape: {Vh_Z.shape}")

    # ------------------------------------------------------------------
    # Extract Chinese mean-pooled hidden states at target layer
    # ------------------------------------------------------------------
    print("\n--- Extracting Chinese hidden states ---")
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h.detach().cpu().squeeze(0)
        return hook

    hook_handle = model.model.layers[LAYER].register_forward_hook(make_hook("target"))

    zh_means = {}  # pair_idx -> (d,) float tensor
    en_means = {}  # pair_idx -> (d,) float tensor

    for i, pair in enumerate(tqdm(PAIRS, desc="Chinese forward passes")):
        inputs = tokenizer(pair["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        zh_means[i] = activations["target"].float().mean(dim=0)

    for i, pair in enumerate(tqdm(PAIRS, desc="English forward passes")):
        inputs = tokenizer(pair["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        en_means[i] = activations["target"].float().mean(dim=0)

    hook_handle.remove()

    # ==================================================================
    # PART 1: Random Subspace Control
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 1: RANDOM SUBSPACE CONTROL")
    print("=" * 70)

    # First, get baseline answers (no patching)
    print("\n--- Running baselines ---")
    baselines = {}
    for i, pair in enumerate(tqdm(PAIRS, desc="Baselines")):
        result = run_patching(model, tokenizer, pair, LAYER, P_Z, P_Zp, zh_means[i], "baseline")
        baselines[i] = result["extracted_answer"]

    # Run real Z patching (z_patch and zperp_patch) for reference
    print("\n--- Running real Z patching ---")
    real_z_changed = 0
    real_zp_changed = 0
    real_z_lang_switch = 0
    real_zp_lang_switch = 0

    for i, pair in enumerate(tqdm(PAIRS, desc="Real Z-patch")):
        r = run_patching(model, tokenizer, pair, LAYER, P_Z, P_Zp, zh_means[i], "z_patch")
        if r["extracted_answer"] != baselines[i]:
            real_z_changed += 1
        if r["output_language"] == "zh":
            real_z_lang_switch += 1

    for i, pair in enumerate(tqdm(PAIRS, desc="Real Z⊥-patch")):
        r = run_patching(model, tokenizer, pair, LAYER, P_Z, P_Zp, zh_means[i], "zperp_patch")
        if r["extracted_answer"] != baselines[i]:
            real_zp_changed += 1
        if r["output_language"] == "zh":
            real_zp_lang_switch += 1

    print(f"\n  Real Z:  ans_changed={real_z_changed}/20, lang_switch={real_z_lang_switch}/20")
    print(f"  Real Z⊥: ans_changed={real_zp_changed}/20, lang_switch={real_zp_lang_switch}/20")

    # Run random subspace controls
    random_results = []
    for ri in range(N_RANDOM):
        print(f"\n--- Random subspace {ri+1}/{N_RANDOM} ---")
        Vh_rand = generate_random_basis(d, K, rng)
        P_R = (Vh_rand.T @ Vh_rand).to(model.device).half()
        P_Rp = (torch.eye(d) - Vh_rand.T @ Vh_rand).to(model.device).half()

        r_z_changed = 0
        r_zp_changed = 0
        r_z_lang = 0
        r_zp_lang = 0

        for i, pair in enumerate(tqdm(PAIRS, desc=f"Rand{ri} Z-patch", leave=False)):
            r = run_patching(model, tokenizer, pair, LAYER, P_R, P_Rp, zh_means[i], "z_patch")
            if r["extracted_answer"] != baselines[i]:
                r_z_changed += 1
            if r["output_language"] == "zh":
                r_z_lang += 1

        for i, pair in enumerate(tqdm(PAIRS, desc=f"Rand{ri} Z⊥-patch", leave=False)):
            r = run_patching(model, tokenizer, pair, LAYER, P_R, P_Rp, zh_means[i], "zperp_patch")
            if r["extracted_answer"] != baselines[i]:
                r_zp_changed += 1
            if r["output_language"] == "zh":
                r_zp_lang += 1

        random_results.append({
            "z_changed": r_z_changed,
            "zp_changed": r_zp_changed,
            "z_lang_switch": r_z_lang,
            "zp_lang_switch": r_zp_lang,
        })
        print(f"  Random {ri}: Z changed={r_z_changed}/20, Z⊥ changed={r_zp_changed}/20, "
              f"Z lang={r_z_lang}/20, Z⊥ lang={r_zp_lang}/20")

    # Summary
    print("\n" + "-" * 60)
    print("RANDOM SUBSPACE CONTROL SUMMARY")
    print("-" * 60)
    print(f"  {'Subspace':<15} {'Z-patch ans Δ':>15} {'Z⊥-patch ans Δ':>16} "
          f"{'Z-patch→zh':>12} {'Z⊥-patch→zh':>13}")
    print(f"  {'Real Z':<15} {real_z_changed:>10}/20    {real_zp_changed:>10}/20     "
          f"{real_z_lang_switch:>7}/20    {real_zp_lang_switch:>7}/20")
    for ri, rr in enumerate(random_results):
        print(f"  {'Random '+str(ri):<15} {rr['z_changed']:>10}/20    {rr['zp_changed']:>10}/20     "
              f"{rr['z_lang_switch']:>7}/20    {rr['zp_lang_switch']:>7}/20")

    rand_z_mean = np.mean([r["z_changed"] for r in random_results])
    rand_zp_mean = np.mean([r["zp_changed"] for r in random_results])
    print(f"\n  Random mean:  Z-patch Δ = {rand_z_mean:.1f}/20, Z⊥-patch Δ = {rand_zp_mean:.1f}/20")
    print(f"  Real Z:       Z-patch Δ = {real_z_changed}/20, Z⊥-patch Δ = {real_zp_changed}/20")

    dissociation_real = real_zp_changed - real_z_changed
    dissociation_rand = [r["zp_changed"] - r["z_changed"] for r in random_results]
    print(f"\n  Double dissociation gap (Z⊥ - Z):")
    print(f"    Real Z:  {dissociation_real}")
    print(f"    Random:  {dissociation_rand} (mean={np.mean(dissociation_rand):.1f})")

    # ==================================================================
    # PART 2: Experiment D — The Bridge
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 2: EXPERIMENT D — THE BRIDGE")
    print("=" * 70)

    # Project means onto Z subspace (extract k active components)
    Vh_Z_cpu = Vh_Z.float()  # (k, d)
    Z_zh = torch.zeros(N, K)
    Z_en = torch.zeros(N, K)

    for i in range(N):
        Z_zh[i] = Vh_Z_cpu @ zh_means[i]  # (k,)
        Z_en[i] = Vh_Z_cpu @ en_means[i]  # (k,)

    Z_zh = Z_zh.numpy()
    Z_en = Z_en.numpy()

    print(f"\n  Z_zh shape: {Z_zh.shape}, Z_en shape: {Z_en.shape}")

    # D.2: Solve W* = (Z_en^T Z_en)^{-1} Z_en^T Z_zh
    ZeTZe = Z_en.T @ Z_en  # (k, k)
    ZeTZz = Z_en.T @ Z_zh  # (k, k)

    # Use pseudo-inverse for numerical stability
    W_star = np.linalg.lstsq(Z_en, Z_zh, rcond=None)[0]  # (k, k)
    print(f"  W* shape: {W_star.shape}")

    # D.3: Metrics
    # R²
    Z_zh_pred = Z_en @ W_star
    ss_res = np.sum((Z_zh - Z_zh_pred) ** 2)
    Z_zh_mean = Z_zh.mean(axis=0, keepdims=True)
    ss_tot = np.sum((Z_zh - Z_zh_mean) ** 2)
    R2 = 1.0 - ss_res / ss_tot
    print(f"\n  R² = {R2:.6f}")

    # Orthogonality error: ||W*^T W* - I||_F / k
    WtW = W_star.T @ W_star
    orth_error = np.linalg.norm(WtW - np.eye(K), "fro") / K
    print(f"  Orthogonality error = ||W*^T W* - I||_F / k = {orth_error:.6f}")

    # SVD of W*
    U_w, S_w, Vht_w = np.linalg.svd(W_star)
    print(f"  W* singular values (top 10): {S_w[:10].round(4)}")
    print(f"  W* singular values (bottom 10): {S_w[-10:].round(4)}")
    print(f"  Condition number: {S_w[0] / S_w[-1]:.4f}")
    print(f"  Spectrum flatness (std/mean): {S_w.std() / S_w.mean():.4f}")

    # Leave-one-out cross-validation R²
    loo_r2s = []
    for hold in range(N):
        mask = np.ones(N, dtype=bool)
        mask[hold] = False
        Z_en_train = Z_en[mask]
        Z_zh_train = Z_zh[mask]
        W_loo = np.linalg.lstsq(Z_en_train, Z_zh_train, rcond=None)[0]
        pred = Z_en[hold:hold+1] @ W_loo
        actual = Z_zh[hold:hold+1]
        ss_res_loo = np.sum((actual - pred) ** 2)
        ss_tot_loo = np.sum((actual - Z_zh_mean) ** 2)
        loo_r2s.append(1.0 - ss_res_loo / ss_tot_loo)

    loo_r2_mean = np.mean(loo_r2s)
    loo_r2_std = np.std(loo_r2s)
    print(f"\n  LOO-CV R²: mean={loo_r2_mean:.6f}, std={loo_r2_std:.6f}")
    print(f"  LOO-CV R² per pair: {[f'{r:.3f}' for r in loo_r2s]}")

    # Per-pair reconstruction error (which pairs are hardest to bridge?)
    pair_errors = np.sqrt(np.sum((Z_zh - Z_zh_pred) ** 2, axis=1))
    print(f"\n  Per-pair reconstruction error (L2):")
    for i in range(N):
        print(f"    Pair {i:2d} ({PAIRS[i]['category']:<14}): error={pair_errors[i]:.4f}, "
              f"LOO-R²={loo_r2s[i]:.3f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Random control comparison (bar chart)
    ax = axes[0, 0]
    labels = ["Real Z"] + [f"Rand {i}" for i in range(N_RANDOM)]
    z_vals = [real_z_changed] + [r["z_changed"] for r in random_results]
    zp_vals = [real_zp_changed] + [r["zp_changed"] for r in random_results]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, z_vals, w, label="Z-patch Δ", color="steelblue")
    ax.bar(x + w/2, zp_vals, w, label="Z⊥-patch Δ", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("Answers changed / 20")
    ax.set_title("Random Subspace Control: Patching Effects")
    ax.legend()
    ax.set_ylim(0, 21)

    # 2. Dissociation gap
    ax = axes[0, 1]
    gaps = [dissociation_real] + dissociation_rand
    colors = ["green"] + ["gray"] * N_RANDOM
    ax.bar(labels, gaps, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Dissociation gap (Z⊥ Δ - Z Δ)")
    ax.set_title("Double Dissociation: Real Z vs Random")
    ax.set_xticklabels(labels, rotation=30)

    # 3. W* singular value spectrum
    ax = axes[0, 2]
    ax.plot(range(1, K+1), S_w, "o-", markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="σ=1 (rotation)")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("σ_i")
    ax.set_title(f"W* SVD spectrum (R²={R2:.3f})")
    ax.legend()

    # 4. WtW heatmap (should be ~identity if orthogonal)
    ax = axes[1, 0]
    im = ax.imshow(WtW, cmap="RdBu_r", vmin=-0.5, vmax=2.0, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"W*ᵀW* (orth err={orth_error:.4f})")
    ax.set_xlabel("Z-dim")
    ax.set_ylabel("Z-dim")

    # 5. LOO-CV R² per pair
    ax = axes[1, 1]
    cats = [PAIRS[i]["category"] for i in range(N)]
    ax.barh(range(N), loo_r2s, color="steelblue")
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"{i}: {c}" for i, c in enumerate(cats)], fontsize=7)
    ax.axvline(loo_r2_mean, color="red", linestyle="--", label=f"mean={loo_r2_mean:.3f}")
    ax.set_xlabel("LOO-CV R²")
    ax.set_title("Bridge generalization per pair")
    ax.legend()
    ax.invert_yaxis()

    # 6. Scatter: Z_zh vs Z_en (first 3 PCs)
    ax = axes[1, 2]
    # PCA on concatenated Z representations
    Z_all = np.vstack([Z_zh, Z_en])
    Z_centered = Z_all - Z_all.mean(axis=0)
    U_pca, S_pca, Vt_pca = np.linalg.svd(Z_centered, full_matrices=False)
    Z_pca = Z_centered @ Vt_pca[:2].T  # (2N, 2)
    ax.scatter(Z_pca[:N, 0], Z_pca[:N, 1], c="red", marker="o", label="Chinese", alpha=0.7)
    ax.scatter(Z_pca[N:, 0], Z_pca[N:, 1], c="blue", marker="s", label="English", alpha=0.7)
    for i in range(N):
        ax.plot([Z_pca[i, 0], Z_pca[N+i, 0]], [Z_pca[i, 1], Z_pca[N+i, 1]],
                "k-", alpha=0.2, linewidth=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Z-projected means in PC space (paired)")
    ax.legend()

    plt.tight_layout()
    out_path = OUTPUT_DIR / "phase3_controls.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved: {out_path}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "part1_random_control": {
            "real_z": {
                "ans_changed": real_z_changed,
                "lang_switch": real_z_lang_switch,
            },
            "real_zperp": {
                "ans_changed": real_zp_changed,
                "lang_switch": real_zp_lang_switch,
            },
            "random": random_results,
            "dissociation_real": dissociation_real,
            "dissociation_random": dissociation_rand,
        },
        "part2_bridge": {
            "R2": float(R2),
            "orth_error": float(orth_error),
            "singular_values": S_w.tolist(),
            "condition_number": float(S_w[0] / S_w[-1]),
            "spectrum_flatness": float(S_w.std() / S_w.mean()),
            "loo_cv_R2_mean": float(loo_r2_mean),
            "loo_cv_R2_std": float(loo_r2_std),
            "loo_cv_R2_per_pair": [float(r) for r in loo_r2s],
            "per_pair_reconstruction_error": pair_errors.tolist(),
            "W_star": W_star.tolist(),
        },
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "k": K,
            "n_random": N_RANDOM,
            "n_pairs": N,
            "seed": SEED,
        },
    }

    out_json = OUTPUT_DIR / "phase3_controls.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_json}")

    # ------------------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"\n  RANDOM CONTROL:")
    if rand_z_mean > real_z_changed + 3:
        print(f"    Random subspaces change MORE answers than Z ({rand_z_mean:.0f} vs {real_z_changed}).")
        print(f"    Z is SPECIAL: it specifically preserves answers across languages.")
    elif abs(rand_z_mean - real_z_changed) <= 3:
        print(f"    Random subspaces change similar answers to Z ({rand_z_mean:.0f} vs {real_z_changed}).")
        print(f"    Need more controls or a different metric to distinguish Z.")
    else:
        print(f"    Random Z-patch changes fewer answers ({rand_z_mean:.0f} vs {real_z_changed}).")
        print(f"    Unexpected — investigate further.")

    dissoc_significant = np.mean(dissociation_rand) < dissociation_real * 0.5
    print(f"\n    Dissociation: Real={dissociation_real}, Random mean={np.mean(dissociation_rand):.1f}")
    if dissoc_significant:
        print(f"    CONFIRMED: Z shows stronger double dissociation than random subspaces.")
    else:
        print(f"    INCONCLUSIVE: Random subspaces also show dissociation. More controls needed.")

    print(f"\n  BRIDGE:")
    if R2 > 0.9:
        print(f"    R² = {R2:.3f} → Languages are near-rotations in Z.")
    elif R2 > 0.8:
        print(f"    R² = {R2:.3f} → Thin wrappers: mostly linear relationship.")
    elif R2 > 0.5:
        print(f"    R² = {R2:.3f} → Moderate linearity. Some nonlinear structure.")
    else:
        print(f"    R² = {R2:.3f} → Relationship is substantially nonlinear.")

    if orth_error < 0.1:
        print(f"    Orth error = {orth_error:.4f} → W* is approximately orthogonal.")
        print(f"    STIEFEL MANIFOLD: Languages are related by rotation in Z!")
    elif orth_error < 0.5:
        print(f"    Orth error = {orth_error:.4f} → W* is not a pure rotation.")
        print(f"    Languages use Z-dimensions with different scales.")
    else:
        print(f"    Orth error = {orth_error:.4f} → W* is far from orthogonal.")

    if loo_r2_mean > 0.5:
        print(f"    LOO-CV R² = {loo_r2_mean:.3f} → Bridge generalizes well to held-out pairs.")
    else:
        print(f"    LOO-CV R² = {loo_r2_mean:.3f} → Bridge overfits. Linear map may not generalize.")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
