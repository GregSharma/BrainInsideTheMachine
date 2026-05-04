"""Phase 3 Controls — k=20 Random Subspace Control + k Sweep.

Run with: MPLBACKEND=Agg .venv/bin/python phase3_controls_k20.py

The make-or-break experiment:
  If random 20-dim subspaces also show 0/20 answer changes under Z-patch,
  then Z is NOT special at any dimensionality — the asymmetry is purely
  about replacing 1% vs 99% of dimensions.

  If random shows 3-8/20 changes while real Z shows 0-1/20, Z IS special:
  it specifically encodes language-invariant reasoning structure.

Part 1: k=20 random control (10 random draws)
Part 2: k sweep (k=10,15,20,25,30,35,40,45,50) with real Z only
Part 3: Per-dimension overlap analysis (random vs real Z)
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
K_MAIN = 20          # primary test dimensionality
N_RANDOM = 10         # number of random subspace controls
K_SWEEP = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # for k sweep
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
# Helpers
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


def make_projectors(Vh, d, device):
    """Given (k, d) orthonormal rows, return P_Z and P_Zp on device in half."""
    P_Z = (Vh.T @ Vh).to(device).half()
    P_Zp = (torch.eye(d) - Vh.T @ Vh).to(device).half()
    return P_Z, P_Zp


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


def run_control_at_k(model, tokenizer, d, h, GQA, layer, k, n_random, rng, zh_means, baselines):
    """Run real Z + n_random random controls at a given k. Returns dict."""
    N = len(PAIRS)

    # Build real Z
    Vh_Z = build_multi_head_z_mask(model, layer, h, GQA, d, k)
    P_Z, P_Zp = make_projectors(Vh_Z, d, model.device)

    # Real Z patching
    real_z_changed = 0
    real_zp_changed = 0
    real_z_lang = 0
    real_zp_lang = 0

    for i, pair in enumerate(tqdm(PAIRS, desc=f"k={k} Real Z-patch", leave=False)):
        r = run_patching(model, tokenizer, pair, layer, P_Z, P_Zp, zh_means[i], "z_patch")
        if r["extracted_answer"] != baselines[i]:
            real_z_changed += 1
        if r["output_language"] == "zh":
            real_z_lang += 1

    for i, pair in enumerate(tqdm(PAIRS, desc=f"k={k} Real Z⊥-patch", leave=False)):
        r = run_patching(model, tokenizer, pair, layer, P_Z, P_Zp, zh_means[i], "zperp_patch")
        if r["extracted_answer"] != baselines[i]:
            real_zp_changed += 1
        if r["output_language"] == "zh":
            real_zp_lang += 1

    result = {
        "k": k,
        "real_z": {"ans_changed": real_z_changed, "lang_switch": real_z_lang},
        "real_zperp": {"ans_changed": real_zp_changed, "lang_switch": real_zp_lang},
        "random": [],
    }

    # Random controls
    for ri in range(n_random):
        Vh_rand = generate_random_basis(d, k, rng)
        P_R, P_Rp = make_projectors(Vh_rand, d, model.device)

        r_z_changed = 0
        r_zp_changed = 0
        r_z_lang = 0
        r_zp_lang = 0

        for i, pair in enumerate(tqdm(PAIRS, desc=f"k={k} Rand{ri} Z-patch", leave=False)):
            r = run_patching(model, tokenizer, pair, layer, P_R, P_Rp, zh_means[i], "z_patch")
            if r["extracted_answer"] != baselines[i]:
                r_z_changed += 1
            if r["output_language"] == "zh":
                r_z_lang += 1

        for i, pair in enumerate(tqdm(PAIRS, desc=f"k={k} Rand{ri} Z⊥-patch", leave=False)):
            r = run_patching(model, tokenizer, pair, layer, P_R, P_Rp, zh_means[i], "zperp_patch")
            if r["extracted_answer"] != baselines[i]:
                r_zp_changed += 1
            if r["output_language"] == "zh":
                r_zp_lang += 1

        result["random"].append({
            "z_changed": r_z_changed, "zp_changed": r_zp_changed,
            "z_lang_switch": r_z_lang, "zp_lang_switch": r_zp_lang,
        })

    # Overlap: cosine similarity between real Z directions and each random basis
    overlaps = []
    for ri in range(n_random):
        Vh_rand = generate_random_basis(d, k, rng)  # regenerate (different seed state)
        # Subspace overlap = ||Vh_Z @ Vh_rand^T||_F^2 / k
        cross = (Vh_Z.float() @ Vh_rand.float().T)
        overlap = (cross ** 2).sum().item() / k
        overlaps.append(overlap)
    result["subspace_overlaps"] = overlaps

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("PHASE 3 CONTROLS: k=20 Random Control + k Sweep")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    print(f"Primary k: {K_MAIN}, Random controls: {N_RANDOM}")
    print(f"k sweep: {K_SWEEP}")
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

    # Extract Chinese + English hidden states at target layer
    print("\n--- Extracting hidden states ---")
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h.detach().cpu().squeeze(0)
        return hook

    hook_handle = model.model.layers[LAYER].register_forward_hook(make_hook("target"))

    zh_means = {}
    for i, pair in enumerate(tqdm(PAIRS, desc="Chinese forward passes")):
        inputs = tokenizer(pair["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        zh_means[i] = activations["target"].float().mean(dim=0)

    hook_handle.remove()

    # Baselines (no patching)
    print("\n--- Running baselines ---")
    # Need a dummy P_Z/P_Zp for baseline call — build at k=20
    Vh_tmp = build_multi_head_z_mask(model, LAYER, h, GQA, d, K_MAIN)
    P_Z_tmp, P_Zp_tmp = make_projectors(Vh_tmp, d, model.device)

    baselines = {}
    for i, pair in enumerate(tqdm(PAIRS, desc="Baselines")):
        result = run_patching(model, tokenizer, pair, LAYER, P_Z_tmp, P_Zp_tmp, zh_means[i], "baseline")
        baselines[i] = result["extracted_answer"]

    # ==================================================================
    # PART 1: k=20 Random Control (THE MAKE-OR-BREAK)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"PART 1: k={K_MAIN} RANDOM CONTROL ({N_RANDOM} draws)")
    print("=" * 70)

    k20_result = run_control_at_k(
        model, tokenizer, d, h, GQA, LAYER, K_MAIN, N_RANDOM, rng, zh_means, baselines
    )

    # Print summary
    real_z = k20_result["real_z"]["ans_changed"]
    real_zp = k20_result["real_zperp"]["ans_changed"]
    rand_z_vals = [r["z_changed"] for r in k20_result["random"]]
    rand_zp_vals = [r["zp_changed"] for r in k20_result["random"]]

    print(f"\n  {'Subspace':<15} {'Z-patch Δ':>12} {'Z⊥-patch Δ':>13}")
    print(f"  {'Real Z':<15} {real_z:>7}/20    {real_zp:>7}/20")
    for ri, rr in enumerate(k20_result["random"]):
        print(f"  {'Random '+str(ri):<15} {rr['z_changed']:>7}/20    {rr['zp_changed']:>7}/20")

    rand_z_mean = np.mean(rand_z_vals)
    rand_z_std = np.std(rand_z_vals)
    print(f"\n  Random Z-patch mean: {rand_z_mean:.1f} ± {rand_z_std:.1f}")
    print(f"  Real Z-patch:        {real_z}/20")

    # Statistical test: is real Z different from random?
    if rand_z_std > 0:
        z_score = (real_z - rand_z_mean) / rand_z_std
        print(f"  Z-score (real vs random): {z_score:.2f}")
    else:
        z_score = 0.0
        print(f"  Z-score: N/A (zero variance in random)")

    # Verdict
    print(f"\n  === VERDICT ===")
    if rand_z_mean >= real_z + 3:
        print(f"  Z IS SPECIAL: Random subspaces change {rand_z_mean:.0f}/20 answers vs {real_z}/20 for real Z.")
        print(f"  The Z subspace specifically preserves reasoning structure.")
        verdict = "Z_SPECIAL"
    elif abs(rand_z_mean - real_z) < 3:
        print(f"  Z is NOT distinguishable from random at k={K_MAIN}.")
        print(f"  Random: {rand_z_mean:.1f}/20, Real Z: {real_z}/20")
        verdict = "INDISTINGUISHABLE"
    else:
        print(f"  Unexpected: Real Z changes MORE than random ({real_z} vs {rand_z_mean:.0f}).")
        verdict = "UNEXPECTED"

    # ==================================================================
    # PART 2: k Sweep (real Z only, 2 random controls per k for reference)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 2: k SWEEP (real Z + 2 random controls per k)")
    print("=" * 70)

    sweep_results = []
    for k in K_SWEEP:
        print(f"\n--- k={k} ---")
        # Skip k=20 if we already ran it with full controls
        if k == K_MAIN:
            # Re-use part 1 results, just add abbreviated version
            sweep_results.append(k20_result)
            print(f"  (reusing Part 1 results)")
            continue

        result = run_control_at_k(
            model, tokenizer, d, h, GQA, LAYER, k, 2, rng, zh_means, baselines
        )
        sweep_results.append(result)

        rz = result["real_z"]["ans_changed"]
        rzp = result["real_zperp"]["ans_changed"]
        rand_mean = np.mean([r["z_changed"] for r in result["random"]])
        print(f"  Real Z: {rz}/20, Real Z⊥: {rzp}/20, Random Z mean: {rand_mean:.0f}/20")

    # Print sweep summary table
    print(f"\n{'='*70}")
    print("k SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'k':>4} {'dims/2048':>10} {'Real Z-Δ':>10} {'Real Z⊥-Δ':>11} {'Rand Z-Δ':>10} {'Gap':>6} {'Z special?':>12}")
    for sr in sweep_results:
        k = sr["k"]
        rz = sr["real_z"]["ans_changed"]
        rzp = sr["real_zperp"]["ans_changed"]
        rand_mean = np.mean([r["z_changed"] for r in sr["random"]])
        gap = rand_mean - rz
        frac = k / d * 100
        special = "YES" if gap >= 3 else "no" if gap < 1 else "maybe"
        print(f"  {k:>4} {frac:>9.1f}% {rz:>7}/20  {rzp:>7}/20   {rand_mean:>7.1f}/20  {gap:>+5.1f} {special:>12}")

    # ==================================================================
    # PART 3: Overlap Analysis
    # ==================================================================
    print(f"\n{'='*70}")
    print("PART 3: SUBSPACE OVERLAP ANALYSIS")
    print(f"{'='*70}")

    # At k=20: compute overlap between real Z and each random subspace
    Vh_Z20 = build_multi_head_z_mask(model, LAYER, h, GQA, d, K_MAIN)
    overlap_rng = np.random.default_rng(12345)
    overlaps_20 = []
    for ri in range(50):
        Vh_rand = generate_random_basis(d, K_MAIN, overlap_rng)
        cross = Vh_Z20.float() @ Vh_rand.float().T
        overlap = (cross ** 2).sum().item() / K_MAIN
        overlaps_20.append(overlap)

    expected_overlap = K_MAIN / d  # expected for random subspaces
    print(f"  k=20 expected random overlap: {expected_overlap:.4f}")
    print(f"  k=20 measured random overlap: {np.mean(overlaps_20):.4f} ± {np.std(overlaps_20):.4f}")
    print(f"  (50 random draws)")

    # ==================================================================
    # Plots
    # ==================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Phase 3 Controls: k={K_MAIN} Random + k Sweep", fontsize=14)

    # 1. k=20 random control bar chart
    ax = axes[0, 0]
    labels = ["Real Z"] + [f"R{i}" for i in range(N_RANDOM)]
    z_vals = [real_z] + rand_z_vals
    zp_vals = [real_zp] + rand_zp_vals
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, z_vals, w, label="Z-patch Δ", color="steelblue")
    ax.bar(x + w/2, zp_vals, w, label="Z⊥-patch Δ", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=7)
    ax.set_ylabel("Answers changed / 20")
    ax.set_title(f"k={K_MAIN} Random Control ({N_RANDOM} draws)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 21)

    # 2. k sweep: real Z vs random
    ax = axes[0, 1]
    ks = [sr["k"] for sr in sweep_results]
    real_z_sweep = [sr["real_z"]["ans_changed"] for sr in sweep_results]
    rand_z_sweep = [np.mean([r["z_changed"] for r in sr["random"]]) for sr in sweep_results]
    real_zp_sweep = [sr["real_zperp"]["ans_changed"] for sr in sweep_results]

    ax.plot(ks, real_z_sweep, "o-", color="steelblue", label="Real Z-patch Δ", linewidth=2)
    ax.plot(ks, rand_z_sweep, "s--", color="gray", label="Random Z-patch Δ (mean)", linewidth=1.5)
    ax.plot(ks, real_zp_sweep, "^-", color="coral", label="Real Z⊥-patch Δ", linewidth=2)
    ax.set_xlabel("k (subspace dimensionality)")
    ax.set_ylabel("Answers changed / 20")
    ax.set_title("k Sweep: Real Z vs Random")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.5, 21)
    ax.grid(True, alpha=0.3)

    # 3. Gap (random - real) across k
    ax = axes[0, 2]
    gaps = [r - z for r, z in zip(rand_z_sweep, real_z_sweep)]
    colors = ["green" if g >= 3 else "orange" if g >= 1 else "red" for g in gaps]
    ax.bar(range(len(ks)), gaps, color=colors, tick_label=[str(k) for k in ks])
    ax.axhline(3, color="green", linestyle="--", alpha=0.5, label="Significance threshold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("k")
    ax.set_ylabel("Gap (random Z-Δ − real Z-Δ)")
    ax.set_title("Is Z Special? (green = yes)")
    ax.legend(fontsize=7)

    # 4. Dimensionality fraction vs effect
    ax = axes[1, 0]
    fracs = [k / d * 100 for k in ks]
    ax.plot(fracs, real_z_sweep, "o-", color="steelblue", label="Real Z-patch Δ")
    ax.plot(fracs, rand_z_sweep, "s--", color="gray", label="Random Z-patch Δ")
    ax.set_xlabel("Fraction of dimensions patched (%)")
    ax.set_ylabel("Answers changed / 20")
    ax.set_title("Effect vs Dimensionality Fraction")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. k=20 random Z-patch histogram
    ax = axes[1, 1]
    ax.hist(rand_z_vals, bins=range(0, 21), color="gray", edgecolor="black", alpha=0.7, label="Random")
    ax.axvline(real_z, color="red", linewidth=2, label=f"Real Z = {real_z}")
    ax.set_xlabel("Answers changed / 20")
    ax.set_ylabel("Count")
    ax.set_title(f"k={K_MAIN} Z-patch: Real vs Random Distribution")
    ax.legend()

    # 6. Overlap histogram
    ax = axes[1, 2]
    ax.hist(overlaps_20, bins=20, color="lightblue", edgecolor="black", alpha=0.7)
    ax.axvline(expected_overlap, color="red", linestyle="--", linewidth=2, label=f"Expected: {expected_overlap:.4f}")
    ax.set_xlabel("Subspace overlap (||V_Z V_R^T||²_F / k)")
    ax.set_ylabel("Count")
    ax.set_title(f"k={K_MAIN} Overlap: Real Z vs Random")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "phase3_controls_k20.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved: {out_path}")

    # ==================================================================
    # Save results
    # ==================================================================
    results = {
        "part1_k20_control": {
            "k": K_MAIN,
            "n_random": N_RANDOM,
            "real_z": k20_result["real_z"],
            "real_zperp": k20_result["real_zperp"],
            "random": k20_result["random"],
            "random_z_mean": float(rand_z_mean),
            "random_z_std": float(rand_z_std),
            "z_score": float(z_score),
            "verdict": verdict,
        },
        "part2_k_sweep": [
            {
                "k": sr["k"],
                "real_z": sr["real_z"],
                "real_zperp": sr["real_zperp"],
                "random_z_mean": float(np.mean([r["z_changed"] for r in sr["random"]])),
                "random_zp_mean": float(np.mean([r["zp_changed"] for r in sr["random"]])),
                "gap": float(np.mean([r["z_changed"] for r in sr["random"]]) - sr["real_z"]["ans_changed"]),
            }
            for sr in sweep_results
        ],
        "part3_overlap": {
            "k": K_MAIN,
            "expected_overlap": float(expected_overlap),
            "measured_overlap_mean": float(np.mean(overlaps_20)),
            "measured_overlap_std": float(np.std(overlaps_20)),
            "n_samples": 50,
        },
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "k_main": K_MAIN,
            "k_sweep": K_SWEEP,
            "n_random": N_RANDOM,
            "n_pairs": len(PAIRS),
            "seed": SEED,
        },
        "baselines": {str(k): v for k, v in baselines.items()},
    }

    out_json = OUTPUT_DIR / "phase3_controls_k20.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_json}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  k=20 Random Control Verdict: {verdict}")
    print(f"  Real Z-patch: {real_z}/20 answers changed")
    print(f"  Random Z-patch: {rand_z_mean:.1f} ± {rand_z_std:.1f}/20 answers changed")
    if verdict == "Z_SPECIAL":
        print(f"\n  CONCLUSION: Z subspace at k=20 IS causally special.")
        print(f"  Random subspaces disrupt answers; Z preserves them.")
        print(f"  The 20-dim reasoning core is a real feature of the model.")
    elif verdict == "INDISTINGUISHABLE":
        print(f"\n  CONCLUSION: Z subspace at k=20 is NOT distinguishable from random.")
        print(f"  The patching asymmetry is a dimensionality artifact at all tested k.")
        print(f"  Need fundamentally different experimental approach.")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
