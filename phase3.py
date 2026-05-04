"""Phase 3, Experiment A: Activation Patching for Causal Z Identification.

Run with: MPLBACKEND=Agg .venv_wsl/bin/python phase3.py

Implements the patching protocol from PHASE3_SPEC.md:
  1. Extract Z masks at L32/L33 via multi-head attention SVD (from Phase 2)
  2. Run Chinese prompts through model, store mean-pooled hidden states
  3. For each English prompt, run 4 conditions:
     - baseline (no patch)
     - Z-patch (replace Z-content with Chinese mean Z)
     - Zperp-patch (replace Z⊥-content with Chinese mean Z⊥)
     - full-patch (replace all dims with Chinese mean)
  4. Generate output, extract answer, classify language
  5. Also runs Experiment B: residual update decomposition (piggybacks on extraction)

Depends on: utils.py (get_attn_subspace, get_model_dims, build_multi_head_z_mask)
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
TARGET_LAYERS = [32, 33]         # L32 primary, L33 secondary
K_VALUES = [20, 50]              # skip 78 (Phase 2 showed signal vanishes)
MAX_NEW_TOKENS = 150             # generation budget per problem
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Prompt pairs — reuse Phase 2's set for direct comparability
# ---------------------------------------------------------------------------
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
# Z mask (reuse Phase 2's build_multi_head_z_mask)
# ---------------------------------------------------------------------------
def build_multi_head_z_mask(
    model, layer_idx: int, h: int, GQA: int, d: int, k: int,
) -> torch.Tensor:
    """Top-k SVD of stacked multi-head attention kernels. Returns (k, d)."""
    all_vh = []
    for head in range(h):
        vh = get_attn_subspace(model, layer_idx, h, GQA, d, head, k=k)
        all_vh.append(vh)
    stacked = torch.cat(all_vh, dim=0)  # (h*k, d)
    _, S, Vh_combined = torch.linalg.svd(stacked, full_matrices=False)
    return Vh_combined[:k, :]  # (k, d)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
def cjk_fraction(text: str) -> float:
    """Fraction of non-whitespace characters that are CJK."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    cjk = sum(1 for c in chars if unicodedata.category(c).startswith("Lo"))
    return cjk / len(chars)


def classify_language(text: str) -> str:
    frac = cjk_fraction(text)
    if frac > 0.3:
        return "zh"
    if frac < 0.05:
        return "en"
    return "mixed"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> str:
    """Best-effort numerical/symbolic answer from generated text."""
    # Take first line (model usually answers immediately)
    first_line = text.strip().split("\n")[0].strip()
    # Try to find a number or simple expression
    # Remove common prefixes
    for prefix in ["Answer:", "答案：", "答案:", "The answer is", "= ", "answer is "]:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip()
    # Extract number-like patterns
    match = re.search(r"[-]?\d+(?:[./]\d+)?(?:π|\\pi)?", first_line)
    if match:
        return match.group(0)
    # Return cleaned first line as fallback
    return first_line[:80]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 3: Causal Z Identification via Activation Patching")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {TARGET_LAYERS}, k values: {K_VALUES}")
    print(f"Prompts: {len(PAIRS)}, max_new_tokens: {MAX_NEW_TOKENS}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load model + tokenizer
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

    # ------------------------------------------------------------------
    # Step 1: Build Z masks
    # ------------------------------------------------------------------
    print("\n--- Step 1: Build Z masks ---")
    z_masks = {}  # (layer, k) -> (k, d) tensor
    projectors = {}  # (layer, k) -> (P_Z, P_Zperp) each (d, d)

    for layer in TARGET_LAYERS:
        for k in K_VALUES:
            Vh = build_multi_head_z_mask(model, layer, h, GQA, d, k)
            z_masks[(layer, k)] = Vh
            P_Z = Vh.T @ Vh            # (d, d)
            P_Zp = torch.eye(d) - P_Z  # (d, d)
            projectors[(layer, k)] = (P_Z, P_Zp)
            print(f"  L{layer} k={k}: mask shape {Vh.shape}")

    # ------------------------------------------------------------------
    # Step 2: Extract Chinese mean-pooled hidden states (patch sources)
    #         + extract ALL layer hidden states for Experiment B
    # ------------------------------------------------------------------
    print("\n--- Step 2: Extract Chinese hidden states ---")

    # Hook ALL layers for Experiment B (residual update decomposition)
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h.detach().cpu().squeeze(0)  # (seq, d)
        return hook

    hooks = []
    for li in range(L):
        handle = model.model.layers[li].register_forward_hook(make_hook(f"L{li}"))
        hooks.append(handle)

    # Store: mean-pooled Chinese states at target layers for patching
    zh_mean_states = {}  # (pair_idx, layer) -> (d,)
    # Store: all-layer states for Experiment B
    all_layer_states = {}  # (lang, pair_idx, layer) -> (d,) mean-pooled

    N = len(PAIRS)

    for i, pair in enumerate(tqdm(PAIRS, desc="Chinese forward passes")):
        inputs = tokenizer(pair["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for li in range(L):
            h_state = activations[f"L{li}"].float()  # (T, d)
            mean_pooled = h_state.mean(dim=0)         # (d,)
            all_layer_states[("zh", i, li)] = mean_pooled
            if li in TARGET_LAYERS:
                zh_mean_states[(i, li)] = mean_pooled

    print(f"  Stored {len(zh_mean_states)} Chinese patch-source vectors")
    print(f"  Stored {L * N} Chinese all-layer vectors for Experiment B")

    # Extract English all-layer states (for Experiment B)
    for i, pair in enumerate(tqdm(PAIRS, desc="English forward passes")):
        inputs = tokenizer(pair["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for li in range(L):
            h_state = activations[f"L{li}"].float()
            all_layer_states[("en", i, li)] = h_state.mean(dim=0)

    # Remove hooks — we'll re-register targeted ones for patching
    for handle in hooks:
        handle.remove()

    # ------------------------------------------------------------------
    # Step 3: Patching experiment
    # ------------------------------------------------------------------
    print("\n--- Step 3: Activation patching ---")

    CONDITIONS = ["baseline", "z_patch", "zperp_patch", "full_patch"]
    all_results = []  # list of dicts

    for layer in TARGET_LAYERS:
        for k in K_VALUES:
            P_Z, P_Zp = projectors[(layer, k)]

            print(f"\n  Config: L{layer} k={k}")

            for i, pair in enumerate(tqdm(
                PAIRS, desc=f"L{layer} k={k}", leave=False,
            )):
                # Compute patch vectors from Chinese mean state
                zh_mean = zh_mean_states[(i, layer)]      # (d,)
                zh_Z = P_Z @ zh_mean       # Chinese Z-content
                zh_Zp = P_Zp @ zh_mean     # Chinese Z⊥-content

                for cond in CONDITIONS:
                    # Build the hook
                    if cond == "baseline":
                        patch_hook = None
                    elif cond == "z_patch":
                        # Keep English Z⊥, inject Chinese Z
                        zh_Z_dev = zh_Z.to(model.device).half()
                        P_Zp_dev = P_Zp.to(model.device).half()
                        def _hook_z(module, input, output,
                                    _pzp=P_Zp_dev, _zhz=zh_Z_dev):
                            # output is a single Tensor (batch, seq, d)
                            h = output if isinstance(output, torch.Tensor) else output[0]
                            patched = h.clone()
                            if patched.dim() == 3:
                                for t in range(patched.shape[1]):
                                    patched[0, t, :] = _pzp @ patched[0, t, :] + _zhz
                            else:
                                for t in range(patched.shape[0]):
                                    patched[t, :] = _pzp @ patched[t, :] + _zhz
                            return patched
                        patch_hook = _hook_z
                    elif cond == "zperp_patch":
                        # Keep English Z, inject Chinese Z⊥
                        zh_Zp_dev = zh_Zp.to(model.device).half()
                        P_Z_dev = P_Z.to(model.device).half()
                        def _hook_zp(module, input, output,
                                     _pz=P_Z_dev, _zhzp=zh_Zp_dev):
                            h = output if isinstance(output, torch.Tensor) else output[0]
                            patched = h.clone()
                            if patched.dim() == 3:
                                for t in range(patched.shape[1]):
                                    patched[0, t, :] = _pz @ patched[0, t, :] + _zhzp
                            else:
                                for t in range(patched.shape[0]):
                                    patched[t, :] = _pz @ patched[t, :] + _zhzp
                            return patched
                        patch_hook = _hook_zp
                    elif cond == "full_patch":
                        # Replace everything with Chinese mean
                        zh_full_dev = zh_mean.to(model.device).half()
                        def _hook_full(module, input, output,
                                       _zhf=zh_full_dev):
                            h = output if isinstance(output, torch.Tensor) else output[0]
                            patched = h.clone()
                            if patched.dim() == 3:
                                for t in range(patched.shape[1]):
                                    patched[0, t, :] = _zhf
                            else:
                                for t in range(patched.shape[0]):
                                    patched[t, :] = _zhf
                            return patched
                        patch_hook = _hook_full

                    # Register hook if needed
                    handle = None
                    if patch_hook is not None:
                        handle = model.model.layers[layer].register_forward_hook(
                            patch_hook
                        )

                    # Generate
                    inputs = tokenizer(
                        pair["en"], return_tensors="pt",
                    ).to(model.device)
                    with torch.no_grad():
                        gen_ids = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,  # greedy for reproducibility
                            temperature=1.0,
                        )

                    if handle is not None:
                        handle.remove()

                    # Decode only the NEW tokens
                    prompt_len = inputs["input_ids"].shape[1]
                    new_ids = gen_ids[0, prompt_len:]
                    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)

                    # Analyze
                    answer = extract_answer(raw_output)
                    lang = classify_language(raw_output)

                    result = {
                        "pair_idx": i,
                        "layer": layer,
                        "k": k,
                        "condition": cond,
                        "category": pair["category"],
                        "expected": pair["answer"],
                        "extracted_answer": answer,
                        "output_language": lang,
                        "raw_output": raw_output[:500],
                    }
                    all_results.append(result)

                    if cond == "baseline":
                        baseline_answer = answer

    # ------------------------------------------------------------------
    # Step 4: Experiment B — Residual update decomposition
    # ------------------------------------------------------------------
    print("\n--- Step 4: Residual update decomposition ---")

    # Use L32 k=50 basis as the fixed Rosetta Stone
    P_Z_ref, P_Zp_ref = projectors[(32, 50)]

    # For each layer transition k -> k+1, compute Z vs Z⊥ update magnitude
    update_ratios = {"zh": np.zeros(L - 1), "en": np.zeros(L - 1)}
    update_z_norms = {"zh": np.zeros(L - 1), "en": np.zeros(L - 1)}
    update_zp_norms = {"zh": np.zeros(L - 1), "en": np.zeros(L - 1)}

    for lang in ["zh", "en"]:
        for li in range(L - 1):
            z_norms = []
            zp_norms = []
            for i in range(N):
                h_curr = all_layer_states[(lang, i, li)]      # (d,)
                h_next = all_layer_states[(lang, i, li + 1)]   # (d,)
                delta = h_next - h_curr                         # (d,)

                dz = P_Z_ref @ delta
                dzp = P_Zp_ref @ delta
                z_norms.append(torch.norm(dz).item())
                zp_norms.append(torch.norm(dzp).item())

            mean_z = np.mean(z_norms)
            mean_zp = np.mean(zp_norms)
            update_z_norms[lang][li] = mean_z
            update_zp_norms[lang][li] = mean_zp
            update_ratios[lang][li] = mean_z / mean_zp if mean_zp > 0 else 0.0

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R(k) vs layer
    ax = axes[0]
    ax.plot(range(L - 1), update_ratios["zh"], "o-", label="Chinese", markersize=3)
    ax.plot(range(L - 1), update_ratios["en"], "s-", label="English", markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="R=1 (equal)")
    for layer in TARGET_LAYERS:
        if layer < L - 1:
            ax.axvline(layer, color="red", linestyle=":", alpha=0.3)
    ax.set_xlabel("Layer transition k → k+1")
    ax.set_ylabel("R(k) = ||Δh_Z|| / ||Δh_Z⊥||")
    ax.set_title("Update decomposition: reasoning vs language per layer")
    ax.legend()

    # Absolute norms
    ax = axes[1]
    ax.plot(range(L - 1), update_z_norms["zh"], "o-", label="zh Z-norm", markersize=3)
    ax.plot(range(L - 1), update_z_norms["en"], "s-", label="en Z-norm", markersize=3)
    ax.plot(range(L - 1), update_zp_norms["zh"], "o--", label="zh Z⊥-norm",
            markersize=3, alpha=0.6)
    ax.plot(range(L - 1), update_zp_norms["en"], "s--", label="en Z⊥-norm",
            markersize=3, alpha=0.6)
    for layer in TARGET_LAYERS:
        if layer < L - 1:
            ax.axvline(layer, color="red", linestyle=":", alpha=0.3)
    ax.set_xlabel("Layer transition k → k+1")
    ax.set_ylabel("||Δh|| (mean across prompts)")
    ax.set_title("Absolute update norms in Z vs Z⊥")
    ax.legend(fontsize=8)

    # Cross-lingual asymmetry
    ax = axes[2]
    diff = update_ratios["zh"] - update_ratios["en"]
    ax.bar(range(L - 1), diff, width=0.8, alpha=0.7)
    ax.axhline(0, color="gray", linestyle="-")
    for layer in TARGET_LAYERS:
        if layer < L - 1:
            ax.axvline(layer, color="red", linestyle=":", alpha=0.3)
    ax.set_xlabel("Layer transition k → k+1")
    ax.set_ylabel("R_zh(k) - R_en(k)")
    ax.set_title("Cross-lingual asymmetry in update decomposition")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_update_decomposition.png", dpi=150)
    print(f"  Saved: phase3_update_decomposition.png")

    # ------------------------------------------------------------------
    # Step 5: Compute summary metrics + save
    # ------------------------------------------------------------------
    print("\n--- Step 5: Summary ---")

    # Save raw results
    with open(OUTPUT_DIR / "phase3_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: phase3_results.json ({len(all_results)} entries)")

    # Save Experiment B data
    exp_b_data = {
        "z_norms_zh": update_z_norms["zh"].tolist(),
        "z_norms_en": update_z_norms["en"].tolist(),
        "zp_norms_zh": update_zp_norms["zh"].tolist(),
        "zp_norms_en": update_zp_norms["en"].tolist(),
        "ratios_zh": update_ratios["zh"].tolist(),
        "ratios_en": update_ratios["en"].tolist(),
        "basis_layer": 32,
        "basis_k": 50,
    }
    with open(OUTPUT_DIR / "phase3_update_decomposition.json", "w") as f:
        json.dump(exp_b_data, f, indent=2)

    # Aggregate patching results
    print("\n" + "=" * 90)
    print("PHASE 3A: PATCHING RESULTS")
    print("=" * 90)

    for layer in TARGET_LAYERS:
        for k in K_VALUES:
            subset = [r for r in all_results
                      if r["layer"] == layer and r["k"] == k]

            print(f"\n  Config: L{layer} k={k}")
            print(f"  {'Condition':<15} {'Ans Changed':>12} {'Lang=en':>10} "
                  f"{'Lang=zh':>10} {'Lang=mix':>10}")
            print("  " + "-" * 60)

            # Get baseline answers for comparison
            baselines = {r["pair_idx"]: r["extracted_answer"]
                         for r in subset if r["condition"] == "baseline"}

            for cond in CONDITIONS:
                cond_results = [r for r in subset if r["condition"] == cond]
                n_total = len(cond_results)
                if n_total == 0:
                    continue

                # Answer changed vs baseline
                ans_changed = sum(
                    1 for r in cond_results
                    if r["extracted_answer"] != baselines.get(r["pair_idx"], "")
                )
                # Language distribution
                n_en = sum(1 for r in cond_results if r["output_language"] == "en")
                n_zh = sum(1 for r in cond_results if r["output_language"] == "zh")
                n_mix = sum(1 for r in cond_results if r["output_language"] == "mixed")

                print(f"  {cond:<15} {ans_changed:>5}/{n_total:<5} "
                      f"{n_en:>5}/{n_total:<4} "
                      f"{n_zh:>5}/{n_total:<4} "
                      f"{n_mix:>5}/{n_total}")

    # Detailed per-problem table for best config (L32 k=50)
    print(f"\n{'=' * 90}")
    print("DETAILED TABLE: L32 k=50")
    print(f"{'=' * 90}")
    print(f"{'#':>2} {'Cat':<13} {'Expected':<10} {'Baseline':<10} "
          f"{'Z-patch':<10} {'Z⊥-patch':<10} {'Full':<10} "
          f"{'BL lang':<8} {'Zp lang':<8} {'Z⊥p lang':<8}")
    print("-" * 110)

    best_subset = [r for r in all_results
                   if r["layer"] == 32 and r["k"] == 50]

    for i in range(N):
        pair_results = {r["condition"]: r for r in best_subset
                        if r["pair_idx"] == i}
        if not pair_results:
            continue

        bl = pair_results.get("baseline", {})
        zp = pair_results.get("z_patch", {})
        zpp = pair_results.get("zperp_patch", {})
        fp = pair_results.get("full_patch", {})

        print(f"{i:>2} {PAIRS[i]['category']:<13} "
              f"{PAIRS[i]['answer']:<10} "
              f"{bl.get('extracted_answer', '?'):<10} "
              f"{zp.get('extracted_answer', '?'):<10} "
              f"{zpp.get('extracted_answer', '?'):<10} "
              f"{fp.get('extracted_answer', '?'):<10} "
              f"{bl.get('output_language', '?'):<8} "
              f"{zp.get('output_language', '?'):<8} "
              f"{zpp.get('output_language', '?'):<8}")

    # Experiment B summary
    print(f"\n{'=' * 90}")
    print("EXPERIMENT B: UPDATE DECOMPOSITION SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Basis: L32 k=50 multi-head Z mask")
    print(f"  R(k) = ||Δh_Z|| / ||Δh_Z⊥|| averaged over {N} prompts")
    print(f"\n  {'Layer':>5} {'R_zh':>8} {'R_en':>8} {'Diff':>8}")
    print("  " + "-" * 30)
    for li in range(L - 1):
        diff_val = update_ratios["zh"][li] - update_ratios["en"][li]
        marker = " <<<" if li in TARGET_LAYERS else ""
        print(f"  {li:>2}→{li+1:<2} {update_ratios['zh'][li]:>8.4f} "
              f"{update_ratios['en'][li]:>8.4f} {diff_val:>+8.4f}{marker}")

    # Peak reasoning layers
    peak_zh = int(np.argmax(update_ratios["zh"]))
    peak_en = int(np.argmax(update_ratios["en"]))
    print(f"\n  Peak R(k): zh at layer {peak_zh}→{peak_zh+1} "
          f"({update_ratios['zh'][peak_zh]:.4f}), "
          f"en at layer {peak_en}→{peak_en+1} "
          f"({update_ratios['en'][peak_en]:.4f})")

    ever_z_dominant_zh = any(r > 1.0 for r in update_ratios["zh"])
    ever_z_dominant_en = any(r > 1.0 for r in update_ratios["en"])
    print(f"  Any layer with R > 1 (Z-dominated update)?  "
          f"zh={'YES' if ever_z_dominant_zh else 'NO'}  "
          f"en={'YES' if ever_z_dominant_en else 'NO'}")
    if not ever_z_dominant_zh and not ever_z_dominant_en:
        print("  → Z is emergent: no single layer has a pure reasoning phase.")
        print("    The representation is built by 30+ layers of mixed computation.")

    print(f"\nDone. All results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
