"""Exp BS: SVD-Truncated Inference — Centered PCA Bottleneck at Equilibrium Layers.

Tests whether the centered Gram rank_90 predicts the operational dimensionality
of reasoning. If accuracy is preserved at k = rank_90 (~20 in 2048-D space),
the remaining 99% of dimensions are chaotic rotation waste.

Key difference from Exp BO (which was catastrophic at ALL k):
- BO used cosine-Gram eigenvectors (contaminated by mean direction)
- BO applied bottleneck at a SINGLE layer
- This uses CENTERED PCA basis and applies at ALL equilibrium layers (L9-L26)

Prediction: k < rank_90 (~20) → catastrophic, k >= rank_90 → preserved.
If confirmed: implies Z-embedding architecture with 100x parameter reduction
in middle layers.
"""

import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import json
import time
import re
import math
import random as pyrandom
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
CACHE_PATH = OUTPUT_DIR / "multilingual_all_layers.npz"
SEED = 42
N_LAYERS = 36
DIM = 2048
MAX_NEW_TOKENS = 128
LANGS_EVAL = ["en", "zh"]
LANGS_CACHE = ["ar", "en", "es", "ja", "ko", "sw", "zh"]

# Layer ranges
EQUILIBRIUM_LAYERS = list(range(9, 27))  # L9-L26 (centered rank_90 = 19-21)
ALL_LAYERS = list(range(0, 36))

# k sweep — prediction: transition at k ~ 20
K_VALUES = [2, 5, 10, 15, 20, 25, 30, 50, 100, 200, 500]

# Test set size (match BO: 4 per category × 5 categories = 20)
N_TEST_PER_CAT = 4

TEMPLATES = {
    "zh": {
        "arithmetic_plus": "计算 {a} + {b} 的值。",
        "arithmetic_times": "计算 {a} × {b} 的值。",
        "combinatorics": "求组合数 C({n}, {k}) 的值。",
        "modular": "{a} 除以 {b} 的余数是多少？",
        "geometry": "一个长方形的长为 {w}，宽为 {h}，求其面积。",
    },
    "en": {
        "arithmetic_plus": "Calculate {a} + {b}.",
        "arithmetic_times": "Calculate {a} × {b}.",
        "combinatorics": "Find the value of C({n}, {k}).",
        "modular": "What is the remainder when {a} is divided by {b}?",
        "geometry": "A rectangle has length {w} and width {h}. Find its area.",
    },
}


def generate_test_problems(n_test=N_TEST_PER_CAT):
    """Generate test problems (identical RNG to BO for comparability)."""
    rng = pyrandom.Random(SEED)
    cats = []

    per_cat = 200 // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        ans = a + b if op == "plus" else a * b
        prompts = {}
        for lang in ["en", "zh"]:
            prompts[lang] = TEMPLATES[lang][f"arithmetic_{op}"].format(a=a, b=b)
        cats.append(("arithmetic", ans, prompts))

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        ans = math.comb(n_val, k_val)
        prompts = {}
        for lang in ["en", "zh"]:
            prompts[lang] = TEMPLATES[lang]["combinatorics"].format(n=n_val, k=k_val)
        cats.append(("combinatorics", ans, prompts))

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        ans = a % b
        prompts = {}
        for lang in ["en", "zh"]:
            prompts[lang] = TEMPLATES[lang]["modular"].format(a=a, b=b)
        cats.append(("modular", ans, prompts))

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        ans = w * h
        prompts = {}
        for lang in ["en", "zh"]:
            prompts[lang] = TEMPLATES[lang]["geometry"].format(w=w, h=h)
        cats.append(("geometry", ans, prompts))

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        cats.append((
            "sequences", ans,
            {
                "en": f"An arithmetic sequence: first term {a1}, common difference {d}. Sum of first {n_terms} terms?",
                "zh": f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
            },
        ))

    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats)))
    rng2.shuffle(indices)
    cats = [cats[i] for i in indices]

    by_cat = {}
    for cat, ans, prompts in cats:
        if cat not in by_cat:
            by_cat[cat] = []
        if len(by_cat[cat]) < n_test:
            by_cat[cat].append((ans, prompts))

    test_set = []
    for cat in by_cat:
        for ans, prompts in by_cat[cat]:
            test_set.append({"category": cat, "answer": ans, "en": prompts["en"], "zh": prompts["zh"]})
    return test_set


def check_answer(text, correct_answer):
    target = str(correct_answer)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return target in numbers


def find_fat(text, correct_answer):
    """Find first-answer-token position."""
    target = str(correct_answer)
    tokens = text.split()
    for i, tok in enumerate(tokens):
        nums = re.findall(r"-?\d+\.?\d*", tok)
        if target in nums:
            return i
    return -1


def detect_lang(text):
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[a-zA-Z]", text))
    if zh_chars > en_chars:
        return "zh"
    elif en_chars > zh_chars:
        return "en"
    return "unk"


# ── PCA computation ───────────────────────────────────────────────────────

def compute_centered_pca(cache_path, n_layers, langs, max_k=500):
    """Compute centered PCA basis at each layer from cached activations.

    Returns per-layer:
        means:     dict[int → np.ndarray (d,)]
        bases:     dict[int → np.ndarray (d, max_k)]  columns = PC directions
        var_fracs: dict[int → np.ndarray (max_k,)]    cumulative variance fraction
    """
    data = np.load(cache_path)
    means = {}
    bases = {}
    var_fracs = {}

    for layer in range(n_layers):
        vecs = [data[f"{lang}_L{layer}"] for lang in langs]
        H = np.vstack(vecs).astype(np.float64)  # (1400, d)

        mu = H.mean(axis=0)
        H_c = H - mu

        # Economy SVD: H_c = U S Vt where Vt is (min(n,d), d)
        _, S, Vt = np.linalg.svd(H_c, full_matrices=False)

        means[layer] = mu.astype(np.float32)
        bases[layer] = Vt[:max_k].T.astype(np.float32)  # (d, max_k)

        eigs = S[:max_k] ** 2
        cum = np.cumsum(eigs)
        total = (S ** 2).sum()
        var_fracs[layer] = (cum / total).astype(np.float32)

    return means, bases, var_fracs


# ── Hook ──────────────────────────────────────────────────────────────────

class CenteredPCAHook:
    """Projects residual stream onto top-k centered PCA directions at one layer."""

    def __init__(self, mean, basis_k, device, dtype):
        """
        mean:    (d,) numpy → tensor on device
        basis_k: (d, k) numpy → tensor on device
        """
        self.mean = torch.from_numpy(mean).to(device=device, dtype=dtype)
        self.basis_k = torch.from_numpy(basis_k).to(device=device, dtype=dtype)

    def __call__(self, module, input, output):
        # Qwen2 decoder layer returns a plain tensor (batch, seq, d)
        hidden = output  # (batch, seq, d)

        # Center → project → reconstruct
        h_c = hidden - self.mean
        coeffs = h_c @ self.basis_k          # (batch, seq, k)
        h_proj = coeffs @ self.basis_k.T     # (batch, seq, d)
        h_new = h_proj + self.mean

        return h_new


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # 1. Compute PCA bases
    print("[1/4] Computing centered PCA bases from cached activations...")
    means, bases, var_fracs = compute_centered_pca(CACHE_PATH, N_LAYERS, LANGS_CACHE)

    # Report variance explained at key k values for equilibrium layers
    print("\n  Variance explained at equilibrium layers:")
    print(f"  {'Layer':>5} | {'k=10':>6} | {'k=20':>6} | {'k=30':>6} | {'k=50':>6} | {'k=100':>6} | {'k=200':>6}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for layer in EQUILIBRIUM_LAYERS:
        vals = []
        for k in [10, 20, 30, 50, 100, 200]:
            idx = min(k, len(var_fracs[layer])) - 1
            vals.append(f"{var_fracs[layer][idx]:.1%}")
        print(f"  L{layer:>3} | {' | '.join(f'{v:>6}' for v in vals)}")

    # 2. Load model
    print("\n[2/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    device = model.device
    dtype = torch.bfloat16

    # 3. Test problems
    print("[3/4] Generating test problems...")
    test_problems = generate_test_problems()
    print(f"  {len(test_problems)} test problems ready")

    # 4. Build condition list
    conditions = [("baseline", [], None)]

    # Main sweep: equilibrium layers (L9-L26), vary k
    for k in K_VALUES:
        conditions.append((f"equil_k{k}", EQUILIBRIUM_LAYERS, k))

    # Control: single layer L16 (peak equilibrium), vary k
    # Direct comparison to BO's single-layer results
    for k in [10, 20, 50, 200]:
        conditions.append((f"single_L16_k{k}", [16], k))

    # Control: all layers, vary k
    for k in [20, 50, 200]:
        conditions.append((f"all_k{k}", ALL_LAYERS, k))

    # Control: build-only (L0-L8), vary k
    for k in [20, 50]:
        conditions.append((f"build_k{k}", list(range(0, 9)), k))

    # Control: output-only (L27-L35), vary k
    for k in [20, 50]:
        conditions.append((f"output_k{k}", list(range(27, 36)), k))

    print(f"\n  Total conditions: {len(conditions)}")

    # 5. Run sweep
    print("\n[4/4] Running SVD-truncation sweep...\n")

    all_results = {}
    pca_metadata = {}

    for i, (name, layers, k) in enumerate(conditions):
        t_cond = time.time()

        # Register hooks
        handles = []
        if layers and k is not None:
            for layer in layers:
                basis_k = bases[layer][:, :k]  # (d, k)
                hook = CenteredPCAHook(means[layer], basis_k, device, dtype)
                h = model.model.layers[layer].register_forward_hook(hook)
                handles.append(h)

        # Evaluate
        results = {}
        for lang in LANGS_EVAL:
            correct = 0
            fats = []
            lang_pres = {"zh": 0, "en": 0, "unk": 0}
            for prob in test_problems:
                inputs = tokenizer(prob[lang], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                gen = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                ok = check_answer(gen, prob["answer"])
                fat = find_fat(gen, prob["answer"])
                if ok:
                    correct += 1
                fats.append(fat)
                dl = detect_lang(gen)
                lang_pres[dl] += 1

            valid_fats = [f for f in fats if f >= 0]
            mfat = float(np.mean(valid_fats)) if valid_fats else -1
            results[lang] = {
                "correct": correct,
                "total": len(test_problems),
                "mean_fat": round(mfat, 1),
                "lang_pres": lang_pres,
            }

        # Remove hooks
        for h in handles:
            h.remove()

        all_results[name] = results
        dt = time.time() - t_cond

        # Compute variance explained for this k at equilibrium layers
        var_info = ""
        if layers and k is not None and layers == EQUILIBRIUM_LAYERS:
            avg_var = np.mean([
                var_fracs[l][min(k, len(var_fracs[l])) - 1]
                for l in layers
            ])
            var_info = f" var={avg_var:.1%}"
            pca_metadata[name] = {
                "k": k,
                "layers": f"L{layers[0]}-L{layers[-1]}",
                "avg_var_explained": round(float(avg_var), 4),
            }

        en_c = results["en"]["correct"]
        zh_c = results["zh"]["correct"]
        en_t = results["en"]["total"]
        en_fat = results["en"]["mean_fat"]
        zh_fat = results["zh"]["mean_fat"]
        print(
            f"  [{i+1:2d}/{len(conditions)}] {name:25s}  "
            f"EN={en_c}/{en_t}  ZH={zh_c}/{en_t}  "
            f"FAT_en={en_fat:5.1f}  FAT_zh={zh_fat:5.1f}  "
            f"({dt:.1f}s){var_info}"
        )

    # 6. Save results
    elapsed = time.time() - t0

    # Collect variance profile for output
    var_profile = {}
    for layer in range(N_LAYERS):
        var_profile[layer] = {
            f"top_{k}": round(float(var_fracs[layer][min(k, len(var_fracs[layer])) - 1]), 4)
            for k in [5, 10, 20, 30, 50, 100, 200, 500]
        }

    output = {
        "experiment": "BS",
        "title": "SVD-Truncated Inference — Centered PCA Bottleneck",
        "hypothesis": "Centered Gram rank_90 (~20) predicts operational dimensionality. k < 20 catastrophic, k >= 20 preserved.",
        "model": MODEL_NAME,
        "dim": DIM,
        "n_layers": N_LAYERS,
        "n_problems": len(test_problems),
        "max_new_tokens": MAX_NEW_TOKENS,
        "runtime_seconds": round(elapsed),
        "conditions": {},
        "variance_profile": var_profile,
        "pca_metadata": pca_metadata,
    }

    for name, layers, k in conditions:
        output["conditions"][name] = {
            "layers": f"L{layers[0]}-L{layers[-1]}" if layers else "none",
            "n_layers_hooked": len(layers),
            "k": k,
            "results": all_results[name],
        }

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY — Equilibrium layers (L9-L26)")
    print("=" * 80)
    print(f"{'Condition':>25s}  {'k':>4}  {'EN':>4}  {'ZH':>4}  {'Var%':>6}  {'FAT_en':>6}  {'FAT_zh':>6}")
    print("-" * 80)

    for name, layers, k in conditions:
        if "equil" in name or name == "baseline":
            r = all_results[name]
            var_str = ""
            if name in pca_metadata:
                var_str = f"{pca_metadata[name]['avg_var_explained']:.1%}"
            k_str = str(k) if k else "-"
            print(
                f"{name:>25s}  {k_str:>4}  "
                f"{r['en']['correct']:>4}  {r['zh']['correct']:>4}  "
                f"{var_str:>6}  {r['en']['mean_fat']:>6.1f}  {r['zh']['mean_fat']:>6.1f}"
            )

    print("\nControls:")
    for name, layers, k in conditions:
        if "equil" not in name and name != "baseline":
            r = all_results[name]
            print(
                f"  {name:>25s}  k={k:>4}  "
                f"EN={r['en']['correct']}/{r['en']['total']}  "
                f"ZH={r['zh']['correct']}/{r['zh']['total']}  "
                f"FAT_en={r['en']['mean_fat']:>5.1f}  FAT_zh={r['zh']['mean_fat']:>5.1f}"
            )

    out_path = OUTPUT_DIR / "expBS_svd_truncation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
