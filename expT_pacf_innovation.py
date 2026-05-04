"""Experiment T: PACF Innovation Analysis of MLP Deltas

For each layer L in L9-L26, capture:
  - MLP delta (the output of the MLP block = its residual stream contribution)
  - MLP input (post-attention residual = what enters the MLP)

Then: regress delta on input (OLS). The residual = INNOVATION — what this layer
uniquely adds that can't be predicted linearly from its input.

Key outputs:
  1. R² per layer: how much of the MLP delta is "inherited" (linearly predictable from input)?
  2. Innovation norm profile: where is genuinely NEW computation happening?
  3. Language direction in innovations vs raw deltas: does the zh/en signal live in
     the inherited or novel component?
  4. Math vs non-math innovation profiles (prediction: math = concentrated, non-math = distributed)

This is the PACF analog for transformer layers. In time series: PACF removes the
contribution of intermediate lags. Here: the "input" to layer L already contains
the accumulated output of layers 0..L-1, so regressing delta on input removes
the inherited component.

Numpy-heavy, no generation needed. Uses cached prefill hidden states.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_NAME = 'Qwen/Qwen2.5-3B'
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

n_layers = model.config.num_hidden_layers
d = model.config.hidden_size
N_TRAIN = 200
BATCH_SIZE = 16
ANALYSIS_LAYERS = list(range(9, 27))  # L9-L26, the "patty"

print(f"Model: {MODEL_NAME} ({n_layers} layers, d={d})")
print(f"Analysis layers: L{ANALYSIS_LAYERS[0]}-L{ANALYSIS_LAYERS[-1]}")
t0 = time.time()


# =============================================================================
# Problem generation (same as all prior experiments)
# =============================================================================
def generate_pca_problems(n=200, seed=42):
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
        problems.append({"zh": zh, "en": en})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                          "en": f"Find the value of C({n_val}, {k_val})."})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？",
                          "en": f"What is the remainder when {a} is divided by {b}?"})
    for _ in range(per_cat):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                          "en": f"A rectangle has length {w} and width {h}. Find its area."})
    for _ in range(per_cat):
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。",
                          "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_terms} terms."})
    return problems


# Also add non-math problems for comparison
nonmath_problems = [
    {"zh": "请翻译成英文：今天天气很好。", "en": "Translate to Chinese: The weather is nice today."},
    {"zh": "请翻译成英文：我喜欢编程。", "en": "Translate to Chinese: I love programming."},
    {"zh": "请翻译成英文：数学是美丽的。", "en": "Translate to Chinese: Math is beautiful."},
    {"zh": "写一首关于春天的短诗。", "en": "Write a short poem about spring."},
    {"zh": "法国的首都是哪里？", "en": "What is the capital of France?"},
    {"zh": "列出前五个质数。", "en": "List the first five prime numbers."},
    {"zh": "用Python写一个阶乘函数。", "en": "Write a factorial function in Python."},
    {"zh": "解释什么是递归。", "en": "Explain what recursion is."},
    {"zh": "地球到月球的距离是多少？", "en": "What is the distance from Earth to the Moon?"},
    {"zh": "描述光合作用的过程。", "en": "Describe the process of photosynthesis."},
    {"zh": "什么是机器学习？", "en": "What is machine learning?"},
    {"zh": "列出太阳系八大行星。", "en": "List the eight planets of the solar system."},
    {"zh": "DNA代表什么？", "en": "What does DNA stand for?"},
    {"zh": "解释供给和需求的关系。", "en": "Explain the relationship between supply and demand."},
    {"zh": "写一个Python冒泡排序。", "en": "Write a bubble sort in Python."},
    {"zh": "什么是量子计算？", "en": "What is quantum computing?"},
    {"zh": "描述水循环的过程。", "en": "Describe the water cycle."},
    {"zh": "什么是GDP？", "en": "What is GDP?"},
    {"zh": "解释牛顿第三定律。", "en": "Explain Newton's third law."},
    {"zh": "什么是区块链？", "en": "What is blockchain?"},
]


# =============================================================================
# Step 1: Extract MLP deltas AND inputs for all layers, all problems
# =============================================================================
print("=" * 70)
print("STEP 1: Extracting MLP deltas and inputs across L9-L26")
print("=" * 70)

def extract_mlp_data(prompts, label=""):
    """Extract MLP delta and MLP input for each layer in ANALYSIS_LAYERS.

    Returns: dict[layer] -> {"delta": (N, d), "input": (N, d)} as float32 numpy.

    MLP input = post-attention residual stream (= what enters the MLP).
    MLP delta = MLP output (the residual contribution).

    In Qwen2: each layer does:
        residual = hidden_states
        hidden_states = attn(layernorm(hidden_states)) + residual   # post-attn residual
        residual = hidden_states
        hidden_states = mlp(layernorm(hidden_states)) + residual    # post-mlp residual

    So: MLP input (pre-layernorm) = hidden_states after attention.
    MLP delta = mlp(layernorm(hidden_states)).

    We hook:
      - mlp: captures delta (output of MLP block)
      - mlp input[0]: this is post-layernorm, not what we want for regression

    Actually, for PACF the right regression is: delta_L ~ f(residual_entering_MLP_L).
    The residual entering MLP = post-attention hidden state = input to post_attention_layernorm.

    Better approach: hook the layer's forward to capture intermediate states.
    Simplest: use output_hidden_states=True to get post-layer residuals, then
    MLP input at layer L = hidden_state[L] after attention = we need a finer hook.

    Let's hook both the MLP (for delta) and use a pre-hook on post_attention_layernorm
    (for the residual stream entering the MLP sub-block).
    """
    layer_data = {li: {"deltas": [], "inputs": []} for li in ANALYSIS_LAYERS}

    # Storage for current forward pass
    captures = {}
    handles = []

    for li in ANALYSIS_LAYERS:
        layer = model.model.layers[li]

        # Hook MLP output (= delta)
        def make_mlp_hook(layer_idx):
            def hook(module, inp, out):
                captures.setdefault(layer_idx, {})["delta"] = out.detach().float()
            return hook
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))

        # Hook the input to post_attention_layernorm (= residual stream before MLP)
        # In Qwen2, this is: post_attention_layernorm.forward(hidden_states)
        # The input to this layernorm IS the post-attention residual stream
        def make_ln_hook(layer_idx):
            def hook(module, inp):
                captures.setdefault(layer_idx, {})["input"] = inp[0].detach().float()
            return hook
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(make_ln_hook(li)))

    n_total = len(prompts)
    for i in range(0, n_total, BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        captures.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        attn_mask = inputs["attention_mask"]
        with torch.no_grad():
            model(**inputs)

        last_idx = attn_mask.sum(dim=1) - 1

        for li in ANALYSIS_LAYERS:
            delta = captures[li]["delta"]  # (batch, seq, d)
            inp = captures[li]["input"]    # (batch, seq, d)
            for j in range(delta.shape[0]):
                layer_data[li]["deltas"].append(delta[j, last_idx[j]].cpu().numpy())
                layer_data[li]["inputs"].append(inp[j, last_idx[j]].cpu().numpy())

    for handle in handles:
        handle.remove()

    # Stack into arrays
    for li in ANALYSIS_LAYERS:
        layer_data[li]["deltas"] = np.stack(layer_data[li]["deltas"])  # (N, d)
        layer_data[li]["inputs"] = np.stack(layer_data[li]["inputs"])  # (N, d)

    print(f"  {label}: extracted {n_total} samples × {len(ANALYSIS_LAYERS)} layers")
    return layer_data


# Generate math problems
math_problems = generate_pca_problems(N_TRAIN, seed=42)

# Extract for math (both languages)
math_zh = [p["zh"] for p in math_problems]
math_en = [p["en"] for p in math_problems]
nonmath_zh = [p["zh"] for p in nonmath_problems]
nonmath_en = [p["en"] for p in nonmath_problems]

data_math_zh = extract_mlp_data(math_zh, "math_zh")
data_math_en = extract_mlp_data(math_en, "math_en")
data_nonmath_zh = extract_mlp_data(nonmath_zh, "nonmath_zh")
data_nonmath_en = extract_mlp_data(nonmath_en, "nonmath_en")

print(f"Extraction done in {time.time()-t0:.1f}s")


# =============================================================================
# Step 2: PACF-style regression — delta_L = A @ input_L + residual
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: OLS regression delta ~ input (innovation decomposition)")
print("=" * 70)

def compute_innovations(layer_data, label=""):
    """For each layer: regress delta on input, return R², innovations, norms."""
    results = {}

    for li in ANALYSIS_LAYERS:
        delta = layer_data[li]["deltas"]   # (N, d)
        inp = layer_data[li]["inputs"]     # (N, d)
        N = delta.shape[0]

        # OLS: delta = inp @ W + b + residual
        # Add bias term: X = [inp, 1]
        X = np.hstack([inp, np.ones((N, 1))])  # (N, d+1)

        # Solve via normal equations with ridge regularization (tiny, for stability)
        # W = (X'X + λI)^{-1} X'Y
        # But d=2048, N=200 or 20 → underdetermined for math! Use min-norm solution.
        # When N < d: use delta @ X' @ (X @ X')^{-1} X instead
        # Or just use np.linalg.lstsq which handles rank-deficient cases.

        # For N=200, d=2048: underdetermined. lstsq gives min-norm solution.
        # For N=20 (nonmath): very underdetermined.
        #
        # The R² from min-norm OLS on underdetermined systems will be 1.0 (perfect fit).
        # That's not informative.
        #
        # BETTER APPROACH: Don't regress delta ~ input with full d×d mapping.
        # Instead, project both delta and input onto a shared low-rank basis,
        # OR use a scalar regression approach:
        #
        # For PACF, what matters is: how much of delta_L's DIRECTION is predictable
        # from input_L? Not the full d-dimensional mapping.
        #
        # Approach:
        #   1. Compute cosine similarity between delta and input per-sample
        #   2. Project delta onto input direction: inherited = (delta·input/||input||²) * input
        #   3. Innovation = delta - inherited (component orthogonal to input direction)
        #   4. R² = ||inherited||² / ||delta||² (fraction of delta variance along input)

        # Per-sample: project delta onto input direction
        # inherited_i = (delta_i · input_i / ||input_i||²) * input_i
        inp_norm_sq = np.sum(inp ** 2, axis=1, keepdims=True) + 1e-10  # (N, 1)
        proj_coeff = np.sum(delta * inp, axis=1, keepdims=True) / inp_norm_sq  # (N, 1)
        inherited = proj_coeff * inp  # (N, d) — the component of delta along input
        innovation = delta - inherited  # (N, d) — the orthogonal residual

        # Norms
        delta_norms = np.linalg.norm(delta, axis=1)          # (N,)
        inherited_norms = np.linalg.norm(inherited, axis=1)   # (N,)
        innovation_norms = np.linalg.norm(innovation, axis=1) # (N,)

        # R² = fraction of delta norm² explained by projection onto input
        r2_per_sample = inherited_norms**2 / (delta_norms**2 + 1e-10)

        # Cosine similarity delta vs input
        cos_sim = np.sum(delta * inp, axis=1) / (delta_norms * np.sqrt(inp_norm_sq.squeeze()) + 1e-10)

        results[li] = {
            "innovation": innovation,          # (N, d) — KEEP for language direction analysis
            "inherited": inherited,            # (N, d)
            "r2_mean": float(r2_per_sample.mean()),
            "r2_std": float(r2_per_sample.std()),
            "delta_norm_mean": float(delta_norms.mean()),
            "inherited_norm_mean": float(inherited_norms.mean()),
            "innovation_norm_mean": float(innovation_norms.mean()),
            "cos_sim_mean": float(cos_sim.mean()),
            "cos_sim_std": float(cos_sim.std()),
            "proj_coeff_mean": float(proj_coeff.mean()),
            "proj_coeff_std": float(proj_coeff.std()),
        }

        print(f"  L{li}: R²={r2_per_sample.mean():.4f} | "
              f"δ_norm={delta_norms.mean():.1f} | "
              f"inherited={inherited_norms.mean():.1f} | "
              f"innovation={innovation_norms.mean():.1f} | "
              f"cos(δ,inp)={cos_sim.mean():.3f}")

    return results


print("\n--- Math (ZH) ---")
innov_math_zh = compute_innovations(data_math_zh, "math_zh")

print("\n--- Math (EN) ---")
innov_math_en = compute_innovations(data_math_en, "math_en")

print("\n--- Non-Math (ZH) ---")
innov_nonmath_zh = compute_innovations(data_nonmath_zh, "nonmath_zh")

print("\n--- Non-Math (EN) ---")
innov_nonmath_en = compute_innovations(data_nonmath_en, "nonmath_en")


# =============================================================================
# Step 3: Language direction in innovations vs raw deltas
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Language direction — innovations vs raw deltas vs inherited")
print("=" * 70)

def lang_direction_analysis(data_zh, data_en, innov_zh, innov_en, label=""):
    """Compare where the zh/en mean-difference signal lives:
    in the raw delta, the inherited component, or the innovation."""
    results = {}

    for li in ANALYSIS_LAYERS:
        delta_zh = data_zh[li]["deltas"]  # (N_zh, d)
        delta_en = data_en[li]["deltas"]  # (N_en, d)
        innov_zh_arr = innov_zh[li]["innovation"]  # (N_zh, d)
        innov_en_arr = innov_en[li]["innovation"]  # (N_en, d)
        inher_zh = innov_zh[li]["inherited"]  # (N_zh, d)
        inher_en = innov_en[li]["inherited"]  # (N_en, d)

        # Mean difference directions
        raw_diff = delta_zh.mean(0) - delta_en.mean(0)
        innov_diff = innov_zh_arr.mean(0) - innov_en_arr.mean(0)
        inher_diff = inher_zh.mean(0) - inher_en.mean(0)

        raw_norm = np.linalg.norm(raw_diff)
        innov_norm = np.linalg.norm(innov_diff)
        inher_norm = np.linalg.norm(inher_diff)

        # What fraction of the raw language signal is in innovations vs inherited?
        # Project raw_diff onto innovation and inherited subspaces
        # But simpler: just compare norms
        # raw_diff ≈ innov_diff + inher_diff (by linearity of mean)
        # Check: is the language signal carried by innovation or inherited component?

        # Cosine between raw and innovation language directions
        cos_raw_innov = np.dot(raw_diff, innov_diff) / (raw_norm * innov_norm + 1e-10)
        cos_raw_inher = np.dot(raw_diff, inher_diff) / (raw_norm * inher_norm + 1e-10)

        # Cohen's d for innovations
        if innov_norm > 1e-10:
            innov_unit = innov_diff / innov_norm
            proj_zh = innov_zh_arr @ innov_unit
            proj_en = innov_en_arr @ innov_unit
            cohens_d_innov = float((proj_zh.mean() - proj_en.mean()) /
                                   np.sqrt((proj_zh.std()**2 + proj_en.std()**2) / 2 + 1e-10))
        else:
            cohens_d_innov = 0.0

        # Cohen's d for raw delta (for comparison)
        if raw_norm > 1e-10:
            raw_unit = raw_diff / raw_norm
            proj_zh_raw = delta_zh @ raw_unit
            proj_en_raw = delta_en @ raw_unit
            cohens_d_raw = float((proj_zh_raw.mean() - proj_en_raw.mean()) /
                                  np.sqrt((proj_zh_raw.std()**2 + proj_en_raw.std()**2) / 2 + 1e-10))
        else:
            cohens_d_raw = 0.0

        results[li] = {
            "raw_lang_norm": float(raw_norm),
            "innov_lang_norm": float(innov_norm),
            "inher_lang_norm": float(inher_norm),
            "cos_raw_innov": float(cos_raw_innov),
            "cos_raw_inher": float(cos_raw_inher),
            "cohens_d_raw": cohens_d_raw,
            "cohens_d_innov": cohens_d_innov,
            "innov_frac_of_raw": float(innov_norm / (raw_norm + 1e-10)),
        }

        print(f"  L{li}: raw_d={cohens_d_raw:.1f} innov_d={cohens_d_innov:.1f} | "
              f"raw_norm={raw_norm:.1f} innov_norm={innov_norm:.1f} inher_norm={inher_norm:.1f} | "
              f"cos(raw,innov)={cos_raw_innov:.3f}")

    return results


print("\n--- Math: language signal decomposition ---")
lang_math = lang_direction_analysis(data_math_zh, data_math_en, innov_math_zh, innov_math_en, "math")

print("\n--- Non-Math: language signal decomposition ---")
lang_nonmath = lang_direction_analysis(data_nonmath_zh, data_nonmath_en, innov_nonmath_zh, innov_nonmath_en, "nonmath")


# =============================================================================
# Step 4: Innovation profile comparison — math vs non-math
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Innovation profiles — math vs non-math")
print("=" * 70)

print(f"\n{'Layer':<8} {'Math_innov':<12} {'NonM_innov':<12} {'Math_R²':<10} {'NonM_R²':<10} {'Math_cos':<10} {'NonM_cos':<10}")
print("-" * 72)

profile_comparison = {}
for li in ANALYSIS_LAYERS:
    # Average math across both languages
    math_innov = (innov_math_zh[li]["innovation_norm_mean"] + innov_math_en[li]["innovation_norm_mean"]) / 2
    math_r2 = (innov_math_zh[li]["r2_mean"] + innov_math_en[li]["r2_mean"]) / 2
    math_cos = (innov_math_zh[li]["cos_sim_mean"] + innov_math_en[li]["cos_sim_mean"]) / 2

    nonm_innov = (innov_nonmath_zh[li]["innovation_norm_mean"] + innov_nonmath_en[li]["innovation_norm_mean"]) / 2
    nonm_r2 = (innov_nonmath_zh[li]["r2_mean"] + innov_nonmath_en[li]["r2_mean"]) / 2
    nonm_cos = (innov_nonmath_zh[li]["cos_sim_mean"] + innov_nonmath_en[li]["cos_sim_mean"]) / 2

    ratio = math_innov / (nonm_innov + 1e-10)

    profile_comparison[li] = {
        "math_innovation_norm": float(math_innov),
        "nonmath_innovation_norm": float(nonm_innov),
        "math_r2": float(math_r2),
        "nonmath_r2": float(nonm_r2),
        "math_cos_delta_input": float(math_cos),
        "nonmath_cos_delta_input": float(nonm_cos),
        "innovation_ratio_math_over_nonmath": float(ratio),
    }

    print(f"  L{li:<4} {math_innov:<12.1f} {nonm_innov:<12.1f} {math_r2:<10.4f} {nonm_r2:<10.4f} {math_cos:<10.3f} {nonm_cos:<10.3f}")


# =============================================================================
# Step 5: Multi-layer regression — does layer L's delta correlate with L-1's delta?
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Cross-layer delta correlation (delta_L ~ delta_{L-1})")
print("=" * 70)

def cross_layer_correlation(layer_data, label=""):
    """Cosine similarity between consecutive layers' MLP deltas."""
    results = {}
    for i, li in enumerate(ANALYSIS_LAYERS[1:], 1):
        prev_li = ANALYSIS_LAYERS[i-1]
        delta_curr = layer_data[li]["deltas"]    # (N, d)
        delta_prev = layer_data[prev_li]["deltas"]  # (N, d)

        # Per-sample cosine similarity
        norms_curr = np.linalg.norm(delta_curr, axis=1, keepdims=True) + 1e-10
        norms_prev = np.linalg.norm(delta_prev, axis=1, keepdims=True) + 1e-10
        cos_sims = np.sum((delta_curr / norms_curr) * (delta_prev / norms_prev), axis=1)

        results[f"L{prev_li}->L{li}"] = {
            "cos_mean": float(cos_sims.mean()),
            "cos_std": float(cos_sims.std()),
        }
    return results


# Combine math zh+en
combined_math = {li: {"deltas": np.vstack([data_math_zh[li]["deltas"], data_math_en[li]["deltas"]])}
                 for li in ANALYSIS_LAYERS}
combined_nonmath = {li: {"deltas": np.vstack([data_nonmath_zh[li]["deltas"], data_nonmath_en[li]["deltas"]])}
                    for li in ANALYSIS_LAYERS}

print("\n--- Math: consecutive delta correlation ---")
cross_math = cross_layer_correlation(combined_math, "math")
for k, v in cross_math.items():
    print(f"  {k}: cos={v['cos_mean']:.3f} ± {v['cos_std']:.3f}")

print("\n--- Non-Math: consecutive delta correlation ---")
cross_nonmath = cross_layer_correlation(combined_nonmath, "nonmath")
for k, v in cross_nonmath.items():
    print(f"  {k}: cos={v['cos_mean']:.3f} ± {v['cos_std']:.3f}")


# =============================================================================
# Save results
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

def serialize_innovations(innov_dict):
    """Strip numpy arrays, keep only scalar stats."""
    return {str(li): {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
            for li, d in innov_dict.items()}

output = {
    "experiment": "T: PACF Innovation Analysis",
    "model": MODEL_NAME,
    "n_math_problems": N_TRAIN,
    "n_nonmath_problems": len(nonmath_problems),
    "analysis_layers": ANALYSIS_LAYERS,
    "method": "Project MLP delta onto MLP input direction. Inherited = projection. Innovation = orthogonal residual.",
    "innovations": {
        "math_zh": serialize_innovations(innov_math_zh),
        "math_en": serialize_innovations(innov_math_en),
        "nonmath_zh": serialize_innovations(innov_nonmath_zh),
        "nonmath_en": serialize_innovations(innov_nonmath_en),
    },
    "language_direction_decomposition": {
        "math": {str(li): v for li, v in lang_math.items()},
        "nonmath": {str(li): v for li, v in lang_nonmath.items()},
    },
    "profile_comparison": {str(li): v for li, v in profile_comparison.items()},
    "cross_layer_delta_correlation": {
        "math": cross_math,
        "nonmath": cross_nonmath,
    },
    "runtime_seconds": time.time() - t0,
}

with open("output/expT_pacf_innovation.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to output/expT_pacf_innovation.json")
print(f"Total runtime: {time.time()-t0:.1f}s")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Find peak innovation layers
math_innov_profile = [(li, profile_comparison[li]["math_innovation_norm"]) for li in ANALYSIS_LAYERS]
nonmath_innov_profile = [(li, profile_comparison[li]["nonmath_innovation_norm"]) for li in ANALYSIS_LAYERS]
math_peak = max(math_innov_profile, key=lambda x: x[1])
nonmath_peak = max(nonmath_innov_profile, key=lambda x: x[1])

print(f"\nMath peak innovation:    L{math_peak[0]} (norm={math_peak[1]:.1f})")
print(f"Non-math peak innovation: L{nonmath_peak[0]} (norm={nonmath_peak[1]:.1f})")

# Average R² across layers
avg_r2_math = np.mean([profile_comparison[li]["math_r2"] for li in ANALYSIS_LAYERS])
avg_r2_nonmath = np.mean([profile_comparison[li]["nonmath_r2"] for li in ANALYSIS_LAYERS])
print(f"\nAvg R² (fraction inherited): math={avg_r2_math:.4f}, nonmath={avg_r2_nonmath:.4f}")

# Where is the language signal? Innovation or inherited?
print(f"\nLanguage signal location (Cohen's d):")
print(f"{'Layer':<8} {'Math_raw':<12} {'Math_innov':<12} {'NonM_raw':<12} {'NonM_innov':<12}")
print("-" * 56)
for li in ANALYSIS_LAYERS:
    m = lang_math[li]
    nm = lang_nonmath[li]
    print(f"  L{li:<4} {m['cohens_d_raw']:<12.1f} {m['cohens_d_innov']:<12.1f} "
          f"{nm['cohens_d_raw']:<12.1f} {nm['cohens_d_innov']:<12.1f}")
