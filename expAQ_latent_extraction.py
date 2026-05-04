"""
Exp AQ: Latent Math Extraction via Precision-Weighted Denoising

Greg's insight: model embedding space as language-specific clusters.
Same math problem → different point in each cluster. Back out the
latent math content using inverse-covariance weighting.

Latent variable model:
    x_lang = A_lang @ z + mu_lang + noise

where z = language-agnostic math content, A_lang = language observation
matrix, mu_lang = cluster center.

Given 7 observations (languages), optimal z estimate is precision-weighted:
    z_hat = (sum_lang Sigma_lang^{-1})^{-1} @ sum_lang Sigma_lang^{-1} @ (x_lang - mu_lang)

This is the Kalman filter / Gaussian posterior mean.

We test:
1. NAIVE AVG: just average (AM baseline — known to fail at L0)
2. CENTERED AVG: subtract cluster centers, then average
3. PRECISION-WEIGHTED: full inverse-covariance weighting (Kalman)
4. WHITENED AVG: whiten each language, average, then project back
5. MMD-STYLE: minimize maximum mean discrepancy across languages

Test at multiple layers (L0 embedding, L9, L18, L26, L35) to find where
the latent space is cleanest.

Then: forward-pass the extracted z through the model and see if it solves math.

On Qwen2.5-3B locally.
"""

import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
device = "cuda"
MAX_NEW_TOKENS = 128

# ── Load pre-computed multilingual data ─────────────────────────────────
print("Loading multilingual hidden states...")
ml_data = np.load("output/multilingual_all_layers.npz", allow_pickle=True)
categories = ml_data["categories"]  # (200,)

LANGS = ["zh", "en", "es", "ar", "ja", "ko", "sw"]  # Match multilingual_all_layers.npz
N_PROBLEMS = 200
N_LAYERS = 36
D = 2048

# Build data tensor: (7, 200, 36, 2048) — but stored as lang_L{i}
# We'll work layer by layer to save memory
def get_layer_data(layer_idx):
    """Return (7, 200, 2048) array for a specific layer."""
    data = []
    for lang in LANGS:
        key = f"{lang}_L{layer_idx}"
        data.append(ml_data[key])  # (200, 2048)
    return np.stack(data)  # (7, 200, 2048)

print(f"Data loaded: {N_PROBLEMS} problems, {len(LANGS)} languages, {N_LAYERS} layers")

# ── Load model for forward pass testing ─────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ── Test problems (first 4 per category = 20 test problems) ────────────
# We need prompts + answers for the test set
# Load from the standard problem set
from pathlib import Path
import importlib.util

# Load problems from existing experiment
MATH_PROBLEMS = [
    {"idx": 0, "answer": "133", "zh": "计算 47 + 86 的值。", "en": "Calculate 47 + 86."},
    {"idx": 1, "answer": "788", "zh": "计算 664 + 124 的值。", "en": "Calculate 664 + 124."},
    {"idx": 2, "answer": "120", "zh": "计算 15 × 8 的值。", "en": "Calculate 15 × 8."},
    {"idx": 3, "answer": "390", "zh": "计算 238 + 152 的值。", "en": "Calculate 238 + 152."},
    {"idx": 4, "answer": "60", "zh": "一个长方形的长为 12，宽为 5，求其面积。",
     "en": "A rectangle has length 12 and width 5. Find its area."},
]

# Use first 20 problems as test (indices 0-19)
TEST_INDICES = list(range(20))
# Use problems 20-199 for estimating cluster statistics
TRAIN_INDICES = list(range(20, 200))


# ══════════════════════════════════════════════════════════════════════════
# PART 1: Estimate cluster statistics at each layer
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: CLUSTER STATISTICS")
print("=" * 70)

TEST_LAYERS = [0, 5, 9, 14, 18, 22, 26, 30, 35]

layer_stats = {}

for layer_idx in TEST_LAYERS:
    print(f"\n--- Layer {layer_idx} ---")
    data = get_layer_data(layer_idx)  # (7, 200, 2048)

    # Train split for statistics
    train_data = data[:, TRAIN_INDICES, :]  # (7, 180, 2048)
    test_data = data[:, TEST_INDICES, :]    # (7, 20, 2048)

    # Cluster centers (per language)
    centers = train_data.mean(axis=1)  # (7, 2048)

    # Center each language
    centered_train = train_data - centers[:, np.newaxis, :]  # (7, 180, 2048)
    centered_test = test_data - centers[:, np.newaxis, :]    # (7, 20, 2048)

    # Per-language covariance (diagonal approximation for 2048D)
    # Full covariance is 2048x2048 — too big. Use diagonal (variance per dim).
    variances = []
    for li in range(7):
        var = centered_train[li].var(axis=0)  # (2048,)
        variances.append(var)
    variances = np.stack(variances)  # (7, 2048)

    # Precision = 1/variance (with floor to avoid division by zero)
    precisions = 1.0 / np.maximum(variances, 1e-8)  # (7, 2048)

    # ── Method 1: Naive average ──
    naive_avg = test_data.mean(axis=0)  # (20, 2048)

    # ── Method 2: Centered average ──
    centered_avg = centered_test.mean(axis=0)  # (20, 2048)

    # ── Method 3: Precision-weighted (Kalman) ──
    # z_hat = (sum precision)^{-1} * sum (precision * centered_x)
    total_precision = precisions.sum(axis=0)  # (2048,)
    weighted_sum = np.zeros((20, D))
    for li in range(7):
        weighted_sum += precisions[li][np.newaxis, :] * centered_test[li]  # (20, 2048)
    precision_weighted = weighted_sum / total_precision[np.newaxis, :]  # (20, 2048)

    # ── Method 4: Whitened average ──
    # Whiten: x_white = (x - mu) / std, then average, then de-whiten with mean std
    stds = np.sqrt(variances)  # (7, 2048)
    whitened = centered_test / np.maximum(stds[:, np.newaxis, :], 1e-8)  # (7, 20, 2048)
    whitened_avg = whitened.mean(axis=0)  # (20, 2048)
    # De-whiten with average std
    mean_std = stds.mean(axis=0)  # (2048,)
    dewhitened_avg = whitened_avg * mean_std  # (20, 2048)

    # ── Cross-language agreement metrics ──
    # How consistent is the latent across languages?
    # Compute pairwise cosine similarity of centered vectors for same problem
    pairwise_cos = []
    for pi in range(20):
        for li in range(7):
            for lj in range(li+1, 7):
                v1 = centered_test[li, pi]
                v2 = centered_test[lj, pi]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                pairwise_cos.append(cos)

    mean_cos = np.mean(pairwise_cos)

    # Cohen's d between languages (using centered vectors)
    # Average L2 distance between same-problem centered vectors / pooled std
    l2_dists = []
    for pi in range(20):
        for li in range(7):
            for lj in range(li+1, 7):
                d = np.linalg.norm(centered_test[li, pi] - centered_test[lj, pi])
                l2_dists.append(d)
    mean_l2 = np.mean(l2_dists)

    # Norms
    naive_norms = np.linalg.norm(naive_avg, axis=1).mean()
    centered_norms = np.linalg.norm(centered_avg, axis=1).mean()
    precision_norms = np.linalg.norm(precision_weighted, axis=1).mean()
    single_norms = np.linalg.norm(test_data[0], axis=1).mean()  # ZH reference

    print(f"  Cross-lang cos (centered, same problem): {mean_cos:.4f}")
    print(f"  Cross-lang L2 (centered): {mean_l2:.1f}")
    print(f"  Norms — single: {single_norms:.1f}, naive: {naive_norms:.1f}, centered: {centered_norms:.1f}, precision: {precision_norms:.1f}")

    # Variance explained by language vs problem
    # Between-language variance vs within-language (across problems) variance
    grand_mean = train_data.mean(axis=(0, 1))  # (2048,)
    between_lang_var = np.mean([(centers[li] - grand_mean)**2 for li in range(7)], axis=0).sum()
    within_lang_var = np.mean([centered_train[li].var(axis=0).sum() for li in range(7)])
    lang_ratio = between_lang_var / (between_lang_var + within_lang_var + 1e-8)

    print(f"  Language variance ratio: {lang_ratio:.4f} ({lang_ratio*100:.1f}% of total variance is language)")

    layer_stats[layer_idx] = {
        "cross_lang_cos": float(mean_cos),
        "cross_lang_l2": float(mean_l2),
        "lang_variance_ratio": float(lang_ratio),
        "norms": {
            "single_zh": float(single_norms),
            "naive_avg": float(naive_norms),
            "centered_avg": float(centered_norms),
            "precision_weighted": float(precision_norms),
        },
        # Store the extracted latents for forward-pass testing
        "_naive": naive_avg,
        "_centered": centered_avg,
        "_precision": precision_weighted,
        "_dewhitened": dewhitened_avg,
    }


# ══════════════════════════════════════════════════════════════════════════
# PART 2: Forward-pass test of extracted latents
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: FORWARD PASS GENERATION FROM EXTRACTED LATENTS")
print("=" * 70)

# We'll inject the extracted latent as a single "token" embedding and generate.
# Also try: inject at the LAYER where it was extracted (not at L0).

def generate_from_single_vector(vec, max_tokens=MAX_NEW_TOKENS):
    """Generate from a single vector treated as a 1-token embedding."""
    embeds = torch.tensor(vec, dtype=torch.bfloat16, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, d)
    try:
        with torch.no_grad():
            out = model.generate(
                inputs_embeds=embeds,
                max_new_tokens=max_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"ERROR: {e}"


def inject_at_layer_and_generate(vec, layer_idx, seq_len_dummy=5, max_tokens=MAX_NEW_TOKENS):
    """
    Inject a vector at a specific layer and generate.
    Uses dummy input for layers 0..layer_idx, then hooks layer_idx to replace output.
    """
    dummy_ids = torch.zeros(1, seq_len_dummy, dtype=torch.long, device=device)
    injection_done = [False]

    def inject_hook(module, input, output):
        if not injection_done[0]:
            injection_done[0] = True
            h = output[0] if isinstance(output, tuple) else output
            # Replace last token position with our vector
            injected = h.clone()
            v = torch.tensor(vec, dtype=h.dtype, device=h.device)
            # Broadcast to all positions (the model might attend to them)
            injected[0, :, :] = v
            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected
        return output

    handle = model.model.layers[layer_idx].register_forward_hook(inject_hook)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=dummy_ids,
                max_new_tokens=max_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        gen = tokenizer.decode(out[0][seq_len_dummy:], skip_special_tokens=True)
    except Exception as e:
        gen = f"ERROR: {e}"
    finally:
        handle.remove()
    return gen


# Test on first 5 problems
gen_results = {}

for test_layer in [0, 9, 18, 26]:
    print(f"\n--- Generating from layer {test_layer} latents ---")
    stats = layer_stats[test_layer]

    for method_name, latent_key in [("naive", "_naive"), ("centered", "_centered"),
                                      ("precision", "_precision"), ("dewhitened", "_dewhitened")]:
        latents = stats[latent_key]  # (20, 2048)

        correct_count = 0
        tested = 0

        for pi in range(5):  # Test first 5 problems
            vec = latents[pi]

            if test_layer == 0:
                gen = generate_from_single_vector(vec)
            else:
                gen = inject_at_layer_and_generate(vec, test_layer)

            # Check if it contains relevant math answer
            # We need answers for test problems 0-4 (matching MATH_PROBLEMS)
            if pi < len(MATH_PROBLEMS):
                answer = MATH_PROBLEMS[pi]["answer"]
                correct = answer in gen
                correct_count += correct
                tested += 1
                marker = "Y" if correct else "N"
            else:
                marker = "?"
                correct = False

            if pi < 3:  # Print first 3
                print(f"  L{test_layer}_{method_name} P{pi}: {marker} — {gen[:50]}...")

        key = f"L{test_layer}_{method_name}"
        gen_results[key] = {
            "correct": correct_count,
            "total": tested,
            "layer": test_layer,
            "method": method_name,
        }
        print(f"  L{test_layer}_{method_name}: {correct_count}/{tested}")


# ══════════════════════════════════════════════════════════════════════════
# PART 3: Linear probe on extracted latents
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: LINEAR PROBE — DO EXTRACTED LATENTS PREDICT CATEGORY?")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

for test_layer in TEST_LAYERS:
    stats = layer_stats[test_layer]

    train_data_layer = get_layer_data(test_layer)[:, TRAIN_INDICES, :]  # (7, 180, 2048)
    test_data_layer = get_layer_data(test_layer)[:, TEST_INDICES, :]    # (7, 20, 2048)

    train_labels = categories[TRAIN_INDICES]  # (180,)
    test_labels = categories[TEST_INDICES]    # (20,)

    centers = train_data_layer.mean(axis=1)  # (7, 2048)

    # Extract precision-weighted latents for train and test
    variances = np.stack([train_data_layer[li].var(axis=0) for li in range(7)])
    precisions = 1.0 / np.maximum(variances, 1e-8)
    total_prec = precisions.sum(axis=0)

    # Train latents
    train_centered = train_data_layer - centers[:, np.newaxis, :]
    train_weighted = np.zeros((180, D))
    for li in range(7):
        train_weighted += precisions[li][np.newaxis, :] * train_centered[li]
    train_latent = train_weighted / total_prec[np.newaxis, :]

    # Test latents (already computed)
    test_latent = stats["_precision"]

    # Also get single-language baselines
    zh_train = train_data_layer[0]  # (180, 2048) — raw ZH
    zh_test = test_data_layer[0]    # (20, 2048)

    # Fit probe
    scaler_z = StandardScaler()
    X_train_z = scaler_z.fit_transform(train_latent)
    X_test_z = scaler_z.transform(test_latent)

    scaler_zh = StandardScaler()
    X_train_zh = scaler_zh.fit_transform(zh_train)
    X_test_zh = scaler_zh.transform(zh_test)

    clf_z = LogisticRegression(max_iter=1000, C=1.0)
    clf_z.fit(X_train_z, train_labels)
    acc_z = clf_z.score(X_test_z, test_labels)

    clf_zh = LogisticRegression(max_iter=1000, C=1.0)
    clf_zh.fit(X_train_zh, train_labels)
    acc_zh = clf_zh.score(X_test_zh, test_labels)

    # Cross-lingual probe: train on ZH, test on latent
    acc_cross = clf_zh.score(X_test_z, test_labels)  # Using ZH scaler on latent — may be unfair

    # Better: train on latent, test on each language
    lang_accs = {}
    for li, lang in enumerate(LANGS):
        lang_test = test_data_layer[li]
        # Center and whiten using train stats
        lang_centered = lang_test - centers[li:li+1, :]
        lang_scaled = scaler_z.transform(lang_centered)  # Approximate
        lang_accs[lang] = clf_z.score(lang_scaled, test_labels)

    print(f"  L{test_layer:2d}: latent_acc={acc_z:.3f}, zh_acc={acc_zh:.3f}, cross={acc_cross:.3f} | per-lang: {' '.join(f'{l}={a:.2f}' for l,a in lang_accs.items())}")

    layer_stats[test_layer]["probe_latent_acc"] = float(acc_z)
    layer_stats[test_layer]["probe_zh_acc"] = float(acc_zh)
    layer_stats[test_layer]["probe_cross_lang"] = {k: float(v) for k, v in lang_accs.items()}


# ── Save ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

# Clean up numpy arrays from stats before saving
clean_stats = {}
for layer_idx, stats in layer_stats.items():
    clean = {k: v for k, v in stats.items() if not k.startswith("_")}
    clean_stats[layer_idx] = clean

output = {
    "experiment": "AQ: Latent Math Extraction via Precision-Weighted Denoising",
    "model": MODEL_NAME,
    "languages": LANGS,
    "n_problems": N_PROBLEMS,
    "n_train": len(TRAIN_INDICES),
    "n_test": len(TEST_INDICES),
    "test_layers": TEST_LAYERS,
    "layer_stats": {str(k): v for k, v in clean_stats.items()},
    "generation_results": gen_results,
}

with open(OUTPUT_DIR / "expAQ_latent_extraction.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("Saved to output/expAQ_latent_extraction.json")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== GRAND SUMMARY ===")
print("\nCluster statistics by layer:")
print(f"{'Layer':>5} {'LangVar%':>8} {'CrossCos':>8} {'CrossL2':>8}")
for l in TEST_LAYERS:
    s = clean_stats[l]
    print(f"  L{l:2d}  {s['lang_variance_ratio']*100:7.1f}%  {s['cross_lang_cos']:8.4f}  {s['cross_lang_l2']:8.1f}")

print("\nGeneration results:")
for k, v in gen_results.items():
    print(f"  {k:25s}: {v['correct']}/{v['total']}")

print("\nDone.")
