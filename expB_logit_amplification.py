"""Experiment B: Logit Amplification
Quantify how much softmax amplifies PC0's tiny bias.
Compute logits → softmax → P(Chinese) vs P(English) before and after PC0 swap.
"""

import json
import numpy as np
from safetensors import safe_open

# Load hidden states at L35 (last layer)
print("Loading hidden states...")
data = np.load("output/all_layers_lasttok.npz")
# Keys are zh_L0..zh_L35, en_L0..en_L35, each (200, 2048)
h_zh_L35 = data["zh_L35"]  # (200, 2048) — Chinese at last layer
h_en_L35 = data["en_L35"]  # (200, 2048) — English at last layer

print(f"zh L35 shape: {h_zh_L35.shape}, en L35 shape: {h_en_L35.shape}")

# Compute PC0 at L35
all_L35 = np.concatenate([h_zh_L35, h_en_L35], axis=0)  # (400, 2048)
mean_L35 = all_L35.mean(axis=0)
centered = all_L35 - mean_L35
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
pc0 = Vt[0]  # (2048,)

# PC0 projections
zh_proj = (h_zh_L35 - mean_L35) @ pc0  # (200,)
en_proj = (h_en_L35 - mean_L35) @ pc0  # (200,)
print(f"Mean PC0 projection — zh: {zh_proj.mean():.4f}, en: {en_proj.mean():.4f}")
print(f"Gap: {en_proj.mean() - zh_proj.mean():.4f}")

# Load LM head weights (= embed_tokens, tied)
print("\nLoading LM head weights...")
model_path = "/home/greg/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
with safe_open(f"{model_path}/model-00001-of-00002.safetensors", framework="pt") as f:
    embed_weights = f.get_tensor("model.embed_tokens.weight").float().numpy()  # (151936, 2048)

print(f"LM head shape: {embed_weights.shape}")

# Token classification: Chinese vs English
# Use Unicode ranges for classification
vocab_size = embed_weights.shape[0]

# We need to decode token IDs to check their content
# For efficiency, project each token embedding onto PC0 and classify by projection
token_pc0_proj = embed_weights @ pc0  # (151936,)

# Build Chinese/English token sets using the analysis from lm_head_pc0_analysis
# Chinese tokens: those below median PC0 projection (the "Chinese-aligned" direction)
# English tokens: those above median
# But more precisely, use Unicode ranges
# Load tokenizer for proper classification
from transformers import AutoTokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

chinese_ids = []
english_ids = []
other_ids = []

for token_id in range(vocab_size):
    try:
        text = tokenizer.decode([token_id])
        # Check if predominantly Chinese characters
        n_chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
        n_latin = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))

        if n_chinese > 0 and n_chinese >= n_latin:
            chinese_ids.append(token_id)
        elif n_latin > 0 and n_latin >= n_chinese:
            english_ids.append(token_id)
        else:
            other_ids.append(token_id)
    except:
        other_ids.append(token_id)

chinese_ids = np.array(chinese_ids)
english_ids = np.array(english_ids)
print(f"\nToken counts — Chinese: {len(chinese_ids)}, English: {len(english_ids)}, Other: {len(other_ids)}")

# Process 20 problems (first 10 zh, first 10 en)
n_test = 10
results = {
    "zh_original": [], "zh_swapped": [],
    "en_original": [], "en_swapped": [],
    "top_tokens_before": [], "top_tokens_after": [],
}

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def compute_lang_probs(logits, chinese_ids, english_ids):
    probs = softmax(logits)
    p_zh = probs[chinese_ids].sum()
    p_en = probs[english_ids].sum()
    return p_zh, p_en, probs

print("\n=== PROCESSING HIDDEN STATES ===")

# PC0 swap vector: the full zh→en displacement along PC0
zh_mean_proj = zh_proj[:n_test].mean()
en_mean_proj = en_proj[:n_test].mean()
swap_delta = (en_mean_proj - zh_mean_proj)  # scalar magnitude
print(f"PC0 swap magnitude: {swap_delta:.4f}")

for i in range(n_test):
    # --- Chinese hidden state ---
    h_zh = h_zh_L35[i]
    logits_zh = embed_weights @ h_zh
    p_zh_orig, p_en_orig, probs_orig = compute_lang_probs(logits_zh, chinese_ids, english_ids)

    # Swap PC0: move projection from zh mean to en mean
    current_proj = (h_zh - mean_L35) @ pc0
    h_zh_swapped = h_zh + (en_mean_proj - current_proj) * pc0
    logits_zh_swap = embed_weights @ h_zh_swapped
    p_zh_swap, p_en_swap, probs_swap = compute_lang_probs(logits_zh_swap, chinese_ids, english_ids)

    results["zh_original"].append({"p_zh": float(p_zh_orig), "p_en": float(p_en_orig)})
    results["zh_swapped"].append({"p_zh": float(p_zh_swap), "p_en": float(p_en_swap)})

    # Top-5 tokens before/after
    top5_before = np.argsort(probs_orig)[-5:][::-1]
    top5_after = np.argsort(probs_swap)[-5:][::-1]

    results["top_tokens_before"].append({
        "problem": i,
        "lang": "zh",
        "tokens": [(int(t), tokenizer.decode([t]), float(probs_orig[t])) for t in top5_before]
    })
    results["top_tokens_after"].append({
        "problem": i,
        "lang": "zh→en_swap",
        "tokens": [(int(t), tokenizer.decode([t]), float(probs_swap[t])) for t in top5_after]
    })

    if i < 3:
        print(f"\nProblem {i} (Chinese):")
        print(f"  Original: P(zh)={p_zh_orig:.4f}, P(en)={p_en_orig:.4f}")
        print(f"  Swapped:  P(zh)={p_zh_swap:.4f}, P(en)={p_en_swap:.4f}")
        print(f"  Δ P(en) = {p_en_swap - p_en_orig:+.4f}")
        print(f"  Top-5 before: {[tokenizer.decode([t]) for t in top5_before]}")
        print(f"  Top-5 after:  {[tokenizer.decode([t]) for t in top5_after]}")

for i in range(n_test):
    # --- English hidden state ---
    h_en = h_en_L35[i]
    logits_en = embed_weights @ h_en
    p_zh_orig, p_en_orig, probs_orig = compute_lang_probs(logits_en, chinese_ids, english_ids)

    # Swap PC0: move projection from en mean to zh mean
    current_proj = (h_en - mean_L35) @ pc0
    h_en_swapped = h_en + (zh_mean_proj - current_proj) * pc0
    logits_en_swap = embed_weights @ h_en_swapped
    p_zh_swap, p_en_swap, probs_swap = compute_lang_probs(logits_en_swap, chinese_ids, english_ids)

    results["en_original"].append({"p_zh": float(p_zh_orig), "p_en": float(p_en_orig)})
    results["en_swapped"].append({"p_zh": float(p_zh_swap), "p_en": float(p_en_swap)})

    top5_before = np.argsort(probs_orig)[-5:][::-1]
    top5_after = np.argsort(probs_swap)[-5:][::-1]

    results["top_tokens_before"].append({
        "problem": i,
        "lang": "en",
        "tokens": [(int(t), tokenizer.decode([t]), float(probs_orig[t])) for t in top5_before]
    })
    results["top_tokens_after"].append({
        "problem": i,
        "lang": "en→zh_swap",
        "tokens": [(int(t), tokenizer.decode([t]), float(probs_swap[t])) for t in top5_after]
    })

    if i < 3:
        print(f"\nProblem {i} (English):")
        print(f"  Original: P(zh)={p_zh_orig:.4f}, P(en)={p_en_orig:.4f}")
        print(f"  Swapped:  P(zh)={p_zh_swap:.4f}, P(en)={p_en_swap:.4f}")
        print(f"  Δ P(zh) = {p_zh_swap - p_zh_orig:+.4f}")
        print(f"  Top-5 before: {[tokenizer.decode([t]) for t in top5_before]}")
        print(f"  Top-5 after:  {[tokenizer.decode([t]) for t in top5_after]}")

# Summary statistics
print("\n\n=== SUMMARY ===")
zh_orig_p_zh = np.mean([r["p_zh"] for r in results["zh_original"]])
zh_orig_p_en = np.mean([r["p_en"] for r in results["zh_original"]])
zh_swap_p_zh = np.mean([r["p_zh"] for r in results["zh_swapped"]])
zh_swap_p_en = np.mean([r["p_en"] for r in results["zh_swapped"]])

en_orig_p_zh = np.mean([r["p_zh"] for r in results["en_original"]])
en_orig_p_en = np.mean([r["p_en"] for r in results["en_original"]])
en_swap_p_zh = np.mean([r["p_zh"] for r in results["en_swapped"]])
en_swap_p_en = np.mean([r["p_en"] for r in results["en_swapped"]])

print(f"\nChinese hidden states (n={n_test}):")
print(f"  Original: P(zh)={zh_orig_p_zh:.4f}, P(en)={zh_orig_p_en:.4f}")
print(f"  PC0→en:   P(zh)={zh_swap_p_zh:.4f}, P(en)={zh_swap_p_en:.4f}")
print(f"  ΔP(en) = {zh_swap_p_en - zh_orig_p_en:+.4f}")
print(f"  ΔP(zh) = {zh_swap_p_zh - zh_orig_p_zh:+.4f}")
print(f"  Probability mass shift: {(zh_swap_p_en - zh_orig_p_en) + (zh_orig_p_zh - zh_swap_p_zh):.4f}")

print(f"\nEnglish hidden states (n={n_test}):")
print(f"  Original: P(zh)={en_orig_p_zh:.4f}, P(en)={en_orig_p_en:.4f}")
print(f"  PC0→zh:   P(zh)={en_swap_p_zh:.4f}, P(en)={en_swap_p_en:.4f}")
print(f"  ΔP(zh) = {en_swap_p_zh - en_orig_p_zh:+.4f}")
print(f"  ΔP(en) = {en_swap_p_en - en_orig_p_en:+.4f}")
print(f"  Probability mass shift: {(en_swap_p_zh - en_orig_p_zh) + (en_orig_p_en - en_swap_p_en):.4f}")

# Amplification factor
# PC0 gap in logit space: 0.018 (from LM head analysis)
# Behavioral effect: 100% language switch
# How much probability mass actually moves?
total_mass_shift = ((zh_swap_p_en - zh_orig_p_en) + (en_swap_p_zh - en_orig_p_zh)) / 2

# Raw logit gap from PC0
raw_logit_gap = 0.018  # from lm_head_pc0_analysis
print(f"\n--- Amplification Analysis ---")
print(f"Mean PC0 logit gap: {raw_logit_gap:.4f}")
print(f"Mean probability mass shift: {total_mass_shift:.4f}")
print(f"Amplification factor: {total_mass_shift / raw_logit_gap:.1f}x")

# Check: did argmax actually change language?
zh_argmax_changed = 0
en_argmax_changed = 0
for i in range(n_test):
    # Check if top token changed from Chinese to English
    before_tokens = results["top_tokens_before"][i]["tokens"]
    after_tokens = results["top_tokens_after"][i]["tokens"]
    before_id = before_tokens[0][0]
    after_id = after_tokens[0][0]
    if before_id in chinese_ids and after_id in english_ids:
        zh_argmax_changed += 1
    elif before_id in english_ids and after_id in chinese_ids:
        zh_argmax_changed += 1  # any language change counts

for i in range(n_test, 2 * n_test):
    before_tokens = results["top_tokens_before"][i]["tokens"]
    after_tokens = results["top_tokens_after"][i]["tokens"]
    before_id = before_tokens[0][0]
    after_id = after_tokens[0][0]
    if before_id in english_ids and after_id in chinese_ids:
        en_argmax_changed += 1
    elif before_id in chinese_ids and after_id in english_ids:
        en_argmax_changed += 1

print(f"\nArgmax language change:")
print(f"  zh→en swap: {zh_argmax_changed}/{n_test} problems changed top token language")
print(f"  en→zh swap: {en_argmax_changed}/{n_test} problems changed top token language")

# Save results
summary = {
    "n_test": n_test,
    "n_chinese_tokens": len(chinese_ids),
    "n_english_tokens": len(english_ids),
    "pc0_swap_magnitude": float(swap_delta),
    "zh_original": {"mean_p_zh": float(zh_orig_p_zh), "mean_p_en": float(zh_orig_p_en)},
    "zh_swapped": {"mean_p_zh": float(zh_swap_p_zh), "mean_p_en": float(zh_swap_p_en)},
    "en_original": {"mean_p_zh": float(en_orig_p_zh), "mean_p_en": float(en_orig_p_en)},
    "en_swapped": {"mean_p_zh": float(en_swap_p_zh), "mean_p_en": float(en_swap_p_en)},
    "zh_to_en_delta_p_en": float(zh_swap_p_en - zh_orig_p_en),
    "en_to_zh_delta_p_zh": float(en_swap_p_zh - en_orig_p_zh),
    "mean_mass_shift": float(total_mass_shift),
    "raw_logit_gap": raw_logit_gap,
    "amplification_factor": float(total_mass_shift / raw_logit_gap),
    "zh_argmax_changed": zh_argmax_changed,
    "en_argmax_changed": en_argmax_changed,
    "per_problem_zh": results["zh_original"],
    "per_problem_zh_swapped": results["zh_swapped"],
    "per_problem_en": results["en_original"],
    "per_problem_en_swapped": results["en_swapped"],
    "top_tokens": results["top_tokens_before"] + results["top_tokens_after"],
}

with open("output/expB_logit_amplification.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to output/expB_logit_amplification.json")
