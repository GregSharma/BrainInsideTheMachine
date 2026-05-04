"""Experiment 2: Early exit at L26 — efficiency test.

Qwen-3B hits z≈25 by L26. The last 10 layers add only +2 to z-score
while adding 28% more compute. Can we read the answer from L26?

Method:
  - Run full forward pass to get baseline first token (L35 output)
  - Hook layer 26 output, apply model.model.norm (final RMSNorm) + model.lm_head
  - Compare top-1 token at L26 vs L35 for all 200 problems × 2 languages
  - Also test L20, L24, L28, L30, L32 for a full early-exit curve

If L26 matches L35 first token 80%+ of the time, early exit is viable.
"""

import numpy as np
import torch
import json
import random as pyrandom
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test exit points: L20 through L34 (L35 = final = baseline)
EXIT_LAYERS = [10, 14, 18, 20, 22, 24, 26, 28, 30, 32, 34]


def generate_problems(n=200, seed=42):
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


def matched_vs_scrambled_z(zh, en, n_perms=500):
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    rng = np.random.RandomState(42)
    scrambled = np.array([
        np.mean(np.sum(zh_u * en_u[rng.permutation(len(en_u))], axis=1))
        for _ in range(n_perms)
    ])
    z = (matched - scrambled.mean()) / scrambled.std()
    return float(z)


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers  # 36
    d = model.config.hidden_size  # 2048
    print(f"Model: {n_layers} layers, d={d}")
    print(f"Exit layers to test: {EXIT_LAYERS}")

    # Get the final norm and lm_head for early projection
    final_norm = model.model.norm  # RMSNorm
    lm_head = model.lm_head  # Linear(d, vocab_size)

    problems = generate_problems(200, seed=42)
    N = len(problems)

    # Hook to capture layer outputs at exit points
    layer_outputs = {}  # layer_idx -> last-token hidden state

    def make_layer_hook(layer_idx):
        def hook(module, input, output):
            # Qwen2DecoderLayer output: (hidden_states, ...)
            h = output[0] if isinstance(output, tuple) else output
            layer_outputs[layer_idx] = h.detach()[:, -1:, :]  # (1, 1, d) — keep on GPU
        return hook

    # Register hooks on all exit layers
    handles = []
    for l in EXIT_LAYERS:
        h = model.model.layers[l].register_forward_hook(make_layer_hook(l))
        handles.append(h)

    # Storage
    # baseline_tokens[lang][i] = argmax token from full model
    # early_tokens[layer][lang][i] = argmax token from early exit at that layer
    # early_hidden[layer][lang][i] = normalized hidden state at that layer
    baseline_zh_tokens = []
    baseline_en_tokens = []
    early_zh_tokens = {l: [] for l in EXIT_LAYERS}
    early_en_tokens = {l: [] for l in EXIT_LAYERS}
    early_zh_hidden = {l: np.zeros((N, d), dtype=np.float32) for l in EXIT_LAYERS}
    early_en_hidden = {l: np.zeros((N, d), dtype=np.float32) for l in EXIT_LAYERS}

    # Also capture top-5 tokens at each exit for richer analysis
    early_zh_top5 = {l: [] for l in EXIT_LAYERS}
    early_en_top5 = {l: [] for l in EXIT_LAYERS}

    print(f"\nExtracting {N} Chinese problems...")
    for i, prob in enumerate(tqdm(problems, desc="zh")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Baseline: full model prediction
        baseline_zh_tokens.append(int(outputs.logits[0, -1].argmax()))

        # Early exit: apply norm + lm_head to each captured layer output
        for l in EXIT_LAYERS:
            h = layer_outputs[l]  # (1, 1, d) on GPU
            with torch.no_grad():
                h_normed = final_norm(h)  # RMSNorm
                logits = lm_head(h_normed)  # (1, 1, vocab_size)
            logits_1d = logits[0, 0]
            early_zh_tokens[l].append(int(logits_1d.argmax()))
            top5 = logits_1d.topk(5)
            early_zh_top5[l].append(top5.indices.cpu().tolist())
            early_zh_hidden[l][i] = h_normed[0, 0].cpu().float().numpy()

        layer_outputs.clear()

    print(f"Extracting {N} English problems...")
    for i, prob in enumerate(tqdm(problems, desc="en")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        baseline_en_tokens.append(int(outputs.logits[0, -1].argmax()))

        for l in EXIT_LAYERS:
            h = layer_outputs[l]
            with torch.no_grad():
                h_normed = final_norm(h)
                logits = lm_head(h_normed)
            logits_1d = logits[0, 0]
            early_en_tokens[l].append(int(logits_1d.argmax()))
            top5 = logits_1d.topk(5)
            early_en_top5[l].append(top5.indices.cpu().tolist())
            early_en_hidden[l][i] = h_normed[0, 0].cpu().float().numpy()

        layer_outputs.clear()

    for h in handles:
        h.remove()

    # ========== ANALYSIS ==========
    print(f"\n{'='*70}")
    print("EARLY EXIT ANALYSIS")
    print(f"{'='*70}")

    # Cross-lingual match at baseline
    cross_match_base = sum(1 for a, b in zip(baseline_zh_tokens, baseline_en_tokens) if a == b) / N

    results = {
        "model": MODEL_NAME,
        "n_problems": N,
        "n_layers": n_layers,
        "cross_lingual_match_baseline": cross_match_base,
        "exit_layers": [],
    }

    print(f"\nBaseline cross-lingual first-token match: {cross_match_base:.1%}")
    print(f"\n{'Layer':>5} | {'zh match':>9} | {'en match':>9} | {'zh top5':>8} | {'en top5':>8} | {'z-score':>8} | {'x-ling match':>12} | {'compute%':>9}")
    print("-" * 90)

    for l in EXIT_LAYERS:
        # Exact first-token match vs baseline
        zh_match = sum(1 for a, b in zip(baseline_zh_tokens, early_zh_tokens[l]) if a == b) / N
        en_match = sum(1 for a, b in zip(baseline_en_tokens, early_en_tokens[l]) if a == b) / N

        # Top-5 containment: does baseline token appear in early exit's top-5?
        zh_top5_match = sum(
            1 for i in range(N)
            if baseline_zh_tokens[i] in early_zh_top5[l][i]
        ) / N
        en_top5_match = sum(
            1 for i in range(N)
            if baseline_en_tokens[i] in early_en_top5[l][i]
        ) / N

        # Cross-lingual z-score at this layer
        z_score = matched_vs_scrambled_z(early_zh_hidden[l], early_en_hidden[l])

        # Cross-lingual first-token match at early exit
        cross_match = sum(1 for a, b in zip(early_zh_tokens[l], early_en_tokens[l]) if a == b) / N

        # Compute savings (layers used / total layers)
        compute_pct = (l + 1) / n_layers * 100

        print(f"  L{l:2d}  | {zh_match:8.1%} | {en_match:8.1%} | {zh_top5_match:7.1%} | {en_top5_match:7.1%} | {z_score:7.1f} | {cross_match:11.1%} | {compute_pct:7.0f}%")

        results["exit_layers"].append({
            "layer": l,
            "zh_top1_match": zh_match,
            "en_top1_match": en_match,
            "zh_top5_match": zh_top5_match,
            "en_top5_match": en_top5_match,
            "z_score": z_score,
            "cross_lingual_match": cross_match,
            "compute_pct": compute_pct,
        })

    # Find the sweet spot
    print(f"\n{'='*70}")
    print("SWEET SPOT ANALYSIS")
    print(f"{'='*70}")
    for entry in results["exit_layers"]:
        avg_match = (entry["zh_top1_match"] + entry["en_top1_match"]) / 2
        if avg_match >= 0.8:
            print(f"  L{entry['layer']}: {avg_match:.1%} avg match at {entry['compute_pct']:.0f}% compute — VIABLE EXIT")
        elif avg_match >= 0.6:
            print(f"  L{entry['layer']}: {avg_match:.1%} avg match at {entry['compute_pct']:.0f}% compute — marginal")

    # Decode some example tokens to see what's happening
    print(f"\n{'='*70}")
    print("EXAMPLE TOKEN COMPARISONS (first 10 problems)")
    print(f"{'='*70}")
    for i in range(min(10, N)):
        base_zh = tokenizer.decode([baseline_zh_tokens[i]])
        base_en = tokenizer.decode([baseline_en_tokens[i]])
        print(f"\n  Problem {i}: zh baseline='{base_zh}', en baseline='{base_en}'")
        for l in [20, 26, 32]:
            if l in EXIT_LAYERS:
                early_zh = tokenizer.decode([early_zh_tokens[l][i]])
                early_en = tokenizer.decode([early_en_tokens[l][i]])
                zh_ok = "✓" if early_zh_tokens[l][i] == baseline_zh_tokens[i] else "✗"
                en_ok = "✓" if early_en_tokens[l][i] == baseline_en_tokens[i] else "✗"
                print(f"    L{l}: zh='{early_zh}' {zh_ok}  en='{early_en}' {en_ok}")

    # Save
    outpath = OUTPUT_DIR / "intervention_early_exit.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
