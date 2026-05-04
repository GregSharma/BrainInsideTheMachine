"""Layer sweep with full controls: decompose KV cache vs PC0 contribution.

At each layer in [8, 12, 16, 20, 24, 26, 28, 30, 32, 34]:
  A) PC0 swap + English KV  (already done, re-run for consistency)
  B) Raw splice + English KV  (no PC0 swap — KV-only baseline)
  C) Random dir + English KV  (noise baseline)
  D) PC0 swap + Chinese KV   (isolates PC0 from KV cache)

Also: save PC0 vectors at each layer and compute cross-layer cosine.
"""

import numpy as np
import torch
import json
import random as pyrandom
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import gc

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
N_PROBLEMS = 20
N_PCA = 200
MAX_TOKENS = 64

SWEEP_LAYERS = [8, 12, 16, 20, 24, 26, 28, 30, 32, 34]


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


def select_problems(problems, n=20):
    selected = []
    cat_count = {i: 0 for i in range(5)}
    for i, p in enumerate(problems):
        c = p['category']
        if cat_count[c] < n // 5:
            selected.append(i)
            cat_count[c] += 1
        if len(selected) == n:
            break
    return selected


def detect_language(text):
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "numeric"
    ratio = chinese_chars / total_alpha
    if ratio > 0.3:
        return "zh"
    elif ratio < 0.1:
        return "en"
    else:
        return "mixed"


def run_with_splice(model, tokenizer, prompt_target, splice_layer, h_inject, max_tokens=64):
    """Run target prompt, inject h_inject at splice_layer output, generate."""
    inputs = tokenizer(prompt_target, return_tensors="pt").to(model.device)

    def splice_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        new_h = h.clone()
        new_h[:, -1:, :] = h_inject.to(h.device)
        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    handle = model.model.layers[splice_layer].register_forward_hook(splice_hook)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

    handle.remove()

    first_token_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[first_token_id]], device=model.device)
    tokens = [first_token_id]

    with torch.no_grad():
        for _ in range(max_tokens - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokens


def extract_hidden_at_layer(model, tokenizer, prompt, layer_idx):
    """Extract last-token hidden state at a specific layer."""
    layer_output = {}

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        layer_output['h'] = h.detach()[:, -1, :]

    handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return layer_output['h'].cpu().float().numpy().squeeze()


def evaluate_condition(tokens, en_baseline_text, en_baseline_first_token, tokenizer):
    """Score a generation against English baseline."""
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    lang = detect_language(text)
    first_tok_match = tokens[0] == en_baseline_first_token

    min_len = min(len(text), len(en_baseline_text))
    exact_chars = min_len
    for j in range(min_len):
        if text[j] != en_baseline_text[j]:
            exact_chars = j
            break

    return {
        "lang": lang,
        "is_english": lang == "en",
        "first_token_match": first_tok_match,
        "exact_chars": exact_chars,
        "total_chars": min_len,
        "exact_text_match": exact_chars == min_len,
        "char_match_pct": exact_chars / max(min_len, 1),
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model: {n_layers} layers, d={d}")

    problems = generate_problems(N_PCA, seed=42)
    selected = select_problems(problems, N_PROBLEMS)

    # ================================================================
    # English baselines (once)
    # ================================================================
    print("\nGenerating English baselines...")
    en_baselines = {}
    for idx in tqdm(selected, desc="en baselines"):
        prob = problems[idx]
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        first_tok = int(outputs.logits[0, -1].argmax())
        next_tok = torch.tensor([[first_tok]], device=model.device)
        tokens = [first_tok]
        with torch.no_grad():
            for _ in range(MAX_TOKENS - 1):
                outputs = model(next_tok, past_key_values=outputs.past_key_values, use_cache=True)
                next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id:
                    break
        en_baselines[idx] = {
            "text": tokenizer.decode(tokens, skip_special_tokens=True),
            "first_token": first_tok,
        }

    # ================================================================
    # Layer sweep with all controls
    # ================================================================
    results = {
        "model": MODEL_NAME,
        "sweep_layers": SWEEP_LAYERS,
        "n_problems": N_PROBLEMS,
        "n_pca": N_PCA,
        "max_tokens": MAX_TOKENS,
        "conditions": ["pc0_swap_en_kv", "raw_splice_en_kv", "random_dir_en_kv", "pc0_swap_zh_kv"],
        "pc0_vectors": {},  # layer -> list (for cross-layer cosine)
        "layer_results": {},
    }

    for sweep_layer in SWEEP_LAYERS:
        print(f"\n{'='*70}")
        print(f"LAYER {sweep_layer}")
        print(f"{'='*70}")

        # --- Extract hidden states for PCA ---
        layer_output = {}
        def capture_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            layer_output['h'] = h.detach()[:, -1, :]

        handle = model.model.layers[sweep_layer].register_forward_hook(capture_hook)

        zh_hidden = np.zeros((N_PCA, d), dtype=np.float32)
        en_hidden = np.zeros((N_PCA, d), dtype=np.float32)

        for i, prob in enumerate(tqdm(problems, desc=f"zh L{sweep_layer}", leave=False)):
            inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            zh_hidden[i] = layer_output['h'].cpu().float().numpy()
            layer_output.clear()

        for i, prob in enumerate(tqdm(problems, desc=f"en L{sweep_layer}", leave=False)):
            inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            en_hidden[i] = layer_output['h'].cpu().float().numpy()
            layer_output.clear()

        handle.remove()

        # --- PCA ---
        zh_norms = np.linalg.norm(zh_hidden, axis=1, keepdims=True)
        en_norms = np.linalg.norm(en_hidden, axis=1, keepdims=True)
        zh_unit = zh_hidden / zh_norms
        en_unit = en_hidden / en_norms

        combined = np.vstack([zh_unit, en_unit])
        pca = PCA(n_components=10)
        pca.fit(combined)

        pc0 = pca.components_[0]
        zh_proj = zh_unit @ pc0
        en_proj = en_unit @ pc0
        cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt((zh_proj.std()**2 + en_proj.std()**2) / 2)
        zh_mean_proj = float(zh_proj.mean())
        en_mean_proj = float(en_proj.mean())

        # Save PC0 vector
        results["pc0_vectors"][str(sweep_layer)] = pc0.tolist()

        print(f"  PC0 var: {pca.explained_variance_ratio_[0]:.1%}, Cohen's d: {cohens_d:.1f}")

        # --- Run 4 conditions on each test problem ---
        layer_result = {
            "pc0_var_explained": float(pca.explained_variance_ratio_[0]),
            "cohens_d": float(cohens_d),
            "zh_mean_proj": zh_mean_proj,
            "en_mean_proj": en_mean_proj,
            "conditions": {},
        }

        for cond_name in results["conditions"]:
            layer_result["conditions"][cond_name] = {
                "n_english": 0, "n_first_tok": 0, "n_exact_text": 0,
                "total_exact_chars": 0, "total_chars": 0,
            }

        for idx in tqdm(selected, desc=f"controls L{sweep_layer}", leave=False):
            prob = problems[idx]
            en_text = en_baselines[idx]["text"]
            en_first = en_baselines[idx]["first_token"]

            # Get zh hidden state at this layer
            zh_h = extract_hidden_at_layer(model, tokenizer, prob["zh"], sweep_layer)
            zh_h_unit = zh_h / np.linalg.norm(zh_h)
            zh_pc0 = float(zh_h_unit @ pc0)
            original_norm = np.linalg.norm(zh_h)

            # === Condition A: PC0 swap + English KV ===
            zh_h_swapped = zh_h_unit - zh_pc0 * pc0 + en_mean_proj * pc0
            zh_h_swapped = zh_h_swapped * original_norm
            h_inject = torch.tensor(zh_h_swapped, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to(model.device)
            tokens = run_with_splice(model, tokenizer, prob["en"], sweep_layer, h_inject, MAX_TOKENS)
            res = evaluate_condition(tokens, en_text, en_first, tokenizer)
            c = layer_result["conditions"]["pc0_swap_en_kv"]
            c["n_english"] += res["is_english"]
            c["n_first_tok"] += res["first_token_match"]
            c["n_exact_text"] += res["exact_text_match"]
            c["total_exact_chars"] += res["exact_chars"]
            c["total_chars"] += res["total_chars"]

            # === Condition B: Raw splice + English KV (no PC0 swap) ===
            h_raw = torch.tensor(zh_h, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to(model.device)
            tokens = run_with_splice(model, tokenizer, prob["en"], sweep_layer, h_raw, MAX_TOKENS)
            res = evaluate_condition(tokens, en_text, en_first, tokenizer)
            c = layer_result["conditions"]["raw_splice_en_kv"]
            c["n_english"] += res["is_english"]
            c["n_first_tok"] += res["first_token_match"]
            c["n_exact_text"] += res["exact_text_match"]
            c["total_exact_chars"] += res["exact_chars"]
            c["total_chars"] += res["total_chars"]

            # === Condition C: Random direction + English KV ===
            rng_dir = np.random.RandomState(idx * 1000 + sweep_layer)
            random_dir = rng_dir.randn(d).astype(np.float32)
            random_dir = random_dir / np.linalg.norm(random_dir)
            swap_mag = abs(zh_pc0 - en_mean_proj)
            zh_h_random = zh_h_unit + swap_mag * random_dir
            zh_h_random = zh_h_random * original_norm
            h_rand = torch.tensor(zh_h_random, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to(model.device)
            tokens = run_with_splice(model, tokenizer, prob["en"], sweep_layer, h_rand, MAX_TOKENS)
            res = evaluate_condition(tokens, en_text, en_first, tokenizer)
            c = layer_result["conditions"]["random_dir_en_kv"]
            c["n_english"] += res["is_english"]
            c["n_first_tok"] += res["first_token_match"]
            c["n_exact_text"] += res["exact_text_match"]
            c["total_exact_chars"] += res["exact_chars"]
            c["total_chars"] += res["total_chars"]

            # === Condition D: PC0 swap + Chinese KV (the critical control) ===
            tokens = run_with_splice(model, tokenizer, prob["zh"], sweep_layer, h_inject, MAX_TOKENS)
            res = evaluate_condition(tokens, en_text, en_first, tokenizer)
            c = layer_result["conditions"]["pc0_swap_zh_kv"]
            c["n_english"] += res["is_english"]
            c["n_first_tok"] += res["first_token_match"]
            c["n_exact_text"] += res["exact_text_match"]
            c["total_exact_chars"] += res["exact_chars"]
            c["total_chars"] += res["total_chars"]

        # Compute summaries
        for cond_name, c in layer_result["conditions"].items():
            c["pct_english"] = c["n_english"] / N_PROBLEMS
            c["first_tok_rate"] = c["n_first_tok"] / N_PROBLEMS
            c["exact_text_rate"] = c["n_exact_text"] / N_PROBLEMS
            c["char_match_pct"] = c["total_exact_chars"] / max(c["total_chars"], 1)

        results["layer_results"][str(sweep_layer)] = layer_result

        # Print summary for this layer
        print(f"  {'Condition':<22s}  {'%En':>5s}  {'1st':>5s}  {'Exact':>5s}  {'Char%':>6s}")
        print(f"  {'-'*50}")
        for cond_name in results["conditions"]:
            c = layer_result["conditions"][cond_name]
            print(f"  {cond_name:<22s}  {c['pct_english']:>4.0%}  {c['first_tok_rate']:>4.0%}  "
                  f"{c['exact_text_rate']:>4.0%}  {c['char_match_pct']:>5.1%}")

        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # PC0 cross-layer cosine similarity
    # ================================================================
    print(f"\n{'='*70}")
    print("PC0 CROSS-LAYER COSINE SIMILARITY")
    print(f"{'='*70}")

    pc0_cosines = {}
    for i, l1 in enumerate(SWEEP_LAYERS):
        for l2 in SWEEP_LAYERS[i+1:]:
            v1 = np.array(results["pc0_vectors"][str(l1)])
            v2 = np.array(results["pc0_vectors"][str(l2)])
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            pc0_cosines[f"L{l1}_L{l2}"] = cos

    results["pc0_cross_layer_cosine"] = pc0_cosines

    # Print adjacent layer cosines
    print(f"  {'Pair':<12s}  {'Cosine':>8s}")
    print(f"  {'-'*22}")
    for i in range(len(SWEEP_LAYERS) - 1):
        l1, l2 = SWEEP_LAYERS[i], SWEEP_LAYERS[i+1]
        cos = pc0_cosines[f"L{l1}_L{l2}"]
        print(f"  L{l1}-L{l2:<4d}  {cos:>8.3f}")

    # Also show L8 vs L34 (extremes)
    cos_extreme = pc0_cosines[f"L{SWEEP_LAYERS[0]}_L{SWEEP_LAYERS[-1]}"]
    print(f"  L{SWEEP_LAYERS[0]}-L{SWEEP_LAYERS[-1]:<4d}  {cos_extreme:>8.3f}  (extreme)")

    # ================================================================
    # Final summary table
    # ================================================================
    print(f"\n{'='*90}")
    print("FULL CONTROL DECOMPOSITION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Layer':>6s}  {'PC0+EnKV':>9s}  {'Raw+EnKV':>9s}  {'Rand+EnKV':>10s}  {'PC0+ZhKV':>9s}  "
          f"{'PC0 delta':>10s}  {'Cohen d':>8s}")
    print("-" * 90)
    for layer in SWEEP_LAYERS:
        r = results["layer_results"][str(layer)]
        pc0_en = r["conditions"]["pc0_swap_en_kv"]["pct_english"]
        raw_en = r["conditions"]["raw_splice_en_kv"]["pct_english"]
        rand_en = r["conditions"]["random_dir_en_kv"]["pct_english"]
        pc0_zh = r["conditions"]["pc0_swap_zh_kv"]["pct_english"]
        # PC0 delta = PC0 swap rate - raw splice rate (PC0's contribution beyond KV cache)
        delta = pc0_en - raw_en
        print(f"  L{layer:<4d}  {pc0_en:>8.0%}  {raw_en:>8.0%}  {rand_en:>9.0%}  {pc0_zh:>8.0%}  "
              f"{delta:>+9.0%}  {r['cohens_d']:>8.1f}")

    # Also print exact text match version
    print(f"\n{'Layer':>6s}  {'PC0+En':>7s}  {'Raw+En':>7s}  {'Rand+En':>8s}  {'PC0+Zh':>7s}  (exact text match rate)")
    print("-" * 70)
    for layer in SWEEP_LAYERS:
        r = results["layer_results"][str(layer)]
        pc0_en = r["conditions"]["pc0_swap_en_kv"]["exact_text_rate"]
        raw_en = r["conditions"]["raw_splice_en_kv"]["exact_text_rate"]
        rand_en = r["conditions"]["random_dir_en_kv"]["exact_text_rate"]
        pc0_zh = r["conditions"]["pc0_swap_zh_kv"]["exact_text_rate"]
        print(f"  L{layer:<4d}  {pc0_en:>6.0%}  {raw_en:>6.0%}  {rand_en:>7.0%}  {pc0_zh:>6.0%}")

    # Save
    # Remove pc0_vectors from JSON (large) — save separately as npz
    pc0_vecs = {k: np.array(v) for k, v in results["pc0_vectors"].items()}
    np.savez_compressed(OUTPUT_DIR / "pc0_vectors_per_layer.npz", **pc0_vecs)
    del results["pc0_vectors"]

    outpath = OUTPUT_DIR / "layer_sweep_controls.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    print(f"PC0 vectors saved to output/pc0_vectors_per_layer.npz")


if __name__ == "__main__":
    main()
