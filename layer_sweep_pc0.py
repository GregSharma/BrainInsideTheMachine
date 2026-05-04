"""Layer sweep: test PC0 swap at multiple layers to find the factorization boundary.

For each layer L in [8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34]:
1. Extract hidden states at L for 200 problems (zh + en)
2. Fit PCA, get PC0 (language axis) at that layer
3. Run PC0 swap on 20 test problems
4. Measure: language of output, first-token match with en baseline, text match

Question: Is L26 special, or does the factorization work everywhere?
"""

import numpy as np
import torch
import json
import random as pyrandom
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import re
import gc

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
N_PROBLEMS = 20
N_PCA = 200
MAX_TOKENS = 64

SWEEP_LAYERS = [8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34]


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


def generate_tokens(model, tokenizer, input_ids, past_key_values, max_tokens):
    """Autoregressive generation from a starting state."""
    tokens = []
    next_token = input_ids
    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            if next_token.item() == eos_id:
                break
    return tokens


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

    problems = generate_problems(N_PCA, seed=42)
    selected = select_problems(problems, N_PROBLEMS)

    # ================================================================
    # First: generate English baselines (once, shared across all layers)
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
        gen_tokens = generate_tokens(model, tokenizer, next_tok, outputs.past_key_values, MAX_TOKENS - 1)
        all_tokens = [first_tok] + gen_tokens
        en_baselines[idx] = {
            "tokens": all_tokens,
            "text": tokenizer.decode(all_tokens, skip_special_tokens=True),
            "first_token": first_tok,
        }

    # ================================================================
    # Layer sweep
    # ================================================================
    results = {
        "model": MODEL_NAME,
        "n_problems": N_PROBLEMS,
        "n_pca": N_PCA,
        "max_tokens": MAX_TOKENS,
        "sweep_layers": SWEEP_LAYERS,
        "en_baselines": {str(k): v for k, v in en_baselines.items()},
        "layer_results": {}
    }

    for sweep_layer in SWEEP_LAYERS:
        print(f"\n{'='*70}")
        print(f"LAYER {sweep_layer}: Extract + PCA + PC0 swap")
        print(f"{'='*70}")

        # Extract hidden states at this layer for all 200 problems
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

        # PCA
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

        zh_mean_proj = zh_proj.mean()
        en_mean_proj = en_proj.mean()

        print(f"  PC0 var explained: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  Cohen's d: {cohens_d:.2f}")

        # PC0 swap on test problems
        layer_result = {
            "pc0_var_explained": float(pca.explained_variance_ratio_[0]),
            "cohens_d": float(cohens_d),
            "zh_mean_proj": float(zh_mean_proj),
            "en_mean_proj": float(en_mean_proj),
            "per_problem": []
        }

        n_en = 0
        n_first_tok_match = 0
        n_exact_text = 0
        total_exact_chars = 0
        total_chars = 0

        for idx in tqdm(selected, desc=f"PC0 swap L{sweep_layer}", leave=False):
            prob = problems[idx]

            # Get zh hidden state at this layer
            inputs_zh = tokenizer(prob["zh"], return_tensors="pt").to(model.device)

            handle = model.model.layers[sweep_layer].register_forward_hook(capture_hook)
            with torch.no_grad():
                model(**inputs_zh)
            zh_h = layer_output['h'].cpu().float().numpy().squeeze()  # (d,)
            layer_output.clear()
            handle.remove()

            # PC0 swap
            zh_h_unit = zh_h / np.linalg.norm(zh_h)
            zh_pc0 = float(zh_h_unit @ pc0)
            zh_h_swapped = zh_h_unit - zh_pc0 * pc0 + en_mean_proj * pc0
            zh_h_swapped = zh_h_swapped * np.linalg.norm(zh_h)
            zh_h_swapped_t = torch.tensor(zh_h_swapped, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to(model.device)

            # Generate with splice
            tokens = run_with_splice(model, tokenizer, prob["en"], sweep_layer, zh_h_swapped_t, MAX_TOKENS)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            lang = detect_language(text)

            # Compare to en baseline
            en_text = en_baselines[idx]["text"]
            en_first = en_baselines[idx]["first_token"]
            first_tok_match = tokens[0] == en_first

            # Exact character match length
            min_len = min(len(text), len(en_text))
            exact_chars = min_len
            for j in range(min_len):
                if text[j] != en_text[j]:
                    exact_chars = j
                    break

            if lang == "en":
                n_en += 1
            if first_tok_match:
                n_first_tok_match += 1
            if exact_chars == min_len:
                n_exact_text += 1
            total_exact_chars += exact_chars
            total_chars += min_len

            layer_result["per_problem"].append({
                "prob_idx": idx,
                "lang": lang,
                "first_token_match_en": first_tok_match,
                "exact_chars": exact_chars,
                "total_chars": min_len,
                "text_snippet": text[:80],
            })

        layer_result["summary"] = {
            "pct_english": n_en / N_PROBLEMS,
            "first_token_match_rate": n_first_tok_match / N_PROBLEMS,
            "exact_text_match_rate": n_exact_text / N_PROBLEMS,
            "mean_exact_char_pct": total_exact_chars / max(total_chars, 1),
        }

        print(f"  English output: {n_en}/{N_PROBLEMS} ({n_en/N_PROBLEMS:.0%})")
        print(f"  First token match: {n_first_tok_match}/{N_PROBLEMS} ({n_first_tok_match/N_PROBLEMS:.0%})")
        print(f"  Exact text match: {n_exact_text}/{N_PROBLEMS} ({n_exact_text/N_PROBLEMS:.0%})")
        print(f"  Mean char match: {total_exact_chars/max(total_chars,1):.1%}")

        results["layer_results"][str(sweep_layer)] = layer_result

        # Clear GPU cache between layers
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    outpath = OUTPUT_DIR / "layer_sweep_pc0.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Final summary table
    print(f"\n{'='*70}")
    print("LAYER SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Layer':>6s}  {'PC0 var':>8s}  {'Cohen d':>8s}  {'%English':>8s}  {'1st tok':>8s}  {'Exact':>8s}  {'Char%':>8s}")
    print("-" * 70)
    for layer in SWEEP_LAYERS:
        r = results["layer_results"][str(layer)]
        s = r["summary"]
        print(f"  L{layer:<4d}  {r['pc0_var_explained']:>7.1%}  {r['cohens_d']:>8.1f}  "
              f"{s['pct_english']:>7.0%}  {s['first_token_match_rate']:>7.0%}  "
              f"{s['exact_text_match_rate']:>7.0%}  {s['mean_exact_char_pct']:>7.1%}")


if __name__ == "__main__":
    main()
