"""Experiment 1: All-Token PC0 Swap with Chinese KV Cache.

The clean test that settles the confound.

Run Chinese prompt normally through all 36 layers for all tokens.
At the injection layer, swap PC0 projection for EVERY token in the sequence.
KV cache is Chinese everywhere — no English signal except the PC0 nudge.
Generate 64+ tokens, measure: output language, text match with English baseline.

Run at L26 first, then at L12, L20, L28, L30, L34.

If output is English: PC0 alone controls language. No confound.
If output is Chinese: English KV cache was doing most work.
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

# L28-L29 is where R² peaks, but test range
SWEEP_LAYERS = [12, 20, 26, 28, 30, 34]


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


def run_alltoken_pc0_swap(model, tokenizer, prompt_zh, splice_layer, pc0, zh_mean_proj, en_mean_proj, max_tokens=64):
    """Run Chinese prompt with ALL tokens' PC0 projections swapped to English mean.

    KV cache is purely Chinese. Only modification is PC0 direction at every position.
    """
    inputs = tokenizer(prompt_zh, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    def alltoken_pc0_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        new_h = h.clone()

        # For EVERY token position, swap PC0 projection
        for pos in range(new_h.shape[1]):
            h_pos = new_h[0, pos, :].float().cpu().numpy()
            h_norm = np.linalg.norm(h_pos)
            if h_norm < 1e-8:
                continue
            h_unit = h_pos / h_norm
            h_pc0 = float(h_unit @ pc0)
            # Swap: remove current PC0, add English mean projection
            h_swapped_unit = h_unit - h_pc0 * pc0 + en_mean_proj * pc0
            h_swapped = h_swapped_unit * h_norm
            new_h[0, pos, :] = torch.tensor(h_swapped, dtype=h.dtype, device=h.device)

        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    handle = model.model.layers[splice_layer].register_forward_hook(alltoken_pc0_hook)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

    handle.remove()

    # Generate autoregressively (no hook — subsequent tokens go through normally)
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


def run_lasttoken_pc0_swap(model, tokenizer, prompt_zh, splice_layer, pc0, zh_mean_proj, en_mean_proj, max_tokens=64):
    """Control: same as existing — only swap LAST token's PC0, Chinese KV."""
    inputs = tokenizer(prompt_zh, return_tensors="pt").to(model.device)

    def lasttoken_pc0_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        new_h = h.clone()

        h_last = new_h[0, -1, :].float().cpu().numpy()
        h_norm = np.linalg.norm(h_last)
        h_unit = h_last / h_norm
        h_pc0 = float(h_unit @ pc0)
        h_swapped_unit = h_unit - h_pc0 * pc0 + en_mean_proj * pc0
        h_swapped = h_swapped_unit * h_norm
        new_h[0, -1, :] = torch.tensor(h_swapped, dtype=h.dtype, device=h.device)

        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    handle = model.model.layers[splice_layer].register_forward_hook(lasttoken_pc0_hook)

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

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model: {n_layers} layers, d={d}")

    problems = generate_problems(N_PCA, seed=42)
    selected = select_problems(problems, N_PROBLEMS)

    # Generate English baselines
    print("\nGenerating English baselines...")
    en_baselines = {}
    for idx in tqdm(selected, desc="en baselines"):
        prob = problems[idx]
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        first_tok = int(outputs.logits[0, -1].argmax())
        next_tok = torch.tensor([[first_tok]], device=model.device)
        gen_tokens = [first_tok]
        pkv = outputs.past_key_values
        with torch.no_grad():
            for _ in range(MAX_TOKENS - 1):
                outputs = model(next_tok, past_key_values=pkv, use_cache=True)
                pkv = outputs.past_key_values
                next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id:
                    break
        en_baselines[idx] = {
            "tokens": gen_tokens,
            "text": tokenizer.decode(gen_tokens, skip_special_tokens=True),
            "first_token": gen_tokens[0],
        }

    # Generate Chinese baselines
    print("Generating Chinese baselines...")
    zh_baselines = {}
    for idx in tqdm(selected, desc="zh baselines"):
        prob = problems[idx]
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
        first_tok = int(outputs.logits[0, -1].argmax())
        next_tok = torch.tensor([[first_tok]], device=model.device)
        gen_tokens = [first_tok]
        pkv = outputs.past_key_values
        with torch.no_grad():
            for _ in range(MAX_TOKENS - 1):
                outputs = model(next_tok, past_key_values=pkv, use_cache=True)
                pkv = outputs.past_key_values
                next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id:
                    break
        zh_baselines[idx] = {
            "tokens": gen_tokens,
            "text": tokenizer.decode(gen_tokens, skip_special_tokens=True),
        }

    results = {
        "model": MODEL_NAME,
        "experiment": "all_token_pc0_swap",
        "sweep_layers": SWEEP_LAYERS,
        "n_problems": N_PROBLEMS,
        "n_pca": N_PCA,
        "max_tokens": MAX_TOKENS,
        "conditions": ["alltoken_pc0_zhkv", "lasttoken_pc0_zhkv"],
        "en_baselines": {str(k): v for k, v in en_baselines.items()},
        "layer_results": {}
    }

    for sweep_layer in SWEEP_LAYERS:
        print(f"\n{'='*70}")
        print(f"LAYER {sweep_layer}")
        print(f"{'='*70}")

        # Extract hidden states and fit PCA
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
        cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt(
            (zh_proj.std()**2 + en_proj.std()**2) / 2
        )
        zh_mean_proj = float(zh_proj.mean())
        en_mean_proj = float(en_proj.mean())

        print(f"  PC0 var: {pca.explained_variance_ratio_[0]:.1%}, Cohen's d: {cohens_d:.1f}")
        print(f"  zh mean proj: {zh_mean_proj:.4f}, en mean proj: {en_mean_proj:.4f}")

        layer_result = {
            "pc0_var_explained": float(pca.explained_variance_ratio_[0]),
            "cohens_d": float(cohens_d),
            "zh_mean_proj": zh_mean_proj,
            "en_mean_proj": en_mean_proj,
            "per_problem": []
        }

        # Run both conditions on each test problem
        for idx in tqdm(selected, desc=f"L{sweep_layer} interventions", leave=False):
            prob = problems[idx]
            en_text = en_baselines[idx]["text"]
            en_first = en_baselines[idx]["first_token"]

            prob_result = {"prob_idx": idx}

            # Condition 1: ALL-TOKEN PC0 swap with Chinese KV
            tokens_all = run_alltoken_pc0_swap(
                model, tokenizer, prob["zh"], sweep_layer,
                pc0, zh_mean_proj, en_mean_proj, MAX_TOKENS
            )
            text_all = tokenizer.decode(tokens_all, skip_special_tokens=True)
            lang_all = detect_language(text_all)

            # Condition 2: LAST-TOKEN only PC0 swap with Chinese KV
            tokens_last = run_lasttoken_pc0_swap(
                model, tokenizer, prob["zh"], sweep_layer,
                pc0, zh_mean_proj, en_mean_proj, MAX_TOKENS
            )
            text_last = tokenizer.decode(tokens_last, skip_special_tokens=True)
            lang_last = detect_language(text_last)

            # Score against English baseline
            for label, tokens, text, lang in [
                ("alltoken_pc0_zhkv", tokens_all, text_all, lang_all),
                ("lasttoken_pc0_zhkv", tokens_last, text_last, lang_last),
            ]:
                first_match = tokens[0] == en_first
                min_len = min(len(text), len(en_text))
                exact_chars = min_len
                for j in range(min_len):
                    if text[j] != en_text[j]:
                        exact_chars = j
                        break

                prob_result[label] = {
                    "lang": lang,
                    "first_token_match": first_match,
                    "exact_chars": exact_chars,
                    "total_chars": min_len,
                    "text_snippet": text[:120],
                }

            layer_result["per_problem"].append(prob_result)

        # Aggregate
        for cond in ["alltoken_pc0_zhkv", "lasttoken_pc0_zhkv"]:
            n_en = sum(1 for p in layer_result["per_problem"] if p[cond]["lang"] == "en")
            n_first = sum(1 for p in layer_result["per_problem"] if p[cond]["first_token_match"])
            n_exact = sum(1 for p in layer_result["per_problem"]
                        if p[cond]["exact_chars"] == p[cond]["total_chars"])
            total_ec = sum(p[cond]["exact_chars"] for p in layer_result["per_problem"])
            total_tc = sum(p[cond]["total_chars"] for p in layer_result["per_problem"])

            layer_result[f"{cond}_summary"] = {
                "pct_english": n_en / N_PROBLEMS,
                "first_tok_rate": n_first / N_PROBLEMS,
                "exact_text_rate": n_exact / N_PROBLEMS,
                "char_match_pct": total_ec / max(total_tc, 1),
            }

            s = layer_result[f"{cond}_summary"]
            print(f"  {cond}: {s['pct_english']:.0%} en, {s['first_tok_rate']:.0%} 1st, "
                  f"{s['exact_text_rate']:.0%} exact, {s['char_match_pct']:.1%} char")

        results["layer_results"][str(sweep_layer)] = layer_result

        gc.collect()
        torch.cuda.empty_cache()

    # Save
    outpath = OUTPUT_DIR / "exp1_alltoken_pc0_swap.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Summary table
    print(f"\n{'='*80}")
    print("EXPERIMENT 1 SUMMARY: ALL-TOKEN vs LAST-TOKEN PC0 SWAP (Chinese KV)")
    print(f"{'='*80}")
    print(f"{'Layer':>6s}  {'Condition':>22s}  {'%En':>5s}  {'1stTok':>6s}  {'Exact':>6s}  {'Char%':>6s}")
    print("-" * 60)
    for layer in SWEEP_LAYERS:
        r = results["layer_results"][str(layer)]
        for cond in ["alltoken_pc0_zhkv", "lasttoken_pc0_zhkv"]:
            s = r[f"{cond}_summary"]
            print(f"  L{layer:<4d}  {cond:>22s}  {s['pct_english']:>4.0%}  "
                  f"{s['first_tok_rate']:>5.0%}  {s['exact_text_rate']:>5.0%}  "
                  f"{s['char_match_pct']:>5.1%}")


if __name__ == "__main__":
    main()
