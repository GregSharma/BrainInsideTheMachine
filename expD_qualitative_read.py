"""Exp D: Qualitative PC0+ZhKV Output Read.

Generate text for 5 problems at L28 under ALL 4 conditions + baselines.
SAVE THE ACTUAL TEXT. We keep saying "100% English, 0% text match" — what does it look like?

Conditions:
  - zh baseline (no intervention)
  - en baseline (no intervention)
  - PC0 swap + English KV (splice zh→en PC0, run on en prompt for KV)
  - PC0 swap + Chinese KV (splice zh→en PC0, run on zh prompt for KV) ← KEY
  - Raw splice + English KV (zh hidden, en KV, no PC0 swap)

Output: full text for each condition × problem, printed and saved to JSON.
"""

import numpy as np
import torch
import json
import random as pyrandom
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import gc

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
LAYER = 28
N_PROBLEMS = 5
N_PCA = 200
MAX_TOKENS = 128  # longer than usual — want to see full reasoning

print("=" * 70)
print(f"EXP D: QUALITATIVE READ — L{LAYER}, {N_PROBLEMS} problems, {MAX_TOKENS} tokens")
print("=" * 70)


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


def select_problems(problems, n=5):
    selected = []
    cat_count = {i: 0 for i in range(5)}
    for i, p in enumerate(problems):
        c = p['category']
        if cat_count[c] < 1:  # 1 per category for 5 total
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


def generate_baseline(model, tokenizer, prompt, max_tokens=128):
    """Generate from a prompt with no intervention. Return tokens + text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    first_tok = int(outputs.logits[0, -1].argmax())
    next_tok = torch.tensor([[first_tok]], device=model.device)
    tokens = [first_tok]
    past_kv = outputs.past_key_values
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            outputs = model(next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_tok.item())
            if next_tok.item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, text


def run_with_splice(model, tokenizer, prompt_target, splice_layer, h_inject, max_tokens=128):
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
        past_kv = outputs.past_key_values
    handle.remove()

    first_tok = int(outputs.logits[0, -1].argmax())
    next_tok = torch.tensor([[first_tok]], device=model.device)
    tokens = [first_tok]
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            outputs = model(next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_tok.item())
            if next_tok.item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, text


def extract_hidden_at_layer(model, tokenizer, prompt, layer_idx):
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


def main():
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    d = model.config.hidden_size

    problems = generate_problems(N_PCA, seed=42)
    selected = select_problems(problems, N_PROBLEMS)

    # --- PCA for PC0 at target layer ---
    print(f"\nFitting PCA at L{LAYER} on {N_PCA} problems...", flush=True)
    zh_hidden = np.zeros((N_PCA, d), dtype=np.float32)
    en_hidden = np.zeros((N_PCA, d), dtype=np.float32)
    for i, prob in enumerate(problems):
        zh_hidden[i] = extract_hidden_at_layer(model, tokenizer, prob["zh"], LAYER)
        en_hidden[i] = extract_hidden_at_layer(model, tokenizer, prob["en"], LAYER)
        if (i + 1) % 50 == 0:
            print(f"  PCA extraction: {i+1}/{N_PCA}", flush=True)

    zh_norms = np.linalg.norm(zh_hidden, axis=1, keepdims=True)
    en_norms = np.linalg.norm(en_hidden, axis=1, keepdims=True)
    zh_unit = zh_hidden / zh_norms
    en_unit = en_hidden / en_norms
    combined = np.vstack([zh_unit, en_unit])
    pca = PCA(n_components=10)
    pca.fit(combined)
    pc0 = pca.components_[0]
    en_mean_proj = float((en_unit @ pc0).mean())
    zh_mean_proj = float((zh_unit @ pc0).mean())
    print(f"  PC0 var: {pca.explained_variance_ratio_[0]:.1%}", flush=True)
    print(f"  zh mean proj: {zh_mean_proj:.4f}, en mean proj: {en_mean_proj:.4f}", flush=True)

    # --- Generate all conditions ---
    results = {
        "layer": LAYER,
        "max_tokens": MAX_TOKENS,
        "pc0_var": float(pca.explained_variance_ratio_[0]),
        "problems": []
    }

    for pi, idx in enumerate(selected):
        prob = problems[idx]
        print(f"\n{'='*70}", flush=True)
        print(f"PROBLEM {pi+1}/{N_PROBLEMS}: {prob['en']}", flush=True)
        print(f"  (Chinese: {prob['zh']})", flush=True)
        print(f"{'='*70}", flush=True)

        prob_result = {
            "idx": idx,
            "category": prob["category"],
            "prompt_en": prob["en"],
            "prompt_zh": prob["zh"],
            "conditions": {}
        }

        # --- Baselines ---
        _, en_text = generate_baseline(model, tokenizer, prob["en"], MAX_TOKENS)
        print(f"\n  EN BASELINE:\n    {en_text[:200]}", flush=True)
        prob_result["conditions"]["en_baseline"] = {"text": en_text, "lang": detect_language(en_text)}

        _, zh_text = generate_baseline(model, tokenizer, prob["zh"], MAX_TOKENS)
        print(f"\n  ZH BASELINE:\n    {zh_text[:200]}", flush=True)
        prob_result["conditions"]["zh_baseline"] = {"text": zh_text, "lang": detect_language(zh_text)}

        # --- Get zh hidden state ---
        zh_h = extract_hidden_at_layer(model, tokenizer, prob["zh"], LAYER)
        zh_h_unit = zh_h / np.linalg.norm(zh_h)
        zh_pc0 = float(zh_h_unit @ pc0)
        original_norm = float(np.linalg.norm(zh_h))

        # --- PC0 swap vector ---
        zh_h_swapped = zh_h_unit - zh_pc0 * pc0 + en_mean_proj * pc0
        zh_h_swapped = zh_h_swapped * original_norm
        h_inject = torch.tensor(zh_h_swapped, dtype=torch.float16).unsqueeze(0).unsqueeze(0)

        # --- Condition A: PC0 swap + English KV ---
        _, text_a = run_with_splice(model, tokenizer, prob["en"], LAYER, h_inject, MAX_TOKENS)
        print(f"\n  PC0_SWAP + EN_KV:\n    {text_a[:200]}", flush=True)
        prob_result["conditions"]["pc0_swap_en_kv"] = {"text": text_a, "lang": detect_language(text_a)}

        # --- Condition D: PC0 swap + Chinese KV (THE KEY ONE) ---
        _, text_d = run_with_splice(model, tokenizer, prob["zh"], LAYER, h_inject, MAX_TOKENS)
        print(f"\n  PC0_SWAP + ZH_KV (KEY):\n    {text_d[:200]}", flush=True)
        prob_result["conditions"]["pc0_swap_zh_kv"] = {"text": text_d, "lang": detect_language(text_d)}

        # --- Condition B: Raw splice + English KV ---
        h_raw = torch.tensor(zh_h, dtype=torch.float16).unsqueeze(0).unsqueeze(0)
        _, text_b = run_with_splice(model, tokenizer, prob["en"], LAYER, h_raw, MAX_TOKENS)
        print(f"\n  RAW_SPLICE + EN_KV:\n    {text_b[:200]}", flush=True)
        prob_result["conditions"]["raw_splice_en_kv"] = {"text": text_b, "lang": detect_language(text_b)}

        results["problems"].append(prob_result)
        gc.collect()
        torch.cuda.empty_cache()

    # --- Save ---
    outpath = OUTPUT_DIR / "expD_qualitative_read.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nSaved to {outpath}", flush=True)

    # --- Summary ---
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for pi, pr in enumerate(results["problems"]):
        print(f"\nProblem {pi+1}: {pr['prompt_en']}", flush=True)
        for cond, data in pr["conditions"].items():
            lang = data["lang"]
            snippet = data["text"][:80].replace('\n', ' ')
            print(f"  {cond:20s}  [{lang:5s}]  {snippet}...", flush=True)


if __name__ == "__main__":
    main()
