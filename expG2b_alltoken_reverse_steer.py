"""Experiment G2b: All-Token Reverse Steer (English → Chinese).

Tests whether the 35% en→zh rate from Exp 3 was suppressed by KV cache contamination.

Exp 3 swapped PC0 on the LAST TOKEN only. All other tokens in the KV cache
retained their original English PC0. The model's attention over those English-PC0
tokens may have pulled generation back toward English.

G2b: swap PC0 on ALL tokens at the intervention layer, not just the last token.
If the 35% was a KV contamination artifact, we expect %zh to jump toward 100%.
If it was genuinely about asymmetric basin widths, %zh stays ~35%.

Also runs last-token-only as a within-experiment control (should replicate Exp 3).

Sweep: L26 (attractor entry) and L30 (Exp 3 peak).
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

SWEEP_LAYERS = [26, 30]


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


def run_reverse_steer(model, tokenizer, prompt_en, splice_layer, pc0,
                      en_mean_proj, zh_mean_proj, max_tokens=64, all_tokens=False):
    """Run English prompt, swap PC0 from English→Chinese direction.

    If all_tokens=True, swap PC0 for EVERY token position.
    If all_tokens=False, swap PC0 for LAST token only (Exp 3 replication).
    """
    inputs = tokenizer(prompt_en, return_tensors="pt").to(model.device)

    def steer_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        new_h = h.clone()

        if all_tokens:
            positions = range(new_h.shape[1])
        else:
            positions = [new_h.shape[1] - 1]

        for pos in positions:
            h_pos = new_h[0, pos, :].float().cpu().numpy()
            h_norm = np.linalg.norm(h_pos)
            if h_norm < 1e-8:
                continue
            h_unit = h_pos / h_norm
            h_pc0 = float(h_unit @ pc0)
            # Remove English PC0 projection, add Chinese mean projection
            h_swapped_unit = h_unit - h_pc0 * pc0 + zh_mean_proj * pc0
            h_swapped = h_swapped_unit * h_norm
            new_h[0, pos, :] = torch.tensor(h_swapped, dtype=h.dtype, device=h.device)

        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    handle = model.model.layers[splice_layer].register_forward_hook(steer_hook)

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
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model: {n_layers} layers, d={d}")

    problems = generate_problems(N_PCA, seed=42)
    selected = select_problems(problems, N_PROBLEMS)

    # Generate Chinese baselines (target)
    print("\nGenerating Chinese baselines...")
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

    # Generate English baselines (source)
    print("Generating English baselines...")
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
        }

    results = {
        "model": MODEL_NAME,
        "experiment": "G2b_alltoken_reverse_steer",
        "hypothesis": "The 35% en→zh rate from Exp 3 was suppressed by KV cache contamination "
                      "from non-steered tokens retaining English PC0. Swapping ALL tokens should "
                      "increase %zh toward 100% if this is the bottleneck.",
        "sweep_layers": SWEEP_LAYERS,
        "n_problems": N_PROBLEMS,
        "n_pca": N_PCA,
        "max_tokens": MAX_TOKENS,
        "conditions": ["alltoken_reverse", "lasttoken_reverse"],
        "layer_results": {}
    }

    for sweep_layer in SWEEP_LAYERS:
        print(f"\n{'='*70}")
        print(f"LAYER {sweep_layer}: Reverse Steer en→zh")
        print(f"{'='*70}")

        # Extract hidden states for PCA at this layer
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

        # PCA on unit-normalized hidden states
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
        zh_mean_proj = float(zh_proj.mean())
        en_mean_proj = float(en_proj.mean())
        cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt(
            (zh_proj.std()**2 + en_proj.std()**2) / 2
        )

        print(f"  PC0 var: {pca.explained_variance_ratio_[0]:.1%}, Cohen's d: {cohens_d:.1f}")
        print(f"  zh mean proj: {zh_mean_proj:.4f}, en mean proj: {en_mean_proj:.4f}")

        layer_result = {
            "pc0_var_explained": float(pca.explained_variance_ratio_[0]),
            "cohens_d": float(cohens_d),
            "zh_mean_proj": zh_mean_proj,
            "en_mean_proj": en_mean_proj,
            "per_problem": []
        }

        for idx in tqdm(selected, desc=f"L{sweep_layer} steer", leave=False):
            prob = problems[idx]
            zh_text = zh_baselines[idx]["text"]

            prob_result = {"prob_idx": idx, "en_prompt": prob["en"][:80]}

            for cond_label, all_tok in [("alltoken_reverse", True), ("lasttoken_reverse", False)]:
                tokens = run_reverse_steer(
                    model, tokenizer, prob["en"], sweep_layer,
                    pc0, en_mean_proj, zh_mean_proj, MAX_TOKENS,
                    all_tokens=all_tok
                )
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                lang = detect_language(text)

                # Character match against Chinese baseline
                min_len = min(len(text), len(zh_text))
                exact_chars = min_len
                for j in range(min_len):
                    if text[j] != zh_text[j]:
                        exact_chars = j
                        break

                prob_result[cond_label] = {
                    "lang": lang,
                    "text_snippet": text[:200],
                    "zh_baseline_match_chars": exact_chars,
                    "zh_baseline_total_chars": min_len,
                }

            layer_result["per_problem"].append(prob_result)

        # Aggregate per condition
        for cond in ["alltoken_reverse", "lasttoken_reverse"]:
            n_zh = sum(1 for p in layer_result["per_problem"] if p[cond]["lang"] == "zh")
            n_en = sum(1 for p in layer_result["per_problem"] if p[cond]["lang"] == "en")
            n_mixed = sum(1 for p in layer_result["per_problem"] if p[cond]["lang"] == "mixed")
            n_num = sum(1 for p in layer_result["per_problem"] if p[cond]["lang"] == "numeric")

            layer_result[f"{cond}_summary"] = {
                "pct_zh": n_zh / N_PROBLEMS,
                "pct_en": n_en / N_PROBLEMS,
                "pct_mixed": n_mixed / N_PROBLEMS,
                "pct_numeric": n_num / N_PROBLEMS,
                "n_zh": n_zh,
                "n_en": n_en,
            }

            s = layer_result[f"{cond}_summary"]
            print(f"  {cond}: {s['pct_zh']:.0%} zh, {s['pct_en']:.0%} en, "
                  f"{s['pct_mixed']:.0%} mixed, {s['pct_numeric']:.0%} numeric")

        results["layer_results"][str(sweep_layer)] = layer_result

        gc.collect()
        torch.cuda.empty_cache()

    # Save
    outpath = OUTPUT_DIR / "expG2b_alltoken_reverse_steer.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Summary table
    print(f"\n{'='*80}")
    print("EXPERIMENT G2b: ALL-TOKEN vs LAST-TOKEN REVERSE STEER (en → zh)")
    print(f"{'='*80}")
    print(f"Hypothesis: if all-token >> last-token, KV contamination suppressed steering.")
    print(f"If all-token ≈ last-token, basin asymmetry is the real explanation.\n")
    print(f"{'Layer':>6s}  {'Condition':>20s}  {'%zh':>5s}  {'%en':>5s}  {'%mix':>5s}")
    print("-" * 55)
    for layer in SWEEP_LAYERS:
        r = results["layer_results"][str(layer)]
        for cond in ["alltoken_reverse", "lasttoken_reverse"]:
            s = r[f"{cond}_summary"]
            print(f"  L{layer:<4d}  {cond:>20s}  {s['pct_zh']:>4.0%}  {s['pct_en']:>4.0%}  "
                  f"{s['pct_mixed']:>4.0%}")
        print()

    # Verdict
    print("VERDICT:")
    for layer in SWEEP_LAYERS:
        r = results["layer_results"][str(layer)]
        all_zh = r["alltoken_reverse_summary"]["pct_zh"]
        last_zh = r["lasttoken_reverse_summary"]["pct_zh"]
        delta = all_zh - last_zh
        if delta > 0.2:
            print(f"  L{layer}: all-token={all_zh:.0%} vs last-token={last_zh:.0%} → "
                  f"KV CONTAMINATION CONFIRMED (Δ={delta:+.0%})")
        elif delta > 0.05:
            print(f"  L{layer}: all-token={all_zh:.0%} vs last-token={last_zh:.0%} → "
                  f"PARTIAL KV EFFECT (Δ={delta:+.0%})")
        else:
            print(f"  L{layer}: all-token={all_zh:.0%} vs last-token={last_zh:.0%} → "
                  f"BASIN ASYMMETRY CONFIRMED (Δ={delta:+.0%})")


if __name__ == "__main__":
    main()
