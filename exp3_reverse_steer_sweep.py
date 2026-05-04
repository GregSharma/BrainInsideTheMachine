"""Experiment 3: Reverse Steer Layer Sweep (English → Chinese).

Tests path-dependent planning hypothesis.
At early layers, steering en→zh should produce NATIVE Chinese (terse, direct).
At late layers, steering en→zh should produce "English in Chinese clothes" (verbose).

The non-PC0 dimensions carry planning/style that accumulates across layers.
Early steering lets the model develop a Chinese plan.
Late steering inherits the English plan.

Run at: L4, L8, L12, L16, L20, L24, L26, L30, L34
Compare: % Chinese, style analysis, token efficiency.
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

SWEEP_LAYERS = [4, 8, 12, 16, 20, 24, 26, 30, 34]


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


def measure_style(text):
    """Measure style characteristics to detect 'English in Chinese clothes'."""
    # Token efficiency: chars per "semantic unit" (rough proxy: sentence length)
    sentences = [s.strip() for s in text.replace('。', '.').replace('，', ',').split('.') if s.strip()]
    mean_sentence_len = np.mean([len(s) for s in sentences]) if sentences else 0

    # Chinese character density (higher = more native Chinese)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text)
    zh_density = chinese_chars / max(total_chars, 1)

    # Step-by-step markers (English-style verbose reasoning)
    step_markers = sum(1 for marker in ['step', 'Step', 'first', 'First', 'then', 'Then',
                                         '首先', '然后', '接下来', '第一', '第二',
                                         'next', 'Next', 'finally', 'Finally']
                      if marker in text)

    # Numeric density (math content, should be similar regardless of style)
    digits = sum(1 for c in text if c.isdigit())
    digit_density = digits / max(total_chars, 1)

    return {
        "total_chars": total_chars,
        "chinese_char_count": chinese_chars,
        "zh_density": zh_density,
        "mean_sentence_len": mean_sentence_len,
        "n_sentences": len(sentences),
        "step_markers": step_markers,
        "digit_density": digit_density,
    }


def run_reverse_steer(model, tokenizer, prompt_en, splice_layer, pc0, en_mean_proj, zh_mean_proj, max_tokens=64):
    """Run English prompt, swap PC0 from English to Chinese direction at last token."""
    inputs = tokenizer(prompt_en, return_tensors="pt").to(model.device)

    def reverse_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        new_h = h.clone()
        h_last = new_h[0, -1, :].float().cpu().numpy()
        h_norm = np.linalg.norm(h_last)
        h_unit = h_last / h_norm
        h_pc0 = float(h_unit @ pc0)
        # Swap: remove English PC0, add Chinese mean
        h_swapped_unit = h_unit - h_pc0 * pc0 + zh_mean_proj * pc0
        h_swapped = h_swapped_unit * h_norm
        new_h[0, -1, :] = torch.tensor(h_swapped, dtype=h.dtype, device=h.device)

        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    handle = model.model.layers[splice_layer].register_forward_hook(reverse_hook)

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

    # Generate Chinese baselines
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
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        zh_baselines[idx] = {
            "tokens": gen_tokens,
            "text": text,
            "style": measure_style(text),
        }

    # Generate English baselines
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
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        en_baselines[idx] = {
            "tokens": gen_tokens,
            "text": text,
            "style": measure_style(text),
        }

    # Baseline style summary
    zh_styles = [zh_baselines[idx]["style"] for idx in selected]
    en_styles = [en_baselines[idx]["style"] for idx in selected]
    print(f"\nBaseline styles:")
    print(f"  ZH: mean chars={np.mean([s['total_chars'] for s in zh_styles]):.0f}, "
          f"zh_density={np.mean([s['zh_density'] for s in zh_styles]):.3f}, "
          f"step_markers={np.mean([s['step_markers'] for s in zh_styles]):.1f}")
    print(f"  EN: mean chars={np.mean([s['total_chars'] for s in en_styles]):.0f}, "
          f"zh_density={np.mean([s['zh_density'] for s in en_styles]):.3f}, "
          f"step_markers={np.mean([s['step_markers'] for s in en_styles]):.1f}")

    results = {
        "model": MODEL_NAME,
        "experiment": "reverse_steer_layer_sweep",
        "sweep_layers": SWEEP_LAYERS,
        "n_problems": N_PROBLEMS,
        "n_pca": N_PCA,
        "max_tokens": MAX_TOKENS,
        "zh_baseline_style": {
            "mean_chars": float(np.mean([s['total_chars'] for s in zh_styles])),
            "mean_zh_density": float(np.mean([s['zh_density'] for s in zh_styles])),
            "mean_step_markers": float(np.mean([s['step_markers'] for s in zh_styles])),
        },
        "en_baseline_style": {
            "mean_chars": float(np.mean([s['total_chars'] for s in en_styles])),
            "mean_zh_density": float(np.mean([s['zh_density'] for s in en_styles])),
            "mean_step_markers": float(np.mean([s['step_markers'] for s in en_styles])),
        },
        "layer_results": {}
    }

    for sweep_layer in SWEEP_LAYERS:
        print(f"\n{'='*70}")
        print(f"LAYER {sweep_layer}: Reverse steer (en → zh)")
        print(f"{'='*70}")

        # Extract and PCA
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
        zh_mean_proj = float(zh_proj.mean())
        en_mean_proj = float(en_proj.mean())
        cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt(
            (zh_proj.std()**2 + en_proj.std()**2) / 2
        )

        print(f"  PC0 var: {pca.explained_variance_ratio_[0]:.1%}, d={cohens_d:.1f}")

        layer_result = {
            "pc0_var_explained": float(pca.explained_variance_ratio_[0]),
            "cohens_d": float(cohens_d),
            "per_problem": []
        }

        for idx in tqdm(selected, desc=f"reverse steer L{sweep_layer}", leave=False):
            prob = problems[idx]

            tokens = run_reverse_steer(
                model, tokenizer, prob["en"], sweep_layer,
                pc0, en_mean_proj, zh_mean_proj, MAX_TOKENS
            )
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            lang = detect_language(text)
            style = measure_style(text)

            # Compare to Chinese baseline
            zh_text = zh_baselines[idx]["text"]
            min_len = min(len(text), len(zh_text))
            exact_chars = min_len
            for j in range(min_len):
                if text[j] != zh_text[j]:
                    exact_chars = j
                    break

            layer_result["per_problem"].append({
                "prob_idx": idx,
                "lang": lang,
                "text_snippet": text[:200],
                "style": style,
                "zh_baseline_match_chars": exact_chars,
                "zh_baseline_total_chars": min_len,
            })

        # Aggregate
        n_zh = sum(1 for p in layer_result["per_problem"] if p["lang"] == "zh")
        n_en = sum(1 for p in layer_result["per_problem"] if p["lang"] == "en")
        n_mixed = sum(1 for p in layer_result["per_problem"] if p["lang"] == "mixed")
        n_num = sum(1 for p in layer_result["per_problem"] if p["lang"] == "numeric")

        styles = [p["style"] for p in layer_result["per_problem"]]
        mean_chars = np.mean([s["total_chars"] for s in styles])
        mean_zh_density = np.mean([s["zh_density"] for s in styles])
        mean_step = np.mean([s["step_markers"] for s in styles])
        mean_sentences = np.mean([s["n_sentences"] for s in styles])

        total_ec = sum(p["zh_baseline_match_chars"] for p in layer_result["per_problem"])
        total_tc = sum(p["zh_baseline_total_chars"] for p in layer_result["per_problem"])

        layer_result["summary"] = {
            "pct_zh": n_zh / N_PROBLEMS,
            "pct_en": n_en / N_PROBLEMS,
            "pct_mixed": n_mixed / N_PROBLEMS,
            "pct_numeric": n_num / N_PROBLEMS,
            "mean_chars": float(mean_chars),
            "mean_zh_density": float(mean_zh_density),
            "mean_step_markers": float(mean_step),
            "mean_sentences": float(mean_sentences),
            "zh_baseline_char_match": float(total_ec / max(total_tc, 1)),
        }

        s = layer_result["summary"]
        print(f"  %zh={s['pct_zh']:.0%}  %en={s['pct_en']:.0%}  %mix={s['pct_mixed']:.0%}  "
              f"chars={s['mean_chars']:.0f}  zh_dens={s['mean_zh_density']:.3f}  "
              f"steps={s['mean_step_markers']:.1f}  zh_match={s['zh_baseline_char_match']:.1%}")

        results["layer_results"][str(sweep_layer)] = layer_result

        gc.collect()
        torch.cuda.empty_cache()

    # Save
    outpath = OUTPUT_DIR / "exp3_reverse_steer_sweep.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Summary table
    print(f"\n{'='*90}")
    print("EXPERIMENT 3 SUMMARY: REVERSE STEER (en → zh) LAYER SWEEP")
    print(f"{'='*90}")
    print(f"{'Layer':>6s}  {'%zh':>5s}  {'%en':>5s}  {'%mix':>5s}  {'chars':>6s}  {'zh_dens':>8s}  {'steps':>6s}  {'zh_match':>9s}")
    print("-" * 70)

    # Baselines
    print(f"  ZH_B  {'100%':>5s}  {'0%':>5s}  {'0%':>5s}  "
          f"{results['zh_baseline_style']['mean_chars']:>5.0f}  "
          f"{results['zh_baseline_style']['mean_zh_density']:>7.3f}  "
          f"{results['zh_baseline_style']['mean_step_markers']:>5.1f}  {'100%':>9s}")
    print(f"  EN_B  {'0%':>5s}  {'100%':>5s}  {'0%':>5s}  "
          f"{results['en_baseline_style']['mean_chars']:>5.0f}  "
          f"{results['en_baseline_style']['mean_zh_density']:>7.3f}  "
          f"{results['en_baseline_style']['mean_step_markers']:>5.1f}  {'0%':>9s}")
    print("-" * 70)

    for layer in SWEEP_LAYERS:
        s = results["layer_results"][str(layer)]["summary"]
        print(f"  L{layer:<4d}  {s['pct_zh']:>4.0%}  {s['pct_en']:>4.0%}  {s['pct_mixed']:>4.0%}  "
              f"{s['mean_chars']:>5.0f}  {s['mean_zh_density']:>7.3f}  "
              f"{s['mean_step_markers']:>5.1f}  {s['zh_baseline_char_match']:>8.1%}")


if __name__ == "__main__":
    main()
