"""Experiment 7: Multi-Language Steering.

Using 7-language activation data at L32:
1. Fit PCA on full 7-language set
2. Extract per-language mean projections onto PC0
3. Run pairwise steering interventions
4. Focus: zh→en, zh→sw, en→ja, sw→zh, en→ar

Tests whether PC0 is a universal language axis or just a zh/en binary switch.
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
STEER_LAYER = 32  # Where PC0 variance is highest
N_PROBLEMS = 20
MAX_TOKENS = 64

# Languages and their detection patterns
LANG_PATTERNS = {
    "zh": lambda t: sum(1 for c in t if '\u4e00' <= c <= '\u9fff'),
    "ja": lambda t: sum(1 for c in t if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff'),
    "ko": lambda t: sum(1 for c in t if '\uac00' <= c <= '\ud7af'),
    "ar": lambda t: sum(1 for c in t if '\u0600' <= c <= '\u06ff'),
}


def detect_language_multi(text):
    """Detect language from generated text."""
    counts = {lang: func(text) for lang, func in LANG_PATTERNS.items()}
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "numeric"

    # Check each non-Latin script
    for lang in ["zh", "ja", "ko", "ar"]:
        if counts[lang] / max(total_alpha, 1) > 0.2:
            return lang

    # Check for Spanish markers
    spanish_markers = sum(1 for w in text.lower().split()
                         if w in ["el", "la", "los", "las", "de", "del", "en", "es",
                                  "que", "por", "para", "con", "una", "uno", "como",
                                  "más", "pero", "este", "esta", "ser", "tiene"])
    if spanish_markers > 3:
        return "es"

    # Check for Swahili markers
    sw_markers = sum(1 for w in text.lower().split()
                    if w in ["ni", "ya", "wa", "na", "kwa", "katika", "za",
                             "au", "hii", "ile", "yake", "wake", "mtu"])
    if sw_markers > 3:
        return "sw"

    return "en"  # default


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


def main():
    # === Step 1: Fit PC0 on 7-language data at L32 ===
    print("Loading 7-language activations at L32...")
    multi_data = np.load("output/multilingual_activations.npz")
    langs = sorted(multi_data.keys())
    print(f"Languages: {langs}")

    all_acts = []
    lang_labels = []
    for lang in langs:
        acts = multi_data[lang].astype(np.float64)  # (200, 2048)
        norms = np.linalg.norm(acts, axis=1, keepdims=True)
        acts_unit = acts / norms
        all_acts.append(acts_unit)
        lang_labels.extend([lang] * acts.shape[0])

    combined = np.vstack(all_acts)  # (1400, 2048)
    print(f"Combined shape: {combined.shape}")

    pca = PCA(n_components=20)
    pca.fit(combined)
    pc0 = pca.components_[0]

    print(f"PC0 variance explained: {pca.explained_variance_ratio_[0]:.1%}")

    # Per-language mean projections onto PC0
    lang_means = {}
    print("\nPer-language PC0 projections:")
    for lang in langs:
        acts = multi_data[lang].astype(np.float64)
        norms = np.linalg.norm(acts, axis=1, keepdims=True)
        acts_unit = acts / norms
        proj = acts_unit @ pc0
        lang_means[lang] = float(proj.mean())
        print(f"  {lang}: mean={proj.mean():+.4f}  std={proj.std():.4f}")

    # === Step 2: Pairwise Cohen's d ===
    print("\nPairwise Cohen's d on PC0:")
    pairwise_d = {}
    for i, l1 in enumerate(langs):
        for l2 in langs[i+1:]:
            a1 = multi_data[l1].astype(np.float64)
            a2 = multi_data[l2].astype(np.float64)
            a1_unit = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
            a2_unit = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
            p1 = a1_unit @ pc0
            p2 = a2_unit @ pc0
            d = (p1.mean() - p2.mean()) / np.sqrt((p1.std()**2 + p2.std()**2) / 2)
            pairwise_d[f"{l1}-{l2}"] = float(d)
            if abs(d) > 3:
                print(f"  {l1} vs {l2}: d={d:+.1f}")

    # === Step 3: GPU interventions ===
    print(f"\nLoading model for interventions...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    problems = generate_problems(200, seed=42)
    selected = select_problems(problems, N_PROBLEMS)

    # Define steering pairs: (source_lang, source_prompt_key, target_lang)
    steer_pairs = [
        ("zh", "zh", "en"),   # zh→en (our standard)
        ("zh", "zh", "es"),   # zh→es
        ("zh", "zh", "ja"),   # zh→ja
        ("zh", "zh", "ar"),   # zh→ar
        ("zh", "zh", "sw"),   # zh→sw (dramatic: CJK → Swahili)
        ("en", "en", "zh"),   # en→zh (reverse)
        ("en", "en", "ja"),   # en→ja
        ("en", "en", "sw"),   # en→sw
    ]

    # Need to extract hidden states at L32 for PCA (reuse cached data for zh/en)
    # For the intervention, we use zh or en prompts and steer PC0 to target lang mean
    layer_output = {}
    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        layer_output['h'] = h.detach()[:, -1, :]

    results = {
        "model": MODEL_NAME,
        "steer_layer": STEER_LAYER,
        "pc0_var_explained": float(pca.explained_variance_ratio_[0]),
        "lang_means": lang_means,
        "pairwise_d": pairwise_d,
        "steer_results": {}
    }

    for src_lang, prompt_key, tgt_lang in steer_pairs:
        pair_name = f"{src_lang}_to_{tgt_lang}"
        print(f"\n--- {pair_name} ---")

        src_mean = lang_means[src_lang]
        tgt_mean = lang_means[tgt_lang]

        pair_results = []

        for idx in tqdm(selected, desc=pair_name, leave=False):
            prob = problems[idx]
            prompt = prob[prompt_key]

            # Get hidden state at L32
            handle = model.model.layers[STEER_LAYER].register_forward_hook(capture_hook)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            h = layer_output['h'].cpu().float().numpy().squeeze()
            layer_output.clear()
            handle.remove()

            # PC0 swap: src_mean → tgt_mean
            h_norm = np.linalg.norm(h)
            h_unit = h / h_norm
            h_pc0 = float(h_unit @ pc0)
            h_swapped_unit = h_unit - h_pc0 * pc0 + tgt_mean * pc0
            h_swapped = h_swapped_unit * h_norm
            h_swapped_t = torch.tensor(h_swapped, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to(model.device)

            # Run with splice
            def splice_hook(module, input, output):
                h_out = output[0] if isinstance(output, tuple) else output
                new_h = h_out.clone()
                new_h[:, -1:, :] = h_swapped_t
                if isinstance(output, tuple):
                    return (new_h,) + output[1:]
                return new_h

            handle = model.model.layers[STEER_LAYER].register_forward_hook(splice_hook)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)
                pkv = outputs.past_key_values
            handle.remove()

            first_tok = int(outputs.logits[0, -1].argmax())
            next_tok = torch.tensor([[first_tok]], device=model.device)
            tokens = [first_tok]

            with torch.no_grad():
                for _ in range(MAX_TOKENS - 1):
                    outputs = model(next_tok, past_key_values=pkv, use_cache=True)
                    pkv = outputs.past_key_values
                    next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    tokens.append(next_tok.item())
                    if next_tok.item() == tokenizer.eos_token_id:
                        break

            text = tokenizer.decode(tokens, skip_special_tokens=True)
            lang = detect_language_multi(text)

            pair_results.append({
                "prob_idx": idx,
                "detected_lang": lang,
                "text_snippet": text[:150],
            })

        # Aggregate
        lang_counts = {}
        for p in pair_results:
            l = p["detected_lang"]
            lang_counts[l] = lang_counts.get(l, 0) + 1

        results["steer_results"][pair_name] = {
            "lang_distribution": lang_counts,
            "n_target_lang": lang_counts.get(tgt_lang, 0),
            "pct_target": lang_counts.get(tgt_lang, 0) / N_PROBLEMS,
            "per_problem": pair_results,
        }

        print(f"  Target: {tgt_lang}  |  Distribution: {lang_counts}  |  "
              f"Hit rate: {lang_counts.get(tgt_lang, 0)}/{N_PROBLEMS}")

    # Save
    outpath = OUTPUT_DIR / "exp7_multilang_steer.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT 7 SUMMARY: MULTI-LANGUAGE STEERING")
    print(f"{'='*60}")
    print(f"{'Steer':>12s}  {'Target%':>8s}  {'Distribution':>30s}")
    print("-" * 55)
    for pair_name, r in results["steer_results"].items():
        print(f"  {pair_name:>10s}  {r['pct_target']:>7.0%}  {str(r['lang_distribution']):>30s}")


if __name__ == "__main__":
    main()
