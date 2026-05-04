#!/usr/bin/env python3
"""
Experiment F: Attention Weight Cross-Lingual Match
===================================================
THE PUMP TEST. If the attention-MLP pump story is correct:
- Attention routes identically for zh and en versions of the same problem
- Same positions attend to same positions (weight matrices match)
- Even though the VALUES (content) are language-specific

Extract attention weight matrices (softmax outputs) for N problems,
both zh and en, at key layers. Measure cross-lingual correlation of
attention patterns for paired problems.

Prediction: if attention is the language-agnostic pump, weight correlation
should be HIGH (>0.9) for same-problem zh/en pairs and LOW for different problems.
"""

import numpy as np
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
N_PROBLEMS = 20  # Keep small — attention matrices are big
LAYERS_TO_EXTRACT = [4, 8, 12, 16, 20, 24, 26, 28, 30, 34]


def generate_problems(n=200, seed=42):
    """Same problem generator as all other experiments."""
    pyrandom.seed(seed)
    np.random.seed(seed)
    problems = []
    categories = []
    cat_names = ['arithmetic', 'combinatorics', 'modular', 'geometry', 'sequences']

    for cat_idx, cat in enumerate(cat_names):
        for _ in range(n // 5):
            if cat == 'arithmetic':
                a, b = pyrandom.randint(100, 9999), pyrandom.randint(100, 9999)
                op = pyrandom.choice(['+', '-', '*'])
                expr = f"{a} {op} {b}"
                ans = eval(expr)
                problems.append({'category': cat, 'expr': expr, 'answer': ans,
                                 'en': f"Calculate: {expr} = ?",
                                 'zh': f"计算：{expr} = ？"})
            elif cat == 'combinatorics':
                n_val = pyrandom.randint(5, 15)
                r_val = pyrandom.randint(2, min(5, n_val))
                from math import comb
                ans = comb(n_val, r_val)
                problems.append({'category': cat, 'expr': f"C({n_val},{r_val})", 'answer': ans,
                                 'en': f"How many ways to choose {r_val} items from {n_val}? Answer with a number.",
                                 'zh': f"从{n_val}个物品中选{r_val}个，有多少种方法？用数字回答。"})
            elif cat == 'modular':
                a = pyrandom.randint(100, 9999)
                m = pyrandom.randint(3, 20)
                ans = a % m
                problems.append({'category': cat, 'expr': f"{a} mod {m}", 'answer': ans,
                                 'en': f"What is {a} mod {m}?",
                                 'zh': f"{a} 除以 {m} 的余数是多少？"})
            elif cat == 'geometry':
                a = pyrandom.randint(3, 50)
                b = pyrandom.randint(3, 50)
                ans = a * b
                problems.append({'category': cat, 'expr': f"{a}x{b}", 'answer': ans,
                                 'en': f"What is the area of a rectangle with sides {a} and {b}?",
                                 'zh': f"边长为{a}和{b}的矩形面积是多少？"})
            elif cat == 'sequences':
                start = pyrandom.randint(1, 20)
                step = pyrandom.randint(2, 10)
                n_terms = pyrandom.randint(3, 8)
                seq = [start + i * step for i in range(n_terms)]
                ans = start + n_terms * step
                seq_str = ", ".join(map(str, seq))
                problems.append({'category': cat, 'expr': seq_str, 'answer': ans,
                                 'en': f"What comes next in the sequence: {seq_str}, ?",
                                 'zh': f"序列 {seq_str} 的下一个数是什么？"})
            categories.append(cat_idx)

    return problems


def extract_attention_weights(model, tokenizer, prompt, layers):
    """Run prompt through model, extract attention weights at specified layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is tuple of (batch, n_heads, seq_len, seq_len) per layer
    result = {}
    for layer in layers:
        attn = outputs.attentions[layer][0]  # (n_heads, seq_len, seq_len)
        # Take last token's attention pattern (what the final position attends to)
        last_tok_attn = attn[:, -1, :]  # (n_heads, seq_len)
        result[layer] = last_tok_attn.cpu().float().numpy()

    return result, inputs['input_ids'].shape[1]


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()

    problems = generate_problems(200)
    # Use first 4 per category = 20 problems (same test set as other experiments)
    selected = []
    for cat_start in range(0, 200, 40):
        selected.extend(range(cat_start, cat_start + 4))
    test_problems = [problems[i] for i in selected]

    print(f"Extracting attention weights for {len(test_problems)} problems at {len(LAYERS_TO_EXTRACT)} layers...")

    results = {
        'per_layer': {str(l): {
            'paired_corr': [],      # correlation between zh[i] and en[i] attention patterns
            'scrambled_corr': [],   # correlation between zh[i] and en[j!=i]
            'paired_cos': [],       # cosine between zh[i] and en[i] attention (flattened across heads)
        } for l in LAYERS_TO_EXTRACT},
        'per_problem': [],
        'seq_lengths': {'zh': [], 'en': []},
    }

    zh_attns = {l: [] for l in LAYERS_TO_EXTRACT}
    en_attns = {l: [] for l in LAYERS_TO_EXTRACT}

    for i, prob in enumerate(test_problems):
        print(f"  Problem {i+1}/{len(test_problems)}: {prob['en'][:50]}...")

        zh_attn, zh_len = extract_attention_weights(model, tokenizer, prob['zh'], LAYERS_TO_EXTRACT)
        en_attn, en_len = extract_attention_weights(model, tokenizer, prob['en'], LAYERS_TO_EXTRACT)

        results['seq_lengths']['zh'].append(zh_len)
        results['seq_lengths']['en'].append(en_len)

        prob_result = {'problem': i, 'zh_len': zh_len, 'en_len': en_len, 'per_layer': {}}

        for layer in LAYERS_TO_EXTRACT:
            zh_a = zh_attn[layer]  # (n_heads, zh_len)
            en_a = en_attn[layer]  # (n_heads, en_len)

            # Since sequences have different lengths, compare the DISTRIBUTION shape
            # Method 1: Truncate to min length (compare attention to first K positions)
            min_len = min(zh_a.shape[1], en_a.shape[1])
            zh_trunc = zh_a[:, :min_len]  # (n_heads, min_len)
            en_trunc = en_a[:, :min_len]

            # Renormalize after truncation
            zh_trunc = zh_trunc / (zh_trunc.sum(axis=1, keepdims=True) + 1e-10)
            en_trunc = en_trunc / (en_trunc.sum(axis=1, keepdims=True) + 1e-10)

            # Per-head correlation
            head_corrs = []
            for h in range(zh_trunc.shape[0]):
                if zh_trunc[h].std() > 1e-10 and en_trunc[h].std() > 1e-10:
                    corr = np.corrcoef(zh_trunc[h], en_trunc[h])[0, 1]
                    head_corrs.append(corr)
                else:
                    head_corrs.append(0.0)

            # Flatten across heads for cosine
            zh_flat = zh_trunc.flatten()
            en_flat = en_trunc.flatten()
            cos = np.dot(zh_flat, en_flat) / (np.linalg.norm(zh_flat) * np.linalg.norm(en_flat) + 1e-10)

            results['per_layer'][str(layer)]['paired_corr'].append(float(np.mean(head_corrs)))
            results['per_layer'][str(layer)]['paired_cos'].append(float(cos))

            prob_result['per_layer'][str(layer)] = {
                'mean_head_corr': float(np.mean(head_corrs)),
                'median_head_corr': float(np.median(head_corrs)),
                'min_head_corr': float(np.min(head_corrs)),
                'max_head_corr': float(np.max(head_corrs)),
                'cosine': float(cos),
                'n_heads': zh_trunc.shape[0],
                'min_len': min_len,
            }

            zh_attns[layer].append(zh_trunc)
            en_attns[layer].append(en_trunc)

        results['per_problem'].append(prob_result)

    # Compute scrambled correlations (zh[i] vs en[j!=i])
    print("\nComputing scrambled baselines...")
    for layer in LAYERS_TO_EXTRACT:
        scrambled = []
        for i in range(len(test_problems)):
            for j in range(len(test_problems)):
                if i == j:
                    continue
                zh_a = zh_attns[layer][i]
                en_a = en_attns[layer][j]
                min_len = min(zh_a.shape[1], en_a.shape[1])
                zh_t = zh_a[:, :min_len]
                en_t = en_a[:, :min_len]
                zh_t = zh_t / (zh_t.sum(axis=1, keepdims=True) + 1e-10)
                en_t = en_t / (en_t.sum(axis=1, keepdims=True) + 1e-10)
                zh_f = zh_t.flatten()
                en_f = en_t.flatten()
                cos = np.dot(zh_f, en_f) / (np.linalg.norm(zh_f) * np.linalg.norm(en_f) + 1e-10)
                scrambled.append(cos)
        results['per_layer'][str(layer)]['scrambled_cos'] = scrambled

    # Summary
    print("\n" + "="*80)
    print("ATTENTION WEIGHT CROSS-LINGUAL MATCH")
    print("="*80)
    print(f"{'Layer':>5} | {'PairedCorr':>10} | {'PairedCos':>10} | {'ScramCos':>10} | {'Gap':>6} | {'Z':>6}")
    print("-"*65)

    summary = []
    for layer in LAYERS_TO_EXTRACT:
        d = results['per_layer'][str(layer)]
        paired_cos = np.mean(d['paired_cos'])
        scram_cos = np.mean(d['scrambled_cos'])
        scram_std = np.std(d['scrambled_cos'])
        z = (paired_cos - scram_cos) / (scram_std + 1e-10)
        paired_corr = np.mean(d['paired_corr'])
        gap = paired_cos - scram_cos

        print(f"  L{layer:>2}  | {paired_corr:>10.4f} | {paired_cos:>10.4f} | {scram_cos:>10.4f} | {gap:>6.4f} | {z:>6.1f}")

        summary.append({
            'layer': layer,
            'paired_corr': float(paired_corr),
            'paired_cos': float(paired_cos),
            'scrambled_cos_mean': float(scram_cos),
            'scrambled_cos_std': float(scram_std),
            'gap': float(gap),
            'z': float(z),
        })

    results['summary'] = summary

    # Verdict
    peak_corr = max(s['paired_corr'] for s in summary)
    peak_layer = max(summary, key=lambda s: s['paired_corr'])['layer']
    mean_gap = np.mean([s['gap'] for s in summary])

    if peak_corr > 0.9:
        print(f"\n>>> ATTENTION ROUTING IS LANGUAGE-AGNOSTIC (peak corr={peak_corr:.3f} at L{peak_layer})")
        print(">>> The pump story is CONFIRMED. Same routing, different content.")
    elif peak_corr > 0.5:
        print(f"\n>>> ATTENTION ROUTING IS PARTIALLY SHARED (peak corr={peak_corr:.3f} at L{peak_layer})")
    else:
        print(f"\n>>> ATTENTION ROUTING IS LANGUAGE-DEPENDENT (peak corr={peak_corr:.3f} at L{peak_layer})")

    with open('output/expF_attn_weight_crosslingual.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nSaved to output/expF_attn_weight_crosslingual.json")


if __name__ == '__main__':
    main()
