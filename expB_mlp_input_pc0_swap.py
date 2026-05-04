#!/usr/bin/env python3
"""
Experiment B: MLP Input PC0 Swap
=================================
Between attention output and MLP input, swap PC0.

At layer L:
1. Run Chinese prompt normally through attention at L
2. BEFORE MLP fires, intercept the pre-MLP vector (post-attention residual)
3. Swap PC0 projection from Chinese mean to English mean
4. Let MLP fire with the modified input
5. Don't touch MLP output — let it pass to residual stream normally
6. Continue through remaining layers, generate tokens

This tests: does MLP read PC0 to decide its computation mode?

Qwen2.5 layer structure:
  h_in → input_layernorm → self_attn → h_mid = h_in + attn_out
  h_mid → post_attention_layernorm → mlp → h_out = h_mid + mlp_out

We hook the MLP module's forward to modify its INPUT (which is post_attention_layernorm(h_mid)).
But we need to modify h_mid BEFORE layernorm, then let layernorm process it.
Actually, simpler: hook the whole layer's forward and modify h_mid between attn and MLP.

Even simpler: register a forward_pre_hook on self.mlp that modifies the input tensor.
But the mlp receives the layernorm'd tensor, not h_mid directly.

CLEANEST APPROACH: Hook the DecoderLayer itself with a custom forward that:
1. Runs attention normally
2. Modifies the residual stream (h_mid) by swapping PC0
3. Runs MLP on the modified residual
"""

import numpy as np
import torch
import json
import random as pyrandom
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import re

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
N_PROBLEMS = 20
N_PCA = 200
MAX_TOKENS = 64
TEST_LAYERS = [12, 20, 26, 28]


def load_cached_pc0(layer):
    """Load PC0 vector and mean projections from cached data."""
    pc0_data = np.load('output/pc0_vectors_per_layer.npz')
    hs_data = np.load('output/all_layers_lasttok.npz')

    pc0 = pc0_data[str(layer)].astype(np.float64)  # (2048,)
    zh_hs = hs_data[f'zh_L{layer}'].astype(np.float64)  # (200, 2048)
    en_hs = hs_data[f'en_L{layer}'].astype(np.float64)

    zh_mean_proj = float((zh_hs @ pc0).mean())
    en_mean_proj = float((en_hs @ pc0).mean())

    # Variance explained (approximate from cached data)
    combined = np.vstack([zh_hs, en_hs])
    var_total = np.var(combined, axis=0).sum()
    var_pc0 = np.var(combined @ pc0)
    var_exp = var_pc0 / var_total

    return pc0, zh_mean_proj, en_mean_proj, var_exp


def generate_problems(n=200, seed=42):
    pyrandom.seed(seed)
    np.random.seed(seed)
    problems = []
    cat_names = ['arithmetic', 'combinatorics', 'modular', 'geometry', 'sequences']
    for cat in cat_names:
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
    return problems


def detect_language(text):
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    total = zh_chars + en_chars
    if total == 0:
        return "numeric"
    zh_frac = zh_chars / total
    if zh_frac > 0.5:
        return "zh"
    elif zh_frac < 0.2:
        return "en"
    return "mixed"


def fit_pc0_at_layer(model, tokenizer, problems, layer):
    """Extract last-token hidden states at a layer for all problems, fit PCA."""
    zh_states = []
    en_states = []

    captured = {}

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured['h'] = h[:, -1:, :].detach().clone()

    handle = model.model.layers[layer].register_forward_hook(capture_hook)

    for prob in problems:
        for lang_key in ['zh', 'en']:
            inputs = tokenizer(prob[lang_key], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            state = captured['h'].squeeze().cpu().float().numpy()
            if lang_key == 'zh':
                zh_states.append(state)
            else:
                en_states.append(state)

    handle.remove()

    zh_arr = np.array(zh_states)  # (N, 2048)
    en_arr = np.array(en_states)

    # Fit PCA on combined
    combined = np.vstack([zh_arr, en_arr])
    pca = PCA(n_components=1).fit(combined)
    pc0 = pca.components_[0]  # (2048,)

    zh_mean_proj = float(zh_arr @ pc0).mean() if len(zh_arr.shape) == 1 else float((zh_arr @ pc0).mean())
    en_mean_proj = float(en_arr @ pc0).mean() if len(en_arr.shape) == 1 else float((en_arr @ pc0).mean())

    return pc0, zh_mean_proj, en_mean_proj, pca.explained_variance_ratio_[0]


def run_with_mlp_pc0_swap(model, tokenizer, prompt, layer, pc0_vec, zh_mean, en_mean,
                           use_en_kv_prompt=None, max_tokens=64):
    """Run prompt, but at layer L, swap PC0 on the pre-MLP hidden state.

    Hook strategy: register a forward hook on model.model.layers[layer] that
    replaces the ENTIRE layer's computation with a modified version:
    1. Run attention normally -> get h_mid
    2. Modify h_mid by swapping PC0
    3. Run MLP on modified h_mid -> get h_out
    """
    # We need to intercept WITHIN the layer. Use a forward_pre_hook on the mlp
    # submodule. But the mlp receives post_attention_layernorm(h_mid), not h_mid.
    # So we hook the mlp and modify its input.
    #
    # Actually, the cleanest way: hook the LAYER's forward and reimplement it.
    # But that's fragile. Instead:
    #
    # Strategy: hook post_attention_layernorm to modify its OUTPUT before MLP sees it.
    # The post_attention_layernorm output IS the mlp input.

    pc0_tensor = torch.tensor(pc0_vec, dtype=torch.float16, device=model.device)
    did_swap = {}

    def layernorm_output_hook(module, input, output):
        """Modify the output of post_attention_layernorm (= MLP input)."""
        if did_swap.get('done'):
            return output
        # output shape: (batch, seq, hidden)
        h = output.clone()
        last = h[:, -1:, :]  # (1, 1, 2048)

        # Project onto PC0
        proj = (last.squeeze() @ pc0_tensor).item()

        # Swap: remove current projection, add english mean
        last_modified = last - proj * pc0_tensor.unsqueeze(0).unsqueeze(0) + en_mean * pc0_tensor.unsqueeze(0).unsqueeze(0)

        h[:, -1:, :] = last_modified
        did_swap['done'] = True
        return h

    # Determine which prompt to use for KV cache
    kv_prompt = use_en_kv_prompt if use_en_kv_prompt else prompt

    inputs = tokenizer(kv_prompt if use_en_kv_prompt else prompt, return_tensors="pt").to(model.device)

    handle = model.model.layers[layer].post_attention_layernorm.register_forward_hook(layernorm_output_hook)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

    handle.remove()

    # Generate tokens
    first_token_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[first_token_id]], device=model.device)
    tokens = [first_token_id]
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_tokens - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            if next_token.item() == eos_id:
                break

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text, detect_language(text)


def run_baseline(model, tokenizer, prompt, max_tokens=64):
    """Normal generation baseline."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

    first_token_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[first_token_id]], device=model.device)
    tokens = [first_token_id]
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_tokens - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            if next_token.item() == eos_id:
                break

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    problems = generate_problems(200)
    # Test set: first 4 per category
    selected_idx = []
    for cat_start in range(0, 200, 40):
        selected_idx.extend(range(cat_start, cat_start + 4))
    test_problems = [problems[i] for i in selected_idx]

    all_results = {}

    for layer in TEST_LAYERS:
        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")

        # Load cached PC0 vector and mean projections (no GPU needed)
        print(f"  Loading cached PC0 at L{layer}...", flush=True)
        pc0, zh_mean, en_mean, var_exp = load_cached_pc0(layer)
        print(f"  PC0 var={var_exp:.3f}, zh_mean={zh_mean:.3f}, en_mean={en_mean:.3f}, gap={zh_mean-en_mean:.3f}")

        layer_results = {
            'pc0_var': float(var_exp),
            'zh_mean_proj': float(zh_mean),
            'en_mean_proj': float(en_mean),
            'per_problem': []
        }

        n_en_baseline = 0
        n_zh_baseline = 0
        n_en_swap = 0
        n_text_match = 0

        for i, prob in enumerate(test_problems):
            print(f"  Problem {i+1}/20: {prob['en'][:50]}...", end=' ', flush=True)

            # Baseline zh
            zh_text = run_baseline(model, tokenizer, prob['zh'])
            zh_lang = detect_language(zh_text)

            # Baseline en
            en_text = run_baseline(model, tokenizer, prob['en'])
            en_lang = detect_language(en_text)

            # MLP input PC0 swap (zh prompt, swap PC0 to en mean before MLP at layer L)
            swap_text, swap_lang = run_with_mlp_pc0_swap(
                model, tokenizer, prob['zh'], layer, pc0, zh_mean, en_mean
            )

            # Compare
            text_match = swap_text.strip() == en_text.strip()
            char_match = sum(a == b for a, b in zip(swap_text, en_text)) / max(len(en_text), 1)

            if zh_lang == 'zh': n_zh_baseline += 1
            if en_lang == 'en': n_en_baseline += 1
            if swap_lang == 'en': n_en_swap += 1
            if text_match: n_text_match += 1

            print(f"zh={zh_lang} en={en_lang} swap={swap_lang} match={text_match}")

            layer_results['per_problem'].append({
                'problem': i,
                'zh_baseline': zh_text[:200],
                'en_baseline': en_text[:200],
                'swap_text': swap_text[:200],
                'zh_lang': zh_lang,
                'en_lang': en_lang,
                'swap_lang': swap_lang,
                'text_match': text_match,
                'char_match': float(char_match),
            })

        layer_results['pct_en_swap'] = n_en_swap / 20
        layer_results['pct_text_match'] = n_text_match / 20
        layer_results['pct_zh_baseline'] = n_zh_baseline / 20
        layer_results['pct_en_baseline'] = n_en_baseline / 20

        print(f"\n  L{layer} SUMMARY: swap→en={n_en_swap}/20 ({n_en_swap/20:.0%}), "
              f"text_match={n_text_match}/20, "
              f"zh_base={n_zh_baseline}/20, en_base={n_en_baseline}/20")

        all_results[str(layer)] = layer_results

    # Save
    output = {
        'experiment': 'B',
        'description': 'MLP Input PC0 Swap — does MLP read PC0 to decide language?',
        'layers_tested': TEST_LAYERS,
        'n_problems': 20,
        'max_tokens': MAX_TOKENS,
        'per_layer': all_results,
    }

    with open('output/expB_mlp_input_pc0_swap.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to output/expB_mlp_input_pc0_swap.json")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for layer in TEST_LAYERS:
        r = all_results[str(layer)]
        print(f"  L{layer}: swap→en={r['pct_en_swap']:.0%}, "
              f"text_match={r['pct_text_match']:.0%}, "
              f"pc0_var={r['pc0_var']:.3f}")


if __name__ == '__main__':
    main()
