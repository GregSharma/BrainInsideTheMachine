"""Gen-time early exit: does L26 language-agnostic peak hold during active reasoning?

The input-pass experiment showed 91% zh-en same-token agreement at L26.
But during generation, the model is ACTUALLY REASONING — producing arithmetic
steps, intermediate tokens, answers. If L26 agreement holds per-token during
generation, it means the model's ACTIVE REASONING is language-blind at 75% depth.

Method:
  - Hook multiple layers (L20, L24, L26, L28, L32) during generation
  - At each generated token, apply final norm + lm_head to each layer's hidden state
  - Compare: (a) Lx-predicted token vs L35-predicted token (actual output)
  - Compare: (b) zh Lx token vs en Lx token for matched problems
  - Run 20 problems × zh + en, up to 256 tokens each
  - Report cross-lingual agreement at each exit layer, per-token and averaged

This is the experiment that turns L26 from "input-pass observation" into
"actionable architectural insight for efficient multilingual inference."
"""

import numpy as np
import torch
import torch.nn.functional as F
import json
import random as pyrandom
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

EXIT_LAYERS = [10, 14, 18, 20, 24, 26, 28, 32]
N_PROBLEMS = 20  # same as gen trajectory scripts
MAX_TOKENS = 256


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
    """Select n problems balanced across categories (4 per cat)."""
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


def generate_with_multi_layer_logit_lens(model, tokenizer, prompt, exit_layers,
                                          final_norm, lm_head, max_new_tokens=256):
    """Generate tokens while capturing logit-lens predictions at multiple exit layers.

    Returns:
        actual_tokens: list of generated token IDs (from full model)
        layer_tokens: dict {layer: list of token IDs predicted by logit lens at that layer}
        n_tokens: number of tokens generated
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eos_id = tokenizer.eos_token_id

    # Hook storage
    hook_data = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hook_data[layer_idx] = h[:, -1:, :].detach()  # (1, 1, d), keep on GPU
        return hook_fn

    handles = []
    for l in exit_layers:
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(h)

    actual_tokens = []
    layer_tokens = {l: [] for l in exit_layers}
    past_key_values = None

    with torch.no_grad():
        # First pass: process entire prompt
        outputs = model(inputs.input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        # Actual next token from full model
        logits_full = outputs.logits[:, -1, :]
        next_token = logits_full.argmax(dim=-1, keepdim=True)
        actual_tokens.append(next_token.item())

        # Logit lens at each exit layer
        for l in exit_layers:
            h = hook_data[l]  # (1, 1, d)
            h_normed = final_norm(h)
            logits_l = lm_head(h_normed)  # (1, 1, vocab)
            layer_tokens[l].append(int(logits_l[0, 0].argmax()))

        if next_token.item() == eos_id:
            for h in handles:
                h.remove()
            return actual_tokens, layer_tokens, 1

        # Autoregressive generation
        for step in range(1, max_new_tokens):
            hook_data.clear()
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values

            logits_full = outputs.logits[:, -1, :]
            next_token = logits_full.argmax(dim=-1, keepdim=True)
            actual_tokens.append(next_token.item())

            for l in exit_layers:
                h = hook_data[l]
                h_normed = final_norm(h)
                logits_l = lm_head(h_normed)
                layer_tokens[l].append(int(logits_l[0, 0].argmax()))

            if next_token.item() == eos_id:
                break

    for h in handles:
        h.remove()

    return actual_tokens, layer_tokens, len(actual_tokens)


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
    final_norm = model.model.norm
    lm_head = model.lm_head
    print(f"Model: {n_layers} layers, d={d}")
    print(f"Exit layers: {EXIT_LAYERS}")

    problems = generate_problems(200, seed=42)
    selected = select_problems(problems, N_PROBLEMS)
    print(f"Selected {len(selected)} problems")

    # Storage
    zh_results = []  # list of {actual_tokens, layer_tokens, n_tokens}
    en_results = []

    print(f"\nGenerating {N_PROBLEMS} Chinese problems...")
    for idx in tqdm(selected, desc="zh gen"):
        prob = problems[idx]
        actual, layer_tok, n_tok = generate_with_multi_layer_logit_lens(
            model, tokenizer, prob["zh"], EXIT_LAYERS, final_norm, lm_head, MAX_TOKENS
        )
        zh_results.append({
            "prob_idx": idx,
            "actual_tokens": actual,
            "layer_tokens": {str(l): lt for l, lt in layer_tok.items()},
            "n_tokens": n_tok,
        })

    print(f"Generating {N_PROBLEMS} English problems...")
    for idx in tqdm(selected, desc="en gen"):
        prob = problems[idx]
        actual, layer_tok, n_tok = generate_with_multi_layer_logit_lens(
            model, tokenizer, prob["en"], EXIT_LAYERS, final_norm, lm_head, MAX_TOKENS
        )
        en_results.append({
            "prob_idx": idx,
            "actual_tokens": actual,
            "layer_tokens": {str(l): lt for l, lt in layer_tok.items()},
            "n_tokens": n_tok,
        })

    # ========== ANALYSIS ==========
    print(f"\n{'='*70}")
    print("GEN-TIME EARLY EXIT ANALYSIS")
    print(f"{'='*70}")

    # 1. Layer vs actual: what % of generation tokens does each exit layer predict correctly?
    print(f"\n1. LOGIT LENS ACCURACY (exit layer token == actual L35 token)")
    print(f"{'Layer':>6} | {'zh accuracy':>12} | {'en accuracy':>12} | {'avg':>8} | {'compute%':>9}")
    print("-" * 60)

    layer_accuracy = {}
    for l in EXIT_LAYERS:
        zh_correct = 0
        zh_total = 0
        en_correct = 0
        en_total = 0

        for r in zh_results:
            lt = r["layer_tokens"][str(l)]
            at = r["actual_tokens"]
            n = min(len(lt), len(at))
            for t in range(n):
                if lt[t] == at[t]:
                    zh_correct += 1
                zh_total += 1

        for r in en_results:
            lt = r["layer_tokens"][str(l)]
            at = r["actual_tokens"]
            n = min(len(lt), len(at))
            for t in range(n):
                if lt[t] == at[t]:
                    en_correct += 1
                en_total += 1

        zh_acc = zh_correct / zh_total if zh_total > 0 else 0
        en_acc = en_correct / en_total if en_total > 0 else 0
        avg_acc = (zh_acc + en_acc) / 2
        compute_pct = (l + 1) / n_layers * 100

        print(f"  L{l:2d}  | {zh_acc:11.1%} | {en_acc:11.1%} | {avg_acc:7.1%} | {compute_pct:7.0f}%")
        layer_accuracy[l] = {"zh": zh_acc, "en": en_acc, "avg": avg_acc}

    # 2. Cross-lingual agreement: at each exit layer, what % of tokens are the SAME
    #    between zh and en (for matched problems at matched generation steps)?
    print(f"\n2. CROSS-LINGUAL TOKEN AGREEMENT AT EACH EXIT LAYER")
    print(f"   (For matched problems: does zh L_x token == en L_x token at each step?)")
    print(f"{'Layer':>6} | {'same token%':>12} | {'both correct%':>14} | {'neither correct%':>16}")
    print("-" * 65)

    cross_lingual = {}
    for l in EXIT_LAYERS:
        same = 0
        both_correct = 0
        neither = 0
        total = 0

        for i in range(N_PROBLEMS):
            zh_lt = zh_results[i]["layer_tokens"][str(l)]
            en_lt = en_results[i]["layer_tokens"][str(l)]
            zh_at = zh_results[i]["actual_tokens"]
            en_at = en_results[i]["actual_tokens"]

            # Compare token by token up to the shorter sequence
            n = min(len(zh_lt), len(en_lt))
            for t in range(n):
                if zh_lt[t] == en_lt[t]:
                    same += 1
                zh_ok = zh_lt[t] == zh_at[t] if t < len(zh_at) else False
                en_ok = en_lt[t] == en_at[t] if t < len(en_at) else False
                if zh_ok and en_ok:
                    both_correct += 1
                if not zh_ok and not en_ok:
                    neither += 1
                total += 1

        same_pct = same / total if total > 0 else 0
        both_pct = both_correct / total if total > 0 else 0
        neither_pct = neither / total if total > 0 else 0

        print(f"  L{l:2d}  | {same_pct:11.1%} | {both_pct:13.1%} | {neither_pct:15.1%}")
        cross_lingual[l] = {"same_token": same_pct, "both_correct": both_pct,
                            "neither_correct": neither_pct, "total_pairs": total}

    # 3. Cross-lingual agreement on ACTUAL tokens (baseline)
    same_actual = 0
    total_actual = 0
    for i in range(N_PROBLEMS):
        zh_at = zh_results[i]["actual_tokens"]
        en_at = en_results[i]["actual_tokens"]
        n = min(len(zh_at), len(en_at))
        for t in range(n):
            if zh_at[t] == en_at[t]:
                same_actual += 1
            total_actual += 1
    actual_agreement = same_actual / total_actual if total_actual > 0 else 0
    print(f"\n  Baseline (actual L35 tokens): zh==en agreement = {actual_agreement:.1%} ({same_actual}/{total_actual})")

    # 4. Per-token position analysis: does agreement change during generation?
    print(f"\n3. AGREEMENT BY GENERATION POSITION (L26 focus)")
    print(f"   (How does L26 cross-lingual agreement change over the generation?)")

    # Bin tokens into phases: prompt response (0-10), early gen (10-30), mid gen (30-100), late gen (100+)
    bins = [(0, 10, "first 10"), (10, 30, "tokens 10-30"), (30, 100, "tokens 30-100"), (100, 999, "tokens 100+")]
    l26 = 26

    print(f"{'Phase':>15} | {'L26 same%':>10} | {'L26→L35 zh':>11} | {'L26→L35 en':>11} | {'n_pairs':>8}")
    print("-" * 65)

    for start, end, label in bins:
        same_26 = 0
        zh_match_26 = 0
        en_match_26 = 0
        count = 0

        for i in range(N_PROBLEMS):
            zh_lt = zh_results[i]["layer_tokens"][str(l26)]
            en_lt = en_results[i]["layer_tokens"][str(l26)]
            zh_at = zh_results[i]["actual_tokens"]
            en_at = en_results[i]["actual_tokens"]
            n = min(len(zh_lt), len(en_lt))

            for t in range(max(0, start), min(n, end)):
                if zh_lt[t] == en_lt[t]:
                    same_26 += 1
                if t < len(zh_at) and zh_lt[t] == zh_at[t]:
                    zh_match_26 += 1
                if t < len(en_at) and en_lt[t] == en_at[t]:
                    en_match_26 += 1
                count += 1

        if count > 0:
            print(f"  {label:>13} | {same_26/count:9.1%} | {zh_match_26/count:10.1%} | {en_match_26/count:10.1%} | {count:>7}")

    # 5. Example decoded tokens
    print(f"\n4. EXAMPLE: Problem 0 — first 20 tokens")
    i = 0
    zh_at = zh_results[i]["actual_tokens"][:20]
    en_at = en_results[i]["actual_tokens"][:20]
    print(f"  zh actual: {tokenizer.decode(zh_at)}")
    print(f"  en actual: {tokenizer.decode(en_at)}")
    for l in [20, 26, 32]:
        if l in EXIT_LAYERS:
            zh_lt = zh_results[i]["layer_tokens"][str(l)][:20]
            en_lt = en_results[i]["layer_tokens"][str(l)][:20]
            print(f"  zh L{l}: {tokenizer.decode(zh_lt)}")
            print(f"  en L{l}: {tokenizer.decode(en_lt)}")

    # Save
    results = {
        "model": MODEL_NAME,
        "n_problems": N_PROBLEMS,
        "exit_layers": EXIT_LAYERS,
        "max_tokens": MAX_TOKENS,
        "layer_accuracy": {str(l): v for l, v in layer_accuracy.items()},
        "cross_lingual_agreement": {str(l): v for l, v in cross_lingual.items()},
        "actual_cross_lingual_agreement": actual_agreement,
        "zh_total_tokens": sum(r["n_tokens"] for r in zh_results),
        "en_total_tokens": sum(r["n_tokens"] for r in en_results),
    }

    outpath = OUTPUT_DIR / "intervention_gentime_early_exit.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Also save raw per-problem data for deeper analysis
    raw_path = OUTPUT_DIR / "gentime_early_exit_raw.json"
    with open(raw_path, "w") as f:
        json.dump({"zh": zh_results, "en": en_results}, f)
    print(f"Saved raw data to {raw_path}")


if __name__ == "__main__":
    main()
