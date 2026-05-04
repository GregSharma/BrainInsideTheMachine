"""
Exp AJ2: Hard Crystallization — AIME-level problems

Same crystallization map as AJ, but on problems that require genuine multi-step reasoning.
If the model "knows" AIME answers at token zero, that's extraordinary.
If it doesn't, we find the boundary between "already knows" and "actually computing."

Key question: does p(answer) grow along the LAYER axis (depth computes)
or the TOKEN axis (chain-of-thought computes) for hard problems?
"""

import json
import numpy as np
import torch
import torch.nn.functional as F_torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
MAX_NEW_TOKENS = 200  # AIME needs more tokens for reasoning

# Mix of difficulty levels:
# - Easy arithmetic (baseline, should crystallize at t=0)
# - Medium multi-step (requires 2-3 steps)
# - Hard AIME-level (requires genuine reasoning chains)
PROBLEMS = [
    # EASY: single-step arithmetic
    {
        "prompt_en": "Calculate 47 + 86.",
        "prompt_zh": "计算 47 + 86 的值。",
        "answer": 133,
        "difficulty": "easy",
        "label": "addition",
    },
    {
        "prompt_en": "Calculate 15 × 8.",
        "prompt_zh": "计算 15 × 8 的值。",
        "answer": 120,
        "difficulty": "easy",
        "label": "multiplication",
    },
    # MEDIUM: multi-step but doable
    {
        "prompt_en": "What is the sum of all positive divisors of 28?",
        "prompt_zh": "28 的所有正因数之和是多少？",
        "answer": 56,
        "difficulty": "medium",
        "label": "divisor_sum",
    },
    {
        "prompt_en": "How many integers between 1 and 100 are divisible by 3 but not by 5?",
        "prompt_zh": "1 到 100 之间有多少个整数能被 3 整除但不能被 5 整除？",
        "answer": 27,
        "difficulty": "medium",
        "label": "inclusion_exclusion",
    },
    {
        "prompt_en": "What is the remainder when 2^10 is divided by 7?",
        "prompt_zh": "2 的 10 次方除以 7 的余数是多少？",
        "answer": 2,
        "difficulty": "medium",
        "label": "modular_exp",
    },
    # HARD: requires genuine multi-step reasoning
    {
        "prompt_en": "Find the number of positive integer divisors of 12! (twelve factorial).",
        "prompt_zh": "求 12!（12 的阶乘）的正因数个数。",
        "answer": 792,
        "difficulty": "hard",
        "label": "factorial_divisors",
    },
    {
        "prompt_en": "How many 4-digit palindromes are there? A palindrome reads the same forwards and backwards.",
        "prompt_zh": "有多少个四位回文数？回文数是指正读和反读都相同的数。",
        "answer": 90,
        "difficulty": "hard",
        "label": "palindromes",
    },
    {
        "prompt_en": "Find the last three digits of 7^2025.",
        "prompt_zh": "求 7 的 2025 次方的最后三位数。",
        "answer": 807,
        "difficulty": "hard",
        "label": "modular_power",
    },
    # AIME-ADJACENT: 2026 AIME I Problem 8 (simplified prompt)
    {
        "prompt_en": "Let n be the number of positive integer divisors of 17017^17 that leave a remainder of 5 upon division by 12. Find the remainder when n is divided by 1000.",
        "prompt_zh": "设 n 为 17017 的 17 次方的正整数因数中除以 12 余 5 的个数。求 n 除以 1000 的余数。",
        "answer": 0,  # placeholder — we'll see if model gets anything coherent
        "difficulty": "aime",
        "label": "aime_2026_p8",
    },
    # Another AIME-level
    {
        "prompt_en": "Find the sum of all positive integers n < 1000 such that n^2 + n + 1 is divisible by both 7 and 13.",
        "prompt_zh": "求所有满足 n² + n + 1 同时被 7 和 13 整除的正整数 n（n < 1000）之和。",
        "answer": 0,  # placeholder
        "difficulty": "aime",
        "label": "quadratic_mod",
    },
]


def get_answer_token_ids(tokenizer, answer):
    """Get token IDs for the answer."""
    ans_str = str(answer)
    ids = set()
    for prefix in ["", " ", "\n"]:
        toks = tokenizer.encode(prefix + ans_str, add_special_tokens=False)
        ids.update(toks)
    toks = tokenizer.encode(ans_str, add_special_tokens=False)
    ids.update(toks)
    return list(ids)


def generate_with_crystallization(model, tokenizer, prompt, answer,
                                  max_new=MAX_NEW_TOKENS):
    """
    Generate token by token. At each step, extract hidden states at every layer
    and compute p(answer_token) via early exit.
    """
    device = model.device
    ALL_LAYERS = list(range(model.config.num_hidden_layers))
    answer_token_ids = get_answer_token_ids(tokenizer, answer)

    final_ln = model.model.norm
    lm_head = model.lm_head

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], device=device)

    crystallization = []
    generated_tokens = []
    # Also track top-1 predicted token at each (layer, token) for qualitative analysis
    top1_tokens = []

    for t in range(max_new):
        layer_hiddens = {}

        def make_capture(l):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                layer_hiddens[l] = h[:, -1:, :].detach()
            return hook

        cap_handles = [model.model.layers[l].register_forward_hook(make_capture(l))
                       for l in ALL_LAYERS]

        with torch.no_grad():
            outputs = model(input_ids)

        for h in cap_handles:
            h.remove()

        next_logits = outputs.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1)

        layer_probs = {}
        layer_top1 = {}
        for l in ALL_LAYERS:
            h_l = layer_hiddens[l]
            h_normed = final_ln(h_l)
            logits_l = lm_head(h_normed).float().squeeze(0).squeeze(0)
            probs = F_torch.softmax(logits_l, dim=-1)
            p_answer = max(probs[tid].item() for tid in answer_token_ids) if answer_token_ids else 0.0
            layer_probs[l] = p_answer
            # Top-1 token at this layer
            top1_id = logits_l.argmax().item()
            layer_top1[l] = top1_id

        crystallization.append(layer_probs)
        top1_tokens.append(layer_top1)
        generated_tokens.append(next_token.item())

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    n_layers = model.config.num_hidden_layers

    n_tok = len(crystallization)
    grid = np.zeros((n_tok, n_layers))
    for t_idx in range(n_tok):
        for l in ALL_LAYERS:
            grid[t_idx, l] = crystallization[t_idx].get(l, 0.0)

    # Decode top-1 tokens at L35 for first 20 gen tokens (what each layer "wants to say")
    top1_L35 = []
    for t_idx in range(min(n_tok, 30)):
        tid = top1_tokens[t_idx].get(n_layers - 1, 0)
        top1_L35.append(tokenizer.decode([tid]))

    return grid, gen_text, generated_tokens, top1_L35


def main():
    print("=== Exp AJ2: Hard Crystallization ===")
    print("AIME-level problems: does the model know before it speaks?\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers

    all_results = []

    for pi, prob in enumerate(PROBLEMS):
        prompt = prob["prompt_en"]  # English to make it harder
        answer = prob["answer"]
        diff = prob["difficulty"]
        label = prob["label"]

        print(f"\nP{pi} [{diff}] {label}: {prompt[:60]}...")
        grid, gen_text, gen_toks, top1_L35 = generate_with_crystallization(
            model, tokenizer, prompt, answer, max_new=MAX_NEW_TOKENS
        )

        # t=0 profile: what does the model "know" before generating?
        if grid.shape[0] > 0:
            t0_max_layer = int(np.argmax(grid[0, :]))
            t0_max_p = float(grid[0, t0_max_layer])
        else:
            t0_max_layer, t0_max_p = -1, 0.0

        # When does p(answer) first exceed thresholds?
        thresholds = [0.01, 0.05, 0.10, 0.50, 0.90]
        first_exceed = {}
        for thresh in thresholds:
            found = False
            for t_idx in range(grid.shape[0]):
                if grid[t_idx, n_layers - 1] > thresh:
                    first_exceed[str(thresh)] = t_idx
                    found = True
                    break
            if not found:
                first_exceed[str(thresh)] = -1

        # Max p(answer) achieved anywhere
        if grid.size > 0:
            max_idx = np.unravel_index(grid.argmax(), grid.shape)
            max_p = float(grid[max_idx])
        else:
            max_idx = (-1, -1)
            max_p = 0.0

        # Answer in generated text?
        ans_str = str(answer)
        answer_found = ans_str in gen_text if answer > 0 else False

        result = {
            "problem_idx": pi,
            "difficulty": diff,
            "label": label,
            "prompt": prompt[:80],
            "answer": answer,
            "gen_text": gen_text[:500],
            "n_tokens_generated": grid.shape[0],
            "answer_found": answer_found,
            "t0_max_p": round(t0_max_p, 6),
            "t0_max_layer": t0_max_layer,
            "t0_profile_L28_35": [round(float(grid[0, l]), 6) for l in range(28, n_layers)]
                if grid.shape[0] > 0 else [],
            "first_exceed": first_exceed,
            "max_p": round(max_p, 6),
            "max_p_token": int(max_idx[0]),
            "max_p_layer": int(max_idx[1]),
            "top1_L35_first20": top1_L35[:20],
            "token_profile_L35": [round(float(grid[t, n_layers - 1]), 6)
                                  for t in range(min(grid.shape[0], 50))],
        }

        all_results.append(result)

        print(f"  t=0 max p(ans): {t0_max_p:.4f} at L{t0_max_layer}")
        print(f"  First p>0.01 at L{n_layers-1}: token {first_exceed.get('0.01', -1)}")
        print(f"  First p>0.10 at L{n_layers-1}: token {first_exceed.get('0.1', -1)}")
        print(f"  First p>0.50 at L{n_layers-1}: token {first_exceed.get('0.5', -1)}")
        print(f"  Max p(ans): {max_p:.4f} at (t={max_idx[0]}, L{max_idx[1]})")
        print(f"  Answer found in text: {answer_found}")
        print(f"  Generated: {gen_text[:120]}...")
        print(f"  Top-1 at L{n_layers-1}, first 10 tokens: {top1_L35[:10]}")

    # Summary by difficulty
    print("\n\n=== SUMMARY BY DIFFICULTY ===")
    for diff in ["easy", "medium", "hard", "aime"]:
        subset = [r for r in all_results if r["difficulty"] == diff]
        if not subset:
            continue
        mean_t0 = np.mean([r["t0_max_p"] for r in subset])
        mean_first_01 = np.mean([r["first_exceed"].get("0.01", -1) for r in subset
                                 if r["first_exceed"].get("0.01", -1) >= 0] or [-1])
        mean_first_50 = np.mean([r["first_exceed"].get("0.5", -1) for r in subset
                                 if r["first_exceed"].get("0.5", -1) >= 0] or [-1])
        n_found = sum(1 for r in subset if r["answer_found"])
        print(f"\n{diff.upper()} (n={len(subset)}):")
        print(f"  Mean p(ans) at t=0, best layer: {mean_t0:.4f}")
        print(f"  Mean token where p>0.01: {mean_first_01:.1f}")
        print(f"  Mean token where p>0.50: {mean_first_50:.1f}")
        print(f"  Answers found: {n_found}/{len(subset)}")

    output = {
        "experiment": "AJ2_hard_crystallization",
        "method": "Per (layer, token) early-exit p(answer) on easy/medium/hard/AIME problems",
        "model": MODEL_NAME,
        "problems": all_results,
    }

    out_path = OUTPUT_DIR / "expAJ_crystallization.json"
    # Save to separate file to not overwrite AJ
    out_path = OUTPUT_DIR / "expAJ2_hard_crystallization.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
