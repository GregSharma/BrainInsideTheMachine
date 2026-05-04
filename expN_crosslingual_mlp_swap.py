"""Experiment N: Cross-Lingual MLP Delta Swap During Generation

Instead of zeroing MLP deltas at L9-L26, SWAP them between Chinese and English
runs of the same problem.

Run each problem in both Chinese and English. Cache the MLP deltas from the
English run at L9-L26. Then re-run the Chinese version, replacing its MLP deltas
at L9-L26 with the cached English deltas (and vice versa).

If both versions still produce correct math in their respective languages:
the MLP computation at L9-L26 is cross-lingually IDENTICAL at the functional level.
The "language dressing" is literally interchangeable.

5 problems. 128 tokens. Full generation.
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
SWAP_LAYERS = list(range(9, 27))  # L9 through L26 inclusive

test_problems = [
    {"en": "Calculate 47 + 86.", "zh": "计算 47 + 86 的值。", "answer": "133", "difficulty": "simple"},
    {"en": "A rectangle has length 12 and width 5. Find its area.", "zh": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60", "difficulty": "simple"},
    {"en": "Find the value of C(10, 3).", "zh": "求组合数 C(10, 3) 的值。", "answer": "120", "difficulty": "hard"},
    {"en": "Calculate 387 × 29.", "zh": "计算 387 × 29 的值。", "answer": "11223", "difficulty": "hard"},
    {"en": "An arithmetic sequence has first term 3 and common difference 7. Find the sum of the first 20 terms.",
     "zh": "等差数列首项为 3，公差为 7，求前 20 项之和。", "answer": "1390", "difficulty": "hard"},
]


def run_baseline(prompt):
    """Run normal generation, return text and correctness."""
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
    past_kv = outputs.past_key_values
    next_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[next_id]], device=device)
    generated_ids = [next_id]

    for step in range(MAX_NEW_TOKENS - 1):
        with torch.no_grad():
            out = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_id = int(out.logits[0, -1].argmax())
        generated_ids.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)
        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_with_cached_mlp(prompt, cached_mlp_deltas):
    """Run generation, replacing MLP deltas at SWAP_LAYERS with cached ones.

    cached_mlp_deltas: dict[layer_idx] -> list of tensors (one per gen step, NOT prefill).
    Prefill runs normally. Only autoregressive steps get swapped.
    """
    input_ids = tokenizer.encode(prompt)
    gen_step = [0]  # tracks autoregressive steps (0-indexed, excludes prefill)
    in_generation = [False]

    # --- Register MLP replacement hooks ---
    handles = []
    for layer_idx in SWAP_LAYERS:
        def make_swap_hook(li):
            def hook_fn(module, input, output):
                if not in_generation[0]:
                    return output  # Don't swap during prefill
                step = gen_step[0]
                cached = cached_mlp_deltas.get(li)
                if cached is not None and step < len(cached):
                    donor = cached[step].to(output.device, output.dtype)
                    # Donor is single-token shape (1,1,d), should match output
                    return donor
                return output
            return hook_fn
        handles.append(
            model.model.layers[layer_idx].mlp.register_forward_hook(make_swap_hook(layer_idx))
        )

    try:
        # Prefill — hooks pass through
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]

        in_generation[0] = True  # Now entering autoregressive mode

        for step in range(MAX_NEW_TOKENS - 1):
            gen_step[0] = step
            with torch.no_grad():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)
            if next_id == tokenizer.eos_token_id:
                break

    finally:
        for h in handles:
            h.remove()

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_and_capture_mlp(prompt):
    """Run generation AND capture MLP deltas at SWAP_LAYERS for each gen step.

    Returns: (generated_text, dict[layer_idx] -> list of tensors per gen step)
    Only captures GENERATION steps (not prefill).
    """
    input_ids = tokenizer.encode(prompt)
    captured = {li: [] for li in SWAP_LAYERS}
    in_generation = [False]

    handles = []
    for layer_idx in SWAP_LAYERS:
        def make_capture_hook(li):
            def hook_fn(module, input, output):
                if in_generation[0]:
                    captured[li].append(output.detach().clone().cpu())
            return hook_fn
        handles.append(
            model.model.layers[layer_idx].mlp.register_forward_hook(make_capture_hook(layer_idx))
        )

    try:
        # Prefill — don't capture
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]

        in_generation[0] = True  # Now capture gen steps

        for step in range(MAX_NEW_TOKENS - 1):
            with torch.no_grad():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)
            if next_id == tokenizer.eos_token_id:
                break
    finally:
        for h in handles:
            h.remove()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, captured


def classify_lang(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = sum(1 for c in text if c.isalpha() and c.isascii())
    if zh > en * 2:
        return "chinese"
    elif en > zh * 2:
        return "english"
    return "mixed"


# =============================================================================
# Run experiments
# =============================================================================
print("=" * 70)
print("EXPERIMENT N: CROSS-LINGUAL MLP DELTA SWAP AT L9-L26")
print(f"  Swap layers: {SWAP_LAYERS[0]}-{SWAP_LAYERS[-1]} ({len(SWAP_LAYERS)} layers)")
print("=" * 70)

results = {"experiment": "N: Cross-Lingual MLP Swap", "swap_layers": f"L{SWAP_LAYERS[0]}-L{SWAP_LAYERS[-1]}", "problems": []}

for prob_idx, prob in enumerate(test_problems):
    print(f"\n{'─'*70}")
    print(f"  Problem {prob_idx}: EN={prob['en'][:50]}  ZH={prob['zh'][:30]}")
    print(f"  Answer: {prob['answer']}")
    print(f"{'─'*70}")

    # Step 1: Baselines
    en_baseline = run_baseline(prob["en"])
    zh_baseline = run_baseline(prob["zh"])
    print(f"  EN baseline: {en_baseline[:100]}")
    print(f"  ZH baseline: {zh_baseline[:100]}")

    # Step 2: Capture MLP deltas from both
    print(f"  Capturing EN MLP deltas...")
    en_text, en_mlp = run_and_capture_mlp(prob["en"])
    print(f"  Capturing ZH MLP deltas...")
    zh_text, zh_mlp = run_and_capture_mlp(prob["zh"])

    # Step 3: Swap — run EN prompt with ZH MLP deltas
    print(f"  Running EN with ZH MLP...")
    en_with_zh_mlp = run_with_cached_mlp(prob["en"], zh_mlp)
    print(f"  Running ZH with EN MLP...")
    zh_with_en_mlp = run_with_cached_mlp(prob["zh"], en_mlp)

    # Evaluate
    en_bl_correct = prob["answer"] in en_baseline
    zh_bl_correct = prob["answer"] in zh_baseline
    en_swap_correct = prob["answer"] in en_with_zh_mlp
    zh_swap_correct = prob["answer"] in zh_with_en_mlp

    en_swap_lang = classify_lang(en_with_zh_mlp)
    zh_swap_lang = classify_lang(zh_with_en_mlp)

    print(f"\n  EN baseline: {'CORRECT' if en_bl_correct else 'WRONG':>7s} | lang={classify_lang(en_baseline)}")
    print(f"  ZH baseline: {'CORRECT' if zh_bl_correct else 'WRONG':>7s} | lang={classify_lang(zh_baseline)}")
    print(f"  EN+ZH_MLP:   {'CORRECT' if en_swap_correct else 'WRONG':>7s} | lang={en_swap_lang} | {en_with_zh_mlp[:100]}")
    print(f"  ZH+EN_MLP:   {'CORRECT' if zh_swap_correct else 'WRONG':>7s} | lang={zh_swap_lang} | {zh_with_en_mlp[:100]}")

    entry = {
        "problem_idx": prob_idx,
        "en_prompt": prob["en"],
        "zh_prompt": prob["zh"],
        "answer": prob["answer"],
        "difficulty": prob["difficulty"],
        "en_baseline": en_baseline,
        "zh_baseline": zh_baseline,
        "en_with_zh_mlp": en_with_zh_mlp,
        "zh_with_en_mlp": zh_with_en_mlp,
        "en_bl_correct": en_bl_correct,
        "zh_bl_correct": zh_bl_correct,
        "en_swap_correct": en_swap_correct,
        "zh_swap_correct": zh_swap_correct,
        "en_swap_lang": en_swap_lang,
        "zh_swap_lang": zh_swap_lang,
        "en_lang_preserved": en_swap_lang == "english",
        "zh_lang_preserved": zh_swap_lang == "chinese",
    }
    results["problems"].append(entry)


# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print("N SUMMARY")
print("=" * 70)

n_en_bl = sum(1 for p in results["problems"] if p["en_bl_correct"])
n_zh_bl = sum(1 for p in results["problems"] if p["zh_bl_correct"])
n_en_swap = sum(1 for p in results["problems"] if p["en_swap_correct"])
n_zh_swap = sum(1 for p in results["problems"] if p["zh_swap_correct"])
n_en_lang = sum(1 for p in results["problems"] if p["en_lang_preserved"])
n_zh_lang = sum(1 for p in results["problems"] if p["zh_lang_preserved"])
total = len(results["problems"])

print(f"  EN baseline correct: {n_en_bl}/{total}")
print(f"  ZH baseline correct: {n_zh_bl}/{total}")
print(f"  EN+ZH_MLP correct:   {n_en_swap}/{total}")
print(f"  ZH+EN_MLP correct:   {n_zh_swap}/{total}")
print(f"  EN lang preserved after ZH MLP: {n_en_lang}/{total}")
print(f"  ZH lang preserved after EN MLP: {n_zh_lang}/{total}")

total_swap_correct = n_en_swap + n_zh_swap
total_bl_correct = n_en_bl + n_zh_bl
total_lang_preserved = n_en_lang + n_zh_lang

if total_swap_correct >= 8:
    verdict = "MLP DELTAS ARE CROSS-LINGUALLY IDENTICAL: Math survives, language determined by prompt/residual"
elif total_swap_correct >= 5:
    verdict = "PARTIAL INTERCHANGEABILITY: MLP deltas mostly language-agnostic"
elif total_swap_correct >= 2:
    verdict = "WEAK INTERCHANGEABILITY: Some math survives but MLP contributes to language"
else:
    verdict = "MLP DELTAS ARE LANGUAGE-SPECIFIC: Swap destroys output"

print(f"\n  VERDICT: {verdict}")

results["summary"] = {
    "en_baseline_correct": n_en_bl,
    "zh_baseline_correct": n_zh_bl,
    "en_swap_correct": n_en_swap,
    "zh_swap_correct": n_zh_swap,
    "en_lang_preserved": n_en_lang,
    "zh_lang_preserved": n_zh_lang,
    "total": total,
    "verdict": verdict,
}

with open("output/expN_crosslingual_mlp_swap.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expN_crosslingual_mlp_swap.json")
