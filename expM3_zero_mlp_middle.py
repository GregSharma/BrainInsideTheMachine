"""Experiment M3: Zero MLP Deltas at L9-L26 During Generation (THE KILL SHOT)

At every layer L9 through L26, attention fires normally (full KV cache consistency),
but the MLP delta is zeroed before it enters the residual stream.

L0-L8: full computation (encoding + early reasoning)
L9-L26: attention only, MLP zeroed (remove language dressing, keep context routing)
L27-L35: full computation (attractor + L30 computation spike)

This is NOT a layer skip. Every layer fires. The KV cache is fully consistent.
The depth is 36 layers. We just kill the MLP's contribution in the middle layers.

Why it might work:
- K2b proved KV cache barely matters for short prompts
- Hidden state still flows through 36 layers of residual accumulation + layernorm
- MLP in L9-L26 is 94% language damping (Exp E), thermostat-following (Exp F)
- L30's computation spike happens AFTER the zeroed zone
- The attractor at L27+ can absorb perturbations (20x norm mismatch tolerance)

10 problems (5 simple, 5 hard). 128 tokens. Full generation. Compare to baseline.
Also capture L30 MLP delta magnitude to verify L30 computation still fires.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
ZERO_LAYERS = list(range(9, 27))  # L9 through L26 inclusive

# =============================================================================
# Test problems: 5 simple + 5 hard
# =============================================================================
test_problems = [
    # Simple
    {"prompt_en": "Calculate 47 + 86.", "prompt_zh": "计算 47 + 86 的值。", "answer": "133", "difficulty": "simple"},
    {"prompt_en": "A rectangle has length 12 and width 5. Find its area.", "prompt_zh": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60", "difficulty": "simple"},
    {"prompt_en": "What is the remainder when 100 is divided by 7?", "prompt_zh": "100 除以 7 的余数是多少？", "answer": "2", "difficulty": "simple"},
    {"prompt_en": "Calculate 15 × 8.", "prompt_zh": "计算 15 × 8 的值。", "answer": "120", "difficulty": "simple"},
    {"prompt_en": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.",
     "prompt_zh": "等差数列首项为 2，公差为 3，求前 5 项之和。", "answer": "40", "difficulty": "simple"},
    # Hard
    {"prompt_en": "Calculate 387 × 29.", "prompt_zh": "计算 387 × 29 的值。", "answer": "11223", "difficulty": "hard"},
    {"prompt_en": "Find the value of C(10, 3).", "prompt_zh": "求组合数 C(10, 3) 的值。", "answer": "120", "difficulty": "hard"},
    {"prompt_en": "What is the remainder when 7654 is divided by 37?", "prompt_zh": "7654 除以 37 的余数是多少？", "answer": "34", "difficulty": "hard"},
    {"prompt_en": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.",
     "prompt_zh": "等差数列首项为 7，公差为 11，求前 25 项之和。", "answer": "3475", "difficulty": "hard"},
    {"prompt_en": "A rectangle has length 47 and width 33. Find its area.", "prompt_zh": "一个长方形的长为 47，宽为 33，求其面积。", "answer": "1551", "difficulty": "hard"},
]


def run_generation(prompt, zero_mlp_layers=None, capture_l30=False):
    """Run autoregressive generation with optional MLP zeroing.

    Args:
        prompt: text prompt
        zero_mlp_layers: list of layer indices where MLP output is zeroed. None = baseline.
        capture_l30: if True, capture L30 MLP delta norms per token.

    Returns:
        dict with generated text, token ids, and optional L30 data.
    """
    input_ids = tokenizer.encode(prompt)
    l30_norms = []

    # --- Register MLP zeroing hooks ---
    handles = []
    if zero_mlp_layers:
        for layer_idx in zero_mlp_layers:
            def make_zero_hook(li):
                def hook_fn(module, input, output):
                    # MLP output is the delta (plain tensor). Zero it.
                    return torch.zeros_like(output)
                return hook_fn
            handles.append(
                model.model.layers[layer_idx].mlp.register_forward_hook(make_zero_hook(layer_idx))
            )

    # --- Register L30 MLP capture hook ---
    l30_handle = None
    if capture_l30:
        def l30_capture_hook(module, input, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            norm = float(mlp_out[0, -1, :].float().norm().item())
            l30_norms.append(norm)
        l30_handle = model.model.layers[30].mlp.register_forward_hook(l30_capture_hook)

    try:
        # Prefill
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        first_token_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[first_token_id]], device=device)
        generated_ids = [first_token_id]

        # Autoregressive generation
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
        if l30_handle:
            l30_handle.remove()

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "text": full_text,
        "n_tokens": len(generated_ids),
        "l30_mlp_norms": l30_norms if capture_l30 else None,
    }


def check_answer(text, answer):
    """Check if the correct answer appears in generated text."""
    return answer in text


# =============================================================================
# Run experiments
# =============================================================================
print("=" * 70)
print("EXPERIMENT M3: ZERO MLP DELTAS AT L9-L26 DURING GENERATION")
print(f"  Zeroing layers: {ZERO_LAYERS[0]}-{ZERO_LAYERS[-1]} ({len(ZERO_LAYERS)} layers)")
print("=" * 70)

results = {
    "experiment": "M3: Zero MLP L9-L26",
    "zeroed_layers": f"L{ZERO_LAYERS[0]}-L{ZERO_LAYERS[-1]}",
    "n_zeroed": len(ZERO_LAYERS),
    "max_new_tokens": MAX_NEW_TOKENS,
    "problems": [],
}

for lang in ["en", "zh"]:
    lang_label = "English" if lang == "en" else "Chinese"
    print(f"\n{'─'*70}")
    print(f"  LANGUAGE: {lang_label}")
    print(f"{'─'*70}")

    for prob_idx, prob in enumerate(test_problems):
        prompt = prob[f"prompt_{lang}"]
        answer = prob["answer"]
        difficulty = prob["difficulty"]

        # Baseline
        baseline = run_generation(prompt, zero_mlp_layers=None, capture_l30=True)
        baseline_correct = check_answer(baseline["text"], answer)

        # MLP-zeroed
        zeroed = run_generation(prompt, zero_mlp_layers=ZERO_LAYERS, capture_l30=True)
        zeroed_correct = check_answer(zeroed["text"], answer)

        # L30 delta comparison
        bl_l30_mean = np.mean(baseline["l30_mlp_norms"]) if baseline["l30_mlp_norms"] else 0
        zr_l30_mean = np.mean(zeroed["l30_mlp_norms"]) if zeroed["l30_mlp_norms"] else 0
        l30_ratio = zr_l30_mean / bl_l30_mean if bl_l30_mean > 0 else float('nan')

        # Language check
        zh_chars_b = sum(1 for c in baseline["text"] if '\u4e00' <= c <= '\u9fff')
        en_chars_b = sum(1 for c in baseline["text"] if c.isalpha() and c.isascii())
        zh_chars_z = sum(1 for c in zeroed["text"] if '\u4e00' <= c <= '\u9fff')
        en_chars_z = sum(1 for c in zeroed["text"] if c.isalpha() and c.isascii())

        bl_lang = "zh" if zh_chars_b > en_chars_b else "en"
        zr_lang = "zh" if zh_chars_z > en_chars_z else "en"
        lang_preserved = (zr_lang == lang)

        status = ""
        if zeroed_correct:
            status += "MATH_OK "
        else:
            status += "MATH_FAIL "
        if lang_preserved:
            status += "LANG_OK "
        else:
            status += f"LANG_FLIP({zr_lang}) "

        print(f"\n  [{prob_idx}] {difficulty.upper()} | {prompt[:50]}...")
        print(f"    Baseline: {'CORRECT' if baseline_correct else 'WRONG':>7s} | {baseline['text'][:80]}")
        print(f"    Zeroed:   {'CORRECT' if zeroed_correct else 'WRONG':>7s} | {zeroed['text'][:80]}")
        print(f"    L30 MLP: baseline={bl_l30_mean:.1f}, zeroed={zr_l30_mean:.1f}, ratio={l30_ratio:.2f}")
        print(f"    Status: {status}")

        entry = {
            "problem_idx": prob_idx,
            "language": lang,
            "difficulty": difficulty,
            "prompt": prompt,
            "answer": answer,
            "baseline_text": baseline["text"],
            "baseline_correct": baseline_correct,
            "zeroed_text": zeroed["text"],
            "zeroed_correct": zeroed_correct,
            "lang_preserved": lang_preserved,
            "zeroed_output_lang": zr_lang,
            "l30_baseline_mean": float(bl_l30_mean),
            "l30_zeroed_mean": float(zr_l30_mean),
            "l30_ratio": float(l30_ratio),
            "baseline_l30_norms": [float(x) for x in baseline["l30_mlp_norms"]],
            "zeroed_l30_norms": [float(x) for x in zeroed["l30_mlp_norms"]],
        }
        results["problems"].append(entry)


# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print("M3 SUMMARY")
print("=" * 70)

for lang in ["en", "zh"]:
    lang_probs = [p for p in results["problems"] if p["language"] == lang]
    n_correct_bl = sum(1 for p in lang_probs if p["baseline_correct"])
    n_correct_zr = sum(1 for p in lang_probs if p["zeroed_correct"])
    n_lang_ok = sum(1 for p in lang_probs if p["lang_preserved"])
    n_total = len(lang_probs)

    simple = [p for p in lang_probs if p["difficulty"] == "simple"]
    hard = [p for p in lang_probs if p["difficulty"] == "hard"]
    s_correct = sum(1 for p in simple if p["zeroed_correct"])
    h_correct = sum(1 for p in hard if p["zeroed_correct"])

    l30_ratios = [p["l30_ratio"] for p in lang_probs if not np.isnan(p["l30_ratio"])]
    avg_l30_ratio = np.mean(l30_ratios) if l30_ratios else float('nan')

    lang_label = "English" if lang == "en" else "Chinese"
    print(f"\n  {lang_label}:")
    print(f"    Baseline correct: {n_correct_bl}/{n_total}")
    print(f"    Zeroed correct:   {n_correct_zr}/{n_total} (simple: {s_correct}/{len(simple)}, hard: {h_correct}/{len(hard)})")
    print(f"    Language preserved: {n_lang_ok}/{n_total}")
    print(f"    L30 MLP ratio (zeroed/baseline): {avg_l30_ratio:.2f}")

all_probs = results["problems"]
total_bl = sum(1 for p in all_probs if p["baseline_correct"])
total_zr = sum(1 for p in all_probs if p["zeroed_correct"])
total_lang = sum(1 for p in all_probs if p["lang_preserved"])
total = len(all_probs)

l30_all_ratios = [p["l30_ratio"] for p in all_probs if not np.isnan(p["l30_ratio"])]
avg_l30_all = np.mean(l30_all_ratios) if l30_all_ratios else float('nan')

print(f"\n  OVERALL:")
print(f"    Baseline correct: {total_bl}/{total}")
print(f"    Zeroed correct:   {total_zr}/{total} = {total_zr/total:.0%}")
print(f"    Language preserved: {total_lang}/{total} = {total_lang/total:.0%}")
print(f"    L30 MLP delta ratio: {avg_l30_all:.2f}")

if total_zr >= 16:
    verdict = "KILL SHOT CONFIRMED: MLP at L9-L26 is overhead. Math correct without it."
elif total_zr >= 12:
    verdict = "STRONG RESULT: Most math survives MLP zeroing at L9-L26"
elif total_zr >= 8:
    verdict = "PARTIAL: Some math survives, MLP partially redundant"
else:
    verdict = "MLP AT L9-L26 IS NEEDED: Math fails without middle MLP"

print(f"\n  VERDICT: {verdict}")

results["summary"] = {
    "baseline_correct": total_bl,
    "zeroed_correct": total_zr,
    "lang_preserved": total_lang,
    "total": total,
    "accuracy_baseline": total_bl / total,
    "accuracy_zeroed": total_zr / total,
    "l30_delta_ratio": float(avg_l30_all),
    "verdict": verdict,
}

with open("output/expM3_zero_mlp_middle.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expM3_zero_mlp_middle.json")
