"""Phase 0: Behavioral verification — does Qwen2.5-3B show Chinese >> English on math?

Run with: MPLBACKEND=Agg .venv_wsl/bin/python phase0_behavioral.py

This is a GATE experiment. If Qwen2.5-3B doesn't show asymmetry,
we either pivot to 3-8B or proceed knowing the behavioral gap may be
too small at 3B (structural Z can still exist without behavioral evidence).

Note: Qwen2.5-3B is a BASE model, not instruct-tuned. We use a simple
few-shot format to get reliable numerical answers.
"""

import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config ---
MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Few-shot examples to prime the base model for Q&A format
FEW_SHOT_ZH = """问题：5 + 7 = ?
答案：12

问题：100 / 4 = ?
答案：25

"""

FEW_SHOT_EN = """Question: 5 + 7 = ?
Answer: 12

Question: 100 / 4 = ?
Answer: 25

"""

# --- Prompt pairs: (Chinese, English, expected_answer, category) ---
PAIRS = [
    {
        "zh": "计算：15 × 17 + 15 × 3 = ?",
        "en": "Calculate: 15 × 17 + 15 × 3 = ?",
        "answer": "300",
        "category": "arithmetic",
    },
    {
        "zh": "求方程 2x + 5 = 17 的解。",
        "en": "Solve the equation 2x + 5 = 17.",
        "answer": "6",
        "category": "algebra",
    },
    {
        "zh": "一个正方形的面积是 144 平方厘米，它的周长是多少厘米？",
        "en": "A square has an area of 144 square centimeters. What is its perimeter in centimeters?",
        "answer": "48",
        "category": "geometry",
    },
    {
        "zh": "从 1 到 100 的整数中，有多少个是 3 的倍数？",
        "en": "Among the integers from 1 to 100, how many are multiples of 3?",
        "answer": "33",
        "category": "counting",
    },
    {
        "zh": "计算 2^10 的值。",
        "en": "Calculate the value of 2^10.",
        "answer": "1024",
        "category": "arithmetic",
    },
    {
        "zh": "如果 f(x) = x² - 4x + 3，求 f(5) 的值。",
        "en": "If f(x) = x² - 4x + 3, find the value of f(5).",
        "answer": "8",
        "category": "algebra",
    },
    {
        "zh": "一个数列的前五项是 2, 6, 18, 54, 162。第六项是多少？",
        "en": "The first five terms of a sequence are 2, 6, 18, 54, 162. What is the sixth term?",
        "answer": "486",
        "category": "sequences",
    },
    {
        "zh": "求 gcd(48, 36) 的值。",
        "en": "Find the value of gcd(48, 36).",
        "answer": "12",
        "category": "number_theory",
    },
    {
        "zh": "从 5 个人中选 2 个人组成委员会，有多少种不同的选法？",
        "en": "How many different ways can you choose 2 people from 5 to form a committee?",
        "answer": "10",
        "category": "combinatorics",
    },
    {
        "zh": "计算 sqrt(169) 的值。",
        "en": "Calculate the value of sqrt(169).",
        "answer": "13",
        "category": "arithmetic",
    },
    {
        "zh": "一个三角形的三条边长分别是 3, 4, 5。这个三角形的面积是多少？",
        "en": "A triangle has sides of length 3, 4, and 5. What is the area of this triangle?",
        "answer": "6",
        "category": "geometry",
    },
    {
        "zh": "求满足 x² = 49 的正整数 x。",
        "en": "Find the positive integer x satisfying x² = 49.",
        "answer": "7",
        "category": "algebra",
    },
    {
        "zh": "如果一个等差数列的首项是 3，公差是 4，求第 10 项。",
        "en": "If an arithmetic sequence has first term 3 and common difference 4, find the 10th term.",
        "answer": "39",
        "category": "sequences",
    },
    {
        "zh": "100 除以 7 的余数是多少？",
        "en": "What is the remainder when 100 is divided by 7?",
        "answer": "2",
        "category": "arithmetic",
    },
    {
        "zh": "解不等式 3x - 7 > 8，求 x 的最小正整数解。",
        "en": "Solve the inequality 3x - 7 > 8. Find the smallest positive integer solution for x.",
        "answer": "6",
        "category": "algebra",
    },
]


def extract_number(text: str) -> str | None:
    """Extract the first clear numerical answer from model output."""
    # Stop at newlines (base model often continues with next "question")
    first_line = text.strip().split("\n")[0].strip()

    patterns = [
        r"(?:答案|answer|result|等于|=)\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
        r"^(-?\d+(?:\.\d+)?)\s*$",  # entire first line is a number
        r"^(-?\d+(?:\.\d+)?)",       # first line starts with a number
    ]
    for pat in patterns:
        match = re.search(pat, first_line, re.IGNORECASE)
        if match:
            return match.group(1)

    # Broader search in first line
    nums = re.findall(r"-?\d+(?:\.\d+)?", first_line)
    return nums[-1] if nums else None


def check_answer(extracted: str | None, expected: str) -> bool:
    """Check if extracted answer matches expected."""
    if extracted is None:
        return False
    expected_vals = {e.strip() for e in expected.split(",")}
    return extracted.strip() in expected_vals


def main():
    print("=" * 60)
    print("PHASE 0: Behavioral Verification")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {len(PAIRS)} pairs (Chinese + English)")
    print("=" * 60)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    results = []

    for i, pair in enumerate(PAIRS):
        print(f"\n--- Problem {i+1}/{len(PAIRS)}: {pair['category']} ---")

        for lang, prompt_key, few_shot in [
            ("zh", "zh", FEW_SHOT_ZH),
            ("en", "en", FEW_SHOT_EN),
        ]:
            prompt = pair[prompt_key]
            # Few-shot format matching the exemplars
            if lang == "zh":
                full_prompt = f"{few_shot}问题：{prompt}\n答案："
            else:
                full_prompt = f"{few_shot}Question: {prompt}\nAnswer: "

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            extracted = extract_number(generated)
            correct = check_answer(extracted, pair["answer"])

            result = {
                "pair_idx": i,
                "lang": lang,
                "category": pair["category"],
                "prompt": prompt,
                "expected": pair["answer"],
                "extracted": extracted,
                "correct": correct,
                "raw_output": generated[:300],
            }
            results.append(result)

            status = "CORRECT" if correct else "WRONG"
            print(f"  [{lang.upper()}] {status} (expected={pair['answer']}, got={extracted})")
            if not correct:
                print(f"         raw: {generated[:100]}")

    # --- Summary ---
    zh_correct = sum(1 for r in results if r["lang"] == "zh" and r["correct"])
    en_correct = sum(1 for r in results if r["lang"] == "en" and r["correct"])
    n = len(PAIRS)

    print("\n" + "=" * 60)
    print("PHASE 0 RESULTS")
    print("=" * 60)
    print(f"Chinese: {zh_correct}/{n} correct ({100*zh_correct/n:.0f}%)")
    print(f"English: {en_correct}/{n} correct ({100*en_correct/n:.0f}%)")
    print(f"Delta:   {zh_correct - en_correct:+d} (positive = Chinese advantage)")

    if zh_correct > en_correct:
        print("\n>> CHINESE ADVANTAGE DETECTED. Proceed with Qwen2.5-3B.")
    elif zh_correct == en_correct:
        print("\n>> NO ASYMMETRY. Consider: (a) harder problems, (b) Qwen3-8B, (c) proceed anyway.")
    else:
        print("\n>> ENGLISH ADVANTAGE (?). Unexpected. Investigate prompt formatting.")

    # Per-category breakdown
    print("\nPer-Category Breakdown:")
    categories = sorted(set(r["category"] for r in results))
    print(f"{'Category':<15} {'ZH':>5} {'EN':>5} {'Delta':>6}")
    for cat in categories:
        zh_cat = sum(1 for r in results if r["lang"] == "zh" and r["category"] == cat and r["correct"])
        en_cat = sum(1 for r in results if r["lang"] == "en" and r["category"] == cat and r["correct"])
        total_cat = sum(1 for r in results if r["lang"] == "zh" and r["category"] == cat)
        print(f"{cat:<15} {zh_cat:>3}/{total_cat} {en_cat:>3}/{total_cat} {zh_cat-en_cat:>+5d}")

    # Per-problem detail
    print("\nPer-Problem Detail:")
    print(f"{'#':>3} {'Cat':<15} {'ZH':>3} {'EN':>3} {'Expected':>10} {'ZH_got':>10} {'EN_got':>10}")
    for i in range(n):
        zh_r = next(r for r in results if r["pair_idx"] == i and r["lang"] == "zh")
        en_r = next(r for r in results if r["pair_idx"] == i and r["lang"] == "en")
        zh_ok = "Y" if zh_r["correct"] else "N"
        en_ok = "Y" if en_r["correct"] else "N"
        print(f"{i+1:>3} {zh_r['category']:<15} {zh_ok:>3} {en_ok:>3} "
              f"{zh_r['expected']:>10} {str(zh_r['extracted']):>10} {str(en_r['extracted']):>10}")

    # Save
    output_path = OUTPUT_DIR / "phase0_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "n_pairs": n,
            "zh_correct": zh_correct,
            "en_correct": en_correct,
            "delta": zh_correct - en_correct,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
