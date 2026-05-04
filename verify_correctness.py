"""Verify numerical correctness of PC0-swapped outputs against ground truth.

For each of the 20 test problems:
1. Compute the exact answer from problem parameters
2. Parse final numerical answer from each condition's text
3. Compare: does the PC0 swap produce the correct number?

If 19/20 texts match the English baseline verbatim, and the English baseline
is correct, then correctness is inherited. But we verify explicitly.
"""

import json
import re
import random as pyrandom
from math import comb, factorial

# ---- Regenerate problems with same seed as intervention script ----

def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            answer = a + b
            zh, en = f"计算 {a} + {b} 的值。", f"Calculate {a} + {b}."
        else:
            answer = a * b
            zh, en = f"计算 {a} × {b} 的值。", f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "category": 0, "answer": answer,
                         "desc": f"{a}{'+'if op=='plus' else'×'}{b}={answer}"})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        answer = comb(n_val, k_val)
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "category": 1, "answer": answer,
                         "desc": f"C({n_val},{k_val})={answer}"})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        answer = a % b
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "category": 2, "answer": answer,
                         "desc": f"{a}%{b}={answer}"})
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        answer = w * h
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "category": 3, "answer": answer,
                         "desc": f"{w}×{h}={answer}"})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        answer = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        problems.append({"zh": zh, "en": en, "category": 4, "answer": answer,
                         "desc": f"AP(a1={a1},d={d},n={n_terms})={answer}"})
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


def extract_numbers(text):
    """Extract all numbers from text, return as list of ints/floats."""
    # Find all number-like patterns (integers and decimals)
    nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', text)
    results = []
    for n in nums:
        n_clean = n.replace(',', '')
        try:
            if '.' in n_clean:
                results.append(float(n_clean))
            else:
                results.append(int(n_clean))
        except ValueError:
            pass
    return results


def check_answer_in_text(text, correct_answer, category):
    """Check if the correct answer appears in the generated text."""
    nums = extract_numbers(text)
    # Direct match
    if correct_answer in nums:
        return True, correct_answer, "exact_match"
    # For floats, check close match
    for n in nums:
        if abs(n - correct_answer) < 0.01:
            return True, n, "float_match"
    return False, None, "not_found"


def main():
    problems = generate_problems(200, seed=42)
    selected_indices = select_problems(problems, 20)

    with open('output/intervention_splice_steer.json') as f:
        data = json.load(f)

    cat_names = {0: 'arithmetic', 1: 'combinatorics', 2: 'modular', 3: 'geometry', 4: 'sequences'}
    conditions = ['baseline_en', 'baseline_zh', 'splice_pc0_swap', 'splice_raw',
                  'splice_reverse_steer', 'splice_random_dir', 'splice_scrambled']

    print("=" * 100)
    print("NUMERICAL CORRECTNESS VERIFICATION")
    print("=" * 100)

    # Per-condition tallies
    correct_counts = {c: 0 for c in conditions}
    total = len(data['per_problem'])

    for i, prob_result in enumerate(data['per_problem']):
        prob_idx = prob_result['prob_idx']
        prob = problems[prob_idx]
        correct = prob['answer']
        cat = cat_names[prob['category']]

        print(f"\nP{i:2d} [{cat:14s}] {prob['desc']:30s}  correct={correct}")

        for cond in conditions:
            text = prob_result[cond]['text']
            found, val, method = check_answer_in_text(text, correct, prob['category'])
            if found:
                correct_counts[cond] += 1
                marker = "OK"
            else:
                marker = "MISS"
                # Show what numbers were found
                nums = extract_numbers(text)
                # Filter out small numbers that are likely parameters, not answers
                # Show last few numbers as likely candidates
                marker = f"MISS (found: {nums[-5:] if nums else 'none'})"

            # Compact output: only show non-OK
            if not found:
                print(f"  {cond:25s}: {marker}")
                print(f"    text: {text[:120]}...")

    print("\n" + "=" * 100)
    print("SUMMARY: Correct answer found in text")
    print("=" * 100)
    for cond in conditions:
        pct = correct_counts[cond] / total * 100
        bar = "#" * correct_counts[cond] + "." * (total - correct_counts[cond])
        print(f"  {cond:25s}: {correct_counts[cond]:2d}/{total} ({pct:5.1f}%)  [{bar}]")

    # Key comparison: PC0 swap vs English baseline
    print("\n" + "=" * 100)
    print("KEY QUESTION: Does PC0 swap inherit English baseline correctness?")
    print("=" * 100)
    pc0_matches_en = 0
    pc0_both_correct = 0
    pc0_correct_en_wrong = 0
    pc0_wrong_en_correct = 0
    for i, prob_result in enumerate(data['per_problem']):
        prob_idx = prob_result['prob_idx']
        prob = problems[prob_idx]
        correct = prob['answer']

        en_found, _, _ = check_answer_in_text(prob_result['baseline_en']['text'], correct, prob['category'])
        pc0_found, _, _ = check_answer_in_text(prob_result['splice_pc0_swap']['text'], correct, prob['category'])

        if en_found and pc0_found:
            pc0_both_correct += 1
        elif en_found and not pc0_found:
            pc0_wrong_en_correct += 1
        elif not en_found and pc0_found:
            pc0_correct_en_wrong += 1

        if en_found == pc0_found:
            pc0_matches_en += 1

    print(f"  Both correct:       {pc0_both_correct}/{total}")
    print(f"  EN correct, PC0 wrong: {pc0_wrong_en_correct}/{total}")
    print(f"  PC0 correct, EN wrong: {pc0_correct_en_wrong}/{total}")
    print(f"  Agreement rate:     {pc0_matches_en}/{total} ({pc0_matches_en/total*100:.0f}%)")

    # Save results
    results = {
        'total_problems': total,
        'correct_counts': correct_counts,
        'ground_truth': [
            {'prob_idx': problems[si]['desc'], 'answer': problems[si]['answer'],
             'category': cat_names[problems[si]['category']]}
            for si in selected_indices
        ],
        'pc0_vs_en': {
            'both_correct': pc0_both_correct,
            'en_correct_pc0_wrong': pc0_wrong_en_correct,
            'pc0_correct_en_wrong': pc0_correct_en_wrong,
            'agreement': pc0_matches_en
        }
    }
    with open('output/correctness_verification.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to output/correctness_verification.json")


if __name__ == "__main__":
    main()
