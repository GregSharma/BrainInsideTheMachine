"""Pre-generative knowledge test: does the model know -3/2 before generating?

MECE discrimination:
  1A: Answer in prompt encoding (logits elevated before generation)
  2A: Answer computed during shared reasoning steps
  3B: Intervention creates fresh computation

Two conditions:
  1. Full prompt WITH answer choices (A-E)
  2. Prompt WITHOUT answer choices (open-ended)

For each: process prompt, read logits at last token, check ranking
of answer-related tokens. No generation.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"

SYS = ("You are solving an AMC 12A multiple choice math problem. "
       "Think step by step, show your work, then clearly state your "
       "final answer as (A), (B), (C), (D), or (E).")

P12_WITH_CHOICES = (
    "The harmonic mean of a collection of numbers is the reciprocal of the "
    "arithmetic mean of the reciprocals of the numbers in the collection. "
    "For example, the harmonic mean of 4, 4, and 5 is\n\n"
    "1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n"
    "What is the harmonic mean of all the real roots of the 4050th degree "
    "polynomial\n\n"
    r"\prod_{k=1}^{2025} (kx^2 - 4x - 3) = "
    "(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)..."
    "(2025x^2 - 4x - 3)?\n\n"
    "(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3"
)

P12_NO_CHOICES = (
    "The harmonic mean of a collection of numbers is the reciprocal of the "
    "arithmetic mean of the reciprocals of the numbers in the collection. "
    "For example, the harmonic mean of 4, 4, and 5 is\n\n"
    "1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n"
    "What is the harmonic mean of all the real roots of the 4050th degree "
    "polynomial\n\n"
    r"\prod_{k=1}^{2025} (kx^2 - 4x - 3) = "
    "(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)..."
    "(2025x^2 - 4x - 3)?"
)

# Also test with a neutral system prompt (no AMC framing)
SYS_NEUTRAL = "You are a careful mathematical reasoner. Show your work step by step."


def make_prompt(sys_msg, problem_text):
    return f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{problem_text}<|im_end|>\n<|im_start|>assistant\n"


def analyze_logits(model, tokenizer, prompt, label):
    """Process prompt, read logits at last token. No generation."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.inference_mode():
        out = model(input_ids)
    logits = out.logits[0, -1, :].float().cpu()  # last token logits

    # Get probabilities
    probs = torch.softmax(logits, dim=0)

    # Check answer-related tokens
    answer_tokens = {
        "-3/2 (correct)": ["-3", "-", "3", "/", "2", "The", "\n"],
        "-5/3": ["-5"],
        "-6/5": ["-6"],
        "-5/6": ["-5"],  # same start as -5/3
        "-2/3": ["-2"],
    }

    # Check letter tokens
    letter_tokens = {}
    for letter in ["A", "B", "C", "D", "E"]:
        for variant in [f"({letter})", f" {letter}", letter, f"\n({letter}"]:
            tids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in tids:
                decoded = tokenizer.decode([tid])
                if letter in decoded and len(decoded.strip()) <= 3:
                    letter_tokens[f"{letter}:{decoded.strip()}"] = tid

    # Check specific fraction tokens
    fraction_strings = ["-3/2", "-5/3", "-6/5", "-5/6", "-2/3",
                        "3/2", "5/3", "6/5", "5/6", "2/3",
                        "-\\frac{3}{2}", "-\\frac{5}{3}"]

    print(f"\n{'='*60}")
    print(f"CONDITION: {label}")
    print(f"Prompt length: {input_ids.shape[1]} tokens")
    print(f"{'='*60}")

    # Top-20 tokens by probability
    top_k = 20
    top_probs, top_ids = probs.topk(top_k)
    print(f"\nTop {top_k} predicted next tokens:")
    for i, (p, tid) in enumerate(zip(top_probs, top_ids)):
        token_str = tokenizer.decode([tid.item()]).replace('\n', '\\n')
        logit_val = logits[tid.item()].item()
        print(f"  {i+1:2d}. [{tid.item():6d}] '{token_str:20s}'  p={p.item():.4f}  logit={logit_val:.2f}")

    # Check specific answer-relevant tokens
    print(f"\nAnswer-relevant token logits:")
    check_strings = [
        "To", "The", "Let", "We", "First",  # reasoning starters
        "(", "A", "B", "C", "D", "E",       # letter answers
        "-", "3", "2", "5", "6",             # digit components
    ]
    for s in check_strings:
        tids = tokenizer.encode(s, add_special_tokens=False)
        if tids:
            tid = tids[0]
            p = probs[tid].item()
            l = logits[tid].item()
            decoded = tokenizer.decode([tid]).replace('\n', '\\n')
            print(f"  '{decoded:10s}' (id={tid:6d})  p={p:.6f}  logit={l:.2f}")

    # Now the critical test: look at what letter the model would pick
    # if forced to output (A), (B), etc.
    print(f"\nForced-choice letter probabilities:")
    letter_probs = {}
    for letter in ["A", "B", "C", "D", "E"]:
        # Try several tokenizations
        best_p = 0
        best_tid = None
        for variant in [f"({letter})", f"({letter}", letter]:
            tids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in tids:
                p = probs[tid].item()
                if p > best_p:
                    best_p = p
                    best_tid = tid
        letter_probs[letter] = best_p
        decoded = tokenizer.decode([best_tid]).replace('\n', '\\n') if best_tid else "?"
        print(f"  ({letter}): p={best_p:.6f}  token='{decoded}'")

    winner = max(letter_probs, key=letter_probs.get)
    print(f"  >>> Highest letter: ({winner}) with p={letter_probs[winner]:.6f}")
    print(f"  >>> Correct is (B). {'MATCH' if winner == 'B' else 'NO MATCH'}")

    # Ratio test: how much does B dominate?
    if letter_probs['B'] > 0:
        others_max = max(v for k, v in letter_probs.items() if k != 'B')
        if others_max > 0:
            print(f"  >>> B/next_best ratio: {letter_probs['B']/others_max:.2f}x")

    # Also check if the model would generate reasoning or answer directly
    print(f"\nFirst-token tendency:")
    reasoning_starters = ["To", "The", "Let", "We", "First", "Step", "##"]
    answer_starters = ["(", "-", "A", "B", "C", "D", "E"]
    r_total = sum(probs[tokenizer.encode(s, add_special_tokens=False)[0]].item()
                  for s in reasoning_starters
                  if tokenizer.encode(s, add_special_tokens=False))
    a_total = sum(probs[tokenizer.encode(s, add_special_tokens=False)[0]].item()
                  for s in answer_starters
                  if tokenizer.encode(s, add_special_tokens=False))
    print(f"  Reasoning starters total prob: {r_total:.4f}")
    print(f"  Answer starters total prob:    {a_total:.4f}")
    print(f"  Model wants to: {'REASON' if r_total > a_total else 'ANSWER DIRECTLY'}")

    return {
        "label": label,
        "letter_probs": letter_probs,
        "winner": winner,
        "correct_match": winner == "B",
        "reasoning_prob": r_total,
        "answer_prob": a_total,
    }


def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print(f"Loaded.\n", flush=True)

    results = []

    # Condition 1: AMC prompt WITH answer choices
    prompt1 = make_prompt(SYS, P12_WITH_CHOICES)
    r1 = analyze_logits(model, tokenizer, prompt1, "AMC_with_choices")
    results.append(r1)

    # Condition 2: AMC prompt WITHOUT answer choices
    prompt2 = make_prompt(SYS, P12_NO_CHOICES)
    r2 = analyze_logits(model, tokenizer, prompt2, "AMC_no_choices")
    results.append(r2)

    # Condition 3: Neutral sys prompt WITH choices
    prompt3 = make_prompt(SYS_NEUTRAL, P12_WITH_CHOICES)
    r3 = analyze_logits(model, tokenizer, prompt3, "neutral_with_choices")
    results.append(r3)

    # Condition 4: Neutral sys prompt WITHOUT choices
    prompt4 = make_prompt(SYS_NEUTRAL, P12_NO_CHOICES)
    r4 = analyze_logits(model, tokenizer, prompt4, "neutral_no_choices")
    results.append(r4)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Pre-generative knowledge test")
    print("=" * 60)
    for r in results:
        status = "B WINS" if r["correct_match"] else f"{r['winner']} WINS"
        print(f"  {r['label']:25s}  {status:10s}  B_prob={r['letter_probs']['B']:.6f}")

    print("\nINTERPRETATION:")
    if results[0]["correct_match"]:
        print("  1A confirmed: model knows -3/2 before generating (with choices).")
        if results[1]["correct_match"]:
            print("  Strong 1A: knows even WITHOUT choices. Genuine pre-generative knowledge.")
        else:
            print("  Weak 1A: only knows WITH choices. May be pattern-matching from options.")
    else:
        print("  1A rejected: model does NOT know -3/2 before generating.")
        print("  Points to 2A: answer computed during chain-of-thought reasoning.")


if __name__ == "__main__":
    main()
