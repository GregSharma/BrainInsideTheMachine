#!/usr/bin/env python3
"""Bias-corrected Zero-CoT probing.

Two corrections over the naive approach:
1. Subtract null-prompt baseline logits (removes frequency bias like the A-dominance)
2. Also probe for actual ANSWER VALUES (e.g. "-3/2", "10", "4:30") not just letters

This tells us: does the model's hidden state carry answer-specific information
beyond the default token frequency bias?
"""
import json, time, os, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"

SYS = (
    "You are solving an AMC 12A multiple choice math problem. "
    "Think step by step, show your work, then clearly state your "
    "final answer as (A), (B), (C), (D), or (E)."
)

SKIP_PROBLEMS = {5, 10, 14, 20, 24, 25}

ANSWER_KEY = {
    1: 'E', 2: 'B', 3: 'A', 4: 'B', 6: 'B', 7: 'C', 8: 'E', 9: 'E',
    11: 'A', 12: 'B', 13: 'D', 15: 'C', 16: 'D', 17: 'A', 18: 'C',
    19: 'E', 21: 'A', 22: 'E', 23: 'C',
}

CHOICE_LETTERS = ['A', 'B', 'C', 'D', 'E']

# Actual answer values per problem — what the model would "think" numerically
# We'll tokenize the first distinctive token of each answer value
VALUE_MAP = {
    1: {'A': '3:30', 'B': '3:45', 'C': '4:00', 'D': '4:15', 'E': '4:30'},
    2: {'A': '3.5', 'B': '4', 'C': '4.5', 'D': '5', 'E': '6'},
    3: {'A': '28', 'B': '29', 'C': '30', 'D': '32', 'E': '33'},
    4: {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4'},
    9: {'A': '3/4', 'B': '1', 'C': '7/5', 'D': '3/2', 'E': '5/3'},
    12: {'A': '-5/3', 'B': '-3/2', 'C': '-6/5', 'D': '-5/6', 'E': '-2/3'},
    15: {'A': '8', 'B': '9', 'C': '10', 'D': '11', 'E': '12'},
    17: {'A': '6', 'B': '8', 'C': '10', 'D': '12', 'E': '14'},
    18: {'A': '36', 'B': '84', 'C': '186', 'D': '336', 'E': '486'},
    21: {'A': '8', 'B': '9', 'C': '10', 'D': '11', 'E': '12'},
    22: {'A': '1/12', 'B': '1/9', 'C': '1/8', 'D': '1/6', 'E': '1/4'},
}


def parse_amc_problems(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    problems = {}
    parts = re.split(r'^Problem (\d+)\s*$', content, flags=re.MULTILINE)
    for i in range(1, len(parts) - 1, 2):
        num = int(parts[i])
        raw = parts[i + 1].strip()
        if num in SKIP_PROBLEMS:
            continue
        raw = re.sub(r'\[Solution\]\([^)]*\)', '', raw)
        raw = re.sub(r'^-+\s*$', '', raw, flags=re.MULTILINE)
        if '[asy]' in raw.lower():
            continue
        raw = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', raw)
        raw = re.sub(r'\n{3,}', '\n\n', raw).strip()
        raw = re.sub(r'\nSee also.*', '', raw, flags=re.DOTALL)
        if num in ANSWER_KEY:
            problems[num] = raw
    return problems


def make_prompt(text):
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def main():
    problems = parse_amc_problems("2025_AMC_12A.md")
    print(f"Parsed {len(problems)} AMC problems")

    print(f"\nLoading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Loaded. {n_layers} layers.\n", flush=True)

    # Answer letter token IDs
    answer_token_ids = []
    for letter in CHOICE_LETTERS:
        tids = tokenizer.encode(letter, add_special_tokens=False)
        answer_token_ids.append(tids[0])
    answer_token_ids_t = torch.tensor(answer_token_ids, device=DEVICE)

    lm_head_weight = model.lm_head.weight.float()
    answer_embeds = lm_head_weight[answer_token_ids_t, :]  # [5, d_model]

    # ============================================================
    # STEP 1: Compute null-prompt baseline
    # ============================================================
    print("Computing null-prompt baseline...", flush=True)
    null_prompt = make_prompt("What is the answer to this multiple choice problem?\n\n(A) (B) (C) (D) (E)")
    null_ids = tokenizer(null_prompt, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        null_out = model(input_ids=null_ids, output_hidden_states=True, use_cache=False)

    # Get baseline logits at each layer
    null_layer_logits = {}
    for li in range(n_layers + 1):
        hs = null_out.hidden_states[li][0, -1, :].float()
        normed = model.model.norm(hs.unsqueeze(0).to(model.model.norm.weight.dtype))
        logits = (normed.float() @ answer_embeds.T).squeeze(0)
        null_layer_logits[li] = logits  # [5] tensor

    print(f"  Null baseline L35: {' '.join(f'{CHOICE_LETTERS[i]}:{null_layer_logits[35][i].item():+.1f}' for i in range(5))}")
    print()

    del null_out
    torch.cuda.empty_cache()

    # ============================================================
    # STEP 2: Run each problem with bias correction
    # ============================================================
    results = []
    probe_layers = [24, 27, 30, 33, 35]

    for pnum in sorted(problems.keys()):
        ptext = problems[pnum]
        correct = ANSWER_KEY[pnum]
        prompt = make_prompt(ptext)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        # --- LETTER PROBING (bias-corrected) ---
        print(f"--- Problem {pnum} (correct={correct}) ---")
        print(f"  Bias-corrected letter logits (problem - null):")
        print(f"  {'Layer':>6s}  {'A':>7s}  {'B':>7s}  {'C':>7s}  {'D':>7s}  {'E':>7s}  Vote")

        layer_votes = {}
        for li in probe_layers:
            hs = out.hidden_states[li][0, -1, :].float()
            normed = model.model.norm(hs.unsqueeze(0).to(model.model.norm.weight.dtype))
            raw_logits = (normed.float() @ answer_embeds.T).squeeze(0)
            corrected = raw_logits - null_layer_logits[li]

            vals = {CHOICE_LETTERS[i]: corrected[i].item() for i in range(5)}
            winner = max(vals, key=vals.get)
            ok = "✓" if winner == correct else " "
            layer_votes[li] = winner
            logit_strs = [f"{vals[c]:+7.2f}" for c in CHOICE_LETTERS]
            print(f"  L{li:2d}     {'  '.join(logit_strs)}  [{winner}] {ok}")

        # --- VALUE PROBING (if available for this problem) ---
        value_vote = None
        if pnum in VALUE_MAP:
            vmap = VALUE_MAP[pnum]
            # Tokenize the first token of each answer value
            value_tokens = {}
            for letter, val_str in vmap.items():
                tids = tokenizer.encode(val_str, add_special_tokens=False)
                value_tokens[letter] = tids[0]  # first token

            val_token_ids = torch.tensor([value_tokens[c] for c in CHOICE_LETTERS], device=DEVICE)
            val_embeds = lm_head_weight[val_token_ids, :]  # [5, d_model]

            # Also need null baseline for value tokens
            null_hs = out.hidden_states[35][0, -1, :].float()  # reuse problem hs
            # Actually compute null for value tokens too
            null_ids2 = tokenizer(null_prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                null_out2 = model(input_ids=null_ids2, output_hidden_states=True, use_cache=False)
            null_hs_35 = null_out2.hidden_states[35][0, -1, :].float()
            null_normed = model.model.norm(null_hs_35.unsqueeze(0).to(model.model.norm.weight.dtype))
            null_val_logits = (null_normed.float() @ val_embeds.T).squeeze(0)
            del null_out2

            print(f"\n  Value probing (answer values, bias-corrected at L35):")
            print(f"  {'Choice':>8s}  {'Value':>8s}  {'TokID':>6s}  {'Raw':>7s}  {'Null':>7s}  {'Delta':>7s}")

            val_deltas = {}
            for i, c in enumerate(CHOICE_LETTERS):
                hs_35 = out.hidden_states[35][0, -1, :].float()
                normed_35 = model.model.norm(hs_35.unsqueeze(0).to(model.model.norm.weight.dtype))
                raw = (normed_35.float() @ val_embeds[i:i+1, :].T).item()
                null = null_val_logits[i].item()
                delta = raw - null
                val_deltas[c] = delta
                ok = "✓" if c == correct else " "
                print(f"  {c:>8s}  {vmap[c]:>8s}  {value_tokens[c]:>6d}  {raw:+7.2f}  {null:+7.2f}  {delta:+7.2f} {ok}")

            value_vote = max(val_deltas, key=val_deltas.get)
            val_ok = "✓" if value_vote == correct else "✗"
            print(f"  Value vote: {value_vote} ({vmap[value_vote]}) {val_ok}")

        # Also get the OUTPUT logits (what the model would actually generate next)
        out_logits = out.logits[0, -1, :]
        # Rank by probability among answer-related tokens
        # Get probabilities for "The answer is (X)" pattern
        # Actually just look at the full output distribution's top tokens
        top_vals, top_ids = out_logits.topk(10)
        top_tokens = [tokenizer.decode([tid.item()]) for tid in top_ids]
        print(f"\n  Top-10 next tokens: {list(zip(top_tokens, [f'{v:.1f}' for v in top_vals.tolist()]))}")

        l35_corrected_vote = layer_votes.get(35, '?')
        results.append({
            "problem": pnum,
            "correct": correct,
            "l35_corrected_vote": l35_corrected_vote,
            "l35_correct": l35_corrected_vote == correct,
            "value_vote": value_vote,
            "value_correct": value_vote == correct if value_vote else None,
        })
        print()

        del out
        torch.cuda.empty_cache()

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("BIAS-CORRECTED ZERO-CoT SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Prob':>5s}  {'Correct':>8s}  {'L35corr':>8s}  {'ValVote':>8s}  {'L35✓':>5s}  {'Val✓':>5s}")
    print("-" * 55)

    l35_correct = 0
    val_correct = 0
    val_total = 0

    for r in results:
        l35_ok = "✓" if r["l35_correct"] else "✗"
        if r["l35_correct"]:
            l35_correct += 1

        val_ok = " "
        val_str = "—"
        if r["value_vote"] is not None:
            val_str = r["value_vote"]
            val_total += 1
            if r["value_correct"]:
                val_correct += 1
                val_ok = "✓"
            else:
                val_ok = "✗"

        print(f"  P{r['problem']:2d}  {r['correct']:>8s}  {r['l35_corrected_vote']:>8s}  "
              f"{val_str:>8s}  {l35_ok:>5s}  {val_ok:>5s}")

    n = len(results)
    print("-" * 55)
    print(f"  L35 corrected: {l35_correct}/{n} ({100*l35_correct/n:.1f}%)")
    if val_total > 0:
        print(f"  Value probe:   {val_correct}/{val_total} ({100*val_correct/val_total:.1f}%)")
    print(f"\n  Baseline CoT:        7/19 (36.8%)")
    print(f"  Soft deflation CoT:  9/19 (47.4%)")
    print(f"  Random chance:       1/5  (20.0%)")
    print(f"  Zero-CoT L35 corr:   {l35_correct}/{n} ({100*l35_correct/n:.1f}%)")
    if val_total > 0:
        print(f"  Zero-CoT Value:      {val_correct}/{val_total} ({100*val_correct/val_total:.1f}%)")

    # Save
    os.makedirs("output", exist_ok=True)
    with open("output/exp_zero_cot_corrected.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to output/exp_zero_cot_corrected.json")


if __name__ == "__main__":
    main()
