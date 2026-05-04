#!/usr/bin/env python3
"""Zero-CoT: Can the model answer AMC problems with NO generation?

Just one forward pass on the prompt. Read the hidden state at every layer.
Project through unembedding for the 5 answer-choice tokens.
See how many it gets right with literally zero chain-of-thought.

This is the maximally aggressive version of "let the latent reasoning sing."
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

# Layers to probe
ALL_LAYERS = list(range(36))


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

    # Answer token IDs
    answer_token_ids = []
    for letter in CHOICE_LETTERS:
        tids = tokenizer.encode(letter, add_special_tokens=False)
        answer_token_ids.append(tids[0])
    answer_token_ids_t = torch.tensor(answer_token_ids, device=DEVICE)
    print(f"Answer token IDs: {dict(zip(CHOICE_LETTERS, answer_token_ids))}\n")

    # Get the unembedding weights for answer tokens
    lm_head_weight = model.lm_head.weight  # [vocab, d_model]
    answer_embeds = lm_head_weight[answer_token_ids_t, :].float()  # [5, d_model]

    results = []

    for pnum in sorted(problems.keys()):
        ptext = problems[pnum]
        correct = ANSWER_KEY[pnum]
        prompt = make_prompt(ptext)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        n_prompt_tokens = input_ids.shape[1]

        print(f"--- Problem {pnum} (correct={correct}, {n_prompt_tokens} prompt tokens) ---",
              flush=True)

        # Single forward pass with hidden states
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        # out.hidden_states is a tuple of (n_layers+1) tensors: [batch, seq, d_model]
        # Index 0 = embedding output, Index i = after layer i-1
        hidden_states = out.hidden_states  # tuple of [1, seq, d_model]

        # Also get the actual output logits (after LM head)
        output_logits = out.logits[0, -1, :]  # [vocab]
        output_answer_logits = output_logits[answer_token_ids_t].float()

        # For each layer, project hidden state at last token through unembed
        layer_results = {}
        for li in range(n_layers + 1):
            hs = hidden_states[li][0, -1, :].float()  # [d_model]
            # Apply RMSNorm (the model's final norm before LM head)
            normed = model.model.norm(hs.unsqueeze(0).to(model.model.norm.weight.dtype))
            logits = (normed.float() @ answer_embeds.T).squeeze(0)  # [5]
            layer_results[li] = {
                CHOICE_LETTERS[i]: logits[i].item()
                for i in range(5)
            }

        # What does each layer vote for?
        print(f"  {'Layer':>6s}  {'A':>7s}  {'B':>7s}  {'C':>7s}  {'D':>7s}  {'E':>7s}  Vote  {'✓/✗':>3s}")
        print(f"  {'-'*56}")

        # Show embedding + selected layers + final
        show_layers = [0, 4, 8, 12, 16, 18, 20, 22, 24, 26, 27, 28, 29, 30,
                       31, 32, 33, 34, 35, 36]
        show_layers = [l for l in show_layers if l <= n_layers]

        for li in show_layers:
            lr = layer_results[li]
            winner = max(lr, key=lr.get)
            ok = "✓" if winner == correct else " "
            label = f"L{li}" if li > 0 else "emb"
            if li == n_layers:
                label = "LMH"
            logit_strs = [f"{lr[c]:+7.1f}" for c in CHOICE_LETTERS]
            print(f"  {label:>6s}  {'  '.join(logit_strs)}  [{winner}]  {ok}")

        # Output logits (after full LM head)
        out_winner = CHOICE_LETTERS[output_answer_logits.argmax().item()]
        out_ok = "✓" if out_winner == correct else " "
        out_strs = [f"{output_answer_logits[i].item():+7.1f}" for i in range(5)]
        print(f"  {'OUT':>6s}  {'  '.join(out_strs)}  [{out_winner}]  {out_ok}")

        # Record
        l35_logits = layer_results[n_layers]  # after last layer
        l35_vote = max(l35_logits, key=l35_logits.get)
        out_vote = out_winner

        results.append({
            "problem": pnum,
            "correct": correct,
            "l35_vote": l35_vote,
            "out_vote": out_vote,
            "l35_correct": l35_vote == correct,
            "out_correct": out_vote == correct,
            "prompt_tokens": n_prompt_tokens,
            "layer_results": {str(k): v for k, v in layer_results.items()},
        })
        print()

    # Clean up
    del out, hidden_states
    torch.cuda.empty_cache()

    # Save
    os.makedirs("output", exist_ok=True)
    outpath = "output/exp_zero_cot.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {outpath}")

    # SUMMARY
    print(f"\n{'='*70}")
    print("ZERO-CoT SUMMARY: One forward pass, no generation")
    print(f"{'='*70}")
    print(f"\n{'Prob':>5s}  {'Correct':>8s}  {'L35 Vote':>9s}  {'OUT Vote':>9s}  {'L35✓':>5s}  {'OUT✓':>5s}")
    print("-" * 50)

    l35_correct = 0
    out_correct = 0
    for r in results:
        l35_ok = "✓" if r["l35_correct"] else "✗"
        out_ok = "✓" if r["out_correct"] else "✗"
        if r["l35_correct"]:
            l35_correct += 1
        if r["out_correct"]:
            out_correct += 1
        print(f"  P{r['problem']:2d}  {r['correct']:>8s}  {r['l35_vote']:>9s}  "
              f"{r['out_vote']:>9s}  {l35_ok:>5s}  {out_ok:>5s}")

    n = len(results)
    print("-" * 50)
    print(f"  L35: {l35_correct}/{n} ({100*l35_correct/n:.1f}%)")
    print(f"  OUT: {out_correct}/{n} ({100*out_correct/n:.1f}%)")
    print(f"\n  Baseline (with CoT): 7/19 (36.8%)")
    print(f"  Soft deflation (with CoT): 9/19 (47.4%)")
    print(f"  Zero-CoT L35: {l35_correct}/{n} ({100*l35_correct/n:.1f}%)")
    print(f"  Zero-CoT OUT: {out_correct}/{n} ({100*out_correct/n:.1f}%)")


if __name__ == "__main__":
    main()
