"""Commitment trajectory via logit rank of correct answer B.

At every generation step, check: what rank is the correct answer token
in the logit distribution? If B is rank 1-5 early and drops to rank 90+
at some point, that's the commitment.

Also tracks attention entropy at late layers (L32-L35) per Ghost's
original Phase 2 suggestion.
"""
import json, time, re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from exp_delayed_deflation_p12 import WindowedDeflation

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 150  # first 150 steps is enough

DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
DEFLATE_REFRESH = 25

SYS = ("You are solving an AMC 12A multiple choice math problem. "
       "Think step by step, show your work, then clearly state your "
       "final answer as (A), (B), (C), (D), or (E).")

P12_TEXT = (
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
PROMPT = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"

# Answer-related tokens to track
ANSWER_TOKENS = {}  # filled after tokenizer loads


def run_trajectory(model, tokenizer, input_ids, deflator=None, label="",
                   answer_token_ids=None, track_layers=(32, 33, 34, 35)):
    """Generate and collect per-step metrics."""
    gen_ids = []
    past_kv = None
    steps = []

    # Hook for attention entropy at specific layers
    entropy_captures = {}
    attn_hooks = []

    for layer_idx in track_layers:
        def make_attn_hook(li):
            def hook(module, input, output):
                # output is (attn_output, attn_weights, past_kv) when output_attentions=True
                # but we don't have output_attentions. Instead compute entropy from Q,K.
                pass
            return hook

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                if deflator:
                    deflator.start_gen()
                    deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[0, -1, :].float()  # (vocab_size,)

            # Rank of each answer token
            sorted_indices = logits.argsort(descending=True)
            ranks = torch.zeros_like(logits, dtype=torch.long)
            ranks[sorted_indices] = torch.arange(len(logits), device=DEVICE)

            step_data = {"step": step}

            # Track rank and logit value of answer tokens
            for name, tid in answer_token_ids.items():
                r = ranks[tid].item()
                v = logits[tid].item()
                step_data[f"rank_{name}"] = r
                step_data[f"logit_{name}"] = round(v, 4)

            # Top-5 tokens
            top5_ids = sorted_indices[:5].tolist()
            top5_tokens = [tokenizer.decode([t]) for t in top5_ids]
            step_data["top5"] = top5_tokens
            step_data["top1_logit"] = round(logits[sorted_indices[0]].item(), 4)

            # Softmax entropy of full distribution
            probs = F.softmax(logits, dim=0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            step_data["entropy"] = round(entropy, 4)

            # KV cache effective rank at L33 (using key norms as proxy)
            keys_L33 = past_kv.layers[33].keys[0, 0, :, :].float()  # (seq, head_dim)
            if keys_L33.shape[0] >= 4:
                sv = torch.linalg.svdvals(keys_L33)
                sv_norm = sv / sv.sum()
                kv_entropy = -(sv_norm * torch.log(sv_norm + 1e-10)).sum().item()
                r90 = (sv.cumsum(0) / sv.sum() < 0.9).sum().item() + 1
                step_data["kv_L33_r90"] = r90
                step_data["kv_L33_entropy"] = round(kv_entropy, 4)
                step_data["kv_L33_sv1_frac"] = round((sv[0] / sv.sum()).item(), 4)

            steps.append(step_data)

            next_id = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            # Decoded token for context
            step_data["token"] = tokenizer.decode([tid])

            if deflator:
                deflator.tick(past_kv)

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return steps, text, len(gen_ids)


def main():
    print("=" * 70)
    print("COMMITMENT TRAJECTORY VIA LOGIT RANK + KV RANK")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    # Find answer-related token IDs
    answer_tokens = {}
    for text, name in [("-3/2", "B_frac"), ("B", "B_letter"),
                        ("-1.5", "B_decimal"), ("(B)", "B_paren"),
                        ("-5/3", "A_frac"), ("A", "A_letter"),
                        ("3", "digit_3"), ("2", "digit_2")]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            answer_tokens[name] = ids[0]
            print(f"  {name}: '{text}' -> token {ids[0]} ({tokenizer.decode([ids[0]])!r})")

    print(f"\nLoaded {MODEL_NAME}\n", flush=True)

    conditions = [
        ("baseline", None),
        ("deflation_onset0", 0),
        ("deflation_onset25", 25),
        ("deflation_onset30", 30),
    ]

    all_results = {}

    for name, onset in conditions:
        print(f"\n=== {name} ===", flush=True)
        if onset is not None:
            defl = WindowedDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R, alpha=0.10,
                                     refresh_every=DEFLATE_REFRESH, active_from=onset)
        else:
            defl = None

        t0 = time.time()
        steps, text, ntok = run_trajectory(
            model, tokenizer, input_ids, deflator=defl,
            label=name, answer_token_ids=answer_tokens)
        dt = time.time() - t0

        if defl:
            defl.remove()

        print(f"  {ntok} tokens, {dt:.1f}s")

        # Print rank of B at key timepoints
        print(f"  B_frac rank trajectory:")
        for t in [0, 5, 10, 15, 20, 25, 27, 30, 35, 40, 50, 75, 100]:
            if t < len(steps) and "rank_B_frac" in steps[t]:
                s = steps[t]
                print(f"    t={t:3d}: rank={s['rank_B_frac']:5d}  "
                      f"logit={s.get('logit_B_frac',0):+8.2f}  "
                      f"top1={s['top5'][0]:10s}  "
                      f"entropy={s['entropy']:.2f}  "
                      f"kv_r90={s.get('kv_L33_r90','?')}  "
                      f"kv_sv1={s.get('kv_L33_sv1_frac','?')}")

        all_results[name] = steps

    # Compare baseline vs onset=25 vs onset=30 at the commitment window
    print(f"\n{'='*70}")
    print("B_frac RANK COMPARISON at commitment window")
    print(f"{'='*70}")
    for t in range(min(50, min(len(all_results[n]) for n in all_results))):
        line = f"  t={t:3d}"
        for name in ["baseline", "deflation_onset25", "deflation_onset30"]:
            if t < len(all_results[name]) and "rank_B_frac" in all_results[name][t]:
                r = all_results[name][t]["rank_B_frac"]
                line += f"  {name[:8]:>8}={r:5d}"
        print(line)

    # Save
    with open("output/exp_commitment_logits.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n-> output/exp_commitment_logits.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
