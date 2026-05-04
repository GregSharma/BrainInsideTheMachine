"""Logit lens on P12 baseline (looping) generation.

At every 50 tokens, probe intermediate layers: project hidden state through
layernorm + lm_head. Track rank of answer letter B vs A/C/D/E.

If B has anomalously high rank at intermediate layers while the model loops,
the answer is PRESENT in the residual stream but buried by later processing.
"""
import json, time, re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 800

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

ANSWER_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}
TARGET_LAYERS = [0, 5, 10, 15, 18, 20, 25, 27, 30, 33, 35]
PROBE_EVERY = 50


def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print(f"Loaded. {len(model.model.layers)} layers.\n", flush=True)

    norm = model.model.norm
    lm_head = model.lm_head

    # Hook storage
    captured = {}

    def make_hook(li):
        def hook(module, inp, output):
            hs = output[0]
            if hs.dim() == 3:
                hs = hs[:, -1, :]
            captured[li] = hs.detach()
        return hook

    hooks = []
    for li in TARGET_LAYERS:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    # Generate
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    past_kv = None
    gen_ids = []
    probes = []  # list of {step, layer, answer_ranks, top5_tokens, top5_probs}

    print("Generating baseline (no deflation, full sys prompt)...", flush=True)
    t0 = time.time()

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)

            past_kv = out.past_key_values
            logits_out = out.logits[:, -1, :]
            next_id = logits_out.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            # Probe at regular intervals
            if (step + 1) % PROBE_EVERY == 0:
                step_num = step + 1
                for li in TARGET_LAYERS:
                    if li not in captured:
                        continue
                    hs = captured[li]
                    normed = norm(hs)
                    layer_logits = lm_head(normed).squeeze(0)  # (vocab,)
                    layer_probs = F.softmax(layer_logits, dim=-1)

                    # Answer letter ranks and probs
                    answer_ranks = {}
                    answer_probs = {}
                    for letter, tok_id in ANSWER_IDS.items():
                        rank = (layer_logits > layer_logits[tok_id]).sum().item() + 1
                        answer_ranks[letter] = rank
                        answer_probs[letter] = layer_probs[tok_id].item()

                    # Top 5
                    top5v, top5i = layer_logits.topk(5)
                    top5_tokens = [tokenizer.decode([t]) for t in top5i.tolist()]
                    top5_probs = [layer_probs[t].item() for t in top5i.tolist()]

                    probes.append({
                        "step": step_num,
                        "layer": li,
                        "answer_ranks": answer_ranks,
                        "answer_probs": answer_probs,
                        "top5_tokens": top5_tokens,
                        "top5_probs": [round(p, 6) for p in top5_probs],
                    })

                captured.clear()

    dt = round(time.time() - t0, 1)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    for h in hooks:
        h.remove()
    del past_kv, out
    torch.cuda.empty_cache()

    print(f"Generated {len(gen_ids)} tokens in {dt}s\n", flush=True)

    # ===== DISPLAY =====
    steps = sorted(set(p["step"] for p in probes))

    for step_num in steps:
        step_probes = sorted([p for p in probes if p["step"] == step_num],
                             key=lambda x: x["layer"])
        # What token was being generated?
        if step_num <= len(gen_ids):
            ctx_snippet = tokenizer.decode(gen_ids[max(0, step_num-20):step_num],
                                           skip_special_tokens=True)[-60:]
        else:
            ctx_snippet = "<end>"

        print(f"\n--- Step {step_num} (context: ...{ctx_snippet}) ---")

        for p in step_probes:
            b_rank = p["answer_ranks"]["B"]
            b_prob = p["answer_probs"]["B"]
            # Which answer letter has best rank?
            best_letter = min(p["answer_ranks"], key=p["answer_ranks"].get)
            best_rank = p["answer_ranks"][best_letter]

            # Marker if B is the best answer letter at this layer
            marker = " <<< B LEADS" if best_letter == "B" else ""

            print(f"  L{p['layer']:>2d}: B_rank={b_rank:>6d}  B_prob={b_prob:.2e}  "
                  f"best={best_letter}(rank={best_rank:>5d})  "
                  f"top5={p['top5_tokens']}{marker}")

    # Summary table: B rank at each step x layer
    print("\n" + "=" * 80)
    print("B RANK AT EACH STEP x LAYER (lower = more prominent)")
    print("=" * 80)
    header = f"{'Step':>6s}" + "".join(f"{'L'+str(li):>8s}" for li in TARGET_LAYERS)
    print(header)
    print("-" * len(header))

    for step_num in steps:
        step_probes = {p["layer"]: p for p in probes if p["step"] == step_num}
        row = f"{step_num:>6d}"
        for li in TARGET_LAYERS:
            if li in step_probes:
                b_rank = step_probes[li]["answer_ranks"]["B"]
                row += f"{b_rank:>8d}"
            else:
                row += f"{'---':>8s}"
        print(row)

    # Is B ever the BEST answer letter at any intermediate layer?
    print("\n" + "=" * 80)
    print("STEPS/LAYERS WHERE B IS THE TOP-RANKED ANSWER LETTER")
    print("=" * 80)
    b_wins = []
    for p in probes:
        best = min(p["answer_ranks"], key=p["answer_ranks"].get)
        if best == "B":
            b_wins.append(p)
            print(f"  Step {p['step']:>4d}, L{p['layer']:>2d}: "
                  f"B_rank={p['answer_ranks']['B']:>5d}  "
                  f"(A={p['answer_ranks']['A']}, C={p['answer_ranks']['C']}, "
                  f"D={p['answer_ranks']['D']}, E={p['answer_ranks']['E']})")

    if not b_wins:
        print("  NONE. B never leads among answer letters.")
    else:
        print(f"\n  {len(b_wins)} instances where B leads.")

    # Save
    out_data = {
        "n_tokens": len(gen_ids),
        "time_s": dt,
        "probes": probes,
        "output_last500": text[-500:] if text else "",
    }
    out_path = "output/exp_logit_lens_p12.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
