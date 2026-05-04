"""Measure computational commitment trajectory.

Ghost's discriminating experiment: track cos(h_L33, loop_template) vs
cos(h_L33, answer_template) at every generation step.

Prediction: sharp phase transition at token ~27.

Also runs WITH deflation to compare onset=25 (works) vs onset=30 (fails).
"""
import json, time, re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from exp_delayed_deflation_p12 import WindowedDeflation

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 200  # only need first ~100 tokens for commitment trajectory

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


def collect_hidden_trajectory(model, tokenizer, input_ids, deflator=None, label=""):
    """Generate and collect h_L33 at every step."""
    gen_ids = []
    past_kv = None
    h_L33_trajectory = []  # list of (step, h_L33_vector)

    # Hook to capture L33 hidden state
    captured = {}
    def capture_hook(module, input, output):
        # output is (hidden_states, ...)
        if isinstance(output, tuple):
            captured['h'] = output[0][:, -1, :].detach().clone()  # last token
        else:
            captured['h'] = output[:, -1, :].detach().clone()

    # Install capture hook on layer 33's output
    hook_handle = model.model.layers[33].register_forward_hook(capture_hook)

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
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()

            # Capture h_L33
            if 'h' in captured:
                h_L33_trajectory.append(captured['h'].cpu().float())

            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            if deflator:
                deflator.tick(past_kv)

    hook_handle.remove()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return h_L33_trajectory, text, len(gen_ids)


def main():
    print("=" * 70)
    print("COMMITMENT TRAJECTORY")
    print("Measuring cos(h_L33, template) at every generation step")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    print(f"Loaded {MODEL_NAME}\n", flush=True)

    # 1. Baseline (no deflation) - generates loop
    print("=== BASELINE (no deflation, loops) ===")
    traj_baseline, text_bl, ntok_bl = collect_hidden_trajectory(
        model, tokenizer, input_ids, deflator=None, label="baseline")
    print(f"  Generated {ntok_bl} tokens\n", flush=True)

    # 2. With deflation onset=0 (works, generates correct answer)
    print("=== DEFLATION onset=0 (works) ===")
    defl_0 = WindowedDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R, alpha=0.10,
                                refresh_every=DEFLATE_REFRESH, active_from=0)
    traj_correct, text_correct, ntok_correct = collect_hidden_trajectory(
        model, tokenizer, input_ids, deflator=defl_0, label="onset=0")
    defl_0.remove()
    print(f"  Generated {ntok_correct} tokens\n", flush=True)

    # 3. With deflation onset=25 (works at delta=0.10)
    print("=== DEFLATION onset=25 (works) ===")
    defl_25 = WindowedDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R, alpha=0.10,
                                 refresh_every=DEFLATE_REFRESH, active_from=25)
    traj_25, text_25, ntok_25 = collect_hidden_trajectory(
        model, tokenizer, input_ids, deflator=defl_25, label="onset=25")
    defl_25.remove()
    print(f"  Generated {ntok_25} tokens\n", flush=True)

    # 4. With deflation onset=30 (fails at delta=0.10)
    print("=== DEFLATION onset=30 (fails) ===")
    defl_30 = WindowedDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R, alpha=0.10,
                                 refresh_every=DEFLATE_REFRESH, active_from=30)
    traj_30, text_30, ntok_30 = collect_hidden_trajectory(
        model, tokenizer, input_ids, deflator=defl_30, label="onset=30")
    defl_30.remove()
    print(f"  Generated {ntok_30} tokens\n", flush=True)

    # Compute templates from the first few steps where they're clearly in each basin
    # Loop template: average h_L33 from baseline steps 50-100 (deep in loop)
    loop_templates = torch.stack(traj_baseline[50:min(100, len(traj_baseline))])
    loop_template = loop_templates.mean(dim=0)  # (1, d)

    # Answer template: average h_L33 from correct-answer run steps 50-100
    if len(traj_correct) > 50:
        ans_templates = torch.stack(traj_correct[50:min(100, len(traj_correct))])
        answer_template = ans_templates.mean(dim=0)
    else:
        answer_template = torch.stack(traj_correct[-20:]).mean(dim=0)

    print(f"Loop template from steps 50-100 of baseline")
    print(f"Answer template from steps 50-100 of correct run")
    cos_templates = F.cosine_similarity(loop_template, answer_template, dim=-1).item()
    print(f"cos(loop_template, answer_template) = {cos_templates:.4f}")
    print()

    # Compute commitment(t) for all four trajectories
    results = {}
    for name, traj in [("baseline", traj_baseline),
                        ("deflation_onset0", traj_correct),
                        ("deflation_onset25", traj_25),
                        ("deflation_onset30", traj_30)]:
        cos_loop = []
        cos_answer = []
        commitment = []
        for h in traj:
            cl = F.cosine_similarity(h, loop_template, dim=-1).item()
            ca = F.cosine_similarity(h, answer_template, dim=-1).item()
            cos_loop.append(cl)
            cos_answer.append(ca)
            commitment.append(cl - ca)
        results[name] = {
            "cos_loop": cos_loop,
            "cos_answer": cos_answer,
            "commitment": commitment,
            "n_steps": len(traj)
        }

    # Print commitment trajectory for baseline
    print("=" * 70)
    print("COMMITMENT TRAJECTORY (baseline, no deflation)")
    print("commitment(t) = cos(h_L33, loop) - cos(h_L33, answer)")
    print("positive = loop-committed, negative = answer-committed")
    print("=" * 70)
    bl = results["baseline"]
    for t in range(min(80, len(bl["commitment"]))):
        c = bl["commitment"][t]
        bar = "|" + "#" * int(abs(c) * 50)
        if c > 0:
            print(f"  t={t:3d}  {c:+.4f}  {'':>25}{bar}")
        else:
            print(f"  t={t:3d}  {c:+.4f}  {bar:>25}")

    # Find crossing point
    print("\n" + "=" * 70)
    print("CROSSING ANALYSIS")
    for name in ["baseline", "deflation_onset25", "deflation_onset30"]:
        c = results[name]["commitment"]
        # Find first step where commitment > 0.1 (clearly loop-committed)
        cross_01 = next((t for t, v in enumerate(c) if v > 0.1), -1)
        cross_02 = next((t for t, v in enumerate(c) if v > 0.2), -1)
        # Find first step where it stabilizes above 0.1 for 5 consecutive steps
        stable = -1
        for t in range(len(c) - 5):
            if all(v > 0.1 for v in c[t:t+5]):
                stable = t
                break
        print(f"  {name:25s}: cross>0.1 at t={cross_01}, cross>0.2 at t={cross_02}, stable>0.1 at t={stable}")

    # Compare onset=25 vs onset=30 at token 27
    print("\n" + "=" * 70)
    print("ONSET=25 vs ONSET=30 at commitment point")
    for t in [20, 25, 27, 30, 35, 40]:
        for name in ["baseline", "deflation_onset25", "deflation_onset30"]:
            c = results[name]["commitment"]
            if t < len(c):
                print(f"  t={t:3d}  {name:25s}: commitment={c[t]:+.4f}")
        print()

    # Save
    # Convert to serializable
    save_results = {}
    for name, data in results.items():
        save_results[name] = {
            "cos_loop": [round(x, 6) for x in data["cos_loop"]],
            "cos_answer": [round(x, 6) for x in data["cos_answer"]],
            "commitment": [round(x, 6) for x in data["commitment"]],
            "n_steps": data["n_steps"]
        }
    save_results["cos_templates"] = round(cos_templates, 6)

    with open("output/exp_commitment_trajectory.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n-> output/exp_commitment_trajectory.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
