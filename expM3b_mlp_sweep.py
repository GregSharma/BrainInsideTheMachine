"""Experiment M3b: MLP Zeroing Layer Sweep

M3 showed zeroing L9-L26 (18 layers) destroys generation.
How many layers of MLP can we zero before things break?

Sweep: zero MLP at progressively wider ranges centered on the middle.
Compare to baseline. Find the tipping point.

Ranges tested (centered around L17-18):
- L14-L22 (9 layers)
- L12-L24 (13 layers)
- L10-L26 (17 layers)
- L9-L26 (18 layers) — already failed in M3

Also test non-centered:
- L9-L14 (6 layers, early middle)
- L15-L20 (6 layers, mid middle)
- L21-L26 (6 layers, late middle)

5 problems (English), 64 tokens. Quick sweep.
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

MAX_NEW_TOKENS = 64

test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120"},
    {"prompt": "Calculate 15 × 8.", "answer": "120"},
    {"prompt": "What is the remainder when 100 is divided by 7?", "answer": "2"},
]

RANGES = [
    ("L15-L20 (6)", list(range(15, 21))),
    ("L9-L14 (6)", list(range(9, 15))),
    ("L21-L26 (6)", list(range(21, 27))),
    ("L14-L22 (9)", list(range(14, 23))),
    ("L12-L24 (13)", list(range(12, 25))),
    ("L10-L26 (17)", list(range(10, 27))),
    ("L9-L26 (18)", list(range(9, 27))),
]


def run_generation(prompt, zero_mlp_layers=None):
    input_ids = tokenizer.encode(prompt)
    handles = []
    if zero_mlp_layers:
        for li in zero_mlp_layers:
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    return torch.zeros_like(output)
                return hook_fn
            handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]
        for _ in range(MAX_NEW_TOKENS - 1):
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


# Baseline
print("=" * 70)
print("EXPERIMENT M3b: MLP ZEROING LAYER SWEEP")
print("=" * 70)

print("\nBaseline:")
baseline_correct = 0
for prob in test_problems:
    text = run_generation(prob["prompt"])
    correct = prob["answer"] in text
    baseline_correct += correct
    print(f"  {'OK' if correct else 'FAIL'} | {prob['prompt'][:40]} → {text[:60]}")
print(f"  Baseline: {baseline_correct}/5")

results = {"experiment": "M3b: MLP Zeroing Layer Sweep", "baseline_correct": baseline_correct, "sweeps": []}

for range_name, layers in RANGES:
    print(f"\n{'─'*70}")
    print(f"  Zeroing: {range_name}")
    n_correct = 0
    coherent = 0
    for prob in test_problems:
        text = run_generation(prob["prompt"], zero_mlp_layers=layers)
        correct = prob["answer"] in text
        n_correct += correct
        # Check coherence: does it have real words?
        words = text.split()
        real_words = sum(1 for w in words if len(w) > 2 and w.isalpha())
        is_coherent = real_words > 3
        coherent += is_coherent
        print(f"    {'OK' if correct else 'FAIL'} {'COH' if is_coherent else 'GIB'} | {text[:70]}")

    print(f"  → Correct: {n_correct}/5, Coherent: {coherent}/5")
    results["sweeps"].append({
        "range": range_name,
        "layers": layers,
        "n_layers": len(layers),
        "correct": n_correct,
        "coherent": coherent,
    })

print(f"\n{'='*70}")
print("M3b SWEEP SUMMARY")
print("=" * 70)
print(f"  {'Range':<20s} {'Layers':>6s} {'Correct':>8s} {'Coherent':>9s}")
print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*9}")
print(f"  {'Baseline':<20s} {'0':>6s} {f'{baseline_correct}/5':>8s} {'5/5':>9s}")
for s in results["sweeps"]:
    c = s['correct']
    co = s['coherent']
    print(f"  {s['range']:<20s} {s['n_layers']:>6d} {c:>5d}/5   {co:>5d}/5")

with open("output/expM3b_mlp_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to output/expM3b_mlp_sweep.json")
