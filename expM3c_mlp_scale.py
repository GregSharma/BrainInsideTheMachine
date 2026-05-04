"""Experiment M3c: MLP SCALING instead of zeroing

M3 showed zeroing MLP at L9-L26 destroys generation. But what if the MLP
contribution is partially redundant? Instead of zeroing, SCALE the MLP delta
by a factor < 1.

Test: scale MLP deltas at L9-L26 by 0.0, 0.25, 0.5, 0.75, 1.0 (baseline).
This finds the minimum MLP contribution needed to maintain coherent generation.

Also test the SWAP interpretation: if N showed language follows MLP,
what happens when we scale only the language COMPONENT of MLP (projection onto
language PCs)?

5 problems, English. 64 tokens.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 64
SCALE_LAYERS = list(range(9, 27))
d = model.config.hidden_size

test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120"},
    {"prompt": "Calculate 387 × 29.", "answer": "11223"},
    {"prompt": "An arithmetic sequence has first term 3 and common difference 7. Find the sum of the first 20 terms.", "answer": "1390"},
]


def run_generation(prompt, scale_factor=1.0, scale_layers=None):
    input_ids = tokenizer.encode(prompt)
    handles = []
    if scale_layers and scale_factor != 1.0:
        for li in scale_layers:
            def make_hook(sf):
                def hook_fn(module, input, output):
                    return output * sf
                return hook_fn
            handles.append(model.model.layers[li].mlp.register_forward_hook(make_hook(scale_factor)))

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


print("=" * 70)
print("EXPERIMENT M3c: MLP SCALING AT L9-L26")
print("=" * 70)

SCALES = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
results = {"experiment": "M3c: MLP Scaling L9-L26", "scales": []}

for sf in SCALES:
    label = f"scale={sf:.2f}"
    print(f"\n{'─'*70}")
    print(f"  {label}")
    n_correct = 0
    coherent = 0
    texts = []
    for prob in test_problems:
        text = run_generation(prob["prompt"], scale_factor=sf, scale_layers=SCALE_LAYERS)
        correct = prob["answer"] in text
        n_correct += correct
        words = text.split()
        real_words = sum(1 for w in words if len(w) > 2 and w.isalpha())
        is_coherent = real_words > 3
        coherent += is_coherent
        texts.append(text[:80])
        print(f"    {'OK' if correct else 'FAIL'} {'COH' if is_coherent else 'GIB'} | {text[:70]}")

    print(f"  → Correct: {n_correct}/5, Coherent: {coherent}/5")
    results["scales"].append({
        "scale_factor": sf,
        "correct": n_correct,
        "coherent": coherent,
        "texts": texts,
    })

print(f"\n{'='*70}")
print("M3c SCALING SUMMARY")
print("=" * 70)
print(f"  {'Scale':<10s} {'Correct':>8s} {'Coherent':>9s}")
print(f"  {'─'*10} {'─'*8} {'─'*9}")
for s in results["scales"]:
    print(f"  {s['scale_factor']:<10.2f} {s['correct']:>5d}/5   {s['coherent']:>5d}/5")

with open("output/expM3c_mlp_scale.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to output/expM3c_mlp_scale.json")
