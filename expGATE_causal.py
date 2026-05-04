#!/usr/bin/env python3
"""
expGATE_causal.py — Causal compression test with universal basis

The centroid SVD tells us which DIRECTIONS to try.
This experiment tests which k is CAUSALLY sufficient:
  - Build top-k projection basis from the combined centroid SVD
  - During generation, project the MLP output at last token onto this subspace
    at layers L6-L31 (option b from discussion)
  - Sweep k = [1, 4, 8, 16, 32, 64, 128, 256]
  - Measure math accuracy (20 problems × 2 languages)

If the causal k << 43 (the statistical r90), then the compression procedure works
even though the statistical dimension climbs with diversity.

Also tests: per-problem SVD basis (should match C3's rank-8 result) as a control.
"""

import json
import re
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
RESULTS_FILE = OUTPUT_DIR / "expGATE_causal_results.json"

# Same 20 test problems from C3
LANGS_TEST = ["en", "zh"]
MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_TOKENS = 128
N_LAYERS = 36
D = 2048

# Compression target layers (option b: L6-L31, leave boundaries alone)
COMPRESS_LAYERS = list(range(6, 32))


def get_test_problems():
    """Return the standard 20 math test problems (first 4 per category)."""
    import sys
    sys.path.insert(0, '.')
    from expC2c_crossmodel_readhead import generate_problems, get_test_subset
    all_problems = generate_problems()
    return get_test_subset(all_problems)


def build_centroid_basis(k):
    """Build top-k basis from combined centroid SVD at each compression layer."""
    langs = ['en', 'zh', 'es']
    bases = {}  # layer -> (k, 2048) array

    math_data = np.load(OUTPUT_DIR / "multilingual_all_layers.npz")
    diverse_data = np.load(OUTPUT_DIR / "diverse_all_layers.npz")
    expanded_data = np.load(OUTPUT_DIR / "expanded_all_layers.npz")
    sat_data = np.load(OUTPUT_DIR / "saturation_all_layers.npz")

    for L in COMPRESS_LAYERS:
        centroids = []
        for data in [math_data, diverse_data, expanded_data, sat_data]:
            cent = np.mean([data[f'{lang}_L{L}'] for lang in langs], axis=0)
            centroids.append(cent)
        all_c = np.vstack(centroids)
        all_c -= all_c.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(all_c, full_matrices=False)
        bases[L] = Vt[:k].astype(np.float32)  # (k, 2048)

    return bases


CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)


def build_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{CHAT_SYSTEM}\n\nProblem: {problem_text}\n\nSolution:"


def check_answer(text, correct):
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)


def generate_text(model, tokenizer, prompt_text, device):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def evaluate_with_projection(model, tokenizer, problems, bases, lang, compress_target="mlp"):
    """Generate with projection hooks active. Returns (n_correct, total, details)."""
    device = model.device

    torch_bases = {}
    for L, basis_np in bases.items():
        B = torch.from_numpy(basis_np).to(device=device, dtype=torch.float16)
        torch_bases[L] = B

    hooks = []

    if compress_target == "mlp":
        for L in COMPRESS_LAYERS:
            if L not in torch_bases:
                continue
            B = torch_bases[L]

            def make_hook(layer_idx, basis):
                def hook(module, input, output):
                    last = output[:, -1:, :]
                    proj = last @ basis.T @ basis
                    output = output.clone()
                    output[:, -1:, :] = proj
                    return output
                return hook

            h = model.model.layers[L].mlp.register_forward_hook(make_hook(L, B))
            hooks.append(h)

    try:
        correct = 0
        total = len(problems)

        for prob in problems:
            prompt_text = build_prompt(tokenizer, prob[lang])
            generated = generate_text(model, tokenizer, prompt_text, device)
            if check_answer(generated, prob['answer']):
                correct += 1

    finally:
        for h in hooks:
            h.remove()

    return correct, total


def run_baseline(model, tokenizer, problems, lang):
    """Run without any hooks (baseline)."""
    device = model.device
    correct = 0
    total = len(problems)

    for prob in problems:
        prompt_text = build_prompt(tokenizer, prob[lang])
        generated = generate_text(model, tokenizer, prompt_text, device)
        if check_answer(generated, prob['answer']):
            correct += 1

    return correct, total


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    print("Loading test problems...")
    problems = get_test_problems()
    print(f"  {len(problems)} test problems")

    results = {"conditions": [], "baselines": {}}

    # Baselines
    print("\n=== BASELINES ===")
    for lang in LANGS_TEST:
        c, t = run_baseline(model, tokenizer, problems, lang)
        results["baselines"][lang] = {"correct": c, "total": t}
        print(f"  {lang}: {c}/{t}")

    # Sweep k values with centroid SVD basis
    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print("\n=== MLP PROJECTION SWEEP (centroid SVD basis, L6-L31) ===")
    for k in k_values:
        print(f"\n  k={k}:")
        bases = build_centroid_basis(k)

        condition = {"k": k, "basis": "centroid_svd", "target": "mlp", "layers": "L6-L31"}
        for lang in LANGS_TEST:
            c, t = evaluate_with_projection(
                model, tokenizer, problems, bases, lang, compress_target="mlp"
            )
            condition[f"{lang}_correct"] = c
            condition[f"{lang}_total"] = t
            print(f"    {lang}: {c}/{t}")

        results["conditions"].append(condition)

        # Early stop if we've matched baseline for both languages
        en_base = results["baselines"]["en"]["correct"]
        zh_base = results["baselines"]["zh"]["correct"]
        if condition["en_correct"] >= en_base and condition["zh_correct"] >= zh_base:
            print(f"  -> Matched baseline at k={k}, continuing to verify...")

    # Summary
    print("\n" + "=" * 60)
    print("CAUSAL COMPRESSION SUMMARY")
    print("=" * 60)
    en_base = results["baselines"]["en"]["correct"]
    zh_base = results["baselines"]["zh"]["correct"]
    print(f"Baselines: EN={en_base}/20, ZH={zh_base}/20")
    print(f"\n{'k':>5s}  {'EN':>4s}  {'ZH':>4s}  {'Status'}")
    print("-" * 35)
    for cond in results["conditions"]:
        k = cond["k"]
        en = cond["en_correct"]
        zh = cond["zh_correct"]
        status = "LOSSLESS" if en >= en_base and zh >= zh_base else ""
        print(f"{k:5d}  {en:4d}  {zh:4d}  {status}")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
