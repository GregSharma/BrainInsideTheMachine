"""Experiment 1: Kill the alignment heads — causal necessity test.

For each target layer, zero out the top-3 pro-alignment attention heads
(by Δz contribution from per_head_alignment.json) during inference.
Then generate the first answer token for 200 math problems in zh and en.

Measures:
  (a) Does the model still produce the same first token? (accuracy match)
  (b) Does the cross-lingual z-score collapse? (alignment necessity)
  (c) Is killing at L0 (where H6 has Δz=+10) more impactful than at L15?

Uses forward hooks on o_proj to zero head slices before projection.
"""

import numpy as np
import torch
import json
import random as pyrandom
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Top-3 alignment heads per layer from per_head_alignment.json
# Format: {layer: [head_indices]} — sorted by dz_added descending
TOP3_HEADS = {
    0: [6, 9, 8],     # dz: +10.0, +7.5, +7.0
    7: [12, 15, 14],   # dz: +0.70, +0.51, +0.40
    8: [2, 0, 1],      # dz: +1.21, +0.39, +0.36
    10: [12, 15, 14],  # will read from JSON
    11: [12, 15, 14],  # will read from JSON
    15: [12, 15, 14],  # will read from JSON
    18: [12, 15, 14],  # will read from JSON
}

def load_top3_from_json():
    """Load actual top-3 heads from saved results."""
    path = OUTPUT_DIR / "per_head_alignment.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        top3 = {}
        for key, val in data.items():
            if key.startswith("L"):
                layer = int(key[1:])
                heads = val["heads"]  # already sorted by dz_added desc
                top3[layer] = [h["head"] for h in heads[:3]]
        return top3
    return TOP3_HEADS

def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh, en = f"计算 {a} + {b} 的值。", f"Calculate {a} + {b}."
        else:
            zh, en = f"计算 {a} × {b} 的值。", f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "category": 0})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "category": 1})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "category": 2})
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "category": 3})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        problems.append({"zh": zh, "en": en, "category": 4})
    rng.shuffle(problems)
    return problems


def matched_vs_scrambled_z(zh, en, n_perms=500):
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    rng = np.random.RandomState(42)
    scrambled = np.array([
        np.mean(np.sum(zh_u * en_u[rng.permutation(len(en_u))], axis=1))
        for _ in range(n_perms)
    ])
    z = (matched - scrambled.mean()) / scrambled.std()
    return float(z)


def get_first_token(model, tokenizer, prompt):
    """Get the model's predicted first output token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # logits shape: (1, seq_len, vocab_size)
    last_logits = outputs.logits[0, -1, :]
    return int(last_logits.argmax())


def main():
    top3 = load_top3_from_json()
    print(f"Loaded top-3 alignment heads for {len(top3)} layers:")
    for l in sorted(top3):
        print(f"  L{l}: heads {top3[l]}")

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_heads = model.config.num_attention_heads  # 16
    d = model.config.hidden_size  # 2048
    d_head = d // n_heads  # 128
    n_layers = model.config.num_hidden_layers  # 36

    problems = generate_problems(200, seed=42)
    N = len(problems)

    # ========== PHASE 1: Baseline (no intervention) ==========
    print(f"\n{'='*70}")
    print("PHASE 1: BASELINE — no intervention")
    print(f"{'='*70}")

    # Collect baseline first tokens AND last-token hidden states
    baseline_zh_tokens = []
    baseline_en_tokens = []
    baseline_zh_hidden = np.zeros((N, d), dtype=np.float32)
    baseline_en_hidden = np.zeros((N, d), dtype=np.float32)

    # Hook to capture final hidden state (before lm_head)
    final_hidden = {}
    def capture_final(module, input, output):
        # model.model (the base transformer) output: (last_hidden_state, ...)
        if isinstance(output, tuple):
            final_hidden['h'] = output[0].detach().cpu().squeeze(0)[-1].float().numpy()
        else:
            final_hidden['h'] = output.detach().cpu().squeeze(0)[-1].float().numpy()

    # Hook on model.model.norm (final RMSNorm, applied after all layers)
    hook_handle = model.model.norm.register_forward_hook(capture_final)

    for i, prob in enumerate(tqdm(problems, desc="Baseline zh")):
        baseline_zh_tokens.append(get_first_token(model, tokenizer, prob["zh"]))
        baseline_zh_hidden[i] = final_hidden['h']

    for i, prob in enumerate(tqdm(problems, desc="Baseline en")):
        baseline_en_tokens.append(get_first_token(model, tokenizer, prob["en"]))
        baseline_en_hidden[i] = final_hidden['h']

    hook_handle.remove()

    baseline_z = matched_vs_scrambled_z(baseline_zh_hidden, baseline_en_hidden)
    print(f"\nBaseline z-score: {baseline_z:.1f}")

    # ========== PHASE 2: Kill heads at each layer ==========
    # Test conditions: kill top-3 at ONE layer at a time
    # Also test: kill top-3 at ALL layers simultaneously

    conditions = []
    for l in sorted(top3):
        conditions.append(("single", l, top3[l]))
    conditions.append(("all", -1, top3))  # all layers simultaneously

    results = {
        "model": MODEL_NAME,
        "n_problems": N,
        "baseline_z": baseline_z,
        "conditions": [],
    }

    for cond_type, target_layer, heads_to_kill in conditions:
        if cond_type == "single":
            label = f"Kill top-3 at L{target_layer} (heads {heads_to_kill})"
            kill_map = {target_layer: heads_to_kill}
        else:
            label = f"Kill top-3 at ALL {len(top3)} layers"
            kill_map = heads_to_kill

        print(f"\n{'='*70}")
        print(f"CONDITION: {label}")
        print(f"{'='*70}")

        # Install kill hooks: zero out specific head slices in o_proj input
        kill_hooks = []

        def make_kill_hook(layer_idx, heads):
            def hook(module, args):
                # o_proj input: (B, seq_len, d)
                x = args[0]
                x_mod = x.clone()
                for h in heads:
                    x_mod[:, :, h * d_head: (h + 1) * d_head] = 0.0
                return (x_mod,) + args[1:] if len(args) > 1 else (x_mod,)
            return hook

        for l, heads in kill_map.items():
            h = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                make_kill_hook(l, heads)
            )
            kill_hooks.append(h)

        # Also capture final hidden states
        killed_zh_hidden = np.zeros((N, d), dtype=np.float32)
        killed_en_hidden = np.zeros((N, d), dtype=np.float32)
        hook_handle = model.model.norm.register_forward_hook(capture_final)

        killed_zh_tokens = []
        killed_en_tokens = []

        for i, prob in enumerate(tqdm(problems, desc=f"zh ({label[:30]})")):
            killed_zh_tokens.append(get_first_token(model, tokenizer, prob["zh"]))
            killed_zh_hidden[i] = final_hidden['h']

        for i, prob in enumerate(tqdm(problems, desc=f"en ({label[:30]})")):
            killed_en_tokens.append(get_first_token(model, tokenizer, prob["en"]))
            killed_en_hidden[i] = final_hidden['h']

        hook_handle.remove()
        for h in kill_hooks:
            h.remove()

        # Metrics
        killed_z = matched_vs_scrambled_z(killed_zh_hidden, killed_en_hidden)
        zh_match = sum(1 for a, b in zip(baseline_zh_tokens, killed_zh_tokens) if a == b) / N
        en_match = sum(1 for a, b in zip(baseline_en_tokens, killed_en_tokens) if a == b) / N

        # Cross-lingual first-token match (zh baseline vs en baseline, and killed)
        cross_match_base = sum(1 for a, b in zip(baseline_zh_tokens, baseline_en_tokens) if a == b) / N
        cross_match_kill = sum(1 for a, b in zip(killed_zh_tokens, killed_en_tokens) if a == b) / N

        print(f"  z-score:  {baseline_z:.1f} → {killed_z:.1f} (Δz = {killed_z - baseline_z:+.1f})")
        print(f"  zh token match (vs baseline): {zh_match:.1%}")
        print(f"  en token match (vs baseline): {en_match:.1%}")
        print(f"  Cross-lingual token match: {cross_match_base:.1%} → {cross_match_kill:.1%}")

        cond_result = {
            "condition": label,
            "type": cond_type,
            "target_layer": target_layer if cond_type == "single" else "all",
            "heads_killed": kill_map if cond_type == "single" else {str(k): v for k, v in kill_map.items()},
            "killed_z": killed_z,
            "dz": killed_z - baseline_z,
            "zh_token_match": zh_match,
            "en_token_match": en_match,
            "cross_lingual_match_baseline": cross_match_base,
            "cross_lingual_match_killed": cross_match_kill,
        }
        results["conditions"].append(cond_result)

    # Save
    outpath = OUTPUT_DIR / "intervention_kill_heads.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
