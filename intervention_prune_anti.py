"""Experiment 3: Prune anti-alignment heads — wasted compute test.

We know from attn_vs_ffn_alignment.json that MLP layers ERODE alignment
(total Δz = -15.4). But we don't have per-head anti-alignment data for
all 36 layers — only for 7 target layers.

Strategy: Run a FULL per-head Δz scan across ALL 36 layers (fast version:
just compute dz_added for each head). Then zero ALL negative-Δz heads
simultaneously during inference. Measure:
  (a) accuracy change (first token match vs baseline)
  (b) z-score change (should increase if anti-heads are noise)
  (c) parameter count of pruned heads (% of model)

This tests: is there wasted compute fighting alignment?
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


def main():
    print(f"Loading {MODEL_NAME}...")
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

    # ========== PHASE 1: Full per-head Δz scan (all 36 layers) ==========
    print(f"\n{'='*70}")
    print("PHASE 1: Full per-head alignment scan (36 layers × 16 heads)")
    print(f"{'='*70}")
    print("Extracting per-head contributions for ALL layers...")

    # Hook to capture o_proj inputs and layer inputs
    layer_inputs = {}
    o_proj_inputs = {}

    def make_layer_pre_hook(layer_idx):
        def hook(module, args):
            h_in = args[0] if isinstance(args, tuple) else args
            if isinstance(h_in, torch.Tensor):
                layer_inputs[layer_idx] = h_in.detach().cpu().squeeze(0)[-1].float()
        return hook

    def make_o_proj_pre_hook(layer_idx):
        def hook(module, args):
            h_in = args[0] if isinstance(args, tuple) else args
            if isinstance(h_in, torch.Tensor):
                o_proj_inputs[layer_idx] = h_in.detach().cpu().squeeze(0)[-1].float()
        return hook

    handles = []
    for l in range(n_layers):
        layer = model.model.layers[l]
        h1 = layer.register_forward_pre_hook(make_layer_pre_hook(l))
        h2 = layer.self_attn.o_proj.register_forward_pre_hook(make_o_proj_pre_hook(l))
        handles.append(h1)
        handles.append(h2)

    # Get o_proj weights
    o_proj_weights = {}
    for l in range(n_layers):
        W = model.model.layers[l].self_attn.o_proj.weight.data.float().cpu()  # (d, d)
        o_proj_weights[l] = W

    # Storage: per-head contributions at last token — use subset for scan
    # Use first 50 problems for fast scan, full 200 for final intervention
    N_SCAN = 50
    zh_head_contribs = np.zeros((n_layers, n_heads, N_SCAN, d), dtype=np.float32)
    en_head_contribs = np.zeros((n_layers, n_heads, N_SCAN, d), dtype=np.float32)
    zh_h_pre_scan = np.zeros((n_layers, N_SCAN, d), dtype=np.float32)
    en_h_pre_scan = np.zeros((n_layers, N_SCAN, d), dtype=np.float32)

    print(f"Extracting {N_SCAN} Chinese problems (scan)...")
    for i, prob in enumerate(tqdm(problems[:N_SCAN], desc="zh scan")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in range(n_layers):
            zh_h_pre_scan[l, i] = layer_inputs[l].numpy()
            concat = o_proj_inputs[l]  # (d,)
            W = o_proj_weights[l]
            for h in range(n_heads):
                head_slice = concat[h * d_head: (h + 1) * d_head]
                W_h = W[:, h * d_head: (h + 1) * d_head]
                zh_head_contribs[l, h, i] = (W_h @ head_slice).numpy()
        layer_inputs.clear()
        o_proj_inputs.clear()

    print(f"Extracting {N_SCAN} English problems (scan)...")
    for i, prob in enumerate(tqdm(problems[:N_SCAN], desc="en scan")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in range(n_layers):
            en_h_pre_scan[l, i] = layer_inputs[l].numpy()
            concat = o_proj_inputs[l]
            W = o_proj_weights[l]
            for h in range(n_heads):
                head_slice = concat[h * d_head: (h + 1) * d_head]
                W_h = W[:, h * d_head: (h + 1) * d_head]
                en_head_contribs[l, h, i] = (W_h @ head_slice).numpy()
        layer_inputs.clear()
        o_proj_inputs.clear()

    for h in handles:
        h.remove()

    # Compute Δz for each head at each layer
    print("\nComputing per-head Δz for all 576 heads...")
    head_dz = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        z_pre = matched_vs_scrambled_z(zh_h_pre_scan[l], en_h_pre_scan[l], n_perms=200)
        for h in range(n_heads):
            zh_with = zh_h_pre_scan[l] + zh_head_contribs[l, h]
            en_with = en_h_pre_scan[l] + en_head_contribs[l, h]
            z_with = matched_vs_scrambled_z(zh_with, en_with, n_perms=200)
            head_dz[l, h] = z_with - z_pre

    # Free scan memory
    del zh_head_contribs, en_head_contribs, zh_h_pre_scan, en_h_pre_scan

    # Identify anti-alignment heads (negative Δz)
    anti_heads = {}  # layer -> list of heads with negative dz
    total_anti = 0
    total_pro = 0
    for l in range(n_layers):
        anti = [int(h) for h in range(n_heads) if head_dz[l, h] < -0.05]  # threshold: meaningful negative
        if anti:
            anti_heads[l] = anti
            total_anti += len(anti)
        pro = [int(h) for h in range(n_heads) if head_dz[l, h] > 0.05]
        total_pro += len(pro)

    print(f"\n  Anti-alignment heads (Δz < -0.05): {total_anti} / {n_layers * n_heads} ({total_anti / (n_layers * n_heads):.1%})")
    print(f"  Pro-alignment heads (Δz > +0.05):  {total_pro} / {n_layers * n_heads} ({total_pro / (n_layers * n_heads):.1%})")

    # Print the worst offenders
    print(f"\n  Top 20 anti-alignment heads:")
    flat = [(l, h, head_dz[l, h]) for l in range(n_layers) for h in range(n_heads) if head_dz[l, h] < 0]
    flat.sort(key=lambda x: x[2])
    for l, h, dz in flat[:20]:
        print(f"    L{l:2d} H{h:2d}: Δz = {dz:+.3f}")

    # Parameter count of anti-alignment heads
    # Each head has: q_proj slice (d × d_head) + k_proj slice (for GQA: shared) + v_proj slice + o_proj slice
    # Q: d × d_head per head = 2048 × 128 = 262,144
    # K: shared across 8 heads (GQA=2 KV heads), so per Q-head: 0 (not prunable individually)
    # V: same as K
    # O: d_head × d per head = 128 × 2048 = 262,144
    # So per head prunable: Q + O = 524,288 params
    params_per_head = d * d_head * 2  # Q projection + O projection
    total_model_params = sum(p.numel() for p in model.parameters())
    prunable_params = total_anti * params_per_head

    print(f"\n  Prunable parameters: {prunable_params:,} ({prunable_params / total_model_params:.1%} of model)")
    print(f"  Total model parameters: {total_model_params:,}")

    # ========== PHASE 2: Intervention — prune all anti-alignment heads ==========
    print(f"\n{'='*70}")
    print("PHASE 2: Pruning intervention (zeroing all anti-alignment heads)")
    print(f"{'='*70}")

    # Baseline first tokens
    baseline_zh_tokens = []
    baseline_en_tokens = []
    baseline_zh_hidden = np.zeros((N, d), dtype=np.float32)
    baseline_en_hidden = np.zeros((N, d), dtype=np.float32)

    final_hidden = {}
    def capture_final(module, input, output):
        if isinstance(output, tuple):
            final_hidden['h'] = output[0].detach().cpu().squeeze(0)[-1].float().numpy()
        else:
            final_hidden['h'] = output.detach().cpu().squeeze(0)[-1].float().numpy()

    hook_handle = model.model.norm.register_forward_hook(capture_final)

    print("Collecting baseline...")
    for i, prob in enumerate(tqdm(problems, desc="Baseline zh")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        baseline_zh_tokens.append(int(outputs.logits[0, -1].argmax()))
        baseline_zh_hidden[i] = final_hidden['h']

    for i, prob in enumerate(tqdm(problems, desc="Baseline en")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        baseline_en_tokens.append(int(outputs.logits[0, -1].argmax()))
        baseline_en_hidden[i] = final_hidden['h']

    hook_handle.remove()
    baseline_z = matched_vs_scrambled_z(baseline_zh_hidden, baseline_en_hidden)
    print(f"Baseline z-score: {baseline_z:.1f}")

    # Install pruning hooks
    def make_prune_hook(layer_idx, heads_to_prune):
        def hook(module, args):
            x = args[0]
            x_mod = x.clone()
            for h in heads_to_prune:
                x_mod[:, :, h * d_head: (h + 1) * d_head] = 0.0
            return (x_mod,) + args[1:] if len(args) > 1 else (x_mod,)
        return hook

    prune_hooks = []
    for l, heads in anti_heads.items():
        h = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
            make_prune_hook(l, heads)
        )
        prune_hooks.append(h)

    hook_handle = model.model.norm.register_forward_hook(capture_final)

    pruned_zh_tokens = []
    pruned_en_tokens = []
    pruned_zh_hidden = np.zeros((N, d), dtype=np.float32)
    pruned_en_hidden = np.zeros((N, d), dtype=np.float32)

    print("Running with anti-alignment heads pruned...")
    for i, prob in enumerate(tqdm(problems, desc="Pruned zh")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        pruned_zh_tokens.append(int(outputs.logits[0, -1].argmax()))
        pruned_zh_hidden[i] = final_hidden['h']

    for i, prob in enumerate(tqdm(problems, desc="Pruned en")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        pruned_en_tokens.append(int(outputs.logits[0, -1].argmax()))
        pruned_en_hidden[i] = final_hidden['h']

    hook_handle.remove()
    for h in prune_hooks:
        h.remove()

    pruned_z = matched_vs_scrambled_z(pruned_zh_hidden, pruned_en_hidden)

    # Metrics
    zh_match = sum(1 for a, b in zip(baseline_zh_tokens, pruned_zh_tokens) if a == b) / N
    en_match = sum(1 for a, b in zip(baseline_en_tokens, pruned_en_tokens) if a == b) / N
    cross_base = sum(1 for a, b in zip(baseline_zh_tokens, baseline_en_tokens) if a == b) / N
    cross_pruned = sum(1 for a, b in zip(pruned_zh_tokens, pruned_en_tokens) if a == b) / N

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  z-score: {baseline_z:.1f} → {pruned_z:.1f} (Δz = {pruned_z - baseline_z:+.1f})")
    print(f"  zh token match: {zh_match:.1%}")
    print(f"  en token match: {en_match:.1%}")
    print(f"  Cross-lingual match: {cross_base:.1%} → {cross_pruned:.1%}")
    print(f"  Anti-heads pruned: {total_anti} across {len(anti_heads)} layers")
    print(f"  Prunable params: {prunable_params:,} ({prunable_params / total_model_params:.1%})")

    if pruned_z > baseline_z and zh_match > 0.85:
        print(f"\n  >>> WASTED COMPUTE CONFIRMED: z-score INCREASED and accuracy held.")
        print(f"  >>> {total_anti} heads ({prunable_params / total_model_params:.1%} of params) fight alignment and can be removed.")
    elif zh_match < 0.5:
        print(f"\n  >>> Anti-alignment heads are FUNCTIONALLY NECESSARY despite fighting alignment.")
    else:
        print(f"\n  >>> Mixed result: accuracy={zh_match:.1%}, Δz={pruned_z - baseline_z:+.1f}")

    # Save full results
    results = {
        "model": MODEL_NAME,
        "n_problems": N,
        "n_scan": N_SCAN,
        "baseline_z": baseline_z,
        "pruned_z": pruned_z,
        "dz": pruned_z - baseline_z,
        "zh_token_match": zh_match,
        "en_token_match": en_match,
        "cross_lingual_match_baseline": cross_base,
        "cross_lingual_match_pruned": cross_pruned,
        "total_anti_heads": total_anti,
        "total_pro_heads": total_pro,
        "total_heads": n_layers * n_heads,
        "prunable_params": prunable_params,
        "total_params": total_model_params,
        "prunable_pct": prunable_params / total_model_params,
        "anti_heads_by_layer": {str(k): v for k, v in anti_heads.items()},
        "head_dz_all": {
            f"L{l}": {f"H{h}": float(head_dz[l, h]) for h in range(n_heads)}
            for l in range(n_layers)
        },
        "top20_anti": [
            {"layer": int(l), "head": int(h), "dz": float(dz)}
            for l, h, dz in flat[:20]
        ],
    }

    outpath = OUTPUT_DIR / "intervention_prune_anti.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
