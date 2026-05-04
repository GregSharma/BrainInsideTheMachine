"""Thread 1: Attention vs FFN decomposition of alignment accumulation.

For each transformer layer, the residual stream update is:
  h_post_attn = h_pre + attn_correction(LayerNorm(h_pre))
  h_post_mlp  = h_post_attn + mlp_correction(LayerNorm(h_post_attn))

This script hooks into each sublayer to extract:
  - h_pre (input to layer)
  - attn_delta = h_post_attn - h_pre
  - mlp_delta = h_post_mlp - h_post_attn

Then computes how each delta contributes to the matched-vs-scrambled z-score
of the cumulative residual stream.

Key question: does cross-lingual alignment come from attention (cross-token mixing)
or MLP (within-token transformation)?
"""

import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom
import json

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_problems(n=200, seed=42):
    """Same deterministic problems as all other scripts."""
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh = f"计算 {a} + {b} 的值。"
            en = f"Calculate {a} + {b}."
        else:
            zh = f"计算 {a} × {b} 的值。"
            en = f"Calculate {a} × {b}."
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


def matched_vs_scrambled_z(zh, en, n_perms=1000):
    """Unit-normalized matched vs scrambled z-score."""
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    rng = np.random.RandomState(42)
    scrambled = np.array([
        np.mean(np.sum(zh_u * en_u[rng.permutation(len(en_u))], axis=1))
        for _ in range(n_perms)
    ])
    z = (matched - scrambled.mean()) / scrambled.std()
    return float(z), float(matched), float(scrambled.mean()), float(scrambled.std())


def cosine_gap(zh, en):
    """Mean matched cosine minus mean scrambled cosine (single permutation for speed)."""
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    perm = np.random.RandomState(42).permutation(len(en_u))
    scrambled = np.mean(np.sum(zh_u * en_u[perm], axis=1))
    return float(matched - scrambled)


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers  # 36
    d = model.config.hidden_size  # 2048
    print(f"Model: {n_layers} layers, d={d}")

    # Inspect the layer structure
    layer0 = model.model.layers[0]
    print(f"Layer structure: {type(layer0).__name__}")
    print(f"  Sublayers: {[n for n, _ in layer0.named_children()]}")

    problems = generate_problems(200, seed=42)
    N = len(problems)

    # We need three things per layer per problem:
    # 1. h_pre (input to the layer — output of previous layer)
    # 2. attn output (before residual add) — need to get from self_attn hook
    # 3. mlp output (before residual add) — need to get from mlp hook

    # Qwen2.5 architecture (Qwen2DecoderLayer):
    #   residual = hidden_states
    #   hidden_states = self.input_layernorm(hidden_states)
    #   hidden_states = self.self_attn(hidden_states, ...)
    #   hidden_states = residual + hidden_states       <-- post-attn
    #   residual = hidden_states
    #   hidden_states = self.post_attention_layernorm(hidden_states)
    #   hidden_states = self.mlp(hidden_states)
    #   hidden_states = residual + hidden_states       <-- post-mlp

    # So self_attn output IS the attention correction (before residual add)
    # And mlp output IS the MLP correction (before residual add)

    # Hook storage
    attn_outputs = {}  # layer_idx -> last_token attn correction
    mlp_outputs = {}   # layer_idx -> last_token mlp correction
    layer_inputs = {}  # layer_idx -> last_token input (h_pre)

    def make_layer_input_hook(layer_idx):
        def hook(module, args, kwargs=None):
            # The first positional argument to the layer is hidden_states
            h_in = args[0] if isinstance(args, tuple) else args
            if isinstance(h_in, torch.Tensor):
                layer_inputs[layer_idx] = h_in.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # self_attn returns (attn_output, ...) or just attn_output
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            attn_outputs[layer_idx] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    def make_mlp_hook(layer_idx):
        def hook(module, input, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            mlp_outputs[layer_idx] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    # Register hooks
    handles = []
    for l in range(n_layers):
        layer = model.model.layers[l]
        # Forward pre-hook on the layer to capture input
        h = layer.register_forward_pre_hook(make_layer_input_hook(l))
        handles.append(h)
        # Forward hook on self_attn to capture attention correction
        h = layer.self_attn.register_forward_hook(make_attn_hook(l))
        handles.append(h)
        # Forward hook on mlp to capture MLP correction
        h = layer.mlp.register_forward_hook(make_mlp_hook(l))
        handles.append(h)

    # Storage: per-layer attn_delta and mlp_delta for all problems
    zh_attn_delta = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
    zh_mlp_delta = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
    zh_h_pre = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}

    en_attn_delta = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
    en_mlp_delta = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
    en_h_pre = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}

    # Extract Chinese
    print(f"\nExtracting {N} Chinese problems...")
    for i, prob in enumerate(tqdm(problems, desc="zh")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in range(n_layers):
            zh_h_pre[l][i] = layer_inputs[l]
            zh_attn_delta[l][i] = attn_outputs[l]
            zh_mlp_delta[l][i] = mlp_outputs[l]
        layer_inputs.clear()
        attn_outputs.clear()
        mlp_outputs.clear()

    # Extract English
    print(f"Extracting {N} English problems...")
    for i, prob in enumerate(tqdm(problems, desc="en")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in range(n_layers):
            en_h_pre[l][i] = layer_inputs[l]
            en_attn_delta[l][i] = attn_outputs[l]
            en_mlp_delta[l][i] = mlp_outputs[l]
        layer_inputs.clear()
        attn_outputs.clear()
        mlp_outputs.clear()

    # Cleanup hooks
    for h in handles:
        h.remove()

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("ANALYSIS: Attention vs FFN Contribution to Cross-Lingual Alignment")
    print("=" * 70)

    results = {
        "model": MODEL_NAME,
        "n_problems": N,
        "n_layers": n_layers,
        "hidden_size": d,
        "methodology": "Unit-normalized matched-vs-scrambled z-score on cumulative residual stream",
    }

    layer_results = []

    for l in range(n_layers):
        # Reconstruct cumulative residual stream at each stage:
        # h_pre[l] is input to layer l (= output of layer l-1, or embedding output for l=0)
        # h_post_attn[l] = h_pre[l] + attn_delta[l]
        # h_post_mlp[l] = h_post_attn[l] + mlp_delta[l] = h_pre[l] + attn_delta[l] + mlp_delta[l]

        # z-score of the full residual at each stage
        z_pre, m_pre, s_pre, std_pre = matched_vs_scrambled_z(zh_h_pre[l], en_h_pre[l], n_perms=500)

        zh_post_attn = zh_h_pre[l] + zh_attn_delta[l]
        en_post_attn = en_h_pre[l] + en_attn_delta[l]
        z_post_attn, m_pa, s_pa, std_pa = matched_vs_scrambled_z(zh_post_attn, en_post_attn, n_perms=500)

        zh_post_mlp = zh_post_attn + zh_mlp_delta[l]
        en_post_mlp = en_post_attn + en_mlp_delta[l]
        z_post_mlp, m_pm, s_pm, std_pm = matched_vs_scrambled_z(zh_post_mlp, en_post_mlp, n_perms=500)

        # Delta z attributable to each sublayer
        dz_attn = z_post_attn - z_pre
        dz_mlp = z_post_mlp - z_post_attn

        # Also compute z-score on the corrections ALONE (are they cross-lingually aligned?)
        z_attn_alone, _, _, _ = matched_vs_scrambled_z(zh_attn_delta[l], en_attn_delta[l], n_perms=500)
        z_mlp_alone, _, _, _ = matched_vs_scrambled_z(zh_mlp_delta[l], en_mlp_delta[l], n_perms=500)

        # Norm of corrections
        attn_norm_zh = np.linalg.norm(zh_attn_delta[l], axis=1).mean()
        mlp_norm_zh = np.linalg.norm(zh_mlp_delta[l], axis=1).mean()
        pre_norm_zh = np.linalg.norm(zh_h_pre[l], axis=1).mean()

        entry = {
            "layer": l,
            "z_pre": z_pre,
            "z_post_attn": z_post_attn,
            "z_post_mlp": z_post_mlp,
            "dz_attn": dz_attn,
            "dz_mlp": dz_mlp,
            "z_attn_correction_alone": z_attn_alone,
            "z_mlp_correction_alone": z_mlp_alone,
            "norm_h_pre": float(pre_norm_zh),
            "norm_attn_delta": float(attn_norm_zh),
            "norm_mlp_delta": float(mlp_norm_zh),
            "matched_cos_pre": m_pre,
            "matched_cos_post_attn": m_pa,
            "matched_cos_post_mlp": m_pm,
        }
        layer_results.append(entry)

        marker = ""
        if abs(dz_attn) > 1.5:
            marker += " *** ATTN"
        if abs(dz_mlp) > 1.5:
            marker += " *** MLP"

        print(f"  L{l:2d}: z_pre={z_pre:5.1f} → +attn={dz_attn:+5.1f} → +mlp={dz_mlp:+5.1f} → z_post={z_post_mlp:5.1f}"
              f"  | attn_alone={z_attn_alone:5.1f} mlp_alone={z_mlp_alone:5.1f}"
              f"  | norms: pre={pre_norm_zh:.0f} attn={attn_norm_zh:.0f} mlp={mlp_norm_zh:.0f}{marker}")

    results["layer_results"] = layer_results

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_dz_attn = sum(r["dz_attn"] for r in layer_results)
    total_dz_mlp = sum(r["dz_mlp"] for r in layer_results)

    print(f"\n  Total Δz from attention: {total_dz_attn:+.1f}")
    print(f"  Total Δz from MLP:       {total_dz_mlp:+.1f}")
    print(f"  Ratio attn/mlp:          {total_dz_attn/total_dz_mlp:.2f}" if total_dz_mlp != 0 else "  MLP Δz is zero")

    # Identify phases
    attn_dominant = sum(1 for r in layer_results if r["dz_attn"] > r["dz_mlp"] and r["dz_attn"] > 0.5)
    mlp_dominant = sum(1 for r in layer_results if r["dz_mlp"] > r["dz_attn"] and r["dz_mlp"] > 0.5)
    print(f"\n  Layers where attention dominates: {attn_dominant}/{n_layers}")
    print(f"  Layers where MLP dominates:       {mlp_dominant}/{n_layers}")

    # Which corrections are themselves cross-lingually aligned?
    attn_aligned = sum(1 for r in layer_results if r["z_attn_correction_alone"] > 3.0)
    mlp_aligned = sum(1 for r in layer_results if r["z_mlp_correction_alone"] > 3.0)
    print(f"\n  Layers with cross-lingual attn corrections (z>3): {attn_aligned}/{n_layers}")
    print(f"  Layers with cross-lingual MLP corrections (z>3):  {mlp_aligned}/{n_layers}")

    results["summary"] = {
        "total_dz_attn": total_dz_attn,
        "total_dz_mlp": total_dz_mlp,
        "attn_dominant_layers": attn_dominant,
        "mlp_dominant_layers": mlp_dominant,
        "attn_aligned_layers": attn_aligned,
        "mlp_aligned_layers": mlp_aligned,
    }

    # Save
    outpath = OUTPUT_DIR / "attn_vs_ffn_alignment.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Also save raw deltas for further analysis
    save_dict = {}
    for l in range(n_layers):
        save_dict[f"zh_attn_delta_L{l}"] = zh_attn_delta[l]
        save_dict[f"zh_mlp_delta_L{l}"] = zh_mlp_delta[l]
        save_dict[f"en_attn_delta_L{l}"] = en_attn_delta[l]
        save_dict[f"en_mlp_delta_L{l}"] = en_mlp_delta[l]
    npz_path = OUTPUT_DIR / "attn_mlp_deltas.npz"
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved raw deltas to {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
