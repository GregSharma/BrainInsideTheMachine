"""Vein 2: Per-attention-head decomposition at highest-delta layers.

L0, L7, L8, L10, L11 had the biggest attention Δz (+1.5 to +2.2).
Qwen-3B has 16 attention heads per layer.
Which heads drive alignment? Are there "cross-lingual reasoning heads"?

For each head at each target layer:
1. Extract that head's output (its contribution to the attention correction)
2. Compute z-score of the residual stream with only that head's contribution
3. Compare: does 2-3 heads carry most of the alignment, or is it distributed?
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

# Target layers: highest attention Δz from Thread 1
TARGET_LAYERS = [0, 7, 8, 10, 11, 15, 18]


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

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    d_head = d // n_heads
    print(f"Model: {n_layers} layers, d={d}, {n_heads} heads, {n_kv_heads} KV heads, d_head={d_head}")

    # Qwen2.5-3B attention architecture:
    # - 16 Q heads, 2 KV heads (GQA: 8 Q heads per KV head)
    # - self_attn forward: Q,K,V projections → attention → output projection (o_proj)
    # - The o_proj maps from (n_heads * d_head) back to d
    # - We need to capture BEFORE o_proj to get per-head contributions

    # Strategy: Hook the attention module's internal computation.
    # After attention scores are applied to V, we get shape (batch, n_heads, seq_len, d_head).
    # Then o_proj maps this to (batch, seq_len, d).
    # To get per-head contribution: for each head h, zero out all other heads,
    # run through o_proj, and that's head h's contribution.

    # But that requires modifying the forward pass. Instead:
    # The o_proj weight has shape (d, d) = (2048, 2048).
    # It's applied to the concatenated head outputs: [h0 h1 ... h15] of shape (2048,).
    # Head h's contribution = o_proj.weight[:, h*d_head:(h+1)*d_head] @ attn_out_h

    # So we need the pre-o_proj attention output (before projection).
    # This is the value `attn_output` inside self_attn, shape (batch, n_heads, seq_len, d_head).

    problems = generate_problems(200, seed=42)
    N = len(problems)

    # We'll hook the attention module to capture the pre-output-projection activations
    # In Qwen2Attention, the flow is:
    #   q, k, v = proj(hidden_states)
    #   attn_output = scaled_dot_product_attention(q, k, v)  # (B, n_heads, seq_len, d_head)
    #   attn_output = attn_output.transpose(1,2).reshape(B, seq_len, d)  # concat heads
    #   attn_output = self.o_proj(attn_output)  # final projection

    # We'll hook o_proj to capture its INPUT (the concatenated head outputs)
    # and also capture the layer input (h_pre) for computing per-head residual contribution

    layer_inputs = {}  # layer_idx -> (seq_len, d)
    o_proj_inputs = {}  # layer_idx -> (seq_len, d) — concatenated head outputs before o_proj

    def make_layer_pre_hook(layer_idx):
        def hook(module, args):
            h_in = args[0] if isinstance(args, tuple) else args
            if isinstance(h_in, torch.Tensor):
                layer_inputs[layer_idx] = h_in.detach().cpu().squeeze(0).float()
        return hook

    def make_o_proj_pre_hook(layer_idx):
        def hook(module, args):
            # o_proj input is the concatenated attention output: (B, seq_len, d)
            h_in = args[0] if isinstance(args, tuple) else args
            if isinstance(h_in, torch.Tensor):
                o_proj_inputs[layer_idx] = h_in.detach().cpu().squeeze(0).float()
        return hook

    handles = []
    for l in TARGET_LAYERS:
        layer = model.model.layers[l]
        h1 = layer.register_forward_pre_hook(make_layer_pre_hook(l))
        h2 = layer.self_attn.o_proj.register_forward_pre_hook(make_o_proj_pre_hook(l))
        handles.append(h1)
        handles.append(h2)

    # Get o_proj weights for per-head decomposition
    o_proj_weights = {}
    for l in TARGET_LAYERS:
        W = model.model.layers[l].self_attn.o_proj.weight.data.float().cpu()  # (d, d)
        o_proj_weights[l] = W

    # Storage: per-head contribution at last token
    # head_contrib[l][h] = o_proj_weights[l][:, h*d_head:(h+1)*d_head] @ concat_attn[h*d_head:(h+1)*d_head]
    zh_head_contribs = {l: np.zeros((n_heads, N, d), dtype=np.float32) for l in TARGET_LAYERS}
    en_head_contribs = {l: np.zeros((n_heads, N, d), dtype=np.float32) for l in TARGET_LAYERS}
    zh_h_pre = {l: np.zeros((N, d), dtype=np.float32) for l in TARGET_LAYERS}
    en_h_pre = {l: np.zeros((N, d), dtype=np.float32) for l in TARGET_LAYERS}

    print(f"\nExtracting {N} Chinese problems (per-head at layers {TARGET_LAYERS})...")
    for i, prob in enumerate(tqdm(problems, desc="zh")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        for l in TARGET_LAYERS:
            zh_h_pre[l][i] = layer_inputs[l][-1].numpy()  # last token
            concat = o_proj_inputs[l][-1]  # (d,) — last token, concatenated head outputs
            W = o_proj_weights[l]  # (d, d)
            for h in range(n_heads):
                head_slice = concat[h * d_head: (h + 1) * d_head]  # (d_head,)
                W_h = W[:, h * d_head: (h + 1) * d_head]  # (d, d_head)
                zh_head_contribs[l][h][i] = (W_h @ head_slice).numpy()

        layer_inputs.clear()
        o_proj_inputs.clear()

    print(f"Extracting {N} English problems (per-head at layers {TARGET_LAYERS})...")
    for i, prob in enumerate(tqdm(problems, desc="en")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        for l in TARGET_LAYERS:
            en_h_pre[l][i] = layer_inputs[l][-1].numpy()
            concat = o_proj_inputs[l][-1]
            W = o_proj_weights[l]
            for h in range(n_heads):
                head_slice = concat[h * d_head: (h + 1) * d_head]
                W_h = W[:, h * d_head: (h + 1) * d_head]
                en_head_contribs[l][h][i] = (W_h @ head_slice).numpy()

        layer_inputs.clear()
        o_proj_inputs.clear()

    for h in handles:
        h.remove()

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("PER-HEAD ALIGNMENT DECOMPOSITION")
    print("=" * 70)

    results = {
        "model": MODEL_NAME,
        "n_heads": n_heads,
        "d_head": d_head,
        "target_layers": TARGET_LAYERS,
    }

    for l in TARGET_LAYERS:
        print(f"\n--- Layer {l} ({n_heads} heads) ---")

        # First: verify total attention correction matches
        total_attn_zh = sum(zh_head_contribs[l][h] for h in range(n_heads))
        total_attn_en = sum(en_head_contribs[l][h] for h in range(n_heads))

        # z of h_pre
        z_pre = matched_vs_scrambled_z(zh_h_pre[l], en_h_pre[l])

        # z of h_pre + total_attn (should match attn_vs_ffn results)
        zh_post = zh_h_pre[l] + total_attn_zh
        en_post = en_h_pre[l] + total_attn_en
        z_post = matched_vs_scrambled_z(zh_post, en_post)
        dz_total = z_post - z_pre

        print(f"  z_pre={z_pre:.1f}, z_post_attn={z_post:.1f}, Δz_total={dz_total:+.1f}")

        # Per-head: z when adding ONLY that head's contribution
        head_results = []
        for h in range(n_heads):
            zh_with_h = zh_h_pre[l] + zh_head_contribs[l][h]
            en_with_h = en_h_pre[l] + en_head_contribs[l][h]
            z_with_h = matched_vs_scrambled_z(zh_with_h, en_with_h)
            dz_h = z_with_h - z_pre

            # Also: z of head correction ALONE
            z_alone = matched_vs_scrambled_z(zh_head_contribs[l][h], en_head_contribs[l][h])

            # Norm of head correction
            norm_h = np.linalg.norm(zh_head_contribs[l][h], axis=1).mean()

            head_results.append({
                "head": h,
                "dz_added": dz_h,
                "z_alone": z_alone,
                "norm": float(norm_h),
            })

        # Sort by dz contribution
        head_results.sort(key=lambda x: x["dz_added"], reverse=True)

        # Leave-one-out: z when removing each head
        for hr in head_results:
            h = hr["head"]
            zh_without = zh_post - zh_head_contribs[l][h]
            en_without = en_post - en_head_contribs[l][h]
            z_without = matched_vs_scrambled_z(zh_without, en_without)
            hr["dz_leave_one_out"] = z_post - z_without

        print(f"\n  Per-head (sorted by Δz contribution):")
        print(f"  {'Head':>4} | {'Δz(add)':>8} | {'Δz(LOO)':>8} | {'z(alone)':>9} | {'norm':>6}")
        print(f"  {'-'*50}")
        for hr in head_results:
            marker = " ***" if hr["dz_added"] > 0.5 else ""
            print(f"  H{hr['head']:>2}  | {hr['dz_added']:>+7.2f} | {hr['dz_leave_one_out']:>+7.2f} | {hr['z_alone']:>8.1f} | {hr['norm']:>6.1f}{marker}")

        # Summary stats
        top3 = sum(hr["dz_added"] for hr in head_results[:3])
        bottom3 = sum(hr["dz_added"] for hr in head_results[-3:])
        print(f"\n  Top-3 heads contribute: {top3:+.2f} (out of total {dz_total:+.1f})")
        print(f"  Bottom-3 heads contribute: {bottom3:+.2f}")
        print(f"  Concentration ratio (top3/total): {top3/dz_total:.1%}" if dz_total != 0 else "")

        results[f"L{l}"] = {
            "z_pre": z_pre,
            "z_post_attn": z_post,
            "dz_total": dz_total,
            "heads": head_results,
            "top3_dz": top3,
            "concentration_ratio": float(top3 / dz_total) if dz_total != 0 else None,
        }

    # Save
    outpath = OUTPUT_DIR / "per_head_alignment.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
