"""Experiment F: MLP Isolation
At L28 and L30: swap PC0 on the residual stream between attention and MLP.
Question: does the MLP's output flip to English when it receives English-addressed input?
Thermostat (reads PC0 → writes consistent) vs Memory (reads KV → fights the swap).
"""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)

# Load hidden states and PC0 vectors
data = np.load("output/all_layers_lasttok.npz")

# Test layers
test_layers = [28, 30]

# Chinese math prompts
zh_prompts = [
    "请计算 2 + 3 × 4 的值。\n",
    "一个矩形的长为8厘米，宽为5厘米，求面积。\n",
    "如果 x + 5 = 12，求 x 的值。\n",
    "计算 100 除以 4 的结果。\n",
    "一个三角形三边长分别为3、4、5，求面积。\n",
]

# English math prompts (same problems)
en_prompts = [
    "Calculate the value of 2 + 3 × 4.\n",
    "A rectangle has length 8 cm and width 5 cm. Find the area.\n",
    "If x + 5 = 12, find the value of x.\n",
    "Calculate the result of 100 divided by 4.\n",
    "A triangle has sides of length 3, 4, and 5. Find the area.\n",
]

results = {}

for target_layer in test_layers:
    print(f"\n{'='*60}")
    print(f"TARGET LAYER: L{target_layer} (gate_z from Exp6)")
    print(f"{'='*60}")

    # Compute PC0 at this layer
    zh_L = data[f"zh_L{target_layer}"]
    en_L = data[f"en_L{target_layer}"]
    all_L = np.concatenate([zh_L, en_L], axis=0)
    mean_L = all_L.mean(axis=0)
    centered = all_L - mean_L
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc0 = Vt[0]

    zh_proj_mean = ((zh_L - mean_L) @ pc0).mean()
    en_proj_mean = ((en_L - mean_L) @ pc0).mean()
    print(f"PC0 projections: zh={zh_proj_mean:.2f}, en={en_proj_mean:.2f}")

    layer_results = {
        "pc0_zh_proj": float(zh_proj_mean),
        "pc0_en_proj": float(en_proj_mean),
        "conditions": {}
    }

    # We need to hook BETWEEN attention and MLP within the target layer
    # Qwen2 decoder layer structure:
    #   residual = hidden_states
    #   hidden_states = self.input_layernorm(hidden_states)
    #   hidden_states = self.self_attn(hidden_states, ...)
    #   hidden_states = residual + hidden_states  # post-attention residual
    #   residual = hidden_states
    #   hidden_states = self.post_attention_layernorm(hidden_states)
    #   hidden_states = self.mlp(hidden_states)
    #   hidden_states = residual + hidden_states  # post-MLP residual

    # Hook the MLP to capture:
    # 1. Its input (post-attention residual, layernormed)
    # 2. Its output (mlp_delta)
    # And to swap PC0 on the post-attention residual

    # Strategy: hook the MLP module itself.
    # Input to MLP = layernormed post-attention residual
    # But we want to swap PC0 on the UN-normed residual stream.
    # The residual (skip connection) is NOT modified by the MLP.
    # So we need to hook the POST_ATTENTION_LAYERNORM's input.

    # Actually, let's hook more carefully:
    # We want to modify the residual BEFORE the layernorm+MLP step.
    # The cleanest way: hook the post_attention_layernorm.

    for condition_name, do_swap in [("natural_zh", False), ("pc0_swapped", True)]:
        print(f"\n  Condition: {condition_name}")

        mlp_inputs = []
        mlp_outputs = []
        residuals_before = []
        residuals_after = []

        def make_pre_hook(do_swap_flag):
            """Hook on post_attention_layernorm to capture/modify its input (= post-attn residual)"""
            def pre_hook(module, input):
                h = input[0]  # (batch, seq, dim)
                last_tok = h[0, -1, :].detach().cpu().float().numpy()
                residuals_before.append(last_tok.copy())

                if do_swap_flag:
                    # Swap PC0 on the residual stream
                    pc0_t = torch.tensor(pc0, dtype=h.dtype, device=h.device)
                    mean_t = torch.tensor(mean_L, dtype=h.dtype, device=h.device)
                    # Current projection
                    current_proj = (h[0, -1, :] - mean_t) @ pc0_t
                    # Target: English mean projection
                    target_proj = torch.tensor(en_proj_mean, dtype=h.dtype, device=h.device)
                    # Swap
                    h[0, -1, :] += (target_proj - current_proj) * pc0_t
                    residuals_after.append(h[0, -1, :].detach().cpu().float().numpy())
                    return (h,)
                else:
                    residuals_after.append(last_tok.copy())
                return None
            return pre_hook

        def mlp_hook(module, input, output):
            """Capture MLP input and output"""
            inp = input[0]  # (batch, seq, dim)
            mlp_inputs.append(inp[0, -1, :].detach().cpu().float().numpy())
            mlp_outputs.append(output[0, -1, :].detach().cpu().float().numpy())

        layer_module = model.model.layers[target_layer]
        h1 = layer_module.post_attention_layernorm.register_forward_pre_hook(
            make_pre_hook(do_swap)
        )
        h2 = layer_module.mlp.register_forward_hook(mlp_hook)

        generations = []
        for i, prompt in enumerate(zh_prompts):
            mlp_inputs.clear()
            mlp_outputs.clear()
            residuals_before.clear()
            residuals_after.clear()

            input_ids = tokenizer.encode(prompt)
            with torch.no_grad():
                outputs = model.generate(
                    torch.tensor([input_ids], device=device),
                    max_new_tokens=64,
                    do_sample=False,
                )
            gen_text = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)

            # Analyze the first-token MLP behavior
            if mlp_outputs:
                mlp_delta = mlp_outputs[0]  # First forward pass MLP output
                mlp_inp = mlp_inputs[0]

                # Project MLP delta onto PC0
                mlp_delta_pc0 = (mlp_delta - mean_L) @ pc0
                mlp_inp_pc0 = (mlp_inp - mean_L) @ pc0

                # Compare to natural baselines
                # Natural Chinese MLP delta PC0 projection
                # (we'd need to compute this separately, but we can compare to the mean)

                generations.append({
                    "prompt_idx": i,
                    "generation": gen_text,
                    "mlp_input_pc0": float(mlp_inp_pc0),
                    "mlp_output_pc0": float(mlp_delta_pc0),
                    "residual_before_pc0": float((residuals_before[0] - mean_L) @ pc0) if residuals_before else None,
                    "residual_after_pc0": float((residuals_after[0] - mean_L) @ pc0) if residuals_after else None,
                })

                if i < 3:
                    print(f"    Problem {i}: {gen_text[:80]}...")
                    if residuals_before:
                        print(f"      Residual PC0: before={((residuals_before[0]-mean_L)@pc0):.2f}, "
                              f"after={((residuals_after[0]-mean_L)@pc0):.2f}")
                    print(f"      MLP input PC0: {mlp_inp_pc0:.2f}")
                    print(f"      MLP output PC0: {mlp_delta_pc0:.2f}")

        h1.remove()
        h2.remove()

        layer_results["conditions"][condition_name] = generations

    # Now run natural English for comparison
    print(f"\n  Condition: natural_en (baseline)")
    mlp_inputs_en = []
    mlp_outputs_en = []

    def mlp_hook_en(module, input, output):
        mlp_inputs_en.append(input[0][0, -1, :].detach().cpu().float().numpy())
        mlp_outputs_en.append(output[0, -1, :].detach().cpu().float().numpy())

    h3 = layer_module.mlp.register_forward_hook(mlp_hook_en)

    en_generations = []
    for i, prompt in enumerate(en_prompts):
        mlp_inputs_en.clear()
        mlp_outputs_en.clear()

        input_ids = tokenizer.encode(prompt)
        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([input_ids], device=device),
                max_new_tokens=64,
                do_sample=False,
            )
        gen_text = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)

        if mlp_outputs_en:
            mlp_delta = mlp_outputs_en[0]
            mlp_inp = mlp_inputs_en[0]
            en_generations.append({
                "prompt_idx": i,
                "generation": gen_text,
                "mlp_input_pc0": float((mlp_inp - mean_L) @ pc0),
                "mlp_output_pc0": float((mlp_delta - mean_L) @ pc0),
            })
            if i < 3:
                print(f"    Problem {i}: {gen_text[:80]}...")
                print(f"      MLP input PC0: {(mlp_inp - mean_L) @ pc0:.2f}")
                print(f"      MLP output PC0: {(mlp_delta - mean_L) @ pc0:.2f}")

    h3.remove()
    layer_results["conditions"]["natural_en"] = en_generations

    # Analysis: compare MLP output PC0 across conditions
    print(f"\n  --- MLP PC0 Analysis at L{target_layer} ---")
    for cond in ["natural_zh", "pc0_swapped", "natural_en"]:
        gens = layer_results["conditions"][cond]
        pc0_outs = [g["mlp_output_pc0"] for g in gens if g.get("mlp_output_pc0") is not None]
        if pc0_outs:
            print(f"    {cond:15s}: MLP output PC0 = {np.mean(pc0_outs):>8.2f} ± {np.std(pc0_outs):.2f}")

    results[f"L{target_layer}"] = layer_results

# Final analysis
print("\n\n" + "="*60)
print("EXPERIMENT F: THERMOSTAT vs MEMORY")
print("="*60)

for L in test_layers:
    lr = results[f"L{L}"]
    zh_outs = [g["mlp_output_pc0"] for g in lr["conditions"]["natural_zh"] if g.get("mlp_output_pc0")]
    swap_outs = [g["mlp_output_pc0"] for g in lr["conditions"]["pc0_swapped"] if g.get("mlp_output_pc0")]
    en_outs = [g["mlp_output_pc0"] for g in lr["conditions"]["natural_en"] if g.get("mlp_output_pc0")]

    zh_mean = np.mean(zh_outs) if zh_outs else 0
    swap_mean = np.mean(swap_outs) if swap_outs else 0
    en_mean = np.mean(en_outs) if en_outs else 0

    print(f"\nL{L}:")
    print(f"  Natural zh MLP output PC0: {zh_mean:.2f}")
    print(f"  PC0-swapped MLP output PC0: {swap_mean:.2f}")
    print(f"  Natural en MLP output PC0: {en_mean:.2f}")

    # Thermostat: swap_mean ≈ en_mean (MLP reads PC0, writes English)
    # Memory: swap_mean ≈ zh_mean (MLP ignores PC0 swap, writes Chinese from KV)
    if en_mean != zh_mean:
        thermostat_frac = (swap_mean - zh_mean) / (en_mean - zh_mean)
        print(f"  Thermostat fraction: {thermostat_frac:.2f}")
        print(f"  (1.0 = pure thermostat, 0.0 = pure memory)")
    else:
        print(f"  Cannot compute (zh == en)")

    # Also check: did the GENERATION language change?
    swap_gens = lr["conditions"]["pc0_swapped"]
    n_english = 0
    for g in swap_gens:
        text = g["generation"]
        n_zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        n_en = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
        if n_en > n_zh:
            n_english += 1
    print(f"  Swapped generations in English: {n_english}/{len(swap_gens)}")

# Save
with open("output/expF_mlp_isolation.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expF_mlp_isolation.json")
