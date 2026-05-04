"""Extract generation-time trajectories from Qwen2.5-1.5B for Z-projection analysis.

Same 20 problems as the original Qwen-3B experiment, but only zh+en (skip es/ja for speed).
Extract h_final (last hidden layer) at each generation step.

Then run Z-basis projection to test if continuous Z alignment exists at generation time.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import json
import random as pyrandom
import gc
from pathlib import Path

OUTPUT_DIR = Path("output")

# Same problems as original
def generate_problems(n=200, seed=42):
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


def extract_gen_trajectories(model_name, problems, n_problems=20, max_tokens=256, target_layer=None):
    """Generate and extract hidden states at target_layer for each token."""
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    if target_layer is None:
        target_layer = n_layers - 4  # ~L24 for 28-layer model, analogous to L32 for 36-layer

    print(f"Model: {n_layers} layers, d={d}, extracting L{target_layer}")

    # Find the layer module
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise ValueError(f"Can't find layers for {model_name}")

    trajectories = {}

    for prob_idx in range(n_problems):
        prob = problems[prob_idx]
        for lang in ['zh', 'en']:
            prompt = prob[lang]
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            h_list = []
            tok_list = []

            # Register hook on target layer
            captured = {}
            def hook_fn(module, input, output):
                h_out = output if isinstance(output, torch.Tensor) else output[0]
                captured['h'] = h_out.detach()

            handle = layers[target_layer].register_forward_hook(hook_fn)

            # Generate token by token
            cur_ids = input_ids
            for step in range(max_tokens):
                with torch.no_grad():
                    outputs = model(cur_ids, use_cache=False)

                # Get hidden state of LAST token
                h = captured['h'][0, -1, :].cpu().float().numpy()
                h_list.append(h)

                # Get next token
                logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                tok_id = next_token.item()
                tok_list.append(tok_id)

                # Check for EOS
                if tok_id == tokenizer.eos_token_id:
                    break

                cur_ids = torch.cat([cur_ids, next_token], dim=1)

            handle.remove()

            h_array = np.array(h_list, dtype=np.float32)
            tok_array = np.array(tok_list, dtype=np.int64)
            key_h = f"h{target_layer}_prob{prob_idx}_{lang}"
            key_t = f"toks_prob{prob_idx}_{lang}"
            trajectories[key_h] = h_array
            trajectories[key_t] = tok_array

            decoded = tokenizer.decode(tok_list[:20])
            print(f"  prob{prob_idx}_{lang}: {len(h_list)} steps, preview: {decoded[:60]}...")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return trajectories, target_layer, d


def analyze_z_projection(trajectories, target_layer, d, input_pass_npz, model_name):
    """Run Z-basis projection analysis on generation trajectories."""
    short = model_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Z-BASIS PROJECTION ANALYSIS: {short}")
    print(f"{'='*60}")

    # Build Z basis from input-pass data
    data = np.load(input_pass_npz)

    # Use the layer closest to target_layer with highest alignment
    n_layers_input = len([k for k in data.keys() if k.startswith('zh_L')])

    # Use target_layer for Z basis (or closest available)
    z_layer = min(target_layer, n_layers_input - 1)
    zh_input = data[f'zh_L{z_layer}']
    en_input = data[f'en_L{z_layer}']
    combined = np.vstack([zh_input, en_input])

    pca = PCA(n_components=20, random_state=42)
    pca.fit(combined)
    Z_basis = pca.components_  # (20, d)
    print(f"Z basis from input-pass L{z_layer}: PCA=20 explains {sum(pca.explained_variance_ratio_):.3f}")

    # Random control basis
    rng = np.random.RandomState(42)
    rand_mat = rng.randn(20, d).astype(np.float32)
    rand_basis, _ = np.linalg.qr(rand_mat.T)
    rand_basis = rand_basis.T[:20]

    # Analyze each problem
    h_key_prefix = f"h{target_layer}"
    all_cos_full = []
    all_cos_z = []
    all_cos_rand = []

    from numpy.linalg import norm
    def cos_batch(a, b):
        dots = np.sum(a * b, axis=1)
        return dots / (norm(a, axis=1) * norm(b, axis=1) + 1e-10)

    prob_results = []
    for prob_idx in range(20):
        zh_key = f"{h_key_prefix}_prob{prob_idx}_zh"
        en_key = f"{h_key_prefix}_prob{prob_idx}_en"
        if zh_key not in trajectories or en_key not in trajectories:
            continue

        zh_h = trajectories[zh_key]
        en_h = trajectories[en_key]
        T = min(len(zh_h), len(en_h))
        if T < 2:
            continue
        zh_h = zh_h[:T]
        en_h = en_h[:T]

        # Full space
        cos_f = cos_batch(zh_h, en_h)
        # Z projected
        zh_proj = zh_h @ Z_basis.T
        en_proj = en_h @ Z_basis.T
        cos_z = cos_batch(zh_proj, en_proj)
        # Random
        zh_rand = zh_h @ rand_basis.T
        en_rand = en_h @ rand_basis.T
        cos_r = cos_batch(zh_rand, en_rand)

        all_cos_full.extend(cos_f.tolist())
        all_cos_z.extend(cos_z.tolist())
        all_cos_rand.extend(cos_r.tolist())

        prob_results.append({
            "prob": prob_idx, "T": T,
            "full_mean": round(float(cos_f.mean()), 4),
            "z_mean": round(float(cos_z.mean()), 4),
            "rand_mean": round(float(cos_r.mean()), 4)
        })
        print(f"  prob{prob_idx}: T={T:3d}  full={cos_f.mean():.4f}  Z={cos_z.mean():.4f}  rand={cos_r.mean():.4f}")

    all_cos_full = np.array(all_cos_full)
    all_cos_z = np.array(all_cos_z)
    all_cos_rand = np.array(all_cos_rand)

    results = {
        "model": model_name,
        "target_layer": target_layer,
        "z_basis_layer": z_layer,
        "n_problems": len(prob_results),
        "overall": {
            "full_mean": round(float(all_cos_full.mean()), 4),
            "full_std": round(float(all_cos_full.std()), 4),
            "z_mean": round(float(all_cos_z.mean()), 4),
            "z_std": round(float(all_cos_z.std()), 4),
            "rand_mean": round(float(all_cos_rand.mean()), 4),
            "rand_std": round(float(all_cos_rand.std()), 4),
            "z_random_gap": round(float(all_cos_z.mean() - all_cos_rand.mean()), 4),
            "z_full_gap": round(float(all_cos_z.mean() - all_cos_full.mean()), 4),
        },
        "per_problem": prob_results
    }

    print(f"\nOVERALL:")
    print(f"  Full-space: {all_cos_full.mean():.4f} ± {all_cos_full.std():.4f}")
    print(f"  Z-projected: {all_cos_z.mean():.4f} ± {all_cos_z.std():.4f}")
    print(f"  Random 20d: {all_cos_rand.mean():.4f} ± {all_cos_rand.std():.4f}")
    print(f"  Z - Random gap: {all_cos_z.mean() - all_cos_rand.mean():+.4f}")
    print(f"  Z - Full gap: {all_cos_z.mean() - all_cos_full.mean():+.4f}")

    if all_cos_z.mean() - all_cos_rand.mean() > 0.2:
        print(f"\n  *** Z SUBSPACE EXISTS — gap > 0.2 over random ***")
    elif all_cos_z.mean() - all_cos_rand.mean() > 0.05:
        print(f"\n  ** Weak Z signal — gap > 0.05 but < 0.2 **")
    else:
        print(f"\n  No Z signal — gap < 0.05")

    return results


def main():
    problems = generate_problems(200, seed=42)

    # Extract gen trajectories from Qwen2.5-1.5B
    traj_15b, layer_15b, d_15b = extract_gen_trajectories(
        "Qwen/Qwen2.5-1.5B", problems,
        n_problems=20, max_tokens=256, target_layer=24  # L24 of 28, analogous to L32 of 36
    )

    # Save trajectories
    np.savez_compressed(OUTPUT_DIR / "qwen15b_gen_trajectories.npz", **traj_15b)

    # Run Z-projection analysis
    results_15b = analyze_z_projection(
        traj_15b, layer_15b, d_15b,
        OUTPUT_DIR / "qwen15b_all_layers.npz",
        "Qwen/Qwen2.5-1.5B"
    )

    # Save results
    outpath = OUTPUT_DIR / "cross_model_gen_z_results.json"
    with open(outpath, "w") as f:
        json.dump({"Qwen2.5-1.5B": results_15b}, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
