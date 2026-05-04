"""Exp E: Propagation Tracking — Does a PC0 swap survive downstream layers?

Swap PC0 on the RESIDUAL STREAM at layer L. At every subsequent layer L+1..L35,
measure the PC0 projection. Does it stay swapped or drift back?

Run at L12 and L26 as swap points. This tests whether:
- Skip connections passively carry the swap (projection stays flipped)
- Subsequent MLPs overwrite it (projection drifts back toward original)

Uses cached PC0 vectors per layer + GPU forward pass with hook.
"""

import numpy as np
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import gc

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SWAP_LAYERS = [12, 26]
N_PROBLEMS = 20
N_PCA = 200
ALL_LAYERS = list(range(36))

print("=" * 70)
print("EXP E: PROPAGATION TRACKING")
print("=" * 70)


def generate_problems(n=200, seed=42):
    import random as pyrandom
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


def select_problems(problems, n=20):
    selected = []
    cat_count = {i: 0 for i in range(5)}
    for i, p in enumerate(problems):
        c = p['category']
        if cat_count[c] < n // 5:
            selected.append(i)
            cat_count[c] += 1
        if len(selected) == n:
            break
    return selected


def main():
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    problems = generate_problems(N_PCA, seed=42)
    selected = select_problems(problems, N_PROBLEMS)

    # --- Load cached data for PC0 vectors at all layers ---
    print("\nLoading cached activations for PC0 computation...", flush=True)
    data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")

    # Compute PC0 at every layer
    pc0_per_layer = {}
    en_mean_proj_per_layer = {}
    zh_mean_proj_per_layer = {}
    for layer in ALL_LAYERS:
        zh_l = data[f"zh_L{layer}"]
        en_l = data[f"en_L{layer}"]
        comb = np.vstack([zh_l, en_l])
        pca = PCA(n_components=1)
        pca.fit(comb)
        v = pca.components_[0]
        # Orient: positive = English
        zh_p = (zh_l @ v).mean()
        en_p = (en_l @ v).mean()
        if zh_p > en_p:
            v = -v
            zh_p, en_p = -zh_p, -en_p
        pc0_per_layer[layer] = v
        en_mean_proj_per_layer[layer] = float(en_p)
        zh_mean_proj_per_layer[layer] = float(zh_p)

    print(f"  Computed PC0 at {len(pc0_per_layer)} layers", flush=True)
    del data

    # --- For each swap layer, run the intervention and track ---
    results = {
        "model": MODEL_NAME,
        "swap_layers": SWAP_LAYERS,
        "n_problems": N_PROBLEMS,
        "tracking": {}
    }

    for swap_layer in SWAP_LAYERS:
        print(f"\n{'='*70}", flush=True)
        print(f"SWAP AT LAYER {swap_layer}", flush=True)
        print(f"{'='*70}", flush=True)

        pc0_swap = pc0_per_layer[swap_layer]
        en_target = en_mean_proj_per_layer[swap_layer]

        # For each problem: run zh forward pass with PC0 swap at swap_layer,
        # capture hidden states at ALL subsequent layers
        # Also run clean zh and en baselines for comparison

        # Track: per-layer PC0 projection for swap, clean_zh, clean_en
        swap_projections = np.zeros((N_PROBLEMS, n_layers))
        clean_zh_projections = np.zeros((N_PROBLEMS, n_layers))
        clean_en_projections = np.zeros((N_PROBLEMS, n_layers))

        for pi, idx in enumerate(selected):
            prob = problems[idx]
            if (pi + 1) % 5 == 0:
                print(f"  Problem {pi+1}/{N_PROBLEMS}...", flush=True)

            # --- Clean zh baseline: capture all layers ---
            all_hidden = {}
            handles = []
            for layer in ALL_LAYERS:
                def make_hook(l):
                    def hook(module, input, output):
                        h = output[0] if isinstance(output, tuple) else output
                        all_hidden[l] = h.detach()[:, -1, :].cpu().float().numpy().squeeze()
                    return hook
                h = model.model.layers[layer].register_forward_hook(make_hook(layer))
                handles.append(h)

            inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            for h in handles:
                h.remove()

            for layer in ALL_LAYERS:
                h_vec = all_hidden[layer]
                h_unit = h_vec / np.linalg.norm(h_vec)
                clean_zh_projections[pi, layer] = float(h_unit @ pc0_per_layer[layer])

            # --- Clean en baseline ---
            all_hidden.clear()
            handles = []
            for layer in ALL_LAYERS:
                def make_hook(l):
                    def hook(module, input, output):
                        h = output[0] if isinstance(output, tuple) else output
                        all_hidden[l] = h.detach()[:, -1, :].cpu().float().numpy().squeeze()
                    return hook
                h = model.model.layers[layer].register_forward_hook(make_hook(layer))
                handles.append(h)

            inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            for h in handles:
                h.remove()

            for layer in ALL_LAYERS:
                h_vec = all_hidden[layer]
                h_unit = h_vec / np.linalg.norm(h_vec)
                clean_en_projections[pi, layer] = float(h_unit @ pc0_per_layer[layer])

            # --- Swapped: zh input, PC0 swap at swap_layer, track all downstream ---
            all_hidden.clear()

            def make_swap_hook(swap_l, pc0_v, en_proj_target):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    new_h = h.clone()
                    # Get last token
                    h_last = h[:, -1, :].float()
                    h_norm = torch.norm(h_last, dim=-1, keepdim=True)
                    h_unit = h_last / h_norm
                    pc0_t = torch.tensor(pc0_v, dtype=torch.float32, device=h.device)
                    proj = (h_unit @ pc0_t).unsqueeze(-1)
                    # Swap PC0
                    h_swapped = h_unit - proj * pc0_t + en_proj_target * pc0_t
                    h_swapped = h_swapped * h_norm
                    new_h[:, -1:, :] = h_swapped.to(h.dtype)
                    # Store the swapped vector
                    all_hidden[swap_l] = h_swapped.cpu().numpy().squeeze()
                    if isinstance(output, tuple):
                        return (new_h,) + output[1:]
                    return new_h
                return hook

            handles = []
            # Register swap hook at swap_layer
            h_swap = model.model.layers[swap_layer].register_forward_hook(
                make_swap_hook(swap_layer, pc0_swap, en_target)
            )
            handles.append(h_swap)

            # Register capture hooks at all layers AFTER swap_layer
            for layer in ALL_LAYERS:
                if layer == swap_layer:
                    continue
                def make_capture_hook(l):
                    def hook(module, input, output):
                        h = output[0] if isinstance(output, tuple) else output
                        all_hidden[l] = h.detach()[:, -1, :].cpu().float().numpy().squeeze()
                    return hook
                h = model.model.layers[layer].register_forward_hook(make_capture_hook(layer))
                handles.append(h)

            inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            for h in handles:
                h.remove()

            for layer in ALL_LAYERS:
                if layer in all_hidden:
                    h_vec = all_hidden[layer]
                    h_unit = h_vec / np.linalg.norm(h_vec)
                    swap_projections[pi, layer] = float(h_unit @ pc0_per_layer[layer])

        # --- Compute per-layer stats ---
        layer_stats = {}
        for layer in ALL_LAYERS:
            zh_mean = float(clean_zh_projections[:, layer].mean())
            en_mean = float(clean_en_projections[:, layer].mean())
            swap_mean = float(swap_projections[:, layer].mean())

            # "Flip fraction": how much of the zh→en gap does the swap close?
            gap = en_mean - zh_mean
            if abs(gap) > 1e-6:
                flip_frac = (swap_mean - zh_mean) / gap
            else:
                flip_frac = 0.0

            layer_stats[str(layer)] = {
                "zh_mean": zh_mean,
                "en_mean": en_mean,
                "swap_mean": swap_mean,
                "gap": float(gap),
                "flip_fraction": flip_frac,
                "zh_std": float(clean_zh_projections[:, layer].std()),
                "en_std": float(clean_en_projections[:, layer].std()),
                "swap_std": float(swap_projections[:, layer].std()),
            }

        results["tracking"][str(swap_layer)] = layer_stats

        # Print summary
        print(f"\n  {'Layer':>6s}  {'zh_mean':>8s}  {'en_mean':>8s}  {'swap':>8s}  {'flip%':>7s}")
        print(f"  {'-'*45}")
        for layer in ALL_LAYERS:
            s = layer_stats[str(layer)]
            marker = " <<<" if layer == swap_layer else ""
            print(f"  L{layer:<4d}  {s['zh_mean']:>8.4f}  {s['en_mean']:>8.4f}  "
                  f"{s['swap_mean']:>8.4f}  {s['flip_fraction']:>6.1%}{marker}", flush=True)

    # --- Save ---
    outpath = OUTPUT_DIR / "expE_propagation_tracking.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
