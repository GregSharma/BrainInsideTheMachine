"""Re-extract activations from ALL 36 layers of Qwen2.5-3B using LAST TOKEN only.

Previous extract_all_layers.py used .mean(axis=0) across token positions, which
caused BOS/position-0 attention sink contamination (dim318=-2858 at pos0 vs ~-2 elsewhere).
This script takes the LAST token's hidden state at each layer instead.

Saves to output/all_layers_lasttok.npz with same key format: zh_L{0..35}, en_L{0..35}, categories.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom
import gc

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_problems(n=200, seed=42):
    """Same as extract_all_layers.py — deterministic, same problems."""
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
    print(f"Model: {n_layers} layers, d={d}")

    problems = generate_problems(200, seed=42)
    categories = np.array([p["category"] for p in problems])
    N = len(problems)

    # Storage
    all_zh = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
    all_en = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}

    # Register hooks on ALL layers
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            # LAST TOKEN ONLY — no mean pooling, no BOS contamination
            layer_outputs[layer_idx] = h_out.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = []
    for l in range(n_layers):
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(h)

    # Extract Chinese
    print(f"Extracting {N} Chinese problems (last token) across {n_layers} layers...")
    for i, prob in enumerate(tqdm(problems, desc="zh")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in range(n_layers):
            all_zh[l][i] = layer_outputs[l]
        layer_outputs.clear()

    # Extract English
    print(f"Extracting {N} English problems (last token) across {n_layers} layers...")
    for i, prob in enumerate(tqdm(problems, desc="en")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in range(n_layers):
            all_en[l][i] = layer_outputs[l]
        layer_outputs.clear()

    # Cleanup
    for h in handles:
        h.remove()

    # Save
    save_dict = {"categories": categories}
    for l in range(n_layers):
        save_dict[f"zh_L{l}"] = all_zh[l]
        save_dict[f"en_L{l}"] = all_en[l]

    outpath = OUTPUT_DIR / "all_layers_lasttok.npz"
    np.savez_compressed(outpath, **save_dict)
    filesize = outpath.stat().st_size / 1e6
    print(f"\nSaved to {outpath} ({filesize:.1f} MB)")
    print(f"Keys: zh_L0..zh_L{n_layers-1}, en_L0..en_L{n_layers-1}, categories")

    # Quick sanity check: dim 318 should NOT dominate
    print("\nSanity check — dim 318 at L15:")
    zh_l15 = all_zh[15]
    print(f"  Mean dim318: {zh_l15[:, 318].mean():.2f} (was ~-179 with mean pooling)")
    print(f"  Std dim318:  {zh_l15[:, 318].std():.2f}")
    print(f"  Mean norm:   {np.linalg.norm(zh_l15, axis=1).mean():.2f}")
    print(f"  Dim318/norm: {np.abs(zh_l15[:, 318].mean()) / np.linalg.norm(zh_l15, axis=1).mean():.4f} (was ~0.74)")

    # Compare variance explained by dim 318
    from sklearn.decomposition import PCA
    combined = np.vstack([all_zh[15], all_en[15]])
    pca = PCA(n_components=5)
    pca.fit(combined)
    print(f"\n  PCA variance explained (L15):")
    for i, v in enumerate(pca.explained_variance_ratio_[:5]):
        # Check if dim 318 is dominant in this component
        d318_loading = np.abs(pca.components_[i, 318])
        print(f"    PC{i}: {v:.3f}  (|loading on dim318|={d318_loading:.3f})")


if __name__ == "__main__":
    main()
