"""Re-extract Qwen-1.5B and phi-2 activations using last-token only.

Same methodology as extract_all_layers_lasttok.py but for the other two models.
Uses the same 200 problems (seed=42).
"""

import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom
import gc
import sys

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_problems(n=200, seed=42):
    """Same as extract_all_layers.py."""
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


def extract_model(model_name, output_name):
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model: {n_layers} layers, d={d}")

    problems = generate_problems(200, seed=42)
    categories = np.array([p["category"] for p in problems])
    N = len(problems)

    all_zh = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
    all_en = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}

    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            layer_outputs[layer_idx] = h_out.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = []
    for l in range(n_layers):
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(h)

    for lang_name, lang_key, lang_store in [("Chinese", "zh", all_zh), ("English", "en", all_en)]:
        print(f"Extracting {N} {lang_name} problems (last token)...")
        for i, prob in enumerate(tqdm(problems, desc=lang_key)):
            inputs = tokenizer(prob[lang_key], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            for l in range(n_layers):
                lang_store[l][i] = layer_outputs[l]
            layer_outputs.clear()

    for h in handles:
        h.remove()

    save_dict = {"categories": categories}
    for l in range(n_layers):
        save_dict[f"zh_L{l}"] = all_zh[l]
        save_dict[f"en_L{l}"] = all_en[l]

    outpath = OUTPUT_DIR / output_name
    np.savez_compressed(outpath, **save_dict)
    filesize = outpath.stat().st_size / 1e6
    print(f"Saved to {outpath} ({filesize:.1f} MB)")

    # Quick z-score check at a few layers
    from scipy.linalg import orthogonal_procrustes
    print("\nQuick alignment check:")
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        zh = all_zh[l]
        en = all_en[l]
        zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
        en_u = en / np.linalg.norm(en, axis=1, keepdims=True)

        matched = np.mean(np.sum(zh_u * en_u, axis=1))
        rng = np.random.RandomState(42)
        scr = [np.mean(np.sum(zh_u * en_u[rng.permutation(N)], axis=1)) for _ in range(500)]
        scr = np.array(scr)
        z = (matched - scr.mean()) / scr.std()
        print(f"  L{l:2d}: z={z:.1f}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return n_layers


def main():
    # Qwen-1.5B
    extract_model("Qwen/Qwen2.5-1.5B", "qwen15b_all_layers_lasttok.npz")

    # phi-2
    extract_model("microsoft/phi-2", "phi2_all_layers_lasttok.npz")


if __name__ == "__main__":
    main()
