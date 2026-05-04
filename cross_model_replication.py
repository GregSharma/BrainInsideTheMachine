"""Cross-model replication: Does Z, two-tower, and backbone neuron exist in other models?

Tests Qwen2.5-1.5B (same family, different scale) and phi-2 (different family).
Extracts all layers, runs the full Z-basis projection pipeline, compares to Qwen2.5-3B findings.

Key questions per model:
1. Does a backbone neuron exist (any single dim >40% variance in middle layers)?
2. Does a two-tower Z-writing pattern exist?
3. Does Z exist in ~20 dimensions with continuous alignment?
4. Is there a dim-132 equivalent at the output layer?
"""

import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import random as pyrandom
import gc
import json

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


def extract_all_layers(model_name, problems, device="cuda"):
    """Extract mean-pooled activations from all layers for zh and en."""
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # phi-2 needs padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model: {n_layers} layers, d={d}")

    N = len(problems)
    all_zh = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
    all_en = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}

    # Find decoder layers - handle different architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers  # Qwen, LLaMA
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h  # GPT-2 style (phi-2 uses this? let's check)
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        layers = model.model.decoder.layers  # some models
    else:
        # phi-2 specific
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise ValueError(f"Can't find layers in {model_name}. Dir: {[x for x in dir(model) if not x.startswith('_')]}")

    layer_outputs = {}
    def make_hook(layer_idx):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            layer_outputs[layer_idx] = h_out.detach().cpu().squeeze(0).float().numpy()
        return hook

    handles = []
    for l in range(n_layers):
        h = layers[l].register_forward_hook(make_hook(l))
        handles.append(h)

    # Extract zh
    print(f"Extracting {N} Chinese problems...")
    for i, prob in enumerate(tqdm(problems, desc="zh")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in range(n_layers):
            all_zh[l][i] = layer_outputs[l].mean(axis=0)
        layer_outputs.clear()

    # Extract en
    print(f"Extracting {N} English problems...")
    for i, prob in enumerate(tqdm(problems, desc="en")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in range(n_layers):
            all_en[l][i] = layer_outputs[l].mean(axis=0)
        layer_outputs.clear()

    for h in handles:
        h.remove()

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return all_zh, all_en, n_layers, d


def analyze_model(model_name, all_zh, all_en, n_layers, d, categories):
    """Run the full Z-basis projection pipeline on extracted activations."""
    short_name = model_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"ANALYZING: {short_name} ({n_layers} layers, d={d})")
    print(f"{'='*60}")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_dim": d,
    }

    # ─── 1. Procrustes R² across all layers ───
    print("\n--- Procrustes R² (Ridge) across layers ---")
    r2_by_layer = []
    for l in range(n_layers):
        zh_l = all_zh[l]
        en_l = all_en[l]
        reg = Ridge(alpha=1.0).fit(zh_l, en_l)
        r2 = reg.score(zh_l, en_l)
        r2_by_layer.append(round(r2, 4))

    results["r2_by_layer"] = r2_by_layer

    # Find phase transitions
    max_drop = 0
    max_drop_layer = 0
    max_jump = 0
    max_jump_layer = 0
    for l in range(1, n_layers):
        delta = r2_by_layer[l] - r2_by_layer[l-1]
        if delta < max_drop:
            max_drop = delta
            max_drop_layer = l
        if delta > max_jump:
            max_jump = delta
            max_jump_layer = l

    print(f"  R² range: {min(r2_by_layer):.3f} - {max(r2_by_layer):.3f}")
    print(f"  Biggest drop: L{max_drop_layer-1}→L{max_drop_layer}: {max_drop:+.3f} (R²: {r2_by_layer[max_drop_layer-1]:.3f}→{r2_by_layer[max_drop_layer]:.3f})")
    print(f"  Biggest jump: L{max_jump_layer-1}→L{max_jump_layer}: {max_jump:+.3f} (R²: {r2_by_layer[max_jump_layer-1]:.3f}→{r2_by_layer[max_jump_layer]:.3f})")
    results["shattering_layer"] = max_drop_layer
    results["shattering_drop"] = round(max_drop, 4)
    results["reassembly_layer"] = max_jump_layer
    results["reassembly_jump"] = round(max_jump, 4)

    # ─── 2. PCA variance sweep — backbone neuron detection ───
    print("\n--- PCA variance sweep + backbone neuron detection ---")
    pca20_explained = []
    top_dim_variance = []
    top5_dims_by_layer = []

    for l in range(n_layers):
        combined = np.vstack([all_zh[l], all_en[l]])  # (400, d)
        # Per-dimension variance
        var = np.var(combined, axis=0)
        total_var = var.sum()
        dim_frac = var / total_var

        # Top dims
        sorted_dims = np.argsort(dim_frac)[::-1]
        top5 = sorted_dims[:5].tolist()
        top5_var = [round(float(dim_frac[j]), 4) for j in top5]
        top5_dims_by_layer.append({"dims": top5, "variances": top5_var})
        top_dim_variance.append(round(float(dim_frac[sorted_dims[0]]), 4))

        # PCA=20 explained variance
        pca = PCA(n_components=min(20, combined.shape[0], combined.shape[1]))
        pca.fit(combined)
        explained = float(sum(pca.explained_variance_ratio_))
        pca20_explained.append(round(explained, 4))

    results["pca20_explained"] = pca20_explained
    results["top_dim_variance"] = top_dim_variance
    results["top5_dims_by_layer"] = top5_dims_by_layer

    # Find backbone neuron
    # Look at middle layers (skip first 2 and last 3)
    mid_start = max(2, n_layers // 8)
    mid_end = n_layers - max(3, n_layers // 8)
    mid_top_vars = top_dim_variance[mid_start:mid_end]
    max_mid_var = max(mid_top_vars)
    max_mid_layer = mid_start + mid_top_vars.index(max_mid_var)
    backbone_dim = top5_dims_by_layer[max_mid_layer]["dims"][0]

    print(f"  Backbone neuron candidate: dim {backbone_dim}")
    print(f"  Peak variance: {max_mid_var:.3f} at L{max_mid_layer}")
    print(f"  Qwen reference: dim 318 at 56% variance")
    has_backbone = max_mid_var > 0.20  # threshold: 20% is already dominant
    print(f"  Has backbone (>20%): {has_backbone}")
    results["backbone_dim"] = backbone_dim
    results["backbone_peak_variance"] = round(max_mid_var, 4)
    results["backbone_peak_layer"] = max_mid_layer
    results["has_backbone"] = has_backbone

    # Print backbone dim trajectory
    backbone_trajectory = []
    for l in range(n_layers):
        combined = np.vstack([all_zh[l], all_en[l]])
        var = np.var(combined, axis=0)
        total_var = var.sum()
        frac = float(var[backbone_dim] / total_var) if total_var > 0 else 0
        backbone_trajectory.append(round(frac, 4))
    results["backbone_trajectory"] = backbone_trajectory
    print(f"  Backbone dim {backbone_dim} trajectory:")
    for l in range(n_layers):
        bar = "#" * int(backbone_trajectory[l] * 100)
        print(f"    L{l:2d}: {backbone_trajectory[l]:.3f} {bar}")

    # ─── 3. Z-basis projection at best alignment layer ───
    # Use the layer with highest R² in the second half as "Z layer" (analogous to L32 for Qwen)
    second_half_r2 = r2_by_layer[n_layers//2:]
    z_layer = n_layers//2 + second_half_r2.index(max(second_half_r2))
    print(f"\n--- Z-basis projection (Z-layer = L{z_layer}, R²={r2_by_layer[z_layer]:.3f}) ---")

    zh_z = all_zh[z_layer]
    en_z = all_en[z_layer]
    combined_z = np.vstack([zh_z, en_z])

    pca = PCA(n_components=20)
    pca.fit(combined_z)
    Z_basis = pca.components_  # (20, d)
    z_explained = float(sum(pca.explained_variance_ratio_))
    print(f"  Z basis: PCA=20 explains {z_explained:.3f} of Z-layer variance")
    results["z_layer"] = z_layer
    results["z_explained_variance"] = round(z_explained, 4)

    # Project zh and en onto Z
    zh_proj = zh_z @ Z_basis.T  # (200, 20)
    en_proj = en_z @ Z_basis.T

    # Cosine in full space vs Z-projected vs residual
    from numpy.linalg import norm

    def cosine_batch(a, b):
        """Row-wise cosine similarity."""
        dot = np.sum(a * b, axis=1)
        na = norm(a, axis=1)
        nb = norm(b, axis=1)
        return dot / (na * nb + 1e-10)

    cos_full = cosine_batch(zh_z, en_z)
    cos_z = cosine_batch(zh_proj, en_proj)

    # Residual
    zh_resid = zh_z - zh_proj @ Z_basis
    en_resid = en_z - en_proj @ Z_basis
    cos_resid = cosine_batch(zh_resid, en_resid)

    # Random 20-dim control
    rng = np.random.RandomState(42)
    rand_basis = rng.randn(20, d).astype(np.float32)
    rand_basis, _ = np.linalg.qr(rand_basis.T)
    rand_basis = rand_basis.T[:20]
    zh_rand = zh_z @ rand_basis.T
    en_rand = en_z @ rand_basis.T
    cos_rand = cosine_batch(zh_rand, en_rand)

    print(f"  Full-space cosine: mean={cos_full.mean():.3f}, std={cos_full.std():.3f}")
    print(f"  Z-projected cosine: mean={cos_z.mean():.3f}, std={cos_z.std():.3f}")
    print(f"  Residual cosine: mean={cos_resid.mean():.3f}, std={cos_resid.std():.3f}")
    print(f"  Random 20-dim cosine: mean={cos_rand.mean():.3f}, std={cos_rand.std():.3f}")
    print(f"  Z - Random gap: {cos_z.mean() - cos_rand.mean():.3f}")

    results["cosine_full_mean"] = round(float(cos_full.mean()), 4)
    results["cosine_z_mean"] = round(float(cos_z.mean()), 4)
    results["cosine_resid_mean"] = round(float(cos_resid.mean()), 4)
    results["cosine_random_mean"] = round(float(cos_rand.mean()), 4)
    results["z_random_gap"] = round(float(cos_z.mean() - cos_rand.mean()), 4)

    # ─── 4. Layer delta Z-fraction (Two Towers test) ───
    print(f"\n--- Layer delta Z-fraction (Two Towers test) ---")
    z_fractions = []
    z_norms = []
    nonz_norms = []

    for l in range(n_layers - 1):
        # Delta = L(l+1) - L(l), averaged over all problems and both languages
        delta_zh = all_zh[l+1] - all_zh[l]  # (200, d)
        delta_en = all_en[l+1] - all_en[l]
        delta_all = np.vstack([delta_zh, delta_en])  # (400, d)

        # Project onto Z basis
        delta_z = delta_all @ Z_basis.T @ Z_basis  # (400, d) in Z
        delta_nonz = delta_all - delta_z  # (400, d) in non-Z

        z_norm = float(np.mean(np.linalg.norm(delta_z, axis=1)))
        nonz_norm = float(np.mean(np.linalg.norm(delta_nonz, axis=1)))
        total_norm = z_norm + nonz_norm
        z_frac = z_norm / total_norm if total_norm > 0 else 0

        z_fractions.append(round(z_frac, 3))
        z_norms.append(round(z_norm, 1))
        nonz_norms.append(round(nonz_norm, 1))

    results["z_fractions"] = z_fractions
    results["z_norms"] = z_norms
    results["nonz_norms"] = nonz_norms

    # Print two-tower analysis
    for l in range(n_layers - 1):
        bar_z = "Z" * int(z_fractions[l] * 20)
        bar_n = "." * (20 - len(bar_z))
        label = ""
        if z_fractions[l] > 0.8:
            label = " ← TOWER"
        elif z_fractions[l] < 0.1:
            label = " ← non-Z dominated"
        print(f"  L{l:2d}→{l+1:2d}: Z={z_fractions[l]:.3f} [{bar_z}{bar_n}] Z-norm={z_norms[l]:6.1f} nonZ={nonz_norms[l]:6.1f}{label}")

    # Detect towers (Z > 0.8)
    towers = [l for l in range(len(z_fractions)) if z_fractions[l] > 0.8]
    results["tower_layers"] = towers
    results["has_two_towers"] = len(towers) >= 2
    print(f"\n  Towers (Z > 0.8): {towers}")
    print(f"  Has two-tower pattern: {results['has_two_towers']}")

    # ─── 5. Output layer analysis (dim-132 equivalent?) ───
    print(f"\n--- Output layer analysis (dim-132 equivalent) ---")
    final_layer = n_layers - 1
    combined_final = np.vstack([all_zh[final_layer], all_en[final_layer]])
    var_final = np.var(combined_final, axis=0)
    total_var_final = var_final.sum()
    frac_final = var_final / total_var_final
    top_final = np.argsort(frac_final)[::-1][:5]

    print(f"  Top 5 dims at L{final_layer}: {top_final.tolist()}")
    print(f"  Variances: {[round(float(frac_final[j]), 4) for j in top_final]}")

    # Check if top dim at final layer is a language polarity marker
    top_final_dim = int(top_final[0])
    zh_vals = all_zh[final_layer][:, top_final_dim]
    en_vals = all_en[final_layer][:, top_final_dim]

    corr = float(np.corrcoef(zh_vals, en_vals)[0, 1])
    zh_mean = float(zh_vals.mean())
    en_mean = float(en_vals.mean())
    polarity = "SAME" if np.sign(zh_mean) == np.sign(en_mean) else "OPPOSITE"

    print(f"  Top dim {top_final_dim} at L{final_layer}:")
    print(f"    zh mean: {zh_mean:+.1f}, en mean: {en_mean:+.1f} → {polarity} polarity")
    print(f"    zh-en correlation: {corr:.3f}")

    results["final_layer_top_dim"] = top_final_dim
    results["final_dim_zh_mean"] = round(zh_mean, 2)
    results["final_dim_en_mean"] = round(en_mean, 2)
    results["final_dim_correlation"] = round(corr, 4)
    results["final_dim_polarity"] = polarity

    # Check dim lifecycle: was it absent in middle layers?
    final_dim_trajectory = []
    for l in range(n_layers):
        combined = np.vstack([all_zh[l], all_en[l]])
        v = np.var(combined, axis=0)
        tv = v.sum()
        frac = float(v[top_final_dim] / tv) if tv > 0 else 0
        final_dim_trajectory.append(round(frac, 4))
    results["final_dim_trajectory"] = final_dim_trajectory

    absent_in_middle = max(final_dim_trajectory[mid_start:mid_end]) < 0.01
    results["final_dim_absent_in_middle"] = absent_in_middle
    print(f"  Dim {top_final_dim} absent in middle layers (<1%): {absent_in_middle}")
    if absent_in_middle:
        print(f"  → ANTI-BACKBONE pattern confirmed (like Qwen dim 132)")

    # ─── 6. Phase structure summary ───
    print(f"\n--- Phase structure summary ---")
    # R² profile
    for l in range(n_layers):
        r2 = r2_by_layer[l]
        bar = "#" * int(r2 * 40)
        print(f"  L{l:2d}: R²={r2:.3f} {bar}")

    return results


def main():
    problems = generate_problems(200, seed=42)
    categories = np.array([p["category"] for p in problems])

    all_results = {}

    # ─── Model 1: Qwen2.5-1.5B ───
    zh1, en1, nl1, d1 = extract_all_layers("Qwen/Qwen2.5-1.5B", problems)
    res1 = analyze_model("Qwen/Qwen2.5-1.5B", zh1, en1, nl1, d1, categories)
    all_results["Qwen2.5-1.5B"] = res1

    # Save intermediate
    np.savez_compressed(OUTPUT_DIR / "qwen15b_all_layers.npz",
                        **{f"zh_L{l}": zh1[l] for l in range(nl1)},
                        **{f"en_L{l}": en1[l] for l in range(nl1)},
                        categories=categories)
    del zh1, en1
    gc.collect()

    # ─── Model 2: phi-2 ───
    zh2, en2, nl2, d2 = extract_all_layers("microsoft/phi-2", problems)
    res2 = analyze_model("microsoft/phi-2", zh2, en2, nl2, d2, categories)
    all_results["phi-2"] = res2

    # Save intermediate
    np.savez_compressed(OUTPUT_DIR / "phi2_all_layers.npz",
                        **{f"zh_L{l}": zh2[l] for l in range(nl2)},
                        **{f"en_L{l}": en2[l] for l in range(nl2)},
                        categories=categories)
    del zh2, en2
    gc.collect()

    # ─── Comparison summary ───
    print(f"\n{'='*60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*60}")

    qwen3b = {
        "shattering": "L1→L2 (R² drop 0.491)",
        "reassembly": "L29→L30 (R² jump 0.275)",
        "backbone": "dim 318, 56% variance L2-L29",
        "towers": "L1→L2 (94% Z), L29→L30 (91% Z)",
        "z_cosine": "0.919 continuous",
        "anti_backbone": "dim 132, absent Phase B, 14.4% at L35, opposite polarity"
    }

    for name, res in all_results.items():
        print(f"\n{name}:")
        print(f"  Shattering: L{res.get('shattering_layer', '?')-1}→L{res.get('shattering_layer', '?')} (R² drop {res.get('shattering_drop', '?')})")
        print(f"  Reassembly: L{res.get('reassembly_layer', '?')-1}→L{res.get('reassembly_layer', '?')} (R² jump {res.get('reassembly_jump', '?')})")
        print(f"  Backbone: dim {res.get('backbone_dim', '?')}, {res.get('backbone_peak_variance', '?'):.1%} peak variance")
        print(f"  Has two towers: {res.get('has_two_towers', '?')} — layers {res.get('tower_layers', [])}")
        print(f"  Z-projected cosine: {res.get('cosine_z_mean', '?'):.3f} (gap over random: {res.get('z_random_gap', '?'):.3f})")
        print(f"  Anti-backbone: dim {res.get('final_layer_top_dim', '?')}, polarity={res.get('final_dim_polarity', '?')}, absent middle={res.get('final_dim_absent_in_middle', '?')}")

    print(f"\nQwen2.5-3B (reference):")
    for k, v in qwen3b.items():
        print(f"  {k}: {v}")

    # Save results
    outpath = OUTPUT_DIR / "cross_model_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
