"""
Exp AH: Equivariance Residual — Does the 1D reflection commute with each layer?

THE measurement: if R_v commutes with F_L, the flip is a symmetry.
If not, it's steering.

For each layer L:
  Path A: h_L -> R_v(h_L) -> F_L -> output_A
  Path B: h_L -> F_L -> h_{L+1} -> R_v(h_{L+1}) = output_B

  ε_L = ||output_A - output_B|| / ||h_L||

Also decompose ε into language-direction component vs orthogonal.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SEED = 42
ALL_LAYERS = list(range(36))
STRIP_LAYERS = list(range(9, 27))

# 20 problems — same test set as other experiments
PROBLEMS = [
    {"en": "Calculate 47 + 86.", "zh": "计算 47 + 86 的值。", "answer": "133"},
    {"en": "A rectangle has length 12 and width 5. Find its area.", "zh": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60"},
    {"en": "What is the remainder when 100 is divided by 7?", "zh": "100 除以 7 的余数是多少？", "answer": "2"},
    {"en": "Calculate 15 × 8.", "zh": "计算 15 × 8 的值。", "answer": "120"},
    {"en": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.",
     "zh": "等差数列首项为 2，公差为 3，求前 5 项之和。", "answer": "40"},
    {"en": "Calculate 387 × 29.", "zh": "计算 387 × 29 的值。", "answer": "11223"},
    {"en": "Find the value of C(10, 3).", "zh": "求组合数 C(10, 3) 的值。", "answer": "120"},
    {"en": "What is the remainder when 7654 is divided by 37?", "zh": "7654 除以 37 的余数是多少？", "answer": "34"},
    {"en": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.",
     "zh": "等差数列首项为 7，公差为 11，求前 25 项之和。", "answer": "3475"},
    {"en": "A rectangle has length 47 and width 33. Find its area.", "zh": "一个长方形的长为 47，宽为 33，求其面积。", "answer": "1551"},
    {"en": "Calculate 256 + 789.", "zh": "计算 256 + 789 的值。", "answer": "1045"},
    {"en": "Find the value of C(8, 2).", "zh": "求组合数 C(8, 2) 的值。", "answer": "28"},
    {"en": "What is the remainder when 500 is divided by 13?", "zh": "500 除以 13 的余数是多少？", "answer": "6"},
    {"en": "Calculate 64 × 15.", "zh": "计算 64 × 15 的值。", "answer": "960"},
    {"en": "A rectangle has length 30 and width 18. Find its area.", "zh": "一个长方形的长为 30，宽为 18，求其面积。", "answer": "540"},
    {"en": "An arithmetic sequence has first term 5 and common difference 4. Find the sum of the first 10 terms.",
     "zh": "等差数列首项为 5，公差为 4，求前 10 项之和。", "answer": "230"},
    {"en": "Calculate 123 + 456.", "zh": "计算 123 + 456 的值。", "answer": "579"},
    {"en": "Find the value of C(12, 5).", "zh": "求组合数 C(12, 5) 的值。", "answer": "792"},
    {"en": "What is the remainder when 2023 is divided by 11?", "zh": "2023 除以 11 的余数是多少？", "answer": "0"},
    {"en": "Calculate 73 × 42.", "zh": "计算 73 × 42 的值。", "answer": "3066"},
]


def fit_1d_dirs(model, tokenizer, layers=ALL_LAYERS):
    """Fit ZH-EN mean-difference direction per layer (on RESIDUAL STREAM output)."""
    rng = pyrandom.Random(SEED)
    problems = []
    per_cat = 40
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        problems.append({"zh": f"计算 {a} + {b} 的值。", "en": f"Calculate {a} + {b}."})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20); k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。", "en": f"Find the value of C({n_val}, {k_val})."})
    for _ in range(per_cat):
        a = rng.randint(50, 9999); b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？", "en": f"What is the remainder when {a} is divided by {b}?"})
    for _ in range(per_cat):
        w = rng.randint(2, 50); h = rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。", "en": f"A rectangle has length {w} and width {h}. Find its area."})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20); d_val = rng.randint(1, 10); n_t = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_t} 项之和。",
                         "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_t} terms."})
    rng.shuffle(problems)

    layer_acts = {l: {"zh": [], "en": []} for l in layers}
    layer_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        for lang in ["zh", "en"]:
            for p in tqdm(problems, desc=f"  fit_1d {lang}", leave=False):
                inp = tokenizer(p[lang], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**inp)
                for l in layers:
                    layer_acts[l][lang].append(layer_out[l].copy())
                layer_out.clear()
    finally:
        for h in handles:
            h.remove()

    dirs = {}
    for l in layers:
        zh_m = np.mean(layer_acts[l]["zh"], axis=0)
        en_m = np.mean(layer_acts[l]["en"], axis=0)
        v = zh_m - en_m
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        dirs[l] = v_norm  # numpy float32
    return dirs


def reflect(h, v):
    """Householder reflection: R_v(h) = h - 2(v·h)v"""
    proj = 2.0 * np.dot(h, v) * v
    return h - proj


def measure_equivariance_residual(model, tokenizer, dirs, problems):
    """
    For each layer L and each problem:
      Path A: capture h_L (input to layer L), apply R_v, run through layer L -> output_A
      Path B: run normally through layer L -> h_{L+1}, apply R_v -> output_B
      ε_L = ||output_A - output_B||

    We need to:
    1. Run the model normally, capturing h_L (input) and h_{L+1} (output) at every layer
    2. For each layer L, inject R_v(h_L) as input to just that layer and capture output
    """
    results_per_problem = []

    for pi, prob in enumerate(tqdm(problems, desc="Equivariance residual")):
        prompt = prob["zh"]  # Chinese prompt — the one where flip matters

        # Step 1: Normal forward pass — capture INPUT and OUTPUT of every layer
        layer_inputs = {}   # l -> h_L (input to layer l, last token)
        layer_outputs = {}  # l -> h_{L+1} (output of layer l, last token)

        def make_input_hook(l):
            def hook(module, inp, out):
                # inp is a tuple; first element is the hidden state
                h_in = inp[0] if isinstance(inp, tuple) else inp
                layer_inputs[l] = h_in.detach().clone()  # keep on device for re-injection
                h_out = out[0] if isinstance(out, tuple) else out
                layer_outputs[l] = h_out.detach().cpu().squeeze(0)[-1].float().numpy()
            return hook

        handles = [model.model.layers[l].register_forward_hook(make_input_hook(l)) for l in ALL_LAYERS]

        # We also need the INPUT to each layer. Use forward pre-hook.
        layer_inputs_pre = {}
        def make_pre_hook(l):
            def hook(module, inp):
                h_in = inp[0] if isinstance(inp, tuple) else inp
                layer_inputs_pre[l] = h_in.detach().cpu().squeeze(0)[-1].float().numpy()
            return hook

        pre_handles = [model.model.layers[l].register_forward_pre_hook(make_pre_hook(l)) for l in ALL_LAYERS]

        inp_tok = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inp_tok)

        for h in handles:
            h.remove()
        for h in pre_handles:
            h.remove()

        # Now we have:
        #   layer_inputs_pre[l] = h_L (input to layer l) as numpy
        #   layer_outputs[l] = output of layer l (= input to layer l+1 approximately) as numpy
        # Note: output of layer l includes the residual connection, so it IS h_{L+1}

        # Step 2: For each layer L, compute equivariance residual
        prob_result = {"problem_idx": pi, "prompt": prompt[:50], "layers": {}}

        for l in ALL_LAYERS:
            v = dirs[l]  # unit direction at this layer
            h_L = layer_inputs_pre[l]  # input to layer l
            h_L1 = layer_outputs[l]    # output of layer l (after residual)

            # Path B (easy): compute then flip
            output_B = reflect(h_L1, v)

            # Path A (needs re-run): flip then compute
            # Apply R_v to the input and re-run just this one layer
            h_L_reflected = reflect(h_L, v)

            # We need to run this reflected input through layer L
            # Use a hook that replaces the input
            output_A_container = {}

            def make_inject_hook(target_l, reflected_h):
                reflected_t = torch.tensor(reflected_h, dtype=torch.bfloat16, device=model.device)
                def pre_hook(module, inp):
                    h_in = inp[0] if isinstance(inp, tuple) else inp
                    # Replace ONLY the last token
                    new_h = h_in.clone()
                    new_h[0, -1, :] = reflected_t
                    if isinstance(inp, tuple):
                        return (new_h,) + inp[1:]
                    return (new_h,)
                return pre_hook

            def make_capture_hook(target_l, container):
                def hook(module, inp, out):
                    h_out = out[0] if isinstance(out, tuple) else out
                    container["out"] = h_out.detach().cpu().squeeze(0)[-1].float().numpy()
                return hook

            inject_h = model.model.layers[l].register_forward_pre_hook(
                make_inject_hook(l, h_L_reflected)
            )
            capture_h = model.model.layers[l].register_forward_hook(
                make_capture_hook(l, output_A_container)
            )

            # Run forward pass again (only need this layer, but have to run full model)
            # To avoid contamination from other layers, we only inject+capture at layer l
            with torch.no_grad():
                model(**inp_tok)

            inject_h.remove()
            capture_h.remove()

            output_A = output_A_container["out"]

            # Compute residual
            epsilon = output_A - output_B
            eps_norm = float(np.linalg.norm(epsilon))
            h_norm = float(np.linalg.norm(h_L))
            relative_eps = eps_norm / (h_norm + 1e-8)

            # Decompose epsilon: how much is in lang direction vs orthogonal?
            eps_lang_component = float(np.dot(epsilon, v))
            eps_lang_norm = abs(eps_lang_component)
            eps_ortho_norm = float(np.sqrt(max(0, eps_norm**2 - eps_lang_component**2)))

            # Also: what fraction of epsilon is in lang direction?
            lang_fraction = eps_lang_norm / (eps_norm + 1e-8)

            prob_result["layers"][str(l)] = {
                "eps_norm": round(eps_norm, 4),
                "h_norm": round(h_norm, 4),
                "relative_eps": round(relative_eps, 6),
                "eps_lang_component": round(eps_lang_component, 4),
                "eps_lang_norm": round(eps_lang_norm, 4),
                "eps_ortho_norm": round(eps_ortho_norm, 4),
                "lang_fraction": round(lang_fraction, 4),
            }

        results_per_problem.append(prob_result)
        print(f"  Problem {pi}: done")

    return results_per_problem


def aggregate_results(results):
    """Compute per-layer means across problems."""
    n_layers = 36
    agg = {}
    for l in range(n_layers):
        sl = str(l)
        vals = [r["layers"][sl] for r in results if sl in r["layers"]]
        if not vals:
            continue
        agg[sl] = {
            "mean_relative_eps": round(float(np.mean([v["relative_eps"] for v in vals])), 6),
            "std_relative_eps": round(float(np.std([v["relative_eps"] for v in vals])), 6),
            "mean_eps_norm": round(float(np.mean([v["eps_norm"] for v in vals])), 4),
            "mean_h_norm": round(float(np.mean([v["h_norm"] for v in vals])), 4),
            "mean_lang_fraction": round(float(np.mean([v["lang_fraction"] for v in vals])), 4),
            "mean_eps_lang_norm": round(float(np.mean([v["eps_lang_norm"] for v in vals])), 4),
            "mean_eps_ortho_norm": round(float(np.mean([v["eps_ortho_norm"] for v in vals])), 4),
        }
    return agg


def main():
    print("=== Exp AH: Equivariance Residual ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    print("Fitting 1D language directions at all 36 layers...")
    dirs = fit_1d_dirs(model, tokenizer, layers=ALL_LAYERS)

    print(f"Running equivariance test on {len(PROBLEMS)} problems...")
    results = measure_equivariance_residual(model, tokenizer, dirs, PROBLEMS)

    agg = aggregate_results(results)

    # Print summary
    print("\n=== EQUIVARIANCE RESIDUAL SUMMARY ===")
    print(f"{'Layer':>5} {'ε/‖h‖':>10} {'ε_lang':>10} {'ε_orth':>10} {'lang_frac':>10}")
    print("-" * 50)
    for l in range(36):
        sl = str(l)
        if sl in agg:
            a = agg[sl]
            phase = ""
            if 9 <= l <= 17:
                phase = " [ADV]"
            elif 18 <= l <= 21:
                phase = " [COOP]"
            elif 22 <= l <= 26:
                phase = " [RAMP]"
            print(f"{l:>5} {a['mean_relative_eps']:>10.6f} {a['mean_eps_lang_norm']:>10.4f} "
                  f"{a['mean_eps_ortho_norm']:>10.4f} {a['mean_lang_fraction']:>10.4f}{phase}")

    # Verdict
    adv_eps = [agg[str(l)]["mean_relative_eps"] for l in range(9, 18) if str(l) in agg]
    coop_eps = [agg[str(l)]["mean_relative_eps"] for l in range(18, 27) if str(l) in agg]
    all_eps = [agg[str(l)]["mean_relative_eps"] for l in range(36) if str(l) in agg]

    print(f"\nMean ε/‖h‖ overall:     {np.mean(all_eps):.6f}")
    print(f"Mean ε/‖h‖ adversarial: {np.mean(adv_eps):.6f}")
    print(f"Mean ε/‖h‖ cooperative: {np.mean(coop_eps):.6f}")

    mean_lang_frac = np.mean([agg[str(l)]["mean_lang_fraction"] for l in range(36) if str(l) in agg])
    print(f"Mean lang_fraction:     {mean_lang_frac:.4f}")

    if np.mean(all_eps) < 0.05:
        verdict = "SYMMETRY — reflection approximately commutes with computation"
    elif np.mean(all_eps) < 0.15:
        verdict = "PARTIAL SYMMETRY — approximate commutation with phase-dependent breaking"
    else:
        verdict = "STEERING — reflection does NOT commute, language participates in computation"

    print(f"\nVERDICT: {verdict}")

    output = {
        "experiment": "AH_equivariance_residual",
        "method": "Per-layer equivariance: ||F_L(R_v(h_L)) - R_v(F_L(h_L))|| / ||h_L||",
        "n_problems": len(PROBLEMS),
        "language": "zh",
        "per_problem": results,
        "aggregate": agg,
        "summary": {
            "mean_relative_eps_all": round(float(np.mean(all_eps)), 6),
            "mean_relative_eps_adversarial": round(float(np.mean(adv_eps)), 6),
            "mean_relative_eps_cooperative": round(float(np.mean(coop_eps)), 6),
            "mean_lang_fraction": round(float(mean_lang_frac), 4),
            "verdict": verdict,
        },
    }

    out_path = OUTPUT_DIR / "expAH_equivariance_residual.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
