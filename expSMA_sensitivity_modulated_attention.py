#!/usr/bin/env python3
"""
expSMA_sensitivity_modulated_attention.py
Sensitivity-Modulated Attention: giving the model gradient proprioception.

At each MLP layer, the SiLU gate has neurons with derivative g = σ(x) + x·σ(x)·(1-σ(x))
that indicates decision confidence. This derivative is NOT recoverable from the activation
value (SiLU is not globally invertible) — information computed and discarded every layer.

The experiment: make this sensitivity visible to the model's own attention by modulating
queries with the per-dimension sensitivity. The model then SEARCHES for information in
the directions where it's most fragile.

Parts:
  0. Diagnostic: measure sensitivity profile across layers/problems
  1. Baseline: standard greedy generation
  2. Modulated: sensitivity-modulated attention queries (intervention)
  3. Controls: random, uniform, inverse modulation
  4. Analysis: per-problem comparison + tension correlation
"""

import json, re, time, sys, os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_NEW = 128
DEVICE = "cuda"

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)


def get_test_problems():
    categories = {
        "algebra": [
            {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22", "answer": "5"},
            {"en": "Solve for x: 2x² - 8 = 0", "zh": "求解x：2x² - 8 = 0", "answer": "2"},
            {"en": "Simplify: (x + 3)(x - 3)", "zh": "化简：(x + 3)(x - 3)", "answer": "x² - 9"},
            {"en": "Solve: |2x - 5| = 3", "zh": "求解：|2x - 5| = 3", "answer": "4"},
        ],
        "arithmetic": [
            {"en": "Calculate: 347 + 658", "zh": "计算：347 + 658", "answer": "1005"},
            {"en": "Calculate: 1000 - 387", "zh": "计算：1000 - 387", "answer": "613"},
            {"en": "Calculate: 23 × 17", "zh": "计算：23 × 17", "answer": "391"},
            {"en": "Calculate: 1728 ÷ 12", "zh": "计算：1728 ÷ 12", "answer": "144"},
        ],
        "geometry": [
            {"en": "Find the area of a circle with radius 7 (use π ≈ 22/7)", "zh": "求半径为7的圆的面积（使用 π ≈ 22/7）", "answer": "154"},
            {"en": "Find the hypotenuse of a right triangle with legs 5 and 12", "zh": "求直角三角形两直角边为5和12时的斜边长", "answer": "13"},
            {"en": "What is the perimeter of a rectangle with length 15 and width 8?", "zh": "长为15宽为8的矩形的周长是多少？", "answer": "46"},
            {"en": "Find the volume of a cube with side length 6", "zh": "求边长为6的正方体的体积", "answer": "216"},
        ],
        "number_theory": [
            {"en": "What is the GCD of 84 and 120?", "zh": "84和120的最大公约数是多少？", "answer": "12"},
            {"en": "Is 97 prime? Answer yes or no, then explain.", "zh": "97是质数吗？回答是或否，然后解释。", "answer": "yes"},
            {"en": "Find the remainder when 2^10 is divided by 7", "zh": "求2^10除以7的余数", "answer": "2"},
            {"en": "What is the sum of all prime numbers less than 20?", "zh": "所有小于20的质数之和是多少？", "answer": "77"},
        ],
        "combinatorics": [
            {"en": "How many ways can you choose 3 items from 7?", "zh": "从7个物品中选3个有多少种方式？", "answer": "35"},
            {"en": "How many ways can 5 people stand in a line?", "zh": "5个人站成一排有多少种方式？", "answer": "120"},
            {"en": "Calculate: 8! / (5! × 3!)", "zh": "计算：8! / (5! × 3!)", "answer": "56"},
            {"en": "How many 3-digit numbers have all distinct digits?", "zh": "有多少个三位数的各位数字互不相同？", "answer": "648"},
        ],
    }
    problems = []
    for cat_name, cat_probs in categories.items():
        for p in cat_probs:
            p["category"] = cat_name
            problems.append(p)
    return problems


def build_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{CHAT_SYSTEM}\n\nProblem: {problem_text}\n\nSolution:"


def check_answer(text, correct):
    if correct in ("yes", "no"):
        return correct.lower() in text.lower()
    if correct == "x² - 9":
        return "x² - 9" in text or "x^2 - 9" in text or "x**2 - 9" in text
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ── SiLU derivative (analytical, no autograd) ────────────────

def silu_derivative(x):
    """g(x) = σ(x) + x·σ(x)·(1 - σ(x)).  Elementwise, analytical."""
    sig = torch.sigmoid(x)
    return sig + x * sig * (1.0 - sig)


# ── Part 0: Diagnostic ───────────────────────────────────────

@torch.inference_mode()
def run_diagnostic(model, tokenizer):
    """Measure sensitivity profile across all layers and problems.
    Uses hooks on MLP to capture gate pre-activations during normal forward."""
    print("\n" + "=" * 60)
    print("PART 0: SENSITIVITY DIAGNOSTIC")
    print("=" * 60)
    n_layers = len(model.model.layers)
    results = []

    # Hooks to capture gate pre-activations
    gate_outputs = {}
    hooks = []

    for ell in range(n_layers):
        def make_hook(layer_idx):
            def hook(module, input, output):
                h_in = input[0]  # MLP input (after post_attn_layernorm)
                with torch.no_grad():
                    x_gate = module.gate_proj(h_in)
                    gate_outputs[layer_idx] = x_gate.detach()
            return hook
        h = model.model.layers[ell].mlp.register_forward_hook(make_hook(ell))
        hooks.append(h)

    for pidx, prob in enumerate(get_test_problems()):
        for lang in ("en", "zh"):
            prompt = build_prompt(tokenizer, prob[lang])
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

            gate_outputs.clear()
            model(ids)  # single forward pass, hooks capture everything

            layer_stats = []
            for ell in range(n_layers):
                x_gate = gate_outputs[ell]  # (1, seq_len, d_ff)
                g = silu_derivative(x_gate)
                tau = g * (1.0 - g)  # Bernoulli variance = tension

                # Stats averaged over sequence positions
                total_tension = tau.sum(dim=-1).mean().item()     # mean over seq of sum over d_ff
                edge_count = ((g > 0.3) & (g < 0.7)).float().sum(dim=-1).mean().item()
                mean_g = g.mean().item()

                layer_stats.append({
                    "layer": ell,
                    "total_tension": round(total_tension, 2),
                    "edge_neurons": round(edge_count, 1),
                    "mean_g": round(mean_g, 4),
                })

            results.append({
                "problem_idx": pidx,
                "lang": lang,
                "category": prob["category"],
                "layer_stats": layer_stats,
            })

        if (pidx + 1) % 5 == 0:
            print(f"  Diagnostic: {pidx + 1}/20 problems done")

    for h in hooks:
        h.remove()

    # Print first problem as sample
    print("\n  Sample (problem 0/en):")
    for ls in results[0]["layer_stats"]:
        bar = "#" * int(ls["total_tension"] / 50)
        print(f"    L{ls['layer']:2d}: T={ls['total_tension']:7.1f}  edge={ls['edge_neurons']:6.0f}  {bar}")

    return results


# ── Sensitivity Modulator (hook-based) ────────────────────────

class SensitivityModulator:
    """Modulates attention Q-projection inputs with MLP gate sensitivity.

    For each MLP at layer ℓ:
      1. Capture gate_proj pre-activations x
      2. Compute g = SiLU'(x), tension τ = g(1-g)
      3. Project to model dim: s_i = Σ_j W_down[i,j]² · τ_j
      4. Normalize s to mean 1
      5. Modulate layer ℓ+1's q_proj input: q = (h ⊙ s) W_Q

    Modes: sensitivity | random | uniform | inverse
    """

    def __init__(self, model, mode="sensitivity"):
        self.model = model
        self.mode = mode
        self.sensitivity = {}   # layer_idx -> (batch, seq, d_model)
        self.hooks = []
        self.enabled = True
        self.n_layers = len(model.model.layers)

        # Precompute W_down² per layer (avoids recomputing each step)
        self.w_down_sq = {}
        for ell in range(self.n_layers):
            W = model.model.layers[ell].mlp.down_proj.weight.detach()  # (d_model, d_ff)
            self.w_down_sq[ell] = W.pow(2)  # cached

        # Register hooks
        for ell in range(self.n_layers):
            # Post-hook on MLP: compute sensitivity from gate pre-activations
            h_mlp = model.model.layers[ell].mlp.register_forward_hook(
                self._make_mlp_hook(ell)
            )
            self.hooks.append(h_mlp)

            # Pre-hook on NEXT layer's Q projection: modulate input
            if ell + 1 < self.n_layers:
                h_q = model.model.layers[ell + 1].self_attn.q_proj.register_forward_pre_hook(
                    self._make_q_hook(ell)
                )
                self.hooks.append(h_q)

    def _make_mlp_hook(self, layer_idx):
        def hook(module, input, output):
            if not self.enabled:
                return
            h_in = input[0]  # MLP input (post_attention_layernorm output)
            with torch.no_grad():
                x_gate = module.gate_proj(h_in)  # (batch, seq, d_ff)
                g = silu_derivative(x_gate)
                tau = g * (1.0 - g)  # tension

                # Project to h-space: s_i = Σ_j W_down[i,j]² · τ_j
                # tau: (B, S, d_ff)  @  W_down_sq.T: (d_ff, d_model) → (B, S, d_model)
                s = torch.matmul(tau, self.w_down_sq[layer_idx].T)

                # Normalize to mean 1 (modulation, not scaling)
                s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)

                if self.mode == "random":
                    s = torch.rand_like(s) + 0.5
                    s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)
                elif self.mode == "uniform":
                    s = torch.ones_like(s)
                elif self.mode == "inverse":
                    s = 1.0 / (s + 1e-8)
                    s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)
                # else "sensitivity": use s as-is

                self.sensitivity[layer_idx] = s
        return hook

    def _make_q_hook(self, prev_layer_idx):
        def hook(module, args):
            if not self.enabled:
                return
            if prev_layer_idx not in self.sensitivity:
                return
            h = args[0]  # input to q_proj: (batch, seq, d_model)
            s = self.sensitivity[prev_layer_idx]

            # Shape alignment (should match, but guard for KV-cache edge cases)
            if s.shape[1] != h.shape[1]:
                s = s[:, -h.shape[1]:, :]

            return (h * s,) + args[1:]
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.sensitivity.clear()
        self.w_down_sq.clear()


# ── Generation + Condition Runner ─────────────────────────────

@torch.inference_mode()
def run_condition(model, tokenizer, name, modulator=None):
    """Run all 20 problems × 2 langs under a condition. Returns results dict."""
    print(f"\n--- {name} ---")
    problems = get_test_problems()
    per_problem = []
    correct_en, correct_zh = 0, 0

    for pidx, prob in enumerate(problems):
        for lang in ("en", "zh"):
            prompt = build_prompt(tokenizer, prob[lang])
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

            out = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=False,
                temperature=None, top_p=None,
            )
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            ok = check_answer(text, prob["answer"])

            if lang == "en" and ok:
                correct_en += 1
            elif lang == "zh" and ok:
                correct_zh += 1

            per_problem.append({
                "problem_idx": pidx,
                "lang": lang,
                "category": prob["category"],
                "correct": ok,
                "answer_expected": prob["answer"],
                "output_preview": text[:300],
            })

    total = correct_en + correct_zh
    print(f"  {name}: EN={correct_en}/20, ZH={correct_zh}/20, total={total}/40")
    return {
        "condition": name,
        "en": correct_en,
        "zh": correct_zh,
        "total": total,
        "per_problem": per_problem,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs("output", exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    d_ff = model.config.intermediate_size
    print(f"  {n_layers} layers, d_model={d_model}, d_ff={d_ff}")

    output = {
        "experiment": "SMA: Sensitivity-Modulated Attention",
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "n_problems": 20,
        "max_new_tokens": MAX_NEW,
    }

    # ── Part 0: Diagnostic ────────────────────────────────────
    diag = run_diagnostic(model, tokenizer)
    output["diagnostic"] = diag

    # Summarize: average tension per layer
    avg_tension = [0.0] * n_layers
    avg_edge = [0.0] * n_layers
    n_samples = len(diag)
    for d in diag:
        for ls in d["layer_stats"]:
            avg_tension[ls["layer"]] += ls["total_tension"] / n_samples
            avg_edge[ls["layer"]] += ls["edge_neurons"] / n_samples

    peak_layer = int(np.argmax(avg_tension))
    print(f"\n  Avg tension per layer (top-5):")
    ranked = sorted(range(n_layers), key=lambda i: avg_tension[i], reverse=True)
    for i in ranked[:5]:
        print(f"    L{i:2d}: tension={avg_tension[i]:.1f}, edge_neurons={avg_edge[i]:.0f}")
    print(f"  Peak: L{peak_layer} ({avg_tension[peak_layer]:.1f})")

    output["diagnostic_summary"] = {
        "avg_tension_per_layer": [round(t, 2) for t in avg_tension],
        "avg_edge_neurons_per_layer": [round(e, 1) for e in avg_edge],
        "peak_tension_layer": peak_layer,
        "peak_tension_value": round(avg_tension[peak_layer], 2),
    }

    if max(avg_tension) < 10:
        print("\n  WARNING: Very low tension — all neurons firmly committed.")
        print("  Mechanism may have nothing to work with. Continuing anyway.")

    # ── Part 1: Baseline ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 1: BASELINE")
    print("=" * 60)
    baseline = run_condition(model, tokenizer, "baseline")
    output["baseline"] = baseline

    # ── Part 2: Sensitivity-modulated attention ───────────────
    print("\n" + "=" * 60)
    print("PART 2: SENSITIVITY-MODULATED ATTENTION")
    print("=" * 60)
    mod = SensitivityModulator(model, mode="sensitivity")
    modulated = run_condition(model, tokenizer, "sensitivity_modulated", modulator=mod)
    mod.remove()
    output["sensitivity_modulated"] = modulated

    # ── Part 3: Controls ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 3: CONTROLS")
    print("=" * 60)

    # 3A: Random (same perturbation magnitude, no information)
    mod_r = SensitivityModulator(model, mode="random")
    ctrl_random = run_condition(model, tokenizer, "random_modulation", modulator=mod_r)
    mod_r.remove()
    output["random_modulation"] = ctrl_random

    # 3B: Uniform (s=1 everywhere, should equal baseline exactly)
    mod_u = SensitivityModulator(model, mode="uniform")
    ctrl_uniform = run_condition(model, tokenizer, "uniform_modulation", modulator=mod_u)
    mod_u.remove()
    output["uniform_modulation"] = ctrl_uniform

    # 3C: Inverse (attend AWAY from sensitive dims)
    mod_i = SensitivityModulator(model, mode="inverse")
    ctrl_inverse = run_condition(model, tokenizer, "inverse_modulation", modulator=mod_i)
    mod_i.remove()
    output["inverse_modulation"] = ctrl_inverse

    # ── Part 4: Analysis ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 4: ANALYSIS")
    print("=" * 60)

    conditions = {
        "baseline": baseline,
        "sensitivity_modulated": modulated,
        "random_modulation": ctrl_random,
        "uniform_modulation": ctrl_uniform,
        "inverse_modulation": ctrl_inverse,
    }

    # Per-problem sensitivity vs baseline
    changes = {"improved": [], "regressed": [], "unchanged": []}
    for i in range(len(baseline["per_problem"])):
        b = baseline["per_problem"][i]["correct"]
        m = modulated["per_problem"][i]["correct"]
        pidx = baseline["per_problem"][i]["problem_idx"]
        lang = baseline["per_problem"][i]["lang"]
        cat = baseline["per_problem"][i]["category"]

        if not b and m:
            changes["improved"].append({"pidx": pidx, "lang": lang, "category": cat})
        elif b and not m:
            changes["regressed"].append({"pidx": pidx, "lang": lang, "category": cat})
        else:
            changes["unchanged"].append({"pidx": pidx, "lang": lang, "category": cat})

    print(f"\n  Sensitivity vs Baseline:")
    print(f"    Improved:  {len(changes['improved'])}")
    print(f"    Regressed: {len(changes['regressed'])}")
    print(f"    Unchanged: {len(changes['unchanged'])}")
    for imp in changes["improved"]:
        print(f"      + P{imp['pidx']}/{imp['lang']} ({imp['category']})")
    for reg in changes["regressed"]:
        print(f"      - P{reg['pidx']}/{reg['lang']} ({reg['category']})")

    # Uniform sanity check
    uniform_match = all(
        baseline["per_problem"][i]["correct"] == ctrl_uniform["per_problem"][i]["correct"]
        for i in range(len(baseline["per_problem"]))
    )
    print(f"\n  Uniform == Baseline (sanity): {uniform_match}")
    if not uniform_match:
        diffs = sum(
            1 for i in range(len(baseline["per_problem"]))
            if baseline["per_problem"][i]["correct"] != ctrl_uniform["per_problem"][i]["correct"]
        )
        print(f"    WARNING: {diffs} problems differ — hook infrastructure may have side effects")

    # Tension correlation with difficulty
    # Group problems by baseline correctness, check if tension differs
    problems = get_test_problems()
    correct_tensions = []  # avg tension for problems answered correctly
    wrong_tensions = []    # avg tension for problems answered incorrectly
    for d_item in diag:
        pidx = d_item["problem_idx"]
        lang = d_item["lang"]
        avg_t = np.mean([ls["total_tension"] for ls in d_item["layer_stats"]])
        # Find matching baseline result
        for bp in baseline["per_problem"]:
            if bp["problem_idx"] == pidx and bp["lang"] == lang:
                if bp["correct"]:
                    correct_tensions.append(avg_t)
                else:
                    wrong_tensions.append(avg_t)
                break

    if correct_tensions and wrong_tensions:
        print(f"\n  Tension by difficulty:")
        print(f"    Correct problems: mean tension = {np.mean(correct_tensions):.1f} (n={len(correct_tensions)})")
        print(f"    Wrong problems:   mean tension = {np.mean(wrong_tensions):.1f} (n={len(wrong_tensions)})")
        print(f"    Delta: {np.mean(wrong_tensions) - np.mean(correct_tensions):+.1f}")

    output["analysis"] = {
        "changes_vs_baseline": changes,
        "uniform_matches_baseline": uniform_match,
        "tension_by_difficulty": {
            "correct_mean": round(float(np.mean(correct_tensions)), 2) if correct_tensions else None,
            "wrong_mean": round(float(np.mean(wrong_tensions)), 2) if wrong_tensions else None,
            "n_correct": len(correct_tensions),
            "n_wrong": len(wrong_tensions),
        },
        "summary_table": {
            name: {"en": res["en"], "zh": res["zh"], "total": res["total"]}
            for name, res in conditions.items()
        },
    }

    # Save
    elapsed = time.time() - t0
    output["wall_time_s"] = round(elapsed, 1)

    out_path = "output/expSMA_sensitivity_modulated_attention.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"Wall time: {elapsed:.0f}s")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, res in conditions.items():
        delta = res["total"] - baseline["total"]
        d_str = f" ({delta:+d})" if name != "baseline" else ""
        print(f"  {name:30s}: EN={res['en']:2d}/20  ZH={res['zh']:2d}/20  total={res['total']:2d}/40{d_str}")


if __name__ == "__main__":
    main()
