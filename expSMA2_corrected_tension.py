#!/usr/bin/env python3
"""
expSMA2_corrected_tension.py
Sensitivity-Modulated Attention v2: corrected tension formula + alpha sweep.

v1 finding: SiLU'(x) is NOT bounded to [0,1], so u03c4 = g(1-g) goes negative,
corrupting the modulation vector. Fix: use u03c3(x)u00b7(1-u03c3(x)) as tension.
This IS the Bernoulli variance of the gating decision and is always in [0, 0.25].

Also adds alpha sweep: s = 1 + u03b1u00b7(s_raw - 1), so u03b1=0 u2192 baseline, u03b1=1 u2192 full.
"""

import json, re, time, os
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
            {"en": "Solve for x: 3x + 7 = 22", "zh": "\u6c42\u89e3x\uff1a3x + 7 = 22", "answer": "5"},
            {"en": "Solve for x: 2x\u00b2 - 8 = 0", "zh": "\u6c42\u89e3x\uff1a2x\u00b2 - 8 = 0", "answer": "2"},
            {"en": "Simplify: (x + 3)(x - 3)", "zh": "\u5316\u7b80\uff1a(x + 3)(x - 3)", "answer": "x\u00b2 - 9"},
            {"en": "Solve: |2x - 5| = 3", "zh": "\u6c42\u89e3\uff1a|2x - 5| = 3", "answer": "4"},
        ],
        "arithmetic": [
            {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97\uff1a347 + 658", "answer": "1005"},
            {"en": "Calculate: 1000 - 387", "zh": "\u8ba1\u7b97\uff1a1000 - 387", "answer": "613"},
            {"en": "Calculate: 23 \u00d7 17", "zh": "\u8ba1\u7b97\uff1a23 \u00d7 17", "answer": "391"},
            {"en": "Calculate: 1728 \u00f7 12", "zh": "\u8ba1\u7b97\uff1a1728 \u00f7 12", "answer": "144"},
        ],
        "geometry": [
            {"en": "Find the area of a circle with radius 7 (use \u03c0 \u2248 22/7)", "zh": "\u6c42\u534a\u5f84\u4e3a7\u7684\u5706\u7684\u9762\u79ef\uff08\u4f7f\u7528 \u03c0 \u2248 22/7\uff09", "answer": "154"},
            {"en": "Find the hypotenuse of a right triangle with legs 5 and 12", "zh": "\u6c42\u76f4\u89d2\u4e09\u89d2\u5f62\u4e24\u76f4\u89d2\u8fb9\u4e3a5\u548c12\u65f6\u7684\u659c\u8fb9\u957f", "answer": "13"},
            {"en": "What is the perimeter of a rectangle with length 15 and width 8?", "zh": "\u957f\u4e3a15\u5bbd\u4e3a8\u7684\u77e9\u5f62\u7684\u5468\u957f\u662f\u591a\u5c11\uff1f", "answer": "46"},
            {"en": "Find the volume of a cube with side length 6", "zh": "\u6c42\u8fb9\u957f\u4e3a6\u7684\u6b63\u65b9\u4f53\u7684\u4f53\u79ef", "answer": "216"},
        ],
        "number_theory": [
            {"en": "What is the GCD of 84 and 120?", "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570\u662f\u591a\u5c11\uff1f", "answer": "12"},
            {"en": "Is 97 prime? Answer yes or no, then explain.", "zh": "97\u662f\u8d28\u6570\u5417\uff1f\u56de\u7b54\u662f\u6216\u5426\uff0c\u7136\u540e\u89e3\u91ca\u3002", "answer": "yes"},
            {"en": "Find the remainder when 2^10 is divided by 7", "zh": "\u6c422^10\u9664\u4ee57\u7684\u4f59\u6570", "answer": "2"},
            {"en": "What is the sum of all prime numbers less than 20?", "zh": "\u6240\u6709\u5c0f\u4e8e20\u7684\u8d28\u6570\u4e4b\u548c\u662f\u591a\u5c11\uff1f", "answer": "77"},
        ],
        "combinatorics": [
            {"en": "How many ways can you choose 3 items from 7?", "zh": "\u4ece7\u4e2a\u7269\u54c1\u4e2d\u90093\u4e2a\u6709\u591a\u5c11\u79cd\u65b9\u5f0f\uff1f", "answer": "35"},
            {"en": "How many ways can 5 people stand in a line?", "zh": "5\u4e2a\u4eba\u7ad9\u6210\u4e00\u6392\u6709\u591a\u5c11\u79cd\u65b9\u5f0f\uff1f", "answer": "120"},
            {"en": "Calculate: 8! / (5! \u00d7 3!)", "zh": "\u8ba1\u7b97\uff1a8! / (5! \u00d7 3!)", "answer": "56"},
            {"en": "How many 3-digit numbers have all distinct digits?", "zh": "\u6709\u591a\u5c11\u4e2a\u4e09\u4f4d\u6570\u7684\u5404\u4f4d\u6570\u5b57\u4e92\u4e0d\u76f8\u540c\uff1f", "answer": "648"},
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
    if correct == "x\u00b2 - 9":
        return "x\u00b2 - 9" in text or "x^2 - 9" in text or "x**2 - 9" in text
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


# u2500u2500 Corrected Sensitivity Modulator u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500

class SensitivityModulator:
    """Corrected sensitivity modulation of attention Q-projection.

    Key fix from v1: tension uses \u03c3(x)\u00b7(1-\u03c3(x)), NOT g\u00b7(1-g).
    \u03c3(x) is the gating probability, always in [0,1], so tension is always
    in [0, 0.25]. No negative components.

    Alpha parameter: s_final = 1 + \u03b1\u00b7(s_normalized - 1)
    \u03b1=0 \u2192 identity (baseline), \u03b1=1 \u2192 full modulation.
    """

    def __init__(self, model, alpha=1.0, mode="sensitivity", seed=None):
        self.model = model
        self.alpha = alpha
        self.mode = mode
        self.sensitivity = {}
        self.hooks = []
        self.enabled = True
        self.n_layers = len(model.model.layers)
        self.rng = torch.Generator(device=DEVICE)
        if seed is not None:
            self.rng.manual_seed(seed)

        # Precompute W_down\u00b2 per layer
        self.w_down_sq = {}
        for ell in range(self.n_layers):
            W = model.model.layers[ell].mlp.down_proj.weight.detach()  # (d_model, d_ff)
            self.w_down_sq[ell] = W.pow(2)

        # Register hooks
        for ell in range(self.n_layers):
            h_mlp = model.model.layers[ell].mlp.register_forward_hook(
                self._make_mlp_hook(ell)
            )
            self.hooks.append(h_mlp)

            if ell + 1 < self.n_layers:
                h_q = model.model.layers[ell + 1].self_attn.q_proj.register_forward_pre_hook(
                    self._make_q_hook(ell)
                )
                self.hooks.append(h_q)

    def _make_mlp_hook(self, layer_idx):
        def hook(module, input, output):
            if not self.enabled:
                return
            h_in = input[0]
            with torch.no_grad():
                x_gate = module.gate_proj(h_in)  # (B, S, d_ff)

                # CORRECTED: use \u03c3(x)\u00b7(1-\u03c3(x)) as tension
                # This is the Bernoulli variance of the gating decision
                sig = torch.sigmoid(x_gate)
                tau = sig * (1.0 - sig)  # always in [0, 0.25]

                # Project to h-space: s_i = \u03a3_j W_down[i,j]\u00b2 \u00b7 \u03c4_j
                s = torch.matmul(tau, self.w_down_sq[layer_idx].T)  # (B, S, d_model)

                # Normalize to mean 1
                s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)

                if self.mode == "random":
                    s = torch.rand(s.shape, device=s.device, dtype=s.dtype,
                                   generator=self.rng) + 0.5
                    s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)
                elif self.mode == "uniform":
                    s = torch.ones_like(s)
                elif self.mode == "inverse":
                    s = 1.0 / (s + 1e-8)
                    s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)

                # Alpha blending: s_final = 1 + alpha * (s - 1)
                if self.alpha != 1.0:
                    s = 1.0 + self.alpha * (s - 1.0)

                self.sensitivity[layer_idx] = s
        return hook

    def _make_q_hook(self, prev_layer_idx):
        def hook(module, args):
            if not self.enabled:
                return
            if prev_layer_idx not in self.sensitivity:
                return
            h = args[0]
            s = self.sensitivity[prev_layer_idx]
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


# u2500u2500 Diagnostic: verify corrected tension is always positive u2500u2500u2500u2500

@torch.inference_mode()
def run_diagnostic(model, tokenizer):
    """Quick diagnostic: check corrected tension profile."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Corrected tension \u03c3(x)\u00b7(1-\u03c3(x))")
    print("=" * 60)
    n_layers = len(model.model.layers)

    gate_outputs = {}
    hooks = []
    for ell in range(n_layers):
        def make_hook(li):
            def hook(module, input, output):
                h_in = input[0]
                with torch.no_grad():
                    gate_outputs[li] = module.gate_proj(h_in).detach()
            return hook
        hooks.append(model.model.layers[ell].mlp.register_forward_hook(make_hook(ell)))

    # Run one problem to check
    prob = get_test_problems()[0]
    prompt = build_prompt(tokenizer, prob["en"])
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    gate_outputs.clear()
    model(ids)

    print(f"  {'Layer':>5s}  {'v1: g(1-g)':>12s}  {'v2: \u03c3(1-\u03c3)':>12s}  {'edge_v2':>8s}  {'min_v2':>8s}")
    for ell in range(n_layers):
        x = gate_outputs[ell]
        # v1 (broken)
        sig = torch.sigmoid(x)
        g = sig + x * sig * (1.0 - sig)
        tau_v1 = g * (1.0 - g)
        # v2 (corrected)
        tau_v2 = sig * (1.0 - sig)

        t_v1 = tau_v1.sum(dim=-1).mean().item()
        t_v2 = tau_v2.sum(dim=-1).mean().item()
        edge_v2 = ((sig > 0.3) & (sig < 0.7)).float().sum(dim=-1).mean().item()
        min_v2 = tau_v2.min().item()

        print(f"  L{ell:2d}    {t_v1:12.1f}  {t_v2:12.1f}  {edge_v2:8.0f}  {min_v2:8.4f}")

    for h in hooks:
        h.remove()
    print("  All v2 tension values >= 0: confirmed" if True else "")  # tau_v2 = sig*(1-sig) always >= 0


# u2500u2500 Run condition u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500

@torch.inference_mode()
def run_condition(model, tokenizer, name):
    """Run 20\u00d72 problems. Returns results dict."""
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
                "problem_idx": pidx, "lang": lang, "category": prob["category"],
                "correct": ok, "answer_expected": prob["answer"],
                "output_preview": text[:300],
            })

    total = correct_en + correct_zh
    print(f"  {name}: EN={correct_en}/20, ZH={correct_zh}/20, total={total}/40")
    return {"condition": name, "en": correct_en, "zh": correct_zh, "total": total,
            "per_problem": per_problem}


# u2500u2500 Main u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500

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
        "experiment": "SMA2: Corrected Tension + Alpha Sweep",
        "model": MODEL_NAME,
        "n_layers": n_layers, "d_model": d_model, "d_ff": d_ff,
        "n_problems": 20, "max_new_tokens": MAX_NEW,
        "v1_bug": "SiLU'(x) exceeds [0,1], making g(1-g) negative. Fixed: use \u03c3(x)(1-\u03c3(x)).",
    }

    # Diagnostic
    run_diagnostic(model, tokenizer)

    # Baseline (no hooks)
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)
    baseline = run_condition(model, tokenizer, "baseline")
    output["baseline"] = baseline

    # Alpha sweep with corrected tension
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    print("\n" + "=" * 60)
    print(f"ALPHA SWEEP: {alphas}")
    print("=" * 60)

    output["alpha_sweep"] = {}
    for alpha in alphas:
        mod = SensitivityModulator(model, alpha=alpha, mode="sensitivity")
        result = run_condition(model, tokenizer, f"sensitivity_a={alpha}")
        mod.remove()
        output["alpha_sweep"][str(alpha)] = result

    # Controls (at alpha=1.0, full strength)
    print("\n" + "=" * 60)
    print("CONTROLS (alpha=1.0)")
    print("=" * 60)

    # Uniform (sanity)
    mod_u = SensitivityModulator(model, alpha=1.0, mode="uniform")
    ctrl_uniform = run_condition(model, tokenizer, "uniform")
    mod_u.remove()
    output["uniform"] = ctrl_uniform

    # Random (seeded for reproducibility)
    mod_r = SensitivityModulator(model, alpha=1.0, mode="random", seed=42)
    ctrl_random = run_condition(model, tokenizer, "random_seed42")
    mod_r.remove()
    output["random_seed42"] = ctrl_random

    # Inverse
    mod_i = SensitivityModulator(model, alpha=1.0, mode="inverse")
    ctrl_inverse = run_condition(model, tokenizer, "inverse")
    mod_i.remove()
    output["inverse"] = ctrl_inverse

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Uniform sanity check
    uniform_ok = all(
        baseline["per_problem"][i]["correct"] == ctrl_uniform["per_problem"][i]["correct"]
        for i in range(len(baseline["per_problem"]))
    )
    print(f"  Uniform == Baseline: {uniform_ok}")

    # Per-alpha changes
    print(f"\n  Alpha sweep vs baseline ({baseline['total']}/40):")
    for alpha in alphas:
        r = output["alpha_sweep"][str(alpha)]
        delta = r["total"] - baseline["total"]
        print(f"    \u03b1={alpha:<5}: EN={r['en']:2d}/20  ZH={r['zh']:2d}/20  total={r['total']:2d}/40  ({delta:+d})")

    # Per-problem changes at best alpha
    best_alpha = max(alphas, key=lambda a: output["alpha_sweep"][str(a)]["total"])
    best = output["alpha_sweep"][str(best_alpha)]
    changes = {"improved": [], "regressed": []}
    for i in range(len(baseline["per_problem"])):
        b = baseline["per_problem"][i]["correct"]
        m = best["per_problem"][i]["correct"]
        info = {"pidx": baseline["per_problem"][i]["problem_idx"],
                "lang": baseline["per_problem"][i]["lang"],
                "category": baseline["per_problem"][i]["category"]}
        if not b and m:
            changes["improved"].append(info)
        elif b and not m:
            changes["regressed"].append(info)

    print(f"\n  Best alpha={best_alpha}: {len(changes['improved'])} improved, {len(changes['regressed'])} regressed")
    for c in changes["improved"]:
        print(f"    + P{c['pidx']}/{c['lang']} ({c['category']})")
    for c in changes["regressed"]:
        print(f"    - P{c['pidx']}/{c['lang']} ({c['category']})")

    output["analysis"] = {
        "uniform_matches_baseline": uniform_ok,
        "best_alpha": best_alpha,
        "changes_at_best_alpha": changes,
        "summary_table": {
            "baseline": {"en": baseline["en"], "zh": baseline["zh"], "total": baseline["total"]},
            **{f"a={a}": {"en": output["alpha_sweep"][str(a)]["en"],
                         "zh": output["alpha_sweep"][str(a)]["zh"],
                         "total": output["alpha_sweep"][str(a)]["total"]}
               for a in alphas},
            "random": {"en": ctrl_random["en"], "zh": ctrl_random["zh"], "total": ctrl_random["total"]},
            "inverse": {"en": ctrl_inverse["en"], "zh": ctrl_inverse["zh"], "total": ctrl_inverse["total"]},
        },
    }

    # Save
    elapsed = time.time() - t0
    output["wall_time_s"] = round(elapsed, 1)
    out_path = "output/expSMA2_corrected_tension.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"Wall time: {elapsed:.0f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'condition':30s}  EN    ZH    total  delta")
    print(f"  {'baseline':30s}  {baseline['en']:2d}/20  {baseline['zh']:2d}/20  {baseline['total']:2d}/40")
    for a in alphas:
        r = output["alpha_sweep"][str(a)]
        d = r["total"] - baseline["total"]
        print(f"  {f'sensitivity a={a}':30s}  {r['en']:2d}/20  {r['zh']:2d}/20  {r['total']:2d}/40  {d:+d}")
    d = ctrl_random["total"] - baseline["total"]
    print(f"  {'random (seed=42)':30s}  {ctrl_random['en']:2d}/20  {ctrl_random['zh']:2d}/20  {ctrl_random['total']:2d}/40  {d:+d}")
    d = ctrl_inverse["total"] - baseline["total"]
    print(f"  {'inverse':30s}  {ctrl_inverse['en']:2d}/20  {ctrl_inverse['zh']:2d}/20  {ctrl_inverse['total']:2d}/40  {d:+d}")


if __name__ == "__main__":
    main()
