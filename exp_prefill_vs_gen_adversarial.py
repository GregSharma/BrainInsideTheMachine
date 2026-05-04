"""Experiment: Prefill vs Generation Adversarial Phase Comparison

THE QUESTION: Does the adversarial zone (L9-L17, negative cross-layer delta cosine)
exist during prefill, worsen during generation, or change character between them?

If adversarial pattern is WORSE during generation than prefill, it suggests the token
round-trip (lm_head → vocab → re-embed) forces recovery computation that manifests
as layers fighting each other — the "Lego daycare" hypothesis.

If identical in both regimes, the adversarial zone is intrinsic to the 3B's layer
computation regardless of the token bottleneck.

METHOD:
- 20 problems (test set: 4 per category = 20), both zh and en = 40 runs
- Prefill: single forward pass, capture MLP delta at L0-L35 (last token)
- Generation: 128 tokens, capture MLP delta at L0-L35 at EVERY generated token
- Compute cross-layer cosine: cos(delta_L, delta_{L-1}) per sample
- Compare prefill profile vs generation profile (mean over tokens)
- Also compute generation profile at early (tok 1-10), mid (tok 50-70), late (tok 100+)
"""

import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_NAME = "Qwen/Qwen2.5-3B"
device = "cuda"
MAX_NEW_TOKENS = 128
N_TEST = 20  # first 4 per category = 20

print(f"Loading {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
# Capture ALL layers, not just L9-L26, to see full profile
ALL_LAYERS = list(range(n_layers))

print(f"Model: {n_layers} layers, d={d_model}")
print(f"Capturing all {n_layers} layers")
t0 = time.time()


# =============================================================================
# Problem generation (same seed as all experiments)
# =============================================================================
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
        problems.append({"zh": zh, "en": en})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                          "en": f"Find the value of C({n_val}, {k_val})."})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？",
                          "en": f"What is the remainder when {a} is divided by {b}?"})
    for _ in range(per_cat):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                          "en": f"A rectangle has length {w} and width {h}. Find its area."})
    for _ in range(per_cat):
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。",
                          "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_terms} terms."})
    return problems

all_problems = generate_problems(200, seed=42)
test_problems = all_problems[:N_TEST]  # first 4 per category = 20


# =============================================================================
# Hook infrastructure: capture MLP delta at every layer, every token
# =============================================================================
class MLPDeltaCapture:
    """Captures MLP delta (= MLP output) at all layers for every forward pass.

    During prefill: captures at the last token position only.
    During generation: captures at the single new token (KV-cached, seq_len=1).

    Stores per-step deltas in pre-allocated GPU buffers.
    """

    def __init__(self, model, layers, max_steps=200):
        self.model = model
        self.layers = layers
        self.max_steps = max_steps
        self.d = model.config.hidden_size
        self.device = next(model.parameters()).device
        self._hooks = []
        self._buffers = {}
        self._counters = {}
        self._is_prefill = True  # flag: first call is prefill, rest are gen

    def _reset(self):
        self._buffers = {}
        self._counters = {}
        for li in self.layers:
            self._buffers[li] = torch.empty(
                (self.max_steps, self.d), device=self.device, dtype=torch.float32
            )
            self._counters[li] = 0

    def _register(self):
        self._hooks = []
        for li in self.layers:
            layer = self.model.model.layers[li]

            def make_hook(idx):
                def hook(module, inp, out):
                    # MLP output = the delta added to residual stream
                    i = self._counters[idx]
                    if i < self.max_steps:
                        # During prefill: take last token. During gen: seq_len=1, take [0]
                        self._buffers[idx][i].copy_(out[0, -1, :].float())
                        self._counters[idx] = i + 1
                return hook

            self._hooks.append(layer.mlp.register_forward_hook(make_hook(li)))

    def _unregister(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run_prefill(self, input_ids):
        """Single forward pass, returns MLP delta at last token for each layer."""
        self._reset()
        self._register()
        with torch.no_grad():
            self.model(input_ids=input_ids)
        self._unregister()

        result = {}
        for li in self.layers:
            n = self._counters[li]
            if n > 0:
                # Prefill: only one capture per layer (last token from batch-1 forward)
                result[li] = self._buffers[li][0].cpu().numpy()
        return result

    def run_generation(self, input_ids, max_new_tokens=128):
        """Generation with KV cache. Returns MLP delta at every gen step for each layer."""
        self._reset()
        self._register()

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        self._unregister()

        prompt_len = input_ids.shape[1]
        gen_ids = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_gen = len(gen_ids)

        result = {"gen_text": gen_text, "n_gen_tokens": n_gen}
        for li in self.layers:
            n = self._counters[li]
            if n > 1:
                # Skip first (prefill), keep generation steps
                result[li] = self._buffers[li][1:n].cpu().numpy()  # (n_gen, d)
            elif n == 1:
                result[li] = self._buffers[li][0:1].cpu().numpy()
        return result


# =============================================================================
# Run experiments
# =============================================================================
capturer = MLPDeltaCapture(model, ALL_LAYERS, max_steps=MAX_NEW_TOKENS + 50)

prefill_results = {"zh": [], "en": []}
gen_results = {"zh": [], "en": []}

for lang in ["zh", "en"]:
    prompts = [p[lang] for p in test_problems]
    print(f"\n{'='*60}")
    print(f"Running {lang.upper()} ({len(prompts)} problems)")
    print(f"{'='*60}")

    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Prefill
        pf = capturer.run_prefill(input_ids)
        prefill_results[lang].append(pf)

        # Generation
        gen = capturer.run_generation(input_ids, max_new_tokens=MAX_NEW_TOKENS)
        gen_results[lang].append(gen)

        if (pi + 1) % 5 == 0:
            print(f"  {lang} {pi+1}/{len(prompts)} done ({time.time()-t0:.0f}s)")


# =============================================================================
# Analysis: cross-layer delta cosine profiles
# =============================================================================
print(f"\n{'='*60}")
print("ANALYSIS: Cross-layer MLP delta cosine similarity")
print(f"{'='*60}")


def cross_layer_cosine_prefill(results_list, layers):
    """Compute cos(delta_L, delta_{L-1}) across problems for prefill.

    Returns dict: "L{prev}->L{curr}" -> {"cos_mean", "cos_std", "cos_values"}
    """
    out = {}
    for i in range(1, len(layers)):
        prev_li, curr_li = layers[i-1], layers[i]
        cos_vals = []
        for pf in results_list:
            if prev_li in pf and curr_li in pf:
                d_prev = pf[prev_li]
                d_curr = pf[curr_li]
                n_prev = np.linalg.norm(d_prev) + 1e-10
                n_curr = np.linalg.norm(d_curr) + 1e-10
                cos = float(np.dot(d_prev, d_curr) / (n_prev * n_curr))
                cos_vals.append(cos)
        if cos_vals:
            out[f"L{prev_li}->L{curr_li}"] = {
                "cos_mean": float(np.mean(cos_vals)),
                "cos_std": float(np.std(cos_vals)),
                "n": len(cos_vals),
            }
    return out


def cross_layer_cosine_generation(results_list, layers, tok_range=None):
    """Compute cos(delta_L, delta_{L-1}) across problems during generation.

    tok_range: (start, end) to restrict to specific token window.
    For each problem, averages cosine over all tokens in range.
    """
    out = {}
    for i in range(1, len(layers)):
        prev_li, curr_li = layers[i-1], layers[i]
        cos_vals = []  # one entry per problem (mean over tokens)
        for gen in results_list:
            if prev_li not in gen or curr_li not in gen:
                continue
            d_prev = gen[prev_li]  # (n_tok, d)
            d_curr = gen[curr_li]  # (n_tok, d)
            n_tok = min(d_prev.shape[0], d_curr.shape[0])
            if n_tok == 0:
                continue

            if tok_range is not None:
                s, e = tok_range
                s = min(s, n_tok)
                e = min(e, n_tok)
                if s >= e:
                    continue
                d_prev = d_prev[s:e]
                d_curr = d_curr[s:e]

            # Per-token cosine then mean
            norms_p = np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-10
            norms_c = np.linalg.norm(d_curr, axis=1, keepdims=True) + 1e-10
            per_tok_cos = np.sum((d_prev / norms_p) * (d_curr / norms_c), axis=1)
            cos_vals.append(float(per_tok_cos.mean()))

        if cos_vals:
            out[f"L{prev_li}->L{curr_li}"] = {
                "cos_mean": float(np.mean(cos_vals)),
                "cos_std": float(np.std(cos_vals)),
                "n": len(cos_vals),
            }
    return out


# Combine zh + en for each condition
all_prefill = prefill_results["zh"] + prefill_results["en"]
all_gen = gen_results["zh"] + gen_results["en"]

print("\n--- PREFILL: cross-layer delta cosine ---")
pf_profile = cross_layer_cosine_prefill(all_prefill, ALL_LAYERS)
for k in sorted(pf_profile.keys(), key=lambda x: int(x.split("->")[0][1:])):
    v = pf_profile[k]
    print(f"  {k}: cos={v['cos_mean']:+.4f} ± {v['cos_std']:.4f}")

print("\n--- GENERATION (all tokens): cross-layer delta cosine ---")
gen_all_profile = cross_layer_cosine_generation(all_gen, ALL_LAYERS)
for k in sorted(gen_all_profile.keys(), key=lambda x: int(x.split("->")[0][1:])):
    v = gen_all_profile[k]
    print(f"  {k}: cos={v['cos_mean']:+.4f} ± {v['cos_std']:.4f}")

print("\n--- GENERATION (early, tok 0-10): cross-layer delta cosine ---")
gen_early = cross_layer_cosine_generation(all_gen, ALL_LAYERS, tok_range=(0, 10))
for k in sorted(gen_early.keys(), key=lambda x: int(x.split("->")[0][1:])):
    v = gen_early[k]
    print(f"  {k}: cos={v['cos_mean']:+.4f} ± {v['cos_std']:.4f}")

print("\n--- GENERATION (late, tok 80+): cross-layer delta cosine ---")
gen_late = cross_layer_cosine_generation(all_gen, ALL_LAYERS, tok_range=(80, 128))
for k in sorted(gen_late.keys(), key=lambda x: int(x.split("->")[0][1:])):
    v = gen_late[k]
    print(f"  {k}: cos={v['cos_mean']:+.4f} ± {v['cos_std']:.4f}")


# =============================================================================
# The key comparison: adversarial zone (L9-L17) prefill vs gen
# =============================================================================
print(f"\n{'='*60}")
print("KEY COMPARISON: Adversarial Zone (L9-L17)")
print(f"{'='*60}")

adv_layers = list(range(9, 18))
coop_layers = list(range(18, 22))
ramp_layers = list(range(22, 27))

def zone_summary(profile, layers, label):
    """Average cosine over a zone of layers."""
    vals = []
    for i in range(1, len(layers)):
        key = f"L{layers[i-1]}->L{layers[i]}"
        if key in profile:
            vals.append(profile[key]["cos_mean"])
    if vals:
        return {"label": label, "mean_cos": float(np.mean(vals)), "n_pairs": len(vals)}
    return {"label": label, "mean_cos": None, "n_pairs": 0}


results_table = []
for zone_name, zone_layers in [("adversarial L9-L17", adv_layers),
                                 ("cooperative L18-L21", coop_layers),
                                 ("ramp L22-L26", ramp_layers)]:
    pf_z = zone_summary(pf_profile, zone_layers, f"prefill_{zone_name}")
    gen_z = zone_summary(gen_all_profile, zone_layers, f"gen_all_{zone_name}")
    gen_e = zone_summary(gen_early, zone_layers, f"gen_early_{zone_name}")
    gen_l = zone_summary(gen_late, zone_layers, f"gen_late_{zone_name}")

    print(f"\n  {zone_name}:")
    print(f"    Prefill:       {pf_z['mean_cos']:+.4f}" if pf_z['mean_cos'] is not None else "    Prefill: N/A")
    print(f"    Gen (all tok):  {gen_z['mean_cos']:+.4f}" if gen_z['mean_cos'] is not None else "    Gen all: N/A")
    print(f"    Gen (tok 0-10): {gen_e['mean_cos']:+.4f}" if gen_e['mean_cos'] is not None else "    Gen early: N/A")
    print(f"    Gen (tok 80+):  {gen_l['mean_cos']:+.4f}" if gen_l['mean_cos'] is not None else "    Gen late: N/A")

    results_table.append({
        "zone": zone_name,
        "prefill": pf_z, "gen_all": gen_z, "gen_early": gen_e, "gen_late": gen_l,
    })


# =============================================================================
# Also: per-language comparison
# =============================================================================
print(f"\n{'='*60}")
print("PER-LANGUAGE: Adversarial zone prefill vs gen")
print(f"{'='*60}")

for lang in ["zh", "en"]:
    pf_lang = cross_layer_cosine_prefill(prefill_results[lang], ALL_LAYERS)
    gen_lang = cross_layer_cosine_generation(gen_results[lang], ALL_LAYERS)
    pf_adv = zone_summary(pf_lang, adv_layers, f"{lang}_prefill_adv")
    gen_adv = zone_summary(gen_lang, adv_layers, f"{lang}_gen_adv")
    pf_coop = zone_summary(pf_lang, coop_layers, f"{lang}_prefill_coop")
    gen_coop = zone_summary(gen_lang, coop_layers, f"{lang}_gen_coop")
    print(f"\n  {lang.upper()}:")
    print(f"    Adversarial - Prefill: {pf_adv['mean_cos']:+.4f}, Gen: {gen_adv['mean_cos']:+.4f}, Delta: {gen_adv['mean_cos'] - pf_adv['mean_cos']:+.4f}" if pf_adv['mean_cos'] and gen_adv['mean_cos'] else f"    {lang} adversarial: N/A")
    print(f"    Cooperative - Prefill: {pf_coop['mean_cos']:+.4f}, Gen: {gen_coop['mean_cos']:+.4f}, Delta: {gen_coop['mean_cos'] - pf_coop['mean_cos']:+.4f}" if pf_coop['mean_cos'] and gen_coop['mean_cos'] else f"    {lang} cooperative: N/A")


# =============================================================================
# Save
# =============================================================================
output = {
    "experiment": "Prefill vs Generation Adversarial Phase Comparison",
    "model": MODEL_NAME,
    "n_problems": N_TEST,
    "max_new_tokens": MAX_NEW_TOKENS,
    "method": "Compare cross-layer MLP delta cosine in prefill (single pass) vs generation (token-by-token with KV cache). Tests whether adversarial zone is intrinsic or worsened by token bottleneck.",
    "prefill_profile": {k: v for k, v in pf_profile.items()},
    "gen_all_profile": {k: v for k, v in gen_all_profile.items()},
    "gen_early_profile": {k: v for k, v in gen_early.items()},
    "gen_late_profile": {k: v for k, v in gen_late.items()},
    "zone_comparison": results_table,
    "runtime_seconds": time.time() - t0,
}

with open("output/exp_prefill_vs_gen_adversarial.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n\nResults saved to output/exp_prefill_vs_gen_adversarial.json")
print(f"Total runtime: {time.time()-t0:.0f}s")
