"""Experiment: Does the MLP language flip restore the cooperative zone during generation?

HYPOTHESIS: The flip works by helping the model maintain cooperation across token
boundaries. If true, generation with flip ON should show L18-L21 recovering positive
cross-layer delta cosine (which collapsed from +0.268 prefill to +0.009 in baseline gen).

METHOD:
- Same 20 problems × 2 langs as the prefill/gen comparison
- Baseline generation: no intervention (already have this data, but re-run for consistency)
- Flip generation: hook MLP deltas at L9-L26, flip the language direction component
- Compare cross-layer cosine profiles: does cooperative zone recover?

The flip: at each layer L9-L26, project the MLP delta onto the mean-difference direction
(zh - en mean over training set), then flip that component (scale by -1).
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
N_TEST = 20
N_TRAIN = 200  # for computing language direction

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
ALL_LAYERS = list(range(n_layers))
FLIP_LAYERS = list(range(9, 27))  # L9-L26

print(f"Model: {n_layers} layers, d={d_model}")
t0 = time.time()


# =============================================================================
# Problem generation
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
test_problems = all_problems[:N_TEST]


# =============================================================================
# Step 1: Compute language direction from training set (prefill, MLP deltas)
# =============================================================================
print(f"\n{'='*60}")
print("STEP 1: Computing language direction from 200 training problems")
print(f"{'='*60}")

lang_directions = {}  # {layer_idx: unit direction vector (d,)}

# Extract MLP deltas for all training problems
train_deltas = {li: {"zh": [], "en": []} for li in FLIP_LAYERS}
handles = []

for li in FLIP_LAYERS:
    layer = model.model.layers[li]

    def make_hook(idx):
        captures = {}
        def hook(module, inp, out):
            captures["delta"] = out.detach().float()
        captures["hook_fn"] = hook
        return captures
    cap = make_hook(li)
    handles.append((li, cap, layer.mlp.register_forward_hook(cap["hook_fn"])))

for lang in ["zh", "en"]:
    prompts = [p[lang] for p in all_problems]
    BATCH = 16
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i:i+BATCH]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        attn_mask = inputs["attention_mask"]
        last_idx = attn_mask.sum(dim=1) - 1

        # Clear captures
        for li, cap, _ in handles:
            cap.pop("delta", None)

        with torch.no_grad():
            model(**inputs)

        for li, cap, _ in handles:
            delta = cap["delta"]  # (batch, seq, d)
            for j in range(delta.shape[0]):
                train_deltas[li][lang].append(delta[j, last_idx[j]].cpu().numpy())

    print(f"  {lang}: {len(train_deltas[FLIP_LAYERS[0]][lang])} samples extracted")

for _, _, h in handles:
    h.remove()

# Compute mean-difference direction
for li in FLIP_LAYERS:
    zh_mean = np.mean(np.stack(train_deltas[li]["zh"]), axis=0)
    en_mean = np.mean(np.stack(train_deltas[li]["en"]), axis=0)
    diff = zh_mean - en_mean
    norm = np.linalg.norm(diff) + 1e-10
    lang_directions[li] = torch.tensor(diff / norm, device=device, dtype=torch.float32)

print(f"  Language directions computed for L{FLIP_LAYERS[0]}-L{FLIP_LAYERS[-1]}")
print(f"  Step 1 took {time.time()-t0:.0f}s")


# =============================================================================
# Step 2: Capture infrastructure (same as prev experiment + flip hooks)
# =============================================================================

class MLPDeltaCaptureWithFlip:
    """Captures MLP delta at all layers. Optionally flips language direction at L9-L26."""

    def __init__(self, model, layers, flip_layers, lang_dirs, max_steps=200, flip_on=False):
        self.model = model
        self.layers = layers
        self.flip_layers = flip_layers
        self.lang_dirs = lang_dirs  # {layer: unit_direction tensor}
        self.max_steps = max_steps
        self.d = model.config.hidden_size
        self.device = next(model.parameters()).device
        self.flip_on = flip_on
        self._capture_hooks = []
        self._flip_hooks = []
        self._buffers = {}
        self._counters = {}

    def _reset(self):
        self._buffers = {}
        self._counters = {}
        for li in self.layers:
            self._buffers[li] = torch.empty(
                (self.max_steps, self.d), device=self.device, dtype=torch.float32
            )
            self._counters[li] = 0

    def _register(self):
        self._capture_hooks = []
        self._flip_hooks = []

        for li in self.layers:
            layer = self.model.model.layers[li]

            # Capture hook: records MLP output AFTER any flip
            def make_capture(idx):
                def hook(module, inp, out):
                    i = self._counters[idx]
                    if i < self.max_steps:
                        self._buffers[idx][i].copy_(out[0, -1, :].float())
                        self._counters[idx] = i + 1
                    return out
                return hook

            # Flip hook: modifies MLP output in-place
            if self.flip_on and li in self.flip_layers and li in self.lang_dirs:
                def make_flip(idx):
                    direction = self.lang_dirs[idx]
                    def hook(module, inp, out):
                        # out shape: (batch, seq, d)
                        # Project onto language direction, flip it (scale by -2 to reverse)
                        dir_cast = direction.to(out.dtype)
                        proj = (out * dir_cast).sum(dim=-1, keepdim=True)
                        out = out - 2 * proj * dir_cast
                        return out
                    return hook
                # Flip hook runs FIRST (modifies output), capture hook runs SECOND (records it)
                self._flip_hooks.append(
                    layer.mlp.register_forward_hook(make_flip(li)))

            self._capture_hooks.append(
                layer.mlp.register_forward_hook(make_capture(li)))

    def _unregister(self):
        for h in self._capture_hooks + self._flip_hooks:
            h.remove()
        self._capture_hooks = []
        self._flip_hooks = []

    def run_generation(self, input_ids, max_new_tokens=128):
        self._reset()
        self._register()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        self._unregister()

        prompt_len = input_ids.shape[1]
        gen_ids = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        result = {"gen_text": gen_text, "n_gen_tokens": len(gen_ids)}
        for li in self.layers:
            n = self._counters[li]
            if n > 1:
                result[li] = self._buffers[li][1:n].cpu().numpy()
            elif n == 1:
                result[li] = self._buffers[li][0:1].cpu().numpy()
        return result


# =============================================================================
# Step 3: Run baseline and flip generation
# =============================================================================
print(f"\n{'='*60}")
print("STEP 2: Running baseline and flip generation")
print(f"{'='*60}")

baseline_cap = MLPDeltaCaptureWithFlip(
    model, ALL_LAYERS, FLIP_LAYERS, lang_directions,
    max_steps=MAX_NEW_TOKENS + 50, flip_on=False
)
flip_cap = MLPDeltaCaptureWithFlip(
    model, ALL_LAYERS, FLIP_LAYERS, lang_directions,
    max_steps=MAX_NEW_TOKENS + 50, flip_on=True
)

baseline_results = {"zh": [], "en": []}
flip_results = {"zh": [], "en": []}

for lang in ["zh", "en"]:
    prompts = [p[lang] for p in test_problems]
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Baseline
        base = baseline_cap.run_generation(input_ids, max_new_tokens=MAX_NEW_TOKENS)
        baseline_results[lang].append(base)

        # Flip
        flipped = flip_cap.run_generation(input_ids, max_new_tokens=MAX_NEW_TOKENS)
        flip_results[lang].append(flipped)

        if (pi + 1) % 5 == 0:
            print(f"  {lang} {pi+1}/{len(prompts)} done ({time.time()-t0:.0f}s)")


# =============================================================================
# Analysis
# =============================================================================
print(f"\n{'='*60}")
print("ANALYSIS: Cross-layer cosine — baseline vs flip")
print(f"{'='*60}")


def cross_layer_cosine_gen(results_list, layers, tok_range=None):
    out = {}
    for i in range(1, len(layers)):
        prev_li, curr_li = layers[i-1], layers[i]
        cos_vals = []
        for gen in results_list:
            if prev_li not in gen or curr_li not in gen:
                continue
            d_prev = gen[prev_li]
            d_curr = gen[curr_li]
            n_tok = min(d_prev.shape[0], d_curr.shape[0])
            if n_tok == 0:
                continue
            if tok_range:
                s, e = min(tok_range[0], n_tok), min(tok_range[1], n_tok)
                if s >= e:
                    continue
                d_prev, d_curr = d_prev[s:e], d_curr[s:e]
            norms_p = np.linalg.norm(d_prev, axis=1, keepdims=True) + 1e-10
            norms_c = np.linalg.norm(d_curr, axis=1, keepdims=True) + 1e-10
            per_tok = np.sum((d_prev / norms_p) * (d_curr / norms_c), axis=1)
            cos_vals.append(float(per_tok.mean()))
        if cos_vals:
            out[f"L{prev_li}->L{curr_li}"] = {
                "cos_mean": float(np.mean(cos_vals)),
                "cos_std": float(np.std(cos_vals)),
                "n": len(cos_vals),
            }
    return out


def zone_summary(profile, layers):
    vals = []
    for i in range(1, len(layers)):
        key = f"L{layers[i-1]}->L{layers[i]}"
        if key in profile:
            vals.append(profile[key]["cos_mean"])
    return float(np.mean(vals)) if vals else None


all_base = baseline_results["zh"] + baseline_results["en"]
all_flip = flip_results["zh"] + flip_results["en"]

adv_layers = list(range(9, 18))
coop_layers = list(range(18, 22))
ramp_layers = list(range(22, 27))

# Full profiles
base_all = cross_layer_cosine_gen(all_base, ALL_LAYERS)
flip_all = cross_layer_cosine_gen(all_flip, ALL_LAYERS)
base_early = cross_layer_cosine_gen(all_base, ALL_LAYERS, tok_range=(0, 10))
flip_early = cross_layer_cosine_gen(all_flip, ALL_LAYERS, tok_range=(0, 10))
base_late = cross_layer_cosine_gen(all_base, ALL_LAYERS, tok_range=(80, 128))
flip_late = cross_layer_cosine_gen(all_flip, ALL_LAYERS, tok_range=(80, 128))

print("\n--- ZONE COMPARISON ---")
print(f"\n{'Zone':<25} {'Base(all)':<12} {'Flip(all)':<12} {'Delta':<12} {'Base(early)':<12} {'Flip(early)':<12} {'Base(late)':<12} {'Flip(late)':<12}")
print("-" * 110)

results_table = []
for zone_name, zlayers in [("Adversarial L9-L17", adv_layers),
                             ("Cooperative L18-L21", coop_layers),
                             ("Ramp L22-L26", ramp_layers)]:
    ba = zone_summary(base_all, zlayers)
    fa = zone_summary(flip_all, zlayers)
    be = zone_summary(base_early, zlayers)
    fe = zone_summary(flip_early, zlayers)
    bl = zone_summary(base_late, zlayers)
    fl = zone_summary(flip_late, zlayers)
    delta = (fa - ba) if ba is not None and fa is not None else None

    print(f"  {zone_name:<23} {ba:+.4f}      {fa:+.4f}      {delta:+.4f}      {be:+.4f}      {fe:+.4f}      {bl:+.4f}      {fl:+.4f}")

    results_table.append({
        "zone": zone_name,
        "baseline_all": ba, "flip_all": fa, "delta_all": delta,
        "baseline_early": be, "flip_early": fe,
        "baseline_late": bl, "flip_late": fl,
    })

# Per-layer detail for the cooperative zone
print(f"\n--- COOPERATIVE ZONE DETAIL (L17-L22) ---")
print(f"{'Pair':<15} {'Base(all)':<12} {'Flip(all)':<12} {'Delta':<12}")
print("-" * 55)
for pair_key in ["L17->L18", "L18->L19", "L19->L20", "L20->L21", "L21->L22"]:
    b = base_all.get(pair_key, {}).get("cos_mean", None)
    f = flip_all.get(pair_key, {}).get("cos_mean", None)
    if b is not None and f is not None:
        print(f"  {pair_key:<13} {b:+.4f}      {f:+.4f}      {f-b:+.4f}")


# =============================================================================
# Save
# =============================================================================
output = {
    "experiment": "Does MLP language flip restore cooperative zone during generation?",
    "model": MODEL_NAME,
    "n_problems": N_TEST,
    "max_new_tokens": MAX_NEW_TOKENS,
    "flip_layers": FLIP_LAYERS,
    "method": "Compare cross-layer MLP delta cosine with and without language-direction flip at L9-L26 during generation.",
    "baseline_all_profile": base_all,
    "flip_all_profile": flip_all,
    "baseline_early_profile": base_early,
    "flip_early_profile": flip_early,
    "baseline_late_profile": base_late,
    "flip_late_profile": flip_late,
    "zone_comparison": results_table,
    "runtime_seconds": time.time() - t0,
}

with open("output/exp_flip_restores_cooperation.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to output/exp_flip_restores_cooperation.json")
print(f"Total runtime: {time.time()-t0:.0f}s")
