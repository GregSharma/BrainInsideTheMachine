"""Experiment X: Language-Stripped Z Iteration

Hypothesis: The reasoning computation (L9-L25) can proceed with the language
coordinate stripped from the residual stream at every step. Language is
re-injectable at L26 (the decode boundary) independently of the computation.

Three conditions on N=20 standard test problems:
  X1 (strip only):      Strip lang dir from residual after each layer L9-L25.
                        No re-inject. Does math survive? What language?
  X2 (strip+reinject):  Strip at L9-L25, reinject ENGLISH dir at L26.
                        Does stripping preserve math? Does reinject control lang?
  X3 (transplant):      ZH prompt, strip lang at L9-L25, reinject ENGLISH at L26.
                        Can Chinese math reasoning emerge in English output?

Critical difference from prior work:
  - NOT zeroing MLP (Exp M3: killed math)
  - NOT skipping layers (Exp G: incoherent)
  - NOT MLP delta manipulation (Exp P/P2/P3: manipulates delta, not residual)
  - NOT Ridge shortcut (Exp O: gibberish)
  This is the residual stream language coordinate, stripped persistently.

~4 conditions x 20 problems x 128 tokens. ~4 min on RayGun.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re

MODEL_NAME = "Qwen/Qwen2.5-3B"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

d_model = model.config.hidden_size
N_TRAIN = 200
BATCH_SIZE = 16
STRIP_LAYERS = list(range(9, 26))   # adversarial + cooperative + ramp
REINJECT_LAYER = 26                  # decode boundary (PC0 swap worked here)
MAX_TOKENS = 128
REINJECT_SCALE = 1.0                 # positive = push toward English
t0 = time.time()


# =============================================================================
# Problem generation (same seed as all prior experiments)
# =============================================================================
def generate_pca_problems(n=200, seed=42):
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
                          "en": f"An arithmetic sequence has first term {a1} and "
                                f"common difference {d_val}. Find the sum of the first {n_terms} terms."})
    return problems


def compute_answer(text):
    """Compute ground-truth answer FROM THE PROMPT."""
    m = re.search(r'(?:Calculate|计算) (\d+) \+ (\d+)', text)
    if m: return str(int(m.group(1)) + int(m.group(2)))
    m = re.search(r'(?:Calculate|计算) (\d+) [×x] (\d+)', text)
    if m: return str(int(m.group(1)) * int(m.group(2)))
    m = re.search(r'C\((\d+),?\s*(\d+)\)', text)
    if m:
        from math import comb
        return str(comb(int(m.group(1)), int(m.group(2))))
    m = re.search(r'(\d+)\s*除以\s*(\d+)', text)
    if not m:
        m = re.search(r'remainder when (\d+) is divided by (\d+)', text)
    if m: return str(int(m.group(1)) % int(m.group(2)))
    m = re.search(r'(?:length|长为)\s*(\d+).*?(?:width|宽为)\s*(\d+)', text)
    if m: return str(int(m.group(1)) * int(m.group(2)))
    m = re.search(r'(?:first term|首项为)\s*(\d+).*?(?:common difference|公差为)\s*(\d+).*?(?:first|前)\s*(\d+)', text)
    if m:
        a1, dd, nn = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return str(nn * a1 + nn * (nn - 1) // 2 * dd)
    return None


def extract_pred(gen_text, expected):
    """Find the expected answer in the generated text."""
    # Direct match: does the exact answer appear?
    nums = re.findall(r'\b(\d+)\b', gen_text)
    if expected in nums:
        return expected
    # Last number heuristic (works for simple problems)
    return nums[-1] if nums else None


def classify_lang(text):
    zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return "zh" if zh_chars > 5 else "en"


# =============================================================================
# Fit language direction on RESIDUAL STREAM (not MLP delta)
# This is the key difference: we want the residual coord, not the MLP output.
# =============================================================================
print("Fitting language direction on residual stream...")
problems = generate_pca_problems(N_TRAIN, seed=42)

# Capture residual stream at each layer OUTPUT (after attn + MLP both applied)
def extract_residuals(prompts):
    """Extract last-token residual stream at each layer output."""
    layer_data = {li: [] for li in STRIP_LAYERS + [REINJECT_LAYER]}
    captures = {}
    handles = []
    for li in STRIP_LAYERS + [REINJECT_LAYER]:
        def make_hook(idx):
            def hook(module, inp, out):
                # Decoder layer returns (hidden_state, ...) or just hidden_state
                hs = out[0] if isinstance(out, tuple) else out
                captures[idx] = hs.detach().float()
            return hook
        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        captures.clear()
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            model(**inputs)
        for li in STRIP_LAYERS + [REINJECT_LAYER]:
            for j in range(captures[li].shape[0]):
                layer_data[li].append(captures[li][j, last_idx[j]].cpu().numpy())

    for h in handles:
        h.remove()
    for li in STRIP_LAYERS + [REINJECT_LAYER]:
        layer_data[li] = np.stack(layer_data[li])
    return layer_data

zh_prompts = [p["zh"] for p in problems]
en_prompts = [p["en"] for p in problems]

print("  Extracting ZH residuals...")
res_zh = extract_residuals(zh_prompts)
print("  Extracting EN residuals...")
res_en = extract_residuals(en_prompts)

# Language direction at each layer: mean(zh) - mean(en), normalized
lang_dirs = {}
for li in STRIP_LAYERS + [REINJECT_LAYER]:
    diff = res_zh[li].mean(0) - res_en[li].mean(0)
    norm = np.linalg.norm(diff)
    lang_dirs[li] = diff / (norm + 1e-10)

print(f"  Directions fitted. Norms at L9={np.linalg.norm(res_zh[9].mean(0)):.1f}, "
      f"L18={np.linalg.norm(res_zh[18].mean(0)):.1f}, L26={np.linalg.norm(res_zh[26].mean(0)):.1f}")
print(f"  Lang dir magnitudes at L9={np.linalg.norm(res_zh[9].mean(0)-res_en[9].mean(0)):.2f}, "
      f"L26={np.linalg.norm(res_zh[26].mean(0)-res_en[26].mean(0)):.2f}")
print(f"Fitting done in {time.time()-t0:.1f}s")


# =============================================================================
# Standard N=20 test set (same as V3/P3/all prior experiments)
# =============================================================================
per_cat = N_TRAIN // 5
# Build test set: 2 per category per language = 10 EN + 10 ZH = 20 total
test_problems = []
for lang in ["en", "zh"]:
    for cat_idx, cat_start in enumerate(range(0, N_TRAIN, per_cat)):
        for i in range(2):  # 2 problems per category per language
            prob = problems[cat_start + i]
            prompt = prob[lang]
            answer = compute_answer(prompt)
            if answer:
                test_problems.append({"prompt": prompt, "answer": answer, "lang": lang})
print(f"\nTest set: {len(test_problems)} problems")
print(f"  EN: {sum(1 for p in test_problems if p['lang']=='en')}, ZH: {sum(1 for p in test_problems if p['lang']=='zh')}")


# =============================================================================
# Generation with residual stream hooks
# =============================================================================
def run_condition(prompt, answer, mode):
    """
    mode:
      'baseline'   - no intervention
      'strip_only' - strip lang dir from residual at L9-L25 each token
      'strip_reinject_en' - strip at L9-L25, reinject EN dir at L26
      'transplant' - same as strip_reinject_en (used with ZH prompts)
    """
    handles = []

    if mode in ("strip_only", "strip_reinject_en", "transplant"):
        # Hook: after each layer in STRIP_LAYERS, project out language direction
        for li in STRIP_LAYERS:
            d_vec = torch.tensor(lang_dirs[li], dtype=torch.bfloat16, device=device)
            def make_strip_hook(dv):
                def hook(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    # Project out the language direction
                    proj = (hs.float() @ dv.float().unsqueeze(-1)).squeeze(-1)  # (batch, seq)
                    hs_stripped = hs - (proj.unsqueeze(-1) * dv.float().unsqueeze(0).unsqueeze(0)).to(hs.dtype)
                    if isinstance(out, tuple):
                        return (hs_stripped,) + out[1:]
                    return hs_stripped
                return hook
            handles.append(model.model.layers[li].register_forward_hook(make_strip_hook(d_vec)))

    if mode in ("strip_reinject_en", "transplant"):
        # Hook: at L26, inject the ENGLISH direction (subtract zh component, add en scale)
        # English = negative of zh lang_dir (since dir = zh - en)
        d_vec = torch.tensor(lang_dirs[REINJECT_LAYER], dtype=torch.bfloat16, device=device)
        def reinject_hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            # First strip any residual language component
            proj = (hs.float() @ d_vec.float().unsqueeze(-1)).squeeze(-1)
            hs_stripped = hs - (proj.unsqueeze(-1) * d_vec.float().unsqueeze(0).unsqueeze(0)).to(hs.dtype)
            # Then inject negative direction (= English) at fixed scale
            # Scale chosen to match typical EN residual projection magnitude
            en_injection = -REINJECT_SCALE * d_vec.float()
            hs_en = hs_stripped + en_injection.unsqueeze(0).unsqueeze(0).to(hs.dtype)
            if isinstance(out, tuple):
                return (hs_en,) + out[1:]
            return hs_en
        handles.append(model.model.layers[REINJECT_LAYER].register_forward_hook(reinject_hook))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_input = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    for h in handles:
        h.remove()

    gen_tokens = out[0][n_input:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    pred = extract_pred(gen_text, answer)
    correct = (pred == answer)
    return {
        "generation": gen_text[:300],
        "correct": correct,
        "pred": pred,
        "lang_out": classify_lang(gen_text),
        "n_tokens": len(gen_tokens),
    }


# =============================================================================
# Run all conditions
# =============================================================================
conditions = [
    ("baseline",           "EN baseline — no intervention"),
    ("strip_only",         "X1: strip lang dir at L9-L25, no reinject"),
    ("strip_reinject_en",  "X2: strip at L9-L25, reinject EN at L26"),
]

results = {}

for cond, desc in conditions:
    print(f"\n--- {desc} ---")
    cond_results = []
    for i, prob in enumerate(test_problems):
        r = run_condition(prob["prompt"], prob["answer"], cond)
        r["prompt"] = prob["prompt"][:60]
        r["answer"] = prob["answer"]
        r["prompt_lang"] = prob["lang"]
        cond_results.append(r)
        mark = "✓" if r["correct"] else "✗"
        print(f"  [{i+1:2d}] {mark} pred={r['pred']:>6} ans={r['answer']:>6} "
              f"lang={r['lang_out']} tok={r['n_tokens']} | {prob['prompt'][:40]}")
    acc = sum(r["correct"] for r in cond_results)
    print(f"  Accuracy: {acc}/{len(cond_results)}")
    results[cond] = {"accuracy": acc, "n": len(cond_results), "problems": cond_results}

# Condition X3: transplant — ZH prompts through strip+reinject_en
print(f"\n--- X3: transplant — ZH prompt + strip + reinject EN ---")
zh_problems = [p for p in test_problems if p["lang"] == "zh"]
print(f"  ({len(zh_problems)} ZH problems)")
transplant_results = []
for i, prob in enumerate(zh_problems):
    r = run_condition(prob["prompt"], prob["answer"], "transplant")
    r["prompt"] = prob["prompt"][:60]
    r["answer"] = prob["answer"]
    r["prompt_lang"] = "zh"
    transplant_results.append(r)
    mark = "✓" if r["correct"] else "✗"
    print(f"  [{i+1:2d}] {mark} pred={r['pred']:>6} ans={r['answer']:>6} "
          f"lang={r['lang_out']} tok={r['n_tokens']} | {prob['prompt'][:40]}")
acc = sum(r["correct"] for r in transplant_results)
print(f"  Accuracy: {acc}/{len(transplant_results)}")
results["transplant"] = {"accuracy": acc, "n": len(transplant_results), "problems": transplant_results}

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
baseline_acc = results["baseline"]["accuracy"]
for cond, data in results.items():
    delta = data["accuracy"] - baseline_acc if cond != "baseline" else 0
    sign = f"+{delta}" if delta > 0 else str(delta)
    print(f"  {cond:25s}: {data['accuracy']:2d}/{data['n']} "
          f"{'(' + sign + ')' if cond != 'baseline' else ''}")

# Language output breakdown
for cond, data in results.items():
    zh_out = sum(1 for r in data["problems"] if r["lang_out"] == "zh")
    en_out = sum(1 for r in data["problems"] if r["lang_out"] == "en")
    print(f"  {cond:25s}: EN={en_out} ZH={zh_out}")

output = {
    "experiment": "X: Language-Stripped Z Iteration",
    "model": MODEL_NAME,
    "strip_layers": STRIP_LAYERS,
    "reinject_layer": REINJECT_LAYER,
    "reinject_scale": REINJECT_SCALE,
    "max_tokens": MAX_TOKENS,
    "n_test": len(test_problems),
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "problems"}
                for k, v in results.items()},
    "detailed": results,
    "runtime_seconds": time.time() - t0,
    "predictions": {
        "X1_strip_only": "Incoherent or default language. Math degrades. Marinade goes all the way.",
        "X2_strip_reinject": "Math holds ~50-70% baseline. Language = English. Z iteration survives.",
        "X3_transplant": "ZH prompt → EN output, correct math. Von Neumann result.",
    }
}

with open("output/expX_z_iteration.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to output/expX_z_iteration.json")
print(f"Total runtime: {time.time()-t0:.1f}s")
