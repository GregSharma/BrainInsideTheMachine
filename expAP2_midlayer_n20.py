"""
Exp AP2: Mid-Layer Arithmetic — Scaled to N=20

AP showed L14 average/signed_max = 5/5 perfect, logit avg = 3/3.
Scale up to the full 20-problem test set (first 4 per category × 5 categories).
Focus on the winning methods: average, signed_max, max, controls.
Drop multiply (0/30 confirmed dead in AP).

Also scale generation-time logit averaging to all 20 problems.

On Qwen2.5-3B locally.
"""

import json, sys, re, time
import random as pyrandom
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
device = "cuda"
MAX_NEW_TOKENS = 128
t0 = time.time()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
d_model = model.config.hidden_size
n_layers = model.config.num_hidden_layers  # 36


# ── Problem generator (same as V3/R3/etc) ──────────────────────────────
def generate_pca_problems(n=200, seed=42):
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


def compute_answer(text):
    m = re.search(r'(?:Calculate|计算) (\d+) \+ (\d+)', text)
    if m: return str(int(m.group(1)) + int(m.group(2)))
    m = re.search(r'(?:Calculate|计算) (\d+) [×x] (\d+)', text)
    if m: return str(int(m.group(1)) * int(m.group(2)))
    m = re.search(r'C\((\d+),?\s*(\d+)\)', text)
    if m:
        from math import comb
        return str(comb(int(m.group(1)), int(m.group(2))))
    m = re.search(r'(?:remainder when|除以)\s*(\d+)\s*(?:is divided by|除以)\s*(\d+)', text)
    if not m:
        m = re.search(r'(\d+)\s*除以\s*(\d+)', text)
    if m: return str(int(m.group(1)) % int(m.group(2)))
    m = re.search(r'(?:length|长为)\s*(\d+).*?(?:width|宽为)\s*(\d+)', text)
    if m: return str(int(m.group(1)) * int(m.group(2)))
    m = re.search(r'(?:first term|首项为)\s*(\d+).*?(?:common difference|公差为)\s*(\d+).*?(?:first|前)\s*(\d+)', text)
    if m:
        a1, dd, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return str(n * a1 + n * (n - 1) // 2 * dd)
    return None


# ── Build test set: first 4 per category = 20 problems ─────────────────
all_problems = generate_pca_problems(200, seed=42)
per_cat = 40
TEST_INDICES = []
for cat in range(5):
    for i in range(4):
        TEST_INDICES.append(cat * per_cat + i)

PROBLEMS = []
for idx in TEST_INDICES:
    p = all_problems[idx]
    # Compute ground truth answer
    answer = compute_answer(p["en"]) or compute_answer(p["zh"])
    PROBLEMS.append({"zh": p["zh"], "en": p["en"], "answer": answer, "idx": idx})

print(f"Test set: {len(PROBLEMS)} problems")
for i, p in enumerate(PROBLEMS):
    print(f"  [{i:2d}] {p['en'][:60]:60s} → {p['answer']}")

# Cut layers: focus on the interesting ones
CUT_LAYERS = [9, 14, 18, 22]  # dropped 26, 30 (all dead in AP)


# ── Core: run partial forward and capture hidden state ──────────────────
def get_hidden_at_layer(prompt, cut_layer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach()
        return hook

    handle = model.model.layers[cut_layer].register_forward_hook(make_hook(cut_layer))
    with torch.no_grad():
        out = model(**inputs)
    handle.remove()

    h_full = captured[cut_layer]
    return {
        "h_last_tok": h_full[0, -1, :].float(),
        "h_full_seq": h_full[0].float(),
        "input_ids": inputs["input_ids"][0],
        "n_tokens": inputs["input_ids"].shape[1],
    }


def continue_from_hidden(h_injected, cut_layer, n_new_tokens=MAX_NEW_TOKENS):
    seq_len = h_injected.shape[1]
    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    injection_done = [False]

    def inject_hook(module, input, output):
        if not injection_done[0]:
            injection_done[0] = True
            h = output[0] if isinstance(output, tuple) else output
            injected = h_injected.to(h.dtype).to(h.device)
            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected
        return output

    handle = model.model.layers[cut_layer].register_forward_hook(inject_hook)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=dummy_ids,
                max_new_tokens=n_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        gen_text = tokenizer.decode(out[0][seq_len:], skip_special_tokens=True)
    except Exception as e:
        gen_text = f"ERROR: {e}"
    finally:
        handle.remove()
    return gen_text


def interpolate_sequences(h_a, h_b):
    len_a, len_b = h_a.shape[0], h_b.shape[0]
    target_len = min(len_a, len_b)

    def interp(h, target):
        if h.shape[0] == target:
            return h
        e = h.T.unsqueeze(0).float()
        e_interp = F.interpolate(e, size=target, mode='linear', align_corners=True)
        return e_interp.squeeze(0).T

    return interp(h_a, target_len), interp(h_b, target_len)


# ── Mixing functions (winners only) ────────────────────────────────────
def mix_average(h_zh, h_en):
    return (h_zh + h_en) / 2

def mix_max(h_zh, h_en):
    return torch.max(h_zh, h_en)

def mix_signed_max(h_zh, h_en):
    mask = h_zh.abs() > h_en.abs()
    return torch.where(mask, h_zh, h_en)

def mix_en_only(h_zh, h_en):
    return h_en

def mix_zh_only(h_zh, h_en):
    return h_zh

MIXERS = {
    "average": mix_average,
    "max": mix_max,
    "signed_max": mix_signed_max,
    "zh_control": mix_zh_only,
    "en_control": mix_en_only,
}


# ══════════════════════════════════════════════════════════════════════════
# PART 1: Mid-layer mixing on 20 problems
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP AP2: MID-LAYER ARITHMETIC — N=20")
print("=" * 70)

all_results = {}

for pi, prob in enumerate(PROBLEMS):
    prompt_zh = prob["zh"]
    prompt_en = prob["en"]
    answer = prob["answer"]

    print(f"\n{'─' * 60}")
    print(f"P{pi}: {prompt_en[:60]} (answer={answer})")
    print(f"{'─' * 60}")

    prob_results = {
        "prompt_zh": prompt_zh, "prompt_en": prompt_en,
        "answer": answer, "idx": prob["idx"],
        "conditions": {},
    }

    for cut_layer in CUT_LAYERS:
        zh_data = get_hidden_at_layer(prompt_zh, cut_layer)
        en_data = get_hidden_at_layer(prompt_en, cut_layer)
        h_zh_interp, h_en_interp = interpolate_sequences(
            zh_data["h_full_seq"], en_data["h_full_seq"]
        )
        common_len = h_zh_interp.shape[0]

        for mixer_name, mixer_fn in MIXERS.items():
            cond_name = f"L{cut_layer}_{mixer_name}"
            try:
                h_mixed = mixer_fn(h_zh_interp, h_en_interp)
                h_inject = h_mixed.unsqueeze(0)
                gen_text = continue_from_hidden(h_inject, cut_layer)
                correct = answer in gen_text
                prob_results["conditions"][cond_name] = {
                    "correct": correct,
                    "gen": gen_text[:250],
                    "cut_layer": cut_layer,
                    "mixer": mixer_name,
                    "common_len": common_len,
                }
                mark = "✓" if correct else "✗"
                print(f"    {cond_name:25s}: {mark} — {gen_text[:50]}...")
            except Exception as e:
                prob_results["conditions"][cond_name] = {"error": str(e)}
                print(f"    {cond_name:25s}: ERROR {str(e)[:50]}")

    all_results[f"problem_{pi}"] = prob_results


# ══════════════════════════════════════════════════════════════════════════
# PART 2: Generation-time logit averaging — all 20 problems
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: GENERATION-TIME LOGIT AVERAGING — N=20")
print("=" * 70)

for pi, prob in enumerate(PROBLEMS):
    prompt_zh = prob["zh"]
    prompt_en = prob["en"]
    answer = prob["answer"]

    print(f"\n  P{pi}: {prompt_en[:50]} (answer={answer})")

    zh_ids = tokenizer(prompt_zh, return_tensors="pt").to(device)["input_ids"]
    en_ids = tokenizer(prompt_en, return_tensors="pt").to(device)["input_ids"]

    generated_ids = []

    for step in range(MAX_NEW_TOKENS):
        gen_tensor = torch.tensor(generated_ids, device=device).unsqueeze(0) if generated_ids else None

        if gen_tensor is not None:
            zh_input = torch.cat([zh_ids, gen_tensor], dim=1)
            en_input = torch.cat([en_ids, gen_tensor], dim=1)
        else:
            zh_input = zh_ids
            en_input = en_ids

        with torch.no_grad():
            out_zh = model(input_ids=zh_input)
            out_en = model(input_ids=en_input)

        zh_logits = out_zh.logits[0, -1, :].float()
        en_logits = out_en.logits[0, -1, :].float()
        avg_logits = (zh_logits + en_logits) / 2

        next_token = avg_logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token)

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    correct = answer in gen_text
    mark = "✓" if correct else "✗"
    print(f"    logit_avg: {mark} — {gen_text[:60]}...")

    all_results[f"problem_{pi}"]["gentime_logit_avg"] = {
        "correct": correct,
        "gen": gen_text[:250],
        "n_steps": len(generated_ids),
        "method": "logit_average_zh_en",
    }


# ══════════════════════════════════════════════════════════════════════════
# PART 3: Baselines (pure ZH and pure EN at 128 tokens, for reference)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: BASELINES — N=20")
print("=" * 70)

for pi, prob in enumerate(PROBLEMS):
    for lang in ["zh", "en"]:
        prompt = prob[lang]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None, top_p=None,
            )
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        correct = prob["answer"] in gen_text
        mark = "✓" if correct else "✗"
        all_results[f"problem_{pi}"][f"baseline_{lang}"] = {
            "correct": correct,
            "gen": gen_text[:250],
        }
        if pi < 5 or correct:  # only print first 5 or correct ones
            print(f"  P{pi} {lang.upper()}: {mark} — {gen_text[:50]}...")


# ── Save ────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.0f}s")

output = {
    "experiment": "AP2: Mid-Layer Arithmetic at N=20",
    "model": MODEL_NAME,
    "cut_layers": CUT_LAYERS,
    "mixers": list(MIXERS.keys()),
    "n_problems": len(PROBLEMS),
    "elapsed_s": elapsed,
    "results": all_results,
}

with open(OUTPUT_DIR / "expAP2_midlayer_n20.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print("Saved to output/expAP2_midlayer_n20.json")


# ── Grand Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GRAND SUMMARY")
print("=" * 70)

# Baselines
for lang in ["zh", "en"]:
    n_correct = sum(1 for k, v in all_results.items()
                    if v.get(f"baseline_{lang}", {}).get("correct", False))
    print(f"  baseline_{lang:2s}                 : {n_correct}/20")

# Mid-layer mixing
for cut_layer in CUT_LAYERS:
    for mixer_name in MIXERS:
        cond = f"L{cut_layer}_{mixer_name}"
        n_correct = sum(1 for k, v in all_results.items()
                        if v.get("conditions", {}).get(cond, {}).get("correct", False))
        print(f"  {cond:25s}: {n_correct}/20")

# Logit avg
n_gt = sum(1 for k, v in all_results.items()
           if v.get("gentime_logit_avg", {}).get("correct", False))
print(f"  {'gentime_logit_avg':25s}: {n_gt}/20")

print(f"\nDone. ({elapsed:.0f}s)")
