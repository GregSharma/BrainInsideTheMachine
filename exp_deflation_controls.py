#!/usr/bin/env python3
"""Deflation controls: robustness, temperature comparison, breadth.

1. p12 robustness: soft_a0.1 with temp=0.01, 3 seeds
2. p12 temperature control: temp=0.3/0.5, no deflation, 3 seeds each
3. Breadth: 5 problems × baseline vs soft_a0.1

The critical question: does temperature alone break the loop and find -3/2,
or does deflation steer toward the correct basin specifically?
"""
import json, time, os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
REFRESH_EVERY = 25

SYS = (
    "You are solving an AMC 12A multiple choice math problem. "
    "Think step by step, show your work, then clearly state your "
    "final answer as (A), (B), (C), (D), or (E)."
)

PROBLEMS = {
    "p3": {
        "text": (
            "A team of students is going to compete against a team of teachers "
            "in a trivia contest. The total number of students and teachers is 15. "
            "Ash, a cousin of one of the students, wants to join the contest. "
            "If Ash plays with the students, the average age on that team will "
            "increase from 12 to 14. If Ash plays with the teachers, the average "
            "age on that team will decrease from 55 to 52. How old is Ash?\n\n"
            "(A) 28  (B) 29  (C) 30  (D) 32  (E) 33"
        ),
        "answer": "A",
    },
    "p4": {
        "text": (
            "Agnes writes the following four statements on a blank piece of paper.\n\n"
            "- At least one of these statements is true.\n"
            "- At least two of these statements are true.\n"
            "- At least two of these statements are false.\n"
            "- At least one of these statements is false.\n\n"
            "Each statement is either true or false. How many false statements "
            "did Agnes write on the paper?\n\n"
            "(A) 0  (B) 1  (C) 2  (D) 3  (E) 4"
        ),
        "answer": "B",
    },
    "p7": {
        "text": (
            "In a certain alien world, the maximum running speed v of an organism "
            "is dependent on its number of toes n and number of eyes m. The "
            "relationship can be expressed as v = k * n^a * m^b centimeters per "
            "hour, where k, a, and b are integer constants. In a population where "
            "all organisms have 5 toes, log v = 4 + 2 log m; and in a population "
            "where all organisms have 25 eyes, log v = 4 + 4 log n, where the "
            "logarithms are base 10. What is k + a + b?\n\n"
            "(A) 20  (B) 21  (C) 22  (D) 23  (E) 24"
        ),
        "answer": "C",
    },
    "p9": {
        "text": (
            "Let w be the complex number 2 + i, where i = sqrt(-1). What real "
            "number r has the property that r, w, and w^2 are three collinear "
            "points in the complex plane?\n\n"
            "(A) 3/4  (B) 1  (C) 7/5  (D) 3/2  (E) 5/3"
        ),
        "answer": "E",
    },
    "p12": {
        "text": (
            "The harmonic mean of a collection of numbers is the reciprocal of the "
            "arithmetic mean of the reciprocals of the numbers in the collection. "
            "For example, the harmonic mean of 4, 4, and 5 is\n\n"
            "1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n"
            "What is the harmonic mean of all the real roots of the 4050th degree "
            "polynomial\n\n"
            r"\prod_{k=1}^{2025} (kx^2 - 4x - 3) = "
            "(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)..."
            "(2025x^2 - 4x - 3)?\n\n"
            "(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3"
        ),
        "answer": "B",
    },
}


def make_prompt(text):
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


class SoftDeflation:
    """q_new = q - alpha * V V^T q at target layers."""

    def __init__(self, model, layers, r=4, alpha=0.1, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.U_r = {}
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_hook(ell)
            )
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
                return output
            q = output
            batch, seq, d = q.shape
            hd, n_q, n_kv = 128, 16, 2
            gs = n_q // n_kv
            q = q.view(batch, seq, n_q, hd)
            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs, (kv_h + 1) * gs
                qg = q[:, :, s:e, :]
                proj = qg @ U @ U.T
                q[:, :, s:e, :] = qg - self.alpha * proj
            return q.view(batch, seq, d)
        return hook

    def refresh_basis(self, past_kv):
        for ell in self.target_layers:
            keys = past_kv.layers[ell].keys
            self.U_r[ell] = {}
            for kv_h in range(keys.shape[1]):
                K = keys[0, kv_h, :, :].float()
                if K.shape[0] < self.r:
                    continue
                _, _, Vh = torch.linalg.svd(K, full_matrices=False)
                self.U_r[ell][kv_h] = Vh[:self.r, :].T.contiguous().to(
                    DEVICE, dtype=torch.float16
                )

    def start_gen(self):
        self.is_generating = True
        self.step_count = 0

    def tick(self, past_kv):
        self.step_count += 1
        if self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.U_r.clear()


def run_one(model, tokenizer, prompt, temp=0, seed=None, deflator=None):
    """Run one generation. Returns (text, n_tokens, time_s)."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None
    t0 = time.time()

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                if deflator:
                    deflator.start_gen()
                    deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv,
                           use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]

            if temp > 0:
                probs = F.softmax(logits / temp, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            if deflator:
                deflator.tick(past_kv)

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids), round(dt, 1)


def extract_answer(text):
    """Pull (A)-(E) from the tail of the output."""
    import re
    # Look for boxed answer first
    m = re.findall(r'\\boxed\{[^}]*\b([A-E])\b[^}]*\}', text)
    if m:
        return m[-1]
    m = re.findall(r'\\boxed\{[^}]*(-?\d+/\d+)[^}]*\}', text)
    if m:
        val = m[-1]
        mapping = {"-5/3": "A", "-3/2": "B", "-6/5": "C", "-5/6": "D", "-2/3": "E",
                   "5/3": "E", "3/4": "A", "7/5": "C", "3/2": "D",
                   "28": "A", "29": "B", "30": "C", "32": "D", "33": "E"}
        return mapping.get(val, "?")
    # Look for explicit (A)-(E)
    m = re.findall(r'\(([A-E])\)', text[-300:])
    if m:
        return m[-1]
    # Look for answer is X
    m = re.findall(r'answer is.*?([A-E])\b', text[-300:], re.IGNORECASE)
    if m:
        return m[-1]
    return "?"


print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
)
model.eval()
print(f"Loaded. {len(model.model.layers)} layers.\n", flush=True)

# ============================================================
# PART 1: p12 control battery
# ============================================================
print("=" * 70)
print("PART 1: p12 CONTROL BATTERY")
print("=" * 70)

p12_prompt = make_prompt(PROBLEMS["p12"]["text"])

p12_conditions = [
    # (name, temp, deflate, seed)
    ("baseline",       0,    False, None),
    ("soft_a0.1",      0,    True,  None),
    ("soft_s42",       0.01, True,  42),
    ("soft_s123",      0.01, True,  123),
    ("soft_s777",      0.01, True,  777),
    ("temp0.3_s42",    0.3,  False, 42),
    ("temp0.3_s123",   0.3,  False, 123),
    ("temp0.3_s777",   0.3,  False, 777),
    ("temp0.5_s42",    0.5,  False, 42),
    ("temp0.5_s123",   0.5,  False, 123),
    ("temp0.5_s777",   0.5,  False, 777),
]

p12_results = []
for cname, temp, do_deflate, seed in p12_conditions:
    print(f"\n--- p12 / {cname} ---", flush=True)
    deflator = None
    if do_deflate:
        deflator = SoftDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R,
                                 alpha=0.1, refresh_every=REFRESH_EVERY)
    text, ntok, dt = run_one(model, tokenizer, p12_prompt,
                             temp=temp, seed=seed, deflator=deflator)
    if deflator:
        deflator.remove()
    ans = extract_answer(text)
    looped = ntok >= 2048
    print(f"    {ntok} tok, {dt}s, ans={ans}, loop={looped}", flush=True)
    print(f"    tail: ...{text[-150:].replace(chr(10), ' ')}", flush=True)
    p12_results.append({
        "condition": cname, "temp": temp, "deflate": do_deflate,
        "seed": seed, "n_tokens": ntok, "time_s": dt,
        "answer": ans, "looped": looped, "output": text,
    })

# ============================================================
# PART 2: 5-problem breadth test
# ============================================================
print(f"\n\n{'='*70}")
print("PART 2: BREADTH TEST (baseline vs soft_a0.1)")
print("=" * 70)

breadth_results = []
for pname in ["p3", "p4", "p7", "p9", "p12"]:
    prompt = make_prompt(PROBLEMS[pname]["text"])
    correct = PROBLEMS[pname]["answer"]
    for cname, do_deflate in [("baseline", False), ("soft_a0.1", True)]:
        # Skip p12 baseline/soft — already have from part 1
        if pname == "p12":
            match = [r for r in p12_results if r["condition"] == cname]
            if match:
                r = match[0]
                breadth_results.append({
                    "problem": pname, "correct": correct,
                    "condition": cname, "n_tokens": r["n_tokens"],
                    "time_s": r["time_s"], "answer": r["answer"],
                    "looped": r["looped"],
                })
                print(f"  {pname}/{cname}: (from part 1) {r['n_tokens']}tok ans={r['answer']}",
                      flush=True)
                continue

        print(f"\n--- {pname} / {cname} ---", flush=True)
        deflator = None
        if do_deflate:
            deflator = SoftDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R,
                                     alpha=0.1, refresh_every=REFRESH_EVERY)
        text, ntok, dt = run_one(model, tokenizer, prompt, deflator=deflator)
        if deflator:
            deflator.remove()
        ans = extract_answer(text)
        looped = ntok >= 2048
        print(f"    {ntok} tok, {dt}s, ans={ans} (correct={correct}), loop={looped}",
              flush=True)
        breadth_results.append({
            "problem": pname, "correct": correct, "condition": cname,
            "n_tokens": ntok, "time_s": dt, "answer": ans, "looped": looped,
            "output": text,
        })

# ============================================================
# SAVE + SUMMARY
# ============================================================
os.makedirs("output", exist_ok=True)
with open("output/exp_deflation_controls.json", "w") as f:
    json.dump({"p12_controls": p12_results, "breadth": breadth_results},
             f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/exp_deflation_controls.json")

print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

print("\n--- P12 CONTROL BATTERY ---")
print(f"{'Condition':20s} {'Tok':>5s} {'Ans':>4s} {'Loop':>5s} {'Correct':>8s}")
for r in p12_results:
    ok = "YES" if r["answer"] == "B" else "no"
    print(f"{r['condition']:20s} {r['n_tokens']:5d} {r['answer']:>4s} "
          f"{'LOOP' if r['looped'] else 'stop':>5s} {ok:>8s}")

print("\n--- BREADTH TEST ---")
print(f"{'Problem':8s} {'Condition':15s} {'Tok':>5s} {'Ans':>4s} {'Correct':>8s} {'Match':>6s}")
for r in breadth_results:
    match = "YES" if r["answer"] == r["correct"] else "no"
    print(f"{r['problem']:8s} {r['condition']:15s} {r['n_tokens']:5d} {r['answer']:>4s} "
          f"{r['correct']:>8s} {match:>6s}")

# Tally
print("\n--- TEMPERATURE vs DEFLATION (p12) ---")
for group, label in [("temp0.3", "temp=0.3"), ("temp0.5", "temp=0.5")]:
    runs = [r for r in p12_results if r["condition"].startswith(group)]
    answers = [r["answer"] for r in runs]
    loops = sum(1 for r in runs if r["looped"])
    print(f"  {label}: answers={answers}, loops={loops}/{len(runs)}")
soft_runs = [r for r in p12_results if r["condition"].startswith("soft")]
soft_answers = [r["answer"] for r in soft_runs]
print(f"  soft_a0.1: answers={soft_answers}")
