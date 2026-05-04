"""Delayed deflation: temporal localization of Q-deflation's mechanism.

Two arms:
  A) DELAYED ONSET: deflation OFF for tokens 0..N, ON for N+.
     If this works at N=200 (after Vieta's formulas), deflation is purely
     a readout protector — it doesn't need to be present during computation.

  B) EARLY CUTOFF: deflation ON for tokens 0..N, OFF for N+.
     Tests whether early deflation alone prevents the loop attractor.

Q-only, alpha=0.1, same params as the known-good condition from dual deflation.
"""
import json, time, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

# Known-good Q-deflation params
Q_ALPHA = 0.1
DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
DEFLATE_REFRESH = 25

SYS = ("You are solving an AMC 12A multiple choice math problem. "
       "Think step by step, show your work, then clearly state your "
       "final answer as (A), (B), (C), (D), or (E).")

P12_TEXT = (
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
)
CORRECT = "B"  # -3/2
PROMPT = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"

# Sweep points
DELAYED_ONSET_N = [0, 50, 100, 150, 200, 300, 500, 750, 1000]
EARLY_CUTOFF_N  = [50, 100, 150, 200, 300, 500, 750, 1000]


class WindowedDeflation:
    """Q-deflation that activates only within a token window.

    active_from..active_until (inclusive). Outside this range, hooks are no-op.
    Set active_until=None for 'from N onward'.
    Set active_from=0 for 'until N'.
    """
    def __init__(self, model, layers, r=4, alpha=0.1, refresh_every=25,
                 active_from=0, active_until=None):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
        self.active_from = active_from
        self.active_until = active_until
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.U_r = {}  # {layer: {kv_head: U_matrix}}
        self.deflation_active = False  # current step active?
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_q_hook(ell))
            self.hooks.append(h)

    def _make_q_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or not self.deflation_active or li not in self.U_r:
                return output
            batch, seq, d = output.shape
            n_heads = 16   # Qwen2.5-3B: 16 Q heads
            head_dim = 128
            n_kv = len(self.U_r[li])
            gs_per_kv = n_heads // n_kv  # 8
            tensor = output.view(batch, seq, n_heads, head_dim)
            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs_per_kv, (kv_h + 1) * gs_per_kv
                qg = tensor[:, :, s:e, :]
                proj = qg @ U @ U.T
                tensor[:, :, s:e, :] = qg - self.alpha * proj
            return tensor.view(batch, seq, d)
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
                    DEVICE, dtype=torch.float16)

    def start_gen(self):
        self.is_generating = True
        self.step_count = 0
        self._update_active()

    def _update_active(self):
        """Check if current step falls within the active window."""
        after_from = self.step_count >= self.active_from
        before_until = (self.active_until is None) or (self.step_count <= self.active_until)
        self.deflation_active = after_from and before_until

    def tick(self, past_kv):
        self.step_count += 1
        self._update_active()
        # Only refresh basis when active and on schedule
        if self.deflation_active and self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.U_r.clear()


def run_p12(model, tokenizer, deflator=None, label=""):
    """Generate P12 with optional windowed deflation.
    Returns (text, n_tokens, time_s, milestones).
    milestones: list of (token_idx, text_so_far_snippet) at key points.
    """
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None
    t0 = time.time()
    milestones = []
    milestone_points = {50, 100, 150, 200, 300, 500}

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                if deflator:
                    deflator.start_gen()
                    deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            if deflator:
                deflator.tick(past_kv)

            # Record milestones
            if len(gen_ids) in milestone_points:
                snippet = tokenizer.decode(gen_ids[-60:], skip_special_tokens=True)
                milestones.append((len(gen_ids), snippet))

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids), round(dt, 1), milestones


def extract_answer(text):
    if not text:
        return "?"
    if re.search(r'\\boxed\{[^}]*-\\frac\{3\}\{2\}', text):
        return "B"
    if re.search(r'\\boxed\{[^}]*-3/2', text):
        return "B"
    m = re.findall(r'\\boxed\{[^}]*\b([A-E])\b[^}]*\}', text)
    if m: return m[-1]
    m = re.findall(r'\\boxed\{[^}]*(-?\d+/\d+)[^}]*\}', text)
    if m:
        mapping = {"-5/3": "A", "-3/2": "B", "-6/5": "C", "-5/6": "D", "-2/3": "E"}
        return mapping.get(m[-1], "?")
    m = re.findall(r'\(([A-E])\)', text[-500:])
    if m: return m[-1]
    if "-3/2" in text[-500:]: return "B"
    m = re.findall(r'answer is.*?([A-E])\b', text[-500:], re.IGNORECASE)
    if m: return m[-1]
    return "?"


def main():
    print("=" * 70)
    print("DELAYED DEFLATION — TEMPORAL LOCALIZATION")
    print("Q-only, alpha=0.1, layers 20-35, r=4, refresh=25")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Loaded {MODEL_NAME}, {n_layers} layers.\n", flush=True)

    results = []

    # ===== ARM A: DELAYED ONSET =====
    print("\n" + "#" * 70)
    print("ARM A: DELAYED ONSET — deflation OFF for 0..N, ON for N+")
    print("If N=200 works, deflation is a readout protector, not computation enabler.")
    print("#" * 70 + "\n", flush=True)

    for delay_n in DELAYED_ONSET_N:
        label = f"delay_{delay_n}"
        print(f"\n{'='*60}", flush=True)
        desc = f"ON from token {delay_n}" if delay_n > 0 else "always ON (control)"
        print(f"  {label}: {desc}", flush=True)
        print(f"{'='*60}", flush=True)

        defl = WindowedDeflation(
            model, DEFLATE_LAYERS, r=DEFLATE_R, alpha=Q_ALPHA,
            refresh_every=DEFLATE_REFRESH,
            active_from=delay_n, active_until=None)

        text, ntok, dt, milestones = run_p12(model, tokenizer, deflator=defl, label=label)
        ans = extract_answer(text)
        looped = ntok >= MAX_TOKENS - 5
        defl.remove()

        status = "CORRECT" if ans == CORRECT else ("LOOP" if looped else f"WRONG({ans})")
        print(f"\n>>> {label}: ans={ans} tok={ntok} {status} t={dt}s", flush=True)

        results.append({
            "arm": "A_delayed_onset",
            "window_param": delay_n,
            "desc": desc,
            "active_from": delay_n,
            "active_until": None,
            "answer": ans,
            "correct": ans == CORRECT,
            "n_tokens": ntok,
            "time_s": dt,
            "looped": looped,
            "milestones": milestones,
            "output_last500": text[-500:] if text else "",
        })

        # Stream progress
        sys.stdout.flush()

    # ===== ARM B: EARLY CUTOFF =====
    print("\n" + "#" * 70)
    print("ARM B: EARLY CUTOFF — deflation ON for 0..N, OFF for N+")
    print("Tests whether early deflation alone prevents the loop attractor.")
    print("#" * 70 + "\n", flush=True)

    for cutoff_n in EARLY_CUTOFF_N:
        label = f"cutoff_{cutoff_n}"
        print(f"\n{'='*60}", flush=True)
        desc = f"ON for tokens 0..{cutoff_n}, then OFF"
        print(f"  {label}: {desc}", flush=True)
        print(f"{'='*60}", flush=True)

        defl = WindowedDeflation(
            model, DEFLATE_LAYERS, r=DEFLATE_R, alpha=Q_ALPHA,
            refresh_every=DEFLATE_REFRESH,
            active_from=0, active_until=cutoff_n)

        text, ntok, dt, milestones = run_p12(model, tokenizer, deflator=defl, label=label)
        ans = extract_answer(text)
        looped = ntok >= MAX_TOKENS - 5
        defl.remove()

        status = "CORRECT" if ans == CORRECT else ("LOOP" if looped else f"WRONG({ans})")
        print(f"\n>>> {label}: ans={ans} tok={ntok} {status} t={dt}s", flush=True)

        results.append({
            "arm": "B_early_cutoff",
            "window_param": cutoff_n,
            "desc": desc,
            "active_from": 0,
            "active_until": cutoff_n,
            "answer": ans,
            "correct": ans == CORRECT,
            "n_tokens": ntok,
            "time_s": dt,
            "looped": looped,
            "milestones": milestones,
            "output_last500": text[-500:] if text else "",
        })

        sys.stdout.flush()

    # ===== BASELINE (no deflation at all) =====
    print(f"\n{'='*60}", flush=True)
    print("  BASELINE: no deflation", flush=True)
    print(f"{'='*60}", flush=True)

    text, ntok, dt, milestones = run_p12(model, tokenizer, deflator=None, label="baseline")
    ans = extract_answer(text)
    looped = ntok >= MAX_TOKENS - 5
    status = "CORRECT" if ans == CORRECT else ("LOOP" if looped else f"WRONG({ans})")
    print(f"\n>>> baseline: ans={ans} tok={ntok} {status} t={dt}s", flush=True)

    results.append({
        "arm": "baseline",
        "window_param": None,
        "desc": "no deflation",
        "active_from": None,
        "active_until": None,
        "answer": ans,
        "correct": ans == CORRECT,
        "n_tokens": ntok,
        "time_s": dt,
        "looped": looped,
        "milestones": milestones,
        "output_last500": text[-500:] if text else "",
    })

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY — TEMPORAL LOCALIZATION OF Q-DEFLATION")
    print("=" * 70)

    print("\nARM A: DELAYED ONSET (deflation starts at token N)")
    print(f"  {'Condition':<20s} {'Answer':>6s} {'Tokens':>6s} {'Status':<12s}")
    for r in results:
        if r["arm"] == "A_delayed_onset":
            st = "CORRECT" if r["correct"] else ("LOOP" if r["looped"] else f"WRONG({r['answer']})")
            print(f"  delay_{r['window_param']:<14d} {r['answer']:>6s} {r['n_tokens']:>6d} {st:<12s}")

    print("\nARM B: EARLY CUTOFF (deflation stops at token N)")
    print(f"  {'Condition':<20s} {'Answer':>6s} {'Tokens':>6s} {'Status':<12s}")
    for r in results:
        if r["arm"] == "B_early_cutoff":
            st = "CORRECT" if r["correct"] else ("LOOP" if r["looped"] else f"WRONG({r['answer']})")
            print(f"  cutoff_{r['window_param']:<13d} {r['answer']:>6s} {r['n_tokens']:>6d} {st:<12s}")

    print("\nBASELINE")
    for r in results:
        if r["arm"] == "baseline":
            st = "CORRECT" if r["correct"] else ("LOOP" if r["looped"] else f"WRONG({r['answer']})")
            print(f"  no_deflation        {r['answer']:>6s} {r['n_tokens']:>6d} {st:<12s}")

    # Determine transition
    arm_a = [r for r in results if r["arm"] == "A_delayed_onset"]
    last_correct_delay = -1
    first_fail_delay = None
    for r in sorted(arm_a, key=lambda x: x["window_param"]):
        if r["correct"]:
            last_correct_delay = r["window_param"]
        elif first_fail_delay is None:
            first_fail_delay = r["window_param"]

    if last_correct_delay >= 0:
        print(f"\n>>> DELAYED ONSET: works up to delay={last_correct_delay} tokens.")
        if first_fail_delay is not None:
            print(f"    Fails at delay={first_fail_delay}. Transition window: [{last_correct_delay}, {first_fail_delay}]")
        else:
            print(f"    Never failed — deflation works even with delay={max(DELAYED_ONSET_N)}!")

    arm_b = [r for r in results if r["arm"] == "B_early_cutoff"]
    any_cutoff_works = any(r["correct"] for r in arm_b)
    if any_cutoff_works:
        min_cutoff = min(r["window_param"] for r in arm_b if r["correct"])
        print(f">>> EARLY CUTOFF: works at cutoff={min_cutoff}+ tokens.")
    else:
        print(">>> EARLY CUTOFF: never works — deflation must persist.")

    out_path = "output/exp_delayed_deflation_p12.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
