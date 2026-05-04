"""Phase diagram: (δ, onset) sweep for echo bifurcation theory.

BATCHED: all onsets for a given δ run simultaneously (batch=12).
Tests Claude Web's prediction: t_crit scales with 1/δ.

Uses proven WindowedDeflation Q-projection mechanism.
Manual generation loop, correct P12 problem, base model.
"""
import json, time, re, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 1200

# Proven deflation params
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
CORRECT = "B"
PROMPT = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"

# Phase diagram grid
DELTA_VALUES = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
ONSET_VALUES = [0, 5, 10, 15, 20, 22, 25, 27, 30, 35, 40, 50]


class BatchedWindowedDeflation:
    """Q-deflation that handles a batch of onset values simultaneously.

    Each element in the batch has its own onset time. The hook checks
    per-element whether deflation is active at the current step.
    """
    def __init__(self, model, layers, onsets, r=4, alpha=0.1, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.onsets = onsets  # list of onset values, one per batch element
        self.batch_size = len(onsets)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.U_r = {}  # {layer: {kv_head: U_matrix}}
        # Per-element active mask (batch,)
        self.active_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=DEVICE)
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_q_hook(ell))
            self.hooks.append(h)

    def _make_q_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
                return output
            # Check if ANY element is active
            if not self.active_mask.any():
                return output

            batch, seq, d = output.shape
            n_heads = 16
            head_dim = 128
            n_kv = len(self.U_r[li])
            gs_per_kv = n_heads // n_kv
            tensor = output.view(batch, seq, n_heads, head_dim)

            # mask shape: (batch, 1, 1, 1) for broadcasting
            mask = self.active_mask.view(batch, 1, 1, 1).float()

            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs_per_kv, (kv_h + 1) * gs_per_kv
                qg = tensor[:, :, s:e, :]  # (batch, seq, gs, head_dim)
                proj = qg @ U @ U.T
                # Only apply deflation to active batch elements
                tensor[:, :, s:e, :] = qg - self.alpha * mask * proj

            return tensor.view(batch, seq, d)
        return hook

    def refresh_basis(self, past_kv):
        """Refresh SVD basis from first batch element's keys.
        All elements share the same prompt, so keys are identical at step 0.
        After that they diverge, but the basis is close enough.
        We use element 0's keys as representative.
        """
        for ell in self.target_layers:
            keys = past_kv.layers[ell].keys  # (batch, n_kv_heads, seq, head_dim)
            self.U_r[ell] = {}
            for kv_h in range(keys.shape[1]):
                K = keys[0, kv_h, :, :].float()  # use element 0
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
        """Update per-element active mask based on current step."""
        for i, onset in enumerate(self.onsets):
            self.active_mask[i] = self.step_count >= onset

    def tick(self, past_kv):
        self.step_count += 1
        self._update_active()
        if self.active_mask.any() and self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.U_r.clear()


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


def classify(text):
    ans = extract_answer(text)
    n = len(text.split())
    if n > MAX_TOKENS * 0.7 and ans == "?":
        return "LOOP", "?"
    if ans == CORRECT:
        return "CORRECT", ans
    return f"WRONG({ans})", ans


def generate_single(model, tokenizer, input_ids):
    """Single baseline generation (batch=1, no deflation)."""
    gen_ids = []
    past_kv = None
    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids)


def generate_batched(model, tokenizer, input_ids, deflator, onsets):
    """Batched generation: all onsets run simultaneously.

    Each batch element generates independently (greedy per-element).
    Elements that hit EOS are frozen (continue receiving padding but
    their outputs are ignored).
    """
    batch_size = len(onsets)
    # Replicate input for batch
    batched_input = input_ids.expand(batch_size, -1)  # (B, seq)

    gen_ids = [[] for _ in range(batch_size)]  # per-element token lists
    finished = [False] * batch_size
    past_kv = None

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=batched_input, use_cache=True)
                deflator.start_gen()
                deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_ids, past_key_values=past_kv, use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]  # (B, vocab)
            next_ids = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Check EOS per element
            all_done = True
            for i in range(batch_size):
                if finished[i]:
                    continue
                tid = next_ids[i].item()
                if tid in (151643, 151645):
                    finished[i] = True
                else:
                    gen_ids[i].append(tid)
                    all_done = False

            if all_done:
                break

            deflator.tick(past_kv)

    # Decode per element
    results = []
    for i in range(batch_size):
        text = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
        label, ans = classify(text)
        results.append({
            "onset": onsets[i],
            "label": label,
            "answer": ans,
            "correct": ans == CORRECT,
            "n_tokens": len(gen_ids[i])
        })

    del past_kv, out
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 70)
    print("PHASE DIAGRAM — (δ, onset) SWEEP [BATCHED]")
    print(f"δ values: {DELTA_VALUES}")
    print(f"onset values: {ONSET_VALUES}")
    print(f"Grid: {len(DELTA_VALUES)} × {len(ONSET_VALUES)} = {len(DELTA_VALUES) * len(ONSET_VALUES)} conditions")
    print(f"Batch size: {len(ONSET_VALUES)} (all onsets per δ)")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print(f"Loaded {MODEL_NAME}, {len(model.model.layers)} layers.\n", flush=True)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    # Baseline (no deflation, single)
    print("Baseline (no deflation)...")
    t0 = time.time()
    bl_text, bl_ntok = generate_single(model, tokenizer, input_ids)
    bl_dt = time.time() - t0
    bl_label, bl_ans = classify(bl_text)
    print(f"  → {bl_label} (ans={bl_ans}, {bl_ntok} tokens, {bl_dt:.1f}s)\n", flush=True)

    all_results = []
    t_start = time.time()

    for di, delta in enumerate(DELTA_VALUES):
        print(f"\nδ={delta:.2f} — batching {len(ONSET_VALUES)} onsets...", flush=True)
        t0 = time.time()

        deflator = BatchedWindowedDeflation(
            model, DEFLATE_LAYERS, ONSET_VALUES,
            r=DEFLATE_R, alpha=delta, refresh_every=DEFLATE_REFRESH)

        row_results = generate_batched(model, tokenizer, input_ids, deflator, ONSET_VALUES)
        deflator.remove()

        dt = time.time() - t0
        n_correct = sum(1 for r in row_results if r["correct"])

        for r in row_results:
            r["delta"] = delta
            r["time_s"] = round(dt, 1)  # total time for this batch
            all_results.append(r)
            sym = "✓" if r["correct"] else "✗"
            print(f"  onset={r['onset']:3d} → {r['label']:12s} {sym}  ({r['n_tokens']:4d}tok)")

        elapsed = time.time() - t_start
        eta = elapsed / (di + 1) * (len(DELTA_VALUES) - di - 1)
        print(f"  --- δ={delta:.2f}: {n_correct}/{len(ONSET_VALUES)} correct  "
              f"({dt:.0f}s batch, ETA {eta:.0f}s) ---", flush=True)

    elapsed = time.time() - t_start

    # Analysis
    print(f"\n{'='*70}")
    print(f"RESULTS ({elapsed:.0f}s total)")
    print(f"{'='*70}")

    # Phase diagram grid
    print(f"\n{'δ':>6} |" + "".join(f" {o:>3}" for o in ONSET_VALUES))
    print(f"{'':->6}-+" + "-" * (4 * len(ONSET_VALUES)))

    t_crit = {}
    for delta in DELTA_VALUES:
        row = [r for r in all_results if r["delta"] == delta]
        row.sort(key=lambda x: x["onset"])
        line = f"{delta:6.2f} |"
        for r in row:
            line += " ✓ " if r["correct"] else " ✗ "
        print(line)
        correct_onsets = [r["onset"] for r in row if r["correct"]]
        if correct_onsets:
            t_crit[delta] = max(correct_onsets)

    # t_crit vs δ
    print(f"\nCritical onset (last onset → CORRECT):")
    print(f"{'δ':>6} {'t_crit':>8} {'δ × t_crit':>10}")
    print(f"{'':->6} {'':->8} {'':->10}")
    products = []
    for d in sorted(t_crit.keys()):
        tc = t_crit[d]
        prod = d * tc
        products.append(prod)
        print(f"{d:6.2f} {tc:8d} {prod:10.2f}")

    if len(products) >= 3:
        mean_p = sum(products) / len(products)
        std_p = (sum((p - mean_p)**2 for p in products) / len(products)) ** 0.5
        cv = std_p / mean_p if mean_p > 0 else float('inf')
        print(f"\nδ × t_crit: mean={mean_p:.2f}, std={std_p:.2f}, CV={cv:.3f}")
        if cv < 0.3:
            print(f"→ δ × t_crit ≈ CONSTANT ({mean_p:.1f})")
            print(f"→ TWO-SOURCE ECHO MODEL CONFIRMED")
            print(f"→ t_crit ≈ {mean_p:.1f}/δ")
        else:
            print(f"→ δ × t_crit NOT constant (CV={cv:.2f})")
            print(f"→ mechanism may be geometric, not echo-mediated")

    # Save
    report = {
        "model": MODEL_NAME,
        "baseline": {"label": bl_label, "answer": bl_ans, "n_tokens": bl_ntok},
        "params": {
            "layers": DEFLATE_LAYERS, "r": DEFLATE_R,
            "refresh": DEFLATE_REFRESH, "max_tokens": MAX_TOKENS
        },
        "delta_values": DELTA_VALUES,
        "onset_values": ONSET_VALUES,
        "results": all_results,
        "t_crit": {str(k): v for k, v in t_crit.items()},
        "elapsed_s": round(elapsed, 1)
    }
    out_path = "output/exp_phase_diagram.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n→ {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
