"""Phase diagram with FIXED BASIS (no refresh during generation).

Tests Ghost's hypothesis: the onset boundary at 25-27 is caused by
the SVD basis refresh contaminating the deflation direction with
loop tokens. If we never refresh, the boundary should disappear.

Batched (safe here: fixed basis = identical for all batch elements).
"""
import json, time, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 1200

DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4

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

DELTA_VALUES = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
ONSET_VALUES = [0, 5, 10, 15, 20, 22, 25, 27, 30, 35, 40, 50]


class FixedBasisDeflation:
    """Q-deflation with basis computed ONCE from prompt KV, never refreshed.

    Batched: per-element onset mask. Basis is shared (identical for all
    elements since it's prompt-only).
    """
    def __init__(self, model, layers, onsets, r=4, alpha=0.1):
        self.model = model
        self.target_layers = set(layers)
        self.onsets = onsets
        self.batch_size = len(onsets)
        self.r = r
        self.alpha = alpha
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.U_r = {}
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
            if not self.active_mask.any():
                return output
            batch, seq, d = output.shape
            n_heads = 16
            head_dim = 128
            n_kv = len(self.U_r[li])
            gs_per_kv = n_heads // n_kv
            tensor = output.view(batch, seq, n_heads, head_dim)
            mask = self.active_mask.view(batch, 1, 1, 1).float()
            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs_per_kv, (kv_h + 1) * gs_per_kv
                qg = tensor[:, :, s:e, :]
                proj = qg @ U @ U.T
                tensor[:, :, s:e, :] = qg - self.alpha * mask * proj
            return tensor.view(batch, seq, d)
        return hook

    def compute_basis_once(self, past_kv):
        """Compute SVD basis from prompt-only KV. Called once, never again."""
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
        print(f"    [fixed basis computed from prompt KV, never refreshing]")

    def start_gen(self):
        self.is_generating = True
        self.step_count = 0
        self._update_active()

    def _update_active(self):
        for i, onset in enumerate(self.onsets):
            self.active_mask[i] = self.step_count >= onset

    def tick(self):
        """Advance step counter. NO basis refresh."""
        self.step_count += 1
        self._update_active()

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


def generate_batched(model, tokenizer, input_ids, deflator, onsets):
    batch_size = len(onsets)
    batched_input = input_ids.expand(batch_size, -1)
    gen_ids = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    past_kv = None

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=batched_input, use_cache=True)
                deflator.compute_basis_once(out.past_key_values)
                deflator.start_gen()
            else:
                out = model(input_ids=next_ids, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_ids = logits.argmax(dim=-1, keepdim=True)
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
            deflator.tick()

    results = []
    for i in range(batch_size):
        text = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
        label, ans = classify(text)
        results.append({
            "onset": onsets[i], "label": label, "answer": ans,
            "correct": ans == CORRECT, "n_tokens": len(gen_ids[i])
        })
    del past_kv, out
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 70)
    print("PHASE DIAGRAM -- FIXED BASIS (no refresh)")
    print("Ghost hypothesis: onset boundary at 25 is caused by basis refresh")
    print("Prediction: boundary should DISAPPEAR with fixed prompt-only basis")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    print(f"Loaded {MODEL_NAME}, {len(model.model.layers)} layers.\n", flush=True)

    all_results = []
    t_start = time.time()

    for di, delta in enumerate(DELTA_VALUES):
        print(f"\ndelta={delta:.2f} -- batching {len(ONSET_VALUES)} onsets...", flush=True)
        t0 = time.time()
        deflator = FixedBasisDeflation(
            model, DEFLATE_LAYERS, ONSET_VALUES, r=DEFLATE_R, alpha=delta)
        row_results = generate_batched(model, tokenizer, input_ids, deflator, ONSET_VALUES)
        deflator.remove()
        dt = time.time() - t0
        n_correct = sum(1 for r in row_results if r["correct"])
        for r in row_results:
            r["delta"] = delta
            r["time_s"] = round(dt, 1)
            all_results.append(r)
            sym = "Y" if r["correct"] else "N"
            print(f"  onset={r['onset']:3d} -> {r['label']:12s} {sym}  ({r['n_tokens']:4d}tok)")
        elapsed = time.time() - t_start
        eta = elapsed / (di + 1) * (len(DELTA_VALUES) - di - 1)
        print(f"  --- delta={delta:.2f}: {n_correct}/{len(ONSET_VALUES)} correct  "
              f"({dt:.0f}s batch, ETA {eta:.0f}s) ---", flush=True)

    elapsed = time.time() - t_start

    # Grid
    print(f"\n{'='*70}")
    print(f"FIXED BASIS RESULTS ({elapsed:.0f}s total)")
    print(f"{'='*70}")
    print(f"\n{'d':>6} |" + "".join(f" {o:>3}" for o in ONSET_VALUES))
    print(f"{'':->6}-+" + "-" * (4 * len(ONSET_VALUES)))

    t_crit = {}
    for delta in DELTA_VALUES:
        row = sorted([r for r in all_results if r["delta"] == delta], key=lambda x: x["onset"])
        line = f"{delta:6.2f} |"
        for r in row:
            line += " Y  " if r["correct"] else " N  "
        print(line)
        correct_onsets = [r["onset"] for r in row if r["correct"]]
        if correct_onsets:
            t_crit[delta] = max(correct_onsets)

    # Compare with refreshing
    print(f"\n{'='*70}")
    print("FIXED vs REFRESHING comparison:")
    try:
        refreshing = json.load(open("output/exp_phase_diagram_sequential.json"))
        ref_r = {(r["delta"], r["onset"]): r["correct"] for r in refreshing["results"]}
        fix_r = {(r["delta"], r["onset"]): r["correct"] for r in all_results}
        disagree = []
        fixed_better = 0
        refresh_better = 0
        for key in sorted(fix_r.keys()):
            if key in ref_r and ref_r[key] != fix_r[key]:
                disagree.append((key, ref_r[key], fix_r[key]))
                if fix_r[key]: fixed_better += 1
                else: refresh_better += 1
        print(f"  Disagreements: {len(disagree)}/{len(fix_r)}")
        print(f"  Fixed wins: {fixed_better}, Refreshing wins: {refresh_better}")
        for (d, o), ref, fix in disagree:
            print(f"    d={d:.2f} onset={o:3d}: refresh={'Y' if ref else 'N'}  fixed={'Y' if fix else 'N'}")

        # Key test: do fixed-basis runs work at onset >= 25?
        fix_late = sum(1 for r in all_results if r["onset"] >= 25 and r["correct"])
        ref_late = sum(1 for r in refreshing["results"] if r["onset"] >= 25 and r["correct"])
        fix_early = sum(1 for r in all_results if r["onset"] < 25 and r["correct"])
        ref_early = sum(1 for r in refreshing["results"] if r["onset"] < 25 and r["correct"])
        print(f"\n  onset < 25:  fixed={fix_early}  refresh={ref_early}")
        print(f"  onset >= 25: fixed={fix_late}  refresh={ref_late}")
        if fix_late > ref_late:
            print(f"  -> FIXED BASIS EXTENDS THE BOUNDARY (Ghost hypothesis supported)")
        elif fix_late == ref_late:
            print(f"  -> NO CHANGE in late-onset success (boundary is intrinsic, not refresh)")
        else:
            print(f"  -> FIXED BASIS WORSE at late onset (refresh was helping)")
    except Exception as e:
        print(f"  Could not load refreshing results: {e}")

    # Save
    report = {
        "model": MODEL_NAME, "method": "fixed_basis",
        "delta_values": DELTA_VALUES, "onset_values": ONSET_VALUES,
        "results": all_results,
        "t_crit": {str(k): v for k, v in t_crit.items()},
        "elapsed_s": round(elapsed, 1)
    }
    with open("output/exp_phase_diagram_fixedbasis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> output/exp_phase_diagram_fixedbasis.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
