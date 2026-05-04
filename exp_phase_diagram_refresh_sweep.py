"""Refresh period sweep: does the onset boundary move with refresh_every?

Ghost's discriminating experiment. If boundary scales with refresh_every,
it's bootstrapping. If fixed at 25-27, it's intrinsic saturation.
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

DELTA_VALUES = [0.10, 0.15, 0.30]
ONSET_VALUES = [0, 5, 10, 15, 20, 22, 25, 27, 30, 35, 40, 50]
REFRESH_VALUES = [5, 10, 15, 25, 50]


class BatchedDeflation:
    def __init__(self, model, layers, onsets, r=4, alpha=0.1, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.onsets = onsets
        self.batch_size = len(onsets)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
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
                deflator.start_gen()
                deflator.refresh_basis(out.past_key_values)
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
            deflator.tick(past_kv)
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
    print("REFRESH PERIOD SWEEP")
    print(f"deltas: {DELTA_VALUES}")
    print(f"refresh_every: {REFRESH_VALUES}")
    print(f"onsets: {ONSET_VALUES}")
    print(f"Total batches: {len(DELTA_VALUES) * len(REFRESH_VALUES)} = {len(DELTA_VALUES)}d x {len(REFRESH_VALUES)}r")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    print(f"Loaded {MODEL_NAME}\n", flush=True)

    all_results = []
    t_start = time.time()
    batch_count = 0
    total_batches = len(DELTA_VALUES) * len(REFRESH_VALUES)

    for delta in DELTA_VALUES:
        for refresh in REFRESH_VALUES:
            batch_count += 1
            print(f"[{batch_count}/{total_batches}] d={delta:.2f} refresh={refresh}...", end="", flush=True)
            t0 = time.time()
            deflator = BatchedDeflation(
                model, DEFLATE_LAYERS, ONSET_VALUES,
                r=DEFLATE_R, alpha=delta, refresh_every=refresh)
            row = generate_batched(model, tokenizer, input_ids, deflator, ONSET_VALUES)
            deflator.remove()
            dt = time.time() - t0
            n_correct = sum(1 for r in row if r["correct"])
            for r in row:
                r["delta"] = delta
                r["refresh_every"] = refresh
                all_results.append(r)
            elapsed = time.time() - t_start
            eta = elapsed / batch_count * (total_batches - batch_count)
            # find t_crit for this row
            correct_onsets = [r["onset"] for r in row if r["correct"]]
            tc = max(correct_onsets) if correct_onsets else -1
            print(f" {n_correct}/12 correct, t_crit={tc}, {dt:.0f}s, ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t_start

    # Analysis: t_crit vs refresh_every for each delta
    print(f"\n{'='*70}")
    print(f"RESULTS ({elapsed:.0f}s total)")
    print(f"{'='*70}")

    for delta in DELTA_VALUES:
        print(f"\ndelta={delta:.2f}:")
        print(f"  {'refresh':>8} |" + "".join(f" {o:>3}" for o in ONSET_VALUES) + " | t_crit  n_correct")
        print(f"  {'':->8}-+" + "-" * (4 * len(ONSET_VALUES)) + "-+" + "-" * 18)
        for refresh in REFRESH_VALUES:
            row = sorted([r for r in all_results if r["delta"]==delta and r["refresh_every"]==refresh],
                         key=lambda x: x["onset"])
            line = f"  {refresh:8d} |"
            for r in row:
                line += " Y  " if r["correct"] else " N  "
            correct_onsets = [r["onset"] for r in row if r["correct"]]
            tc = max(correct_onsets) if correct_onsets else -1
            nc = len(correct_onsets)
            line += f" | {tc:>6}  {nc:>2}/12"
            print(line)

    # Summary table: t_crit(delta, refresh)
    print(f"\n{'='*70}")
    print("t_crit summary (last onset -> CORRECT):")
    print(f"  {'':>8} |" + "".join(f" r={r:>3}" for r in REFRESH_VALUES))
    print(f"  {'':->8}-+" + "-" * (6 * len(REFRESH_VALUES)))
    for delta in DELTA_VALUES:
        line = f"  d={delta:.2f} |"
        for refresh in REFRESH_VALUES:
            row = [r for r in all_results if r["delta"]==delta and r["refresh_every"]==refresh]
            correct_onsets = [r["onset"] for r in row if r["correct"]]
            tc = max(correct_onsets) if correct_onsets else -1
            line += f"  {tc:>4}"
        print(line)

    # Does boundary move with refresh?
    print(f"\nKey question: does t_crit scale with refresh_every?")
    for delta in DELTA_VALUES:
        tcrits = []
        for refresh in REFRESH_VALUES:
            row = [r for r in all_results if r["delta"]==delta and r["refresh_every"]==refresh]
            correct_onsets = [r["onset"] for r in row if r["correct"]]
            tcrits.append(max(correct_onsets) if correct_onsets else -1)
        valid = [(r, t) for r, t in zip(REFRESH_VALUES, tcrits) if t >= 0]
        if len(valid) >= 2:
            rs, ts = zip(*valid)
            corr = sum((r - sum(rs)/len(rs)) * (t - sum(ts)/len(ts)) for r, t in valid)
            corr /= (sum((r - sum(rs)/len(rs))**2 for r in rs) * sum((t - sum(ts)/len(ts))**2 for t in ts)) ** 0.5 if sum((r - sum(rs)/len(rs))**2 for r in rs) > 0 and sum((t - sum(ts)/len(ts))**2 for t in ts) > 0 else 1
            print(f"  d={delta:.2f}: corr(refresh, t_crit) = {corr:.3f}")
        else:
            print(f"  d={delta:.2f}: insufficient data")

    # Save
    report = {
        "model": MODEL_NAME, "method": "refresh_sweep",
        "delta_values": DELTA_VALUES, "onset_values": ONSET_VALUES,
        "refresh_values": REFRESH_VALUES,
        "results": all_results,
        "elapsed_s": round(elapsed, 1)
    }
    with open("output/exp_phase_diagram_refresh_sweep.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> output/exp_phase_diagram_refresh_sweep.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
