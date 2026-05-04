"""Layer-specific deflation: which layers actually need Q-deflation?

Motivation:
- Template matching shows L33 holds the -3/2 signal (cos=0.73) but L35 loses it.
- C6b showed attention at last token is a constant bias (read head).
- If only the read-head layers (L34-L35) need deflation, computation layers
  (L20-L33) don't need the intervention at all.

Sweep:
1. Single-layer deflation at each of L20, 25, 30, 33, 34, 35
2. Read-head only (L34-L35)
3. Computation only (L20-L33)
4. Full range (L20-L35) as control
5. Combine with temporal: L34-L35 only, first 50 tokens only
"""
import json, time, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

Q_ALPHA = 0.1
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


class LayerSpecificDeflation:
    """Q-deflation at specified layers, optionally windowed in time."""
    def __init__(self, model, layers, r=4, alpha=0.1, refresh_every=25,
                 active_until=None):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
        self.active_until = active_until  # None = always on
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.is_active = True
        self.U_r = {}
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_hook(ell))
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or not self.is_active or li not in self.U_r:
                return output
            batch, seq, d = output.shape
            n_heads = 16
            head_dim = 128
            n_kv = len(self.U_r[li])
            gs_per_kv = n_heads // n_kv
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
        self.is_active = True

    def tick(self, past_kv):
        self.step_count += 1
        if self.active_until is not None and self.step_count > self.active_until:
            self.is_active = False
        if self.is_active and self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.U_r.clear()


def run_p12(model, tokenizer, deflator=None):
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
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

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids), round(dt, 1)


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
    print("LAYER-SPECIFIC Q-DEFLATION ON P12")
    print("Which layers actually need deflation?")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print(f"Loaded. {len(model.model.layers)} layers.\n", flush=True)

    # Define conditions: (name, layers, active_until)
    conditions = [
        # Controls
        ("baseline",           [],              None),
        ("full_L20-L35",       list(range(20,36)), None),

        # Single-layer sweep
        ("only_L20",           [20],            None),
        ("only_L25",           [25],            None),
        ("only_L30",           [30],            None),
        ("only_L33",           [33],            None),
        ("only_L34",           [34],            None),
        ("only_L35",           [35],            None),

        # Read head vs computation
        ("readhead_L34-L35",   [34, 35],        None),
        ("readhead_L33-L35",   [33, 34, 35],    None),
        ("compute_L20-L33",    list(range(20,34)), None),
        ("late_L30-L35",       list(range(30,36)), None),

        # Temporal + layer combos
        ("readhead_50tok",     [34, 35],        50),    # read head, first 50 tokens only
        ("full_50tok",         list(range(20,36)), 50),  # full range, first 50 tokens
        ("L33-35_50tok",       [33, 34, 35],    50),    # L33-35, first 50 tokens
    ]

    results = []

    for name, layers, active_until in conditions:
        print(f"\n{'='*60}", flush=True)
        layer_desc = f"L{layers}" if layers else "none"
        time_desc = f", first {active_until} tok" if active_until else ""
        print(f"  {name}: {layer_desc}{time_desc}", flush=True)
        print(f"{'='*60}", flush=True)

        if layers:
            defl = LayerSpecificDeflation(
                model, layers, r=DEFLATE_R, alpha=Q_ALPHA,
                refresh_every=DEFLATE_REFRESH, active_until=active_until)
        else:
            defl = None

        text, ntok, dt = run_p12(model, tokenizer, deflator=defl)
        ans = extract_answer(text)
        looped = ntok >= MAX_TOKENS - 5

        if defl:
            defl.remove()

        status = "CORRECT" if ans == CORRECT else ("LOOP" if looped else f"WRONG({ans})")
        print(f">>> {name}: ans={ans} tok={ntok} {status} t={dt}s", flush=True)

        results.append({
            "condition": name,
            "layers": layers,
            "active_until": active_until,
            "answer": ans,
            "correct": ans == CORRECT,
            "n_tokens": ntok,
            "time_s": dt,
            "looped": looped,
            "output_last200": text[-200:] if text else "",
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<25s} {'Layers':>12s} {'Window':>8s} {'Ans':>4s} {'Tok':>5s} {'Status':<12s}")
    print("-" * 70)
    for r in results:
        ly = f"{len(r['layers'])}L" if r['layers'] else "-"
        win = str(r['active_until']) if r['active_until'] else "all"
        st = "CORRECT" if r["correct"] else ("LOOP" if r["looped"] else f"WRONG({r['answer']})")
        print(f"{r['condition']:<25s} {ly:>12s} {win:>8s} {r['answer']:>4s} {r['n_tokens']:>5d} {st:<12s}")

    out_path = "output/exp_layer_specific_deflation_p12.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
