"""Stack all three interventions on P12: surgery + SMA + deflation.

Three independent bottleneck-widening mechanisms:
  1. Surgery: remove convention direction e_c from W_down (L13-L35)
  2. SMA: additive sensitivity modulation h + alpha_s * s * h on q_proj input
  3. Deflation: soft query steering q - alpha_d * V V^T q on q_proj output

Conditions: baseline, surgery, sma, deflation, ALL_THREE
"""
import json, time, re, copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

# --- Intervention params ---
SURGERY_LAYERS = list(range(13, 36))   # above l_c
SMA_ALPHA = 0.05                       # reasoning-path shift, not catastrophic
DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
DEFLATE_ALPHA = 0.1
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

# --- Bilingual problems for e_c computation ---
SURGERY_PROBLEMS = [
    {"en": "Solve for x: 3x + 7 = 22", "zh": "\u6c42\u89e3x\uff1a3x + 7 = 22"},
    {"en": "Solve for x: 2x\u00b2 - 8 = 0", "zh": "\u6c42\u89e3x\uff1a2x\u00b2 - 8 = 0"},
    {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97\uff1a347 + 658"},
    {"en": "Calculate: 23 \u00d7 17", "zh": "\u8ba1\u7b97\uff1a23 \u00d7 17"},
    {"en": "Find the area of a circle with radius 7", "zh": "\u6c42\u534a\u5f84\u4e3a7\u7684\u5706\u7684\u9762\u79ef"},
    {"en": "Find the hypotenuse of a right triangle with legs 5 and 12",
     "zh": "\u6c42\u76f4\u89d2\u4e09\u89d2\u5f62\u4e24\u76f4\u89d2\u8fb9\u4e3a5\u548c12\u65f6\u7684\u659c\u8fb9\u957f"},
    {"en": "What is the GCD of 84 and 120?", "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570\u662f\u591a\u5c11\uff1f"},
    {"en": "How many ways can you choose 3 items from 7?",
     "zh": "\u4ece7\u4e2a\u7269\u54c1\u4e2d\u90093\u4e2a\u6709\u591a\u5c11\u79cd\u65b9\u5f0f\uff1f"},
]

SURGERY_SYS = ("You are a careful mathematical reasoner. When given a problem, "
               "think step by step, show your work clearly, and then state the "
               "final numerical answer on its own line.")


# ============ SURGERY ============
def extract_layer_acts(model, tokenizer, problems, lang, device):
    """Get last hidden state at each layer for each problem."""
    n_layers = len(model.model.layers)
    all_acts = {L: [] for L in range(n_layers)}
    captures = [None] * n_layers

    class Cap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[:, -1, :].detach().float().cpu().numpy()

    caps = [Cap() for _ in range(n_layers)]
    hooks = [model.model.layers[L].register_forward_hook(caps[L])
             for L in range(n_layers)]

    with torch.inference_mode():
        for p in problems:
            msg = [{"role": "system", "content": SURGERY_SYS},
                   {"role": "user", "content": p[lang]}]
            prompt = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            model(ids)
            for L in range(n_layers):
                all_acts[L].append(caps[L].out[0])

    for h in hooks:
        h.remove()
    return {L: np.array(all_acts[L]) for L in range(n_layers)}


def compute_ec(en_acts, zh_acts, n_layers):
    directions = {}
    for L in range(n_layers):
        diff = zh_acts[L].mean(axis=0) - en_acts[L].mean(axis=0)
        directions[L] = diff / (np.linalg.norm(diff) + 1e-12)
    return directions


def apply_surgery(model, directions, layers, device):
    for L in layers:
        e_c = torch.tensor(directions[L], dtype=torch.float16, device=device)
        W = model.model.layers[L].mlp.down_proj.weight.data
        proj = e_c.unsqueeze(0) @ W
        W.sub_(e_c.unsqueeze(1) @ proj)


def restore_weights(model, saved_weights):
    for L, w in saved_weights.items():
        model.model.layers[L].mlp.down_proj.weight.data.copy_(w)


# ============ SMA ============
class SMAHook:
    def __init__(self, model, alpha=0.05):
        self.alpha = alpha
        self.hooks = []
        self.n_layers = len(model.model.layers)
        self.w_down_sq_T = {}
        for ell in range(self.n_layers):
            W = model.model.layers[ell].mlp.down_proj.weight.detach()
            self.w_down_sq_T[ell] = W.pow(2).T.contiguous()
        self.gate_projs = [
            model.model.layers[ell].mlp.gate_proj
            for ell in range(self.n_layers)
        ]
        for ell in range(self.n_layers):
            h = model.model.layers[ell].self_attn.q_proj.register_forward_pre_hook(
                self._make_hook(ell)
            )
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, args):
            h = args[0]
            with torch.no_grad():
                x_gate = self.gate_projs[li](h)
                sig = torch.sigmoid(x_gate)
                tau = sig * (1.0 - sig)
                s = torch.matmul(tau, self.w_down_sq_T[li])
            return (h + self.alpha * s * h,) + args[1:]
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.w_down_sq_T.clear()


# ============ DEFLATION ============
class SoftDeflation:
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


# ============ GENERATION ============
def run_p12(model, tokenizer, deflator=None):
    """Greedy generation with optional deflation. Returns (text, n_tokens, time_s)."""
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
                out = model(input_ids=next_id, past_key_values=past_kv,
                           use_cache=True)

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
    m = re.findall(r'\\boxed\{[^}]*\b([A-E])\b[^}]*\}', text)
    if m:
        return m[-1]
    m = re.findall(r'\\boxed\{[^}]*(-?\d+/\d+)[^}]*\}', text)
    if m:
        mapping = {"-5/3": "A", "-3/2": "B", "-6/5": "C", "-5/6": "D", "-2/3": "E"}
        return mapping.get(m[-1], "?")
    m = re.findall(r'\(([A-E])\)', text[-300:])
    if m:
        return m[-1]
    m = re.findall(r'answer is.*?([A-E])\b', text[-300:], re.IGNORECASE)
    if m:
        return m[-1]
    return "?"


# ============ MAIN ============
def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Loaded. {n_layers} layers.\n", flush=True)

    # Save original weights for surgery restore
    print("Saving original W_down...", flush=True)
    saved_w = {}
    for L in SURGERY_LAYERS:
        saved_w[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    # Compute convention direction e_c
    print("Extracting encoding activations for e_c...", flush=True)
    en_acts = extract_layer_acts(model, tokenizer, SURGERY_PROBLEMS, "en", DEVICE)
    zh_acts = extract_layer_acts(model, tokenizer, SURGERY_PROBLEMS, "zh", DEVICE)
    directions = compute_ec(en_acts, zh_acts, n_layers)
    del en_acts, zh_acts
    print("e_c computed.\n", flush=True)

    results = []

    # --- 1. BASELINE ---
    print("=" * 60)
    print("CONDITION: baseline")
    print("=" * 60, flush=True)
    text, ntok, dt = run_p12(model, tokenizer)
    ans = extract_answer(text)
    looped = ntok >= MAX_TOKENS - 5
    print(f"  Answer: {ans}  Correct: {CORRECT}  Tokens: {ntok}  Time: {dt}s  Loop: {looped}")
    results.append({"condition": "baseline", "answer": ans, "correct": ans == CORRECT,
                    "n_tokens": ntok, "time_s": dt, "looped": looped, "output": text})

    # --- 2. SURGERY ONLY ---
    print("\n" + "=" * 60)
    print("CONDITION: surgery_only (e_c removed from W_down, L13-L35)")
    print("=" * 60, flush=True)
    apply_surgery(model, directions, SURGERY_LAYERS, DEVICE)
    text, ntok, dt = run_p12(model, tokenizer)
    ans = extract_answer(text)
    looped = ntok >= MAX_TOKENS - 5
    print(f"  Answer: {ans}  Correct: {CORRECT}  Tokens: {ntok}  Time: {dt}s  Loop: {looped}")
    results.append({"condition": "surgery_only", "answer": ans, "correct": ans == CORRECT,
                    "n_tokens": ntok, "time_s": dt, "looped": looped, "output": text})
    restore_weights(model, saved_w)

    # --- 3. SMA ONLY ---
    print("\n" + "=" * 60)
    print(f"CONDITION: sma_only (additive alpha={SMA_ALPHA})")
    print("=" * 60, flush=True)
    sma = SMAHook(model, alpha=SMA_ALPHA)
    text, ntok, dt = run_p12(model, tokenizer)
    ans = extract_answer(text)
    looped = ntok >= MAX_TOKENS - 5
    print(f"  Answer: {ans}  Correct: {CORRECT}  Tokens: {ntok}  Time: {dt}s  Loop: {looped}")
    results.append({"condition": "sma_only", "answer": ans, "correct": ans == CORRECT,
                    "n_tokens": ntok, "time_s": dt, "looped": looped, "output": text})
    sma.remove()

    # --- 4. DEFLATION ONLY ---
    print("\n" + "=" * 60)
    print(f"CONDITION: deflation_only (alpha={DEFLATE_ALPHA}, r={DEFLATE_R})")
    print("=" * 60, flush=True)
    defl = SoftDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R,
                         alpha=DEFLATE_ALPHA, refresh_every=DEFLATE_REFRESH)
    text, ntok, dt = run_p12(model, tokenizer, deflator=defl)
    ans = extract_answer(text)
    looped = ntok >= MAX_TOKENS - 5
    print(f"  Answer: {ans}  Correct: {CORRECT}  Tokens: {ntok}  Time: {dt}s  Loop: {looped}")
    results.append({"condition": "deflation_only", "answer": ans, "correct": ans == CORRECT,
                    "n_tokens": ntok, "time_s": dt, "looped": looped, "output": text})
    defl.remove()

    # --- 5. ALL THREE STACKED ---
    print("\n" + "=" * 60)
    print("CONDITION: STACKED (surgery + SMA + deflation)")
    print("=" * 60, flush=True)
    # Surgery (weight mod)
    apply_surgery(model, directions, SURGERY_LAYERS, DEVICE)
    # SMA (pre-hook) — note: surgery changed W_down, so SMA's w_down_sq_T
    # needs to reflect the SURGERED weights
    sma = SMAHook(model, alpha=SMA_ALPHA)
    # Deflation (post-hook)
    defl = SoftDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R,
                         alpha=DEFLATE_ALPHA, refresh_every=DEFLATE_REFRESH)
    text, ntok, dt = run_p12(model, tokenizer, deflator=defl)
    ans = extract_answer(text)
    looped = ntok >= MAX_TOKENS - 5
    print(f"  Answer: {ans}  Correct: {CORRECT}  Tokens: {ntok}  Time: {dt}s  Loop: {looped}")
    results.append({"condition": "stacked", "answer": ans, "correct": ans == CORRECT,
                    "n_tokens": ntok, "time_s": dt, "looped": looped, "output": text})
    defl.remove()
    sma.remove()
    restore_weights(model, saved_w)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY — P12 Harmonic Mean (correct = B / -3/2)")
    print("=" * 60)
    for r in results:
        status = "CORRECT" if r["correct"] else ("LOOP" if r["looped"] else "WRONG")
        print(f"  {r['condition']:20s}  ans={r['answer']}  tok={r['n_tokens']:4d}  {status}")

    # Save
    out_path = "output/exp_stacked_p12.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
