"""Dual deflation: all 2^3 subsets of {Q-deflation, K-deflation, Surgery} on P12.

Query deflation: q_new = q - alpha * V V^T q  (attend to what's new)
Key deflation:   k_new = k - beta  * V V^T k  (be something new)
Surgery:         remove e_c from W_down        (remove convention)

8 conditions run sequentially, output streamed inline.
"""
import json, time, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

# Intervention params
Q_ALPHA = 0.1
K_BETA = 0.1   # start symmetric, can tune
DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
DEFLATE_REFRESH = 25
SURGERY_LAYERS = list(range(13, 36))

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
    n_layers = len(model.model.layers)
    all_acts = {L: [] for L in range(n_layers)}
    class Cap:
        def __init__(self): self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[:, -1, :].detach().float().cpu().numpy()
    caps = [Cap() for _ in range(n_layers)]
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in range(n_layers)]
    with torch.inference_mode():
        for p in problems:
            msg = [{"role": "system", "content": SURGERY_SYS},
                   {"role": "user", "content": p[lang]}]
            prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            model(ids)
            for L in range(n_layers):
                all_acts[L].append(caps[L].out[0])
    for h in hooks: h.remove()
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


# ============ DUAL DEFLATION ============
class DualDeflation:
    """Query and/or key deflation using shared SVD basis from key cache.

    Query: q_new = q - alpha * V V^T q  (at q_proj output)
    Key:   k_new = k - beta  * V V^T k  (at k_proj output)

    Both use the same basis V = top-r right SVs of cached keys.
    """
    def __init__(self, model, layers, r=4, q_alpha=0.0, k_beta=0.0, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.q_alpha = q_alpha
        self.k_beta = k_beta
        self.refresh_every = refresh_every
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.U_r = {}  # {layer: {kv_head: U_matrix}}
        self._install()

    def _install(self):
        for ell in self.target_layers:
            if self.q_alpha > 0:
                h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                    self._make_q_hook(ell))
                self.hooks.append(h)
            if self.k_beta > 0:
                h = self.model.model.layers[ell].self_attn.k_proj.register_forward_hook(
                    self._make_k_hook(ell))
                self.hooks.append(h)

    def _deflate(self, tensor, li, n_heads, head_dim, gs_per_kv=None):
        """Shared deflation logic for both q and k."""
        if li not in self.U_r:
            return tensor
        batch, seq, d = tensor.shape
        n_kv = len(self.U_r[li])  # number of KV heads with basis
        if gs_per_kv is None:
            # Key: 1 head per KV head
            tensor = tensor.view(batch, seq, n_heads, head_dim)
            for kv_h in range(n_heads):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                kg = tensor[:, :, kv_h:kv_h+1, :]
                proj = kg @ U @ U.T
                tensor[:, :, kv_h:kv_h+1, :] = kg - self.k_beta * proj
        else:
            # Query: gs_per_kv query heads per KV head
            tensor = tensor.view(batch, seq, n_heads, head_dim)
            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs_per_kv, (kv_h + 1) * gs_per_kv
                qg = tensor[:, :, s:e, :]
                proj = qg @ U @ U.T
                tensor[:, :, s:e, :] = qg - self.q_alpha * proj
        return tensor.view(batch, seq, d)

    def _make_q_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
                return output
            # Qwen2.5-3B: 16 Q heads, 2 KV heads, head_dim=128
            return self._deflate(output, li, n_heads=16, head_dim=128, gs_per_kv=8)
        return hook

    def _make_k_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
                return output
            # Qwen2.5-3B: 2 KV heads, head_dim=128
            return self._deflate(output, li, n_heads=2, head_dim=128, gs_per_kv=None)
        return hook

    def refresh_basis(self, past_kv):
        for ell in self.target_layers:
            keys = past_kv.layers[ell].keys  # (batch, n_kv_heads, seq, head_dim)
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

    def tick(self, past_kv):
        self.step_count += 1
        if self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.U_r.clear()


# ============ GENERATION WITH STREAMING ============
def run_p12(model, tokenizer, deflator=None, label=""):
    """Generate with inline streaming. Returns (text, n_tokens, time_s)."""
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None
    t0 = time.time()
    # Buffer for streaming
    stream_buf = []
    printed_chars = 0

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

            # Stream output every 50 tokens
            if len(gen_ids) % 50 == 0:
                full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                new_text = full_text[printed_chars:]
                if new_text:
                    print(new_text, end="", flush=True)
                    printed_chars = len(full_text)

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Print remaining
    remaining = text[printed_chars:]
    if remaining:
        print(remaining, end="", flush=True)
    print(flush=True)

    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids), round(dt, 1)


def extract_answer(text):
    if not text:
        return "?"
    # Check for -3/2 in boxed
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


# ============ MAIN ============
def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Loaded. {n_layers} layers.\n", flush=True)

    # Save weights for surgery
    saved_w = {}
    for L in SURGERY_LAYERS:
        saved_w[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    # Compute e_c
    print("Computing e_c for surgery...", flush=True)
    en_acts = extract_layer_acts(model, tokenizer, SURGERY_PROBLEMS, "en", DEVICE)
    zh_acts = extract_layer_acts(model, tokenizer, SURGERY_PROBLEMS, "zh", DEVICE)
    directions = compute_ec(en_acts, zh_acts, n_layers)
    del en_acts, zh_acts
    print("Done.\n", flush=True)

    # All 2^3 conditions
    conditions = [
        # (name,            q_alpha, k_beta, surgery)
        ("1_baseline",       0.0,    0.0,    False),
        ("2_Q_only",         Q_ALPHA, 0.0,   False),
        ("3_K_only",         0.0,    K_BETA,  False),
        ("4_QK_dual",        Q_ALPHA, K_BETA, False),
        ("5_surgery_only",   0.0,    0.0,    True),
        ("6_surgery_Q",      Q_ALPHA, 0.0,   True),
        ("7_surgery_K",      0.0,    K_BETA,  True),
        ("8_surgery_QK",     Q_ALPHA, K_BETA, True),
    ]

    results = []

    for cname, qa, kb, do_surgery in conditions:
        print("\n" + "=" * 70, flush=True)
        print(f"CONDITION: {cname}", flush=True)
        desc_parts = []
        if qa > 0: desc_parts.append(f"Q-deflation \u03b1={qa}")
        if kb > 0: desc_parts.append(f"K-deflation \u03b2={kb}")
        if do_surgery: desc_parts.append("surgery L13-L35")
        if not desc_parts: desc_parts.append("no intervention")
        print(f"  [{', '.join(desc_parts)}]")
        print("=" * 70, flush=True)

        # Apply/restore surgery
        if do_surgery:
            restore_weights(model, saved_w)  # clean slate
            apply_surgery(model, directions, SURGERY_LAYERS, DEVICE)
        else:
            restore_weights(model, saved_w)  # always restore to clean

        # Set up deflation
        defl = None
        if qa > 0 or kb > 0:
            defl = DualDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R,
                                 q_alpha=qa, k_beta=kb, refresh_every=DEFLATE_REFRESH)

        # Run
        text, ntok, dt = run_p12(model, tokenizer, deflator=defl, label=cname)
        ans = extract_answer(text)
        looped = ntok >= MAX_TOKENS - 5

        # Cleanup
        if defl:
            defl.remove()

        # Report
        status = "CORRECT" if ans == CORRECT else ("LOOP" if looped else f"WRONG({ans})")
        print(f"\n>>> {cname}: ans={ans} tok={ntok} {status} t={dt}s", flush=True)

        results.append({
            "condition": cname,
            "q_alpha": qa, "k_beta": kb, "surgery": do_surgery,
            "answer": ans, "correct": ans == CORRECT,
            "n_tokens": ntok, "time_s": dt, "looped": looped,
            "output": text,
        })

    # Restore weights
    restore_weights(model, saved_w)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — P12 Harmonic Mean (correct = B / -3/2)")
    print("=" * 70)
    for r in results:
        status = "CORRECT" if r["correct"] else ("LOOP" if r["looped"] else f"WRONG({r['answer']})")
        flags = []
        if r["q_alpha"] > 0: flags.append("Q")
        if r["k_beta"] > 0: flags.append("K")
        if r["surgery"]: flags.append("S")
        flagstr = "+".join(flags) if flags else "-"
        print(f"  {r['condition']:20s} [{flagstr:5s}]  ans={r['answer']}  tok={r['n_tokens']:4d}  {status}")

    out_path = "output/exp_dual_deflation_p12.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
