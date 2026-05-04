"""Stacked v2: Surgery (full) + conditional gentle deflation.

Surgery produces the best REASONING (self-correction, finds -3/2 through checking).
Deflation produces CONVERGENCE (breaks loops).
Stack them with deflation at 1/3 dose, activated only when perseveration detected.

Conditions:
  1. baseline
  2. surgery_only (full e_c removal, L13-L35)
  3. surgery + gentle_deflation (alpha=0.03, conditional on L27 cos_sim > 0.95)
  4. surgery + medium_deflation (alpha=0.05, conditional)
  5. surgery + original_deflation (alpha=0.1, conditional)
"""
import json, time, re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048
SURGERY_LAYERS = list(range(13, 36))
DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
DEFLATE_REFRESH = 25
STICKINESS_LAYER = 27  # L27 is our perseveration detector
STICKINESS_THRESHOLD = 0.95
STICKINESS_WINDOW = 5   # consecutive steps above threshold to trigger

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
    "(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3" # Should we get rid of this?!???!??!?!
)
CORRECT = "B"
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
SURGERY_SYS = "You are a careful mathematical reasoner. Think step by step."


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


# ============ CONDITIONAL DEFLATION ============
class ConditionalDeflation:
    """Deflation that only activates when perseveration is detected.

    Monitors attention cos_sim at STICKINESS_LAYER. When cos_sim > threshold
    for STICKINESS_WINDOW consecutive steps, activates deflation. When it
    drops below, deactivates.
    """
    def __init__(self, model, layers, r=4, alpha=0.1, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
        self.hooks = []
        self.attn_hook = None
        self.step_count = 0
        self.is_generating = False
        self.deflation_active = False
        self.U_r = {}
        # Stickiness tracking
        self.prev_attn = None
        self.sticky_count = 0
        self.n_deflation_steps = 0
        self.n_total_steps = 0
        self._install()

    def _install(self):
        # Deflation hooks on q_proj output
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_deflation_hook(ell)
            )
            self.hooks.append(h)
        # Attention capture hook on L27 to monitor stickiness
        self.attn_hook = self.model.model.layers[STICKINESS_LAYER].self_attn.register_forward_hook(
            self._attn_capture_hook
        )

    def _make_deflation_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or not self.deflation_active or li not in self.U_r:
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

    def _attn_capture_hook(self, module, input, output):
        """Capture the attention output at L27 for stickiness detection."""
        if not self.is_generating:
            return
        # output is (attn_out, attn_weights, past_kv) or just attn_out
        # We just need the output hidden state for cos_sim tracking
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            curr = h[0, -1, :].detach().float()  # last token
        else:
            curr = h[-1, :].detach().float()

        if self.prev_attn is not None:
            cos = F.cosine_similarity(self.prev_attn.unsqueeze(0),
                                       curr.unsqueeze(0)).item()
            if cos > STICKINESS_THRESHOLD:
                self.sticky_count += 1
            else:
                self.sticky_count = 0

            was_active = self.deflation_active
            if self.sticky_count >= STICKINESS_WINDOW:
                self.deflation_active = True
            else:
                self.deflation_active = False

            if self.deflation_active and not was_active:
                pass  # silently activate
            if was_active and not self.deflation_active:
                pass  # silently deactivate

        if self.deflation_active:
            self.n_deflation_steps += 1
        self.n_total_steps += 1
        self.prev_attn = curr

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
        self.prev_attn = None
        self.sticky_count = 0
        self.deflation_active = False
        self.n_deflation_steps = 0
        self.n_total_steps = 0

    def tick(self, past_kv):
        self.step_count += 1
        if self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()
        if self.attn_hook: self.attn_hook.remove()
        self.U_r.clear()


# ============ GENERATION ============
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
            if tid in (151643, 151645): break
            gen_ids.append(tid)
            if deflator: deflator.tick(past_kv)
    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    defl_stats = None
    if deflator:
        defl_stats = {
            "deflation_steps": deflator.n_deflation_steps,
            "total_steps": deflator.n_total_steps,
            "pct_deflated": round(100 * deflator.n_deflation_steps / max(1, deflator.n_total_steps), 1),
        }
    return text, len(gen_ids), round(dt, 1), defl_stats


def extract_answer(text):
    import re
    # LaTeX fraction
    m = re.findall(r'\\boxed\{[^}]*?-\\frac\{3\}\{2\}[^}]*?\}', text)
    if m: return "B"
    m = re.findall(r'\\boxed\{[^}]*?-3/2[^}]*?\}', text)
    if m: return "B"
    m = re.findall(r'\\boxed\{[^}]*\b([A-E])\b[^}]*\}', text)
    if m: return m[-1]
    m = re.findall(r'\\boxed\{[^}]*(-?\d+/\d+)[^}]*\}', text)
    if m:
        mapping = {"-5/3": "A", "-3/2": "B", "-6/5": "C", "-5/6": "D", "-2/3": "E"}
        return mapping.get(m[-1], "?")
    m = re.findall(r'\(([A-E])\)', text[-500:])
    if m: return m[-1]
    # Check for -3/2 anywhere in last 500 chars
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

    # Save weights
    saved_w = {}
    for L in SURGERY_LAYERS:
        saved_w[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    # Compute e_c
    print("Computing e_c...", flush=True)
    en_acts = extract_layer_acts(model, tokenizer, SURGERY_PROBLEMS, "en", DEVICE)
    zh_acts = extract_layer_acts(model, tokenizer, SURGERY_PROBLEMS, "zh", DEVICE)
    directions = compute_ec(en_acts, zh_acts, n_layers)
    del en_acts, zh_acts
    print("Done.\n", flush=True)

    results = []

    # 1. BASELINE
    print("=" * 60 + "\nbaseline\n" + "=" * 60, flush=True)
    text, ntok, dt, _ = run_p12(model, tokenizer)
    ans = extract_answer(text)
    looped = ntok >= MAX_TOKENS - 5
    print(f"  ans={ans} tok={ntok} loop={looped} t={dt}s")
    results.append({"condition": "baseline", "answer": ans, "correct": ans == CORRECT,
                    "n_tokens": ntok, "time_s": dt, "looped": looped, "output": text})
    print(text)

    # 2. SURGERY ONLY
    print("\n" + "=" * 60 + "\nsurgery_only\n" + "=" * 60, flush=True)
    apply_surgery(model, directions, SURGERY_LAYERS, DEVICE)
    text, ntok, dt, _ = run_p12(model, tokenizer)
    ans = extract_answer(text)
    looped = ntok >= MAX_TOKENS - 5
    print(f"  ans={ans} tok={ntok} loop={looped} t={dt}s")
    results.append({"condition": "surgery_only", "answer": ans, "correct": ans == CORRECT,
                    "n_tokens": ntok, "time_s": dt, "looped": looped, "output": text})
    print(text)

    # 3-5: SURGERY + CONDITIONAL DEFLATION at varying alpha
    for alpha in [0.03, 0.05, 0.1]:
        cname = f"surgery+cond_defl_a{alpha}"
        print(f"\n{'=' * 60}\n{cname}\n{'=' * 60}", flush=True)
        # Surgery is still applied from condition 2
        defl = ConditionalDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R,
                                     alpha=alpha, refresh_every=DEFLATE_REFRESH)
        text, ntok, dt, dstats = run_p12(model, tokenizer, deflator=defl)
        ans = extract_answer(text)
        looped = ntok >= MAX_TOKENS - 5
        print(f"  ans={ans} tok={ntok} loop={looped} t={dt}s")
        if dstats:
            print(f"  deflation active: {dstats['deflation_steps']}/{dstats['total_steps']} steps ({dstats['pct_deflated']}%)")
        defl.remove()
        results.append({"condition": cname, "answer": ans, "correct": ans == CORRECT,
                        "n_tokens": ntok, "time_s": dt, "looped": looped,
                        "deflation_stats": dstats, "output": text})
        print(text)

    restore_weights(model, saved_w)

    # Summary
    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    for r in results:
        status = "CORRECT" if r["correct"] else ("LOOP" if r["looped"] else "WRONG")
        ds = r.get("deflation_stats")
        extra = f"  defl={ds['pct_deflated']}%" if ds else ""
        print(f"  {r['condition']:35s}  ans={r['answer']}  tok={r['n_tokens']:4d}  {status}{extra}")

    with open("output/exp_stacked_v2_p12.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to output/exp_stacked_v2_p12.json")


if __name__ == "__main__":
    main()
