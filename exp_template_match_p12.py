"""Template matching: does the baseline's residual stream ever pass near
the deflated run's "-3/2 moment"?

Plan:
1. Run DEFLATED P12 (known good), capture hidden state at every layer
   at the step where B first becomes rank 1 at L35.
2. Run BASELINE P12 (looping), capture hidden state at every step
   at the same key layers.
3. Cosine similarity between baseline states and deflated template.

If baseline briefly approaches the template then diverges, the
computation reaches the right place but the cache pulls it off.
"""
import json, time, re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

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
PROMPT = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"

B_TOKEN_ID = 33
ANSWER_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}

# Layers to capture templates at
TEMPLATE_LAYERS = [15, 20, 25, 27, 30, 33, 35]


class QDeflation:
    """Q-only deflation (from existing experiments)."""
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
                self._make_hook(ell))
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
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

    def tick(self, past_kv):
        self.step_count += 1
        if self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.U_r.clear()


def generate_with_capture(model, tokenizer, deflator=None, capture_layers=None,
                          capture_every=1, max_tokens=MAX_TOKENS, find_b_moment=False):
    """Generate and capture hidden states.

    Args:
        capture_layers: which layers to capture hidden states at
        capture_every: capture every N steps (1 = every step)
        find_b_moment: if True, also track when B becomes rank 1 at L35
    Returns:
        text, gen_ids, states_by_layer, b_moment_step
    """
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    past_kv = None
    gen_ids = []
    captured = {}
    capture_hooks = []

    norm = model.model.norm
    lm_head = model.lm_head

    def make_capture_hook(li):
        def hook(module, inp, output):
            hs = output[0]
            if hs.dim() == 3:
                hs = hs[:, -1, :]
            captured[li] = hs.detach()
        return hook

    for li in (capture_layers or []):
        h = model.model.layers[li].register_forward_hook(make_capture_hook(li))
        capture_hooks.append(h)

    # Storage: {layer: [(step, hidden_state_cpu), ...]}
    states = {li: [] for li in (capture_layers or [])}
    b_moment_step = None
    b_moment_states = {}  # {layer: hidden_state} at the B-moment

    t0 = time.time()
    for step in range(max_tokens):
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

            # Capture hidden states
            if (step + 1) % capture_every == 0 and captured:
                for li in capture_layers or []:
                    if li in captured:
                        states[li].append((step + 1, captured[li].cpu().float()))

                # Check if B is rank 1 at L35 (for finding the B-moment)
                if find_b_moment and b_moment_step is None and 35 in captured:
                    hs35 = captured[35]
                    normed = norm(hs35)
                    out_logits = lm_head(normed).squeeze(0)
                    b_rank = (out_logits > out_logits[B_TOKEN_ID]).sum().item() + 1
                    if b_rank == 1:
                        b_moment_step = step + 1
                        for li in capture_layers or []:
                            if li in captured:
                                b_moment_states[li] = captured[li].cpu().float()
                        print(f"  B-moment found at step {step + 1}!", flush=True)

                captured.clear()

    dt = round(time.time() - t0, 1)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    for h in capture_hooks:
        h.remove()
    del past_kv, out
    torch.cuda.empty_cache()

    return text, gen_ids, states, b_moment_step, b_moment_states, dt


def main():
    print("=" * 70)
    print("TEMPLATE MATCHING: DEFLATED B-MOMENT vs BASELINE TRAJECTORY")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print(f"Loaded {MODEL_NAME}\n", flush=True)

    # ===== STEP 1: Run deflated, find B-moment =====
    print("STEP 1: Running deflated P12 to find B-moment...", flush=True)
    defl = QDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R, alpha=Q_ALPHA,
                      refresh_every=DEFLATE_REFRESH)

    text_d, ids_d, states_d, b_step, b_states, dt_d = generate_with_capture(
        model, tokenizer, deflator=defl, capture_layers=TEMPLATE_LAYERS,
        capture_every=1, find_b_moment=True)
    defl.remove()

    ans_d = "B" if "-3/2" in text_d[-500:] or "frac{3}{2}" in text_d[-500:] else "?"
    print(f"  Deflated: {len(ids_d)} tokens, ans={ans_d}, t={dt_d}s", flush=True)

    if b_step is None:
        # B might not be rank 1 at any single step — find the step where B
        # rank is lowest at L35
        print("  B never reached rank 1. Finding best B rank...", flush=True)
        norm = model.model.norm
        lm_head = model.lm_head
        best_rank = 999999
        best_step_idx = 0
        for idx, (step, hs) in enumerate(states_d[35]):
            hs_gpu = hs.to(DEVICE, dtype=torch.float16)
            normed = norm(hs_gpu)
            logits = lm_head(normed).squeeze(0)
            rank = (logits > logits[B_TOKEN_ID]).sum().item() + 1
            if rank < best_rank:
                best_rank = rank
                best_step_idx = idx
                b_step = step
        # Collect states at best step
        for li in TEMPLATE_LAYERS:
            if states_d[li]:
                # Find the entry closest to b_step
                for s, hs in states_d[li]:
                    if s == b_step:
                        b_states[li] = hs
                        break
        print(f"  Best B rank: {best_rank} at step {b_step}", flush=True)

    print(f"\n  B-moment: step {b_step}", flush=True)
    if b_step and b_step <= len(ids_d):
        ctx = tokenizer.decode(ids_d[max(0, b_step-30):b_step], skip_special_tokens=True)
        print(f"  Context at B-moment: ...{ctx[-100:]}", flush=True)

    # Free deflated states (keep only b_states as templates)
    del states_d
    torch.cuda.empty_cache()

    # ===== STEP 2: Run baseline, capture at B-moment layers =====
    print("\nSTEP 2: Running baseline P12 with hidden state capture...", flush=True)

    # Capture every 5 steps to keep memory manageable
    text_b, ids_b, states_b, _, _, dt_b = generate_with_capture(
        model, tokenizer, deflator=None, capture_layers=TEMPLATE_LAYERS,
        capture_every=5, find_b_moment=False)

    looped = len(ids_b) >= MAX_TOKENS - 5
    print(f"  Baseline: {len(ids_b)} tokens, looped={looped}, t={dt_b}s\n", flush=True)

    # ===== STEP 3: Cosine similarity between templates and baseline trajectory =====
    print("STEP 3: Computing cosine similarity...\n", flush=True)

    results = {}
    for li in TEMPLATE_LAYERS:
        if li not in b_states or not states_b[li]:
            continue

        template = b_states[li].squeeze(0)  # (d,)
        steps_list = []
        cos_list = []

        for step, hs in states_b[li]:
            h = hs.squeeze(0)  # (d,)
            cos = F.cosine_similarity(template.unsqueeze(0), h.unsqueeze(0)).item()
            steps_list.append(step)
            cos_list.append(cos)

        results[li] = {"steps": steps_list, "cosines": cos_list}

        # Find peak
        max_cos = max(cos_list)
        max_step = steps_list[cos_list.index(max_cos)]
        min_cos = min(cos_list)
        mean_cos = sum(cos_list) / len(cos_list)

        print(f"  L{li:>2d}: peak cos={max_cos:.4f} at step {max_step:>4d}  "
              f"mean={mean_cos:.4f}  min={min_cos:.4f}  range={max_cos-min_cos:.4f}")

    # ===== STEP 4: Detailed trajectory for key layers =====
    print("\n" + "=" * 70)
    print("COSINE TRAJECTORY TO B-MOMENT TEMPLATE (selected layers)")
    print("=" * 70)

    for li in [25, 30, 33, 35]:
        if li not in results:
            continue
        print(f"\nL{li}:")
        steps = results[li]["steps"]
        cos = results[li]["cosines"]
        # Print every 50 steps plus peak
        peak_idx = cos.index(max(cos))
        for i, (s, c) in enumerate(zip(steps, cos)):
            if s % 50 == 0 or s <= 50 or i == peak_idx:
                marker = " <<< PEAK" if i == peak_idx else ""
                # Context
                if s <= len(ids_b):
                    snippet = tokenizer.decode(
                        ids_b[max(0, s-15):s], skip_special_tokens=True)[-40:]
                else:
                    snippet = "<end>"
                print(f"  step {s:>5d}: cos={c:.4f}  ...{snippet}{marker}")

    # ===== Save =====
    out_data = {
        "b_moment_step": b_step,
        "deflated_tokens": len(ids_d),
        "baseline_tokens": len(ids_b),
        "baseline_looped": looped,
        "trajectories": {str(li): {
            "steps": results[li]["steps"],
            "cosines": [round(c, 5) for c in results[li]["cosines"]],
            "peak_cos": round(max(results[li]["cosines"]), 5),
            "peak_step": results[li]["steps"][results[li]["cosines"].index(max(results[li]["cosines"]))],
        } for li in results},
    }
    out_path = "output/exp_template_match_p12.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
