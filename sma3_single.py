#!/usr/bin/env python3
"""SMA3 single-problem run. No straitjacket.

Same-layer modulation, no sys prompt, no answer choices,
no max token limit, no repeat penalty. Just the problem and the model.
Streaming output via TextStreamer.
"""
import json, time, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"

PROBLEM_TEXT = (
    "The harmonic mean of a collection of numbers is the reciprocal of the "
    "arithmetic mean of the reciprocals of the numbers in the collection. "
    "For example, the harmonic mean of 4, 4, and 5 is\n\n"
    "1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n"
    "What is the harmonic mean of all the real roots of the 4050th degree polynomial\n\n"
    r"\prod_{k=1}^{2025} (kx^2 - 4x - 3) = "
    "(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)...(2025x^2 - 4x - 3)?"
)

CORRECT = "-3/2"

SYS = "You are a careful mathematical reasoner. Show your work step by step."
PROMPT = (
    "<|im_start|>system\n" + SYS + "<|im_end|>\n"
    "<|im_start|>user\n" + PROBLEM_TEXT + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class SensitivityModulator:
    """Same-layer: pre-hook on layer L's q_proj peeks at layer L's own gate_proj(h)."""

    def __init__(self, model, alpha=1.0, mode="sensitivity", seed=None):
        self.alpha = alpha
        self.mode = mode
        self.hooks = []
        self.n_layers = len(model.model.layers)
        self.rng = torch.Generator(device=DEVICE)
        if seed is not None:
            self.rng.manual_seed(seed)
        self.w_down_sq_T = {}
        for ell in range(self.n_layers):
            W = model.model.layers[ell].mlp.down_proj.weight.detach()
            self.w_down_sq_T[ell] = W.pow(2).T.contiguous()
        self.gate_projs = [
            model.model.layers[ell].mlp.gate_proj for ell in range(self.n_layers)
        ]
        for ell in range(self.n_layers):
            h = model.model.layers[ell].self_attn.q_proj.register_forward_pre_hook(
                self._make_q_hook(ell)
            )
            self.hooks.append(h)

    def _make_q_hook(self, li):
        def hook(module, args):
            h = args[0]
            with torch.no_grad():
                x_gate = self.gate_projs[li](h)
                sig = torch.sigmoid(x_gate)
                tau = sig * (1.0 - sig)
                s = torch.matmul(tau, self.w_down_sq_T[li])
                # No normalization — raw tension magnitude is the signal
                if self.mode == "uniform":
                    s = torch.ones_like(s)
            return (h + self.alpha * s * h,) + args[1:]
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.w_down_sq_T.clear()


print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
)
model.eval()
print(f"Loaded. {len(model.model.layers)} layers.\n", flush=True)

print(">>>>>> RAW PROMPT <<<<<<", flush=True)
print(repr(PROMPT), flush=True)
print(">>>>>> END <<<<<<\n", flush=True)

streamer = TextStreamer(tokenizer, skip_special_tokens=True)

conditions = [
    ("baseline",       None,  None),
    ("additive_a0.01", 0.01,  "sensitivity"),
    ("additive_a0.05", 0.05,  "sensitivity"),
    ("additive_a0.1",  0.1,   "sensitivity"),
    ("additive_a0.3",  0.3,   "sensitivity"),
    ("additive_a0.5",  0.5,   "sensitivity"),
    ("additive_a1.0",  1.0,   "sensitivity"),
    ("uniform",        0.5,   "uniform"),
]

results = []
for cname, alpha, mode in conditions:
    print("\n" + "=" * 70, flush=True)
    print(f"CONDITION: {cname}", flush=True)
    print("=" * 70, flush=True)

    mod = None
    if alpha is not None:
        seed = 42 if mode == "random" else None
        mod = SensitivityModulator(model, alpha=alpha, mode=mode, seed=seed)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    out = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=2048,
        eos_token_id=[151643, 151645],  # <|endoftext|> + <|im_end|>
        streamer=streamer,
    )
    dt = time.time() - t0
    text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    ntok = out.shape[1] - inputs.input_ids.shape[1]

    if mod:
        mod.remove()
    torch.cuda.empty_cache()

    print(f"\n--- {ntok} tokens, {dt:.1f}s ---", flush=True)
    results.append(
        {"condition": cname, "tokens": int(ntok), "time": round(dt, 1), "output": text}
    )

print("\n" + "=" * 70, flush=True)
print(f"DONE. Correct answer: {CORRECT}", flush=True)
os.makedirs("output", exist_ok=True)
with open("output/expSMA3_single_p12.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved to output/expSMA3_single_p12.json", flush=True)
