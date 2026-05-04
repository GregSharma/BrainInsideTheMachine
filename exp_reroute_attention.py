#!/usr/bin/env python3
"""Rank-preserving deflated attention: deflate q, reroute removed through o_proj.

Two modes:
- soft: q_new = q - alpha * V V^T q (partial deflation, small rank loss)
- reroute: q_new = q - V V^T q, then add beta * o_proj(removed) to attn output
  (full deflation, zero rank loss)

V = top-r right SVs of the cached key matrix (refreshed every N steps).
"""
import json, time, os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
REFRESH_EVERY = 25

SYS = (
    "You are solving an AMC 12A multiple choice math problem. "
    "Think step by step, show your work, then clearly state your "
    "final answer as (A), (B), (C), (D), or (E)."
)

PROBLEMS = {
    "p9_complex": {
        "text": (
            "Let w be the complex number 2 + i, where i = sqrt(-1). What real "
            "number r has the property that r, w, and w^2 are three collinear "
            "points in the complex plane?\n\n"
            "(A) 3/4  (B) 1  (C) 7/5  (D) 3/2  (E) 5/3"
        ),
        "answer": "E",
    },
    "p3_age": {
        "text": (
            "A team of students is going to compete against a team of teachers "
            "in a trivia contest. The total number of students and teachers is 15. "
            "Ash, a cousin of one of the students, wants to join the contest. "
            "If Ash plays with the students, the average age on that team will "
            "increase from 12 to 14. If Ash plays with the teachers, the average "
            "age on that team will decrease from 55 to 52. How old is Ash?\n\n"
            "(A) 28  (B) 29  (C) 30  (D) 32  (E) 33"
        ),
        "answer": "A",
    },
    "p12_harmonic": {
        "text": (
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
        ),
        "answer": "B",
    },
}


def make_prompt(text):
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


class RerouteDeflation:
    """Deflate q, reroute removed component through o_proj."""

    def __init__(self, model, layers, r=4, refresh_every=25,
                 mode="soft", alpha=0.1, beta=1.0):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.refresh_every = refresh_every
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.U_r = {}       # layer -> {kv_head -> (head_dim, r)}
        self.removed_q = {}  # layer -> removed q tensor
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h1 = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_deflate_hook(ell)
            )
            self.hooks.append(h1)
            if self.mode == "reroute":
                h2 = self.model.model.layers[ell].self_attn.register_forward_hook(
                    self._make_compensate_hook(ell)
                )
                self.hooks.append(h2)

    def _make_deflate_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
                return output
            q = output
            batch, seq, d = q.shape
            head_dim = 128
            n_q, n_kv = 16, 2
            gs = n_q // n_kv

            q = q.view(batch, seq, n_q, head_dim)
            removed = torch.zeros_like(q)

            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs, (kv_h + 1) * gs
                qg = q[:, :, s:e, :]
                proj = qg @ U @ U.T

                if self.mode == "soft":
                    q[:, :, s:e, :] = qg - self.alpha * proj
                elif self.mode == "reroute":
                    q[:, :, s:e, :] = qg - proj
                    removed[:, :, s:e, :] = proj

            if self.mode == "reroute":
                self.removed_q[li] = removed.view(batch, seq, d)

            return q.view(batch, seq, d)
        return hook

    def _make_compensate_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.removed_q:
                return output
            removed = self.removed_q.pop(li)
            o_proj = self.model.model.layers[li].self_attn.o_proj
            with torch.no_grad():
                comp = o_proj(removed)
            if isinstance(output, tuple):
                return (output[0] + self.beta * comp,) + output[1:]
            return output + self.beta * comp
        return hook

    def refresh_basis(self, past_kv):
        for ell in self.target_layers:
            keys = past_kv.layers[ell].keys
            n_kv = keys.shape[1]
            self.U_r[ell] = {}
            for kv_h in range(n_kv):
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
        self.removed_q.clear()


print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
)
model.eval()
print(f"Loaded. {len(model.model.layers)} layers.\n", flush=True)

conditions = [
    ("baseline",      None),
    ("soft_a0.05",    {"mode": "soft",    "alpha": 0.05, "r": 4}),
    ("soft_a0.1",     {"mode": "soft",    "alpha": 0.1,  "r": 4}),
    ("soft_a0.2",     {"mode": "soft",    "alpha": 0.2,  "r": 4}),
    ("reroute_b0.5",  {"mode": "reroute", "beta": 0.5,   "r": 4}),
    ("reroute_b1.0",  {"mode": "reroute", "beta": 1.0,   "r": 4}),
]

results = {}

for pname, pinfo in PROBLEMS.items():
    prompt = make_prompt(pinfo["text"])
    results[pname] = {"correct": pinfo["answer"], "conditions": {}}

    for cname, cfg in conditions:
        print(f"\n{'='*70}", flush=True)
        print(f"{pname} / {cname}", flush=True)
        print(f"{'='*70}", flush=True)

        deflator = None
        if cfg is not None:
            deflator = RerouteDeflation(
                model, DEFLATE_LAYERS,
                r=cfg.get("r", 4),
                refresh_every=REFRESH_EVERY,
                mode=cfg["mode"],
                alpha=cfg.get("alpha", 0.1),
                beta=cfg.get("beta", 1.0),
            )

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
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
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tid = next_id.item()

                if tid in (151643, 151645):
                    break
                gen_ids.append(tid)
                print(tokenizer.decode([tid]), end="", flush=True)

                if deflator:
                    deflator.tick(past_kv)

        dt = time.time() - t0
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n = len(gen_ids)

        if deflator:
            deflator.remove()
        del past_kv, out
        torch.cuda.empty_cache()

        print(f"\n--- {n} tokens, {dt:.1f}s ---", flush=True)
        results[pname]["conditions"][cname] = {
            "n_tokens": n, "time_s": round(dt, 1),
            "output": text, "tail": text[-400:],
        }

os.makedirs("output", exist_ok=True)
with open("output/exp_reroute_attention.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/exp_reroute_attention.json")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for pname, pr in results.items():
    print(f"\n{pname} (correct={pr['correct']}):")
    for cname, cr in pr["conditions"].items():
        tail = cr["tail"][-120:].replace("\n", " ")
        print(f"  {cname:15s}: {cr['n_tokens']:4d} tok, {cr['time_s']:5.1f}s | ...{tail}")
