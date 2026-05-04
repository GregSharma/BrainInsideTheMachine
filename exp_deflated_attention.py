#!/usr/bin/env python3
"""Deflated Attention: project out dominant key subspace from queries.

At layers where attention perseverates (L27, L30), maintain a running
low-rank approximation of the key Gram matrix K K^T. Before computing
attention, deflate the query by projecting out the top-r directions
of K K^T. The query then responds only to what's structurally NEW
in the cache, not what's redundant.

Mechanics:
  q_eff = q - V_r V_r^T q
  where V_r = top-r right singular vectors of the accumulated keys.

We recompute V_r every REFRESH_EVERY steps from the actual key cache
(cheap: SVD of a (T, 128) matrix).

No SMA. No gate peek. Pure attention geometry.
"""
import json, time, os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

# Deflation config
DEFLATE_LAYERS = list(range(20, 36))  # L20-L35: where stickiness builds
DEFLATE_R = 4       # number of top key directions to project out
REFRESH_EVERY = 25  # recompute SVD every N steps

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


class DeflatedAttention:
    """Project out top-r key directions from q_proj input at target layers."""

    def __init__(self, model, layers, r=4, refresh_every=25):
        self.model = model
        self.layers = set(layers)
        self.r = r
        self.refresh_every = refresh_every
        self.hooks = []
        self.step = 0
        # V_r per layer: (d_model, r) — top-r right SVs of accumulated keys
        self.V_r = {}
        self._install_hooks()

    def _install_hooks(self):
        for ell in self.layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_pre_hook(
                self._make_hook(ell)
            )
            self.hooks.append(h)

    def _make_hook(self, layer_idx):
        def hook(module, args):
            if layer_idx not in self.V_r:
                return args  # no basis yet, pass through
            h = args[0]  # (batch, seq, d_model)
            V = self.V_r[layer_idx]  # (d_model, r)
            # Deflate: h_new = h - V V^T h
            proj = h @ V  # (batch, seq, r)
            h_deflated = h - proj @ V.T  # (batch, seq, d_model)
            return (h_deflated,) + args[1:]
        return hook

    def refresh_basis(self, past_kv):
        """Recompute top-r right SVs from the accumulated key cache."""
        for ell in self.layers:
            keys = past_kv.layers[ell].keys  # (1, n_kv, T, d_k)
            # Concatenate KV heads: (T, n_kv * d_k)
            T = keys.shape[2]
            K = keys[0].permute(1, 0, 2).reshape(T, -1).float()  # (T, n_kv*d_k)
            # But we want to deflate in d_model space, not d_k space.
            # The key projection is K = W_K @ h, so the key-loud directions
            # in h-space are W_K^T's top left singular vectors = W_K's top
            # right singular vectors. These are FIXED (precomputed).
            # BUT we want the directions that the ACTUAL cached keys span,
            # which changes as the cache grows.
            #
            # Approach: compute SVD of the cached keys in key-space,
            # then map back to h-space via W_K^+ (pseudoinverse).
            # Simpler: just deflate in key-space by hooking k_proj output
            # instead of q_proj input.
            #
            # Actually simplest correct approach: the query q = W_Q h attends
            # to keys k = W_K h. The attention score is q^T k = h^T W_Q^T W_K h'.
            # To deflate, we want to remove from q the components that align
            # with the dominant key directions. In q-space, the dominant
            # directions are the top left SVs of the key matrix K (T x d_k).
            # Deflating q by these: q_new = q - U_r U_r^T q where U_r = top-r
            # left SVs of K.
            #
            # But we're hooking q_proj INPUT (h-space), not q_proj OUTPUT
            # (q-space). To deflate in h-space such that the resulting q
            # is deflated in key-space... we need W_Q's structure.
            #
            # Cleanest: hook q_proj OUTPUT instead.
            pass

        # Actually, let me just hook the q_proj output directly.
        # Re-architect: remove current hooks, install output hooks.
        # For now, let's use the simpler approach: deflate in key-space
        # by hooking AFTER q_proj.
        pass

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.V_r.clear()


class DeflatedAttentionV2:
    """Deflate q_proj OUTPUT by top-r left SVs of the cached key matrix.

    Hook on q_proj output. At each step, if we have a cached basis U_r
    (top-r left SVs of accumulated keys), project it out of q:
      q_new = q - U_r U_r^T q

    This removes the components of the query that would attend to the
    dominant (redundant) key directions.
    """

    def __init__(self, model, layers, r=4, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.refresh_every = refresh_every
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        # U_r per layer: (d_k_total, r) — top-r left SVs of K cache
        # d_k_total = n_kv_heads * head_dim (256 for Qwen2.5-3B)
        self.U_r = {}
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_hook(ell)
            )
            self.hooks.append(h)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if not self.is_generating:
                return output  # don't touch prompt encoding
            if layer_idx not in self.U_r:
                return output  # no basis yet
            # output shape: (batch, seq, n_heads * head_dim)
            # But GQA: n_q_heads=16, n_kv_heads=2, head_dim=128
            # q_proj output is (batch, seq, 16*128=2048) = full d_model
            # We need to reshape to per-head, deflate in head_dim space
            # where keys live, then reshape back.
            #
            # Actually, with GQA, each group of 8 q-heads shares 1 kv-head.
            # The key-space per kv-head is (head_dim,). We deflate each
            # q-head by the basis of its corresponding kv-head's keys.
            #
            # For Qwen2.5-3B: 16 q-heads, 2 kv-heads, groups of 8.
            # U_r[layer] is a dict: {kv_head_idx: (head_dim, r)}
            q = output  # (batch, seq, 2048)
            batch, seq, d = q.shape
            head_dim = 128
            n_q = 16
            n_kv = 2
            group_size = n_q // n_kv  # 8

            q = q.view(batch, seq, n_q, head_dim)  # (B, S, 16, 128)
            for kv_h in range(n_kv):
                if kv_h not in self.U_r[layer_idx]:
                    continue
                U = self.U_r[layer_idx][kv_h]  # (head_dim, r)
                # Deflate all q-heads in this group
                start = kv_h * group_size
                end = start + group_size
                q_group = q[:, :, start:end, :]  # (B, S, 8, 128)
                proj = q_group @ U  # (B, S, 8, r)
                q_group_deflated = q_group - proj @ U.T  # (B, S, 8, 128)
                q[:, :, start:end, :] = q_group_deflated

            return q.view(batch, seq, d)
        return hook

    def refresh_basis(self, past_kv):
        """Recompute U_r from accumulated key cache."""
        for ell in self.target_layers:
            keys = past_kv.layers[ell].keys  # (1, n_kv, T, head_dim)
            n_kv = keys.shape[1]
            self.U_r[ell] = {}
            for kv_h in range(n_kv):
                K = keys[0, kv_h, :, :].float()  # (T, head_dim)
                if K.shape[0] < self.r:
                    continue
                U, S, Vh = torch.linalg.svd(K, full_matrices=False)
                # Top-r left SVs of K = top-r columns of U... but U is (T, min(T,d))
                # We want directions in head_dim space, which are Vh rows.
                # K = U S Vh => K^T K = Vh^T S^2 Vh => top eigenvecs of K^T K = Vh rows
                # The key-space directions are Vh[:r, :] (rows).
                # To deflate q in head_dim space: project out Vh[:r, :].T
                V_top = Vh[:self.r, :].T.contiguous().to(DEVICE, dtype=torch.float16)
                self.U_r[ell][kv_h] = V_top  # (head_dim, r)

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


print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()
n_layers = len(model.model.layers)
print(f"Loaded. {n_layers} layers.\n", flush=True)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

# Conditions
conditions = [
    ("baseline",    None),
    ("deflate_r4",  {"r": 4, "layers": DEFLATE_LAYERS, "refresh": REFRESH_EVERY}),
    ("deflate_r8",  {"r": 8, "layers": DEFLATE_LAYERS, "refresh": REFRESH_EVERY}),
    ("deflate_r2",  {"r": 2, "layers": DEFLATE_LAYERS, "refresh": REFRESH_EVERY}),
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
            deflator = DeflatedAttentionV2(
                model, cfg["layers"], r=cfg["r"], refresh_every=cfg["refresh"]
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

                # Print token
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
with open("output/exp_deflated_attention.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/exp_deflated_attention.json")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for pname, pr in results.items():
    print(f"\n{pname} (correct={pr['correct']}):")
    for cname, cr in pr["conditions"].items():
        tail = cr["tail"][-100:].replace("\n", " ")
        print(f"  {cname:15s}: {cr['n_tokens']:4d} tok, {cr['time_s']:5.1f}s | ...{tail}")
