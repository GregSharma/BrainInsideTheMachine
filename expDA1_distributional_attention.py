#!/usr/bin/env python3
"""
Exp DA1: Distributional Attention — Kernel Trick Variance Bonus
===============================================================
Tests whether adding an uncertainty-driven exploration bonus to attention
logits improves math reasoning at inference time, without retraining.

Core idea: Attention IS kernel regression. If the query has diagonal
covariance Σ_q, the expected kernel score under a Gaussian query is:

  α̃_j ∝ exp(μ_q·k_j/√d + k_j^T Σ_q k_j / (2d))
                                ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                                 variance bonus

The bonus is larger for keys aligned with UNCERTAIN query directions,
implementing automatic UCB-like exploration. From the moment-generating
function of a Gaussian — exact, no approximation.

Implementation:
  - Monkey-patch self_attn.forward to add variance bonus before softmax
  - Track diagonal variance σ² in residual stream space
  - Project σ² through W_q² to get per-head query variance
  - Bonus per key: b_j = σ²_q · k_j² / (2·head_dim), normalized by α
  - Optionally propagate σ² via Herfindahl (attention concentration)

Conditions:
  baseline:     Standard attention (no bonus)
  fixed_α:      Fixed diagonal variance, sweep bonus strength α
  prop_α:       Herfindahl-propagated variance with process noise

Model: Qwen2.5-3B, 20 math problems × 2 langs, greedy decoding

Usage:
  python3 expDA1_distributional_attention.py           # full (~15 min)
  python3 expDA1_distributional_attention.py --dry      # quick test
"""

import json, time, re, argparse, types
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb, repeat_kv,
)

from expC2c_crossmodel_readhead import (
    generate_problems, get_test_subset, NumpyEncoder,
)

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"

MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_NEW = 128
LANGS = ["en", "zh"]

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)


def build_prompt(tokenizer, text):
    msgs = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def check_answer(text, correct):
    # Strip trailing periods from matches to avoid "348." != "348" bug
    nums = [n.rstrip(".") for n in re.findall(r"-?\d+\.?\d*", text)]
    return str(correct) in nums


# ═══════════════════════════════════════════════════════════════
# DISTRIBUTIONAL ATTENTION STATE
# ═══════════════════════════════════════════════════════════════


class DistributionalState:
    """Manages diagonal variance σ² and computes the distributional
    attention bonus.

    The bonus for key j at head h is:
      b_{h,j} = Σ_i (W_q[h,i,:]² · σ²) · k_j[i]² / (2 · head_dim)

    This is the variance term from the moment-generating function of
    a Gaussian query under the softmax-exponential kernel.
    """

    def __init__(self, model, alpha=0.1, propagate=False, sigma_init=1.0):
        self.alpha = alpha
        self.propagate = propagate
        self.sigma_init = sigma_init
        self.active = False
        self.sigma_sq = None

        cfg = model.config
        self.d_model = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = self.d_model // self.n_heads
        self.n_layers = cfg.num_hidden_layers
        self.kv_groups = self.n_heads // self.n_kv_heads

        # Store references to q_proj weights (already on GPU, no extra memory)
        self.q_proj_weights = [
            layer.self_attn.q_proj.weight
            for layer in model.model.layers
        ]

        # Process noise for propagated mode
        self.process_noise = sigma_init / self.n_layers

    def reset(self, device):
        """Reset variance to uniform at start of each token."""
        self.sigma_sq = torch.full(
            (self.d_model,), self.sigma_init,
            device=device, dtype=torch.float32,
        )

    def compute_bonus(self, layer_idx, key_expanded):
        """
        Compute variance bonus for attention logits.

        Args:
            key_expanded: (B, n_heads, seq, head_dim) — GQA-expanded, post-RoPE

        Returns:
            (B, n_heads, 1, seq) — bonus to add before softmax

        Note: We compute W_q² on the fly from the model weights to avoid
        storing 36 × (16, 128, 2048) float32 tensors (~600 MB).
        RoPE effect on variance is ignored (RoPE is orthogonal, preserves
        total variance per head; diagonal approximation is redistributed
        but the sum-over-dimensions bonus is similar).
        """
        W_q = self.q_proj_weights[layer_idx].data.float()  # (d_model, d_model)
        W_q = W_q.view(self.n_heads, self.head_dim, self.d_model)

        # Per-head query variance: σ²_q[h, i] = Σ_j W_q[h,i,j]² · σ²[j]
        sigma_q = torch.matmul(W_q ** 2, self.sigma_sq)  # (n_heads, head_dim)

        # Bonus: b_{h,j} = Σ_i σ²_q[h,i] · k[h,j,i]² / (2·head_dim)
        k_sq = key_expanded.float() ** 2  # (B, n_heads, seq, head_dim)
        sq = sigma_q.unsqueeze(0).unsqueeze(2)  # (1, n_heads, 1, head_dim)

        bonus = (k_sq * sq).sum(-1, keepdim=True)  # (B, n_heads, seq, 1)
        bonus = bonus.permute(0, 1, 3, 2) / (2 * self.head_dim)  # (B, n_heads, 1, seq)

        return bonus

    def update_after_attention(self, attn_weights):
        """Update σ² using Herfindahl index of attention weights.

        Herfindahl H = Σ_j α_j². Measures concentration:
        - Uniform over n keys: H = 1/n (low → variance drops)
        - Concentrated on 1 key: H = 1 (high → variance preserved)

        σ²_new = H · σ²_old + σ²_process
        """
        if not self.propagate:
            return
        herf = (attn_weights.float() ** 2).sum(-1).mean().item()
        self.sigma_sq = self.sigma_sq * herf + self.process_noise


# ═══════════════════════════════════════════════════════════════
# MONKEY-PATCH ATTENTION
# ═══════════════════════════════════════════════════════════════


def install_distributional_attention(model, dist_state):
    """Replace each layer's self_attn.forward with distributional version.

    The patched forward is identical to eager_attention_forward when
    dist_state.active is False. When active and seq_len=1 (generation),
    it adds the variance bonus before softmax.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        attn_mod = layer.self_attn

        def make_fwd(m, lidx):
            def fwd(
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                **kwargs,
            ):
                inp_shape = hidden_states.shape[:-1]
                hid_shape = (*inp_shape, -1, m.head_dim)

                q = m.q_proj(hidden_states).view(hid_shape).transpose(1, 2)
                k = m.k_proj(hidden_states).view(hid_shape).transpose(1, 2)
                v = m.v_proj(hidden_states).view(hid_shape).transpose(1, 2)

                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                if past_key_values is not None:
                    ck = {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }
                    k, v = past_key_values.update(k, v, m.layer_idx, ck)

                ke = repeat_kv(k, m.num_key_value_groups)
                ve = repeat_kv(v, m.num_key_value_groups)

                # Standard attention logits
                aw = torch.matmul(q, ke.transpose(2, 3)) * m.scaling

                # ═══ DISTRIBUTIONAL VARIANCE BONUS ═══
                if dist_state.active and q.shape[2] == 1:
                    bonus_raw = dist_state.compute_bonus(lidx, ke)

                    # Per-head normalization: max bonus = α × |max logit|
                    # Shape: (B, n_heads)
                    max_logit = (
                        aw.abs()
                        .flatten(2)
                        .max(-1)
                        .values
                        .clamp(min=1e-6)
                    )
                    bonus_max = (
                        bonus_raw
                        .flatten(2)
                        .max(-1)
                        .values
                        .clamp(min=1e-10)
                    )
                    scale = (
                        dist_state.alpha * max_logit / bonus_max
                    ).unsqueeze(-1).unsqueeze(-1)  # (B, n_heads, 1, 1)

                    aw = aw + (bonus_raw * scale).to(aw.dtype)

                if attention_mask is not None:
                    aw = aw + attention_mask

                aw = nn.functional.softmax(
                    aw, dim=-1, dtype=torch.float32
                ).to(q.dtype)

                # Update variance state
                if dist_state.active and q.shape[2] == 1:
                    dist_state.update_after_attention(aw)

                ao = torch.matmul(aw, ve)
                ao = ao.transpose(1, 2).contiguous()
                ao = ao.reshape(*inp_shape, -1).contiguous()
                ao = m.o_proj(ao)

                return ao, None

            return fwd

        attn_mod.forward = make_fwd(attn_mod, layer_idx)


# ═══════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════


def generate(model, tokenizer, prompt, device, dist_state, use_bonus, max_new):
    """Token-by-token greedy generation.

    If use_bonus is True, the distributional variance bonus is applied
    during generation steps (seq_len=1). Prompt encoding is always
    unmodified (the shape check in the patched forward skips the bonus
    when seq_len > 1).
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = []
    past_kv = None
    cur = ids

    dist_state.active = use_bonus

    with torch.inference_mode():
        for _ in range(max_new):
            if use_bonus:
                dist_state.reset(device)

            out = model(cur, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax().item())
            generated.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break
            cur = torch.tensor([[next_id]], device=device)

    dist_state.active = False
    return tokenizer.decode(generated, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="3 problems, 64 tokens")
    args = parser.parse_args()

    device = "cuda"
    max_new = 64 if args.dry else MAX_NEW

    print("=" * 60)
    print("Exp DA1: Distributional Attention — Variance Bonus")
    print("=" * 60)
    print(f"Model:     {MODEL_NAME}")
    print(f"Max tokens: {max_new}")
    print()

    # ── Load model ──
    print("Loading model...")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",  # must match monkey-patch
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    # ── Problems ──
    all_problems = generate_problems()
    test_problems = get_test_subset(all_problems)
    if args.dry:
        test_problems = test_problems[:3]
    n_eval = len(test_problems) * len(LANGS)
    print(f"  {len(test_problems)} problems × {len(LANGS)} langs = {n_eval} evals/condition")

    # ── Install distributional attention ──
    print("Installing distributional attention hooks...")
    dist_state = DistributionalState(model, alpha=0.1, propagate=False)
    install_distributional_attention(model, dist_state)
    print(f"  Patched {dist_state.n_layers} layers")
    print(f"  Config: d={dist_state.d_model}, heads={dist_state.n_heads}, "
          f"kv_heads={dist_state.n_kv_heads}, head_dim={dist_state.head_dim}")

    # ── Conditions ──
    conditions = [
        # (name, alpha, propagate, use_bonus)
        ("baseline",   0.0,  False, False),
        ("fixed_0.01", 0.01, False, True),
        ("fixed_0.05", 0.05, False, True),
        ("fixed_0.1",  0.1,  False, True),
        ("fixed_0.5",  0.5,  False, True),
        ("prop_0.1",   0.1,  True,  True),
    ]

    results = {
        "experiment": "DA1_distributional_attention",
        "model": MODEL_NAME,
        "max_new_tokens": max_new,
        "n_problems": len(test_problems),
        "langs": LANGS,
    }
    per_problem = []

    for cond_name, alpha, propagate, use_bonus in conditions:
        print(f"\n{'─'*50}")
        print(f"Condition: {cond_name}  (α={alpha}, propagate={propagate})")
        print(f"{'─'*50}")

        dist_state.alpha = alpha
        dist_state.propagate = propagate

        correct = {lang: 0 for lang in LANGS}
        total = {lang: 0 for lang in LANGS}
        t0 = time.time()

        for pi, prob in enumerate(test_problems):
            for lang in LANGS:
                prompt = build_prompt(tokenizer, prob[lang])
                text = generate(
                    model, tokenizer, prompt, device,
                    dist_state, use_bonus, max_new,
                )

                ok = check_answer(text, prob["answer"])
                correct[lang] += int(ok)
                total[lang] += 1

                per_problem.append({
                    "condition": cond_name,
                    "problem_idx": pi,
                    "lang": lang,
                    "category": prob["category"],
                    "correct": ok,
                    "answer": prob["answer"],
                    "output_snippet": text[:300],
                })

                status = "✓" if ok else "✗"
                print(f"  [{pi+1:2d}/{len(test_problems)}] {lang.upper()} {status}  "
                      f"ans={prob['answer']}  "
                      f"({prob['category']})")

        elapsed = time.time() - t0
        scores = {lang: correct[lang] for lang in LANGS}
        scores["total"] = sum(correct.values())
        scores["max"] = sum(total.values())
        scores["time_s"] = round(elapsed, 1)
        results[cond_name] = scores

        summary = " | ".join(
            f"{l.upper()}={correct[l]}/{total[l]}" for l in LANGS
        )
        print(f"\n  ► {summary} | Total={scores['total']}/{scores['max']} | {elapsed:.1f}s")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'EN':>5} {'ZH':>5} {'Total':>7} {'Time':>7}")
    print(f"{'-'*15} {'-'*5} {'-'*5} {'-'*7} {'-'*7}")
    for cond_name, _, _, _ in conditions:
        s = results[cond_name]
        print(f"{cond_name:<15} {s['en']:>5} {s['zh']:>5} "
              f"{s['total']:>4}/{s['max']:<2} {s['time_s']:>6.1f}s")

    # ── Save ──
    results["per_problem"] = per_problem
    out_path = OUTPUT_DIR / "expDA1_distributional_attention.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
