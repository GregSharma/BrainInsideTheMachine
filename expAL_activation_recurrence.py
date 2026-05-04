"""
Exp AL: Activation Recurrence — The GFOAMS

Von Neumann's ultimate question: can the model compute WITHOUT generating tokens?

Standard autoregressive: embed → layers → logit → token → embed → layers → logit → ...
The token bottleneck forces 151936-way discretization at every step.

THIS EXPERIMENT: skip the bottleneck. Feed the model's final-layer activation
back as input and iterate. Track whether the answer emerges, stabilizes, or diverges.

Setup:
  1. Forward pass on math prompt → get h_L (last layer, last token) = the model's "intent"
  2. Define mixing function f that combines original input embeddings with h_L
  3. Replace input embeddings with f(original, h_L)
  4. Forward pass again → new h_L' → iterate
  5. At each iteration: what token would the model predict? Does accuracy change?
     Track trajectory in activation space.

Mixing functions to test:
  A. REPLACE_LAST: replace last-token embedding with projected h_L. Simple feedback.
  B. LINEAR_MIX: a'_m = w * a_m + (1-w) * proj(h_L), sweep w from 0 to 1.
  C. APPEND: don't replace — add h_L as a NEW position (sequence grows by 1 each iter).
     This is closest to autoregressive but without tokenization.
  D. GREG_GRADIENT: spread the feedback across positions with linearly varying weights.
     Position i gets: ((m-1-i)/(m-1)) * a_i + (i/(m-1)) * proj(h_L)
     The last position = pure h_L, first position = pure original.

Key insight from Exp K2b: KV cache is expendable. So we can rebuild it each iteration.
Key insight from Exp Y: raw layer iteration diverges. But Y didn't re-embed through
the full model — it just looped L9-L25. This experiment re-runs ALL 36 layers each time.

On Qwen2.5-3B. 5 math problems, up to 20 iterations each.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import random as pyrandom
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
device = "cuda"
MAX_ITERATIONS = 20

# ── Load Model ──────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="eager"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

d = model.config.hidden_size          # 2048
n_layers = model.config.num_hidden_layers  # 36
V = model.config.vocab_size           # 151936

# Get embedding and lm_head weights
embed_weight = model.model.embed_tokens.weight.data  # (V, d)
lm_head_weight = model.lm_head.weight.data           # (V, d) — tied with embed in Qwen

print(f"Model loaded: d={d}, L={n_layers}, V={V}")
print(f"Embeddings tied: {model.lm_head.weight.data_ptr() == embed_weight.data_ptr()}")


# ── Test Problems ───────────────────────────────────────────────────────
PROBLEMS = [
    {"prompt": "Calculate 47 + 86.", "answer": "133", "lang": "en"},
    {"prompt": "Calculate 664 + 124.", "answer": "788", "lang": "en"},
    {"prompt": "计算 47 + 86 的值。", "answer": "133", "lang": "zh"},
    {"prompt": "计算 15 × 8 的值。", "answer": "120", "lang": "zh"},
    {"prompt": "What is the remainder when 100 is divided by 7?", "answer": "2", "lang": "en"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60", "lang": "en"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120", "lang": "en"},
    {"prompt": "计算 238 + 152 的值。", "answer": "390", "lang": "zh"},
]


# ── Projection: h_L → embedding space ──────────────────────────────────
def project_to_embed_space(h_L, method="soft"):
    """
    Project a last-layer hidden state back to embedding space.

    method="hard": argmax → embed(token). Full discretization.
    method="soft": softmax(lm_head(h_L)) @ embed_weight. Soft projection.
    method="raw": just use h_L directly (it's the same dimensionality).
                  Apply final layernorm inverse approximately.
    """
    h = h_L.float()

    if method == "hard":
        # Discretize through vocabulary
        logits = model.lm_head(model.model.norm(h.unsqueeze(0).to(embed_weight.dtype)))
        token_id = logits.argmax(dim=-1).item()
        return embed_weight[token_id].float(), token_id

    elif method == "soft":
        # Soft mixture of embeddings weighted by logit probabilities
        normed = model.model.norm(h.unsqueeze(0).to(embed_weight.dtype))
        logits = model.lm_head(normed).float().squeeze(0)  # (V,)
        # Temperature-scaled softmax (low temp = more peaked = closer to hard)
        probs = F.softmax(logits / 0.1, dim=-1)  # (V,)
        soft_embed = probs @ embed_weight.float()  # (d,)
        top_token = logits.argmax().item()
        return soft_embed, top_token

    elif method == "raw":
        # Use h_L directly — same dim as embedding. No projection needed.
        # But we should normalize to embedding scale
        embed_norm = embed_weight.float().norm(dim=-1).mean()
        h_normalized = h * (embed_norm / h.norm())
        return h_normalized, -1

    else:
        raise ValueError(f"Unknown method: {method}")


# ── Core: Single forward pass with custom embeddings ────────────────────
def forward_with_custom_embeds(model, custom_embeds):
    """
    Run model forward pass using custom embedding tensor instead of token IDs.

    custom_embeds: (1, seq_len, d) tensor in model dtype
    Returns: h_last_layer_last_token (d,), logits_last_token (V,), all_layer_states
    """
    embeds = custom_embeds.to(embed_weight.dtype)

    # Hook last layer + every 6th for trajectory tracking — ONE forward pass
    captured = {}
    hooks = []

    for l in list(range(0, n_layers, 6)) + [n_layers - 1]:
        def make_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h[0, -1, :].detach().float().cpu().numpy()
            return hook
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    with torch.no_grad():
        out = model(inputs_embeds=embeds)

    for h in hooks:
        h.remove()

    # h_final = last layer output (pre-norm). For projection back to embed space,
    # we use the logits which already go through norm + lm_head.
    h_final = torch.tensor(captured[n_layers - 1], device=device)  # (d,)
    logits_last = out.logits[0, -1, :].detach().float()  # (V,)
    state_list = [captured.get(l, np.zeros(d)) for l in range(0, n_layers, 6)]

    return h_final, logits_last, state_list


# ── Mixing Functions ────────────────────────────────────────────────────
def mix_replace_last(original_embeds, h_projected, iteration):
    """Replace the last-token embedding with projected h_L."""
    new_embeds = original_embeds.clone()
    new_embeds[0, -1, :] = h_projected.to(new_embeds.dtype)
    return new_embeds


def mix_linear(original_embeds, h_projected, iteration, w=0.5):
    """Mix last-token embedding: w*original + (1-w)*h_projected."""
    new_embeds = original_embeds.clone()
    orig_last = original_embeds[0, -1, :].float()
    mixed = w * orig_last + (1 - w) * h_projected
    new_embeds[0, -1, :] = mixed.to(new_embeds.dtype)
    return new_embeds


def mix_append(original_embeds, h_projected, iteration):
    """Append h_projected as a new position (sequence grows)."""
    new_pos = h_projected.unsqueeze(0).unsqueeze(0).to(original_embeds.dtype)  # (1, 1, d)
    return torch.cat([original_embeds, new_pos], dim=1)


def mix_greg_gradient(original_embeds, h_projected, iteration):
    """
    Greg's gradient scheme: spread feedback across positions.
    Position i gets: ((m-1-i)/(m-1)) * a_i + (i/(m-1)) * proj(h_L)
    First position = pure original, last position = pure h_L.
    """
    new_embeds = original_embeds.clone().float()
    m = new_embeds.shape[1]
    h = h_projected.float()

    for i in range(m):
        w_orig = (m - 1 - i) / max(m - 1, 1)
        w_new = i / max(m - 1, 1)
        new_embeds[0, i, :] = w_orig * original_embeds[0, i, :].float() + w_new * h

    return new_embeds.to(original_embeds.dtype)


def mix_decay_gradient(original_embeds, h_projected, iteration):
    """
    Like greg_gradient but the mixing strength increases with iteration.
    At iteration 0: all original. At iteration MAX: full gradient.
    This lets the model gradually "absorb" its own output.
    """
    new_embeds = original_embeds.clone().float()
    m = new_embeds.shape[1]
    h = h_projected.float()

    # Iteration-dependent strength (0 at iter 0, 1 at iter MAX)
    strength = min(iteration / 10.0, 1.0)

    for i in range(m):
        w_new = strength * (i / max(m - 1, 1))
        w_orig = 1.0 - w_new
        new_embeds[0, i, :] = w_orig * original_embeds[0, i, :].float() + w_new * h

    return new_embeds.to(original_embeds.dtype)


# ── Run one problem through the recurrence ──────────────────────────────
def run_recurrence(model, tokenizer, problem, mix_fn, mix_name, proj_method="soft",
                   max_iter=MAX_ITERATIONS):
    """
    Run the activation recurrence for one problem.
    Returns trajectory data.
    """
    prompt = problem["prompt"]
    answer = problem["answer"]

    # Step 0: Get original embeddings
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    original_embeds = model.model.embed_tokens(input_ids).detach()  # (1, seq, d)

    # Track trajectory
    trajectory = []

    # Initial state: standard forward pass
    current_embeds = original_embeds.clone()

    for iteration in range(max_iter + 1):
        # Forward pass
        h_final, logits_last, layer_states = forward_with_custom_embeds(model, current_embeds)

        # What token would the model predict?
        top5_tokens = logits_last.topk(5)
        top5_ids = top5_tokens.indices.cpu().tolist()
        top5_probs = F.softmax(logits_last, dim=-1)[top5_ids].cpu().tolist()
        top5_text = [tokenizer.decode([t]) for t in top5_ids]

        predicted_token = top5_ids[0]
        predicted_text = top5_text[0]

        # Check if answer appears in top prediction
        answer_in_top1 = answer in predicted_text
        answer_in_top5 = any(answer in t for t in top5_text)

        # Also check: what's the rank of the answer token?
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        answer_rank = -1
        if answer_tokens:
            target_id = answer_tokens[0]
            sorted_logits = logits_last.argsort(descending=True)
            rank_positions = (sorted_logits == target_id).nonzero()
            if len(rank_positions) > 0:
                answer_rank = rank_positions[0].item()

        # Trajectory metrics
        h_np = h_final.cpu().numpy()
        step_data = {
            "iteration": iteration,
            "seq_len": current_embeds.shape[1],
            "h_norm": float(np.linalg.norm(h_np)),
            "predicted_token": predicted_text,
            "predicted_token_id": predicted_token,
            "top5": list(zip(top5_text, [round(p, 4) for p in top5_probs])),
            "answer_in_top1": answer_in_top1,
            "answer_in_top5": answer_in_top5,
            "answer_rank": answer_rank,
            "logit_entropy": float(-(F.softmax(logits_last, dim=-1) *
                                     F.log_softmax(logits_last, dim=-1)).sum()),
        }

        # Track cosine with previous iteration
        if iteration > 0:
            prev_h = trajectory[-1]["_h"]
            cos_sim = float(F.cosine_similarity(
                torch.tensor(h_np).unsqueeze(0),
                torch.tensor(prev_h).unsqueeze(0)
            ))
            step_data["cos_with_prev"] = round(cos_sim, 6)

        # Track cosine with iteration 0
        if iteration > 0:
            h0 = trajectory[0]["_h"]
            cos_sim_0 = float(F.cosine_similarity(
                torch.tensor(h_np).unsqueeze(0),
                torch.tensor(h0).unsqueeze(0)
            ))
            step_data["cos_with_iter0"] = round(cos_sim_0, 6)

        step_data["_h"] = h_np  # Keep for trajectory analysis, won't save to JSON

        # Layer-by-layer norms for divergence tracking
        step_data["layer_norms"] = [round(float(np.linalg.norm(s)), 2) for s in layer_states[::6]]

        trajectory.append(step_data)

        print(f"    iter {iteration}: pred='{predicted_text}' "
              f"rank={answer_rank} norm={step_data['h_norm']:.1f} "
              f"entropy={step_data['logit_entropy']:.2f}"
              + (f" cos_prev={step_data.get('cos_with_prev', 'N/A')}" if iteration > 0 else ""))

        # Prepare for next iteration
        if iteration < max_iter:
            # Project h_final back to embedding space
            h_projected, _ = project_to_embed_space(h_final, method=proj_method)

            # Apply mixing function
            current_embeds = mix_fn(original_embeds, h_projected, iteration)

    return trajectory


# ── Also: standard autoregressive baseline for comparison ───────────────
def run_baseline(model, tokenizer, problem, n_tokens=5):
    """Standard autoregressive generation for comparison."""
    inputs = tokenizer(problem["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=n_tokens,
            do_sample=False, temperature=None, top_p=None
        )
    gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen_text


# ══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP AL: ACTIVATION RECURRENCE — THE GFOAMS")
print("="*70)

mixing_functions = {
    "replace_last": (mix_replace_last, "soft"),
    "linear_w05": (lambda o, h, i: mix_linear(o, h, i, w=0.5), "soft"),
    "linear_w08": (lambda o, h, i: mix_linear(o, h, i, w=0.8), "soft"),
    "greg_gradient": (mix_greg_gradient, "soft"),
    "decay_gradient": (mix_decay_gradient, "soft"),
    "replace_last_hard": (mix_replace_last, "hard"),
    "replace_last_raw": (mix_replace_last, "raw"),
    "append_soft": (mix_append, "soft"),
}

all_results = {}

for prob_idx, problem in enumerate(PROBLEMS):
    print(f"\n{'─'*60}")
    print(f"Problem {prob_idx}: {problem['prompt']} (answer={problem['answer']})")
    print(f"{'─'*60}")

    # Baseline
    baseline = run_baseline(model, tokenizer, problem, n_tokens=10)
    print(f"  Baseline (10 tokens): {baseline[:60]}...")

    prob_results = {"prompt": problem["prompt"], "answer": problem["answer"],
                    "lang": problem["lang"], "baseline": baseline, "conditions": {}}

    for mix_name, (mix_fn, proj_method) in mixing_functions.items():
        print(f"\n  === {mix_name} (proj={proj_method}) ===")

        try:
            trajectory = run_recurrence(
                model, tokenizer, problem, mix_fn, mix_name,
                proj_method=proj_method, max_iter=15
            )

            # Clean trajectory for JSON (remove numpy arrays)
            clean_trajectory = []
            for step in trajectory:
                clean_step = {k: v for k, v in step.items() if k != "_h"}
                clean_trajectory.append(clean_step)

            # Summary metrics
            first_correct_iter = -1
            last_correct_iter = -1
            n_correct = 0
            for step in trajectory:
                if step["answer_rank"] >= 0 and step["answer_rank"] < 10:
                    n_correct += 1
                    if first_correct_iter == -1:
                        first_correct_iter = step["iteration"]
                    last_correct_iter = step["iteration"]

            # Did it converge? (cosine with prev > 0.999 for 3+ consecutive steps)
            converged_at = -1
            if len(trajectory) > 3:
                for i in range(2, len(trajectory)):
                    if all(trajectory[j].get("cos_with_prev", 0) > 0.999
                           for j in range(i-1, i+1) if "cos_with_prev" in trajectory[j]):
                        converged_at = i - 1
                        break

            # Norm trajectory
            norms = [step["h_norm"] for step in trajectory]
            norm_growth = norms[-1] / max(norms[0], 1e-8)

            summary = {
                "n_iterations": len(trajectory),
                "first_correct_iter": first_correct_iter,
                "last_correct_iter": last_correct_iter,
                "n_correct_in_top10": n_correct,
                "converged_at": converged_at,
                "norm_growth": round(norm_growth, 4),
                "final_predicted": trajectory[-1]["predicted_token"],
                "final_answer_rank": trajectory[-1]["answer_rank"],
            }

            prob_results["conditions"][mix_name] = {
                "summary": summary,
                "trajectory": clean_trajectory,
            }

            print(f"    Summary: correct_in_top10={n_correct}/{len(trajectory)}, "
                  f"norm_growth={norm_growth:.2f}x, converged_at={converged_at}")

        except Exception as e:
            print(f"    ERROR: {e}")
            prob_results["conditions"][mix_name] = {"error": str(e)}

    all_results[f"problem_{prob_idx}"] = prob_results


# ── Save ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output = {
    "experiment": "AL: Activation Recurrence — The GFOAMS",
    "model": MODEL_NAME,
    "description": "Can the model compute without generating tokens? "
                   "Feed last-layer activation back as input embedding, iterate.",
    "mixing_functions": list(mixing_functions.keys()),
    "projection_methods": ["soft", "hard", "raw"],
    "max_iterations": MAX_ITERATIONS,
    "results": all_results,
}

output_path = OUTPUT_DIR / "expAL_activation_recurrence.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to {output_path}")

# ── Grand Summary ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("GRAND SUMMARY")
print("="*70)

for prob_key, prob_data in all_results.items():
    print(f"\n{prob_data['prompt']} (ans={prob_data['answer']})")
    for cond_name, cond_data in prob_data.get("conditions", {}).items():
        if "summary" in cond_data:
            s = cond_data["summary"]
            print(f"  {cond_name}: top10={s['n_correct_in_top10']}/16 "
                  f"norm={s['norm_growth']:.1f}x conv@{s['converged_at']} "
                  f"final='{s['final_predicted']}' rank={s['final_answer_rank']}")
        elif "error" in cond_data:
            print(f"  {cond_name}: ERROR — {cond_data['error'][:60]}")

print("\nDone.")
