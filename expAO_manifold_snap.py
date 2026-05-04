"""
Exp AO: Manifold-Snapped Token Compression

AN showed that compressing tokens to k arbitrary vectors fails — they leave
the embedding manifold. AM showed that differences (translations) work because
they stay on it. AL showed that re-embedding through vocabulary works.

This experiment: compress, then SNAP BACK to the manifold by finding the
nearest vocabulary token for each compressed vector. The question becomes:
does the "nearest token" sequence encode enough of the original problem to solve it?

Also: what if we use soft-tokens (weighted average of top-k nearest tokens)?

Methods:
1. MEAN-POOL + SNAP: Pool windows → find nearest vocab token → use those tokens
2. SVD + SNAP: SVD compress → snap each component to nearest vocab token
3. SOFT-PROMPT: Learn (via gradient descent) k token embeddings that minimize
   the KL divergence between full-prompt logits and compressed-prompt logits
4. CENTROID: Find the single vocab token closest to the mean of all embeddings
5. GREEDY-IMPORTANT: Use attention weights from a forward pass to pick the k
   most important tokens (attention-weighted selection)

On Qwen2.5-3B locally.
"""

import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
device = "cuda"
MAX_NEW_TOKENS = 128

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    attn_implementation="eager",  # Need attention weights
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
d_model = model.config.hidden_size
n_layers = model.config.num_hidden_layers
embed_weight = model.model.embed_tokens.weight.float()  # (V, d)

# Normalize embed_weight for cosine search
embed_norms = embed_weight.norm(dim=-1, keepdim=True)
embed_normed = embed_weight / embed_norms.clamp(min=1e-8)


# ── Problems ────────────────────────────────────────────────────────────
PROBLEMS = [
    {"prompt": "计算 47 + 86 的值。", "answer": "133", "lang": "zh"},
    {"prompt": "Calculate 664 + 124.", "answer": "788", "lang": "en"},
    {"prompt": "计算 15 × 8 的值。", "answer": "120", "lang": "zh"},
    {"prompt": "Calculate 238 + 152.", "answer": "390", "lang": "en"},
    {"prompt": "What is the remainder when 100 is divided by 7?", "answer": "2", "lang": "en"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60", "lang": "en"},
    {"prompt": "计算 664 + 124 的值。", "answer": "788", "lang": "zh"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120", "lang": "en"},
    {"prompt": "计算 238 + 152 的值。", "answer": "390", "lang": "zh"},
    {"prompt": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60", "lang": "zh"},
]


# ── Helpers ─────────────────────────────────────────────────────────────

def get_token_ids(prompt):
    return tokenizer(prompt, return_tensors="pt").to(device)["input_ids"][0]


def get_embeddings(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model.model.embed_tokens(inputs["input_ids"])
    return embeds.squeeze(0).float()


def nearest_token(vec):
    """Find the nearest vocabulary token to a vector (cosine similarity)."""
    vec_normed = vec / vec.norm().clamp(min=1e-8)
    sims = vec_normed @ embed_normed.T  # (V,)
    idx = sims.argmax().item()
    return idx, sims[idx].item()


def generate_from_ids(token_ids):
    """Generate from token IDs."""
    if isinstance(token_ids, list):
        token_ids = torch.tensor(token_ids, device=device).unsqueeze(0)
    elif token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    with torch.no_grad():
        out = model.generate(
            input_ids=token_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_from_embeds(embeds_2d, max_tokens=MAX_NEW_TOKENS):
    embeds = embeds_2d.unsqueeze(0).to(model.model.embed_tokens.weight.dtype).to(device)
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=embeds,
            max_new_tokens=max_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def check_answer(gen_text, answer):
    return answer in gen_text


# ── Compression Methods ─────────────────────────────────────────────────

def compress_mean_pool_snap(embeds, k):
    """Mean-pool to k vectors, snap each to nearest vocab token."""
    seq_len = embeds.shape[0]
    if k >= seq_len:
        return get_token_ids(None), embeds  # identity

    indices = torch.linspace(0, seq_len, k + 1).long()
    token_ids = []
    snapped_embeds = []
    snap_info = []

    for i in range(k):
        start, end = indices[i].item(), indices[i + 1].item()
        if end <= start:
            end = start + 1
        pooled = embeds[start:end].mean(dim=0)
        tid, sim = nearest_token(pooled)
        token_ids.append(tid)
        snapped_embeds.append(embed_weight[tid])
        snap_info.append({
            "token": tokenizer.decode([tid]),
            "cosine_sim": round(sim, 4),
            "window": f"{start}:{end}",
        })

    return torch.tensor(token_ids, device=device), torch.stack(snapped_embeds), snap_info


def compress_attention_select(embeds, k, prompt):
    """Use attention weights to select the k most important tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    # Average attention across heads and layers for the last token
    # Shape per layer: (1, n_heads, seq, seq)
    attn_weights = []
    for layer_attn in out.attentions:
        # Mean across heads, take last token's attention to all positions
        last_tok_attn = layer_attn[0].mean(dim=0)[-1, :]  # (seq,)
        attn_weights.append(last_tok_attn)

    # Average across layers
    avg_attn = torch.stack(attn_weights).mean(dim=0)  # (seq,)

    # Select top-k positions
    _, top_indices = avg_attn.topk(min(k, len(avg_attn)))
    top_indices = top_indices.sort().values  # maintain order

    selected_ids = inputs["input_ids"][0][top_indices]
    selected_embeds = embeds[top_indices]

    selected_tokens = [tokenizer.decode([tid]) for tid in selected_ids.tolist()]

    return selected_ids, selected_embeds, {
        "selected_positions": top_indices.tolist(),
        "selected_tokens": selected_tokens,
        "attention_scores": avg_attn[top_indices].tolist(),
    }


def compress_soft_prompt(embeds, k, prompt, n_steps=50, lr=0.1):
    """
    Gradient-optimize k token embeddings to match the full-prompt logits.
    This is a mini soft-prompt tuning.
    """
    seq_len = embeds.shape[0]

    # Get target logits from full prompt
    full_embeds = embeds.unsqueeze(0).to(model.model.embed_tokens.weight.dtype).to(device)
    with torch.no_grad():
        target_out = model(inputs_embeds=full_embeds)
        target_logits = target_out.logits[0, -1, :].float()
        target_probs = F.softmax(target_logits, dim=-1)

    # Initialize soft tokens from mean-pooled embeddings
    indices = torch.linspace(0, seq_len, k + 1).long()
    init_embeds = []
    for i in range(k):
        start, end = indices[i].item(), indices[i + 1].item()
        if end <= start:
            end = start + 1
        init_embeds.append(embeds[start:end].mean(dim=0))

    soft_embeds = torch.stack(init_embeds).to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([soft_embeds], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        inp = soft_embeds.unsqueeze(0).to(model.model.embed_tokens.weight.dtype)
        out = model(inputs_embeds=inp)
        pred_logits = out.logits[0, -1, :].float()
        pred_probs = F.log_softmax(pred_logits, dim=-1)

        loss = F.kl_div(pred_probs, target_probs, reduction='batchmean')
        loss.backward()
        optimizer.step()

    return soft_embeds.detach().float(), loss.item()


K_VALUES = [1, 2, 3, 5, 8]


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP AO: MANIFOLD-SNAPPED TOKEN COMPRESSION")
print("=" * 70)

all_results = {}

for pi, prob in enumerate(PROBLEMS):
    prompt = prob["prompt"]
    answer = prob["answer"]
    lang = prob["lang"]

    print(f"\n{'─' * 60}")
    print(f"P{pi}: {prompt} (answer={answer})")
    print(f"{'─' * 60}")

    full_embeds = get_embeddings(prompt)
    n_tokens = full_embeds.shape[0]
    original_ids = get_token_ids(prompt)
    print(f"  Full: {n_tokens} tokens: {tokenizer.decode(original_ids)}")

    # Baseline
    try:
        baseline_gen = generate_from_ids(original_ids)
        baseline_correct = check_answer(baseline_gen, answer)
    except Exception as e:
        baseline_gen = f"ERROR: {e}"
        baseline_correct = False
    print(f"  Baseline: {'Y' if baseline_correct else 'N'} — {baseline_gen[:50]}...")

    prob_results = {
        "prompt": prompt, "answer": answer, "lang": lang,
        "n_tokens": n_tokens,
        "baseline": {"correct": baseline_correct, "gen": baseline_gen[:150]},
        "conditions": {},
    }

    for k in K_VALUES:
        if k > n_tokens:
            continue

        # Method 1: Mean-pool + snap to nearest token
        try:
            snap_ids, snap_embeds, snap_info = compress_mean_pool_snap(full_embeds, k)
            snap_text = tokenizer.decode(snap_ids)
            gen = generate_from_ids(snap_ids)
            correct = check_answer(gen, answer)
            prob_results["conditions"][f"mean_snap_k{k}"] = {
                "correct": correct,
                "gen": gen[:150],
                "snap_text": snap_text,
                "snap_info": snap_info,
            }
            print(f"  mean_snap_k{k:2d}: {'Y' if correct else 'N'} tokens='{snap_text}' → {gen[:40]}...")
        except Exception as e:
            print(f"  mean_snap_k{k:2d}: ERROR {e}")
            prob_results["conditions"][f"mean_snap_k{k}"] = {"error": str(e)}

        # Method 2: Attention-weighted selection
        try:
            attn_ids, attn_embeds, attn_info = compress_attention_select(full_embeds, k, prompt)
            attn_text = tokenizer.decode(attn_ids)
            gen = generate_from_ids(attn_ids)
            correct = check_answer(gen, answer)
            prob_results["conditions"][f"attn_select_k{k}"] = {
                "correct": correct,
                "gen": gen[:150],
                "selected_text": attn_text,
                "info": attn_info,
            }
            print(f"  attn_sel_k{k:2d} : {'Y' if correct else 'N'} tokens='{attn_text}' → {gen[:40]}...")
        except Exception as e:
            print(f"  attn_sel_k{k:2d} : ERROR {e}")
            prob_results["conditions"][f"attn_select_k{k}"] = {"error": str(e)}

        # Method 3: Soft-prompt optimization (only for k=3,5 — expensive)
        if k in [3, 5]:
            try:
                soft_embeds, final_loss = compress_soft_prompt(full_embeds, k, prompt)
                gen = generate_from_embeds(soft_embeds)
                correct = check_answer(gen, answer)

                # Also snap the optimized soft tokens to nearest vocab
                soft_snapped_ids = []
                for i in range(k):
                    tid, sim = nearest_token(soft_embeds[i])
                    soft_snapped_ids.append(tid)
                soft_snapped_text = tokenizer.decode(soft_snapped_ids)

                prob_results["conditions"][f"soft_prompt_k{k}"] = {
                    "correct": correct,
                    "gen": gen[:150],
                    "final_loss": round(final_loss, 4),
                    "snapped_text": soft_snapped_text,
                }
                print(f"  soft_pr_k{k:2d}  : {'Y' if correct else 'N'} loss={final_loss:.4f} snap='{soft_snapped_text}' → {gen[:40]}...")
            except Exception as e:
                print(f"  soft_pr_k{k:2d}  : ERROR {e}")
                prob_results["conditions"][f"soft_prompt_k{k}"] = {"error": str(e)}

    # Method 4: Centroid token (k=1 special case)
    try:
        mean_embed = full_embeds.mean(dim=0)
        centroid_id, centroid_sim = nearest_token(mean_embed)
        centroid_text = tokenizer.decode([centroid_id])
        gen = generate_from_ids(torch.tensor([centroid_id], device=device))
        correct = check_answer(gen, answer)
        prob_results["conditions"]["centroid"] = {
            "correct": correct,
            "gen": gen[:150],
            "centroid_token": centroid_text,
            "cosine_sim": round(centroid_sim, 4),
        }
        print(f"  centroid     : {'Y' if correct else 'N'} token='{centroid_text}' sim={centroid_sim:.4f} → {gen[:40]}...")
    except Exception as e:
        print(f"  centroid     : ERROR {e}")

    all_results[f"problem_{pi}"] = prob_results


# ── Save ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

output = {
    "experiment": "AO: Manifold-Snapped Token Compression",
    "model": MODEL_NAME,
    "k_values": K_VALUES,
    "methods": ["mean_snap", "attn_select", "soft_prompt", "centroid"],
    "n_problems": len(PROBLEMS),
    "results": all_results,
}

with open(OUTPUT_DIR / "expAO_manifold_snap.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("Saved to output/expAO_manifold_snap.json")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== GRAND SUMMARY ===")
for method in ["mean_snap", "attn_select", "soft_prompt"]:
    scores = []
    for k in K_VALUES:
        cond = f"{method}_k{k}"
        n_correct = sum(1 for pv in all_results.values()
                        if pv["conditions"].get(cond, {}).get("correct", False))
        n_total = sum(1 for pv in all_results.values()
                      if cond in pv["conditions"] and "error" not in pv["conditions"][cond])
        if n_total > 0:
            scores.append(f"{n_correct}/{n_total}")
        else:
            scores.append("  - ")
    print(f"{method:20s}: " + " ".join(f"k={k}:{s}" for k, s in zip(K_VALUES, scores)))

# Centroid
n_c = sum(1 for pv in all_results.values() if pv["conditions"].get("centroid", {}).get("correct", False))
print(f"{'centroid':20s}: {n_c}/{len(PROBLEMS)}")

n_b = sum(1 for pv in all_results.values() if pv["baseline"]["correct"])
print(f"{'baseline':20s}: {n_b}/{len(PROBLEMS)}")

print("\nDone.")
