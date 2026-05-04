"""
Exp AN: Hierarchical Token Compression — "The Tok Sliding Experiment"

Can a math problem be compressed to fewer tokens and still solved?

If KV cache is expendable (K2b) and token sequence is a scaffold (Y/AL),
maybe fewer tokens suffice if they encode the right hidden state.

Methods:
1. MEAN-POOL: Sliding window mean-pool of token embeddings → k tokens
2. SVD-COMPRESS: Take all token embeddings, SVD, keep top-k left singular vectors
3. LAST-K: Just use the last k token embeddings (autoregressive context)
4. STRIDE: Take every N-th token embedding
5. PCA-PROJECT: Project all tokens onto top-k PCA dimensions, use k synthetic tokens
6. ORACLE: Use the hidden state at last token from layer L as a single "token"

For each: embed original prompt → compress to k vectors → forward pass → check answer.

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
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
d_model = model.config.hidden_size
embed_weight = model.model.embed_tokens.weight  # (V, d)

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

K_VALUES = [1, 2, 3, 4, 5, 8]  # Number of compressed tokens


# ── Helpers ─────────────────────────────────────────────────────────────

def get_embeddings(prompt):
    """Get raw token embeddings. Returns (seq_len, d) float tensor."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model.model.embed_tokens(inputs["input_ids"])
    return embeds.squeeze(0).float()  # (seq, d)


def generate_from_embeds(embeds_2d, max_tokens=MAX_NEW_TOKENS):
    """Generate from (seq, d) embeddings."""
    embeds = embeds_2d.unsqueeze(0).to(model.model.embed_tokens.weight.dtype).to(device)
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=embeds,
            max_new_tokens=max_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def forward_logits(embeds_2d):
    """Single forward pass, return top-5 and answer rank info."""
    embeds = embeds_2d.unsqueeze(0).to(model.model.embed_tokens.weight.dtype).to(device)
    with torch.no_grad():
        out = model(inputs_embeds=embeds)
    logits = out.logits[0, -1, :].float()
    top5 = logits.topk(5)
    top5_text = [tokenizer.decode([t]) for t in top5.indices.tolist()]
    return {
        "top5": list(zip(top5_text, [round(p, 4) for p in F.softmax(logits, dim=-1)[top5.indices].tolist()])),
        "top1": top5_text[0],
        "entropy": round(float(-(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum()), 2),
    }


def check_answer(gen_text, answer):
    """Check if answer appears in generated text."""
    return answer in gen_text


# ── Compression Methods ─────────────────────────────────────────────────

def compress_mean_pool(embeds, k):
    """Sliding window mean-pool to k tokens."""
    seq_len = embeds.shape[0]
    if k >= seq_len:
        return embeds
    # Split into k equal-ish windows
    indices = torch.linspace(0, seq_len, k + 1).long()
    compressed = []
    for i in range(k):
        start, end = indices[i].item(), indices[i + 1].item()
        if end <= start:
            end = start + 1
        compressed.append(embeds[start:end].mean(dim=0))
    return torch.stack(compressed)  # (k, d)


def compress_svd(embeds, k):
    """SVD of embedding matrix, keep top-k left singular vectors scaled by singular values."""
    seq_len = embeds.shape[0]
    if k >= seq_len:
        return embeds
    U, S, Vh = torch.linalg.svd(embeds.cpu(), full_matrices=False)
    # Top-k: U[:, :k] @ diag(S[:k]) — but we want k "tokens", so use Vh[:k] as basis
    # Actually: reconstruct as k synthetic tokens = S[:k] * Vh[:k, :] — these are (k, d) vectors
    compressed = (S[:k].unsqueeze(1) * Vh[:k, :]).to(device)  # (k, d)
    # Normalize to match original embedding scale
    orig_scale = embeds.norm(dim=-1).mean()
    comp_scale = compressed.norm(dim=-1).mean()
    if comp_scale > 0:
        compressed = compressed * (orig_scale / comp_scale)
    return compressed.float()


def compress_last_k(embeds, k):
    """Take the last k token embeddings."""
    return embeds[-k:]


def compress_stride(embeds, k):
    """Take every N-th token to get k tokens."""
    seq_len = embeds.shape[0]
    if k >= seq_len:
        return embeds
    indices = torch.linspace(0, seq_len - 1, k).long()
    return embeds[indices]


def compress_pca_project(embeds, k):
    """
    PCA the (seq, d) matrix along seq dimension.
    Top-k principal components become k "tokens".
    """
    seq_len = embeds.shape[0]
    if k >= seq_len:
        return embeds
    # Center
    mean = embeds.mean(dim=0, keepdim=True)
    centered = embeds - mean
    # SVD of centered matrix
    U, S, Vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
    # Project: scores = U[:, :k] @ diag(S[:k]) — these are the coordinates
    # But we want k "tokens" in d-space: Vh[:k, :] scaled by S[:k]
    compressed = (S[:k].unsqueeze(1) * Vh[:k, :]).to(device) + mean  # add mean back
    orig_scale = embeds.norm(dim=-1).mean()
    comp_scale = compressed.norm(dim=-1).mean()
    if comp_scale > 0:
        compressed = compressed * (orig_scale / comp_scale)
    return compressed.float()


def compress_oracle(embeds, k, model, prompt):
    """
    Run full forward pass, extract hidden state at last token from layer 18 (mid-network).
    Use this single vector as a "token". If k>1, extract from multiple layers.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    captured = {}
    hooks = []

    # Pick k layers evenly spaced
    n_layers = model.config.num_hidden_layers
    layer_indices = torch.linspace(0, n_layers - 1, k).long().tolist()

    for l in layer_indices:
        def make_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h[0, -1, :].detach().float()
            return hook
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Stack the captured states → (k, d)
    compressed = torch.stack([captured[l] for l in layer_indices])

    # Project back to embedding space scale
    orig_scale = embeds.norm(dim=-1).mean()
    comp_scale = compressed.norm(dim=-1).mean()
    if comp_scale > 0:
        compressed = compressed * (orig_scale / comp_scale)

    return compressed


METHODS = {
    "mean_pool": compress_mean_pool,
    "svd": compress_svd,
    "last_k": compress_last_k,
    "stride": compress_stride,
    "pca_project": compress_pca_project,
    # oracle handled separately (needs model + prompt)
}


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP AN: HIERARCHICAL TOKEN COMPRESSION")
print("=" * 70)

all_results = {}

for pi, prob in enumerate(PROBLEMS):
    prompt = prob["prompt"]
    answer = prob["answer"]
    lang = prob["lang"]

    print(f"\n{'─' * 60}")
    print(f"P{pi}: {prompt} (answer={answer})")
    print(f"{'─' * 60}")

    # Get full embeddings
    full_embeds = get_embeddings(prompt)
    n_tokens = full_embeds.shape[0]
    print(f"  Full: {n_tokens} tokens")

    # Baseline
    try:
        baseline_gen = generate_from_embeds(full_embeds)
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

    for method_name, compress_fn in METHODS.items():
        for k in K_VALUES:
            if k > n_tokens:
                continue

            cond_name = f"{method_name}_k{k}"

            try:
                compressed = compress_fn(full_embeds, k)
                comp_norm = compressed.norm(dim=-1).mean().item()
                orig_norm = full_embeds.norm(dim=-1).mean().item()

                # Forward pass for logits
                logit_info = forward_logits(compressed)

                # Generate
                gen = generate_from_embeds(compressed)
                correct = check_answer(gen, answer)

                result = {
                    "correct": correct,
                    "gen": gen[:150],
                    "k": k,
                    "method": method_name,
                    "comp_ratio": round(n_tokens / k, 1),
                    "norm_ratio": round(comp_norm / orig_norm, 3),
                    "logits": logit_info,
                }
                prob_results["conditions"][cond_name] = result

                marker = "Y" if correct else "N"
                print(f"  {cond_name:25s}: {marker} top1='{logit_info['top1']}' ent={logit_info['entropy']}")

            except Exception as e:
                prob_results["conditions"][cond_name] = {"error": str(e)}
                print(f"  {cond_name:25s}: ERROR {e}")

    # Oracle method (separate because it needs model + prompt)
    for k in K_VALUES:
        if k > n_tokens:
            continue
        cond_name = f"oracle_k{k}"
        try:
            compressed = compress_oracle(full_embeds, k, model, prompt)
            logit_info = forward_logits(compressed)
            gen = generate_from_embeds(compressed)
            correct = check_answer(gen, answer)

            prob_results["conditions"][cond_name] = {
                "correct": correct,
                "gen": gen[:150],
                "k": k,
                "method": "oracle",
                "comp_ratio": round(n_tokens / k, 1),
                "logits": logit_info,
            }
            marker = "Y" if correct else "N"
            print(f"  {cond_name:25s}: {marker} top1='{logit_info['top1']}' ent={logit_info['entropy']}")
        except Exception as e:
            prob_results["conditions"][cond_name] = {"error": str(e)}
            print(f"  {cond_name:25s}: ERROR {e}")

    all_results[f"problem_{pi}"] = prob_results


# ── Save ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

output = {
    "experiment": "AN: Hierarchical Token Compression",
    "model": MODEL_NAME,
    "k_values": K_VALUES,
    "methods": list(METHODS.keys()) + ["oracle"],
    "n_problems": len(PROBLEMS),
    "results": all_results,
}

with open(OUTPUT_DIR / "expAN_token_compression.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("Saved to output/expAN_token_compression.json")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== GRAND SUMMARY ===")
print(f"{'Method':25s} " + " ".join(f"k={k}" for k in K_VALUES))
print("-" * 70)

method_scores = {}
for method in list(METHODS.keys()) + ["oracle"]:
    scores = []
    for k in K_VALUES:
        cond_name = f"{method}_k{k}"
        n_correct = sum(
            1 for pv in all_results.values()
            if pv["conditions"].get(cond_name, {}).get("correct", False)
        )
        n_total = sum(
            1 for pv in all_results.values()
            if cond_name in pv["conditions"] and "error" not in pv["conditions"][cond_name]
        )
        if n_total > 0:
            scores.append(f"{n_correct}/{n_total}")
        else:
            scores.append("  - ")
    method_scores[method] = scores
    print(f"{method:25s} " + " ".join(f"{s:>4}" for s in scores))

n_baseline = sum(1 for pv in all_results.values() if pv["baseline"]["correct"])
print(f"\nBaseline: {n_baseline}/{len(PROBLEMS)}")

print("\nDone.")
