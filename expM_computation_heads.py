"""Experiment M: Computation Heads at L30.

L2 showed L30 delta is 1.9x larger on decisive tokens. Which attention heads
at L30 activate during computation? Do they attend differently on math vs template tokens?

Also check: is the L30 signal from attention or MLP? If MLP, which neurons fire?
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
d = model.config.hidden_size
n_heads = model.config.num_attention_heads
n_kv_heads = model.config.num_key_value_heads
head_dim = d // n_heads

print(f"Model: d={d}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")

test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133"},
    {"prompt": "Calculate 123 × 45.", "answer": "5535"},
    {"prompt": "What is the remainder when 7654 is divided by 37?", "answer": "34"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120"},
    {"prompt": "An arithmetic sequence has first term 3 and common difference 7. Find the sum of the first 20 terms.",
     "answer": "1390"},
]

results = {"experiment": "M: Computation Heads at L30", "problems": []}

for prob_idx, prob in enumerate(test_problems):
    print(f"\n  Problem {prob_idx}: {prob['prompt'][:60]}")

    input_ids = tokenizer.encode(prob["prompt"])
    prompt_len = len(input_ids)

    # Prefill
    with torch.no_grad():
        outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
    past_kv = outputs.past_key_values
    first_token_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[first_token_id]], device=device)
    generated_ids = [first_token_id]
    token_data = []

    for step in range(MAX_NEW_TOKENS - 1):
        # Capture attention output (per-head) and MLP output at L30
        captures = {}

        def make_attn_hook():
            def hook_fn(module, input, output):
                # Self-attention output before projection: (batch, seq, d)
                # We want per-head norms of the attention output
                attn_out = output[0] if isinstance(output, tuple) else output
                captures['attn_out'] = attn_out[0, -1, :].detach().cpu().float().numpy()
            return hook_fn

        def make_mlp_hook():
            def hook_fn(module, input, output):
                mlp_out = output[0] if isinstance(output, tuple) else output
                captures['mlp_out'] = mlp_out[0, -1, :].detach().cpu().float().numpy()
            return hook_fn

        h_attn = model.model.layers[30].self_attn.register_forward_hook(make_attn_hook())
        h_mlp = model.model.layers[30].mlp.register_forward_hook(make_mlp_hook())

        with torch.no_grad():
            out = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )

        h_attn.remove()
        h_mlp.remove()

        past_kv = out.past_key_values
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        logit_entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        top1_prob = float(probs.max().item())
        next_id = int(logits.argmax())

        # Per-head attention pattern at L30
        attn_L30 = out.attentions[30][0].cpu().float().numpy()  # (n_heads, 1, total_seq)
        total_seq = attn_L30.shape[2]

        # Per-head analysis
        head_stats = []
        for head_idx in range(n_heads):
            head_attn = attn_L30[head_idx, 0, :]  # (total_seq,)
            attn_prompt = float(head_attn[:prompt_len].sum())
            attn_gen = float(head_attn[prompt_len:].sum())
            # Entropy per head
            hc = np.clip(head_attn, 1e-10, 1.0)
            entropy = float(-np.sum(hc * np.log(hc)))
            # Peak position
            peak_pos = int(np.argmax(head_attn))
            peak_val = float(head_attn[peak_pos])
            head_stats.append({
                "attn_prompt": attn_prompt,
                "attn_gen": attn_gen,
                "entropy": entropy,
                "peak_pos": peak_pos,
                "peak_val": peak_val,
            })

        # Attention vs MLP contribution norms
        attn_norm = float(np.linalg.norm(captures.get('attn_out', np.zeros(d))))
        mlp_norm = float(np.linalg.norm(captures.get('mlp_out', np.zeros(d))))

        decoded_token = tokenizer.decode([next_id])
        is_math = any(c.isdigit() for c in decoded_token) or any(c in decoded_token for c in '=+-×÷*/')

        entry = {
            "step": step,
            "token_text": decoded_token,
            "is_math": is_math,
            "logit_entropy": logit_entropy,
            "top1_prob": top1_prob,
            "attn_out_norm": attn_norm,
            "mlp_out_norm": mlp_norm,
            "attn_mlp_ratio": attn_norm / max(mlp_norm, 1e-8),
            "head_stats": head_stats,
        }
        token_data.append(entry)
        generated_ids.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)
        if next_id == tokenizer.eos_token_id:
            break

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"    Generated ({len(token_data)} tokens): {full_text[:80]}...")

    # Classify tokens by logit entropy
    entropies = [t["logit_entropy"] for t in token_data]
    median_entropy = np.median(entropies)
    for t in token_data:
        t["is_decisive"] = bool(t["logit_entropy"] < median_entropy)

    decisive = [t for t in token_data if t["is_decisive"]]
    template = [t for t in token_data if not t["is_decisive"]]

    # Is the L30 spike from attention or MLP?
    d_attn = np.mean([t["attn_out_norm"] for t in decisive])
    t_attn = np.mean([t["attn_out_norm"] for t in template])
    d_mlp = np.mean([t["mlp_out_norm"] for t in decisive])
    t_mlp = np.mean([t["mlp_out_norm"] for t in template])

    print(f"    Attention norm: decisive={d_attn:.1f} vs template={t_attn:.1f} "
          f"(ratio={d_attn/max(t_attn,1):.2f}x)")
    print(f"    MLP norm:       decisive={d_mlp:.1f} vs template={t_mlp:.1f} "
          f"(ratio={d_mlp/max(t_mlp,1):.2f}x)")

    # Per-head analysis: which heads differ most between decisive and template?
    head_decisive_entropy = np.zeros(n_heads)
    head_template_entropy = np.zeros(n_heads)
    head_decisive_prompt = np.zeros(n_heads)
    head_template_prompt = np.zeros(n_heads)

    for h in range(n_heads):
        d_ent = [t["head_stats"][h]["entropy"] for t in decisive]
        t_ent = [t["head_stats"][h]["entropy"] for t in template]
        d_prompt = [t["head_stats"][h]["attn_prompt"] for t in decisive]
        t_prompt = [t["head_stats"][h]["attn_prompt"] for t in template]
        head_decisive_entropy[h] = np.mean(d_ent)
        head_template_entropy[h] = np.mean(t_ent)
        head_decisive_prompt[h] = np.mean(d_prompt)
        head_template_prompt[h] = np.mean(t_prompt)

    # Find heads with biggest entropy difference (focused during computation)
    entropy_diff = head_template_entropy - head_decisive_entropy  # positive = more focused on decisive
    prompt_diff = head_decisive_prompt - head_template_prompt  # positive = attends more to prompt on decisive

    top_focus_heads = np.argsort(entropy_diff)[-3:][::-1]  # most focused during computation
    top_prompt_heads = np.argsort(prompt_diff)[-3:][::-1]  # most prompt-attending during computation

    print(f"    Top computation-focused heads (lower entropy on decisive):")
    for h in top_focus_heads:
        print(f"      Head {h}: entropy diff={entropy_diff[h]:.3f} "
              f"(decisive={head_decisive_entropy[h]:.2f}, template={head_template_entropy[h]:.2f})")

    print(f"    Top prompt-attending heads during computation:")
    for h in top_prompt_heads:
        print(f"      Head {h}: prompt diff={prompt_diff[h]:+.3f} "
              f"(decisive={head_decisive_prompt[h]:.0%}, template={head_template_prompt[h]:.0%})")

    prob_result = {
        "prompt": prob["prompt"],
        "answer": prob["answer"],
        "total_tokens": len(token_data),
        "decisive_attn_norm": float(d_attn),
        "template_attn_norm": float(t_attn),
        "decisive_mlp_norm": float(d_mlp),
        "template_mlp_norm": float(t_mlp),
        "attn_ratio": float(d_attn / max(t_attn, 1)),
        "mlp_ratio": float(d_mlp / max(t_mlp, 1)),
        "top_focus_heads": [int(h) for h in top_focus_heads],
        "top_prompt_heads": [int(h) for h in top_prompt_heads],
        "entropy_diff_per_head": entropy_diff.tolist(),
        "prompt_diff_per_head": prompt_diff.tolist(),
    }
    results["problems"].append(prob_result)

# =============================================================================
# Summary: which heads are consistently computation-focused?
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT M — SUMMARY")
print("=" * 70)

# Aggregate entropy_diff across problems
all_entropy_diffs = np.array([p["entropy_diff_per_head"] for p in results["problems"]])
mean_entropy_diff = all_entropy_diffs.mean(axis=0)

# Is the L30 spike from attention or MLP?
all_attn_ratios = [p["attn_ratio"] for p in results["problems"]]
all_mlp_ratios = [p["mlp_ratio"] for p in results["problems"]]
print(f"  Attention ratio (decisive/template): {np.mean(all_attn_ratios):.2f}x")
print(f"  MLP ratio (decisive/template):       {np.mean(all_mlp_ratios):.2f}x")

if np.mean(all_mlp_ratios) > np.mean(all_attn_ratios):
    print(f"  → L30 computation spike is driven by MLP")
else:
    print(f"  → L30 computation spike is driven by ATTENTION")

print(f"\n  Per-head entropy difference (template - decisive, positive = more focused on computation):")
ranked = np.argsort(mean_entropy_diff)[::-1]
for i, h in enumerate(ranked[:5]):
    print(f"    Head {h}: mean entropy diff = {mean_entropy_diff[h]:+.3f}")
print(f"  ...")
for i, h in enumerate(ranked[-3:]):
    print(f"    Head {h}: mean entropy diff = {mean_entropy_diff[h]:+.3f}")

# Consistency: do the same heads appear across problems?
top_heads_per_problem = [set(p["top_focus_heads"]) for p in results["problems"]]
consistent_heads = set.intersection(*top_heads_per_problem) if top_heads_per_problem else set()
print(f"\n  Heads consistently in top-3 across ALL problems: {sorted(consistent_heads) if consistent_heads else 'NONE'}")

# Union with count
from collections import Counter
head_counts = Counter()
for hs in top_heads_per_problem:
    head_counts.update(hs)
print(f"  Head frequency in top-3 (out of {len(results['problems'])} problems):")
for h, c in head_counts.most_common(5):
    print(f"    Head {h}: appears {c}/{len(results['problems'])} times")

results["summary"] = {
    "mean_attn_ratio": float(np.mean(all_attn_ratios)),
    "mean_mlp_ratio": float(np.mean(all_mlp_ratios)),
    "spike_source": "MLP" if np.mean(all_mlp_ratios) > np.mean(all_attn_ratios) else "ATTENTION",
    "mean_entropy_diff_per_head": mean_entropy_diff.tolist(),
    "consistent_heads": sorted(consistent_heads) if consistent_heads else [],
    "head_frequency": dict(head_counts.most_common()),
}

with open("output/expM_computation_heads.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to output/expM_computation_heads.json")
