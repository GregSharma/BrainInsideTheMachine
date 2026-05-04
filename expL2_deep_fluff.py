"""Experiment L2: Deep Fluff Analysis — Multi-Layer MLP Delta Classification.

L showed 76% fluff by token count but MLP delta at L8 couldn't distinguish
computation from narration. The real action is in deeper layers (L27+)
where the tug-of-war happens.

For each generated token, capture:
1. Full hidden state delta (layer output - layer input) at L8, L20, L27, L30, L34
2. Attention entropy at L27 (where the attractor is)
3. Logit entropy (how "decided" the model is about the next token)
4. Token-level classification: is this a "decisive" step (low logit entropy,
   big hidden state change) or a "template" step (high logit entropy, small change)?
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
CAPTURE_LAYERS = [8, 20, 27, 30, 34]

test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133", "category": "easy_add"},
    {"prompt": "Calculate 123 × 45.", "answer": "5535", "category": "medium_mult"},
    {"prompt": "What is the remainder when 7654 is divided by 37?", "answer": "34", "category": "division"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120", "category": "combo"},
    {"prompt": "An arithmetic sequence has first term 3 and common difference 7. Find the sum of the first 20 terms.",
     "answer": "1390", "category": "sequence"},
]


print("=" * 70)
print("EXPERIMENT L2: DEEP FLUFF ANALYSIS")
print("=" * 70)

d = model.config.hidden_size
results = {"experiment": "L2: Deep Fluff Analysis", "capture_layers": CAPTURE_LAYERS, "problems": []}

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
        # Set up hooks to capture layer-level deltas
        layer_captures = {}

        handles = []
        for layer_idx in CAPTURE_LAYERS:
            def make_hook(lidx):
                def hook_fn(module, input, output):
                    h_in = input[0] if isinstance(input, tuple) else input
                    h_out = output[0] if isinstance(output, tuple) else output
                    delta = (h_out[0, -1, :] - h_in[0, -1, :]).detach().cpu().float().numpy()
                    layer_captures[lidx] = {
                        "delta_norm": float(np.linalg.norm(delta)),
                        "h_out_norm": float(torch.norm(h_out[0, -1, :].float()).item()),
                    }
                return hook_fn
            handles.append(
                model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            )

        with torch.no_grad():
            out = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )

        for h in handles:
            h.remove()

        past_kv = out.past_key_values
        logits = out.logits[0, -1, :]

        # Logit entropy — how decided is the model?
        probs = torch.softmax(logits.float(), dim=-1)
        logit_entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())

        # Top-1 probability
        top1_prob = float(probs.max().item())

        next_id = int(logits.argmax())

        # Attention entropy at L27
        attn_L27 = out.attentions[27][0].cpu().float().numpy()  # (n_heads, 1, total_seq)
        mean_attn_27 = attn_L27.mean(axis=0)[0]  # (total_seq,)
        attn_clipped = np.clip(mean_attn_27, 1e-10, 1.0)
        attn_entropy_27 = float(-np.sum(attn_clipped * np.log(attn_clipped)))

        # Attention on prompt vs generated at L27
        attn_on_prompt_27 = float(mean_attn_27[:prompt_len].sum())

        decoded_token = tokenizer.decode([next_id])
        is_math = any(c.isdigit() for c in decoded_token) or any(c in decoded_token for c in '=+-×÷*/')
        is_newline = decoded_token.strip() == '' or decoded_token == '\n'

        entry = {
            "step": step,
            "token_id": next_id,
            "token_text": decoded_token,
            "is_math": is_math,
            "is_newline": is_newline,
            "logit_entropy": logit_entropy,
            "top1_prob": top1_prob,
            "attn_entropy_L27": attn_entropy_27,
            "attn_on_prompt_L27": attn_on_prompt_27,
        }

        # Layer deltas
        for lidx in CAPTURE_LAYERS:
            if lidx in layer_captures:
                entry[f"delta_norm_L{lidx}"] = layer_captures[lidx]["delta_norm"]
                entry[f"h_norm_L{lidx}"] = layer_captures[lidx]["h_out_norm"]

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

    # "Decisive" = low entropy (model knows what to output)
    # "Template" = high entropy (model is navigating boilerplate)
    for t in token_data:
        t["is_decisive"] = t["logit_entropy"] < median_entropy

    n_decisive = sum(1 for t in token_data if t["is_decisive"])
    n_template = len(token_data) - n_decisive

    # Compare decisive vs template tokens
    decisive_tokens = [t for t in token_data if t["is_decisive"]]
    template_tokens = [t for t in token_data if not t["is_decisive"]]

    decisive_math = sum(1 for t in decisive_tokens if t["is_math"])
    template_math = sum(1 for t in template_tokens if t["is_math"])

    # Layer delta comparison
    for lidx in CAPTURE_LAYERS:
        key = f"delta_norm_L{lidx}"
        d_vals = [t[key] for t in decisive_tokens if key in t]
        t_vals = [t[key] for t in template_tokens if key in t]
        if d_vals and t_vals:
            print(f"    L{lidx} delta: decisive={np.mean(d_vals):.1f} vs template={np.mean(t_vals):.1f} "
                  f"(ratio={np.mean(d_vals)/max(np.mean(t_vals),1e-8):.2f}x)")

    # Attention comparison
    d_attn = [t["attn_on_prompt_L27"] for t in decisive_tokens]
    t_attn = [t["attn_on_prompt_L27"] for t in template_tokens]
    print(f"    L27 attn on prompt: decisive={np.mean(d_attn):.0%} vs template={np.mean(t_attn):.0%}")

    d_entropy_attn = [t["attn_entropy_L27"] for t in decisive_tokens]
    t_entropy_attn = [t["attn_entropy_L27"] for t in template_tokens]
    print(f"    L27 attn entropy: decisive={np.mean(d_entropy_attn):.2f} vs template={np.mean(t_entropy_attn):.2f}")

    print(f"    Decisive: {n_decisive} tokens ({decisive_math} math) | Template: {n_template} tokens ({template_math} math)")
    print(f"    Median logit entropy: {median_entropy:.2f}")

    prob_result = {
        "prompt": prob["prompt"],
        "answer": prob["answer"],
        "category": prob["category"],
        "full_text": full_text,
        "total_tokens": len(token_data),
        "n_decisive": n_decisive,
        "n_template": n_template,
        "decisive_math_count": decisive_math,
        "template_math_count": template_math,
        "median_logit_entropy": float(median_entropy),
        "per_token": token_data,
    }

    # Aggregate layer deltas
    for lidx in CAPTURE_LAYERS:
        key = f"delta_norm_L{lidx}"
        d_vals = [t[key] for t in decisive_tokens if key in t]
        t_vals = [t[key] for t in template_tokens if key in t]
        prob_result[f"decisive_delta_L{lidx}"] = float(np.mean(d_vals)) if d_vals else None
        prob_result[f"template_delta_L{lidx}"] = float(np.mean(t_vals)) if t_vals else None

    results["problems"].append(prob_result)

# =============================================================================
# Overall Summary
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT L2 — SUMMARY")
print("=" * 70)

# Aggregate across all problems
all_decisive = []
all_template = []
for p in results["problems"]:
    for t in p["per_token"]:
        if t["is_decisive"]:
            all_decisive.append(t)
        else:
            all_template.append(t)

print(f"  Total: {len(all_decisive)} decisive + {len(all_template)} template tokens")

for lidx in CAPTURE_LAYERS:
    key = f"delta_norm_L{lidx}"
    d_vals = [t[key] for t in all_decisive if key in t]
    t_vals = [t[key] for t in all_template if key in t]
    if d_vals and t_vals:
        ratio = np.mean(d_vals) / max(np.mean(t_vals), 1e-8)
        print(f"  L{lidx} delta: decisive={np.mean(d_vals):.1f} vs template={np.mean(t_vals):.1f} ({ratio:.2f}x)")

# Attention
d_attn = [t["attn_on_prompt_L27"] for t in all_decisive]
t_attn = [t["attn_on_prompt_L27"] for t in all_template]
print(f"  L27 attn on prompt: decisive={np.mean(d_attn):.0%} vs template={np.mean(t_attn):.0%}")

# Key question: do decisive tokens cluster around math content?
d_math = sum(1 for t in all_decisive if t["is_math"])
t_math = sum(1 for t in all_template if t["is_math"])
print(f"  Math tokens in decisive: {d_math}/{len(all_decisive)} = {d_math/max(len(all_decisive),1):.0%}")
print(f"  Math tokens in template: {t_math}/{len(all_template)} = {t_math/max(len(all_template),1):.0%}")

# Logit entropy comparison
d_ent = [t["logit_entropy"] for t in all_decisive]
t_ent = [t["logit_entropy"] for t in all_template]
print(f"  Logit entropy: decisive={np.mean(d_ent):.2f} vs template={np.mean(t_ent):.2f}")
print(f"  Top-1 prob: decisive={np.mean([t['top1_prob'] for t in all_decisive]):.0%} "
      f"vs template={np.mean([t['top1_prob'] for t in all_template]):.0%}")

results["summary"] = {
    "total_decisive": len(all_decisive),
    "total_template": len(all_template),
    "fluff_fraction": len(all_template) / (len(all_decisive) + len(all_template)),
    "decisive_math_pct": d_math / max(len(all_decisive), 1),
    "template_math_pct": t_math / max(len(all_template), 1),
    "decisive_mean_entropy": float(np.mean(d_ent)),
    "template_mean_entropy": float(np.mean(t_ent)),
    "decisive_mean_top1": float(np.mean([t["top1_prob"] for t in all_decisive])),
    "template_mean_top1": float(np.mean([t["top1_prob"] for t in all_template])),
}

for lidx in CAPTURE_LAYERS:
    key = f"delta_norm_L{lidx}"
    d_vals = [t[key] for t in all_decisive if key in t]
    t_vals = [t[key] for t in all_template if key in t]
    results["summary"][f"decisive_delta_L{lidx}"] = float(np.mean(d_vals)) if d_vals else None
    results["summary"][f"template_delta_L{lidx}"] = float(np.mean(t_vals)) if t_vals else None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open("output/expL2_deep_fluff.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print(f"\nSaved to output/expL2_deep_fluff.json")
