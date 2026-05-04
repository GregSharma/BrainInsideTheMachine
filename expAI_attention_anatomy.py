"""
Exp AI: Attention Anatomy — What Von Neumann Would Have Done

Decompose attention into:
1. Content-only QK similarity (no RoPE)
2. Full QK similarity (with RoPE)
3. The "complement vector" — softmax-weighted value sum per token
4. How all of this differs ZH vs EN for same math problem

Track layer by layer, head by head. See what the model is actually thinking.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")

# One problem, deeply analyzed
PROBLEM = {
    "en": "Calculate 47 + 86.",
    "zh": "计算 47 + 86 的值。",
    "answer": "133",
}

# A few more for averaging
PROBLEMS = [
    {"en": "Calculate 47 + 86.", "zh": "计算 47 + 86 的值。", "answer": "133"},
    {"en": "Calculate 15 × 8.", "zh": "计算 15 × 8 的值。", "answer": "120"},
    {"en": "Find the value of C(10, 3).", "zh": "求组合数 C(10, 3) 的值。", "answer": "120"},
    {"en": "What is the remainder when 100 is divided by 7?", "zh": "100 除以 7 的余数是多少？", "answer": "2"},
    {"en": "A rectangle has length 12 and width 5. Find its area.", "zh": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60"},
]


def get_rope_params(model, seq_len, device):
    """Extract RoPE cos/sin for given sequence length."""
    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = rotary_emb(
        torch.zeros(1, seq_len, model.config.hidden_size, device=device, dtype=torch.bfloat16),
        position_ids
    )
    return cos, sin, position_ids


def decompose_attention_at_layer(model, tokenizer, prompt, target_layer):
    """
    For a single prompt at a single layer, extract:
    - Q, K, V projections (pre-RoPE and post-RoPE)
    - Content-only attention weights (Q_content @ K_content^T)
    - Full attention weights (Q_rope @ K_rope^T)
    - The complement vector per token
    - Attention entropy per token
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    tokens = [tokenizer.decode([t]) for t in input_ids[0]]

    # We need to hook into the attention layer BEFORE and AFTER RoPE
    captured = {}

    def attn_hook(module, args, kwargs, output):
        # This won't work — we need to go deeper
        pass

    # Instead: hook the self_attn forward to capture Q, K, V pre/post RoPE
    layer = model.model.layers[target_layer]
    attn = layer.self_attn

    # Capture the hidden state input to attention
    hidden_capture = {}
    def capture_hidden(module, inp, out):
        # inp to the full layer
        h = inp[0] if isinstance(inp, tuple) else inp
        hidden_capture["h"] = h.detach()

    h_handle = layer.register_forward_pre_hook(
        lambda module, inp: hidden_capture.update({"h": (inp[0] if isinstance(inp, tuple) else inp).detach()})
    )

    # Run forward to get hidden state
    with torch.no_grad():
        model(**inputs, output_attentions=False)
    h_handle.remove()

    hidden_state = hidden_capture["h"]  # (1, seq_len, hidden_size)

    # Now manually compute Q, K, V
    # Apply input layernorm first (Qwen uses pre-norm)
    h_normed = layer.input_layernorm(hidden_state)

    # Project Q, K, V
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads

    q_proj = attn.q_proj(h_normed)  # (1, seq_len, n_heads * head_dim)
    k_proj = attn.k_proj(h_normed)  # (1, seq_len, n_kv_heads * head_dim)
    v_proj = attn.v_proj(h_normed)  # (1, seq_len, n_kv_heads * head_dim)

    # Reshape to (batch, n_heads, seq_len, head_dim)
    q = q_proj.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k_proj.view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v_proj.view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    # CONTENT-ONLY attention (no RoPE)
    q_content = q.float()
    k_content = k.float()

    # Expand k for GQA if needed
    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k_content_exp = k_content.repeat_interleave(n_rep, dim=1)
        v_exp = v.float().repeat_interleave(n_rep, dim=1)
    else:
        k_content_exp = k_content
        v_exp = v.float()

    content_scores = torch.matmul(q_content, k_content_exp.transpose(-2, -1)) / (head_dim ** 0.5)
    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    content_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    content_attn = F.softmax(content_scores, dim=-1)  # (1, n_heads, seq_len, seq_len)

    # FULL attention (with RoPE)
    cos, sin, position_ids = get_rope_params(model, seq_len, device)
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    q_rope = q_rope.float()
    k_rope = k_rope.float()

    if n_rep > 1:
        k_rope_exp = k_rope.repeat_interleave(n_rep, dim=1)
    else:
        k_rope_exp = k_rope

    full_scores = torch.matmul(q_rope, k_rope_exp.transpose(-2, -1)) / (head_dim ** 0.5)
    full_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    full_attn = F.softmax(full_scores, dim=-1)  # (1, n_heads, seq_len, seq_len)

    # Complement vector: softmax-weighted sum of values for each token
    # complement[i] = sum_j attn[i,j] * v[j]
    complement_content = torch.matmul(content_attn, v_exp)  # (1, n_heads, seq_len, head_dim)
    complement_full = torch.matmul(full_attn, v_exp)

    # Entropy of attention distribution per token per head
    def attn_entropy(attn_weights):
        # attn_weights: (1, n_heads, seq_len, seq_len)
        # Avoid log(0)
        a = attn_weights.clamp(min=1e-10)
        ent = -(a * a.log()).sum(dim=-1)  # (1, n_heads, seq_len)
        return ent.squeeze(0).detach().cpu().numpy()  # (n_heads, seq_len)

    content_entropy = attn_entropy(content_attn)
    full_entropy = attn_entropy(full_attn)

    # RoPE effect: KL divergence between content-only and full attention
    def kl_div_per_token(p, q_dist):
        # p, q: (1, n_heads, seq_len, seq_len)
        p = p.clamp(min=1e-10)
        q_dist = q_dist.clamp(min=1e-10)
        kl = (p * (p.log() - q_dist.log())).sum(dim=-1)  # (1, n_heads, seq_len)
        return kl.squeeze(0).detach().cpu().numpy()

    rope_kl = kl_div_per_token(full_attn, content_attn)  # how much RoPE changes things

    # Complement cosine similarity between content-only and full
    c1 = complement_content.squeeze(0)  # (n_heads, seq_len, head_dim)
    c2 = complement_full.squeeze(0)
    cos_sim = F.cosine_similarity(c1, c2, dim=-1).detach().cpu().numpy()  # (n_heads, seq_len)

    # Last-token attention pattern (most relevant for generation)
    last_tok_content = content_attn[0, :, -1, :].detach().cpu().numpy()  # (n_heads, seq_len)
    last_tok_full = full_attn[0, :, -1, :].detach().cpu().numpy()

    return {
        "tokens": tokens,
        "seq_len": seq_len,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "content_entropy": content_entropy,  # (n_heads, seq_len)
        "full_entropy": full_entropy,
        "rope_kl": rope_kl,  # (n_heads, seq_len)
        "complement_cosine": cos_sim,  # (n_heads, seq_len)
        "last_tok_content_attn": last_tok_content,
        "last_tok_full_attn": last_tok_full,
    }


def compare_languages(model, tokenizer, problem, layers):
    """Compare ZH vs EN attention anatomy at selected layers."""
    results = {}
    for l in layers:
        print(f"  Layer {l}...")
        en_data = decompose_attention_at_layer(model, tokenizer, problem["en"], l)
        zh_data = decompose_attention_at_layer(model, tokenizer, problem["zh"], l)

        # Per-head summary for last token
        head_summaries = []
        for h in range(en_data["n_heads"]):
            # Content entropy at last token
            en_content_ent = float(en_data["content_entropy"][h, -1])
            zh_content_ent = float(zh_data["content_entropy"][h, -1])
            en_full_ent = float(en_data["full_entropy"][h, -1])
            zh_full_ent = float(zh_data["full_entropy"][h, -1])

            # RoPE effect at last token
            en_rope_kl_last = float(en_data["rope_kl"][h, -1])
            zh_rope_kl_last = float(zh_data["rope_kl"][h, -1])

            # Complement stability
            en_comp_cos = float(en_data["complement_cosine"][h, -1])
            zh_comp_cos = float(zh_data["complement_cosine"][h, -1])

            head_summaries.append({
                "head": h,
                "en_content_entropy": round(en_content_ent, 4),
                "zh_content_entropy": round(zh_content_ent, 4),
                "en_full_entropy": round(en_full_ent, 4),
                "zh_full_entropy": round(zh_full_ent, 4),
                "en_rope_kl": round(en_rope_kl_last, 4),
                "zh_rope_kl": round(zh_rope_kl_last, 4),
                "en_complement_cos": round(en_comp_cos, 4),
                "zh_complement_cos": round(zh_comp_cos, 4),
            })

        # Aggregate across heads
        mean_en_rope_kl = float(np.mean([h["en_rope_kl"] for h in head_summaries]))
        mean_zh_rope_kl = float(np.mean([h["zh_rope_kl"] for h in head_summaries]))
        mean_en_content_ent = float(np.mean([h["en_content_entropy"] for h in head_summaries]))
        mean_zh_content_ent = float(np.mean([h["zh_content_entropy"] for h in head_summaries]))
        mean_en_full_ent = float(np.mean([h["en_full_entropy"] for h in head_summaries]))
        mean_zh_full_ent = float(np.mean([h["zh_full_entropy"] for h in head_summaries]))
        mean_comp_cos_en = float(np.mean([h["en_complement_cos"] for h in head_summaries]))
        mean_comp_cos_zh = float(np.mean([h["zh_complement_cos"] for h in head_summaries]))

        results[str(l)] = {
            "en_tokens": en_data["tokens"],
            "zh_tokens": zh_data["tokens"],
            "en_seq_len": en_data["seq_len"],
            "zh_seq_len": zh_data["seq_len"],
            "mean_en_rope_kl": round(mean_en_rope_kl, 4),
            "mean_zh_rope_kl": round(mean_zh_rope_kl, 4),
            "mean_en_content_entropy": round(mean_en_content_ent, 4),
            "mean_zh_content_entropy": round(mean_zh_content_ent, 4),
            "mean_en_full_entropy": round(mean_en_full_ent, 4),
            "mean_zh_full_entropy": round(mean_zh_full_ent, 4),
            "mean_complement_cos_en": round(mean_comp_cos_en, 4),
            "mean_complement_cos_zh": round(mean_comp_cos_zh, 4),
            "rope_kl_ratio_zh_en": round(mean_zh_rope_kl / (mean_en_rope_kl + 1e-8), 4),
            "per_head": head_summaries,
        }
    return results


def multi_problem_sweep(model, tokenizer, problems, layers):
    """Run across multiple problems, aggregate per-layer stats."""
    all_results = []
    for i, prob in enumerate(problems):
        print(f"Problem {i}: {prob['en'][:40]}...")
        r = compare_languages(model, tokenizer, prob, layers)
        all_results.append(r)

    # Aggregate
    agg = {}
    for l in [str(x) for x in layers]:
        vals = [r[l] for r in all_results if l in r]
        agg[l] = {
            "mean_en_rope_kl": round(float(np.mean([v["mean_en_rope_kl"] for v in vals])), 4),
            "mean_zh_rope_kl": round(float(np.mean([v["mean_zh_rope_kl"] for v in vals])), 4),
            "mean_en_content_entropy": round(float(np.mean([v["mean_en_content_entropy"] for v in vals])), 4),
            "mean_zh_content_entropy": round(float(np.mean([v["mean_zh_content_entropy"] for v in vals])), 4),
            "mean_en_full_entropy": round(float(np.mean([v["mean_en_full_entropy"] for v in vals])), 4),
            "mean_zh_full_entropy": round(float(np.mean([v["mean_zh_full_entropy"] for v in vals])), 4),
            "mean_complement_cos_en": round(float(np.mean([v["mean_complement_cos_en"] for v in vals])), 4),
            "mean_complement_cos_zh": round(float(np.mean([v["mean_complement_cos_zh"] for v in vals])), 4),
            "rope_kl_ratio_zh_en": round(float(np.mean([v["rope_kl_ratio_zh_en"] for v in vals])), 4),
        }
    return agg, all_results


def main():
    print("=== Exp AI: Attention Anatomy ===")
    print("What Von Neumann would have done: look at every computation.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",  # need this for manual attention access
    )
    model.eval()

    # All 36 layers
    layers = list(range(36))

    print("=== Multi-problem sweep (5 problems × 36 layers) ===")
    agg, all_results = multi_problem_sweep(model, tokenizer, PROBLEMS, layers)

    # Print summary
    print("\n=== ATTENTION ANATOMY SUMMARY ===")
    print(f"{'Layer':>5} {'EN_RoPE_KL':>10} {'ZH_RoPE_KL':>10} {'ZH/EN':>8} "
          f"{'EN_ent_c':>8} {'EN_ent_f':>8} {'ZH_ent_c':>8} {'ZH_ent_f':>8} "
          f"{'comp_EN':>8} {'comp_ZH':>8}")
    print("-" * 100)
    for l in range(36):
        sl = str(l)
        a = agg[sl]
        phase = ""
        if 9 <= l <= 17: phase = " [ADV]"
        elif 18 <= l <= 21: phase = " [COOP]"
        elif 22 <= l <= 26: phase = " [RAMP]"
        print(f"{l:>5} {a['mean_en_rope_kl']:>10.4f} {a['mean_zh_rope_kl']:>10.4f} "
              f"{a['rope_kl_ratio_zh_en']:>8.2f} "
              f"{a['mean_en_content_entropy']:>8.4f} {a['mean_en_full_entropy']:>8.4f} "
              f"{a['mean_zh_content_entropy']:>8.4f} {a['mean_zh_full_entropy']:>8.4f} "
              f"{a['mean_complement_cos_en']:>8.4f} {a['mean_complement_cos_zh']:>8.4f}{phase}")

    # Key questions answered:
    print("\n=== KEY FINDINGS ===")

    # 1. Does RoPE affect ZH more than EN?
    adv_en = np.mean([agg[str(l)]["mean_en_rope_kl"] for l in range(9, 18)])
    adv_zh = np.mean([agg[str(l)]["mean_zh_rope_kl"] for l in range(9, 18)])
    coop_en = np.mean([agg[str(l)]["mean_en_rope_kl"] for l in range(18, 22)])
    coop_zh = np.mean([agg[str(l)]["mean_zh_rope_kl"] for l in range(18, 22)])
    print(f"RoPE KL (adversarial): EN={adv_en:.4f}, ZH={adv_zh:.4f}, ratio={adv_zh/adv_en:.2f}")
    print(f"RoPE KL (cooperative): EN={coop_en:.4f}, ZH={coop_zh:.4f}, ratio={coop_zh/coop_en:.2f}")

    # 2. Complement stability (content vs full)
    mean_comp_en = np.mean([agg[str(l)]["mean_complement_cos_en"] for l in range(36)])
    mean_comp_zh = np.mean([agg[str(l)]["mean_complement_cos_zh"] for l in range(36)])
    print(f"Complement cos(content,full): EN={mean_comp_en:.4f}, ZH={mean_comp_zh:.4f}")

    # 3. Entropy: does attention get sharper (more Dirac-like) in later layers?
    early_ent_en = np.mean([agg[str(l)]["mean_en_full_entropy"] for l in range(0, 9)])
    late_ent_en = np.mean([agg[str(l)]["mean_en_full_entropy"] for l in range(27, 36)])
    early_ent_zh = np.mean([agg[str(l)]["mean_zh_full_entropy"] for l in range(0, 9)])
    late_ent_zh = np.mean([agg[str(l)]["mean_zh_full_entropy"] for l in range(27, 36)])
    print(f"Entropy early (L0-8): EN={early_ent_en:.4f}, ZH={early_ent_zh:.4f}")
    print(f"Entropy late (L27-35): EN={late_ent_en:.4f}, ZH={late_ent_zh:.4f}")

    output = {
        "experiment": "AI_attention_anatomy",
        "method": "Decompose attention into content-only (no RoPE) vs full (with RoPE). "
                  "Track entropy, KL divergence, complement vector stability across layers.",
        "n_problems": len(PROBLEMS),
        "aggregate": agg,
        "per_problem": [{
            "en": PROBLEMS[i]["en"],
            "zh": PROBLEMS[i]["zh"],
            "layers": {l: {
                "en_tokens": all_results[i][l]["en_tokens"],
                "zh_tokens": all_results[i][l]["zh_tokens"],
                "mean_en_rope_kl": all_results[i][l]["mean_en_rope_kl"],
                "mean_zh_rope_kl": all_results[i][l]["mean_zh_rope_kl"],
                "mean_complement_cos_en": all_results[i][l]["mean_complement_cos_en"],
                "mean_complement_cos_zh": all_results[i][l]["mean_complement_cos_zh"],
            } for l in [str(x) for x in layers]}
        } for i in range(len(PROBLEMS))],
    }

    out_path = OUTPUT_DIR / "expAI_attention_anatomy.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
