"""
Exp AP: Mid-Layer Arithmetic — Greg's Idea

Instead of averaging/differencing at the embedding level (L0), do it at
INTERMEDIATE layers where the computation is actually happening.

Both ZH and EN representations at layer ~18 are ON THE MANIFOLD because
they were produced by the same transformer. Averaging here shouldn't
leave the manifold like L0 averaging does (AM: 0/5).

Method:
1. Run ZH and EN prompts separately through L0 → L_cut (capture hidden states)
2. At L_cut: average, difference, multiply (Hadamard), or interpolate
3. Continue L_cut → L35 with the mixed state → generate

Also tests MULTIPLICATIVE mixing (Greg's idea): element-wise multiply
instead of add. SwiGLU is multiplicative — maybe the structure is too.

Also tests GENERATION-TIME SUBSTITUTION (Greg's idea 3): generate normally
but swap hidden states mid-generation from a parallel language run.

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
n_layers = model.config.num_hidden_layers  # 36

# ── Problems (paired ZH/EN) ────────────────────────────────────────────
PROBLEMS = [
    {"zh": "计算 47 + 86 的值。", "en": "Calculate 47 + 86.", "answer": "133"},
    {"zh": "计算 664 + 124 的值。", "en": "Calculate 664 + 124.", "answer": "788"},
    {"zh": "计算 15 × 8 的值。", "en": "Calculate 15 × 8.", "answer": "120"},
    {"zh": "计算 238 + 152 的值。", "en": "Calculate 238 + 152.", "answer": "390"},
    {"zh": "一个长方形的长为 12，宽为 5，求其面积。",
     "en": "A rectangle has length 12 and width 5. Find its area.", "answer": "60"},
]

# Cut points to test: adversarial zone, cooperative zone, late
CUT_LAYERS = [9, 14, 18, 22, 26, 30]


# ── Core: run partial forward and capture hidden state ──────────────────

def get_hidden_at_layer(prompt, cut_layer):
    """
    Run prompt through model up to cut_layer, return hidden state at last token.
    Returns: (h_cut, full_seq_embeds, input_ids, all captured states)
    h_cut: (d,) tensor — last token hidden state after layer `cut_layer`
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = h.detach()  # (1, seq, d) — keep full seq
        return hook

    hooks = []
    hooks.append(model.model.layers[cut_layer].register_forward_hook(make_hook(cut_layer)))

    with torch.no_grad():
        out = model(**inputs)

    for h in hooks:
        h.remove()

    h_full = captured[cut_layer]  # (1, seq, d)
    return {
        "h_last_tok": h_full[0, -1, :].float(),  # (d,) last token
        "h_full_seq": h_full[0].float(),  # (seq, d) all tokens
        "input_ids": inputs["input_ids"][0],
        "logits": out.logits[0, -1, :].float(),
        "n_tokens": inputs["input_ids"].shape[1],
    }


def continue_from_hidden(h_injected, cut_layer, n_new_tokens=MAX_NEW_TOKENS):
    """
    Inject a hidden state at cut_layer and continue forward through remaining layers.
    h_injected: (1, seq, d) tensor to inject.
    Returns generated text.

    Strategy: hook layer `cut_layer` to REPLACE its output, then run model.generate
    with dummy input. The hook fires on the first forward pass (prefill).
    """
    # We need a dummy input of the right length
    seq_len = h_injected.shape[1]
    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)

    injection_done = [False]

    def inject_hook(module, input, output):
        if not injection_done[0]:
            injection_done[0] = True
            h = output[0] if isinstance(output, tuple) else output
            # Replace with our injected hidden state
            injected = h_injected.to(h.dtype).to(h.device)
            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected
        return output  # Don't modify during generation steps

    handle = model.model.layers[cut_layer].register_forward_hook(inject_hook)

    # Also need to zero out all layers BEFORE cut_layer (they'll compute garbage from dummy input)
    # Better approach: hook the FIRST layer to inject, and let subsequent layers process normally
    # Actually: we need to inject AFTER cut_layer's output. So we hook cut_layer.
    # But layers 0..cut_layer will still run on dummy input. Their output gets overwritten by our hook.
    # Layers cut_layer+1..35 then process our injected hidden state normally. This works.

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=dummy_ids,
                max_new_tokens=n_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        gen_text = tokenizer.decode(out[0][seq_len:], skip_special_tokens=True)
    except Exception as e:
        gen_text = f"ERROR: {e}"
    finally:
        handle.remove()

    return gen_text


def interpolate_sequences(h_a, h_b):
    """
    Interpolate two hidden state sequences of different lengths to the shorter one.
    h_a: (seq_a, d), h_b: (seq_b, d)
    Returns: (h_a_interp, h_b_interp) both of shape (min_len, d)
    """
    len_a, len_b = h_a.shape[0], h_b.shape[0]
    target_len = min(len_a, len_b)

    def interp(h, target):
        if h.shape[0] == target:
            return h
        e = h.T.unsqueeze(0).float()  # (1, d, seq)
        e_interp = F.interpolate(e, size=target, mode='linear', align_corners=True)
        return e_interp.squeeze(0).T  # (target, d)

    return interp(h_a, target_len), interp(h_b, target_len)


# ── Mixing functions ────────────────────────────────────────────────────

def mix_average(h_zh, h_en):
    """Simple average: (h_zh + h_en) / 2"""
    return (h_zh + h_en) / 2

def mix_difference_zh_plus_en(h_zh, h_en):
    """ZH + 0.5*(EN - ZH) = 0.5*ZH + 0.5*EN — same as average but framed as displacement"""
    return h_zh + 0.5 * (h_en - h_zh)

def mix_multiply(h_zh, h_en):
    """Element-wise multiply (Hadamard). Normalized to preserve scale."""
    product = h_zh * h_en
    # Normalize to geometric mean of norms
    target_norm = (h_zh.norm(dim=-1, keepdim=True) * h_en.norm(dim=-1, keepdim=True)).sqrt()
    product_norm = product.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return product * (target_norm / product_norm)

def mix_multiply_raw(h_zh, h_en):
    """Element-wise multiply, no normalization."""
    return h_zh * h_en

def mix_en_only(h_zh, h_en):
    """Control: just use EN (should match EN baseline)."""
    return h_en

def mix_zh_only(h_zh, h_en):
    """Control: just use ZH (should match ZH baseline)."""
    return h_zh

def mix_max(h_zh, h_en):
    """Element-wise max — keep the larger activation."""
    return torch.max(h_zh, h_en)

def mix_signed_max(h_zh, h_en):
    """Keep the element with larger absolute value (preserving sign)."""
    mask = h_zh.abs() > h_en.abs()
    return torch.where(mask, h_zh, h_en)


MIXERS = {
    "average": mix_average,
    "multiply_norm": mix_multiply,
    "multiply_raw": mix_multiply_raw,
    "max": mix_max,
    "signed_max": mix_signed_max,
    "zh_control": mix_zh_only,
    "en_control": mix_en_only,
}


# ══════════════════════════════════════════════════════════════════════════
# PART 1: Mid-layer mixing
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP AP: MID-LAYER ARITHMETIC")
print("=" * 70)

all_results = {}

for pi, prob in enumerate(PROBLEMS):
    prompt_zh = prob["zh"]
    prompt_en = prob["en"]
    answer = prob["answer"]

    print(f"\n{'─' * 60}")
    print(f"P{pi}: {prompt_en} (answer={answer})")
    print(f"{'─' * 60}")

    prob_results = {
        "prompt_zh": prompt_zh, "prompt_en": prompt_en, "answer": answer,
        "conditions": {},
    }

    for cut_layer in CUT_LAYERS:
        print(f"\n  --- Cut at layer {cut_layer} ---")

        # Get hidden states at cut_layer for both languages
        zh_data = get_hidden_at_layer(prompt_zh, cut_layer)
        en_data = get_hidden_at_layer(prompt_en, cut_layer)

        print(f"  ZH: {zh_data['n_tokens']} tokens, EN: {en_data['n_tokens']} tokens")

        # Interpolate to common length
        h_zh_interp, h_en_interp = interpolate_sequences(
            zh_data["h_full_seq"], en_data["h_full_seq"]
        )
        common_len = h_zh_interp.shape[0]
        print(f"  Common length: {common_len}")

        for mixer_name, mixer_fn in MIXERS.items():
            cond_name = f"L{cut_layer}_{mixer_name}"

            try:
                # Mix at every position
                h_mixed = mixer_fn(h_zh_interp, h_en_interp)  # (common_len, d)

                # Continue from mixed hidden state
                h_inject = h_mixed.unsqueeze(0)  # (1, common_len, d)
                gen_text = continue_from_hidden(h_inject, cut_layer)

                correct = answer in gen_text
                prob_results["conditions"][cond_name] = {
                    "correct": correct,
                    "gen": gen_text[:200],
                    "cut_layer": cut_layer,
                    "mixer": mixer_name,
                    "common_len": common_len,
                }
                print(f"    {mixer_name:15s}: {'Y' if correct else 'N'} — {gen_text[:50]}...")

            except Exception as e:
                prob_results["conditions"][cond_name] = {"error": str(e)}
                print(f"    {mixer_name:15s}: ERROR {str(e)[:50]}")

    all_results[f"problem_{pi}"] = prob_results


# ══════════════════════════════════════════════════════════════════════════
# PART 2: Generation-time substitution
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: GENERATION-TIME CROSS-LINGUAL SUBSTITUTION")
print("=" * 70)

# For this, we generate from ZH prompt, but at each generation step,
# we also run the EN prompt and mix the hidden states at a specific layer
# before continuing. This requires token-by-token generation with hooks.

SUBST_LAYER = 18  # Cooperative zone

for pi, prob in enumerate(PROBLEMS[:3]):  # Only first 3 (expensive)
    prompt_zh = prob["zh"]
    prompt_en = prob["en"]
    answer = prob["answer"]

    print(f"\n{'─' * 60}")
    print(f"P{pi}: {prompt_en} (answer={answer})")
    print(f"  Generation-time substitution at L{SUBST_LAYER}")
    print(f"{'─' * 60}")

    # Strategy: generate from ZH. At each step, also forward EN with same
    # generated tokens appended. Average the hidden states at SUBST_LAYER.
    # This is expensive (2x forward per generated token) but reveals
    # whether the reasoning geometry is compositional.

    zh_ids = tokenizer(prompt_zh, return_tensors="pt").to(device)["input_ids"]
    en_ids = tokenizer(prompt_en, return_tensors="pt").to(device)["input_ids"]

    generated_ids = []
    gen_text_parts = []

    for step in range(32):  # Generate up to 32 tokens
        # Build current sequences: original prompt + generated tokens so far
        gen_tensor = torch.tensor(generated_ids, device=device).unsqueeze(0) if generated_ids else None

        if gen_tensor is not None:
            zh_input = torch.cat([zh_ids, gen_tensor], dim=1)
            en_input = torch.cat([en_ids, gen_tensor], dim=1)
        else:
            zh_input = zh_ids
            en_input = en_ids

        # Forward both
        captured_zh = {}
        captured_en = {}

        def make_hook_zh(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured_zh[layer_idx] = h[0, -1, :].detach().float()
            return hook

        def make_hook_en(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured_en[layer_idx] = h[0, -1, :].detach().float()
            return hook

        # Run ZH
        hook_zh = model.model.layers[SUBST_LAYER].register_forward_hook(make_hook_zh(SUBST_LAYER))
        with torch.no_grad():
            out_zh = model(input_ids=zh_input)
        hook_zh.remove()

        # Run EN
        hook_en = model.model.layers[SUBST_LAYER].register_forward_hook(make_hook_en(SUBST_LAYER))
        with torch.no_grad():
            out_en = model(input_ids=en_input)
        hook_en.remove()

        # Average the logits (simple approach — average at output level)
        zh_logits = out_zh.logits[0, -1, :].float()
        en_logits = out_en.logits[0, -1, :].float()
        avg_logits = (zh_logits + en_logits) / 2

        # Also try: use ZH logits but with EN hidden state influence
        # For now, just pick from averaged logits
        next_token = avg_logits.argmax().item()

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token)
        gen_text_parts.append(tokenizer.decode([next_token]))

    gen_text = "".join(gen_text_parts)
    correct = answer in gen_text
    print(f"  logit_avg: {'Y' if correct else 'N'} — {gen_text[:60]}...")

    all_results[f"problem_{pi}"]["gentime_logit_avg"] = {
        "correct": correct,
        "gen": gen_text[:200],
        "n_steps": len(generated_ids),
        "method": "logit_average_zh_en",
        "subst_layer": SUBST_LAYER,
    }


# ── Save ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

output = {
    "experiment": "AP: Mid-Layer Arithmetic",
    "model": MODEL_NAME,
    "cut_layers": CUT_LAYERS,
    "mixers": list(MIXERS.keys()),
    "n_problems": len(PROBLEMS),
    "results": all_results,
}

with open(OUTPUT_DIR / "expAP_midlayer_arithmetic.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("Saved to output/expAP_midlayer_arithmetic.json")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== GRAND SUMMARY ===")
print(f"{'Condition':30s} Correct/Total")
for cut_layer in CUT_LAYERS:
    for mixer_name in MIXERS:
        cond = f"L{cut_layer}_{mixer_name}"
        n_correct = sum(1 for pv in all_results.values()
                        if pv.get("conditions", {}).get(cond, {}).get("correct", False))
        n_total = sum(1 for pv in all_results.values()
                      if cond in pv.get("conditions", {}))
        if n_total > 0:
            print(f"  {cond:30s}: {n_correct}/{n_total}")

# Gentime
n_gt = sum(1 for pv in all_results.values()
           if pv.get("gentime_logit_avg", {}).get("correct", False))
print(f"  {'gentime_logit_avg':30s}: {n_gt}/3")

print("\nDone.")
