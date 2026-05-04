"""
Exp AM: Meta-Lingual Embedding — The Token Sliding Experiment

Take the same math problem in N languages. Embed each (different token counts).
Interpolate all to a common sequence length. Average position-wise.
Feed this "universal prompt" to the model.

Questions:
1. Does the averaged embedding produce coherent output? Or gibberish?
2. Does it produce the CORRECT ANSWER?
3. Which language does the output come in?
4. What about DIFFERENCES between language embeddings? (ZH - EN added to JA)
5. What about taking the SVD null space of the multi-lingual embeddings?
   (Remove the top-k language directions → pure math embedding)

Connection to toy SVD theorem: the null space of cross-language variance
IS the math kernel. The average should approach this null space.

On Qwen2.5-3B locally.
"""

import json, sys
import numpy as np
import torch
import torch.nn.functional as F
import random as pyrandom
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
d = model.config.hidden_size

# ── Problems in 7 languages ────────────────────────────────────────────
# Same math problems, hand-translated
PROBLEMS = [
    {
        "answer": "133",
        "prompts": {
            "zh": "计算 47 + 86 的值。",
            "en": "Calculate 47 + 86.",
            "es": "Calcula 47 + 86.",
            "fr": "Calculez 47 + 86.",
            "de": "Berechne 47 + 86.",
            "ja": "47 + 86 を計算してください。",
            "ko": "47 + 86을 계산하세요.",
        }
    },
    {
        "answer": "788",
        "prompts": {
            "zh": "计算 664 + 124 的值。",
            "en": "Calculate 664 + 124.",
            "es": "Calcula 664 + 124.",
            "fr": "Calculez 664 + 124.",
            "de": "Berechne 664 + 124.",
            "ja": "664 + 124 を計算してください。",
            "ko": "664 + 124를 계산하세요.",
        }
    },
    {
        "answer": "120",
        "prompts": {
            "zh": "计算 15 × 8 的值。",
            "en": "Calculate 15 × 8.",
            "es": "Calcula 15 × 8.",
            "fr": "Calculez 15 × 8.",
            "de": "Berechne 15 × 8.",
            "ja": "15 × 8 を計算してください。",
            "ko": "15 × 8을 계산하세요.",
        }
    },
    {
        "answer": "60",
        "prompts": {
            "zh": "一个长方形的长为 12，宽为 5，求其面积。",
            "en": "A rectangle has length 12 and width 5. Find its area.",
            "es": "Un rectángulo tiene largo 12 y ancho 5. Encuentra su área.",
            "fr": "Un rectangle a une longueur de 12 et une largeur de 5. Trouvez son aire.",
            "de": "Ein Rechteck hat die Länge 12 und die Breite 5. Berechne seine Fläche.",
            "ja": "長さ12、幅5の長方形の面積を求めてください。",
            "ko": "길이 12, 너비 5인 직사각형의 넓이를 구하세요.",
        }
    },
    {
        "answer": "390",
        "prompts": {
            "zh": "计算 238 + 152 的值。",
            "en": "Calculate 238 + 152.",
            "es": "Calcula 238 + 152.",
            "fr": "Calculez 238 + 152.",
            "de": "Berechne 238 + 152.",
            "ja": "238 + 152 を計算してください。",
            "ko": "238 + 152를 계산하세요.",
        }
    },
]

LANGS = ["zh", "en", "es", "fr", "de", "ja", "ko"]
REF_LANG = "zh"  # Shortest token count typically


# ── Embed and interpolate ──────────────────────────────────────────────

def get_embeddings(prompt):
    """Get raw token embeddings for a prompt. Returns (seq_len, d) tensor."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model.model.embed_tokens(inputs["input_ids"])  # (1, seq, d)
    return embeds.squeeze(0).float()  # (seq, d)


def interpolate_to_length(embeds, target_len):
    """
    Interpolate embedding sequence to target length using linear interpolation.
    embeds: (seq, d) tensor
    Returns: (target_len, d) tensor
    """
    seq_len = embeds.shape[0]
    if seq_len == target_len:
        return embeds

    # Transpose to (1, d, seq) for F.interpolate, then back
    e = embeds.T.unsqueeze(0)  # (1, d, seq)
    e_interp = F.interpolate(e, size=target_len, mode='linear', align_corners=True)
    return e_interp.squeeze(0).T  # (target_len, d)


def generate_from_embeds(model, embeds_2d):
    """
    Generate text from raw embeddings (seq, d) → model output.
    Returns generated text and logits info.
    """
    embeds = embeds_2d.unsqueeze(0).to(model.model.embed_tokens.weight.dtype).to(device)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
        )

    # out includes generated tokens only (no input since we used inputs_embeds)
    gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return gen_text


def forward_embeds_logits(model, embeds_2d):
    """
    Single forward pass from embeddings. Return top-5 predictions and answer rank.
    """
    embeds = embeds_2d.unsqueeze(0).to(model.model.embed_tokens.weight.dtype).to(device)

    with torch.no_grad():
        out = model(inputs_embeds=embeds)

    logits = out.logits[0, -1, :].float()  # (V,) last position
    top5 = logits.topk(5)
    top5_text = [tokenizer.decode([t]) for t in top5.indices.tolist()]
    top5_probs = F.softmax(logits, dim=-1)[top5.indices].tolist()

    entropy = float(-(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum())

    return {
        "top5": list(zip(top5_text, [round(p, 4) for p in top5_probs])),
        "top1": top5_text[0],
        "entropy": round(entropy, 2),
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXP AM: META-LINGUAL EMBEDDING")
print("="*70)

all_results = {}

for pi, problem in enumerate(PROBLEMS):
    answer = problem["answer"]
    prompts = problem["prompts"]
    print(f"\n{'─'*60}")
    print(f"Problem {pi}: {prompts['en']} (answer={answer})")
    print(f"{'─'*60}")

    # Step 1: Embed all languages
    lang_embeds = {}
    lang_lengths = {}
    for lang in LANGS:
        e = get_embeddings(prompts[lang])
        lang_embeds[lang] = e
        lang_lengths[lang] = e.shape[0]
        print(f"  {lang}: {e.shape[0]} tokens")

    # Reference length = Chinese (typically shortest)
    ref_len = lang_lengths[REF_LANG]
    min_len = min(lang_lengths.values())
    max_len = max(lang_lengths.values())
    print(f"  Ref length ({REF_LANG}): {ref_len}, min: {min_len}, max: {max_len}")

    # Step 2: Interpolate all to reference length
    interp_embeds = {}
    for lang in LANGS:
        interp_embeds[lang] = interpolate_to_length(lang_embeds[lang], ref_len)

    # ── Condition 1: Single-language baselines ──
    print(f"\n  === BASELINES ===")
    baselines = {}
    for lang in ["zh", "en", "ja"]:
        gen = generate_from_embeds(model, lang_embeds[lang])
        has_answer = answer in gen
        baselines[lang] = {"gen": gen[:80], "correct": has_answer}
        print(f"    {lang}: {'✓' if has_answer else '✗'} {gen[:60]}...")

    # ── Condition 2: Simple average of all 7 languages ──
    print(f"\n  === AVERAGE (all 7 langs, interp to {ref_len} tokens) ===")
    avg_embed = torch.stack([interp_embeds[l] for l in LANGS]).mean(dim=0)  # (ref_len, d)

    # Norm comparison: how does average compare to individual?
    avg_norm = avg_embed.norm(dim=-1).mean().item()
    ref_norm = interp_embeds["zh"].norm(dim=-1).mean().item()
    print(f"    Avg embed norm: {avg_norm:.1f} (zh ref: {ref_norm:.1f}, ratio: {avg_norm/ref_norm:.2f})")

    avg_logits = forward_embeds_logits(model, avg_embed)
    print(f"    Next token prediction: {avg_logits['top5'][:3]}")

    try:
        avg_gen = generate_from_embeds(model, avg_embed)
        avg_correct = answer in avg_gen
        print(f"    Generated: {'✓' if avg_correct else '✗'} {avg_gen[:60]}...")
    except Exception as e:
        avg_gen = f"ERROR: {e}"
        avg_correct = False
        print(f"    ERROR: {e}")

    # ── Condition 3: Average of just ZH + EN ──
    print(f"\n  === AVERAGE (zh + en only) ===")
    avg2_embed = (interp_embeds["zh"] + interp_embeds["en"]) / 2
    avg2_logits = forward_embeds_logits(model, avg2_embed)
    print(f"    Next token: {avg2_logits['top5'][:3]}")
    try:
        avg2_gen = generate_from_embeds(model, avg2_embed)
        avg2_correct = answer in avg2_gen
        print(f"    Generated: {'✓' if avg2_correct else '✗'} {avg2_gen[:60]}...")
    except Exception as e:
        avg2_gen = f"ERROR: {e}"
        avg2_correct = False
        print(f"    ERROR: {e}")

    # ── Condition 4: DIFFERENCE — ZH embed + (EN - JA) direction ──
    # Hypothesis: EN-JA captures the "switch from JA to EN" direction in embed space
    # Adding it to ZH should push ZH toward EN
    print(f"\n  === DIFFERENCE (zh + (en - ja)) ===")
    en_minus_ja = interp_embeds["en"] - interp_embeds["ja"]
    diff_embed = interp_embeds["zh"] + 0.5 * en_minus_ja  # half-strength
    try:
        diff_gen = generate_from_embeds(model, diff_embed)
        diff_correct = answer in diff_gen
        print(f"    Generated: {'✓' if diff_correct else '✗'} {diff_gen[:60]}...")
    except Exception as e:
        diff_gen = f"ERROR: {e}"
        diff_correct = False
        print(f"    ERROR: {e}")

    # ── Condition 5: SVD NULL SPACE — remove top-k language directions ──
    # Stack all 7 interpolated embeddings: (7, ref_len, d) → reshape to (7*ref_len, d)
    # SVD → top-k directions = language variation → project them out
    print(f"\n  === SVD NULL SPACE (remove top-k lang directions from avg) ===")
    all_stacked = torch.stack([interp_embeds[l] for l in LANGS])  # (7, ref_len, d)
    # Compute per-position cross-language mean
    mean_embed = all_stacked.mean(dim=0)  # (ref_len, d) — this is the average

    # Deviation from mean per language: (7, ref_len, d)
    deviations = all_stacked - mean_embed.unsqueeze(0)
    # Reshape to (7*ref_len, d) for SVD
    dev_flat = deviations.reshape(-1, d).cpu().numpy()

    U, S, Vh = np.linalg.svd(dev_flat, full_matrices=False)
    # Top-k language directions
    for k in [1, 3, 5, 10]:
        lang_dirs = torch.tensor(Vh[:k], device=device, dtype=torch.float32)  # (k, d)
        P_lang = lang_dirs.T @ lang_dirs  # (d, d) — projects ONTO language space
        P_null = torch.eye(d, device=device) - P_lang  # projects INTO null space

        # Apply null-space projection to the average embedding
        null_embed = (avg_embed @ P_null.T)  # (ref_len, d)

        # Renormalize to original scale
        null_embed = null_embed * (ref_norm / null_embed.norm(dim=-1, keepdim=True).mean())

        null_logits = forward_embeds_logits(model, null_embed)
        print(f"    k={k}: top1='{null_logits['top1']}', entropy={null_logits['entropy']}")

        if k == 3:  # Generate for k=3 (moderate)
            try:
                null_gen = generate_from_embeds(model, null_embed)
                null_correct = answer in null_gen
                print(f"      Generated: {'✓' if null_correct else '✗'} {null_gen[:60]}...")
            except Exception as e:
                null_gen = f"ERROR: {e}"
                null_correct = False
                print(f"      ERROR: {e}")

    # ── Condition 6: TRUNCATION instead of interpolation ──
    # Just take the first ref_len tokens of each language (no interp distortion)
    print(f"\n  === TRUNCATED AVERAGE (first {ref_len} tokens, no interpolation) ===")
    trunc_embeds = []
    for lang in LANGS:
        e = lang_embeds[lang][:ref_len]  # truncate to ref_len
        if e.shape[0] < ref_len:
            # Pad with zeros
            pad = torch.zeros(ref_len - e.shape[0], d, device=device)
            e = torch.cat([e, pad])
        trunc_embeds.append(e)
    trunc_avg = torch.stack(trunc_embeds).mean(dim=0)

    try:
        trunc_gen = generate_from_embeds(model, trunc_avg)
        trunc_correct = answer in trunc_gen
        print(f"    Generated: {'✓' if trunc_correct else '✗'} {trunc_gen[:60]}...")
    except Exception as e:
        trunc_gen = f"ERROR: {e}"
        trunc_correct = False
        print(f"    ERROR: {e}")

    # ── Condition 7: Cosine similarity between averaged and individual embeds ──
    print(f"\n  === EMBEDDING GEOMETRY ===")
    for lang in LANGS:
        cos = F.cosine_similarity(avg_embed.mean(dim=0, keepdim=True),
                                   interp_embeds[lang].mean(dim=0, keepdim=True)).item()
        print(f"    cos(avg, {lang}) = {cos:.4f}")

    # Store results
    all_results[f"problem_{pi}"] = {
        "prompt_en": prompts["en"],
        "answer": answer,
        "token_lengths": lang_lengths,
        "baselines": {l: baselines.get(l, {}) for l in baselines},
        "avg_7lang": {"gen": avg_gen[:200] if isinstance(avg_gen, str) else str(avg_gen)[:200], "correct": avg_correct,
                      "logits": avg_logits},
        "avg_2lang": {"gen": avg2_gen[:200] if isinstance(avg2_gen, str) else str(avg2_gen)[:200], "correct": avg2_correct},
        "difference": {"gen": diff_gen[:200] if isinstance(diff_gen, str) else str(diff_gen)[:200], "correct": diff_correct},
        "truncated_avg": {"gen": trunc_gen[:200] if isinstance(trunc_gen, str) else str(trunc_gen)[:200], "correct": trunc_correct},
    }


# ── Save ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SAVING")
print("="*70)

output = {
    "experiment": "AM: Meta-Lingual Embedding",
    "model": MODEL_NAME,
    "languages": LANGS,
    "ref_language": REF_LANG,
    "conditions": ["baselines", "avg_7lang", "avg_2lang_zh_en", "difference_zh+(en-ja)",
                    "svd_null_space_k1_3_5_10", "truncated_avg"],
    "results": all_results,
}

with open(OUTPUT_DIR / "expAM_metalingual_embedding.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("Saved to output/expAM_metalingual_embedding.json")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== GRAND SUMMARY ===")
for pk, pv in all_results.items():
    print(f"\n{pv['prompt_en']} (ans={pv['answer']})")
    for bl, bd in pv.get("baselines", {}).items():
        print(f"  {bl}: {'✓' if bd.get('correct') else '✗'}")
    print(f"  avg_7: {'✓' if pv.get('avg_7lang',{}).get('correct') else '✗'}")
    print(f"  avg_2: {'✓' if pv.get('avg_2lang',{}).get('correct') else '✗'}")
    print(f"  diff: {'✓' if pv.get('difference',{}).get('correct') else '✗'}")
    print(f"  trunc: {'✓' if pv.get('truncated_avg',{}).get('correct') else '✗'}")

print("\nDone.")
