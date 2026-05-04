"""expG1c: Soft-Blend Inverted-F — Smooth the Seam

G1b showed exact problem recovery at L*=27 zh→zh (geometry: base 10,
height 7, area 35 — all from math KV, describe prompt blind). But 6/12
garbage from the hard seam at L*=27 (only 8 describe layers).

G1c smooths the transition: instead of a hard cutoff at L*, blend math
and describe KV over a transition zone. Layers below the zone are pure
math KV, layers above are pure describe KV, and layers in the zone are
a weighted average.

If this keeps L*=27's content recovery while cutting the garbage rate,
we have the right architecture for 7B replication.

Also tests the best conditions from G1b more thoroughly:
- zh→zh (cleanest, same-language)
- en→en (partial structural recovery)
- More L* values around the sweet spot
"""
import json
import time
import copy
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_GEN = 256

PROBLEMS = [
    {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22",
     "answer": "5", "category": "algebra"},
    {"en": "What is the area of a triangle with base 10 and height 7?",
     "zh": "底边为10、高为7的三角形面积是多少？",
     "answer": "35", "category": "geometry"},
    {"en": "Find the GCD of 84 and 120", "zh": "求84和120的最大公约数",
     "answer": "12", "category": "number_theory"},
]

DESCRIBE_GENERIC = {
    "en": "Describe what mathematical reasoning is being performed:",
    "zh": "描述正在进行什么数学推理：",
}

SOLVE_TEMPLATE = {"en": "{problem}", "zh": "{problem}"}


def build_chat_prompt(tokenizer, text):
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def contains_math_content(text, problem):
    answer = problem["answer"]
    keywords = {
        "algebra": ["solve", "equation", "variable", "x", answer, "求解", "方程",
                     "linear", "isolat", "subtract", "3x", "22"],
        "geometry": ["area", "triangle", "base", "height", answer, "面积", "三角",
                     "底", "formula", "multiply", "10", "7"],
        "number_theory": ["gcd", "greatest common", "divisor", answer, "最大公约数",
                          "euclidean", "remainder", "84", "120"],
    }
    cat_words = keywords.get(problem["category"], [answer])
    text_lower = text.lower()
    hits = sum(1 for w in cat_words if w.lower() in text_lower)
    return hits, len(cat_words)


def is_garbage(text):
    if len(text.strip()) < 5:
        return True
    for emoji in ["🎓", "🕹", "🥗"]:
        if text.count(emoji) > 5:
            return True
    if "0000000" in text or text.count("玳瑁") > 3:
        return True
    if len(text) > 60:
        for i in range(0, min(len(text) - 8, 40)):
            chunk = text[i:i + 8]
            if chunk.strip() and text.count(chunk) > 6:
                return True
    return False


def encode_with_cache(model, tokenizer, text, device):
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    return {
        "past_kv": outputs.past_key_values,
        "logits": outputs.logits,
        "seq_len": inputs["input_ids"].shape[1],
    }


def manual_generate(model, tokenizer, cache, first_logits, max_gen=MAX_GEN):
    kv = copy.deepcopy(cache)
    next_token = first_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    eos = tokenizer.eos_token_id
    for _ in range(max_gen - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tid = next_token.item()
        if tid == eos:
            break
        generated.append(tid)
    return tokenizer.decode(generated, skip_special_tokens=True)


def build_soft_blend_cache(math_kv, desc_kv, math_seq_len, desc_seq_len,
                           L_center, width):
    """Build a soft-blended KV cache.

    Layers 0..L_center-width:   pure math KV
    Layers L_center-width+1..L_center+width: linear blend (math→describe)
    Layers L_center+width+1..35: pure describe KV

    The blend weight at layer L in the transition zone:
      alpha = (L - (L_center - width)) / (2 * width)
      KV = (1-alpha) * math_KV + alpha * desc_KV

    Seq len mismatch: truncate to min, pad with last-token repeat.
    """
    hybrid = copy.deepcopy(desc_kv)
    min_seq = min(math_seq_len, desc_seq_len)
    target_seq = desc_seq_len

    zone_start = max(0, L_center - width)
    zone_end = min(N_LAYERS - 1, L_center + width)
    zone_len = zone_end - zone_start

    for i in range(N_LAYERS):
        if i > zone_end:
            # Pure describe — already in hybrid from deepcopy
            continue

        # Get math KV for this layer
        mk = math_kv.layers[i].keys[:, :, :min_seq, :].clone()
        mv = math_kv.layers[i].values[:, :, :min_seq, :].clone()
        if min_seq < target_seq:
            pad = target_seq - min_seq
            mk = torch.cat([mk, mk[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
            mv = torch.cat([mv, mv[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
        elif math_seq_len > target_seq:
            mk = mk[:, :, :target_seq, :]
            mv = mv[:, :, :target_seq, :]

        if i < zone_start:
            # Pure math
            hybrid.layers[i].keys = mk
            hybrid.layers[i].values = mv
        else:
            # Blend zone: alpha goes from 0 (pure math) to 1 (pure describe)
            if zone_len > 0:
                alpha = (i - zone_start) / zone_len
            else:
                alpha = 0.5
            dk = hybrid.layers[i].keys[:, :, :target_seq, :].clone()
            dv = hybrid.layers[i].values[:, :, :target_seq, :].clone()
            hybrid.layers[i].keys = (1 - alpha) * mk + alpha * dk
            hybrid.layers[i].values = (1 - alpha) * mv + alpha * dv

    return hybrid


def build_hard_cache(math_kv, desc_kv, math_seq_len, desc_seq_len, L_star):
    """Hard cutoff: math 0..L_star, describe L_star+1..35."""
    hybrid = copy.deepcopy(desc_kv)
    min_seq = min(math_seq_len, desc_seq_len)
    target_seq = desc_seq_len

    for i in range(L_star + 1):
        mk = math_kv.layers[i].keys[:, :, :min_seq, :].clone()
        mv = math_kv.layers[i].values[:, :, :min_seq, :].clone()
        if min_seq < target_seq:
            pad = target_seq - min_seq
            mk = torch.cat([mk, mk[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
            mv = torch.cat([mv, mv[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
        elif math_seq_len > target_seq:
            mk = mk[:, :, :target_seq, :]
            mv = mv[:, :, :target_seq, :]
        hybrid.layers[i].keys = mk
        hybrid.layers[i].values = mv

    return hybrid


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # Encode
    print("Encoding...", flush=True)
    desc_caches = {}
    for lang in ["en", "zh"]:
        desc_caches[lang] = encode_with_cache(model, tokenizer, DESCRIBE_GENERIC[lang], device)
        print(f"  Describe ({lang}): seq_len={desc_caches[lang]['seq_len']}")

    math_caches = {}
    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            key = f"p{pi}_{lang}"
            math_caches[key] = encode_with_cache(
                model, tokenizer, SOLVE_TEMPLATE[lang].format(problem=prob[lang]), device
            )
            print(f"  Math {key}: seq_len={math_caches[key]['seq_len']}")
    print()

    all_results = {}

    # ==================================================================
    # PART 1: SOFT BLEND — sweep L_center and width
    # ==================================================================
    print(f"{'='*70}")
    print(f"  PART 1: SOFT BLEND (smooth seam)")
    print(f"  Transition zone around L_center, width layers each side")
    print(f"{'='*70}\n")

    # Configurations: (L_center, width)
    BLEND_CONFIGS = [
        (24, 3),   # math 0-21, blend 21-27, describe 28-35
        (25, 4),   # math 0-21, blend 21-29, describe 30-35
        (27, 3),   # math 0-24, blend 24-30, describe 31-35
        (27, 5),   # math 0-22, blend 22-32, describe 33-35
        (22, 4),   # math 0-18, blend 18-26, describe 27-35
    ]

    blend_results = []

    for L_center, width in BLEND_CONFIGS:
        zone_start = max(0, L_center - width)
        zone_end = min(35, L_center + width)
        label = f"C{L_center}_W{width} (math 0-{zone_start-1 if zone_start>0 else 0}, blend {zone_start}-{zone_end}, desc {zone_end+1}-35)"
        print(f"  --- {label} ---")

        for pi, prob in enumerate(PROBLEMS):
            # Focus on the conditions that worked in G1b: zh→zh and en→en
            for lang in ["zh", "en"]:
                mc = math_caches[f"p{pi}_{lang}"]
                dc = desc_caches[lang]

                hybrid = build_soft_blend_cache(
                    mc["past_kv"], dc["past_kv"],
                    mc["seq_len"], dc["seq_len"],
                    L_center, width
                )

                output = manual_generate(model, tokenizer, hybrid, dc["logits"])
                math_hits, math_total = contains_math_content(output, prob)
                garbage = is_garbage(output)

                result = {
                    "L_center": L_center,
                    "width": width,
                    "zone": f"{zone_start}-{zone_end}",
                    "problem_idx": pi,
                    "category": prob["category"],
                    "lang": lang,
                    "output": output[:600],
                    "math_hits": math_hits,
                    "math_total": math_total,
                    "is_garbage": garbage,
                    "has_answer": prob["answer"] in output,
                }
                blend_results.append(result)

                tag = f"p{pi}_{lang}({prob['category']})"
                status = "GARBAGE" if garbage else f"hits={math_hits}/{math_total}"
                print(f"    {tag} [{status}] ans={result['has_answer']}: {output[:90]}...")

        # Per-config summary
        cr = [r for r in blend_results if r["L_center"] == L_center and r["width"] == width]
        n = len(cr)
        garb = sum(1 for r in cr if r["is_garbage"])
        ans = sum(1 for r in cr if r["has_answer"])
        avg = sum(r["math_hits"] for r in cr) / n
        print(f"  {label}: garbage={garb}/{n}, has_answer={ans}/{n}, avg_hits={avg:.1f}\n")

    all_results["soft_blend"] = blend_results

    # ==================================================================
    # PART 2: HARD COMPARISON at same L* for reference
    # ==================================================================
    print(f"{'='*70}")
    print(f"  PART 2: HARD CUTOFF COMPARISON (same L* values)")
    print(f"{'='*70}\n")

    hard_results = []

    for L_star in [22, 24, 27]:
        print(f"  --- Hard L* = {L_star} ---")

        for pi, prob in enumerate(PROBLEMS):
            for lang in ["zh", "en"]:
                mc = math_caches[f"p{pi}_{lang}"]
                dc = desc_caches[lang]

                hybrid = build_hard_cache(
                    mc["past_kv"], dc["past_kv"],
                    mc["seq_len"], dc["seq_len"],
                    L_star
                )

                output = manual_generate(model, tokenizer, hybrid, dc["logits"])
                math_hits, math_total = contains_math_content(output, prob)
                garbage = is_garbage(output)

                result = {
                    "L_star": L_star,
                    "problem_idx": pi,
                    "category": prob["category"],
                    "lang": lang,
                    "output": output[:600],
                    "math_hits": math_hits,
                    "math_total": math_total,
                    "is_garbage": garbage,
                    "has_answer": prob["answer"] in output,
                }
                hard_results.append(result)

                tag = f"p{pi}_{lang}({prob['category']})"
                status = "GARBAGE" if garbage else f"hits={math_hits}/{math_total}"
                print(f"    {tag} [{status}] ans={result['has_answer']}: {output[:90]}...")

        cr = [r for r in hard_results if r["L_star"] == L_star]
        n = len(cr)
        garb = sum(1 for r in cr if r["is_garbage"])
        ans = sum(1 for r in cr if r["has_answer"])
        avg = sum(r["math_hits"] for r in cr) / n
        print(f"  Hard L*={L_star}: garbage={garb}/{n}, has_answer={ans}/{n}, avg_hits={avg:.1f}\n")

    all_results["hard_cutoff"] = hard_results

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0

    # Compare blend vs hard
    print(f"\n{'='*70}")
    print(f"  COMPARISON: SOFT BLEND vs HARD CUTOFF")
    print(f"{'='*70}\n")

    print(f"  {'Config':>25} | {'garbage':>7} | {'has_ans':>7} | {'avg_hits':>8} | {'max_hits':>8}")
    print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    for L_center, width in BLEND_CONFIGS:
        cr = [r for r in blend_results if r["L_center"] == L_center and r["width"] == width]
        n = len(cr)
        label = f"Blend C{L_center} W{width}"
        print(f"  {label:>25} | {sum(1 for r in cr if r['is_garbage']):>4}/{n} | "
              f"{sum(1 for r in cr if r['has_answer']):>4}/{n} | "
              f"{sum(r['math_hits'] for r in cr)/n:>8.1f} | "
              f"{max(r['math_hits'] for r in cr):>8}")

    for L_star in [22, 24, 27]:
        cr = [r for r in hard_results if r["L_star"] == L_star]
        n = len(cr)
        label = f"Hard L*={L_star}"
        print(f"  {label:>25} | {sum(1 for r in cr if r['is_garbage']):>4}/{n} | "
              f"{sum(1 for r in cr if r['has_answer']):>4}/{n} | "
              f"{sum(r['math_hits'] for r in cr)/n:>8.1f} | "
              f"{max(r['math_hits'] for r in cr):>8}")

    # Save
    all_results["summary"] = {
        "elapsed_s": round(elapsed, 1),
        "blend_configs": [{"L_center": c, "width": w} for c, w in BLEND_CONFIGS],
        "hard_L_stars": [22, 24, 27],
    }

    out_path = OUTPUT_DIR / "expG1c_soft_blend.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
