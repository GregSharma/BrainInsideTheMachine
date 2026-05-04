"""expG1: Inverted-F KV Surgery + Natural Description Baseline (G4)

G4: Natural description baseline — prompt model with "Describe what math
    operation is performed in: [problem]". Ground truth for what descriptions
    look like at 3B scale.

G1: Inverted-F — math KV at layers 0..L*, describe KV at layers L*+1..L35.
    Lower layers feed math content through residual stream, upper layers
    set describe mode through attention KV. L* sweep: [12, 15, 18, 22, 27].

    If the layer boundary hypothesis is right (content below, mode above),
    this should produce descriptions referencing actual math content.
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

L_STAR_SWEEP = [12, 15, 18, 22, 27]

PROBLEMS = [
    {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22",
     "answer": "5", "category": "algebra"},
    {"en": "What is the area of a triangle with base 10 and height 7?",
     "zh": "底边为10、高为7的三角形面积是多少？",
     "answer": "35", "category": "geometry"},
    {"en": "Find the GCD of 84 and 120", "zh": "求84和120的最大公约数",
     "answer": "12", "category": "number_theory"},
]

# Describe prompt: asks for description of the math operation, NOT to solve it
DESCRIBE_TEMPLATE = {
    "en": "Describe in detail what mathematical operation or reasoning is needed to solve this problem (do NOT solve it, just describe the approach): {problem}",
    "zh": "详细描述解决这个问题需要什么数学运算或推理方法（不要求解，只描述方法）：{problem}",
}

# Solve prompt: standard math prompt
SOLVE_TEMPLATE = {
    "en": "{problem}",
    "zh": "{problem}",
}


def build_chat_prompt(tokenizer, text):
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def detect_lang(text):
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if cjk > len(text) * 0.1 else "en"


def contains_math_content(text, problem):
    """Check if output references the specific math problem's content."""
    answer = problem["answer"]
    keywords = {
        "algebra": ["solve", "equation", "variable", "x", answer, "求解", "方程", "linear", "线性"],
        "geometry": ["area", "triangle", "base", "height", answer, "面积", "三角", "底", "formula"],
        "number_theory": ["gcd", "greatest common", "divisor", answer, "最大公约数", "公约数", "euclidean"],
    }
    cat_words = keywords.get(problem["category"], [answer])
    text_lower = text.lower()
    hits = sum(1 for w in cat_words if w.lower() in text_lower)
    return hits, len(cat_words)


def is_garbage(text):
    if len(text.strip()) < 5:
        return True
    if text.count("🎓") > 5 or text.count("🕹") > 5 or text.count("🥗") > 5:
        return True
    if "0000000" in text or text.count("玳瑁") > 3:
        return True
    if len(text) > 40:
        for i in range(0, min(len(text) - 4, 40)):
            chunk = text[i:i+4]
            if chunk.strip() and text.count(chunk) > 10:
                return True
    return False


def classify_output(text, problem):
    """Classify output as: describes, solves, garbage, or other."""
    if is_garbage(text):
        return "garbage"

    # Description indicators: talks ABOUT the math without giving the answer directly
    desc_keywords = [
        "this problem", "this involves", "we need to", "the approach",
        "the operation", "requires", "involves", "method", "technique",
        "strategy", "step", "first", "identify", "recognize",
        "这个问题", "需要", "方法", "运算", "步骤", "首先", "识别",
        "involves solving", "requires finding", "mathematical operation",
    ]
    solve_keywords = [
        "the answer is", "= " + problem["answer"], "equals " + problem["answer"],
        "x = " + problem["answer"], "答案是", "等于",
        "therefore", "so x", "所以",
    ]

    text_lower = text.lower()
    desc_hits = sum(1 for k in desc_keywords if k.lower() in text_lower)
    solve_hits = sum(1 for k in solve_keywords if k.lower() in text_lower)

    has_answer = problem["answer"] in text
    math_hits, _ = contains_math_content(text, problem)

    if desc_hits >= 2 and not has_answer:
        return "describes"
    elif desc_hits >= 2 and has_answer:
        return "describes_with_answer"  # describes but also gives answer
    elif has_answer and solve_hits >= 1:
        return "solves"
    elif math_hits >= 2:
        return "math_content"
    else:
        return "other"


def encode_with_cache(model, tokenizer, text, device):
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    return {
        "past_kv": outputs.past_key_values,
        "hidden_states": outputs.hidden_states,
        "logits": outputs.logits,
        "input_ids": inputs["input_ids"],
        "seq_len": inputs["input_ids"].shape[1],
    }


def generate_baseline(model, tokenizer, text, device):
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_GEN, do_sample=False)
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def manual_generate(model, tokenizer, cache, first_logits, max_gen=MAX_GEN):
    """Manual greedy decode from a cache + first-token logits."""
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


def build_inverted_f_cache(math_kv, desc_kv, math_seq_len, desc_seq_len, L_star):
    """Build Inverted-F cache: math KV at layers 0..L_star, describe KV at layers L_star+1..35.

    Lower layers get math content (feeds into residual stream).
    Upper layers get describe mode (sets generation mode via attention).

    Seq len mismatch handled by truncation to min + padding with last-token repeat.
    """
    hybrid = copy.deepcopy(desc_kv)  # Start from describe (upper layers keep theirs)
    min_seq = min(math_seq_len, desc_seq_len)

    # Replace layers 0..L_star with math KV
    for i in range(L_star + 1):
        mk = math_kv.layers[i].keys[:, :, :min_seq, :].clone()
        mv = math_kv.layers[i].values[:, :, :min_seq, :].clone()
        # Pad to desc_seq_len if math is shorter
        if min_seq < desc_seq_len:
            pad = desc_seq_len - min_seq
            mk = torch.cat([mk, mk[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
            mv = torch.cat([mv, mv[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
        # Truncate to desc_seq_len if math is longer
        elif math_seq_len > desc_seq_len:
            mk = mk[:, :, :desc_seq_len, :]
            mv = mv[:, :, :desc_seq_len, :]
        hybrid.layers[i].keys = mk
        hybrid.layers[i].values = mv

    return hybrid


def run_g4_baseline(model, tokenizer, device):
    """G4: Natural description baseline + residual stream mode divergence.

    Part 1: What does the model produce when asked to describe vs solve?
    Part 2: Where in the residual stream do solve/describe diverge?
             (Informs L* selection for G1 and G5 viability.)
    """
    print(f"\n{'='*70}")
    print(f"  G4: NATURAL DESCRIPTION BASELINE")
    print(f"{'='*70}\n")

    results = []
    mode_divergence = []  # Per-layer cosine between solve/describe hidden states

    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            # Description prompt
            desc_prompt = DESCRIBE_TEMPLATE[lang].format(problem=prob[lang])
            desc_output = generate_baseline(model, tokenizer, desc_prompt, device)
            desc_class = classify_output(desc_output, prob)

            # Solve prompt (for comparison)
            solve_prompt = SOLVE_TEMPLATE[lang].format(problem=prob[lang])
            solve_output = generate_baseline(model, tokenizer, solve_prompt, device)
            solve_class = classify_output(solve_output, prob)

            result = {
                "problem_idx": pi,
                "problem": prob[lang],
                "lang": lang,
                "category": prob["category"],
                "describe_output": desc_output[:500],
                "describe_class": desc_class,
                "solve_output": solve_output[:500],
                "solve_class": solve_class,
            }
            results.append(result)

            print(f"  p{pi}_{lang} ({prob['category']})")
            print(f"    DESCRIBE [{desc_class}]: {desc_output[:120]}...")
            print(f"    SOLVE    [{solve_class}]: {solve_output[:120]}...")

            # --- Residual stream mode divergence ---
            # Encode both prompts with hidden states to compare per-layer
            desc_cache = encode_with_cache(model, tokenizer, desc_prompt, device)
            solve_cache = encode_with_cache(model, tokenizer, solve_prompt, device)

            layer_cos = []
            for L in range(N_LAYERS):
                # Last-token hidden state at each layer
                h_desc = desc_cache["hidden_states"][L + 1][:, -1, :].float()
                h_solve = solve_cache["hidden_states"][L + 1][:, -1, :].float()
                cos = torch.nn.functional.cosine_similarity(h_desc, h_solve, dim=-1).item()
                layer_cos.append(round(cos, 4))

            mode_divergence.append({
                "problem_idx": pi,
                "lang": lang,
                "layer_cos_solve_vs_describe": layer_cos,
            })

            # Print key divergence points
            min_cos = min(layer_cos)
            min_layer = layer_cos.index(min_cos)
            print(f"    MODE DIVERGENCE: min cos={min_cos:.3f} at L{min_layer}, "
                  f"L15={layer_cos[15]:.3f}, L22={layer_cos[22]:.3f}, L27={layer_cos[27]:.3f}")
            print()

    # Summary
    desc_classes = [r["describe_class"] for r in results]
    solve_classes = [r["solve_class"] for r in results]
    print(f"  G4 SUMMARY:")
    for cls in set(desc_classes):
        print(f"    Describe → {cls}: {desc_classes.count(cls)}/{len(desc_classes)}")
    for cls in set(solve_classes):
        print(f"    Solve → {cls}: {solve_classes.count(cls)}/{len(solve_classes)}")

    # Mode divergence summary: mean cos per layer
    import numpy as np
    all_cos = np.array([m["layer_cos_solve_vs_describe"] for m in mode_divergence])
    mean_cos = all_cos.mean(axis=0)
    print(f"\n  MODE DIVERGENCE (mean cos solve vs describe):")
    print(f"    Early (L0-L5):  {mean_cos[:6].mean():.3f}")
    print(f"    Mid   (L10-L17): {mean_cos[10:18].mean():.3f}")
    print(f"    Late  (L22-L27): {mean_cos[22:28].mean():.3f}")
    print(f"    Final (L30-L35): {mean_cos[30:36].mean():.3f}")
    diverge_layer = int(np.argmin(mean_cos))
    print(f"    Max divergence at L{diverge_layer} (mean cos={mean_cos[diverge_layer]:.3f})")
    print(f"    → Suggested L* for G1: {diverge_layer} (mode split point)")

    return results, mode_divergence


def run_g1_inverted_f(model, tokenizer, device):
    """G1: Inverted-F — math KV low, describe KV high. Sweep L*."""
    print(f"\n{'='*70}")
    print(f"  G1: INVERTED-F KV SURGERY")
    print(f"{'='*70}\n")

    # Pre-encode all prompts
    print("  Encoding prompts...", flush=True)
    caches = {}
    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            # Math (solve) prompt
            solve_text = SOLVE_TEMPLATE[lang].format(problem=prob[lang])
            caches[f"math_p{pi}_{lang}"] = encode_with_cache(model, tokenizer, solve_text, device)

            # Describe prompt
            desc_text = DESCRIBE_TEMPLATE[lang].format(problem=prob[lang])
            caches[f"desc_p{pi}_{lang}"] = encode_with_cache(model, tokenizer, desc_text, device)

    print(f"  Encoded {len(caches)} prompts.\n", flush=True)

    results = []

    for L_star in L_STAR_SWEEP:
        print(f"  --- L* = {L_star} (math KV 0..{L_star}, describe KV {L_star+1}..35) ---")

        for pi, prob in enumerate(PROBLEMS):
            for math_lang in ["en", "zh"]:
                for desc_lang in ["en", "zh"]:
                    math_cache = caches[f"math_p{pi}_{math_lang}"]
                    desc_cache = caches[f"desc_p{pi}_{desc_lang}"]

                    # Build inverted-F hybrid: math low, describe high
                    hybrid_kv = build_inverted_f_cache(
                        math_cache["past_kv"], desc_cache["past_kv"],
                        math_cache["seq_len"], desc_cache["seq_len"],
                        L_star
                    )

                    # Generate using describe prompt's first-token logits
                    # (the desc prompt sets the initial generation token)
                    output = manual_generate(
                        model, tokenizer, hybrid_kv,
                        desc_cache["logits"], MAX_GEN
                    )

                    output_class = classify_output(output, prob)
                    math_hits, math_total = contains_math_content(output, prob)

                    result = {
                        "L_star": L_star,
                        "problem_idx": pi,
                        "category": prob["category"],
                        "math_lang": math_lang,
                        "desc_lang": desc_lang,
                        "condition": f"{math_lang}_math→{desc_lang}_desc",
                        "output": output[:500],
                        "output_class": output_class,
                        "math_hits": math_hits,
                        "math_total": math_total,
                        "is_garbage": is_garbage(output),
                        "has_answer": prob["answer"] in output,
                    }
                    results.append(result)

                    tag = f"p{pi}_{math_lang}→{desc_lang}"
                    print(f"    {tag} [{output_class}] math={math_hits}/{math_total}: {output[:100]}...")

        # Per-L* summary
        l_results = [r for r in results if r["L_star"] == L_star]
        classes = [r["output_class"] for r in l_results]
        garbage_n = sum(1 for r in l_results if r["is_garbage"])
        answer_n = sum(1 for r in l_results if r["has_answer"])
        print(f"\n  L*={L_star} SUMMARY: {len(l_results)} gens, garbage={garbage_n}, has_answer={answer_n}")
        for cls in sorted(set(classes)):
            print(f"    {cls}: {classes.count(cls)}")
        print()

    return results


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # G4: Natural description baseline + mode divergence
    g4_results, mode_divergence = run_g4_baseline(model, tokenizer, device)

    # G1: Inverted-F
    g1_results = run_g1_inverted_f(model, tokenizer, device)

    # Global summary
    elapsed = time.time() - t0

    # Aggregate G1
    g1_by_lstar = {}
    for L in L_STAR_SWEEP:
        lr = [r for r in g1_results if r["L_star"] == L]
        g1_by_lstar[str(L)] = {
            "n": len(lr),
            "describes": sum(1 for r in lr if r["output_class"] == "describes"),
            "describes_with_answer": sum(1 for r in lr if r["output_class"] == "describes_with_answer"),
            "solves": sum(1 for r in lr if r["output_class"] == "solves"),
            "math_content": sum(1 for r in lr if r["output_class"] == "math_content"),
            "garbage": sum(1 for r in lr if r["output_class"] == "garbage"),
            "other": sum(1 for r in lr if r["output_class"] == "other"),
            "has_answer_frac": sum(1 for r in lr if r["has_answer"]) / max(len(lr), 1),
        }

    output = {
        "experiment": "G4+G1: Natural Description Baseline + Inverted-F KV Surgery",
        "model": MODEL_NAME,
        "n_layers": N_LAYERS,
        "L_star_sweep": L_STAR_SWEEP,
        "problems": [p["en"] for p in PROBLEMS],
        "elapsed_seconds": round(elapsed, 1),
        "g4_baseline": g4_results,
        "g4_mode_divergence": mode_divergence,
        "g1_results": g1_results,
        "g1_summary_by_lstar": g1_by_lstar,
    }

    out_path = OUTPUT_DIR / "expG1_inverted_f.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Saved: {out_path}")
    print(f"{'='*70}")

    # Print G1 summary table
    print(f"\n  G1 SUMMARY TABLE:")
    print(f"  {'L*':>4} | {'describes':>9} | {'desc+ans':>8} | {'solves':>6} | {'garbage':>7} | {'other':>5} | {'ans%':>5}")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}")
    for L in L_STAR_SWEEP:
        s = g1_by_lstar[str(L)]
        print(f"  {L:>4} | {s['describes']:>9} | {s['describes_with_answer']:>8} | {s['solves']:>6} | {s['garbage']:>7} | {s['other']:>5} | {s['has_answer_frac']:>5.0%}")


if __name__ == "__main__":
    main()
