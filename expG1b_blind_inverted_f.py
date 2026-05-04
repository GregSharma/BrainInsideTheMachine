"""expG1b: Blind Inverted-F — The Clean Bun Inversion via KV

Same as G1 (math KV at layers 0..L*, describe KV at layers L*+1..35),
but the describe prompt is PROBLEM-AGNOSTIC. Zero information about
which problem is being solved.

Math content enters through attention over the full math KV sequence
(30+ tokens per layer per head) at the lower layers. The describe KV
at upper layers sets mode. The token walks past math shelves below,
describe shelves above.

If the model produces problem-specific descriptions, the content
traveled upward through the residual stream from lower-layer KV
attention — not from the prompt text.
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

L_STAR_SWEEP = [10, 12, 15, 18, 22, 27]

PROBLEMS = [
    {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22",
     "answer": "5", "category": "algebra"},
    {"en": "What is the area of a triangle with base 10 and height 7?",
     "zh": "底边为10、高为7的三角形面积是多少？",
     "answer": "35", "category": "geometry"},
    {"en": "Find the GCD of 84 and 120", "zh": "求84和120的最大公约数",
     "answer": "12", "category": "number_theory"},
]

# BLIND describe prompt — no problem text, no hint about category
DESCRIBE_GENERIC = {
    "en": "Describe what mathematical reasoning is being performed:",
    "zh": "描述正在进行什么数学推理：",
}

# Math prompt — standard solve prompt
SOLVE_TEMPLATE = {
    "en": "{problem}",
    "zh": "{problem}",
}


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
        "hidden_states": outputs.hidden_states,
        "logits": outputs.logits,
        "input_ids": inputs["input_ids"],
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


def build_inverted_f_cache(math_kv, desc_kv, math_seq_len, desc_seq_len, L_star):
    """Math KV at layers 0..L_star, describe KV at layers L_star+1..35."""
    hybrid = copy.deepcopy(desc_kv)
    min_seq = min(math_seq_len, desc_seq_len)

    for i in range(L_star + 1):
        mk = math_kv.layers[i].keys[:, :, :min_seq, :].clone()
        mv = math_kv.layers[i].values[:, :, :min_seq, :].clone()
        if min_seq < desc_seq_len:
            pad = desc_seq_len - min_seq
            mk = torch.cat([mk, mk[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
            mv = torch.cat([mv, mv[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
        elif math_seq_len > desc_seq_len:
            mk = mk[:, :, :desc_seq_len, :]
            mv = mv[:, :, :desc_seq_len, :]
        hybrid.layers[i].keys = mk
        hybrid.layers[i].values = mv

    return hybrid


def generate_baseline(model, tokenizer, text, device):
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_GEN, do_sample=False)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


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

    # ================================================================
    # BASELINES
    # ================================================================
    print(f"{'='*70}")
    print(f"  BASELINES")
    print(f"{'='*70}\n")

    # Generic describe baseline (what the model says with NO math info)
    for lang in ["en", "zh"]:
        text = generate_baseline(model, tokenizer, DESCRIBE_GENERIC[lang], device)
        print(f"  Describe ({lang}): {text[:150]}...")

    # Solve baselines (what the model does with the actual problems)
    for pi, prob in enumerate(PROBLEMS):
        text = generate_baseline(model, tokenizer, prob["en"], device)
        print(f"  Solve p{pi}({prob['category']}): {text[:120]}...")
    print()

    # ================================================================
    # ENCODE PROMPTS
    # ================================================================
    print("Encoding prompts...", flush=True)

    # One generic describe cache per language (problem-agnostic)
    desc_caches = {}
    for lang in ["en", "zh"]:
        desc_caches[lang] = encode_with_cache(
            model, tokenizer, DESCRIBE_GENERIC[lang], device
        )
        print(f"  Describe ({lang}): seq_len={desc_caches[lang]['seq_len']}")

    # Math caches per problem per language
    math_caches = {}
    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            key = f"p{pi}_{lang}"
            math_caches[key] = encode_with_cache(
                model, tokenizer, SOLVE_TEMPLATE[lang].format(problem=prob[lang]), device
            )
            print(f"  Math {key}: seq_len={math_caches[key]['seq_len']}")

    print()

    # ================================================================
    # BLIND INVERTED-F: L* sweep × problems × lang combos
    # ================================================================
    print(f"{'='*70}")
    print(f"  BLIND INVERTED-F: math KV low, generic describe KV high")
    print(f"  Describe prompt has ZERO info about the specific problem.")
    print(f"{'='*70}\n")

    results = []

    for L_star in L_STAR_SWEEP:
        print(f"  --- L* = {L_star} (math KV 0..{L_star}, describe KV {L_star+1}..35) ---")

        for pi, prob in enumerate(PROBLEMS):
            for math_lang in ["en", "zh"]:
                for desc_lang in ["en", "zh"]:
                    mc = math_caches[f"p{pi}_{math_lang}"]
                    dc = desc_caches[desc_lang]

                    hybrid_kv = build_inverted_f_cache(
                        mc["past_kv"], dc["past_kv"],
                        mc["seq_len"], dc["seq_len"],
                        L_star
                    )

                    output = manual_generate(
                        model, tokenizer, hybrid_kv,
                        dc["logits"], MAX_GEN
                    )

                    math_hits, math_total = contains_math_content(output, prob)
                    garbage = is_garbage(output)

                    result = {
                        "L_star": L_star,
                        "problem_idx": pi,
                        "category": prob["category"],
                        "math_lang": math_lang,
                        "desc_lang": desc_lang,
                        "condition": f"{math_lang}_math→{desc_lang}_desc",
                        "output": output[:600],
                        "math_hits": math_hits,
                        "math_total": math_total,
                        "is_garbage": garbage,
                        "has_answer": prob["answer"] in output,
                    }
                    results.append(result)

                    tag = f"p{pi}_{math_lang}→{desc_lang}"
                    print(f"    {tag} math={math_hits}/{math_total} "
                          f"{'GARBAGE' if garbage else 'ans='+str(result['has_answer'])}: "
                          f"{output[:100]}...")

        # Per-L* summary
        lr = [r for r in results if r["L_star"] == L_star]
        n = len(lr)
        garbage_n = sum(1 for r in lr if r["is_garbage"])
        answer_n = sum(1 for r in lr if r["has_answer"])
        avg_hits = sum(r["math_hits"] for r in lr) / n
        print(f"\n  L*={L_star}: n={n}, garbage={garbage_n}, has_answer={answer_n}, "
              f"avg_math_hits={avg_hits:.1f}\n")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0

    # Per-L* aggregate
    summary_by_l = {}
    for L in L_STAR_SWEEP:
        lr = [r for r in results if r["L_star"] == L]
        n = len(lr)
        summary_by_l[str(L)] = {
            "n": n,
            "garbage": sum(1 for r in lr if r["is_garbage"]),
            "has_answer": sum(1 for r in lr if r["has_answer"]),
            "avg_math_hits": round(sum(r["math_hits"] for r in lr) / n, 2),
            "max_math_hits": max(r["math_hits"] for r in lr),
            "math_content_any": sum(1 for r in lr if r["math_hits"] >= 3),
        }

    # Per-problem aggregate (across all L* and langs)
    summary_by_prob = {}
    for pi, prob in enumerate(PROBLEMS):
        pr = [r for r in results if r["problem_idx"] == pi]
        summary_by_prob[prob["category"]] = {
            "avg_hits": round(sum(r["math_hits"] for r in pr) / len(pr), 2),
            "has_answer": sum(1 for r in pr if r["has_answer"]),
            "garbage": sum(1 for r in pr if r["is_garbage"]),
        }

    output_data = {
        "experiment": "G1b: Blind Inverted-F (problem-agnostic describe prompt)",
        "model": MODEL_NAME,
        "n_layers": N_LAYERS,
        "L_star_sweep": L_STAR_SWEEP,
        "describe_prompts": DESCRIBE_GENERIC,
        "problems": [p["en"] for p in PROBLEMS],
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
        "summary_by_lstar": summary_by_l,
        "summary_by_problem": summary_by_prob,
    }

    out_path = OUTPUT_DIR / "expG1b_blind_inverted_f.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  DONE in {elapsed:.1f}s — Saved: {out_path}")
    print(f"{'='*70}\n")

    print(f"  SUMMARY BY L*:")
    print(f"  {'L*':>4} | {'garbage':>7} | {'has_ans':>7} | {'avg_hits':>8} | {'max_hits':>8} | {'content≥3':>9}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")
    for L in L_STAR_SWEEP:
        s = summary_by_l[str(L)]
        print(f"  {L:>4} | {s['garbage']:>4}/{s['n']} | {s['has_answer']:>4}/{s['n']} | "
              f"{s['avg_math_hits']:>8.1f} | {s['max_math_hits']:>8} | {s['math_content_any']:>6}/{s['n']}")

    print(f"\n  SUMMARY BY PROBLEM:")
    for cat, s in summary_by_prob.items():
        print(f"    {cat}: avg_hits={s['avg_hits']:.1f}, has_answer={s['has_answer']}, garbage={s['garbage']}")


if __name__ == "__main__":
    main()
