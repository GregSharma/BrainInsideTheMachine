"""expG1e_14b: 14B Blind Inverted-F — The Bun Inversion at Scale

Replicates G1b/G1c on Qwen2.5-14B (28 layers, d=3584, tied embeddings).
Proportional L* mapping: 3B L*=27 → 7B L*≈21.

Tests:
  1. Hard cutoff: L* sweep [15, 18, 19, 21, 24]
  2. Soft blend: C21_W4 (proportional to 3B C27_W5)
  3. Full 4-language matrix for best configs
  4. Cross-problem identification

7B has tighter cross-lingual representations (cos=0.998 vs 0.956 at 3B),
which should produce cleaner content recovery and possibly enable
cross-lingual bun inversion.

Run on A100 (Colab or remote GPU with ≥24GB VRAM).
"""
import json
import time
import copy
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-14B"
N_LAYERS = 48
D_MODEL = 5120
MAX_GEN = 256

PROBLEMS = [
    {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22",
     "answer": "5", "category": "algebra"},
    {"en": "What is the area of a triangle with base 10 and height 7?",
     "zh": "底边为10、高为7的三角形面积是多少？",
     "answer": "35", "category": "geometry"},
    {"en": "Find the GCD of 84 and 120", "zh": "求84和120的最大公约数",
     "answer": "12", "category": "number_theory"},
    {"en": "What is 17 × 23?", "zh": "17乘以23等于多少？",
     "answer": "391", "category": "arithmetic"},
    {"en": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
     "zh": "如果火车以每小时60英里的速度行驶2.5小时，它走了多远？",
     "answer": "150", "category": "word_problem"},
]

DESCRIBE_GENERIC = {
    "en": "Describe what mathematical reasoning is being performed:",
    "zh": "描述正在进行什么数学推理：",
}

SOLVE_TEMPLATE = {"en": "{problem}", "zh": "{problem}"}

# L* sweep (proportional: 3B 27/36=0.75 → 7B 21/28=0.75)
L_STAR_SWEEP = [30, 36, 38, 41, 44]

# Blend config (proportional to 3B C27_W5)
BLEND_CENTER = 41
BLEND_WIDTH = 6


def build_chat_prompt(tokenizer, text):
    msgs = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def contains_math_content(text, problem):
    answer = problem["answer"]
    generic_kw = ["equation", "variable", "solve", "formula", "calculate",
                  "方程", "变量", "求解", "公式", "计算"]
    specific = {
        "algebra": ["x", "isolat", "subtract", "3x", "22", answer,
                     "方程", "线性"],
        "geometry": ["triangle", "area", "base", "height", "10", "7", answer,
                     "三角", "面积", "底", "高"],
        "number_theory": ["gcd", "greatest common", "divisor", "84", "120", answer,
                          "euclidean", "最大公约数", "辗转"],
        "arithmetic": ["multiply", "17", "23", answer, "乘", "积"],
        "word_problem": ["distance", "speed", "time", "60", "2.5", answer,
                         "距离", "速度", "时间"],
    }
    kws = specific.get(problem["category"], [answer])
    text_lower = text.lower()
    hits = sum(1 for w in kws if w.lower() in text_lower)
    return hits, len(kws)


def is_garbage(text):
    if len(text.strip()) < 5:
        return True
    for emoji in ["🎓", "🕹", "🥗", "🐉"]:
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


def encode_with_cache(model, tokenizer, text, device, raw=False):
    if raw:
        prompt = text  # No chat template — raw text
    else:
        prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
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


def build_hard_cache(math_kv, desc_kv, math_seq_len, desc_seq_len, L_star,
                     n_layers):
    hybrid = copy.deepcopy(desc_kv)
    min_seq = min(math_seq_len, desc_seq_len)
    target_seq = desc_seq_len

    for i in range(min(L_star + 1, n_layers)):
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


def build_soft_blend_cache(math_kv, desc_kv, math_seq_len, desc_seq_len,
                           L_center, width, n_layers):
    hybrid = copy.deepcopy(desc_kv)
    min_seq = min(math_seq_len, desc_seq_len)
    target_seq = desc_seq_len
    zone_start = max(0, L_center - width)
    zone_end = min(n_layers - 1, L_center + width)
    zone_len = zone_end - zone_start

    for i in range(n_layers):
        if i > zone_end:
            continue
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
            hybrid.layers[i].keys = mk
            hybrid.layers[i].values = mv
        else:
            alpha = (i - zone_start) / max(zone_len, 1)
            dk = hybrid.layers[i].keys[:, :, :target_seq, :].clone()
            dv = hybrid.layers[i].values[:, :, :target_seq, :].clone()
            hybrid.layers[i].keys = (1 - alpha) * mk + alpha * dk
            hybrid.layers[i].values = (1 - alpha) * mv + alpha * dv
    return hybrid


def run_experiment(model, tokenizer, device, math_caches, desc_caches, problems,
                   config_name, build_fn, build_kwargs_fn, langs=None):
    """Generic experiment runner.

    build_kwargs_fn(mc, dc, pi, lang) returns kwargs for build_fn.
    """
    if langs is None:
        langs = [("zh", "zh"), ("en", "en")]

    results = []
    for pi, prob in enumerate(problems):
        for math_lang, desc_lang in langs:
            mc = math_caches.get(f"p{pi}_{math_lang}")
            dc = desc_caches.get(desc_lang)
            if mc is None or dc is None:
                continue

            kwargs = build_kwargs_fn(mc, dc, pi, math_lang)
            hybrid = build_fn(**kwargs)
            output = manual_generate(model, tokenizer, hybrid, dc["logits"])
            math_hits, math_total = contains_math_content(output, prob)
            garbage = is_garbage(output)

            result = {
                "config": config_name,
                "problem_idx": pi,
                "category": prob["category"],
                "math_lang": math_lang,
                "desc_lang": desc_lang,
                "output": output[:600],
                "math_hits": math_hits,
                "math_total": math_total,
                "is_garbage": garbage,
                "has_answer": prob["answer"] in output,
            }
            results.append(result)

            tag = f"p{pi}_{math_lang}→{desc_lang}({prob['category']})"
            status = "GARBAGE" if garbage else f"hits={math_hits}/{math_total}"
            print(f"    {tag} [{status}] ans={result['has_answer']}: {output[:80]}...")

    n = len(results)
    if n > 0:
        garb = sum(1 for r in results if r["is_garbage"])
        ans = sum(1 for r in results if r["has_answer"])
        avg = sum(r["math_hits"] for r in results) / n
        print(f"  {config_name}: garbage={garb}/{n}, has_answer={ans}/{n}, avg_hits={avg:.1f}\n")

    return results


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
    print("Encoding prompts...", flush=True)
    desc_caches = {}
    for lang in ["en", "zh"]:
        desc_caches[lang] = encode_with_cache(
            model, tokenizer, DESCRIBE_GENERIC[lang], device
        )
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
    # PART 1: Hard cutoff L* sweep (zh→zh and en→en)
    # ==================================================================
    print(f"{'='*70}")
    print(f"  PART 1: HARD CUTOFF L* SWEEP")
    print(f"{'='*70}\n")

    hard_results = []
    for L_star in L_STAR_SWEEP:
        print(f"  --- Hard L* = {L_star} ---")
        r = run_experiment(
            model, tokenizer, device, math_caches, desc_caches, PROBLEMS,
            config_name=f"hard_L{L_star}",
            build_fn=build_hard_cache,
            build_kwargs_fn=lambda mc, dc, pi, ml, L=L_star: {
                "math_kv": mc["past_kv"], "desc_kv": dc["past_kv"],
                "math_seq_len": mc["seq_len"], "desc_seq_len": dc["seq_len"],
                "L_star": L, "n_layers": N_LAYERS,
            },
        )
        hard_results.extend(r)

    all_results["hard_cutoff"] = hard_results

    # ==================================================================
    # PART 2: Soft blend (C21_W4, proportional to 3B C27_W5)
    # ==================================================================
    print(f"{'='*70}")
    print(f"  PART 2: SOFT BLEND (C{BLEND_CENTER} W{BLEND_WIDTH})")
    print(f"{'='*70}\n")

    print(f"  --- Blend C{BLEND_CENTER} W{BLEND_WIDTH} ---")
    blend_results = run_experiment(
        model, tokenizer, device, math_caches, desc_caches, PROBLEMS,
        config_name=f"blend_C{BLEND_CENTER}_W{BLEND_WIDTH}",
        build_fn=build_soft_blend_cache,
        build_kwargs_fn=lambda mc, dc, pi, ml: {
            "math_kv": mc["past_kv"], "desc_kv": dc["past_kv"],
            "math_seq_len": mc["seq_len"], "desc_seq_len": dc["seq_len"],
            "L_center": BLEND_CENTER, "width": BLEND_WIDTH,
            "n_layers": N_LAYERS,
        },
    )
    all_results["soft_blend"] = blend_results

    # ==================================================================
    # PART 3: Best config with full language matrix
    # ==================================================================
    # Find best L* from Part 1
    best_L = max(L_STAR_SWEEP, key=lambda L: sum(
        r["math_hits"] for r in hard_results if r["config"] == f"hard_L{L}"
    ))
    print(f"\n{'='*70}")
    print(f"  PART 3: FULL LANGUAGE MATRIX (best hard L*={best_L})")
    print(f"{'='*70}\n")

    all_langs = [("en", "en"), ("en", "zh"), ("zh", "en"), ("zh", "zh")]
    full_lang = run_experiment(
        model, tokenizer, device, math_caches, desc_caches, PROBLEMS,
        config_name=f"full_lang_L{best_L}",
        build_fn=build_hard_cache,
        build_kwargs_fn=lambda mc, dc, pi, ml, L=best_L: {
            "math_kv": mc["past_kv"], "desc_kv": dc["past_kv"],
            "math_seq_len": mc["seq_len"], "desc_seq_len": dc["seq_len"],
            "L_star": L, "n_layers": N_LAYERS,
        },
        langs=all_langs,
    )
    all_results["full_lang_matrix"] = full_lang

    # ==================================================================
    # PART 4: RAW ENCODING (no chat template) — fixes zh→zh leakage
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  PART 4: RAW ENCODING (no chat template)")
    print(f"  Tests whether zh→zh system prompt leakage disappears")
    print(f"{'='*70}\n")

    raw_desc_caches = {}
    raw_math_caches = {}
    for lang in ["en", "zh"]:
        raw_desc_caches[lang] = encode_with_cache(
            model, tokenizer, DESCRIBE_GENERIC[lang], device, raw=True
        )
        print(f"  Raw describe ({lang}): seq_len={raw_desc_caches[lang]['seq_len']}")
    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            key = f"p{pi}_{lang}"
            raw_math_caches[key] = encode_with_cache(
                model, tokenizer, SOLVE_TEMPLATE[lang].format(problem=prob[lang]),
                device, raw=True
            )

    raw_results = run_experiment(
        model, tokenizer, device, raw_math_caches, raw_desc_caches, PROBLEMS,
        config_name=f"raw_L{best_L}",
        build_fn=build_hard_cache,
        build_kwargs_fn=lambda mc, dc, pi, ml, L=best_L: {
            "math_kv": mc["past_kv"], "desc_kv": dc["past_kv"],
            "math_seq_len": mc["seq_len"], "desc_seq_len": dc["seq_len"],
            "L_star": L, "n_layers": N_LAYERS,
        },
        langs=[("zh", "zh"), ("en", "en"), ("zh", "en")],
    )
    all_results["raw_encoding"] = raw_results

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  14B BUN INVERSION — COMPLETE in {elapsed:.1f}s")
    print(f"{'='*70}\n")

    # Summary table
    print(f"  {'Config':>25} | {'garbage':>7} | {'has_ans':>7} | {'avg_hits':>8} | {'max_hits':>8}")
    print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    for L in L_STAR_SWEEP:
        cr = [r for r in hard_results if r["config"] == f"hard_L{L}"]
        if cr:
            n = len(cr)
            print(f"  {'Hard L*='+str(L):>25} | {sum(1 for r in cr if r['is_garbage']):>4}/{n} | "
                  f"{sum(1 for r in cr if r['has_answer']):>4}/{n} | "
                  f"{sum(r['math_hits'] for r in cr)/n:>8.1f} | "
                  f"{max(r['math_hits'] for r in cr):>8}")

    n = len(blend_results)
    if n > 0:
        label = f"Blend C{BLEND_CENTER} W{BLEND_WIDTH}"
        print(f"  {label:>25} | {sum(1 for r in blend_results if r['is_garbage']):>4}/{n} | "
              f"{sum(1 for r in blend_results if r['has_answer']):>4}/{n} | "
              f"{sum(r['math_hits'] for r in blend_results)/n:>8.1f} | "
              f"{max(r['math_hits'] for r in blend_results):>8}")

    if raw_results:
        n = len(raw_results)
        label = f"Raw L*={best_L}"
        print(f"  {label:>25} | {sum(1 for r in raw_results if r['is_garbage']):>4}/{n} | "
              f"{sum(1 for r in raw_results if r['has_answer']):>4}/{n} | "
              f"{sum(r['math_hits'] for r in raw_results)/n:>8.1f} | "
              f"{max(r['math_hits'] for r in raw_results):>8}")

    # Per-problem breakdown for best config
    print(f"\n  PER-PROBLEM (best hard L*={best_L}):")
    for pi, prob in enumerate(PROBLEMS):
        pr = [r for r in hard_results if r["config"] == f"hard_L{best_L}" and r["problem_idx"] == pi]
        if pr:
            hits = sum(r["math_hits"] for r in pr) / len(pr)
            ans = sum(1 for r in pr if r["has_answer"])
            garb = sum(1 for r in pr if r["is_garbage"])
            print(f"    p{pi}({prob['category']}): avg_hits={hits:.1f}, has_answer={ans}/{len(pr)}, garbage={garb}/{len(pr)}")

    # Save
    summary = {
        "model": MODEL_NAME,
        "n_layers": N_LAYERS,
        "d_model": D_MODEL,
        "elapsed_s": round(elapsed, 1),
        "best_hard_L": best_L,
        "n_problems": len(PROBLEMS),
    }
    all_results["summary"] = summary

    out_path = OUTPUT_DIR / "expG1e_14b_bun_inversion.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
