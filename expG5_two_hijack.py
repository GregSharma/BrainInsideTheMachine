"""expG5: Two-Hijack Bun Inversion — F2 KV + Describe Steering Vector (14B, A100)

The content/mode separation hypothesis:
- KV cache → what model attends to (content). Give it 100% math.
- Residual stream → what model does (mode). Steer toward describe.

Prior exps (G1-G1e) traded content for mode through L* splitting.
G5 decouples: full math KV for content + steering vector for mode.

Steering vector delta^(l) = mean(h_desc) - mean(h_solve), averaged over trivial
calibration problems. Problem-agnostic, computed once.

Raw encoding throughout (no chat template — per G1e finding).
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

# Test problems (same as G1e)
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

# Calibration problems for steering vector (NOT in test set)
CALIBRATION_PROBLEMS = [
    {"en": "What is 2 + 2?", "zh": "2加2等于多少？"},
    {"en": "What is 5 times 3?", "zh": "5乘3等于多少？"},
    {"en": "What is 10 minus 4?", "zh": "10减4等于多少？"},
    {"en": "What is 100 divided by 5?", "zh": "100除以5等于多少？"},
    {"en": "Solve for y: y + 1 = 3", "zh": "求解y：y + 1 = 3"},
]

DESCRIBE_GENERIC = {
    "en": "Describe what mathematical reasoning is being performed:",
    "zh": "描述正在进行什么数学推理：",
}

ALPHA_SWEEP = [0.5, 1.0, 2.0, 3.0, 5.0]
L_STAR_CONTROL = 41  # Best from G1e

LANG_COMBOS = [("zh", "en"), ("en", "en"), ("zh", "zh")]


def contains_math_content(text, problem):
    answer = problem["answer"]
    specific = {
        "algebra": ["x", "isolat", "subtract", "3x", "22", answer, "方程", "线性"],
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


def is_descriptive(text):
    """Count descriptive vs solve-mode keywords."""
    desc_kw = [
        "involves", "requires", "perform", "reasoning", "problem", "asks",
        "determine", "approach", "method", "step", "operation", "technique",
        "涉及", "需要", "执行", "推理", "问题", "要求", "方法", "步骤",
    ]
    solve_kw = [
        "= ", "answer is", "therefore", "thus", "so x", "the answer",
        "答案是", "所以", "因此", "等于",
    ]
    text_lower = text.lower()
    desc_hits = sum(1 for w in desc_kw if w.lower() in text_lower)
    solve_hits = sum(1 for w in solve_kw if w.lower() in text_lower)
    return desc_hits, solve_hits


def extract_steering_vector(model, tokenizer, device):
    """Extract per-layer describe-solve direction from calibration problems.

    delta^(l) = mean over (problems x languages) of:
        mean_tokens(h_desc^(l)) - mean_tokens(h_solve^(l))
    """
    print("Extracting steering vector from calibration problems...")
    delta = [torch.zeros(D_MODEL, device=device, dtype=torch.float32)
             for _ in range(N_LAYERS)]
    n_samples = 0

    for pi, prob in enumerate(CALIBRATION_PROBLEMS):
        for lang in ["en", "zh"]:
            solve_text = prob[lang]
            solve_inputs = tokenizer(solve_text, return_tensors="pt").to(device)
            with torch.no_grad():
                solve_out = model(**solve_inputs, output_hidden_states=True)

            desc_text = DESCRIBE_GENERIC[lang]
            desc_inputs = tokenizer(desc_text, return_tensors="pt").to(device)
            with torch.no_grad():
                desc_out = model(**desc_inputs, output_hidden_states=True)

            for ell in range(N_LAYERS):
                h_solve = solve_out.hidden_states[ell + 1].float().mean(dim=1).squeeze(0)
                h_desc = desc_out.hidden_states[ell + 1].float().mean(dim=1).squeeze(0)
                delta[ell] += (h_desc - h_solve)

            n_samples += 1
            print(f"  cal p{pi}_{lang}: solve_len={solve_inputs['input_ids'].shape[1]}, "
                  f"desc_len={desc_inputs['input_ids'].shape[1]}")

    for ell in range(N_LAYERS):
        delta[ell] /= n_samples

    norms = [delta[ell].norm().item() for ell in range(N_LAYERS)]
    print(f"\n  Steering norms (min/max/mean): "
          f"{min(norms):.1f} / {max(norms):.1f} / {sum(norms)/len(norms):.1f}")
    top5 = sorted(range(N_LAYERS), key=lambda i: -norms[i])[:5]
    print(f"  Peak norm layers: {top5} (norms: {[round(norms[i],1) for i in top5]})")

    delta_f16 = [d.half() for d in delta]
    return delta_f16, norms


def encode_raw(model, tokenizer, text, device):
    """Encode text WITHOUT chat template."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return {
        "past_kv": outputs.past_key_values,
        "logits": outputs.logits,
        "seq_len": inputs["input_ids"].shape[1],
    }


def build_hard_cache(math_kv, desc_kv, math_seq_len, desc_seq_len, L_star,
                     n_layers):
    """Build hybrid KV: math for layers 0..L*, describe for L*+1..N."""
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


def manual_generate(model, tokenizer, cache, first_logits, max_gen=MAX_GEN):
    """Standard generation without steering (for controls)."""
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


def generate_with_steering(model, tokenizer, cache, first_logits, delta, alpha,
                           max_gen=MAX_GEN):
    """Generate with math KV + residual stream steering.

    At each decoder layer l, adds alpha * delta^(l) to hidden state output.
    Hooks active only during generation, not encoding.
    """
    kv = copy.deepcopy(cache)

    hooks = []
    for ell in range(N_LAYERS):
        steer_vec = delta[ell]

        def make_hook(sv):
            def hook_fn(module, input, output):
                hs = output[0]
                hs_modified = hs + alpha * sv.unsqueeze(0).unsqueeze(0)
                return (hs_modified,) + output[1:]
            return hook_fn

        h = model.model.layers[ell].register_forward_hook(make_hook(steer_vec))
        hooks.append(h)

    try:
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
    finally:
        for h in hooks:
            h.remove()

    return tokenizer.decode(generated, skip_special_tokens=True)


def run_config(model, tokenizer, device, math_caches, desc_caches, problems,
               config_name, use_steering=False, alpha=None, delta=None,
               L_star=None):
    """Run one config across all problems and lang combos."""
    results = []
    for pi, prob in enumerate(problems):
        for math_lang, desc_lang in LANG_COMBOS:
            mc = math_caches[f"p{pi}_{math_lang}"]
            dc = desc_caches[desc_lang]

            if L_star is None:
                L_star_use = N_LAYERS - 1  # All math
            else:
                L_star_use = L_star

            hybrid = build_hard_cache(
                mc["past_kv"], dc["past_kv"],
                mc["seq_len"], dc["seq_len"],
                L_star=L_star_use, n_layers=N_LAYERS
            )

            if use_steering and delta is not None:
                output = generate_with_steering(
                    model, tokenizer, hybrid, dc["logits"],
                    delta, alpha, max_gen=MAX_GEN
                )
            else:
                output = manual_generate(
                    model, tokenizer, hybrid, dc["logits"],
                    max_gen=MAX_GEN
                )

            hits, total = contains_math_content(output, prob)
            desc_hits, solve_hits = is_descriptive(output)
            garbage = is_garbage(output)

            result = {
                "config": config_name,
                "problem_idx": pi,
                "category": prob["category"],
                "math_lang": math_lang,
                "desc_lang": desc_lang,
                "output": output[:800],
                "math_hits": hits,
                "math_total": total,
                "is_garbage": garbage,
                "has_answer": prob["answer"] in output,
                "desc_keywords": desc_hits,
                "solve_keywords": solve_hits,
            }
            results.append(result)

            tag = f"p{pi}_{math_lang}->{desc_lang}({prob['category']})"
            status = "GARBAGE" if garbage else f"h={hits}/{total}"
            print(f"    {tag} [{status}] ans={result['has_answer']} "
                  f"d={desc_hits} s={solve_hits}: {output[:120]}...")

    n = len(results)
    if n > 0:
        garb = sum(1 for r in results if r["is_garbage"])
        ans = sum(1 for r in results if r["has_answer"])
        desc_t = sum(r["desc_keywords"] for r in results)
        solve_t = sum(r["solve_keywords"] for r in results)
        avg_h = sum(r["math_hits"] for r in results) / n
        print(f"  >> {config_name}: garbage={garb}/{n}, has_answer={ans}/{n}, "
              f"avg_hits={avg_h:.1f}, desc_kw={desc_t}, solve_kw={solve_t}\n")

    return results


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"expG5: Two-Hijack Bun Inversion (14B)")
    print(f"{'='*70}\n")

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # ============================================================
    # STEP 1: Extract steering vector
    # ============================================================
    delta, delta_norms = extract_steering_vector(model, tokenizer, device)

    # ============================================================
    # STEP 2: Encode test prompts (raw, no chat template)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Encoding test prompts (raw, no chat template)")
    print(f"{'='*70}\n")

    desc_caches = {}
    for lang in ["en", "zh"]:
        desc_caches[lang] = encode_raw(model, tokenizer, DESCRIBE_GENERIC[lang], device)
        print(f"  Describe ({lang}): seq_len={desc_caches[lang]['seq_len']}")

    math_caches = {}
    for pi, prob in enumerate(PROBLEMS):
        for lang in ["en", "zh"]:
            key = f"p{pi}_{lang}"
            math_caches[key] = encode_raw(model, tokenizer, prob[lang], device)
            print(f"  Math {key}: seq_len={math_caches[key]['seq_len']}")
    print()

    all_results = {}

    # ============================================================
    # CONTROL 1: Pure F2 (all math KV, no steering) — should SOLVE
    # ============================================================
    print(f"{'='*70}")
    print(f"  CONTROL 1: Pure F2 (all math KV, no steering)")
    print(f"  Expected: model solves (no describe mode)")
    print(f"{'='*70}\n")

    all_results["f2_no_steer"] = run_config(
        model, tokenizer, device, math_caches, desc_caches, PROBLEMS,
        config_name="f2_no_steer", L_star=N_LAYERS - 1
    )

    # ============================================================
    # CONTROL 2: G1b at L*=41 (no steering) — baseline describe
    # ============================================================
    print(f"{'='*70}")
    print(f"  CONTROL 2: G1b Hard L*={L_STAR_CONTROL} (no steering)")
    print(f"  Expected: describes but may lose content")
    print(f"{'='*70}\n")

    all_results[f"g1b_L{L_STAR_CONTROL}"] = run_config(
        model, tokenizer, device, math_caches, desc_caches, PROBLEMS,
        config_name=f"g1b_L{L_STAR_CONTROL}", L_star=L_STAR_CONTROL
    )

    # ============================================================
    # G5: F2 + STEERING at multiple alpha values
    # ============================================================
    for alpha in ALPHA_SWEEP:
        print(f"{'='*70}")
        print(f"  G5: F2 + STEERING alpha={alpha}")
        print(f"{'='*70}\n")

        all_results[f"g5_alpha{alpha}"] = run_config(
            model, tokenizer, device, math_caches, desc_caches, PROBLEMS,
            config_name=f"g5_alpha{alpha}",
            use_steering=True, alpha=alpha, delta=delta,
            L_star=N_LAYERS - 1  # All math KV
        )

    # ============================================================
    # BONUS: Steering on G1b L*=41 (steering + partial describe KV)
    # ============================================================
    best_alpha_f2 = max(ALPHA_SWEEP, key=lambda a: (
        sum(r["math_hits"] + r["desc_keywords"] * 2
            for r in all_results.get(f"g5_alpha{a}", []))
        - sum(1 for r in all_results.get(f"g5_alpha{a}", []) if r["is_garbage"]) * 10
    ))
    print(f"{'='*70}")
    print(f"  BONUS: G1b L*={L_STAR_CONTROL} + steering alpha={best_alpha_f2}")
    print(f"  (best alpha from F2, applied to L*-split architecture)")
    print(f"{'='*70}\n")

    all_results[f"g1b_L{L_STAR_CONTROL}_steer{best_alpha_f2}"] = run_config(
        model, tokenizer, device, math_caches, desc_caches, PROBLEMS,
        config_name=f"g1b_L{L_STAR_CONTROL}_steer{best_alpha_f2}",
        use_steering=True, alpha=best_alpha_f2, delta=delta,
        L_star=L_STAR_CONTROL
    )

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  G5 TWO-HIJACK SUMMARY — {elapsed:.1f}s")
    print(f"{'='*70}\n")

    configs = (
        ["f2_no_steer", f"g1b_L{L_STAR_CONTROL}"]
        + [f"g5_alpha{a}" for a in ALPHA_SWEEP]
        + [f"g1b_L{L_STAR_CONTROL}_steer{best_alpha_f2}"]
    )

    print(f"  {'Config':>30} | {'garb':>5} | {'ans':>5} | {'hits':>5} | "
          f"{'desc':>5} | {'solve':>5}")
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")

    for cfg in configs:
        cr = all_results.get(cfg, [])
        if not cr:
            continue
        n = len(cr)
        garb = sum(1 for r in cr if r["is_garbage"])
        ans = sum(1 for r in cr if r["has_answer"])
        avg_h = sum(r["math_hits"] for r in cr) / n
        desc_kw = sum(r["desc_keywords"] for r in cr) / n
        solve_kw = sum(r["solve_keywords"] for r in cr) / n
        print(f"  {cfg:>30} | {garb:>2}/{n:<2} | {ans:>2}/{n:<2} | "
              f"{avg_h:>5.1f} | {desc_kw:>5.1f} | {solve_kw:>5.1f}")

    # Per-problem detail for best steered config
    best_alpha = max(ALPHA_SWEEP, key=lambda a: (
        sum(r["math_hits"] + r["desc_keywords"] * 2
            for r in all_results.get(f"g5_alpha{a}", []))
        - sum(1 for r in all_results.get(f"g5_alpha{a}", []) if r["is_garbage"]) * 10
    ))
    print(f"\n  BEST STEERED: alpha={best_alpha}")
    best_key = f"g5_alpha{best_alpha}"
    for pi, prob in enumerate(PROBLEMS):
        pr = [r for r in all_results[best_key] if r["problem_idx"] == pi]
        if pr:
            hits = sum(r["math_hits"] for r in pr) / len(pr)
            ans = sum(1 for r in pr if r["has_answer"])
            desc = sum(r["desc_keywords"] for r in pr) / len(pr)
            print(f"    p{pi}({prob['category']}): avg_hits={hits:.1f}, "
                  f"has_answer={ans}/{len(pr)}, desc_kw={desc:.1f}")
            for r in pr:
                print(f"      {r['math_lang']}->{r['desc_lang']}: "
                      f"{r['output'][:150]}...")

    # Steering vector profile
    print(f"\n  STEERING VECTOR NORM PROFILE:")
    for ell in range(0, N_LAYERS, 4):
        bar = "#" * int(delta_norms[ell] / max(delta_norms) * 40)
        print(f"    L{ell:>2}: {delta_norms[ell]:>7.1f} {bar}")

    # Save
    summary = {
        "model": MODEL_NAME,
        "n_layers": N_LAYERS,
        "d_model": D_MODEL,
        "elapsed_s": round(elapsed, 1),
        "alpha_sweep": ALPHA_SWEEP,
        "best_alpha": best_alpha,
        "best_alpha_f2": best_alpha_f2,
        "n_test_problems": len(PROBLEMS),
        "n_calibration_problems": len(CALIBRATION_PROBLEMS),
        "steering_vector_norms": delta_norms,
        "lang_combos": LANG_COMBOS,
    }
    all_results["summary"] = summary

    out_path = OUTPUT_DIR / "expG5_two_hijack.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
