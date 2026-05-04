"""expMS1d: System prompt confound control for kernel surgery.

Tests whether MS1's +6 at 512tok survives under three system prompt conditions:
  1. en_only:  English system prompt for both EN and ZH problems (original MS1)
  2. matched:  English sys for EN, Chinese sys for ZH
  3. none:     No system prompt (user message only in chat template)

Crossed with: baseline (no surgery) vs above_lc (L13-L35 surgery).
Total: 3 × 2 × 40 = 240 evals.
"""
import json, re, time, copy
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_NEW = 512

SYSTEM_EN = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)
SYSTEM_ZH = (
    "你是一个严谨的数学推理者。遇到问题时，请逐步思考，清晰地展示你的推导过程，"
    "然后在单独的一行给出最终的数值答案。"
)


def get_test_problems():
    categories = {
        "algebra": [
            {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22", "answer": "5"},
            {"en": "Solve for x: 2x² - 8 = 0", "zh": "求解x：2x² - 8 = 0", "answer": "2"},
            {"en": "Simplify: (x + 3)(x - 3)", "zh": "化简：(x + 3)(x - 3)", "answer": "x² - 9"},
            {"en": "Solve: |2x - 5| = 3", "zh": "求解：|2x - 5| = 3", "answer": "4"},
        ],
        "arithmetic": [
            {"en": "Calculate: 347 + 658", "zh": "计算：347 + 658", "answer": "1005"},
            {"en": "Calculate: 1000 - 387", "zh": "计算：1000 - 387", "answer": "613"},
            {"en": "Calculate: 23 × 17", "zh": "计算：23 × 17", "answer": "391"},
            {"en": "Calculate: 1728 ÷ 12", "zh": "计算：1728 ÷ 12", "answer": "144"},
        ],
        "geometry": [
            {"en": "Find the area of a circle with radius 7 (use π ≈ 22/7)", "zh": "求半径为7的圆的面积（使用 π ≈ 22/7）", "answer": "154"},
            {"en": "Find the hypotenuse of a right triangle with legs 5 and 12", "zh": "求直角三角形两直角边为5和12时的斜边长", "answer": "13"},
            {"en": "What is the perimeter of a rectangle with length 15 and width 8?", "zh": "长为15宽为8的矩形的周长是多少？", "answer": "46"},
            {"en": "Find the volume of a cube with side length 6", "zh": "求边长为6的正方体的体积", "answer": "216"},
        ],
        "number_theory": [
            {"en": "What is the GCD of 84 and 120?", "zh": "84和120的最大公约数是多少？", "answer": "12"},
            {"en": "Is 97 prime? Answer yes or no, then explain.", "zh": "97是质数吗？回答是或否，然后解释。", "answer": "yes"},
            {"en": "Find the remainder when 2^10 is divided by 7", "zh": "求2^10除以7的余数", "answer": "2"},
            {"en": "What is the sum of all prime numbers less than 20?", "zh": "所有小于20的质数之和是多少？", "answer": "77"},
        ],
        "combinatorics": [
            {"en": "How many ways can you choose 3 items from 7?", "zh": "从7个物品中选3个有多少种方式？", "answer": "35"},
            {"en": "How many ways can 5 people stand in a line?", "zh": "5个人站成一排有多少种方式？", "answer": "120"},
            {"en": "Calculate: 8! / (5! × 3!)", "zh": "计算：8! / (5! × 3!)", "answer": "56"},
            {"en": "How many 3-digit numbers have all distinct digits?", "zh": "有多少个三位数的各位数字互不相同？", "answer": "648"},
        ],
    }
    problems = []
    for cat_name, cat_probs in categories.items():
        for p in cat_probs:
            p["category"] = cat_name
            problems.append(p)
    return problems


def build_prompt(tokenizer, problem_text, sys_mode, lang):
    """Build prompt with specified system prompt mode.

    sys_mode: "en_only" | "matched" | "none"
    lang: "en" | "zh" (used for matched mode)
    """
    if sys_mode == "none":
        messages = [{"role": "user", "content": problem_text}]
    elif sys_mode == "matched":
        sys_content = SYSTEM_ZH if lang == "zh" else SYSTEM_EN
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": problem_text},
        ]
    else:  # en_only (original)
        messages = [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": problem_text},
        ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if sys_mode == "none":
            return f"Problem: {problem_text}\n\nSolution:"
        sys = SYSTEM_EN if sys_mode == "en_only" else (SYSTEM_ZH if lang == "zh" else SYSTEM_EN)
        return f"{sys}\n\nProblem: {problem_text}\n\nSolution:"


def check_answer(text, correct):
    if correct in ("yes", "no"):
        return correct.lower() in text.lower()
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)


def extract_encoding_activations(model, tokenizer, problems, lang, sys_mode, device):
    """Get last hidden state at each layer for each problem."""
    all_acts = {L: [] for L in range(N_LAYERS)}

    class LayerCapture:
        def __init__(self):
            self.out = None
        def __call__(self, module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                self.out = h[:, -1, :].detach().float().cpu().numpy()
            else:
                self.out = h[-1:, :].detach().float().cpu().numpy()

    captures = [LayerCapture() for _ in range(N_LAYERS)]
    hooks = []
    for L in range(N_LAYERS):
        h = model.model.layers[L].register_forward_hook(captures[L])
        hooks.append(h)

    with torch.inference_mode():
        for prob in problems:
            prompt = build_prompt(tokenizer, prob[lang], sys_mode, lang)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            model(ids)
            for L in range(N_LAYERS):
                all_acts[L].append(captures[L].out[0])

    for h in hooks:
        h.remove()
    return {L: np.array(all_acts[L]) for L in range(N_LAYERS)}


def compute_convention_directions(en_acts, zh_acts):
    """Compute e_c at each layer as normalized mean(zh) - mean(en)."""
    directions = {}
    for L in range(N_LAYERS):
        diff = zh_acts[L].mean(axis=0) - en_acts[L].mean(axis=0)
        norm = np.linalg.norm(diff)
        directions[L] = diff / (norm + 1e-12)
    return directions


def apply_surgery(model, directions, layers_to_modify, device):
    """Project out e_c from W_down at specified layers. Modifies model IN PLACE."""
    for L in layers_to_modify:
        e_c = torch.tensor(directions[L], dtype=torch.float16, device=device)
        W = model.model.layers[L].mlp.down_proj.weight.data  # (2048, 11008)
        proj = e_c.unsqueeze(0) @ W  # (1, 11008)
        W.sub_(e_c.unsqueeze(1) @ proj)


def evaluate(model, tokenizer, problems, sys_mode, device):
    """Run eval with specified system prompt mode."""
    results = {"en": [], "zh": []}
    scores = {"en": 0, "zh": 0}
    for lang in ["en", "zh"]:
        for prob in problems:
            prompt = build_prompt(tokenizer, prob[lang], sys_mode, lang)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.inference_mode():
                out = model.generate(
                    ids, max_new_tokens=MAX_NEW, do_sample=False,
                    temperature=None, top_p=None,
                )
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            correct = check_answer(gen, prob["answer"])
            if correct:
                scores[lang] += 1
            results[lang].append({
                "problem": prob[lang],
                "answer": prob["answer"],
                "category": prob["category"],
                "correct": correct,
                "gen_tokens": out.shape[1] - ids.shape[1],
                "output_preview": gen[:200],
            })
    return scores, results


def main():
    device = "cuda"
    problems = get_test_problems()
    above_lc = list(range(13, 36))

    SYS_MODES = ["en_only", "matched", "none"]
    SURGERY_MODES = ["baseline", "above_lc"]

    print(f"{'=' * 70}")
    print(f"Exp MS1d: System Prompt Confound Control")
    print(f"{'=' * 70}")
    print(f"Model:        {MODEL_NAME}")
    print(f"Problems:     {len(problems)} × 2 langs = {len(problems)*2} evals per cell")
    print(f"Sys modes:    {SYS_MODES}")
    print(f"Surgery:      {SURGERY_MODES}")
    print(f"Max tokens:   {MAX_NEW}")
    print(f"Total evals:  {len(SYS_MODES) * len(SURGERY_MODES) * len(problems) * 2}")
    print(flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Save original weights
    print("Saving original weights...", flush=True)
    original_weights = {}
    for L in range(N_LAYERS):
        original_weights[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    # For each sys_mode, we need a separate e_c (convention direction depends on prompt format)
    all_directions = {}
    for sys_mode in SYS_MODES:
        print(f"\nExtracting encoding activations for sys_mode={sys_mode}...", flush=True)
        en_acts = extract_encoding_activations(model, tokenizer, problems, "en", sys_mode, device)
        zh_acts = extract_encoding_activations(model, tokenizer, problems, "zh", sys_mode, device)
        dirs = compute_convention_directions(en_acts, zh_acts)
        all_directions[sys_mode] = dirs

        # Report cos between this e_c and en_only e_c
        if sys_mode != "en_only" and "en_only" in all_directions:
            cos_vals = []
            for L in range(N_LAYERS):
                c = np.dot(all_directions["en_only"][L], dirs[L])
                cos_vals.append(c)
            print(f"  cos(e_c_{sys_mode}, e_c_en_only): "
                  f"mean={np.mean(cos_vals):.3f}, min={np.min(cos_vals):.3f}, "
                  f"max={np.max(cos_vals):.3f}", flush=True)

    # Run all 6 cells
    all_results = {}

    for sys_mode in SYS_MODES:
        for surgery_mode in SURGERY_MODES:
            cell_name = f"{sys_mode}__{surgery_mode}"

            # Restore original weights
            for L in range(N_LAYERS):
                model.model.layers[L].mlp.down_proj.weight.data.copy_(original_weights[L])

            # Apply surgery if needed
            if surgery_mode == "above_lc":
                apply_surgery(model, all_directions[sys_mode], above_lc, device)

            # Evaluate
            print(f"\n--- {cell_name} ---", flush=True)
            t_c = time.time()
            scores, details = evaluate(model, tokenizer, problems, sys_mode, device)

            avg_tok_en = np.mean([d["gen_tokens"] for d in details["en"]])
            avg_tok_zh = np.mean([d["gen_tokens"] for d in details["zh"]])

            print(f"  EN={scores['en']}/20, ZH={scores['zh']}/20, "
                  f"total={scores['en']+scores['zh']}/40  "
                  f"(avg_tok: EN={avg_tok_en:.0f}, ZH={avg_tok_zh:.0f})  "
                  f"({time.time()-t_c:.0f}s)", flush=True)

            all_results[cell_name] = {
                "scores": scores,
                "details": {lang: details[lang] for lang in ["en", "zh"]},
                "avg_tokens": {"en": float(avg_tok_en), "zh": float(avg_tok_zh)},
            }

    # Restore weights
    for L in range(N_LAYERS):
        model.model.layers[L].mlp.down_proj.weight.data.copy_(original_weights[L])

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"SUMMARY (MS1d System Prompt Control)")
    print(f"{'=' * 70}")
    print(f"{'Cell':30s} {'EN':>6s} {'ZH':>6s} {'Total':>7s} {'Delta':>7s} {'AvgTok':>10s}")
    print(f"{'-' * 70}")

    for sys_mode in SYS_MODES:
        bl_key = f"{sys_mode}__baseline"
        sg_key = f"{sys_mode}__above_lc"
        bl = all_results[bl_key]["scores"]
        sg = all_results[sg_key]["scores"]
        bl_tot = bl["en"] + bl["zh"]
        sg_tot = sg["en"] + sg["zh"]
        delta = sg_tot - bl_tot

        bl_tok = f"{all_results[bl_key]['avg_tokens']['en']:.0f}/{all_results[bl_key]['avg_tokens']['zh']:.0f}"
        sg_tok = f"{all_results[sg_key]['avg_tokens']['en']:.0f}/{all_results[sg_key]['avg_tokens']['zh']:.0f}"

        print(f"  {bl_key:30s} {bl['en']:2d}/20  {bl['zh']:2d}/20  {bl_tot:2d}/40")
        print(f"  {sg_key:30s} {sg['en']:2d}/20  {sg['zh']:2d}/20  {sg_tot:2d}/40  {delta:+d}      {sg_tok}")
        print()

    # Key question: does the surgery delta persist across sys modes?
    print(f"\nSURGERY EFFECT BY SYSTEM PROMPT MODE:")
    for sys_mode in SYS_MODES:
        bl = all_results[f"{sys_mode}__baseline"]["scores"]
        sg = all_results[f"{sys_mode}__above_lc"]["scores"]
        delta_en = sg["en"] - bl["en"]
        delta_zh = sg["zh"] - bl["zh"]
        delta_tot = (sg["en"] + sg["zh"]) - (bl["en"] + bl["zh"])
        print(f"  {sys_mode:10s}: EN {delta_en:+d}, ZH {delta_zh:+d}, total {delta_tot:+d}")
    print(f"{'=' * 70}", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Also save e_c cosine similarities between modes
    ec_cosines = {}
    for sm1 in SYS_MODES:
        for sm2 in SYS_MODES:
            if sm1 >= sm2:
                continue
            key = f"{sm1}_vs_{sm2}"
            ec_cosines[key] = {
                str(L): float(np.dot(all_directions[sm1][L], all_directions[sm2][L]))
                for L in range(N_LAYERS)
            }

    out = {
        "experiment": "MS1d_sysprompt_control",
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW,
        "n_problems": len(problems),
        "sys_modes": SYS_MODES,
        "surgery_layers": {"above_lc": above_lc},
        "results": all_results,
        "ec_cosines_between_modes": ec_cosines,
        "wall_time_s": time.time() - t0,
    }
    out_file = OUTPUT_DIR / "expMS1d_sysprompt_control.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
