"""expMS1b: Kernel surgery with robust e_c from 202 problems × 3 languages.

MS1 used 20 problems × 2 languages for e_c estimation. Web suggests:
better e_c should give equal or better results if the finding is real.

Uses SVD of language deviations (the proper AB-style computation) rather
than simple mean difference.
"""

import json, re, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_NEW = 128
LANGS = ["en", "zh", "es"]

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)


def get_test_problems():
    """The standard 20 test problems."""
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


def build_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{CHAT_SYSTEM}\n\nProblem: {problem_text}\n\nSolution:"


def check_answer(text, correct):
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)


def extract_last_hidden(model, tokenizer, problems, lang, device):
    """Get last-token hidden state at each layer for each problem."""
    all_acts = {L: [] for L in range(N_LAYERS)}

    class Cap:
        def __init__(self):
            self.out = None
        def __call__(self, module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                self.out = h[:, -1, :].detach().float().cpu().numpy()[0]
            else:
                self.out = h[-1, :].detach().float().cpu().numpy()

    caps = [Cap() for _ in range(N_LAYERS)]
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in range(N_LAYERS)]

    with torch.inference_mode():
        for prob in problems:
            text = prob.get(lang)
            if text is None:
                continue
            prompt = build_prompt(tokenizer, text)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            model(ids)
            for L in range(N_LAYERS):
                all_acts[L].append(caps[L].out.copy())

    for h in hooks:
        h.remove()
    return {L: np.array(all_acts[L]) for L in range(N_LAYERS)}


def compute_ec_svd(acts_by_lang, n_dims=1):
    """Compute e_c at each layer via SVD of per-problem language deviations (AB-style).

    For each problem, compute the mean across languages, then the deviation.
    SVD the deviations → top directions are the convention axes.
    """
    directions = {}
    var_explained = {}
    langs = list(acts_by_lang.keys())
    n_probs = acts_by_lang[langs[0]][0].shape[0]  # layer 0, first lang

    for L in range(N_LAYERS):
        # Stack: (n_problems, n_langs, d_model)
        stacked = np.stack([acts_by_lang[lang][L] for lang in langs], axis=1)
        # Per-problem mean across languages
        prob_means = stacked.mean(axis=1, keepdims=True)
        # Language deviations
        devs = (stacked - prob_means).reshape(-1, D_MODEL)
        # SVD
        _, S, Vt = np.linalg.svd(devs, full_matrices=False)
        directions[L] = Vt[:n_dims]  # (n_dims, d_model)
        var_explained[L] = float((S[:n_dims]**2).sum() / (S**2).sum())

    return directions, var_explained


def apply_surgery(model, directions, layers, device, n_dims=1):
    """Project out top n_dims convention directions from W_down at specified layers."""
    for L in layers:
        U = torch.tensor(directions[L][:n_dims].T, dtype=torch.float16, device=device)  # (d, n_dims)
        W = model.model.layers[L].mlp.down_proj.weight.data  # (d, m)
        # P = I - UU^T; W_new = P @ W = W - U(U^T @ W)
        proj = U.T @ W  # (n_dims, m)
        W.sub_(U @ proj)


def evaluate(model, tokenizer, problems, device):
    scores = {"en": 0, "zh": 0}
    for lang in ["en", "zh"]:
        for prob in problems:
            prompt = build_prompt(tokenizer, prob[lang])
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     temperature=None, top_p=None)
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            if check_answer(gen, prob["answer"]):
                scores[lang] += 1
    return scores


def main():
    device = "cuda"
    test_problems = get_test_problems()

    # Load expanded problems for e_c estimation
    with open("output/expanded_problems.json") as f:
        basis_problems = json.load(f)
    print(f"Basis: {len(basis_problems)} problems × {len(LANGS)} langs", flush=True)

    CONDITIONS = {
        "all_36": list(range(36)),
        "above_lc": list(range(13, 36)),
        "all_except_L5_L12": [L for L in range(36) if L not in (5, 12)],
    }

    print(f"{'=' * 60}")
    print(f"Exp MS1b: Robust Kernel Surgery (202 × 3 langs)")
    print(f"{'=' * 60}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Save original weights
    original_weights = {}
    for L in range(N_LAYERS):
        original_weights[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    # Extract activations for all 3 languages from basis problems
    acts_by_lang = {}
    for lang in LANGS:
        print(f"Extracting activations ({lang}, {len(basis_problems)} problems)...", flush=True)
        t_ext = time.time()
        acts_by_lang[lang] = extract_last_hidden(model, tokenizer, basis_problems, lang, device)
        print(f"  Done in {time.time() - t_ext:.1f}s", flush=True)

    # Compute e_c via SVD (AB-style)
    print("Computing convention directions (SVD of language deviations)...", flush=True)
    directions, var_explained = compute_ec_svd(acts_by_lang, n_dims=1)
    for L in [0, 5, 12, 15, 20, 25, 30, 35]:
        print(f"  L{L}: var_explained by e_c = {var_explained[L]:.3f}", flush=True)

    # Baseline
    print(f"\n--- Baseline ---", flush=True)
    t_bl = time.time()
    baseline = evaluate(model, tokenizer, test_problems, device)
    print(f"  EN={baseline['en']}/20, ZH={baseline['zh']}/20, "
          f"total={baseline['en']+baseline['zh']}/40  ({time.time()-t_bl:.0f}s)", flush=True)

    results = {"baseline": baseline}

    for cond_name, layers in CONDITIONS.items():
        # Restore
        for L in range(N_LAYERS):
            model.model.layers[L].mlp.down_proj.weight.data.copy_(original_weights[L])

        apply_surgery(model, directions, layers, device)

        print(f"\n--- {cond_name} ({len(layers)} layers) ---", flush=True)
        t_c = time.time()
        scores = evaluate(model, tokenizer, test_problems, device)
        bl = baseline["en"] + baseline["zh"]
        total = scores["en"] + scores["zh"]
        print(f"  EN={scores['en']}/20 ({scores['en']-baseline['en']:+d}), "
              f"ZH={scores['zh']}/20 ({scores['zh']-baseline['zh']:+d}), "
              f"total={total}/40 ({total-bl:+d})  ({time.time()-t_c:.0f}s)", flush=True)
        results[cond_name] = scores

    # Restore
    for L in range(N_LAYERS):
        model.model.layers[L].mlp.down_proj.weight.data.copy_(original_weights[L])

    # Summary
    bl = baseline["en"] + baseline["zh"]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY (MS1b vs MS1):")
    for name, scores in results.items():
        total = scores["en"] + scores["zh"]
        print(f"  {name:25s}: EN={scores['en']:2d}/20  ZH={scores['zh']:2d}/20  "
              f"total={total:2d}/40  ({total-bl:+d})")
    print(f"{'=' * 60}", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = {
        "experiment": "MS1b_robust_surgery",
        "model": MODEL_NAME,
        "n_basis_problems": len(basis_problems),
        "n_langs": len(LANGS),
        "n_test_problems": len(test_problems),
        "var_explained_by_ec": {str(L): var_explained[L] for L in range(N_LAYERS)},
        "results": results,
        "wall_time_s": time.time() - t0,
    }
    with open(OUTPUT_DIR / "expMS1b_robust_surgery.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: output/expMS1b_robust_surgery.json", flush=True)


if __name__ == "__main__":
    main()
