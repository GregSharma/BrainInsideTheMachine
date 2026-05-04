"""expC7c: Anatomize L30's rank-1 direction.

Key questions:
  1. What IS v1 at L30? cos with convention direction e_c, language dir, mean.
  2. Is L30 convention-invariant? Do en-only and zh-only v1 align?
  3. If e_c ⊥ v1, kernel surgery at L30 is automatically safe.

Also characterizes v1 at every layer for context (full trajectory of cos(v1, e_c)).
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

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
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


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def eff_rank(S, t):
    v = S ** 2
    return int(np.searchsorted(np.cumsum(v) / v.sum(), t) + 1)


def main():
    device = "cuda"
    problems = get_test_problems()
    langs = ["en", "zh"]

    print(f"{'=' * 60}")
    print(f"Exp C7c: L30 Direction Anatomy")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_NAME}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Collect MLP outputs at ALL layers during generation, separated by language
    captures = {lang: {L: [] for L in range(N_LAYERS)} for lang in langs}

    class MultiLayerCapture:
        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.active = False
            self.current_lang = None

        def __call__(self, module, input, output):
            if not self.active:
                return
            if output.shape[1] == 1:
                vec = output[0, 0].float().cpu().numpy()
                captures[self.current_lang][self.layer_idx].append(vec)

    hooks = []
    hook_objs = []
    for L in range(N_LAYERS):
        ch = MultiLayerCapture(L)
        h = model.model.layers[L].mlp.register_forward_hook(ch)
        hook_objs.append(ch)
        hooks.append(h)

    print(f"\nCollecting MLP outputs from {len(problems)} problems × {len(langs)} langs...", flush=True)
    t_gen = time.time()

    for pi, prob in enumerate(problems):
        for lang in langs:
            for ch in hook_objs:
                ch.active = True
                ch.current_lang = lang

            prompt_text = build_prompt(tokenizer, prob[lang])
            ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            generated_ids = []
            past_kv = None
            cur_input = ids

            with torch.inference_mode():
                for _ in range(MAX_NEW):
                    out = model(cur_input, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    next_id = int(out.logits[0, -1].argmax().item())
                    generated_ids.append(next_id)
                    if next_id == tokenizer.eos_token_id:
                        break
                    cur_input = torch.tensor([[next_id]], device=device)

            for ch in hook_objs:
                ch.active = False

            if (pi * 2 + langs.index(lang) + 1) % 10 == 0:
                done = pi * 2 + langs.index(lang) + 1
                n_steps = len(captures[lang][30])
                print(f"  [{done}/40] prob {pi} {lang}: L30 has {n_steps} vecs so far", flush=True)

    for h in hooks:
        h.remove()

    gen_time = time.time() - t_gen
    n_en = len(captures["en"][30])
    n_zh = len(captures["zh"][30])
    print(f"  Generation: {gen_time:.1f}s. EN: {n_en} vecs, ZH: {n_zh} vecs.", flush=True)

    # Analysis at every layer
    print(f"\nAnalyzing all {N_LAYERS} layers...", flush=True)
    results_per_layer = {}

    # Get unembedding for token identification
    W_U = model.model.embed_tokens.weight.detach().float().cpu().numpy()  # (vocab, d_model) for tied

    for L in range(N_LAYERS):
        en_vecs = np.array(captures["en"][L])
        zh_vecs = np.array(captures["zh"][L])
        all_vecs = np.vstack([en_vecs, zh_vecs])

        # Convention direction: mean(zh) - mean(en), normalized
        e_c = zh_vecs.mean(axis=0) - en_vecs.mean(axis=0)
        e_c_norm = e_c / (np.linalg.norm(e_c) + 1e-12)

        # SVD of all (centered)
        mean_all = all_vecs.mean(axis=0)
        centered = all_vecs - mean_all
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        v1 = Vt[0]

        # SVD of en-only and zh-only (centered separately)
        en_c = en_vecs - en_vecs.mean(axis=0)
        _, S_en, Vt_en = np.linalg.svd(en_c, full_matrices=False)
        v1_en = Vt_en[0]

        zh_c = zh_vecs - zh_vecs.mean(axis=0)
        _, S_zh, Vt_zh = np.linalg.svd(zh_c, full_matrices=False)
        v1_zh = Vt_zh[0]

        # Key cosines
        cos_v1_ec = cos(v1, e_c_norm)
        cos_v1_mean = cos(v1, mean_all)
        cos_v1en_v1zh = abs(cos(v1_en, v1_zh))  # abs because sign is arbitrary in SVD
        cos_v1en_ec = cos(v1_en, e_c_norm)
        cos_v1zh_ec = cos(v1_zh, e_c_norm)

        # How much variance does e_c explain?
        proj_on_ec = centered @ e_c_norm
        var_ec = np.var(proj_on_ec)
        var_total = np.sum(S ** 2) / len(all_vecs)
        frac_ec = var_ec / (var_total + 1e-12)

        # Top unembed tokens for v1 (what does this direction map to in vocab?)
        logits_v1 = W_U @ v1
        top_ids = np.argsort(logits_v1)[-5:][::-1]
        bot_ids = np.argsort(logits_v1)[:5]
        top_tokens = [tokenizer.decode([int(i)]) for i in top_ids]
        bot_tokens = [tokenizer.decode([int(i)]) for i in bot_ids]

        r = {
            "r50": eff_rank(S, 0.5),
            "r90": eff_rank(S, 0.9),
            "sv1_sv2": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
            "cos_v1_ec": cos_v1_ec,
            "cos_v1_mean": cos_v1_mean,
            "cos_v1en_v1zh": cos_v1en_v1zh,
            "cos_v1en_ec": cos_v1en_ec,
            "cos_v1zh_ec": cos_v1zh_ec,
            "frac_var_ec": float(frac_ec),
            "top_tokens_v1": top_tokens,
            "bot_tokens_v1": bot_tokens,
        }
        results_per_layer[L] = r

        flag = " <<<" if L == 30 else ""
        print(f"  L{L:2d}: cos(v1,e_c)={cos_v1_ec:+.3f}  cos(v1_en,v1_zh)={cos_v1en_v1zh:.3f}  "
              f"cos(v1,mean)={cos_v1_mean:+.3f}  var(e_c)={frac_ec:.3f}  "
              f"sv1/2={S[0]/S[1]:.1f}  r90={eff_rank(S, 0.9)}{flag}", flush=True)

    # Summary for L30
    L30 = results_per_layer[30]
    print(f"\n{'=' * 60}")
    print(f"L30 ANATOMY:")
    print(f"  cos(v1, e_c)       = {L30['cos_v1_ec']:+.4f}")
    print(f"  cos(v1_en, v1_zh)  = {L30['cos_v1en_v1zh']:.4f}  (convention-invariance)")
    print(f"  cos(v1, mean)      = {L30['cos_v1_mean']:+.4f}")
    print(f"  frac_var(e_c)      = {L30['frac_var_ec']:.4f}")
    print(f"  sv1/sv2            = {L30['sv1_sv2']:.1f}")
    print(f"  top tokens (v1)    = {L30['top_tokens_v1']}")
    print(f"  bottom tokens (v1) = {L30['bot_tokens_v1']}")
    print(f"{'=' * 60}", flush=True)

    # Convention direction trajectory
    ec_trajectory = [results_per_layer[L]["cos_v1_ec"] for L in range(N_LAYERS)]
    inv_trajectory = [results_per_layer[L]["cos_v1en_v1zh"] for L in range(N_LAYERS)]
    print(f"\ncos(v1, e_c) trajectory:      {[f'{x:+.3f}' for x in ec_trajectory]}")
    print(f"cos(v1_en, v1_zh) trajectory: {[f'{x:.3f}' for x in inv_trajectory]}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {
        "experiment": "C7c_L30_anatomy",
        "model": MODEL_NAME,
        "n_problems": len(problems),
        "n_en_vecs": n_en,
        "n_zh_vecs": n_zh,
        "per_layer": {str(L): results_per_layer[L] for L in range(N_LAYERS)},
        "cos_v1_ec_trajectory": ec_trajectory,
        "convention_invariance_trajectory": inv_trajectory,
        "wall_time_s": time.time() - t0,
    }
    out_file = OUTPUT_DIR / "expC7c_L30_anatomy.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
