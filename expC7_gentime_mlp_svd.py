"""expC7_gentime_mlp_svd.py — Measure Z_comp: generation-time MLP output effective rank.

The GATE causal failure (0/20 at all k) showed encoding-time centroid SVD is the wrong
basis for generation-time compression. This experiment measures the RIGHT object: what
subspace do MLP outputs actually occupy during generation?

Design:
  - 20 test problems × 2 languages × up to 128 generation steps
  - Hook every MLP layer, capture output at last token during each gen step
  - Center, SVD, report effective rank per layer
  - Compare with C3's attention-output rank (was rank-1 at 7B)

If Z_comp is low-rank, the MLP width m can be compressed.
If it's full-rank, the compression thesis needs a different angle.
"""

import json
import re
import time
import argparse
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
    """Standard 20 test problems (first 4 per category)."""
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


class MLPCaptureHook:
    """Captures MLP output at last token during each generation step."""

    def __init__(self):
        self.captured = []
        self.active = False

    def __call__(self, module, input, output):
        if not self.active:
            return
        # MLP output is (batch, seq, d_model)
        if output.shape[1] == 1:  # generation step only
            self.captured.append(output[0, 0].float().cpu().numpy())

    def reset(self):
        self.captured = []


def effective_rank(S, threshold):
    """Number of singular values needed to explain `threshold` fraction of total variance."""
    var = S ** 2
    cumvar = np.cumsum(var) / var.sum()
    return int(np.searchsorted(cumvar, threshold) + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="3 problems, 64 tokens")
    args = parser.parse_args()

    device = "cuda"
    max_new = 64 if args.dry else MAX_NEW
    problems = get_test_problems()
    if args.dry:
        problems = problems[:3]
    langs = ["en", "zh"]

    print(f"{'=' * 60}")
    print(f"Exp C7: Generation-Time MLP Output SVD")
    print(f"{'=' * 60}")
    print(f"Model:      {MODEL_NAME}")
    print(f"Problems:   {len(problems)} × {len(langs)} langs")
    print(f"Max tokens: {max_new}")
    print(f"Layers:     all {N_LAYERS}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Register capture hooks on all MLP layers
    hooks = []
    capture_hooks = []
    for L in range(N_LAYERS):
        ch = MLPCaptureHook()
        h = model.model.layers[L].mlp.register_forward_hook(ch)
        capture_hooks.append(ch)
        hooks.append(h)

    # Collect MLP outputs during generation
    all_outputs = {L: [] for L in range(N_LAYERS)}  # layer -> list of vectors
    per_problem_outputs = {}  # (prob_idx, lang) -> {layer: list of vectors}
    baseline_correct = {"en": 0, "zh": 0}

    print("\nGenerating and capturing MLP outputs...")
    t_gen = time.time()

    for pi, prob in enumerate(problems):
        for lang in langs:
            prompt_text = build_prompt(tokenizer, prob[lang])
            ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            generated_ids = []
            past_kv = None
            cur_input = ids

            # Reset all capture hooks
            for ch in capture_hooks:
                ch.reset()
                ch.active = True

            with torch.inference_mode():
                for step in range(max_new):
                    out = model(cur_input, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    next_id = int(out.logits[0, -1].argmax().item())
                    generated_ids.append(next_id)
                    if next_id == tokenizer.eos_token_id:
                        break
                    cur_input = torch.tensor([[next_id]], device=device)

            # Deactivate hooks
            for ch in capture_hooks:
                ch.active = False

            # Check answer
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            correct = check_answer(gen_text, prob["answer"])
            if correct:
                baseline_correct[lang] += 1

            # Store per-problem captures
            n_steps = len(capture_hooks[0].captured)
            per_problem_outputs[(pi, lang)] = {}
            for L in range(N_LAYERS):
                vecs = np.array(capture_hooks[L].captured)  # (n_steps, d_model)
                all_outputs[L].append(vecs)
                per_problem_outputs[(pi, lang)][L] = vecs

            if (pi * len(langs) + langs.index(lang) + 1) % 5 == 0:
                done = pi * len(langs) + langs.index(lang) + 1
                total = len(problems) * len(langs)
                print(f"  [{done}/{total}] prob {pi} {lang}: {n_steps} steps, "
                      f"{'CORRECT' if correct else 'wrong'}")

    gen_time = time.time() - t_gen
    print(f"\nGeneration: {gen_time:.1f}s")
    print(f"Baseline: EN={baseline_correct['en']}/{len(problems)}, "
          f"ZH={baseline_correct['zh']}/{len(problems)}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # SVD analysis per layer
    print("\nComputing SVD per layer...")
    results_per_layer = {}

    for L in range(N_LAYERS):
        # Stack all vectors for this layer
        M = np.vstack(all_outputs[L])  # (total_steps, d_model)
        n_samples = M.shape[0]

        # Raw stats
        raw_mean = np.mean(M, axis=0)
        raw_norms = np.linalg.norm(M, axis=1)

        # Center
        Mc = M - M.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)

        r50 = effective_rank(S, 0.5)
        r90 = effective_rank(S, 0.9)
        r95 = effective_rank(S, 0.95)
        r99 = effective_rank(S, 0.99)

        # Per-problem effective rank at this layer
        per_prob_r90 = []
        for key, layer_dict in per_problem_outputs.items():
            vecs = layer_dict[L]
            if vecs.shape[0] < 3:
                continue
            vc = vecs - vecs.mean(axis=0, keepdims=True)
            _, Sp, _ = np.linalg.svd(vc, full_matrices=False)
            per_prob_r90.append(effective_rank(Sp, 0.9))

        # Cosine of MLP output with its mean (how constant is it?)
        cos_with_mean = []
        mn = raw_mean / (np.linalg.norm(raw_mean) + 1e-12)
        for row in M:
            c = np.dot(row / (np.linalg.norm(row) + 1e-12), mn)
            cos_with_mean.append(float(c))

        results_per_layer[L] = {
            "n_samples": n_samples,
            "r50": r50,
            "r90": r90,
            "r95": r95,
            "r99": r99,
            "top10_sv": S[:10].tolist(),
            "sv_ratio_1_2": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
            "per_problem_r90_mean": float(np.mean(per_prob_r90)),
            "per_problem_r90_median": float(np.median(per_prob_r90)),
            "per_problem_r90_max": float(np.max(per_prob_r90)),
            "mean_cos_with_mean": float(np.mean(cos_with_mean)),
            "mean_norm": float(np.mean(raw_norms)),
        }

        print(f"  L{L:2d}: r50={r50:3d}  r90={r90:3d}  r95={r95:3d}  r99={r99:3d}  "
              f"sv1/sv2={S[0]/S[1]:.1f}  per_prob_r90={np.mean(per_prob_r90):.1f}  "
              f"cos_mean={np.mean(cos_with_mean):.3f}")

    # Summary
    r90_trajectory = [results_per_layer[L]["r90"] for L in range(N_LAYERS)]
    per_prob_trajectory = [results_per_layer[L]["per_problem_r90_mean"] for L in range(N_LAYERS)]

    print(f"\n{'=' * 60}")
    print(f"CROSS-PROBLEM r90 trajectory: {r90_trajectory}")
    print(f"PER-PROBLEM r90 trajectory:   {[f'{x:.1f}' for x in per_prob_trajectory]}")
    print(f"Baseline: EN={baseline_correct['en']}/{len(problems)}, "
          f"ZH={baseline_correct['zh']}/{len(problems)}")

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {
        "experiment": "C7_gentime_mlp_svd",
        "model": MODEL_NAME,
        "n_problems": len(problems),
        "n_langs": len(langs),
        "max_new_tokens": max_new,
        "baseline": baseline_correct,
        "per_layer": {str(L): results_per_layer[L] for L in range(N_LAYERS)},
        "r90_trajectory": r90_trajectory,
        "per_problem_r90_trajectory": per_prob_trajectory,
        "wall_time_s": time.time() - t0,
        "generation_time_s": gen_time,
    }

    out_file = OUTPUT_DIR / "expC7_gentime_mlp_svd.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
