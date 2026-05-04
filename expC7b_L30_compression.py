"""expC7b: Targeted MLP compression at L30 — the 4D bottleneck.

C7 found L30 has r90=4 (cross-problem) for MLP output during generation.
This tests whether projecting L30 MLP output onto top-k during generation
preserves math accuracy.

Design:
  Phase 1: Collect MLP outputs at L30 during generation of 10 basis problems.
  Phase 2: Build SVD basis from collected outputs.
  Phase 3: Re-run all 20 problems with L30 MLP output projected onto top-k.
  Sweep k = [1, 2, 4, 8, 16, 32, 64, 256, 1024].
"""

import json, re, time, argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_NEW = 128
TARGET_LAYER = 30

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)

K_VALUES = [1, 2, 4, 8, 16, 32, 64, 256, 1024]


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


def check_answer(text, correct):
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)


class MLPCaptureHook:
    """Captures MLP output at last token during generation."""
    def __init__(self):
        self.captured = []
        self.active = False

    def __call__(self, module, input, output):
        if not self.active:
            return
        if output.shape[1] == 1:
            self.captured.append(output[0, 0].float().cpu().numpy())

    def reset(self):
        self.captured = []


class MLPCompressionHook:
    """Projects MLP output onto top-k affine subspace during generation."""
    def __init__(self, mean_vec, basis_vecs, k, device):
        self.mean = torch.tensor(mean_vec, dtype=torch.float32, device=device)
        self.basis = torch.tensor(basis_vecs[:k], dtype=torch.float32, device=device)
        self.active = False

    def __call__(self, module, input, output):
        if not self.active:
            return output
        if output.shape[1] != 1:
            return output  # prompt pass-through
        x = output[0, 0].float()
        centered = x - self.mean
        coeffs = centered @ self.basis.T
        projected = self.mean + coeffs @ self.basis
        new_out = projected.to(output.dtype).unsqueeze(0).unsqueeze(0)
        # Replace only the last-token MLP output
        return new_out


def generate(model, tokenizer, prompt_text, device, max_new=MAX_NEW):
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    generated_ids = []
    past_kv = None
    cur_input = ids
    with torch.inference_mode():
        for _ in range(max_new):
            out = model(cur_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax().item())
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            cur_input = torch.tensor([[next_id]], device=device)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    args = parser.parse_args()

    device = "cuda"
    target_layer = args.layer
    problems = get_test_problems()
    langs = ["en", "zh"]

    # Stratified split: 2 per category for basis, 2 per test
    basis_problems = []
    test_problems = []
    cats = {}
    for p in problems:
        cats.setdefault(p["category"], []).append(p)
    for cat, probs in cats.items():
        basis_problems.extend(probs[:2])
        test_problems.extend(probs[2:])

    if args.dry:
        basis_problems = basis_problems[:2]
        test_problems = test_problems[:2]
        K_VALUES_USE = [1, 4, 32]
    else:
        K_VALUES_USE = K_VALUES

    print(f"{'=' * 60}")
    print(f"Exp C7b: Targeted MLP Compression at L{target_layer}")
    print(f"{'=' * 60}")
    print(f"Model:       {MODEL_NAME}")
    print(f"Target:      L{target_layer} MLP output")
    print(f"Basis probs: {len(basis_problems)} (stratified 2/category)")
    print(f"Test probs:  {len(test_problems)}")
    print(f"k-values:    {K_VALUES_USE}")
    print(flush=True)

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # ── Phase 1: Collect MLP outputs at target layer ──
    print(f"\nPhase 1: Collecting L{target_layer} MLP outputs from {len(basis_problems)} basis problems...", flush=True)
    capture = MLPCaptureHook()
    h = model.model.layers[target_layer].mlp.register_forward_hook(capture)

    for pi, prob in enumerate(basis_problems):
        for lang in langs:
            capture.reset()
            capture.active = True
            prompt_text = build_prompt(tokenizer, prob[lang])
            generate(model, tokenizer, prompt_text, device)
            capture.active = False

    h.remove()

    all_vecs = np.array(capture.captured)
    print(f"  Collected {all_vecs.shape[0]} vectors at L{target_layer}", flush=True)

    # ── Phase 2: Build SVD basis ──
    mean_vec = all_vecs.mean(axis=0)
    centered = all_vecs - mean_vec
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    def eff_rank(S, t):
        v = S**2
        return int(np.searchsorted(np.cumsum(v) / v.sum(), t) + 1)

    print(f"  Basis SVD: r50={eff_rank(S, 0.5)}, r90={eff_rank(S, 0.9)}, "
          f"r95={eff_rank(S, 0.95)}, sv1/sv2={S[0]/S[1]:.1f}", flush=True)

    # ── Phase 3: Baseline (no hooks) ──
    print(f"\nPhase 3: Baseline on {len(test_problems)} test problems...", flush=True)
    baseline = {"en": 0, "zh": 0}
    for prob in test_problems:
        for lang in langs:
            text = generate(model, tokenizer, build_prompt(tokenizer, prob[lang]), device)
            if check_answer(text, prob["answer"]):
                baseline[lang] += 1
    n_test = len(test_problems)
    print(f"  Baseline: EN={baseline['en']}/{n_test}, ZH={baseline['zh']}/{n_test}", flush=True)

    # ── Phase 4: Compression sweep ──
    print(f"\nPhase 4: Compression sweep at L{target_layer}...", flush=True)
    results_by_k = {}

    for k in K_VALUES_USE:
        if k > Vt.shape[0]:
            k = Vt.shape[0]

        comp_hook = MLPCompressionHook(mean_vec, Vt, k, device)
        h = model.model.layers[target_layer].mlp.register_forward_hook(comp_hook)

        scores = {"en": 0, "zh": 0}
        for prob in test_problems:
            for lang in langs:
                comp_hook.active = True
                text = generate(model, tokenizer, build_prompt(tokenizer, prob[lang]), device)
                comp_hook.active = False
                if check_answer(text, prob["answer"]):
                    scores[lang] += 1

        h.remove()

        total = scores["en"] + scores["zh"]
        bl_total = baseline["en"] + baseline["zh"]
        print(f"  k={k:4d}: EN={scores['en']}/{n_test}, ZH={scores['zh']}/{n_test}, "
              f"total={total}/{2*n_test} (baseline {bl_total}/{2*n_test})", flush=True)

        results_by_k[k] = {
            "en": scores["en"],
            "zh": scores["zh"],
            "total": total,
        }

    # ── Save results ──
    wall_time = time.time() - t0
    results = {
        "experiment": "C7b_L30_compression",
        "model": MODEL_NAME,
        "target_layer": target_layer,
        "n_basis": len(basis_problems),
        "n_test": len(test_problems),
        "n_basis_vectors": int(all_vecs.shape[0]),
        "basis_svd": {
            "r50": eff_rank(S, 0.5),
            "r90": eff_rank(S, 0.9),
            "r95": eff_rank(S, 0.95),
            "top10_sv": S[:10].tolist(),
        },
        "baseline": baseline,
        "results_by_k": {str(k): v for k, v in results_by_k.items()},
        "wall_time_s": wall_time,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / f"expC7b_L{target_layer}_compression.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file} ({wall_time:.0f}s wall time)", flush=True)


if __name__ == "__main__":
    main()
