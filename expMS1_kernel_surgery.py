"""expMS1: Global kernel surgery pilot — remove convention direction from W_down at every layer.

Prediction from C7c:
  - L14-L29: safe (cos(v1_en,v1_zh) > 0.93, convention-invariant computation)
  - L30: safe (e_c ⊥ v1, cos=0.09)
  - L5, L12: might break (cos(v1_en,v1_zh) ≈ 0.03, languages diverge)
  - L0-L4, L6-L11: likely safe but less certain

Conditions:
  1. Baseline (no surgery)
  2. Surgery ALL 36 layers
  3. Surgery L0-L12 only (below l_c)
  4. Surgery L13-L35 only (above l_c)
  5. Surgery all EXCEPT L5 and L12 (predicted risk layers)
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


def check_answer(text, correct):
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)


def extract_encoding_activations(model, tokenizer, problems, lang, device):
    """Get last hidden state at each layer for each problem in the given language."""
    all_acts = {L: [] for L in range(N_LAYERS)}

    class LayerCapture:
        def __init__(self):
            self.out = None
        def __call__(self, module, input, output):
            # output may be tuple (hidden_states, ...) or just hidden_states
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
            prompt = build_prompt(tokenizer, prob[lang])
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
        # P = I - e_c e_c^T applied to W_down
        # W_down is (d_model, intermediate_size) = (2048, 11008)
        W = model.model.layers[L].mlp.down_proj.weight.data  # (2048, 11008)
        # Project out: W_new = W - e_c (e_c^T @ W)
        proj = e_c.unsqueeze(0) @ W  # (1, 11008)
        W.sub_(e_c.unsqueeze(1) @ proj)  # in-place: W -= e_c * (e_c^T @ W)


def evaluate(model, tokenizer, problems, device):
    """Run eval, return {en: n_correct, zh: n_correct}."""
    scores = {"en": 0, "zh": 0}
    for lang in ["en", "zh"]:
        for prob in problems:
            prompt = build_prompt(tokenizer, prob[lang])
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.inference_mode():
                out = model.generate(
                    ids, max_new_tokens=MAX_NEW, do_sample=False,
                    temperature=None, top_p=None,
                )
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            if check_answer(gen, prob["answer"]):
                scores[lang] += 1
    return scores


def main():
    device = "cuda"
    problems = get_test_problems()

    CONDITIONS = {
        "all_36": list(range(36)),
        "below_lc": list(range(0, 13)),        # L0-L12
        "above_lc": list(range(13, 36)),        # L13-L35
        "safe_zone": list(range(14, 30)),        # L14-L29 (high convention-invariance)
        "all_except_L5_L12": [L for L in range(36) if L not in (5, 12)],
    }

    print(f"{'=' * 60}")
    print(f"Exp MS1: Global Kernel Surgery Pilot")
    print(f"{'=' * 60}")
    print(f"Model:      {MODEL_NAME}")
    print(f"Problems:   {len(problems)}")
    print(f"Conditions: baseline + {list(CONDITIONS.keys())}")
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

    # Save original W_down weights for restoration
    print("Saving original weights...", flush=True)
    original_weights = {}
    for L in range(N_LAYERS):
        original_weights[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    # Extract encoding-time activations for e_c computation
    print("Extracting encoding activations (EN)...", flush=True)
    en_acts = extract_encoding_activations(model, tokenizer, problems, "en", device)
    print("Extracting encoding activations (ZH)...", flush=True)
    zh_acts = extract_encoding_activations(model, tokenizer, problems, "zh", device)

    # Compute convention direction at every layer
    print("Computing convention directions...", flush=True)
    directions = compute_convention_directions(en_acts, zh_acts)

    # Report cos(e_c, v1) would need C7c data — skip for now, C7c already measured it.

    # Baseline
    print("\n--- Baseline (no surgery) ---", flush=True)
    t_bl = time.time()
    baseline = evaluate(model, tokenizer, problems, device)
    print(f"  EN={baseline['en']}/20, ZH={baseline['zh']}/20, "
          f"total={baseline['en']+baseline['zh']}/40  ({time.time()-t_bl:.0f}s)", flush=True)

    # Run each condition
    results = {"baseline": baseline}

    for cond_name, layers in CONDITIONS.items():
        # Restore original weights
        for L in range(N_LAYERS):
            model.model.layers[L].mlp.down_proj.weight.data.copy_(original_weights[L])

        # Apply surgery
        apply_surgery(model, directions, layers, device)

        # Evaluate
        print(f"\n--- {cond_name} (surgery at {len(layers)} layers) ---", flush=True)
        t_c = time.time()
        scores = evaluate(model, tokenizer, problems, device)
        delta_en = scores["en"] - baseline["en"]
        delta_zh = scores["zh"] - baseline["zh"]
        delta_tot = (scores["en"] + scores["zh"]) - (baseline["en"] + baseline["zh"])
        print(f"  EN={scores['en']}/20 ({delta_en:+d}), ZH={scores['zh']}/20 ({delta_zh:+d}), "
              f"total={scores['en']+scores['zh']}/40 ({delta_tot:+d})  ({time.time()-t_c:.0f}s)", flush=True)
        results[cond_name] = scores

    # Restore weights
    for L in range(N_LAYERS):
        model.model.layers[L].mlp.down_proj.weight.data.copy_(original_weights[L])

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY:")
    bl = baseline["en"] + baseline["zh"]
    for name, scores in results.items():
        total = scores["en"] + scores["zh"]
        delta = total - bl
        print(f"  {name:25s}: EN={scores['en']:2d}/20  ZH={scores['zh']:2d}/20  "
              f"total={total:2d}/40  ({delta:+d})")
    print(f"{'=' * 60}", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = {
        "experiment": "MS1_kernel_surgery",
        "model": MODEL_NAME,
        "n_problems": len(problems),
        "conditions": {k: v for k, v in CONDITIONS.items()},
        "results": results,
        "wall_time_s": time.time() - t0,
    }
    out_file = OUTPUT_DIR / "expMS1_kernel_surgery.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
