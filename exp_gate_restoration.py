"""Exp GATE-R: Clipped-Feature Restoration — The Silent Columns of W_down

The MLP gate (SiLU) clips features below threshold. The clipped features
are computed (W_up · x exists) but multiplied by ~zero and discarded.
W_down's corresponding columns never contribute to the residual stream.

This experiment restores α of the clipped contribution: the features the
gate decided to suppress, projected back through W_down.

Design:
  - α sweep: 0.01, 0.05, 0.1, 0.2, 0.5
  - Controls: random features, below convention boundary, no temporal window
  - All on instruct model with 20 math problems × 2 languages

The 2×2 factorial with QK deflation is deferred to a separate script
once we know whether gate restoration has any signal at all.
"""

import json
import time
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True

# === Config ===
MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_NEW = 512
SEED = 42
DEVICE = "cuda"
LANGS = ["en", "zh"]

# Gate restoration params
ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]
GATE_THRESHOLD = 0.1  # |SiLU(z)| below this = "clipped"
RESTORATION_LAYERS = list(range(13, 36))  # above convention boundary l_c
PRIMING_WINDOW = 50  # only restore during first 50 tokens

CHAT_SYSTEM = ("You are a careful mathematical reasoner. "
               "Show your work step by step.")


def generate_problems():
    """20 math problems, 5 categories, bilingual."""
    return [
        {"en": "Find the value of C(17, 1).", "zh": "求组合数 C(17, 1) 的值。",
         "answer": "17", "category": "combinatorics"},
        {"en": "Find the value of C(5, 1).", "zh": "求组合数 C(5, 1) 的值。",
         "answer": "5", "category": "combinatorics"},
        {"en": "Find the value of C(10, 2).", "zh": "求组合数 C(10, 2) 的值。",
         "answer": "45", "category": "combinatorics"},
        {"en": "Find the value of C(8, 3).", "zh": "求组合数 C(8, 3) 的值。",
         "answer": "56", "category": "combinatorics"},
        {"en": "Solve for x: 3x + 7 = 22", "zh": "求解：3x + 7 = 22",
         "answer": "5", "category": "algebra"},
        {"en": "Solve for x: 2x² - 8 = 0", "zh": "求解：2x² - 8 = 0",
         "answer": "2", "category": "algebra"},
        {"en": "Simplify: (x + 3)(x - 3)", "zh": "化简：(x + 3)(x - 3)",
         "answer": "x² - 9", "category": "algebra"},
        {"en": "Solve: |2x - 5| = 3", "zh": "求解：|2x - 5| = 3",
         "answer": "4", "category": "algebra"},
        {"en": "Calculate: 347 + 658", "zh": "计算：347 + 658",
         "answer": "1005", "category": "arithmetic"},
        {"en": "Calculate: 1000 - 387", "zh": "计算：1000 - 387",
         "answer": "613", "category": "arithmetic"},
        {"en": "Calculate: 23 × 17", "zh": "计算：23 × 17",
         "answer": "391", "category": "arithmetic"},
        {"en": "Calculate: 1728 ÷ 12", "zh": "计算：1728 ÷ 12",
         "answer": "144", "category": "arithmetic"},
        {"en": "Find the area of a circle with radius 7 (use π ≈ 22/7).",
         "zh": "求半径为7的圆的面积（使用 π ≈ 22/7）。",
         "answer": "154", "category": "geometry"},
        {"en": "A right triangle has legs 5 and 12. Find the hypotenuse.",
         "zh": "求直角三角形两直角边为5和12时的斜边长",
         "answer": "13", "category": "geometry"},
        {"en": "Find the perimeter of a rectangle with length 15 and width 8.",
         "zh": "长为15宽为8的矩形的周长是多少？",
         "answer": "46", "category": "geometry"},
        {"en": "Find the volume of a cube with side length 6.",
         "zh": "求边长为6的正方体的体积",
         "answer": "216", "category": "geometry"},
        {"en": "What is the remainder when 17 is divided by 5?",
         "zh": "17 除以 5 的余数是多少？",
         "answer": "2", "category": "modular"},
        {"en": "What is the remainder when 100 is divided by 7?",
         "zh": "100 除以 7 的余数是多少？",
         "answer": "2", "category": "modular"},
        {"en": "What is the remainder when 256 is divided by 10?",
         "zh": "256 除以 10 的余数是多少？",
         "answer": "6", "category": "modular"},
        {"en": "What is the remainder when 1000 is divided by 13?",
         "zh": "1000 除以 13 的余数是多少？",
         "answer": "12", "category": "modular"},
    ]


def build_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def check_answer(text, correct):
    """Check if correct answer appears in the LAST 200 chars of output."""
    tail = text[-200:] if len(text) > 200 else text
    correct_nums = re.findall(r'-?\d+\.?\d*', str(correct))
    if not correct_nums:
        return correct.lower() in tail.lower()
    target = correct_nums[0]
    # Check if target appears as a standalone number in the tail
    numbers_in_tail = re.findall(r'-?\d+\.?\d*', tail)
    return target in numbers_in_tail


# === Gate Restoration Hook ===
class GateRestorationHook:
    """Restore clipped MLP features at specified layers.

    Intervention: W_down · (clip_mask ⊙ up_proj)
    This gives voice to the up-projection values that the gate silenced.
    It is NOT the same as recovering the gated intermediate (gate * up);
    it deliberately bypasses the gate to project raw up values through W_down.
    """
    def __init__(self, alpha, threshold, active_layers, priming_window,
                 mode="clipped"):
        self.alpha = alpha
        self.threshold = threshold
        self.active_layers = set(active_layers)
        self.priming_window = priming_window
        self.mode = mode  # "clipped", "random"
        self.step = 0
        self.active = True
        # Accumulate stats across all problems (not reset per problem)
        self.all_clipped_fracs = []
        self.total_interventions = 0

    def make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if not self.active:
                return output
            if layer_idx not in self.active_layers:
                return output
            if self.step >= self.priming_window:
                return output

            x = input[0]
            # Only intervene on single-token steps (generation, not prefill)
            if x.shape[1] > 1:
                return output

            # Recompute gate and up projections
            with torch.no_grad():
                gate_proj = module.gate_proj(x)
                up_proj = module.up_proj(x)
                gate_values = F.silu(gate_proj)

                # Identify clipped features
                clip_mask = (gate_values.abs() < self.threshold).to(
                    gate_values.dtype)
                clipped_frac = clip_mask.mean().item()
                self.all_clipped_fracs.append(clipped_frac)

                if self.mode == "clipped":
                    masked_up = clip_mask * up_proj
                elif self.mode == "random":
                    rand_mask = torch.zeros_like(clip_mask)
                    n_clipped = int(clip_mask.sum().item())
                    if n_clipped > 0:
                        indices = torch.randperm(
                            clip_mask.shape[-1],
                            device=clip_mask.device)[:n_clipped]
                        rand_mask[..., indices] = 1.0
                    masked_up = rand_mask * up_proj
                else:
                    return output

                clipped_output = module.down_proj(masked_up)
                self.total_interventions += 1

            # MLP output is a plain tensor in Qwen2
            return output + self.alpha * clipped_output

        return hook_fn

    def increment_step(self):
        self.step += 1

    def reset_step(self):
        """Reset step counter for new generation. Does NOT reset stats."""
        self.step = 0

    def get_stats(self):
        if not self.all_clipped_fracs:
            return {}
        return {
            "total_interventions": self.total_interventions,
            "mean_clipped_frac": float(np.mean(self.all_clipped_fracs)),
            "std_clipped_frac": float(np.std(self.all_clipped_fracs)),
            "n_samples": len(self.all_clipped_fracs),
        }


# === Manual Generation ===
def generate_manual(model, tokenizer, prompt_text, device,
                    gate_hook=None, max_new=MAX_NEW):
    """Manual token-by-token generation with hook support."""
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    generated = []
    past_kv = None

    if gate_hook:
        gate_hook.reset_step()

    for step in range(max_new):
        with torch.no_grad():
            if past_kv is None:
                out = model(input_ids, use_cache=True)
            else:
                out = model(
                    input_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True
                )

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            tid = next_token.item()
            generated.append(tid)

            if gate_hook:
                gate_hook.increment_step()

            if tid in (151643, 151645):  # Qwen EOS tokens
                break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return text


def run_condition(model, tokenizer, problems, device, label, gate_hook=None):
    """Run one condition across all problems."""
    handles = []

    if gate_hook:
        for layer_idx in gate_hook.active_layers:
            mlp_module = model.model.layers[layer_idx].mlp
            h = mlp_module.register_forward_hook(gate_hook.make_hook(layer_idx))
            handles.append(h)

    results = []
    for pi, prob in enumerate(problems):
        for lang in LANGS:
            prompt = build_prompt(tokenizer, prob[lang])
            text = generate_manual(
                model, tokenizer, prompt, device, gate_hook=gate_hook)
            correct = check_answer(text, prob["answer"])
            results.append({
                "pi": pi, "lang": lang, "cat": prob["category"],
                "correct": correct, "text": text[:300]
            })

    for h in handles:
        h.remove()

    n_correct = sum(r["correct"] for r in results)
    en = sum(r["correct"] for r in results if r["lang"] == "en")
    zh = sum(r["correct"] for r in results if r["lang"] == "zh")
    total = len(results)

    gate_stats = gate_hook.get_stats() if gate_hook else {}
    clip_str = f"  clip={gate_stats.get('mean_clipped_frac', 0):.3f}" if gate_stats else ""
    print(f"  {label}: {n_correct}/{total} (EN={en}, ZH={zh}){clip_str}")

    return {
        "label": label,
        "correct": n_correct, "total": total,
        "en": en, "zh": zh,
        "gate_stats": gate_stats,
        "details": results
    }


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Exp GATE-R: Clipped-Feature Restoration")
    print("  The silent columns of W_down speak.")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Model loaded: {MODEL_NAME}")

    problems = generate_problems()
    print(f"  {len(problems)} problems × {len(LANGS)} langs = "
          f"{len(problems)*len(LANGS)} evals per condition")

    all_results = {}

    # === 1. Baseline ===
    print("\n--- Baseline ---")
    all_results["baseline"] = run_condition(
        model, tokenizer, problems, DEVICE, "baseline")

    # === 2. Gate restoration α sweep (L13-35, t<50, clipped features) ===
    for alpha in ALPHA_VALUES:
        print(f"\n--- Gate Restoration α={alpha} ---")
        hook = GateRestorationHook(
            alpha=alpha, threshold=GATE_THRESHOLD,
            active_layers=RESTORATION_LAYERS,
            priming_window=PRIMING_WINDOW, mode="clipped")
        all_results[f"gate_a{alpha}"] = run_condition(
            model, tokenizer, problems, DEVICE,
            f"gate_a{alpha}", gate_hook=hook)

    # === 3. Random control (same α as middle of sweep) ===
    print("\n--- Random Features Control (α=0.1) ---")
    hook = GateRestorationHook(
        alpha=0.1, threshold=GATE_THRESHOLD,
        active_layers=RESTORATION_LAYERS,
        priming_window=PRIMING_WINDOW, mode="random")
    all_results["random_control"] = run_condition(
        model, tokenizer, problems, DEVICE,
        "random_control", gate_hook=hook)

    # === 4. Below convention boundary (L0-L12) ===
    print("\n--- Below Convention Boundary (L0-L12, α=0.1) ---")
    hook = GateRestorationHook(
        alpha=0.1, threshold=GATE_THRESHOLD,
        active_layers=list(range(0, 13)),
        priming_window=PRIMING_WINDOW, mode="clipped")
    all_results["below_lc"] = run_condition(
        model, tokenizer, problems, DEVICE,
        "below_lc", gate_hook=hook)

    # === 5. No temporal window (all generation steps, not just t<50) ===
    print("\n--- No Window Limit (α=0.1, all steps) ---")
    hook = GateRestorationHook(
        alpha=0.1, threshold=GATE_THRESHOLD,
        active_layers=RESTORATION_LAYERS,
        priming_window=999999, mode="clipped")
    all_results["no_window"] = run_condition(
        model, tokenizer, problems, DEVICE,
        "no_window", gate_hook=hook)

    # === Summary ===
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SUMMARY (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"{'Condition':>25s}  {'Score':>7s}  {'EN':>4s}  {'ZH':>4s}  "
          f"{'ClipFrac':>8s}")
    for key, res in all_results.items():
        cf = res['gate_stats'].get('mean_clipped_frac', 0)
        print(f"{key:>25s}  {res['correct']:>3d}/{res['total']:<3d}  "
              f"{res['en']:>4d}  {res['zh']:>4d}  {cf:>8.3f}")

    # Save
    output_path = Path("output/exp_gate_restoration.json")
    output_path.parent.mkdir(exist_ok=True)
    save_data = {
        "experiment": "GATE-R: Clipped-Feature Restoration",
        "model": MODEL_NAME,
        "alpha_values": ALPHA_VALUES,
        "gate_threshold": GATE_THRESHOLD,
        "restoration_layers": RESTORATION_LAYERS,
        "priming_window": PRIMING_WINDOW,
        "results": {k: {kk: vv for kk, vv in v.items()}
                    for k, v in all_results.items()},
        "wall_time_s": elapsed,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
