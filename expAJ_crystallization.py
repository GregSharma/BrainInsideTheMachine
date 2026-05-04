"""
Exp AJ: Answer Crystallization Map

For each (layer L, generated token t), compute p(correct_answer_token | h_L,t).
This gives a 2D heatmap showing WHERE in the layer×token grid the answer emerges.

Three conditions: baseline ZH, flip ZH, baseline EN.

The diffusion analogy: each layer is one step of learned reverse diffusion.
The heatmap shows the denoising trajectory — when does the "image" (answer) emerge?
"""

import json
import numpy as np
import torch
import torch.nn.functional as F_torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SEED = 42
ALL_LAYERS = list(range(36))
STRIP_LAYERS = list(range(9, 27))
MAX_NEW_TOKENS = 80  # enough to capture answer, not too many


PROBLEMS = [
    {"en": "Calculate 47 + 86.", "zh": "计算 47 + 86 的值。", "answer": 133},
    {"en": "Calculate 15 × 8.", "zh": "计算 15 × 8 的值。", "answer": 120},
    {"en": "Find the value of C(10, 3).", "zh": "求组合数 C(10, 3) 的值。", "answer": 120},
    {"en": "What is the remainder when 100 is divided by 7?",
     "zh": "100 除以 7 的余数是多少？", "answer": 2},
    {"en": "A rectangle has length 12 and width 5. Find its area.",
     "zh": "一个长方形的长为 12，宽为 5，求其面积。", "answer": 60},
    {"en": "Calculate 256 + 789.", "zh": "计算 256 + 789 的值。", "answer": 1045},
    {"en": "Find the value of C(8, 2).", "zh": "求组合数 C(8, 2) 的值。", "answer": 28},
    {"en": "What is the remainder when 500 is divided by 13?",
     "zh": "500 除以 13 的余数是多少？", "answer": 6},
    {"en": "Calculate 64 × 15.", "zh": "计算 64 × 15 的值。", "answer": 960},
    {"en": "A rectangle has length 30 and width 18. Find its area.",
     "zh": "一个长方形的长为 30，宽为 18，求其面积。", "answer": 540},
]


def fit_1d_dirs(model, tokenizer):
    """Fit ZH-EN mean-difference direction per layer for STRIP_LAYERS."""
    rng = pyrandom.Random(SEED)
    problems = []
    for _ in range(40):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        problems.append({"zh": f"计算 {a} + {b} 的值。", "en": f"Calculate {a} + {b}."})
    for _ in range(40):
        n_val = rng.randint(5, 20); k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                         "en": f"Find the value of C({n_val}, {k_val})."})
    for _ in range(40):
        a = rng.randint(50, 9999); b_val = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b_val} 的余数是多少？",
                         "en": f"What is the remainder when {a} is divided by {b_val}?"})
    for _ in range(40):
        w = rng.randint(2, 50); h = rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                         "en": f"A rectangle has length {w} and width {h}. Find its area."})
    for _ in range(40):
        a1 = rng.randint(1, 20); d = rng.randint(1, 10); n_t = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d}，求前 {n_t} 项之和。",
                         "en": f"An arithmetic sequence: first term {a1}, common diff {d}. Sum of first {n_t} terms?"})
    rng.shuffle(problems)

    layer_acts = {l: {"zh": [], "en": []} for l in STRIP_LAYERS}
    layer_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in STRIP_LAYERS]
    try:
        for lang in ["zh", "en"]:
            for p in problems:
                inp = tokenizer(p[lang], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**inp)
                for l in STRIP_LAYERS:
                    layer_acts[l][lang].append(layer_out[l].copy())
                layer_out.clear()
    finally:
        for h in handles:
            h.remove()

    dirs = {}
    for l in STRIP_LAYERS:
        zh_m = np.mean(layer_acts[l]["zh"], axis=0)
        en_m = np.mean(layer_acts[l]["en"], axis=0)
        v = zh_m - en_m
        dirs[l] = torch.tensor(v / (np.linalg.norm(v) + 1e-8), dtype=torch.bfloat16)
    return dirs


def get_answer_token_ids(tokenizer, answer):
    """Get token IDs that represent the answer."""
    ans_str = str(answer)
    # Tokenize just the answer string in different contexts
    ids = set()
    for prefix in ["", " ", "\n"]:
        toks = tokenizer.encode(prefix + ans_str, add_special_tokens=False)
        ids.update(toks)
    # Also tokenize the bare number
    toks = tokenizer.encode(ans_str, add_special_tokens=False)
    ids.update(toks)
    return list(ids)


def generate_with_crystallization(model, tokenizer, prompt, answer, dirs_1d=None,
                                  scale=-1.0, max_new=MAX_NEW_TOKENS):
    """
    Generate token by token. At each step, extract hidden states at every layer
    and compute p(answer_token) via early exit (apply final layernorm + lm_head).

    Returns: (layer×token) grid of answer probabilities, plus generated text.
    """
    device = model.device
    answer_token_ids = get_answer_token_ids(tokenizer, answer)

    # We need the final layernorm and lm_head for early exit
    final_ln = model.model.norm
    lm_head = model.lm_head

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], device=device)
    prompt_len = input_ids.shape[1]

    # Storage: crystallization[t][l] = max p(answer_tok) across answer_token_ids
    crystallization = []
    generated_tokens = []

    # Set up MLP flip hooks if dirs provided
    flip_handles = []
    if dirs_1d is not None:
        def make_flip(l):
            v = dirs_1d[l].to(device)
            def hook(module, inp, out):
                h = out
                proj = torch.einsum("...d,d->...", h, v)
                h_new = h + scale * proj.unsqueeze(-1) * v
                return h_new
            return hook

        for l in STRIP_LAYERS:
            if l in dirs_1d:
                handle = model.model.layers[l].mlp.register_forward_hook(make_flip(l))
                flip_handles.append(handle)

    try:
        for t in range(max_new):
            # Capture hidden states at each layer
            layer_hiddens = {}

            def make_capture(l):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    # Only last token
                    layer_hiddens[l] = h[:, -1:, :].detach()
                return hook

            cap_handles = [model.model.layers[l].register_forward_hook(make_capture(l))
                           for l in ALL_LAYERS]

            with torch.no_grad():
                outputs = model(input_ids)

            for h in cap_handles:
                h.remove()

            # Get next token (greedy)
            next_logits = outputs.logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1)

            # Compute early-exit probabilities at each layer
            layer_probs = {}
            for l in ALL_LAYERS:
                h_l = layer_hiddens[l]  # (1, 1, 2048) bfloat16
                h_normed = final_ln(h_l)
                logits_l = lm_head(h_normed).float().squeeze(0).squeeze(0)  # (vocab_size,)
                probs = F_torch.softmax(logits_l, dim=-1)
                # Max probability across all answer token representations
                p_answer = max(probs[tid].item() for tid in answer_token_ids)
                layer_probs[l] = p_answer

            crystallization.append(layer_probs)
            generated_tokens.append(next_token.item())

            # Append and continue
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    finally:
        for h in flip_handles:
            h.remove()

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Convert to array: (n_tokens, 36)
    n_tok = len(crystallization)
    grid = np.zeros((n_tok, 36))
    for t_idx in range(n_tok):
        for l in ALL_LAYERS:
            grid[t_idx, l] = crystallization[t_idx].get(l, 0.0)

    return grid, gen_text, generated_tokens


def main():
    print("=== Exp AJ: Answer Crystallization Map ===\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    print("Fitting 1D language directions...")
    dirs = fit_1d_dirs(model, tokenizer)

    conditions = {
        "baseline_zh": {"lang": "zh", "dirs": None, "scale": 0},
        "flip_zh": {"lang": "zh", "dirs": dirs, "scale": -1.0},
        "baseline_en": {"lang": "en", "dirs": None, "scale": 0},
    }

    all_results = {}

    for cond_name, cond in conditions.items():
        print(f"\n=== Condition: {cond_name} ===")
        cond_results = []

        for pi, prob in enumerate(PROBLEMS):
            prompt = prob[cond["lang"]]
            answer = prob["answer"]

            grid, gen_text, gen_toks = generate_with_crystallization(
                model, tokenizer, prompt, answer,
                dirs_1d=cond["dirs"], scale=cond["scale"],
                max_new=MAX_NEW_TOKENS
            )

            # Find crystallization point: first (token, layer) where p > 0.1
            cryst_tok, cryst_layer = -1, -1
            for t_idx in range(grid.shape[0]):
                for l in range(36):
                    if grid[t_idx, l] > 0.1:
                        cryst_tok, cryst_layer = t_idx, l
                        break
                if cryst_tok >= 0:
                    break

            # Max probability and its location
            if grid.size > 0:
                max_idx = np.unravel_index(grid.argmax(), grid.shape)
                max_p = grid[max_idx]
            else:
                max_idx = (-1, -1)
                max_p = 0.0

            # Answer found in text?
            ans_str = str(answer)
            answer_found = ans_str in gen_text
            answer_token_pos = -1
            for i, tok_id in enumerate(gen_toks):
                tok_str = tokenizer.decode([tok_id])
                if ans_str in tok_str or tok_str.strip() == ans_str:
                    answer_token_pos = i
                    break

            result = {
                "problem_idx": pi,
                "prompt": prompt[:50],
                "answer": answer,
                "gen_text": gen_text[:200],
                "n_tokens": grid.shape[0],
                "answer_found": answer_found,
                "answer_token_pos": answer_token_pos,
                "crystallization_token": cryst_tok,
                "crystallization_layer": cryst_layer,
                "max_p": round(float(max_p), 4),
                "max_p_location": [int(max_idx[0]), int(max_idx[1])],
                # Layer profile at first token
                "layer_profile_t0": [round(float(grid[0, l]), 6) for l in range(36)]
                    if grid.shape[0] > 0 else [],
                # Layer profile at answer token (if found)
                "layer_profile_answer": [round(float(grid[answer_token_pos, l]), 6)
                                         for l in range(36)]
                    if answer_token_pos >= 0 and answer_token_pos < grid.shape[0] else [],
                # Token profile at L30 (concentration layer) and L35 (final)
                "token_profile_L30": [round(float(grid[t, 30]), 6)
                                      for t in range(min(grid.shape[0], 40))],
                "token_profile_L35": [round(float(grid[t, 35]), 6)
                                      for t in range(min(grid.shape[0], 40))],
            }

            cond_results.append(result)
            print(f"  P{pi}: cryst=({cryst_tok},{cryst_layer}) max_p={max_p:.4f} "
                  f"at ({max_idx[0]},{max_idx[1]}) ans_tok={answer_token_pos} "
                  f"found={answer_found}")

        all_results[cond_name] = cond_results

    # Aggregate: mean crystallization token per condition
    print("\n=== SUMMARY ===")
    for cond_name in conditions:
        results = all_results[cond_name]
        cryst_toks = [r["crystallization_token"] for r in results if r["crystallization_token"] >= 0]
        ans_toks = [r["answer_token_pos"] for r in results if r["answer_token_pos"] >= 0]
        n_found = sum(1 for r in results if r["answer_found"])

        print(f"{cond_name}:")
        print(f"  Answers found: {n_found}/{len(results)}")
        if cryst_toks:
            print(f"  Mean crystallization token: {np.mean(cryst_toks):.1f} "
                  f"(earliest: {min(cryst_toks)}, latest: {max(cryst_toks)})")
        if ans_toks:
            print(f"  Mean answer token position: {np.mean(ans_toks):.1f}")

    # Key comparison: does flip shift crystallization earlier?
    bl_cryst = [r["crystallization_token"] for r in all_results["baseline_zh"]
                if r["crystallization_token"] >= 0]
    fl_cryst = [r["crystallization_token"] for r in all_results["flip_zh"]
                if r["crystallization_token"] >= 0]
    if bl_cryst and fl_cryst:
        print(f"\n  Baseline ZH mean crystallization: {np.mean(bl_cryst):.1f}")
        print(f"  Flip ZH mean crystallization:     {np.mean(fl_cryst):.1f}")
        print(f"  Shift: {np.mean(bl_cryst) - np.mean(fl_cryst):.1f} tokens earlier with flip")

    # Does crystallization happen along layers (vertical) or tokens (horizontal)?
    # Check: at crystallization token, which layers have p > 0.05?
    print("\n=== CRYSTALLIZATION STRUCTURE ===")
    for cond_name in ["baseline_zh", "flip_zh"]:
        results = all_results[cond_name]
        print(f"\n{cond_name} — Layer profile at t=0 (before any generation):")
        t0_profiles = [r["layer_profile_t0"] for r in results if r["layer_profile_t0"]]
        if t0_profiles:
            mean_t0 = np.mean(t0_profiles, axis=0)
            for l in range(36):
                bar = "#" * int(mean_t0[l] * 200)
                if mean_t0[l] > 0.001:
                    print(f"  L{l:>2}: {mean_t0[l]:.6f} {bar}")

    output = {
        "experiment": "AJ_crystallization",
        "method": "Per (layer, token) early-exit p(answer). 2D heatmap of answer emergence.",
        "n_problems": len(PROBLEMS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "conditions": all_results,
    }

    out_path = OUTPUT_DIR / "expAJ_crystallization.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
