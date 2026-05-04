"""Experiment P2: Causal Language Direction Strip from MLP Deltas

P showed that PCA-derived language PCs don't capture the causal language signal
in MLP deltas. The variance-maximizing directions aren't the operative ones.

New approach: compute the MEAN DIFFERENCE between zh and en MLP deltas at each
layer. This difference vector IS the causal language direction (by construction).
Then project it out.

Also try: instead of projecting out, REPLACE the MLP delta's projection onto
the mean-difference direction with zero (single-direction strip) vs with the
opposite language's projection (cross-lingual swap along the causal direction).

10 problems × 2 languages. 128 tokens.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
STRIP_LAYERS = list(range(9, 27))
N_TRAIN = 200
d = model.config.hidden_size


def generate_pca_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh, en = f"计算 {a} + {b} 的值。", f"Calculate {a} + {b}."
        else:
            zh, en = f"计算 {a} × {b} 的值。", f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                          "en": f"Find the value of C({n_val}, {k_val})."})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？",
                          "en": f"What is the remainder when {a} is divided by {b}?"})
    for _ in range(per_cat):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                          "en": f"A rectangle has length {w} and width {h}. Find its area."})
    for _ in range(per_cat):
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。",
                          "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_terms} terms."})
    return problems


# =============================================================================
# Step 1: Compute per-layer MEAN DIFFERENCE direction for MLP deltas
# =============================================================================
print("=" * 70)
print("COMPUTING CAUSAL LANGUAGE DIRECTIONS FROM MLP DELTAS")
print("=" * 70)

problems = generate_pca_problems(N_TRAIN, seed=42)

# For efficiency, fit at a few representative layers
fit_layers = [10, 14, 18, 22, 26]
lang_dirs = {}  # layer -> unit direction vector (torch tensor on GPU)

for fl in fit_layers:
    mlp_out = {}
    def mlp_cap(module, input, output):
        mlp_out['d'] = output.detach()[:, -1, :]
    handle = model.model.layers[fl].mlp.register_forward_hook(mlp_cap)

    zh_deltas = np.zeros((N_TRAIN, d), dtype=np.float32)
    en_deltas = np.zeros((N_TRAIN, d), dtype=np.float32)

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        zh_deltas[i] = mlp_out['d'].cpu().float().numpy()
        mlp_out.clear()

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        en_deltas[i] = mlp_out['d'].cpu().float().numpy()
        mlp_out.clear()

    handle.remove()

    # Mean difference: zh - en
    diff = zh_deltas.mean(axis=0) - en_deltas.mean(axis=0)
    diff_norm = np.linalg.norm(diff)
    diff_unit = diff / diff_norm

    # Check: how much of MLP delta variance is along this direction?
    zh_proj = zh_deltas @ diff_unit
    en_proj = en_deltas @ diff_unit
    cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt((zh_proj.std()**2 + en_proj.std()**2) / 2)

    # Fraction of total MLP delta norm in this direction
    zh_frac = np.abs(zh_proj).mean() / np.linalg.norm(zh_deltas, axis=1).mean()
    en_frac = np.abs(en_proj).mean() / np.linalg.norm(en_deltas, axis=1).mean()

    print(f"  L{fl}: diff_norm={diff_norm:.1f}, Cohen's d={cohens_d:.1f}, "
          f"zh_frac={zh_frac:.1%}, en_frac={en_frac:.1%}")

    lang_dirs[fl] = torch.tensor(diff_unit, dtype=torch.float32, device=device)

# Map each strip layer to nearest fit layer
def get_lang_dir(layer_idx):
    nearest = min(fit_layers, key=lambda x: abs(x - layer_idx))
    return lang_dirs[nearest]

# Precompute for all strip layers
strip_dirs = {li: get_lang_dir(li) for li in STRIP_LAYERS}


# =============================================================================
# Test problems
# =============================================================================
test_problems = [
    {"prompt_en": "Calculate 47 + 86.", "prompt_zh": "计算 47 + 86 的值。", "answer": "133", "difficulty": "simple"},
    {"prompt_en": "A rectangle has length 12 and width 5. Find its area.", "prompt_zh": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60", "difficulty": "simple"},
    {"prompt_en": "What is the remainder when 100 is divided by 7?", "prompt_zh": "100 除以 7 的余数是多少？", "answer": "2", "difficulty": "simple"},
    {"prompt_en": "Calculate 15 × 8.", "prompt_zh": "计算 15 × 8 的值。", "answer": "120", "difficulty": "simple"},
    {"prompt_en": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.",
     "prompt_zh": "等差数列首项为 2，公差为 3，求前 5 项之和。", "answer": "40", "difficulty": "simple"},
    {"prompt_en": "Calculate 387 × 29.", "prompt_zh": "计算 387 × 29 的值。", "answer": "11223", "difficulty": "hard"},
    {"prompt_en": "Find the value of C(10, 3).", "prompt_zh": "求组合数 C(10, 3) 的值。", "answer": "120", "difficulty": "hard"},
    {"prompt_en": "What is the remainder when 7654 is divided by 37?", "prompt_zh": "7654 除以 37 的余数是多少？", "answer": "34", "difficulty": "hard"},
    {"prompt_en": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.",
     "prompt_zh": "等差数列首项为 7，公差为 11，求前 25 项之和。", "answer": "3475", "difficulty": "hard"},
    {"prompt_en": "A rectangle has length 47 and width 33. Find its area.", "prompt_zh": "一个长方形的长为 47，宽为 33，求其面积。", "answer": "1551", "difficulty": "hard"},
]


def run_generation(prompt, strip_mode=None, strip_scale=1.0):
    """Run generation with optional causal language stripping from MLP deltas.

    strip_mode:
        None = baseline
        "zero" = project out the causal language direction from MLP delta
        "flip" = flip the sign of the language component (zh→en or en→zh)
        "amplify" = amplify the language component by strip_scale
    """
    input_ids = tokenizer.encode(prompt)
    handles = []

    if strip_mode:
        for li in STRIP_LAYERS:
            def make_hook(layer_idx, lang_dir, mode, scale):
                def hook_fn(module, input, output):
                    delta = output.float()  # (batch, seq, d)
                    # Project delta onto language direction
                    proj = torch.sum(delta * lang_dir, dim=-1, keepdim=True)  # (batch, seq, 1)
                    lang_component = proj * lang_dir  # (batch, seq, d)

                    if mode == "zero":
                        stripped = delta - lang_component
                    elif mode == "flip":
                        stripped = delta - 2 * lang_component  # flip sign
                    elif mode == "amplify":
                        stripped = delta + (scale - 1) * lang_component
                    else:
                        stripped = delta

                    return stripped.to(output.dtype)
                return hook_fn
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_hook(li, strip_dirs[li], strip_mode, strip_scale)
                )
            )

    try:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]

        for _ in range(MAX_NEW_TOKENS - 1):
            with torch.no_grad():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)
            if next_id == tokenizer.eos_token_id:
                break
    finally:
        for h in handles:
            h.remove()

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def classify_lang(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = sum(1 for c in text if c.isalpha() and c.isascii())
    if zh > en * 2:
        return "zh"
    elif en > zh * 2:
        return "en"
    return "mixed"


# =============================================================================
# Run experiments
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT P2: CAUSAL LANGUAGE DIRECTION STRIP FROM MLP DELTAS")
print("=" * 70)

modes = [
    (None, 1.0, "baseline"),
    ("zero", 1.0, "zero_lang_dir"),
    ("flip", 1.0, "flip_lang_dir"),
    ("amplify", 2.0, "amplify_2x"),
]

results = {"experiment": "P2: Causal Language Strip", "modes": {}}

for strip_mode, scale, mode_label in modes:
    print(f"\n{'━'*70}")
    print(f"  MODE: {mode_label}")
    print(f"{'━'*70}")

    mode_results = []

    for lang in ["en", "zh"]:
        lang_label = "English" if lang == "en" else "Chinese"
        print(f"\n  ── {lang_label} ──")

        for prob_idx, prob in enumerate(test_problems):
            prompt = prob[f"prompt_{lang}"]
            answer = prob["answer"]

            text = run_generation(prompt, strip_mode=strip_mode, strip_scale=scale)
            correct = answer in text
            out_lang = classify_lang(text)
            lang_ok = out_lang == lang

            status = f"{'MATH_OK' if correct else 'MATH_FAIL'} {'LANG_OK' if lang_ok else f'LANG→{out_lang}'}"
            print(f"    [{prob_idx}] {status:<22s} | {text[:70]}")

            mode_results.append({
                "problem_idx": prob_idx,
                "language": lang,
                "difficulty": prob["difficulty"],
                "prompt": prompt,
                "answer": answer,
                "text": text,
                "correct": correct,
                "output_lang": out_lang,
                "lang_preserved": lang_ok,
            })

    n_correct = sum(1 for r in mode_results if r["correct"])
    n_lang = sum(1 for r in mode_results if r["lang_preserved"])
    total = len(mode_results)

    # Language flips (zh->en or en->zh)
    en_to_zh = sum(1 for r in mode_results if r["language"] == "en" and r["output_lang"] == "zh")
    zh_to_en = sum(1 for r in mode_results if r["language"] == "zh" and r["output_lang"] == "en")

    print(f"\n  {mode_label}: {n_correct}/{total} correct, {n_lang}/{total} lang OK")
    if en_to_zh > 0 or zh_to_en > 0:
        print(f"    Language flips: EN→ZH={en_to_zh}, ZH→EN={zh_to_en}")

    results["modes"][mode_label] = {
        "results": mode_results,
        "total_correct": n_correct,
        "total_lang": n_lang,
        "total": total,
        "en_to_zh_flips": en_to_zh,
        "zh_to_en_flips": zh_to_en,
    }

# =============================================================================
# Final comparison
# =============================================================================
print(f"\n{'='*70}")
print("P2 SUMMARY")
print("=" * 70)

print(f"  {'Mode':<20s} {'Math':>6s} {'Lang':>6s} {'EN→ZH':>6s} {'ZH→EN':>6s}")
print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
for mode_label, data in results["modes"].items():
    mc = f"{data['total_correct']}/20"
    lp = f"{data['total_lang']}/20"
    ez = str(data['en_to_zh_flips'])
    ze = str(data['zh_to_en_flips'])
    print(f"  {mode_label:<20s} {mc:>6s} {lp:>6s} {ez:>6s} {ze:>6s}")

bl = results["modes"]["baseline"]
zero = results["modes"]["zero_lang_dir"]
flip = results["modes"]["flip_lang_dir"]

if flip["en_to_zh_flips"] + flip["zh_to_en_flips"] > 5:
    verdict = "CAUSAL DIRECTION FOUND: Flipping it switches language. Math + language are separable along this axis."
elif zero["total_lang"] < bl["total_lang"] - 3:
    verdict = "PARTIAL: Zeroing disrupts language but doesn't cleanly flip it"
else:
    verdict = "CAUSAL DIRECTION IS NOT A SINGLE VECTOR: Language in MLP is distributed"

print(f"\n  VERDICT: {verdict}")
results["verdict"] = verdict

with open("output/expP2_causal_lang_strip.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expP2_causal_lang_strip.json")
