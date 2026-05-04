"""
Exp AC: Kernel-Projected MLP Generation — The Galactic Swish Candidate

The 1D flip (Exp R4): strips EN-ZH mean-difference from MLP deltas at L9-L26.
  - +160% math accuracy at 128 tok, 6x FAT speedup
  - BUT: breaks language when applied to ZH prompts (language follows MLP donor)

The kernel (Exp AB): projects out 10D language subspace fitted from 7 languages.
  - +71% answer probe transfer vs raw
  - Preserves category at 1.0 through L35

The question: does the 10D kernel projector DURING GENERATION:
  1. Preserve or improve math efficiency (FAT metric)?
  2. PRESERVE translation fidelity (the 1D flip's fatal flaw)?

If yes to both — the kernel separates what the 1D flip couldn't.
This is the generation-time test of Greg's "invert the metric" idea.

Conditions:
  A. baseline — no intervention
  B. flip_1d — 1D EN-ZH mean-diff flip (scale=-1.0), replicates R4
  C. kernel_10d — project out 10D language subspace during MLP deltas
  D. kernel_5d — project out 5D (ablation — is 10 necessary?)

Metrics:
  - FAT: first-appearance token (lower = faster)
  - Correctness: does answer appear in output?
  - Language preserved: does output match prompt language?

Translation canary: run ZH prompts and check if output stays ZH.
  The 1D flip switches ZH→EN. The kernel should NOT.
"""

import json
import numpy as np
import torch
import random as pyrandom
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SEED = 42
STRIP_LAYERS = list(range(9, 27))  # L9-L26 inclusive
MAX_TOKENS = 256
LANGUAGES = ['zh', 'en', 'es', 'ar', 'ja', 'ko', 'sw']

TEST_PROBLEMS = [
    # EN problems
    {'prompt': 'Calculate 47 + 86.', 'answer': '133', 'lang': 'en'},
    {'prompt': 'A rectangle has length 12 and width 5. Find its area.', 'answer': '60', 'lang': 'en'},
    {'prompt': 'What is the remainder when 100 is divided by 7?', 'answer': '2', 'lang': 'en'},
    {'prompt': 'Calculate 15 × 8.', 'answer': '120', 'lang': 'en'},
    {'prompt': 'An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.', 'answer': '40', 'lang': 'en'},
    {'prompt': 'Calculate 387 × 29.', 'answer': '11223', 'lang': 'en'},
    {'prompt': 'Find the value of C(10, 3).', 'answer': '120', 'lang': 'en'},
    {'prompt': 'What is the remainder when 7654 is divided by 37?', 'answer': '34', 'lang': 'en'},
    {'prompt': 'An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.', 'answer': '3475', 'lang': 'en'},
    {'prompt': 'A rectangle has length 47 and width 33. Find its area.', 'answer': '1551', 'lang': 'en'},
    # ZH problems — translation canary
    {'prompt': '计算 47 + 86 的值。', 'answer': '133', 'lang': 'zh'},
    {'prompt': '一个长方形的长为 12，宽为 5，求其面积。', 'answer': '60', 'lang': 'zh'},
    {'prompt': '100 除以 7 的余数是多少？', 'answer': '2', 'lang': 'zh'},
    {'prompt': '计算 15 × 8 的值。', 'answer': '120', 'lang': 'zh'},
    {'prompt': '等差数列首项为 2，公差为 3，求前 5 项之和。', 'answer': '40', 'lang': 'zh'},
    {'prompt': '计算 387 × 29 的值。', 'answer': '11223', 'lang': 'zh'},
    {'prompt': '求组合数 C(10, 3) 的值。', 'answer': '120', 'lang': 'zh'},
    {'prompt': '7654 除以 37 的余数是多少？', 'answer': '34', 'lang': 'zh'},
    {'prompt': '等差数列首项为 7，公差为 11，求前 25 项之和。', 'answer': '3475', 'lang': 'zh'},
    {'prompt': '一个长方形的长为 47，宽为 33，求其面积。', 'answer': '1551', 'lang': 'zh'},
]


def generate_fit_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            problems.append({"zh": f"计算 {a} + {b} 的值。", "en": f"Calculate {a} + {b}."})
        else:
            problems.append({"zh": f"计算 {a} × {b} 的值。", "en": f"Calculate {a} × {b}."})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20); k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。", "en": f"Find the value of C({n_val}, {k_val})."})
    for _ in range(per_cat):
        a = rng.randint(50, 9999); b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？", "en": f"What is the remainder when {a} is divided by {b}?"})
    for _ in range(per_cat):
        w = rng.randint(2, 50); h = rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。", "en": f"A rectangle has length {w} and width {h}. Find its area."})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20); d = rng.randint(1, 10); n_t = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d}，求前 {n_t} 项之和。",
                         "en": f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_t} terms."})
    rng.shuffle(problems)
    return problems


def fit_1d_lang_dirs(model, tokenizer, problems, layers):
    """Fit 1D EN-ZH mean-difference direction per layer (same as R4)."""
    print("  Fitting 1D language directions...")
    layer_acts = {l: {'zh': [], 'en': []} for l in layers}
    layer_outputs = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out if isinstance(out, torch.Tensor) else out[0]
            layer_outputs[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        for lang in ['zh', 'en']:
            for p in tqdm(problems, desc=f"  1D fit {lang}", leave=False):
                inp = tokenizer(p[lang], return_tensors='pt').to(model.device)
                with torch.no_grad(): model(**inp)
                for l in layers:
                    layer_acts[l][lang].append(layer_outputs[l].copy())
                layer_outputs.clear()
    finally:
        for h in handles: h.remove()

    dirs = {}
    for l in layers:
        zh = np.mean(layer_acts[l]['zh'], axis=0)
        en = np.mean(layer_acts[l]['en'], axis=0)
        d = zh - en
        dirs[l] = torch.tensor(d / (np.linalg.norm(d) + 1e-8), dtype=torch.bfloat16)
    return dirs


def fit_kernel_projectors(n_lang_dims=10):
    """Load precomputed multilingual activations and fit kernel projectors per layer."""
    print("  Loading multilingual_all_layers.npz for kernel fit...")
    ml = np.load("output/multilingual_all_layers.npz")
    n_layers = 36
    d = 2048
    projectors = {}

    for l in STRIP_LAYERS:
        per_lang = np.stack([ml[f"{lang}_L{l}"] for lang in LANGUAGES], axis=1)  # (200, 7, 2048)
        prob_means = per_lang.mean(axis=1, keepdims=True)
        deviations = (per_lang - prob_means).reshape(-1, d)
        _, _, Vt = np.linalg.svd(deviations, full_matrices=False)
        lang_axes = Vt[:n_lang_dims]
        U_mat = lang_axes.T.astype(np.float32)
        P = np.eye(d, dtype=np.float32) - (U_mat @ U_mat.T)
        projectors[l] = torch.tensor(P, dtype=torch.bfloat16)

    return projectors


def classify_lang(text):
    """Rough language detection: zh/en/other."""
    zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total = max(len(text.strip()), 1)
    if zh_chars / total > 0.15:
        return 'zh'
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / total > 0.3:
        return 'en'
    return 'other'


def generate_with_tracking(prompt, answer, model, tokenizer, mode, mode_data, scale=-1.0):
    """
    Generate up to MAX_TOKENS, tracking first-appearance token (FAT).

    mode: 'baseline' | 'flip_1d' | 'kernel_10d' | 'kernel_5d'
    mode_data: dirs dict (flip) or projectors dict (kernel)
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    prompt_len = inputs['input_ids'].shape[1]

    mlp_hook_handles = []
    current_token_idx = [0]

    if mode == 'flip_1d' and mode_data:
        def make_mlp_hook(layer_idx):
            def hook(module, inp, out):
                if layer_idx not in mode_data:
                    return out
                d_vec = mode_data[layer_idx].to(out.device)
                # Project out and flip: delta_new = delta - 2*(delta·d)*d * (-scale)
                # Equivalent: delta += scale * 2 * (delta·d) * d  [scale<0 = flip]
                proj = (out * d_vec).sum(dim=-1, keepdim=True)
                return out + scale * 2.0 * proj * d_vec
            return hook

        for l in STRIP_LAYERS:
            h = model.model.layers[l].mlp.register_forward_hook(make_mlp_hook(l))
            mlp_hook_handles.append(h)

    elif mode in ('kernel_10d', 'kernel_5d') and mode_data:
        def make_kernel_hook(layer_idx):
            def hook(module, inp, out):
                if layer_idx not in mode_data:
                    return out
                P = mode_data[layer_idx].to(out.device)
                # Project MLP delta onto kernel (remove language component)
                return out @ P.T
            return hook

        for l in STRIP_LAYERS:
            h = model.model.layers[l].mlp.register_forward_hook(make_kernel_hook(l))
            mlp_hook_handles.append(h)

    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in mlp_hook_handles:
            h.remove()

    generated_ids = out_ids[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_lang = classify_lang(generated_text)

    # FAT: first token index where answer string appears in decoded prefix
    fat = -1
    for t in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:t], skip_special_tokens=True)
        if answer in prefix:
            fat = t
            break

    return {
        'output': generated_text[:200],
        'fat': fat,
        'correct': fat > 0,
        'output_lang': output_lang,
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True
    )
    model.eval()
    print("Model loaded.")

    # Fit language directions
    fit_problems = generate_fit_problems(200, seed=SEED)
    dirs_1d = fit_1d_lang_dirs(model, tokenizer, fit_problems, STRIP_LAYERS)

    # Load precomputed kernel projectors
    print("  Fitting 10D kernel projectors from multilingual data...")
    proj_10d = fit_kernel_projectors(n_lang_dims=10)
    print("  Fitting 5D kernel projectors...")
    proj_5d = fit_kernel_projectors(n_lang_dims=5)
    print("All directions/projectors ready.")

    conditions = [
        ('baseline',    None,      1.0),
        ('flip_1d',     dirs_1d,  -1.0),
        ('kernel_10d',  proj_10d,  1.0),
        ('kernel_5d',   proj_5d,   1.0),
    ]

    results = {}
    for cname, cdata, cscale in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cname}")
        print(f"{'='*60}")
        results[cname] = []
        for prob in tqdm(TEST_PROBLEMS, desc=cname):
            r = generate_with_tracking(prob['prompt'], prob['answer'],
                                       model, tokenizer, cname, cdata, cscale)
            r.update({'prompt': prob['prompt'], 'answer': prob['answer'],
                      'prompt_lang': prob['lang']})
            results[cname].append(r)

    # ── Analysis ──
    print("\n" + "="*70)
    print("EXP AC: KERNEL GENERATION RESULTS")
    print("="*70)

    summary = {}
    for cname, cres in results.items():
        en_res = [r for r in cres if r['prompt_lang'] == 'en']
        zh_res = [r for r in cres if r['prompt_lang'] == 'zh']

        en_correct = sum(r['correct'] for r in en_res)
        zh_correct = sum(r['correct'] for r in zh_res)

        en_fats = [r['fat'] for r in en_res if r['fat'] > 0]
        zh_fats = [r['fat'] for r in zh_res if r['fat'] > 0]

        # Language preservation: ZH prompts → ZH output?
        zh_lang_preserved = sum(1 for r in zh_res if r['output_lang'] == 'zh')

        summary[cname] = {
            'en_correct': en_correct, 'en_n': len(en_res),
            'zh_correct': zh_correct, 'zh_n': len(zh_res),
            'en_median_fat': int(np.median(en_fats)) if en_fats else -1,
            'zh_median_fat': int(np.median(zh_fats)) if zh_fats else -1,
            'zh_lang_preserved': zh_lang_preserved,
        }

    print(f"\n{'Condition':>12} | {'EN correct':>10} {'EN FAT':>7} | {'ZH correct':>10} {'ZH FAT':>7} | {'ZH lang ok':>10}")
    print("-"*75)
    for cname, s in summary.items():
        en_acc = f"{s['en_correct']}/{s['en_n']}"
        zh_acc = f"{s['zh_correct']}/{s['zh_n']}"
        zh_lang = f"{s['zh_lang_preserved']}/{s['zh_n']}"
        en_fat = str(s['en_median_fat']) if s['en_median_fat'] > 0 else 'N/A'
        zh_fat = str(s['zh_median_fat']) if s['zh_median_fat'] > 0 else 'N/A'
        print(f"{cname:>12} | {en_acc:>10} {en_fat:>7} | {zh_acc:>10} {zh_fat:>7} | {zh_lang:>10}")

    # Key comparison
    print("\nKEY COMPARISON (vs baseline):")
    bl = summary['baseline']
    for cname, s in summary.items():
        if cname == 'baseline':
            continue
        delta_en = s['en_correct'] - bl['en_correct']
        delta_zh = s['zh_correct'] - bl['zh_correct']
        fat_change_en = (s['en_median_fat'] - bl['en_median_fat']) if bl['en_median_fat'] > 0 and s['en_median_fat'] > 0 else 'N/A'
        zh_lang_loss = bl.get('zh_lang_preserved', 10) - s['zh_lang_preserved']
        print(f"  {cname}: EN Δ={delta_en:+d}, ZH Δ={delta_zh:+d}, "
              f"EN FAT Δ={fat_change_en}, ZH lang lost={zh_lang_loss}/{s['zh_n']}")

    print("\nVERDICT:")
    k10 = summary.get('kernel_10d', {})
    f1d = summary.get('flip_1d', {})
    bl = summary.get('baseline', {})

    kernel_math_ok = k10.get('en_correct', 0) >= bl.get('en_correct', 0)
    kernel_fat_better = k10.get('en_median_fat', 999) < bl.get('en_median_fat', 999)
    kernel_lang_preserved = k10.get('zh_lang_preserved', 0) >= f1d.get('zh_lang_preserved', 0)
    flip_lang_broken = f1d.get('zh_lang_preserved', 10) < bl.get('zh_lang_preserved', 10)

    if kernel_math_ok and kernel_lang_preserved and flip_lang_broken:
        print("  *** GALACTIC SWISH: kernel preserves language AND maintains/improves math ***")
    elif kernel_math_ok and kernel_lang_preserved:
        print("  Kernel preserves math + language. But 1D flip also preserved language (unusual).")
    elif kernel_math_ok and not kernel_lang_preserved:
        print("  Kernel helps math but breaks language similarly to 1D flip.")
    elif not kernel_math_ok:
        print("  Kernel hurts math. The language subspace was carrying math signal.")

    # Save full results
    conditions_meta = [{'name': c[0], 'scale': c[2]} for c in conditions]
    out = {
        'conditions': conditions_meta,
        'summary': summary,
        'raw': {cname: [{k: v for k, v in r.items() if k != 'output'} for r in cres]
                for cname, cres in results.items()},
        'raw_outputs': {cname: [r['output'] for r in cres] for cname, cres in results.items()},
    }
    path = OUTPUT_DIR / "expAC_kernel_generation.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
