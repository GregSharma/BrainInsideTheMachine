"""
Exp AD: MLP-Delta Kernel — The Correct Space for Generation Intervention

Problem with Exp AC: kernel was fitted on residual stream activations (layer outputs),
but the intervention applies to MLP deltas (module output before residual addition).
These are different spaces. The cross-talk (language dims 6-10 hurting math) comes
from fitting the SVD in the wrong space.

Fix: fit SVD on MLP OUTPUT DELTAS across 7 languages.
The projector is now calibrated for exactly the space being intervened on.

Prediction:
- mlp_kernel_10d may now be neutral on accuracy (vs residual kernel_10d which hurt)
- ZH language preservation maintained (≥9/10)
- FAT speedup comparable to flip_1d
- If all three hold: galactic swish

Conditions:
  A. baseline
  B. flip_1d (same as AC)
  C. mlp_kernel_5d
  D. mlp_kernel_10d
  E. mlp_kernel_15d (ablation — how far can we go?)
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SEED = 42
STRIP_LAYERS = list(range(9, 27))
MAX_TOKENS = 256
LANGUAGES = ['zh', 'en', 'es', 'ar', 'ja', 'ko', 'sw']

TEMPLATES = {
    'zh': {
        'arith_plus': "计算 {a} + {b} 的值。",
        'arith_times': "计算 {a} × {b} 的值。",
        'comb': "求组合数 C({n}, {k}) 的值。",
        'mod': "{a} 除以 {b} 的余数是多少？",
        'geo': "一个长方形的长为 {w}，宽为 {h}，求其面积。",
        'seq': "等差数列首项为 {a1}，公差为 {d}，求前 {n} 项之和。",
    },
    'en': {
        'arith_plus': "Calculate {a} + {b}.",
        'arith_times': "Calculate {a} × {b}.",
        'comb': "Find the value of C({n}, {k}).",
        'mod': "What is the remainder when {a} is divided by {b}?",
        'geo': "A rectangle has length {w} and width {h}. Find its area.",
        'seq': "An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n} terms.",
    },
    'es': {
        'arith_plus': "Calcula {a} + {b}.",
        'arith_times': "Calcula {a} × {b}.",
        'comb': "Encuentra el valor de C({n}, {k}).",
        'mod': "¿Cuál es el resto cuando {a} se divide entre {b}?",
        'geo': "Un rectángulo tiene largo {w} y ancho {h}. Encuentra su área.",
        'seq': "Una sucesión aritmética tiene primer término {a1} y diferencia común {d}. Encuentra la suma de los primeros {n} términos.",
    },
    'ar': {
        'arith_plus': "احسب {a} + {b}.",
        'arith_times': "احسب {a} × {b}.",
        'comb': "أوجد قيمة C({n}, {k}).",
        'mod': "ما هو باقي قسمة {a} على {b}؟",
        'geo': "مستطيل طوله {w} وعرضه {h}. أوجد مساحته.",
        'seq': "متتالية حسابية أول حد فيها {a1} وأساسها {d}. أوجد مجموع أول {n} حدود.",
    },
    'ja': {
        'arith_plus': "{a} + {b} を計算せよ。",
        'arith_times': "{a} × {b} を計算せよ。",
        'comb': "C({n}, {k}) の値を求めよ。",
        'mod': "{a} を {b} で割ったときの余りを求めよ。",
        'geo': "縦 {w}、横 {h} の長方形の面積を求めよ。",
        'seq': "初項 {a1}、公差 {d} の等差数列の初め {n} 項の和を求めよ。",
    },
    'ko': {
        'arith_plus': "{a} + {b} 를 계산하시오.",
        'arith_times': "{a} × {b} 를 계산하시오.",
        'comb': "C({n}, {k}) 의 값을 구하시오.",
        'mod': "{a} 를 {b} 로 나눈 나머지를 구하시오.",
        'geo': "가로 {w}, 세로 {h} 인 직사각형의 넓이를 구하시오.",
        'seq': "첫째 항이 {a1} 이고 공차가 {d} 인 등차수열의 앞 {n} 항의 합을 구하시오.",
    },
    'sw': {
        'arith_plus': "Hesabu {a} + {b}.",
        'arith_times': "Hesabu {a} × {b}.",
        'comb': "Tafuta thamani ya C({n}, {k}).",
        'mod': "Nini ni mabaki wakati {a} inagawanywa na {b}?",
        'geo': "Mstatili una urefu {w} na upana {h}. Tafuta eneo lake.",
        'seq': "Mfululizo wa hesabu una neno la kwanza {a1} na tofauti ya kawaida {d}. Tafuta jumla ya maneno {n} ya kwanza.",
    },
}

TEST_PROBLEMS = [
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


def generate_problems_multilingual(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        row = {}
        for lang in LANGUAGES:
            if op == "plus":
                row[lang] = TEMPLATES[lang]['arith_plus'].format(a=a, b=b)
            else:
                row[lang] = TEMPLATES[lang]['arith_times'].format(a=a, b=b)
        problems.append(row)
    for _ in range(per_cat):
        n_val = rng.randint(5, 20); k_val = rng.randint(1, min(n_val - 1, 8))
        row = {lang: TEMPLATES[lang]['comb'].format(n=n_val, k=k_val) for lang in LANGUAGES}
        problems.append(row)
    for _ in range(per_cat):
        a = rng.randint(50, 9999); b = rng.randint(3, 37)
        row = {lang: TEMPLATES[lang]['mod'].format(a=a, b=b) for lang in LANGUAGES}
        problems.append(row)
    for _ in range(per_cat):
        w = rng.randint(2, 50); h = rng.randint(2, 50)
        row = {lang: TEMPLATES[lang]['geo'].format(w=w, h=h) for lang in LANGUAGES}
        problems.append(row)
    for _ in range(per_cat):
        a1 = rng.randint(1, 20); d = rng.randint(1, 10); n_t = rng.randint(5, 30)
        row = {lang: TEMPLATES[lang]['seq'].format(a1=a1, d=d, n=n_t) for lang in LANGUAGES}
        problems.append(row)
    rng.shuffle(problems)
    return problems


def extract_mlp_deltas(model, tokenizer, problems, layers):
    """Extract MLP output (delta) at last token position for all 7 languages × all strip layers."""
    N = len(problems)
    d = model.config.hidden_size
    # {lang: {layer: (N, d)}}
    deltas = {lang: {l: np.zeros((N, d), dtype=np.float32) for l in layers} for lang in LANGUAGES}

    layer_mlp_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out if isinstance(out, torch.Tensor) else out[0]
            layer_mlp_out[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].mlp.register_forward_hook(make_hook(l)) for l in layers]
    try:
        for lang in LANGUAGES:
            print(f"  Extracting MLP deltas: {lang}...")
            for i, prob in enumerate(tqdm(problems, desc=lang, leave=False)):
                inp = tokenizer(prob[lang], return_tensors='pt').to(model.device)
                with torch.no_grad(): model(**inp)
                for l in layers:
                    deltas[lang][l][i] = layer_mlp_out[l].copy()
                layer_mlp_out.clear()
    finally:
        for h in handles: h.remove()

    return deltas


def fit_mlp_kernel_projectors(deltas, layers, n_lang_dims):
    """Fit SVD on MLP delta cross-language deviations per layer."""
    d = 2048
    projectors = {}
    for l in tqdm(layers, desc=f"  SVD {n_lang_dims}D"):
        per_lang = np.stack([deltas[lang][l] for lang in LANGUAGES], axis=1)  # (200, 7, d)
        prob_means = per_lang.mean(axis=1, keepdims=True)
        deviations = (per_lang - prob_means).reshape(-1, d)
        _, _, Vt = np.linalg.svd(deviations, full_matrices=False)
        lang_axes = Vt[:n_lang_dims]
        U_mat = lang_axes.T.astype(np.float32)
        P = np.eye(d, dtype=np.float32) - (U_mat @ U_mat.T)
        projectors[l] = torch.tensor(P, dtype=torch.bfloat16)
    return projectors


def fit_1d_dirs(model, tokenizer, layers):
    """Fit 1D EN-ZH mean-difference per layer."""
    rng = pyrandom.Random(SEED)
    problems = []
    per_cat = 40
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

    layer_acts = {l: {'zh': [], 'en': []} for l in layers}
    layer_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out if isinstance(out, torch.Tensor) else out[0]
            layer_out[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        for lang in ['zh', 'en']:
            for p in tqdm(problems, desc=f"  1D {lang}", leave=False):
                inp = tokenizer(p[lang], return_tensors='pt').to(model.device)
                with torch.no_grad(): model(**inp)
                for l in layers:
                    layer_acts[l][lang].append(layer_out[l].copy())
                layer_out.clear()
    finally:
        for h in handles: h.remove()

    dirs = {}
    for l in layers:
        zh_m = np.mean(layer_acts[l]['zh'], axis=0)
        en_m = np.mean(layer_acts[l]['en'], axis=0)
        v = zh_m - en_m
        dirs[l] = torch.tensor(v / (np.linalg.norm(v) + 1e-8), dtype=torch.bfloat16)
    return dirs


def classify_lang(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total = max(len(text.strip()), 1)
    if zh / total > 0.15: return 'zh'
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / total > 0.3: return 'en'
    return 'other'


def generate_track(prompt, answer, model, tokenizer, mode, mode_data, scale=-1.0):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    prompt_len = inputs['input_ids'].shape[1]
    handles = []

    if mode == 'flip_1d' and mode_data:
        def make_flip(l):
            def hook(module, inp, out):
                if l not in mode_data: return out
                d_vec = mode_data[l].to(out.device)
                proj = (out * d_vec).sum(dim=-1, keepdim=True)
                return out + scale * 2.0 * proj * d_vec
            return hook
        handles = [model.model.layers[l].mlp.register_forward_hook(make_flip(l)) for l in STRIP_LAYERS]

    elif mode.startswith('mlp_kernel') and mode_data:
        def make_proj(l):
            def hook(module, inp, out):
                if l not in mode_data: return out
                P = mode_data[l].to(out.device)
                return out @ P.T
            return hook
        handles = [model.model.layers[l].mlp.register_forward_hook(make_proj(l)) for l in STRIP_LAYERS]

    try:
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    finally:
        for h in handles: h.remove()

    gen = out_ids[0][prompt_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    fat = next((t for t in range(1, len(gen)+1)
                if answer in tokenizer.decode(gen[:t], skip_special_tokens=True)), -1)
    return {'output': text[:200], 'fat': fat, 'correct': fat > 0, 'output_lang': classify_lang(text)}


def main():
    cache = OUTPUT_DIR / "multilingual_mlp_deltas.npz"

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                                  device_map='cuda', trust_remote_code=True)
    model.eval()
    print("Model loaded.")

    # ── Step 1: MLP delta extraction ──
    if cache.exists():
        print(f"Loading cached MLP deltas from {cache}...")
        data = np.load(cache)
        deltas = {lang: {l: data[f"{lang}_L{l}"] for l in STRIP_LAYERS} for lang in LANGUAGES}
        print("Loaded.")
    else:
        print("Extracting MLP deltas (7 langs × 18 layers × 200 probs)...")
        problems = generate_problems_multilingual(200, seed=SEED)
        deltas = extract_mlp_deltas(model, tokenizer, problems, STRIP_LAYERS)
        save_dict = {}
        for lang in LANGUAGES:
            for l in STRIP_LAYERS:
                save_dict[f"{lang}_L{l}"] = deltas[lang][l]
        np.savez_compressed(cache, **save_dict)
        print(f"Saved {cache} ({cache.stat().st_size/1e6:.1f} MB)")

    # ── Step 2: Fit projectors ──
    print("\nFitting MLP-delta kernel projectors...")
    proj_5d  = fit_mlp_kernel_projectors(deltas, STRIP_LAYERS, n_lang_dims=5)
    proj_10d = fit_mlp_kernel_projectors(deltas, STRIP_LAYERS, n_lang_dims=10)
    proj_15d = fit_mlp_kernel_projectors(deltas, STRIP_LAYERS, n_lang_dims=15)

    print("Fitting 1D directions...")
    dirs_1d = fit_1d_dirs(model, tokenizer, STRIP_LAYERS)
    print("All ready.")

    # ── Step 3: Generation test ──
    conditions = [
        ('baseline',      None,     1.0),
        ('flip_1d',       dirs_1d, -1.0),
        ('mlp_kernel_5d', proj_5d,  1.0),
        ('mlp_kernel_10d',proj_10d, 1.0),
        ('mlp_kernel_15d',proj_15d, 1.0),
    ]

    results = {}
    for cname, cdata, cscale in conditions:
        print(f"\nCondition: {cname}")
        results[cname] = []
        for prob in tqdm(TEST_PROBLEMS, desc=cname, leave=False):
            r = generate_track(prob['prompt'], prob['answer'], model, tokenizer, cname, cdata, cscale)
            r.update({'prompt_lang': prob['lang'], 'answer': prob['answer']})
            results[cname].append(r)

    # ── Results ──
    print("\n" + "="*70)
    print("EXP AD: MLP-DELTA KERNEL GENERATION RESULTS")
    print("="*70)

    summary = {}
    for cname, cres in results.items():
        en = [r for r in cres if r['prompt_lang'] == 'en']
        zh = [r for r in cres if r['prompt_lang'] == 'zh']
        en_fats = [r['fat'] for r in en if r['fat'] > 0]
        zh_fats = [r['fat'] for r in zh if r['fat'] > 0]
        summary[cname] = {
            'en_correct': sum(r['correct'] for r in en), 'en_n': len(en),
            'zh_correct': sum(r['correct'] for r in zh), 'zh_n': len(zh),
            'en_median_fat': int(np.median(en_fats)) if en_fats else -1,
            'zh_median_fat': int(np.median(zh_fats)) if zh_fats else -1,
            'zh_lang_preserved': sum(1 for r in zh if r['output_lang'] == 'zh'),
        }

    print(f"\n{'Condition':>15} | {'EN ok':>6} {'EN FAT':>7} | {'ZH ok':>6} {'ZH FAT':>7} | {'ZH lang':>8}")
    print("-"*65)
    bl = summary['baseline']
    for cname, s in summary.items():
        d_en = s['en_correct'] - bl['en_correct']
        flag = ""
        if cname != 'baseline':
            if d_en >= 0 and s['zh_lang_preserved'] >= bl['zh_lang_preserved'] and s['en_median_fat'] < bl['en_median_fat']:
                flag = "  *** SWISH"
            elif d_en >= 0 and s['zh_lang_preserved'] >= bl['zh_lang_preserved']:
                flag = "  (acc+lang ok)"
        en_f = str(s['en_median_fat']) if s['en_median_fat'] > 0 else 'N/A'
        zh_f = str(s['zh_median_fat']) if s['zh_median_fat'] > 0 else 'N/A'
        print(f"{cname:>15} | {s['en_correct']:>3}/{s['en_n']:<2} {en_f:>7} | "
              f"{s['zh_correct']:>3}/{s['zh_n']:<2} {zh_f:>7} | "
              f"{s['zh_lang_preserved']:>3}/{s['zh_n']:<2}{flag}")

    print("\nVS EXP AC (residual stream kernel):")
    print("  AC kernel_5d:  EN 5/10, FAT 66, ZH lang 9/10")
    print("  AC kernel_10d: EN 4/10, FAT 22, ZH lang 9/10")

    # Save
    out_path = OUTPUT_DIR / "expAD_mlp_delta_kernel.json"
    with open(out_path, 'w') as f:
        json.dump({'summary': summary,
                   'raw': {c: [{k: v for k, v in r.items() if k != 'output'} for r in rs]
                           for c, rs in results.items()},
                   'outputs': {c: [r['output'] for r in rs] for c, rs in results.items()}
                   }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
