"""
Exp AE: Procrustes Rotation in Language Subspace

Hypothesis: the 1D flip works because it ROTATES (reflects), not PROJECTS.
The kernel PROJECTS (destroys signal). A 10D ROTATION in the language subspace
should give both: precision of kernel + signal preservation of flip.

Method:
  1. Fit 10D language subspace U per layer (same SVD as AC/AD on residual stream).
  2. Compute ZH and EN centroids projected into U.
  3. Find the rotation R (Procrustes: scipy.linalg.orthogonal_procrustes) that maps
     ZH centroid → EN centroid within the 10D subspace.
  4. During generation: for each MLP output at layer l,
       h_lang = U @ (U^T @ h)          # language component (10D coords → full space)
       h_math = h - h_lang             # math complement (untouched)
       h_new  = h_math + U @ R @ (U^T @ h)  # rotated language back to full space
     This is norm-preserving if R is orthogonal.

Conditions:
  A. baseline
  B. flip_1d        (reference: the current best)
  C. procrustes_5d  (rotation in 5D language subspace)
  D. procrustes_10d (rotation in 10D language subspace)
  E. procrustes_15d (ablation)
  F. procrustes_1d  (should recover flip_1d approximately)

If procrustes_10d ≥ flip_1d on EN accuracy + FAT + lang preservation → GALACTIC SWISH.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import orthogonal_procrustes
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
            key = 'arith_plus' if op == 'plus' else 'arith_times'
            row[lang] = TEMPLATES[lang][key].format(a=a, b=b)
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


def extract_residual_activations(model, tokenizer, problems, layers):
    """Extract residual stream (layer output) at last token for 7 langs × all strip layers."""
    N = len(problems)
    d = model.config.hidden_size
    acts = {lang: {l: np.zeros((N, d), dtype=np.float32) for l in layers} for lang in LANGUAGES}
    layer_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        for lang in LANGUAGES:
            print(f"  Extracting residual: {lang}...")
            for i, prob in enumerate(tqdm(problems, desc=lang, leave=False)):
                inp = tokenizer(prob[lang], return_tensors='pt').to(model.device)
                with torch.no_grad(): model(**inp)
                for l in layers:
                    acts[lang][l][i] = layer_out[l].copy()
                layer_out.clear()
    finally:
        for h in handles: h.remove()
    return acts


def fit_procrustes_rotators(acts, layers, n_lang_dims):
    """
    For each layer:
      1. Fit SVD on cross-language deviations → U (d × n_lang_dims), the language subspace.
      2. Project ZH and EN activations into U coords.
      3. Compute centroids in U space: zh_c (n_lang_dims,), en_c (n_lang_dims,).
      4. Procrustes: R = argmin ||zh_proj @ R - en_proj|| subject to R orthogonal.
         (scipy orthogonal_procrustes: maps source → target)

    Returns dict: layer → {
        'U': torch.tensor (d, n_lang_dims) bfloat16,
        'R': torch.tensor (n_lang_dims, n_lang_dims) float32,
    }
    """
    d = 2048
    rotators = {}
    for l in tqdm(layers, desc=f"  Procrustes {n_lang_dims}D"):
        per_lang = np.stack([acts[lang][l] for lang in LANGUAGES], axis=1)  # (200, 7, d)
        prob_means = per_lang.mean(axis=1, keepdims=True)
        deviations = (per_lang - prob_means).reshape(-1, d)
        _, _, Vt = np.linalg.svd(deviations, full_matrices=False)
        U = Vt[:n_lang_dims].T.astype(np.float32)  # (d, n_lang_dims)

        # Project ZH and EN into U-space
        zh_proj = acts['zh'][l] @ U   # (200, n_lang_dims)
        en_proj = acts['en'][l] @ U   # (200, n_lang_dims)

        # Procrustes: find R such that zh_proj @ R ≈ en_proj
        R, _ = orthogonal_procrustes(zh_proj, en_proj)  # R: (n_lang_dims, n_lang_dims)

        rotators[l] = {
            'U': torch.tensor(U, dtype=torch.bfloat16),
            'R': torch.tensor(R, dtype=torch.float32),
        }
    return rotators


def fit_1d_dirs(model, tokenizer, layers):
    """Fit 1D EN-ZH mean-difference on residual stream."""
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
        a1 = rng.randint(1, 20); d_val = rng.randint(1, 10); n_t = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_t} 项之和。",
                         "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_t} terms."})
    rng.shuffle(problems)

    layer_acts = {l: {'zh': [], 'en': []} for l in layers}
    layer_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
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

    if mode == 'flip_1d':
        # Standard 1D flip on MLP output
        def make_flip(l):
            def hook(module, inp, out):
                if l not in mode_data: return out
                d_vec = mode_data[l].to(out.device)
                proj = (out * d_vec).sum(dim=-1, keepdim=True)
                return out + scale * 2.0 * proj * d_vec
            return hook
        handles = [model.model.layers[l].mlp.register_forward_hook(make_flip(l)) for l in STRIP_LAYERS]

    elif mode.startswith('procrustes'):
        # Procrustes rotation: applied to residual stream AFTER MLP addition
        # Hook on layer output (post-residual), not MLP output
        def make_rotate(l):
            def hook(module, inp, out):
                if l not in mode_data: return out
                h = out[0] if isinstance(out, tuple) else out  # (batch, seq, d)
                rot = mode_data[l]
                U = rot['U'].to(h.device)   # (d, k) bfloat16
                R = rot['R'].to(h.dtype).to(h.device)  # (k, k) bfloat16

                # Project into language subspace
                coords = h @ U              # (batch, seq, k)
                # Rotate: apply R to language coords
                rotated_coords = coords @ R  # (batch, seq, k)
                # Reconstruct: replace language component
                h_lang_orig = coords @ U.T      # (batch, seq, d) — original lang component
                h_lang_new  = rotated_coords @ U.T  # (batch, seq, d) — rotated lang component
                h_new = h - h_lang_orig + h_lang_new

                if isinstance(out, tuple):
                    return (h_new,) + out[1:]
                return h_new
            return hook
        handles = [model.model.layers[l].register_forward_hook(make_rotate(l)) for l in STRIP_LAYERS]

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
    cache_residual = OUTPUT_DIR / "multilingual_all_layers.npz"

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                                  device_map='cuda', trust_remote_code=True)
    model.eval()
    print("Model loaded.")

    # ── Step 1: Residual stream extraction ──
    if cache_residual.exists():
        print(f"Loading cached residual activations from {cache_residual}...")
        data = np.load(cache_residual)
        acts = {lang: {l: data[f"{lang}_L{l}"] for l in STRIP_LAYERS} for lang in LANGUAGES}
        print(f"Loaded. Keys sample: {list(data.keys())[:3]}")
    else:
        print("Extracting residual stream (7 langs × 18 layers × 200 probs)...")
        problems = generate_problems_multilingual(200, seed=SEED)
        acts = extract_residual_activations(model, tokenizer, problems, STRIP_LAYERS)
        save_dict = {}
        for lang in LANGUAGES:
            for l in STRIP_LAYERS:
                save_dict[f"{lang}_L{l}"] = acts[lang][l]
        np.savez_compressed(cache_residual, **save_dict)
        print(f"Saved {cache_residual}")

    # ── Step 2: Fit Procrustes rotators ──
    print("\nFitting Procrustes rotators...")
    rot_1d_approx = fit_procrustes_rotators(acts, STRIP_LAYERS, n_lang_dims=1)
    rot_5d  = fit_procrustes_rotators(acts, STRIP_LAYERS, n_lang_dims=5)
    rot_10d = fit_procrustes_rotators(acts, STRIP_LAYERS, n_lang_dims=10)
    rot_15d = fit_procrustes_rotators(acts, STRIP_LAYERS, n_lang_dims=15)

    print("\nFitting 1D flip direction...")
    dirs_1d = fit_1d_dirs(model, tokenizer, STRIP_LAYERS)
    print("All ready.")

    # ── Step 3: Generation test ──
    conditions = [
        ('baseline',        None,       1.0),
        ('flip_1d',         dirs_1d,   -1.0),
        ('procrustes_1d',   rot_1d_approx, 1.0),
        ('procrustes_5d',   rot_5d,     1.0),
        ('procrustes_10d',  rot_10d,    1.0),
        ('procrustes_15d',  rot_15d,    1.0),
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
    print("EXP AE: PROCRUSTES ROTATION RESULTS")
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
        if cname not in ('baseline',):
            # Galactic swish: EN acc ≥ baseline, ZH lang preserved, FAT lower
            is_swish = (d_en >= 0
                        and s['zh_lang_preserved'] >= bl['zh_lang_preserved']
                        and s['en_median_fat'] > 0
                        and s['en_median_fat'] < bl['en_median_fat'])
            if is_swish:
                flag = "  *** GALACTIC SWISH"
            elif d_en >= 0 and s['zh_lang_preserved'] >= bl['zh_lang_preserved']:
                flag = "  (acc+lang ok)"
            elif d_en >= 0:
                flag = "  (acc ok)"
        en_f = str(s['en_median_fat']) if s['en_median_fat'] > 0 else 'N/A'
        zh_f = str(s['zh_median_fat']) if s['zh_median_fat'] > 0 else 'N/A'
        print(f"{cname:>15} | {s['en_correct']:>3}/{s['en_n']:<2} {en_f:>7} | "
              f"{s['zh_correct']:>3}/{s['zh_n']:<2} {zh_f:>7} | "
              f"{s['zh_lang_preserved']:>3}/{s['zh_n']:<2}{flag}")

    print("\nVS PRIOR EXPERIMENTS (reference):")
    print("  flip_1d (AC):      EN 5/10 FAT=14, ZH 6/10 FAT=18, lang 6/10")
    print("  kernel_5d (AC):    EN 5/10 FAT=66, ZH 5/10 FAT=26, lang 9/10")
    print("  kernel_10d (AC):   EN 4/10 FAT=22, ZH 4/10 FAT=17, lang 9/10")

    # Verbose outputs for inspection
    print("\n--- ZH outputs (procrustes_10d) ---")
    if 'procrustes_10d' in results:
        for r in results['procrustes_10d']:
            if r['prompt_lang'] == 'zh':
                ok = "✓" if r['correct'] else "✗"
                print(f"  {ok} [{r['output_lang']}] FAT={r['fat']:>4} | {r['output'][:80]}")

    # Save
    out_path = OUTPUT_DIR / "expAE_procrustes_rotation.json"
    with open(out_path, 'w') as f:
        json.dump({
            'summary': summary,
            'raw': {c: [{k: v for k, v in r.items() if k != 'output'} for r in rs]
                    for c, rs in results.items()},
            'outputs': {c: [r['output'] for r in rs] for c, rs in results.items()},
            'method': 'procrustes_rotation_in_language_subspace',
            'note': ('Procrustes rotation maps ZH centroid → EN centroid within k-D language '
                     'subspace while preserving norm and math complement. '
                     'Applied to residual stream (layer output), not MLP delta.')
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
