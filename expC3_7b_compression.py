#!/usr/bin/env python3
"""
Exp C3-7B: Constructive Compression Test on Qwen2.5-7B at L27
==============================================================
C5 spectral data: L27 has 95.7% variance in ONE direction (r90=1).
This is the sharpest binary test: if k=1 is lossless, rank-1 readout
is confirmed cross-model. If k=1 fails despite 95.7% spectral
concentration, spectral != causal.

Self-contained: no imports from local modules. Runs on A100.

Design (same as C3-3B):
  Phase 1: 10 basis problems (2/category, stratified). Capture self_attn
    post-o_proj at L27 during generation. SVD -> basis.
  Phase 2: 10 test problems. Generate with compression hook projecting
    L27 attention output onto k-D subspace. Sweep k={1,2,3,4,5,6,8,12,20}.
  Only modifies last token during generation. Prompt untouched.
"""

import json, math, time, re, argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

torch.backends.cuda.matmul.allow_tf32 = True

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen2.5-7B"
N_LAYERS = 28
D_MODEL = 3584
TARGET_LAYER = 27  # C5: top1_frac=0.9571, r90=1
MAX_NEW = 128
K_VALUES = [1, 2, 3, 4, 5, 6, 8, 12, 20, 50]
SEED = 42

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)

LANGS = ['en', 'zh']  # bilingual test (matches C3-3B)

# ═══════════════════════════════════════════════════════════════════
# PROBLEM GENERATION (self-contained, identical to C2c/C3)
# ═══════════════════════════════════════════════════════════════════

TEMPLATES = {
    'en': {
        'arithmetic_plus': "Calculate {a} + {b}.",
        'arithmetic_times': "Calculate {a} × {b}.",
        'combinatorics': "Find the value of C({n}, {k}).",
        'modular': "What is the remainder when {a} is divided by {b}?",
        'geometry': "A rectangle has length {w} and width {h}. Find its area.",
        'sequences': "An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n} terms.",
    },
    'zh': {
        'arithmetic_plus': "计算 {a} + {b} 的值。",
        'arithmetic_times': "计算 {a} × {b} 的值。",
        'combinatorics': "求组合数 C({n}, {k}) 的值。",
        'modular': "{a} 除以 {b} 的余数是多少？",
        'geometry': "一个长方形的长为 {w}，宽为 {h}，求其面积。",
        'sequences': "等差数列首项为 {a1}，公差为 {d}，求前 {n} 项之和。",
    },
}

ALL_TEMPLATES = {
    'ar': {
        'arithmetic_plus': "احسب {a} + {b}.",
        'arithmetic_times': "احسب {a} × {b}.",
        'combinatorics': "أوجد قيمة C({n}, {k}).",
        'modular': "ما هو باقي قسمة {a} على {b}؟",
        'geometry': "مستطيل طوله {w} وعرضه {h}، أوجد مساحته.",
        'sequences': "متتالية حسابية حدها الأول {a1} وفرقها {d}، أوجد مجموع أول {n} حد.",
    },
    'en': TEMPLATES['en'],
    'es': {
        'arithmetic_plus': "Calcula {a} + {b}.",
        'arithmetic_times': "Calcula {a} × {b}.",
        'combinatorics': "Encuentra el valor de C({n}, {k}).",
        'modular': "¿Cuál es el resto de dividir {a} entre {b}?",
        'geometry': "Un rectángulo tiene largo {w} y ancho {h}. Encuentra su área.",
        'sequences': "Una sucesión aritmética tiene primer término {a1} y diferencia común {d}. Encuentra la suma de los primeros {n} términos.",
    },
    'ja': {
        'arithmetic_plus': "{a} + {b} を計算せよ。",
        'arithmetic_times': "{a} × {b} を計算せよ。",
        'combinatorics': "C({n}, {k}) の値を求めよ。",
        'modular': "{a} を {b} で割った余りを求めよ。",
        'geometry': "縦 {w}、横 {h} の長方形の面積を求めよ。",
        'sequences': "初項 {a1}、公差 {d} の等差数列の初めの {n} 項の和を求めよ。",
    },
    'ko': {
        'arithmetic_plus': "{a} + {b} 를 계산하시오.",
        'arithmetic_times': "{a} × {b} 를 계산하시오.",
        'combinatorics': "C({n}, {k}) 의 값을 구하시오.",
        'modular': "{a} 를 {b} 로 나눈 나머지를 구하시오.",
        'geometry': "가로 {w}, 세로 {h} 인 직사각형의 넓이를 구하시오.",
        'sequences': "첫째 항이 {a1} 이고 공차가 {d} 인 등차수열의 앞 {n} 항의 합을 구하시오.",
    },
    'sw': {
        'arithmetic_plus': "Hesabu {a} + {b}.",
        'arithmetic_times': "Hesabu {a} × {b}.",
        'combinatorics': "Tafuta thamani ya C({n}, {k}).",
        'modular': "Nini ni mabaki wakati {a} inagawanywa na {b}?",
        'geometry': "Mstatili una urefu {w} na upana {h}. Tafuta eneo lake.",
        'sequences': "Mfululizo wa hesabu una neno la kwanza {a1} na tofauti ya kawaida {d}. Tafuta jumla ya maneno {n} ya kwanza.",
    },
    'zh': TEMPLATES['zh'],
}

ALL_LANGS = ['ar', 'en', 'es', 'ja', 'ko', 'sw', 'zh']


def generate_problems(n_per_cat=40):
    """Generate math problems deterministically (seed=42). Identical to C2c."""
    rng = np.random.RandomState(SEED)
    cats = []

    for _ in range(n_per_cat):
        a, b = int(rng.randint(10, 999)), int(rng.randint(10, 999))
        op = rng.choice(['plus', 'times'])
        ans = a + b if op == 'plus' else a * b
        row = {'category': 'arithmetic', 'answer': int(ans)}
        for lang in ALL_LANGS:
            row[lang] = ALL_TEMPLATES[lang][f'arithmetic_{op}'].format(a=a, b=b)
        cats.append(row)

    for _ in range(n_per_cat):
        n_val = int(rng.randint(5, 20))
        k_val = int(rng.randint(1, min(n_val - 1, 8)))
        ans = math.comb(n_val, k_val)
        row = {'category': 'combinatorics', 'answer': int(ans)}
        for lang in ALL_LANGS:
            row[lang] = ALL_TEMPLATES[lang]['combinatorics'].format(n=n_val, k=k_val)
        cats.append(row)

    for _ in range(n_per_cat):
        a = int(rng.randint(50, 9999))
        b = int(rng.randint(3, 37))
        ans = a % b
        row = {'category': 'modular', 'answer': int(ans)}
        for lang in ALL_LANGS:
            row[lang] = ALL_TEMPLATES[lang]['modular'].format(a=a, b=b)
        cats.append(row)

    for _ in range(n_per_cat):
        w = int(rng.randint(2, 50))
        h = int(rng.randint(2, 50))
        ans = w * h
        row = {'category': 'geometry', 'answer': int(ans)}
        for lang in ALL_LANGS:
            row[lang] = ALL_TEMPLATES[lang]['geometry'].format(w=w, h=h)
        cats.append(row)

    for _ in range(n_per_cat):
        a1 = int(rng.randint(1, 20))
        d = int(rng.randint(1, 10))
        n_terms = int(rng.randint(5, 30))
        ans = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        row = {'category': 'sequences', 'answer': int(ans)}
        for lang in ALL_LANGS:
            row[lang] = ALL_TEMPLATES[lang]['sequences'].format(a1=a1, d=d, n=n_terms)
        cats.append(row)

    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats)))
    rng2.shuffle(indices)
    return [cats[i] for i in indices]


def get_test_subset(problems, n_per_cat=4):
    """First n_per_cat problems per category (after shuffle) = test set."""
    by_cat = {}
    for p in problems:
        cat = p['category']
        by_cat.setdefault(cat, [])
        if len(by_cat[cat]) < n_per_cat:
            by_cat[cat].append(p)
    return [p for cat_probs in by_cat.values() for p in cat_probs]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


# ═══════════════════════════════════════════════════════════════════
# HOOKS
# ═══════════════════════════════════════════════════════════════════

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


class CaptureHook:
    """Captures self_attn output (post-o_proj) for last token at each gen step."""
    def __init__(self):
        self.captured = []
        self.active = False

    def __call__(self, module, input, output):
        if not self.active:
            return
        attn_out = output[0]
        if attn_out.shape[1] == 1:
            self.captured.append(attn_out[0, 0].float().cpu().numpy())


class CompressionHook:
    """Projects self_attn output onto k-D affine subspace during generation."""
    def __init__(self, mean_vec, basis_vecs, k, device):
        self.mean = torch.tensor(mean_vec, dtype=torch.float32, device=device)
        self.basis = torch.tensor(
            basis_vecs[:k], dtype=torch.float32, device=device
        )
        self.active = False

    def __call__(self, module, input, output):
        if not self.active:
            return output
        attn_out = output[0]
        if attn_out.shape[1] != 1:
            return output
        x = attn_out[0, 0].float()
        centered = x - self.mean
        coeffs = centered @ self.basis.T
        projected = self.mean + coeffs @ self.basis
        new_out = projected.to(attn_out.dtype).unsqueeze(0).unsqueeze(0)
        return (new_out,) + output[1:]


# ═══════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--also-l26", action="store_true",
                        help="Also run at L26 (second-highest spectral)")
    args = parser.parse_args()

    device = "cuda"
    max_new = 64 if args.dry else MAX_NEW
    target_layer = args.layer
    layers_to_test = [target_layer]
    if args.also_l26 and target_layer == 27:
        layers_to_test.append(26)

    print(f"{'='*60}")
    print(f"Exp C3-7B: Constructive Compression at L{target_layer}")
    print(f"{'='*60}")
    print(f"Model:     {MODEL_NAME}")
    print(f"Layers:    {layers_to_test}")
    print(f"k-values:  {K_VALUES}")
    print(f"C5 spectral: L27 top1_frac=0.9571, r90=1 (RANK-1)")
    print(f"Prediction: k=1 should be LOSSLESS if spectral=causal")
    print()

    # ── Load model ──
    print("Loading model...")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t_load:.0f}s")
    print(f"  Model layers: {len(model.model.layers)}")
    print(f"  d_model: {model.config.hidden_size}")
    print()

    # ── Problems (stratified: 2/cat basis, 2/cat test) ──
    all_problems = generate_problems()
    all_test = get_test_subset(all_problems)
    by_cat = OrderedDict()
    for p in all_test:
        cat = p.get("category", "?")
        by_cat.setdefault(cat, []).append(p)
    basis_problems, test_problems = [], []
    for cat, probs in by_cat.items():
        half = max(1, len(probs) // 2)
        basis_problems.extend(probs[:half])
        test_problems.extend(probs[half:])
    if args.dry:
        basis_problems = basis_problems[:3]
        test_problems = test_problems[:3]
    print(f"  {len(basis_problems)} basis + {len(test_problems)} test problems")
    cats_basis = set(p.get("category", "?") for p in basis_problems)
    cats_test = set(p.get("category", "?") for p in test_problems)
    print(f"  Basis categories: {sorted(cats_basis)}")
    print(f"  Test categories:  {sorted(cats_test)}")
    n_basis = len(basis_problems)
    n_test = len(test_problems)
    print()

    all_layer_results = {}

    for target_l in layers_to_test:
        print(f"\n{'#'*60}")
        print(f"### TARGET LAYER: L{target_l}")
        print(f"{'#'*60}\n")

        attn_module = model.model.layers[target_l].self_attn

        # ═════════════════════════════════════════════════════════
        # PHASE 1: BASIS EXTRACTION
        # ═════════════════════════════════════════════════════════
        print(f"PHASE 1: Extract SVD basis at L{target_l}")
        capture = CaptureHook()
        handle = attn_module.register_forward_hook(capture)

        all_captures = []
        t0 = time.time()

        for pi, prob in enumerate(basis_problems):
            for lang in LANGS:
                prompt_text = build_prompt(tokenizer, prob[lang])
                capture.captured = []
                capture.active = True
                generate(model, tokenizer, prompt_text, device, max_new)
                capture.active = False
                all_captures.extend(capture.captured)
                print(f"  Basis P{pi}/{lang}: {len(capture.captured)} steps")

        handle.remove()

        all_readouts = np.stack(all_captures)
        print(f"  Readout matrix: {all_readouts.shape}")

        mean_vec = all_readouts.mean(axis=0)
        centered = all_readouts - mean_vec
        _, S, Vh = np.linalg.svd(centered, full_matrices=False)
        basis_vecs = Vh

        cumvar = np.cumsum(S**2) / (S**2).sum()
        rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        print(f"\n  Cross-problem effective rank at 90% variance: {rank_90}")
        print(f"  Per-problem effective rank: {all_readouts.shape[0] / (2 * n_basis):.1f} avg steps")
        print(f"  Variance explained:")
        for k in [1, 2, 3, 5, 8, 12, 20, rank_90]:
            if k <= len(cumvar):
                print(f"    k={k:3d}: {cumvar[k-1]*100:6.2f}%")

        basis_time = time.time() - t0
        print(f"  Basis extraction: {basis_time:.0f}s\n")

        # ═════════════════════════════════════════════════════════
        # PHASE 2: COMPRESSION TEST
        # ═════════════════════════════════════════════════════════
        print(f"PHASE 2: Compression test at L{target_l}")

        results = {}

        # ── Baseline ──
        print(f"\n--- Baseline (no compression) ---")
        bl = []
        for pi, prob in enumerate(test_problems):
            for lang in LANGS:
                prompt_text = build_prompt(tokenizer, prob[lang])
                text = generate(model, tokenizer, prompt_text, device, max_new)
                correct = check_answer(text, prob["answer"])
                bl.append({
                    "problem_idx": pi,
                    "lang": lang,
                    "category": prob.get("category", "?"),
                    "correct": correct,
                    "text": text[:300],
                })
                mark = "\u2713" if correct else "\u2717"
                cat = prob.get("category", "?")[:4]
                print(f"  T{pi}/{lang}({cat}): {mark}  {text[:55]}...")

        bl_correct = sum(r["correct"] for r in bl)
        results["baseline"] = {
            "accuracy": bl_correct / len(bl),
            "correct": bl_correct,
            "total": len(bl),
            "en_correct": sum(r["correct"] for r in bl if r["lang"] == "en"),
            "zh_correct": sum(r["correct"] for r in bl if r["lang"] == "zh"),
            "per_problem": bl,
        }
        print(f"  Baseline: {bl_correct}/{len(bl)} "
              f"(EN={results['baseline']['en_correct']}, ZH={results['baseline']['zh_correct']})\n")

        # ── Compression sweep ──
        for k in K_VALUES:
            print(f"--- k={k} ---")
            compressor = CompressionHook(mean_vec, basis_vecs, k, device)
            handle = attn_module.register_forward_hook(compressor)

            kr = []
            for pi, prob in enumerate(test_problems):
                for lang in LANGS:
                    prompt_text = build_prompt(tokenizer, prob[lang])
                    compressor.active = True
                    text = generate(model, tokenizer, prompt_text, device, max_new)
                    compressor.active = False
                    correct = check_answer(text, prob["answer"])
                    kr.append({
                        "problem_idx": n_basis + pi,
                        "lang": lang,
                        "category": prob.get("category", "?"),
                        "correct": correct,
                        "text": text[:300],
                    })
                    mark = "\u2713" if correct else "\u2717"
                    cat = prob.get("category", "?")[:4]
                    print(f"  P{n_basis+pi}/{lang}({cat}): {mark}  {text[:60]}...")

            handle.remove()
            k_correct = sum(r["correct"] for r in kr)
            en_correct = sum(r["correct"] for r in kr if r["lang"] == "en")
            zh_correct = sum(r["correct"] for r in kr if r["lang"] == "zh")
            results[f"k={k}"] = {
                "accuracy": k_correct / len(kr),
                "correct": k_correct,
                "total": len(kr),
                "en_correct": en_correct,
                "zh_correct": zh_correct,
                "per_problem": kr,
            }
            print(f"  k={k}: {k_correct}/{len(kr)} (EN={en_correct}, ZH={zh_correct})\n")

        # Store results for this layer
        all_layer_results[f"L{target_l}"] = {
            "target_layer": target_l,
            "basis_readout_shape": list(all_readouts.shape),
            "singular_values_top50": S[:50].tolist(),
            "cross_problem_rank_90pct": rank_90,
            "basis_variance_explained": {
                str(k): float(cumvar[k-1])
                for k in sorted(set(K_VALUES + [rank_90]))
                if k <= len(cumvar)
            },
            "results": results,
        }

    # ═════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for layer_key, layer_data in all_layer_results.items():
        tl = layer_data["target_layer"]
        res = layer_data["results"]
        cumv = np.array([layer_data["basis_variance_explained"].get(str(k), 0)
                        for k in K_VALUES])
        print(f"\n  {layer_key} (rank_90={layer_data['cross_problem_rank_90pct']}):")
        print(f"  {'Condition':>12} | {'Score':>8} | {'EN':>4} | {'ZH':>4} | {'VarExpl':>8}")
        print(f"  {'-'*55}")
        for label in ["baseline"] + [f"k={k}" for k in K_VALUES]:
            r = res[label]
            if label == "baseline":
                ve = "  ---"
            else:
                kv = int(label.split("=")[1])
                ve_val = layer_data["basis_variance_explained"].get(str(kv), None)
                ve = f"{ve_val*100:5.1f}%" if ve_val else "  ---"
            print(
                f"  {label:>12} | {r['correct']:>3}/{r['total']:<4} | "
                f"{r.get('en_correct','?'):>4} | {r.get('zh_correct','?'):>4} | {ve:>8}"
            )

    # ── Save ──
    output = {
        "experiment": "C3-7B: Constructive Compression Test",
        "model": MODEL_NAME,
        "n_layers": N_LAYERS,
        "d_model": D_MODEL,
        "c5_spectral_ref": {
            "L27": {"top1_frac": 0.9571, "sigma1": 2162.5, "r90": 1},
            "L26": {"top1_frac": 0.8367, "sigma1": 1029.2, "r90": 75},
        },
        "prediction": "k=1 lossless at L27 if spectral=causal",
        "n_basis_problems": n_basis,
        "n_test_problems": n_test,
        "max_new": max_new,
        "k_values": K_VALUES,
        "elapsed_s": total_time,
        "layers": all_layer_results,
    }
    outpath = Path(f"expC3_7b_compression_L{target_layer}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
