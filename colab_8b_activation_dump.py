"""
8B Cross-Model Activation Dump — Colab Script
==============================================
Self-contained script for Google Colab (T4/A100).
Captures 7-language × all-layer activations for Qwen2.5-7B.
Outputs: multilingual_all_layers_7b.npz (~700 MB)

Usage:
  1. Upload this file to Colab or paste into a cell
  2. Run. Takes ~30 min on T4, ~15 min on A100.
  3. Download the .npz file from /content/output/

Same format as the 3B cache: keys = {lang}_L{0..27}, categories
"""

import numpy as np
import torch
import time
import random as pyrandom
import gc
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──
MODEL_NAME = "Qwen/Qwen2.5-7B"  # 28 layers, d=3584
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42

LANGUAGES = ["ar", "en", "es", "ja", "ko", "sw", "zh"]

# ── Problem templates (identical to 3B: 200 math problems, 5 categories, 7 languages) ──
TEMPLATES = {
    'ar': {
        'arithmetic_plus': "احسب {a} + {b}.",
        'arithmetic_times': "احسب {a} × {b}.",
        'combinatorics': "أوجد قيمة C({n}, {k}).",
        'modular': "ما هو باقي قسمة {a} على {b}؟",
        'geometry': "مستطيل طوله {w} وعرضه {h}، أوجد مساحته.",
        'sequences': "متتالية حسابية حدها الأول {a1} وفرقها {d}، أوجد مجموع أول {n} حد.",
    },
    'en': {
        'arithmetic_plus': "Calculate {a} + {b}.",
        'arithmetic_times': "Calculate {a} × {b}.",
        'combinatorics': "Find the value of C({n}, {k}).",
        'modular': "What is the remainder when {a} is divided by {b}?",
        'geometry': "A rectangle has length {w} and width {h}. Find its area.",
        'sequences': "An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n} terms.",
    },
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
    'zh': {
        'arithmetic_plus': "计算 {a} + {b} 的值。",
        'arithmetic_times': "计算 {a} × {b} 的值。",
        'combinatorics': "求组合数 C({n}, {k}) 的值。",
        'modular': "{a} 除以 {b} 的余数是多少？",
        'geometry': "一个长方形的长为 {w}，宽为 {h}，求其面积。",
        'sequences': "等差数列首项为 {a1}，公差为 {d}，求前 {n} 项之和。",
    },
}


def generate_problems_multilingual(n=200, seed=42):
    """Generate 200 problems in ALL 7 languages, same seed as base dataset."""
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        row = {'category': 0}
        for lang in LANGUAGES:
            if op == "plus":
                row[lang] = TEMPLATES[lang]['arithmetic_plus'].format(a=a, b=b)
            else:
                row[lang] = TEMPLATES[lang]['arithmetic_times'].format(a=a, b=b)
        problems.append(row)

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        row = {'category': 1}
        for lang in LANGUAGES:
            row[lang] = TEMPLATES[lang]['combinatorics'].format(n=n_val, k=k_val)
        problems.append(row)

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        row = {'category': 2}
        for lang in LANGUAGES:
            row[lang] = TEMPLATES[lang]['modular'].format(a=a, b=b)
        problems.append(row)

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        row = {'category': 3}
        for lang in LANGUAGES:
            row[lang] = TEMPLATES[lang]['geometry'].format(w=w, h=h)
        problems.append(row)

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        row = {'category': 4}
        for lang in LANGUAGES:
            row[lang] = TEMPLATES[lang]['sequences'].format(a1=a1, d=d, n=n_terms)
        problems.append(row)

    rng.shuffle(problems)
    return problems


def extract_multilingual_all_layers(model, tokenizer, problems, n_layers, d):
    """Extract last-token hidden states for all 7 languages × all layers."""
    N = len(problems)
    all_acts = {lang: {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
                for lang in LANGUAGES}

    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            layer_outputs[layer_idx] = h_out.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l))
               for l in range(n_layers)]

    try:
        for lang in LANGUAGES:
            print(f"  Extracting {lang} ({N} problems)...")
            for i, prob in enumerate(tqdm(problems, desc=lang, leave=False)):
                inputs = tokenizer(prob[lang], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**inputs)
                for l in range(n_layers):
                    all_acts[lang][l][i] = layer_outputs[l]
                layer_outputs.clear()
    finally:
        for h in handles:
            h.remove()

    return all_acts


def main():
    t0 = time.time()

    # ── Load model ──
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model loaded: {n_layers} layers, d={d}")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")

    # ── Generate problems ──
    problems = generate_problems_multilingual(200, seed=SEED)
    categories = np.array([p['category'] for p in problems])
    print(f"Generated {len(problems)} problems in {len(LANGUAGES)} languages")

    # ── Extract activations ──
    print(f"\nExtracting activations ({len(LANGUAGES)} langs × {n_layers} layers × {len(problems)} problems)...")
    acts = extract_multilingual_all_layers(model, tokenizer, problems, n_layers, d)

    # ── Save ──
    print("\nSaving activation cache...")
    save_dict = {'categories': categories}
    for lang in LANGUAGES:
        for l in range(n_layers):
            save_dict[f"{lang}_L{l}"] = acts[lang][l]

    model_tag = MODEL_NAME.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = OUTPUT_DIR / f"multilingual_all_layers_{model_tag}.npz"
    np.savez_compressed(outpath, **save_dict)
    filesize = outpath.stat().st_size / 1e6
    print(f"Saved {outpath} ({filesize:.1f} MB)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Keys: {LANGUAGES[0]}_L0 .. {LANGUAGES[-1]}_L{n_layers-1}, categories")
    print(f"Shape per key: ({len(problems)}, {d})")

    # ── Quick sanity check ──
    print("\nSanity check (EN L0 vs ZH L0 cosine):")
    from sklearn.metrics.pairwise import cosine_similarity
    en0 = acts['en'][0]
    zh0 = acts['zh'][0]
    cos = cosine_similarity(en0[:5], zh0[:5])
    print(f"  Mean cosine (first 5 probs): {cos.diagonal().mean():.4f}")

    # Cleanup
    del model, acts
    torch.cuda.empty_cache()
    gc.collect()

    # ── Also run BQ-style Gram analysis if numpy is fast enough ──
    print("\n" + "="*60)
    print("Running quick Gram eigenanalysis on 7B data...")
    print("="*60)

    data = np.load(outpath, allow_pickle=True)
    for L_check in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        arrays = [data[f"{lang}_L{L_check}"] for lang in LANGUAGES]
        H = np.vstack(arrays)
        G = cosine_similarity(H)
        eigenvals = np.linalg.eigvalsh(G)[::-1]
        cumsum = np.cumsum(eigenvals) / eigenvals.sum()
        rank_50 = int(np.searchsorted(cumsum, 0.50) + 1)
        rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)
        print(f"  L{L_check}: rank_50={rank_50}, rank_90={rank_90}, "
              f"top eigenval={eigenvals[0]:.1f}")

    print("\nDone. Download the .npz file for full BQ analysis on local machine.")


if __name__ == "__main__":
    main()
