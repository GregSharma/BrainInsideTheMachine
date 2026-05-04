"""Exp BO: Z-Bottleneck Inference Test.

The question: at an intermediate layer, how many dimensions of the residual
stream are needed for the model to continue reasoning correctly?

At layer L, hook the residual stream and project it to a k-dimensional
subspace (null-space aware ordering: most language-invariant dims first).
Then let the model continue from L+1 to the final layer and generate.

If accuracy survives at k << 2048, it proves a Z-bottleneck architecture
is viable: encode to k dims, reason, decode back to language.

Projection: eigenvectors of the cross-language difference gram matrix,
ordered ascending by eigenvalue. Bottom k eigenvecs = most null (math).
Top eigenvecs = most language. Keeping bottom k = bottleneck to k dims.

RMSNorm at each layer's input compensates for norm drop, so the test
is purely about DIRECTION, not scale.
"""

import json
import time
import numpy as np
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SEED = 42
LANGUAGES = ['en', 'zh', 'es', 'ar', 'ja', 'ko', 'sw']
DIM = 2048
MAX_NEW_TOKENS = 128

t0 = time.time()

print("=" * 60)
print("  Exp BO: Z-Bottleneck Inference Test")
print("=" * 60)


# ── 1. Build eigenbases per layer from cached data ──────────────

print("\n[1/4] Building eigenbases from cached multilingual data...")

multi = np.load(OUTPUT_DIR / "multilingual_all_layers.npz")
ALL_LANGS = sorted(set(k.split("_L")[0] for k in multi.files if "_L" in k))

# Preload to RAM
H_cache = {}
for lang in ALL_LANGS:
    H_cache[lang] = {}
    for L in range(36):
        H_cache[lang][L] = multi[f"{lang}_L{L}"].astype(np.float32)
del multi

# Eigenbases: ordered ascending by eigenvalue (bottom = null, top = language)
eigenbases = {}  # L -> (DIM, DIM) full eigenvector matrix, columns ordered ascending
for L in tqdm(range(36), desc="  Eigenbases"):
    diffs = []
    for i, la in enumerate(ALL_LANGS):
        for j, lb in enumerate(ALL_LANGS):
            if i >= j:
                continue
            diffs.append(H_cache[la][L] - H_cache[lb][L])
    diffs = np.vstack(diffs)
    gram = diffs.T @ diffs
    eigenvalues, eigenvectors = np.linalg.eigh(gram)  # ascending
    # eigenvectors columns are ordered ascending: col 0 = most null
    eigenbases[L] = eigenvectors  # (DIM, DIM), columns ordered
del H_cache
print(f"  Done. {len(eigenbases)} layer eigenbases computed.")


# ── 2. Test problem generation ──────────────────────────────────

TEMPLATES = {
    'zh': {
        'arithmetic_plus': "计算 {a} + {b} 的值。",
        'arithmetic_times': "计算 {a} × {b} 的值。",
        'combinatorics': "求组合数 C({n}, {k}) 的值。",
        'modular': "{a} 除以 {b} 的余数是多少？",
        'geometry': "一个长方形的长为 {w}，宽为 {h}，求其面积。",
    },
    'en': {
        'arithmetic_plus': "Calculate {a} + {b}.",
        'arithmetic_times': "Calculate {a} × {b}.",
        'combinatorics': "Find the value of C({n}, {k}).",
        'modular': "What is the remainder when {a} is divided by {b}?",
        'geometry': "A rectangle has length {w} and width {h}. Find its area.",
    },
}


def generate_test_problems(n_test=4):
    import math
    rng = pyrandom.Random(SEED)
    problems = []
    per_cat = 200 // 5

    cats = []
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        ans = a + b if op == "plus" else a * b
        prompts = {}
        for lang in ['en', 'zh']:
            key = f'arithmetic_{op}'
            prompts[lang] = TEMPLATES[lang][key].format(a=a, b=b)
        cats.append(('arithmetic', ans, prompts))

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        ans = math.comb(n_val, k_val)
        prompts = {}
        for lang in ['en', 'zh']:
            prompts[lang] = TEMPLATES[lang]['combinatorics'].format(n=n_val, k=k_val)
        cats.append(('combinatorics', ans, prompts))

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        ans = a % b
        prompts = {}
        for lang in ['en', 'zh']:
            prompts[lang] = TEMPLATES[lang]['modular'].format(a=a, b=b)
        cats.append(('modular', ans, prompts))

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        ans = w * h
        prompts = {}
        for lang in ['en', 'zh']:
            prompts[lang] = TEMPLATES[lang]['geometry'].format(w=w, h=h)
        cats.append(('geometry', ans, prompts))

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        cats.append(('sequences', ans, {
            'en': f"An arithmetic sequence: first term {a1}, common difference {d}. Sum of first {n_terms} terms?",
            'zh': f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
        }))

    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats)))
    rng2.shuffle(indices)
    cats = [cats[i] for i in indices]

    by_cat = {}
    for cat, ans, prompts in cats:
        if cat not in by_cat:
            by_cat[cat] = []
        if len(by_cat[cat]) < n_test:
            by_cat[cat].append((ans, prompts))

    test_set = []
    for cat in by_cat:
        for ans, prompts in by_cat[cat]:
            test_set.append({'category': cat, 'answer': ans, 'en': prompts['en'], 'zh': prompts['zh']})
    return test_set


def check_answer(text, correct_answer):
    target = str(correct_answer)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return target in numbers


def find_fat(text, correct_answer):
    target = str(correct_answer)
    tokens = text.split()
    for i, tok in enumerate(tokens):
        nums = re.findall(r'-?\d+\.?\d*', tok)
        if target in nums:
            return i
    return -1


def detect_lang(text):
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return 'zh' if cjk > latin else ('en' if latin > 0 else 'unk')


# ── 3. Load model ───────────────────────────────────────────────

print("\n[2/4] Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()
print(f"  Model loaded on CUDA")


# ── 4. Bottleneck hook machinery ────────────────────────────────

class BottleneckHook:
    """Projects residual stream at a given layer to k-dimensional subspace."""

    def __init__(self, eigenbasis, k, mode="null_aware"):
        """
        eigenbasis: (DIM, DIM) columns ordered ascending eigenvalue.
          - Column 0 = most language-invariant (null-space).
          - Column 2047 = most language-variant.
        k: number of dims to keep.
        mode: "null_aware" keeps bottom k (most math).
              "random" keeps random k dims.
              "reverse" keeps top k (most language) — control.
        """
        if mode == "null_aware":
            V_k = eigenbasis[:, :k]  # bottom k cols
        elif mode == "random":
            rng = np.random.RandomState(42)
            rand_basis = rng.randn(DIM, k).astype(np.float32)
            V_k, _ = np.linalg.qr(rand_basis)  # orthonormalize
        elif mode == "reverse":
            V_k = eigenbasis[:, -k:]  # top k cols (most language)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Projector: V_k V_k^T
        self.P = torch.tensor(V_k @ V_k.T, dtype=torch.bfloat16, device="cuda")
        self.k = k
        self.mode = mode

    def __call__(self, module, input, output):
        # output is a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            h = output[0]  # (batch, seq_len, DIM)
            h_proj = torch.einsum("bsd,de->bse", h, self.P)
            return (h_proj,) + output[1:]
        else:
            h_proj = torch.einsum("bsd,de->bse", output, self.P)
            return h_proj


# ── 5. Run experiments ──────────────────────────────────────────

print("\n[3/4] Generating test problems...")
test_problems = generate_test_problems(n_test=4)
print(f"  {len(test_problems)} test problems ready")

# Condition list: (name, layer, k, mode)
conditions = [
    ("baseline", None, None, None),
]

# Null-aware bottleneck sweep
for L in [8, 16, 28]:
    for k in [3, 10, 20, 50, 100, 200, 500]:
        conditions.append((f"null_L{L}_k{k}", L, k, "null_aware"))

# Random control at key k values
for L in [8, 16]:
    for k in [20, 100]:
        conditions.append((f"rand_L{L}_k{k}", L, k, "random"))

# Reverse control (keep language, discard math)
for L in [8, 16]:
    conditions.append((f"rev_L{L}_k20", L, 20, "reverse"))

print(f"\n  Total conditions: {len(conditions)}")

all_results = {}


def run_condition(name, layer, k, mode):
    """Run one experimental condition."""
    handle = None
    if layer is not None:
        hook = BottleneckHook(eigenbases[layer], k, mode)
        handle = model.model.layers[layer].register_forward_hook(hook)

    results = {}
    for lang in ['en', 'zh']:
        correct = 0
        fats = []
        lang_pres = {'zh': 0, 'en': 0, 'unk': 0}
        for prob in test_problems:
            inputs = tokenizer(prob[lang], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            ok = check_answer(gen, prob['answer'])
            fat = find_fat(gen, prob['answer'])
            if ok:
                correct += 1
            fats.append(fat)
            dl = detect_lang(gen)
            lang_pres[dl] += 1

        valid_fats = [f for f in fats if f >= 0]
        mfat = float(np.mean(valid_fats)) if valid_fats else -1
        results[lang] = {
            'correct': correct, 'total': len(test_problems),
            'mean_fat': round(mfat, 1), 'lang_pres': lang_pres,
        }

    if handle is not None:
        handle.remove()

    return results


print("\n[4/4] Running bottleneck sweep...\n")

for i, (name, layer, k, mode) in enumerate(conditions):
    t_cond = time.time()
    desc = f"[{i+1}/{len(conditions)}] {name}"
    print(f"  {desc}...", end="", flush=True)

    res = run_condition(name, layer, k, mode)
    all_results[name] = res

    en_c = res['en']['correct']
    zh_c = res['zh']['correct']
    en_t = res['en']['total']
    en_fat = res['en']['mean_fat']
    zh_fat = res['zh']['mean_fat']
    elapsed_cond = time.time() - t_cond
    print(f"  EN={en_c}/{en_t} FAT={en_fat:.0f}  ZH={zh_c}/{en_t} FAT={zh_fat:.0f}  ({elapsed_cond:.0f}s)")


# ── 6. Summary ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  BOTTLENECK SWEEP RESULTS")
print("=" * 70)

print(f"\n  {'Condition':>22s} | {'EN':>4s} {'EN_FAT':>7s} | {'ZH':>4s} {'ZH_FAT':>7s} | {'EN_lang':>8s} {'ZH_lang':>8s}")
print(f"  {'-'*22}-+-{'-'*4}-{'-'*7}-+-{'-'*4}-{'-'*7}-+-{'-'*8}-{'-'*8}")

for name, _, _, _ in conditions:
    r = all_results[name]
    en, zh = r['en'], r['zh']
    en_lp = f"{en['lang_pres'].get('en',0)}/{en['total']}"
    zh_lp = f"{zh['lang_pres'].get('zh',0)}/{zh['total']}"
    print(f"  {name:>22s} | {en['correct']:>3}/{en['total']} {en['mean_fat']:>7.1f} | "
          f"{zh['correct']:>3}/{zh['total']} {zh['mean_fat']:>7.1f} | {en_lp:>8s} {zh_lp:>8s}")

# Phase transition detection
print("\n  Null-aware bottleneck phase transition:")
for L in [8, 16, 28]:
    print(f"\n  Layer {L}:")
    baseline_en = all_results["baseline"]["en"]["correct"]
    baseline_zh = all_results["baseline"]["zh"]["correct"]
    for k in [3, 10, 20, 50, 100, 200, 500]:
        key = f"null_L{L}_k{k}"
        if key in all_results:
            en = all_results[key]["en"]["correct"]
            zh = all_results[key]["zh"]["correct"]
            en_pct = en / max(baseline_en, 1) * 100
            zh_pct = zh / max(baseline_zh, 1) * 100
            bar_en = "#" * int(en_pct / 5)
            bar_zh = "#" * int(zh_pct / 5)
            print(f"    k={k:>4d}: EN {en:>2d}/{baseline_en} ({en_pct:>5.0f}%) {bar_en}")
            print(f"           ZH {zh:>2d}/{baseline_zh} ({zh_pct:>5.0f}%) {bar_zh}")

elapsed = time.time() - t0
print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

# Save
output = {
    "experiment": "BO",
    "title": "Z-Bottleneck Inference Test",
    "model": MODEL_NAME,
    "n_problems": len(test_problems),
    "max_new_tokens": MAX_NEW_TOKENS,
    "runtime_seconds": round(elapsed),
    "conditions": {name: {
        "layer": layer, "k": k, "mode": mode,
        "results": all_results[name],
    } for name, layer, k, mode in conditions},
}

with open(OUTPUT_DIR / "expBO_z_bottleneck.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n  Saved to output/expBO_z_bottleneck.json")

del model
torch.cuda.empty_cache()
