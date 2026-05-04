#!/usr/bin/env python3
"""
Exp C6b: Mean Dissection — Is it v1 or the mean doing the work?
================================================================
C6 showed cos(v1, mean_readout) = -0.954 and k=1 lossless at ALL layers.
This means the affine compression (mean + project onto v1) is approximately
"project onto the mean direction." But is the mean SUFFICIENT?

4 conditions at L27:
  1. mean-only:     replace attn output with constant mean vector
  2. zero-attn:     replace attn output with zeros
  3. random-1d:     project onto random unit direction (is v1 special?)
  4. anti-mean-1d:  project onto direction ORTHOGONAL to mean

If mean-only = baseline → the mean vector carries everything
If zero-attn = baseline → attention output is irrelevant at last token
If random-1d = baseline → any 1D affine subspace works (structure doesn't matter)
If anti-mean-1d breaks → mean direction specifically is load-bearing

Quick run: ~5 conditions × 20 problems × ~10s = ~15 min
"""

import json, math, time, re
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

torch.backends.cuda.matmul.allow_tf32 = True

MODEL_NAME = "Qwen/Qwen2.5-7B"
TARGET_LAYER = 27
MAX_NEW = 128
SEED = 42
LANGS = ['en', 'zh']

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)

ALL_LANGS = ['ar', 'en', 'es', 'ja', 'ko', 'sw', 'zh']
ALL_TEMPLATES = {
    'ar': {'arithmetic_plus': "احسب {a} + {b}.", 'arithmetic_times': "احسب {a} × {b}.", 'combinatorics': "أوجد قيمة C({n}, {k}).", 'modular': "ما هو باقي قسمة {a} على {b}؟", 'geometry': "مستطيل طوله {w} وعرضه {h}، أوجد مساحته.", 'sequences': "متتالية حسابية حدها الأول {a1} وفرقها {d}، أوجد مجموع أول {n} حد."},
    'en': {'arithmetic_plus': "Calculate {a} + {b}.", 'arithmetic_times': "Calculate {a} × {b}.", 'combinatorics': "Find the value of C({n}, {k}).", 'modular': "What is the remainder when {a} is divided by {b}?", 'geometry': "A rectangle has length {w} and width {h}. Find its area.", 'sequences': "An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n} terms."},
    'es': {'arithmetic_plus': "Calcula {a} + {b}.", 'arithmetic_times': "Calcula {a} × {b}.", 'combinatorics': "Encuentra el valor de C({n}, {k}).", 'modular': "¿Cuál es el resto de dividir {a} entre {b}?", 'geometry': "Un rectángulo tiene largo {w} y ancho {h}. Encuentra su área.", 'sequences': "Una sucesión aritmética tiene primer término {a1} y diferencia común {d}. Encuentra la suma de los primeros {n} términos."},
    'ja': {'arithmetic_plus': "{a} + {b} を計算せよ。", 'arithmetic_times': "{a} × {b} を計算せよ。", 'combinatorics': "C({n}, {k}) の値を求めよ。", 'modular': "{a} を {b} で割った余りを求めよ。", 'geometry': "縦 {w}、横 {h} の長方形の面積を求めよ。", 'sequences': "初項 {a1}、公差 {d} の等差数列の初めの {n} 項の和を求めよ。"},
    'ko': {'arithmetic_plus': "{a} + {b} 를 계산하시오.", 'arithmetic_times': "{a} × {b} 를 계산하시오.", 'combinatorics': "C({n}, {k}) 의 값을 구하시오.", 'modular': "{a} 를 {b} 로 나눈 나머지를 구하시오.", 'geometry': "가로 {w}, 세로 {h} 인 직사각형의 넓이를 구하시오.", 'sequences': "첫째 항이 {a1} 이고 공차가 {d} 인 등차수열의 앞 {n} 항의 합을 구하시오."},
    'sw': {'arithmetic_plus': "Hesabu {a} + {b}.", 'arithmetic_times': "Hesabu {a} × {b}.", 'combinatorics': "Tafuta thamani ya C({n}, {k}).", 'modular': "Nini ni mabaki wakati {a} inagawanywa na {b}?", 'geometry': "Mstatili una urefu {w} na upana {h}. Tafuta eneo lake.", 'sequences': "Mfululizo wa hesabu una neno la kwanza {a1} na tofauti ya kawaida {d}. Tafuta jumla ya maneno {n} ya kwanza."},
    'zh': {'arithmetic_plus': "计算 {a} + {b} 的值。", 'arithmetic_times': "计算 {a} × {b} 的值。", 'combinatorics': "求组合数 C({n}, {k}) 的值。", 'modular': "{a} 除以 {b} 的余数是多少？", 'geometry': "一个长方形的长为 {w}，宽为 {h}，求其面积。", 'sequences': "等差数列首项为 {a1}，公差为 {d}，求前 {n} 项之和。"},
}

def generate_problems(n_per_cat=40):
    rng = np.random.RandomState(SEED)
    cats = []
    for _ in range(n_per_cat):
        a, b = int(rng.randint(10, 999)), int(rng.randint(10, 999))
        op = rng.choice(['plus', 'times'])
        ans = a + b if op == 'plus' else a * b
        row = {'category': 'arithmetic', 'answer': int(ans)}
        for lang in ALL_LANGS: row[lang] = ALL_TEMPLATES[lang][f'arithmetic_{op}'].format(a=a, b=b)
        cats.append(row)
    for _ in range(n_per_cat):
        n_val = int(rng.randint(5, 20)); k_val = int(rng.randint(1, min(n_val-1, 8)))
        ans = math.comb(n_val, k_val)
        row = {'category': 'combinatorics', 'answer': int(ans)}
        for lang in ALL_LANGS: row[lang] = ALL_TEMPLATES[lang]['combinatorics'].format(n=n_val, k=k_val)
        cats.append(row)
    for _ in range(n_per_cat):
        a = int(rng.randint(50, 9999)); b = int(rng.randint(3, 37)); ans = a % b
        row = {'category': 'modular', 'answer': int(ans)}
        for lang in ALL_LANGS: row[lang] = ALL_TEMPLATES[lang]['modular'].format(a=a, b=b)
        cats.append(row)
    for _ in range(n_per_cat):
        w = int(rng.randint(2, 50)); h = int(rng.randint(2, 50)); ans = w * h
        row = {'category': 'geometry', 'answer': int(ans)}
        for lang in ALL_LANGS: row[lang] = ALL_TEMPLATES[lang]['geometry'].format(w=w, h=h)
        cats.append(row)
    for _ in range(n_per_cat):
        a1 = int(rng.randint(1, 20)); d = int(rng.randint(1, 10)); n_terms = int(rng.randint(5, 30))
        ans = n_terms * (2*a1 + (n_terms-1)*d) // 2
        row = {'category': 'sequences', 'answer': int(ans)}
        for lang in ALL_LANGS: row[lang] = ALL_TEMPLATES[lang]['sequences'].format(a1=a1, d=d, n=n_terms)
        cats.append(row)
    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats))); rng2.shuffle(indices)
    return [cats[i] for i in indices]

def get_test_subset(problems, n_per_cat=4):
    by_cat = {}
    for p in problems:
        cat = p['category']; by_cat.setdefault(cat, [])
        if len(by_cat[cat]) < n_per_cat: by_cat[cat].append(p)
    return [p for cat_probs in by_cat.values() for p in cat_probs]

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.bool_): return bool(o)
        return super().default(o)

def build_prompt(tokenizer, problem_text):
    messages = [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": problem_text}]
    try: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except: return f"{CHAT_SYSTEM}\n\nProblem: {problem_text}\n\nSolution:"

def check_answer(text, correct):
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)

def generate(model, tokenizer, prompt_text, device, max_new=MAX_NEW):
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    generated_ids, past_kv, cur_input = [], None, ids
    with torch.inference_mode():
        for _ in range(max_new):
            out = model(cur_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax().item())
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id: break
            cur_input = torch.tensor([[next_id]], device=device)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════
# INTERVENTION HOOKS
# ═══════════════════════════════════════════════════════════════════

class MeanOnlyHook:
    """Replace attn output with constant mean vector."""
    def __init__(self, mean_vec, device):
        self.mean = torch.tensor(mean_vec, dtype=torch.float32, device=device)
        self.active = False
    def __call__(self, module, input, output):
        if not self.active: return output
        attn_out = output[0]
        if attn_out.shape[1] != 1: return output
        new_out = self.mean.to(attn_out.dtype).unsqueeze(0).unsqueeze(0)
        return (new_out,) + output[1:]

class ZeroAttnHook:
    """Replace attn output with zeros."""
    def __init__(self):
        self.active = False
    def __call__(self, module, input, output):
        if not self.active: return output
        attn_out = output[0]
        if attn_out.shape[1] != 1: return output
        new_out = torch.zeros_like(attn_out)
        return (new_out,) + output[1:]

class RandomProjectHook:
    """Project onto random 1D affine subspace (mean + random direction)."""
    def __init__(self, mean_vec, device, seed=123):
        self.mean = torch.tensor(mean_vec, dtype=torch.float32, device=device)
        rng = np.random.RandomState(seed)
        rand_dir = rng.randn(len(mean_vec)).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir)
        self.direction = torch.tensor(rand_dir, dtype=torch.float32, device=device)
        self.active = False
    def __call__(self, module, input, output):
        if not self.active: return output
        attn_out = output[0]
        if attn_out.shape[1] != 1: return output
        x = attn_out[0, 0].float()
        centered = x - self.mean
        coeff = centered @ self.direction
        projected = self.mean + coeff * self.direction
        new_out = projected.to(attn_out.dtype).unsqueeze(0).unsqueeze(0)
        return (new_out,) + output[1:]

class OrthMeanHook:
    """Project onto 1D direction orthogonal to mean."""
    def __init__(self, mean_vec, v1, device):
        self.mean = torch.tensor(mean_vec, dtype=torch.float32, device=device)
        # Gram-Schmidt: remove mean component from v1
        mean_norm = mean_vec / np.linalg.norm(mean_vec)
        v1_orth = v1 - np.dot(v1, mean_norm) * mean_norm
        v1_orth /= np.linalg.norm(v1_orth)
        self.direction = torch.tensor(v1_orth, dtype=torch.float32, device=device)
        self.active = False
    def __call__(self, module, input, output):
        if not self.active: return output
        attn_out = output[0]
        if attn_out.shape[1] != 1: return output
        x = attn_out[0, 0].float()
        centered = x - self.mean
        coeff = centered @ self.direction
        projected = self.mean + coeff * self.direction
        new_out = projected.to(attn_out.dtype).unsqueeze(0).unsqueeze(0)
        return (new_out,) + output[1:]

class CaptureHook:
    def __init__(self):
        self.captured = []; self.active = False
    def __call__(self, module, input, output):
        if not self.active: return
        attn_out = output[0]
        if attn_out.shape[1] == 1:
            self.captured.append(attn_out[0, 0].float().cpu().numpy())


def run_condition(model, tokenizer, test_problems, hook, attn_module, device, label):
    """Run one experimental condition."""
    handle = attn_module.register_forward_hook(hook)
    results = []
    for pi, prob in enumerate(test_problems):
        for lang in LANGS:
            prompt_text = build_prompt(tokenizer, prob[lang])
            hook.active = True
            text = generate(model, tokenizer, prompt_text, device)
            hook.active = False
            correct = check_answer(text, prob["answer"])
            results.append({"pi": pi, "lang": lang, "cat": prob["category"],
                          "correct": correct, "text": text[:200]})
    handle.remove()
    n = sum(r["correct"] for r in results)
    en = sum(r["correct"] for r in results if r["lang"] == "en")
    zh = sum(r["correct"] for r in results if r["lang"] == "zh")
    total = len(results)
    print(f"  {label}: {n}/{total} (EN={en}, ZH={zh})")
    return {"label": label, "correct": n, "total": total, "en": en, "zh": zh, "details": results}


def main():
    device = "cuda"
    t_start = time.time()

    print(f"{'='*60}")
    print(f"Exp C6b: Mean Dissection at L{TARGET_LAYER}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Model loaded")

    all_problems = generate_problems()
    all_test = get_test_subset(all_problems)
    by_cat = OrderedDict()
    for p in all_test: by_cat.setdefault(p["category"], []).append(p)
    basis_problems, test_problems = [], []
    for cat, probs in by_cat.items():
        half = max(1, len(probs) // 2)
        basis_problems.extend(probs[:half]); test_problems.extend(probs[half:])
    print(f"  {len(basis_problems)} basis + {len(test_problems)} test")

    attn_module = model.model.layers[TARGET_LAYER].self_attn

    # Extract basis
    print(f"\nExtracting basis at L{TARGET_LAYER}...")
    capture = CaptureHook()
    handle = attn_module.register_forward_hook(capture)
    for prob in basis_problems:
        for lang in LANGS:
            capture.captured = []; capture.active = True
            generate(model, tokenizer, build_prompt(tokenizer, prob[lang]), device)
            capture.active = False
    handle.remove()
    # Need all captures from all basis problems
    capture2 = CaptureHook()
    handle2 = attn_module.register_forward_hook(capture2)
    all_caps = []
    for prob in basis_problems:
        for lang in LANGS:
            capture2.captured = []; capture2.active = True
            generate(model, tokenizer, build_prompt(tokenizer, prob[lang]), device)
            capture2.active = False
            all_caps.extend(capture2.captured)
    handle2.remove()

    all_readouts = np.stack(all_caps)
    mean_vec = all_readouts.mean(axis=0)
    centered = all_readouts - mean_vec
    _, S, Vh = np.linalg.svd(centered, full_matrices=False)
    v1 = Vh[0]
    print(f"  Readout: {all_readouts.shape}, ||mean||={np.linalg.norm(mean_vec):.1f}")
    print(f"  cos(v1, mean_hat) = {np.dot(v1, mean_vec/np.linalg.norm(mean_vec)):.4f}")

    # Baseline
    print(f"\n--- Conditions ---")
    bl_results = []
    for pi, prob in enumerate(test_problems):
        for lang in LANGS:
            text = generate(model, tokenizer, build_prompt(tokenizer, prob[lang]), device)
            correct = check_answer(text, prob["answer"])
            bl_results.append({"pi": pi, "lang": lang, "correct": correct})
    bl_n = sum(r["correct"] for r in bl_results)
    bl_en = sum(r["correct"] for r in bl_results if r["lang"] == "en")
    bl_zh = sum(r["correct"] for r in bl_results if r["lang"] == "zh")
    print(f"  baseline: {bl_n}/{len(bl_results)} (EN={bl_en}, ZH={bl_zh})")

    results = {"baseline": {"correct": bl_n, "total": len(bl_results), "en": bl_en, "zh": bl_zh}}

    # Condition 1: mean-only
    hook1 = MeanOnlyHook(mean_vec, device)
    r1 = run_condition(model, tokenizer, test_problems, hook1, attn_module, device, "mean_only")
    results["mean_only"] = r1

    # Condition 2: zero-attn
    hook2 = ZeroAttnHook()
    r2 = run_condition(model, tokenizer, test_problems, hook2, attn_module, device, "zero_attn")
    results["zero_attn"] = r2

    # Condition 3: random 1D projection
    hook3 = RandomProjectHook(mean_vec, device, seed=123)
    r3 = run_condition(model, tokenizer, test_problems, hook3, attn_module, device, "random_1d")
    results["random_1d"] = r3

    # Condition 4: orthogonal-to-mean 1D projection
    hook4 = OrthMeanHook(mean_vec, v1, device)
    r4 = run_condition(model, tokenizer, test_problems, hook4, attn_module, device, "orth_mean_1d")
    results["orth_mean_1d"] = r4

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — L{TARGET_LAYER} Mean Dissection")
    print(f"{'='*60}")
    print(f"  {'Condition':>15} | {'Score':>8} | {'EN':>4} | {'ZH':>4}")
    print(f"  {'-'*42}")
    for label in ["baseline", "mean_only", "zero_attn", "random_1d", "orth_mean_1d"]:
        r = results[label]
        c = r["correct"]; t = r.get("total", len(bl_results))
        print(f"  {label:>15} | {c:>3}/{t:<4} | {r['en']:>4} | {r['zh']:>4}")

    output = {
        "experiment": "C6b: Mean Dissection",
        "model": MODEL_NAME, "target_layer": TARGET_LAYER,
        "elapsed_s": total_time,
        "mean_norm": float(np.linalg.norm(mean_vec)),
        "cos_v1_mean": float(np.dot(v1, mean_vec/np.linalg.norm(mean_vec))),
        "results": results,
    }
    outpath = Path("expC6b_mean_dissection_7b.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {total_time:.0f}s")

    # Kill the compute
    print("\n*** Releasing Colab GPU ***")
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as e:
        print(f"  Could not unassign (not in Colab notebook?): {e}")
        print("  Run 'from google.colab import runtime; runtime.unassign()' manually")


if __name__ == "__main__":
    main()
