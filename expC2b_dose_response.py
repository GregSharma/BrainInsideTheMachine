"""Exp C2b: Dose-Response — Read Head vs Context, Multi-Layer Transplant

C2 showed: replacing last token's V⊥ at one layer → no effect (baseline).
BS showed:  truncating ALL tokens' V⊥ at one layer → 0/20.

The difference: the LAST TOKEN is a read head. It doesn't need V⊥ — it reads
answer information from CONTEXT TOKENS via attention. Context tokens need the
full 2048-D space to carry the computation.

This experiment tests the hypothesis with a clean 2×2 + control:
  Mode A (last_only):   transplant LAST TOKEN at N layers → should stay baseline
  Mode B (all_tokens):  truncate ALL TOKENS at N layers → should degrade to 0
  Mode C (context_only): truncate all EXCEPT last token → isolates context role

Layer counts: 1, 3, 9, 18, 36 (dose-response curve)
k = 20 (fixed — C2 showed no k-dependence for last-token transplant)

Predictions:
  Mode A: flat at baseline regardless of N (attention heals read head)
  Mode B: 0/20 even at N=1 (matching BS)
  Mode C at N=1: 0/20 if context is essential carrier
"""

import json
import math
import time
import re
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = 'Qwen/Qwen2.5-3B'
CACHE_PATH = Path('output/multilingual_all_layers.npz')
OUTPUT_DIR = Path('output')
SEED = 42

LANGS_CACHE = ['ar', 'en', 'es', 'ja', 'ko', 'sw', 'zh']
N_PROB_CACHE = 200
N_LAYERS = 36
K_FIXED = 20       # centered Gram rank_90 ≈ 17-21
MAX_NEW = 128
N_TEST = 4          # per category → 20 total

LAYER_COUNTS = [1, 3, 9, 18, 36]

TEMPLATES = {
    'en': {
        'arithmetic_plus': 'Calculate {a} + {b}.',
        'arithmetic_times': 'Calculate {a} × {b}.',
        'combinatorics': 'Find the value of C({n}, {k}).',
        'modular': 'What is the remainder when {a} is divided by {b}?',
        'geometry': 'A rectangle has length {w} and width {h}. Find its area.',
    },
    'zh': {
        'arithmetic_plus': '计算 {a} + {b} 的值。',
        'arithmetic_times': '计算 {a} × {b} 的值。',
        'combinatorics': '求组合数 C({n}, {k}) 的值。',
        'modular': '{a} 除以 {b} 的余数是多少？',
        'geometry': '一个长方形的长为 {w}，宽为 {h}，求其面积。',
    },
}


# ── Utilities ───────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_test_problems(n_test=4):
    rng = np.random.RandomState(SEED)
    per_cat = 40
    cats = []
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(['plus', 'times'])
        ans = a + b if op == 'plus' else a * b
        prompts = {l: TEMPLATES[l][f'arithmetic_{op}'].format(a=a, b=b)
                   for l in ['en', 'zh']}
        cats.append(('arithmetic', ans, prompts))
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        ans = math.comb(n_val, k_val)
        prompts = {l: TEMPLATES[l]['combinatorics'].format(n=n_val, k=k_val)
                   for l in ['en', 'zh']}
        cats.append(('combinatorics', ans, prompts))
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        ans = a % b
        prompts = {l: TEMPLATES[l]['modular'].format(a=a, b=b)
                   for l in ['en', 'zh']}
        cats.append(('modular', ans, prompts))
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        ans = w * h
        prompts = {l: TEMPLATES[l]['geometry'].format(w=w, h=h)
                   for l in ['en', 'zh']}
        cats.append(('geometry', ans, prompts))
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        cats.append(('sequences', ans, {
            'en': f"An arithmetic sequence: first term {a1}, common difference "
                  f"{d}. Sum of first {n_terms} terms?",
            'zh': f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
        }))
    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats)))
    rng2.shuffle(indices)
    cats = [cats[i] for i in indices]
    by_cat = {}
    for cat, ans, prompts in cats:
        by_cat.setdefault(cat, [])
        if len(by_cat[cat]) < n_test:
            by_cat[cat].append((ans, prompts))
    test_set = []
    for cat in by_cat:
        for ans, prompts in by_cat[cat]:
            test_set.append({
                'category': cat, 'answer': ans,
                'en': prompts['en'], 'zh': prompts['zh'],
            })
    return test_set


def check_answer(text, correct_answer):
    return str(correct_answer) in re.findall(r'-?\d+\.?\d*', text)


def detect_lang(text):
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    return 'zh' if cjk > lat else ('en' if lat > 0 else 'unk')


# ── Subspace computation ────────────────────────────────────────────

def load_layer_states(cache, layer):
    return np.concatenate(
        [cache[f'{lang}_L{layer}'] for lang in LANGS_CACHE], axis=0
    )


def compute_subspace(H, k):
    h_mean = H.mean(axis=0)
    H_c = H - h_mean
    G = H_c @ H_c.T
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx[:k]]
    eigvecs = eigvecs[:, idx[:k]]
    sigma = np.sqrt(np.maximum(eigvals, 1e-12))
    U_k = (H_c.T @ eigvecs) / sigma[None, :]
    return h_mean, U_k


# ── Hook factories ──────────────────────────────────────────────────

def make_last_token_hook(h_mean_t, U_k_t):
    """Zero-perp for LAST TOKEN only. Context tokens pass through."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        # Only modify last token
        last = new_h[0, -1, :].float()
        h_c = last - h_mean_t
        h_V = U_k_t @ (U_k_t.T @ h_c)
        new_h[0, -1, :] = (h_mean_t + h_V).to(h.dtype)
        if isinstance(out, tuple):
            return (new_h,) + out[1:]
        return new_h
    return hook


def make_all_tokens_hook(h_mean_t, U_k_t):
    """Zero-perp for ALL TOKENS. Matches BS methodology."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # Truncate all tokens: center → project → reconstruct
        h_c = h.float() - h_mean_t
        coeffs = h_c @ U_k_t               # (batch, seq, k)
        h_V = coeffs @ U_k_t.T             # (batch, seq, d)
        new_h = (h_V + h_mean_t).to(h.dtype)
        if isinstance(out, tuple):
            return (new_h,) + out[1:]
        return new_h
    return hook


def make_context_only_hook(h_mean_t, U_k_t):
    """Zero-perp for all tokens EXCEPT the last. Isolates context role."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        # Truncate all except last
        ctx = new_h[0, :-1, :].float()
        ctx_c = ctx - h_mean_t
        coeffs = ctx_c @ U_k_t
        ctx_V = coeffs @ U_k_t.T
        new_h[0, :-1, :] = (ctx_V + h_mean_t).to(h.dtype)
        # Last token passes through untouched
        if isinstance(out, tuple):
            return (new_h,) + out[1:]
        return new_h
    return hook


# ── Generation ──────────────────────────────────────────────────────

def generate_tokens(model, tokenizer, input_ids, max_new=128):
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
    tok = out.logits[0, -1].argmax().item()
    tokens = [tok]
    nxt = torch.tensor([[tok]], device=model.device)
    with torch.no_grad():
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[0, -1].argmax().item()
            tokens.append(tok)
            nxt = torch.tensor([[tok]], device=model.device)
            if tok == tokenizer.eos_token_id:
                break
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_with_hooks(model, tokenizer, input_ids, hooks_by_layer, max_new=128):
    """Forward + generate with multiple hooks at specified layers."""
    handles = []
    for layer, hook_fn in hooks_by_layer.items():
        handles.append(
            model.model.layers[layer].register_forward_hook(hook_fn)
        )

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    for h in handles:
        h.remove()

    tok = out.logits[0, -1].argmax().item()
    tokens = [tok]
    nxt = torch.tensor([[tok]], device=model.device)
    with torch.no_grad():
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[0, -1].argmax().item()
            tokens.append(tok)
            nxt = torch.tensor([[tok]], device=model.device)
            if tok == tokenizer.eos_token_id:
                break
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ── Layer selection ─────────────────────────────────────────────────

def select_layers(n, total=36):
    """Select n layers evenly spaced from 0 to total-1."""
    if n >= total:
        return list(range(total))
    if n == 1:
        return [total // 2]  # middle layer
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXP C2b: DOSE-RESPONSE — READ HEAD vs CONTEXT")
    print("  Mode A: last-token transplant at N layers (should be flat)")
    print("  Mode B: all-token truncation at N layers (should match BS)")
    print("  Mode C: context-only truncation (isolates context role)")
    print("=" * 70)

    t0 = time.time()

    # ── [1] Precompute subspaces at ALL 36 layers ──
    print("\n[1] Loading cache and computing subspaces (k=20 at all layers)...",
          flush=True)
    cache = np.load(CACHE_PATH)
    subs = {}
    for layer in range(N_LAYERS):
        H = load_layer_states(cache, layer)
        h_mean, U_k = compute_subspace(H, K_FIXED)
        subs[layer] = {'h_mean': h_mean, 'U_k': U_k}
    del cache
    print(f"  Done. {N_LAYERS} subspaces computed ({time.time()-t0:.1f}s)",
          flush=True)

    # ── [2] Test problems + model ──
    print("\n[2] Test problems...", flush=True)
    problems = generate_test_problems(N_TEST)
    print(f"  {len(problems)} problems", flush=True)

    print("\n[3] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    device = model.device
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    # Pre-convert subspaces to tensors on device
    subs_t = {}
    for layer in range(N_LAYERS):
        h_mean_t = torch.from_numpy(subs[layer]['h_mean']).float().to(device)
        U_k_t = torch.from_numpy(subs[layer]['U_k']).float().to(device)
        subs_t[layer] = (h_mean_t, U_k_t)

    # ── [4] Baselines ──
    print("\n[4] Baselines...", flush=True)
    baselines = {}
    for lang in ['en', 'zh']:
        correct = 0
        for p in problems:
            ids = tokenizer(p[lang], return_tensors="pt").input_ids.to(device)
            text = generate_tokens(model, tokenizer, ids, MAX_NEW)
            if check_answer(text, p['answer']):
                correct += 1
        baselines[lang] = correct
        print(f"  {lang.upper()}: {correct}/{len(problems)}", flush=True)

    # ── [5] Build condition list ──
    conditions = []

    for n_layers in LAYER_COUNTS:
        layers = select_layers(n_layers)
        label_layers = ','.join(str(l) for l in layers)

        # Mode A: last-token only
        conditions.append({
            'name': f'last_only_N{n_layers}',
            'mode': 'last_only',
            'layers': layers,
            'layer_str': label_layers,
            'n_layers': n_layers,
        })

        # Mode B: all tokens
        conditions.append({
            'name': f'all_tokens_N{n_layers}',
            'mode': 'all_tokens',
            'layers': layers,
            'layer_str': label_layers,
            'n_layers': n_layers,
        })

    # Mode C: context-only at single layer (L18, middle)
    conditions.append({
        'name': 'context_only_L18',
        'mode': 'context_only',
        'layers': [18],
        'layer_str': '18',
        'n_layers': 1,
    })
    # Mode C at 9 layers for good measure
    ctx9 = select_layers(9)
    conditions.append({
        'name': 'context_only_N9',
        'mode': 'context_only',
        'layers': ctx9,
        'layer_str': ','.join(str(l) for l in ctx9),
        'n_layers': 9,
    })

    print(f"\n[5] Running {len(conditions)} conditions × 40 problems...",
          flush=True)

    # ── [6] Run ──
    all_results = {}
    for ci, cond in enumerate(conditions):
        name = cond['name']
        mode = cond['mode']
        layers = cond['layers']

        # Build hooks
        hook_factory = {
            'last_only': make_last_token_hook,
            'all_tokens': make_all_tokens_hook,
            'context_only': make_context_only_hook,
        }[mode]

        hooks = {l: hook_factory(*subs_t[l]) for l in layers}

        results_by_lang = {}
        for lang in ['en', 'zh']:
            correct = 0
            details = []
            for i, p in enumerate(problems):
                ids = tokenizer(p[lang], return_tensors="pt").input_ids.to(device)
                text = generate_with_hooks(model, tokenizer, ids, hooks, MAX_NEW)
                ok = check_answer(text, p['answer'])
                if ok:
                    correct += 1
                details.append({
                    'i': i, 'ok': ok, 'gen': text[:120],
                    'lang_out': detect_lang(text),
                })
            results_by_lang[lang] = {
                'correct': correct, 'total': len(problems), 'details': details,
            }

        all_results[name] = {
            'mode': mode,
            'n_layers': cond['n_layers'],
            'layers': layers,
            'en': results_by_lang['en'],
            'zh': results_by_lang['zh'],
        }

        en_c = results_by_lang['en']['correct']
        zh_c = results_by_lang['zh']['correct']
        print(f"  [{ci+1:2d}/{len(conditions)}] {name:25s}  "
              f"EN={en_c:>2}/20  ZH={zh_c:>2}/20  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ── [7] Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaselines: EN={baselines['en']}/20, ZH={baselines['zh']}/20")

    print(f"\n{'Condition':>25s} | {'Mode':>12s} | {'N_layers':>4s} | "
          f"{'EN':>5s} | {'ZH':>5s}")
    print("-" * 70)
    for name in [c['name'] for c in conditions]:
        r = all_results[name]
        print(f"  {name:>23s} | {r['mode']:>12s} | {r['n_layers']:>4d} | "
              f"{r['en']['correct']:>2}/20 | {r['zh']['correct']:>2}/20")

    # ── [8] Save ──
    elapsed = time.time() - t0
    output = {
        'config': {
            'model': MODEL_NAME, 'k': K_FIXED,
            'layer_counts': LAYER_COUNTS, 'max_new': MAX_NEW,
        },
        'baselines': baselines,
        'conditions': all_results,
        'elapsed_s': elapsed,
    }
    out_path = OUTPUT_DIR / 'expC2b_dose_response.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
