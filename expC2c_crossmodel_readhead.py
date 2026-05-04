#!/usr/bin/env python3
"""
Exp C2c: Cross-Model Read Head vs Context Replication
=====================================================
Replicates C2b finding across Qwen model family:
  Last token = low-rank read head (V⊥ irrelevant at all layers)
  Context tokens = computation substrate (V⊥ essential at 1 layer)

Self-contained: collects activation cache inline, no file uploads needed.
Optimized for A100-80GB.

3B reference (C2b):
  Baselines: EN=5/20, ZH=12/20
  last_only_N36:   EN=4/20, ZH=13/20  (≈ baseline — read head is resilient)
  all_tokens_N1:   EN=0/20, ZH=0/20   (context destruction = total failure)
  context_only_L18: EN=0/20, ZH=0/20  (context IS the computation)

Usage:
  python3 expC2c_crossmodel_readhead.py --model 7b           # ~8 min
  python3 expC2c_crossmodel_readhead.py --model 14b          # ~20 min
  python3 expC2c_crossmodel_readhead.py --model qwen3-8b     # ~12 min
  python3 expC2c_crossmodel_readhead.py --model all           # sequential
  python3 expC2c_crossmodel_readhead.py --model 7b --quick    # essential only ~5 min
"""

import json, math, time, re, sys, gc, argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

# ── Performance ──
torch.backends.cuda.matmul.allow_tf32 = True

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    '7b':       {'name': 'Qwen/Qwen2.5-7B',  'n_layers': 28, 'd': 3584, 'tied': True},
    '14b':      {'name': 'Qwen/Qwen2.5-14B', 'n_layers': 48, 'd': 5120, 'tied': False},
    'qwen3-8b': {'name': 'Qwen/Qwen3-8B',    'n_layers': 36, 'd': 4096, 'tied': False},
}

SEED = 42
K_FIXED = 20       # centered Gram rank_90 ≈ 17-21 on 3B
MAX_NEW = 128
N_TEST = 4          # per category → 20 total
N_PROBLEMS = 200    # for subspace computation
LANGS = ['ar', 'en', 'es', 'ja', 'ko', 'sw', 'zh']

# ── 7-language templates (same as BQ2 cache collection) ──
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


# ═══════════════════════════════════════════════════════════════════
# PROBLEM GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_problems(n_per_cat=40):
    """Generate math problems deterministically (seed=42).
    Same RNG sequence as C2b for identical test set.
    Returns list of 200 dicts with all 7 languages + answer + category.
    """
    rng = np.random.RandomState(SEED)
    cats = []

    # Arithmetic (40)
    for _ in range(n_per_cat):
        a, b = int(rng.randint(10, 999)), int(rng.randint(10, 999))
        op = rng.choice(['plus', 'times'])
        ans = a + b if op == 'plus' else a * b
        row = {'category': 'arithmetic', 'answer': int(ans)}
        for lang in LANGS:
            row[lang] = TEMPLATES[lang][f'arithmetic_{op}'].format(a=a, b=b)
        cats.append(row)

    # Combinatorics (40)
    for _ in range(n_per_cat):
        n_val = int(rng.randint(5, 20))
        k_val = int(rng.randint(1, min(n_val - 1, 8)))
        ans = math.comb(n_val, k_val)
        row = {'category': 'combinatorics', 'answer': int(ans)}
        for lang in LANGS:
            row[lang] = TEMPLATES[lang]['combinatorics'].format(n=n_val, k=k_val)
        cats.append(row)

    # Modular (40)
    for _ in range(n_per_cat):
        a = int(rng.randint(50, 9999))
        b = int(rng.randint(3, 37))
        ans = a % b
        row = {'category': 'modular', 'answer': int(ans)}
        for lang in LANGS:
            row[lang] = TEMPLATES[lang]['modular'].format(a=a, b=b)
        cats.append(row)

    # Geometry (40)
    for _ in range(n_per_cat):
        w = int(rng.randint(2, 50))
        h = int(rng.randint(2, 50))
        ans = w * h
        row = {'category': 'geometry', 'answer': int(ans)}
        for lang in LANGS:
            row[lang] = TEMPLATES[lang]['geometry'].format(w=w, h=h)
        cats.append(row)

    # Sequences (40)
    for _ in range(n_per_cat):
        a1 = int(rng.randint(1, 20))
        d = int(rng.randint(1, 10))
        n_terms = int(rng.randint(5, 30))
        ans = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        row = {'category': 'sequences', 'answer': int(ans)}
        for lang in LANGS:
            row[lang] = TEMPLATES[lang]['sequences'].format(a1=a1, d=d, n=n_terms)
        cats.append(row)

    # Shuffle (pyrandom, same as C2b)
    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats)))
    rng2.shuffle(indices)
    cats = [cats[i] for i in indices]
    return cats


def get_test_subset(problems, n_per_cat=N_TEST):
    """First n_per_cat problems per category (after shuffle) = test set."""
    by_cat = {}
    for p in problems:
        cat = p['category']
        by_cat.setdefault(cat, [])
        if len(by_cat[cat]) < n_per_cat:
            by_cat[cat].append(p)
    return [p for cat_probs in by_cat.values() for p in cat_probs]


# ═══════════════════════════════════════════════════════════════════
# CACHE COLLECTION & SUBSPACE
# ═══════════════════════════════════════════════════════════════════

def collect_cache(model, tokenizer, problems, n_layers, device):
    """Collect last-token hidden states at all layers via output_hidden_states.
    Also collects context-token mean for World E diagnostic.
    Returns: (last_token_cache, context_mean_cache) — both dicts of {lang_L{i}: (N, d)}
    """
    lt_cache = {}  # last-token
    ct_cache = {}  # context-mean
    n_total = len(problems) * len(LANGS)
    done = 0
    t0 = time.time()

    for lang in LANGS:
        lt_by_layer = [[] for _ in range(n_layers)]
        ct_by_layer = [[] for _ in range(n_layers)]
        for prob in problems:
            ids = tokenizer(prob[lang], return_tensors="pt").input_ids.to(device)
            seq_len = ids.shape[1]
            with torch.inference_mode():
                out = model(ids, output_hidden_states=True)
            for layer in range(n_layers):
                hs = out.hidden_states[layer + 1]  # (1, seq, d)
                lt_by_layer[layer].append(hs[0, -1, :].float().cpu().numpy())
                if seq_len > 1:
                    ct_by_layer[layer].append(hs[0, :-1, :].float().mean(dim=0).cpu().numpy())
                else:
                    ct_by_layer[layer].append(hs[0, 0, :].float().cpu().numpy())
            done += 1

        for layer in range(n_layers):
            lt_cache[f'{lang}_L{layer}'] = np.stack(lt_by_layer[layer])
            ct_cache[f'{lang}_L{layer}'] = np.stack(ct_by_layer[layer])

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 1
        remaining = (n_total - done) / rate
        print(f"    {lang}: done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)", flush=True)

    return lt_cache, ct_cache


def compute_subspace(H, k):
    """Centered PCA → top-k directions. Returns (h_mean, U_k)."""
    h_mean = H.mean(axis=0)
    H_c = H - h_mean
    N, d = H_c.shape
    if N <= d:
        G = H_c @ H_c.T
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(-eigvals)[:k]
        sigma = np.sqrt(np.maximum(eigvals[idx], 1e-12))
        U_k = (H_c.T @ eigvecs[:, idx]) / sigma[None, :]
    else:
        C = H_c.T @ H_c / N
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(-eigvals)[:k]
        U_k = eigvecs[:, idx]
    return h_mean, U_k


def compute_effective_ranks(H):
    """Compute effective rank at 50/90/95 thresholds from centered Gram."""
    h_mean = H.mean(axis=0)
    H_c = H - h_mean
    N = H_c.shape[0]
    if N <= H_c.shape[1]:
        G = H_c @ H_c.T
        eigvals = np.linalg.eigvalsh(G)
    else:
        C = H_c.T @ H_c / N
        eigvals = np.linalg.eigvalsh(C)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return {'rank_50': 0, 'rank_90': 0, 'rank_95': 0}
    cumvar = np.cumsum(eigvals) / eigvals.sum()
    return {
        'rank_50': int(np.searchsorted(cumvar, 0.5)) + 1,
        'rank_90': int(np.searchsorted(cumvar, 0.9)) + 1,
        'rank_95': int(np.searchsorted(cumvar, 0.95)) + 1,
    }


def compute_all_subspaces(lt_cache, ct_cache, n_layers, k=K_FIXED):
    """Compute subspaces + effective ranks for both last-token and context-mean."""
    subs = {}
    lt_ranks = {}
    ct_ranks = {}

    for layer in range(n_layers):
        # Last-token subspace (for hooks)
        H_lt = np.concatenate([lt_cache[f'{lang}_L{layer}'] for lang in LANGS], axis=0)
        h_mean, U_k = compute_subspace(H_lt, k)
        subs[layer] = (h_mean, U_k)
        lt_ranks[layer] = compute_effective_ranks(H_lt)

        # Context-mean ranks (World E diagnostic)
        H_ct = np.concatenate([ct_cache[f'{lang}_L{layer}'] for lang in LANGS], axis=0)
        ct_ranks[layer] = compute_effective_ranks(H_ct)

    return subs, lt_ranks, ct_ranks


# ═══════════════════════════════════════════════════════════════════
# HOOKS (identical to C2b)
# ═══════════════════════════════════════════════════════════════════

def make_last_token_hook(h_mean_t, U_k_t):
    """Zero V⊥ for LAST TOKEN only. Context tokens pass through."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        last = new_h[0, -1, :].float()
        h_c = last - h_mean_t
        h_V = U_k_t @ (U_k_t.T @ h_c)
        new_h[0, -1, :] = (h_mean_t + h_V).to(h.dtype)
        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h
    return hook


def make_all_tokens_hook(h_mean_t, U_k_t):
    """Zero V⊥ for ALL TOKENS. Matches BS methodology."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h_c = h.float() - h_mean_t
        coeffs = h_c @ U_k_t            # (batch, seq, k)
        h_V = coeffs @ U_k_t.T          # (batch, seq, d)
        new_h = (h_V + h_mean_t).to(h.dtype)
        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h
    return hook


def make_context_only_hook(h_mean_t, U_k_t):
    """Zero V⊥ for all tokens EXCEPT the last. Isolates context role."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        ctx = new_h[0, :-1, :].float()
        ctx_c = ctx - h_mean_t
        coeffs = ctx_c @ U_k_t
        ctx_V = coeffs @ U_k_t.T
        new_h[0, :-1, :] = (ctx_V + h_mean_t).to(h.dtype)
        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h
    return hook


# ═══════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_tokens(model, tokenizer, input_ids, max_new=MAX_NEW):
    """Greedy generation, no hooks."""
    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
    tok = out.logits[0, -1].argmax().item()
    tokens = [tok]
    nxt = torch.tensor([[tok]], device=input_ids.device)
    with torch.inference_mode():
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[0, -1].argmax().item()
            tokens.append(tok)
            nxt = torch.tensor([[tok]], device=input_ids.device)
            if tok == tokenizer.eos_token_id:
                break
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_with_hooks(model, tokenizer, input_ids, hooks_by_layer, max_new=MAX_NEW):
    """Forward + generate with hooks active during prefill only."""
    handles = []
    for layer, hook_fn in hooks_by_layer.items():
        handles.append(model.model.layers[layer].register_forward_hook(hook_fn))

    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    for h in handles:
        h.remove()

    tok = out.logits[0, -1].argmax().item()
    tokens = [tok]
    nxt = torch.tensor([[tok]], device=input_ids.device)
    with torch.inference_mode():
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[0, -1].argmax().item()
            tokens.append(tok)
            nxt = torch.tensor([[tok]], device=input_ids.device)
            if tok == tokenizer.eos_token_id:
                break
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def check_answer(text, correct):
    return str(correct) in re.findall(r'-?\d+\.?\d*', text)


def detect_lang(text):
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    return 'zh' if cjk > lat else ('en' if lat > 0 else 'unk')


def select_layers(n, total):
    """Select n layers evenly spaced from 0 to total-1."""
    if n >= total:
        return list(range(total))
    if n == 1:
        return [total // 2]
    return sorted(set(int(round(i * (total - 1) / (n - 1))) for i in range(n)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════

def run_model(model_key, quick=False):
    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg['n_layers']
    model_name = cfg['name']

    print(f"\n{'='*70}")
    print(f"EXP C2c: {model_name}")
    print(f"  {n_layers} layers, d={cfg['d']}, tied={cfg['tied']}")
    print(f"  Read head vs context replication (C2b methodology)")
    print(f"{'='*70}")
    t0 = time.time()

    # ── [1] Problems ──
    print("\n[1] Generating problems...", flush=True)
    all_problems = generate_problems(n_per_cat=40)
    test_problems = get_test_subset(all_problems, n_per_cat=N_TEST)
    print(f"    {len(all_problems)} for cache, {len(test_problems)} for test", flush=True)

    # ── [2] Model ──
    print(f"\n[2] Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"    Loaded. VRAM: {vram:.1f} GB ({time.time()-t0:.0f}s)", flush=True)

    # ── [3] Cache collection ──
    print(f"\n[3] Collecting activation cache ({len(all_problems)}×{len(LANGS)}={len(all_problems)*len(LANGS)} fwd passes)...", flush=True)
    lt_cache, ct_cache = collect_cache(model, tokenizer, all_problems, n_layers, device)
    print(f"    Done ({time.time()-t0:.0f}s)", flush=True)

    # ── [4] Subspaces ──
    print(f"\n[4] Computing subspaces (k={K_FIXED}) + World E diagnostic...", flush=True)
    subs_np, lt_ranks, ct_ranks = compute_all_subspaces(lt_cache, ct_cache, n_layers, K_FIXED)
    del lt_cache, ct_cache
    gc.collect()

    # Convert to GPU tensors
    subs_t = {}
    for layer in range(n_layers):
        h_mean, U_k = subs_np[layer]
        subs_t[layer] = (
            torch.from_numpy(h_mean).float().to(device),
            torch.from_numpy(U_k).float().to(device),
        )
    del subs_np
    gc.collect()

    # Print rank comparison (World E diagnostic)
    print(f"\n    WORLD E DIAGNOSTIC: last-token vs context-mean effective rank")
    print(f"    {'Layer':>6s}  {'LT_r50':>6s}  {'LT_r90':>6s}  {'CT_r50':>6s}  {'CT_r90':>6s}  {'ratio_r90':>9s}")
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    for l in sample_layers:
        lt_r90 = lt_ranks[l]['rank_90']
        ct_r90 = ct_ranks[l]['rank_90']
        ratio = ct_r90 / lt_r90 if lt_r90 > 0 else float('inf')
        print(f"    L{l:>3d}    {lt_ranks[l]['rank_50']:>5d}   {lt_r90:>5d}   "
              f"{ct_ranks[l]['rank_50']:>5d}   {ct_r90:>5d}   {ratio:>8.1f}x", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

    # ── [5] Baselines ──
    print(f"\n[5] Baselines...", flush=True)
    baselines = {}
    for lang in ['en', 'zh']:
        correct = 0
        details = []
        for p in test_problems:
            ids = tokenizer(p[lang], return_tensors="pt").input_ids.to(device)
            text = generate_tokens(model, tokenizer, ids, MAX_NEW)
            ok = check_answer(text, p['answer'])
            if ok:
                correct += 1
            details.append({'ok': ok, 'gen': text[:150], 'cat': p['category']})
        baselines[lang] = {'correct': correct, 'total': len(test_problems), 'details': details}
        print(f"    {lang.upper()}: {correct}/{len(test_problems)}", flush=True)

    # ── [6] Build conditions ──
    mid = n_layers // 2
    conditions = []

    if quick:
        # Essential 3 conditions
        lc_last = [n_layers]
        lc_all = [1]
    else:
        # Full dose-response (deduplicated)
        lc_last = sorted(set([1, 3, max(1, n_layers // 4), max(1, n_layers // 2), n_layers]))
        lc_all = sorted(set([1, 3, max(1, n_layers // 4), max(1, n_layers // 2), n_layers]))

    for n_l in lc_last:
        conditions.append({
            'name': f'last_only_N{n_l}',
            'mode': 'last_only',
            'layers': select_layers(n_l, n_layers),
            'n_layers': n_l,
        })
    for n_l in lc_all:
        conditions.append({
            'name': f'all_tokens_N{n_l}',
            'mode': 'all_tokens',
            'layers': select_layers(n_l, n_layers),
            'n_layers': n_l,
        })

    # Context-only
    conditions.append({
        'name': f'context_only_L{mid}',
        'mode': 'context_only',
        'layers': [mid],
        'n_layers': 1,
    })
    if not quick:
        n_ctx = max(1, n_layers // 4)
        conditions.append({
            'name': f'context_only_N{n_ctx}',
            'mode': 'context_only',
            'layers': select_layers(n_ctx, n_layers),
            'n_layers': n_ctx,
        })

    hook_factories = {
        'last_only': make_last_token_hook,
        'all_tokens': make_all_tokens_hook,
        'context_only': make_context_only_hook,
    }

    # ── [7] Run conditions ──
    n_evals = len(conditions) * len(test_problems) * 2
    print(f"\n[6] Running {len(conditions)} conditions × {len(test_problems)*2} evals = {n_evals} generations...", flush=True)
    all_results = {}

    # Incremental save path — survives tunnel drops
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)
    partial_path = out_dir / f'expC2c_{model_key}_partial.json'

    def save_partial():
        partial = {
            'config': {
                'model': model_name, 'model_key': model_key,
                'n_layers': n_layers, 'd': cfg['d'], 'tied': cfg['tied'],
                'k': K_FIXED, 'max_new': MAX_NEW, 'seed': SEED,
            },
            'baselines': {lang: baselines[lang]['correct'] for lang in baselines},
            'rank_trajectory': {
                'last_token': {f'L{l}': lt_ranks[l] for l in range(n_layers)},
                'context_mean': {f'L{l}': ct_ranks[l] for l in range(n_layers)},
            },
            'conditions_completed': len(all_results),
            'conditions_total': len(conditions),
            'conditions': all_results,
            'elapsed_s_so_far': time.time() - t0,
        }
        with open(partial_path, 'w') as f:
            json.dump(partial, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

    for ci, cond in enumerate(conditions):
        name = cond['name']
        factory = hook_factories[cond['mode']]
        hooks = {l: factory(*subs_t[l]) for l in cond['layers']}

        results_by_lang = {}
        for lang in ['en', 'zh']:
            correct = 0
            details = []
            for i, p in enumerate(test_problems):
                ids = tokenizer(p[lang], return_tensors="pt").input_ids.to(device)
                text = generate_with_hooks(model, tokenizer, ids, hooks, MAX_NEW)
                ok = check_answer(text, p['answer'])
                if ok:
                    correct += 1
                details.append({
                    'i': i, 'ok': ok, 'gen': text[:120],
                    'lang_out': detect_lang(text), 'cat': p['category'],
                })
            results_by_lang[lang] = {
                'correct': correct, 'total': len(test_problems), 'details': details,
            }

        all_results[name] = {
            'mode': cond['mode'],
            'n_layers': cond['n_layers'],
            'layers': cond['layers'],
            'en': results_by_lang['en'],
            'zh': results_by_lang['zh'],
        }

        en_c = results_by_lang['en']['correct']
        zh_c = results_by_lang['zh']['correct']
        n_test = len(test_problems)
        print(f"    [{ci+1:2d}/{len(conditions)}] {name:25s}  "
              f"EN={en_c:>2}/{n_test}  ZH={zh_c:>2}/{n_test}  "
              f"({time.time()-t0:.0f}s)", flush=True)

        # Save incrementally — survives tunnel drops
        save_partial()

    # ── [8] Summary ──
    elapsed = time.time() - t0
    n_test = len(test_problems)

    print(f"\n{'='*70}")
    print(f"RESULTS: {model_name} ({model_key})")
    print(f"{'='*70}")
    print(f"Baselines: EN={baselines['en']['correct']}/{n_test}, ZH={baselines['zh']['correct']}/{n_test}")
    print(f"\n{'Condition':>25s} | {'Mode':>12s} | {'N':>4s} | {'EN':>6s} | {'ZH':>6s}")
    print("-" * 68)
    for name, r in all_results.items():
        print(f"  {name:>23s} | {r['mode']:>12s} | {r['n_layers']:>4d} | "
              f"{r['en']['correct']:>2}/{n_test}  | {r['zh']['correct']:>2}/{n_test}")

    # Key diagnostic
    print(f"\n── KEY TEST ──")
    if f'last_only_N{n_layers}' in all_results:
        lo = all_results[f'last_only_N{n_layers}']
        print(f"  last_only_N{n_layers}: EN={lo['en']['correct']}/{n_test}, ZH={lo['zh']['correct']}/{n_test}")
        print(f"    → {'PASS (≈baseline)' if abs(lo['en']['correct'] - baselines['en']['correct']) <= 2 and abs(lo['zh']['correct'] - baselines['zh']['correct']) <= 2 else 'CHECK'}: last token V⊥ irrelevant")
    if 'all_tokens_N1' in all_results:
        at = all_results['all_tokens_N1']
        print(f"  all_tokens_N1:    EN={at['en']['correct']}/{n_test}, ZH={at['zh']['correct']}/{n_test}")
        print(f"    → {'PASS (≈0)' if at['en']['correct'] <= 2 and at['zh']['correct'] <= 2 else 'CHECK'}: single-layer all-token truncation kills accuracy")
    if f'context_only_L{mid}' in all_results:
        co = all_results[f'context_only_L{mid}']
        print(f"  context_only_L{mid}: EN={co['en']['correct']}/{n_test}, ZH={co['zh']['correct']}/{n_test}")
        print(f"    → {'PASS (≈0)' if co['en']['correct'] <= 2 and co['zh']['correct'] <= 2 else 'CHECK'}: context IS the computation")

    # World E summary
    print(f"\n── WORLD E: context-mean rank vs last-token rank ──")
    mid_lt = lt_ranks[mid]['rank_90']
    mid_ct = ct_ranks[mid]['rank_90']
    print(f"  Mid-layer (L{mid}): last-token rank_90={mid_lt}, context-mean rank_90={mid_ct}, ratio={mid_ct/mid_lt:.1f}x")
    print(f"  → {'CONFIRMS' if mid_ct > mid_lt * 1.5 else 'WEAK/CHECK'}: context tokens live in higher-rank space than read head")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # ── [9] Save ──
    output = {
        'config': {
            'model': model_name, 'model_key': model_key,
            'n_layers': n_layers, 'd': cfg['d'], 'tied': cfg['tied'],
            'k': K_FIXED, 'max_new': MAX_NEW, 'seed': SEED,
            'n_cache_problems': len(all_problems), 'n_test_problems': n_test,
        },
        'baselines': {lang: baselines[lang]['correct'] for lang in baselines},
        'baseline_details': baselines,
        'rank_trajectory': {
            'last_token': {f'L{l}': lt_ranks[l] for l in range(n_layers)},
            'context_mean': {f'L{l}': ct_ranks[l] for l in range(n_layers)},
        },
        'conditions': all_results,
        'elapsed_s': elapsed,
    }

    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'expC2c_{model_key}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"Saved to {out_path}")

    # Cleanup
    del model, tokenizer, subs_t
    gc.collect()
    torch.cuda.empty_cache()

    return output


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Cross-model C2b replication: read head vs context')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()) + ['all'],
                        default='7b', help='Model to test (default: 7b)')
    parser.add_argument('--quick', action='store_true',
                        help='Essential conditions only (3 instead of ~12)')
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == 'all' else [args.model]

    all_outputs = {}
    for mk in models:
        all_outputs[mk] = run_model(mk, quick=args.quick)

    # Cross-model comparison if multiple
    if len(all_outputs) > 1:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"{'Model':>15s} | {'Base EN':>7s} | {'Base ZH':>7s} | {'LO_all':>10s} | {'AT_1':>10s} | {'CO_mid':>10s}")
        print("-" * 75)
        for mk, out in all_outputs.items():
            cfg = MODEL_CONFIGS[mk]
            n_l = cfg['n_layers']
            mid = n_l // 2
            b_en = out['baselines']['en']
            b_zh = out['baselines']['zh']
            lo = out['conditions'].get(f'last_only_N{n_l}', {})
            at = out['conditions'].get('all_tokens_N1', {})
            co = out['conditions'].get(f'context_only_L{mid}', {})
            lo_s = f"{lo.get('en',{}).get('correct','?')}/{lo.get('zh',{}).get('correct','?')}" if lo else "N/A"
            at_s = f"{at.get('en',{}).get('correct','?')}/{at.get('zh',{}).get('correct','?')}" if at else "N/A"
            co_s = f"{co.get('en',{}).get('correct','?')}/{co.get('zh',{}).get('correct','?')}" if co else "N/A"
            print(f"  {cfg['name']:>13s} | {b_en:>3}/20  | {b_zh:>3}/20  | {lo_s:>10s} | {at_s:>10s} | {co_s:>10s}")


if __name__ == '__main__':
    main()
