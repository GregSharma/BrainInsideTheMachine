"""Exp C2: Tail Transplant — Is V⊥ generic infrastructure or problem-specific?

Central question: The centered Gram matrix shows ~20-D inter-problem geometry.
SVD truncation to k=500 (99.9% variance) kills accuracy (BS). C1/C1b show this
isn't about hidden rank or gate control — it's error amplification.

This experiment asks: is the V⊥ "tail" (directions outside the Gram's top-k
subspace) generic infrastructure that any problem can use, or does it carry
problem-specific information invisible to the Gram?

Method: At layer L, decompose hidden states into V (top-k Gram eigenvectors)
and V⊥. Swap V⊥ between problems and measure accuracy.

Conditions:
  cross:  keep problem i's V, use problem j's V⊥ (real tail from another problem)
  zero:   keep problem i's V, set V⊥ = 0 (pure projection into V)
  noise:  keep problem i's V, V⊥ = random noise restricted to V⊥ (matched norm)

Interpretation:
  cross ≈ baseline, zero < baseline → tail is generic infrastructure
  cross ≈ zero ≈ 0 → tail is problem-specific
  noise ≈ cross → any valid-norm V⊥ works (manifold not required)
  noise < cross → need a "natural" tail (manifold matters)
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
N_TOTAL_CACHE = N_PROB_CACHE * len(LANGS_CACHE)  # 1400

PROBE_LAYERS = [9, 20, 30]
K_VALUES = [5, 20, 50]
N_DONORS = 3
N_TEST = 4          # per category → 20 total
MAX_NEW = 128

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
    """Same generator as AG2 for cross-experiment consistency."""
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
    """Stack all 1400 cached hidden states at a layer → (1400, 2048)."""
    return np.concatenate(
        [cache[f'{lang}_L{layer}'] for lang in LANGS_CACHE], axis=0
    )


def compute_subspace(H, k):
    """Centered PCA → top-k directions in R^d.
    Returns: h_mean (d,), U_k (d, k)
    """
    h_mean = H.mean(axis=0)
    H_c = H - h_mean
    # N×N Gram is more efficient when N < d (1400 < 2048)
    G = H_c @ H_c.T
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx[:k]]
    eigvecs = eigvecs[:, idx[:k]]
    sigma = np.sqrt(np.maximum(eigvals, 1e-12))
    U_k = (H_c.T @ eigvecs) / sigma[None, :]
    return h_mean, U_k


def decompose(h, h_mean, U_k):
    """Split h into V-component and V⊥-component (both centered)."""
    h_c = h - h_mean
    h_V = U_k @ (U_k.T @ h_c)
    h_perp = h_c - h_V
    return h_V, h_perp


# ── Model interaction ───────────────────────────────────────────────

def generate_tokens(model, tokenizer, input_ids, max_new=128):
    """Greedy generation, no hooks."""
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


def capture_all_layers(model, input_ids, layers):
    """Single forward pass → last-token hidden states at all probe layers."""
    caps = {l: [None] for l in layers}
    handles = []

    for l in layers:
        def make_hook(cap):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[0] = h[0, -1, :].float().cpu().numpy()
                return out
            return hook
        handles.append(
            model.model.layers[l].register_forward_hook(make_hook(caps[l]))
        )

    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()

    return {l: caps[l][0] for l in layers}


def generate_transplanted(model, tokenizer, input_ids, layer, h_new, max_new=128):
    """Forward + generate with last-token hidden state replaced at `layer`."""
    h_tensor = torch.tensor(h_new, dtype=torch.float16)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        new_h = h.clone()
        new_h[0, -1, :] = h_tensor.to(h.device)
        return (new_h,) + out[1:] if isinstance(out, tuple) else new_h

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
    handle.remove()

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


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXP C2: TAIL TRANSPLANT")
    print("  Is V⊥ generic infrastructure or problem-specific information?")
    print("=" * 70)

    t0 = time.time()
    rng = np.random.RandomState(SEED)

    # ── [1] Precompute subspaces from cache ──
    print("\n[1] Loading cache and computing subspaces...")
    cache = np.load(CACHE_PATH)
    subs = {}   # layer → k → {h_mean, U_k, H_perp, perp_norms}
    diag = {}   # layer → k → variance fractions

    for layer in PROBE_LAYERS:
        H = load_layer_states(cache, layer)
        subs[layer] = {}
        diag[layer] = {}
        for k in K_VALUES:
            h_mean, U_k = compute_subspace(H, k)
            H_c = H - h_mean
            H_V = (H_c @ U_k) @ U_k.T
            H_perp = H_c - H_V

            var_V = np.mean(np.sum(H_V ** 2, axis=1))
            var_perp = np.mean(np.sum(H_perp ** 2, axis=1))
            v_frac = var_V / (var_V + var_perp)

            subs[layer][k] = {
                'h_mean': h_mean, 'U_k': U_k,
                'H_perp': H_perp,
                'perp_norms': np.linalg.norm(H_perp, axis=1),
            }
            diag[layer][k] = {
                'var_V_frac': float(v_frac),
                'mean_perp_norm': float(np.mean(np.linalg.norm(H_perp, axis=1))),
            }
            print(f"  L{layer} k={k:>3d}: V captures {v_frac:.1%} of centered variance")

    del cache  # free ~250 MB

    # ── [2] Test problems ──
    print("\n[2] Generating test problems...")
    problems = generate_test_problems(N_TEST)
    print(f"  {len(problems)} problems (4 per category × 5 categories)")

    # ── [3] Model ──
    print("\n[3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ── [4] Baselines ──
    print("\n[4] Baselines...")
    baselines = {}
    for lang in ['en', 'zh']:
        correct = 0
        details = []
        for i, p in enumerate(problems):
            ids = tokenizer(p[lang], return_tensors="pt").input_ids.to(model.device)
            text = generate_tokens(model, tokenizer, ids, MAX_NEW)
            ok = check_answer(text, p['answer'])
            if ok:
                correct += 1
            details.append({
                'i': i, 'cat': p['category'], 'ans': p['answer'],
                'ok': ok, 'gen': text[:150], 'lang_out': detect_lang(text),
            })
        baselines[lang] = {'correct': correct, 'total': len(problems), 'details': details}
        print(f"  {lang.upper()}: {correct}/{len(problems)}")

    # ── [5] Sanity: self-reconstruction at L20 k=20 ──
    print("\n[5] Sanity check: self-reconstruction (should match baseline)...")
    sub = subs[20][20]
    n_match = 0
    for lang in ['en']:
        for i, p in enumerate(problems):
            ids = tokenizer(p[lang], return_tensors="pt").input_ids.to(model.device)
            h_live = capture_all_layers(model, ids, [20])
            h_V, h_perp = decompose(h_live[20], sub['h_mean'], sub['U_k'])
            h_recon = (sub['h_mean'] + h_V + h_perp).astype(np.float32)
            text = generate_transplanted(model, tokenizer, ids, 20, h_recon, MAX_NEW)
            ok = check_answer(text, p['answer'])
            baseline_ok = baselines['en']['details'][i]['ok']
            if ok == baseline_ok:
                n_match += 1
    print(f"  Self-reconstruct matches baseline: {n_match}/20")
    if n_match < 18:
        print("  WARNING: decomposition may have numerical issues!")

    # ── [6] Main experiment ──
    print("\n[6] Running transplant conditions...")
    all_runs = []   # flat list, aggregate later
    n_done = 0
    n_total = (len(problems) * 2 * len(PROBE_LAYERS) * len(K_VALUES)
               * (N_DONORS + 2))  # cross×donors + zero + noise

    for lang in ['en', 'zh']:
        for i, p in enumerate(problems):
            ids = tokenizer(p[lang], return_tensors="pt").input_ids.to(model.device)
            h_live_all = capture_all_layers(model, ids, PROBE_LAYERS)

            for layer in PROBE_LAYERS:
                h_live = h_live_all[layer]

                for k in K_VALUES:
                    sub = subs[layer][k]
                    h_mean = sub['h_mean']
                    U_k = sub['U_k']
                    H_perp_pool = sub['H_perp']

                    h_V, h_perp = decompose(h_live, h_mean, U_k)
                    perp_norm = float(np.linalg.norm(h_perp))

                    # ── Cross-transplant ──
                    donors = rng.choice(N_TOTAL_CACHE, N_DONORS, replace=False)
                    for d_idx in donors:
                        h_new = (h_mean + h_V + H_perp_pool[d_idx]).astype(np.float32)
                        text = generate_transplanted(
                            model, tokenizer, ids, layer, h_new, MAX_NEW
                        )
                        ok = check_answer(text, p['answer'])
                        d_lang = LANGS_CACHE[d_idx // N_PROB_CACHE]
                        all_runs.append({
                            'cond': 'cross', 'layer': layer, 'k': k,
                            'prob': i, 'lang': lang, 'cat': p['category'],
                            'ans': p['answer'], 'ok': ok,
                            'donor': int(d_idx), 'donor_lang': d_lang,
                            'gen': text[:120],
                        })
                        n_done += 1

                    # ── Zero-perp ──
                    h_new = (h_mean + h_V).astype(np.float32)
                    text = generate_transplanted(
                        model, tokenizer, ids, layer, h_new, MAX_NEW
                    )
                    ok = check_answer(text, p['answer'])
                    all_runs.append({
                        'cond': 'zero', 'layer': layer, 'k': k,
                        'prob': i, 'lang': lang, 'cat': p['category'],
                        'ans': p['answer'], 'ok': ok,
                        'gen': text[:120],
                    })
                    n_done += 1

                    # ── Noise-perp (random in V⊥, matched norm) ──
                    noise = rng.randn(h_mean.shape[0]).astype(np.float32)
                    noise_V = U_k @ (U_k.T @ noise)
                    noise_perp = noise - noise_V
                    noise_perp *= perp_norm / (np.linalg.norm(noise_perp) + 1e-12)
                    h_new = (h_mean + h_V + noise_perp).astype(np.float32)
                    text = generate_transplanted(
                        model, tokenizer, ids, layer, h_new, MAX_NEW
                    )
                    ok = check_answer(text, p['answer'])
                    all_runs.append({
                        'cond': 'noise', 'layer': layer, 'k': k,
                        'prob': i, 'lang': lang, 'cat': p['category'],
                        'ans': p['answer'], 'ok': ok,
                        'gen': text[:120],
                    })
                    n_done += 1

            # Progress
            if (i + 1) % 5 == 0 or i == 0:
                el = time.time() - t0
                # exclude first 30s of setup from rate calc
                rate = n_done / max(el - 30, 1)
                eta = (n_total - n_done) / max(rate, 0.01)
                print(f"  [{lang.upper()} {i + 1:>2}/{len(problems)}] "
                      f"{n_done:>5}/{n_total} runs  "
                      f"{el:>6.0f}s elapsed  ~{eta:.0f}s left")

    # ── [7] Aggregate and print summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    b_en = baselines['en']['correct']
    b_zh = baselines['zh']['correct']
    print(f"\nBaselines: EN={b_en}/20, ZH={b_zh}/20")

    header = (f"{'':>12} | {'cross_EN':>10} {'cross_ZH':>10} | "
              f"{'zero_EN':>9} {'zero_ZH':>9} | "
              f"{'noise_EN':>9} {'noise_ZH':>9}")
    print(f"\n{header}")
    print("-" * len(header))

    for layer in PROBE_LAYERS:
        for k in K_VALUES:
            tag = f"L{layer}_k{k}"
            cells = []
            for cond in ['cross', 'zero', 'noise']:
                for lang in ['en', 'zh']:
                    subset = [r for r in all_runs
                              if r['layer'] == layer and r['k'] == k
                              and r['cond'] == cond and r['lang'] == lang]
                    c = sum(r['ok'] for r in subset)
                    t = len(subset)
                    cells.append(f"{c:>3}/{t:<3}")
            print(f"  {tag:>10} | {cells[0]} {cells[1]}   | "
                  f"{cells[2]} {cells[3]}  | "
                  f"{cells[4]} {cells[5]}")

    # Cross-transplant: same-lang donor vs cross-lang donor breakdown
    print("\n── Cross-transplant: same-lang vs cross-lang donor ──")
    for layer in PROBE_LAYERS:
        for k in K_VALUES:
            cross = [r for r in all_runs
                     if r['layer'] == layer and r['k'] == k and r['cond'] == 'cross']
            same = [r for r in cross if r['donor_lang'] == r['lang']]
            diff = [r for r in cross if r['donor_lang'] != r['lang']]
            s_acc = sum(r['ok'] for r in same) / max(len(same), 1)
            d_acc = sum(r['ok'] for r in diff) / max(len(diff), 1)
            print(f"  L{layer}_k{k}: same_lang={s_acc:.2f} ({len(same)}), "
                  f"cross_lang={d_acc:.2f} ({len(diff)})")

    # ── [8] Save ──
    elapsed = time.time() - t0
    output = {
        'config': {
            'model': MODEL_NAME, 'probe_layers': PROBE_LAYERS,
            'k_values': K_VALUES, 'n_donors': N_DONORS,
            'n_test_per_cat': N_TEST, 'max_new': MAX_NEW, 'seed': SEED,
        },
        'baselines': baselines,
        'diagnostics': diag,
        'runs': all_runs,
        'elapsed_s': elapsed,
    }
    out_path = OUTPUT_DIR / 'expC2_tail_transplant.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
