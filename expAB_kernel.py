"""
Exp AB: Math Kernel Extraction via Multilingual Null Space

Greg's kernel idea: run the same math problem in N languages simultaneously.
The subspace INVARIANT across all languages = pure math representation.
This is the null space of the "language operator" — richer than the 1D EN-ZH mean diff.

Pipeline:
1. Extract all-layer activations for 7 languages × 200 problems
   (cached in output/multilingual_all_layers.npz if exists)
2. At each layer, compute within-problem cross-language covariance:
   For each problem i, form matrix X_i = [h_zh, h_en, h_es, ...] (7 × 2048)
   Stack across all problems: language variation matrix
3. SVD to find top language axes (these span the "language subspace")
4. Project all hidden states onto the null space (orthogonal complement)
5. Rerun linear probe (EN→ZH transfer) in kernel-projected space
   Baseline (Exp Z): category=1.0 from L4, answer=0.35 at L2

Key question: does answer transfer improve above 0.35 in the kernel subspace?
If yes: the EN-ZH probe was missing signal that the full 7-language kernel unlocks.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random as pyrandom
import gc

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42

LANGUAGES = ['zh', 'en', 'es', 'ar', 'ja', 'ko', 'sw']

# Language templates — same math problem in each language
# Same 5 categories as the base dataset (generate_problems seed=42)
TEMPLATES = {
    'zh': {
        'arithmetic_plus': "计算 {a} + {b} 的值。",
        'arithmetic_times': "计算 {a} × {b} 的值。",
        'combinatorics': "求组合数 C({n}, {k}) 的值。",
        'modular': "{a} 除以 {b} 的余数是多少？",
        'geometry': "一个长方形的长为 {w}，宽为 {h}，求其面积。",
        'sequences': "等差数列首项为 {a1}，公差为 {d}，求前 {n} 项之和。",
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
        'modular': "¿Cuál es el resto cuando {a} se divide entre {b}?",
        'geometry': "Un rectángulo tiene largo {w} y ancho {h}. Encuentra su área.",
        'sequences': "Una sucesión aritmética tiene primer término {a1} y diferencia común {d}. Encuentra la suma de los primeros {n} términos.",
    },
    'ar': {
        'arithmetic_plus': "احسب {a} + {b}.",
        'arithmetic_times': "احسب {a} × {b}.",
        'combinatorics': "أوجد قيمة C({n}, {k}).",
        'modular': "ما هو باقي قسمة {a} على {b}؟",
        'geometry': "مستطيل طوله {w} وعرضه {h}. أوجد مساحته.",
        'sequences': "متتالية حسابية أول حد فيها {a1} وأساسها {d}. أوجد مجموع أول {n} حدود.",
    },
    'ja': {
        'arithmetic_plus': "{a} + {b} を計算せよ。",
        'arithmetic_times': "{a} × {b} を計算せよ。",
        'combinatorics': "C({n}, {k}) の値を求めよ。",
        'modular': "{a} を {b} で割ったときの余りを求めよ。",
        'geometry': "縦 {w}、横 {h} の長方形の面積を求めよ。",
        'sequences': "初項 {a1}、公差 {d} の等差数列の初め {n} 項の和を求めよ。",
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
    """Extract last-token hidden states for all 7 languages × all 36 layers."""
    N = len(problems)
    # {lang: {layer: (N, d)}}
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


def compute_language_kernel(acts_by_lang, n_layers, d, n_lang_dims=10):
    """
    At each layer, find the null space of the language operator.

    Method:
    - Stack within-problem cross-language differences: for each problem i,
      compute centered deviations from problem mean: h_lang_i - mean_across_langs_i
    - These 7*200 deviation vectors span the language subspace
    - SVD: top n_lang_dims singular vectors = language axes
    - Null space = complement = kernel projector P = I - U U^T

    Returns:
    - kernel_projectors: list of (2048, 2048) projectors per layer
    - lang_var_explained: variance explained by top n_lang_dims dims per layer
    """
    kernel_projectors = []
    lang_var_explained = []

    for l in tqdm(range(n_layers), desc="Computing kernel per layer"):
        # Stack all languages at this layer: (7*200, 2048)
        stacked = np.vstack([acts_by_lang[lang][l] for lang in LANGUAGES])  # (1400, 2048)

        # For each problem, center within-language (remove problem mean)
        # Reshape to (200, 7, 2048), compute per-problem mean, subtract
        per_lang = np.stack([acts_by_lang[lang][l] for lang in LANGUAGES], axis=1)  # (200, 7, 2048)
        prob_means = per_lang.mean(axis=1, keepdims=True)  # (200, 1, 2048)
        deviations = (per_lang - prob_means).reshape(-1, d)  # (1400, 2048) — pure language variation

        # SVD of within-problem language deviations
        # Top singular vectors = language axes
        U, S, Vt = np.linalg.svd(deviations, full_matrices=False)
        lang_axes = Vt[:n_lang_dims]  # (n_lang_dims, 2048) — top language directions

        # Variance explained
        total_var = (S**2).sum()
        lang_var = (S[:n_lang_dims]**2).sum() / total_var if total_var > 0 else 0
        lang_var_explained.append(float(lang_var))

        # Kernel projector: I - U U^T where U = lang_axes^T
        # Projects out the language subspace
        U_mat = lang_axes.T  # (2048, n_lang_dims)
        P_lang = np.eye(d, dtype=np.float32) - (U_mat @ U_mat.T).astype(np.float32)
        kernel_projectors.append(P_lang)

    return kernel_projectors, lang_var_explained


def run_probe_in_kernel(acts_by_lang, kernel_projectors, problems, n_layers, n_pca=64,
                        train_lang='en', test_lang='zh'):
    """
    Replicate Exp Z probe but in kernel-projected space.

    Exp Z baseline: category=1.0 from L4, answer=0.35 at L2.
    If kernel helps: answer accuracy should increase.

    Labels: (a) category 0-4, (b) answer log-bucket (20 bins)
    Train on train_lang, test on test_lang (cross-lingual transfer).
    """
    categories = np.array([p['category'] for p in problems])
    N = len(problems)

    # Compute answer labels (same as Exp Z)
    rng = pyrandom.Random(SEED)
    answers = []
    per_cat = N // 5
    # Re-generate to get exact answers (parallel to generate_problems_multilingual)
    # Just use categories × index to back out the answer
    # Actually easier: rerun the generator and compute answers
    answer_vals = []
    rng2 = pyrandom.Random(SEED)
    temp_problems = []
    for _ in range(per_cat):
        a, b = rng2.randint(10, 999), rng2.randint(10, 999)
        op = rng2.choice(["plus", "times"])
        temp_problems.append(a + b if op == "plus" else a * b)
    for _ in range(per_cat):
        n_val = rng2.randint(5, 20)
        k_val = rng2.randint(1, min(n_val - 1, 8))
        import math
        temp_problems.append(math.comb(n_val, k_val))
    for _ in range(per_cat):
        a = rng2.randint(50, 9999)
        b = rng2.randint(3, 37)
        temp_problems.append(a % b)
    for _ in range(per_cat):
        w = rng2.randint(2, 50)
        h = rng2.randint(2, 50)
        temp_problems.append(w * h)
    for _ in range(per_cat):
        a1 = rng2.randint(1, 20)
        d = rng2.randint(1, 10)
        n_terms = rng2.randint(5, 30)
        temp_problems.append(n_terms * (2 * a1 + (n_terms - 1) * d) // 2)
    # NOTE: temp_problems is pre-shuffle. We need the shuffled order.
    # The shuffle is applied in generate_problems_multilingual but we can't recover
    # the exact shuffle mapping easily. Use categories as the label instead.
    # For answer, just use the problem index within category (ordinal rank in log space)

    # Simpler: use log-bucketed answer from the known formula per problem.
    # Re-derive answers in shuffled order by matching categories.
    # Actually: the SHUFFLE seed is fixed (same rng), so we can just re-run it.
    all_problems_ordered = []
    rng3 = pyrandom.Random(SEED)
    import math
    for _ in range(per_cat):
        a, b = rng3.randint(10, 999), rng3.randint(10, 999)
        op = rng3.choice(["plus", "times"])
        all_problems_ordered.append(a + b if op == "plus" else a * b)
    for _ in range(per_cat):
        n_val = rng3.randint(5, 20)
        k_val = rng3.randint(1, min(n_val - 1, 8))
        all_problems_ordered.append(math.comb(n_val, k_val))
    for _ in range(per_cat):
        a = rng3.randint(50, 9999)
        b = rng3.randint(3, 37)
        all_problems_ordered.append(a % b)
    for _ in range(per_cat):
        w = rng3.randint(2, 50)
        h = rng3.randint(2, 50)
        all_problems_ordered.append(w * h)
    for _ in range(per_cat):
        a1 = rng3.randint(1, 20)
        d_val = rng3.randint(1, 10)
        n_terms = rng3.randint(5, 30)
        all_problems_ordered.append(n_terms * (2 * a1 + (n_terms - 1) * d_val) // 2)
    # shuffle with same rng
    indices = list(range(N))
    rng3.shuffle(indices)
    answers_shuffled = [all_problems_ordered[i] for i in indices]
    log_answers = np.array([np.log1p(max(0, a)) for a in answers_shuffled])
    # 20 equal-width bins on log scale
    bins = np.linspace(log_answers.min(), log_answers.max(), 21)
    answer_bins = np.digitize(log_answers, bins[1:-1])  # 0..19

    # Train/test split: 160/40 (same as Exp Z)
    train_idx = np.arange(160)
    test_idx = np.arange(160, 200)

    cat_accs_raw = []        # category accuracy without kernel (baseline)
    cat_accs_kernel = []     # category accuracy in kernel
    ans_accs_raw = []        # answer accuracy without kernel
    ans_accs_kernel = []     # answer accuracy in kernel

    for l in range(n_layers):
        h_train = acts_by_lang[train_lang][l][train_idx]
        h_test = acts_by_lang[test_lang][l][test_idx]

        # === RAW (no kernel) ===
        acc_cat_raw, acc_ans_raw = _probe(
            h_train, h_test, categories[train_idx], categories[test_idx],
            answer_bins[train_idx], answer_bins[test_idx], n_pca
        )
        cat_accs_raw.append(acc_cat_raw)
        ans_accs_raw.append(acc_ans_raw)

        # === KERNEL PROJECTED ===
        P = kernel_projectors[l]  # (2048, 2048)
        h_train_k = (h_train @ P.T)
        h_test_k = (h_test @ P.T)
        acc_cat_k, acc_ans_k = _probe(
            h_train_k, h_test_k, categories[train_idx], categories[test_idx],
            answer_bins[train_idx], answer_bins[test_idx], n_pca
        )
        cat_accs_kernel.append(acc_cat_k)
        ans_accs_kernel.append(acc_ans_k)

    return {
        'cat_raw': cat_accs_raw,
        'cat_kernel': cat_accs_kernel,
        'ans_raw': ans_accs_raw,
        'ans_kernel': ans_accs_kernel,
    }


def _probe(h_train, h_test, cat_train, cat_test, ans_train, ans_test, n_pca=64):
    """PCA-64 + LogReg probe for both category and answer bucket."""
    try:
        scaler = StandardScaler()
        h_tr_s = scaler.fit_transform(h_train)
        h_te_s = scaler.transform(h_test)

        pca = PCA(n_components=min(n_pca, h_tr_s.shape[0] - 1, h_tr_s.shape[1]))
        h_tr_p = pca.fit_transform(h_tr_s)
        h_te_p = pca.transform(h_te_s)

        # Category probe
        clf_cat = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf_cat.fit(h_tr_p, cat_train)
        acc_cat = clf_cat.score(h_te_p, cat_test)

        # Answer bucket probe
        clf_ans = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf_ans.fit(h_tr_p, ans_train)
        acc_ans = clf_ans.score(h_te_p, ans_test)

        return acc_cat, acc_ans
    except Exception as e:
        return 0.0, 0.0


def main():
    cache_path = OUTPUT_DIR / "multilingual_all_layers.npz"

    # ── Step 1: Load or extract multilingual all-layer activations ──
    if cache_path.exists():
        print(f"Loading cached multilingual activations from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        # Reconstruct {lang: {layer: array}}
        n_layers = 36
        d = 2048
        acts = {lang: {l: data[f"{lang}_L{l}"] for l in range(n_layers)}
                for lang in LANGUAGES}
        problems = None  # will regenerate for labels
        categories = data['categories']
        print(f"Loaded. Languages: {LANGUAGES}, Layers: {n_layers}, N=200")
    else:
        print(f"Extracting multilingual activations (7 languages × 36 layers × 200 problems)...")
        print(f"Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
            trust_remote_code=True
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        d = model.config.hidden_size
        print(f"Model: {n_layers} layers, d={d}")

        problems_list = generate_problems_multilingual(200, seed=SEED)
        categories = np.array([p['category'] for p in problems_list])

        acts = extract_multilingual_all_layers(model, tokenizer, problems_list, n_layers, d)

        # Save cache
        print("Saving cache...")
        save_dict = {'categories': categories}
        for lang in LANGUAGES:
            for l in range(n_layers):
                save_dict[f"{lang}_L{l}"] = acts[lang][l]
        np.savez_compressed(cache_path, **save_dict)
        filesize = cache_path.stat().st_size / 1e6
        print(f"Saved {cache_path} ({filesize:.1f} MB)")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Regenerate problems for label computation
    problems_list = generate_problems_multilingual(200, seed=SEED)
    n_layers = 36
    d = 2048

    # ── Step 2: Compute language kernel projectors at each layer ──
    print(f"\nComputing language kernel (null space of {len(LANGUAGES)}-language operator)...")
    N_LANG_DIMS = 10  # project out top 10 language directions (matches Exp 7 finding: ~5-10 dims span language)
    kernel_projectors, lang_var = compute_language_kernel(acts, n_layers, d, n_lang_dims=N_LANG_DIMS)

    print(f"\nLanguage variance explained by top {N_LANG_DIMS} dims (per layer):")
    for l in [0, 4, 8, 13, 17, 18, 22, 26, 30, 35]:
        print(f"  L{l:2d}: {lang_var[l]:.3f}")

    # ── Step 3: Run probe in raw vs kernel-projected space ──
    print(f"\nRunning EN→ZH cross-lingual probe (raw vs kernel)...")
    probe_results = run_probe_in_kernel(acts, kernel_projectors, problems_list, n_layers)

    # ── Step 4: Print results ──
    print("\n" + "="*70)
    print("EXP AB: MATH KERNEL — RESULTS")
    print("="*70)
    print(f"{'Layer':>5} | {'cat_raw':>8} {'cat_kern':>9} | {'ans_raw':>8} {'ans_kern':>9} | {'lang_var':>9}")
    print("-"*70)
    for l in range(n_layers):
        cr = probe_results['cat_raw'][l]
        ck = probe_results['cat_kernel'][l]
        ar = probe_results['ans_raw'][l]
        ak = probe_results['ans_kernel'][l]
        lv = lang_var[l]
        delta_ans = ak - ar
        flag = " *** KERNEL WINS" if delta_ans > 0.05 else (" *** KERNEL KILLS" if delta_ans < -0.05 else "")
        print(f"  L{l:2d} | {cr:8.3f} {ck:9.3f} | {ar:8.3f} {ak:9.3f} | {lv:9.3f}{flag}")

    # Key summary stats
    best_ans_raw = max(probe_results['ans_raw'])
    best_ans_kernel = max(probe_results['ans_kernel'])
    best_layer_raw = np.argmax(probe_results['ans_raw'])
    best_layer_kernel = np.argmax(probe_results['ans_kernel'])

    print("\n" + "="*70)
    print("SUMMARY")
    print(f"  Best answer accuracy (raw):    {best_ans_raw:.3f} at L{best_layer_raw}")
    print(f"  Best answer accuracy (kernel): {best_ans_kernel:.3f} at L{best_layer_kernel}")
    print(f"  Exp Z baseline (EN→ZH):        0.350 at L2  (7x chance=0.048)")
    delta = best_ans_kernel - best_ans_raw
    if delta > 0.05:
        print(f"  KERNEL UNLOCKS +{delta:.3f} — Greg's kernel idea is REAL")
    elif delta > 0:
        print(f"  Kernel marginally better (+{delta:.3f}) — weak signal")
    elif delta > -0.05:
        print(f"  Kernel neutral (Δ={delta:.3f}) — language subspace doesn't help answer probe")
    else:
        print(f"  KERNEL HURTS (Δ={delta:.3f}) — language subspace was carrying answer info")

    print(f"\n  Category accuracy (raw peak):    {max(probe_results['cat_raw']):.3f}")
    print(f"  Category accuracy (kernel peak): {max(probe_results['cat_kernel']):.3f}")
    print(f"  (Exp Z baseline: category=1.0 from L4)")

    # Save results
    results = {
        'n_lang_dims': N_LANG_DIMS,
        'languages': LANGUAGES,
        'lang_var_explained': lang_var,
        'cat_raw': probe_results['cat_raw'],
        'cat_kernel': probe_results['cat_kernel'],
        'ans_raw': probe_results['ans_raw'],
        'ans_kernel': probe_results['ans_kernel'],
        'best_ans_raw': float(best_ans_raw),
        'best_ans_kernel': float(best_ans_kernel),
        'best_layer_raw': int(best_layer_raw),
        'best_layer_kernel': int(best_layer_kernel),
        'exp_z_baseline_answer': 0.350,
        'exp_z_baseline_category': 1.0,
        'interpretation': (
            'kernel_wins' if best_ans_kernel > best_ans_raw + 0.05 else
            'kernel_neutral' if abs(best_ans_kernel - best_ans_raw) <= 0.05 else
            'kernel_hurts'
        )
    }

    out_path = OUTPUT_DIR / "expAB_kernel.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
