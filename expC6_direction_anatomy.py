#!/usr/bin/env python3
"""
Exp C6: Direction Anatomy — What IS the Rank-1 Readout?
=======================================================
C3-7B confirmed: k=1 is lossless at L27. But what IS that direction?

Three analyses in one script:

PART A: Direction Identity
  Extract the rank-1 SVD direction at L27. Compute cosine similarity with:
  1. Digit token embeddings (0-9) — are we reading out "which number?"
  2. Language mean-difference direction (EN vs ZH mean embed) — language ID?
  3. Individual answer-token embeddings for the test problems
  4. W_U rows for digit tokens — does the readout project into unembedding space?

PART B: Layer Sweep
  Run C3 (k=1 only, fast) at L26, L25, L20, L15, L10, L5 to find where
  rank-1 compression breaks. Maps the "read head formation" trajectory.

PART C: Subspace Overlap
  Compute principal angles between:
  - C3 top-8 subspace at L27 vs digit-embedding span(embed(0)..embed(9))
  - C3 top-8 subspace at L27 vs answer-token embedding span
  If overlap > 0.8 → readout IS the output vocabulary, not abstract.

Self-contained. Runs on A100 with 7B loaded.
"""

import json, math, time, re, gc
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

torch.backends.cuda.matmul.allow_tf32 = True

MODEL_NAME = "Qwen/Qwen2.5-7B"
N_LAYERS = 28
D_MODEL = 3584
MAX_NEW = 128
SEED = 42
PRIMARY_LAYER = 27
SWEEP_LAYERS = [26, 25, 20, 15, 10, 5]

CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)

LANGS = ['en', 'zh']

# ═══════════════════════════════════════════════════════════════════
# PROBLEM GENERATION (identical to C2c/C3)
# ═══════════════════════════════════════════════════════════════════

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
    by_cat = {}
    for p in problems:
        cat = p['category']
        by_cat.setdefault(cat, [])
        if len(by_cat[cat]) < n_per_cat:
            by_cat[cat].append(p)
    return [p for cat_probs in by_cat.values() for p in cat_probs]

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.bool_): return bool(o)
        return super().default(o)

# ═══════════════════════════════════════════════════════════════════
# HOOKS & GENERATION
# ═══════════════════════════════════════════════════════════════════

def build_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{CHAT_SYSTEM}\n\nProblem: {problem_text}\n\nSolution:"

def check_answer(text, correct):
    return str(correct) in re.findall(r"-?\d+\.?\d*", text)

class CaptureHook:
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
    def __init__(self, mean_vec, basis_vecs, k, device):
        self.mean = torch.tensor(mean_vec, dtype=torch.float32, device=device)
        self.basis = torch.tensor(basis_vecs[:k], dtype=torch.float32, device=device)
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

def extract_basis(model, tokenizer, problems, target_layer, device, max_new=MAX_NEW):
    """Extract SVD basis at target_layer from basis problems. Returns (mean, Vh, S, all_readouts)."""
    attn_module = model.model.layers[target_layer].self_attn
    capture = CaptureHook()
    handle = attn_module.register_forward_hook(capture)
    all_captures = []
    for pi, prob in enumerate(problems):
        for lang in LANGS:
            prompt_text = build_prompt(tokenizer, prob[lang])
            capture.captured = []
            capture.active = True
            generate(model, tokenizer, prompt_text, device, max_new)
            capture.active = False
            all_captures.extend(capture.captured)
    handle.remove()
    all_readouts = np.stack(all_captures)
    mean_vec = all_readouts.mean(axis=0)
    centered = all_readouts - mean_vec
    _, S, Vh = np.linalg.svd(centered, full_matrices=False)
    return mean_vec, Vh, S, all_readouts

def run_compression_test(model, tokenizer, problems, target_layer, mean_vec, basis_vecs, k, device, max_new=MAX_NEW):
    """Run k-D compression at target_layer. Returns (n_correct, total, per_lang, details)."""
    attn_module = model.model.layers[target_layer].self_attn
    compressor = CompressionHook(mean_vec, basis_vecs, k, device)
    handle = attn_module.register_forward_hook(compressor)
    results = []
    for pi, prob in enumerate(problems):
        for lang in LANGS:
            prompt_text = build_prompt(tokenizer, prob[lang])
            compressor.active = True
            text = generate(model, tokenizer, prompt_text, device, max_new)
            compressor.active = False
            correct = check_answer(text, prob["answer"])
            results.append({"pi": pi, "lang": lang, "cat": prob["category"], "correct": correct, "text": text[:200]})
    handle.remove()
    n_correct = sum(r["correct"] for r in results)
    total = len(results)
    en = sum(r["correct"] for r in results if r["lang"] == "en")
    zh = sum(r["correct"] for r in results if r["lang"] == "zh")
    return n_correct, total, {"en": en, "zh": zh}, results


# ═══════════════════════════════════════════════════════════════════
# PART A: Direction Identity
# ═══════════════════════════════════════════════════════════════════

def analyze_direction_identity(model, tokenizer, v1, top8, mean_vec, device):
    """What IS the rank-1 direction? Project onto known subspaces."""
    print(f"\n{'='*60}")
    print("PART A: Direction Identity — What is v1?")
    print(f"{'='*60}")

    results = {}

    # 1. Get embedding matrix
    embed_weight = model.model.embed_tokens.weight.float().cpu()  # (vocab, d)
    # For tied models, lm_head.weight = embed_tokens.weight
    if hasattr(model, 'lm_head') and model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        print("  Model has TIED embeddings (lm_head = embed_tokens)")
        unembed_weight = embed_weight  # same matrix
        results["tied"] = True
    else:
        unembed_weight = model.lm_head.weight.float().cpu()
        print("  Model has UNTIED embeddings")
        results["tied"] = False

    v1_t = torch.tensor(v1, dtype=torch.float32)
    v1_norm = v1_t / v1_t.norm()

    # 2. Digit token embeddings (0-9)
    print("\n  --- Digit Token Cosines (embed & unembed) ---")
    digit_cosines_embed = {}
    digit_cosines_unembed = {}
    digit_ids = {}
    for digit in range(10):
        tok_ids = tokenizer.encode(str(digit), add_special_tokens=False)
        # Take the first token that encodes just the digit
        tid = tok_ids[0] if tok_ids else None
        if tid is not None:
            digit_ids[str(digit)] = tid
            e = embed_weight[tid]
            e_norm = e / e.norm()
            cos_e = float(v1_norm @ e_norm)
            digit_cosines_embed[str(digit)] = cos_e

            u = unembed_weight[tid]
            u_norm = u / u.norm()
            cos_u = float(v1_norm @ u_norm)
            digit_cosines_unembed[str(digit)] = cos_u

            print(f"    '{digit}' (id={tid}): embed_cos={cos_e:.4f}, unembed_cos={cos_u:.4f}")

    results["digit_cosines_embed"] = digit_cosines_embed
    results["digit_cosines_unembed"] = digit_cosines_unembed
    results["digit_token_ids"] = digit_ids

    # 3. Digit embedding subspace overlap
    digit_embeds = torch.stack([embed_weight[digit_ids[str(d)]] for d in range(10)])  # (10, d)
    digit_centered = digit_embeds - digit_embeds.mean(dim=0)
    U_digit, S_digit, _ = torch.linalg.svd(digit_centered, full_matrices=False)
    digit_basis = torch.linalg.svd(digit_centered, full_matrices=False)[2][:6]  # top-6 of digits
    # Project v1 onto digit subspace
    proj_coeff = v1_norm @ digit_basis.T  # (6,)
    v1_in_digit = float((proj_coeff ** 2).sum())  # fraction of v1 in digit span
    print(f"\n  v1 projection onto digit-embed span (top-6): {v1_in_digit:.4f}")
    results["v1_in_digit_span"] = v1_in_digit

    # 4. Language direction
    print("\n  --- Language Direction ---")
    # Compute mean embedding for EN vs ZH common tokens
    # Use a simple approach: mean of first 1000 tokens in each script
    en_toks = [tokenizer.encode(c, add_special_tokens=False)[0]
               for c in "the of and to in is that it for was on are as with his they be at one have this from or had by not but some what there we can out other were all your when up use how said an each she which do their time if will way about many then them would write like so these her long make thing see him two has look more day could go come did my no most number".split()
               if tokenizer.encode(c, add_special_tokens=False)]
    zh_chars = "的一是不了人我在有他这中大来上个国到说们为子和你地出会也时要就学那能开去过家后多里方自心前面水如行走很见对然好十定没比实体现做同月行从当没全建于起真好正到新两些样过同前等回因又向此所以想只手已但把把"
    zh_toks = [tokenizer.encode(c, add_special_tokens=False)[0] for c in zh_chars
               if tokenizer.encode(c, add_special_tokens=False)]

    en_mean = embed_weight[en_toks[:50]].mean(dim=0)
    zh_mean = embed_weight[zh_toks[:50]].mean(dim=0)
    lang_dir = zh_mean - en_mean
    lang_dir_norm = lang_dir / lang_dir.norm()

    cos_lang = float(v1_norm @ lang_dir_norm)
    print(f"  cos(v1, zh_mean - en_mean) = {cos_lang:.4f}")
    results["cos_v1_lang_direction"] = cos_lang

    # 5. Special tokens
    print("\n  --- Special Token Cosines ---")
    special = {}
    for name in ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "\n", "=", "+", "×", "答案", "answer", "Answer"]:
        toks = tokenizer.encode(name, add_special_tokens=False)
        if toks:
            e = embed_weight[toks[0]]
            cos_val = float(v1_norm @ (e / e.norm()))
            special[name] = {"id": toks[0], "cos": cos_val}
            print(f"    '{name}' (id={toks[0]}): cos={cos_val:.4f}")
    results["special_token_cosines"] = special

    # 6. Top-k tokens by cosine with v1 (in unembed space — what does v1 "vote for"?)
    print("\n  --- Top 20 Tokens by |cos(v1, unembed[t])| ---")
    all_cos = (unembed_weight / unembed_weight.norm(dim=1, keepdim=True)) @ v1_norm
    top_idx = torch.argsort(all_cos.abs(), descending=True)[:20]
    top_tokens = []
    for idx in top_idx:
        tid = int(idx)
        cos_val = float(all_cos[tid])
        tok_str = tokenizer.decode([tid])
        top_tokens.append({"id": tid, "token": tok_str, "cos": cos_val})
        print(f"    id={tid:>6} '{tok_str:>12}': cos={cos_val:+.4f}")
    results["top20_unembed_cosines"] = top_tokens

    # 7. Top-k tokens by cos with v1 in embed space
    print("\n  --- Top 20 Tokens by |cos(v1, embed[t])| ---")
    all_cos_e = (embed_weight / embed_weight.norm(dim=1, keepdim=True)) @ v1_norm
    top_idx_e = torch.argsort(all_cos_e.abs(), descending=True)[:20]
    top_tokens_e = []
    for idx in top_idx_e:
        tid = int(idx)
        cos_val = float(all_cos_e[tid])
        tok_str = tokenizer.decode([tid])
        top_tokens_e.append({"id": tid, "token": tok_str, "cos": cos_val})
        print(f"    id={tid:>6} '{tok_str:>12}': cos={cos_val:+.4f}")
    results["top20_embed_cosines"] = top_tokens_e

    # 8. Principal angles between top-8 SVD subspace and digit subspace
    print("\n  --- Subspace Overlap: top-8 readout vs digit-embed ---")
    top8_t = torch.tensor(top8, dtype=torch.float32)  # (8, d)
    # Principal angles via SVD of A^T B
    M = top8_t @ digit_basis.T  # (8, 6)
    svals = torch.linalg.svdvals(M)
    angles = torch.acos(torch.clamp(svals, -1, 1)) * 180 / np.pi
    print(f"  Principal angles (degrees): {[f'{a:.1f}' for a in angles.tolist()]}")
    print(f"  Cosines of principal angles: {[f'{s:.4f}' for s in svals.tolist()]}")
    overlap_metric = float(svals[:min(6, len(svals))].sum() / min(6, len(svals)))
    print(f"  Mean cosine (overlap metric): {overlap_metric:.4f}")
    results["principal_angles_deg"] = angles.tolist()
    results["principal_cosines"] = svals.tolist()
    results["mean_overlap"] = overlap_metric

    # 9. Is v1 approximately the mean readout direction? (sanity check)
    mean_norm = np.linalg.norm(mean_vec)
    cos_v1_mean = float(v1_norm @ torch.tensor(mean_vec / mean_norm, dtype=torch.float32))
    print(f"\n  cos(v1, mean_readout) = {cos_v1_mean:.4f}")
    results["cos_v1_mean_readout"] = cos_v1_mean

    return results


# ═══════════════════════════════════════════════════════════════════
# PART B: Layer Sweep (k=1 only)
# ═══════════════════════════════════════════════════════════════════

def layer_sweep_k1(model, tokenizer, basis_problems, test_problems, device, max_new=MAX_NEW):
    """Run k=1 compression at multiple layers to find where it breaks."""
    print(f"\n{'='*60}")
    print("PART B: Layer Sweep — Where does k=1 break?")
    print(f"{'='*60}")

    # First get baseline
    print("\n  --- Baseline ---")
    bl_correct = 0
    bl_total = 0
    for prob in test_problems:
        for lang in LANGS:
            text = generate(model, tokenizer, build_prompt(tokenizer, prob[lang]), device, max_new)
            if check_answer(text, prob["answer"]):
                bl_correct += 1
            bl_total += 1
    print(f"  Baseline: {bl_correct}/{bl_total}")

    results = {"baseline": {"correct": bl_correct, "total": bl_total}}

    for layer in [PRIMARY_LAYER] + SWEEP_LAYERS:
        print(f"\n  --- L{layer} ---")
        t0 = time.time()
        mean_vec, Vh, S, _ = extract_basis(model, tokenizer, basis_problems, layer, device, max_new)
        cumvar = np.cumsum(S**2) / (S**2).sum()
        rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        top1_frac = float(cumvar[0])
        print(f"    rank_90={rank_90}, top1_frac={top1_frac:.4f}")

        n_correct, total, per_lang, details = run_compression_test(
            model, tokenizer, test_problems, layer, mean_vec, Vh, 1, device, max_new
        )
        elapsed = time.time() - t0
        print(f"    k=1: {n_correct}/{total} (EN={per_lang['en']}, ZH={per_lang['zh']}) [{elapsed:.0f}s]")

        results[f"L{layer}"] = {
            "layer": layer,
            "k1_correct": n_correct,
            "k1_total": total,
            "k1_en": per_lang["en"],
            "k1_zh": per_lang["zh"],
            "rank_90": rank_90,
            "top1_frac": top1_frac,
            "top5_singular": S[:5].tolist(),
            "elapsed_s": elapsed,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    device = "cuda"
    t_start = time.time()

    print(f"{'='*60}")
    print(f"Exp C6: Direction Anatomy on Qwen2.5-7B")
    print(f"{'='*60}")
    print(f"Primary layer: L{PRIMARY_LAYER}")
    print(f"Sweep layers:  {SWEEP_LAYERS}")
    print()

    # ── Load model ──
    print("Loading model...")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t_load:.0f}s, d={model.config.hidden_size}, layers={len(model.model.layers)}")

    # ── Problems ──
    all_problems = generate_problems()
    all_test = get_test_subset(all_problems)
    by_cat = OrderedDict()
    for p in all_test:
        by_cat.setdefault(p["category"], []).append(p)
    basis_problems, test_problems = [], []
    for cat, probs in by_cat.items():
        half = max(1, len(probs) // 2)
        basis_problems.extend(probs[:half])
        test_problems.extend(probs[half:])
    print(f"  {len(basis_problems)} basis + {len(test_problems)} test")
    print()

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Extract basis at primary layer
    # ══════════════════════════════════════════════════════════
    print(f"Extracting SVD basis at L{PRIMARY_LAYER}...")
    mean_vec, Vh, S, all_readouts = extract_basis(
        model, tokenizer, basis_problems, PRIMARY_LAYER, device
    )
    cumvar = np.cumsum(S**2) / (S**2).sum()
    rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f"  Readout shape: {all_readouts.shape}")
    print(f"  rank_90={rank_90}, top1_frac={cumvar[0]:.4f}")
    print(f"  Top-5 singular values: {S[:5].tolist()}")

    v1 = Vh[0]  # THE rank-1 direction
    top8 = Vh[:8]

    # ══════════════════════════════════════════════════════════
    # PART A: Direction Identity
    # ══════════════════════════════════════════════════════════
    part_a = analyze_direction_identity(model, tokenizer, v1, top8, mean_vec, device)

    # ══════════════════════════════════════════════════════════
    # PART B: Layer Sweep
    # ══════════════════════════════════════════════════════════
    part_b = layer_sweep_k1(model, tokenizer, basis_problems, test_problems, device)

    # ══════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════
    total_time = time.time() - t_start

    output = {
        "experiment": "C6: Direction Anatomy on 7B",
        "model": MODEL_NAME,
        "primary_layer": PRIMARY_LAYER,
        "sweep_layers": SWEEP_LAYERS,
        "elapsed_s": total_time,
        "primary_basis": {
            "readout_shape": list(all_readouts.shape),
            "rank_90": rank_90,
            "top1_frac": float(cumvar[0]),
            "top10_singular_values": S[:10].tolist(),
            "cumvar_at_k": {str(k): float(cumvar[k-1]) for k in [1,2,3,5,8,12,20] if k <= len(cumvar)},
        },
        "part_a_direction_identity": part_a,
        "part_b_layer_sweep": part_b,
    }

    outpath = Path("expC6_direction_anatomy_7b.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {total_time:.0f}s")

    # ── Quick summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nPart A — Direction Identity:")
    print(f"  cos(v1, lang_dir):     {part_a['cos_v1_lang_direction']:.4f}")
    print(f"  v1 in digit span:      {part_a['v1_in_digit_span']:.4f}")
    print(f"  cos(v1, mean_readout): {part_a['cos_v1_mean_readout']:.4f}")
    print(f"  Subspace overlap:      {part_a['mean_overlap']:.4f}")
    print(f"  Top unembed token:     '{part_a['top20_unembed_cosines'][0]['token']}' (cos={part_a['top20_unembed_cosines'][0]['cos']:.4f})")

    print(f"\nPart B — Layer Sweep (k=1):")
    print(f"  {'Layer':>6} | {'k=1':>6} | {'top1%':>6} | {'r90':>4}")
    print(f"  {'-'*35}")
    for key, data in part_b.items():
        if key == "baseline":
            print(f"  {'base':>6} | {data['correct']:>3}/{data['total']:<2} |    --- |  ---")
        else:
            print(f"  L{data['layer']:>4} | {data['k1_correct']:>3}/{data['k1_total']:<2} | {data['top1_frac']*100:>5.1f}% | {data['rank_90']:>4}")


if __name__ == "__main__":
    main()
