"""Experiment O: The Learned Shortcut (Generation Version)

Train a Ridge regression mapping L8 hidden states to L27 hidden states.
Training data: cached 200 zh + 200 en states from all_layers_lasttok.npz.

Then: use the Ridge map during generation. Every forward pass:
- L0-L8 fire normally
- Apply Ridge map: h_L27_approx = W @ h_L8 + b
- L27-L35 fire normally with h_L27_approx as input

The Ridge map produces states IN the L27 manifold (trained on real L27 states).
No norm mismatch. No coordinate mismatch. The KV cache at L27+ is built from
Ridge-mapped states, which look like natural L27 states.

10 problems. 128 tokens. Compare to baseline AND to Exp M3.
"""
import json
import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
SRC_LAYER = 8
DST_LAYER = 27
SKIP_LAYERS = list(range(SRC_LAYER + 1, DST_LAYER))  # L9 through L26

# =============================================================================
# Step 1: Check for cached data, otherwise extract
# =============================================================================
import os
npz_path = "output/all_layers_lasttok.npz"

if os.path.exists(npz_path):
    print("Loading cached hidden states from all_layers_lasttok.npz...")
    data = np.load(npz_path)
    # Expected keys: zh_L8, en_L8, zh_L27, en_L27 (or similar naming)
    available_keys = list(data.keys())
    print(f"  Available keys: {available_keys[:10]}...")

    # Try to find the right arrays
    # The file likely has arrays named like 'zh' and 'en' with shape (200, 36, 2048)
    # or individual layer arrays
    if 'zh' in data and 'en' in data:
        zh_all = data['zh']  # (200, 36, 2048) or similar
        en_all = data['en']
        print(f"  zh shape: {zh_all.shape}, en shape: {en_all.shape}")
        if zh_all.ndim == 3:
            zh_L8 = zh_all[:, SRC_LAYER, :]
            en_L8 = en_all[:, SRC_LAYER, :]
            zh_L27 = zh_all[:, DST_LAYER, :]
            en_L27 = en_all[:, DST_LAYER, :]
        else:
            raise ValueError(f"Unexpected shape: {zh_all.shape}")
    else:
        # Try layer-specific keys
        zh_L8 = data.get(f'zh_L{SRC_LAYER}', data.get(f'zh_{SRC_LAYER}'))
        en_L8 = data.get(f'en_L{SRC_LAYER}', data.get(f'en_{SRC_LAYER}'))
        zh_L27 = data.get(f'zh_L{DST_LAYER}', data.get(f'zh_{DST_LAYER}'))
        en_L27 = data.get(f'en_L{DST_LAYER}', data.get(f'en_{DST_LAYER}'))
        if zh_L8 is None:
            print(f"  Can't find layer-specific arrays. Available: {available_keys}")
            raise KeyError("Missing layer data")
else:
    print("No cached data found. Extracting L8 and L27 hidden states...")
    import random as pyrandom

    def generate_pca_problems(n=200, seed=42):
        rng = pyrandom.Random(seed)
        problems = []
        per_cat = n // 5
        for _ in range(per_cat):
            a, b = rng.randint(10, 999), rng.randint(10, 999)
            op = rng.choice(["plus", "times"])
            if op == "plus":
                zh, en = f"计算 {a} + {b} 的值。", f"Calculate {a} + {b}."
            else:
                zh, en = f"计算 {a} × {b} 的值。", f"Calculate {a} × {b}."
            problems.append({"zh": zh, "en": en})
        for _ in range(per_cat):
            n_val = rng.randint(5, 20)
            k_val = rng.randint(1, min(n_val - 1, 8))
            problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                              "en": f"Find the value of C({n_val}, {k_val})."})
        for _ in range(per_cat):
            a = rng.randint(50, 9999)
            b = rng.randint(3, 37)
            problems.append({"zh": f"{a} 除以 {b} 的余数是多少？",
                              "en": f"What is the remainder when {a} is divided by {b}?"})
        for _ in range(per_cat):
            w, h = rng.randint(2, 50), rng.randint(2, 50)
            problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                              "en": f"A rectangle has length {w} and width {h}. Find its area."})
        for _ in range(per_cat):
            a1, d = rng.randint(1, 20), rng.randint(1, 10)
            n_terms = rng.randint(5, 30)
            problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
                              "en": f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."})
        return problems

    problems = generate_pca_problems(200, seed=42)
    d = model.config.hidden_size

    zh_L8 = np.zeros((200, d), dtype=np.float32)
    en_L8 = np.zeros((200, d), dtype=np.float32)
    zh_L27 = np.zeros((200, d), dtype=np.float32)
    en_L27 = np.zeros((200, d), dtype=np.float32)

    captures = {}
    def make_capture(layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captures[layer_idx] = h.detach()[:, -1, :].cpu().float().numpy()
        return hook_fn

    h8 = model.model.layers[SRC_LAYER].register_forward_hook(make_capture(SRC_LAYER))
    h27 = model.model.layers[DST_LAYER].register_forward_hook(make_capture(DST_LAYER))

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        zh_L8[i] = captures[SRC_LAYER]
        zh_L27[i] = captures[DST_LAYER]
        captures.clear()

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        en_L8[i] = captures[SRC_LAYER]
        en_L27[i] = captures[DST_LAYER]
        captures.clear()

    h8.remove()
    h27.remove()
    print(f"  Extracted {len(problems)} zh + en states at L{SRC_LAYER} and L{DST_LAYER}")

# =============================================================================
# Step 2: Train Ridge regression L8 -> L27
# =============================================================================
print(f"\nTraining Ridge regression L{SRC_LAYER} -> L{DST_LAYER}...")

X_train = np.vstack([zh_L8, en_L8])  # (400, 2048)
Y_train = np.vstack([zh_L27, en_L27])  # (400, 2048)

print(f"  X shape: {X_train.shape}, Y shape: {Y_train.shape}")
print(f"  X norms: mean={np.linalg.norm(X_train, axis=1).mean():.1f}")
print(f"  Y norms: mean={np.linalg.norm(Y_train, axis=1).mean():.1f}")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)

# Evaluate fit quality
Y_pred = ridge.predict(X_train)
residuals = Y_train - Y_pred
r2 = 1 - np.sum(residuals**2) / np.sum((Y_train - Y_train.mean(axis=0))**2)
cos_sims = np.array([np.dot(Y_train[i], Y_pred[i]) / (np.linalg.norm(Y_train[i]) * np.linalg.norm(Y_pred[i]))
                      for i in range(len(Y_train))])
norm_ratios = np.array([np.linalg.norm(Y_pred[i]) / np.linalg.norm(Y_train[i]) for i in range(len(Y_train))])

print(f"  Ridge R²: {r2:.4f}")
print(f"  Cosine similarity: mean={cos_sims.mean():.4f}, min={cos_sims.min():.4f}")
print(f"  Norm ratio (pred/true): mean={norm_ratios.mean():.3f}, std={norm_ratios.std():.3f}")

# Convert to torch tensors for fast inference
W = torch.tensor(ridge.coef_, dtype=torch.bfloat16, device=device)  # (2048, 2048)
b = torch.tensor(ridge.intercept_, dtype=torch.bfloat16, device=device)  # (2048,)

# =============================================================================
# Step 3: Generation with Ridge shortcut
# =============================================================================
test_problems = [
    # Simple
    {"prompt_en": "Calculate 47 + 86.", "prompt_zh": "计算 47 + 86 的值。", "answer": "133", "difficulty": "simple"},
    {"prompt_en": "A rectangle has length 12 and width 5. Find its area.", "prompt_zh": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60", "difficulty": "simple"},
    {"prompt_en": "What is the remainder when 100 is divided by 7?", "prompt_zh": "100 除以 7 的余数是多少？", "answer": "2", "difficulty": "simple"},
    {"prompt_en": "Calculate 15 × 8.", "prompt_zh": "计算 15 × 8 的值。", "answer": "120", "difficulty": "simple"},
    {"prompt_en": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.",
     "prompt_zh": "等差数列首项为 2，公差为 3，求前 5 项之和。", "answer": "40", "difficulty": "simple"},
    # Hard
    {"prompt_en": "Calculate 387 × 29.", "prompt_zh": "计算 387 × 29 的值。", "answer": "11223", "difficulty": "hard"},
    {"prompt_en": "Find the value of C(10, 3).", "prompt_zh": "求组合数 C(10, 3) 的值。", "answer": "120", "difficulty": "hard"},
    {"prompt_en": "What is the remainder when 7654 is divided by 37?", "prompt_zh": "7654 除以 37 的余数是多少？", "answer": "34", "difficulty": "hard"},
    {"prompt_en": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.",
     "prompt_zh": "等差数列首项为 7，公差为 11，求前 25 项之和。", "answer": "3475", "difficulty": "hard"},
    {"prompt_en": "A rectangle has length 47 and width 33. Find its area.", "prompt_zh": "一个长方形的长为 47，宽为 33，求其面积。", "answer": "1551", "difficulty": "hard"},
]


def run_baseline(prompt):
    """Normal generation."""
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
    past_kv = outputs.past_key_values
    next_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[next_id]], device=device)
    generated_ids = [next_id]
    for _ in range(MAX_NEW_TOKENS - 1):
        with torch.no_grad():
            out = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_id = int(out.logits[0, -1].argmax())
        generated_ids.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_ridge_shortcut(prompt):
    """Generation with Ridge shortcut: L0-L8 normal, Ridge map to L27, L27-L35 normal.

    Implementation: Hook on L8 output to capture h_L8. Hook on each of L9-L26 to
    replace their output with a passthrough (the residual stream from L8, untouched).
    Hook on L27 input to inject the Ridge-mapped state.

    Actually, the cleanest approach: hook the output of EACH layer L9-L26 to replace
    the hidden state with the L8 hidden state (passthrough — skip the computation
    but maintain shape/cache). Then hook L27 input to inject the Ridge-mapped state
    from L8.

    Even cleaner: hook L8 output to capture. Hook each L9-L26 to zero their residual
    contributions (both attn and MLP). Then inject at L27 by hooking L26's output to
    be the Ridge-mapped state.
    """
    input_ids = tokenizer.encode(prompt)
    l8_hidden = {}

    # Hook L8 to capture
    def l8_capture(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        l8_hidden['h'] = h.detach()  # (batch, seq, d)

    # Hook L26 (last skipped layer) to inject Ridge-mapped state
    def l26_inject(module, input, output):
        h_l8 = l8_hidden['h']  # (batch, seq, d)
        # Apply Ridge: h_L27_approx = h_L8 @ W^T + b
        h_mapped = h_l8.float() @ W.float().T + b.float()
        h_mapped = h_mapped.to(output.dtype if not isinstance(output, tuple) else output[0].dtype)
        if isinstance(output, tuple):
            return (h_mapped,) + output[1:]
        return h_mapped

    # For L9-L25: pass through (layer computes but we replace output with input)
    # We want the KV cache to be built from the actual computations (attention fires),
    # but the residual stream should be frozen at h_L8
    # Actually — if we freeze the residual at L8 through L9-L25, the KV cache will
    # reflect the L8 state at every position, which is internally consistent.
    def make_passthrough(layer_idx):
        def hook_fn(module, input, output):
            h_l8 = l8_hidden['h']
            if isinstance(output, tuple):
                return (h_l8.to(output[0].dtype),) + output[1:]
            return h_l8.to(output.dtype)
        return hook_fn

    handles = []
    handles.append(model.model.layers[SRC_LAYER].register_forward_hook(l8_capture))
    for li in range(SRC_LAYER + 1, DST_LAYER - 1):  # L9 through L25
        handles.append(model.model.layers[li].register_forward_hook(make_passthrough(li)))
    handles.append(model.model.layers[DST_LAYER - 1].register_forward_hook(l26_inject))  # L26

    try:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]

        for _ in range(MAX_NEW_TOKENS - 1):
            with torch.no_grad():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)
            if next_id == tokenizer.eos_token_id:
                break
    finally:
        for h in handles:
            h.remove()

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# =============================================================================
# Run
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT O: RIDGE SHORTCUT (L8 -> L27) DURING GENERATION")
print("=" * 70)

results = {
    "experiment": "O: Ridge Shortcut L8->L27",
    "ridge_r2": float(r2),
    "ridge_cos_mean": float(cos_sims.mean()),
    "ridge_norm_ratio_mean": float(norm_ratios.mean()),
    "problems": [],
}

for lang in ["en", "zh"]:
    lang_label = "English" if lang == "en" else "Chinese"
    print(f"\n{'─'*70}")
    print(f"  LANGUAGE: {lang_label}")
    print(f"{'─'*70}")

    for prob_idx, prob in enumerate(test_problems):
        prompt = prob[f"prompt_{lang}"]
        answer = prob["answer"]
        difficulty = prob["difficulty"]

        baseline_text = run_baseline(prompt)
        ridge_text = run_ridge_shortcut(prompt)

        bl_correct = answer in baseline_text
        ridge_correct = answer in ridge_text

        zh_r = sum(1 for c in ridge_text if '\u4e00' <= c <= '\u9fff')
        en_r = sum(1 for c in ridge_text if c.isalpha() and c.isascii())
        ridge_lang = "zh" if zh_r > en_r else "en"
        lang_ok = ridge_lang == lang

        status = f"{'MATH_OK' if ridge_correct else 'MATH_FAIL'} {'LANG_OK' if lang_ok else f'LANG_FLIP({ridge_lang})'}"

        print(f"\n  [{prob_idx}] {difficulty.upper()} | {prompt[:50]}...")
        print(f"    Baseline: {'CORRECT' if bl_correct else 'WRONG':>7s} | {baseline_text[:80]}")
        print(f"    Ridge:    {'CORRECT' if ridge_correct else 'WRONG':>7s} | {ridge_text[:80]}")
        print(f"    Status: {status}")

        entry = {
            "problem_idx": prob_idx,
            "language": lang,
            "difficulty": difficulty,
            "prompt": prompt,
            "answer": answer,
            "baseline_text": baseline_text,
            "ridge_text": ridge_text,
            "baseline_correct": bl_correct,
            "ridge_correct": ridge_correct,
            "lang_preserved": lang_ok,
            "ridge_output_lang": ridge_lang,
        }
        results["problems"].append(entry)

# Summary
print(f"\n{'='*70}")
print("O SUMMARY")
print("=" * 70)

for lang in ["en", "zh"]:
    lp = [p for p in results["problems"] if p["language"] == lang]
    bl_c = sum(1 for p in lp if p["baseline_correct"])
    rd_c = sum(1 for p in lp if p["ridge_correct"])
    lg_ok = sum(1 for p in lp if p["lang_preserved"])
    n = len(lp)
    simple_c = sum(1 for p in lp if p["difficulty"] == "simple" and p["ridge_correct"])
    hard_c = sum(1 for p in lp if p["difficulty"] == "hard" and p["ridge_correct"])
    lang_label = "English" if lang == "en" else "Chinese"
    print(f"\n  {lang_label}:")
    print(f"    Baseline correct: {bl_c}/{n}")
    print(f"    Ridge correct:    {rd_c}/{n} (simple: {simple_c}/5, hard: {hard_c}/5)")
    print(f"    Language preserved: {lg_ok}/{n}")

all_p = results["problems"]
total_bl = sum(1 for p in all_p if p["baseline_correct"])
total_rd = sum(1 for p in all_p if p["ridge_correct"])
total_lg = sum(1 for p in all_p if p["lang_preserved"])
total = len(all_p)

print(f"\n  OVERALL:")
print(f"    Baseline correct: {total_bl}/{total}")
print(f"    Ridge correct:    {total_rd}/{total} = {total_rd/total:.0%}")
print(f"    Language preserved: {total_lg}/{total} = {total_lg/total:.0%}")

if total_rd >= 16:
    verdict = "RIDGE SHORTCUT WORKS: L9-L26 can be replaced by a single matrix multiply"
elif total_rd >= 12:
    verdict = "STRONG: Ridge shortcut mostly preserves math reasoning"
elif total_rd >= 8:
    verdict = "PARTIAL: Some math survives the shortcut"
else:
    verdict = "RIDGE SHORTCUT FAILS: L9-L26 computation is not linearly approximable"

print(f"\n  VERDICT: {verdict}")

results["summary"] = {
    "baseline_correct": total_bl,
    "ridge_correct": total_rd,
    "lang_preserved": total_lg,
    "total": total,
    "verdict": verdict,
}

with open("output/expO_ridge_shortcut.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expO_ridge_shortcut.json")
