"""Experiment P: Strip Language From MLP Deltas During Generation

At L9-L26, let MLP fire normally, then subtract its projection onto the top
language PCs before the delta enters the residual stream. The math component
passes through; the language dressing is removed.

PC0 on the residual stream (from skip connection) is the only language signal.
The attractor at L27+ reads PC0 and formats the output.

M3 failed (zeroed everything). N proved math and language are functionally separable
in the MLP delta. P exploits the separation: keep math, strip language.

10 problems × 2 languages. 128 tokens. Full generation.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
STRIP_LAYERS = list(range(9, 27))  # L9-L26
N_LANG_PCS = 10
N_PCA = 200
d = model.config.hidden_size


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
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。",
                          "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_terms} terms."})
    return problems


# =============================================================================
# Step 1: Fit per-layer language PCA at each strip layer
# =============================================================================
print("=" * 70)
print("FITTING PER-LAYER LANGUAGE PCA")
print("=" * 70)

problems = generate_pca_problems(N_PCA, seed=42)

# Fit PCA at representative layers, use nearest for the rest
pca_fit_layers = [8, 12, 16, 20, 24, 27]
lang_pcs_by_layer = {}

for pca_layer in pca_fit_layers:
    layer_output = {}
    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        layer_output['h'] = h.detach()[:, -1, :]

    handle = model.model.layers[pca_layer].register_forward_hook(capture_hook)

    zh_hidden = np.zeros((N_PCA, d), dtype=np.float32)
    en_hidden = np.zeros((N_PCA, d), dtype=np.float32)

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        zh_hidden[i] = layer_output['h'].cpu().float().numpy()
        layer_output.clear()

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        en_hidden[i] = layer_output['h'].cpu().float().numpy()
        layer_output.clear()

    handle.remove()

    # Fit PCA on normalized hidden states
    zh_norms = np.linalg.norm(zh_hidden, axis=1, keepdims=True)
    en_norms = np.linalg.norm(en_hidden, axis=1, keepdims=True)
    combined = np.vstack([zh_hidden / zh_norms, en_hidden / en_norms])

    pca = PCA(n_components=N_LANG_PCS)
    pca.fit(combined)
    lang_pcs_by_layer[pca_layer] = pca.components_

    zh_proj = (zh_hidden / zh_norms) @ pca.components_[0]
    en_proj = (en_hidden / en_norms) @ pca.components_[0]
    cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt((zh_proj.std()**2 + en_proj.std()**2) / 2)
    print(f"  L{pca_layer}: PC0 Cohen's d = {cohens_d:.1f}, top {N_LANG_PCS} PCs = {sum(pca.explained_variance_ratio_):.1%}")

# Map each strip layer to nearest PCA layer
def get_pcs_for_layer(layer_idx):
    nearest = min(pca_fit_layers, key=lambda x: abs(x - layer_idx))
    return lang_pcs_by_layer[nearest]

# Precompute PC tensors on GPU for each strip layer
pc_tensors = {}
for li in STRIP_LAYERS:
    pcs = get_pcs_for_layer(li)
    pc_tensors[li] = torch.tensor(pcs, dtype=torch.float32, device=device)  # (N_PCS, d)

# Also fit MLP-output-space PCA: the MLP delta lives in a different subspace than
# the layer output. Let's capture MLP deltas directly and fit PCA on those.
print("\nFitting PCA on MLP DELTAS (not layer outputs)...")
mlp_pca_fit_layers = [12, 18, 24]
mlp_lang_pcs = {}

for pca_layer in mlp_pca_fit_layers:
    mlp_output = {}
    def mlp_capture(module, input, output):
        mlp_output['d'] = output.detach()[:, -1, :]  # MLP delta, last token

    handle = model.model.layers[pca_layer].mlp.register_forward_hook(mlp_capture)

    zh_deltas = np.zeros((N_PCA, d), dtype=np.float32)
    en_deltas = np.zeros((N_PCA, d), dtype=np.float32)

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        zh_deltas[i] = mlp_output['d'].cpu().float().numpy()
        mlp_output.clear()

    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        en_deltas[i] = mlp_output['d'].cpu().float().numpy()
        mlp_output.clear()

    handle.remove()

    # Normalize and fit PCA on MLP deltas
    zh_n = np.linalg.norm(zh_deltas, axis=1, keepdims=True) + 1e-8
    en_n = np.linalg.norm(en_deltas, axis=1, keepdims=True) + 1e-8
    combined_d = np.vstack([zh_deltas / zh_n, en_deltas / en_n])

    pca_d = PCA(n_components=N_LANG_PCS)
    pca_d.fit(combined_d)
    mlp_lang_pcs[pca_layer] = pca_d.components_

    zh_p = (zh_deltas / zh_n) @ pca_d.components_[0]
    en_p = (en_deltas / en_n) @ pca_d.components_[0]
    cd = (zh_p.mean() - en_p.mean()) / np.sqrt((zh_p.std()**2 + en_p.std()**2) / 2)
    print(f"  L{pca_layer} MLP delta: PC0 Cohen's d = {cd:.1f}, top {N_LANG_PCS} PCs = {sum(pca_d.explained_variance_ratio_):.1%}")

# Map strip layers to nearest MLP PCA layer
def get_mlp_pcs_for_layer(layer_idx):
    nearest = min(mlp_pca_fit_layers, key=lambda x: abs(x - layer_idx))
    return mlp_lang_pcs[nearest]

mlp_pc_tensors = {}
for li in STRIP_LAYERS:
    pcs = get_mlp_pcs_for_layer(li)
    mlp_pc_tensors[li] = torch.tensor(pcs, dtype=torch.float32, device=device)


# =============================================================================
# Test problems
# =============================================================================
test_problems = [
    {"prompt_en": "Calculate 47 + 86.", "prompt_zh": "计算 47 + 86 的值。", "answer": "133", "difficulty": "simple"},
    {"prompt_en": "A rectangle has length 12 and width 5. Find its area.", "prompt_zh": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60", "difficulty": "simple"},
    {"prompt_en": "What is the remainder when 100 is divided by 7?", "prompt_zh": "100 除以 7 的余数是多少？", "answer": "2", "difficulty": "simple"},
    {"prompt_en": "Calculate 15 × 8.", "prompt_zh": "计算 15 × 8 的值。", "answer": "120", "difficulty": "simple"},
    {"prompt_en": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.",
     "prompt_zh": "等差数列首项为 2，公差为 3，求前 5 项之和。", "answer": "40", "difficulty": "simple"},
    {"prompt_en": "Calculate 387 × 29.", "prompt_zh": "计算 387 × 29 的值。", "answer": "11223", "difficulty": "hard"},
    {"prompt_en": "Find the value of C(10, 3).", "prompt_zh": "求组合数 C(10, 3) 的值。", "answer": "120", "difficulty": "hard"},
    {"prompt_en": "What is the remainder when 7654 is divided by 37?", "prompt_zh": "7654 除以 37 的余数是多少？", "answer": "34", "difficulty": "hard"},
    {"prompt_en": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.",
     "prompt_zh": "等差数列首项为 7，公差为 11，求前 25 项之和。", "answer": "3475", "difficulty": "hard"},
    {"prompt_en": "A rectangle has length 47 and width 33. Find its area.", "prompt_zh": "一个长方形的长为 47，宽为 33，求其面积。", "answer": "1551", "difficulty": "hard"},
]


def run_generation(prompt, strip_mode=None):
    """Run generation with optional MLP language stripping.

    strip_mode: None=baseline, "layer_pcs"=strip using layer-output PCs, "mlp_pcs"=strip using MLP-delta PCs
    """
    input_ids = tokenizer.encode(prompt)
    handles = []

    if strip_mode:
        pcs_dict = mlp_pc_tensors if strip_mode == "mlp_pcs" else pc_tensors
        for li in STRIP_LAYERS:
            def make_strip_hook(layer_idx, pc_t):
                def hook_fn(module, input, output):
                    # output is the MLP delta (plain tensor, shape batch,seq,d)
                    delta = output.float()
                    # Project onto language PCs and subtract
                    # delta: (1, seq, d), pc_t: (N_PCS, d)
                    projections = torch.einsum('bsd,pd->bsp', delta, pc_t)  # (1, seq, N_PCS)
                    lang_component = torch.einsum('bsp,pd->bsd', projections, pc_t)  # (1, seq, d)
                    stripped = delta - lang_component
                    return stripped.to(output.dtype)
                return hook_fn
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_strip_hook(li, pcs_dict[li])
                )
            )

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


def classify_lang(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = sum(1 for c in text if c.isalpha() and c.isascii())
    if zh > en * 2:
        return "zh"
    elif en > zh * 2:
        return "en"
    return "mixed"


# =============================================================================
# Run experiments: two strip modes
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT P: STRIP LANGUAGE FROM MLP DELTAS AT L9-L26")
print("=" * 70)

results = {"experiment": "P: Strip Language from MLP", "modes": {}}

for strip_mode, mode_label in [(None, "baseline"), ("mlp_pcs", "mlp_delta_PCs"), ("layer_pcs", "layer_output_PCs")]:
    print(f"\n{'━'*70}")
    print(f"  MODE: {mode_label}")
    print(f"{'━'*70}")

    mode_results = []

    for lang in ["en", "zh"]:
        lang_label = "English" if lang == "en" else "Chinese"
        print(f"\n  ── {lang_label} ──")

        for prob_idx, prob in enumerate(test_problems):
            prompt = prob[f"prompt_{lang}"]
            answer = prob["answer"]

            text = run_generation(prompt, strip_mode=strip_mode)
            correct = answer in text
            out_lang = classify_lang(text)
            lang_ok = out_lang == lang

            status = f"{'MATH_OK' if correct else 'MATH_FAIL'} {'LANG_OK' if lang_ok else f'LANG→{out_lang}'}"
            print(f"    [{prob_idx}] {status:<20s} | {text[:70]}")

            mode_results.append({
                "problem_idx": prob_idx,
                "language": lang,
                "difficulty": prob["difficulty"],
                "prompt": prompt,
                "answer": answer,
                "text": text,
                "correct": correct,
                "output_lang": out_lang,
                "lang_preserved": lang_ok,
            })

    n_correct = sum(1 for r in mode_results if r["correct"])
    n_lang = sum(1 for r in mode_results if r["lang_preserved"])
    total = len(mode_results)

    en_results = [r for r in mode_results if r["language"] == "en"]
    zh_results = [r for r in mode_results if r["language"] == "zh"]
    en_correct = sum(1 for r in en_results if r["correct"])
    zh_correct = sum(1 for r in zh_results if r["correct"])
    en_lang = sum(1 for r in en_results if r["lang_preserved"])
    zh_lang = sum(1 for r in zh_results if r["lang_preserved"])

    print(f"\n  {mode_label} summary:")
    print(f"    EN: {en_correct}/10 correct, {en_lang}/10 lang OK")
    print(f"    ZH: {zh_correct}/10 correct, {zh_lang}/10 lang OK")
    print(f"    Total: {n_correct}/{total} correct = {n_correct/total:.0%}, {n_lang}/{total} lang = {n_lang/total:.0%}")

    results["modes"][mode_label] = {
        "results": mode_results,
        "en_correct": en_correct,
        "zh_correct": zh_correct,
        "en_lang": en_lang,
        "zh_lang": zh_lang,
        "total_correct": n_correct,
        "total_lang": n_lang,
        "total": total,
    }

# =============================================================================
# Final comparison
# =============================================================================
print(f"\n{'='*70}")
print("P SUMMARY — COMPARISON")
print("=" * 70)

print(f"  {'Mode':<22s} {'Math Correct':>13s} {'Lang Preserved':>15s}")
print(f"  {'─'*22} {'─'*13} {'─'*15}")
for mode_label, data in results["modes"].items():
    mc = f"{data['total_correct']}/{data['total']}"
    lp = f"{data['total_lang']}/{data['total']}"
    print(f"  {mode_label:<22s} {mc:>13s} {lp:>15s}")

bl = results["modes"]["baseline"]
mlp_mode = results["modes"]["mlp_delta_PCs"]
layer_mode = results["modes"]["layer_output_PCs"]

if mlp_mode["total_correct"] >= bl["total_correct"] - 1:
    verdict_math = "MATH PRESERVED: Stripping language PCs from MLP delta keeps math intact"
elif mlp_mode["total_correct"] >= bl["total_correct"] * 0.5:
    verdict_math = "PARTIAL MATH: Some degradation from language stripping"
else:
    verdict_math = "MATH FAILS: Language PCs in MLP delta are entangled with math computation"

if mlp_mode["total_lang"] < bl["total_lang"] * 0.5:
    verdict_lang = "LANGUAGE DISRUPTED: Stripping works — language signal in MLP is removed"
else:
    verdict_lang = "LANGUAGE SURVIVES: PC projection doesn't capture the language signal in MLP"

print(f"\n  Math verdict: {verdict_math}")
print(f"  Lang verdict: {verdict_lang}")

results["verdict_math"] = verdict_math
results["verdict_lang"] = verdict_lang

with open("output/expP_strip_lang_mlp.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expP_strip_lang_mlp.json")
