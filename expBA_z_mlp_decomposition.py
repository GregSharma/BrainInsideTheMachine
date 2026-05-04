"""Exp BA: Z-MLP Decomposition — Can the MLP reason on Z alone?

Tests whether the MLP's computation can be factored into reasoning (Z subspace)
and language (complement) pathways.

Two conditions:
  A) Z-input MLP: project MLP input onto Z before feeding through MLP.
     Complement bypasses MLP unchanged. Tests if MLP can reason on 20 dims.
  B) Z-output MLP: let MLP see full input, but only keep the Z-projected
     component of the MLP delta. Complement of delta is zeroed.
     Tests if the reasoning output lives in Z.

Condition B is the more likely to work (closer to Exp P2 which showed
zeroing the language component of the delta preserves math).

Uses the contrastive Z basis from Phase 5 (k=20, layer 32).
Per-layer Z via the pc0 vectors where available, falls back to L32 Z.
"""
import json, re, time
import random as pyrandom
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import comb

OUTPUT_DIR = Path('output')
MAX_NEW_TOKENS = 128
device = 'cuda'

# ── Load model ──────────────────────────────────────────────────────────
print("Loading Qwen2.5-3B...")
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation='sdpa'
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"  {n_layers} layers, d={d_model}")


# ── Load Z basis ────────────────────────────────────────────────────────
print("\nLoading Z basis...")
# Get contrastive Z from multilingual data
data = np.load('output/multilingual_all_layers.npz', allow_pickle=True)

def compute_z_basis(layer, k=20):
    """Compute Z basis at a given layer via contrastive PCA (EN-ZH pairs)."""
    en = data[f'en_L{layer}']  # (200, 2048)
    zh = data[f'zh_L{layer}']  # (200, 2048)
    # Contrastive: within-problem cross-language covariance
    diffs = en - zh  # (200, 2048)
    # SVD of the difference matrix
    U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
    # Top k right singular vectors = language-varying directions
    # Z = null space of these = directions that DON'T vary across languages
    lang_dirs = Vt[:k]  # (k, 2048)
    # Z basis = complement of language directions
    # But actually we want: the directions that ARE shared across languages
    # Use within-language PCA on the mean-centered combined data
    combined = np.vstack([en, zh])  # (400, 2048)
    mean = combined.mean(axis=0)
    centered = combined - mean
    _, _, Vt_all = np.linalg.svd(centered, full_matrices=False)
    # Z = top PCA directions MINUS the language directions
    # Simpler: project out language directions from top PCA
    top_pca = Vt_all[:k+10]  # over-sample
    # Project out language subspace
    for ld in lang_dirs:
        ld = ld / (np.linalg.norm(ld) + 1e-10)
        top_pca = top_pca - np.outer(top_pca @ ld, ld)
    # Re-orthogonalize via SVD
    U_z, S_z, Vt_z = np.linalg.svd(top_pca, full_matrices=False)
    z_basis = Vt_z[:k]  # (k, 2048)
    return z_basis

# Compute Z basis at representative layers
Z_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
z_bases = {}
for l in Z_LAYERS:
    z_bases[l] = compute_z_basis(l, k=20)
    print(f"  L{l}: Z basis shape {z_bases[l].shape}")

def get_z_basis(layer):
    """Get Z basis for a layer, interpolating from nearest computed."""
    nearest = min(Z_LAYERS, key=lambda x: abs(x - layer))
    return z_bases[nearest]

# Pre-compute Z projection matrices as tensors
z_projectors = {}
for l in range(n_layers):
    Z = get_z_basis(l)
    Z_t = torch.tensor(Z, dtype=torch.float32, device=device)  # (20, 2048)
    # Projector onto Z: P_Z = Z^T @ Z (since Z rows are orthonormal)
    P_Z = Z_t.T @ Z_t  # (2048, 2048)
    z_projectors[l] = P_Z

del data  # free memory
torch.cuda.empty_cache()
print("Z bases loaded for all layers.")


# ── Problem sets ────────────────────────────────────────────────────────
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

test_en = [
    {"prompt": "Calculate 47 + 86.", "answer": "133"},
    {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answer": "60"},
    {"prompt": "What is the remainder when 100 is divided by 7?", "answer": "2"},
    {"prompt": "Calculate 15 × 8.", "answer": "120"},
    {"prompt": "An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.", "answer": "40"},
    {"prompt": "Calculate 387 × 29.", "answer": "11223"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120"},
    {"prompt": "What is the remainder when 7654 is divided by 37?", "answer": "34"},
    {"prompt": "An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.", "answer": "3475"},
    {"prompt": "A rectangle has length 47 and width 33. Find its area.", "answer": "1551"},
]
test_zh = [
    {"prompt": "计算 47 + 86 的值。", "answer": "133"},
    {"prompt": "一个长方形的长为 12，宽为 5，求其面积。", "answer": "60"},
    {"prompt": "100 除以 7 的余数是多少？", "answer": "2"},
    {"prompt": "计算 15 × 8 的值。", "answer": "120"},
    {"prompt": "等差数列首项为 2，公差为 3，求前 5 项之和。", "answer": "40"},
    {"prompt": "计算 387 × 29 的值。", "answer": "11223"},
    {"prompt": "求组合数 C(10, 3) 的值。", "answer": "120"},
    {"prompt": "7654 除以 37 的余数是多少？", "answer": "34"},
    {"prompt": "等差数列首项为 7，公差为 11，求前 25 项之和。", "answer": "3475"},
    {"prompt": "一个长方形的长为 47，宽为 33，求其面积。", "answer": "1551"},
]


# ── Generation with MLP decomposition hooks ─────────────────────────────
def generate_decomposed(prompt, mode="baseline", layers=None, max_tokens=128):
    """
    mode:
      "baseline" — no intervention
      "z_input"  — project MLP input onto Z, complement bypasses MLP
      "z_output" — full MLP input, but only keep Z component of MLP delta
      "z_output_complement" — full MLP input, zero out Z component of delta (keep only language)
    layers: which layers to intervene on (default: all)
    """
    if layers is None:
        layers = list(range(n_layers))

    input_ids = tokenizer.encode(prompt)
    handles = []
    per_layer_cosines = {}  # track how much the intervention changes each layer's output

    if mode == "z_input":
        # For z_input, we need to re-run the MLP on projected input.
        # To avoid recursion (hook calling MLP which triggers hook),
        # we use a re-entrancy guard.
        _in_hook = {}
        for li in layers:
            P_Z = z_projectors[li]
            def make_hook(layer_idx, proj):
                def hook_fn(module, input, output):
                    if _in_hook.get(layer_idx, False):
                        return  # re-entrant call, pass through
                    h_in = input[0].float()  # (batch, seq, d)
                    h_z = torch.einsum('...d,de->...e', h_in, proj)
                    _in_hook[layer_idx] = True
                    try:
                        with torch.no_grad():
                            z_output = module(h_z.to(output.dtype))
                    finally:
                        _in_hook[layer_idx] = False
                    return z_output
                return hook_fn
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_hook(li, P_Z)
                )
            )

    elif mode == "z_output":
        for li in layers:
            P_Z = z_projectors[li]
            def make_hook(layer_idx, proj):
                def hook_fn(module, input, output):
                    # Keep only Z component of MLP delta
                    delta = output.float()
                    z_delta = torch.einsum('...d,de->...e', delta, proj)
                    return z_delta.to(output.dtype)
                return hook_fn
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_hook(li, P_Z)
                )
            )

    elif mode == "z_output_complement":
        for li in layers:
            P_Z = z_projectors[li]
            I_minus_PZ = torch.eye(d_model, device=device, dtype=torch.float32) - P_Z
            def make_hook(layer_idx, proj_compl):
                def hook_fn(module, input, output):
                    # Keep only complement (language) component, zero out Z
                    delta = output.float()
                    compl_delta = torch.einsum('...d,de->...e', delta, proj_compl)
                    return compl_delta.to(output.dtype)
                return hook_fn
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_hook(li, I_minus_PZ)
                )
            )

    try:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]
        for _ in range(max_tokens - 1):
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


def extract_answer(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    ans_match = re.findall(r'(?:answer|result|value)\s*(?:is|=|:)\s*[\\$]*\s*([\d,]+)', text, re.I)
    if ans_match:
        return ans_match[-1].replace(',', '').strip()
    eq_match = re.findall(r'=\s*([\d,]+)', text)
    if eq_match:
        return eq_match[-1].replace(',', '').strip()
    nums = re.findall(r'\b\d+\b', text)
    if nums:
        return nums[-1]
    return ""


def check_correct(response, answer):
    return extract_answer(response) == str(answer)


def detect_language(text):
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return 'zh' if cjk > latin else 'en'


# ── Run experiments ─────────────────────────────────────────────────────
results = {
    "experiment": "BA: Z-MLP Decomposition",
    "model": "Qwen/Qwen2.5-3B",
    "n_layers": n_layers,
    "d_model": d_model,
    "z_dim": 20,
    "conditions": {}
}

# All conditions: baseline, z_input, z_output, z_output_complement
# Also: z_output with only adversarial/cooperative/ramp layers
CONDITIONS = [
    ("baseline", "baseline", None),
    ("z_input_all", "z_input", list(range(n_layers))),
    ("z_output_all", "z_output", list(range(n_layers))),
    ("z_output_complement_all", "z_output_complement", list(range(n_layers))),
    ("z_output_adversarial", "z_output", list(range(9, 18))),
    ("z_output_cooperative", "z_output", list(range(18, 27))),
    ("z_output_L9_L26", "z_output", list(range(9, 27))),
]

t0 = time.time()

for cond_name, mode, layers in CONDITIONS:
    print(f"\n{'='*60}")
    print(f"Condition: {cond_name} (mode={mode}, layers={layers})")
    print(f"{'='*60}")

    en_correct = 0
    zh_correct = 0
    en_details = []
    zh_details = []

    for i, prob in enumerate(test_en):
        resp = generate_decomposed(prob["prompt"], mode, layers, MAX_NEW_TOKENS)
        correct = check_correct(resp, prob["answer"])
        lang = detect_language(resp)
        en_correct += int(correct)
        en_details.append({
            "idx": i, "correct": correct, "answer": prob["answer"],
            "extracted": extract_answer(resp), "lang": lang,
            "first_80": resp[:80]
        })
        mark = "✓" if correct else "✗"
        print(f"  EN {i}: {mark} (ans={prob['answer']}, got={extract_answer(resp)}, lang={lang})")

    for i, prob in enumerate(test_zh):
        resp = generate_decomposed(prob["prompt"], mode, layers, MAX_NEW_TOKENS)
        correct = check_correct(resp, prob["answer"])
        lang = detect_language(resp)
        zh_correct += int(correct)
        zh_details.append({
            "idx": i, "correct": correct, "answer": prob["answer"],
            "extracted": extract_answer(resp), "lang": lang,
            "first_80": resp[:80]
        })
        mark = "✓" if correct else "✗"
        print(f"  ZH {i}: {mark} (ans={prob['answer']}, got={extract_answer(resp)}, lang={lang})")

    results["conditions"][cond_name] = {
        "en_correct": en_correct, "en_total": len(test_en),
        "zh_correct": zh_correct, "zh_total": len(test_zh),
        "total_correct": en_correct + zh_correct,
        "total": len(test_en) + len(test_zh),
        "en_details": en_details, "zh_details": zh_details,
    }
    print(f"\n  EN: {en_correct}/{len(test_en)}, ZH: {zh_correct}/{len(test_zh)}, "
          f"Total: {en_correct+zh_correct}/{len(test_en)+len(test_zh)}")

# ── Summary ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0
results["elapsed_s"] = elapsed

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for cond, data in results["conditions"].items():
    print(f"  {cond:30s}: EN={data['en_correct']}/{data['en_total']}, "
          f"ZH={data['zh_correct']}/{data['zh_total']}, "
          f"Total={data['total_correct']}/{data['total']}")
print(f"\nTotal time: {elapsed:.0f}s")

out_path = OUTPUT_DIR / "expBA_z_mlp_decomposition.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: {out_path}")
