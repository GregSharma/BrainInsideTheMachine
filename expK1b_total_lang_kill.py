"""Experiment K1b: Total Language Kill — ablate language PCs from ALL tokens at ALL layers.

K1 failed to kill language because it only ablated at L27 on the last token.
The KV cache from earlier layers still carried language signal.

K1b: register hooks on EVERY layer (L0-L35) that project out the top 10 language PCs
from EVERY token position during prefill. This should completely erase language
information from the residual stream.

If format still survives, independence is truly confirmed.
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
N_PCA = 200
N_LANG_PCS = 10

test_prompts = {
    "zh_math": "如果 x + 5 = 12，求 x 的值。\n",
    "en_math": "If x + 5 = 12, find the value of x.\n",
    "zh_prose": "请描述一下春天的景色。\n",
    "en_prose": "Describe the scenery of spring.\n",
    "code": "Write a Python function to compute factorial.\n",
    "zh_list": "列出三种常见的水果。\n",
    "en_list": "List three common fruits.\n",
    "zh_logic": "如果所有的猫都是动物，而小白是一只猫，那么小白是什么？\n",
}

PROMPT_EXPECTED_LANG = {
    "zh_math": "chinese", "en_math": "english",
    "zh_prose": "chinese", "en_prose": "english",
    "code": "english",
    "zh_list": "chinese", "en_list": "english",
    "zh_logic": "chinese",
}
PROMPT_EXPECTED_FMT = {
    "zh_math": "math", "en_math": "math",
    "zh_prose": "prose", "en_prose": "prose",
    "code": "code",
    "zh_list": "list", "en_list": "list",
    "zh_logic": "prose",
}


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


def classify_output(text):
    zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_chars = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
    if zh_chars > en_chars * 2:
        lang = "chinese"
    elif en_chars > zh_chars * 2:
        lang = "english"
    else:
        lang = "mixed"

    has_equation = any(c in text for c in ['=', '×', '÷', '+']) and any(c.isdigit() for c in text)
    has_code = 'def ' in text or 'return ' in text or 'import ' in text
    has_bullets = any(line.strip().startswith(('1.', '2.', '3.', '1、', '2、', '3、', '-', '•', '*'))
                      for line in text.split('\n') if line.strip())
    has_math_steps = any(p in text for p in ['因此', '所以', 'therefore', 'Thus', '得到', '解：',
                                              'Hence', 'Answer'])
    if has_code:
        fmt = "code"
    elif has_equation and has_math_steps:
        fmt = "math"
    elif has_equation:
        fmt = "math"
    elif has_bullets:
        fmt = "list"
    else:
        fmt = "prose"
    return {"language": lang, "format": fmt}


# =============================================================================
# Step 1: Fit language PCA at each layer
# =============================================================================
print("=" * 70)
print("FITTING LANGUAGE PCA AT MULTIPLE LAYERS")
print("=" * 70)

problems = generate_pca_problems(N_PCA, seed=42)
d = model.config.hidden_size

# We'll fit PCA at layers 0, 8, 16, 24, 32 and use the PCs from the nearest
# Actually, for a clean test, fit PCA at EVERY layer and project out at that layer
# But that's expensive. Instead, fit at L8 and L27 (the two key layers) and use
# L8 PCs for L0-L17, L27 PCs for L18-L35.

pca_layers = [8, 27]
lang_pcs_by_layer = {}

for pca_layer in pca_layers:
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

    zh_norms = np.linalg.norm(zh_hidden, axis=1, keepdims=True)
    en_norms = np.linalg.norm(en_hidden, axis=1, keepdims=True)
    combined = np.vstack([zh_hidden / zh_norms, en_hidden / en_norms])

    pca = PCA(n_components=N_LANG_PCS)
    pca.fit(combined)
    lang_pcs_by_layer[pca_layer] = pca.components_

    zh_proj = (zh_hidden / zh_norms) @ pca.components_[0]
    en_proj = (en_hidden / en_norms) @ pca.components_[0]
    cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt(
        (zh_proj.std()**2 + en_proj.std()**2) / 2
    )
    print(f"  L{pca_layer}: PC0 Cohen's d = {cohens_d:.1f}, "
          f"top {N_LANG_PCS} PCs = {sum(pca.explained_variance_ratio_):.1%}")

# Precompute PC tensors for speed
lang_pc_tensors = {}
for pca_layer, pcs in lang_pcs_by_layer.items():
    lang_pc_tensors[pca_layer] = torch.tensor(pcs, dtype=torch.float32, device=device)

# =============================================================================
# Step 2: Total Language Kill — hooks on ALL layers
# =============================================================================
print(f"\n{'='*70}")
print("K1b: TOTAL LANGUAGE KILL (ALL LAYERS, ALL TOKENS)")
print("=" * 70)

k1b_results = []

for name, prompt in test_prompts.items():
    input_ids = tokenizer.encode(prompt)

    def make_alltoken_kill_hook(pc_tensor):
        """Project out language PCs from ALL token positions."""
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h = hidden.float()  # (batch, seq, d)
            # Project out each PC from all positions at once
            # h: (1, seq, d), pc_tensor: (N_PCS, d)
            projections = torch.einsum('bsd,pd->bsp', h, pc_tensor)  # (1, seq, N_PCS)
            correction = torch.einsum('bsp,pd->bsd', projections, pc_tensor)  # (1, seq, d)
            h_cleaned = h - correction
            hidden = h_cleaned.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook_fn

    handles = []
    for layer_idx in range(36):
        # Use L8 PCs for L0-L17, L27 PCs for L18-L35
        pca_layer = 8 if layer_idx < 18 else 27
        pc_tensor = lang_pc_tensors[pca_layer]
        handles.append(
            model.model.layers[layer_idx].register_forward_hook(
                make_alltoken_kill_hook(pc_tensor)
            )
        )

    try:
        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([input_ids], device=device),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    cls = classify_output(text)
    expected_fmt = PROMPT_EXPECTED_FMT[name]
    expected_lang = PROMPT_EXPECTED_LANG[name]

    fmt_survived = cls["format"] == expected_fmt
    lang_destroyed = cls["language"] != expected_lang

    entry = {
        "prompt": name,
        "expected_lang": expected_lang,
        "expected_fmt": expected_fmt,
        "output_lang": cls["language"],
        "output_fmt": cls["format"],
        "fmt_survived": fmt_survived,
        "lang_destroyed": lang_destroyed,
        "text": text,
    }
    k1b_results.append(entry)

    marker = ""
    if fmt_survived:
        marker += " [FMT OK]"
    else:
        marker += " [FMT FAIL]"
    if lang_destroyed:
        marker += " [LANG KILLED]"
    else:
        marker += " [LANG SURVIVED]"

    print(f"  {name:>12s}: lang={cls['language']:>8s} fmt={cls['format']:>5s}{marker}")
    print(f"    {text[:100]}...")

k1b_fmt_survived = sum(1 for r in k1b_results if r["fmt_survived"])
k1b_lang_destroyed = sum(1 for r in k1b_results if r["lang_destroyed"])

print(f"\n{'='*70}")
print("K1b SUMMARY")
print("=" * 70)
print(f"  Format survived total language kill: {k1b_fmt_survived}/{len(k1b_results)} "
      f"= {k1b_fmt_survived/len(k1b_results):.0%}")
print(f"  Language destroyed:                  {k1b_lang_destroyed}/{len(k1b_results)} "
      f"= {k1b_lang_destroyed/len(k1b_results):.0%}")

if k1b_fmt_survived >= 6 and k1b_lang_destroyed >= 6:
    verdict = "STRONG INDEPENDENCE: Format survives complete language ablation"
elif k1b_fmt_survived >= 6:
    verdict = "FORMAT ROBUST but language PCs are redundant (not fully killed)"
else:
    verdict = "FORMAT AND LANGUAGE ARE ENTANGLED (format collapsed with language)"

print(f"  VERDICT: {verdict}")

results = {
    "experiment": "K1b: Total Language Kill",
    "n_pca_layers": len(pca_layers),
    "n_lang_pcs": N_LANG_PCS,
    "ablation_scope": "ALL 36 layers, ALL token positions, during ALL generation steps",
    "results": k1b_results,
    "fmt_survived_count": k1b_fmt_survived,
    "lang_destroyed_count": k1b_lang_destroyed,
    "total": len(k1b_results),
    "verdict": verdict,
}

with open("output/expK1b_total_lang_kill.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expK1b_total_lang_kill.json")
