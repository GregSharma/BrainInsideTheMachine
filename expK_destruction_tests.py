"""Experiment K: Destruction Tests for Channel Independence.

If language and format are independent channels in the residual stream:

K1: Kill Language, Check Format Survives
    At L27, project out PC0 and top language PCs. Zero out the language channel.
    Prediction: correct FORMAT but random/default language.

K2: Kill Format, Check Language Survives
    Scramble KV cache at L27+ (randomly permute token positions).
    Prediction: correct LANGUAGE but garbled format.

Both are ablation experiments that test whether destroying one channel
leaves the other intact.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import gc

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
N_PCA = 200
N_LANG_PCS = 10  # Number of language PCs to ablate

# Test prompts — diverse formats in both languages
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
    "zh_logic": "prose",  # logic is prose-like
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
    has_code = '```' in text or 'def ' in text or 'return ' in text or 'print(' in text
    has_bullets = any(line.strip().startswith(('1.', '2.', '3.', '1、', '2、', '3、', '-', '•', '*'))
                      for line in text.split('\n') if line.strip())
    has_math_steps = any(p in text for p in ['因此', '所以', 'therefore', 'Thus', '得到', '解：',
                                              'Hence', 'So,', 'we get', 'Answer'])

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

    return {"language": lang, "format": fmt, "zh_chars": zh_chars, "en_chars": en_chars,
            "total_chars": len(text), "coherent": len(text) > 10 and not all(c in ' \n\t' for c in text)}


# =============================================================================
# Step 1: Fit language PCA at L27
# =============================================================================
print("=" * 70)
print("FITTING LANGUAGE PCA AT L27")
print("=" * 70)

problems = generate_pca_problems(N_PCA, seed=42)
d = model.config.hidden_size

layer_output = {}
def capture_hook(module, input, output):
    h = output[0] if isinstance(output, tuple) else output
    layer_output['h'] = h.detach()[:, -1, :]

handle = model.model.layers[27].register_forward_hook(capture_hook)

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

# PCA on unit-normalized
zh_norms = np.linalg.norm(zh_hidden, axis=1, keepdims=True)
en_norms = np.linalg.norm(en_hidden, axis=1, keepdims=True)
zh_unit = zh_hidden / zh_norms
en_unit = en_hidden / en_norms
combined = np.vstack([zh_unit, en_unit])

pca = PCA(n_components=N_LANG_PCS)
pca.fit(combined)

# The language subspace: top N_LANG_PCS principal components
lang_subspace = pca.components_  # (N_LANG_PCS, d)

# Verify PC0 separates languages
zh_proj = zh_unit @ pca.components_[0]
en_proj = en_unit @ pca.components_[0]
cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt(
    (zh_proj.std()**2 + en_proj.std()**2) / 2
)
print(f"  PC0 Cohen's d: {cohens_d:.1f}")
print(f"  Variance explained by top {N_LANG_PCS} PCs: {sum(pca.explained_variance_ratio_):.1%}")

# =============================================================================
# Step 2: Generate baselines
# =============================================================================
print(f"\n{'='*70}")
print("GENERATING BASELINES")
print("=" * 70)

baselines = {}
for name, prompt in test_prompts.items():
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        outputs = model.generate(
            torch.tensor([input_ids], device=device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    cls = classify_output(text)
    baselines[name] = {"text": text, "cls": cls}
    print(f"  {name:>12s}: lang={cls['language']:>8s} fmt={cls['format']:>5s} | {text[:60]}...")

# =============================================================================
# K1: Kill Language Channel, Check Format Survives
# =============================================================================
print(f"\n{'='*70}")
print("K1: KILL LANGUAGE CHANNEL AT L27")
print("Projecting out top 10 language PCs from hidden state")
print("=" * 70)

k1_results = []

for name, prompt in test_prompts.items():
    input_ids = tokenizer.encode(prompt)
    injected = [False]

    def make_kill_lang_hook(subspace, flag):
        def hook_fn(module, input, output):
            if not flag[0]:
                hidden = output if not isinstance(output, tuple) else output[0]
                # Project out language subspace from last token
                h_last = hidden[0, -1, :].float()
                for pc in subspace:
                    pc_t = torch.tensor(pc, dtype=torch.float32, device=device)
                    proj = torch.dot(h_last, pc_t)
                    h_last = h_last - proj * pc_t
                hidden[0, -1, :] = h_last.to(hidden.dtype)
                flag[0] = True
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return output
        return hook_fn

    injected = [False]
    handle = model.model.layers[27].register_forward_hook(
        make_kill_lang_hook(lang_subspace, injected)
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
        handle.remove()

    cls = classify_output(text)
    expected_fmt = PROMPT_EXPECTED_FMT[name]
    expected_lang = PROMPT_EXPECTED_LANG[name]

    # Format should survive, language should be random/default
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
    k1_results.append(entry)

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
    print(f"    {text[:80]}...")

k1_fmt_survived = sum(1 for r in k1_results if r["fmt_survived"])
k1_lang_destroyed = sum(1 for r in k1_results if r["lang_destroyed"])
print(f"\n  K1 Summary: Format survived {k1_fmt_survived}/{len(k1_results)}, "
      f"Language destroyed {k1_lang_destroyed}/{len(k1_results)}")

# =============================================================================
# K2: Kill Format Channel (Scramble KV Cache), Check Language Survives
# =============================================================================
print(f"\n{'='*70}")
print("K2: KILL FORMAT CHANNEL (SCRAMBLE KV CACHE AT L27+)")
print("Randomly permuting token positions in KV cache after prefill")
print("=" * 70)

k2_results = []

for name, prompt in test_prompts.items():
    input_ids = tokenizer.encode(prompt)
    seq_len = len(input_ids)

    # Run prefill normally to get KV cache
    with torch.no_grad():
        outputs = model(
            torch.tensor([input_ids], device=device),
            use_cache=True,
        )
        past_kv = outputs.past_key_values

    # Scramble KV cache at layers 27+ (permute token positions)
    rng = np.random.RandomState(42)
    perm = torch.tensor(rng.permutation(seq_len), device=device)

    # Handle both DynamicCache and tuple-of-tuples formats
    from transformers.cache_utils import DynamicCache
    if isinstance(past_kv, DynamicCache):
        for layer_idx in range(len(past_kv.key_cache)):
            if layer_idx >= 27:
                past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, perm, :]
                past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, perm, :]
        scrambled_kv = past_kv
    else:
        scrambled_kv = []
        for layer_idx in range(len(past_kv)):
            layer_kv = past_kv[layer_idx]
            k, v = layer_kv[0], layer_kv[1]
            if layer_idx >= 27:
                k = k[:, :, perm, :]
                v = v[:, :, perm, :]
            scrambled_kv.append((k, v) + layer_kv[2:])
        scrambled_kv = tuple(scrambled_kv)

    # Generate with scrambled KV cache
    first_token_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[first_token_id]], device=device)
    tokens = [first_token_id]

    pkv = scrambled_kv
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS - 1):
            out = model(next_token, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    cls = classify_output(text)

    expected_lang = PROMPT_EXPECTED_LANG[name]
    expected_fmt = PROMPT_EXPECTED_FMT[name]

    # Language should survive, format should be garbled
    lang_survived = cls["language"] == expected_lang
    fmt_destroyed = cls["format"] != expected_fmt

    entry = {
        "prompt": name,
        "expected_lang": expected_lang,
        "expected_fmt": expected_fmt,
        "output_lang": cls["language"],
        "output_fmt": cls["format"],
        "lang_survived": lang_survived,
        "fmt_destroyed": fmt_destroyed,
        "text": text,
        "coherent": cls["coherent"],
    }
    k2_results.append(entry)

    marker = ""
    if lang_survived:
        marker += " [LANG OK]"
    else:
        marker += " [LANG FAIL]"
    if fmt_destroyed:
        marker += " [FMT KILLED]"
    else:
        marker += " [FMT SURVIVED]"

    print(f"  {name:>12s}: lang={cls['language']:>8s} fmt={cls['format']:>5s}{marker}")
    print(f"    {text[:80]}...")

k2_lang_survived = sum(1 for r in k2_results if r["lang_survived"])
k2_fmt_destroyed = sum(1 for r in k2_results if r["fmt_destroyed"])
print(f"\n  K2 Summary: Language survived {k2_lang_survived}/{len(k2_results)}, "
      f"Format destroyed {k2_fmt_destroyed}/{len(k2_results)}")

# =============================================================================
# Overall Summary
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT K — DESTRUCTION TEST SUMMARY")
print("=" * 70)
print(f"  K1 (Kill Language, Check Format):")
print(f"    Format survived:    {k1_fmt_survived}/{len(k1_results)} = {k1_fmt_survived/len(k1_results):.0%}")
print(f"    Language destroyed:  {k1_lang_destroyed}/{len(k1_results)} = {k1_lang_destroyed/len(k1_results):.0%}")
print(f"  K2 (Kill Format, Check Language):")
print(f"    Language survived:  {k2_lang_survived}/{len(k2_results)} = {k2_lang_survived/len(k2_results):.0%}")
print(f"    Format destroyed:   {k2_fmt_destroyed}/{len(k2_results)} = {k2_fmt_destroyed/len(k2_results):.0%}")

if k1_fmt_survived >= 6 and k2_lang_survived >= 6:
    verdict = "INDEPENDENCE CONFIRMED: Each channel survives destruction of the other"
elif k1_fmt_survived >= 6:
    verdict = "PARTIAL: Format survives language death, but language doesn't survive format death"
elif k2_lang_survived >= 6:
    verdict = "PARTIAL: Language survives format death, but format doesn't survive language death"
else:
    verdict = "INDEPENDENCE REJECTED: Both channels are entangled"

print(f"\n  VERDICT: {verdict}")

results = {
    "experiment": "K: Destruction Tests",
    "lang_pca_layer": 27,
    "n_lang_pcs_ablated": N_LANG_PCS,
    "pca_variance_explained": float(sum(pca.explained_variance_ratio_)),
    "pc0_cohens_d": float(cohens_d),
    "K1_kill_language": {
        "results": k1_results,
        "fmt_survived_count": k1_fmt_survived,
        "lang_destroyed_count": k1_lang_destroyed,
        "total": len(k1_results),
    },
    "K2_kill_format": {
        "results": k2_results,
        "lang_survived_count": k2_lang_survived,
        "fmt_destroyed_count": k2_fmt_destroyed,
        "total": len(k2_results),
    },
    "verdict": verdict,
}

with open("output/expK_destruction_tests.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expK_destruction_tests.json")
