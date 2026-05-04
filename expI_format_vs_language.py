"""Experiment I: Format Attractor vs Language Attractor

If L27+ is a FORMAT attractor (not a language attractor), then:
1. Injecting zh math L8 state into an ENGLISH prompt should produce Chinese math formatting
2. Injecting zh math L8 state into a PROSE prompt should produce math formatting, not prose
3. Injecting L8 from a simple problem into a hard problem's prompt should produce
   simple-style output (wrong content, right format)

If L27+ is a LANGUAGE attractor, then:
1. The output language should match the L8 state's language (Chinese)
2. But the output FORMAT should match the prompt's format (prose vs math)

This disentangles: does the attractor preserve language, format, or both?
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128

# --- Diverse prompts covering different formats ---
# Math prompts (Chinese and English)
zh_math = "如果 x + 5 = 12，求 x 的值。\n"
en_math = "If x + 5 = 12, find the value of x.\n"

# Prose/narrative prompts (Chinese and English)
zh_prose = "请描述一下春天的景色。\n"  # "Describe the scenery of spring"
en_prose = "Describe the scenery of spring.\n"

# Code prompt
code_prompt = "Write a Python function to compute factorial.\n"

# List/enumeration prompt
zh_list = "列出三种常见的水果。\n"  # "List three common fruits"
en_list = "List three common fruits.\n"

# Translation prompt
translate_prompt = "Translate '你好世界' to English.\n"

# Reasoning/logic prompt (Chinese)
zh_logic = "如果所有的猫都是动物，而小白是一只猫，那么小白是什么？\n"
# "If all cats are animals, and Xiaobai is a cat, then what is Xiaobai?"

all_prompts = {
    "zh_math": zh_math,
    "en_math": en_math,
    "zh_prose": zh_prose,
    "en_prose": en_prose,
    "code": code_prompt,
    "zh_list": zh_list,
    "en_list": en_list,
    "translate": translate_prompt,
    "zh_logic": zh_logic,
}


def extract_L8_hidden(prompt):
    """Extract L8 hidden state for the last token of a prompt."""
    input_ids = tokenizer.encode(prompt)
    captured = {}
    def hook_fn(module, input, output):
        if not captured:
            hidden = output if not isinstance(output, tuple) else output[0]
            captured["h"] = hidden[0, -1, :].detach().cpu().float().numpy()
    handle = model.model.layers[8].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(torch.tensor([input_ids], device=device))
    finally:
        handle.remove()
    return captured["h"]


def generate_baseline(prompt, max_new_tokens=MAX_NEW_TOKENS):
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        outputs = model.generate(
            torch.tensor([input_ids], device=device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)


def generate_with_injection(prompt, h_vector, inject_layer=27, max_new_tokens=MAX_NEW_TOKENS):
    """Inject h_vector at inject_layer on prefill only."""
    input_ids = tokenizer.encode(prompt)
    injected = [False]
    def hook_fn(module, input, output):
        if not injected[0]:
            hidden = output if not isinstance(output, tuple) else output[0]
            vec = torch.tensor(h_vector, dtype=hidden.dtype, device=hidden.device)
            hidden[0, -1, :] = vec
            injected[0] = True
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return output
    handle = model.model.layers[inject_layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([input_ids], device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    finally:
        handle.remove()
    return gen


def classify_output(text):
    """Classify output along multiple dimensions."""
    zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_chars = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))

    # Language
    if zh_chars > en_chars * 2:
        lang = "chinese"
    elif en_chars > zh_chars * 2:
        lang = "english"
    else:
        lang = "mixed"

    # Format detection
    has_equation = any(c in text for c in ['=', '×', '÷', '+']) and any(c.isdigit() for c in text)
    has_code = '```' in text or 'def ' in text or 'return ' in text or 'print(' in text
    has_list = text.count('\n') > 2 and any(text.strip().startswith(p) for p in ['1.', '1、', '-', '•', '*'])
    has_bullets = any(line.strip().startswith(('1.', '2.', '3.', '1、', '2、', '3、', '-', '•'))
                      for line in text.split('\n') if line.strip())
    has_math_steps = any(p in text for p in ['因此', '所以', 'therefore', 'Thus', '得到', '解：'])

    if has_code:
        fmt = "code"
    elif has_equation and has_math_steps:
        fmt = "math_solution"
    elif has_equation:
        fmt = "math_expression"
    elif has_bullets or has_list:
        fmt = "list"
    else:
        fmt = "prose"

    return {"language": lang, "format": fmt, "zh_chars": zh_chars, "en_chars": en_chars}


# =============================================================================
# Extract L8 states from diverse prompts
# =============================================================================
print("=" * 70)
print("EXTRACTING L8 HIDDEN STATES FROM ALL PROMPTS")
print("=" * 70)

h_L8_states = {}
baselines = {}
for name, prompt in all_prompts.items():
    h = extract_L8_hidden(prompt)
    h_L8_states[name] = h
    baseline = generate_baseline(prompt)
    baselines[name] = baseline
    cls = classify_output(baseline)
    print(f"  {name}: norm={np.linalg.norm(h):.1f}, lang={cls['language']}, fmt={cls['format']}")
    print(f"    Baseline: {baseline[:80]}...")

# =============================================================================
# CROSS-INJECTION MATRIX: inject every L8 state into every prompt
# =============================================================================
print(f"\n{'='*70}")
print("CROSS-INJECTION MATRIX")
print("="*70)

# Key test cases (not exhaustive — focus on format vs language disentanglement)
test_cases = [
    # (source_state, target_prompt, hypothesis)
    ("zh_math", "en_math", "Same task different language → tests language control"),
    ("zh_math", "en_prose", "Math state into prose prompt → tests format control"),
    ("zh_math", "code", "Math state into code prompt → tests format control"),
    ("en_math", "zh_math", "Reverse language swap → tests symmetry"),
    ("zh_prose", "zh_math", "Prose state into math prompt → tests format control"),
    ("zh_prose", "en_prose", "Same task different language → tests language control"),
    ("code", "zh_math", "Code state into math prompt → tests format control"),
    ("zh_list", "en_prose", "List state into prose prompt → tests format control"),
    ("zh_logic", "en_math", "Logic state into math prompt → tests content preservation"),
    ("en_list", "zh_list", "Same task different language → tests language control"),
]

results = {
    "experiment": "I: Format Attractor vs Language Attractor",
    "baselines": {},
    "cross_injections": [],
}

for name, prompt in all_prompts.items():
    cls = classify_output(baselines[name])
    results["baselines"][name] = {
        "prompt": prompt.strip(),
        "generation": baselines[name],
        "classification": cls,
    }

for src_name, tgt_name, hypothesis in test_cases:
    h_src = h_L8_states[src_name]
    tgt_prompt = all_prompts[tgt_name]

    gen = generate_with_injection(tgt_prompt, h_src, inject_layer=27)
    cls = classify_output(gen)

    src_cls = classify_output(baselines[src_name])
    tgt_cls = classify_output(baselines[tgt_name])

    # What did the output follow?
    lang_follows_source = cls["language"] == src_cls["language"]
    lang_follows_target = cls["language"] == tgt_cls["language"]
    fmt_follows_source = cls["format"] == src_cls["format"]
    fmt_follows_target = cls["format"] == tgt_cls["format"]

    print(f"\n  {src_name} L8 → {tgt_name} prompt:")
    print(f"    Hypothesis: {hypothesis}")
    print(f"    Source baseline: lang={src_cls['language']}, fmt={src_cls['format']}")
    print(f"    Target baseline: lang={tgt_cls['language']}, fmt={tgt_cls['format']}")
    print(f"    OUTPUT:          lang={cls['language']}, fmt={cls['format']}")
    print(f"    Language follows: {'SOURCE' if lang_follows_source else 'TARGET' if lang_follows_target else 'NEITHER'}")
    print(f"    Format follows:   {'SOURCE' if fmt_follows_source else 'TARGET' if fmt_follows_target else 'NEITHER'}")
    print(f"    Text: {gen[:100]}...")

    results["cross_injections"].append({
        "source": src_name,
        "target": tgt_name,
        "hypothesis": hypothesis,
        "generation": gen,
        "output_classification": cls,
        "source_baseline_classification": src_cls,
        "target_baseline_classification": tgt_cls,
        "language_follows_source": lang_follows_source,
        "language_follows_target": lang_follows_target,
        "format_follows_source": fmt_follows_source,
        "format_follows_target": fmt_follows_target,
    })

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT I — SUMMARY")
print("="*70)

lang_source = sum(1 for r in results["cross_injections"] if r["language_follows_source"])
lang_target = sum(1 for r in results["cross_injections"] if r["language_follows_target"])
fmt_source = sum(1 for r in results["cross_injections"] if r["format_follows_source"])
fmt_target = sum(1 for r in results["cross_injections"] if r["format_follows_target"])
total = len(results["cross_injections"])

print(f"  Language follows SOURCE (L8 state): {lang_source}/{total}")
print(f"  Language follows TARGET (prompt):    {lang_target}/{total}")
print(f"  Format follows SOURCE (L8 state):   {fmt_source}/{total}")
print(f"  Format follows TARGET (prompt):     {fmt_target}/{total}")

if lang_source > lang_target and fmt_target > fmt_source:
    print("\n  → LANGUAGE ATTRACTOR: L8 state controls language, prompt controls format")
elif fmt_source > fmt_target and lang_source > lang_target:
    print("\n  → FORMAT+LANGUAGE ATTRACTOR: L8 state controls both format and language")
elif fmt_target > fmt_source and lang_target > lang_source:
    print("\n  → PROMPT DOMINATES: The attractor mostly follows the prompt, not the injected state")
else:
    print("\n  → MIXED: Neither clean language nor clean format attractor")

results["summary"] = {
    "language_follows_source": lang_source,
    "language_follows_target": lang_target,
    "format_follows_source": fmt_source,
    "format_follows_target": fmt_target,
    "total_tests": total,
}

with open("output/expI_format_vs_language.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expI_format_vs_language.json")
