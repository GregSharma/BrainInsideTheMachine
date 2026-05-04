"""Experiment J: Full Independence Matrix (von Neumann Battery).

Exp I showed language follows L8 state (7/10), format follows prompt (7/10) on
10 cherry-picked cases. The theory predicts these are INDEPENDENT control channels.

Test: run the FULL 9x9 cross-injection matrix. All 72 off-diagonal combinations.
Score each for language-follows-source and format-follows-target.
If independence holds, both should be 85%+ across the full matrix.
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

# --- All 9 prompts from Exp I ---
all_prompts = {
    "zh_math": "如果 x + 5 = 12，求 x 的值。\n",
    "en_math": "If x + 5 = 12, find the value of x.\n",
    "zh_prose": "请描述一下春天的景色。\n",
    "en_prose": "Describe the scenery of spring.\n",
    "code": "Write a Python function to compute factorial.\n",
    "zh_list": "列出三种常见的水果。\n",
    "en_list": "List three common fruits.\n",
    "translate": "Translate '你好世界' to English.\n",
    "zh_logic": "如果所有的猫都是动物，而小白是一只猫，那么小白是什么？\n",
}

# Expected properties of each prompt
PROMPT_LANGUAGE = {
    "zh_math": "chinese", "en_math": "english",
    "zh_prose": "chinese", "en_prose": "english",
    "code": "english",
    "zh_list": "chinese", "en_list": "english",
    "translate": "english",  # prompt is English, even though it contains Chinese
    "zh_logic": "chinese",
}


def extract_L8_hidden(prompt):
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


def generate_baseline(prompt):
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        outputs = model.generate(
            torch.tensor([input_ids], device=device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)


def generate_with_injection(prompt, h_vector, inject_layer=27):
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
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    finally:
        handle.remove()
    return gen


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
        fmt = "math_solution"
    elif has_equation:
        fmt = "math_expression"
    elif has_bullets:
        fmt = "list"
    else:
        fmt = "prose"

    return {"language": lang, "format": fmt, "zh_chars": zh_chars, "en_chars": en_chars}


# =============================================================================
# Step 1: Extract all L8 states and baselines
# =============================================================================
print("=" * 70)
print("EXTRACTING L8 HIDDEN STATES AND BASELINES")
print("=" * 70)

h_L8_states = {}
baselines = {}
baseline_cls = {}
prompt_names = list(all_prompts.keys())

for name in prompt_names:
    h = extract_L8_hidden(all_prompts[name])
    h_L8_states[name] = h
    bl = generate_baseline(all_prompts[name])
    baselines[name] = bl
    cls = classify_output(bl)
    baseline_cls[name] = cls
    print(f"  {name:>12s}: lang={cls['language']:>8s}, fmt={cls['format']:>15s}  |  {bl[:60]}...")

# =============================================================================
# Step 2: Full 9x9 cross-injection matrix
# =============================================================================
print(f"\n{'='*70}")
print("FULL CROSS-INJECTION MATRIX (72 off-diagonal)")
print("=" * 70)

results = {
    "experiment": "J: Full Independence Matrix",
    "baselines": {},
    "matrix": [],
}

for name in prompt_names:
    results["baselines"][name] = {
        "prompt": all_prompts[name].strip(),
        "generation": baselines[name],
        "classification": baseline_cls[name],
    }

lang_follows_source_count = 0
fmt_follows_target_count = 0
both_independent_count = 0
total_cross = 0

# Track failures by category
failures_lang = []
failures_fmt = []

for src_name in prompt_names:
    for tgt_name in prompt_names:
        if src_name == tgt_name:
            continue

        total_cross += 1
        h_src = h_L8_states[src_name]
        gen = generate_with_injection(all_prompts[tgt_name], h_src, inject_layer=27)
        cls = classify_output(gen)

        src_cls = baseline_cls[src_name]
        tgt_cls = baseline_cls[tgt_name]

        lang_follows_source = cls["language"] == src_cls["language"]
        fmt_follows_target = cls["format"] == tgt_cls["format"]

        if lang_follows_source:
            lang_follows_source_count += 1
        else:
            failures_lang.append((src_name, tgt_name, src_cls["language"],
                                  tgt_cls["language"], cls["language"]))

        if fmt_follows_target:
            fmt_follows_target_count += 1
        else:
            failures_fmt.append((src_name, tgt_name, src_cls["format"],
                                 tgt_cls["format"], cls["format"]))

        if lang_follows_source and fmt_follows_target:
            both_independent_count += 1

        entry = {
            "source": src_name,
            "target": tgt_name,
            "generation": gen,
            "output_cls": cls,
            "src_baseline_cls": src_cls,
            "tgt_baseline_cls": tgt_cls,
            "lang_follows_source": lang_follows_source,
            "fmt_follows_target": fmt_follows_target,
        }
        results["matrix"].append(entry)

        marker = ""
        if not lang_follows_source:
            marker += " [LANG FAIL]"
        if not fmt_follows_target:
            marker += " [FMT FAIL]"

        print(f"  {src_name:>12s} → {tgt_name:<12s}: "
              f"lang={cls['language']:>8s} (src={src_cls['language']:>8s}) "
              f"fmt={cls['format']:>15s} (tgt={tgt_cls['format']:>15s})"
              f"{marker}")

# =============================================================================
# Step 3: Summary
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT J — SUMMARY")
print("=" * 70)
print(f"  Total cross-injections: {total_cross}")
print(f"  Language follows SOURCE (L8 state): {lang_follows_source_count}/{total_cross} "
      f"= {lang_follows_source_count/total_cross:.0%}")
print(f"  Format follows TARGET (prompt):     {fmt_follows_target_count}/{total_cross} "
      f"= {fmt_follows_target_count/total_cross:.0%}")
print(f"  BOTH independent:                   {both_independent_count}/{total_cross} "
      f"= {both_independent_count/total_cross:.0%}")

# Failure analysis
if failures_lang:
    print(f"\n  LANGUAGE FAILURES ({len(failures_lang)}):")
    for src, tgt, src_l, tgt_l, out_l in failures_lang:
        print(f"    {src}→{tgt}: expected {src_l}, got {out_l} (target was {tgt_l})")

if failures_fmt:
    print(f"\n  FORMAT FAILURES ({len(failures_fmt)}):")
    for src, tgt, src_f, tgt_f, out_f in failures_fmt:
        print(f"    {src}→{tgt}: expected {tgt_f}, got {out_f} (source was {src_f})")

# Check if failures cluster around Chinese-overwhelms-English
zh_src_lang_fails = sum(1 for s, t, sl, tl, ol in failures_lang
                        if PROMPT_LANGUAGE.get(s) == "chinese")
en_src_lang_fails = sum(1 for s, t, sl, tl, ol in failures_lang
                        if PROMPT_LANGUAGE.get(s) == "english")
print(f"\n  Language failures with zh source: {zh_src_lang_fails}")
print(f"  Language failures with en source: {en_src_lang_fails}")
if en_src_lang_fails > zh_src_lang_fails:
    print("  → Failures cluster where English source tries to override Chinese target (asymmetry)")
else:
    print("  → No clear asymmetry pattern in failures")

results["summary"] = {
    "total_cross": total_cross,
    "lang_follows_source": lang_follows_source_count,
    "lang_follows_source_pct": lang_follows_source_count / total_cross,
    "fmt_follows_target": fmt_follows_target_count,
    "fmt_follows_target_pct": fmt_follows_target_count / total_cross,
    "both_independent": both_independent_count,
    "both_independent_pct": both_independent_count / total_cross,
    "lang_failures": [{"src": s, "tgt": t, "expected": sl, "got": ol}
                      for s, t, sl, tl, ol in failures_lang],
    "fmt_failures": [{"src": s, "tgt": t, "expected": tf, "got": of}
                     for s, t, sf, tf, of in failures_fmt],
}

# Independence test: if lang and fmt are truly independent, the joint probability
# should equal the product of marginals
p_lang = lang_follows_source_count / total_cross
p_fmt = fmt_follows_target_count / total_cross
p_both_observed = both_independent_count / total_cross
p_both_expected = p_lang * p_fmt
independence_ratio = p_both_observed / max(p_both_expected, 0.01)

print(f"\n  INDEPENDENCE TEST:")
print(f"    P(lang correct) = {p_lang:.3f}")
print(f"    P(fmt correct)  = {p_fmt:.3f}")
print(f"    P(both) observed = {p_both_observed:.3f}")
print(f"    P(both) expected (if independent) = {p_both_expected:.3f}")
print(f"    Ratio observed/expected = {independence_ratio:.2f}")
if 0.8 < independence_ratio < 1.2:
    print("    → INDEPENDENCE CONFIRMED (ratio ≈ 1.0)")
elif independence_ratio > 1.2:
    print("    → POSITIVE COUPLING (they help each other)")
else:
    print("    → NEGATIVE COUPLING (they interfere)")

results["summary"]["independence_ratio"] = independence_ratio
results["summary"]["p_lang"] = p_lang
results["summary"]["p_fmt"] = p_fmt
results["summary"]["p_both_observed"] = p_both_observed
results["summary"]["p_both_expected"] = p_both_expected

with open("output/expJ_independence_matrix.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expJ_independence_matrix.json")
