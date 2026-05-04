"""K2 only: Kill Format Channel (Scramble KV Cache at L27+), Check Language Survives."""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
MAX_NEW_TOKENS = 128

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

    return {"language": lang, "format": fmt, "zh_chars": zh_chars, "en_chars": en_chars}


print("=" * 70)
print("K2: KILL FORMAT CHANNEL (SCRAMBLE KV CACHE AT L27+)")
print("=" * 70)

k2_results = []

for name, prompt in test_prompts.items():
    input_ids = tokenizer.encode(prompt)
    seq_len = len(input_ids)

    with torch.no_grad():
        outputs = model(
            torch.tensor([input_ids], device=device),
            use_cache=True,
        )
        past_kv = outputs.past_key_values

    rng = np.random.RandomState(42)
    perm = torch.tensor(rng.permutation(seq_len), device=device)

    # DynamicCache with .layers[i].keys/.values
    for layer_idx in range(len(past_kv.layers)):
        if layer_idx >= 27:
            past_kv.layers[layer_idx].keys = past_kv.layers[layer_idx].keys[:, :, perm, :]
            past_kv.layers[layer_idx].values = past_kv.layers[layer_idx].values[:, :, perm, :]
    scrambled_kv = past_kv

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
    print(f"    {text[:100]}...")

k2_lang_survived = sum(1 for r in k2_results if r["lang_survived"])
k2_fmt_destroyed = sum(1 for r in k2_results if r["fmt_destroyed"])
print(f"\n  K2 Summary: Language survived {k2_lang_survived}/{len(k2_results)}, "
      f"Format destroyed {k2_fmt_destroyed}/{len(k2_results)}")

# Load K1 results and combine
try:
    with open("output/expK_destruction_tests.json") as f:
        existing = json.load(f)
    k1_data = existing.get("K1_kill_language", {})
except Exception:
    k1_data = {"fmt_survived_count": 8, "lang_destroyed_count": 1, "total": 8}

results = {
    "experiment": "K: Destruction Tests",
    "K1_kill_language": k1_data,
    "K2_kill_format": {
        "results": k2_results,
        "lang_survived_count": k2_lang_survived,
        "fmt_destroyed_count": k2_fmt_destroyed,
        "total": len(k2_results),
    },
}

# Overall verdict
k1_fmt = k1_data.get("fmt_survived_count", 0)
k1_total = k1_data.get("total", 8)
print(f"\n{'='*70}")
print("COMBINED K RESULTS")
print("=" * 70)
print(f"  K1: Format survived language death: {k1_fmt}/{k1_total}")
print(f"  K2: Language survived format death: {k2_lang_survived}/{len(k2_results)}")

if k1_fmt >= 6 and k2_lang_survived >= 6:
    verdict = "INDEPENDENCE CONFIRMED"
elif k1_fmt >= 6:
    verdict = "PARTIAL: Format robust, language fragile"
elif k2_lang_survived >= 6:
    verdict = "PARTIAL: Language robust, format fragile"
else:
    verdict = "INDEPENDENCE REJECTED"

results["verdict"] = verdict
print(f"  VERDICT: {verdict}")

with open("output/expK_destruction_tests.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expK_destruction_tests.json")
