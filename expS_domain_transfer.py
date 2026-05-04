"""Experiment S: Strategy Switch Across Domains

Does the language-direction flip generalize beyond math?
Same mechanism (causal language direction at L9-L26, scale -0.5), same FAT metric.
512 token budget. 5 problems per domain.

Domains: factual extraction, structured output, translation, code.
Prediction: compact-answer domains (factual, structured) → big speedup.
Code (answer IS long) → moderate. Translation → mixed (cross-lingual interference).
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re

MODEL_NAME = 'Qwen/Qwen2.5-3B'
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

n_layers = model.config.num_hidden_layers
d = model.config.hidden_size
N_TRAIN = 200
BATCH_SIZE = 16
STRIP_LAYERS = list(range(9, 27))
MAX_TOKENS = 512

print(f"Model: {MODEL_NAME} ({n_layers} layers, d={d})")
t0 = time.time()


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
# Batched direction fitting
# =============================================================================
print(f"Fitting directions...")
problems = generate_pca_problems(N_TRAIN, seed=42)
fit_layers = [10, 14, 18, 22, 26]

lang_dirs = {}
for fl in fit_layers:
    mlp_out_list = []
    def mlp_cap(module, input, output):
        mlp_out_list.append(output.detach().float())
    handle = model.model.layers[fl].mlp.register_forward_hook(mlp_cap)

    zh_deltas, en_deltas = [], []
    for lang, deltas in [("zh", zh_deltas), ("en", en_deltas)]:
        for i in range(0, N_TRAIN, BATCH_SIZE):
            batch = [p[lang] for p in problems[i:i+BATCH_SIZE]]
            mlp_out_list.clear()
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad(): model(**inputs)
            attn_mask = inputs["attention_mask"]
            last_idx = attn_mask.sum(dim=1) - 1
            out = mlp_out_list[0]
            for j in range(out.shape[0]):
                deltas.append(out[j, last_idx[j]].cpu().numpy())

    handle.remove()
    diff = np.stack(zh_deltas).mean(0) - np.stack(en_deltas).mean(0)
    norm = np.linalg.norm(diff)
    lang_dirs[fl] = torch.tensor(diff / norm, dtype=torch.float32, device=device)

def get_lang_dir(li):
    return lang_dirs[min(fit_layers, key=lambda x: abs(x - li))]

strip_dirs = {li: get_lang_dir(li) for li in STRIP_LAYERS}
print(f"Directions fitted in {time.time()-t0:.1f}s")


# =============================================================================
# Generation with tracking
# =============================================================================
def generate_with_tracking(prompt, answer_strings, dirs_dict=None, scale=1.0):
    """Generate up to MAX_TOKENS, tracking when any answer string first appears."""
    input_ids = tokenizer.encode(prompt)
    handles = []

    if dirs_dict and scale != 1.0:
        for li in STRIP_LAYERS:
            def make_hook(layer_idx, lang_dir, sc):
                def hook_fn(module, input, output):
                    delta = output.float()
                    proj = torch.sum(delta * lang_dir, dim=-1, keepdim=True)
                    lang_component = proj * lang_dir
                    modified = delta - lang_component + sc * lang_component
                    return modified.to(output.dtype)
                return hook_fn
            handles.append(
                model.model.layers[li].mlp.register_forward_hook(
                    make_hook(li, strip_dirs[li], scale)
                )
            )

    first_appearance = None
    matched_answer = None
    try:
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
        past_kv = outputs.past_key_values
        next_id = int(outputs.logits[0, -1].argmax())
        next_token = torch.tensor([[next_id]], device=device)
        generated_ids = [next_id]

        for step in range(MAX_TOKENS - 1):
            text_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if first_appearance is None:
                for ans in answer_strings:
                    if ans.lower() in text_so_far.lower():
                        first_appearance = step + 1
                        matched_answer = ans
                        break

            with torch.no_grad():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], device=device)
            if next_id == tokenizer.eos_token_id:
                text_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
                if first_appearance is None:
                    for ans in answer_strings:
                        if ans.lower() in text_so_far.lower():
                            first_appearance = len(generated_ids)
                            matched_answer = ans
                            break
                break
    finally:
        for h in handles:
            h.remove()

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if first_appearance is None:
        for ans in answer_strings:
            if ans.lower() in full_text.lower():
                first_appearance = len(generated_ids)
                matched_answer = ans
                break

    return {
        "fat": first_appearance,
        "total_tokens": len(generated_ids),
        "found": first_appearance is not None,
        "matched": matched_answer,
        "output": full_text[:400]
    }


# =============================================================================
# Domain definitions
# =============================================================================
domains = {
    "factual": [
        {"prompt": "What is the capital of France? Answer with just the city name.",
         "answers": ["Paris"]},
        {"prompt": "What is 2^10? Answer with just the number.",
         "answers": ["1024"]},
        {"prompt": "How many days in a leap year? Answer with just the number.",
         "answers": ["366"]},
        {"prompt": "What is the chemical formula for water? Answer with just the formula.",
         "answers": ["H2O"]},
        {"prompt": "What is the speed of light in m/s? Answer with just the number.",
         "answers": ["299792458", "3 × 10^8", "3×10^8", "3e8", "300000000"]},
    ],
    "structured": [
        {"prompt": "List the first 5 prime numbers, comma-separated.",
         "answers": ["2, 3, 5, 7, 11", "2,3,5,7,11"]},
        {"prompt": "List the days of the week, comma-separated.",
         "answers": ["Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday",
                     "Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday"]},
        {"prompt": "List the first 4 planets from the sun, comma-separated.",
         "answers": ["Mercury, Venus, Earth, Mars", "Mercury,Venus,Earth,Mars"]},
        {"prompt": "What are the RGB values for pure red? Format: (R,G,B)",
         "answers": ["(255,0,0)", "(255, 0, 0)", "255, 0, 0", "255,0,0"]},
        {"prompt": "What is pi to 4 decimal places?",
         "answers": ["3.1416", "3.1415"]},
    ],
    "translation": [
        {"prompt": "Translate '你好世界' to English.",
         "answers": ["Hello World", "Hello, World", "hello world"]},
        {"prompt": "Translate 'the cat sat on the mat' to Chinese.",
         "answers": ["猫坐在垫子上", "猫", "坐在"]},
        {"prompt": "Translate '今天天气很好' to English.",
         "answers": ["weather is good", "nice weather", "good weather", "beautiful weather",
                     "weather is nice", "weather is very good", "weather is great"]},
        {"prompt": "Translate 'I love programming' to Chinese.",
         "answers": ["我喜欢编程", "我爱编程", "编程"]},
        {"prompt": "Translate '数学是美丽的' to English.",
         "answers": ["math is beautiful", "mathematics is beautiful", "Math is beautiful"]},
    ],
    "code": [
        {"prompt": "Write a Python function that returns the factorial of n.",
         "answers": ["def factorial", "def fact"]},
        {"prompt": "Write a Python function that checks if a string is a palindrome.",
         "answers": ["def is_palindrome", "def palindrome", "def check_palindrome"]},
        {"prompt": "Write a Python function that returns the nth Fibonacci number.",
         "answers": ["def fibonacci", "def fib"]},
        {"prompt": "Write a Python function that sorts a list of integers.",
         "answers": ["def sort", "sorted(", ".sort()"]},
        {"prompt": "Write a Python function that counts vowels in a string.",
         "answers": ["def count_vowels", "def vowel"]},
    ],
    "math_reference": [
        {"prompt": "Calculate 47 + 86.", "answers": ["133"]},
        {"prompt": "What is the remainder when 100 is divided by 7?", "answers": ["2"]},
        {"prompt": "Find the value of C(10, 3).", "answers": ["120"]},
        {"prompt": "A rectangle has length 12 and width 5. Find its area.", "answers": ["60"]},
        {"prompt": "Calculate 15 × 8.", "answers": ["120"]},
    ],
}


# =============================================================================
# Run all domains × conditions
# =============================================================================
conditions = [
    ("baseline", None, 1.0),
    ("flip_-0.5", strip_dirs, -0.5),
]

results = {
    "experiment": "S: Domain Transfer",
    "model": MODEL_NAME,
    "max_tokens": MAX_TOKENS,
    "domains": {}
}

for domain_name, problems in domains.items():
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain_name} ({len(problems)} problems)")
    print("="*70)

    domain_results = {}
    for cond_name, dirs, scale in conditions:
        cond_data = []
        for prob in problems:
            r = generate_with_tracking(prob["prompt"], prob["answers"], dirs, scale)
            fat_str = f"FAT={r['fat']}" if r['fat'] else "NEVER"
            print(f"  [{cond_name:>10s}] {prob['prompt'][:45]:45s} → {fat_str:>10s} ({r['total_tokens']} tok)")
            cond_data.append({
                "prompt": prob["prompt"],
                "expected": prob["answers"][0],
                **r
            })
        domain_results[cond_name] = cond_data

    results["domains"][domain_name] = domain_results

# =============================================================================
# Analysis
# =============================================================================
print(f"\n{'='*70}")
print(f"EXPERIMENT S: DOMAIN TRANSFER ANALYSIS [{time.time()-t0:.0f}s]")
print("="*70)

print(f"\n  {'Domain':<15s} {'Cond':<12s} {'Found':>6s} {'Mean FAT':>10s} {'Med FAT':>10s}")
print(f"  {'─'*15} {'─'*12} {'─'*6} {'─'*10} {'─'*10}")

domain_summaries = {}
for domain_name in domains:
    for cond_name in ["baseline", "flip_-0.5"]:
        items = results["domains"][domain_name][cond_name]
        found = [x for x in items if x["found"]]
        fats = [x["fat"] for x in found]
        n_found = len(found)
        mean_fat = np.mean(fats) if fats else float('inf')
        med_fat = np.median(fats) if fats else float('inf')
        print(f"  {domain_name:<15s} {cond_name:<12s} {n_found:>4d}/5 {mean_fat:>10.1f} {med_fat:>10.1f}")

        if domain_name not in domain_summaries:
            domain_summaries[domain_name] = {}
        domain_summaries[domain_name][cond_name] = {
            "found": n_found, "mean_fat": mean_fat, "med_fat": med_fat
        }

# Speedup table
print(f"\n  {'Domain':<15s} {'Base Found':>11s} {'Flip Found':>11s} {'Base MedFAT':>12s} {'Flip MedFAT':>12s} {'Speedup':>8s}")
print(f"  {'─'*15} {'─'*11} {'─'*11} {'─'*12} {'─'*12} {'─'*8}")

for domain_name in domains:
    bs = domain_summaries[domain_name]["baseline"]
    fs = domain_summaries[domain_name]["flip_-0.5"]
    speedup = bs["med_fat"] / fs["med_fat"] if fs["med_fat"] > 0 and fs["med_fat"] != float('inf') else 0
    print(f"  {domain_name:<15s} {bs['found']:>9d}/5 {fs['found']:>9d}/5 {bs['med_fat']:>12.1f} {fs['med_fat']:>12.1f} {speedup:>7.1f}x")

results["summary"] = domain_summaries

with open("output/expS_domain_transfer.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expS_domain_transfer.json [{time.time()-t0:.0f}s total]")


# =============================================================================
# The Nugget
# =============================================================================
print(f"\n{'='*70}")
print("WHAT I SEE AFTER 50+ EXPERIMENTS")
print("="*70)
print("""
The model doesn't have two separate systems for language and math.
It has ONE system that compresses everything into a shared representational
space. Language, math, format — they're all directions in the same vector space.

What we call "verbose CoT" is the model unfolding its compressed representation
back into natural language token by token. What the flip does is suppress that
unfolding — it lets the compressed answer leak through directly.

The tug-of-war (attention pushes language up, MLP dampens it) isn't a bug —
it's the model negotiating between "express the answer in words" and "just
output the answer." Both impulses exist simultaneously in every forward pass.

The causal language direction isn't "where language lives." It's the axis along
which the model decides HOW MUCH to explain. Flip it, and you get less
explanation, more answer. Zero it, and you get something in between.

Von Neumann was right: the brain and the computer use different "languages"
internally, and the mathematical formalism we use to describe them may be
fundamentally inadequate. PCA — our best variance-based tool — misses the
causal structure entirely. The dimensions that EXPLAIN the most variance are
NOT the dimensions that CONTROL behavior. That's not just a finding about
this model. That's a finding about how we study neural networks.

The nugget: the model has a verbosity dial, and we found it. It's one
direction in a 2048-dimensional space. Turn it down, answers come faster.
Turn it up, explanations come longer. It works across languages because
math doesn't care what language you speak. It fails for reasoning because
reasoning IS the explanation — there's no shortcut to skip.
""")
