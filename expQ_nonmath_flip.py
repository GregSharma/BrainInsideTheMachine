"""Experiment Q: Language Direction Flip on Non-Math Tasks (Qwen-3B)

P3 showed flipping the causal language direction improves math by 86%.
Does it generalize to non-math reasoning? Logic, factual, code, general knowledge.

If yes: the language direction interferes with ALL computation, not just arithmetic.
If no: the interference is math-specific.

10 non-math problems. Scale -1.5 (optimal from P3). EN + ZH. 128 tokens.
"""
import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

MAX_NEW_TOKENS = 128
STRIP_LAYERS = list(range(9, 27))
N_TRAIN = 200
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


# Compute causal language directions (reuse from P3)
print("Computing causal language directions from math MLP deltas...")
problems = generate_pca_problems(N_TRAIN, seed=42)
fit_layers = [10, 14, 18, 22, 26]
lang_dirs = {}

for fl in fit_layers:
    mlp_out = {}
    def mlp_cap(module, input, output):
        mlp_out['d'] = output.detach()[:, -1, :]
    handle = model.model.layers[fl].mlp.register_forward_hook(mlp_cap)
    zh_deltas = np.zeros((N_TRAIN, d), dtype=np.float32)
    en_deltas = np.zeros((N_TRAIN, d), dtype=np.float32)
    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
        zh_deltas[i] = mlp_out['d'].cpu().float().numpy(); mlp_out.clear()
    for i, prob in enumerate(problems):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
        en_deltas[i] = mlp_out['d'].cpu().float().numpy(); mlp_out.clear()
    handle.remove()
    diff = zh_deltas.mean(axis=0) - en_deltas.mean(axis=0)
    lang_dirs[fl] = torch.tensor(diff / np.linalg.norm(diff), dtype=torch.float32, device=device)

strip_dirs = {li: lang_dirs[min(fit_layers, key=lambda x: abs(x - li))] for li in STRIP_LAYERS}

# Non-math test problems with verifiable answers
test_problems = [
    # Logic
    {"en": "If all cats are animals and Whiskers is a cat, what is Whiskers?",
     "zh": "如果所有的猫都是动物，而小白是一只猫，那么小白是什么？",
     "answer_contains": ["animal", "动物"], "category": "logic"},
    {"en": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
     "zh": "如果5台机器5分钟生产5个零件，100台机器生产100个零件需要多少分钟？",
     "answer_contains": ["5"], "category": "logic"},
    # Factual
    {"en": "What is the capital of France?",
     "zh": "法国的首都是哪里？",
     "answer_contains": ["Paris", "巴黎"], "category": "factual"},
    {"en": "What is the chemical symbol for water?",
     "zh": "水的化学式是什么？",
     "answer_contains": ["H2O"], "category": "factual"},
    {"en": "How many days are there in a leap year?",
     "zh": "闰年有多少天？",
     "answer_contains": ["366"], "category": "factual"},
    # Reasoning
    {"en": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
     "zh": "一个球棒和一个球一共1.10美元。球棒比球贵1.00美元。球多少钱？",
     "answer_contains": ["0.05", "5 cents", "5分", "0.05美元", "$0.05"], "category": "reasoning"},
    {"en": "If you rearrange the letters 'CIFAIPC' you get the name of a what? Ocean.",
     "zh": "如果重新排列字母'CIFAIPC'，你会得到什么名字？大洋。",
     "answer_contains": ["Pacific", "太平洋", "PACIFIC"], "category": "reasoning"},
    # Code-like
    {"en": "What is the output of: print(len('hello'))",
     "zh": "以下代码的输出是什么：print(len('hello'))",
     "answer_contains": ["5"], "category": "code"},
    # General knowledge
    {"en": "How many continents are there on Earth?",
     "zh": "地球上有几个大洲？",
     "answer_contains": ["7", "七"], "category": "factual"},
    {"en": "What is the square root of 144?",
     "zh": "144的平方根是多少？",
     "answer_contains": ["12"], "category": "math_simple"},
]


def run_generation(prompt, scale=1.0):
    input_ids = tokenizer.encode(prompt)
    handles = []
    if scale != 1.0:
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


def check_answer(text, answer_contains):
    text_lower = text.lower()
    return any(a.lower() in text_lower for a in answer_contains)


# =============================================================================
# Run
# =============================================================================
print(f"\n{'='*70}")
print("EXPERIMENT Q: NON-MATH LANGUAGE FLIP (GENERALIZATION TEST)")
print("=" * 70)

SCALES = [1.0, -1.5, -1.0, 0.0]  # baseline, optimal flip, exact flip, zero
results = {"experiment": "Q: Non-Math Flip Generalization", "modes": {}}

for scale in SCALES:
    label = f"scale={scale:+.1f}" if scale != 1.0 else "baseline"
    print(f"\n{'━'*70}")
    print(f"  {label}")
    print(f"{'━'*70}")

    mode_data = []
    for lang in ["en", "zh"]:
        lang_label = "EN" if lang == "en" else "ZH"
        for prob_idx, prob in enumerate(test_problems):
            prompt = prob[lang]
            text = run_generation(prompt, scale=scale)
            correct = check_answer(text, prob["answer_contains"])
            print(f"    {lang_label} [{prob['category']:>10s}] {'OK' if correct else 'FAIL'} | {text[:70]}")
            mode_data.append({
                "lang": lang, "category": prob["category"],
                "prompt": prompt, "text": text, "correct": correct,
            })

    n_correct = sum(1 for r in mode_data if r["correct"])
    total = len(mode_data)
    en_c = sum(1 for r in mode_data if r["lang"] == "en" and r["correct"])
    zh_c = sum(1 for r in mode_data if r["lang"] == "zh" and r["correct"])
    print(f"\n  {label}: {n_correct}/{total} ({en_c}/10 EN, {zh_c}/10 ZH)")

    # Per-category
    cats = sorted(set(r["category"] for r in mode_data))
    for cat in cats:
        cat_c = sum(1 for r in mode_data if r["category"] == cat and r["correct"])
        cat_t = sum(1 for r in mode_data if r["category"] == cat)
        print(f"    {cat}: {cat_c}/{cat_t}")

    results["modes"][label] = {
        "results": mode_data, "total_correct": n_correct, "total": total,
        "en_correct": en_c, "zh_correct": zh_c,
    }

# Summary
print(f"\n{'='*70}")
print("Q SUMMARY")
print("=" * 70)
print(f"  {'Mode':<15s} {'Total':>7s} {'EN':>5s} {'ZH':>5s}")
print(f"  {'─'*15} {'─'*7} {'─'*5} {'─'*5}")
for label, data in results["modes"].items():
    print(f"  {label:<15s} {data['total_correct']:>4d}/20 {data['en_correct']:>3d}/10 {data['zh_correct']:>3d}/10")

bl = results["modes"]["baseline"]["total_correct"]
flip = results["modes"].get("scale=-1.5", {}).get("total_correct", 0)
if flip > bl + 2:
    verdict = "GENERALIZES: Language flip improves non-math reasoning too"
elif flip >= bl:
    verdict = "NEUTRAL: Flip doesn't hurt non-math but doesn't help either"
else:
    verdict = "MATH-SPECIFIC: Flip hurts non-math tasks"

print(f"\n  VERDICT: {verdict}")
results["verdict"] = verdict

with open("output/expQ_nonmath_flip.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expQ_nonmath_flip.json")
