"""Experiment M2: Is L30 attention-MLP cooperation or conflict during computation?

Exp M showed L30 MLP drives the computation spike (3.05x). But does attention
COOPERATE with MLP during computation (both push toward the answer) or does
the tug-of-war persist (attention pushes language, MLP corrects)?

Measure: cosine similarity between attention output and MLP output at L30,
separately for decisive vs template tokens. If cooperation → cos > 0 on decisive.
If conflict → cos < 0 (like the tug-of-war).

Also: decompose L30 MLP output along the language PCs to see if the 3x spike
is language-related or pure computation.
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
d = model.config.hidden_size
N_PCA = 200


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


# Fit language PCA at L30
print("Fitting language PCA at L30...")
problems = generate_pca_problems(N_PCA, seed=42)
layer_output = {}
def capture_hook(module, input, output):
    h = output[0] if isinstance(output, tuple) else output
    layer_output['h'] = h.detach()[:, -1, :]

handle = model.model.layers[30].register_forward_hook(capture_hook)

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
pca = PCA(n_components=10)
pca.fit(combined)
lang_pcs = pca.components_
print(f"  L30 PC0 variance: {pca.explained_variance_ratio_[0]:.1%}, top 10: {sum(pca.explained_variance_ratio_):.1%}")

# Test problems
test_problems = [
    {"prompt": "Calculate 47 + 86.", "answer": "133"},
    {"prompt": "Calculate 123 × 45.", "answer": "5535"},
    {"prompt": "What is the remainder when 7654 is divided by 37?", "answer": "34"},
    {"prompt": "Find the value of C(10, 3).", "answer": "120"},
    {"prompt": "An arithmetic sequence has first term 3 and common difference 7. Find the sum of the first 20 terms.",
     "answer": "1390"},
]

print(f"\n{'='*70}")
print("L30 ATTENTION-MLP COOPERATION ANALYSIS")
print("=" * 70)

results = {"experiment": "M2: L30 Cooperation", "problems": []}

for prob_idx, prob in enumerate(test_problems):
    print(f"\n  Problem {prob_idx}: {prob['prompt'][:60]}")

    input_ids = tokenizer.encode(prob["prompt"])
    with torch.no_grad():
        outputs = model(torch.tensor([input_ids], device=device), use_cache=True)
    past_kv = outputs.past_key_values
    first_token_id = int(outputs.logits[0, -1].argmax())
    next_token = torch.tensor([[first_token_id]], device=device)
    generated_ids = [first_token_id]
    token_data = []

    for step in range(MAX_NEW_TOKENS - 1):
        captures = {}

        def make_attn_hook():
            def hook_fn(module, input, output):
                attn_out = output[0] if isinstance(output, tuple) else output
                captures['attn'] = attn_out[0, -1, :].detach().cpu().float().numpy()
            return hook_fn

        def make_mlp_hook():
            def hook_fn(module, input, output):
                mlp_out = output[0] if isinstance(output, tuple) else output
                captures['mlp'] = mlp_out[0, -1, :].detach().cpu().float().numpy()
            return hook_fn

        h1 = model.model.layers[30].self_attn.register_forward_hook(make_attn_hook())
        h2 = model.model.layers[30].mlp.register_forward_hook(make_mlp_hook())

        with torch.no_grad():
            out = model(next_token, past_key_values=past_kv, use_cache=True)

        h1.remove()
        h2.remove()

        past_kv = out.past_key_values
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        logit_entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        next_id = int(logits.argmax())

        attn_vec = captures.get('attn', np.zeros(d))
        mlp_vec = captures.get('mlp', np.zeros(d))

        attn_norm = float(np.linalg.norm(attn_vec))
        mlp_norm = float(np.linalg.norm(mlp_vec))

        # Cosine similarity between attention and MLP outputs
        if attn_norm > 1e-8 and mlp_norm > 1e-8:
            cos_sim = float(np.dot(attn_vec, mlp_vec) / (attn_norm * mlp_norm))
        else:
            cos_sim = 0.0

        # Language content of each component
        if attn_norm > 1e-8:
            attn_unit = attn_vec / attn_norm
            attn_lang_frac = sum(float(attn_unit @ pc)**2 for pc in lang_pcs)
        else:
            attn_lang_frac = 0.0

        if mlp_norm > 1e-8:
            mlp_unit = mlp_vec / mlp_norm
            mlp_lang_frac = sum(float(mlp_unit @ pc)**2 for pc in lang_pcs)
        else:
            mlp_lang_frac = 0.0

        # PC0 projection of each (signed — positive or negative language push)
        attn_pc0 = float(np.dot(attn_vec, lang_pcs[0]))
        mlp_pc0 = float(np.dot(mlp_vec, lang_pcs[0]))

        decoded_token = tokenizer.decode([next_id])
        is_math = any(c.isdigit() for c in decoded_token) or any(c in decoded_token for c in '=+-×÷*/')

        entry = {
            "step": step,
            "token_text": decoded_token,
            "is_math": is_math,
            "logit_entropy": logit_entropy,
            "cos_sim_attn_mlp": cos_sim,
            "attn_norm": attn_norm,
            "mlp_norm": mlp_norm,
            "attn_lang_frac": float(attn_lang_frac),
            "mlp_lang_frac": float(mlp_lang_frac),
            "attn_pc0": attn_pc0,
            "mlp_pc0": mlp_pc0,
            "tug_of_war": bool(np.sign(attn_pc0) != np.sign(mlp_pc0)),
        }
        token_data.append(entry)
        generated_ids.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)
        if next_id == tokenizer.eos_token_id:
            break

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    entropies = [t["logit_entropy"] for t in token_data]
    median_entropy = np.median(entropies)
    for t in token_data:
        t["is_decisive"] = bool(t["logit_entropy"] < median_entropy)

    decisive = [t for t in token_data if t["is_decisive"]]
    template = [t for t in token_data if not t["is_decisive"]]

    d_cos = np.mean([t["cos_sim_attn_mlp"] for t in decisive])
    t_cos = np.mean([t["cos_sim_attn_mlp"] for t in template])
    d_lang_attn = np.mean([t["attn_lang_frac"] for t in decisive])
    t_lang_attn = np.mean([t["attn_lang_frac"] for t in template])
    d_lang_mlp = np.mean([t["mlp_lang_frac"] for t in decisive])
    t_lang_mlp = np.mean([t["mlp_lang_frac"] for t in template])
    d_tow = sum(1 for t in decisive if t["tug_of_war"]) / max(len(decisive), 1)
    t_tow = sum(1 for t in template if t["tug_of_war"]) / max(len(template), 1)

    print(f"    Cos(attn, mlp): decisive={d_cos:+.3f} vs template={t_cos:+.3f}")
    print(f"    Language frac (attn): decisive={d_lang_attn:.1%} vs template={t_lang_attn:.1%}")
    print(f"    Language frac (mlp):  decisive={d_lang_mlp:.1%} vs template={t_lang_mlp:.1%}")
    print(f"    Tug-of-war (opposite PC0 signs): decisive={d_tow:.0%} vs template={t_tow:.0%}")

    prob_result = {
        "prompt": prob["prompt"],
        "cos_decisive": float(d_cos),
        "cos_template": float(t_cos),
        "lang_attn_decisive": float(d_lang_attn),
        "lang_attn_template": float(t_lang_attn),
        "lang_mlp_decisive": float(d_lang_mlp),
        "lang_mlp_template": float(t_lang_mlp),
        "tow_decisive": float(d_tow),
        "tow_template": float(t_tow),
    }
    results["problems"].append(prob_result)

# Summary
print(f"\n{'='*70}")
print("M2 SUMMARY")
print("=" * 70)

avg_d_cos = np.mean([p["cos_decisive"] for p in results["problems"]])
avg_t_cos = np.mean([p["cos_template"] for p in results["problems"]])
avg_d_lang_mlp = np.mean([p["lang_mlp_decisive"] for p in results["problems"]])
avg_t_lang_mlp = np.mean([p["lang_mlp_template"] for p in results["problems"]])
avg_d_tow = np.mean([p["tow_decisive"] for p in results["problems"]])
avg_t_tow = np.mean([p["tow_template"] for p in results["problems"]])

print(f"  Cos(attn, mlp): decisive={avg_d_cos:+.3f} vs template={avg_t_cos:+.3f}")
print(f"  MLP language frac: decisive={avg_d_lang_mlp:.1%} vs template={avg_t_lang_mlp:.1%}")
print(f"  Tug-of-war rate: decisive={avg_d_tow:.0%} vs template={avg_t_tow:.0%}")

if avg_d_cos > avg_t_cos + 0.05:
    print(f"\n  → COOPERATION during computation: attn and MLP align MORE on decisive tokens")
elif avg_d_cos < avg_t_cos - 0.05:
    print(f"\n  → INCREASED CONFLICT during computation")
else:
    print(f"\n  → NO CHANGE in cooperation vs conflict pattern")

if avg_d_lang_mlp < avg_t_lang_mlp * 0.8:
    print(f"  → MLP SHIFTS from language to computation on decisive tokens")
elif avg_d_lang_mlp > avg_t_lang_mlp * 1.2:
    print(f"  → MLP does MORE language work on decisive tokens")
else:
    print(f"  → MLP language content similar across token types")

results["summary"] = {
    "cos_decisive": float(avg_d_cos),
    "cos_template": float(avg_t_cos),
    "mlp_lang_decisive": float(avg_d_lang_mlp),
    "mlp_lang_template": float(avg_t_lang_mlp),
    "tow_decisive": float(avg_d_tow),
    "tow_template": float(avg_t_tow),
}

with open("output/expM2_l30_cooperation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to output/expM2_l30_cooperation.json")
