"""
Exp AK: Triple Probe — Attention Procrustes + MLP Decomposition + Rank-Forcing FOAMS

OPTIMIZED: Single forward pass captures ALL layers simultaneously.
Original: 1530 forward passes. Now: ~90 forward passes (17x speedup on Parts 1+2).

PART 1 — Attention Procrustes (Greg's hailmary)
PART 2 — MLP Sub-Layer Decomposition
PART 3 — Rank-Forcing FOAMS Intervention
"""

import json, sys
import numpy as np
import torch
import torch.nn.functional as F
import random as pyrandom
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scipy.linalg import orthogonal_procrustes

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
device = "cuda"
MAX_NEW_TOKENS = 64  # Reduced from 128 — enough to detect answer

# ── Load Model ──────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="eager"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

n_heads = model.config.num_attention_heads
n_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // n_heads
d = model.config.hidden_size
d_ff = model.config.intermediate_size
n_layers = model.config.num_hidden_layers
n_rep = n_heads // n_kv_heads

print(f"Model: d={d}, d_ff={d_ff}, L={n_layers}, heads={n_heads}, kv={n_kv_heads}")


# ── Problem Sets ────────────────────────────────────────────────────────
def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    from math import comb
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            problems.append({"zh": f"计算 {a} + {b} 的值。", "en": f"Calculate {a} + {b}.", "answer": str(a+b), "cat": "arith"})
        else:
            problems.append({"zh": f"计算 {a} × {b} 的值。", "en": f"Calculate {a} × {b}.", "answer": str(a*b), "cat": "arith"})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20); k_val = rng.randint(1, min(n_val-1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。", "en": f"Find the value of C({n_val}, {k_val}).", "answer": str(comb(n_val, k_val)), "cat": "comb"})
    for _ in range(per_cat):
        a = rng.randint(50, 9999); b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？", "en": f"What is the remainder when {a} is divided by {b}?", "answer": str(a%b), "cat": "mod"})
    for _ in range(per_cat):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。", "en": f"A rectangle has length {w} and width {h}. Find its area.", "answer": str(w*h), "cat": "geom"})
    for _ in range(per_cat):
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10); n_terms = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。", "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_terms} terms.", "answer": str(n_terms*(2*a1+(n_terms-1)*d_val)//2), "cat": "seq"})
    return problems

FACTUAL_PROBLEMS = [
    {"prompt": "What is the capital of France?", "answer": "Paris", "cat": "factual"},
    {"prompt": "What is the chemical symbol for gold?", "answer": "Au", "cat": "factual"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "cat": "factual"},
    {"prompt": "What planet is closest to the Sun?", "answer": "Mercury", "cat": "factual"},
    {"prompt": "What is the largest ocean on Earth?", "answer": "Pacific", "cat": "factual"},
    {"prompt": "What is the speed of light in km/s?", "answer": "300000", "cat": "factual"},
    {"prompt": "What gas do plants absorb from the atmosphere?", "answer": "CO2", "cat": "factual"},
    {"prompt": "What is the freezing point of water in Celsius?", "answer": "0", "cat": "factual"},
    {"prompt": "How many continents are there?", "answer": "7", "cat": "factual"},
    {"prompt": "What is the smallest prime number?", "answer": "2", "cat": "factual"},
]

LOGIC_PROBLEMS = [
    {"prompt": "If all roses are flowers and all flowers need water, do all roses need water? Answer yes or no.", "answer": "yes", "cat": "logic"},
    {"prompt": "A is taller than B. B is taller than C. Is A taller than C? Answer yes or no.", "answer": "yes", "cat": "logic"},
    {"prompt": "If it rains, the ground is wet. The ground is wet. Did it rain? Answer yes, no, or uncertain.", "answer": "uncertain", "cat": "logic"},
    {"prompt": "All dogs are animals. Some animals are pets. Are all dogs pets? Answer yes, no, or uncertain.", "answer": "uncertain", "cat": "logic"},
    {"prompt": "If no fish can fly and all salmon are fish, can salmon fly? Answer yes or no.", "answer": "no", "cat": "logic"},
]

# Build test sets
all_math = generate_problems(200, seed=42)
cats = ["arith", "comb", "mod", "geom", "seq"]
test_indices = []
for cat in cats:
    cat_idxs = [i for i, p in enumerate(all_math) if p["cat"] == cat]
    test_indices.extend(cat_idxs[:4])

zh_math_test = [{"prompt": all_math[i]["zh"], "answer": all_math[i]["answer"], "cat": all_math[i]["cat"], "lang": "zh"} for i in test_indices]
en_math_test = [{"prompt": all_math[i]["en"], "answer": all_math[i]["answer"], "cat": all_math[i]["cat"], "lang": "en"} for i in test_indices]


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZED: Capture ALL layers in ONE forward pass
# ══════════════════════════════════════════════════════════════════════════

def capture_all_layers(model, tokenizer, prompt):
    """
    ONE forward pass → capture pre-layer hidden states and MLP inputs at ALL layers.
    Returns dict with layer_idx → {h_pre, mlp_input} as numpy arrays (last token only).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]

    layer_h_pre = {}   # Hidden state INPUT to each layer (for attention analysis)
    layer_mlp_in = {}  # Hidden state INPUT to each layer's MLP (post-attn residual)

    hooks = []

    for l in range(n_layers):
        layer = model.model.layers[l]

        # Capture pre-layer hidden state
        def make_pre_hook(layer_idx):
            def hook(module, inp):
                h = inp[0] if isinstance(inp, tuple) else inp
                layer_h_pre[layer_idx] = h.detach()
            return hook

        # Capture MLP input
        def make_mlp_hook(layer_idx):
            def hook(module, inp):
                h = inp[0] if isinstance(inp, tuple) else inp
                layer_mlp_in[layer_idx] = h.detach()
            return hook

        hooks.append(layer.register_forward_pre_hook(make_pre_hook(l)))
        hooks.append(layer.mlp.register_forward_pre_hook(make_mlp_hook(l)))

    with torch.no_grad():
        model(**inputs, output_attentions=False)

    for h in hooks:
        h.remove()

    return layer_h_pre, layer_mlp_in, seq_len


def compute_attention_vectors(model, layer_idx, h_pre, seq_len):
    """
    Given captured h_pre for a layer, compute a_i and c_i WITHOUT another forward pass.
    h_pre: (1, seq_len, d) tensor
    Returns a_i (d,) and c_full (d,) as numpy.
    """
    layer = model.model.layers[layer_idx]

    # Apply input layernorm
    h_normed = layer.input_layernorm(h_pre)
    a_i = h_normed[0, -1, :].float()  # (d,) last token

    # Project Q, K, V
    attn = layer.self_attn
    with torch.no_grad():
        q = attn.q_proj(h_normed)
        k = attn.k_proj(h_normed)
        v = attn.v_proj(h_normed)

    q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2).float()
    k = k.view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2).float()
    v = v.view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2).float()

    # RoPE
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(
        torch.zeros(1, seq_len, d, device=device, dtype=torch.bfloat16), pos_ids
    )
    q_rope, k_rope = apply_rotary_pos_emb(q.to(cos.dtype), k.to(cos.dtype), cos, sin)
    q_rope, k_rope = q_rope.float(), k_rope.float()

    # GQA expansion
    k_exp = k_rope.repeat_interleave(n_rep, dim=1)
    v_exp = v.repeat_interleave(n_rep, dim=1)

    # Attention for last token
    q_last = q_rope[:, :, -1:, :]
    scores = torch.matmul(q_last, k_exp.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)

    # Context vector
    c_i = torch.matmul(attn_weights, v_exp).squeeze(0).squeeze(1)  # (n_heads, head_dim)
    c_concat = c_i.reshape(1, -1)
    c_full = attn.o_proj(c_concat.to(h_normed.dtype)).float().squeeze(0)

    return a_i.detach().cpu().numpy(), c_full.detach().cpu().numpy()


def decompose_mlp(model, layer_idx, mlp_input):
    """
    Given captured MLP input, decompose gate/up/down WITHOUT another forward pass.
    mlp_input: (1, seq_len, d) tensor
    """
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    x = mlp_input[0, -1:, :].float()  # (1, d)

    with torch.no_grad():
        gate_out = mlp.gate_proj(x.to(mlp.gate_proj.weight.dtype)).float()
        up_out = mlp.up_proj(x.to(mlp.up_proj.weight.dtype)).float()
        gate_activated = F.silu(gate_out)
        gated_product = gate_activated * up_out
        mlp_output = mlp.down_proj(gated_product.to(mlp.down_proj.weight.dtype)).float()

    gate_activated = gate_activated.squeeze(0)
    gated_product = gated_product.squeeze(0)
    mlp_output = mlp_output.squeeze(0)
    up_out = up_out.squeeze(0)
    x_flat = x.squeeze(0)

    return {
        "x_norm": float(x_flat.norm()),
        "gate_norm": float(gate_out.squeeze(0).norm()),
        "up_norm": float(up_out.norm()),
        "gate_activated_norm": float(gate_activated.norm()),
        "gated_product_norm": float(gated_product.norm()),
        "mlp_output_norm": float(mlp_output.norm()),
        "gate_active_frac": float((gate_activated.abs() > 0.1).float().mean()),
        "gating_cos": float(F.cosine_similarity(up_out.unsqueeze(0), gated_product.unsqueeze(0)).item()),
        "mlp_output_vec": mlp_output.detach().cpu().numpy(),
    }


# ══════════════════════════════════════════════════════════════════════════
# PARTS 1 + 2: SINGLE-PASS COLLECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PARTS 1+2: COLLECTING ALL DATA (single pass per problem)")
print("="*70)

all_problems = (
    [{"prompt": p["prompt"], "cat": "zh_math"} for p in zh_math_test] +
    [{"prompt": p["prompt"], "cat": "en_math"} for p in en_math_test] +
    [{"prompt": p["prompt"], "cat": "factual"} for p in FACTUAL_PROBLEMS] +
    [{"prompt": p["prompt"], "cat": "logic"} for p in LOGIC_PROBLEMS]
)
print(f"Total problems: {len(all_problems)}")

# Layers to analyze
probe_layers = sorted(set([0, 3, 6, 8, 9, 10, 12, 14, 17, 18, 20, 22, 25, 27, 30, 33, 35]))

# Storage
n_prob = len(all_problems)
attn_A = {l: np.zeros((n_prob, d)) for l in probe_layers}  # attention input
attn_C = {l: np.zeros((n_prob, d)) for l in probe_layers}  # attention context
mlp_data = {l: [] for l in probe_layers}  # MLP decomposition per problem

for pi, prob in enumerate(all_problems):
    if pi % 10 == 0:
        print(f"  Problem {pi}/{n_prob}: {prob['prompt'][:40]}...")

    # ONE forward pass captures all layers
    layer_h_pre, layer_mlp_in, seq_len = capture_all_layers(model, tokenizer, prob["prompt"])

    # Extract attention + MLP data for each probe layer
    for l in probe_layers:
        if l in layer_h_pre:
            a_i, c_full = compute_attention_vectors(model, l, layer_h_pre[l], seq_len)
            attn_A[l][pi] = a_i
            attn_C[l][pi] = c_full

        if l in layer_mlp_in:
            decomp = decompose_mlp(model, l, layer_mlp_in[l])
            mlp_data[l].append({"cat": prob["cat"], **{k: v for k, v in decomp.items() if k != "mlp_output_vec"}})

    # Free captured tensors
    del layer_h_pre, layer_mlp_in
    torch.cuda.empty_cache()

print(f"Collection complete: {n_prob} problems × {len(probe_layers)} layers = {n_prob * len(probe_layers)} data points")


# ══════════════════════════════════════════════════════════════════════════
# PART 1: PROCRUSTES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 1: ATTENTION PROCRUSTES")
print("="*70)

task_groups = {
    "zh_math": [i for i, p in enumerate(all_problems) if p["cat"] == "zh_math"],
    "en_math": [i for i, p in enumerate(all_problems) if p["cat"] == "en_math"],
    "factual": [i for i, p in enumerate(all_problems) if p["cat"] == "factual"],
    "logic": [i for i, p in enumerate(all_problems) if p["cat"] == "logic"],
}

procrustes_results = {}
for layer_idx in probe_layers:
    A_full = attn_A[layer_idx]
    C_full = attn_C[layer_idx]

    layer_result = {}
    rotations = {}

    for group_name, indices in task_groups.items():
        A_group = A_full[indices]
        C_group = C_full[indices]

        A_c = A_group - A_group.mean(axis=0, keepdims=True)
        C_c = C_group - C_group.mean(axis=0, keepdims=True)
        A_norm = np.linalg.norm(A_c, 'fro')
        C_norm = np.linalg.norm(C_c, 'fro')
        if A_norm > 1e-8: A_c = A_c / A_norm
        if C_norm > 1e-8: C_c = C_c / C_norm

        R, scale = orthogonal_procrustes(A_c, C_c)
        residual = float(np.linalg.norm(A_c @ R - C_c, 'fro'))
        rotations[group_name] = R
        layer_result[group_name] = {"residual": round(residual, 6), "scale": round(float(scale), 6), "n": len(indices)}

    # Cross-group: apply each R to other groups
    cross_fit = {}
    for src, R_src in rotations.items():
        for tgt, tgt_idx in task_groups.items():
            A_t = A_full[tgt_idx]; C_t = C_full[tgt_idx]
            A_c = A_t - A_t.mean(axis=0, keepdims=True)
            C_c = C_t - C_t.mean(axis=0, keepdims=True)
            an = np.linalg.norm(A_c, 'fro'); cn = np.linalg.norm(C_c, 'fro')
            if an > 1e-8: A_c /= an
            if cn > 1e-8: C_c /= cn
            cross_fit[f"{src}_R_on_{tgt}"] = round(float(np.linalg.norm(A_c @ R_src - C_c, 'fro')), 6)

    # Rotation similarity
    rot_sims = {}
    groups = list(rotations.keys())
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1, g2 = groups[i], groups[j]
            frob = float(np.linalg.norm(rotations[g1] - rotations[g2], 'fro'))
            trace = float(np.trace(rotations[g1].T @ rotations[g2])) / d
            rot_sims[f"{g1}_vs_{g2}"] = {"frobenius_dist": round(frob, 4), "trace_similarity": round(trace, 6)}

    procrustes_results[str(layer_idx)] = {"self_fit": layer_result, "cross_fit": cross_fit, "rotation_similarity": rot_sims}

    # Key print
    zm_en = rot_sims.get("zh_math_vs_en_math", {}).get("trace_similarity", "?")
    zm_fact = rot_sims.get("zh_math_vs_factual", {}).get("trace_similarity", "?")
    print(f"  L{layer_idx:2d}: zh_math vs en_math trace_sim={zm_en}, zh_math vs factual trace_sim={zm_fact}")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: MLP DECOMPOSITION SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 2: MLP SUB-LAYER DECOMPOSITION")
print("="*70)

mlp_decomp_results = {}
for layer_idx in probe_layers:
    group_stats = {}
    for entry in mlp_data[layer_idx]:
        cat = entry["cat"]
        if cat not in group_stats:
            group_stats[cat] = {k: [] for k in ["x_norm", "gate_norm", "up_norm", "gate_activated_norm",
                                                  "gated_product_norm", "mlp_output_norm", "gate_active_frac", "gating_cos"]}
        for k in group_stats[cat]:
            group_stats[cat][k].append(entry[k])

    layer_result = {}
    for group, stats in group_stats.items():
        layer_result[group] = {k: round(float(np.mean(v)), 4) for k, v in stats.items()}

    mlp_decomp_results[str(layer_idx)] = layer_result

    # Print key metric: gate activation fraction differs by task type?
    fracs = {g: layer_result[g]["gate_active_frac"] for g in layer_result}
    print(f"  L{layer_idx:2d}: gate_active — " + ", ".join(f"{g}={v:.3f}" for g, v in fracs.items()))


# ══════════════════════════════════════════════════════════════════════════
# PART 3: RANK-FORCING FOAMS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 3: RANK-FORCING FOAMS INTERVENTION")
print("="*70)

# Collect hidden states for SVD basis (use training problems)
train_indices = []
for cat in cats:
    cat_idxs = [i for i, p in enumerate(all_math) if p["cat"] == cat]
    train_indices.extend(cat_idxs[4:12])

FOAMS_LAYER = 10  # mid-adversarial zone

print(f"Collecting SVD bases at L{FOAMS_LAYER}...")


def collect_hidden_at_layer(model, tokenizer, prompts, target_layer):
    states = []
    captured = {}
    def hook_fn(module, input, output):
        h = output if isinstance(output, torch.Tensor) else output[0]
        captured["h"] = h.detach()
    handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
    for prompt in prompts:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)
        states.append(captured["h"][0, -1, :].float().cpu().numpy())
    handle.remove()
    return np.stack(states)


zh_train = collect_hidden_at_layer(model, tokenizer, [all_math[i]["zh"] for i in train_indices], FOAMS_LAYER)
en_train = collect_hidden_at_layer(model, tokenizer, [all_math[i]["en"] for i in train_indices], FOAMS_LAYER)
fact_train = collect_hidden_at_layer(model, tokenizer, [p["prompt"] for p in FACTUAL_PROBLEMS], FOAMS_LAYER)

def svd_basis(states, k=10):
    U, S, Vh = np.linalg.svd(states - states.mean(axis=0, keepdims=True), full_matrices=False)
    return Vh[:k], S[:k]

zh_basis, zh_sv = svd_basis(zh_train)
en_basis, en_sv = svd_basis(en_train)
fact_basis, fact_sv = svd_basis(fact_train)

def pr(sv):
    sv2 = sv**2; sv2 = sv2 / sv2.sum()
    return float(1.0 / (sv2**2).sum())

print(f"  PR: zh_math={pr(zh_sv):.1f}, en_math={pr(en_sv):.1f}, factual={pr(fact_sv):.1f}")

# Intervention
def eval_rank_forcing(model, tokenizer, problems, basis, rank_k, check_fn):
    basis_t = torch.tensor(basis[:rank_k], dtype=torch.float32, device=device)
    P = basis_t.T @ basis_t

    def hook(module, input, output):
        h = output if isinstance(output, torch.Tensor) else output[0]
        h_proj = h.float() @ P.to(h.device)
        if isinstance(output, tuple):
            return (h_proj.to(output[0].dtype),) + output[1:]
        return h_proj.to(output.dtype)

    handle = model.model.layers[FOAMS_LAYER].register_forward_hook(hook)
    correct = 0
    for prob in problems:
        inputs = tokenizer(prob["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if check_fn(gen, prob["answer"]):
            correct += 1
    handle.remove()
    return correct, len(problems)

check_math = lambda g, a: a in g
check_fact = lambda g, a: a.lower() in g.lower()

# Baselines
print("\n  BASELINES:")
bl = {}
for name, probs, check_fn in [("zh_math", zh_math_test, check_math), ("en_math", en_math_test, check_math), ("factual", FACTUAL_PROBLEMS, check_fact)]:
    c = 0
    for p in probs:
        inp = tokenizer(p["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None)
        gen = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if check_fn(gen, p["answer"]): c += 1
    bl[name] = f"{c}/{len(probs)}"
    print(f"    {name}: {c}/{len(probs)}")

# Interventions
print(f"\n  INTERVENTIONS at L{FOAMS_LAYER}:")
foams_results = {"baselines": bl}
interventions = [
    ("zh_math_own_rank1", zh_math_test, zh_basis, 1, check_math),
    ("zh_math_own_rank3", zh_math_test, zh_basis, 3, check_math),
    ("zh_math_fact_rank8", zh_math_test, fact_basis, 8, check_math),
    ("zh_math_fact_rank1", zh_math_test, fact_basis, 1, check_math),
    ("factual_own_rank8", FACTUAL_PROBLEMS, fact_basis, 8, check_fact),
    ("factual_own_rank3", FACTUAL_PROBLEMS, fact_basis, 3, check_fact),
    ("factual_zh_rank1", FACTUAL_PROBLEMS, zh_basis, 1, check_fact),
    ("en_math_zh_rank1", en_math_test, zh_basis, 1, check_math),
    ("en_math_own_rank3", en_math_test, en_basis, 3, check_math),
]

for label, probs, basis, rank_k, check_fn in interventions:
    c, n = eval_rank_forcing(model, tokenizer, probs, basis, rank_k, check_fn)
    foams_results[label] = f"{c}/{n}"
    print(f"    {label}: {c}/{n}")


# ══════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SAVING")
print("="*70)

output = {
    "experiment": "AK: Triple Probe (Optimized)",
    "model": MODEL_NAME,
    "dims": {"d": d, "d_ff": d_ff, "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv_heads, "head_dim": head_dim},
    "part1_procrustes": procrustes_results,
    "part2_mlp_decomposition": mlp_decomp_results,
    "part3_foams": {"layer": FOAMS_LAYER, "svd_pr": {"zh": pr(zh_sv), "en": pr(en_sv), "fact": pr(fact_sv)}, "results": foams_results},
}

with open(OUTPUT_DIR / "expAK_triple_probe.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"Saved to output/expAK_triple_probe.json")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== KEY FINDINGS ===")
print("\nPart 1 — Procrustes (is attention doing different rotations per task?):")
for l in [9, 17, 18, 30]:
    if str(l) in procrustes_results:
        r = procrustes_results[str(l)]["rotation_similarity"]
        print(f"  L{l}: " + ", ".join(f"{k}={v['trace_similarity']:.4f}" for k, v in r.items()))

print("\nPart 2 — MLP (gate activation selectivity per task):")
for l in [9, 17, 18, 30]:
    if str(l) in mlp_decomp_results:
        r = mlp_decomp_results[str(l)]
        print(f"  L{l}: " + ", ".join(f"{g}={r[g]['gate_active_frac']:.3f}" for g in r))

print("\nPart 3 — FOAMS (rank-forcing):")
for k, v in foams_results.items():
    print(f"  {k}: {v}")

print("\nDone.")
