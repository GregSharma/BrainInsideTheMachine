"""Exp BP: Sublayer null-space rotation decomposition.

BN showed 77°/layer rotation of the null-space. But each layer has TWO sublayers:
  h → h + Attn(LN(h))  →  h' + MLP(LN(h'))  → h_next
       [attention]              [MLP]

KG evidence predicts: attention is language-blind (82D, low-rank), MLP is the
language engine. So MLP should do ALL the rotating and attention should preserve
the null-space.

Measurement:
  For each layer L, capture three points:
    h_pre:  residual stream BEFORE layer L (= output of layer L-1)
    h_mid:  residual stream AFTER attention, BEFORE MLP
    h_post: residual stream AFTER MLP (= output of layer L)

  Build null-space at each point. Measure rotation:
    attn_rotation: null-space(h_pre) vs null-space(h_mid)
    mlp_rotation:  null-space(h_mid) vs null-space(h_post)

  Prediction: attn_rotation ≈ 0°, mlp_rotation ≈ 77°.

Also re-measures BL's ρ in the CORRECT basis: each MLP's output measured
in its OWN input null-space, not L32's fixed projector.
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUT = Path("output")
N_LAYERS = 36
DIM = 2048
N_NULL = 20
LANGUAGES = ['en', 'zh', 'es', 'ar', 'ja', 'ko', 'sw']

t0 = time.time()

print("=" * 60)
print("  Exp BP: Sublayer Null-Space Rotation Decomposition")
print("=" * 60)

# ── 1. Load model and tokenizer ─────────────────────────────────

print("\n[1/5] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()

# ── 2. Load problems (same as multilingual cache) ───────────────

print("\n[2/5] Loading problems and building prompts...")

# We need the same 200 problems used in the multilingual cache
# Load from the existing multilingual data to get problem count
multi_check = np.load(OUT / "multilingual_all_layers.npz")
N_PROBLEMS = multi_check["en_L0"].shape[0]
print(f"  {N_PROBLEMS} problems, {len(LANGUAGES)} languages")

# Load the problem prompts — reconstruct from the generation script
import random as pyrandom, math
SEED = 42

TEMPLATES = {
    'zh': {'arithmetic_plus': "计算 {a} + {b} 的值。", 'arithmetic_times': "计算 {a} × {b} 的值。",
            'combinatorics': "求组合数 C({n}, {k}) 的值。", 'modular': "{a} 除以 {b} 的余数是多少？",
            'geometry': "一个长方形的长为 {w}，宽为 {h}，求其面积。"},
    'en': {'arithmetic_plus': "Calculate {a} + {b}.", 'arithmetic_times': "Calculate {a} × {b}.",
            'combinatorics': "Find the value of C({n}, {k}).", 'modular': "What is the remainder when {a} is divided by {b}?",
            'geometry': "A rectangle has length {w} and width {h}. Find its area."},
    'es': {'arithmetic_plus': "Calcula {a} + {b}.", 'arithmetic_times': "Calcula {a} × {b}.",
            'combinatorics': "Halla el valor de C({n}, {k}).", 'modular': "¿Cuál es el residuo de {a} entre {b}?",
            'geometry': "Un rectángulo tiene largo {w} y ancho {h}. Halla su área."},
    'ar': {'arithmetic_plus': "احسب {a} + {b}.", 'arithmetic_times': "احسب {a} × {b}.",
            'combinatorics': "أوجد قيمة C({n}, {k}).", 'modular': "ما باقي قسمة {a} على {b}؟",
            'geometry': "مستطيل طوله {w} وعرضه {h}، أوجد مساحته."},
    'ja': {'arithmetic_plus': "{a} + {b} を計算してください。", 'arithmetic_times': "{a} × {b} を計算してください。",
            'combinatorics': "C({n}, {k}) の値を求めてください。", 'modular': "{a} を {b} で割った余りは？",
            'geometry': "長さ {w}、幅 {h} の長方形の面積を求めてください。"},
    'ko': {'arithmetic_plus': "{a} + {b}를 계산하세요.", 'arithmetic_times': "{a} × {b}를 계산하세요.",
            'combinatorics': "C({n}, {k})의 값을 구하세요.", 'modular': "{a}을 {b}로 나눈 나머지는?",
            'geometry': "가로 {w}, 세로 {h}인 직사각형의 넓이를 구하세요."},
    'sw': {'arithmetic_plus': "Kokotoa {a} + {b}.", 'arithmetic_times': "Kokotoa {a} × {b}.",
            'combinatorics': "Tafuta thamani ya C({n}, {k}).", 'modular': "Baki ya {a} gawanywa na {b} ni nini?",
            'geometry': "Mstatili una urefu {w} na upana {h}. Tafuta eneo lake."},
}

def generate_problems():
    rng = pyrandom.Random(SEED)
    problems = []
    per_cat = N_PROBLEMS // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        ans = a + b if op == "plus" else a * b
        prompts = {lang: TEMPLATES[lang][f'arithmetic_{op}'].format(a=a, b=b) for lang in LANGUAGES}
        problems.append(prompts)
    for _ in range(per_cat):
        n, k = rng.randint(5, 20), 0
        k = rng.randint(1, min(n-1, 8))
        prompts = {lang: TEMPLATES[lang]['combinatorics'].format(n=n, k=k) for lang in LANGUAGES}
        problems.append(prompts)
    for _ in range(per_cat):
        a, b = rng.randint(50, 9999), rng.randint(3, 37)
        prompts = {lang: TEMPLATES[lang]['modular'].format(a=a, b=b) for lang in LANGUAGES}
        problems.append(prompts)
    for _ in range(per_cat):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        prompts = {lang: TEMPLATES[lang]['geometry'].format(w=w, h=h) for lang in LANGUAGES}
        problems.append(prompts)
    for _ in range(per_cat):
        a1 = rng.randint(1, 20); d = rng.randint(1, 10); n_terms = rng.randint(5, 30)
        prompts = {
            lang: (f"An arithmetic sequence: first term {a1}, common difference {d}. Sum of first {n_terms} terms?"
                   if lang == 'en' else
                   f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。" if lang == 'zh' else
                   f"Sucesión aritmética: primer término {a1}, diferencia {d}. Suma de {n_terms} términos?" if lang == 'es' else
                   f"متتالية حسابية: الحد الأول {a1}، الفرق {d}. مجموع أول {n_terms} حدود؟" if lang == 'ar' else
                   f"等差数列：初項 {a1}、公差 {d}。初項から第 {n_terms} 項までの和は？" if lang == 'ja' else
                   f"등차수열: 첫째항 {a1}, 공차 {d}. 처음 {n_terms}항의 합?" if lang == 'ko' else
                   f"Mfuatano wa hesabu: neno la kwanza {a1}, tofauti {d}. Jumla ya maneno {n_terms} ya kwanza?")
            for lang in LANGUAGES
        }
        problems.append(prompts)
    # Shuffle with same seed
    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(problems)))
    rng2.shuffle(indices)
    return [problems[i] for i in indices]

problems = generate_problems()
print(f"  Generated {len(problems)} problems")
del multi_check


# ── 3. Capture h_pre, h_mid, h_post for all inputs ──────────────

print("\n[3/5] Capturing sublayer activations (h_pre, h_mid, h_post)...")

# Storage: h_mid[lang][layer] = (N_PROBLEMS, DIM) numpy array
h_mid_all = {lang: {} for lang in LANGUAGES}
h_pre_all = {lang: {} for lang in LANGUAGES}  # = h_post of previous layer
h_post_all = {lang: {} for lang in LANGUAGES}

# We capture h_mid by hooking post_attention_layernorm (its input = h_mid)
# We capture h_post by hooking the layer output

for lang in tqdm(LANGUAGES, desc="  Languages"):
    # Collect activations for all problems in this language
    mid_by_layer = {L: [] for L in range(N_LAYERS)}
    post_by_layer = {L: [] for L in range(N_LAYERS)}

    mid_hooks = []
    post_hooks = []

    # Register hooks
    for L in range(N_LAYERS):
        def make_mid_hook(layer_idx):
            def hook_fn(module, input, output):
                # post_attention_layernorm receives h_mid as input
                h = input[0] if isinstance(input, tuple) else input
                mid_by_layer[layer_idx].append(h[:, -1, :].detach().cpu().float().numpy())
            return hook_fn

        def make_post_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                post_by_layer[layer_idx].append(h[:, -1, :].detach().cpu().float().numpy())
            return hook_fn

        mid_hooks.append(
            model.model.layers[L].post_attention_layernorm.register_forward_hook(make_mid_hook(L))
        )
        post_hooks.append(
            model.model.layers[L].register_forward_hook(make_post_hook(L))
        )

    # Run all problems
    for prob in problems:
        inputs = tokenizer(prob[lang], return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inputs)

    # Remove hooks
    for h in mid_hooks + post_hooks:
        h.remove()

    # Stack into arrays
    for L in range(N_LAYERS):
        h_mid_all[lang][L] = np.vstack(mid_by_layer[L])    # (N_PROBLEMS, DIM)
        h_post_all[lang][L] = np.vstack(post_by_layer[L])   # (N_PROBLEMS, DIM)
        # h_pre for layer L = h_post of layer L-1 (or embedding for L=0)
        if L > 0:
            h_pre_all[lang][L] = h_post_all[lang][L-1]

    del mid_by_layer, post_by_layer

print(f"  Captured. Shape check: h_mid['en'][0] = {h_mid_all['en'][0].shape}")

# Free GPU memory
del model
torch.cuda.empty_cache()


# ── 4. Build null-spaces and measure rotation ────────────────────

print("\n[4/5] Building null-spaces at each sublayer point and measuring rotation...")

def build_nullspace(H_dict, layer_or_point):
    """Build null-space from cross-language differences."""
    diffs = []
    for i, la in enumerate(LANGUAGES):
        for j, lb in enumerate(LANGUAGES):
            if i >= j:
                continue
            diffs.append(H_dict[la][layer_or_point] - H_dict[lb][layer_or_point])
    diffs = np.vstack(diffs).astype(np.float32)
    gram = diffs.T @ diffs
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    return eigenvectors[:, :N_NULL].T  # (N_NULL, DIM), ascending eigenval


def subspace_alignment(B1, B2):
    """Mean cos(principal angles). 1.0=identical, 0.0=orthogonal."""
    M = B1 @ B2.T
    sv = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(np.minimum(sv, 1.0)))


def procrustes_angle(B1, B2):
    """Mean principal angle in degrees between two subspaces."""
    M = B2 @ B1.T
    sv = np.linalg.svd(M, compute_uv=False)
    angles_deg = np.degrees(np.arccos(np.clip(sv, -1, 1)))
    return float(np.mean(angles_deg))


# Build null-spaces at h_mid and h_post for each layer
null_mid = {}   # L -> (N_NULL, DIM)
null_post = {}  # L -> (N_NULL, DIM)

for L in tqdm(range(N_LAYERS), desc="  Null-space (mid)"):
    null_mid[L] = build_nullspace(h_mid_all, L)

for L in tqdm(range(N_LAYERS), desc="  Null-space (post)"):
    null_post[L] = build_nullspace(h_post_all, L)

# Measure rotations
attn_rotations = {}   # rotation from h_pre to h_mid (attention's effect)
mlp_rotations = {}    # rotation from h_mid to h_post (MLP's effect)
full_rotations = {}   # rotation from h_pre to h_post (full layer = BN replication)

for L in range(1, N_LAYERS):
    # h_pre null-space = h_post of layer L-1
    pre_null = null_post[L-1]
    mid_null = null_mid[L]
    post_null = null_post[L]

    attn_rotations[L] = procrustes_angle(pre_null, mid_null)
    mlp_rotations[L] = procrustes_angle(mid_null, post_null)
    full_rotations[L] = procrustes_angle(pre_null, post_null)


# ── 4b. Re-measure BL's ρ in correct basis ──────────────────────

print("\n  Re-measuring MLP ρ in own-layer null-space...")

rho_correct = {}
for L in tqdm(range(N_LAYERS), desc="  ρ (own basis)"):
    # MLP delta = h_post - h_mid
    H_mid = np.vstack([h_mid_all[la][L] for la in LANGUAGES])
    H_post = np.vstack([h_post_all[la][L] for la in LANGUAGES])
    mlp_delta = H_post - H_mid  # (1400, DIM)

    # Project MLP delta into h_mid's null-space (the INPUT null-space for this MLP)
    Pi_mid = null_mid[L]  # (N_NULL, DIM)
    # Projector: V V^T
    delta_proj = mlp_delta @ Pi_mid.T @ Pi_mid  # project into null-space

    total_e = np.mean(np.sum(mlp_delta ** 2, axis=1))
    null_e = np.mean(np.sum(delta_proj ** 2, axis=1))
    rho_correct[L] = float(null_e / (total_e + 1e-10))


# ── 5. Results ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SUBLAYER ROTATION DECOMPOSITION")
print("=" * 70)

print(f"\n  {'Layer':<6s} {'Attn°':>8s} {'MLP°':>8s} {'Full°':>8s} {'Attn%':>8s} {'ρ_own':>8s}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for L in range(1, N_LAYERS):
    a = attn_rotations[L]
    m = mlp_rotations[L]
    f = full_rotations[L]
    attn_pct = a / (a + m + 1e-10) * 100
    rho = rho_correct[L]
    print(f"  L{L:<4d} {a:>8.2f} {m:>8.2f} {f:>8.2f} {attn_pct:>7.1f}% {rho:>8.4f}")

# Phase averages
phases = {
    "early (L1-L8)":   range(1, 9),
    "advers (L9-L17)":  range(9, 18),
    "coop (L18-L26)":   range(18, 27),
    "late (L27-L35)":   range(27, 36),
}

print(f"\n  Phase averages:")
print(f"  {'Phase':<20s} {'Attn°':>8s} {'MLP°':>8s} {'Full°':>8s} {'Attn%':>8s} {'ρ_own':>8s}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for name, rng in phases.items():
    avg_a = np.mean([attn_rotations[L] for L in rng])
    avg_m = np.mean([mlp_rotations[L] for L in rng])
    avg_f = np.mean([full_rotations[L] for L in rng])
    avg_pct = avg_a / (avg_a + avg_m + 1e-10) * 100
    avg_rho = np.mean([rho_correct[L] for L in rng])
    print(f"  {name:<20s} {avg_a:>8.2f} {avg_m:>8.2f} {avg_f:>8.2f} {avg_pct:>7.1f}% {avg_rho:>8.4f}")

# Verdicts
avg_attn = np.mean([attn_rotations[L] for L in range(1, N_LAYERS)])
avg_mlp = np.mean([mlp_rotations[L] for L in range(1, N_LAYERS)])
avg_rho_own = np.mean([rho_correct[L] for L in range(N_LAYERS)])
avg_rho_bl = 0.005  # BL's average with L32 projector

print(f"\n  Overall: Attn avg={avg_attn:.2f}°  MLP avg={avg_mlp:.2f}°")
print(f"  Attn fraction of total rotation: {avg_attn/(avg_attn+avg_mlp)*100:.1f}%")
print(f"  ρ (own null-space): {avg_rho_own:.4f}  vs  ρ (L32 fixed, BL): ~{avg_rho_bl:.4f}")

if avg_attn < 30 and avg_mlp > 50:
    verdict = "CONFIRMED: Attention preserves null-space, MLP rotates it"
elif avg_attn > avg_mlp:
    verdict = "SURPRISE: Attention rotates MORE than MLP"
else:
    verdict = "MIXED: Both sublayers contribute to rotation"

print(f"\n  VERDICT: {verdict}")

if avg_rho_own > 0.05:
    verdict_rho = f"BL REHABILITATED: MLP ρ={avg_rho_own:.3f} in own basis (was 0.005 in L32 basis)"
else:
    verdict_rho = f"BL CONFIRMED: MLP ρ={avg_rho_own:.4f} even in own basis"

print(f"  VERDICT (ρ): {verdict_rho}")

elapsed = time.time() - t0
print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")


# ── Save ─────────────────────────────────────────────────────────

output = {
    "experiment": "BP",
    "title": "Sublayer null-space rotation decomposition",
    "n_null": N_NULL,
    "runtime_seconds": round(elapsed),
    "attn_rotations": {str(k): v for k, v in attn_rotations.items()},
    "mlp_rotations": {str(k): v for k, v in mlp_rotations.items()},
    "full_rotations": {str(k): v for k, v in full_rotations.items()},
    "rho_own_basis": {str(k): v for k, v in rho_correct.items()},
    "phase_averages": {name: {
        "attn": float(np.mean([attn_rotations[L] for L in rng])),
        "mlp": float(np.mean([mlp_rotations[L] for L in rng])),
        "full": float(np.mean([full_rotations[L] for L in rng])),
        "rho_own": float(np.mean([rho_correct[L] for L in rng])),
    } for name, rng in phases.items()},
    "avg_attn_rotation": float(avg_attn),
    "avg_mlp_rotation": float(avg_mlp),
    "avg_rho_own_basis": float(avg_rho_own),
    "verdict": verdict,
    "verdict_rho": verdict_rho,
}

with open(OUT / "expBP_sublayer_rotation.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBP_sublayer_rotation.json")
