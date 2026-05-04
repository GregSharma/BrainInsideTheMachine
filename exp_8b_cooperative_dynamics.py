#!/usr/bin/env python3
"""8B Cooperative Dynamics — Prefill/Gen comparison + Cooperation-Crystallization

Designed for Colab GPU (A100/H100). Aggressive timeouts, progress probing, fast-fail.

Two experiments in one script:
  1. Prefill vs Generation adversarial comparison (does cooperative collapse exist at 8B?)
  2. Cooperation-crystallization by difficulty (does difficulty invert the pattern at 8B?)

Usage:
  python3 exp_8b_cooperative_dynamics.py [--model Qwen/Qwen3-8B] [--max-tokens 128] [--quantize none]

Outputs progress to stdout every problem. Saves results incrementally.
"""

import argparse
import json
import numpy as np
import torch
import random as pyrandom
import time
import sys
import os
from math import comb

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--quantize", default="none", choices=["none", "4bit", "8bit"])
parser.add_argument("--chat-template", action="store_true", default=True)
parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
parser.add_argument("--output-dir", default="output")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# =============================================================================
# GPU check — fast fail
# =============================================================================
print("=" * 60)
print(f"GPU CHECK")
print("=" * 60)
if not torch.cuda.is_available():
    print("FATAL: No CUDA GPU available. Exiting.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"  GPU: {gpu_name}")
print(f"  VRAM: {gpu_mem_gb:.1f} GB")

if args.quantize == "none" and gpu_mem_gb < 20:
    print(f"  WARNING: {gpu_mem_gb:.1f} GB may not fit {args.model} unquantized (~16GB).")
    print(f"  Consider --quantize 4bit or 8bit.")

# =============================================================================
# Load model with timeout awareness
# =============================================================================
print(f"\n{'='*60}")
print(f"LOADING MODEL: {args.model} (quantize={args.quantize})")
print(f"{'='*60}")
t_load = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

load_kwargs = {
    "trust_remote_code": True,
    "attn_implementation": "sdpa",
}

if args.quantize == "4bit":
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
elif args.quantize == "8bit":
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
else:
    load_kwargs["torch_dtype"] = torch.bfloat16
    load_kwargs["device_map"] = "cuda"

model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = next(model.parameters()).device
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size

print(f"  Loaded in {time.time()-t_load:.1f}s")
print(f"  Layers: {n_layers}, d_model: {d_model}, device: {device}")
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB / {gpu_mem_gb:.1f} GB")

ALL_LAYERS = list(range(n_layers))
COOP_LAYERS = list(range(n_layers*47//100, n_layers*61//100 + 1))  # ~47%-61% depth = cooperative zone analog
# For 36-layer model: L17-L22. For 32-layer: L15-L19.
print(f"  Cooperative zone layers: L{COOP_LAYERS[0]}-L{COOP_LAYERS[-1]}")

t0 = time.time()


# =============================================================================
# Chat template helper
# =============================================================================
def format_prompt(text):
    if args.chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


# =============================================================================
# Problem set (same as v2 — easy/medium/hard × 3 langs)
# =============================================================================
def build_problems():
    problems = []
    rng = pyrandom.Random(42)

    # Easy (10)
    for _ in range(2):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            problems.append({"en": f"Calculate {a} + {b}.", "zh": f"计算 {a} + {b} 的值。",
                              "es": f"Calcula {a} + {b}.", "answer": str(a+b), "difficulty": "easy"})
        else:
            problems.append({"en": f"Calculate {a} × {b}.", "zh": f"计算 {a} × {b} 的值。",
                              "es": f"Calcula {a} × {b}.", "answer": str(a*b), "difficulty": "easy"})
    for _ in range(2):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"en": f"Find the value of C({n_val}, {k_val}).",
                          "zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                          "es": f"Encuentra el valor de C({n_val}, {k_val}).",
                          "answer": str(comb(n_val, k_val)), "difficulty": "easy"})
    for _ in range(2):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        problems.append({"en": f"What is the remainder when {a} is divided by {b}?",
                          "zh": f"{a} 除以 {b} 的余数是多少？",
                          "es": f"¿Cuál es el resto cuando {a} se divide por {b}?",
                          "answer": str(a % b), "difficulty": "easy"})
    for _ in range(2):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        problems.append({"en": f"A rectangle has length {w} and width {h}. Find its area.",
                          "zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                          "es": f"Un rectángulo tiene largo {w} y ancho {h}. Encuentra su área.",
                          "answer": str(w*h), "difficulty": "easy"})
    for _ in range(2):
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2*a1 + (n_terms-1)*d_val) // 2
        problems.append({"en": f"Arithmetic sequence: first term {a1}, common difference {d_val}. Sum of first {n_terms} terms?",
                          "zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。",
                          "es": f"Sucesión aritmética: primer término {a1}, diferencia {d_val}. Suma de los primeros {n_terms} términos?",
                          "answer": str(ans), "difficulty": "easy"})

    # Medium (10)
    rng2 = pyrandom.Random(99)
    for _ in range(3):
        a, b = rng2.randint(100, 9999), rng2.randint(100, 9999)
        problems.append({"en": f"Calculate {a} × {b}.", "zh": f"计算 {a} × {b} 的值。",
                          "es": f"Calcula {a} × {b}.", "answer": str(a*b), "difficulty": "medium"})
    for _ in range(3):
        n_val = rng2.randint(12, 20)
        k_val = rng2.randint(3, 6)
        problems.append({"en": f"Find the value of C({n_val}, {k_val}).",
                          "zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                          "es": f"Encuentra el valor de C({n_val}, {k_val}).",
                          "answer": str(comb(n_val, k_val)), "difficulty": "medium"})
    for _ in range(4):
        a = rng2.randint(1000, 99999)
        b = rng2.randint(7, 97)
        problems.append({"en": f"What is the remainder when {a} is divided by {b}?",
                          "zh": f"{a} 除以 {b} 的余数是多少？",
                          "es": f"¿Cuál es el resto cuando {a} se divide por {b}?",
                          "answer": str(a % b), "difficulty": "medium"})

    # Hard (AMC) — EN only for now, add translations inline
    amc_problems = [
        {"en": "Andy and Betsy both live in Mathville. Andy leaves Mathville on his bicycle at 1:30, traveling due north at a steady 8 miles per hour. Betsy leaves on her bicycle from the same point at 2:30, traveling due east at a steady 12 miles per hour. At what time will they be exactly the same distance from their common starting point?",
         "zh": "安迪和贝琪都住在数学城。安迪下午1:30骑自行车离开数学城，以每小时8英里的速度向正北方向匀速前进。贝琪从同一个出发点于下午2:30骑自行车出发，以每小时12英里的速度向正东方向匀速前进。他们什么时候离出发点的距离完全相同？",
         "es": "Andy sale en bicicleta a la 1:30 hacia el norte a 8 mph. Betsy sale del mismo punto a las 2:30 hacia el este a 12 mph. ¿A qué hora estarán a la misma distancia del punto de partida?",
         "answer": "4:30"},
        {"en": "A box contains 10 pounds of a nut mix that is 50% peanuts, 20% cashews, 30% almonds. A second mix (20% peanuts, 40% cashews, 40% almonds) is added so peanuts become 40%. How many pounds of cashews are now in the box?",
         "zh": "一个箱子里有10磅坚果混合物(50%花生,20%腰果,30%杏仁)。加入第二种混合物(20%花生,40%腰果,40%杏仁)使花生含量为40%。现在有多少磅腰果？",
         "es": "Una caja tiene 10 libras de mezcla (50% maní, 20% anacardos, 30% almendras). Se agrega otra mezcla (20% maní, 40% anacardos, 40% almendras) para obtener 40% maní. ¿Cuántas libras de anacardos hay?",
         "answer": "4"},
        {"en": "Students vs teachers trivia. Total 15 people. Ash joining students raises avg age from 12 to 14. Ash joining teachers drops avg age from 55 to 52. How old is Ash?",
         "zh": "学生队和教师队共15人。阿什加入学生队，平均年龄从12升到14。加入教师队，平均年龄从55降到52。阿什几岁？",
         "es": "Estudiantes vs maestros, total 15. Si Ash se une a estudiantes, promedio sube de 12 a 14. Si se une a maestros, baja de 55 a 52. ¿Cuántos años tiene Ash?",
         "answer": "28"},
        {"en": "Agnes writes 4 statements: (1) At least one is true. (2) At least two are true. (3) At least two are false. (4) At least one is false. How many are false?",
         "zh": "四句话：(1)至少一句是真的 (2)至少两句是真的 (3)至少两句是假的 (4)至少一句是假的。几句是假的？",
         "es": "Cuatro afirmaciones: (1) Al menos una verdadera. (2) Al menos dos verdaderas. (3) Al menos dos falsas. (4) Al menos una falsa. ¿Cuántas son falsas?",
         "answer": "1"},
        {"en": "Alien world: v = k * n^a * m^b cm/hr. With 5 toes: log(v) = 4 + 2*log(m). With 25 eyes: log(v) = 4 + 4*log(n). What is k + a + b?",
         "zh": "外星世界：v = k * n^a * m^b厘米/小时。5个脚趾时log(v) = 4 + 2*log(m)；25只眼睛时log(v) = 4 + 4*log(n)。求k+a+b。",
         "es": "Mundo alienígena: v = k*n^a*m^b cm/h. Con 5 dedos: log(v)=4+2*log(m). Con 25 ojos: log(v)=4+4*log(n). ¿k+a+b?",
         "answer": "22"},
    ]
    for p in amc_problems:
        p["difficulty"] = "hard"
        problems.append(p)

    return problems


problems = build_problems()
print(f"\nProblem set: {len(problems)} ({sum(1 for p in problems if p['difficulty']=='easy')} easy, "
      f"{sum(1 for p in problems if p['difficulty']=='medium')} medium, "
      f"{sum(1 for p in problems if p['difficulty']=='hard')} hard)")


# =============================================================================
# MLP delta capture (supports both prefill and generation)
# =============================================================================
class MLPDeltaCapture:
    def __init__(self, model, layers, max_steps=200):
        self.model = model
        self.layers = layers
        self.max_steps = max_steps
        self.d = model.config.hidden_size
        self.device = next(model.parameters()).device
        self._hooks = []
        self._buffers = {}
        self._counters = {}

    def _reset(self):
        self._buffers = {}
        self._counters = {}
        for li in self.layers:
            self._buffers[li] = torch.empty((self.max_steps, self.d), device=self.device, dtype=torch.float32)
            self._counters[li] = 0

    def _find_mlp(self, layer):
        if hasattr(layer, "mlp"):
            return layer.mlp
        for name, mod in layer.named_children():
            if "mlp" in name.lower() or "feed_forward" in name.lower():
                return mod
        return None

    def _register(self):
        self._hooks = []
        for li in self.layers:
            layer = self.model.model.layers[li]
            mlp = self._find_mlp(layer)
            if mlp is None:
                print(f"  WARNING: No MLP found for layer {li}")
                continue
            def make_hook(idx):
                def hook(module, inp, out):
                    i = self._counters[idx]
                    if i < self.max_steps:
                        self._buffers[idx][i].copy_(out[0, -1, :].float())
                        self._counters[idx] = i + 1
                return hook
            self._hooks.append(mlp.register_forward_hook(make_hook(li)))

    def _unregister(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run_prefill(self, input_ids):
        self._reset()
        self._register()
        with torch.no_grad():
            self.model(input_ids=input_ids)
        self._unregister()
        result = {}
        for li in self.layers:
            n = self._counters[li]
            if n > 0:
                result[li] = self._buffers[li][0].cpu().numpy()
        return result

    def run_generation(self, input_ids, max_new_tokens=128, tokenizer=None):
        self._reset()
        self._register()
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
        # Qwen3 thinking: disable if possible
        if hasattr(self.model.config, "thinking") or "qwen3" in args.model.lower():
            gen_kwargs["thinking"] = False
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, **gen_kwargs)
        self._unregister()
        prompt_len = input_ids.shape[1]
        gen_ids = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True) if tokenizer else ""
        result = {"gen_text": gen_text, "n_gen_tokens": len(gen_ids)}
        for li in self.layers:
            n = self._counters[li]
            if n > 1:
                result[li] = self._buffers[li][1:n].cpu().numpy()
        return result


# =============================================================================
# EXPERIMENT 1: Prefill vs Generation adversarial comparison
# =============================================================================
print(f"\n{'='*60}")
print("EXPERIMENT 1: Prefill vs Generation (adversarial comparison)")
print(f"{'='*60}")

capturer = MLPDeltaCapture(model, ALL_LAYERS, max_steps=args.max_tokens + 50)

# Use 10 easy problems × 2 langs for speed
exp1_problems = [p for p in problems if p["difficulty"] == "easy"][:10]
exp1_prefill = {"en": [], "zh": []}
exp1_gen = {"en": [], "zh": []}

for lang in ["en", "zh"]:
    for pi, prob in enumerate(exp1_problems):
        prompt = format_prompt(prob[lang])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        t_start = time.time()
        pf = capturer.run_prefill(input_ids)
        exp1_prefill[lang].append(pf)

        gen = capturer.run_generation(input_ids, max_new_tokens=args.max_tokens, tokenizer=tokenizer)
        exp1_gen[lang].append(gen)

        elapsed = time.time() - t_start
        tok_per_sec = gen["n_gen_tokens"] / max(elapsed - 0.1, 0.01)  # rough
        print(f"  [{lang}] {pi+1}/{len(exp1_problems)} | {gen['n_gen_tokens']} tok | {elapsed:.1f}s | ~{tok_per_sec:.0f} tok/s | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

        # Fast-fail: if single problem takes > 60s, warn
        if elapsed > 60:
            print(f"  ⚠️  Problem took {elapsed:.0f}s — may timeout on full run")


# Compute cross-layer cosine profiles
def cross_layer_cos_prefill(results_list, layers):
    out = {}
    for i in range(1, len(layers)):
        prev_li, curr_li = layers[i-1], layers[i]
        cos_vals = []
        for pf in results_list:
            if prev_li in pf and curr_li in pf:
                d_prev, d_curr = pf[prev_li], pf[curr_li]
                n_p, n_c = np.linalg.norm(d_prev)+1e-10, np.linalg.norm(d_curr)+1e-10
                cos_vals.append(float(np.dot(d_prev, d_curr)/(n_p*n_c)))
        if cos_vals:
            out[f"L{prev_li}->L{curr_li}"] = {"cos_mean": float(np.mean(cos_vals)), "cos_std": float(np.std(cos_vals))}
    return out

def cross_layer_cos_gen(results_list, layers, tok_range=None):
    out = {}
    for i in range(1, len(layers)):
        prev_li, curr_li = layers[i-1], layers[i]
        cos_vals = []
        for gen in results_list:
            if prev_li not in gen or curr_li not in gen: continue
            d_p, d_c = gen[prev_li], gen[curr_li]
            n_tok = min(d_p.shape[0], d_c.shape[0])
            if n_tok == 0: continue
            if tok_range:
                s, e = min(tok_range[0], n_tok), min(tok_range[1], n_tok)
                if s >= e: continue
                d_p, d_c = d_p[s:e], d_c[s:e]
            np_p = np.linalg.norm(d_p, axis=1, keepdims=True)+1e-10
            np_c = np.linalg.norm(d_c, axis=1, keepdims=True)+1e-10
            cos_vals.append(float(np.mean(np.sum((d_p/np_p)*(d_c/np_c), axis=1))))
        if cos_vals:
            out[f"L{prev_li}->L{curr_li}"] = {"cos_mean": float(np.mean(cos_vals)), "cos_std": float(np.std(cos_vals))}
    return out

def zone_summary(profile, layers):
    vals = [profile[f"L{layers[i-1]}->L{layers[i]}"]["cos_mean"]
            for i in range(1, len(layers)) if f"L{layers[i-1]}->L{layers[i]}" in profile]
    return float(np.mean(vals)) if vals else None

all_pf = exp1_prefill["zh"] + exp1_prefill["en"]
all_gen = exp1_gen["zh"] + exp1_gen["en"]

pf_profile = cross_layer_cos_prefill(all_pf, ALL_LAYERS)
gen_all = cross_layer_cos_gen(all_gen, ALL_LAYERS)
gen_early = cross_layer_cos_gen(all_gen, ALL_LAYERS, tok_range=(0, 10))
gen_late = cross_layer_cos_gen(all_gen, ALL_LAYERS, tok_range=(80, args.max_tokens))

# Determine zones based on layer count
adv_start = n_layers * 25 // 100  # ~25% depth
adv_end = n_layers * 47 // 100    # ~47% depth
coop_start = adv_end
coop_end = n_layers * 61 // 100   # ~61% depth
ramp_end = n_layers * 75 // 100   # ~75% depth

adv_layers = list(range(adv_start, adv_end + 1))
coop_layers = list(range(coop_start, coop_end + 1))
ramp_layers = list(range(coop_end, ramp_end + 1))

print(f"\n--- EXP 1 RESULTS ---")
print(f"  Zones: adv=L{adv_start}-L{adv_end}, coop=L{coop_start}-L{coop_end}, ramp=L{coop_end}-L{ramp_end}")
for zone_name, zlayers in [("adversarial", adv_layers), ("cooperative", coop_layers), ("ramp", ramp_layers)]:
    pf_z = zone_summary(pf_profile, zlayers)
    ga_z = zone_summary(gen_all, zlayers)
    ge_z = zone_summary(gen_early, zlayers)
    gl_z = zone_summary(gen_late, zlayers)
    pf_s = f"{pf_z:+.4f}" if pf_z is not None else "N/A"
    ga_s = f"{ga_z:+.4f}" if ga_z is not None else "N/A"
    ge_s = f"{ge_z:+.4f}" if ge_z is not None else "N/A"
    gl_s = f"{gl_z:+.4f}" if gl_z is not None else "N/A"
    print(f"  {zone_name:<12} Prefill:{pf_s}  Gen(all):{ga_s}  Gen(early):{ge_s}  Gen(late):{gl_s}")


# =============================================================================
# EXPERIMENT 2: Cooperation-Crystallization by difficulty
# =============================================================================
print(f"\n{'='*60}")
print("EXPERIMENT 2: Cooperation-Crystallization by difficulty")
print(f"{'='*60}")

coop_capturer = MLPDeltaCapture(model, COOP_LAYERS, max_steps=args.max_tokens + 50)
exp2_results = []

for lang in ["en", "zh", "es"]:
    print(f"\n  --- {lang.upper()} ---")
    for pi, prob in enumerate(problems):
        prompt = format_prompt(prob[lang])
        answer_str = prob["answer"]
        answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        t_start = time.time()
        result = coop_capturer.run_generation(input_ids, max_new_tokens=args.max_tokens, tokenizer=tokenizer)
        elapsed = time.time() - t_start

        # Per-token cooperation
        n_tok = min(*(result.get(li, np.empty((0, d_model))).shape[0] for li in COOP_LAYERS)) \
                if all(li in result for li in COOP_LAYERS) else 0
        per_tok_coop = []
        for t in range(n_tok):
            cos_vals = []
            for i in range(1, len(COOP_LAYERS)):
                pl, cl = COOP_LAYERS[i-1], COOP_LAYERS[i]
                if pl in result and cl in result:
                    dp, dc = result[pl][t], result[cl][t]
                    cos_vals.append(float(np.dot(dp, dc) / (np.linalg.norm(dp)*np.linalg.norm(dc) + 1e-10)))
            per_tok_coop.append(float(np.mean(cos_vals)) if cos_vals else 0.0)

        n_paired = min(len(per_tok_coop), n_tok)
        corr = None
        coop_first_half = coop_second_half = None
        if n_paired > 5:
            ca = np.array(per_tok_coop[:n_paired])
            half = n_paired // 2
            coop_first_half = float(ca[:half].mean())
            coop_second_half = float(ca[half:].mean())

        got_correct = answer_str in result.get("gen_text", "")
        entry = {"problem_idx": pi, "lang": lang, "difficulty": prob["difficulty"],
                 "answer": answer_str, "got_correct": got_correct,
                 "n_tokens": n_paired, "coop_first_half": coop_first_half,
                 "coop_second_half": coop_second_half, "elapsed": elapsed}
        exp2_results.append(entry)

        status = "✓" if got_correct else "✗"
        c1 = f"{coop_first_half:+.3f}" if coop_first_half is not None else "N/A"
        c2 = f"{coop_second_half:+.3f}" if coop_second_half is not None else "N/A"
        print(f"    {status} [{prob['difficulty'][:3]}] p{pi:02d} | {elapsed:.1f}s | coop1H={c1} coop2H={c2} | VRAM:{torch.cuda.memory_allocated()/1e9:.1f}GB")


# =============================================================================
# Analysis
# =============================================================================
print(f"\n{'='*60}")
print("EXP 2 ANALYSIS")
print(f"{'='*60}")

for diff in ["easy", "medium", "hard"]:
    subset = [r for r in exp2_results if r["difficulty"] == diff]
    correct = sum(1 for r in subset if r["got_correct"])
    c1_vals = [r["coop_first_half"] for r in subset if r["coop_first_half"] is not None]
    c2_vals = [r["coop_second_half"] for r in subset if r["coop_second_half"] is not None]
    c1 = f"{np.mean(c1_vals):.4f}" if c1_vals else "N/A"
    c2 = f"{np.mean(c2_vals):.4f}" if c2_vals else "N/A"
    print(f"  {diff:8s} ({len(subset)} runs, {correct} correct): coop1H={c1}, coop2H={c2}")


# =============================================================================
# Save
# =============================================================================
output = {
    "model": args.model, "n_layers": n_layers, "d_model": d_model,
    "quantize": args.quantize, "chat_template": args.chat_template,
    "gpu": gpu_name, "gpu_mem_gb": gpu_mem_gb,
    "max_new_tokens": args.max_tokens,
    "exp1_prefill_vs_gen": {
        "prefill_profile": pf_profile, "gen_all_profile": gen_all,
        "gen_early_profile": gen_early, "gen_late_profile": gen_late,
        "zones": {"adversarial": adv_layers, "cooperative": coop_layers, "ramp": ramp_layers},
    },
    "exp2_cooperation_crystallization": exp2_results,
    "runtime_seconds": time.time() - t0,
}

outfile = f"{args.output_dir}/exp_8b_cooperative_dynamics.json"
with open(outfile, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {outfile}")
print(f"Total runtime: {time.time()-t0:.0f}s")
