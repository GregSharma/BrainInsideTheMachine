"""Experiment: Cooperation-Crystallization v2 — Broader scope

v1 used 20 easy arithmetic problems. Too narrow.
v2 uses:
  - 10 AMC 2025 12A problems (hard, multi-step)
  - 10 easy arithmetic (first 2 per category)
  - 10 medium arithmetic (problems 20-29, larger numbers)
  = 30 problems × 3 languages (EN, ZH, ES) = 90 runs

Measures per-token cooperation (L17-L22 cross-layer cosine) vs crystallization (p(answer)).
"""

import json
import numpy as np
import torch
import random as pyrandom
from math import comb
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys
sys.path.insert(0, "perturbation/lib")
from problems_amc2025 import AMC_2025_12A

MODEL_NAME = "Qwen/Qwen2.5-3B"
device = "cuda"
MAX_NEW_TOKENS = 128

print(f"Loading {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
COOP_LAYERS = list(range(17, 23))
t0 = time.time()


# =============================================================================
# Build problem set: easy arith + medium arith + AMC (hard)
# =============================================================================
def build_problems():
    """Build 3-tier problem set with answers and 3-language prompts."""
    problems = []
    rng = pyrandom.Random(42)

    # --- EASY arithmetic (10 problems, first 2 per category) ---
    easy_specs = []
    for _ in range(2):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            easy_specs.append((f"Calculate {a} + {b}.", f"计算 {a} + {b} 的值。", f"Calcula {a} + {b}.", str(a + b)))
        else:
            easy_specs.append((f"Calculate {a} × {b}.", f"计算 {a} × {b} 的值。", f"Calcula {a} × {b}.", str(a * b)))
    for _ in range(2):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        easy_specs.append((f"Find the value of C({n_val}, {k_val}).",
                           f"求组合数 C({n_val}, {k_val}) 的值。",
                           f"Encuentra el valor de C({n_val}, {k_val}).",
                           str(comb(n_val, k_val))))
    for _ in range(2):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        easy_specs.append((f"What is the remainder when {a} is divided by {b}?",
                           f"{a} 除以 {b} 的余数是多少？",
                           f"¿Cuál es el resto cuando {a} se divide por {b}?",
                           str(a % b)))
    for _ in range(2):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        easy_specs.append((f"A rectangle has length {w} and width {h}. Find its area.",
                           f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                           f"Un rectángulo tiene largo {w} y ancho {h}. Encuentra su área.",
                           str(w * h)))
    for _ in range(2):
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2 * a1 + (n_terms - 1) * d_val) // 2
        easy_specs.append((
            f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_terms} terms.",
            f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。",
            f"Una sucesión aritmética tiene primer término {a1} y diferencia común {d_val}. Encuentra la suma de los primeros {n_terms} términos.",
            str(ans)))

    for en, zh, es, ans in easy_specs:
        problems.append({"en": en, "zh": zh, "es": es, "answer": ans, "difficulty": "easy"})

    # --- MEDIUM arithmetic (10 problems, harder numbers) ---
    rng2 = pyrandom.Random(99)
    for _ in range(3):
        a, b = rng2.randint(100, 9999), rng2.randint(100, 9999)
        problems.append({
            "en": f"Calculate {a} × {b}.",
            "zh": f"计算 {a} × {b} 的值。",
            "es": f"Calcula {a} × {b}.",
            "answer": str(a * b), "difficulty": "medium"
        })
    for _ in range(3):
        n_val = rng2.randint(12, 20)
        k_val = rng2.randint(3, 6)
        problems.append({
            "en": f"Find the value of C({n_val}, {k_val}).",
            "zh": f"求组合数 C({n_val}, {k_val}) 的值。",
            "es": f"Encuentra el valor de C({n_val}, {k_val}).",
            "answer": str(comb(n_val, k_val)), "difficulty": "medium"
        })
    for _ in range(4):
        a = rng2.randint(1000, 99999)
        b = rng2.randint(7, 97)
        problems.append({
            "en": f"What is the remainder when {a} is divided by {b}?",
            "zh": f"{a} 除以 {b} 的余数是多少？",
            "es": f"¿Cuál es el resto cuando {a} se divide por {b}?",
            "answer": str(a % b), "difficulty": "medium"
        })

    # --- HARD: AMC 2025 12A (10 problems) ---
    # Need to add ZH and ES translations
    amc_translations = {
        1: {"zh": "安迪和贝琪都住在数学城。安迪下午1:30骑自行车离开数学城，以每小时8英里的速度向正北方向匀速前进。贝琪从同一个出发点于下午2:30骑自行车出发，以每小时12英里的速度向正东方向匀速前进。他们什么时候离出发点的距离完全相同？",
             "es": "Andy y Betsy viven en Mathville. Andy sale en bicicleta a la 1:30, viajando hacia el norte a 8 millas por hora. Betsy sale del mismo punto a las 2:30, viajando hacia el este a 12 millas por hora. ¿A qué hora estarán exactamente a la misma distancia del punto de partida?"},
        2: {"zh": "一个箱子里有10磅坚果混合物，其中50%是花生，20%是腰果，30%是杏仁。加入第二种坚果混合物（20%花生，40%腰果，40%杏仁），使新混合物中花生含量为40%。现在箱子里有多少磅腰果？",
             "es": "Una caja contiene 10 libras de mezcla de nueces: 50% maní, 20% anacardos, 30% almendras. Se agrega una segunda mezcla (20% maní, 40% anacardos, 40% almendras) para obtener 40% de maní. ¿Cuántas libras de anacardos hay ahora?"},
        3: {"zh": "一组学生将与一组教师进行知识竞赛。学生和教师的总人数为15人。阿什是一名学生的表亲，想参加比赛。如果阿什加入学生队，该队平均年龄从12岁增加到14岁。如果阿什加入教师队，该队平均年龄从55岁降低到52岁。阿什几岁？",
             "es": "Un equipo de estudiantes competirá contra maestros en trivia. El total es 15 personas. Ash quiere unirse. Si juega con estudiantes, la edad promedio sube de 12 a 14. Si juega con maestros, baja de 55 a 52. ¿Cuántos años tiene Ash?"},
        4: {"zh": "阿格尼斯在一张白纸上写了四句话：(1)这些话中至少有一句是真的。(2)这些话中至少有两句是真的。(3)这些话中至少有两句是假的。(4)这些话中至少有一句是假的。每句话要么真要么假。阿格尼斯写了多少句假话？",
             "es": "Agnes escribe cuatro afirmaciones: (1) Al menos una es verdadera. (2) Al menos dos son verdaderas. (3) Al menos dos son falsas. (4) Al menos una es falsa. ¿Cuántas son falsas?"},
        7: {"zh": "在某个外星世界中，生物的最大奔跑速度v取决于脚趾数n和眼睛数m。关系为v = k * n^a * m^b厘米/小时，其中k、a、b是整数常数。在所有生物都有5个脚趾的群体中，log(v) = 4 + 2*log(m)；在所有生物都有25只眼睛的群体中，log(v) = 4 + 4*log(n)。求k + a + b。",
             "es": "En un mundo alienígena, la velocidad máxima v depende del número de dedos n y ojos m: v = k * n^a * m^b cm/h. Con 5 dedos: log(v) = 4 + 2*log(m). Con 25 ojos: log(v) = 4 + 4*log(n). ¿Cuánto es k + a + b?"},
        9: {"zh": "设w是复数2 + i，其中i = sqrt(-1)。哪个实数r使得r、w和w²是复平面上三个共线的点？",
             "es": "Sea w = 2 + i, donde i = sqrt(-1). ¿Qué número real r tiene la propiedad de que r, w y w² son tres puntos colineales en el plano complejo?"},
        12: {"zh": "一组数的调和平均值是这组数倒数的算术平均值的倒数。求4050次多项式（等于从k=1到2025的乘积k*x²-4x-3）的所有实根的调和平均值。",
              "es": "La media armónica es el recíproco de la media aritmética de los recíprocos. ¿Cuál es la media armónica de todas las raíces reales del polinomio de grado 4050 igual al producto de k=1 a 2025 de (k*x²-4x-3)?"},
        15: {"zh": "如果集合中任意两个元素（可以相同）x和y的和x+y都不是集合的元素，则称该集合为无和集。{1,2,3,...,20}的无和子集最多能有多少个元素？",
              "es": "Un conjunto es libre de sumas si para cualesquiera x, y del conjunto, x+y no pertenece al conjunto. ¿Cuál es el mayor número de elementos en un subconjunto libre de sumas de {1,2,...,20}?"},
        18: {"zh": "有多少个有序三元组(x,y,z)，其中x、y、z是不同的正整数且都不超过8，满足xy > z、xz > y且yz > x？",
              "es": "¿Cuántas tripletas ordenadas (x,y,z) de enteros positivos distintos ≤ 8 satisfacen xy > z, xz > y, y yz > x?"},
        19: {"zh": "设a、b、c是多项式x³+kx+1的根。求a³b²+a²b³+b³c²+b²c³+c³a²+c²a³。",
              "es": "Sea a, b, c las raíces de x³+kx+1. ¿Cuánto es a³b²+a²b³+b³c²+b²c³+c³a²+c²a³?"},
    }

    for amc in AMC_2025_12A:
        num = amc["number"]
        trans = amc_translations.get(num, {})
        problems.append({
            "en": amc["en"],
            "zh": trans.get("zh", amc["en"]),  # fallback to EN if missing
            "es": trans.get("es", amc["en"]),
            "answer": str(amc["answer"]),
            "difficulty": "hard",
            "amc_number": num,
        })

    return problems


problems = build_problems()
print(f"Problem set: {len(problems)} problems")
print(f"  Easy: {sum(1 for p in problems if p['difficulty']=='easy')}")
print(f"  Medium: {sum(1 for p in problems if p['difficulty']=='medium')}")
print(f"  Hard (AMC): {sum(1 for p in problems if p['difficulty']=='hard')}")


# =============================================================================
# Capture infrastructure (same as v1)
# =============================================================================
class CoopCrystalCapture:
    def __init__(self, model, coop_layers, max_steps=200):
        self.model = model
        self.coop_layers = coop_layers
        self.max_steps = max_steps
        self.d = model.config.hidden_size
        self.device = next(model.parameters()).device
        self._hooks = []
        self._mlp_buffers = {}
        self._mlp_counters = {}
        self._logit_buffer = []

    def _reset(self):
        self._mlp_buffers = {}
        self._mlp_counters = {}
        for li in self.coop_layers:
            self._mlp_buffers[li] = torch.empty(
                (self.max_steps, self.d), device=self.device, dtype=torch.float32
            )
            self._mlp_counters[li] = 0
        self._logit_buffer = []

    def _register(self):
        self._hooks = []
        for li in self.coop_layers:
            layer = self.model.model.layers[li]
            def make_hook(idx):
                def hook(module, inp, out):
                    i = self._mlp_counters[idx]
                    if i < self.max_steps:
                        self._mlp_buffers[idx][i].copy_(out[0, -1, :].float())
                        self._mlp_counters[idx] = i + 1
                return hook
            self._hooks.append(layer.mlp.register_forward_hook(make_hook(li)))

        def logit_hook(module, inp, out):
            self._logit_buffer.append(out[0, -1, :].detach().float().cpu())
        self._hooks.append(self.model.lm_head.register_forward_hook(logit_hook))

    def _unregister(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run(self, input_ids, max_new_tokens=128):
        self._reset()
        self._register()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        self._unregister()
        prompt_len = input_ids.shape[1]
        gen_ids = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        mlp_deltas = {}
        for li in self.coop_layers:
            n = self._mlp_counters[li]
            if n > 1:
                mlp_deltas[li] = self._mlp_buffers[li][1:n].cpu().numpy()
        logits = self._logit_buffer[1:] if len(self._logit_buffer) > 1 else []
        return {"gen_text": gen_text, "n_gen_tokens": len(gen_ids),
                "gen_ids": gen_ids.cpu().numpy(), "mlp_deltas": mlp_deltas, "logits": logits}


# =============================================================================
# Run
# =============================================================================
capturer = CoopCrystalCapture(model, COOP_LAYERS, max_steps=MAX_NEW_TOKENS + 50)
all_results = []

for lang in ["en", "zh", "es"]:
    print(f"\n{'='*60}")
    print(f"Running {lang.upper()} ({len(problems)} problems)")
    print(f"{'='*60}")

    for pi, prob in enumerate(problems):
        prompt = prob[lang]
        answer_str = prob["answer"]
        answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        result = capturer.run(input_ids, max_new_tokens=MAX_NEW_TOKENS)

        # Per-token cooperation
        n_tok = min(*(result["mlp_deltas"].get(li, np.empty((0, d_model))).shape[0]
                      for li in COOP_LAYERS)) if all(li in result["mlp_deltas"] for li in COOP_LAYERS) else 0

        per_tok_coop = []
        for t in range(n_tok):
            cos_vals = []
            for i in range(1, len(COOP_LAYERS)):
                prev_li, curr_li = COOP_LAYERS[i-1], COOP_LAYERS[i]
                if prev_li in result["mlp_deltas"] and curr_li in result["mlp_deltas"]:
                    d_prev = result["mlp_deltas"][prev_li][t]
                    d_curr = result["mlp_deltas"][curr_li][t]
                    n_p = np.linalg.norm(d_prev) + 1e-10
                    n_c = np.linalg.norm(d_curr) + 1e-10
                    cos_vals.append(float(np.dot(d_prev, d_curr) / (n_p * n_c)))
            per_tok_coop.append(float(np.mean(cos_vals)) if cos_vals else 0.0)

        # Per-token p(answer)
        per_tok_p_answer = []
        for t_idx, logit_vec in enumerate(result["logits"]):
            if t_idx >= n_tok:
                break
            probs = torch.softmax(logit_vec, dim=0)
            p_ans = sum(probs[tid].item() for tid in answer_tokens)
            per_tok_p_answer.append(p_ans)

        n_paired = min(len(per_tok_coop), len(per_tok_p_answer))
        per_tok_coop = per_tok_coop[:n_paired]
        per_tok_p_answer = per_tok_p_answer[:n_paired]

        # Stats
        corr = None
        coop_peak_tok = None
        cryst_threshold_tok = None
        coop_first_half = None
        coop_second_half = None
        max_p_answer = 0.0

        if n_paired > 5:
            coop_arr = np.array(per_tok_coop)
            cryst_arr = np.array(per_tok_p_answer)
            max_p_answer = float(cryst_arr.max())
            if coop_arr.std() > 1e-10 and cryst_arr.std() > 1e-10:
                corr = float(np.corrcoef(coop_arr, cryst_arr)[0, 1])
            else:
                corr = 0.0
            coop_peak_tok = int(np.argmax(coop_arr[:min(50, n_paired)]))
            for t in range(n_paired):
                if cryst_arr[t] > 0.1:
                    cryst_threshold_tok = t
                    break
            half = n_paired // 2
            coop_first_half = float(coop_arr[:half].mean())
            coop_second_half = float(coop_arr[half:].mean())

        entry = {
            "problem_idx": pi, "lang": lang, "difficulty": prob["difficulty"],
            "answer": answer_str, "n_paired_tokens": n_paired,
            "correlation": corr, "coop_peak_token": coop_peak_tok,
            "cryst_threshold_token": cryst_threshold_tok,
            "coop_first_half_mean": coop_first_half,
            "coop_second_half_mean": coop_second_half,
            "max_p_answer": max_p_answer,
            "got_correct": answer_str in result["gen_text"],
            "per_tok_coop_first20": per_tok_coop[:20],
            "per_tok_p_answer_first20": per_tok_p_answer[:20],
        }
        if "amc_number" in prob:
            entry["amc_number"] = prob["amc_number"]
        all_results.append(entry)

        if (pi + 1) % 10 == 0:
            print(f"  {lang} {pi+1}/{len(problems)} done ({time.time()-t0:.0f}s)")


# =============================================================================
# Analysis
# =============================================================================
print(f"\n{'='*60}")
print("ANALYSIS")
print(f"{'='*60}")

valid = [r for r in all_results if r["correlation"] is not None]

# By difficulty
for diff in ["easy", "medium", "hard"]:
    subset = [r for r in valid if r["difficulty"] == diff]
    if subset:
        corrs = [r["correlation"] for r in subset]
        correct = [r for r in subset if r["got_correct"]]
        c1 = [r["coop_first_half_mean"] for r in subset if r["coop_first_half_mean"] is not None]
        c2 = [r["coop_second_half_mean"] for r in subset if r["coop_second_half_mean"] is not None]
        print(f"\n  {diff.upper()} ({len(subset)} runs, {len(correct)} correct):")
        print(f"    Correlation: mean={np.mean(corrs):.4f}, median={np.median(corrs):.4f}")
        if c1 and c2:
            print(f"    Coop 1st half: {np.mean(c1):.4f}, 2nd half: {np.mean(c2):.4f}")

# By language
for lang in ["en", "zh", "es"]:
    subset = [r for r in valid if r["lang"] == lang]
    if subset:
        corrs = [r["correlation"] for r in subset]
        correct = [r for r in subset if r["got_correct"]]
        print(f"\n  {lang.upper()} ({len(subset)} runs, {len(correct)} correct):")
        print(f"    Correlation: mean={np.mean(corrs):.4f}, median={np.median(corrs):.4f}")

# By difficulty × language
print(f"\n--- DIFFICULTY × LANGUAGE ---")
print(f"{'Diff':<8} {'Lang':<5} {'N':<5} {'Correct':<8} {'r_mean':<10} {'Coop1H':<10} {'Coop2H':<10} {'MaxP':<8}")
print("-" * 70)
for diff in ["easy", "medium", "hard"]:
    for lang in ["en", "zh", "es"]:
        subset = [r for r in valid if r["difficulty"] == diff and r["lang"] == lang]
        if subset:
            corrs = [r["correlation"] for r in subset]
            correct = sum(1 for r in subset if r["got_correct"])
            c1 = np.mean([r["coop_first_half_mean"] for r in subset if r["coop_first_half_mean"] is not None])
            c2 = np.mean([r["coop_second_half_mean"] for r in subset if r["coop_second_half_mean"] is not None])
            mp = np.mean([r["max_p_answer"] for r in subset])
            print(f"  {diff:<6} {lang:<5} {len(subset):<5} {correct:<8} {np.mean(corrs):<10.4f} {c1:<10.4f} {c2:<10.4f} {mp:<8.4f}")

# Correct vs incorrect across all difficulties
correct_all = [r for r in valid if r["got_correct"]]
incorrect_all = [r for r in valid if not r["got_correct"]]
print(f"\n  CORRECT ({len(correct_all)}):")
if correct_all:
    print(f"    r = {np.mean([r['correlation'] for r in correct_all]):.4f}")
    print(f"    Coop 1H = {np.mean([r['coop_first_half_mean'] for r in correct_all if r['coop_first_half_mean']]):.4f}")
print(f"  INCORRECT ({len(incorrect_all)}):")
if incorrect_all:
    print(f"    r = {np.mean([r['correlation'] for r in incorrect_all]):.4f}")
    print(f"    Coop 1H = {np.mean([r['coop_first_half_mean'] for r in incorrect_all if r['coop_first_half_mean']]):.4f}")


# =============================================================================
# Save
# =============================================================================
output = {
    "experiment": "Cooperation-Crystallization v2 (broader scope)",
    "model": MODEL_NAME,
    "n_problems": len(problems),
    "n_runs": len(all_results),
    "max_new_tokens": MAX_NEW_TOKENS,
    "coop_layers": COOP_LAYERS,
    "difficulties": {"easy": 10, "medium": 10, "hard": 10},
    "languages": ["en", "zh", "es"],
    "results": all_results,
    "runtime_seconds": time.time() - t0,
}

with open("output/exp_cooperation_crystallization_v2.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n\nResults saved to output/exp_cooperation_crystallization_v2.json")
print(f"Total runtime: {time.time()-t0:.0f}s")
