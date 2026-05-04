"""expMS2: 7-language centroid SVD surgery.

Convention direction from pure geometry: centroid SVD across 7 languages
using the existing multilingual cache (200 problems x 7 langs x 36 layers).
Surgery above l_c. Test on 20 problems in all 7 languages.

Prints FULL input/output for every problem as it runs.
"""
import json, re, time, sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_NEW = 1024
LANGS = ["en", "zh", "ar", "es", "ja", "ko", "sw"]

# System prompts per language (matched)
SYSTEM_PROMPTS = {
    "en": "You are a careful mathematical reasoner. When given a problem, think step by step, show your work clearly, and then state the final numerical answer on its own line.",
    "zh": "你是一个严谨的数学推理者。遇到问题时，请逐步思考，清晰地展示你的推导过程，然后在单独的一行给出最终的数值答案。",
    "ar": "أنت مفكر رياضي دقيق. عند تقديم مسألة، فكر خطوة بخطوة، واعرض عملك بوضوح، ثم اذكر الإجابة الرقمية النهائية في سطر منفصل.",
    "es": "Eres un razonador matemático cuidadoso. Cuando se te da un problema, piensa paso a paso, muestra tu trabajo claramente, y luego indica la respuesta numérica final en una línea aparte.",
    "ja": "あなたは慎重な数学的推論者です。問題が与えられたら、段階的に考え、作業を明確に示し、最終的な数値回答を独立した行に記述してください。",
    "ko": "당신은 신중한 수학적 추론자입니다. 문제가 주어지면 단계별로 생각하고, 작업 과정을 명확히 보여주고, 최종 수치 답을 별도의 줄에 기재하세요.",
    "sw": "Wewe ni mfikiriaji wa hisabati makini. Unapopewa tatizo, fikiria hatua kwa hatua, onyesha kazi yako kwa uwazi, kisha taja jibu la mwisho la nambari kwenye mstari wake.",
}


def get_test_problems():
    """20 test problems in all 7 languages."""
    # EN and ZH from MS1. Other langs generated via GPT-4.1 translation.
    problems = [
        {
            "en": "Solve for x: 3x + 7 = 22",
            "zh": "求解x：3x + 7 = 22",
            "ar": "أوجد قيمة x: 3x + 7 = 22",
            "es": "Resuelve para x: 3x + 7 = 22",
            "ja": "xについて解け：3x + 7 = 22",
            "ko": "x에 대해 풀어라: 3x + 7 = 22",
            "sw": "Tatua x: 3x + 7 = 22",
            "answer": "5", "category": "algebra",
        },
        {
            "en": "Solve for x: 2x² - 8 = 0",
            "zh": "求解x：2x² - 8 = 0",
            "ar": "أوجد قيمة x: 2x² - 8 = 0",
            "es": "Resuelve para x: 2x² - 8 = 0",
            "ja": "xについて解け：2x² - 8 = 0",
            "ko": "x에 대해 풀어라: 2x² - 8 = 0",
            "sw": "Tatua x: 2x² - 8 = 0",
            "answer": "2", "category": "algebra",
        },
        {
            "en": "Simplify: (x + 3)(x - 3)",
            "zh": "化简：(x + 3)(x - 3)",
            "ar": "بسّط: (x + 3)(x - 3)",
            "es": "Simplifica: (x + 3)(x - 3)",
            "ja": "簡略化せよ：(x + 3)(x - 3)",
            "ko": "간단히 하시오: (x + 3)(x - 3)",
            "sw": "Rahisisha: (x + 3)(x - 3)",
            "answer": "x² - 9", "category": "algebra",
        },
        {
            "en": "Solve: |2x - 5| = 3",
            "zh": "求解：|2x - 5| = 3",
            "ar": "حل: |2x - 5| = 3",
            "es": "Resuelve: |2x - 5| = 3",
            "ja": "解け：|2x - 5| = 3",
            "ko": "풀어라: |2x - 5| = 3",
            "sw": "Tatua: |2x - 5| = 3",
            "answer": "4", "category": "algebra",
        },
        {
            "en": "Calculate: 347 + 658",
            "zh": "计算：347 + 658",
            "ar": "احسب: 347 + 658",
            "es": "Calcula: 347 + 658",
            "ja": "計算せよ：347 + 658",
            "ko": "계산하시오: 347 + 658",
            "sw": "Hesabu: 347 + 658",
            "answer": "1005", "category": "arithmetic",
        },
        {
            "en": "Calculate: 1000 - 387",
            "zh": "计算：1000 - 387",
            "ar": "احسب: 1000 - 387",
            "es": "Calcula: 1000 - 387",
            "ja": "計算せよ：1000 - 387",
            "ko": "계산하시오: 1000 - 387",
            "sw": "Hesabu: 1000 - 387",
            "answer": "613", "category": "arithmetic",
        },
        {
            "en": "Calculate: 23 × 17",
            "zh": "计算：23 × 17",
            "ar": "احسب: 23 × 17",
            "es": "Calcula: 23 × 17",
            "ja": "計算せよ：23 × 17",
            "ko": "계산하시오: 23 × 17",
            "sw": "Hesabu: 23 × 17",
            "answer": "391", "category": "arithmetic",
        },
        {
            "en": "Calculate: 1728 ÷ 12",
            "zh": "计算：1728 ÷ 12",
            "ar": "احسب: 1728 ÷ 12",
            "es": "Calcula: 1728 ÷ 12",
            "ja": "計算せよ：1728 ÷ 12",
            "ko": "계산하시오: 1728 ÷ 12",
            "sw": "Hesabu: 1728 ÷ 12",
            "answer": "144", "category": "arithmetic",
        },
        {
            "en": "Find the area of a circle with radius 7 (use π ≈ 22/7)",
            "zh": "求半径为7的圆的面积（使用 π ≈ 22/7）",
            "ar": "أوجد مساحة دائرة نصف قطرها 7 (استخدم π ≈ 22/7)",
            "es": "Encuentra el área de un círculo con radio 7 (usa π ≈ 22/7)",
            "ja": "半径7の円の面積を求めよ（π ≈ 22/7を使用）",
            "ko": "반지름 7인 원의 넓이를 구하시오 (π ≈ 22/7 사용)",
            "sw": "Tafuta eneo la duara lenye radius 7 (tumia π ≈ 22/7)",
            "answer": "154", "category": "geometry",
        },
        {
            "en": "Find the hypotenuse of a right triangle with legs 5 and 12",
            "zh": "求直角三角形两直角边为5和12时的斜边长",
            "ar": "أوجد الوتر في مثلث قائم الزاوية طول ضلعيه 5 و12",
            "es": "Encuentra la hipotenusa de un triángulo rectángulo con catetos 5 y 12",
            "ja": "辺の長さが5と12の直角三角形の斜辺を求めよ",
            "ko": "두 변의 길이가 5와 12인 직각삼각형의 빗변을 구하시오",
            "sw": "Tafuta hypothenuse ya pembetatu ya pembe ya kulia yenye miguu 5 na 12",
            "answer": "13", "category": "geometry",
        },
        {
            "en": "What is the perimeter of a rectangle with length 15 and width 8?",
            "zh": "长为15宽为8的矩形的周长是多少？",
            "ar": "ما محيط مستطيل طوله 15 وعرضه 8؟",
            "es": "¿Cuál es el perímetro de un rectángulo de largo 15 y ancho 8?",
            "ja": "長さ15、幅8の長方形の周囲の長さは？",
            "ko": "길이 15, 너비 8인 직사각형의 둘레는?",
            "sw": "Mzunguko wa mstatili wenye urefu 15 na upana 8 ni upi?",
            "answer": "46", "category": "geometry",
        },
        {
            "en": "Find the volume of a cube with side length 6",
            "zh": "求边长为6的正方体的体积",
            "ar": "أوجد حجم مكعب طول ضلعه 6",
            "es": "Encuentra el volumen de un cubo con lado 6",
            "ja": "一辺の長さが6の立方体の体積を求めよ",
            "ko": "한 변의 길이가 6인 정육면체의 부피를 구하시오",
            "sw": "Tafuta ujazo wa mchemraba wenye upande wa 6",
            "answer": "216", "category": "geometry",
        },
        {
            "en": "What is the GCD of 84 and 120?",
            "zh": "84和120的最大公约数是多少？",
            "ar": "ما القاسم المشترك الأكبر لـ 84 و120؟",
            "es": "¿Cuál es el MCD de 84 y 120?",
            "ja": "84と120の最大公約数は？",
            "ko": "84와 120의 최대공약수는?",
            "sw": "Kigawanyo kikubwa cha pamoja cha 84 na 120 ni nini?",
            "answer": "12", "category": "number_theory",
        },
        {
            "en": "Is 97 prime? Answer yes or no, then explain.",
            "zh": "97是质数吗？回答是或否，然后解释。",
            "ar": "هل 97 عدد أولي؟ أجب بنعم أو لا، ثم اشرح.",
            "es": "¿Es 97 primo? Responde sí o no, luego explica.",
            "ja": "97は素数ですか？はいかいいえで答え、説明してください。",
            "ko": "97은 소수인가? 예 또는 아니오로 답한 후 설명하시오.",
            "sw": "Je, 97 ni nambari kuu? Jibu ndiyo au hapana, kisha eleza.",
            "answer": "yes", "category": "number_theory",
        },
        {
            "en": "Find the remainder when 2^10 is divided by 7",
            "zh": "求2^10除以7的余数",
            "ar": "أوجد الباقي عند قسمة 2^10 على 7",
            "es": "Encuentra el residuo cuando 2^10 se divide por 7",
            "ja": "2^10を7で割った余りを求めよ",
            "ko": "2^10을 7로 나눈 나머지를 구하시오",
            "sw": "Tafuta salio unapogawanya 2^10 kwa 7",
            "answer": "2", "category": "number_theory",
        },
        {
            "en": "What is the sum of all prime numbers less than 20?",
            "zh": "所有小于20的质数之和是多少？",
            "ar": "ما مجموع جميع الأعداد الأولية الأقل من 20؟",
            "es": "¿Cuál es la suma de todos los números primos menores que 20?",
            "ja": "20未満のすべての素数の和は？",
            "ko": "20 미만의 모든 소수의 합은?",
            "sw": "Jumla ya nambari zote kuu chini ya 20 ni ngapi?",
            "answer": "77", "category": "number_theory",
        },
        {
            "en": "How many ways can you choose 3 items from 7?",
            "zh": "从7个物品中选3个有多少种方式？",
            "ar": "كم عدد الطرق لاختيار 3 عناصر من 7؟",
            "es": "¿De cuántas maneras puedes elegir 3 elementos de 7?",
            "ja": "7つから3つを選ぶ方法は何通り？",
            "ko": "7개에서 3개를 선택하는 방법의 수는?",
            "sw": "Kuna njia ngapi za kuchagua vitu 3 kutoka 7?",
            "answer": "35", "category": "combinatorics",
        },
        {
            "en": "How many ways can 5 people stand in a line?",
            "zh": "5个人站成一排有多少种方式？",
            "ar": "كم عدد الطرق التي يمكن لـ 5 أشخاص الوقوف في صف؟",
            "es": "¿De cuántas maneras pueden 5 personas formar una fila?",
            "ja": "5人が一列に並ぶ方法は何通り？",
            "ko": "5명이 한 줄로 서는 방법의 수는?",
            "sw": "Kuna njia ngapi ambazo watu 5 wanaweza kusimama kwenye mstari?",
            "answer": "120", "category": "combinatorics",
        },
        {
            "en": "Calculate: 8! / (5! × 3!)",
            "zh": "计算：8! / (5! × 3!)",
            "ar": "احسب: 8! / (5! × 3!)",
            "es": "Calcula: 8! / (5! × 3!)",
            "ja": "計算せよ：8! / (5! × 3!)",
            "ko": "계산하시오: 8! / (5! × 3!)",
            "sw": "Hesabu: 8! / (5! × 3!)",
            "answer": "56", "category": "combinatorics",
        },
        {
            "en": "How many 3-digit numbers have all distinct digits?",
            "zh": "有多少个三位数的各位数字互不相同？",
            "ar": "كم عدداً مكوناً من 3 أرقام تكون جميع أرقامه مختلفة؟",
            "es": "¿Cuántos números de 3 dígitos tienen todos los dígitos distintos?",
            "ja": "すべての桁の数字が異なる3桁の数はいくつ？",
            "ko": "모든 자릿수가 서로 다른 3자리 수는 몇 개인가?",
            "sw": "Kuna nambari ngapi za tarakimu 3 zenye tarakimu zote tofauti?",
            "answer": "648", "category": "combinatorics",
        },
    ]
    return problems


def build_prompt(tokenizer, problem_text, lang):
    """Chat template with matched system prompt. lang=None for no system prompt."""
    sys_content = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{sys_content}\n\nProblem: {problem_text}\n\nSolution:"


def check_answer(text, correct):
    text_lower = text.lower().strip()
    if correct in ("yes", "no"):
        # Direct match
        if correct.lower() in text_lower:
            return True
        # "97 is prime" / "97 is a prime number" / equivalents in other languages
        if correct == "yes":
            # Match affirmative statements about being prime in multiple languages
            prime_patterns = [
                r"97\s+(is\s+)?(a\s+)?prime",       # EN
                r"97\s*是\s*质数", r"97\s*是\s*素数",   # ZH
                r"97\s*(هو\s*)?عدد\s*أولي",           # AR
                r"97\s+(es\s+)?(un\s+)?(número\s+)?primo",  # ES
                r"97\s*は\s*素数",                     # JA
                r"97\s*(은|는)\s*소수",                 # KO
                r"97\s+ni\s+nambari\s+kuu",           # SW
                r"是\s*的",                            # ZH: "是的" (= "yes")
                r"نعم",                               # AR: "nعm" (= "yes")
                r"sí",                                # ES
                r"はい",                               # JA
                r"예",                                 # KO
                r"ndiyo",                             # SW
            ]
            for pat in prime_patterns:
                if re.search(pat, text_lower if pat.isascii() else text):
                    return True
        return False
    # For x^2 - 9 type answers
    if "²" in correct or "x" in correct:
        return correct in text or correct.replace("²", "^2") in text
    # Numeric: extract numbers, strip trailing periods/commas
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    nums_clean = [n.rstrip(".,").replace(",", "") for n in nums]
    return str(correct) in nums_clean


def compute_convention_directions_centroid_svd(cache_path):
    """Compute convention directions from 7-language centroid SVD.

    For each problem, compute centroid across 7 languages.
    Stack all deviations from centroid. SVD gives convention subspace.
    Top-1 direction = e_c at each layer.
    """
    print("Loading 7-language cache for centroid SVD...", flush=True)
    data = np.load(cache_path, allow_pickle=True)

    directions = {}
    svd_info = {}

    for L in range(N_LAYERS):
        # Stack all 7 languages: (7, 200, 2048)
        lang_acts = []
        for lang in LANGS:
            key = f"{lang}_L{L}"
            lang_acts.append(data[key])  # (200, 2048)
        lang_acts = np.stack(lang_acts, axis=0)  # (7, 200, 2048)

        # Centroid per problem: average across 7 languages
        centroids = lang_acts.mean(axis=0)  # (200, 2048)

        # Deviations from centroid: (7, 200, 2048) - (1, 200, 2048)
        deviations = lang_acts - centroids[np.newaxis, :, :]  # (7, 200, 2048)

        # Reshape to (1400, 2048) and center
        dev_flat = deviations.reshape(-1, D_MODEL)  # (1400, 2048)
        dev_flat -= dev_flat.mean(axis=0, keepdims=True)

        # SVD
        U, S, Vt = np.linalg.svd(dev_flat, full_matrices=False)

        # Top-1 convention direction
        directions[L] = Vt[0]  # (2048,)

        # Info for reporting
        total_var = (S ** 2).sum()
        top1_var = S[0] ** 2 / total_var
        top5_var = (S[:5] ** 2).sum() / total_var
        r90 = np.searchsorted(np.cumsum(S ** 2) / total_var, 0.90) + 1

        svd_info[L] = {
            "sv1_sv2": float(S[0] / S[1]) if S[1] > 0 else float("inf"),
            "top1_var": float(top1_var),
            "top5_var": float(top5_var),
            "r90": int(r90),
        }

        if L % 6 == 0 or L == N_LAYERS - 1:
            print(f"  L{L:2d}: sv1/sv2={S[0]/S[1]:.2f}, "
                  f"top1={top1_var:.3f}, r90={r90}", flush=True)

    return directions, svd_info


def apply_surgery(model, directions, layers_to_modify, device):
    """Project out top-1 convention direction from W_down at specified layers."""
    for L in layers_to_modify:
        e_c = torch.tensor(directions[L], dtype=torch.float16, device=device)
        W = model.model.layers[L].mlp.down_proj.weight.data  # (2048, 11008)
        proj = e_c.unsqueeze(0) @ W  # (1, 11008)
        W.sub_(e_c.unsqueeze(1) @ proj)


def run_eval(model, tokenizer, problems, langs_to_eval, label, device):
    """Run evaluation, printing full I/O for every problem."""
    results = {}
    total_correct = 0
    total_count = 0

    for lang in langs_to_eval:
        lang_correct = 0
        lang_results = []

        print(f"\n{'=' * 80}", flush=True)
        print(f"  [{label}] Language: {lang.upper()}", flush=True)
        print(f"{'=' * 80}", flush=True)

        for i, prob in enumerate(problems):
            prompt = build_prompt(tokenizer, prob[lang], lang)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            with torch.inference_mode():
                out = model.generate(
                    ids, max_new_tokens=MAX_NEW, do_sample=False,
                    temperature=None, top_p=None,
                )
            gen_tokens = out.shape[1] - ids.shape[1]
            gen_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            correct = check_answer(gen_text, prob["answer"])

            if correct:
                lang_correct += 1
            total_correct += int(correct)
            total_count += 1

            # Print full I/O
            mark = "OK" if correct else "XX"
            print(f"\n--- [{mark}] Problem {i+1}/{len(problems)} "
                  f"({prob['category']}) | answer={prob['answer']} | "
                  f"{gen_tokens} tokens ---", flush=True)
            print(f"  INPUT:  {prob[lang]}", flush=True)
            print(f"  OUTPUT: {gen_text[:500]}", flush=True)
            if len(gen_text) > 500:
                print(f"  ... ({len(gen_text)} chars total)", flush=True)

            lang_results.append({
                "problem_idx": i,
                "category": prob["category"],
                "input": prob[lang],
                "answer": prob["answer"],
                "correct": correct,
                "gen_tokens": gen_tokens,
                "output": gen_text,
            })

        print(f"\n  >>> {lang.upper()} score: {lang_correct}/{len(problems)}", flush=True)
        results[lang] = {
            "correct": lang_correct,
            "total": len(problems),
            "details": lang_results,
        }

    print(f"\n{'=' * 80}", flush=True)
    print(f"  [{label}] TOTAL: {total_correct}/{total_count} "
          f"({total_correct/total_count*100:.1f}%)", flush=True)
    for lang in langs_to_eval:
        r = results[lang]
        print(f"    {lang.upper()}: {r['correct']}/{r['total']}", flush=True)
    print(f"{'=' * 80}", flush=True)

    return results


def main():
    device = "cuda"
    problems = get_test_problems()
    above_lc = list(range(13, 36))  # L13-L35

    print(f"{'#' * 80}", flush=True)
    print(f"  Exp MS2: 7-Language Centroid SVD Surgery", flush=True)
    print(f"{'#' * 80}", flush=True)
    print(f"Model:      {MODEL_NAME}", flush=True)
    print(f"Problems:   {len(problems)} x {len(LANGS)} langs = "
          f"{len(problems)*len(LANGS)} evals per condition", flush=True)
    print(f"Languages:  {LANGS}", flush=True)
    print(f"Max tokens: {MAX_NEW}", flush=True)
    print(f"Surgery:    above l_c (L13-L35)", flush=True)
    print(f"Convention: centroid SVD from 200-problem 7-lang cache", flush=True)
    print(flush=True)

    t0 = time.time()

    # Step 1: Compute convention directions from cache
    cache_path = "output/multilingual_all_layers.npz"
    directions, svd_info = compute_convention_directions_centroid_svd(cache_path)

    # Step 2: Load model
    print("\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Save original weights
    print("Saving original weights...", flush=True)
    original_weights = {}
    for L in range(N_LAYERS):
        original_weights[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    # Step 3: Baseline evaluation (all 7 languages)
    print("\n\n" + "#" * 80, flush=True)
    print("  BASELINE (no surgery)", flush=True)
    print("#" * 80, flush=True)
    baseline = run_eval(model, tokenizer, problems, LANGS, "BASELINE", device)

    # Step 4: Apply surgery and evaluate
    print("\nApplying centroid SVD surgery above l_c...", flush=True)
    apply_surgery(model, directions, above_lc, device)

    print("\n\n" + "#" * 80, flush=True)
    print("  SURGERY (centroid SVD, above l_c)", flush=True)
    print("#" * 80, flush=True)
    surgery = run_eval(model, tokenizer, problems, LANGS, "SURGERY", device)

    # Restore weights
    for L in range(N_LAYERS):
        model.model.layers[L].mlp.down_proj.weight.data.copy_(original_weights[L])

    # Summary
    print(f"\n\n{'#' * 80}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'#' * 80}", flush=True)
    print(f"{'Lang':<6s} {'Baseline':>10s} {'Surgery':>10s} {'Delta':>8s}", flush=True)
    print(f"{'-' * 40}", flush=True)
    total_bl = 0
    total_sg = 0
    for lang in LANGS:
        bl = baseline[lang]["correct"]
        sg = surgery[lang]["correct"]
        total_bl += bl
        total_sg += sg
        delta = sg - bl
        print(f"{lang.upper():<6s} {bl:>5d}/20   {sg:>5d}/20   {delta:+d}", flush=True)
    print(f"{'-' * 40}", flush=True)
    print(f"{'TOTAL':<6s} {total_bl:>5d}/140  {total_sg:>5d}/140  "
          f"{total_sg - total_bl:+d}", flush=True)
    print(f"{'#' * 80}", flush=True)

    wall = time.time() - t0
    print(f"\nWall time: {wall:.0f}s ({wall/60:.1f}min)", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = {
        "experiment": "MS2_7lang_centroid_svd_surgery",
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW,
        "n_problems": len(problems),
        "languages": LANGS,
        "surgery_layers": above_lc,
        "svd_info": {str(k): v for k, v in svd_info.items()},
        "baseline": {
            lang: {"correct": baseline[lang]["correct"],
                   "total": baseline[lang]["total"],
                   "details": baseline[lang]["details"]}
            for lang in LANGS
        },
        "surgery": {
            lang: {"correct": surgery[lang]["correct"],
                   "total": surgery[lang]["total"],
                   "details": surgery[lang]["details"]}
            for lang in LANGS
        },
        "wall_time_s": wall,
    }
    out_file = OUTPUT_DIR / "expMS2_7lang_surgery.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
