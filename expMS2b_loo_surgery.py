"""expMS2b: Leave-one-out centroid SVD surgery.

Fix for the centroid bias in MS2: for each eval language, compute the centroid
from the OTHER 6 languages only. This removes self-reinforcing bias where
the language being evaluated contributes to its own convention direction.

Also includes: bilateral mean-diff surgery (MS1-style, but averaged over all
6 partner languages instead of just EN-ZH).

Fixed grader from MS2 (handles "97 is prime", trailing dots, etc.)
"""
import json, re, time, copy, sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.sparse.linalg import svds

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_NEW = 1024
LANGS = ["en", "zh", "ar", "es", "ja", "ko", "sw"]

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
    problems = [
        {"en": "Solve for x: 3x + 7 = 22", "zh": "求解x：3x + 7 = 22", "ar": "أوجد قيمة x: 3x + 7 = 22", "es": "Resuelve para x: 3x + 7 = 22", "ja": "xについて解け：3x + 7 = 22", "ko": "x에 대해 풀어라: 3x + 7 = 22", "sw": "Tatua x: 3x + 7 = 22", "answer": "5", "category": "algebra"},
        {"en": "Solve for x: 2x² - 8 = 0", "zh": "求解x：2x² - 8 = 0", "ar": "أوجد قيمة x: 2x² - 8 = 0", "es": "Resuelve para x: 2x² - 8 = 0", "ja": "xについて解け：2x² - 8 = 0", "ko": "x에 대해 풀어라: 2x² - 8 = 0", "sw": "Tatua x: 2x² - 8 = 0", "answer": "2", "category": "algebra"},
        {"en": "Simplify: (x + 3)(x - 3)", "zh": "化简：(x + 3)(x - 3)", "ar": "بسّط: (x + 3)(x - 3)", "es": "Simplifica: (x + 3)(x - 3)", "ja": "簡略化せよ：(x + 3)(x - 3)", "ko": "간단히 하시오: (x + 3)(x - 3)", "sw": "Rahisisha: (x + 3)(x - 3)", "answer": "x² - 9", "category": "algebra"},
        {"en": "Solve: |2x - 5| = 3", "zh": "求解：|2x - 5| = 3", "ar": "حل: |2x - 5| = 3", "es": "Resuelve: |2x - 5| = 3", "ja": "解け：|2x - 5| = 3", "ko": "풀어라: |2x - 5| = 3", "sw": "Tatua: |2x - 5| = 3", "answer": "4", "category": "algebra"},
        {"en": "Calculate: 347 + 658", "zh": "计算：347 + 658", "ar": "احسب: 347 + 658", "es": "Calcula: 347 + 658", "ja": "計算せよ：347 + 658", "ko": "계산하시오: 347 + 658", "sw": "Hesabu: 347 + 658", "answer": "1005", "category": "arithmetic"},
        {"en": "Calculate: 1000 - 387", "zh": "计算：1000 - 387", "ar": "احسب: 1000 - 387", "es": "Calcula: 1000 - 387", "ja": "計算せよ：1000 - 387", "ko": "계산하시오: 1000 - 387", "sw": "Hesabu: 1000 - 387", "answer": "613", "category": "arithmetic"},
        {"en": "Calculate: 23 × 17", "zh": "计算：23 × 17", "ar": "احسب: 23 × 17", "es": "Calcula: 23 × 17", "ja": "計算せよ：23 × 17", "ko": "계산하시오: 23 × 17", "sw": "Hesabu: 23 × 17", "answer": "391", "category": "arithmetic"},
        {"en": "Calculate: 1728 ÷ 12", "zh": "计算：1728 ÷ 12", "ar": "احسب: 1728 ÷ 12", "es": "Calcula: 1728 ÷ 12", "ja": "計算せよ：1728 ÷ 12", "ko": "계산하시오: 1728 ÷ 12", "sw": "Hesabu: 1728 ÷ 12", "answer": "144", "category": "arithmetic"},
        {"en": "Find the area of a circle with radius 7 (use π ≈ 22/7)", "zh": "求半径为7的圆的面积（使用 π ≈ 22/7）", "ar": "أوجد مساحة دائرة نصف قطرها 7 (استخدم π ≈ 22/7)", "es": "Encuentra el área de un círculo con radio 7 (usa π ≈ 22/7)", "ja": "半径7の円の面積を求めよ（π ≈ 22/7を使用）", "ko": "반지름 7인 원의 넓이를 구하시오 (π ≈ 22/7 사용)", "sw": "Tafuta eneo la duara lenye radius 7 (tumia π ≈ 22/7)", "answer": "154", "category": "geometry"},
        {"en": "Find the hypotenuse: right triangle with legs 5 and 12", "zh": "求直角三角形两直角边为5和12时的斜边长", "ar": "أوجد الوتر في مثلث قائم الزاوية طول ضلعيه 5 و12", "es": "Encuentra la hipotenusa de un triángulo rectángulo con catetos 5 y 12", "ja": "辺の長さが5と12の直角三角形の斜辺を求めよ", "ko": "두 변의 길이가 5와 12인 직각삼각형의 빗변을 구하시오", "sw": "Tafuta hypothenuse ya pembetatu ya pembe ya kulia yenye miguu 5 na 12", "answer": "13", "category": "geometry"},
        {"en": "What is the perimeter of a rectangle with length 15 and width 8?", "zh": "长为15宽为8的矩形的周长是多少？", "ar": "ما محيط مستطيل طوله 15 وعرضه 8؟", "es": "¿Cuál es el perímetro de un rectángulo de largo 15 y ancho 8?", "ja": "長さ15、幅8の長方形の周囲の長さは？", "ko": "길이 15, 너비 8인 직사각형의 둘레는?", "sw": "Mzunguko wa mstatili wenye urefu 15 na upana 8 ni upi?", "answer": "46", "category": "geometry"},
        {"en": "Find the volume of a cube with side length 6", "zh": "求边长为6的正方体的体积", "ar": "أوجد حجم مكعب طول ضلعه 6", "es": "Encuentra el volumen de un cubo con lado 6", "ja": "一辺の長さが6の立方体の体積を求めよ", "ko": "한 변의 길이가 6인 정육면체의 부피를 구하시오", "sw": "Tafuta ujazo wa mchemraba wenye upande wa 6", "answer": "216", "category": "geometry"},
        {"en": "What is the GCD of 84 and 120?", "zh": "84和120的最大公约数是多少？", "ar": "ما القاسم المشترك الأكبر لـ 84 و120؟", "es": "¿Cuál es el MCD de 84 y 120?", "ja": "84と120の最大公約数は？", "ko": "84와 120의 최대공약수는?", "sw": "Kigawanyo kikubwa cha pamoja cha 84 na 120 ni nini?", "answer": "12", "category": "number_theory"},
        {"en": "Is 97 prime? Answer yes or no, then explain.", "zh": "97是质数吗？回答是或否，然后解释。", "ar": "هل 97 عدد أولي؟ أجب بنعم أو لا، ثم اشرح.", "es": "¿Es 97 primo? Responde sí o no, luego explica.", "ja": "97は素数ですか？はいかいいえで答え、説明してください。", "ko": "97은 소수인가? 예 또는 아니오로 답한 후 설명하시오.", "sw": "Je, 97 ni nambari kuu? Jibu ndiyo au hapana, kisha eleza.", "answer": "yes", "category": "number_theory"},
        {"en": "Find the remainder when 2^10 is divided by 7", "zh": "求2^10除以7的余数", "ar": "أوجد الباقي عند قسمة 2^10 على 7", "es": "Encuentra el residuo cuando 2^10 se divide por 7", "ja": "2^10を7で割った余りを求めよ", "ko": "2^10을 7로 나눈 나머지를 구하시오", "sw": "Tafuta salio unapogawanya 2^10 kwa 7", "answer": "2", "category": "number_theory"},
        {"en": "What is the sum of all prime numbers less than 20?", "zh": "所有小于20的质数之和是多少？", "ar": "ما مجموع جميع الأعداد الأولية الأقل من 20؟", "es": "¿Cuál es la suma de todos los números primos menores que 20?", "ja": "20未満のすべての素数の和は？", "ko": "20 미만의 모든 소수의 합은?", "sw": "Jumla ya nambari zote kuu chini ya 20 ni ngapi?", "answer": "77", "category": "number_theory"},
        {"en": "How many ways can you choose 3 items from 7?", "zh": "从7个物品中选3个有多少种方式？", "ar": "كم عدد الطرق لاختيار 3 عناصر من 7؟", "es": "¿De cuántas maneras puedes elegir 3 elementos de 7?", "ja": "7つから3つを選ぶ方法は何通り？", "ko": "7개에서 3개를 선택하는 방법의 수는?", "sw": "Kuna njia ngapi za kuchagua vitu 3 kutoka 7?", "answer": "35", "category": "combinatorics"},
        {"en": "How many ways can 5 people stand in a line?", "zh": "5个人站成一排有多少种方式？", "ar": "كم عدد الطرق التي يمكن لـ 5 أشخاص الوقوف في صف؟", "es": "¿De cuántas maneras pueden 5 personas formar una fila?", "ja": "5人が一列に並ぶ方法は何通り？", "ko": "5명이 한 줄로 서는 방법의 수는?", "sw": "Kuna njia ngapi ambazo watu 5 wanaweza kusimama kwenye mstari?", "answer": "120", "category": "combinatorics"},
        {"en": "Calculate: 8! / (5! × 3!)", "zh": "计算：8! / (5! × 3!)", "ar": "احسب: 8! / (5! × 3!)", "es": "Calcula: 8! / (5! × 3!)", "ja": "計算せよ：8! / (5! × 3!)", "ko": "계산하시오: 8! / (5! × 3!)", "sw": "Hesabu: 8! / (5! × 3!)", "answer": "56", "category": "combinatorics"},
        {"en": "How many 3-digit numbers have all distinct digits?", "zh": "有多少个三位数的各位数字互不相同？", "ar": "كم عدداً مكوناً من 3 أرقام تكون جميع أرقامه مختلفة؟", "es": "¿Cuántos números de 3 dígitos tienen todos los dígitos distintos?", "ja": "すべての桁の数字が異なる3桁の数はいくつ？", "ko": "모든 자릿수가 서로 다른 3자리 수는 몇 개인가?", "sw": "Kuna nambari ngapi za tarakimu 3 zenye tarakimu zote tofauti?", "answer": "648", "category": "combinatorics"},
    ]
    return problems


def build_prompt(tokenizer, problem_text, lang):
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
    """Fixed grader: handles 'X is prime', trailing dots, comma-separated numbers."""
    text_lower = text.lower().strip()
    if correct in ("yes", "no"):
        if correct.lower() in text_lower:
            return True
        if correct == "yes":
            prime_patterns = [
                r"97\s+(is\s+)?(a\s+)?prime",
                r"97\s*是\s*质数", r"97\s*是\s*素数",
                r"97\s*(هو\s*)?عدد\s*أولي",
                r"97\s+(es\s+)?(un\s+)?(número\s+)?primo",
                r"97\s*は\s*素数",
                r"97\s*(은|는)\s*소수",
                r"97\s+ni\s+nambari\s+kuu",
                r"是\s*的", r"نعم", r"sí", r"はい", r"예", r"ndiyo",
            ]
            for pat in prime_patterns:
                if re.search(pat, text_lower if pat.isascii() else text):
                    return True
        return False
    if "²" in correct or "x" in correct:
        return correct in text or correct.replace("²", "^2") in text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    nums_clean = [n.rstrip(".,").replace(",", "") for n in nums]
    return str(correct) in nums_clean


# --------------- Convention direction methods ---------------

def compute_loo_centroid_directions(cache_path):
    """Leave-one-out centroid SVD: for each language, centroid from other 6.

    Memory-efficient: loads one layer at a time, uses float32.
    Returns dict: lang -> {layer: direction_vector}
    """
    print("Loading 7-language cache for LOO centroid SVD...", flush=True)
    data = np.load(cache_path, allow_pickle=True)

    loo_directions = {lang: {} for lang in LANGS}
    loo_svd_info = {lang: {} for lang in LANGS}

    for L in range(N_LAYERS):
        # Load this layer for all languages: (7, 200, 2048) in float32
        lang_acts = np.stack(
            [data[f"{lang}_L{L}"].astype(np.float32) for lang in LANGS], axis=0
        )

        for ti, target_lang in enumerate(LANGS):
            # Centroid from other 6
            mask = np.ones(7, dtype=bool)
            mask[ti] = False
            centroids = lang_acts[mask].mean(axis=0)  # (200, 2048)

            # Deviations of all 7 from this centroid
            deviations = lang_acts - centroids[np.newaxis, :, :]  # (7, 200, 2048)
            dev_flat = deviations.reshape(-1, D_MODEL).astype(np.float64)
            dev_flat -= dev_flat.mean(axis=0, keepdims=True)

            # Truncated SVD: only top-2 (need sv1/sv2 ratio)
            U2, S2, Vt2 = svds(dev_flat, k=2, which='LM')
            # svds returns in ascending order; reverse
            idx = np.argsort(-S2)
            S2 = S2[idx]
            Vt2 = Vt2[idx]

            loo_directions[target_lang][L] = Vt2[0].astype(np.float32).copy()

            loo_svd_info[target_lang][L] = {
                "sv1_sv2": float(S2[0] / S2[1]) if S2[1] > 0 else float("inf"),
                "r90": -1,  # not computed with truncated SVD
            }

        if L % 6 == 0 or L == N_LAYERS - 1:
            info_str = ", ".join(
                f"{lang.upper()}={loo_svd_info[lang][L]['sv1_sv2']:.2f}"
                for lang in LANGS[:3]
            )
            print(f"  L{L:2d}: sv1/sv2 [{info_str}, ...]", flush=True)

    return loo_directions, loo_svd_info


def compute_bilateral_directions(cache_path):
    """Bilateral mean-diff: for each target language, average of 6 pairwise
    mean-diff directions (target - other_i), normalized.

    Memory-efficient: one layer at a time.
    Returns dict: lang -> {layer: direction_vector}
    """
    print("Computing bilateral mean-diff directions...", flush=True)
    data = np.load(cache_path, allow_pickle=True)

    bilateral_directions = {lang: {} for lang in LANGS}

    for L in range(N_LAYERS):
        # Load means for this layer
        lang_means = {}
        for lang in LANGS:
            lang_means[lang] = data[f"{lang}_L{L}"].astype(np.float32).mean(axis=0)

        for ti, target_lang in enumerate(LANGS):
            other_langs = [l for l in LANGS if l != target_lang]
            diffs = []
            for other in other_langs:
                diff = lang_means[target_lang] - lang_means[other]
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    diffs.append(diff / norm)
            avg_diff = np.mean(diffs, axis=0)
            avg_diff /= np.linalg.norm(avg_diff) + 1e-10
            bilateral_directions[target_lang][L] = avg_diff

    print("  Bilateral: done (all layers, all languages)", flush=True)
    return bilateral_directions


def apply_surgery(model, directions, layers_to_modify, device):
    """Project out convention direction from W_down at specified layers."""
    for L in layers_to_modify:
        e_c = torch.tensor(directions[L], dtype=torch.float16, device=device)
        W = model.model.layers[L].mlp.down_proj.weight.data
        proj = e_c.unsqueeze(0) @ W
        W.sub_(e_c.unsqueeze(1) @ proj)


def restore_weights(model, original_weights):
    for L in range(N_LAYERS):
        model.model.layers[L].mlp.down_proj.weight.data.copy_(original_weights[L])


def run_eval(model, tokenizer, problems, langs_to_eval, label, device):
    """Run evaluation with full I/O printing."""
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

    total = sum(results[l]["correct"] for l in langs_to_eval)
    n = sum(results[l]["total"] for l in langs_to_eval)
    print(f"\n  [{label}] TOTAL: {total}/{n} ({total/n*100:.1f}%)", flush=True)
    for lang in langs_to_eval:
        r = results[lang]
        print(f"    {lang.upper()}: {r['correct']}/{r['total']}", flush=True)

    return results


def run_lang_specific_surgery_eval(model, tokenizer, problems, lang_directions,
                                   layers_to_modify, original_weights,
                                   label, device):
    """For each language, apply that language's specific surgery, eval, restore."""
    results = {}

    for lang in LANGS:
        # Restore clean weights
        restore_weights(model, original_weights)
        # Apply this language's surgery
        apply_surgery(model, lang_directions[lang], layers_to_modify, device)
        # Eval just this language
        lang_result = run_eval(model, tokenizer, problems, [lang],
                               f"{label}-{lang.upper()}", device)
        results[lang] = lang_result[lang]

    # Restore clean
    restore_weights(model, original_weights)

    # Summary
    total = sum(results[l]["correct"] for l in LANGS)
    n = sum(results[l]["total"] for l in LANGS)
    print(f"\n{'=' * 80}", flush=True)
    print(f"  [{label}] COMBINED: {total}/{n} ({total/n*100:.1f}%)", flush=True)
    for lang in LANGS:
        r = results[lang]
        print(f"    {lang.upper()}: {r['correct']}/{r['total']}", flush=True)
    print(f"{'=' * 80}", flush=True)

    return results


def main():
    device = "cuda"
    problems = get_test_problems()
    above_lc = list(range(13, 36))
    cache_path = "output/multilingual_all_layers.npz"

    print(f"{'#' * 80}", flush=True)
    print(f"  Exp MS2b: Leave-One-Out + Bilateral Surgery", flush=True)
    print(f"{'#' * 80}", flush=True)
    print(f"Model:      {MODEL_NAME}", flush=True)
    print(f"Problems:   {len(problems)} x {len(LANGS)} langs", flush=True)
    print(f"Max tokens: {MAX_NEW}", flush=True)
    print(f"Surgery:    L13-L35 (above l_c)", flush=True)
    print(f"Methods:    (1) LOO centroid SVD, (2) Bilateral mean-diff", flush=True)
    print(flush=True)

    t0 = time.time()

    # Step 1: Compute directions
    loo_dirs, loo_info = compute_loo_centroid_directions(cache_path)
    bilateral_dirs = compute_bilateral_directions(cache_path)

    # Step 2: Load model
    print("\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Save original weights
    original_weights = {}
    for L in range(N_LAYERS):
        original_weights[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    # Step 3: Baseline (fixed grader)
    print("\n\n" + "#" * 80, flush=True)
    print("  BASELINE (no surgery, fixed grader)", flush=True)
    print("#" * 80, flush=True)
    baseline = run_eval(model, tokenizer, problems, LANGS, "BASELINE", device)

    # Step 4: LOO centroid surgery
    print("\n\n" + "#" * 80, flush=True)
    print("  LOO CENTROID SVD SURGERY (per-language centroid from other 6)", flush=True)
    print("#" * 80, flush=True)
    loo_results = run_lang_specific_surgery_eval(
        model, tokenizer, problems, loo_dirs, above_lc, original_weights,
        "LOO-SVD", device
    )

    # Step 5: Bilateral mean-diff surgery
    print("\n\n" + "#" * 80, flush=True)
    print("  BILATERAL MEAN-DIFF SURGERY (avg of 6 pairwise diffs)", flush=True)
    print("#" * 80, flush=True)
    bilateral_results = run_lang_specific_surgery_eval(
        model, tokenizer, problems, bilateral_dirs, above_lc, original_weights,
        "BILATERAL", device
    )

    # Final summary
    wall = time.time() - t0
    print(f"\n\n{'#' * 80}", flush=True)
    print(f"  FINAL SUMMARY (fixed grader)", flush=True)
    print(f"{'#' * 80}", flush=True)
    print(f"{'Lang':<6s} {'Baseline':>10s} {'LOO-SVD':>10s} {'Δ':>5s} {'Bilateral':>10s} {'Δ':>5s}", flush=True)
    print(f"{'-' * 52}", flush=True)
    t_bl, t_loo, t_bi = 0, 0, 0
    for lang in LANGS:
        bl = baseline[lang]["correct"]
        lo = loo_results[lang]["correct"]
        bi = bilateral_results[lang]["correct"]
        t_bl += bl; t_loo += lo; t_bi += bi
        print(f"{lang.upper():<6s} {bl:>5d}/20   {lo:>5d}/20  {lo-bl:>+3d}  "
              f"{bi:>5d}/20  {bi-bl:>+3d}", flush=True)
    print(f"{'-' * 52}", flush=True)
    print(f"{'TOTAL':<6s} {t_bl:>5d}/140  {t_loo:>5d}/140 {t_loo-t_bl:>+4d}  "
          f"{t_bi:>5d}/140 {t_bi-t_bl:>+4d}", flush=True)
    print(f"\nWall time: {wall:.0f}s ({wall/60:.1f}min)", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = {
        "experiment": "MS2b_loo_bilateral_surgery",
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW,
        "n_problems": len(problems),
        "languages": LANGS,
        "surgery_layers": above_lc,
        "methods": ["loo_centroid_svd", "bilateral_mean_diff"],
        "loo_svd_info": {
            lang: {str(k): v for k, v in info.items()}
            for lang, info in loo_info.items()
        },
        "baseline": {
            lang: {"correct": baseline[lang]["correct"],
                   "total": baseline[lang]["total"],
                   "details": baseline[lang]["details"]}
            for lang in LANGS
        },
        "loo_surgery": {
            lang: {"correct": loo_results[lang]["correct"],
                   "total": loo_results[lang]["total"],
                   "details": loo_results[lang]["details"]}
            for lang in LANGS
        },
        "bilateral_surgery": {
            lang: {"correct": bilateral_results[lang]["correct"],
                   "total": bilateral_results[lang]["total"],
                   "details": bilateral_results[lang]["details"]}
            for lang in LANGS
        },
        "wall_time_s": wall,
    }
    out_file = OUTPUT_DIR / "expMS2b_loo_bilateral_surgery.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
