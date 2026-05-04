"""exp_z_encoder: Can a linear map from L30 activations predict answers cross-lingually?

The $1M test: if f* is rank-1 at L30, then a linear projection from
R^2048 -> R^1 should predict the numerical answer from activations alone.

Train on 10 problems in {EN, ZH, ES} (30 samples).
Test on held-out 10 problems in ALL 7 languages (70 samples).
If R^2 > 0 on held-out problems in unseen languages (AR, JA, KO, SW),
f* is extractable and cross-lingual.

Also test: can the Z representation distinguish WHICH problem
is being computed, not just the answer value?
"""
import json, time
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
N_LAYERS = 36
D_MODEL = 2048
DEVICE = 'cuda'

SYS_PROMPTS = {
    'en': 'You are a careful mathematical reasoner. Think step by step.',
    'zh': '\u4f60\u662f\u4e00\u4e2a\u4e25\u8c28\u7684\u6570\u5b66\u63a8\u7406\u8005\u3002\u8bf7\u9010\u6b65\u601d\u8003\u3002',
    'es': 'Eres un razonador matem\u00e1tico cuidadoso. Piensa paso a paso.',
    'ar': '\u0623\u0646\u062a \u0645\u0641\u0643\u0631 \u0631\u064a\u0627\u0636\u064a \u062f\u0642\u064a\u0642. \u0641\u0643\u0631 \u062e\u0637\u0648\u0629 \u0628\u062e\u0637\u0648\u0629.',
    'ja': '\u3042\u306a\u305f\u306f\u6ce8\u610f\u6df1\u3044\u6570\u5b66\u7684\u63a8\u8ad6\u8005\u3067\u3059\u3002\u6bb5\u968e\u7684\u306b\u8003\u3048\u3066\u304f\u3060\u3055\u3044\u3002',
    'ko': '\ub2f9\uc2e0\uc740 \uc2e0\uc911\ud55c \uc218\ud559\uc801 \ucd94\ub860\uc790\uc785\ub2c8\ub2e4. \ub2e8\uacc4\ubcc4\ub85c \uc0dd\uac01\ud558\uc138\uc694.',
    'sw': 'Wewe ni mfikiria wa hisabati makini. Fikiria hatua kwa hatua.',
}


def get_problems_7lang():
    """20 problems with 7-language translations. Numeric answers only (skip string answers)."""
    problems = [
        {"en": "Solve for x: 3x + 7 = 22", "zh": "\u89e3\u65b9\u7a0b\uff1a3x + 7 = 22",
         "es": "Resuelve para x: 3x + 7 = 22", "ar": "\u062d\u0644 \u0644\u0640 x: 3x + 7 = 22",
         "ja": "x\u3092\u89e3\u304d\u306a\u3055\u3044: 3x + 7 = 22", "ko": "x\ub97c \ud480\uc5b4\ub77c: 3x + 7 = 22",
         "sw": "Tatua x: 3x + 7 = 22", "answer": 5, "category": "algebra"},
        {"en": "Solve for x: 2x\u00b2 - 8 = 0", "zh": "\u89e3\u65b9\u7a0b\uff1a2x\u00b2 - 8 = 0",
         "es": "Resuelve para x: 2x\u00b2 - 8 = 0", "ar": "\u062d\u0644 \u0644\u0640 x: 2x\u00b2 - 8 = 0",
         "ja": "x\u3092\u89e3\u304d\u306a\u3055\u3044: 2x\u00b2 - 8 = 0", "ko": "x\ub97c \ud480\uc5b4\ub77c: 2x\u00b2 - 8 = 0",
         "sw": "Tatua x: 2x\u00b2 - 8 = 0", "answer": 2, "category": "algebra"},
        {"en": "Solve: |2x - 5| = 3", "zh": "\u89e3\u65b9\u7a0b\uff1a|2x - 5| = 3",
         "es": "Resuelve: |2x - 5| = 3", "ar": "\u062d\u0644: |2x - 5| = 3",
         "ja": "\u89e3\u304d\u306a\u3055\u3044: |2x - 5| = 3", "ko": "\ud480\uc5b4\ub77c: |2x - 5| = 3",
         "sw": "Tatua: |2x - 5| = 3", "answer": 4, "category": "algebra"},
        {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97\uff1a347 + 658",
         "es": "Calcula: 347 + 658", "ar": "\u0627\u062d\u0633\u0628: 347 + 658",
         "ja": "\u8a08\u7b97: 347 + 658", "ko": "\uacc4\uc0b0: 347 + 658",
         "sw": "Hesabu: 347 + 658", "answer": 1005, "category": "arithmetic"},
        {"en": "Calculate: 1000 - 387", "zh": "\u8ba1\u7b97\uff1a1000 - 387",
         "es": "Calcula: 1000 - 387", "ar": "\u0627\u062d\u0633\u0628: 1000 - 387",
         "ja": "\u8a08\u7b97: 1000 - 387", "ko": "\uacc4\uc0b0: 1000 - 387",
         "sw": "Hesabu: 1000 - 387", "answer": 613, "category": "arithmetic"},
        {"en": "Calculate: 23 \u00d7 17", "zh": "\u8ba1\u7b97\uff1a23 \u00d7 17",
         "es": "Calcula: 23 \u00d7 17", "ar": "\u0627\u062d\u0633\u0628: 23 \u00d7 17",
         "ja": "\u8a08\u7b97: 23 \u00d7 17", "ko": "\uacc4\uc0b0: 23 \u00d7 17",
         "sw": "Hesabu: 23 \u00d7 17", "answer": 391, "category": "arithmetic"},
        {"en": "Calculate: 1728 \u00f7 12", "zh": "\u8ba1\u7b97\uff1a1728 \u00f7 12",
         "es": "Calcula: 1728 \u00f7 12", "ar": "\u0627\u062d\u0633\u0628: 1728 \u00f7 12",
         "ja": "\u8a08\u7b97: 1728 \u00f7 12", "ko": "\uacc4\uc0b0: 1728 \u00f7 12",
         "sw": "Hesabu: 1728 \u00f7 12", "answer": 144, "category": "arithmetic"},
        {"en": "Find the area of a circle with radius 7 (use \u03c0 \u2248 22/7)",
         "zh": "\u6c42\u534a\u5f84\u4e3a7\u7684\u5706\u7684\u9762\u79ef\uff08\u4f7f\u7528\u03c0 \u2248 22/7\uff09",
         "es": "Halla el \u00e1rea de un c\u00edrculo con radio 7 (usa \u03c0 \u2248 22/7)",
         "ar": "\u0623\u0648\u062c\u062f \u0645\u0633\u0627\u062d\u0629 \u062f\u0627\u0626\u0631\u0629 \u0646\u0635\u0641 \u0642\u0637\u0631\u0647\u0627 7 (\u0627\u0633\u062a\u062e\u062f\u0645 \u03c0 \u2248 22/7)",
         "ja": "\u534a\u5f847\u306e\u5186\u306e\u9762\u7a4d\u3092\u6c42\u3081\u3088(\u03c0 \u2248 22/7)",
         "ko": "\ubc18\uc9c0\ub984 7\uc778 \uc6d0\uc758 \ub113\uc774\ub97c \uad6c\ud558\ub77c (\u03c0 \u2248 22/7)",
         "sw": "Tafuta eneo la mduara wenye radius 7 (tumia \u03c0 \u2248 22/7)",
         "answer": 154, "category": "geometry"},
        {"en": "Find the hypotenuse of a right triangle with legs 5 and 12",
         "zh": "\u6c42\u76f4\u89d2\u4e09\u89d2\u5f62\u7684\u659c\u8fb9\uff0c\u4e24\u76f4\u89d2\u8fb9\u5206\u522b\u4e3a5\u548c12",
         "es": "Halla la hipotenusa de un tri\u00e1ngulo rect\u00e1ngulo con catetos 5 y 12",
         "ar": "\u0623\u0648\u062c\u062f \u0648\u062a\u0631 \u0645\u062b\u0644\u062b \u0642\u0627\u0626\u0645 \u0627\u0644\u0632\u0627\u0648\u064a\u0629 \u0636\u0644\u0639\u0627\u0647 5 \u06488 12",
         "ja": "\u8db3\u304c5\u306812\u306e\u76f4\u89d2\u4e09\u89d2\u5f62\u306e\u659c\u8fba\u3092\u6c42\u3081\u3088",
         "ko": "\ub450 \ubcc0\uc774 5\uc640 12\uc778 \uc9c1\uac01\uc0bc\uac01\ud615\uc758 \ube57\ubcc0\uc744 \uad6c\ud558\ub77c",
         "sw": "Tafuta hypotenuse ya pembetatu ya kulia yenye miguu 5 na 12",
         "answer": 13, "category": "geometry"},
        {"en": "What is the perimeter of a rectangle with length 15 and width 8?",
         "zh": "\u957f15\u5bbd8\u7684\u77e9\u5f62\u7684\u5468\u957f\u662f\u591a\u5c11\uff1f",
         "es": "\u00bfCu\u00e1l es el per\u00edmetro de un rect\u00e1ngulo de largo 15 y ancho 8?",
         "ar": "\u0645\u0627 \u0645\u062d\u064a\u0637 \u0645\u0633\u062a\u0637\u064a\u0644 \u0637\u0648\u0644\u0647 15 \u0648\u0639\u0631\u0636\u0647 8\u061f",
         "ja": "\u7e2615\u3001\u5e458\u306e\u9577\u65b9\u5f62\u306e\u5468\u56f2\u306f\uff1f",
         "ko": "\uae38\uc774 15, \ub108\ube44 8\uc778 \uc9c1\uc0ac\uac01\ud615\uc758 \ub458\ub808\ub294?",
         "sw": "Mzunguko wa mstatili wenye urefu 15 na upana 8 ni nini?",
         "answer": 46, "category": "geometry"},
        {"en": "Find the volume of a cube with side length 6",
         "zh": "\u6c42\u8fb9\u957f\u4e3a6\u7684\u6b63\u65b9\u4f53\u7684\u4f53\u79ef",
         "es": "Halla el volumen de un cubo con lado 6",
         "ar": "\u0623\u0648\u062c\u062f \u062d\u062c\u0645 \u0645\u0643\u0639\u0628 \u0637\u0648\u0644 \u0636\u0644\u0639\u0647 6",
         "ja": "\u4e00\u8fba6\u306e\u7acb\u65b9\u4f53\u306e\u4f53\u7a4d\u3092\u6c42\u3081\u3088",
         "ko": "\ud55c \ubcc0\uc758 \uae38\uc774\uac00 6\uc778 \uc815\uc721\uba74\uc758 \ubd80\ud53c\ub97c \uad6c\ud558\ub77c",
         "sw": "Tafuta ujazo wa mchemraba wenye upande wa 6",
         "answer": 216, "category": "geometry"},
        {"en": "What is the GCD of 84 and 120?",
         "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570\u662f\u591a\u5c11\uff1f",
         "es": "\u00bfCu\u00e1l es el MCD de 84 y 120?",
         "ar": "\u0645\u0627 \u0627\u0644\u0642\u0627\u0633\u0645 \u0627\u0644\u0645\u0634\u062a\u0631\u0643 \u0627\u0644\u0623\u0643\u0628\u0631 \u0644\u0640 84 \u0648 120\u061f",
         "ja": "84\u3068120\u306e\u6700\u5927\u516c\u7d04\u6570\u306f\uff1f",
         "ko": "84\uc640 120\uc758 \ucd5c\ub300\uacf5\uc57d\uc218\ub294?",
         "sw": "GCD ya 84 na 120 ni nini?",
         "answer": 12, "category": "number_theory"},
        {"en": "Find the remainder when 2^10 is divided by 7",
         "zh": "\u6c422^10\u9664\u4ee57\u7684\u4f59\u6570",
         "es": "Encuentra el resto cuando 2^10 se divide por 7",
         "ar": "\u0623\u0648\u062c\u062f \u0627\u0644\u0628\u0627\u0642\u064a \u0639\u0646\u062f \u0642\u0633\u0645\u0629 2^10 \u0639\u0644\u0649 7",
         "ja": "2^10\u30927\u3067\u5272\u3063\u305f\u4f59\u308a\u3092\u6c42\u3081\u3088",
         "ko": "2^10\uc744 7\ub85c \ub098\ub208 \ub098\uba38\uc9c0\ub97c \uad6c\ud558\ub77c",
         "sw": "Tafuta salio 2^10 ikigawanywa na 7",
         "answer": 2, "category": "number_theory"},
        {"en": "What is the sum of all prime numbers less than 20?",
         "zh": "\u6c4220\u4ee5\u5185\u6240\u6709\u8d28\u6570\u7684\u548c",
         "es": "\u00bfCu\u00e1l es la suma de todos los primos menores que 20?",
         "ar": "\u0645\u0627 \u0645\u062c\u0645\u0648\u0639 \u062c\u0645\u064a\u0639 \u0627\u0644\u0623\u0639\u062f\u0627\u062f \u0627\u0644\u0623\u0648\u0644\u064a\u0629 \u0627\u0644\u0623\u0642\u0644 \u0645\u0646 20\u061f",
         "ja": "20\u672a\u6e80\u306e\u7d20\u6570\u306e\u548c\u306f\uff1f",
         "ko": "20 \ubbf8\ub9cc\uc758 \ubaa8\ub4e0 \uc18c\uc218\uc758 \ud569\uc740?",
         "sw": "Jumla ya namba zote za kwanza chini ya 20 ni nini?",
         "answer": 77, "category": "number_theory"},
        {"en": "How many ways can you choose 3 items from 7?",
         "zh": "\u4ece7\u4e2a\u7269\u54c1\u4e2d\u9009\u62e93\u4e2a\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f",
         "es": "\u00bfDe cu\u00e1ntas formas puedes elegir 3 de 7?",
         "ar": "\u0643\u0645 \u0639\u062f\u062f \u0637\u0631\u0642 \u0627\u062e\u062a\u064a\u0627\u0631 3 \u0639\u0646\u0627\u0635\u0631 \u0645\u0646 7\u061f",
         "ja": "7\u500b\u304b\u30893\u500b\u9078\u3076\u65b9\u6cd5\u306f\u4f55\u901a\u308a\uff1f",
         "ko": "7\uac1c\uc5d0\uc11c 3\uac1c\ub97c \uace0\ub974\ub294 \ubc29\ubc95\uc758 \uc218\ub294?",
         "sw": "Njia ngapi za kuchagua vitu 3 kutoka 7?",
         "answer": 35, "category": "combinatorics"},
        {"en": "How many ways can 5 people stand in a line?",
         "zh": "5\u4e2a\u4eba\u7ad9\u6210\u4e00\u6392\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f",
         "es": "\u00bfDe cu\u00e1ntas formas pueden 5 personas hacer fila?",
         "ar": "\u0643\u0645 \u0639\u062f\u062f \u0637\u0631\u0642 \u0648\u0642\u0648\u0641 5 \u0623\u0634\u062e\u0627\u0635 \u0641\u064a \u0635\u0641\u061f",
         "ja": "5\u4eba\u304c\u4e00\u5217\u306b\u4e26\u3076\u65b9\u6cd5\u306f\u4f55\u901a\u308a\uff1f",
         "ko": "5\uba85\uc774 \uc904\uc744 \uc11c\ub294 \ubc29\ubc95\uc758 \uc218\ub294?",
         "sw": "Njia ngapi za watu 5 kusimama kwenye mstari?",
         "answer": 120, "category": "combinatorics"},
        {"en": "Calculate: 8! / (5! \u00d7 3!)",
         "zh": "\u8ba1\u7b97\uff1a8! / (5! \u00d7 3!)",
         "es": "Calcula: 8! / (5! \u00d7 3!)",
         "ar": "\u0627\u062d\u0633\u0628: 8! / (5! \u00d7 3!)",
         "ja": "\u8a08\u7b97: 8! / (5! \u00d7 3!)",
         "ko": "\uacc4\uc0b0: 8! / (5! \u00d7 3!)",
         "sw": "Hesabu: 8! / (5! \u00d7 3!)",
         "answer": 56, "category": "combinatorics"},
        {"en": "How many 3-digit numbers have all distinct digits?",
         "zh": "\u6709\u591a\u5c11\u4e2a\u4e09\u4f4d\u6570\u7684\u5404\u4f4d\u6570\u5b57\u5747\u4e0d\u76f8\u540c\uff1f",
         "es": "\u00bfCu\u00e1ntos n\u00fameros de 3 cifras tienen todos sus d\u00edgitos distintos?",
         "ar": "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0623\u0631\u0642\u0627\u0645 \u0627\u0644\u0645\u0643\u0648\u0646\u0629 \u0645\u0646 3 \u0623\u0631\u0642\u0627\u0645 \u0628\u0623\u0631\u0642\u0627\u0645 \u0645\u062e\u062a\u0644\u0641\u0629\u061f",
         "ja": "\u5404\u4f4d\u306e\u6570\u5b57\u304c\u3059\u3079\u3066\u7570\u306a\u308b3\u6841\u306e\u6570\u306f\u3044\u304f\u3064\uff1f",
         "ko": "\ubaa8\ub4e0 \uc790\ub9bf\uc218\uac00 \ub2e4\ub978 3\uc790\ub9ac \uc218\ub294 \uba87 \uac1c?",
         "sw": "Nambari ngapi za tarakimu 3 zina tarakimu zote tofauti?",
         "answer": 648, "category": "combinatorics"},
    ]
    return problems


def extract_activations(model, tokenizer, problems, layers):
    """Extract last-token hidden states at specified layers for all problems in all languages."""
    langs = ['en', 'zh', 'es', 'ar', 'ja', 'ko', 'sw']
    acts = {L: [] for L in layers}
    labels = []  # (problem_idx, lang, answer, category)

    class Cap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[:, -1, :].detach().float().cpu().numpy()

    caps = {L: Cap() for L in layers}
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in layers]

    for pi, prob in enumerate(problems):
        for lang in langs:
            if lang not in prob:
                continue
            sys = SYS_PROMPTS.get(lang, SYS_PROMPTS['en'])
            messages = [{"role": "system", "content": sys}, {"role": "user", "content": prob[lang]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
            with torch.inference_mode():
                model(ids)
            for L in layers:
                acts[L].append(caps[L].out.squeeze())
            labels.append((pi, lang, prob['answer'], prob['category']))

    for h in hooks:
        h.remove()

    for L in layers:
        acts[L] = np.stack(acts[L])
    return acts, labels


def main():
    print('=' * 60)
    print('Exp Z-Encoder: The $1M Test')
    print('  Can a linear map from L30 predict answers cross-lingually?')
    print('=' * 60)

    # Layers to test
    test_layers = [5, 10, 13, 15, 18, 20, 25, 27, 30, 33, 35]

    print(f'\nLoading {MODEL_NAME}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print('  Loaded.', flush=True)

    problems = get_problems_7lang()
    print(f'  {len(problems)} problems \u00d7 7 langs = {len(problems)*7} activations', flush=True)

    # Extract activations
    print('\nExtracting activations...', flush=True)
    t0 = time.time()
    acts, labels = extract_activations(model, tokenizer, problems, test_layers)
    print(f'  Done in {time.time()-t0:.1f}s. Shape: {acts[test_layers[0]].shape}', flush=True)

    # Split: train on problems 0-9 in {en, zh, es}, test on problems 10-19 in ALL languages
    # Also test on problems 0-9 in UNSEEN languages {ar, ja, ko, sw}
    train_langs = {'en', 'zh', 'es'}
    test_langs_seen = {'en', 'zh', 'es'}
    test_langs_unseen = {'ar', 'ja', 'ko', 'sw'}
    train_problems = set(range(10))
    test_problems = set(range(10, len(problems)))

    train_idx = [i for i, (pi, lang, _, _) in enumerate(labels) if pi in train_problems and lang in train_langs]
    test_new_prob_idx = [i for i, (pi, lang, _, _) in enumerate(labels) if pi in test_problems]
    test_new_lang_idx = [i for i, (pi, lang, _, _) in enumerate(labels) if pi in train_problems and lang in test_langs_unseen]
    test_full_unseen_idx = [i for i, (pi, lang, _, _) in enumerate(labels) if pi in test_problems and lang in test_langs_unseen]

    print(f'\n  Train: {len(train_idx)} samples (problems 0-9, langs en/zh/es)')
    print(f'  Test new problems: {len(test_new_prob_idx)} (problems 10-{len(problems)-1}, all langs)')
    print(f'  Test new languages: {len(test_new_lang_idx)} (problems 0-9, ar/ja/ko/sw)')
    print(f'  Test fully unseen: {len(test_full_unseen_idx)} (problems 10+, unseen langs)', flush=True)

    results = {}

    for L in test_layers:
        X = acts[L]
        y_answer = np.array([labels[i][2] for i in range(len(labels))], dtype=float)
        y_problem = np.array([labels[i][0] for i in range(len(labels))])

        X_train = X[train_idx]
        y_train_ans = y_answer[train_idx]
        y_train_prob = y_problem[train_idx]

        # === TEST 1: Predict numerical answer (regression) ===
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train_ans)

        r2_results = {}
        for split_name, split_idx in [('train', train_idx), ('new_problems', test_new_prob_idx),
                                       ('new_languages', test_new_lang_idx), ('fully_unseen', test_full_unseen_idx)]:
            if len(split_idx) == 0:
                continue
            y_pred = reg.predict(X[split_idx])
            y_true = y_answer[split_idx]
            r2 = r2_score(y_true, y_pred)
            r2_results[split_name] = float(r2)

        # === TEST 2: Classify which problem (18-class classification) ===
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train_prob)

        acc_results = {}
        for split_name, split_idx in [('train', train_idx), ('new_languages', test_new_lang_idx)]:
            if len(split_idx) == 0:
                continue
            y_pred = clf.predict(X[split_idx])
            y_true = y_problem[split_idx]
            acc = accuracy_score(y_true, y_pred)
            acc_results[split_name] = float(acc)

        results[str(L)] = {
            'r2_answer': r2_results,
            'acc_problem_id': acc_results,
        }

        # Print key numbers
        r2_new = r2_results.get('new_problems', '?')
        r2_unseen = r2_results.get('fully_unseen', '?')
        acc_lang = acc_results.get('new_languages', '?')
        print(f'  L{L:2d}: R\u00b2(new_prob)={r2_new:>7.3f}  R\u00b2(unseen_lang+prob)={r2_unseen:>7.3f}  '
              f'acc(problem_id, new_lang)={acc_lang:>5.3f}' if isinstance(r2_new, float) else
              f'  L{L:2d}: R\u00b2={r2_results}  acc={acc_results}', flush=True)

    # === CROSS-LINGUAL RETRIEVAL TEST ===
    # For each test activation, find nearest neighbor in training set
    # If NN is the SAME problem in a different language, retrieval succeeds
    print(f'\n{"=" * 60}')
    print('CROSS-LINGUAL NEAREST-NEIGHBOR RETRIEVAL')
    print(f'{"=" * 60}')

    for L in [20, 25, 27, 30, 33]:
        X = acts[L]
        # Use new_languages test set: problems 0-9 in ar/ja/ko/sw
        correct = 0
        total = 0
        for idx in test_new_lang_idx:
            pi_test = labels[idx][0]
            # Find nearest training sample
            dists = np.linalg.norm(X[train_idx] - X[idx], axis=1)
            nn_idx = train_idx[np.argmin(dists)]
            pi_nn = labels[nn_idx][0]
            if pi_nn == pi_test:
                correct += 1
            total += 1
        retrieval_acc = correct / total if total > 0 else 0
        results[str(L)]['retrieval_acc'] = float(retrieval_acc)
        print(f'  L{L:2d}: {correct}/{total} = {retrieval_acc:.3f} (same problem, different language)', flush=True)

    # Save
    outpath = OUTPUT_DIR / 'exp_z_encoder.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
