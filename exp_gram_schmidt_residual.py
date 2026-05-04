"""exp_gram_schmidt_residual: Training-free Z-encoder via orthogonal projection.

The idea (Greg, 2026-04-25): At each layer, project h_last onto the
complement of span(context_token_hidden_states). The residual is what
the model ADDED beyond what the input said. Should be language-invariant
by construction because different languages produce different input
subspaces but the same computation residual.

Key controls informed by prior findings:
- Raw h_last cosines (baseline)
- Mean-centered h_last cosines (since centered Gram >> raw Gram, centering matters)
- Random subspace projection of same rank (is it specific to context tokens?)
- Shuffled context tokens from different problem (does it need the CORRECT input?)

Key layers informed by evidence chain:
- L5, L10: pre-convention-invariant, convention still in activations
- L13: convention boundary l_c
- L18: cooperative zone start (adversarial→cooperative transition)
- L22: mid-cooperative
- L26: cooperative zone exit
- L30: rank-1 MLP bottleneck
- L33: classifier peak (100% cross-lingual accuracy on raw)

What we expect given prior findings:
- C6b: attn at last token = constant bias. MLP does the work.
- L14-L29 convention-invariant (cos>0.93), so GS at L26 may not help much.
- L5-L13 still has convention, so GS there should remove more language.
- Random control should be WORSE than real context (or gram-schmidt is just denoising).
"""
import json, time, sys
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

LANGS = ['en', 'zh', 'es', 'ar', 'ja', 'ko', 'sw']
TEST_LAYERS = [5, 10, 13, 18, 22, 26, 30, 33]


def get_problems():
    """Same 20 problems from z_encoder."""
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
         "ar": "\u0623\u0648\u062c\u062f \u0648\u062a\u0631 \u0645\u062b\u0644\u062b \u0642\u0627\u0626\u0645 \u0627\u0644\u0632\u0627\u0648\u064a\u0629 \u0636\u0644\u0639\u0627\u0647 5 \u0648 12",
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


def project_out_subspace(h, basis_matrix, threshold=1e-6):
    """Project h onto the complement of span(basis_matrix).

    Uses SVD for numerical stability (better than Gram-Schmidt).
    basis_matrix: (n_tokens, d_model) — the context token hidden states.
    h: (d_model,) — the last token hidden state.

    Returns: residual (d_model,), effective_rank (int), projection (d_model,)
    """
    # Center the basis (remove mean — this separates mean-removal from subspace projection)
    # Actually NO: don't center. We want the raw span of the context token states.
    # Centering would change the subspace. The mean IS part of the input.

    U, S, Vt = np.linalg.svd(basis_matrix, full_matrices=False)
    # Keep singular vectors with non-negligible singular values
    mask = S > threshold * S[0]
    effective_rank = int(mask.sum())
    V_keep = Vt[mask]  # (rank, d_model)

    # Project h onto span(V_keep)
    coeffs = V_keep @ h  # (rank,)
    projection = coeffs @ V_keep  # (d_model,)
    residual = h - projection

    return residual, effective_rank, projection


def extract_all_token_activations(model, tokenizer, problems, layers):
    """Extract hidden states for ALL tokens at specified layers.

    Returns:
        data: list of dicts, one per (problem, lang) pair:
            {'problem_idx': int, 'lang': str, 'answer': int, 'category': str,
             'h_last': {L: array(d,)}, 'h_context': {L: array(n_ctx, d)},
             'n_tokens': int}
    """
    data = []

    class AllTokenCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[0].detach().float().cpu().numpy()  # (seq_len, d_model)

    caps = {L: AllTokenCap() for L in layers}
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in layers]

    total = len(problems) * len(LANGS)
    done = 0

    for pi, prob in enumerate(problems):
        for lang in LANGS:
            if lang not in prob:
                continue
            sys = SYS_PROMPTS.get(lang, SYS_PROMPTS['en'])
            messages = [{"role": "system", "content": sys}, {"role": "user", "content": prob[lang]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
            n_tokens = ids.shape[1]

            with torch.inference_mode():
                model(ids)

            entry = {
                'problem_idx': pi,
                'lang': lang,
                'answer': prob['answer'],
                'category': prob['category'],
                'n_tokens': n_tokens,
                'h_last': {},
                'h_context': {},
            }
            for L in layers:
                all_h = caps[L].out  # (seq_len, d_model)
                entry['h_last'][L] = all_h[-1].copy()   # last token
                entry['h_context'][L] = all_h[:-1].copy()  # all context tokens

            data.append(entry)
            done += 1
            if done % 14 == 0:
                print(f'  extracted {done}/{total}', flush=True)

    for h in hooks:
        h.remove()

    return data


def compute_cosine_matrix(vectors):
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = vectors / norms
    return normed @ normed.T


def analyze_cosines(cos_matrix, problem_indices, lang_indices):
    """Compute mean cosines for same-problem-diff-lang and diff-problem-same-lang."""
    n = len(problem_indices)
    same_prob_diff_lang = []
    diff_prob_same_lang = []
    diff_prob_diff_lang = []

    for i in range(n):
        for j in range(i+1, n):
            same_p = (problem_indices[i] == problem_indices[j])
            same_l = (lang_indices[i] == lang_indices[j])
            c = cos_matrix[i, j]
            if same_p and not same_l:
                same_prob_diff_lang.append(c)
            elif not same_p and same_l:
                diff_prob_same_lang.append(c)
            elif not same_p and not same_l:
                diff_prob_diff_lang.append(c)

    return {
        'same_prob_diff_lang': float(np.mean(same_prob_diff_lang)) if same_prob_diff_lang else None,
        'diff_prob_same_lang': float(np.mean(diff_prob_same_lang)) if diff_prob_same_lang else None,
        'diff_prob_diff_lang': float(np.mean(diff_prob_diff_lang)) if diff_prob_diff_lang else None,
        'n_same_prob_diff_lang': len(same_prob_diff_lang),
        'n_diff_prob_same_lang': len(diff_prob_same_lang),
    }


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('=' * 70)
    print('EXP GRAM-SCHMIDT RESIDUAL: Training-free Z-encoder')
    print('  project h_last onto complement of context token subspace')
    print('  the residual = what the model ADDED beyond the input')
    print('=' * 70)

    print(f'\nLoading {MODEL_NAME}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print('  loaded.', flush=True)

    problems = get_problems()
    n_problems = len(problems)
    print(f'  {n_problems} problems x {len(LANGS)} langs = {n_problems * len(LANGS)} samples', flush=True)

    # === EXTRACT ALL-TOKEN ACTIVATIONS ===
    print('\nextracting all-token activations...', flush=True)
    t0 = time.time()
    data = extract_all_token_activations(model, tokenizer, problems, TEST_LAYERS)
    print(f'  done in {time.time()-t0:.1f}s. {len(data)} samples.', flush=True)

    # Free model
    del model
    torch.cuda.empty_cache()

    problem_indices = np.array([d['problem_idx'] for d in data])
    lang_indices = np.array([LANGS.index(d['lang']) for d in data])

    results = {}

    for L in TEST_LAYERS:
        print(f'\n{"=" * 70}')
        print(f'LAYER {L}')
        print(f'{"=" * 70}')

        # === 1. RAW h_last cosines ===
        raw_vecs = np.stack([d['h_last'][L] for d in data])
        raw_cos = compute_cosine_matrix(raw_vecs)
        raw_stats = analyze_cosines(raw_cos, problem_indices, lang_indices)

        # === 2. MEAN-CENTERED h_last cosines ===
        centered_vecs = raw_vecs - raw_vecs.mean(axis=0, keepdims=True)
        centered_cos = compute_cosine_matrix(centered_vecs)
        centered_stats = analyze_cosines(centered_cos, problem_indices, lang_indices)

        # === 3. GRAM-SCHMIDT RESIDUAL cosines ===
        residuals = []
        eff_ranks = []
        proj_fracs = []  # fraction of h_last that was projected out

        for d in data:
            h_last = d['h_last'][L]
            h_ctx = d['h_context'][L]
            residual, eff_rank, projection = project_out_subspace(h_last, h_ctx)
            residuals.append(residual)
            eff_ranks.append(eff_rank)
            proj_frac = np.linalg.norm(projection) / (np.linalg.norm(h_last) + 1e-10)
            proj_fracs.append(proj_frac)

        residual_vecs = np.stack(residuals)
        gs_cos = compute_cosine_matrix(residual_vecs)
        gs_stats = analyze_cosines(gs_cos, problem_indices, lang_indices)

        mean_eff_rank = float(np.mean(eff_ranks))
        mean_proj_frac = float(np.mean(proj_fracs))

        # === 4. RANDOM SUBSPACE CONTROL ===
        # Project out a random subspace of the same average rank
        rng = np.random.RandomState(42)
        k_rand = int(round(mean_eff_rank))
        random_basis = rng.randn(k_rand, D_MODEL)
        random_basis, _ = np.linalg.qr(random_basis.T)  # orthonormalize
        random_basis = random_basis[:, :k_rand].T  # (k_rand, d_model)

        random_residuals = []
        for d in data:
            h = d['h_last'][L]
            coeffs = random_basis @ h
            proj = coeffs @ random_basis
            random_residuals.append(h - proj)

        random_vecs = np.stack(random_residuals)
        rand_cos = compute_cosine_matrix(random_vecs)
        rand_stats = analyze_cosines(rand_cos, problem_indices, lang_indices)

        # === 5. SHUFFLED CONTEXT CONTROL ===
        # Use context tokens from a DIFFERENT problem (same language)
        shuffled_residuals = []
        for i, d in enumerate(data):
            # Find another problem in the same language
            lang = d['lang']
            pi = d['problem_idx']
            candidates = [j for j, d2 in enumerate(data)
                         if d2['lang'] == lang and d2['problem_idx'] != pi]
            if candidates:
                donor_idx = candidates[rng.randint(len(candidates))]
                donor_ctx = data[donor_idx]['h_context'][L]
            else:
                donor_ctx = d['h_context'][L]  # fallback

            h_last = d['h_last'][L]
            residual, _, _ = project_out_subspace(h_last, donor_ctx)
            shuffled_residuals.append(residual)

        shuffled_vecs = np.stack(shuffled_residuals)
        shuf_cos = compute_cosine_matrix(shuffled_vecs)
        shuf_stats = analyze_cosines(shuf_cos, problem_indices, lang_indices)

        # === 6. SVD OF RESIDUALS — dimensionality ===
        residual_centered = residual_vecs - residual_vecs.mean(axis=0, keepdims=True)
        _, S_res, _ = np.linalg.svd(residual_centered, full_matrices=False)
        cumvar = np.cumsum(S_res**2) / np.sum(S_res**2)
        r50 = int(np.searchsorted(cumvar, 0.5) + 1)
        r90 = int(np.searchsorted(cumvar, 0.9) + 1)
        r99 = int(np.searchsorted(cumvar, 0.99) + 1)
        top1_frac = float(S_res[0]**2 / np.sum(S_res**2))

        # === 7. CLASSIFIER ON RESIDUALS ===
        # Train on problems 0-9 in en/zh/es, test on problems 0-9 in ar/ja/ko/sw
        train_langs = {'en', 'zh', 'es'}
        train_probs = set(range(10))

        train_idx = [i for i, d in enumerate(data)
                    if d['problem_idx'] in train_probs and d['lang'] in train_langs]
        test_idx = [i for i, d in enumerate(data)
                   if d['problem_idx'] in train_probs and d['lang'] not in train_langs]

        if len(train_idx) >= 10 and len(test_idx) >= 10:
            X_train = residual_vecs[train_idx]
            y_train = problem_indices[train_idx]
            X_test = residual_vecs[test_idx]
            y_test = problem_indices[test_idx]

            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X_train, y_train)
            acc_train = float(accuracy_score(y_train, clf.predict(X_train)))
            acc_test = float(accuracy_score(y_test, clf.predict(X_test)))

            # Also on raw for comparison
            clf_raw = LogisticRegression(max_iter=2000, C=1.0)
            clf_raw.fit(raw_vecs[train_idx], y_train)
            acc_raw_test = float(accuracy_score(y_test, clf_raw.predict(raw_vecs[test_idx])))
        else:
            acc_train = acc_test = acc_raw_test = None

        # === REPORT ===
        # The key metric: SEPARATION = same_prob_diff_lang - diff_prob_same_lang
        # Higher = better separation of problem identity from language
        sep_raw = (raw_stats['same_prob_diff_lang'] or 0) - (raw_stats['diff_prob_same_lang'] or 0)
        sep_gs = (gs_stats['same_prob_diff_lang'] or 0) - (gs_stats['diff_prob_same_lang'] or 0)
        sep_centered = (centered_stats['same_prob_diff_lang'] or 0) - (centered_stats['diff_prob_same_lang'] or 0)
        sep_rand = (rand_stats['same_prob_diff_lang'] or 0) - (rand_stats['diff_prob_same_lang'] or 0)
        sep_shuf = (shuf_stats['same_prob_diff_lang'] or 0) - (shuf_stats['diff_prob_same_lang'] or 0)

        print(f'  context token effective rank: {mean_eff_rank:.1f} / {D_MODEL}')
        print(f'  projection fraction (|proj|/|h|): {mean_proj_frac:.4f}')
        print(f'  residual SVD: r50={r50}, r90={r90}, r99={r99}, top1={top1_frac:.4f}')
        print(f'')
        print(f'  {"condition":<20s} {"same_p_diff_l":>14s} {"diff_p_same_l":>14s} {"separation":>12s}')
        print(f'  {"-"*20} {"-"*14} {"-"*14} {"-"*12}')
        print(f'  {"raw":<20s} {raw_stats["same_prob_diff_lang"]:>14.4f} {raw_stats["diff_prob_same_lang"]:>14.4f} {sep_raw:>12.4f}')
        print(f'  {"centered":<20s} {centered_stats["same_prob_diff_lang"]:>14.4f} {centered_stats["diff_prob_same_lang"]:>14.4f} {sep_centered:>12.4f}')
        print(f'  {"gram-schmidt":<20s} {gs_stats["same_prob_diff_lang"]:>14.4f} {gs_stats["diff_prob_same_lang"]:>14.4f} {sep_gs:>12.4f}')
        print(f'  {"random subspace":<20s} {rand_stats["same_prob_diff_lang"]:>14.4f} {rand_stats["diff_prob_same_lang"]:>14.4f} {sep_rand:>12.4f}')
        print(f'  {"shuffled context":<20s} {shuf_stats["same_prob_diff_lang"]:>14.4f} {shuf_stats["diff_prob_same_lang"]:>14.4f} {sep_shuf:>12.4f}')

        if acc_test is not None:
            print(f'')
            print(f'  classifier (unseen langs): raw={acc_raw_test:.3f}  residual={acc_test:.3f}')

        results[str(L)] = {
            'effective_rank': mean_eff_rank,
            'proj_fraction': mean_proj_frac,
            'residual_svd': {'r50': r50, 'r90': r90, 'r99': r99, 'top1_frac': top1_frac},
            'raw': raw_stats,
            'centered': centered_stats,
            'gram_schmidt': gs_stats,
            'random_control': rand_stats,
            'shuffled_control': shuf_stats,
            'separation': {
                'raw': float(sep_raw),
                'centered': float(sep_centered),
                'gram_schmidt': float(sep_gs),
                'random': float(sep_rand),
                'shuffled': float(sep_shuf),
            },
            'classifier': {
                'raw_test_acc': acc_raw_test,
                'residual_train_acc': acc_train,
                'residual_test_acc': acc_test,
            },
        }
        sys.stdout.flush()

    # === SUMMARY ===
    print(f'\n{"=" * 70}')
    print('SUMMARY: gram-schmidt separation gain over raw')
    print(f'{"=" * 70}')
    print(f'  {"layer":>5s} {"raw_sep":>10s} {"gs_sep":>10s} {"delta":>10s} {"rand_sep":>10s} {"clf_raw":>10s} {"clf_gs":>10s}')
    for L in TEST_LAYERS:
        r = results[str(L)]
        delta = r['separation']['gram_schmidt'] - r['separation']['raw']
        clf_r = r['classifier']['raw_test_acc'] or 0
        clf_g = r['classifier']['residual_test_acc'] or 0
        print(f'  L{L:>3d} {r["separation"]["raw"]:>10.4f} {r["separation"]["gram_schmidt"]:>10.4f} '
              f'{delta:>+10.4f} {r["separation"]["random"]:>10.4f} {clf_r:>10.3f} {clf_g:>10.3f}')

    # Save
    outpath = OUTPUT_DIR / 'exp_gram_schmidt_residual.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved to {outpath}')


if __name__ == '__main__':
    main()
