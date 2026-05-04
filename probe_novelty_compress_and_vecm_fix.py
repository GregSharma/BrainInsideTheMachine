"""Two experiments, no cathedrals.

1. NOVELTY-WEIGHTED KV COMPRESSION: compress echo tokens more, preserve novel tokens.
   Use the seed score as the compression weight.

2. VECM WITH PROPER SAMPLE SIZE: 20 problems x 7 languages = 140 samples.
   PCA to 50D. Check if R^2 < 1 and if rank trajectory holds.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

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
    return [
        {"en": "Solve for x: 3x + 7 = 22", "zh": "\u89e3\u65b9\u7a0b\uff1a3x + 7 = 22", "es": "Resuelve: 3x + 7 = 22",
         "ar": "\u062d\u0644: 3x + 7 = 22", "ja": "x\u3092\u89e3\u3051: 3x + 7 = 22", "ko": "x\ub97c \ud480\uc5b4\ub77c: 3x + 7 = 22", "sw": "Tatua: 3x + 7 = 22"},
        {"en": "Solve for x: 2x\u00b2 - 8 = 0", "zh": "\u89e3: 2x\u00b2 - 8 = 0", "es": "Resuelve: 2x\u00b2 - 8 = 0",
         "ar": "\u062d\u0644: 2x\u00b2 - 8 = 0", "ja": "\u89e3\u3051: 2x\u00b2 - 8 = 0", "ko": "\ud480\uc5b4\ub77c: 2x\u00b2 - 8 = 0", "sw": "Tatua: 2x\u00b2 - 8 = 0"},
        {"en": "Solve: |2x - 5| = 3", "zh": "\u89e3: |2x - 5| = 3", "es": "Resuelve: |2x - 5| = 3",
         "ar": "\u062d\u0644: |2x - 5| = 3", "ja": "\u89e3\u3051: |2x - 5| = 3", "ko": "\ud480\uc5b4\ub77c: |2x - 5| = 3", "sw": "Tatua: |2x - 5| = 3"},
        {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97: 347 + 658", "es": "Calcula: 347 + 658",
         "ar": "\u0627\u062d\u0633\u0628: 347 + 658", "ja": "\u8a08\u7b97: 347 + 658", "ko": "\uacc4\uc0b0: 347 + 658", "sw": "Hesabu: 347 + 658"},
        {"en": "Calculate: 1000 - 387", "zh": "\u8ba1\u7b97: 1000 - 387", "es": "Calcula: 1000 - 387",
         "ar": "\u0627\u062d\u0633\u0628: 1000 - 387", "ja": "\u8a08\u7b97: 1000 - 387", "ko": "\uacc4\uc0b0: 1000 - 387", "sw": "Hesabu: 1000 - 387"},
        {"en": "Calculate: 23 \u00d7 17", "zh": "\u8ba1\u7b97: 23 \u00d7 17", "es": "Calcula: 23 \u00d7 17",
         "ar": "\u0627\u062d\u0633\u0628: 23 \u00d7 17", "ja": "\u8a08\u7b97: 23 \u00d7 17", "ko": "\uacc4\uc0b0: 23 \u00d7 17", "sw": "Hesabu: 23 \u00d7 17"},
        {"en": "Calculate: 1728 \u00f7 12", "zh": "\u8ba1\u7b97: 1728 \u00f7 12", "es": "Calcula: 1728 \u00f7 12",
         "ar": "\u0627\u062d\u0633\u0628: 1728 \u00f7 12", "ja": "\u8a08\u7b97: 1728 \u00f7 12", "ko": "\uacc4\uc0b0: 1728 \u00f7 12", "sw": "Hesabu: 1728 \u00f7 12"},
        {"en": "Area of circle with radius 7 (\u03c0\u224822/7)", "zh": "\u534a\u5f847\u7684\u5706\u9762\u79ef(\u03c0\u224822/7)", "es": "\u00c1rea c\u00edrculo radio 7",
         "ar": "\u0645\u0633\u0627\u062d\u0629 \u062f\u0627\u0626\u0631\u0629 \u0646\u0635\u0641 \u0642\u0637\u0631\u0647\u0627 7", "ja": "\u534a\u5f847\u306e\u5186\u306e\u9762\u79ef", "ko": "\ubc18\uc9c0\ub984 7\uc778 \uc6d0\uc758 \ub113\uc774", "sw": "Eneo la mduara radius 7"},
        {"en": "Hypotenuse with legs 5 and 12", "zh": "\u76f4\u89d2\u4e09\u89d2\u5f62\u659c\u8fb9 5\u548c12", "es": "Hipotenusa catetos 5 y 12",
         "ar": "\u0648\u062a\u0631 \u0636\u0644\u0639\u0627\u0647 5 \u0648 12", "ja": "\u8db3\u304c5\u306812\u306e\u659c\u8fba", "ko": "\ubcc0 5\uc640 12\uc758 \ube57\ubcc0", "sw": "Hypotenuse miguu 5 na 12"},
        {"en": "Perimeter of rectangle 15 by 8", "zh": "\u957f15\u5bbd8\u77e9\u5f62\u5468\u957f", "es": "Per\u00edmetro rect\u00e1ngulo 15x8",
         "ar": "\u0645\u062d\u064a\u0637 \u0645\u0633\u062a\u0637\u064a\u0644 15\u00d78", "ja": "\u7e2615\u5e458\u306e\u5468\u56f2", "ko": "\uae38\uc774 15 \ub108\ube44 8 \ub458\ub808", "sw": "Mzunguko mstatili 15x8"},
        {"en": "Volume of cube side 6", "zh": "\u8fb9\u957f6\u7684\u6b63\u65b9\u4f53\u4f53\u79ef", "es": "Volumen cubo lado 6",
         "ar": "\u062d\u062c\u0645 \u0645\u0643\u0639\u0628 \u0636\u0644\u0639 6", "ja": "\u4e00\u8fba6\u306e\u7acb\u65b9\u4f53\u306e\u4f53\u7a4d", "ko": "\ud55c \ubcc0 6 \uc815\uc721\uba74\uc758 \ubd80\ud53c", "sw": "Ujazo mchemraba upande 6"},
        {"en": "GCD of 84 and 120", "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570", "es": "MCD de 84 y 120",
         "ar": "\u0642\u0627\u0633\u0645 \u0645\u0634\u062a\u0631\u0643 \u0623\u0643\u0628\u0631 84 \u0648 120", "ja": "84\u3068120\u306e\u6700\u5927\u516c\u7d04\u6570", "ko": "84\uc640 120\uc758 \ucd5c\ub300\uacf5\uc57d\uc218", "sw": "GCD ya 84 na 120"},
        {"en": "Remainder 2^10 divided by 7", "zh": "2^10\u96647\u7684\u4f59\u6570", "es": "Resto 2^10 entre 7",
         "ar": "\u0628\u0627\u0642\u064a 2^10 \u0639\u0644\u0649 7", "ja": "2^10\u30927\u3067\u5272\u3063\u305f\u4f59\u308a", "ko": "2^10\uc744 7\ub85c \ub098\ub208 \ub098\uba38\uc9c0", "sw": "Salio 2^10 kwa 7"},
        {"en": "Sum of primes less than 20", "zh": "20\u4ee5\u5185\u8d28\u6570\u4e4b\u548c", "es": "Suma primos menores que 20",
         "ar": "\u0645\u062c\u0645\u0648\u0639 \u0623\u0648\u0644\u064a\u0629 \u0623\u0642\u0644 \u0645\u0646 20", "ja": "20\u672a\u6e80\u306e\u7d20\u6570\u306e\u548c", "ko": "20 \ubbf8\ub9cc \uc18c\uc218\uc758 \ud569", "sw": "Jumla namba za kwanza chini ya 20"},
        {"en": "Choose 3 from 7", "zh": "\u4ece7\u4e2a\u4e2d\u9009\u62e93\u4e2a", "es": "Elegir 3 de 7",
         "ar": "\u0627\u062e\u062a\u064a\u0627\u0631 3 \u0645\u0646 7", "ja": "7\u500b\u304b\u30893\u500b\u9078\u3076", "ko": "7\uac1c\uc5d0\uc11c 3\uac1c \uace0\ub974\uae30", "sw": "Chagua 3 kutoka 7"},
        {"en": "5 people in a line", "zh": "5\u4eba\u7ad9\u6210\u4e00\u6392", "es": "5 personas en fila",
         "ar": "5 \u0623\u0634\u062e\u0627\u0635 \u0641\u064a \u0635\u0641", "ja": "5\u4eba\u304c\u4e00\u5217\u306b\u4e26\u3076", "ko": "5\uba85\uc774 \uc904 \uc11c\uae30", "sw": "Watu 5 kwenye mstari"},
        {"en": "8!/(5!\u00d73!)", "zh": "8!/(5!\u00d73!)", "es": "8!/(5!\u00d73!)",
         "ar": "8!/(5!\u00d73!)", "ja": "8!/(5!\u00d73!)", "ko": "8!/(5!\u00d73!)", "sw": "8!/(5!\u00d73!)"},
        {"en": "3-digit numbers all distinct digits", "zh": "\u4e09\u4f4d\u6570\u5404\u4f4d\u4e0d\u540c", "es": "N\u00fameros 3 cifras distintas",
         "ar": "\u0623\u0631\u0642\u0627\u0645 3 \u062e\u0627\u0646\u0627\u062a \u0645\u062e\u062a\u0644\u0641\u0629", "ja": "\u5404\u4f4d\u7570\u306a\u308b3\u6841\u306e\u6570", "ko": "\uc790\ub9bf\uc218 \ub2e4\ub978 3\uc790\ub9ac \uc218", "sw": "Nambari 3 tarakimu tofauti"},
    ]

def main():
    import warnings; warnings.filterwarnings('ignore')
    print('loading...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # === PART 2: VECM WITH PROPER SAMPLES ===
    print(f'\n{"="*70}')
    print('VECM WITH 18 PROBLEMS x 7 LANGUAGES = 126 SAMPLES')
    print(f'{"="*70}')

    problems = get_problems_7lang()
    langs = list(SYS_PROMPTS.keys())
    ALL_LAYERS = list(range(36))

    class HCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            self.out = h[0, -1].detach().float().cpu().numpy()

    caps = {L: HCap() for L in ALL_LAYERS}
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in ALL_LAYERS]

    trajectories = []; labels = []
    for pi, prob in enumerate(problems):
        for lang in langs:
            if lang not in prob: continue
            sys = SYS_PROMPTS[lang]
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": prob[lang]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors='pt').input_ids.to(DEV)
            with torch.inference_mode(): model(ids)
            traj = np.stack([caps[L].out for L in ALL_LAYERS])
            trajectories.append(traj); labels.append((pi, lang))
        if (pi+1) % 5 == 0:
            print(f'  {pi+1}/{len(problems)} problems encoded', flush=True)

    for h in hooks: h.remove()

    trajectories = np.stack(trajectories)  # (n, 36, 2048)
    n = len(trajectories)
    print(f'  {n} trajectories collected', flush=True)

    # PCA
    K_PCA = 50
    all_h = trajectories.reshape(-1, 2048)
    pca = PCA(n_components=K_PCA)
    traj_r = pca.fit_transform(all_h).reshape(n, 36, K_PCA)
    print(f'  PCA {K_PCA}D, var={pca.explained_variance_ratio_.sum():.4f}', flush=True)

    # VECM with TRAIN/TEST split
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    print(f'  train: {n_train}, test: {n - n_train}', flush=True)

    print(f'\n  {"layer":>5s} {"rank_90":>8s} {"R2_train":>10s} {"R2_test":>10s} {"||Pi||":>8s} {"sv1":>8s} {"sv5":>8s} {"sv9":>8s}  zone')

    for L in range(35):
        H_L_train = traj_r[train_idx, L, :]
        delta_train = traj_r[train_idx, L+1, :] - H_L_train
        H_L_test = traj_r[test_idx, L, :]
        delta_test = traj_r[test_idx, L+1, :] - H_L_test

        # Fit on train
        Pi_T, _, _, _ = np.linalg.lstsq(H_L_train, delta_train, rcond=None)
        Pi = Pi_T.T

        # R^2 on train
        pred_train = H_L_train @ Pi_T
        ss_res_tr = np.sum((delta_train - pred_train)**2)
        ss_tot_tr = np.sum((delta_train - delta_train.mean(axis=0))**2)
        r2_train = 1 - ss_res_tr / (ss_tot_tr + 1e-10)

        # R^2 on TEST (the real number)
        pred_test = H_L_test @ Pi_T
        ss_res_te = np.sum((delta_test - pred_test)**2)
        ss_tot_te = np.sum((delta_test - delta_test.mean(axis=0))**2)
        r2_test = 1 - ss_res_te / (ss_tot_te + 1e-10)

        # SVD of Pi
        _, S, _ = np.linalg.svd(Pi)
        cumvar = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
        rank_90 = int(np.searchsorted(cumvar, 0.9) + 1)

        zone = ''
        if L < 9: zone = 'early'
        elif L < 18: zone = 'adversarial'
        elif L < 27: zone = 'cooperative'
        elif L < 33: zone = 'canyon'
        else: zone = 'readout'

        sv5 = S[4] if len(S) > 4 else 0
        sv9 = S[8] if len(S) > 8 else 0

        print(f'  L{L:>2d}->{L+1:<2d} {rank_90:>8d} {r2_train:>10.4f} {r2_test:>10.4f} '
              f'{np.linalg.norm(Pi, "fro"):>8.2f} {S[0]:>8.4f} {sv5:>8.4f} {sv9:>8.4f}  {zone}')

    # Summary
    print(f'\n  R\u00b2 test trajectory:')
    r2_tests = []
    for L in range(35):
        H_L_train = traj_r[train_idx, L, :]
        delta_train = traj_r[train_idx, L+1, :] - H_L_train
        H_L_test = traj_r[test_idx, L, :]
        delta_test = traj_r[test_idx, L+1, :] - H_L_test
        Pi_T, _, _, _ = np.linalg.lstsq(H_L_train, delta_train, rcond=None)
        pred_test = H_L_test @ Pi_T
        ss_res = np.sum((delta_test - pred_test)**2)
        ss_tot = np.sum((delta_test - delta_test.mean(axis=0))**2)
        r2_tests.append(1 - ss_res / (ss_tot + 1e-10))

    zones = {'early': r2_tests[0:9], 'adversarial': r2_tests[9:18],
             'cooperative': r2_tests[18:27], 'canyon': r2_tests[27:33], 'readout': r2_tests[33:35]}
    for zone, vals in zones.items():
        print(f'  {zone:>15s}: mean={np.mean(vals):.4f}, range=[{np.min(vals):.4f}, {np.max(vals):.4f}]')


if __name__ == '__main__':
    main()
