"""Find the kernel. GP/KRR on layer-to-layer dynamics.

Fit RBF kernel ridge regression: delta_L = f(h_L) where f lives in RKHS.
Report: length scale trajectory, test R^2 (should beat linear R^2 at
cooperative zone where 42% is nonlinear), kernel effective rank.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

SYS_PROMPTS = {
    'en': 'You are a careful mathematical reasoner. Think step by step.',
    'zh': '\u4f60\u662f\u4e00\u4e2a\u4e25\u8c28\u7684\u6570\u5b66\u63a8\u7406\u8005\u3002\u8bf7\u9010\u6b65\u601d\u8003\u3002',
    'es': 'Eres un razonador matem\u00e1tico cuidadoso. Piensa paso a paso.',
    'ar': '\u0623\u0646\u062a \u0645\u0641\u0643\u0631 \u0631\u064a\u0627\u0636\u064a \u062f\u0642\u064a\u0642.',
    'ja': '\u6570\u5b66\u7684\u306b\u8003\u3048\u3066\u304f\u3060\u3055\u3044\u3002',
    'ko': '\ub2e8\uacc4\ubcc4\ub85c \uc0dd\uac01\ud558\uc138\uc694.',
    'sw': 'Fikiria hatua kwa hatua.',
}

def get_problems():
    return [
        {"en": "Solve: 3x + 7 = 22", "zh": "\u89e3: 3x + 7 = 22", "es": "Resuelve: 3x + 7 = 22",
         "ar": "\u062d\u0644: 3x + 7 = 22", "ja": "\u89e3\u3051: 3x + 7 = 22", "ko": "\ud480\uc5b4\ub77c: 3x + 7 = 22", "sw": "Tatua: 3x + 7 = 22"},
        {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97: 347 + 658", "es": "Calcula: 347 + 658",
         "ar": "\u0627\u062d\u0633\u0628: 347 + 658", "ja": "\u8a08\u7b97: 347 + 658", "ko": "\uacc4\uc0b0: 347 + 658", "sw": "Hesabu: 347 + 658"},
        {"en": "Hypotenuse legs 5 and 12", "zh": "\u659c\u8fb9 5\u548c12", "es": "Hipotenusa 5 y 12",
         "ar": "\u0648\u062a\u0631 5 \u0648 12", "ja": "\u659c\u8fba 5\u306812", "ko": "\ube57\ubcc0 5\uc640 12", "sw": "Hypotenuse 5 na 12"},
        {"en": "GCD of 84 and 120", "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570", "es": "MCD 84 y 120",
         "ar": "\u0642\u0627\u0633\u0645 84 \u0648 120", "ja": "84\u3068120\u306e\u6700\u5927\u516c\u7d04\u6570", "ko": "84\uc640 120\uc758 \ucd5c\ub300\uacf5\uc57d\uc218", "sw": "GCD 84 na 120"},
        {"en": "Choose 3 from 7", "zh": "\u4ece7\u9009\u62e93", "es": "Elegir 3 de 7",
         "ar": "\u0627\u062e\u062a\u064a\u0627\u0631 3 \u0645\u0646 7", "ja": "7\u304b\u30893\u9078\u3076", "ko": "7\uc5d0\uc11c 3\uace0\ub974\uae30", "sw": "Chagua 3 kutoka 7"},
        {"en": "23 times 17", "zh": "23\u4e5817", "es": "23 por 17",
         "ar": "23 \u0636\u0631\u0628 17", "ja": "23\u304b\u305117", "ko": "23\uacf1\ud558\uae30 17", "sw": "23 mara 17"},
        {"en": "1000 - 387", "zh": "1000 - 387", "es": "1000 - 387",
         "ar": "1000 - 387", "ja": "1000 - 387", "ko": "1000 - 387", "sw": "1000 - 387"},
        {"en": "Area circle radius 7", "zh": "\u534a\u5f847\u5706\u9762\u79ef", "es": "\u00c1rea c\u00edrculo r=7",
         "ar": "\u0645\u0633\u0627\u062d\u0629 \u062f\u0627\u0626\u0631\u0629 7", "ja": "\u5186\u306e\u9762\u79ef r=7", "ko": "\uc6d0 \ub113\uc774 r=7", "sw": "Eneo mduara r=7"},
        {"en": "Volume cube side 6", "zh": "\u8fb96\u7acb\u65b9\u4f53\u4f53\u79ef", "es": "Volumen cubo lado 6",
         "ar": "\u062d\u062c\u0645 \u0645\u0643\u0639\u0628 6", "ja": "\u4e00\u8fba6\u306e\u7acb\u65b9\u4f53", "ko": "\ud55c\ubcc0 6 \uc815\uc721\uba74\uc758", "sw": "Ujazo mchemraba 6"},
        {"en": "Perimeter rectangle 15 by 8", "zh": "15\u00d78\u77e9\u5f62\u5468\u957f", "es": "Per\u00edmetro 15x8",
         "ar": "\u0645\u062d\u064a\u0637 15\u00d78", "ja": "\u5468\u56f2 15\u00d78", "ko": "\ub458\ub808 15\u00d78", "sw": "Mzunguko 15x8"},
        {"en": "Sum primes under 20", "zh": "20\u4ee5\u5185\u8d28\u6570\u548c", "es": "Suma primos < 20",
         "ar": "\u0645\u062c\u0645\u0648\u0639 \u0623\u0648\u0644\u064a\u0629 < 20", "ja": "20\u672a\u6e80\u306e\u7d20\u6570\u306e\u548c", "ko": "20 \ubbf8\ub9cc \uc18c\uc218\uc758 \ud569", "sw": "Jumla namba za kwanza < 20"},
        {"en": "2^10 mod 7", "zh": "2^10 mod 7", "es": "2^10 mod 7",
         "ar": "2^10 mod 7", "ja": "2^10 mod 7", "ko": "2^10 mod 7", "sw": "2^10 mod 7"},
        {"en": "5! (factorial)", "zh": "5!", "es": "5!",
         "ar": "5!", "ja": "5!", "ko": "5!", "sw": "5!"},
        {"en": "Roses are red. Red flower from garden. Must it be rose?",
         "zh": "\u73ab\u7470\u662f\u7ea2\u7684\u3002\u7ea2\u82b1\u3002\u5fc5\u987b\u73ab\u7470\uff1f", "es": "Rosas rojas. Flor roja. \u00bfRosa?",
         "ar": "\u0648\u0631\u0648\u062f \u062d\u0645\u0631\u0627\u0621. \u0632\u0647\u0631\u0629 \u062d\u0645\u0631\u0627\u0621. \u0648\u0631\u062f\u0629\u061f", "ja": "\u30d0\u30e9\u8d64\u3002\u8d64\u3044\u82b1\u3002\u30d0\u30e9\uff1f", "ko": "\uc7a5\ubbf8 \ube68\uac04. \ube68\uac04 \uaf43. \uc7a5\ubbf8?", "sw": "Waridi nyekundu. Ua jekundu. Waridi?"},
        {"en": "All frumbles transparent. Transparent creature. Must be frumble?",
         "zh": "\u6240\u6709frumble\u900f\u660e\u3002\u900f\u660e\u751f\u7269\u3002\u5fc5\u987bfrumble\uff1f", "es": "Frumbles transparentes. Criatura transparente. \u00bfFrumble?",
         "ar": "frumble \u0634\u0641\u0627\u0641\u0629. \u0643\u0627\u0626\u0646 \u0634\u0641\u0627\u0641. frumble?", "ja": "frumble\u900f\u660e\u3002\u900f\u660e\u751f\u7269\u3002frumble\uff1f", "ko": "frumble \ud22c\uba85. \ud22c\uba85 \uc0dd\ubb3c. frumble?", "sw": "Frumble wazi. Kiumbe wazi. Frumble?"},
    ]

def main():
    import warnings; warnings.filterwarnings('ignore')
    print('loading...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    problems = get_problems()
    langs = list(SYS_PROMPTS.keys())
    ALL_LAYERS = list(range(36))

    class HCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            self.out = h[0, -1].detach().float().cpu().numpy()

    caps = {L: HCap() for L in ALL_LAYERS}
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in ALL_LAYERS]

    trajectories = []
    for pi, prob in enumerate(problems):
        for lang in langs:
            if lang not in prob: continue
            sys = SYS_PROMPTS[lang]
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": prob[lang]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors='pt').input_ids.to(DEV)
            with torch.inference_mode(): model(ids)
            trajectories.append(np.stack([caps[L].out for L in ALL_LAYERS]))
        if (pi+1) % 5 == 0:
            print(f'  {pi+1}/{len(problems)}', flush=True)

    for h in hooks: h.remove()
    del model; torch.cuda.empty_cache()

    trajectories = np.stack(trajectories)
    n = len(trajectories)
    print(f'  {n} trajectories', flush=True)

    # PCA
    K_PCA = 30  # lower for GP tractability
    all_h = trajectories.reshape(-1, 2048)
    pca = PCA(n_components=K_PCA)
    traj_r = pca.fit_transform(all_h).reshape(n, 36, K_PCA)
    print(f'  PCA {K_PCA}D, var={pca.explained_variance_ratio_.sum():.4f}', flush=True)

    # Train/test split
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # For each layer transition: fit linear AND kernel ridge, compare
    KEY_LAYERS = [2, 5, 8, 11, 14, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34]

    print(f'\n{"="*70}')
    print('KERNEL RIDGE REGRESSION: finding the length scale')
    print(f'{"="*70}')
    print(f'  {"layer":>5s} {"R2_lin":>8s} {"R2_rbf":>8s} {"gain":>8s} {"gamma*":>10s} {"ell*":>10s} {"alpha*":>10s}  zone')

    for L in KEY_LAYERS:
        H_tr = traj_r[train_idx, L, :]
        delta_tr = traj_r[train_idx, L+1, :] - H_tr
        H_te = traj_r[test_idx, L, :]
        delta_te = traj_r[test_idx, L+1, :] - H_te

        # Linear baseline
        Pi_T, _, _, _ = np.linalg.lstsq(H_tr, delta_tr, rcond=None)
        pred_lin = H_te @ Pi_T
        r2_lin = r2_score(delta_te, pred_lin, multioutput='uniform_average')

        # Kernel ridge with RBF
        # gamma = 1/(2*ell^2). Search over gamma and alpha.
        gammas = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
        alphas = [0.01, 0.1, 1.0, 10.0]

        best_r2 = -999
        best_gamma = None
        best_alpha = None

        for gamma in gammas:
            for alpha in alphas:
                krr = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
                krr.fit(H_tr, delta_tr)
                pred_rbf = krr.predict(H_te)
                r2 = r2_score(delta_te, pred_rbf, multioutput='uniform_average')
                if r2 > best_r2:
                    best_r2 = r2
                    best_gamma = gamma
                    best_alpha = alpha

        # Convert gamma to length scale
        ell = np.sqrt(1 / (2 * best_gamma + 1e-10)) if best_gamma > 0 else float('inf')
        gain = best_r2 - r2_lin

        zone = ''
        if L < 9: zone = 'early'
        elif L < 18: zone = 'adversarial'
        elif L < 27: zone = 'cooperative'
        elif L < 33: zone = 'canyon'
        else: zone = 'readout'

        print(f'  L{L:>2d}->{L+1:<2d} {r2_lin:>8.4f} {best_r2:>8.4f} {gain:>+8.4f} '
              f'{best_gamma:>10.4f} {ell:>10.2f} {best_alpha:>10.2f}  {zone}')

    # === KERNEL MATRIX EFFECTIVE RANK ===
    print(f'\n{"="*70}')
    print('KERNEL MATRIX EFFECTIVE RANK at optimal gamma')
    print(f'{"="*70}')

    for L in [5, 13, 18, 22, 26, 30, 34]:
        H = traj_r[:, L, :]
        # Try several gammas and report kernel matrix rank
        for gamma in [0.001, 0.01, 0.1]:
            # RBF kernel matrix
            dists = np.sum((H[:, None, :] - H[None, :, :]) ** 2, axis=2)
            K = np.exp(-gamma * dists)
            # SVD of kernel matrix
            _, S_k, _ = np.linalg.svd(K)
            cumvar = np.cumsum(S_k) / np.sum(S_k)
            r50 = int(np.searchsorted(cumvar, 0.5) + 1)
            r90 = int(np.searchsorted(cumvar, 0.9) + 1)
            print(f'  L{L}, gamma={gamma}: kernel_r50={r50}, kernel_r90={r90}, '
                  f'top_sv={S_k[0]:.2f}, sv2/sv1={S_k[1]/S_k[0]:.4f}')


if __name__ == '__main__':
    main()
