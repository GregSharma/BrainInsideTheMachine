"""VECM on the residual stream: cointegration rank across layers.

Treat h_0, h_1, ..., h_35 as an integrated vector time series (36 "time steps").
Each "observation" is one problem x language (1400 series of 36 steps each).
Work in PCA-reduced space (top 50D) for tractability.

Measure:
- Johansen cointegration rank at each layer transition
- How rank evolves: early -> adversarial -> cooperative -> canyon
- Whether cointegrating vectors align with known directions (f*, e_c, seed)
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'
ALL_LAYERS = list(range(36))

def main():
    import warnings; warnings.filterwarnings('ignore')
    print('loading...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # Problems x languages
    SYS = {}
    problems = [
        {"en": "Solve for x: 3x + 7 = 22", "zh": "\u89e3\u65b9\u7a0b\uff1a3x + 7 = 22",
         "es": "Resuelve para x: 3x + 7 = 22"},
        {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97\uff1a347 + 658",
         "es": "Calcula: 347 + 658"},
        {"en": "Find the hypotenuse of a right triangle with legs 5 and 12",
         "zh": "\u6c42\u76f4\u89d2\u4e09\u89d2\u5f62\u7684\u659c\u8fb9\uff0c\u4e24\u76f4\u89d2\u8fb9\u5206\u522b\u4e3a5\u548c12",
         "es": "Halla la hipotenusa de un tri\u00e1ngulo rect\u00e1ngulo con catetos 5 y 12"},
        {"en": "What is the GCD of 84 and 120?", "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570\u662f\u591a\u5c11\uff1f",
         "es": "\u00bfCu\u00e1l es el MCD de 84 y 120?"},
        {"en": "How many ways can you choose 3 items from 7?",
         "zh": "\u4ece7\u4e2a\u7269\u54c1\u4e2d\u9009\u62e93\u4e2a\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f",
         "es": "\u00bfDe cu\u00e1ntas formas puedes elegir 3 de 7?"},
        {"en": "What is 23 times 17?", "zh": "23\u4e5817\u7b49\u4e8e\u591a\u5c11\uff1f",
         "es": "\u00bfCu\u00e1nto es 23 por 17?"},
        {"en": "Calculate: 1000 - 387", "zh": "\u8ba1\u7b97\uff1a1000 - 387",
         "es": "Calcula: 1000 - 387"},
        {"en": "Find the area of a circle with radius 7",
         "zh": "\u6c42\u534a\u5f84\u4e3a7\u7684\u5706\u7684\u9762\u79ef",
         "es": "Halla el \u00e1rea de un c\u00edrculo con radio 7"},
        # Logic problems (non-math)
        {"en": "All roses in my garden are red. I got a flower from my garden and it is red. Must it be a rose?",
         "zh": "\u6211\u82b1\u56ed\u91cc\u6240\u6709\u7684\u73ab\u7470\u90fd\u662f\u7ea2\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u82b1\uff0c\u5b83\u662f\u7ea2\u8272\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u73ab\u7470\u5417\uff1f",
         "es": "Todas las rosas de mi jard\u00edn son rojas. Cog\u00ed una flor de mi jard\u00edn y es roja. \u00bfTiene que ser una rosa?"},
        {"en": "Every frumble in a glasshouse is transparent. I found a transparent creature in a glasshouse. Must it be a frumble?",
         "zh": "\u6240\u6709\u5728\u73bb\u7483\u623f\u91cc\u7684frumble\u90fd\u662f\u900f\u660e\u7684\u3002\u6211\u5728\u73bb\u7483\u623f\u91cc\u53d1\u73b0\u4e86\u4e00\u4e2a\u900f\u660e\u7684\u751f\u7269\u3002\u5b83\u4e00\u5b9a\u662ffrumble\u5417\uff1f",
         "es": "Todo frumble en un invernadero es transparente. Encontr\u00e9 una criatura transparente en un invernadero. \u00bfDebe ser un frumble?"},
    ]
    langs = ['en', 'zh', 'es']

    # Hooks to capture h at ALL layers
    class HCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            self.out = h[0, -1].detach().float().cpu().numpy()

    caps = {L: HCap() for L in ALL_LAYERS}
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in ALL_LAYERS]

    # Collect h trajectories: (n_samples, 36 layers, 2048)
    trajectories = []  # list of (36, 2048) arrays
    labels = []  # (problem_idx, lang)

    for pi, prob in enumerate(problems):
        for lang in langs:
            if lang not in prob:
                continue
            ids = tok(prob[lang], return_tensors='pt').input_ids.to(DEV)
            with torch.inference_mode():
                model(ids)
            traj = np.stack([caps[L].out for L in ALL_LAYERS])  # (36, 2048)
            trajectories.append(traj)
            labels.append((pi, lang))

    for h in hooks: h.remove()
    del model; torch.cuda.empty_cache()

    trajectories = np.stack(trajectories)  # (n_samples, 36, 2048)
    n_samples = len(trajectories)
    print(f'collected {n_samples} trajectories, shape {trajectories.shape}', flush=True)

    # PCA to reduce dimensions
    K_PCA = 50
    all_h = trajectories.reshape(-1, 2048)  # (n_samples * 36, 2048)
    pca = PCA(n_components=K_PCA)
    all_h_reduced = pca.fit_transform(all_h)  # (n_samples * 36, K_PCA)
    trajectories_r = all_h_reduced.reshape(n_samples, 36, K_PCA)  # (n_samples, 36, K_PCA)
    print(f'PCA to {K_PCA}D, explained variance: {pca.explained_variance_ratio_.sum():.4f}', flush=True)

    # === VECM ANALYSIS ===
    # For each layer transition L -> L+1:
    # delta_h = h_{L+1} - h_L (the "first difference")
    # Fit: delta_h = Pi @ h_L + epsilon
    # rank(Pi) = number of cointegrating relationships

    print(f'\n{"="*70}')
    print('VECM: cointegration rank across layer transitions')
    print(f'{"="*70}')
    print(f'  {"layer":>5s} {"rank_Pi":>8s} {"R2":>8s} {"||Pi||":>8s} {"top_sv":>8s} {"sv2":>8s} {"sv5":>8s} {"sv9":>8s}  zone')

    pi_matrices = {}  # store for later analysis

    for L in range(35):
        # h_L across all samples: (n_samples, K_PCA)
        H_L = trajectories_r[:, L, :]    # (n, K_PCA)
        H_L1 = trajectories_r[:, L+1, :]  # (n, K_PCA)
        delta = H_L1 - H_L  # (n, K_PCA)

        # Fit Pi: delta = H_L @ Pi.T + residual
        # Pi = (delta.T @ H_L) @ (H_L.T @ H_L)^{-1}
        # Using least squares:
        # delta = H_L @ Pi.T  =>  Pi.T = lstsq(H_L, delta)
        Pi_T, residuals, rank_HL, sv_HL = np.linalg.lstsq(H_L, delta, rcond=None)
        Pi = Pi_T.T  # (K_PCA, K_PCA)

        # R^2
        delta_pred = H_L @ Pi_T
        ss_res = np.sum((delta - delta_pred)**2)
        ss_tot = np.sum((delta - delta.mean(axis=0))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # SVD of Pi to get rank
        U_pi, S_pi, Vt_pi = np.linalg.svd(Pi)

        # Effective rank (how many SVs are significant)
        cumvar_pi = np.cumsum(S_pi**2) / (np.sum(S_pi**2) + 1e-10)
        rank_50 = int(np.searchsorted(cumvar_pi, 0.5) + 1)
        rank_90 = int(np.searchsorted(cumvar_pi, 0.9) + 1)

        pi_norm = np.linalg.norm(Pi, 'fro')

        zone = ''
        if L < 9: zone = 'early'
        elif L < 18: zone = 'adversarial'
        elif L < 27: zone = 'cooperative'
        elif L < 33: zone = 'canyon'
        else: zone = 'readout'

        sv2 = S_pi[1] if len(S_pi) > 1 else 0
        sv5 = S_pi[4] if len(S_pi) > 4 else 0
        sv9 = S_pi[8] if len(S_pi) > 8 else 0

        print(f'  L{L:>2d}->{L+1:<2d} {rank_90:>8d} {r2:>8.4f} {pi_norm:>8.2f} '
              f'{S_pi[0]:>8.4f} {sv2:>8.4f} {sv5:>8.4f} {sv9:>8.4f}  {zone}')

        pi_matrices[L] = {
            'Pi': Pi,
            'S': S_pi,
            'U': U_pi,
            'Vt': Vt_pi,
            'r2': r2,
            'rank_90': rank_90,
        }

    # === COINTEGRATION RANK TRAJECTORY ===
    print(f'\n  rank_90 trajectory (number of cointegrating relationships):')
    ranks = [pi_matrices[L]['rank_90'] for L in range(35)]
    print(f'  early(0-8):       {ranks[0:9]}')
    print(f'  adversarial(9-17): {ranks[9:18]}')
    print(f'  cooperative(18-26):{ranks[18:27]}')
    print(f'  canyon(27-32):     {ranks[27:33]}')
    print(f'  readout(33-34):    {ranks[33:35]}')

    # === DO PI'S EIGENVECTORS ALIGN WITH KNOWN DIRECTIONS? ===
    print(f'\n{"="*70}')
    print('ALIGNMENT: do Pi eigenvectors match known directions?')
    print(f'{"="*70}')

    # Known directions in PCA space:
    # 1. Language mean difference (en vs zh)
    en_idx = [i for i, (p, l) in enumerate(labels) if l == 'en']
    zh_idx = [i for i, (p, l) in enumerate(labels) if l == 'zh']

    for L in [13, 18, 22, 26, 30, 33]:
        en_mean = trajectories_r[en_idx, L, :].mean(axis=0)
        zh_mean = trajectories_r[zh_idx, L, :].mean(axis=0)
        lang_dir = en_mean - zh_mean
        lang_dir = lang_dir / (np.linalg.norm(lang_dir) + 1e-10)

        # Project lang_dir onto Pi's top singular vectors
        if L < 35:
            Vt = pi_matrices[L]['Vt']  # right singular vectors of Pi
            S = pi_matrices[L]['S']
            # How much of lang_dir is in the top-k right singular vectors?
            projections = [abs(float(np.dot(lang_dir, Vt[k]))) for k in range(min(10, len(Vt)))]
            top_proj = max(projections)
            top_k = projections.index(top_proj)
            print(f'  L{L}: lang_dir alignment with Pi SVs: '
                  f'max={top_proj:.4f} at SV{top_k} (sv={S[top_k]:.4f}), '
                  f'top3=[{projections[0]:.3f}, {projections[1]:.3f}, {projections[2]:.3f}]')

    # R^2 trajectory
    print(f'\n  R\u00b2 trajectory (how linear is each layer transition?):')
    r2s = [pi_matrices[L]['r2'] for L in range(35)]
    print(f'  early(0-8):       {[f"{r:.3f}" for r in r2s[0:9]]}')
    print(f'  adversarial(9-17): {[f"{r:.3f}" for r in r2s[9:18]]}')
    print(f'  cooperative(18-26):{[f"{r:.3f}" for r in r2s[18:27]]}')
    print(f'  canyon(27-32):     {[f"{r:.3f}" for r in r2s[27:33]]}')
    print(f'  readout(33-34):    {[f"{r:.3f}" for r in r2s[33:35]]}')


if __name__ == '__main__':
    main()
