"""VECM part 2: what ARE the cointegrating vectors?

Extract Pi's right singular vectors at key layer transitions.
Project back to original space. Check alignment with known directions.
Track rotation across layers. What survives to L34->L35?
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    import warnings; warnings.filterwarnings('ignore')
    print('loading...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    problems = [
        {"en": "Solve for x: 3x + 7 = 22", "zh": "\u89e3\u65b9\u7a0b\uff1a3x + 7 = 22", "es": "Resuelve para x: 3x + 7 = 22"},
        {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97\uff1a347 + 658", "es": "Calcula: 347 + 658"},
        {"en": "Find the hypotenuse with legs 5 and 12", "zh": "\u6c42\u76f4\u89d2\u4e09\u89d2\u5f62\u7684\u659c\u8fb9\uff0c\u4e24\u76f4\u89d2\u8fb9\u5206\u522b\u4e3a5\u548c12", "es": "Halla la hipotenusa con catetos 5 y 12"},
        {"en": "What is the GCD of 84 and 120?", "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570", "es": "MCD de 84 y 120"},
        {"en": "How many ways to choose 3 from 7?", "zh": "\u4ece7\u4e2a\u4e2d\u9009\u62e93\u4e2a", "es": "Elegir 3 de 7"},
        {"en": "What is 23 times 17?", "zh": "23\u4e5817", "es": "23 por 17"},
        {"en": "Calculate: 1000 - 387", "zh": "\u8ba1\u7b97\uff1a1000 - 387", "es": "1000 - 387"},
        {"en": "Area of circle radius 7", "zh": "\u534a\u5f847\u7684\u5706\u9762\u79ef", "es": "\u00c1rea c\u00edrculo radio 7"},
        {"en": "All roses are red. Got a red flower. Must it be a rose?",
         "zh": "\u6240\u6709\u73ab\u7470\u662f\u7ea2\u7684\u3002\u62ff\u5230\u7ea2\u82b1\u3002\u5fc5\u987b\u662f\u73ab\u7470\u5417\uff1f",
         "es": "Todas rosas rojas. Flor roja. \u00bfDebe ser rosa?"},
        {"en": "Every frumble in a glasshouse is transparent. Found transparent creature. Must it be frumble?",
         "zh": "\u73bb\u7483\u623f\u91ccfrumble\u900f\u660e\u3002\u53d1\u73b0\u900f\u660e\u751f\u7269\u3002\u5fc5\u987b\u662ffrumble\u5417\uff1f",
         "es": "Frumble en invernadero transparente. Criatura transparente. \u00bfDebe ser frumble?"},
    ]
    langs = ['en', 'zh', 'es']

    class HCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            self.out = h[0, -1].detach().float().cpu().numpy()

    ALL_LAYERS = list(range(36))
    caps = {L: HCap() for L in ALL_LAYERS}
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in ALL_LAYERS]

    trajectories = []; labels = []
    for pi, prob in enumerate(problems):
        for lang in langs:
            if lang not in prob: continue
            ids = tok(prob[lang], return_tensors='pt').input_ids.to(DEV)
            with torch.inference_mode(): model(ids)
            traj = np.stack([caps[L].out for L in ALL_LAYERS])
            trajectories.append(traj); labels.append((pi, lang))

    for h in hooks: h.remove()
    del model; torch.cuda.empty_cache()

    trajectories = np.stack(trajectories)  # (n, 36, 2048)
    n = len(trajectories)
    print(f'{n} trajectories', flush=True)

    # PCA
    K_PCA = 50
    all_h = trajectories.reshape(-1, 2048)
    pca = PCA(n_components=K_PCA)
    traj_r = pca.fit_transform(all_h).reshape(n, 36, K_PCA)
    print(f'PCA {K_PCA}D, var={pca.explained_variance_ratio_.sum():.4f}', flush=True)

    # Known directions in PCA space
    en_idx = [i for i, (p, l) in enumerate(labels) if l == 'en']
    zh_idx = [i for i, (p, l) in enumerate(labels) if l == 'zh']

    # Per-problem centroids (across languages) at each layer
    prob_indices = sorted(set(p for p, l in labels))

    # Fit Pi at each layer and extract cointegrating vectors
    KEY_TRANSITIONS = [8, 13, 17, 18, 22, 26, 29, 33, 34]

    print(f'\n{"="*70}')
    print('COINTEGRATING VECTORS: what are the equilibrium relationships?')
    print(f'{"="*70}')

    coint_vectors = {}  # layer -> (rank, vectors in PCA space)

    for L in range(35):
        H_L = traj_r[:, L, :]
        delta = traj_r[:, L+1, :] - H_L
        Pi_T, _, _, _ = np.linalg.lstsq(H_L, delta, rcond=None)
        Pi = Pi_T.T
        U, S, Vt = np.linalg.svd(Pi)
        cumvar = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
        rank_90 = int(np.searchsorted(cumvar, 0.9) + 1)
        coint_vectors[L] = {'Vt': Vt, 'S': S, 'U': U, 'rank': rank_90}

    # For key transitions, analyze what the top cointegrating vectors represent
    for L in KEY_TRANSITIONS:
        cv = coint_vectors[L]
        rank = cv['rank']
        Vt = cv['Vt']
        S = cv['S']

        print(f'\n  L{L}->{L+1} (rank_90={rank}):')

        # Language direction at this layer
        en_h = traj_r[en_idx, L, :].mean(axis=0)
        zh_h = traj_r[zh_idx, L, :].mean(axis=0)
        lang_dir = en_h - zh_h
        lang_dir /= np.linalg.norm(lang_dir) + 1e-10

        # Per-problem directions (problem centroids)
        prob_centroids = []
        for pi in prob_indices:
            pidx = [i for i, (p, l) in enumerate(labels) if p == pi]
            prob_centroids.append(traj_r[pidx, L, :].mean(axis=0))
        prob_centroids = np.stack(prob_centroids)  # (n_probs, K_PCA)
        prob_centered = prob_centroids - prob_centroids.mean(axis=0)
        _, S_prob, Vt_prob = np.linalg.svd(prob_centered, full_matrices=False)

        # For each top cointegrating vector, check:
        # 1. Alignment with language direction
        # 2. Alignment with top problem directions (f* proxy)
        # 3. What it "looks like" (top PCA components)
        print(f'    {"SV":>3s} {"sv_val":>8s} {"cos_lang":>10s} {"cos_prob1":>10s} {"cos_prob2":>10s} {"cos_prob3":>10s} | top_3_pca_loadings')

        for k in range(min(rank + 2, len(Vt))):
            v = Vt[k]  # right singular vector of Pi
            c_lang = cos(v, lang_dir)
            c_prob = [cos(v, Vt_prob[j]) for j in range(min(3, len(Vt_prob)))]

            # Top PCA component loadings
            top3_idx = np.argsort(np.abs(v))[-3:][::-1]
            loadings = ', '.join(f'PC{idx}={v[idx]:+.3f}' for idx in top3_idx)

            marker = ''
            if abs(c_lang) > 0.3: marker += ' <-- LANG'
            if any(abs(c) > 0.3 for c in c_prob): marker += ' <-- PROB'

            print(f'    {k:>3d} {S[k]:>8.4f} {c_lang:>+10.4f}'
                  f' {c_prob[0]:>+10.4f} {c_prob[1]:>+10.4f} {c_prob[2]:>+10.4f}'
                  f' | {loadings}{marker}')

    # === CROSS-LAYER STABILITY ===
    print(f'\n{"="*70}')
    print('STABILITY: do cointegrating vectors rotate across layers?')
    print(f'{"="*70}')
    print(f'  cos(top-5 vectors at L_a, top-5 vectors at L_b):')

    ref_layers = [13, 18, 22, 26, 33]
    for i, La in enumerate(ref_layers):
        for Lb in ref_layers[i+1:]:
            Vt_a = coint_vectors[La]['Vt'][:5]
            Vt_b = coint_vectors[Lb]['Vt'][:5]
            # Subspace overlap: sum of squared cosines
            overlap = sum(cos(Vt_a[j], Vt_b[k])**2
                         for j in range(5) for k in range(5))
            print(f'  L{La}->L{Lb}: overlap={overlap:.4f} (max=5.0)')

    # === THE FINAL 5: what survives to readout? ===
    print(f'\n{"="*70}')
    print('THE FINAL 5: what cointegrating vectors survive to L34->L35?')
    print(f'{"="*70}')

    final_Vt = coint_vectors[34]['Vt'][:5]
    final_S = coint_vectors[34]['S'][:5]

    # Project back to full space for interpretation
    for k in range(5):
        v_pca = final_Vt[k]  # in PCA space
        v_full = pca.inverse_transform(v_pca.reshape(1, -1))[0] - pca.mean_  # in original 2048D

        # Check against language and problem directions in full space
        en_full = trajectories[en_idx, 34, :].mean(axis=0)
        zh_full = trajectories[zh_idx, 34, :].mean(axis=0)
        lang_full = en_full - zh_full
        lang_full /= np.linalg.norm(lang_full) + 1e-10

        c_lang = cos(v_full, lang_full)

        # Problem centroids in full space
        prob_c_full = []
        for pi in prob_indices:
            pidx = [i for i, (p, l) in enumerate(labels) if p == pi]
            prob_c_full.append(trajectories[pidx, 34, :].mean(axis=0))
        prob_c_full = np.stack(prob_c_full)
        pc_full = prob_c_full - prob_c_full.mean(axis=0)
        _, _, Vt_pf = np.linalg.svd(pc_full, full_matrices=False)

        c_probs = [cos(v_full, Vt_pf[j]) for j in range(min(3, len(Vt_pf)))]

        tag = ''
        if abs(c_lang) > 0.3: tag = 'LANGUAGE'
        elif any(abs(c) > 0.3 for c in c_probs): tag = 'PROBLEM'
        else: tag = 'unknown'

        print(f'  vector {k}: sv={final_S[k]:.4f}, cos_lang={c_lang:+.4f}, '
              f'cos_prob=[{c_probs[0]:+.3f}, {c_probs[1]:+.3f}, {c_probs[2]:+.3f}] -> {tag}')


if __name__ == '__main__':
    main()
