"""Weight spectra: zero forward passes. Pure weight analysis.

For each layer, extract:
- W_down effective rank (r90, r50)
- W_up spectral norm and sv1/sv2
- W_gate spectral norm
- Combined Lipschitz = ||W_down||_2 * ||W_up||_2
- Attention kernel norm = ||W_Q^T W_K||_F per head

Overlay with kernel length scale and linear R^2 from prior experiments.
This is the convergence test: do weights predict dynamics?
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cpu'  # weights only, no forward pass needed

def main():
    import warnings; warnings.filterwarnings('ignore')
    print('loading weights (CPU, no GPU needed)...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, trust_remote_code=True)

    # Kernel length scales from probe_kernel.py (hardcoded from results)
    kernel_ell = {2: 2.24, 5: 2.24, 8: 7.07, 11: 7.07, 14: 22.36, 17: 7.07,
                  18: 10.0, 20: 22.36, 22: 10.0, 24: 70.71, 26: 22.36,
                  28: 70.71, 30: 70.71, 32: 70.71, 34: 70.71}
    # Linear R^2 test from probe_vecm (hardcoded)
    linear_r2 = {0: 0.72, 1: 0.84, 2: 0.21, 3: 0.45, 4: 0.64, 5: 0.46, 6: 0.56, 7: 0.46, 8: 0.75,
                 9: 0.47, 10: 0.47, 11: 0.56, 12: 0.74, 13: 0.69, 14: 0.60, 15: 0.64, 16: 0.65, 17: 0.64,
                 18: 0.71, 19: 0.49, 20: 0.53, 21: 0.55, 22: 0.47, 23: 0.43, 24: 0.72, 25: 0.50, 26: 0.84,
                 27: 0.83, 28: 0.86, 29: 0.95, 30: 0.96, 31: 0.93, 32: 0.95, 33: 0.97, 34: 0.97}

    print(f'\n{"="*90}')
    print('WEIGHT SPECTRA: layer-by-layer analysis')
    print(f'{"="*90}')
    print(f'  {"L":>3s} {"Wd_r50":>6s} {"Wd_r90":>6s} {"Wu_sn":>7s} {"Wu_12":>6s} '
          f'{"Wg_sn":>7s} {"Lip":>8s} {"QK_F":>8s} '
          f'{"ell":>6s} {"R2":>6s}  zone')

    results = []

    for L in range(36):
        layer = model.model.layers[L]

        # MLP weights
        W_down = layer.mlp.down_proj.weight.detach().numpy()  # (2048, 11008)
        W_up = layer.mlp.up_proj.weight.detach().numpy()      # (11008, 2048)
        W_gate = layer.mlp.gate_proj.weight.detach().numpy()   # (11008, 2048)

        # W_down effective rank
        _, S_down, _ = np.linalg.svd(W_down, full_matrices=False)
        cumvar_d = np.cumsum(S_down**2) / np.sum(S_down**2)
        wd_r50 = int(np.searchsorted(cumvar_d, 0.5) + 1)
        wd_r90 = int(np.searchsorted(cumvar_d, 0.9) + 1)

        # W_up spectral norm and sv1/sv2
        S_up = np.linalg.svd(W_up, compute_uv=False)
        wu_sn = float(S_up[0])
        wu_ratio = float(S_up[0] / S_up[1]) if S_up[1] > 0 else float('inf')

        # W_gate spectral norm
        S_gate = np.linalg.svd(W_gate, compute_uv=False)
        wg_sn = float(S_gate[0])

        # Combined Lipschitz
        lip = float(S_down[0] * S_up[0])

        # Attention kernel norm: ||W_Q^T W_K||_F
        W_Q = layer.self_attn.q_proj.weight.detach().numpy()  # (2048, 2048)
        W_K = layer.self_attn.k_proj.weight.detach().numpy()  # (256, 2048)
        # QK product for each KV head group
        # Q is (2048, 2048), K is (256, 2048)
        # W_Q^T @ W_K^T = (2048, 2048) @ (2048, 256) = (2048, 256)
        QK = W_Q.T @ W_K.T  # (2048, 256)
        qk_frob = float(np.linalg.norm(QK, 'fro'))

        ell = kernel_ell.get(L, None)
        r2 = linear_r2.get(L, None)

        zone = ''
        if L < 9: zone = 'early'
        elif L < 18: zone = 'adversarial'
        elif L < 27: zone = 'cooperative'
        elif L < 33: zone = 'canyon'
        else: zone = 'readout'

        ell_str = f'{ell:.1f}' if ell is not None else '---'
        r2_str = f'{r2:.2f}' if r2 is not None else '---'

        print(f'  {L:>3d} {wd_r50:>6d} {wd_r90:>6d} {wu_sn:>7.3f} {wu_ratio:>6.3f} '
              f'{wg_sn:>7.3f} {lip:>8.1f} {qk_frob:>8.1f} '
              f'{ell_str:>6s} {r2_str:>6s}  {zone}')

        results.append({
            'layer': L, 'zone': zone,
            'wd_r50': wd_r50, 'wd_r90': wd_r90,
            'wu_sn': wu_sn, 'wu_ratio': wu_ratio, 'wg_sn': wg_sn,
            'lip': lip, 'qk_frob': qk_frob,
            'ell': ell, 'r2': r2,
        })

    # === CORRELATION ANALYSIS ===
    print(f'\n{"="*90}')
    print('CORRELATION: do weight spectra predict kernel dynamics?')
    print(f'{"="*90}')

    # Layers where we have both kernel ell and weight data
    matched = [(r['lip'], r['ell']) for r in results if r['ell'] is not None]
    if matched:
        lips, ells = zip(*matched)
        from scipy.stats import spearmanr, pearsonr
        rho_lip_ell, p_lip_ell = spearmanr(lips, ells)
        print(f'  Spearman(Lipschitz, ell):     rho={rho_lip_ell:+.4f}, p={p_lip_ell:.6f}')
        # Expect NEGATIVE: high Lipschitz = short ell = nonlinear

    matched_qk = [(r['qk_frob'], r['ell']) for r in results if r['ell'] is not None]
    if matched_qk:
        qks, ells = zip(*matched_qk)
        rho_qk_ell, p_qk_ell = spearmanr(qks, ells)
        print(f'  Spearman(QK_frob, ell):       rho={rho_qk_ell:+.4f}, p={p_qk_ell:.6f}')

    matched_gate = [(r['wg_sn'], r['ell']) for r in results if r['ell'] is not None]
    if matched_gate:
        gates, ells = zip(*matched_gate)
        rho_gate_ell, p_gate_ell = spearmanr(gates, ells)
        print(f'  Spearman(W_gate_sn, ell):     rho={rho_gate_ell:+.4f}, p={p_gate_ell:.6f}')
        # Expect NEGATIVE: high gate sensitivity = short ell = nonlinear

    matched_r2 = [(r['lip'], r['r2']) for r in results if r['r2'] is not None]
    if matched_r2:
        lips_r2, r2s = zip(*matched_r2)
        rho_lip_r2, p_lip_r2 = spearmanr(lips_r2, r2s)
        print(f'  Spearman(Lipschitz, R2_test): rho={rho_lip_r2:+.4f}, p={p_lip_r2:.6f}')
        # Expect NEGATIVE: high Lipschitz = more nonlinear = lower linear R2

    # Zone averages
    print(f'\n  zone averages:')
    zone_data = {}
    for r in results:
        z = r['zone']
        if z not in zone_data:
            zone_data[z] = {'lip': [], 'qk': [], 'gate': [], 'wd_r90': [], 'wu_sn': []}
        zone_data[z]['lip'].append(r['lip'])
        zone_data[z]['qk'].append(r['qk_frob'])
        zone_data[z]['gate'].append(r['wg_sn'])
        zone_data[z]['wd_r90'].append(r['wd_r90'])
        zone_data[z]['wu_sn'].append(r['wu_sn'])

    print(f'  {"zone":>15s} {"mean_lip":>10s} {"mean_qk":>10s} {"mean_gate":>10s} {"mean_wd_r90":>12s} {"mean_wu_sn":>10s}')
    for z in ['early', 'adversarial', 'cooperative', 'canyon', 'readout']:
        if z in zone_data:
            d = zone_data[z]
            print(f'  {z:>15s} {np.mean(d["lip"]):>10.1f} {np.mean(d["qk"]):>10.1f} '
                  f'{np.mean(d["gate"]):>10.3f} {np.mean(d["wd_r90"]):>12.1f} {np.mean(d["wu_sn"]):>10.3f}')


if __name__ == '__main__':
    main()
