"""Weight spectra FAST: use torch SVD on GPU, skip full decomposition.

For the large MLP matrices, use randomized SVD (top-k only) or
just compute spectral norm via power iteration.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def spectral_norm_power(W, n_iter=20):
    """Fast spectral norm via power iteration. O(n_iter * m * n)."""
    v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
    for _ in range(n_iter):
        u = W @ v; u = u / (u.norm() + 1e-10)
        v = W.T @ u; v = v / (v.norm() + 1e-10)
    return float((W @ v).norm())

def effective_rank_fast(W, k=50):
    """Effective rank from top-k SVD via torch.svd_lowrank."""
    U, S, V = torch.svd_lowrank(W.float(), q=k)
    cumvar = torch.cumsum(S**2, 0) / (S**2).sum()
    r50 = int((cumvar < 0.5).sum()) + 1
    r90 = int((cumvar < 0.9).sum()) + 1
    return r50, r90, S[:10].cpu().numpy()

def main():
    import warnings; warnings.filterwarnings('ignore')
    print('loading on GPU...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)

    print(f'\n{"="*100}')
    print('WEIGHT SPECTRA (fast: GPU power iteration + low-rank SVD)')
    print(f'{"="*100}')
    print(f'  {"L":>3s} {"Wd_r50":>6s} {"Wd_r90":>6s} {"Wd_sn":>7s} '
          f'{"Wu_sn":>7s} {"Wg_sn":>7s} {"Lip":>8s} {"QK_sn":>8s} '
          f'{"Wd_1/2":>7s}  zone')

    all_results = []

    for L in range(36):
        layer = model.model.layers[L]

        W_down = layer.mlp.down_proj.weight.detach()  # (2048, 11008)
        W_up = layer.mlp.up_proj.weight.detach()      # (11008, 2048)
        W_gate = layer.mlp.gate_proj.weight.detach()   # (11008, 2048)
        W_Q = layer.self_attn.q_proj.weight.detach()   # (2048, 2048)
        W_K = layer.self_attn.k_proj.weight.detach()   # (256, 2048)

        # Spectral norms (fast)
        wd_sn = spectral_norm_power(W_down)
        wu_sn = spectral_norm_power(W_up)
        wg_sn = spectral_norm_power(W_gate)

        # Lipschitz
        lip = wd_sn * wu_sn

        # W_down effective rank (top-50 SVD)
        wd_r50, wd_r90, wd_sv = effective_rank_fast(W_down, k=50)
        wd_ratio = float(wd_sv[0] / wd_sv[1]) if wd_sv[1] > 0 else float('inf')

        # QK spectral norm: ||W_Q^T W_K^T||_2
        QK = W_Q.float().T @ W_K.float().T  # (2048, 256)
        qk_sn = spectral_norm_power(QK.half())

        zone = ''
        if L < 9: zone = 'early'
        elif L < 18: zone = 'adversarial'
        elif L < 27: zone = 'cooperative'
        elif L < 33: zone = 'canyon'
        else: zone = 'readout'

        print(f'  {L:>3d} {wd_r50:>6d} {wd_r90:>6d} {wd_sn:>7.3f} '
              f'{wu_sn:>7.3f} {wg_sn:>7.3f} {lip:>8.1f} {qk_sn:>8.1f} '
              f'{wd_ratio:>7.3f}  {zone}', flush=True)

        all_results.append({
            'L': L, 'zone': zone,
            'wd_r50': wd_r50, 'wd_r90': wd_r90, 'wd_sn': wd_sn,
            'wu_sn': wu_sn, 'wg_sn': wg_sn, 'lip': lip, 'qk_sn': qk_sn,
            'wd_ratio': wd_ratio,
        })

    # Zone averages
    print(f'\n  zone averages:')
    print(f'  {"zone":>15s} {"Lip":>8s} {"Wg_sn":>7s} {"QK_sn":>8s} {"Wd_r90":>7s} {"Wd_sn":>7s}')
    for z in ['early', 'adversarial', 'cooperative', 'canyon', 'readout']:
        zr = [r for r in all_results if r['zone'] == z]
        if zr:
            print(f'  {z:>15s} {np.mean([r["lip"] for r in zr]):>8.1f} '
                  f'{np.mean([r["wg_sn"] for r in zr]):>7.3f} '
                  f'{np.mean([r["qk_sn"] for r in zr]):>8.1f} '
                  f'{np.mean([r["wd_r90"] for r in zr]):>7.1f} '
                  f'{np.mean([r["wd_sn"] for r in zr]):>7.3f}')

    # Predictions check
    print(f'\n  DEEPSEEK PREDICTIONS:')
    lips = [r['lip'] for r in all_results]
    gates = [r['wg_sn'] for r in all_results]
    qks = [r['qk_sn'] for r in all_results]
    wd_r90s = [r['wd_r90'] for r in all_results]

    print(f'  P1: W_gate sn should be highest early, lowest canyon/readout')
    print(f'    early mean={np.mean(gates[:9]):.3f}, advers={np.mean(gates[9:18]):.3f}, '
          f'coop={np.mean(gates[18:27]):.3f}, canyon={np.mean(gates[27:33]):.3f}, '
          f'read={np.mean(gates[33:]):.3f}')

    print(f'  P3: Lipschitz should correlate inversely with ell (nonlinear=high Lip, linear=low Lip)')
    print(f'    early mean={np.mean(lips[:9]):.1f}, advers={np.mean(lips[9:18]):.1f}, '
          f'coop={np.mean(lips[18:27]):.1f}, canyon={np.mean(lips[27:33]):.1f}, '
          f'read={np.mean(lips[33:]):.1f}')

    print(f'  P4: QK_sn should be highest adversarial (picky attention), lowest canyon (diffuse)')
    print(f'    early mean={np.mean(qks[:9]):.1f}, advers={np.mean(qks[9:18]):.1f}, '
          f'coop={np.mean(qks[18:27]):.1f}, canyon={np.mean(qks[27:33]):.1f}, '
          f'read={np.mean(qks[33:]):.1f}')


if __name__ == '__main__':
    main()
