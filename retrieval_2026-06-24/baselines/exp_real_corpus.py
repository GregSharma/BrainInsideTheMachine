#!/usr/bin/env python3
"""Real-corpus retrieval benchmark v4 — Grassmannian + covariance + Frobenius.

48 real paragraphs, independent queries. Compares pooling-based (mean/late/last)
against matrix-level scorers that use the FULL token path:
  1. Grassmannian: principal angles between V-subspaces
  2. Covariance alignment: tr(Σ_Q Σ_C) / (||Σ_Q|| ||Σ_C||)
  3. Frobenius: normalized inner product in shared SVD basis
Plus the prior cosine-pooling families.

MRR with honest train/test layer selection. Sig kernel SKIPPED (needs vectorization).
"""
import json, random, gc, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from exp_unbiased_sweep import Hooks

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
DRAW_SEED = 12061

PAIRS = [(p['chunk'], p['query']) for p in json.load(open('output/real_pairs.json'))]


# ── matrix-level scoring functions ──────────────────────────

def grassmann_score(V_Q, V_C, k=8):
    """Sum of squared cosines of principal angles between top-k column spans."""
    k = min(k, V_Q.shape[0], V_C.shape[0], V_Q.shape[1])
    Q_Q = torch.linalg.qr(V_Q.T)[0][:, :k]
    Q_C = torch.linalg.qr(V_C.T)[0][:, :k]
    S = torch.linalg.svdvals(Q_Q.T @ Q_C)
    return (S ** 2).sum().item()


def cov_score(V_Q, V_C):
    """Centered covariance alignment: tr(Σ_Q Σ_C) / (||Σ_Q||_F ||Σ_C||_F)."""
    Qc = V_Q - V_Q.mean(0, keepdim=True)
    Cc = V_C - V_C.mean(0, keepdim=True)
    S_Q = Qc.T @ Qc / max(V_Q.shape[0] - 1, 1)
    S_C = Cc.T @ Cc / max(V_C.shape[0] - 1, 1)
    num = (S_Q * S_C).sum()
    den = S_Q.norm() * S_C.norm()
    return (num / den).item() if den > 1e-12 else 0.0


def frob_score(V_Q, V_C, k=16):
    """Normalized Frobenius inner product in shared top-k SVD basis."""
    V_cat = torch.cat([V_Q, V_C], dim=0)
    _, S, Wt = torch.linalg.svd(V_cat, full_matrices=False)
    k = min(k, Wt.shape[0])
    P_Q = V_Q @ Wt[:k].T
    P_C = V_C @ Wt[:k].T
    score = 0.0
    for j in range(k):
        mq, mc = P_Q[:, j].mean(), P_C[:, j].mean()
        sq, sc = P_Q[:, j].std() + 1e-8, P_C[:, j].std() + 1e-8
        score += abs(mq * mc) / (sq * sc)
    return score / k


# ── encoding ────────────────────────────────────────────────

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    hooks = Hooks(model)

    h_cap = {}
    def make_h_hook(li):
        def hook(m, inp):
            h_cap[li] = inp[0].detach()[0].float().cpu()
        return hook
    h_hooks = []
    for li, layer in enumerate(model.model.layers):
        h_hooks.append(layer.register_forward_pre_hook(make_h_hook(li)))
    def final_hook(m, inp, out):
        h_cap['final'] = out[0][0].float().cpu()
    h_hooks.append(model.model.norm.register_forward_hook(final_hook))

    @torch.no_grad()
    def encode_one(text):
        hooks.cur = {}; h_cap.clear()
        ids = tok(text, return_tensors='pt').input_ids.to(DEVICE)
        model(input_ids=ids)
        H = [h_cap[li] for li in range(len(model.model.layers))]
        if 'final' in h_cap: H.append(h_cap['final'])
        V = [hooks.cur[('v', li)] for li in range(len(model.model.layers))]
        torch.cuda.empty_cache()
        return {'H': H, 'V': V}

    print(f"encoding {len(PAIRS)} pairs...", flush=True)
    chunks = [encode_one(c) for c, q in PAIRS]
    queries = [encode_one(q) for c, q in PAIRS]
    for h in h_hooks: h.remove()
    hooks.close(); del model; gc.collect(); torch.cuda.empty_cache()

    N = len(PAIRS)
    nL_h = len(chunks[0]['H'])
    nL_v = len(chunks[0]['V'])

    rng = random.Random(DRAW_SEED)
    idx = list(range(N)); rng.shuffle(idx)
    TR, TE = idx[:N//2], idx[N//2:]

    def mrr_from_scores(score_matrix, eval_idx):
        rr = []; r1 = 0; r3 = 0; r5 = 0
        for qi in eval_idx:
            sims = score_matrix[qi]
            rank = (sims.argsort(descending=True) == qi).nonzero().item() + 1
            rr.append(1.0/rank); r1 += (rank==1); r3 += (rank<=3); r5 += (rank<=5)
        n = len(eval_idx)
        return sum(rr)/n, r1/n, r3/n, r5/n

    # ── cosine families ─────────────────────────────────────
    def cosine_matrix(seqs_q, seqs_c, l, key, pool):
        S = torch.zeros(N, N)
        for qi in range(N):
            Tq = seqs_q[qi][key][l]
            if pool == 'mean': qv = Tq.mean(0)
            elif pool == 'late': qv = Tq[max(0, int(0.75*Tq.shape[0])):].mean(0)
            elif pool == 'last': qv = Tq[-1]
            qv = qv.view(1, -1)
            for ci in range(N):
                Tc = seqs_c[ci][key][l]
                if pool == 'mean': cv = Tc.mean(0)
                elif pool == 'late': cv = Tc[max(0, int(0.75*Tc.shape[0])):].mean(0)
                elif pool == 'last': cv = Tc[-1]
                cv = cv.view(1, -1)
                S[qi, ci] = F.cosine_similarity(qv, cv).item()
        return S

    # ── matrix-level families ───────────────────────────────
    def matrix_scores(seqs_q, seqs_c, l, key, scorer_fn):
        S = torch.zeros(N, N)
        for qi in range(N):
            Vq = seqs_q[qi][key][l]
            if Vq.dim() < 2: return None
            for ci in range(N):
                Vc = seqs_c[ci][key][l]
                if Vc.dim() < 2: return None
                S[qi, ci] = scorer_fn(Vq, Vc)
        return S

    configs = []
    # cosine-pooling
    for key, nL, label in [('H', nL_h, 'h'), ('V', nL_v, 'v')]:
        for pool in ['mean', 'late', 'last']:
            configs.append((f'{label}|{pool}', key, nL, 'cosine', pool, None))
    # innovation (depth-delta mean)
    configs.append(('eh|mean', 'H', nL_h, 'eh', 'mean', None))
    # matrix-level V-space
    configs.append(('v|grassmann', 'V', nL_v, 'matrix', None, grassmann_score))
    configs.append(('v|cov', 'V', nL_v, 'matrix', None, cov_score))
    configs.append(('v|frob', 'V', nL_v, 'matrix', None, frob_score))
    # matrix-level H-space
    configs.append(('h|grassmann', 'H', nL_h, 'matrix', None, grassmann_score))
    configs.append(('h|cov', 'H', nL_h, 'matrix', None, cov_score))

    print(f"\nN={N}, train={len(TR)}, test={len(TE)}", flush=True)
    print(f"random baseline: MRR~{sum(1/r for r in range(1,N+1))/N:.3f}\n", flush=True)
    print(f"{'family':>14} {'L*':>3} | {'tr MRR':>7} | {'te MRR':>7} {'R@1':>5} {'R@3':>5} {'R@5':>5}", flush=True)

    out = {}
    for name, key, nL, mode, pool, scorer_fn in configs:
        best = None
        # for matrix-level, sample every 4th layer + last to keep runtime sane
        layers = range(nL) if mode in ('cosine', 'eh') else list(range(0, nL, 4)) + [nL-1]
        for l in layers:
            if mode == 'cosine':
                S = cosine_matrix(queries, chunks, l, key, pool)
            elif mode == 'eh':
                # depth-delta mean
                S = torch.zeros(N, N)
                for qi in range(N):
                    if l == 0: qv = queries[qi]['H'][0].mean(0)
                    else: qv = (queries[qi]['H'][l] - queries[qi]['H'][l-1]).mean(0)
                    qv = qv.view(1, -1)
                    for ci in range(N):
                        if l == 0: cv = chunks[ci]['H'][0].mean(0)
                        else: cv = (chunks[ci]['H'][l] - chunks[ci]['H'][l-1]).mean(0)
                        cv = cv.view(1, -1)
                        S[qi, ci] = F.cosine_similarity(qv, cv).item()
            elif mode == 'matrix':
                print(f"  {name} L{l}...", end='', flush=True)
                S = matrix_scores(queries, chunks, l, key, scorer_fn)
                if S is None: print(" skip(1D)", flush=True); continue
                print(f" done", flush=True)
            m = mrr_from_scores(S, TR)
            if best is None or m[0] > best[1]:
                best = (l, m[0], S)
        L, _, S_best = best
        tr = mrr_from_scores(S_best, TR)
        te = mrr_from_scores(S_best, TE)
        out[name] = {'L': L, 'tr': round(tr[0],3), 'te': round(te[0],3),
                     'r1': round(te[1],3), 'r3': round(te[2],3), 'r5': round(te[3],3)}
        print(f"{name:>14} L{L:>2} | {tr[0]:>7.3f} | {te[0]:>7.3f} {te[1]:>5.2f} {te[2]:>5.2f} {te[3]:>5.2f}", flush=True)

    top = sorted(out.items(), key=lambda kv: kv[1]['te'], reverse=True)
    print(f"\nFinal ranking:", flush=True)
    for k, v in top:
        print(f"  {k:>14} L{v['L']:>2}  MRR={v['te']:.3f} R@1={v['r1']:.2f} R@3={v['r3']:.2f} R@5={v['r5']:.2f}", flush=True)

    with open('output/exp_real_corpus.json', 'w') as f:
        json.dump({'N': N, 'results': out, 'ranking': top}, f, indent=2)
    print("\nsaved output/exp_real_corpus.json", flush=True)


if __name__ == '__main__':
    main()
