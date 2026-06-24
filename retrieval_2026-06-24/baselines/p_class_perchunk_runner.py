"""
Lower-time-frame chart-on-a-chart: run the unified SBERT/TFIDF/QK probe on
Greg's ORIGINAL stimuli (synthetic_probe grief 8+8, reverse_attention flirt 4+4),
report PER-CHUNK which method gets each chunk right vs wrong.

This is the case-by-case look that the v3/v4 aggregate sweep can't give us.
"""
import os, json, numpy as np, torch
from numpy.random import default_rng
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen2.5-3B"
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LAYERS = [0, 6, 12, 18, 22, 26, 27, 30, 33, 34]
QK_LAYERS = [18, 22, 26, 27, 30, 33]
DEVICE = "cuda"
OUT = "output/exp_original_perchunk.json"


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    o = np.argsort(-s, kind='mergesort'); y = y[o]
    P, N = (y == 1).sum(), (y == 0).sum()
    if P == 0 or N == 0: return float('nan')
    tps = np.cumsum(y == 1) / P; fps = np.cumsum(y == 0) / N
    return float(np.trapezoid(tps, fps))


@torch.no_grad()
def capture(model, tok, text, layers, qk_layers, max_len=64):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
    qk_buf = {L: {} for L in qk_layers}
    handles = []
    def make_hook(L, kind):
        def h(mod, inp, out): qk_buf[L][kind] = out[0].float().cpu().numpy()
        return h
    for L in qk_layers:
        layer = model.model.layers[L]
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(L, 'q')))
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(L, 'k')))
    o = model(input_ids=ids, output_hidden_states=True)
    for h in handles: h.remove()
    last_resid = {L: o.hidden_states[L][0, -1].float().cpu().numpy() for L in layers}
    n_head = model.config.num_attention_heads
    n_kv = getattr(model.config, 'num_key_value_heads', n_head)
    d_h = model.config.hidden_size // n_head
    qk = {}
    for L in qk_layers:
        q, k = qk_buf[L]['q'], qk_buf[L]['k']
        T = q.shape[0]
        q_last = q[-1].reshape(n_head, d_h)
        k_all = k.reshape(T, n_kv, d_h)
        if n_kv != n_head:
            k_all = np.repeat(k_all, n_head // n_kv, axis=1)
        qk[L] = {'q': q_last, 'k_mean': k_all.mean(0)}
    return {'last_resid': last_resid, 'qk': qk}


def load_cells():
    cells = {}
    # grief / not-grief from synthetic_probe
    d = json.load(open('output/exp_synthetic_probe.json'))
    cells['grief_v1'] = {'pos': d['pos_exemplars'], 'neg': d['neg_exemplars']}
    # flirt / neutral from reverse_attention_adjoint_flirt
    d = json.load(open('output/exp_reverse_attention_adjoint_flirt.json'))
    pos = [x['chunk'] for x in d['results'] if x['label'] == 'FLIRT']
    neg = [x['chunk'] for x in d['results'] if x['label'] == 'NEUTRAL']
    cells['flirt_v1'] = {'pos': pos, 'neg': neg}
    return cells


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def main():
    rng = default_rng(0)
    cells = load_cells()
    print(f'[battery] {len(cells)} cells: {list(cells.keys())}')
    for name, bd in cells.items():
        print(f'  {name}: n_pos={len(bd["pos"])} n_neg={len(bd["neg"])}')

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f'[model] loading {MODEL}')
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEVICE,
                                                 trust_remote_code=True, attn_implementation='eager').eval()
    print(f'[sbert] loading {SBERT_MODEL}')
    sbert = SentenceTransformer(SBERT_MODEL)

    out = {'model': MODEL, 'sbert': SBERT_MODEL, 'per_cell': {}}
    n_head = model.config.num_attention_heads

    for cell, bd in cells.items():
        pos, neg = bd['pos'], bd['neg']
        texts = pos + neg
        Y = np.array([1] * len(pos) + [0] * len(neg))
        n = len(texts)
        print(f'\n=== {cell}: n={n} ({len(pos)}+{len(neg)}) ===')

        # capture activations for every chunk
        caches = [capture(model, tok, t, LAYERS, QK_LAYERS) for t in texts]
        print(f'  [capture] done')

        # SBERT embeddings
        sb_emb = sbert.encode(texts, batch_size=32, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)

        # TFIDF: 5-fold logistic if n big enough, else LOO
        if n >= 10:
            skf = StratifiedKFold(n_splits=min(5, len(pos), len(neg)), shuffle=True, random_state=0)
            tf_aucs = []
            for tr, te in skf.split(texts, Y):
                v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000, lowercase=True)
                Xtr = v.fit_transform([texts[i] for i in tr])
                Xte = v.transform([texts[i] for i in te])
                clf = LogisticRegression(C=1.0, max_iter=1000)
                clf.fit(Xtr, Y[tr])
                tf_aucs.append(auc(clf.decision_function(Xte), Y[te]))
            tfidf_auc = float(np.mean(tf_aucs))
        else:
            tfidf_auc = float('nan')

        # Leave-One-Out per-chunk evaluation
        # For each query chunk i, exemplars are everything else (pos as pos pool, neg as neg pool)
        rows = []
        for i in range(n):
            qtext = texts[i]
            qlab = Y[i]
            other_pos = [j for j in range(n) if j != i and Y[j] == 1]
            other_neg = [j for j in range(n) if j != i and Y[j] == 0]
            if not other_pos or not other_neg: continue

            # SBERT scores
            sb_pos_mean = sb_emb[other_pos].mean(0); sb_pos_mean /= (np.linalg.norm(sb_pos_mean) + 1e-12)
            sb_neg_mean = sb_emb[other_neg].mean(0); sb_neg_mean /= (np.linalg.norm(sb_neg_mean) + 1e-12)
            sb_diff = sb_pos_mean - sb_neg_mean
            sb_score_hyde = float(sb_emb[i] @ sb_pos_mean)
            sb_score_diff = float(sb_emb[i] @ sb_diff)

            # Per-layer mean_resid_cos and diff_resid_cos
            resid_by_layer = {}
            for L in LAYERS:
                pm = np.mean([caches[j]['last_resid'][L] for j in other_pos], axis=0)
                nm = np.mean([caches[j]['last_resid'][L] for j in other_neg], axis=0)
                resid_by_layer[L] = {
                    'hyde_cos': cosine(caches[i]['last_resid'][L], pm),
                    'diff_cos': cosine(caches[i]['last_resid'][L], pm - nm),
                }

            # Per-(layer,head) QK scores
            qk_by_layer = {}
            for L in QK_LAYERS:
                kpos = np.mean([caches[j]['qk'][L]['k_mean'] for j in other_pos], axis=0)
                kneg = np.mean([caches[j]['qk'][L]['k_mean'] for j in other_neg], axis=0)
                head_pos = []; head_diff = []
                for h in range(n_head):
                    sp = float(caches[i]['qk'][L]['q'][h] @ kpos[h])
                    sd = float(caches[i]['qk'][L]['q'][h] @ (kpos[h] - kneg[h]))
                    head_pos.append(sp); head_diff.append(sd)
                qk_by_layer[L] = {'head_pos': head_pos, 'head_diff': head_diff}

            rows.append({
                'i': i, 'label': int(qlab), 'text': qtext,
                'sbert': {'hyde': sb_score_hyde, 'diff': sb_score_diff},
                'resid': resid_by_layer,
                'qk': qk_by_layer,
            })

        # Compute LOO AUCs from rows
        aucs = {}
        # SBERT
        aucs['sbert_hyde'] = auc([r['sbert']['hyde'] for r in rows], [r['label'] for r in rows])
        aucs['sbert_diff'] = auc([r['sbert']['diff'] for r in rows], [r['label'] for r in rows])
        # Per-layer resid
        for L in LAYERS:
            aucs[f'resid_hyde_L{L}'] = auc([r['resid'][L]['hyde_cos'] for r in rows], [r['label'] for r in rows])
            aucs[f'resid_diff_L{L}'] = auc([r['resid'][L]['diff_cos'] for r in rows], [r['label'] for r in rows])
        # Per-(layer,head) QK
        for L in QK_LAYERS:
            for h in range(n_head):
                aucs[f'qk_pos_L{L}_h{h}'] = auc([r['qk'][L]['head_pos'][h] for r in rows], [r['label'] for r in rows])
                aucs[f'qk_diff_L{L}_h{h}'] = auc([r['qk'][L]['head_diff'][h] for r in rows], [r['label'] for r in rows])
        # Take best per family using |auc-0.5|
        def best(prefix):
            keys = [k for k in aucs if k.startswith(prefix) and not np.isnan(aucs[k])]
            if not keys: return None, float('nan')
            k = max(keys, key=lambda kk: abs(aucs[kk] - 0.5))
            return k, aucs[k]
        sbert_best_k, sbert_best = best('sbert')
        resid_diff_best_k, resid_diff_best = best('resid_diff')
        qk_diff_best_k, qk_diff_best = best('qk_diff')
        qk_pos_best_k, qk_pos_best = best('qk_pos')

        # Per-chunk winner analysis
        # Use the BEST QK and BEST SBERT scores per chunk
        # For SBERT: pick whichever (hyde, diff) has higher abs(auc-0.5)
        sbert_pick = 'diff' if abs(aucs['sbert_diff']-0.5) > abs(aucs['sbert_hyde']-0.5) else 'hyde'
        # For QK: parse the best key
        def parse_qk_key(k):
            # qk_diff_L{L}_h{H}
            parts = k.split('_')
            L = int(parts[2][1:]); H = int(parts[3][1:])
            kind = parts[1]
            return L, H, kind
        if qk_diff_best_k:
            qkL, qkH, qkkind = parse_qk_key(qk_diff_best_k)
        else:
            qkL, qkH, qkkind = None, None, None

        # Per-chunk: did SBERT call it right? did QK call it right?
        # "Right" = score above median for class 1, below median for class 0 (or use ranks)
        per_chunk = []
        sb_scores = [r['sbert'][sbert_pick] for r in rows]
        qk_scores = [r['qk'][qkL]['head_'+qkkind][qkH] for r in rows] if qkL is not None else [0]*len(rows)
        sb_med = np.median(sb_scores); qk_med = np.median(qk_scores)
        for r, sb, qk_s in zip(rows, sb_scores, qk_scores):
            sb_call = int(sb > sb_med)
            qk_call = int(qk_s > qk_med)
            per_chunk.append({
                'text': r['text'],
                'label': r['label'],
                'sbert_score': sb,
                'sbert_call_correct': sb_call == r['label'],
                'qk_score': qk_s,
                'qk_call_correct': qk_call == r['label'],
                'differential': 'QK_only' if qk_call == r['label'] and sb_call != r['label']
                               else 'SBERT_only' if sb_call == r['label'] and qk_call != r['label']
                               else 'both' if qk_call == r['label']
                               else 'neither',
            })

        cell_result = {
            'n_pos': len(pos), 'n_neg': len(neg),
            'tfidf_auc': tfidf_auc,
            'sbert_hyde_auc': aucs['sbert_hyde'],
            'sbert_diff_auc': aucs['sbert_diff'],
            'sbert_best': {'pick': sbert_pick, 'auc': sbert_best},
            'resid_diff_best': {'key': resid_diff_best_k, 'auc': resid_diff_best},
            'qk_pos_best': {'key': qk_pos_best_k, 'auc': qk_pos_best},
            'qk_diff_best': {'key': qk_diff_best_k, 'auc': qk_diff_best},
            'per_chunk': per_chunk,
        }
        out['per_cell'][cell] = cell_result

        # print summary
        print(f'  TFIDF_AUC = {tfidf_auc:.4f}')
        print(f'  SBERT(hyde) = {aucs["sbert_hyde"]:.4f}  SBERT(diff) = {aucs["sbert_diff"]:.4f}')
        print(f'  best resid_diff: {resid_diff_best_k} = {resid_diff_best:.4f}')
        print(f'  best qk_pos:     {qk_pos_best_k} = {qk_pos_best:.4f}')
        print(f'  best qk_diff:    {qk_diff_best_k} = {qk_diff_best:.4f}')
        # tally
        diffs = [c['differential'] for c in per_chunk]
        print(f'  per-chunk tally: QK_only={diffs.count("QK_only")} SBERT_only={diffs.count("SBERT_only")} both={diffs.count("both")} neither={diffs.count("neither")}')

    json.dump(out, open(OUT, 'w'), indent=2, default=str)
    print(f'\n[done] -> {OUT}')


if __name__ == '__main__':
    main()
