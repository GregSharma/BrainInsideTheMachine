"""
exp_unified_intent_sweep.py — ONE model load, ONE forward per chunk, ZERO-LEARNING retrieval.

The pitch (Greg, the whole point):
  - Corpus activations are PRECOMPUTED once per chunk. They never change.
  - A new query (e.g. "find sincere apology" / "find grief" / "find flirting") is encoded ONCE.
  - Scoring is a dot product. No fitting, no probing, no per-corpus learning.
  - That's the P-vs-NP asymmetry: index once, query forever.
  - We INCLUDE a supervised linear probe ONLY as a CEILING reference — it's not the headline.

CONTROLS Greg explicitly demanded (v3 update — replaces silent reliance on the LLM L0 readout):
  - TFIDF_AUC — Logistic regression on TF-IDF features (5-fold). The surface-vocabulary baseline.
    If TFIDF beats query-only QK, geometry adds nothing and the leakage audit failed for that cell.
  - SBERT_AUC — sentence-transformers paraphrase-multilingual-MiniLM cos against held-out POS exemplar mean.
    This is "RAG": precomputed embeddings, cosine to a query exemplar mean, NO LLM inference per chunk.
    Greg's prior retrieval work claimed to beat RAG. On v3 anti-leak stimuli we re-prove (or refute) that.
    If RAG also wins, "something changed and faith in this code is lower" — his exact words.

For each chunk we capture once (single forward, all heads/layers cached):
  - last-token residual at each layer L (the "embed" of the chunk)
  - per-(L,H) last-token Q vector and MEAN K vector over chunk tokens (the read/write directions)

For each (cell, query-type, layer, head) we score every chunk WITHOUT learning, then compute AUC.

QUERY TYPES (all use the SAME cached chunk activations; only the query side changes):

  mean_resid_cos:      query = mean residual of K_EXEM held-out POS exemplars at layer L
                       score = cos(chunk_resid_L, query)
                       (your HyDE in activation space)

  diff_resid_cos:      query = (mean POS resid - mean NEG resid) at layer L
                       score = cos(chunk_resid_L, query)
                       (cancels register noise -> the "concept-pure" direction)

  qk_perhead:          for each head h at layer L:
                       query = mean K vector of POS exemplars at (L, h)
                       score = q_chunk[L, h] @ query
                       (your Q/K-attention readout — per head)

  qk_perhead_diff:     query = mean K of POS - mean K of NEG at (L, h)
                       score = q_chunk[L, h] @ query
                       (the concept-pure version, per head)

  qkov_supervised:     LINEAR PROBE on last-tok resid_L — CEILING ONLY, not part of the pitch.

The output JSON gives per (cell, query_type, layer[, head]): AUC + ACC. No k-fold for query-only methods
(query exemplars are held out from scoring, that's the only "split"). Supervised gets k-fold.

Output: output/exp_unified_intent_sweep.json — incremental save per cell.
Local 4070, Qwen2.5-3B fp16 eager.
"""
import os, json, glob, gc, time
import numpy as np
import torch
from numpy.random import default_rng
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen2.5-3B"
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # 384-d, covers EN/ES/ZH
BATTERY = "/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623"
LAYERS = [0, 6, 12, 18, 22, 26, 30, 33, 34]
QK_LAYERS = [18, 22, 26, 27, 30, 33]   # zone where per-head intent specialization lives (project's L27 result)
PATH_LAYERS = [12, 22, 26, 30, 34]      # full per-token path saved here for signature features
K_EXEM = 8                              # POS exemplar pool size (matches Greg's K_EX in exp_synthetic_probe.py)
KFOLD_SUPERVISED = 5
DP_SIG = 16                             # PCA dim for path signature
SEED = 0
OUT_JSON = "output/exp_unified_intent_sweep.json"
DEVICE = "cuda"


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    # handle ties via average rank trick; this matches sklearn AUC well enough
    order = np.argsort(-s, kind='mergesort'); y = y[order]
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return float('nan')
    tps = np.cumsum(y == 1) / P; fps = np.cumsum(y == 0) / N
    return float(np.trapezoid(tps, fps) if hasattr(np, 'trapezoid') else np.trapz(tps, fps))


def acc_at_thresh(scores, labels, thresh=None):
    s = np.asarray(scores); y = np.asarray(labels)
    if thresh is None: thresh = float(np.median(s))
    return float(np.mean((s > thresh) == (y == 1)))


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def load_battery():
    """Load cells. Derive contrast/lang from filename if not in JSON.
    Skip non-cell files (pool, build, gen, iter, _v*)."""
    SKIP = ('_pool', '_gen', 'build_', '_iter')
    cells = {}
    for fp in sorted(glob.glob(f"{BATTERY}/*.json")):
        name = os.path.basename(fp).replace('.json', '')
        if any(s in name for s in SKIP):
            continue
        # name format expected: {contrast}_{lang} or {contrast}_{lang}_v{N}
        # accept 2-token base; reject anything else
        parts = name.split('_')
        if len(parts) < 2:
            continue
        contrast, lang = parts[0], parts[1]
        if lang not in ('en', 'es', 'zh', 'ja'):
            continue
        try:
            d = json.load(open(fp))
            if not isinstance(d, dict):
                continue
            pos = [s for s in d.get("pos", []) if isinstance(s, str) and 4 <= len(s.split()) <= 60]
            neg = [s for s in d.get("neg", []) if isinstance(s, str) and 4 <= len(s.split()) <= 60]
            # ZH is character-script, not word-spaced; relax for ZH
            if lang == 'zh':
                pos = [s for s in d.get("pos", []) if isinstance(s, str) and 8 <= len(s) <= 200]
                neg = [s for s in d.get("neg", []) if isinstance(s, str) and 8 <= len(s) <= 200]
            k = min(len(pos), len(neg))
            if k >= 20:
                cell_key = f"{contrast}_{lang}"
                cells[cell_key] = {"pos": pos[:k], "neg": neg[:k]}
        except Exception as e:
            print(f"  [skip] {fp}: {e}", flush=True)
    return cells


@torch.no_grad()
def capture_one(model, tok, text, layers, qk_layers, path_layers, max_len=64):
    """ONE forward. Cache:
       - last_resid[L] = last-token residual at each requested layer (1D, hidden_size)
       - path[L]       = full per-token residual path at chosen layers (T, hidden_size)
       - qk[L][h]['q'] = last-token Q for head h (1D, d_h)
       - qk[L][h]['k_mean'] = mean over tokens of K for head h (1D, d_h)
    """
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
    qk_buf = {L: {} for L in qk_layers}
    handles = []
    def make_hook(L, kind):
        def h(mod, inp, out):
            qk_buf[L][kind] = out[0].float().cpu().numpy()
        return h
    for L in qk_layers:
        layer = model.model.layers[L].self_attn
        handles.append(layer.q_proj.register_forward_hook(make_hook(L, 'q')))
        handles.append(layer.k_proj.register_forward_hook(make_hook(L, 'k')))
    o = model(input_ids=ids, output_hidden_states=True)
    for h in handles: h.remove()
    last_resid = {L: o.hidden_states[L][0, -1].float().cpu().numpy() for L in layers}
    path = {L: o.hidden_states[L][0].float().cpu().numpy() for L in path_layers}
    n_head = model.config.num_attention_heads
    n_kv = getattr(model.config, 'num_key_value_heads', n_head)
    d_h = model.config.hidden_size // n_head
    qk = {}
    for L in qk_layers:
        q = qk_buf[L]['q']; k = qk_buf[L]['k']            # (T, *)
        T = q.shape[0]
        q_last = q[-1].reshape(n_head, d_h)                # (n_head, d_h)
        k_all = k.reshape(T, n_kv, d_h)
        if n_kv != n_head:
            rep = n_head // n_kv
            k_all = np.repeat(k_all, rep, axis=1)          # (T, n_head, d_h)
        k_mean = k_all.mean(0)                              # (n_head, d_h)
        qk[L] = {'q': q_last, 'k_mean': k_mean}
    return {'last_resid': last_resid, 'path': path, 'qk': qk}


def kfold_linprobe_auc(X, Y, rng, k=KFOLD_SUPERVISED, ridge=30.0):
    idx = rng.permutation(len(Y)); folds = np.array_split(idx, k); aucs, accs = [], []
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(k) if g != f])
        if len(set(Y[tr].tolist())) < 2: continue
        Xtr = X[tr]; Xte = X[te]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))]); Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
        p = Xtr.shape[1]
        w = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(p), Xtr.T @ (2 * Y[tr] - 1.0))
        sc = Xte @ w
        aucs.append(auc(sc, Y[te])); accs.append(float(np.mean((sc > 0) == (Y[te] == 1))))
    return float(np.nanmean(aucs)), float(np.mean(accs))


def main():
    rng = default_rng(SEED)
    cells = load_battery()
    print(f"[battery] {len(cells)} cells: {sorted(cells.keys())}", flush=True)
    if not cells: return

    print(f"[model] loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEVICE,
        trust_remote_code=True, attn_implementation="eager",
    ).eval()
    n_head = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    d_h = model.config.hidden_size // n_head
    print(f"[model] L={n_layers} heads={n_head} d_h={d_h}", flush=True)

    # Sentence-transformer for the RAG baseline. Loads once, lives on CPU.
    print(f"[sbert] loading {SBERT_MODEL}", flush=True)
    sbert = SentenceTransformer(SBERT_MODEL)
    print(f"[sbert] dim={sbert.get_sentence_embedding_dimension()}", flush=True)

    out = {"model": MODEL, "sbert_model": SBERT_MODEL,
           "battery_dir": BATTERY, "layers": LAYERS, "qk_layers": QK_LAYERS,
           "n_head": n_head, "K_EXEM": K_EXEM,
           "note": "Query-only retrieval AUC. Exemplars held out of scoring. Supervised linprobe = CEILING ONLY. "
                   "TFIDF_AUC = surface-vocab control (must be << QK AUC for the geometric claim to hold). "
                   "SBERT_AUC = sentence-transformer RAG baseline. If SBERT >= QK on v3, the QK win shrinks "
                   "(this is the control Greg explicitly demanded — his prior work claimed to beat RAG).",
           "per_cell": {}}
    t0 = time.time()

    for cell, bd in cells.items():
        pos, neg = bd["pos"], bd["neg"]
        texts = pos + neg
        Y = np.array([1] * len(pos) + [0] * len(neg))
        n = len(texts)
        print(f"\n=== {cell}: n_pos={len(pos)} n_neg={len(neg)} ===", flush=True)

        t_cap = time.time()
        caches = [capture_one(model, tok, t, LAYERS, QK_LAYERS, PATH_LAYERS) for t in texts]
        print(f"  [capture] {n} chunks in {time.time()-t_cap:.1f}s", flush=True)

        # held-out exemplars (POS and NEG), pulled once per cell
        idx_pos = np.where(Y == 1)[0]; idx_neg = np.where(Y == 0)[0]
        ex_pos = rng.choice(idx_pos, size=min(K_EXEM, len(idx_pos)//4), replace=False)
        ex_neg = rng.choice(idx_neg, size=min(K_EXEM, len(idx_neg)//4), replace=False)
        score_mask = np.ones(n, dtype=bool)
        score_mask[ex_pos] = False; score_mask[ex_neg] = False
        score_idx = np.where(score_mask)[0]
        Y_score = Y[score_idx]

        cr = {"n": n, "n_pos": len(pos), "n_score": len(score_idx),
              "mean_resid_cos": {}, "diff_resid_cos": {}, "qk_perhead_pos": {},
              "qk_perhead_diff": {}, "path_signature": {}, "linprobe_ceiling": {},
              "controls": {}}

        # ---------- CONTROL #1: TF-IDF surface baseline (5-fold logistic, full text) ----------
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            tf_aucs = []
            for tr, te in skf.split(texts, Y):
                v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, lowercase=True)
                Xtr = v.fit_transform([texts[i] for i in tr])
                Xte = v.transform([texts[i] for i in te])
                clf = LogisticRegression(C=1.0, max_iter=1000)
                clf.fit(Xtr, Y[tr])
                sc = clf.decision_function(Xte)
                tf_aucs.append(auc(sc, Y[te]))
            tfidf_auc_raw = float(np.mean(tf_aucs))
            cr["controls"]["tfidf_auc_raw"] = round(tfidf_auc_raw, 4)
            cr["controls"]["tfidf_auc"] = round(max(tfidf_auc_raw, 1 - tfidf_auc_raw), 4)
            cr["controls"]["tfidf_verdict"] = ("LEAKAGE" if cr["controls"]["tfidf_auc"] > 0.70
                                               else "BORDERLINE" if cr["controls"]["tfidf_auc"] > 0.60
                                               else "CLEAN")
        except Exception as e:
            cr["controls"]["tfidf_error"] = str(e)

        # ---------- CONTROL #2: SBERT RAG baseline — cos against held-out POS exemplar mean ----------
        # This is the THING Greg's retrieval claim was supposed to beat. NO LLM inference per chunk.
        try:
            sb_emb = sbert.encode(texts, batch_size=32, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)
            sb_pos_mean = sb_emb[ex_pos].mean(0); sb_pos_mean = sb_pos_mean / (np.linalg.norm(sb_pos_mean) + 1e-12)
            sb_neg_mean = sb_emb[ex_neg].mean(0); sb_neg_mean = sb_neg_mean / (np.linalg.norm(sb_neg_mean) + 1e-12)
            sb_diff = sb_pos_mean - sb_neg_mean
            sc_sb_hyde = sb_emb[score_idx] @ sb_pos_mean
            sc_sb_diff = sb_emb[score_idx] @ sb_diff
            sb_hyde_raw = auc(sc_sb_hyde, Y_score)
            sb_diff_raw = auc(sc_sb_diff, Y_score)
            cr["controls"]["sbert_hyde_auc_raw"]  = round(sb_hyde_raw, 4)
            cr["controls"]["sbert_diff_auc_raw"]  = round(sb_diff_raw, 4)
            cr["controls"]["sbert_hyde_auc"]      = round(max(sb_hyde_raw, 1 - sb_hyde_raw), 4)
            cr["controls"]["sbert_diff_auc"]      = round(max(sb_diff_raw, 1 - sb_diff_raw), 4)
            cr["controls"]["sbert_best_auc"]      = round(max(cr["controls"]["sbert_hyde_auc"],
                                                              cr["controls"]["sbert_diff_auc"]), 4)
        except Exception as e:
            cr["controls"]["sbert_error"] = str(e)

        print(f"  [controls] TFIDF_AUC={cr['controls'].get('tfidf_auc','?')} "
              f"({cr['controls'].get('tfidf_verdict','?')}) "
              f"SBERT_AUC={cr['controls'].get('sbert_best_auc','?')}", flush=True)

        # ---------- mean_resid_cos and diff_resid_cos (per layer) ----------
        for L in LAYERS:
            pos_mean = np.mean([caches[i]['last_resid'][L] for i in ex_pos], axis=0)
            neg_mean = np.mean([caches[i]['last_resid'][L] for i in ex_neg], axis=0)
            d_dir = pos_mean - neg_mean
            sc_hyde = np.array([cosine(caches[i]['last_resid'][L], pos_mean) for i in score_idx])
            sc_diff = np.array([cosine(caches[i]['last_resid'][L], d_dir) for i in score_idx])
            cr["mean_resid_cos"][str(L)] = {"auc": round(auc(sc_hyde, Y_score), 4),
                                            "acc": round(acc_at_thresh(sc_hyde, Y_score), 4)}
            cr["diff_resid_cos"][str(L)] = {"auc": round(auc(sc_diff, Y_score), 4),
                                            "acc": round(acc_at_thresh(sc_diff, Y_score), 4)}

        # ---------- per-head Q/K: POS-only key and POS-minus-NEG key ----------
        for L in QK_LAYERS:
            kpos = np.mean([caches[i]['qk'][L]['k_mean'] for i in ex_pos], axis=0)   # (H, d_h)
            kneg = np.mean([caches[i]['qk'][L]['k_mean'] for i in ex_neg], axis=0)
            kdir = kpos - kneg
            head_pos = []; head_diff = []
            for h in range(n_head):
                sc_p = np.array([caches[i]['qk'][L]['q'][h] @ kpos[h] for i in score_idx])
                sc_d = np.array([caches[i]['qk'][L]['q'][h] @ kdir[h] for i in score_idx])
                head_pos.append(round(auc(sc_p, Y_score), 4))
                head_diff.append(round(auc(sc_d, Y_score), 4))
            cr["qk_perhead_pos"][str(L)] = {"head_aucs": head_pos,
                                            "best_head": int(np.argmax(np.abs(np.array(head_pos) - 0.5))),
                                            "max_abs_auc": round(max(head_pos, key=lambda x: abs(x - 0.5)), 4)}
            cr["qk_perhead_diff"][str(L)] = {"head_aucs": head_diff,
                                             "best_head": int(np.argmax(np.abs(np.array(head_diff) - 0.5))),
                                             "max_abs_auc": round(max(head_diff, key=lambda x: abs(x - 0.5)), 4)}

        # ---------- path signature (END / MEAN / SIG / SURR), SUPERVISED ridge AUC at PATH_LAYERS ----------
        def sig_lvl12(z):
            dz = np.diff(z, axis=0)
            if len(dz) < 2:
                return np.concatenate([z[-1] - z[0], np.zeros(z.shape[1] ** 2)])
            csum = np.cumsum(dz, axis=0) - dz
            return np.concatenate([z[-1] - z[0], (csum.T @ dz).ravel()])
        for L in PATH_LAYERS:
            Hs = [c['path'][L] for c in caches]
            allh = np.vstack(Hs); mn = allh.mean(0)
            U, S, Vt = np.linalg.svd(allh - mn, full_matrices=False); comp = Vt[:DP_SIG]
            END, MEAN, SIG, SURR = [], [], [], []
            rng_p = default_rng(SEED)
            for H in Hs:
                END.append(H[-1] - mn); MEAN.append(H.mean(0) - mn)
                z = (H - mn) @ comp.T
                SIG.append(sig_lvl12(z))
                dz = np.diff(z, axis=0)
                if len(dz) > 0: rng_p.shuffle(dz)
                z_s = np.vstack([z[0], z[0] + np.cumsum(dz, axis=0)]) if len(dz) > 0 else z
                SURR.append(sig_lvl12(z_s))
            END, MEAN, SIG, SURR = map(np.asarray, (END, MEAN, SIG, SURR))
            ae, ce = kfold_linprobe_auc(END, Y, default_rng(SEED))
            am, cm = kfold_linprobe_auc(MEAN, Y, default_rng(SEED))
            as_, cs = kfold_linprobe_auc(SIG, Y, default_rng(SEED))
            au, cu = kfold_linprobe_auc(SURR, Y, default_rng(SEED))
            cr["path_signature"][str(L)] = {
                "END":  {"auc": round(ae, 4), "acc": round(ce, 4)},
                "MEAN": {"auc": round(am, 4), "acc": round(cm, 4)},
                "SIG":  {"auc": round(as_, 4), "acc": round(cs, 4)},
                "SURR": {"auc": round(au, 4), "acc": round(cu, 4)},
                "SIG_minus_SURR_auc": round(as_ - au, 4),
                "SIG_minus_END_auc":  round(as_ - ae, 4),
            }

        # ---------- supervised linprobe CEILING (last-tok resid per layer) ----------
        for L in LAYERS:
            X = np.stack([c['last_resid'][L] for c in caches])
            a, ac = kfold_linprobe_auc(X, Y, default_rng(SEED))
            cr["linprobe_ceiling"][str(L)] = {"auc": round(a, 4), "acc": round(ac, 4)}

        # ---------- summary: best across families ----------
        best = {}
        for fam in ["mean_resid_cos", "diff_resid_cos", "linprobe_ceiling"]:
            entries = cr[fam]
            bL, bA = max(entries.items(), key=lambda kv: kv[1]["auc"])
            best[fam] = {"layer": bL, **bA}
        # path_signature: best SIG auc
        ps = cr["path_signature"]
        if ps:
            bL, bA = max(ps.items(), key=lambda kv: kv[1]["SIG"]["auc"])
            best["path_signature"] = {"layer": bL, "SIG_auc": bA["SIG"]["auc"],
                                      "END_auc": bA["END"]["auc"], "MEAN_auc": bA["MEAN"]["auc"],
                                      "SURR_auc": bA["SURR"]["auc"],
                                      "SIG_minus_SURR_auc": bA["SIG_minus_SURR_auc"]}
        for fam in ["qk_perhead_pos", "qk_perhead_diff"]:
            entries = cr[fam]
            bL, bA = max(entries.items(), key=lambda kv: kv[1]["max_abs_auc"])
            best[fam] = {"layer": bL, "best_head": bA["best_head"], "max_abs_auc": bA["max_abs_auc"]}
        cr["best"] = best

        # print headline
        ctl = cr.get("controls", {})
        tfa = ctl.get("tfidf_auc", "?"); tfv = ctl.get("tfidf_verdict", "?")
        sba = ctl.get("sbert_best_auc", "?")
        print(f"  [summary] {cell}  ||  CONTROL TFIDF={tfa} [{tfv}]  SBERT={sba}", flush=True)
        # gap of best QK over the controls — the headline number for the geometric claim
        try:
            qk_best = max(best["qk_perhead_pos"]["max_abs_auc"], best["qk_perhead_diff"]["max_abs_auc"])
            if isinstance(tfa, float) or isinstance(tfa, int):
                cr["controls"]["qk_minus_tfidf"] = round(qk_best - tfa, 4)
            if isinstance(sba, float) or isinstance(sba, int):
                cr["controls"]["qk_minus_sbert"] = round(qk_best - sba, 4)
            print(f"    GEOMETRIC GAP: QK_best={qk_best:.4f}  -  TFIDF={tfa}  =  Δ_surface={cr['controls'].get('qk_minus_tfidf','?')}", flush=True)
            print(f"                   QK_best={qk_best:.4f}  -  SBERT={sba}  =  Δ_RAG    ={cr['controls'].get('qk_minus_sbert','?')}", flush=True)
        except Exception:
            pass
        print(f"    QUERY-ONLY mean_resid_cos best L{best['mean_resid_cos']['layer']} AUC={best['mean_resid_cos']['auc']}", flush=True)
        print(f"    QUERY-ONLY diff_resid_cos best L{best['diff_resid_cos']['layer']} AUC={best['diff_resid_cos']['auc']}", flush=True)
        print(f"    QUERY-ONLY qk_perhead_pos  best L{best['qk_perhead_pos']['layer']} head{best['qk_perhead_pos']['best_head']} AUC={best['qk_perhead_pos']['max_abs_auc']}", flush=True)
        print(f"    QUERY-ONLY qk_perhead_diff best L{best['qk_perhead_diff']['layer']} head{best['qk_perhead_diff']['best_head']} AUC={best['qk_perhead_diff']['max_abs_auc']}", flush=True)
        if "path_signature" in best:
            b = best["path_signature"]
            print(f"    SUPERVISED path_sig best L{b['layer']} SIG_AUC={b['SIG_auc']} END_AUC={b['END_auc']} MEAN_AUC={b['MEAN_auc']} SIG-SURR={b['SIG_minus_SURR_auc']:+.4f}", flush=True)
        print(f"    SUPERVISED linprobe ceiling    best L{best['linprobe_ceiling']['layer']} AUC={best['linprobe_ceiling']['auc']}", flush=True)

        out["per_cell"][cell] = cr
        out["wall_s"] = time.time() - t0
        json.dump(out, open(OUT_JSON, "w"), indent=2)
        del caches; gc.collect(); torch.cuda.empty_cache()

    print(f"\n[done] wall {time.time() - t0:.0f}s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
