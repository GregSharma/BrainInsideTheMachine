#!/usr/bin/env python3
"""Unbiased feature sweep — no early/late layer prior, ranked by OOS.

Hunch under test: "the model knows what it can't say." So we privilege NOTHING —
not hidden over Q/K, not last-token over pooled, not innovation over raw, not any
layer. Every lawful (object x pool x query x comparator) family is scored at EVERY
layer; winners are ranked by out-of-sample AUC on a fixed stratified split, and we
report WHERE (which layer) each winner lands rather than assuming.

OOM-safe: one model load, capture all activations to CPU, free model, score on CPU.

Objects (chunk & exemplar & query):
  residual:   h, dh, eh        (eh = innovation k=2 of dh over depth)
  attention:  q, dq, eq / k, dk, ek / v, dv, ev
Pools: last, mean, late_mean, max
Query objects: bare question, hyde exemplar mean
Comparators:
  same-object cosine (legal same-space pairs)
  qQ->kX, eqQ->kX, qQ->ekX  (addressing pairs, GQA-mapped)
Datasets: grief, MM
"""
import json, random, time, gc, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from exp_order_asymmetry import QUOTES, auc
from exp_synthetic_probe import sample_exemplars, POS_PROMPT
from exp_mm_relevance import P as PASSAGES, HYDE_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
DRAW_SEED = 12061
K_EX = 8
LADDER_K = 2
NQ, NKV, HD = 16, 2, 128
GS = NQ // NKV

QGRIEF = "Does this sound like something a grieving person would say?"
QMM = ("Would this passage be useful to a market maker quoting two-sided prices on NBA "
       "game outcome contracts, modeling toxicity, adverse selection, inventory risk?")

OBJS = ['h','dh','eh','q','dq','eq','k','dk','ek','v','dv','ev']


def cell_auc(s, cells, pos, negs):
    pr = ([(x,1) for x,c in zip(s,cells) if c==pos] +
          [(x,0) for x,c in zip(s,cells) if c in negs])
    return auc([p[0] for p in pr],[p[1] for p in pr])


def decompose_seq(X, k):
    L = X.shape[0]
    eps = torch.zeros_like(X)
    for l in range(L):
        hist = X[max(0,l-k):l]
        if len(hist)==0:
            eps[l]=X[l]; continue
        B = torch.linalg.qr(hist.T)[0]
        eps[l] = X[l] - B@(B.T@X[l])
    return eps


def pools(T):
    n=len(T); st=max(0,int(0.75*n))
    return {'last':T[-1],'mean':T.mean(0),
            'late_mean':T[st:].mean(0),'max':T.abs().max(0).values*T.mean(0).sign()}


class Hooks:
    def __init__(self,model):
        self.cur={}; self.hooks=[]
        for li,layer in enumerate(model.model.layers):
            a=layer.self_attn
            self.hooks.append(a.q_proj.register_forward_hook(self._m('q',li)))
            self.hooks.append(a.k_proj.register_forward_hook(self._m('k',li)))
            self.hooks.append(a.v_proj.register_forward_hook(self._m('v',li)))
    def _m(self,kind,li):
        def h(m,i,o): self.cur[(kind,li)]=o.detach()[0].float().cpu()
        return h
    def close(self):
        for h in self.hooks: h.remove()


@torch.no_grad()
def encode(model, tok, hooks, text):
    """Return dict obj-> {pool-> (Lp1or L, d)} capturing all 12 objects, all pools."""
    hooks.cur={}
    ids=tok(text,return_tensors='pt').input_ids.to(DEVICE)
    out=model(input_ids=ids,output_hidden_states=True)
    H=[h[0].float().cpu() for h in out.hidden_states]       # list (L+1) of (seq,d)
    nL=len(model.model.layers)
    Q=[hooks.cur[('q',li)] for li in range(nL)]
    K=[hooks.cur[('k',li)] for li in range(nL)]
    V=[hooks.cur[('v',li)] for li in range(nL)]
    del out;
    # pooled per layer
    def stack_pool(seqlist):
        # seqlist: list over layers of (seq,d) -> dict pool-> (nlayer,d)
        pl={p:[] for p in ['last','mean','late_mean','max']}
        for T in seqlist:
            pp=pools(T)
            for p in pl: pl[p].append(pp[p])
        return {p:torch.stack(v) for p,v in pl.items()}
    hp=stack_pool(H); qp=stack_pool(Q); kp=stack_pool(K); vp=stack_pool(V)
    res={}
    for p in ['last','mean','late_mean','max']:
        h=hp[p]; dh=h[1:]-h[:-1]; eh=decompose_seq(dh,LADDER_K)
        q=qp[p]; dq=q[1:]-q[:-1]; eq=decompose_seq(dq,LADDER_K)
        k=kp[p]; dk=k[1:]-k[:-1]; ek=decompose_seq(dk,LADDER_K)
        v=vp[p]; dv=v[1:]-v[:-1]; ev=decompose_seq(dv,LADDER_K)
        res[p]={'h':h,'dh':dh,'eh':eh,'q':q,'dq':dq,'eq':eq,
                'k':k,'dk':dk,'ek':ek,'v':v,'dv':dv,'ev':ev}
    return res


def q2k(vv):
    if vv.numel()!=NQ*HD: return vv
    qh=vv.view(NQ,HD)
    return torch.cat([qh[kv*GS:(kv+1)*GS].mean(0) for kv in range(NKV)],0)


def score_family(qobj, cobjs, layers, map_q2k=False):
    """qobj (Lq,d); cobjs list of (Lc,d). returns per-layer list of score-lists."""
    L=min(qobj.shape[0], cobjs[0].shape[0])
    rows=[]
    for l in range(L):
        qv=q2k(qobj[l]) if map_q2k else qobj[l]
        if qv.norm()<1e-8: rows.append(None); continue
        rows.append([F.cosine_similarity(qv.unsqueeze(0),c[l].unsqueeze(0)).item() for c in cobjs])
    return rows


def run(name, data, question, pos, negs, qbias, model, tok, hooks):
    cells=[x[0] for x in data]; labels=[x[1] for x in data]
    print(f"\n████ {name} ████ capturing...", flush=True)
    chunks=[encode(model,tok,hooks,x[2]) for x in data]
    qenc=encode(model,tok,hooks,question)
    ex=sample_exemplars(model,tok,qbias,K_EX)
    if 'MM' in name: ex=[e for e in ex if len(e)>100]
    exenc=[encode(model,tok,hooks,e) for e in ex]
    print(f"  captured {len(chunks)} chunks, {len(exenc)} exemplars", flush=True)

    rng=random.Random(DRAW_SEED)
    bycell={}
    for i,c in enumerate(cells): bycell.setdefault(c,[]).append(i)
    IS=[];
    for c,idx in bycell.items(): IS+=rng.sample(idx,len(idx)//2)
    OOS=[i for i in range(len(cells)) if i not in set(IS)]
    def sub(s,idx): return auc([s[i] for i in idx],[labels[i] for i in idx])

    def ex_mean(p,o): return torch.stack([e[p][o] for e in exenc]).mean(0)

    families=[]
    for p in ['last','mean','late_mean','max']:
        for o in OBJS:
            families.append((f'{o}|q|{p}', qenc[p][o], [c[p][o] for c in chunks], False))
            families.append((f'{o}|hyde|{p}', ex_mean(p,o), [c[p][o] for c in chunks], False))
        # addressing pairs
        families.append((f'qQ->kX|q|{p}', qenc[p]['q'], [c[p]['k'] for c in chunks], True))
        families.append((f'qQ->kX|hyde|{p}', ex_mean(p,'q'), [c[p]['k'] for c in chunks], True))
        families.append((f'eqQ->kX|q|{p}', qenc[p]['eq'], [c[p]['k'] for c in chunks], True))
        families.append((f'eqQ->kX|hyde|{p}', ex_mean(p,'eq'), [c[p]['k'] for c in chunks], True))
        families.append((f'qQ->ekX|q|{p}', qenc[p]['q'], [c[p]['ek'] for c in chunks], True))

    results={}
    for fn,qo,co,mp in families:
        rows=score_family(qo,co,None,map_q2k=mp)
        # HONEST: select layer on IS only, then report that layer's OOS (no peeking)
        best=None
        for l,s in enumerate(rows):
            if s is None: continue
            isc=sub(s,IS)
            if best is None or isc>best['IS']:
                best={'L':l,'IS':round(isc,4),'OOS':round(sub(s,OOS),4),
                      'full':round(auc(s,labels),4),
                      'uncued':round(cell_auc(s,cells,pos,negs),4)}
        results[fn]=best
    # rank by OOS of the IS-selected layer (honest generalization)
    top=sorted(results.items(),key=lambda kv:kv[1]['OOS'],reverse=True)[:20]
    print(f"  Top 20 (layer chosen on IS, ranked by OOS):", flush=True)
    for k,v in top:
        print(f"   {k:22s} L{v['L']:>2}  IS={v['IS']:.3f} OOS={v['OOS']:.3f} "
              f"full={v['full']:.3f} unc={v['uncued']:.3f}", flush=True)
    # where do winners land? layer histogram of top-20
    layhist={}
    for k,v in top: layhist[v['L']]=layhist.get(v['L'],0)+1
    print(f"  top-20 winning layers: {dict(sorted(layhist.items()))}", flush=True)
    return {'top20':top,'all':results,'layer_hist':layhist}


def main():
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,dtype=torch.float16,device_map=DEVICE,trust_remote_code=True)
    model.eval()
    hooks=Hooks(model)
    out={}
    out['grief']=run('GRIEF',QUOTES,QGRIEF,'grief_uncued',
                     ['cued_nongrief','neutral'],POS_PROMPT,model,tok,hooks)
    out['mm']=run('MM',PASSAGES,QMM,'mm_uncued',
                  ['keyword_trap','neutral'],HYDE_PROMPT,model,tok,hooks)
    hooks.close(); del model; gc.collect(); torch.cuda.empty_cache()
    with open('output/exp_unbiased_sweep.json','w') as f:
        json.dump(out,f,indent=2)
    print(f"\nsaved output/exp_unbiased_sweep.json in {time.time()-t0:.0f}s",flush=True)


if __name__=='__main__':
    main()
