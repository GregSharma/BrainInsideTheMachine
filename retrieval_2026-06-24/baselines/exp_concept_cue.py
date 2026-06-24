#!/usr/bin/env python3
"""Concept-cue vs instructed-question vs hyde (Greg's confound, 2026-06-12).

Worry: the instructed yes/no question collapses the model into a narrow helper
basin that may destroy the latent witness. Also temperature only enters hyde gen.
Test: a bare, ungrammatical CONCEPT CUE ("grief loss mourning"), encoded once,
deterministically, matched by cosine — does it beat the instructed question and
approach hyde? If yes: kills instruction-basin AND temperature confounds at once.

Query objects (all bare-encoded, NO generation, NO temperature):
  q_full   : "Does this sound like something a grieving person would say?"
  q_desc   : "someone who is grieving"
  q_phrase : "grief loss mourning bereavement"
  q_word   : "grief"
  hyde     : 8 sampled exemplars, mean (the stochastic baseline to beat)
Chunk object: residual h (the canonical one), all pools, all layers.
Metrics: 4-cell AUC and HARD CORE (uncued vs cued_nongrief).
"""
import json, random, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from exp_order_asymmetry import QUOTES, auc
from exp_synthetic_probe import sample_exemplars, POS_PROMPT
from exp_unbiased_sweep import Hooks, encode

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
DRAW_SEED = 12061
K_EX = 8

CUES = {
    'q_full': "Does this sound like something a grieving person would say?",
    'q_desc': "someone who is grieving",
    'q_phrase': "grief loss mourning bereavement",
    'q_word': "grief",
}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    hooks = Hooks(model)

    cells = [q[0] for q in QUOTES]
    labels = [q[1] for q in QUOTES]
    chunks = [encode(model, tok, hooks, q[2]) for q in QUOTES]
    cue_enc = {name: encode(model, tok, hooks, text) for name, text in CUES.items()}
    ex = sample_exemplars(model, tok, POS_PROMPT, K_EX)
    exenc = [encode(model, tok, hooks, e) for e in ex]
    hooks.close(); del model; torch.cuda.empty_cache()

    # splits
    rng = random.Random(DRAW_SEED)
    bycell = {}
    for i, c in enumerate(cells): bycell.setdefault(c, []).append(i)
    IS4 = []
    for c, idx in bycell.items(): IS4 += rng.sample(idx, len(idx)//2)
    OOS4 = [i for i in range(len(cells)) if i not in set(IS4)]

    core = [i for i in range(len(cells)) if cells[i] in ('grief_uncued', 'cued_nongrief')]
    clab = {i: (1 if cells[i] == 'grief_uncued' else 0) for i in core}
    rng2 = random.Random(DRAW_SEED)
    cu = [i for i in core if clab[i] == 1]; cn = [i for i in core if clab[i] == 0]
    ISc = rng2.sample(cu, len(cu)//2) + rng2.sample(cn, len(cn)//2)
    OOSc = [i for i in core if i not in set(ISc)]

    def sub4(s, idx): return auc([s[i] for i in idx], [labels[i] for i in idx])
    def subc(s, idx): return auc([s[i] for i in idx], [clab[i] for i in idx])

    def query_obj(name, pool):
        if name == 'hyde':
            return torch.stack([e[pool]['h'] for e in exenc]).mean(0)
        return cue_enc[name][pool]['h']

    print(f"{'cue':>9} {'pool':>10} {'L':>3} | {'4cell IS':>9} {'4cell OOS':>9} | "
          f"{'core IS':>8} {'core OOS':>9}", flush=True)
    out = {}
    for name in list(CUES) + ['hyde']:
        best4 = None; bestc = None
        for pool in ['last', 'mean', 'late_mean', 'max']:
            qo = query_obj(name, pool)
            for l in range(qo.shape[0]):
                if qo[l].norm() < 1e-8: continue
                s = [F.cosine_similarity(qo[l].unsqueeze(0), chunks[i][pool]['h'][l].unsqueeze(0)).item()
                     for i in range(len(chunks))]
                i4 = sub4(s, IS4)
                if best4 is None or i4 > best4[0]:
                    best4 = (i4, sub4(s, OOS4), pool, l)
                ic = subc(s, ISc)
                if bestc is None or ic > bestc[0]:
                    bestc = (ic, subc(s, OOSc), pool, l)
        out[name] = {'4cell': {'IS': round(best4[0],3), 'OOS': round(best4[1],3),
                               'pool': best4[2], 'L': best4[3]},
                     'core': {'IS': round(bestc[0],3), 'OOS': round(bestc[1],3),
                              'pool': bestc[2], 'L': bestc[3]}}
        print(f"{name:>9} {best4[2]:>10} L{best4[3]:>2} | {best4[0]:>9.3f} {best4[1]:>9.3f} | "
              f"core@{bestc[2]}/L{bestc[3]} {bestc[0]:>5.3f} {bestc[1]:>9.3f}", flush=True)

    with open('output/exp_concept_cue.json', 'w') as f:
        json.dump({'cues': CUES, 'results': out}, f, indent=2)
    print("\nsaved output/exp_concept_cue.json", flush=True)


if __name__ == '__main__':
    main()
