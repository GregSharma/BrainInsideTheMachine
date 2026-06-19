"""exp_subway_concept: is the carriers' shared axis ABOUT THE SENTENCE, or just
generic recite/prefix geometry? (the decisive novel-vs-mundane test)

Carriers for one sentence share a small positive component (exp_subway_shell).
Confound: same task + same prefix/suffix could create that axis regardless of
content. Decisive test: optimize carriers for TWO different sentences (same
prefix/suffix) and compare:

  within-A cosine  vs  within-B cosine  vs  across-A-B cosine

If within > across  -> the shared structure is SENTENCE-SPECIFIC (a concept
direction on the shell = the interesting result).
If within ~ across  -> it is generic recite/prefix geometry (mundane).
We also report the cosine between the per-sentence MEAN carrier directions and
whether projecting out the global shared axis leaves a per-sentence remainder.
"""
import sys, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INSTR = "Repeat the following sentence back to me, word for word, with nothing else:"
PLACEHOLDER = "@@SENT@@"
M_EACH = 7
SENT_A = "do not lean on car doors on the subway"
SENT_B = "the early train always arrives before sunrise today"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32); model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb = model.get_input_embeddings()
d = model.config.hidden_size
full = tok.apply_chat_template([{"role":"user","content":f'{INSTR}\n{PLACEHOLDER}'}],
                               tokenize=False, add_generation_prompt=True)
a,b = full.split(PLACEHOLDER)
PE = emb(torch.tensor([tok(a, add_special_tokens=False).input_ids]))
SE = emb(torch.tensor([tok(b, add_special_tokens=False).input_ids]))

def carriers_for(sentence, m):
    sid = tok(sentence, add_special_tokens=False).input_ids
    TGT = torch.tensor([sid + [tok.eos_token_id]]); TE = emb(TGT)
    tn = emb(torch.tensor([sid])).squeeze(0).norm(dim=-1).mean()
    out = []
    for s in range(m):
        torch.manual_seed(7000 + s)
        P = (torch.randn(1, d) * tn).requires_grad_(True)
        o = torch.optim.Adam([P], lr=0.05)
        for _ in range(220):
            o.zero_grad()
            f_ = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
            ctx = PE.shape[1]+1+SE.shape[1]
            lg = model(inputs_embeds=f_).logits[:, ctx-1:ctx-1+TGT.shape[1], :]
            loss = F.cross_entropy(lg.reshape(-1,lg.shape[-1]), TGT.reshape(-1))
            loss.backward(); o.step()
        out.append(P.detach().squeeze(0))
    print(f"  done {sentence[:30]!r} ({m} carriers, last loss {loss.item():.3f})", flush=True)
    return torch.stack(out)

print(f"optimizing carriers (1/sqrt(d)={1/math.sqrt(d):.4f})...", flush=True)
ZA = carriers_for(SENT_A, M_EACH)
ZB = carriers_for(SENT_B, M_EACH)
UA = ZA / ZA.norm(dim=-1, keepdim=True)
UB = ZB / ZB.norm(dim=-1, keepdim=True)

def mean_offdiag(G):
    return G[~torch.eye(G.shape[0], dtype=bool)].mean().item()

within_A = mean_offdiag(UA @ UA.T)
within_B = mean_offdiag(UB @ UB.T)
across   = (UA @ UB.T).mean().item()

print("\n" + "="*60)
print("DECISIVE TEST: within-sentence vs across-sentence cosine")
print("="*60)
print(f"  within-A (subway)   mean cos = {within_A:+.4f}")
print(f"  within-B (train)    mean cos = {within_B:+.4f}")
print(f"  across  A vs B      mean cos = {across:+.4f}")
print(f"  cos(meanDirA, meanDirB)      = {F.cosine_similarity(UA.mean(0,keepdim=True), UB.mean(0,keepdim=True)).item():+.4f}")
gap = (within_A + within_B)/2 - across
print(f"  within - across gap          = {gap:+.4f}")
print("  within >> across  => sentence-specific structure (interesting)")
print("  within  ~ across  => generic recite/prefix geometry (mundane)")
print("Done.")
