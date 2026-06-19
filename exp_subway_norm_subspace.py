"""exp_subway_norm_subspace: WHY does the off-manifold carrier work?

Two follow-ups to the 98.6%-orthogonal / 25x-norm finding:

  (1) NORM-SENSITIVITY: rescale the carrier to a range of magnitudes and recite.
      Is the huge norm load-bearing, or incidental? If recall collapses when
      scaled down to a normal token norm, magnitude is part of the mechanism.

  (2) SUBSPACE: eigen-decompose the token-embedding covariance. Does the carrier
      put its energy in the LOW-variance directions that real tokens barely use
      (which would explain how it is decodable yet orthogonal to tokens)?
      Compare the carrier's energy spectrum to a typical token's.
"""
import sys, re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SENTENCE = "do not lean on car doors on the subway"
INSTR = "Repeat the following sentence back to me, word for word, with nothing else:"
PLACEHOLDER = "@@SENT@@"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32); model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb = model.get_input_embeddings()
W = emb.weight.detach()
WORDS = SENTENCE.split(); N = len(WORDS)
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids

def ps(instr):
    full = tok.apply_chat_template([{"role":"user","content":f'{instr}\n{PLACEHOLDER}'}],
                                   tokenize=False, add_generation_prompt=True)
    a,b = full.split(PLACEHOLDER)
    return tok(a, add_special_tokens=False).input_ids, tok(b, add_special_tokens=False).input_ids
PRE, SUF = ps(INSTR)
PE = emb(torch.tensor([PRE])); SE = emb(torch.tensor([SUF]))
TGT = torch.tensor([SENT_IDS + [tok.eos_token_id]]); TE = emb(TGT)
E = emb(torch.tensor([SENT_IDS])).squeeze(0).detach()

def score(text):
    words = re.findall(r"[a-z]+", text.lower()); truth=[w.lower() for w in WORDS]
    bs,b=0,-1
    for s in range(max(1,len(words)-N+1)):
        win=words[s:s+N]; sc=sum(1 for i in range(min(N,len(win))) if win[i]==truth[i])
        if sc>b: b,bs=sc,s
    win=words[bs:bs+N]
    return sum(1 for i in range(N) if i<len(win) and win[i]==truth[i])

def opt_one(steps=150, lr=0.05):
    P = E.mean(0,keepdim=True).clone().requires_grad_(True)
    o = torch.optim.Adam([P], lr=lr)
    for _ in range(steps):
        o.zero_grad()
        full = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1]+1+SE.shape[1]
        lg = model(inputs_embeds=full).logits[:, ctx-1:ctx-1+TGT.shape[1], :]
        loss = F.cross_entropy(lg.reshape(-1,lg.shape[-1]), TGT.reshape(-1))
        loss.backward(); o.step()
    return P.detach().squeeze(0)

@torch.no_grad()
def recite(vec, max_new=24):
    ce = torch.cat([PE, vec.view(1,1,-1), SE], dim=1)
    am = torch.ones(ce.shape[:2], dtype=torch.long)
    out = model.generate(inputs_embeds=ce, attention_mask=am, max_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()

c = opt_one()
n0 = c.norm()
print(f"\ncarrier norm {n0:.2f}; mean token norm {E.norm(dim=-1).mean():.3f}", flush=True)

print("\n" + "="*60)
print("(1) NORM-SENSITIVITY: rescale carrier direction to target norm")
print("="*60)
unit = c / n0
for scale in [0.42/n0.item(), 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    v = unit * (scale * n0)
    tag = "(=token norm)" if abs(scale - 0.42/n0.item()) < 1e-6 else ""
    print(f"  norm {v.norm():6.2f} ({scale:.2f}x) {tag:14s} -> {score(recite(v))}/{N}")

print("\n" + "="*60)
print("(2) SUBSPACE: carrier energy in token-embedding eigenbasis")
print("="*60)
# covariance eigenbasis of a sample of token embeddings
idx = torch.randperm(W.shape[0])[:20000]
S = W[idx]; mu = S.mean(0); Sc = S - mu
cov = (Sc.T @ Sc) / Sc.shape[0]
evals, evecs = torch.linalg.eigh(cov)          # ascending
evecs = evecs.flip(1); evals = evals.flip(0)   # descending (top variance first)
d = W.shape[1]
def energy_profile(v):
    coeff = evecs.T @ (v - mu*0)               # project (carrier not mean-centered)
    e = coeff**2
    e = e / e.sum()
    tops = [int(0.1*d), int(0.25*d), int(0.5*d)]
    return [e[:t].sum().item() for t in tops], e
tok_vec = E[0]
ce_c, _ = energy_profile(c)
ce_t, _ = energy_profile(tok_vec)
print(f"  cumulative energy in top 10% / 25% / 50% variance directions:")
print(f"    carrier      : {ce_c[0]*100:5.1f}% / {ce_c[1]*100:5.1f}% / {ce_c[2]*100:5.1f}%")
print(f"    token ('do') : {ce_t[0]*100:5.1f}% / {ce_t[1]*100:5.1f}% / {ce_t[2]*100:5.1f}%")
print("  (if carrier's bars are LOWER, it lives more in low-variance / unused directions)")
print("\nDone.")
