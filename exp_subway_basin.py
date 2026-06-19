"""exp_subway_basin: is the NORM special, or is the solution just a narrow basin?

The carrier recites 9/9 only at its trained norm (radial sweep collapses on both
sides). Two explanations: (a) magnitude is a genuine resonance, or (b) the
optimum sits in a narrow basin in ALL directions and rescaling merely leaves it.
Distinguish them: perturb the carrier DIRECTIONALLY at FIXED norm to target
cosine similarities, measure recall. If recall survives moderate angular noise
but dies under rescaling, the norm is special. If small angular noise also kills
it, the basin is just narrow everywhere.
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

c = opt_one(); n0 = c.norm(); unit = c / n0
print(f"\ncarrier norm {n0:.2f}", flush=True)
print("\n" + "="*60)
print("DIRECTIONAL perturbation at FIXED norm: recall vs cos(perturbed, c)")
print("="*60)
print("(median over 3 random perturbation directions per target cosine)")
for target_cos in [1.0, 0.999, 0.99, 0.95, 0.9, 0.8]:
    scores = []
    for trial in range(3):
        g = torch.randn_like(c); g = g - (g @ unit) * unit   # orthogonal component
        g = g / g.norm()
        # build v = a*unit + b*g  with cos(v,c)=target_cos, ||v||=n0
        a = target_cos; b = (1 - target_cos**2) ** 0.5
        v = (a * unit + b * g) * n0
        scores.append(score(recite(v)))
    scores.sort()
    print(f"  cos={target_cos:.3f}  recall(median)={scores[1]}/{N}   trials={scores}")
print("\nCompare to radial sweep (prior exp): 0.5x->2/9, 1.0x->9/9, 1.5x->2/9.")
print("Done.")
