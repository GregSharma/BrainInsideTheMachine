"""exp_subway_multiplicity: is the carrier THE point, or one of many?

The solution is a sharp isolated basin. Does the model admit ONE such carrier
for this sentence, or many mutually-distinct ones? Optimize several carriers
from different random initialisations; check each recites 9/9, then measure
pairwise cosine similarity. Near-orthogonal distinct solutions => the embedding
space holds a large constellation of isolated codes for the same content
(high capacity of carriers); collapse to one direction => the carrier is unique.
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

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32); model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb = model.get_input_embeddings()
WORDS = SENTENCE.split(); N = len(WORDS)
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids

def psf(instr):
    full = tok.apply_chat_template([{"role":"user","content":f'{instr}\n{PLACEHOLDER}'}],
                                   tokenize=False, add_generation_prompt=True)
    a,b = full.split(PLACEHOLDER)
    return tok(a, add_special_tokens=False).input_ids, tok(b, add_special_tokens=False).input_ids
PRE, SUF = psf(INSTR)
PE = emb(torch.tensor([PRE])); SE = emb(torch.tensor([SUF]))
TGT = torch.tensor([SENT_IDS + [tok.eos_token_id]]); TE = emb(TGT)
E = emb(torch.tensor([SENT_IDS])).squeeze(0).detach()
mean_emb = E.mean(0)

def score(text):
    words = re.findall(r"[a-z]+", text.lower()); truth=[w.lower() for w in WORDS]
    bs,b=0,-1
    for s in range(max(1,len(words)-N+1)):
        win=words[s:s+N]; sc=sum(1 for i in range(min(N,len(win))) if win[i]==truth[i])
        if sc>b: b,bs=sc,s
    win=words[bs:bs+N]
    return sum(1 for i in range(N) if i<len(win) and win[i]==truth[i])

def opt_from(init, steps=200, lr=0.05):
    P = init.clone().reshape(1, -1).requires_grad_(True)
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

carriers = []
print("\noptimizing carriers from different inits...", flush=True)
for s in range(4):
    torch.manual_seed(100 + s)
    if s == 0:
        init = mean_emb.clone()                       # the usual warm start
    else:
        init = torch.randn_like(mean_emb) * E.norm(dim=-1).mean()  # random init
    c = opt_from(init)
    carriers.append(c)
    print(f"  carrier {s}: recall {score(recite(c))}/{N}  norm {c.norm():.2f}", flush=True)

print("\npairwise cosine similarity between carriers:")
print("     " + " ".join(f"c{j}" for j in range(len(carriers))))
for i, ci in enumerate(carriers):
    row = [F.cosine_similarity(ci.unsqueeze(0), cj.unsqueeze(0)).item() for cj in carriers]
    print(f"  c{i} " + " ".join(f"{v:+.2f}" for v in row))
print("\nDone.")
