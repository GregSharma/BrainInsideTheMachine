"""exp_subway_concept4: length control for the content-signature finding.

concept3 confirmed within>across (p<0.0005), controlled for token overlap. Last
confound: sentence LENGTH (within-pairs share length trivially). Here every
recitation target is TRUNCATED to the same token length K, so length is constant
across all carriers. If within>across persists, length is ruled out and the
signature is content-specific.
"""
import sys, itertools
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INSTR = "Repeat the following sentence back to me, word for word, with nothing else:"
PLACEHOLDER = "@@SENT@@"
M_EACH = 4
SENTS = [
    "do not lean on car doors on the subway",
    "the early train always arrives before sunrise today",
    "please water the garden plants every single morning",
    "loud thunder frightened the small dog last night",
    "she carefully painted the wooden fence bright blue",
    "our team finally won the championship game in overtime",
]

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32); model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb = model.get_input_embeddings()
dd = model.config.hidden_size
full = tok.apply_chat_template([{"role":"user","content":f'{INSTR}\n{PLACEHOLDER}'}],
                               tokenize=False, add_generation_prompt=True)
a,b = full.split(PLACEHOLDER)
PE = emb(torch.tensor([tok(a, add_special_tokens=False).input_ids]))
SE = emb(torch.tensor([tok(b, add_special_tokens=False).input_ids]))

raw = [tok(s, add_special_tokens=False).input_ids for s in SENTS]
K = min(len(r) for r in raw)
targets = [r[:K] for r in raw]                  # all length K -> length controlled
print(f"length-controlled K = {K} tokens; sentence raw lengths {[len(r) for r in raw]}", flush=True)

_seed = [0]
def opt_one(sid):
    TGT = torch.tensor([sid + [tok.eos_token_id]]); TE = emb(TGT)
    tn = emb(torch.tensor([sid])).squeeze(0).norm(dim=-1).mean()
    torch.manual_seed(95000 + _seed[0]); _seed[0] += 1
    P = (torch.randn(1, dd) * tn).requires_grad_(True)
    o = torch.optim.Adam([P], lr=0.05)
    for _ in range(200):
        o.zero_grad()
        f_ = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1]+1+SE.shape[1]
        lg = model(inputs_embeds=f_).logits[:, ctx-1:ctx-1+TGT.shape[1], :]
        loss = F.cross_entropy(lg.reshape(-1,lg.shape[-1]), TGT.reshape(-1))
        loss.backward(); o.step()
    return P.detach().squeeze(0), loss.item()

vecs, labels = [], []
for si, sid in enumerate(targets):
    for _ in range(M_EACH):
        c, l = opt_one(sid)
        if l < 0.05: vecs.append(c); labels.append(si)
    print(f"  target {si}: kept {labels.count(si)}/{M_EACH} (loss {l:.3f})", flush=True)

U = F.normalize(torch.stack(vecs), dim=1); labels = torch.tensor(labels); Cos = U @ U.T; n = len(U)
def gap_for(lab):
    same = lab.unsqueeze(0)==lab.unsqueeze(1); eye = torch.eye(n, dtype=bool)
    return Cos[same&~eye].mean().item(), Cos[~same].mean().item()
w, a = gap_for(labels); gap = w - a
torch.manual_seed(0)
null = torch.tensor([ (lambda L: gap_for(L)[0]-gap_for(L)[1])(labels[torch.randperm(n)]) for _ in range(2000)])
pval = (null >= gap).float().mean().item()
print("\n" + "="*60)
print("LENGTH-CONTROLLED within>across test")
print("="*60)
print(f"  carriers {n}, all targets length {K}")
print(f"  within = {w:+.4f}  across = {a:+.4f}  gap = {gap:+.4f}")
print(f"  permutation null gap {null.mean():+.4f} +/- {null.std():.4f};  p = {pval:.4f}")
print("  gap still positive & significant => content, not length.")
print("Done.")
