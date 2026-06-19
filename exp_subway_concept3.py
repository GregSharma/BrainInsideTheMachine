"""exp_subway_concept3: is the within>across signal REAL? (powered + controlled)

concept2 (flaw fixed) showed same-sentence carriers align more than cross-sentence
(gap ~0.04). Underpowered (n=3) and uncontrolled. Here:
  - 6 sentences x 4 carriers, independent seeds (keep only carriers that recite,
    loss < 0.05);
  - PERMUTATION TEST: shuffle sentence labels, recompute the within-across gap,
    2000x -> p-value;
  - TOKEN-OVERLAP CONTROL: correlate each sentence-pair's Jaccard token overlap
    with its cross-cosine (does shared vocabulary explain alignment?).
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
d = model.config.hidden_size
full = tok.apply_chat_template([{"role":"user","content":f'{INSTR}\n{PLACEHOLDER}'}],
                               tokenize=False, add_generation_prompt=True)
a,b = full.split(PLACEHOLDER)
PE = emb(torch.tensor([tok(a, add_special_tokens=False).input_ids]))
SE = emb(torch.tensor([tok(b, add_special_tokens=False).input_ids]))

_seed = [0]
def opt_one(sid):
    TGT = torch.tensor([sid + [tok.eos_token_id]]); TE = emb(TGT)
    tn = emb(torch.tensor([sid])).squeeze(0).norm(dim=-1).mean()
    torch.manual_seed(90000 + _seed[0]); _seed[0] += 1
    P = (torch.randn(1, d) * tn).requires_grad_(True)
    o = torch.optim.Adam([P], lr=0.05)
    for _ in range(200):
        o.zero_grad()
        f_ = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1]+1+SE.shape[1]
        lg = model(inputs_embeds=f_).logits[:, ctx-1:ctx-1+TGT.shape[1], :]
        loss = F.cross_entropy(lg.reshape(-1,lg.shape[-1]), TGT.reshape(-1))
        loss.backward(); o.step()
    return P.detach().squeeze(0), loss.item()

vecs, labels, sids = [], [], []
for si, s in enumerate(SENTS):
    sid = tok(s, add_special_tokens=False).input_ids; sids.append(set(sid))
    for _ in range(M_EACH):
        c, l = opt_one(sid)
        if l < 0.05:                      # keep only carriers that actually recite
            vecs.append(c); labels.append(si)
    print(f"  sentence {si}: kept {labels.count(si)}/{M_EACH} (last loss {l:.3f})", flush=True)

U = F.normalize(torch.stack(vecs), dim=1)
labels = torch.tensor(labels)
Cos = U @ U.T
n = len(U)

def gap_for(lab):
    same = lab.unsqueeze(0) == lab.unsqueeze(1)
    eye = torch.eye(n, dtype=bool)
    w = Cos[same & ~eye].mean().item()
    a_ = Cos[~same].mean().item()
    return w, a_, w - a_

w, a, gap = gap_for(labels)

# permutation test
torch.manual_seed(0)
null = []
for _ in range(2000):
    perm = labels[torch.randperm(n)]
    null.append(gap_for(perm)[2])
null = torch.tensor(null)
pval = (null >= gap).float().mean().item()

# token-overlap control
pair_overlap, pair_cos = [], []
for i, j in itertools.combinations(range(len(SENTS)), 2):
    jac = len(sids[i] & sids[j]) / len(sids[i] | sids[j])
    mi = (labels == i); mj = (labels == j)
    if mi.any() and mj.any():
        pair_overlap.append(jac)
        pair_cos.append(Cos[mi][:, mj].mean().item())
po = torch.tensor(pair_overlap); pc = torch.tensor(pair_cos)
if len(po) > 2 and po.std() > 0:
    corr = (((po-po.mean())*(pc-pc.mean())).mean() / (po.std()*pc.std())).item()
else:
    corr = float('nan')

print("\n" + "="*60)
print("POWERED + CONTROLLED within>across test")
print("="*60)
print(f"  carriers kept: {n}  across {len(set(labels.tolist()))} sentences")
print(f"  within = {w:+.4f}   across = {a:+.4f}   gap = {gap:+.4f}")
print(f"  permutation null gap: mean {null.mean():+.4f}  std {null.std():.4f}")
print(f"  p-value (null gap >= observed): {pval:.4f}")
print(f"  token-overlap vs cross-cosine correlation: {corr:+.3f}  (high => overlap confound)")
print(f"  pair overlaps: {[round(x,2) for x in pair_overlap]}")
print("Done.")
