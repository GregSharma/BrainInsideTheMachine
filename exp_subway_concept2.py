"""exp_subway_concept2: definitive content-structure test (flaw fixed).

The first concept test had a flaw: carrier s of sentence A and of B shared an
init seed, inflating across-sentence cosine. Here: 3 sentences, FULLY INDEPENDENT
seeds. And the sharper question: after PROJECTING OUT the global shared
"recite-mode" axis, is there residual content structure (within > across)?

Reports:
  - raw within-sentence vs across-sentence mean cosine
  - the 3 per-sentence mean-direction pairwise cosines (the shared axis)
  - after removing the global mean carrier direction: within vs across on residuals
If residual within > across -> content structure survives the generic axis.
If residual within ~ across -> content geometry is isotropic (mundane, robust).
"""
import sys, re, itertools
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INSTR = "Repeat the following sentence back to me, word for word, with nothing else:"
PLACEHOLDER = "@@SENT@@"
M_EACH = 6
SENTS = [
    "do not lean on car doors on the subway",
    "the early train always arrives before sunrise today",
    "please water the garden plants every single morning",
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
    torch.manual_seed(50000 + _seed[0]); _seed[0] += 1   # globally unique seed
    P = (torch.randn(1, d) * tn).requires_grad_(True)
    o = torch.optim.Adam([P], lr=0.05)
    for _ in range(220):
        o.zero_grad()
        f_ = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1]+1+SE.shape[1]
        lg = model(inputs_embeds=f_).logits[:, ctx-1:ctx-1+TGT.shape[1], :]
        loss = F.cross_entropy(lg.reshape(-1,lg.shape[-1]), TGT.reshape(-1))
        loss.backward(); o.step()
    return P.detach().squeeze(0), loss.item()

groups = []
for si, s in enumerate(SENTS):
    sid = tok(s, add_special_tokens=False).input_ids
    Z = []
    for _ in range(M_EACH):
        c, l = opt_one(sid); Z.append(c)
    Z = torch.stack(Z)
    groups.append(Z)
    print(f"  sentence {si} ({len(sid)} tok): {M_EACH} carriers, last loss {l:.3f}", flush=True)

def within_across(units_per_group):
    wt = []
    for U in units_per_group:
        G = U @ U.T; wt.append(G[~torch.eye(len(U), dtype=bool)].mean().item())
    ac = []
    for i,j in itertools.combinations(range(len(units_per_group)), 2):
        ac.append((units_per_group[i] @ units_per_group[j].T).mean().item())
    return sum(wt)/len(wt), sum(ac)/len(ac)

units = [Z / Z.norm(dim=-1, keepdim=True) for Z in groups]
w_raw, a_raw = within_across(units)
mean_dirs = [F.normalize(U.mean(0), dim=0) for U in units]
md_cos = [F.cosine_similarity(mean_dirs[i].unsqueeze(0), mean_dirs[j].unsqueeze(0)).item()
          for i,j in itertools.combinations(range(len(SENTS)),2)]

# project out the GLOBAL mean carrier direction, then re-measure
allU = torch.cat(units, 0)
g = F.normalize(allU.mean(0), dim=0)
res = [F.normalize(U - (U @ g).unsqueeze(1) * g, dim=1) for U in units]
w_res, a_res = within_across(res)

print("\n" + "="*60)
print("DEFINITIVE content-structure test (3 sentences, indep seeds)")
print("="*60)
print(f"  1/sqrt(d) = {1/d**0.5:.4f}")
print(f"  RAW      within = {w_raw:+.4f}   across = {a_raw:+.4f}   gap = {w_raw-a_raw:+.4f}")
print(f"  mean-dir pairwise cosines (shared axis): {[round(x,3) for x in md_cos]}")
print(f"  AFTER removing global axis:")
print(f"  RESIDUAL within = {w_res:+.4f}   across = {a_res:+.4f}   gap = {w_res-a_res:+.4f}")
print("  residual within > across => content structure survives (interesting)")
print("  residual within ~ across => content geometry isotropic (mundane, robust)")
print("Done.")
