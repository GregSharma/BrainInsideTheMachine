"""exp_subway_shell: is S_t a thinned ISOTROPIC SHELL? (cross-validation)

Two parts.

PART 1 (ambient law, no model): sum k iid unit vectors in R^d, many times.
Confirm the CLT geometry that the roots-of-unity toy predicts:
  - ||S_k|| concentrates at ~sqrt(k)
  - directions are uniform on the sphere -> pairwise cosine ~ N(0, 1/sqrt(d))
  - no preferred axis

PART 2 (real carriers): optimize many independent soft-token carriers for the
subway sentence and ask whether they are statistically indistinguishable from
uniform random directions on a high-norm shell:
  - pairwise cosine distribution vs the 1/sqrt(d) null
  - is there a shared/preferred axis? (top singular value of the unit-carrier
    matrix vs the random null)
If carriers look like random shell directions, S_t is a thinned isotropic shell
and the February roots-of-unity graph IS the ambient law.
"""
import sys, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
torch.manual_seed(0)

# ---------------- PART 1: ambient law ----------------
print("="*64)
print("PART 1  ambient law: sum of k unit vectors in R^d (no model)")
print("="*64)
def sim(d, k, trials=3000):
    X = torch.randn(trials, k, d)
    X = X / X.norm(dim=-1, keepdim=True)        # k unit vectors each
    S = X.sum(1)                                 # (trials, d) sums
    norms = S.norm(dim=-1)
    U = S / norms.unsqueeze(-1)                   # directions
    # pairwise cosines among a sample of directions
    G = U[:400] @ U[:400].T
    off = G[~torch.eye(400, dtype=bool)]
    # preferred axis: top eigenvalue fraction of direction covariance
    C = (U.T @ U) / U.shape[0]
    evals = torch.linalg.eigvalsh(C)
    top_frac = (evals.max() / evals.sum()).item()
    return norms.mean().item(), norms.std().item(), off.mean().item(), off.std().item(), top_frac

for d in [896, 2048]:
    for k in [10, 50]:
        mn, sn, cm, cs, tf = sim(d, k)
        print(f"  d={d:>4} k={k:>2}: ||S||={mn:5.2f}±{sn:4.2f} (sqrt(k)={math.sqrt(k):.2f})  "
              f"cos={cm:+.4f}±{cs:.4f} (1/sqrt(d)={1/math.sqrt(d):.4f})  top-axis frac={tf:.4f} (iso~{1/d:.4f})")

# ---------------- PART 2: real carriers ----------------
print("\n" + "="*64)
print("PART 2  real carriers: are they uniform shell directions?")
print("="*64)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SENTENCE = "do not lean on car doors on the subway"
INSTR = "Repeat the following sentence back to me, word for word, with nothing else:"
PLACEHOLDER = "@@SENT@@"
M_CARRIERS = 12

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32); model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb = model.get_input_embeddings()
d = model.config.hidden_size
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids
full = tok.apply_chat_template([{"role":"user","content":f'{INSTR}\n{PLACEHOLDER}'}],
                               tokenize=False, add_generation_prompt=True)
a,b = full.split(PLACEHOLDER)
PE = emb(torch.tensor([tok(a, add_special_tokens=False).input_ids]))
SE = emb(torch.tensor([tok(b, add_special_tokens=False).input_ids]))
TGT = torch.tensor([SENT_IDS + [tok.eos_token_id]]); TE = emb(TGT)
tok_norm = emb(torch.tensor([SENT_IDS])).squeeze(0).norm(dim=-1).mean()

def opt(seed, steps=220, lr=0.05):
    torch.manual_seed(1000+seed)
    P = (torch.randn(1, d) * tok_norm).requires_grad_(True)
    o = torch.optim.Adam([P], lr=lr)
    for _ in range(steps):
        o.zero_grad()
        f_ = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1]+1+SE.shape[1]
        lg = model(inputs_embeds=f_).logits[:, ctx-1:ctx-1+TGT.shape[1], :]
        loss = F.cross_entropy(lg.reshape(-1,lg.shape[-1]), TGT.reshape(-1))
        loss.backward(); o.step()
    return P.detach().squeeze(0), loss.item()

carriers = []
print(f"optimizing {M_CARRIERS} carriers...", flush=True)
for s in range(M_CARRIERS):
    c, l = opt(s)
    carriers.append(c)
    print(f"  carrier {s:>2}: loss {l:.3f}  norm {c.norm():.2f}", flush=True)

Z = torch.stack(carriers)
U = Z / Z.norm(dim=-1, keepdim=True)
G = U @ U.T
off = G[~torch.eye(len(U), dtype=bool)]
C = (U.T @ U) / U.shape[0]
top_frac = (torch.linalg.eigvalsh(C).max() / U.shape[0] * U.shape[0]).item()  # eigval max / sum (=1)
evals = torch.linalg.eigvalsh(C); top_frac = (evals.max()/evals.sum()).item()

# null: same count of random unit dirs in d
Ur = torch.randn(len(U), d); Ur = Ur / Ur.norm(dim=-1, keepdim=True)
Gr = Ur @ Ur.T; offr = Gr[~torch.eye(len(U), dtype=bool)]
Cr = (Ur.T @ Ur)/Ur.shape[0]; tfr = (torch.linalg.eigvalsh(Cr).max()/torch.linalg.eigvalsh(Cr).sum()).item()

print(f"\nd = {d},  1/sqrt(d) = {1/math.sqrt(d):.4f}")
print(f"  carrier norms        : mean {Z.norm(dim=-1).mean():.2f}  range [{Z.norm(dim=-1).min():.2f}, {Z.norm(dim=-1).max():.2f}]")
print(f"  carrier pairwise cos : mean {off.mean():+.4f}  std {off.std():.4f}  max {off.abs().max():.4f}")
print(f"  RANDOM pairwise cos  : mean {offr.mean():+.4f}  std {offr.std():.4f}  max {offr.abs().max():.4f}")
print(f"  carrier top-axis frac: {top_frac:.4f}   RANDOM top-axis frac: {tfr:.4f}   (iso lower ~{1/len(U):.4f})")
print("\nIf carrier rows ~ random rows -> S_t is a thinned isotropic shell.")
print("Done.")
