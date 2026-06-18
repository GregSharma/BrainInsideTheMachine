"""exp_subway_capacity: how many tokens fit in ONE injected vector?

Pull the thread: stop guessing, measure the capacity curve. For a single
optimized vector (k=1) injected in place of an n-token target, sweep n and
measure free-generation reconstruction. Two regimes:

  NATURAL : first-n tokens of an English paragraph (linguistically compressible)
  RANDOM  : n random vocabulary tokens (worst case, ~log2(V) bits each)

JL / superposition prediction: a d-dim vector linearly decodes ~ d/log2(V)
near-orthogonal slots before interference crosses the argmax margin.
For d=896, log2(V)~17 -> k* ~ 50 RANDOM tokens. NATURAL should go much further
(redundancy). We also do a min-k sweep: fix a long target, grow k, find the
smallest number of vectors that reconstructs it.

Metric: free-running greedy generation token match vs target (no teacher forcing).
"""
import sys, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INSTR = "Repeat the following exactly, with nothing else:"
PLACEHOLDER = "@@SENT@@"
torch.manual_seed(0); random.seed(0)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32); model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb = model.get_input_embeddings()
d = model.config.hidden_size
V = emb.weight.shape[0]
import math
print(f"loaded d={d} V={V} log2V={math.log2(V):.1f}  JL k*~{d/math.log2(V):.0f} (random)", flush=True)

NATURAL = ("The transformer reads a sentence as a sequence of discrete tokens, "
           "but inside the network every token becomes a continuous vector that "
           "flows through many layers of attention and feed forward blocks before "
           "the final layer projects it back onto the vocabulary to choose the next word.")
NAT_IDS = tok(NATURAL, add_special_tokens=False).input_ids
print(f"natural pool = {len(NAT_IDS)} tokens", flush=True)

def prefix_suffix():
    full = tok.apply_chat_template([{"role":"user","content":f'{INSTR}\n{PLACEHOLDER}'}],
                                   tokenize=False, add_generation_prompt=True)
    a,b = full.split(PLACEHOLDER)
    return (torch.tensor([tok(a, add_special_tokens=False).input_ids]),
            torch.tensor([tok(b, add_special_tokens=False).input_ids]))
PRE, SUF = prefix_suffix()
with torch.no_grad(): PE, SE = emb(PRE), emb(SUF)

def optimize(target_ids, k, steps=250, lr=0.05):
    tgt = torch.tensor([target_ids + [tok.eos_token_id]])
    with torch.no_grad(): TE = emb(tgt)
    # warm start: mean-pool target embeddings into k slots
    Etg = emb(torch.tensor([target_ids])).squeeze(0).detach()
    idx = torch.linspace(0, len(target_ids), k+1).long()
    init = torch.stack([Etg[idx[i]:max(idx[i+1],idx[i]+1)].mean(0) for i in range(k)])
    P = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        full = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1] + k + SE.shape[1]
        logits = model(inputs_embeds=full).logits
        pred = logits[:, ctx-1:ctx-1+tgt.shape[1], :]
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1))
        loss.backward(); opt.step()
    return P.detach(), loss.item()

@torch.no_grad()
def gen_match(P, target_ids):
    ce = torch.cat([PE, P.unsqueeze(0), SE], dim=1)
    am = torch.ones(ce.shape[:2], dtype=torch.long)
    out = model.generate(inputs_embeds=ce, attention_mask=am,
                         max_new_tokens=len(target_ids)+4, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    g = out[0].tolist()[:len(target_ids)]
    matches = [(i < len(g) and g[i]==target_ids[i]) for i in range(len(target_ids))]
    # first mismatch position
    fm = next((i for i,m in enumerate(matches) if not m), len(target_ids))
    return sum(matches), fm

print("\n" + "="*72)
print("CAPACITY of a SINGLE vector (k=1): tokens reconstructed vs length")
print("="*72)
print(f"{'n':>4} {'natural':>14} {'random':>14}")
for n in [4, 8, 12, 16, 24, 32, 48, 64]:
    if n > len(NAT_IDS): break
    nat = NAT_IDS[:n]
    rnd = [random.randint(1000, V-1000) for _ in range(n)]
    Pn,_ = optimize(nat, 1); cn, fmn = gen_match(Pn, nat)
    Pr,_ = optimize(rnd, 1); cr, fmr = gen_match(Pr, rnd)
    print(f"{n:>4} {f'{cn}/{n} (fm@{fmn})':>14} {f'{cr}/{n} (fm@{fmr})':>14}")

print("\n" + "="*72)
print("MIN-k: fix a 32-token natural target, grow k until full reconstruction")
print("="*72)
tgt32 = NAT_IDS[:32]
for k in [1, 2, 3, 4, 6, 8]:
    P,l = optimize(tgt32, k, steps=300); c, fm = gen_match(P, tgt32)
    print(f"  k={k:>2} ({32/k:.1f}x): {c}/32 reconstructed (fm@{fm})  loss={l:.4f}")
print("\nDone.")
