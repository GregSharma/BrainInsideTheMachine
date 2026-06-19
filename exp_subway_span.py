"""exp_subway_span: is the carrier a SUPERPOSITION of the token directions?

The trained vector recites 9/9 but is off-lattice, off-manifold, off-readout.
Central question (Greg's VSA intuition): is it nonetheless built FROM the 9
sentence-token embedding directions (a reweighted bundle + correction), or is
it orthogonal/exotic? Least-squares project the carrier onto span(E_sentence)
and measure the residual fraction. Controls: mean-pool (in-span by
construction), and span of 9 RANDOM token embeddings.
"""
import sys
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
WORDS = SENTENCE.split()
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids
N = len(SENT_IDS)

def ps(instr):
    full = tok.apply_chat_template([{"role":"user","content":f'{instr}\n{PLACEHOLDER}'}],
                                   tokenize=False, add_generation_prompt=True)
    a,b = full.split(PLACEHOLDER)
    return tok(a, add_special_tokens=False).input_ids, tok(b, add_special_tokens=False).input_ids
PRE, SUF = ps(INSTR)
PE = emb(torch.tensor([PRE])); SE = emb(torch.tensor([SUF]))
TGT = torch.tensor([SENT_IDS + [tok.eos_token_id]]); TE = emb(TGT)
E = emb(torch.tensor([SENT_IDS])).squeeze(0).detach()  # (9, d)

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

def resid_frac(c, basis):
    """fraction of c's norm NOT explained by least-squares fit in span(basis rows)."""
    B = basis.T                          # (d, k)
    a = torch.linalg.lstsq(B, c).solution
    fit = B @ a
    return (c - fit).norm().item() / c.norm().item(), a

c = opt_one()
mean = E.mean(0)
rand_ids = torch.randint(1000, W.shape[0]-1000, (N,))
Erand = W[rand_ids]

print(f"\ncarrier ||c|| = {c.norm():.2f}   mean token norm = {E.norm(dim=-1).mean():.2f}")
rf_sent, a_sent = resid_frac(c, E)
rf_mean, _      = resid_frac(mean, E)
rf_rand, _      = resid_frac(c, Erand)
print("\n" + "="*60)
print("Is the carrier a superposition of the 9 token directions?")
print("="*60)
print(f"  residual NOT in span(sentence embeddings) : {rf_sent*100:5.1f}%  -> {100-rf_sent*100:.1f}% explained")
print(f"  (control) mean-pool vec, same span        : {rf_mean*100:5.1f}%")
print(f"  (control) carrier vs 9 RANDOM embeddings   : {rf_rand*100:5.1f}%")
print("\nLeast-squares coefficients on each sentence token:")
for w, a in zip(WORDS, a_sent.tolist()):
    print(f"    {w:8s} {a:+.3f}")
print("\nPer-token cosine(carrier, embedding):")
cos = F.cosine_similarity(c.unsqueeze(0), E, dim=-1)
for w, cv in zip(WORDS, cos.tolist()):
    print(f"    {w:8s} {cv:+.3f}")
print("\nDone.")
