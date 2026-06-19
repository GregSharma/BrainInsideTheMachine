"""exp_subway_decode: is the per-content carrier direction INTERPRETABLE?

We have: carriers for the same sentence share a direction d_X (content signature).
Question: does d_X align with the model's OWN natural representation of that
sentence m_X (semantics), or is it an orthogonal code?

For each sentence:
  d_X = normalized mean of its carriers (content direction)
  m_X = model's natural representation: mean of last-layer hidden states over the
        sentence tokens (read normally)
Build the cross-alignment matrix cos(d_X, m_Y). If the diagonal (same sentence)
dominates -> the carrier direction tracks the model's semantics (interpretable).
If d_X is ~orthogonal to all m_Y -> it's an off-manifold code (continues the
off-everything theme; more surprising).
"""
import sys, itertools
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INSTR = "Repeat the following sentence back to me, word for word, with nothing else:"
PLACEHOLDER = "@@SENT@@"
M_EACH = 3
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

_seed = [0]
def opt_one(sid):
    TGT = torch.tensor([sid + [tok.eos_token_id]]); TE = emb(TGT)
    tn = emb(torch.tensor([sid])).squeeze(0).norm(dim=-1).mean()
    torch.manual_seed(97000 + _seed[0]); _seed[0] += 1
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

@torch.no_grad()
def meaning_vec(sid):
    """model's natural rep: mean last-layer hidden over the sentence tokens."""
    out = model(input_ids=torch.tensor([sid]), output_hidden_states=True)
    return out.hidden_states[-1][0].mean(0)        # (d,)

d_dirs, m_dirs = [], []
for si, s in enumerate(SENTS):
    sid = tok(s, add_special_tokens=False).input_ids
    cs = [opt_one(sid)[0] for _ in range(M_EACH)]
    d_dirs.append(F.normalize(torch.stack(cs).mean(0), dim=0))
    m_dirs.append(F.normalize(meaning_vec(sid), dim=0))
    print(f"  sentence {si} done", flush=True)

D = torch.stack(d_dirs); M = torch.stack(m_dirs)
X = D @ M.T                                          # cos(d_X, m_Y)
print("\n" + "="*60)
print("cross-alignment  cos(carrier-dir d_X , model-meaning m_Y)")
print("="*60)
print("      " + " ".join(f"m{j}" for j in range(len(SENTS))))
for i in range(len(SENTS)):
    print(f"  d{i} " + " ".join(f"{X[i,j]:+.2f}" for j in range(len(SENTS))))
diag = X.diag().mean().item()
off = X[~torch.eye(len(SENTS), dtype=bool)].mean().item()
print(f"\n  diagonal (same)   mean = {diag:+.4f}")
print(f"  off-diag (diff)   mean = {off:+.4f}")
print(f"  diag - off = {diag-off:+.4f}")
# also: do model-meaning vectors themselves separate? (sanity)
MM = M @ M.T; mm_off = MM[~torch.eye(len(SENTS),dtype=bool)].mean().item()
print(f"  (sanity) mean off-diag cos among model-meaning vecs = {mm_off:+.3f}")
print("\n  diag>>off => carrier dir tracks model semantics (interpretable)")
print("  diag~off~0 => carrier dir is an off-manifold code")
print("Done.")
