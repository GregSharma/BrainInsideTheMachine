"""exp_subway_lens: where does the trained vector POINT? Logit-lens the SGD
soft token through the layers and contrast with the failing mean-pool vector.

We inject the single vector at one position in the recite prompt, capture that
position's residual stream at every layer, and logit-lens it (final norm +
lm_head -> vocabulary). This shows whether the trained off-lattice vector
'unfolds' into readable token directions as it climbs the stack, and at which
layer it first looks like the sentence -- vs the naive vector that never does.
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
NL = model.config.num_hidden_layers
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
E = emb(torch.tensor([SENT_IDS])).squeeze(0).detach()

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
    return P.detach()

slot_pos = PE.shape[1]
def lens_trajectory(vec):
    """Return per-layer top-1 token at the slot position, logit-lensed."""
    ce = torch.cat([PE, vec.view(1,1,-1), SE], dim=1)
    with torch.no_grad():
        out = model(inputs_embeds=ce, output_hidden_states=True)
    norm, head = model.model.norm, model.lm_head
    traj = []
    for L in range(NL+1):
        h = out.hidden_states[L][0, slot_pos, :]
        with torch.no_grad():
            lg = head(norm(h))
        top = lg.topk(3).indices.tolist()
        traj.append([tok.decode([t]).strip() for t in top])
    return traj

opt = opt_one()
mean = E.mean(0)
print("loaded; opt vector trained (recites 9/9 by construction).", flush=True)
print("\n" + "="*72)
print("LOGIT-LENS of the slot position, layer by layer  (top-1 / top-3)")
print("="*72)
to = lens_trajectory(opt); tm = lens_trajectory(mean)
print(f"{'L':>3}  {'OPT (works)':<28} {'MEANPOOL (fails)':<28}")
for L in range(NL+1):
    print(f"{L:>3}  {str(to[L]):<28} {str(tm[L]):<28}")
print("\nNote: the sentence starts 'do not lean ...' — watch which column ever shows it.")
print("Done.")
