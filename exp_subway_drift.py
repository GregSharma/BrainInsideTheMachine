"""exp_subway_drift: does the slot 'freeze' at high norm? (test the refined story)

After refuting the attention-concentration theory, the new hypothesis: the norm
controls the balance between the injected direction and the per-layer updates.
Prediction across a norm sweep of a trained carrier:
  - HIGH norm  -> slot residual stays ~= raw z (updates negligible): little
                 layer-to-layer drift, high cos(z, h_final)  [FROZEN]
  - LOW norm   -> updates swamp z: large relative drift, low cos(z, h_final)
  - OPTIMUM    -> intermediate drift, AND best recall.
We capture the slot residual at every layer and measure drift + freezing.
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
NL = model.config.num_hidden_layers
WORDS = SENTENCE.split(); N = len(WORDS)
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids
full = tok.apply_chat_template([{"role":"user","content":f'{INSTR}\n{PLACEHOLDER}'}],
                               tokenize=False, add_generation_prompt=True)
a,b = full.split(PLACEHOLDER)
PRE = tok(a, add_special_tokens=False).input_ids; SUF = tok(b, add_special_tokens=False).input_ids
PE = emb(torch.tensor([PRE])); SE = emb(torch.tensor([SUF]))
TGT = torch.tensor([SENT_IDS + [tok.eos_token_id]]); TE = emb(TGT)
E = emb(torch.tensor([SENT_IDS])).squeeze(0).detach()
slot_pos = len(PRE)

def score(text):
    words = re.findall(r"[a-z]+", text.lower()); truth=[w.lower() for w in WORDS]
    bs,bb=0,-1
    for s in range(max(1,len(words)-N+1)):
        win=words[s:s+N]; sc=sum(1 for i in range(min(N,len(win))) if win[i]==truth[i])
        if sc>bb: bb,bs=sc,s
    win=words[bs:bs+N]
    return sum(1 for i in range(N) if i<len(win) and win[i]==truth[i])

def opt_one(steps=150, lr=0.05):
    P = E.mean(0,keepdim=True).clone().requires_grad_(True)
    o = torch.optim.Adam([P], lr=lr)
    for _ in range(steps):
        o.zero_grad()
        f_ = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1]+1+SE.shape[1]
        lg = model(inputs_embeds=f_).logits[:, ctx-1:ctx-1+TGT.shape[1], :]
        loss = F.cross_entropy(lg.reshape(-1,lg.shape[-1]), TGT.reshape(-1))
        loss.backward(); o.step()
    return P.detach().squeeze(0)

@torch.no_grad()
def drift(vec):
    ce = torch.cat([PE, vec.view(1,1,-1), SE], dim=1)
    out = model(inputs_embeds=ce, output_hidden_states=True)
    H = [h[0, slot_pos, :] for h in out.hidden_states]   # NL+1 states at slot
    z = H[0]
    rel = [ (H[i+1]-H[i]).norm().item() / H[i].norm().item() for i in range(len(H)-1) ]
    cos_in_final = F.cosine_similarity(z.unsqueeze(0), H[-1].unsqueeze(0)).item()
    return sum(rel)/len(rel), cos_in_final

@torch.no_grad()
def recite(vec, max_new=24):
    ce = torch.cat([PE, vec.view(1,1,-1), SE], dim=1)
    am = torch.ones(ce.shape[:2], dtype=torch.long)
    out = model.generate(inputs_embeds=ce, attention_mask=am, max_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()

c = opt_one(); n0 = c.norm(); unit = c / n0
print(f"\ncarrier norm {n0:.2f}", flush=True)
print("\n" + "="*64)
print("slot residual drift & freezing vs norm")
print("="*64)
print(f"{'norm':>7} {'x':>5} {'mean rel drift/layer':>21} {'cos(z,h_final)':>15} {'recall':>7}")
for scale in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    v = unit * (scale * n0)
    rel, cif = drift(v); rc = score(recite(v))
    print(f"{v.norm():7.2f} {scale:5.2f} {rel:21.4f} {cif:15.3f} {rc:5d}/{N}", flush=True)
print("\nFrozen (high cos(z,h_final), low drift) at high norm => slot can't be sculpted.")
print("Done.")
