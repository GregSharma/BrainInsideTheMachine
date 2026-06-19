"""exp_subway_attention: is the norm resonance the softmax/vMF CONCENTRATION knob?

Theory claim: the slot is read via softmax attention exp(q . W_K z); scaling ||z||
scales the logit -> ||z|| acts as the vMF concentration kappa. Prediction:
  - attention mass the downstream pays to the slot RISES with ||z||
    (ignored at low norm -> saturated at high norm),
  - recall PEAKS at intermediate ||z|| (slot is read but doesn't starve the
    suffix's reads of the prompt / its own partial recitation).

Measure, over a norm sweep of a trained carrier's direction:
  (1) mean attention-to-slot (over heads, layers, downstream query positions),
  (2) recall (free generation), on the same axis.
"""
import sys, re, math
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
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32,
                                             attn_implementation="eager")
model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb = model.get_input_embeddings()
d = model.config.hidden_size
WORDS = SENTENCE.split(); N = len(WORDS)
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids

full = tok.apply_chat_template([{"role":"user","content":f'{INSTR}\n{PLACEHOLDER}'}],
                               tokenize=False, add_generation_prompt=True)
a,b = full.split(PLACEHOLDER)
PRE = tok(a, add_special_tokens=False).input_ids
SUF = tok(b, add_special_tokens=False).input_ids
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
def attn_to_slot(vec):
    """mean attention weight placed on the slot by downstream (suffix) queries,
    averaged over heads and layers."""
    ce = torch.cat([PE, vec.view(1,1,-1), SE], dim=1)
    out = model(inputs_embeds=ce, output_attentions=True)
    qpos = list(range(slot_pos+1, ce.shape[1]))   # suffix positions (after slot)
    vals = []
    for A in out.attentions:                       # (1, heads, seq, seq)
        w = A[0, :, qpos, slot_pos]                # (heads, |qpos|)
        vals.append(w.mean().item())
    return sum(vals)/len(vals)

@torch.no_grad()
def recite(vec, max_new=24):
    ce = torch.cat([PE, vec.view(1,1,-1), SE], dim=1)
    am = torch.ones(ce.shape[:2], dtype=torch.long)
    out = model.generate(inputs_embeds=ce, attention_mask=am, max_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()

c = opt_one(); n0 = c.norm(); unit = c / n0
# baseline: a normal token gets this much attention on average (uniform-ish ref)
print(f"\ncarrier norm {n0:.2f};  slot_pos={slot_pos};  suffix len={len(SUF)}", flush=True)
print("\n" + "="*64)
print("attention-to-slot  and  recall   vs   carrier norm")
print("="*64)
print(f"{'norm':>7} {'x':>5} {'attn->slot':>11} {'recall':>7}")
for scale in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
    v = unit * (scale * n0)
    at = attn_to_slot(v); rc = score(recite(v))
    bar = "#" * int(at*60)
    print(f"{v.norm():7.2f} {scale:5.2f} {at:11.4f} {rc:5d}/{N}  {bar}", flush=True)
print("\nPrediction: attn->slot rises monotonically with norm (vMF concentration);")
print("recall peaks at intermediate norm (read but not saturated).")
print("Done.")
