"""exp_subway_midlayer: is the sentence recoverable from ONE MID-LAYER residual
vector, with NO training? (the 'free compression' route)

The input-layer closed form failed. The deeper conjecture is about mid-layer
representations. Test directly, expINJ-style:

  1. Forward [prefix][sentence][suffix]; capture the residual stream at the LAST
     sentence-token position at every layer (causal attention => it has seen the
     whole sentence).
  2. Build [prefix][1 neutral slot][suffix]; during prefill, OVERWRITE the slot's
     layer-L output with the captured vector s_L; let layers L+1.. run; recite.

If the model recites from s_L alone, the sentence's information is linearly
present and READABLE at a single mid-layer position -- compression for the cost
of one forward pass, no gradient descent. Sweep L. Controls: norm-matched random
vector (should fail), and no-injection (slot is neutral -> should fail).
"""
import sys, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SENTENCE = "do not lean on car doors on the subway"
INSTR = "Repeat the sentence I gave you, word for word, with nothing else:"
PLACEHOLDER = "@@SENT@@"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32); model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb = model.get_input_embeddings()
NL = model.config.num_hidden_layers
WORDS = SENTENCE.split(); N = len(WORDS)
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids
NEUTRAL = emb.weight.mean(0).detach()   # uninformative seed for the slot
print(f"loaded layers={NL} sentence={N} tok", flush=True)

def ps(instr):
    full = tok.apply_chat_template([{"role":"user","content":f'{instr}\n{PLACEHOLDER}'}],
                                   tokenize=False, add_generation_prompt=True)
    a,b = full.split(PLACEHOLDER)
    return (tok(a, add_special_tokens=False).input_ids, tok(b, add_special_tokens=False).input_ids)
PRE, SUF = ps(INSTR)

def score(text):
    words = re.findall(r"[a-z]+", text.lower()); truth=[w.lower() for w in WORDS]
    bs,b=0,-1
    for s in range(max(1,len(words)-N+1)):
        win=words[s:s+N]; sc=sum(1 for i in range(min(N,len(win))) if win[i]==truth[i])
        if sc>b: b,bs=sc,s
    win=words[bs:bs+N]
    return sum(1 for i in range(N) if i<len(win) and win[i]==truth[i])

# 1) capture last-sentence-token residual at every layer
cap = {}
def mk_cap(L, pos):
    def h(m,i,o):
        hid = o[0] if isinstance(o,tuple) else o
        cap[L] = hid[0, pos, :].detach().clone()
    return h
ids = torch.tensor([PRE + SENT_IDS + SUF])
sent_last = len(PRE) + N - 1
hs = [model.model.layers[L].register_forward_hook(mk_cap(L, sent_last)) for L in range(NL)]
with torch.no_grad(): model(input_ids=ids)
for h in hs: h.remove()

# 2) inject s_L at the slot during prefill, generate
slot_pos = len(PRE)
def inject_seq_embeds():
    pe = emb(torch.tensor([PRE]))
    se = emb(torch.tensor([SUF]))
    slot = NEUTRAL.view(1,1,-1)
    return torch.cat([pe, slot, se], dim=1)

class Injector:
    def __init__(self, L, vec): self.L=L; self.vec=vec; self.fired=False; self.h=None
    def __enter__(self):
        def hook(m,i,o):
            if self.fired: return o
            hid = o[0] if isinstance(o,tuple) else o
            if hid.shape[1] > 1:
                hid[0, slot_pos, :] = self.vec; self.fired=True
            return o
        self.h = model.model.layers[self.L].register_forward_hook(hook); return self
    def __exit__(self,*a): self.h.remove()

@torch.no_grad()
def recite_inject(L, vec, max_new=24):
    ce = inject_seq_embeds()
    am = torch.ones(ce.shape[:2], dtype=torch.long)
    with Injector(L, vec):
        out = model.generate(inputs_embeds=ce, attention_mask=am, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()

print("\n" + "="*72)
print("MID-LAYER single-vector injection (no training): recite from s_L alone")
print("="*72)
for L in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
    if L >= NL: break
    real = recite_inject(L, cap[L])
    rnd_v = torch.randn_like(cap[L]); rnd_v = rnd_v / rnd_v.norm() * cap[L].norm()
    rnd = recite_inject(L, rnd_v)
    print(f"  L={L:>2}: real {score(real)}/{N}  rand {score(rnd)}/{N}   real->{real[:42]!r}")

# no-injection control
ce = inject_seq_embeds(); am = torch.ones(ce.shape[:2], dtype=torch.long)
with torch.no_grad():
    out = model.generate(inputs_embeds=ce, attention_mask=am, max_new_tokens=24,
                        do_sample=False, pad_token_id=tok.eos_token_id)
print(f"\n  no-injection (neutral slot): {score(tok.decode(out[0], skip_special_tokens=True))}/{N}")
print("Done.")
