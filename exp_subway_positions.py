"""exp_subway_positions: the control that pins down the mid-layer negative.

B showed: one natural mid-layer residual -> 0/9. A skeptic says "mid-layer
injection just doesn't work." This rules that out. We transplant the model's
own layer-L residuals for the LAST m of the 9 sentence positions into m of 9
slots (rest neutral) and recite. Sweep m.

Prediction if the sentence is DISTRIBUTED across positions (not compressible
for free): recall rises with m, ~9/9 at m=9 (injection works), and falls toward
0 as m->1 (B). That curve = the information is spread across positions; no
single position holds it.
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
NEUTRAL = emb.weight.mean(0).detach()

def ps(instr):
    full = tok.apply_chat_template([{"role":"user","content":f'{instr}\n{PLACEHOLDER}'}],
                                   tokenize=False, add_generation_prompt=True)
    a,b = full.split(PLACEHOLDER)
    return tok(a, add_special_tokens=False).input_ids, tok(b, add_special_tokens=False).input_ids
PRE, SUF = ps(INSTR)

def score(text):
    words = re.findall(r"[a-z]+", text.lower()); truth=[w.lower() for w in WORDS]
    bs,b=0,-1
    for s in range(max(1,len(words)-N+1)):
        win=words[s:s+N]; sc=sum(1 for i in range(min(N,len(win))) if win[i]==truth[i])
        if sc>b: b,bs=sc,s
    win=words[bs:bs+N]
    return sum(1 for i in range(N) if i<len(win) and win[i]==truth[i])

# capture all 9 sentence-position residuals at every layer
cap = {L: None for L in range(NL)}
def mk(L, lo, hi):
    def h(m,i,o):
        hid = o[0] if isinstance(o,tuple) else o
        cap[L] = hid[0, lo:hi, :].detach().clone()   # (9, d)
    return h
ids = torch.tensor([PRE + SENT_IDS + SUF])
lo = len(PRE); hi = lo + N
hs = [model.model.layers[L].register_forward_hook(mk(L,lo,hi)) for L in range(NL)]
with torch.no_grad(): model(input_ids=ids)
for h in hs: h.remove()

# build [PRE][9 neutral slots][SUF]; overwrite last m slots' layer-L output
slot0 = len(PRE)
def seq_embeds():
    pe = emb(torch.tensor([PRE])); se = emb(torch.tensor([SUF]))
    slots = NEUTRAL.view(1,1,-1).repeat(1, N, 1)
    return torch.cat([pe, slots, se], dim=1)

class InjectM:
    def __init__(self, L, vecs, m): self.L=L; self.vecs=vecs; self.m=m; self.fired=False; self.h=None
    def __enter__(self):
        def hook(mod,i,o):
            if self.fired: return o
            hid = o[0] if isinstance(o,tuple) else o
            if hid.shape[1] > 1:
                for j in range(N - self.m, N):           # last m positions
                    hid[0, slot0 + j, :] = self.vecs[j]
                self.fired = True
            return o
        self.h = model.model.layers[self.L].register_forward_hook(hook); return self
    def __exit__(self,*a): self.h.remove()

@torch.no_grad()
def recite_m(L, m, max_new=24):
    ce = seq_embeds(); am = torch.ones(ce.shape[:2], dtype=torch.long)
    with InjectM(L, cap[L], m):
        out = model.generate(inputs_embeds=ce, attention_mask=am, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()

print("\n" + "="*72)
print("RECALL vs #positions injected (last m of 9 natural residuals at layer L)")
print("="*72)
print(f"{'L':>3} " + " ".join(f"m={m}" for m in [1,3,5,7,9]))
for L in [4, 8, 12, 16, 20]:
    row = [score(recite_m(L, m)) for m in [1,3,5,7,9]]
    print(f"{L:>3} " + " ".join(f"{s:>3}" for s in row), flush=True)
    if L == 12:
        print(f"     (m=9 sample: {recite_m(12,9)[:48]!r})")
print("\nDone.")
