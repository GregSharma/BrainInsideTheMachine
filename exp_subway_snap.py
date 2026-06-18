"""exp_subway_snap: can the optimized soft token be ARGMAX'd onto a real token?
And can we do this WITHOUT gradient descent at all?

Follow-up to exp_subway_token_compression.py. Two tests:

  TEST 1 (snap).   Optimize the soft vector(s), then snap each to its nearest
                   vocabulary embedding (argmax cosine over the 151k tokens),
                   i.e. turn the continuous "soft prompt" into a real discrete
                   token. Recite & score. Asks: does the magic survive a trip
                   back to the token grid?

  TEST 2 (search). Pure gradient-free DISCRETE search: pick the k REAL tokens
                   (from a candidate pool, forward passes only, no gradients)
                   that maximize P(sentence | tokens). Asks: can a short real-
                   token prompt carry the sentence with no training?

Same 0.5B CPU setup and same recite-and-score protocol.
"""
import time, re, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SENTENCE   = "do not lean on car doors on the subway"
TRAIN_INSTR = "Repeat the following sentence back to me, word for word, with nothing else:"
PLACEHOLDER = "@@SENT@@"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32); model.eval()
for p in model.parameters(): p.requires_grad_(False)
emb_layer = model.get_input_embeddings()
W = emb_layer.weight.detach()                       # (V, d)
Wn = F.normalize(W, dim=-1)
WORDS = SENTENCE.split()
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids
N = len(SENT_IDS)
print(f"loaded  V={W.shape[0]}  d={W.shape[1]}  sentence={N} tokens", flush=True)


def build_prefix_suffix(instruction):
    content = f'{instruction}\n{PLACEHOLDER}'
    full = tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True)
    pre_txt, suf_txt = full.split(PLACEHOLDER)
    return (torch.tensor([tok(pre_txt, add_special_tokens=False).input_ids]),
            torch.tensor([tok(suf_txt, add_special_tokens=False).input_ids]))

PRE, SUF = build_prefix_suffix(TRAIN_INSTR)
with torch.no_grad():
    PE, SE = emb_layer(PRE), emb_layer(SUF)
TGT = torch.tensor([SENT_IDS + [tok.eos_token_id]])
with torch.no_grad():
    TE = emb_layer(TGT)


@torch.no_grad()
def recite(sent_vecs, max_new=24):
    ce = torch.cat([PE, sent_vecs.unsqueeze(0), SE], dim=1)
    am = torch.ones(ce.shape[:2], dtype=torch.long)
    out = model.generate(inputs_embeds=ce, attention_mask=am, max_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def score(text):
    words = re.findall(r"[a-z]+", text.lower()); truth = [w.lower() for w in WORDS]
    best_s, best = 0, -1
    for s in range(max(1, len(words)-N+1)):
        win = words[s:s+N]; sc = sum(1 for i in range(min(N, len(win))) if win[i]==truth[i])
        if sc > best: best, best_s = sc, s
    win = words[best_s:best_s+N]
    return sum(1 for i in range(N) if i < len(win) and win[i]==truth[i])


@torch.no_grad()
def recite_loss(sent_vecs):
    """teacher-forced CE of the sentence given the k injected vectors."""
    full = torch.cat([PE, sent_vecs.unsqueeze(0), SE, TE], dim=1)
    ctx = PE.shape[1] + sent_vecs.shape[0] + SE.shape[1]
    logits = model(inputs_embeds=full).logits
    pred = logits[:, ctx-1:ctx-1+TGT.shape[1], :]
    return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), TGT.reshape(-1)).item()


def optimize(k, steps=150, lr=0.05):
    idx = torch.linspace(0, N, k+1).long()
    init = torch.stack([emb_layer(torch.tensor(SENT_IDS[idx[i]:max(idx[i+1],idx[i]+1)])).mean(0)
                        for i in range(k)])
    P = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        full = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1] + k + SE.shape[1]
        logits = model(inputs_embeds=full).logits
        pred = logits[:, ctx-1:ctx-1+TGT.shape[1], :]
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), TGT.reshape(-1))
        loss.backward(); opt.step()
    return P.detach()


print("\n" + "="*72)
print("TEST 1 — snap the optimized vector onto the nearest real token (argmax)")
print("="*72)
for k in [1, 2, 4]:
    P = optimize(k)
    soft_hits = score(recite(P))
    # snap each of the k vectors to nearest vocab token by cosine
    Pn = F.normalize(P, dim=-1)
    sims = Pn @ Wn.T                       # (k, V)
    best = sims.max(dim=-1)
    ids = best.indices.tolist(); cos = [round(c, 3) for c in best.values.tolist()]
    snapped_vecs = W[best.indices]         # real embeddings of the snapped tokens
    snap_hits = score(recite(snapped_vecs))
    toks = [repr(tok.decode([i])) for i in ids]
    print(f"  k={k}: soft={soft_hits}/{N}  ->  snapped tokens {toks} (cos {cos})  ->  snap={snap_hits}/{N}")
    print(f"        recite(snapped): {recite(snapped_vecs)[:60]!r}")


print("\n" + "="*72)
print("TEST 2 — gradient-free discrete search: best real token(s), forward only")
print("="*72)
# Candidate pool: the sentence's own tokens + a generic set, score each as a 1-token prompt.
# (Pure forward passes, no gradients.) k=1 then greedy add a 2nd from the pool.
pool = sorted(set(SENT_IDS + tok(" the a is not on do car train bus stop hold rail").input_ids
                  + list(range(0, 2000, 23))))
print(f"  pool size = {len(pool)}")
# k=1: best single real token
t0 = time.time()
scored = []
for tid in pool:
    L = recite_loss(W[tid:tid+1])
    scored.append((L, tid))
scored.sort()
bestL, best1 = scored[0]
print(f"  k=1 best token {tok.decode([best1])!r}  loss={bestL:.3f}  "
      f"hits={score(recite(W[best1:best1+1]))}/{N}  ({time.time()-t0:.0f}s, {len(pool)} fwd)")
print(f"      recite: {recite(W[best1:best1+1])[:60]!r}")
# k=2: keep best1, greedily search 2nd token
t0 = time.time()
scored2 = []
for tid in pool:
    vecs = torch.stack([W[best1], W[tid]])
    scored2.append((recite_loss(vecs), tid))
scored2.sort()
bestL2, best2 = scored2[0]
v2 = torch.stack([W[best1], W[best2]])
print(f"  k=2 best pair {tok.decode([best1])!r}+{tok.decode([best2])!r}  loss={bestL2:.3f}  "
      f"hits={score(recite(v2))}/{N}  ({time.time()-t0:.0f}s)")
print(f"      recite: {recite(v2)[:60]!r}")
print("\nDone.")
