"""exp_subway_closedform: can the carrier be COMPUTED, not trained?

Follows the thread from exp_subway_token_compression.py. The SGD soft token
hit 9/9 from a single off-lattice vector. Question: is there a training-free,
closed-form construction that recovers the sentence — exploiting near-additivity
and the VSA reading (RoPE = multiplicative position-binding, residual sum =
bundling)?

All methods produce ONE vector, injected at ONE position in place of the 9
sentence tokens, then we recite & score word-by-word. Ladder:

  meanpool_1   : mean of the 9 embeddings           (additive bag, control)
  bag_sum      : raw sum of the 9 embeddings         (additive bag)
  bag_norm     : sum, rescaled to mean token norm    (additive bag)
  rope_bind    : sum_i RoPE_rotate(E[x_i], i)        (VSA bind+bundle, closed-form)
  rope_bind_n  : rope_bind, rescaled to token norm
  opt_1        : 150-step SGD soft token             (positive reference, expect 9/9)

If rope_bind recovers materially more than bag_sum, multiplicative binding is
doing real work and the closed-form direction is live. If everything but opt_1
fails, the middle is not additive enough for a naive analytic encoder at this
layer and SGD (or a learned encoder) is still required.
"""
import re, sys
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
d = model.config.hidden_size
THETA = float(getattr(model.config, "rope_theta", 1e6))
WORDS = SENTENCE.split()
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids
N = len(SENT_IDS)
print(f"loaded d={d} theta={THETA:g} sentence={N} tokens", flush=True)


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
with torch.no_grad(): TE = emb_layer(TGT)
E = emb_layer(torch.tensor([SENT_IDS])).squeeze(0).detach()   # (N, d)
tok_norm = E.norm(dim=-1).mean()


@torch.no_grad()
def recite(vecs, max_new=24):
    ce = torch.cat([PE, vecs.unsqueeze(0), SE], dim=1)
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


def rope_rotate(vec, pos, theta=THETA):
    """VSA-style bind: rotate dim-pairs of `vec` by angle = pos * theta^(-2j/d)."""
    half = vec.shape[-1] // 2
    j = torch.arange(half, dtype=torch.float32)
    ang = pos * (theta ** (-2.0 * j / vec.shape[-1]))
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = vec[..., 0::2], vec[..., 1::2]
    out = torch.empty_like(vec)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def opt_one(steps=150, lr=0.05):
    P = E.mean(0, keepdim=True).clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        full = torch.cat([PE, P.unsqueeze(0), SE, TE], dim=1)
        ctx = PE.shape[1] + 1 + SE.shape[1]
        logits = model(inputs_embeds=full).logits
        pred = logits[:, ctx-1:ctx-1+TGT.shape[1], :]
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), TGT.reshape(-1))
        loss.backward(); opt.step()
    return P.detach()


# ── closed-form single-vector constructions ────────────────────────────────
bag_sum  = E.sum(0, keepdim=True)
bag_norm = bag_sum * (tok_norm / bag_sum.norm())
rope     = torch.stack([rope_rotate(E[i], i) for i in range(N)]).sum(0, keepdim=True)
rope_n   = rope * (tok_norm / rope.norm())

methods = {
    "meanpool_1": E.mean(0, keepdim=True),
    "bag_sum":    bag_sum,
    "bag_norm":   bag_norm,
    "rope_bind":  rope,
    "rope_bind_n":rope_n,
}

print("\n" + "="*72)
print("CLOSED-FORM / TRAINING-FREE single-vector constructions (1 token, 9x)")
print("="*72)
for name, v in methods.items():
    t = recite(v)
    print(f"  {name:13s} hits {score(t)}/{N}   {t[:50]!r}")

print("\n-- positive reference --")
P = opt_one()
t = recite(P)
print(f"  {'opt_1 (SGD)':13s} hits {score(t)}/{N}   {t[:50]!r}")
print("\nDone.")
