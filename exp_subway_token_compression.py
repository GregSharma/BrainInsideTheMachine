"""exp_subway: Can a sentence be conveyed in FEWER tokens than it has words?

Greg's question: take a short sentence like
    "do not lean on car doors on the subway"   (9 words -> 9 tokens here)
and replace its token embeddings with k < 9 injected vectors (activation
injection via `inputs_embeds`).  Then verify the information survived by
asking the model to recite the sentence and checking it word-by-word:
    what was the 1st word? the 2nd? ... the 9th?   <- all should hit.

This is the token-compression / soft-prompt ("gist token") question, run at
the smallest scale that fits on CPU: Qwen2.5-0.5B-Instruct.

Conditions (each = a way to represent the 9-token sentence inside the prompt):
  - baseline_text : the real 9 embeddings, no compression (upper bound)
  - meanpool_k    : sliding-window mean-pool to k vectors          (naive, expAN)
  - svd_k         : top-k right singular vectors, rescaled         (naive, expAN)
  - stride_k      : every-Nth token embedding                      (naive, expAN)
  - opt_k         : k continuous vectors OPTIMIZED by gradient descent so the
                    frozen model recites the sentence (this is the "encoder")

Verification (the literal "what was the Nth word" test):
  PRIMARY  : recite with the SAME instruction opt_k was trained on.
  GENERAL  : recite with a DIFFERENT instruction phrasing opt_k never saw
             -> shows the k vectors encode the *content*, not a memorized
                output string.

We score each of the 9 word positions independently and report hits.

Why a 0.5B model: no GPU here (4 CPU, 15GB). The repo's mainline runs use
Qwen2.5-3B on CUDA; the method is identical, only the scale differs. The 0.5B
can recite a sentence verbatim but cannot do positional *counting* ("the 7th
word") even from clean text, so we verify via recitation + per-position
alignment rather than asking it to count.
"""
import json, time, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("output")
SENTENCE   = "do not lean on car doors on the subway"
K_VALUES   = [1, 2, 3, 4]
OPT_STEPS  = 150
OPT_LR     = 0.05
SEED       = 0

# Instruction used to elicit recitation.  opt_k is trained on TRAIN_INSTR only;
# GEN_INSTR is held out to test that the vectors encode content, not a string.
TRAIN_INSTR = "Repeat the following sentence back to me, word for word, with nothing else:"
GEN_INSTR   = "Echo this sentence verbatim, output only the sentence:"
PLACEHOLDER = "@@SENT@@"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading model...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
emb_layer = model.get_input_embeddings()
d_model = model.config.hidden_size
WORDS = SENTENCE.split()
SENT_IDS = tok(SENTENCE, add_special_tokens=False).input_ids  # the 9 tokens
print(f"  loaded in {time.time()-t0:.1f}s  d={d_model}  layers={model.config.num_hidden_layers}", flush=True)
print(f"  sentence: {len(WORDS)} words / {len(SENT_IDS)} tokens", flush=True)
print(f"  tokens: {[tok.decode([i]) for i in SENT_IDS]}", flush=True)
assert len(SENT_IDS) == len(WORDS), "expected 1 token per word for this sentence"
N = len(SENT_IDS)


# ── Prompt assembly: prefix-text | [sentence vectors] | suffix-text ─────────
def build_prefix_suffix(instruction):
    """Return (prefix_ids, suffix_ids) surrounding the sentence slot."""
    content = f'{instruction}\n{PLACEHOLDER}'
    full = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    pre_txt, suf_txt = full.split(PLACEHOLDER)
    pre = tok(pre_txt, add_special_tokens=False).input_ids
    suf = tok(suf_txt, add_special_tokens=False).input_ids
    return torch.tensor([pre]), torch.tensor([suf])


def sentence_embeds():
    """The genuine (N, d) embeddings of the sentence tokens."""
    ids = torch.tensor([SENT_IDS])
    with torch.no_grad():
        return emb_layer(ids).squeeze(0).clone()  # (N, d)


def ctx_embeds(sent_vecs, instruction):
    """prefix | sent_vecs(k,d) | suffix  ->  (1, L, d)"""
    pre, suf = build_prefix_suffix(instruction)
    with torch.no_grad():
        pe = emb_layer(pre)            # (1, P, d)
        se = emb_layer(suf)            # (1, S, d)
    sv = sent_vecs.unsqueeze(0)        # (1, k, d)  (may carry grad)
    return torch.cat([pe, sv, se], dim=1)


@torch.no_grad()
def recite(sent_vecs, instruction, max_new=24):
    """Generate the model's recitation given the sentence-slot vectors."""
    ce = ctx_embeds(sent_vecs, instruction)
    am = torch.ones(ce.shape[:2], dtype=torch.long)
    out = model.generate(inputs_embeds=ce, attention_mask=am,
                         max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()


# ── Naive compressors (mirrors expAN) ──────────────────────────────────────
def c_meanpool(E, k):
    idx = torch.linspace(0, E.shape[0], k + 1).long()
    return torch.stack([E[idx[i]:max(idx[i+1], idx[i]+1)].mean(0) for i in range(k)])

def c_stride(E, k):
    idx = torch.linspace(0, E.shape[0]-1, k).long()
    return E[idx]

def c_svd(E, k):
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    comp = S[:k].unsqueeze(1) * Vh[:k, :]
    comp = comp * (E.norm(dim=-1).mean() / comp.norm(dim=-1).mean())
    return comp

NAIVE = {"meanpool": c_meanpool, "stride": c_stride, "svd": c_svd}


# ── The optimizer: find k vectors that make the model recite the sentence ───
def optimize_soft_tokens(k, steps=OPT_STEPS, lr=OPT_LR):
    """k continuous vectors, warm-started from mean-pool, trained so the frozen
    model outputs the sentence under TRAIN_INSTR.  Only the k vectors learn."""
    E = sentence_embeds()
    P = c_meanpool(E, k).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    pre, suf = build_prefix_suffix(TRAIN_INSTR)
    tgt = torch.tensor([SENT_IDS + [tok.eos_token_id]])  # recite sentence then stop
    with torch.no_grad():
        pe, se, te = emb_layer(pre), emb_layer(suf), emb_layer(tgt)
    losses = []
    for step in range(steps):
        opt.zero_grad()
        full = torch.cat([pe, P.unsqueeze(0), se, te], dim=1)
        ctx_len = pe.shape[1] + k + se.shape[1]
        logits = model(inputs_embeds=full).logits
        pred = logits[:, ctx_len-1:ctx_len-1+tgt.shape[1], :]
        loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return P.detach(), losses


# ── Scoring: read the recitation off word-by-word ───────────────────────────
def score_positions(text):
    """Return (per_position_hit list, recovered_words) aligned to WORDS.

    Find the contiguous window of the output that best matches the sentence,
    then compare each of the 9 positions.  Robust to a leading 'Sure,' etc.
    """
    words = re.findall(r"[a-z]+", text.lower())
    truth = [w.lower() for w in WORDS]
    best_start, best_score = 0, -1
    for s in range(0, max(1, len(words) - N + 1)):
        window = words[s:s+N]
        score = sum(1 for i in range(min(N, len(window))) if window[i] == truth[i])
        if score > best_score:
            best_score, best_start = score, s
    window = words[best_start:best_start+N]
    hits = [(i < len(window) and window[i] == truth[i]) for i in range(N)]
    return hits, window


def hits_str(hits):
    return "".join("Y" if h else "." for h in hits)


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUBWAY TOKEN COMPRESSION  —  recite & verify word-by-word")
print("=" * 72)
ords = ["1st","2nd","3rd","4th","5th","6th","7th","8th","9th"]
results = {"model": MODEL_NAME, "sentence": SENTENCE, "words": WORDS,
           "n_tokens": N, "k_values": K_VALUES, "conditions": {}}

def run_condition(name, sent_vecs, k, instr=TRAIN_INSTR):
    text = recite(sent_vecs, instr)
    hits, window = score_positions(text)
    nhit = sum(hits)
    results["conditions"][name] = {
        "k": k, "comp_ratio": round(N / k, 2), "n_hit": nhit,
        "hits": [bool(h) for h in hits], "recovered": window, "raw": text,
    }
    print(f"  {name:18s} k={k} ratio={N/k:.2f}x  hits {nhit}/{N} [{hits_str(hits)}]  {text[:46]!r}")
    return nhit

# 0) Upper bound: the real 9 embeddings, no compression
E = sentence_embeds()
print("\n-- baseline (uncompressed, k=9) --")
run_condition("baseline_text", E, N)

# 1) Naive compression
print("\n-- naive compression --")
for mname, fn in NAIVE.items():
    for k in K_VALUES:
        run_condition(f"{mname}_k{k}", fn(E, k), k)

# 2) Optimized soft tokens (activation injection)
print("\n-- optimized soft tokens (gradient descent on frozen model) --")
opt_vectors = {}
for k in K_VALUES:
    P, losses = optimize_soft_tokens(k)
    opt_vectors[k] = P
    results["conditions"].setdefault("_opt_loss", {})[f"k{k}"] = round(losses[-1], 4)
    print(f"  [opt k={k}] final loss {losses[-1]:.4f} (start {losses[0]:.3f})")
    run_condition(f"opt_k{k}", P, k)

# 3) Generalization: optimized vectors under a DIFFERENT instruction
print("\n-- generalization: opt vectors + UNSEEN instruction --")
for k in K_VALUES:
    run_condition(f"opt_k{k}_GEN", opt_vectors[k], k, instr=GEN_INSTR)

# ── The user's literal question, on the best small-k optimized condition ────
best_k = K_VALUES[0]  # most aggressive compression
P = opt_vectors[best_k]
text = recite(P, TRAIN_INSTR)
hits, window = score_positions(text)
print("\n" + "=" * 72)
print(f"VERIFY @ k={best_k}  ({N}->{best_k} tokens, {N/best_k:.1f}x compression)")
print("=" * 72)
print(f"  model recited: {text!r}")
for i in range(N):
    print(f"  what was the {ords[i]} word?  truth={WORDS[i]:7s} got={window[i] if i<len(window) else '<none>':7s}  {'HIT' if hits[i] else 'MISS'}")
print(f"  => {sum(hits)}/{N} positions correct")

OUTPUT_DIR.mkdir(exist_ok=True)
with open(OUTPUT_DIR / "exp_subway_token_compression.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved output/exp_subway_token_compression.json")
print("Done.")
