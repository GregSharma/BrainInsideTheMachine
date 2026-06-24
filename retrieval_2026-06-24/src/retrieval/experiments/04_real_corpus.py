#!/usr/bin/env python3
"""Real-corpus retrieval: 209-session BITM index vs real research queries.

Method (frozen ship spec from prior holdout):
  - Chunk feature: Q(h) at every token
  - Cue feature:   Q(h) at every token
  - Scorer:        cross_cos_mean (mean of cosine similarities across (i,j) position pairs)
  - Layer:         L27 on Qwen2.5-3B

For each query, print top-K sessions by score and the date/objective of each
for Greg to eyeball-grade.
"""
import sys, json, re, time, torch
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/greg/Desktop/Projects/BrainInsideTheMachine")

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from retrieval.features import HiddenState, QProj
from retrieval.scorers import CrossCosineMean, CosineLast
from retrieval.pipeline import encode_bare

INDEX_PATH = "/home/greg/Desktop/Projects/Claude_Transcripts_to_Md/last_index/Desktop_Projects_BrainInsideTheMachine/_index_oldestfirst.md"
MAX_TOKENS_PER_CHUNK = 384      # truncate session entries for uniform encoding cost
LAYER = 27

# Real research queries — what Greg might ask the corpus
QUERIES = {
    "read_head_discovery": "when was the read head discovered and what was the causal proof that last-token computation is rank-1",
    "convention_surgery": "convention surgery W_down kernel removal accuracy gains across languages and models",
    "innovation_ladder": "depth innovation Gram-Schmidt residual stream retrieval feature",
    "deflation_loop_escape": "soft query deflation breaking attention loop attractor on AMC problems",
    "kalshi_pricing": "Kalshi generative pricing news chunks LOO sensitivity Musk apology",
    "killed_paths": "experiments that were killed null results dead paths Z-bottleneck procrustes",
}


def split_sessions(text):
    """Split the index by '## 2026-' headers. Returns list of (header, body)."""
    parts = re.split(r"(?=^## 2026-)", text, flags=re.MULTILINE)
    sessions = []
    for p in parts:
        if not p.strip().startswith("## 2026-"):
            continue
        header_match = re.match(r"^## (.+?)$", p, flags=re.MULTILINE)
        header = header_match.group(1).strip() if header_match else "?"
        sessions.append((header, p.strip()))
    return sessions


def main():
    t0 = time.time()
    print("loading model...", flush=True)
    MODEL = "Qwen/Qwen2.5-3B"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    print(f"loading corpus from {INDEX_PATH}...", flush=True)
    with open(INDEX_PATH) as f:
        text = f.read()
    sessions = split_sessions(text)
    print(f"  {len(sessions)} sessions", flush=True)

    # Truncate each session to MAX_TOKENS_PER_CHUNK
    print(f"truncating chunks to {MAX_TOKENS_PER_CHUNK} tokens each...", flush=True)
    chunks = []
    for header, body in sessions:
        ids = tok(body, return_tensors="pt").input_ids[0][:MAX_TOKENS_PER_CHUNK]
        truncated = tok.decode(ids, skip_special_tokens=True)
        chunks.append((header, truncated))

    # Encode each chunk
    print(f"encoding {len(chunks)} chunks (L{LAYER}, Q(h) projection)...", flush=True)
    h_feat = HiddenState()
    q_feat = QProj(h_feat)
    device = "cuda"

    chunk_qs = []  # list of (T, d) tensors on GPU
    for header, body in tqdm(chunks, desc="encode"):
        c = encode_bare(body, tok, model)
        q = q_feat.extract(c, LAYER, model).to(device, dtype=torch.float32)
        # also store raw hidden state for baseline
        h_last = c.hidden_states[LAYER][-1].to(device, dtype=torch.float32)
        chunk_qs.append((q, h_last, header))

    # Encode each query
    print(f"\nencoding {len(QUERIES)} queries...", flush=True)
    query_data = {}
    for name, qtext in QUERIES.items():
        c = encode_bare(qtext, tok, model)
        q_proj = q_feat.extract(c, LAYER, model).to(device, dtype=torch.float32)
        h_last = c.hidden_states[LAYER][-1].to(device, dtype=torch.float32)
        query_data[name] = (q_proj, h_last, qtext)

    from retrieval.scorers import CrossMax, CrossCosineMax
    scorers = {
        "Q.Q cross_cos_mean":  (CrossCosineMean(), True),   # (scorer, uses_Q_projection)
        "Q.Q cross_cos_max":   (CrossCosineMax(),  True),
        "Q.Q cos_last":        (CosineLast(),      True),
        "h.h cos_last":        (CosineLast(),      False),  # vanilla baseline
    }

    print("\n" + "="*80, flush=True)
    print("TOP-10 SESSIONS PER QUERY, ALL SCORERS", flush=True)
    print("="*80, flush=True)

    results = {}
    for qname, (q_proj, q_h, qtext) in query_data.items():
        print(f"\n### QUERY: {qname}", flush=True)
        print(f"    \"{qtext}\"", flush=True)

        # Score under each scorer
        per_scorer_scores = {}
        for sname, (scorer, uses_q) in scorers.items():
            scores = []
            for q, h, header in chunk_qs:
                if uses_q:
                    s = scorer(q, q_proj, d_h=q.shape[-1])
                else:
                    # h is single (d,) — fake position dim for last-token cosine
                    s = scorer(h.unsqueeze(0), q_h.unsqueeze(0))
                scores.append(s)
            per_scorer_scores[sname] = scores

        # Show top-10 per scorer
        for sname, scores in per_scorer_scores.items():
            ranked = sorted(zip(scores, [c[2] for c in chunk_qs]), key=lambda x: -x[0])
            print(f"\n  >>> {sname}", flush=True)
            for i, (s, hdr) in enumerate(ranked[:10]):
                print(f"    {i+1:>3}. {s:+.4f}  {hdr[:95]}", flush=True)

        results[qname] = {sname: sorted(zip(scores, [c[2] for c in chunk_qs]), key=lambda x: -x[0])[:20]
                           for sname, scores in per_scorer_scores.items()}

    out = {
        "model": MODEL, "layer": LAYER,
        "n_chunks": len(chunks),
        "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK,
        "queries": QUERIES,
        "method": "Q(h).Q(h) cross_cos_mean L27",
        "top20_per_query": {k: {sn: [(float(s), h) for s, h in entries]
                                for sn, entries in scorers_d.items()}
                            for k, scorers_d in results.items()},
    }
    with open("output/exp_real_corpus.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
    print("saved to output/exp_real_corpus.json", flush=True)


if __name__ == "__main__":
    main()
