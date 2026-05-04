#!/usr/bin/env python3
"""
Exp: Attention Readout Anatomy — Two-Timescale Generation Test
==============================================================
Directly tests whether the late-layer read head sweeps a frozen workspace
with STICKY attention during glue tokens and SHIFTING attention during
content tokens.

Core metric: cos(a_{t-1}, a_t) over STATIC-PROMPT context positions only
(self-generated prefix and BOS excluded), per layer, per head, per
generation step. Compared to glue/content classification via
self-surprisal + tokenizer-functional tags.

Two regimes:
  A) LAYERWISE (not tested here — use output_attentions at prompt pass)
  B) STEPWISE (this script) — autoregressive generation with attention
     capture at every step.

Predictions:
  - Late layers show cos(a_{t-1}, a_t) > 0.9 for glue steps and < 0.7
    for content steps, with a visible bimodal gap.
  - Entropy stays roughly constant across glue runs, spikes briefly at
    content tokens (head momentarily reads more of the workspace).
  - Shuffled-label permutation test: real gap must sit outside 95% of
    the null distribution.

Kill conditions:
  - Uniform cos everywhere (>0.95 or <0.5): no sweep / no structure.
  - Gap only under surprisal labels but not tokenizer labels: we're
    detecting predictability, not function.
  - Gap disappears after excluding self-generated prefix positions:
    contamination by causal-mask recency.

Usage:
  python3 exp_attention_anatomy.py --dry          # 3 problems EN × 64 tokens, sanity
  python3 exp_attention_anatomy.py                # 20 problems × 2 langs × 128 tokens
  python3 exp_attention_anatomy.py --model 7b     # A100 (same script, bigger model)
"""

import json
import time
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse C2c problem generator for identical test set (seed=42)
from expC2c_crossmodel_readhead import (
    generate_problems, get_test_subset, TEMPLATES, SEED, NumpyEncoder,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    '3b': {'name': 'Qwen/Qwen2.5-3B', 'n_layers': 36, 'd': 2048},
    '7b': {'name': 'Qwen/Qwen2.5-7B', 'n_layers': 28, 'd': 3584},
}

MAX_NEW_DEFAULT = 128
MAX_NEW_DRY = 96

# Chat template wrap to extend prompt length beyond bare math statements.
# This gives the read head an actual workspace to sweep.
CHAT_SYSTEM = (
    "You are a careful mathematical reasoner. When given a problem, think "
    "step by step, show your work clearly, and then state the final numerical "
    "answer on its own line."
)

# Glue token heuristics
GLUE_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from', 'as',
    'and', 'or', 'but', 'so', 'if', 'then', 'that', 'this', 'it', 'its',
    'we', 'you', 'i', 'he', 'she', 'they', 'them', 'his', 'her', 'their',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can',
    'could', 'should', 'may', 'might', 'must', 'shall', 'not', 'no',
    'yes', 'there', 'here', 'now', 'when', 'where', 'why', 'how', 'what',
    'which', 'who', 'whom', 'whose', 'than', 'just', 'only', 'also',
    'therefore', 'thus', 'hence', 'so',
}
GLUE_PUNCT = set(' .,;:!?\'"()[]{}—-\n\t')


# ═══════════════════════════════════════════════════════════════════
# STEPWISE ATTENTION CAPTURE
# ═══════════════════════════════════════════════════════════════════

def build_prompt(tokenizer, problem_text, style='chat'):
    """Return the tokenized prompt. 'chat' uses Qwen chat template for a
    longer workspace; 'bare' uses the raw math string."""
    if style == 'chat':
        messages = [
            {'role': 'system', 'content': CHAT_SYSTEM},
            {'role': 'user', 'content': problem_text},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = f"{CHAT_SYSTEM}\n\nProblem: {problem_text}\n\nSolution:"
        return text
    return problem_text


def capture_stepwise(model, tokenizer, prompt_text, max_new, device):
    """Autoregressive generation with attention capture at every step.

    Returns dict with:
      - prompt_text, prompt_len, token_ids, tokens, surprisals
      - attn: np.array of shape (T, n_layers, n_heads, prompt_len)
              (last-token attention over the STATIC prompt only)
    """
    ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
    prompt_len = ids.shape[1]

    attn_trace = []
    token_ids = []
    surprisals = []
    tokens_decoded = []

    past_kv = None
    cur_input = ids
    with torch.inference_mode():
        for step in range(max_new):
            out = model(
                cur_input,
                past_key_values=past_kv,
                output_attentions=True,
                use_cache=True,
            )
            past_kv = out.past_key_values

            # out.attentions: tuple of length n_layers, each (1, n_heads, q_len, k_len)
            # On step 0, q_len = prompt_len. After, q_len = 1.
            # We always grab the last query's attention over the first prompt_len keys.
            step_layer_attn = []
            for lattn in out.attentions:
                a = lattn[0, :, -1, :prompt_len].float().cpu().numpy()
                step_layer_attn.append(a)
            attn_trace.append(np.stack(step_layer_attn, axis=0))  # (n_layers, n_heads, prompt_len)

            logits = out.logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            next_id = int(logits.argmax().item())
            surp = -float(torch.log(probs[next_id] + 1e-20).item())

            token_ids.append(next_id)
            surprisals.append(surp)
            tokens_decoded.append(tokenizer.decode([next_id]))

            if next_id == tokenizer.eos_token_id:
                break
            cur_input = torch.tensor([[next_id]], device=device)

    attn_np = np.stack(attn_trace, axis=0)  # (T, n_layers, n_heads, prompt_len)
    return {
        'prompt_text': prompt_text,
        'prompt_len': int(prompt_len),
        'token_ids': token_ids,
        'tokens': tokens_decoded,
        'surprisals': surprisals,
        'attn': attn_np,
    }


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def _safe_renorm(a, eps=1e-12):
    s = a.sum(axis=-1, keepdims=True)
    return a / (s + eps)


def compute_metrics(result, skip_sink=True):
    """From the raw attn trace, compute per-step per-layer metrics.

    - entropy:   (T, n_layers)       head-mean Shannon entropy
    - cos_trace: (T-1, n_layers)     head-mean cos(a_{t-1}, a_t)
    - argmax_pos:(T, n_layers)       head-mean argmax context position (float)
    """
    attn = result['attn']  # (T, L, H, P)
    T, L, H, P = attn.shape

    if skip_sink and P > 1:
        attn = attn[:, :, :, 1:]   # drop BOS
        attn = _safe_renorm(attn)
        P = P - 1

    # Entropy (per t, layer, head) → head-mean
    eps = 1e-12
    H_ent = -np.sum(attn * np.log(attn + eps), axis=-1)   # (T, L, H)
    entropy = H_ent.mean(axis=-1)                          # (T, L)

    # Cosine similarity between consecutive steps
    norms = np.linalg.norm(attn, axis=-1) + eps            # (T, L, H)
    cos_full = np.zeros((T - 1, L, H), dtype=np.float32)
    for t in range(1, T):
        dot = np.sum(attn[t - 1] * attn[t], axis=-1)       # (L, H)
        cos_full[t - 1] = dot / (norms[t - 1] * norms[t])
    cos_trace = cos_full.mean(axis=-1)                     # (T-1, L)

    # Argmax position (head-mean of per-head argmax)
    argmax = np.argmax(attn, axis=-1).astype(np.float32)   # (T, L, H)
    argmax_pos = argmax.mean(axis=-1)                      # (T, L)

    return {
        'entropy': entropy,
        'cos_trace': cos_trace,
        'argmax_pos': argmax_pos,
        'T': int(T),
        'L': int(L),
        'H': int(H),
        'P_used': int(P),
    }


# ═══════════════════════════════════════════════════════════════════
# GLUE / CONTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def classify_tokens(tokens, surprisals, low_pct=40, high_pct=60):
    """Return three label arrays of shape (T,):
       surp_label, tok_label, both_label (where they agree).
       Values: 'glue', 'content', or 'middle' / 'disagree'.
    """
    surp = np.asarray(surprisals)
    T = len(tokens)

    # Surprisal-based
    lo = np.percentile(surp, low_pct)
    hi = np.percentile(surp, high_pct)
    surp_label = np.where(surp <= lo, 'glue',
                          np.where(surp >= hi, 'content', 'middle'))

    # Tokenizer-functional
    tok_label = np.empty(T, dtype='<U8')
    for i, t in enumerate(tokens):
        s = t.strip()
        if len(s) == 0 or all(c in GLUE_PUNCT for c in s):
            tok_label[i] = 'glue'
        elif any(c.isdigit() for c in s):
            tok_label[i] = 'content'
        elif s.lower() in GLUE_STOPWORDS:
            tok_label[i] = 'glue'
        else:
            tok_label[i] = 'content'

    # Agreement
    both_label = np.where(
        surp_label == tok_label, surp_label, 'disagree'
    )
    return surp_label, tok_label, both_label


def glue_content_gap(cos_trace, labels_aligned):
    """cos_trace: (T-1, L).  labels_aligned: (T-1,) — labels for emitted token at step t+1."""
    L = cos_trace.shape[1]
    gap = np.full(L, np.nan, dtype=np.float32)
    glue_mask = labels_aligned == 'glue'
    content_mask = labels_aligned == 'content'
    if glue_mask.sum() > 0 and content_mask.sum() > 0:
        glue_mean = cos_trace[glue_mask].mean(axis=0)
        content_mean = cos_trace[content_mask].mean(axis=0)
        gap = glue_mean - content_mean
    return gap, int(glue_mask.sum()), int(content_mask.sum())


def bootstrap_gap(cos_trace, labels_aligned, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    real_gap, n_glue, n_content = glue_content_gap(cos_trace, labels_aligned)
    if n_glue == 0 or n_content == 0:
        return real_gap, np.zeros((n_boot, cos_trace.shape[1])), np.ones(cos_trace.shape[1])
    L = cos_trace.shape[1]
    null_gaps = np.zeros((n_boot, L), dtype=np.float32)
    labels_copy = labels_aligned.copy()
    for b in range(n_boot):
        rng.shuffle(labels_copy)
        g, _, _ = glue_content_gap(cos_trace, labels_copy)
        null_gaps[b] = g
    p_per_layer = (null_gaps >= real_gap[None, :]).mean(axis=0)  # one-sided
    return real_gap, null_gaps, p_per_layer


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run(model_key='3b', dry=False, max_new=None, langs=None, prompt_style='chat'):
    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg['name']
    n_layers_cfg = cfg['n_layers']

    if max_new is None:
        max_new = MAX_NEW_DRY if dry else MAX_NEW_DEFAULT
    if langs is None:
        langs = ['en'] if dry else ['en', 'zh']

    print(f"\n{'='*70}")
    print(f"EXP: Attention Readout Anatomy — {model_name}")
    print(f"  dry={dry}  max_new={max_new}  langs={langs}  prompt_style={prompt_style}")
    print(f"{'='*70}")
    t0 = time.time()

    # ── [1] Problems ──
    print("\n[1] Generating problems (seed=42, matches C2c)...", flush=True)
    all_problems = generate_problems(n_per_cat=40)
    test_problems = get_test_subset(all_problems, n_per_cat=4)  # 20 total
    if dry:
        test_problems = test_problems[:3]
    print(f"    {len(test_problems)} test problems", flush=True)

    # ── [2] Model ──
    print(f"\n[2] Loading {model_name} (eager attention)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
        attn_implementation='eager',
    )
    model.eval()
    device = next(model.parameters()).device
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"    Loaded. VRAM: {vram:.1f} GB ({time.time()-t0:.0f}s)", flush=True)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"    n_layers={n_layers}  n_heads={n_heads}", flush=True)

    # ── [3] Per-problem generation + capture ──
    print(f"\n[3] Generating with attention capture...", flush=True)
    per_problem = []   # list of dicts keyed by (problem_idx, lang)
    for p_idx, p in enumerate(test_problems):
        for lang in langs:
            t_start = time.time()
            wrapped = build_prompt(tokenizer, p[lang], style=prompt_style)
            result = capture_stepwise(
                model, tokenizer, wrapped, max_new=max_new, device=device,
            )
            m = compute_metrics(result, skip_sink=True)
            surp_lab, tok_lab, both_lab = classify_tokens(
                result['tokens'], result['surprisals']
            )
            T = m['T']
            # align: cos_trace[i] corresponds to transition from step i to step i+1
            # so label the emitted token at step i+1
            aligned_surp = surp_lab[1:]
            aligned_tok = tok_lab[1:]
            aligned_both = both_lab[1:]

            gap_s, ng_s, nc_s = glue_content_gap(m['cos_trace'], aligned_surp)
            gap_t, ng_t, nc_t = glue_content_gap(m['cos_trace'], aligned_tok)
            gap_b, ng_b, nc_b = glue_content_gap(m['cos_trace'], aligned_both)

            per_problem.append({
                'problem_idx': p_idx,
                'category': p['category'],
                'lang': lang,
                'prompt_text': result['prompt_text'],
                'prompt_len': result['prompt_len'],
                'T': T,
                'n_layers': m['L'],
                'n_heads': n_heads,
                'tokens': result['tokens'],
                'token_ids': result['token_ids'],
                'surprisals': result['surprisals'],
                'surp_labels': surp_lab.tolist(),
                'tok_labels': tok_lab.tolist(),
                'both_labels': both_lab.tolist(),
                'entropy_per_layer': m['entropy'].tolist(),
                'cos_trace_per_layer': m['cos_trace'].tolist(),
                'argmax_pos_per_layer': m['argmax_pos'].tolist(),
                'gap_surp': gap_s.tolist(),
                'gap_tok': gap_t.tolist(),
                'gap_both': gap_b.tolist(),
                'n_glue_surp': ng_s, 'n_content_surp': nc_s,
                'n_glue_tok': ng_t, 'n_content_tok': nc_t,
                'n_glue_both': ng_b, 'n_content_both': nc_b,
                'elapsed_s': time.time() - t_start,
            })
            print(
                f"    [{p_idx+1:2d}/{len(test_problems)}] {lang} T={T} "
                f"glue/content(surp)={ng_s}/{nc_s} "
                f"glue/content(tok)={ng_t}/{nc_t} "
                f"({time.time()-t_start:.1f}s)",
                flush=True,
            )

    # ── [4] Aggregate ──
    print(f"\n[4] Aggregating across problems...", flush=True)
    # Pool cos_trace and aligned labels across all problems per layer
    def pool(labels_name):
        cos_all = []
        lab_all = []
        for rec in per_problem:
            ct = np.asarray(rec['cos_trace_per_layer'])         # (T-1, L)
            lab = np.asarray(rec[labels_name])[1:]               # align to transitions
            # Truncate to matching length
            n = min(ct.shape[0], len(lab))
            cos_all.append(ct[:n])
            lab_all.append(lab[:n])
        cos_all = np.concatenate(cos_all, axis=0) if cos_all else np.zeros((0, n_layers))
        lab_all = np.concatenate(lab_all, axis=0) if lab_all else np.zeros(0, dtype='<U8')
        return cos_all, lab_all

    pooled_surp_cos, pooled_surp_lab = pool('surp_labels')
    pooled_tok_cos, pooled_tok_lab = pool('tok_labels')
    pooled_both_cos, pooled_both_lab = pool('both_labels')

    agg = {}
    for label_name, cos, lab in [
        ('surp', pooled_surp_cos, pooled_surp_lab),
        ('tok', pooled_tok_cos, pooled_tok_lab),
        ('both', pooled_both_cos, pooled_both_lab),
    ]:
        gap, n_g, n_c = glue_content_gap(cos, lab)
        # Bootstrap (shuffle labels)
        real_gap, null_gaps, p_per = bootstrap_gap(cos, lab, n_boot=1000, seed=42)
        agg[label_name] = {
            'n_glue': int(n_g),
            'n_content': int(n_c),
            'gap_per_layer': gap.tolist(),
            'p_one_sided_per_layer': p_per.tolist(),
            'null_gap_mean_per_layer': null_gaps.mean(axis=0).tolist(),
            'null_gap_std_per_layer': null_gaps.std(axis=0).tolist(),
        }

    # ── [5] Summary print ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for label_name in ['surp', 'tok', 'both']:
        info = agg[label_name]
        gap = np.asarray(info['gap_per_layer'])
        pv = np.asarray(info['p_one_sided_per_layer'])
        # Focus on late layers: last 30% of layers
        late_start = int(0.7 * n_layers)
        late_gap = gap[late_start:]
        late_pv = pv[late_start:]
        print(f"\n  [{label_name}] n_glue={info['n_glue']} n_content={info['n_content']}")
        print(f"    late-layer (L{late_start}–L{n_layers-1}) mean gap: {np.nanmean(late_gap):.4f}")
        print(f"    late-layer max gap: L{late_start + int(np.nanargmax(late_gap))} = {np.nanmax(late_gap):.4f}")
        print(f"    late-layer min p:   L{late_start + int(np.nanargmin(late_pv))} = {np.nanmin(late_pv):.4f}")
        print(f"    full-range mean gap: {np.nanmean(gap):.4f}")
        sig_layers = np.where(pv < 0.05)[0]
        print(f"    layers with p<0.05: {len(sig_layers)}/{n_layers}  → {sig_layers.tolist()}")

    # ── [6] Save ──
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)
    suffix = '_dry' if dry else ''
    out_path = out_dir / f'exp_attention_anatomy_{model_key}{suffix}.json'

    output = {
        'config': {
            'model': model_name,
            'model_key': model_key,
            'n_layers': int(n_layers),
            'n_heads': int(n_heads),
            'max_new': int(max_new),
            'langs': langs,
            'dry': dry,
            'seed': SEED,
            'skip_sink': True,
        },
        'per_problem': per_problem,
        'aggregate': agg,
        'elapsed_total_s': time.time() - t0,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output


def main():
    parser = argparse.ArgumentParser(description='Attention Readout Anatomy')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), default='3b')
    parser.add_argument('--dry', action='store_true', help='3 problems × EN × 96 tokens')
    parser.add_argument('--max-new', type=int, default=None)
    parser.add_argument('--langs', nargs='+', default=None)
    parser.add_argument('--prompt-style', choices=['chat', 'bare'], default='chat')
    args = parser.parse_args()
    run(
        model_key=args.model,
        dry=args.dry,
        max_new=args.max_new,
        langs=args.langs,
        prompt_style=args.prompt_style,
    )


if __name__ == '__main__':
    main()
