#!/usr/bin/env python3
"""
Exp: Causal Source Map — context→read-head operator via position ablation.
==========================================================================

GOAL
----
Test the operator ansatz (GPT Web reframe): if there is a canonical
language-agnostic reasoning geometry, it should live in the operator that
maps context-token computation into the last-token read head, not in the
raw last-token state's PCA.

METHOD (true ablation, not fitted regression)
---------------------------------------------
For each problem × language at a set of late layers ℓ ∈ L_TARGETS:
  1. Prefill the chat-wrapped prompt. Record baseline h_last^(ℓ) and next
     token logits.
  2. For each prompt position t (skipping BOS), run a second forward pass
     with position t attention-masked out. Record h_last^(ℓ) and logits.
  3. Source-map influence of position t on layer ℓ:
         s^(ℓ)(t) = || h_last_base^(ℓ) - h_last_ablated(t)^(ℓ) ||_2
     Second metric: Δlogit on baseline argmax next token.

OP-2 (CKA across languages)
---------------------------
Each language produces an influence profile s^(ℓ) over positions. Languages
differ in prompt length so position-index alignment is meaningless. Instead,
resample each profile onto a fixed grid of 20 τ-fractions and compute:
  a) linear CKA between same-problem cross-language profiles (en vs zh)
  b) linear CKA between same-language cross-problem profiles (en vs en on
     different problems) as a null
  c) linear CKA of raw last-token states (existing measurement) for comparison

If (a) >> (b), the operator is more language-invariant than the substrate —
evidence for a canonical operator.

COST
----
20 problems × 2 langs × ~150-position ablations × 6 layers.
Batched: ~60-80 batched forwards on RayGun 3B fp16. Target < 15 minutes wall.
If OOM, drop batch size. If still OOM, move to Colab A100.

USAGE
-----
python3 exp_causal_source_map.py --dry        # 3 problems × EN × L32 only
python3 exp_causal_source_map.py              # 20 problems × EN+ZH × 6 layers
python3 exp_causal_source_map.py --model 7b   # A100 (same script, bigger model)
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse problem generator + chat template from existing infrastructure
from expC2c_crossmodel_readhead import (
    generate_problems, get_test_subset, TEMPLATES, SEED, NumpyEncoder,
)
from exp_attention_anatomy import CHAT_SYSTEM, build_prompt


MODEL_CONFIGS = {
    '3b': {'name': 'Qwen/Qwen2.5-3B', 'n_layers': 36, 'd': 2048},
    '7b': {'name': 'Qwen/Qwen2.5-7B', 'n_layers': 28, 'd': 3584},
}

# Late layers: read-tip assembly zone. L20 bracket (pre-assembly), L24/27
# (assembly), L30/32/35 (mature read tip). These are the 6 layers that
# most directly probe the context→read-head operator.
L_TARGETS_3B = [20, 24, 27, 30, 32, 35]
L_TARGETS_7B = [16, 19, 21, 23, 25, 27]  # Qwen2.5-7B has 28 layers

BATCH_SIZE_DEFAULT = 16  # ablations per forward pass; drop if OOM


def get_hooks(model, layer_indices):
    """Register forward hooks to capture hidden states at given layer indices.
    Returns (hooks_list, storage_dict). storage gets overwritten each forward."""
    storage = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output can be a tensor or tuple depending on layer impl
            h = output[0] if isinstance(output, tuple) else output
            storage[layer_idx] = h.detach()
        return hook

    hooks = []
    for idx in layer_indices:
        layer = model.model.layers[idx]
        hooks.append(layer.register_forward_hook(make_hook(idx)))
    return hooks, storage


def run_batched_ablations(model, input_ids, attention_masks, storage, device):
    """Run a batch of forward passes with different attention masks.

    input_ids: (1, P) — single prompt, shared across batch
    attention_masks: (B, P) — one row per ablation condition
    storage: dict filled by hooks with tensors of shape (B, P, d)

    Returns:
      last_hidden: dict {layer_idx: (B, d) last-position hidden states}
      logits: (B, V) last-position logits
    """
    B = attention_masks.shape[0]
    P = input_ids.shape[1]

    # Broadcast input_ids to batch dim
    ids_batched = input_ids.expand(B, P).contiguous()

    with torch.inference_mode():
        out = model(
            input_ids=ids_batched,
            attention_mask=attention_masks,
            use_cache=False,
        )

    # Pull last-token hidden state per layer per batch row
    last_hidden = {}
    for layer_idx, h in storage.items():
        # h: (B, P, d). Last token is position P-1.
        last_hidden[layer_idx] = h[:, -1, :].float().cpu().numpy()

    logits = out.logits[:, -1, :].float().cpu().numpy()
    return last_hidden, logits


def compute_source_map(model, tokenizer, prompt_text, layer_targets, batch_size, device):
    """For one prompt, compute the per-position source map at each target layer.

    Returns:
      dict with:
        prompt_len, token_ids, tokens,
        baseline_hidden: {layer: (d,) np array}
        baseline_logits: (V,) np
        baseline_next_id: int
        source_map: {layer: (P_ablatable,) np array of L2 distances}
        logit_delta: (P_ablatable,) np of Δlogit on baseline next token
        ablated_positions: list of ints (positions 1..P-1, BOS=0 excluded)
    """
    ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
    P = ids.shape[1]

    # Register hooks
    hooks, storage = get_hooks(model, layer_targets)

    try:
        # Baseline forward with full attention
        baseline_mask = torch.ones((1, P), dtype=torch.long, device=device)
        base_hidden, base_logits = run_batched_ablations(
            model, ids, baseline_mask, storage, device
        )
        baseline_hidden = {l: base_hidden[l][0] for l in layer_targets}  # (d,)
        baseline_logits_row = base_logits[0]
        baseline_next_id = int(baseline_logits_row.argmax())
        baseline_next_logit = float(baseline_logits_row[baseline_next_id])

        # Ablation positions: 1..P-1 (skip BOS). We also skip ablating the
        # last position itself since we measure effect ON the last position.
        ablate_positions = list(range(1, P - 1))
        n_ablate = len(ablate_positions)

        # Prepare all ablation masks: (n_ablate, P)
        all_masks = torch.ones((n_ablate, P), dtype=torch.long, device=device)
        for i, t in enumerate(ablate_positions):
            all_masks[i, t] = 0

        # Process in batches
        source_map = {l: np.zeros(n_ablate, dtype=np.float32) for l in layer_targets}
        logit_delta = np.zeros(n_ablate, dtype=np.float32)

        for start in range(0, n_ablate, batch_size):
            end = min(start + batch_size, n_ablate)
            batch_mask = all_masks[start:end]
            ablated_hidden, ablated_logits = run_batched_ablations(
                model, ids, batch_mask, storage, device
            )
            for l in layer_targets:
                diff = ablated_hidden[l] - baseline_hidden[l][None, :]  # (b, d)
                source_map[l][start:end] = np.linalg.norm(diff, axis=1)
            # Δlogit on baseline next token (positive = ablation lowered the
            # baseline's preferred token, i.e. that position was supportive)
            logit_delta[start:end] = (
                baseline_next_logit - ablated_logits[:, baseline_next_id]
            )
    finally:
        for h in hooks:
            h.remove()

    tokens = [tokenizer.decode([int(t)]) for t in ids[0].tolist()]

    return {
        'prompt_text': prompt_text,
        'prompt_len': int(P),
        'token_ids': ids[0].tolist(),
        'tokens': tokens,
        'baseline_hidden': {int(l): baseline_hidden[l].tolist() for l in layer_targets},
        'baseline_next_id': baseline_next_id,
        'baseline_next_token': tokenizer.decode([baseline_next_id]),
        'baseline_next_logit': baseline_next_logit,
        'ablated_positions': ablate_positions,
        'source_map': {int(l): source_map[l].tolist() for l in layer_targets},
        'logit_delta': logit_delta.tolist(),
    }


# ============================================================================
# OP-2: CKA cross-language alignment
# ============================================================================

def linear_cka(X, Y):
    """Linear CKA between (n, d) matrices."""
    n = X.shape[0]
    if n < 2:
        return float('nan')
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    XtX = Xc @ Xc.T
    YtY = Yc @ Yc.T
    hsic = np.trace(XtX @ YtY) / ((n - 1) ** 2)
    vx = np.trace(XtX @ XtX) / ((n - 1) ** 2)
    vy = np.trace(YtY @ YtY) / ((n - 1) ** 2)
    if vx < 1e-12 or vy < 1e-12:
        return 0.0
    return float(hsic / np.sqrt(vx * vy))


def resample_to_grid(profile, n_grid=20):
    """Resample a 1D profile onto a fixed grid of n_grid points via linear interp."""
    profile = np.asarray(profile, dtype=np.float64)
    if len(profile) == 0:
        return np.zeros(n_grid)
    if len(profile) == 1:
        return np.full(n_grid, profile[0])
    x_src = np.linspace(0, 1, len(profile))
    x_dst = np.linspace(0, 1, n_grid)
    return np.interp(x_dst, x_src, profile)


def compute_op2_metrics(per_problem_results, layer_targets, langs, n_grid=20):
    """For each layer, compute:
      - within_problem_cross_lang CKA (en vs zh on same problem)
      - across_problem_same_lang CKA (en vs en on different problems)
      - raw last-token-state CKA (baseline substrate measurement)

    Uses the source_map profiles, resampled onto n_grid τ-points.
    """
    # Organize: {(problem_idx, lang): result}
    by_key = {(r['problem_idx'], r['lang']): r for r in per_problem_results}

    # Problems that have all langs
    problem_indices = sorted(set(k[0] for k in by_key.keys()))
    complete = [p for p in problem_indices if all((p, l) in by_key for l in langs)]

    out = {}
    for layer in layer_targets:
        layer = int(layer)
        # Build per-lang profile matrix: (n_complete, n_grid)
        lang_profiles = {}
        for lang in langs:
            mats = []
            for p in complete:
                sm = np.asarray(by_key[(p, lang)]['source_map'][str(layer)]
                                if str(layer) in by_key[(p, lang)]['source_map']
                                else by_key[(p, lang)]['source_map'][layer])
                mats.append(resample_to_grid(sm, n_grid))
            lang_profiles[lang] = np.stack(mats, axis=0)  # (n_complete, n_grid)

        # (a) Within-problem cross-language CKA
        # Stack: X = en profiles, Y = zh profiles, row-aligned by problem
        if len(langs) >= 2:
            en_mat = lang_profiles[langs[0]]
            zh_mat = lang_profiles[langs[1]]
            cka_within_problem = linear_cka(en_mat, zh_mat)
        else:
            cka_within_problem = float('nan')

        # (b) Across-problem same-language CKA (null: shuffle)
        # Compare en[:n//2] to en[n//2:] — if problems are drawn iid, this
        # measures how much a single-language profile is shared across problems.
        # If within-problem cross-language > across-problem same-language,
        # there is a genuine cross-language operator signal.
        en_mat = lang_profiles[langs[0]]
        n = en_mat.shape[0]
        if n >= 4:
            cka_null_same_lang = linear_cka(en_mat[: n // 2], en_mat[n // 2:])
        else:
            cka_null_same_lang = float('nan')

        # (c) Raw last-token state CKA — substrate baseline
        en_h = np.stack([
            np.asarray(by_key[(p, langs[0])]['baseline_hidden'][str(layer)]
                       if str(layer) in by_key[(p, langs[0])]['baseline_hidden']
                       else by_key[(p, langs[0])]['baseline_hidden'][layer])
            for p in complete
        ], axis=0)
        zh_h = np.stack([
            np.asarray(by_key[(p, langs[1])]['baseline_hidden'][str(layer)]
                       if str(layer) in by_key[(p, langs[1])]['baseline_hidden']
                       else by_key[(p, langs[1])]['baseline_hidden'][layer])
            for p in complete
        ], axis=0)
        cka_raw_substrate = linear_cka(en_h, zh_h)

        out[layer] = {
            'cka_within_problem_cross_lang': cka_within_problem,
            'cka_across_problem_same_lang': cka_null_same_lang,
            'cka_raw_last_token_substrate': cka_raw_substrate,
            'n_complete_problems': len(complete),
        }
    return out


# ============================================================================
# MAIN
# ============================================================================

def run(model_key='3b', dry=False, langs=None, batch_size=BATCH_SIZE_DEFAULT):
    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg['name']

    if langs is None:
        langs = ['en'] if dry else ['en', 'zh']

    if model_key == '3b':
        layer_targets = L_TARGETS_3B if not dry else [32]
    else:
        layer_targets = L_TARGETS_7B if not dry else [L_TARGETS_7B[3]]

    print(f"\n{'=' * 70}")
    print(f"EXP: Causal Source Map — {model_name}")
    print(f"  dry={dry}  langs={langs}  layers={layer_targets}  batch={batch_size}")
    print(f"{'=' * 70}")
    t0 = time.time()

    # Problems
    print("\n[1] Generating problems (seed=42, matches C2c)...", flush=True)
    all_problems = generate_problems(n_per_cat=40)
    test_problems = get_test_subset(all_problems, n_per_cat=4)  # 20
    if dry:
        test_problems = test_problems[:3]
    print(f"    {len(test_problems)} test problems", flush=True)

    # Model
    print(f"\n[2] Loading {model_name}...", flush=True)
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
    print(f"    Loaded. VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB "
          f"({time.time() - t0:.0f}s)", flush=True)

    # Per-problem ablation
    print(f"\n[3] Running causal ablations...", flush=True)
    per_problem = []
    for p_idx, p in enumerate(test_problems):
        for lang in langs:
            t_start = time.time()
            prompt = build_prompt(tokenizer, p[lang], style='chat')
            try:
                result = compute_source_map(
                    model, tokenizer, prompt, layer_targets, batch_size, device
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    OOM at batch={batch_size}, retrying batch={batch_size // 2}",
                      flush=True)
                result = compute_source_map(
                    model, tokenizer, prompt, layer_targets, batch_size // 2, device
                )
            result['problem_idx'] = p_idx
            result['category'] = p['category']
            result['lang'] = lang
            result['elapsed_s'] = time.time() - t_start
            per_problem.append(result)

            # Print top-3 most influential positions at L32 (or first target)
            l_report = layer_targets[min(4, len(layer_targets) - 1)]
            sm = np.asarray(result['source_map'][int(l_report)])
            top3 = np.argsort(-sm)[:3]
            top3_tokens = [result['tokens'][result['ablated_positions'][i]] for i in top3]
            print(f"    [{p_idx + 1:2d}/{len(test_problems)}] {lang} "
                  f"P={result['prompt_len']} "
                  f"L{l_report} top-3: {top3_tokens} "
                  f"({result['elapsed_s']:.1f}s)", flush=True)

    # Op-2 metrics
    print(f"\n[4] Computing Op-2 cross-language CKA...", flush=True)
    if len(langs) >= 2:
        op2 = compute_op2_metrics(per_problem, layer_targets, langs)
        print(f"\n{'=' * 70}")
        print("SUMMARY — cross-language CKA of source maps vs substrate")
        print(f"{'=' * 70}")
        print(f"{'Layer':>6}  {'within-prob xLang':>18}  "
              f"{'null (xProb same-L)':>20}  {'raw substrate':>14}")
        for layer in layer_targets:
            m = op2[layer]
            print(f"  L{layer:02d}  "
                  f"{m['cka_within_problem_cross_lang']:>18.4f}  "
                  f"{m['cka_across_problem_same_lang']:>20.4f}  "
                  f"{m['cka_raw_last_token_substrate']:>14.4f}")
    else:
        op2 = None
        print("    (skipped — need ≥2 languages)", flush=True)

    # Save
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)
    suffix = '_dry' if dry else ''
    out_path = out_dir / f'exp_causal_source_map_{model_key}{suffix}.json'

    output = {
        'config': {
            'model': model_name,
            'model_key': model_key,
            'layer_targets': layer_targets,
            'langs': langs,
            'dry': dry,
            'batch_size': batch_size,
            'seed': SEED,
        },
        'per_problem': per_problem,
        'op2': op2,
        'elapsed_total_s': time.time() - t0,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output


def main():
    parser = argparse.ArgumentParser(description='Causal source map experiment')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), default='3b')
    parser.add_argument('--dry', action='store_true',
                        help='3 problems × EN × L32 only')
    parser.add_argument('--langs', nargs='+', default=None)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE_DEFAULT)
    args = parser.parse_args()
    run(
        model_key=args.model,
        dry=args.dry,
        langs=args.langs,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
