"""Post-hoc analysis: redo Op-2 CKA on the USER-CONTENT slice only.

Reads output/exp_causal_source_map_3b.json. The previous content-only filter
(analyze_source_map_content_only.py) stripped chat-template scaffolding but
LEFT the system prompt intact, which is byte-identical across languages and
inflates BOTH cross-language similarity AND the within-language null. Result:
contaminated.

This script finds the slice of token positions that correspond ONLY to the
user's math problem text — between the second <|im_start|> + 'user\\n' marker
and the next <|im_end|> — and recomputes:

  (1) CKA of source maps on the user-content slice
        - cross-language same-problem (Op-2 signal)
        - same-language across-problem (null)
        - cross-language across-problem (stricter null)

  (2) PAIRWISE COSINE — explicit Vision A test
        For every pair of problems i, j compute cosine between resampled
        source-map profiles. Compare:
          mean cos(prob_i^en, prob_i^zh)        — translated paraphrase
          mean cos(prob_i^en, prob_j^en), i!=j  — same-lang diff problem
          mean cos(prob_i^en, prob_j^zh), i!=j  — cross-lang diff problem
        Vision A predicts:  xLang_same  >  sameLang_diff  >  xLang_diff.

  (3) Diagnostic: per-problem (xLang_same - mean sameLang_diff). Positive
      on majority of problems = Vision A signal real per-instance, not just
      population aggregate.

NO GPU NEEDED. Pure post-hoc on stored source maps.
"""
import json
from itertools import combinations

import numpy as np


# ============================================================================
# Span detection — find user-message slice in stored tokens
# ============================================================================

def find_user_content_span(tokens):
    """Return (start, end) half-open interval of token positions belonging to
    the user's message body, exclusive of role markers and <|im_end|>.

    Qwen2.5 chat template structure:
        <|im_start|> system \\n  ... <|im_end|> \\n
        <|im_start|> user   \\n  <user content>  <|im_end|> \\n
        <|im_start|> assistant \\n  (generation prompt)

    We find the SECOND <|im_start|> (= user role), skip the role tokens,
    and stop at the next <|im_end|>.
    """
    im_start_positions = [i for i, t in enumerate(tokens) if t == '<|im_start|>']
    if len(im_start_positions) < 2:
        raise ValueError(f"Expected ≥2 <|im_start|> markers; found {len(im_start_positions)}")
    user_marker = im_start_positions[1]
    # Skip <|im_start|>, 'user', '\n'
    start = user_marker + 3
    # Find next <|im_end|>
    end = None
    for i in range(start, len(tokens)):
        if tokens[i] == '<|im_end|>':
            end = i
            break
    if end is None:
        raise ValueError("No <|im_end|> after user role marker")
    return start, end


def slice_source_map(result, layer):
    """Return the source-map values at the user-content positions for a layer.

    The stored source_map array is indexed 0..n_ablate-1, where index i
    corresponds to ablating absolute position ablated_positions[i].
    Since ablated_positions == list(range(1, P-1)), source_map index = pos - 1.
    """
    sm = np.asarray(result['source_map'][str(layer)])
    tokens = result['tokens']
    start, end = find_user_content_span(tokens)
    # Convert absolute token positions to source_map indices.
    # ablated_positions[i] = i + 1, so source_map index for position p = p - 1.
    sm_start = max(0, start - 1)
    sm_end = max(sm_start, end - 1)
    user_slice = sm[sm_start:sm_end]
    user_tokens = tokens[start:end]
    return user_slice, user_tokens


# ============================================================================
# Resampling + CKA + cosine
# ============================================================================

def resample(profile, n_grid=20):
    profile = np.asarray(profile, dtype=np.float64)
    if len(profile) == 0:
        return np.zeros(n_grid)
    if len(profile) == 1:
        return np.full(n_grid, profile[0])
    x_src = np.linspace(0, 1, len(profile))
    x_dst = np.linspace(0, 1, n_grid)
    return np.interp(x_dst, x_src, profile)


def linear_cka(X, Y):
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


def cosine(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# ============================================================================
# Main
# ============================================================================

def main():
    in_path = '/home/greg/Desktop/Projects/BrainInsideTheMachine/output/exp_causal_source_map_3b.json'
    out_path = '/home/greg/Desktop/Projects/BrainInsideTheMachine/output/source_map_user_content_cka.json'

    with open(in_path) as f:
        d = json.load(f)

    per_problem = d['per_problem']
    layer_targets = d['config']['layer_targets']
    langs = d['config']['langs']

    by_key = {(r['problem_idx'], r['lang']): r for r in per_problem}
    problem_indices = sorted(set(k[0] for k in by_key.keys()))
    complete = [p for p in problem_indices if all((p, l) in by_key for l in langs)]

    print(f"Complete problems: {len(complete)}")
    print(f"Languages: {langs}")
    print(f"Layers: {layer_targets}")

    # ---- Diagnostic: show what got sliced for first 5 problems ----
    print(f"\n{'=' * 80}")
    print("USER-CONTENT SLICE — sample of what survives")
    print(f"{'=' * 80}")
    sample_meta = []
    for p in complete[:5]:
        for lang in langs:
            r = by_key[(p, lang)]
            try:
                user_slice, user_tokens = slice_source_map(r, layer_targets[0])
                P_total = len(r['ablated_positions'])
                P_user = len(user_slice)
                preview = ''.join(user_tokens)[:60]
                print(f"  prob{p}/{lang}: ablated={P_total}  user_slice={P_user}  "
                      f"text={preview!r}")
                sample_meta.append({
                    'problem': p, 'lang': lang,
                    'P_total': P_total, 'P_user': P_user,
                    'user_text': ''.join(user_tokens),
                })
            except ValueError as e:
                print(f"  prob{p}/{lang}: SLICE FAILED — {e}")

    # ---- Build per-language profile matrices on the user-content slice ----
    print(f"\n{'=' * 80}")
    print("CKA — source maps on USER-CONTENT slice (system + template excluded)")
    print(f"{'=' * 80}")
    header = (f"{'Layer':>6}  "
              f"{'xLang same-prob':>16}  "
              f"{'sameL diff-prob':>16}  "
              f"{'xLang diff-prob':>16}  "
              f"{'raw substrate':>14}")
    print(header)
    print('-' * len(header))

    layer_results = {}
    for layer in layer_targets:
        # User-content slice profiles, resampled to a common grid
        user_profiles = {lang: [] for lang in langs}
        for p in complete:
            for lang in langs:
                r = by_key[(p, lang)]
                try:
                    sm_user, _ = slice_source_map(r, layer)
                except ValueError:
                    sm_user = np.array([])
                user_profiles[lang].append(resample(sm_user))
        # (n_complete, n_grid) per language
        mats = {lang: np.stack(user_profiles[lang], axis=0) for lang in langs}

        # (a) Cross-language same-problem CKA — Op-2 headline
        cka_xLang_same = linear_cka(mats[langs[0]], mats[langs[1]])

        # (b) Same-language across-problem CKA — naive null (split half)
        en = mats[langs[0]]
        n_c = en.shape[0]
        cka_sameL_diff = linear_cka(en[: n_c // 2], en[n_c // 2:])

        # (c) Cross-language across-problem CKA — stricter null
        # Take en[:n//2] vs zh[n//2:]: diff problems AND diff language
        cka_xLang_diff = linear_cka(mats[langs[0]][: n_c // 2],
                                    mats[langs[1]][n_c // 2:])

        # Raw substrate (last-token hidden state) — unchanged from before
        en_h = np.stack([
            np.asarray(by_key[(p, langs[0])]['baseline_hidden'][str(layer)])
            for p in complete
        ], axis=0)
        zh_h = np.stack([
            np.asarray(by_key[(p, langs[1])]['baseline_hidden'][str(layer)])
            for p in complete
        ], axis=0)
        cka_raw = linear_cka(en_h, zh_h)

        print(f"  L{layer:02d}  "
              f"{cka_xLang_same:>16.4f}  "
              f"{cka_sameL_diff:>16.4f}  "
              f"{cka_xLang_diff:>16.4f}  "
              f"{cka_raw:>14.4f}")

        layer_results[layer] = {
            'cka_xLang_same_problem': cka_xLang_same,
            'cka_sameLang_diff_problem': cka_sameL_diff,
            'cka_xLang_diff_problem': cka_xLang_diff,
            'cka_raw_substrate': cka_raw,
        }

    # ---- Pairwise cosine — explicit Vision A test ----
    print(f"\n{'=' * 80}")
    print("PAIRWISE COSINE on user-content slice — Vision A test")
    print("(translated paraphrase vs same-lang diff problem vs diff-lang diff problem)")
    print(f"{'=' * 80}")
    header2 = (f"{'Layer':>6}  "
               f"{'xL same μ':>11}  "
               f"{'sameL diff μ':>13}  "
               f"{'xL diff μ':>11}  "
               f"{'paired Δ':>10}  "
               f"{'sign':>6}")
    print(header2)
    print('-' * len(header2))

    for layer in layer_targets:
        # Build resampled user-content profiles (already done above; redo for clarity)
        prof = {lang: [] for lang in langs}
        for p in complete:
            for lang in langs:
                r = by_key[(p, lang)]
                try:
                    sm_user, _ = slice_source_map(r, layer)
                except ValueError:
                    sm_user = np.array([])
                prof[lang].append(resample(sm_user))
        en = np.stack(prof[langs[0]], axis=0)
        zh = np.stack(prof[langs[1]], axis=0)

        # (a) xLang same-problem: cos(en_i, zh_i) for each i
        xLang_same = np.array([cosine(en[i], zh[i]) for i in range(len(complete))])

        # (b) sameLang diff-problem: cos(en_i, en_j) for i<j  AND  cos(zh_i, zh_j)
        sameL_diff = []
        for i, j in combinations(range(len(complete)), 2):
            sameL_diff.append(cosine(en[i], en[j]))
            sameL_diff.append(cosine(zh[i], zh[j]))
        sameL_diff = np.array(sameL_diff)

        # (c) xLang diff-problem: cos(en_i, zh_j) for i!=j
        xLang_diff = []
        for i in range(len(complete)):
            for j in range(len(complete)):
                if i == j:
                    continue
                xLang_diff.append(cosine(en[i], zh[j]))
        xLang_diff = np.array(xLang_diff)

        # Per-problem paired delta: xLang_same_i - mean over j!=i of sameLang_diff
        paired_deltas = []
        for i in range(len(complete)):
            others_en = [cosine(en[i], en[j]) for j in range(len(complete)) if j != i]
            others_zh = [cosine(zh[i], zh[j]) for j in range(len(complete)) if j != i]
            ref = np.mean(others_en + others_zh)
            paired_deltas.append(xLang_same[i] - ref)
        paired_deltas = np.array(paired_deltas)
        n_pos = int((paired_deltas > 0).sum())

        print(f"  L{layer:02d}  "
              f"{xLang_same.mean():>11.4f}  "
              f"{sameL_diff.mean():>13.4f}  "
              f"{xLang_diff.mean():>11.4f}  "
              f"{paired_deltas.mean():>+10.4f}  "
              f"{n_pos:>3d}/{len(complete)}")

        layer_results[layer].update({
            'cos_xLang_same_mean': float(xLang_same.mean()),
            'cos_xLang_same_std': float(xLang_same.std()),
            'cos_sameLang_diff_mean': float(sameL_diff.mean()),
            'cos_sameLang_diff_std': float(sameL_diff.std()),
            'cos_xLang_diff_mean': float(xLang_diff.mean()),
            'cos_xLang_diff_std': float(xLang_diff.std()),
            'paired_delta_mean': float(paired_deltas.mean()),
            'paired_delta_n_positive': n_pos,
            'paired_delta_n_total': int(len(complete)),
        })

    # ---- Interpretation guide ----
    print(f"\n{'=' * 80}")
    print("Interpretation")
    print(f"{'=' * 80}")
    print("""
  CKA TABLE
    xLang same-prob >> sameL diff-prob:
      Operator carries cross-language structure on math content alone.
      Strong evidence for Vision A (language-agnostic core 'f') AND for
      Vision B (canonical context→read-head operator).

    xLang same-prob ≈ sameL diff-prob:
      Source maps are problem-specific within a language but the
      translated-paraphrase signal does not exceed it. Naive operator
      ansatz dies on math problems in this regime.

    xLang same-prob > xLang diff-prob >> 0:
      Even diff-problem cross-lang shows residual structure (possibly
      from math notation '(', ')', digits). Worth a digit-only ablation.

  PAIRWISE COSINE
    paired Δ > 0 on N/N problems = unanimous Vision A signal.
    paired Δ > 0 on majority but not all = Vision A holds in aggregate.
    paired Δ < 0 on majority = problems are MORE similar within language,
      meaning the source map is dominated by language-specific reading
      strategy and 'f' lives elsewhere.

  L35 SPECIFIC
    If xLang same-prob drops vs L30/L32 in user-content slice (it did in
    contaminated run, 0.79→0.62), the read head becomes language-specific
    at the final layer. Predicted by PC0 swap and tied-embedding rupture.
""")

    # Save
    out = {
        'analysis': 'user_content_slice_cka_and_pairwise',
        'sample_meta': sample_meta,
        'layer_results': {int(k): v for k, v in layer_results.items()},
        'config': {
            'source': in_path,
            'n_complete_problems': len(complete),
            'languages': langs,
            'layers': layer_targets,
            'n_grid': 20,
        },
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
