"""Notation vs language token split on the user-content slice.

Reads exp_causal_source_map_3b.json. Within the user-content slice (the math
problem text only), partitions positions into:

  NOTATION  — digits, parentheses, comma, math operators, dot, equals
  LANGUAGE  — words ('Find', 'value', 'of', '求', '组合数', '的值', etc.)

Then runs the same Op-2 CKA + pairwise cosine on each subset separately.

Question: is the weak Vision A signal at L20 (CKA xLang same=0.79 vs sameL
diff=0.36) being driven by NOTATION tokens (digits/parens are byte-identical
across en/zh and could trivially align) or by LANGUAGE tokens (which differ
across languages and thus require operator-level invariance)?

Outcome reading:
  notation drives signal      → trivial. xLang same on language tokens ≈ null
  language drives signal      → real Vision A. xLang same on language tokens > null
  both contribute             → Vision A real but mediated partly by notation
  neither shows separation    → contaminated entirely; signal was at digit level
"""
import json
from itertools import combinations

import numpy as np


def find_user_content_span(tokens):
    im_start = [i for i, t in enumerate(tokens) if t == '<|im_start|>']
    if len(im_start) < 2:
        raise ValueError(f"need 2+ <|im_start|>, got {len(im_start)}")
    user_marker = im_start[1]
    start = user_marker + 3  # skip <|im_start|>, 'user', '\n'
    end = None
    for i in range(start, len(tokens)):
        if tokens[i] == '<|im_end|>':
            end = i
            break
    if end is None:
        raise ValueError("no <|im_end|> after user marker")
    return start, end


def is_notation(tok):
    """Notation tokens: anything that's not a word in any language.

    Empirically: digits, single ASCII punctuation, parens, dots, commas,
    operators. Words (including CJK characters) get classified as language.
    """
    s = tok.strip()
    if not s:
        return False  # pure whitespace — drop entirely from both buckets
    # Pure-digit token
    if s.isdigit():
        return True
    # Single non-alphanum char (punct/operator/paren)
    if len(s) == 1 and not s.isalnum():
        return True
    # Multi-char token where every char is non-alphanum (e.g. ').', '),')
    if all(not c.isalnum() for c in s):
        return True
    # Mixed digit + punctuation (e.g. '18,', '3).')
    has_alpha = any(c.isalpha() for c in s)
    if not has_alpha:
        return True
    return False  # has at least one alphabetic / CJK character → language


def slice_subsets(result, layer):
    """Return (notation_profile, language_profile) — source-map values
    restricted to notation and language tokens within the user-content span."""
    sm = np.asarray(result['source_map'][str(layer)])
    tokens = result['tokens']
    start, end = find_user_content_span(tokens)
    notation_vals = []
    language_vals = []
    notation_toks = []
    language_toks = []
    for pos in range(start, end):
        sm_idx = pos - 1  # ablated_positions[i] = i+1
        if sm_idx < 0 or sm_idx >= len(sm):
            continue
        tok = tokens[pos]
        if is_notation(tok):
            notation_vals.append(sm[sm_idx])
            notation_toks.append(tok)
        else:
            language_vals.append(sm[sm_idx])
            language_toks.append(tok)
    return (np.asarray(notation_vals), np.asarray(language_vals),
            notation_toks, language_toks)


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
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def analyze_subset(by_key, complete, langs, layer, get_subset, label):
    """Generic analysis for one subset (notation or language)."""
    prof = {lang: [] for lang in langs}
    n_empty = 0
    for p in complete:
        for lang in langs:
            r = by_key[(p, lang)]
            try:
                vals = get_subset(r, layer)
            except ValueError:
                vals = np.array([])
            if len(vals) == 0:
                n_empty += 1
            prof[lang].append(resample(vals))

    en = np.stack(prof[langs[0]], axis=0)
    zh = np.stack(prof[langs[1]], axis=0)
    n = en.shape[0]

    cka_xLang_same = linear_cka(en, zh)
    cka_sameL_diff = linear_cka(en[: n // 2], en[n // 2:])
    cka_xLang_diff = linear_cka(en[: n // 2], zh[n // 2:])

    xLang_same_pairs = np.array([cosine(en[i], zh[i]) for i in range(n)])
    sameL_diff_pairs = []
    for i, j in combinations(range(n), 2):
        sameL_diff_pairs.append(cosine(en[i], en[j]))
        sameL_diff_pairs.append(cosine(zh[i], zh[j]))
    sameL_diff_pairs = np.array(sameL_diff_pairs)

    paired_deltas = []
    for i in range(n):
        others_en = [cosine(en[i], en[j]) for j in range(n) if j != i]
        others_zh = [cosine(zh[i], zh[j]) for j in range(n) if j != i]
        ref = np.mean(others_en + others_zh)
        paired_deltas.append(xLang_same_pairs[i] - ref)
    paired_deltas = np.array(paired_deltas)

    return {
        'subset': label,
        'layer': layer,
        'n_empty_profiles': n_empty,
        'cka_xLang_same': float(cka_xLang_same),
        'cka_sameLang_diff': float(cka_sameL_diff),
        'cka_xLang_diff': float(cka_xLang_diff),
        'cos_xLang_same_mean': float(xLang_same_pairs.mean()),
        'cos_sameLang_diff_mean': float(sameL_diff_pairs.mean()),
        'paired_delta_mean': float(paired_deltas.mean()),
        'paired_delta_n_positive': int((paired_deltas > 0).sum()),
        'paired_delta_n_total': int(n),
    }


def main():
    in_path = '/home/greg/Desktop/Projects/BrainInsideTheMachine/output/exp_causal_source_map_3b.json'
    out_path = '/home/greg/Desktop/Projects/BrainInsideTheMachine/output/source_map_split_notation_vs_language.json'

    with open(in_path) as f:
        d = json.load(f)

    per_problem = d['per_problem']
    layer_targets = d['config']['layer_targets']
    langs = d['config']['langs']

    by_key = {(r['problem_idx'], r['lang']): r for r in per_problem}
    problem_indices = sorted(set(k[0] for k in by_key.keys()))
    complete = [p for p in problem_indices if all((p, l) in by_key for l in langs)]

    # Diagnostic: show partition for first 3 problems
    print(f"Complete problems: {len(complete)}\n")
    print("=" * 80)
    print("PARTITION DIAGNOSTIC — first 3 problems")
    print("=" * 80)
    for p in complete[:3]:
        for lang in langs:
            r = by_key[(p, lang)]
            _, _, n_toks, l_toks = slice_subsets(r, layer_targets[0])
            print(f"  prob{p}/{lang}:")
            print(f"    notation ({len(n_toks)}): {n_toks}")
            print(f"    language ({len(l_toks)}): {l_toks}")

    # Helper closures for the two subsets
    def get_notation(r, layer):
        return slice_subsets(r, layer)[0]

    def get_language(r, layer):
        return slice_subsets(r, layer)[1]

    print(f"\n{'=' * 80}")
    print("CKA + PAIRWISE — NOTATION subset (digits, parens, punctuation)")
    print(f"{'=' * 80}")
    header = (f"{'Layer':>6}  "
              f"{'CKA xL same':>12}  {'CKA sameL':>10}  {'CKA xL diff':>12}  "
              f"{'cos xL same':>12}  {'cos sameL':>10}  {'paired Δ':>10}  {'sign':>6}")
    print(header)
    print('-' * len(header))

    notation_results = []
    for layer in layer_targets:
        m = analyze_subset(by_key, complete, langs, layer, get_notation, 'notation')
        notation_results.append(m)
        sign = f"{m['paired_delta_n_positive']}/{m['paired_delta_n_total']}"
        print(f"  L{layer:02d}  "
              f"{m['cka_xLang_same']:>12.4f}  {m['cka_sameLang_diff']:>10.4f}  "
              f"{m['cka_xLang_diff']:>12.4f}  "
              f"{m['cos_xLang_same_mean']:>12.4f}  {m['cos_sameLang_diff_mean']:>10.4f}  "
              f"{m['paired_delta_mean']:>+10.4f}  {sign:>6}")

    print(f"\n{'=' * 80}")
    print("CKA + PAIRWISE — LANGUAGE subset (words: 'Find', 'value', '求', '组合数', ...)")
    print(f"{'=' * 80}")
    print(header)
    print('-' * len(header))

    language_results = []
    for layer in layer_targets:
        m = analyze_subset(by_key, complete, langs, layer, get_language, 'language')
        language_results.append(m)
        sign = f"{m['paired_delta_n_positive']}/{m['paired_delta_n_total']}"
        print(f"  L{layer:02d}  "
              f"{m['cka_xLang_same']:>12.4f}  {m['cka_sameLang_diff']:>10.4f}  "
              f"{m['cka_xLang_diff']:>12.4f}  "
              f"{m['cos_xLang_same_mean']:>12.4f}  {m['cos_sameLang_diff_mean']:>10.4f}  "
              f"{m['paired_delta_mean']:>+10.4f}  {sign:>6}")

    # Side-by-side delta interpretation
    print(f"\n{'=' * 80}")
    print("HEAD-TO-HEAD: gap (xLang same − sameLang diff) on each subset")
    print(f"{'=' * 80}")
    print(f"{'Layer':>6}  {'NOTATION CKA Δ':>16}  {'LANGUAGE CKA Δ':>16}  {'verdict':<35}")
    print('-' * 80)
    for nr, lr in zip(notation_results, language_results):
        n_delta = nr['cka_xLang_same'] - nr['cka_sameLang_diff']
        l_delta = lr['cka_xLang_same'] - lr['cka_sameLang_diff']
        if l_delta > n_delta + 0.05:
            verdict = "language carries it (real Vision A)"
        elif n_delta > l_delta + 0.05:
            verdict = "notation carries it (trivial)"
        elif l_delta > 0.05 and n_delta > 0.05:
            verdict = "both contribute"
        else:
            verdict = "neither shows clean separation"
        print(f"  L{nr['layer']:02d}  {n_delta:>+16.4f}  {l_delta:>+16.4f}  {verdict}")

    # Save
    out = {
        'analysis': 'notation_vs_language_split',
        'notation_results': notation_results,
        'language_results': language_results,
        'config': {
            'source': in_path,
            'n_complete': len(complete),
            'languages': langs,
            'layers': layer_targets,
        },
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
