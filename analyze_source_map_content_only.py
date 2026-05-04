"""Post-hoc analysis: redo Op-2 CKA on content-only positions.

Reads output/exp_causal_source_map_3b.json. For each problem × language,
identifies which ablated positions correspond to chat-template scaffolding
(user, assistant, system, newlines, punctuation) vs actual content tokens
(math statement). Recomputes cross-language CKA using only content-token
positions.

If within-problem cross-lang CKA remains elevated on the content-only slice,
the operator signal is real. If it collapses to the null, it was all template.
"""
import json
import numpy as np

TEMPLATE_STOPWORDS = {
    'user', 'assistant', 'system', 'im_start', 'im_end',
    '<|im_start|>', '<|im_end|>', '<|endoftext|>',
    '\n', ' \n', '\n\n', '', ' ',
}


def is_template_token(tok):
    s = tok.strip().lower()
    if s in TEMPLATE_STOPWORDS:
        return True
    if s.startswith('<|') and s.endswith('|>'):
        return True
    # Whitespace / punctuation only
    if all(not c.isalnum() for c in s):
        return True
    return False


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


def resample(profile, n_grid=20):
    profile = np.asarray(profile, dtype=np.float64)
    if len(profile) == 0:
        return np.zeros(n_grid)
    if len(profile) == 1:
        return np.full(n_grid, profile[0])
    x_src = np.linspace(0, 1, len(profile))
    x_dst = np.linspace(0, 1, n_grid)
    return np.interp(x_dst, x_src, profile)


def build_content_profile(result, layer):
    """Extract the source map at layer, keeping only content-token positions."""
    sm = np.asarray(result['source_map'][str(layer)])
    ablated_positions = result['ablated_positions']
    tokens = result['tokens']

    keep_idx = []
    content_tokens = []
    for i, pos in enumerate(ablated_positions):
        tok = tokens[pos]
        if not is_template_token(tok):
            keep_idx.append(i)
            content_tokens.append(tok)
    return sm[keep_idx], content_tokens


def main():
    with open('/home/greg/Desktop/Projects/BrainInsideTheMachine/output/exp_causal_source_map_3b.json') as f:
        d = json.load(f)

    per_problem = d['per_problem']
    layer_targets = d['config']['layer_targets']
    langs = d['config']['langs']

    by_key = {(r['problem_idx'], r['lang']): r for r in per_problem}
    problem_indices = sorted(set(k[0] for k in by_key.keys()))
    complete = [p for p in problem_indices if all((p, l) in by_key for l in langs)]

    print(f"Complete problems: {len(complete)}")
    print(f"Languages: {langs}")

    # Diagnose content token counts per problem per language
    print("\nContent-token counts (P_total / P_content):")
    for p in complete[:5]:
        for lang in langs:
            r = by_key[(p, lang)]
            _, ct = build_content_profile(r, layer_targets[0])
            print(f"  prob{p}/{lang}: {len(r['ablated_positions'])} / {len(ct)} — sample: {ct[:8]}")

    # For each layer, redo the CKA on content-only profiles
    print(f"\n{'=' * 80}")
    print("CKA of source maps — content tokens only (template excluded)")
    print(f"{'=' * 80}")
    header = (f"{'Layer':>6}  "
              f"{'ALL xLang':>10}  {'CONTENT xLang':>14}  "
              f"{'ALL null':>10}  {'CONTENT null':>14}  "
              f"{'raw subst':>10}")
    print(header)
    print('-' * len(header))

    results = {}
    for layer in layer_targets:
        # Full (all positions) cross-language CKA — already in Op-2 output but recompute for sanity
        lang_mats_all = {}
        lang_mats_content = {}
        for lang in langs:
            all_profiles = []
            content_profiles = []
            for p in complete:
                r = by_key[(p, lang)]
                sm_all = np.asarray(r['source_map'][str(layer)])
                sm_content, _ = build_content_profile(r, layer)
                all_profiles.append(resample(sm_all))
                content_profiles.append(resample(sm_content))
            lang_mats_all[lang] = np.stack(all_profiles, axis=0)
            lang_mats_content[lang] = np.stack(content_profiles, axis=0)

        # Within-problem cross-language
        cka_all_xlang = linear_cka(lang_mats_all[langs[0]], lang_mats_all[langs[1]])
        cka_content_xlang = linear_cka(lang_mats_content[langs[0]], lang_mats_content[langs[1]])

        # Null: across-problem same-language (split half)
        en_all = lang_mats_all[langs[0]]
        en_content = lang_mats_content[langs[0]]
        n = en_all.shape[0]
        cka_all_null = linear_cka(en_all[: n // 2], en_all[n // 2:])
        cka_content_null = linear_cka(en_content[: n // 2], en_content[n // 2:])

        # Raw substrate (from baseline hidden states)
        en_h = np.stack([
            np.asarray(by_key[(p, langs[0])]['baseline_hidden'][str(layer)])
            for p in complete
        ], axis=0)
        zh_h = np.stack([
            np.asarray(by_key[(p, langs[1])]['baseline_hidden'][str(layer)])
            for p in complete
        ], axis=0)
        cka_substrate = linear_cka(en_h, zh_h)

        print(f"  L{layer:02d}  "
              f"{cka_all_xlang:>10.4f}  {cka_content_xlang:>14.4f}  "
              f"{cka_all_null:>10.4f}  {cka_content_null:>14.4f}  "
              f"{cka_substrate:>10.4f}")

        results[layer] = {
            'cka_all_xlang': cka_all_xlang,
            'cka_content_xlang': cka_content_xlang,
            'cka_all_null': cka_all_null,
            'cka_content_null': cka_content_null,
            'cka_raw_substrate': cka_substrate,
        }

    print(f"\n{'=' * 80}")
    print("Interpretation guide:")
    print(f"{'=' * 80}")
    print("""
  CONTENT xLang vs CONTENT null:
    If CONTENT xLang >> CONTENT null, the operator signal survives the
    template confound — cross-language source maps share structure on
    actual content positions. Strong evidence for the operator ansatz.

    If CONTENT xLang ≈ CONTENT null, the original signal was all template
    and the operator does not carry language-invariant structure.

  CONTENT xLang vs raw substrate:
    If CONTENT xLang > raw substrate on the same problems, the operator
    is MORE language-invariant than the substrate. Strongest possible
    outcome for the operator ansatz.

    If raw substrate > CONTENT xLang, the substrate is already as
    language-invariant as the operator; promoting to operator-thinking
    adds no invariance but may add mechanistic clarity.
""")

    # Save
    out = {
        'analysis': 'content_only_cka',
        'layer_results': results,
        'per_layer_content_count_sample': [
            (p, lang, len(build_content_profile(by_key[(p, lang)], layer_targets[0])[1]))
            for p in complete[:5] for lang in langs
        ],
    }
    with open('/home/greg/Desktop/Projects/BrainInsideTheMachine/output/source_map_content_only_cka.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: output/source_map_content_only_cka.json")


if __name__ == '__main__':
    main()
