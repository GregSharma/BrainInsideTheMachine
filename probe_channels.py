"""probe_channels: Where does verbatim memory live vs reasoning?

The north star question: does the model maintain dual channels
(semantic in h, verbatim in attention) or one monolithic space?

Approach: vary one dimension at a time. Report raw numbers.

Part A: TASK PREFIX — does query intent live in h?
  bare sentence vs "repeat:" vs "what was the 3rd word?" vs reasoning
  If h changes with prefix, task intent is in residual stream.
  If h stays same, task intent is elsewhere (attention, output head).

Part B: LOGICAL QUANTIFIER — pure reasoning variation
  Fix entity, property, domain, language. Change only the logic.
  "all" vs "none" vs "some" vs "only"

Part C: NEGATION SYLLOGISM — does flipping logic move h?
  Same syllogism structure, same entities, different truth value.

Part D: GENERATION — repeat task fixed point
  Ask model to repeat. Compare gen-time h to encoding-time h
  for the SAME tokens. How close is the "copy"?

No system prompt. No chat template. Raw text.
"""
import json, time, sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
DEVICE = 'cuda'
KEY_LAYERS = [5, 13, 18, 26, 30, 33]


class Probe:
    def __init__(self):
        print('loading model...', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
        self.model.eval()
        self._caps = {}
        self._hooks = []
        for L in KEY_LAYERS:
            cap = self._Cap()
            self._caps[L] = cap
            self._hooks.append(self.model.model.layers[L].register_forward_hook(cap))
        print('ready.', flush=True)

    class _Cap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[0].detach().float().cpu().numpy()  # (seq, d)

    def encode(self, text):
        """Raw text -> hidden states at all layers. No chat template."""
        ids = self.tokenizer(text, return_tensors='pt').input_ids.to(DEVICE)
        with torch.inference_mode():
            self.model(ids)
        result = {L: self._caps[L].out.copy() for L in KEY_LAYERS}
        tokens = [self.tokenizer.decode(ids[0, i]) for i in range(ids.shape[1])]
        return result, tokens, ids

    def generate_with_capture(self, text, max_new=64):
        """Generate tokens and capture h at each step.

        Returns: gen_tokens, gen_h_per_step (list of {layer: array(d,)})
        """
        ids = self.tokenizer(text, return_tensors='pt').input_ids.to(DEVICE)
        prompt_len = ids.shape[1]

        gen_tokens = []
        gen_h = []  # list of {layer: array(d,)} per generated token

        for step in range(max_new):
            with torch.inference_mode():
                out = self.model(ids)
            # capture h_last at this step
            h_step = {L: self._caps[L].out[-1].copy() for L in KEY_LAYERS}
            gen_h.append(h_step)

            # greedy next token
            next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            next_tok = self.tokenizer.decode(next_id[0, 0])
            gen_tokens.append(next_tok)

            # check for eos
            if next_id.item() == self.tokenizer.eos_token_id:
                break

            ids = torch.cat([ids, next_id], dim=1)

        return gen_tokens, gen_h, prompt_len

    def close(self):
        for h in self._hooks:
            h.remove()


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_part_a(probe):
    """TASK PREFIX: does query intent live in h?"""
    print('\n' + '='*70)
    print('PART A: TASK PREFIX — does query intent change h?')
    print('='*70)

    content = "the quick brown fox jumps over the lazy dog"

    prompts = [
        ("bare",      content),
        ("repeat",    f"repeat exactly: {content}"),
        ("query_3rd", f"what was the third word in the following sentence? {content}"),
        ("grammar",   f"is this sentence grammatically correct? {content}"),
        ("count",     f"how many words are in this sentence? {content}"),
    ]

    results = {}
    h_data = {}
    for name, text in prompts:
        h_all, tokens, ids = probe.encode(text)
        h_data[name] = {'h_last': {L: h_all[L][-1] for L in KEY_LAYERS},
                        'h_all': h_all, 'tokens': tokens}
        print(f'  {name:12s}: {len(tokens)} tokens — "{text[:60]}..."' if len(text) > 60
              else f'  {name:12s}: {len(tokens)} tokens — "{text}"', flush=True)

    # Compare h_last across conditions
    print(f'\n  h_last cosines (raw):')
    names = [n for n, _ in prompts]
    print(f'  {"":12s}', end='')
    for L in KEY_LAYERS:
        print(f'  L{L:>2d}    ', end='')
    print()

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            label = f'{n1}-{n2}'
            print(f'  {label:24s}', end='')
            for L in KEY_LAYERS:
                c = cos(h_data[n1]['h_last'][L], h_data[n2]['h_last'][L])
                print(f'  {c:.4f}', end='')
            print()

    # Now check: do the CONTENT TOKENS change?
    # Find "the quick brown fox..." tokens in each prompt
    # by matching the last N tokens (content appears at the end)
    bare_tokens = h_data['bare']['tokens']
    n_content = len(bare_tokens)

    print(f'\n  content token alignment ({n_content} content tokens):')
    print(f'  comparing h of same content tokens across conditions')

    for name in ['repeat', 'query_3rd', 'grammar', 'count']:
        other_tokens = h_data[name]['tokens']
        n_other = len(other_tokens)
        # content is at the end of the prompted version
        # find where content starts in the other prompt
        offset = n_other - n_content
        if offset < 0:
            print(f'  {name}: content longer than prompt?? skip')
            continue

        # check token alignment
        aligned = all(bare_tokens[k] == other_tokens[offset + k] for k in range(n_content))

        print(f'\n  bare vs {name} (offset={offset}, aligned={aligned}):')
        print(f'  {"token":>20s}', end='')
        for L in KEY_LAYERS:
            print(f'  L{L:>2d}    ', end='')
        print()

        token_cos = {L: [] for L in KEY_LAYERS}
        for k in range(n_content):
            tok = bare_tokens[k]
            print(f'  {tok:>20s}', end='')
            for L in KEY_LAYERS:
                h_bare = h_data['bare']['h_all'][L][k]
                h_other = h_data[name]['h_all'][L][offset + k]
                c = cos(h_bare, h_other)
                token_cos[L].append(c)
                print(f'  {c:.4f}', end='')
            print()

        print(f'  {"MEAN":>20s}', end='')
        for L in KEY_LAYERS:
            print(f'  {np.mean(token_cos[L]):.4f}', end='')
        print()

    return results


def run_part_b(probe):
    """LOGICAL QUANTIFIER: fix everything, change only the logic."""
    print('\n' + '='*70)
    print('PART B: LOGICAL QUANTIFIER — pure reasoning variation')
    print('='*70)

    # Same entity, property, domain, language. Only quantifier changes.
    quantifiers = [
        ("all",   "all the roses in my garden are red"),
        ("none",  "none of the roses in my garden are red"),
        ("some",  "some of the roses in my garden are red"),
        ("most",  "most of the roses in my garden are red"),
        ("only",  "only the roses in my garden are red"),
        ("no_not","the roses in my garden are not red"),
    ]

    h_data = {}
    for name, text in quantifiers:
        h_all, tokens, ids = probe.encode(text)
        h_data[name] = {L: h_all[L][-1] for L in KEY_LAYERS}
        print(f'  {name:8s}: "{text}"', flush=True)

    print(f'\n  h_last cosines (raw):')
    names = [n for n, _ in quantifiers]
    print(f'  {"":16s}', end='')
    for L in KEY_LAYERS:
        print(f'  L{L:>2d}    ', end='')
    print()

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            label = f'{names[i]}-{names[j]}'
            print(f'  {label:16s}', end='')
            for L in KEY_LAYERS:
                c = cos(h_data[names[i]][L], h_data[names[j]][L])
                print(f'  {c:.4f}', end='')
            print()

    # Centered cosines
    print(f'\n  h_last cosines (centered):')
    for L in KEY_LAYERS:
        vecs = np.stack([h_data[n][L] for n in names])
        mean = vecs.mean(axis=0)
        centered = vecs - mean
        print(f'  L{L}:', end='')
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                c = cos(centered[i], centered[j])
                print(f'  {names[i][:3]}-{names[j][:3]}={c:.3f}', end='')
        print()


def run_part_c(probe):
    """NEGATION SYLLOGISM: same structure, entities; different logic & truth value."""
    print('\n' + '='*70)
    print('PART C: NEGATION SYLLOGISM — same entities, different logic')
    print('='*70)

    syllogisms = [
        ("all_false",
         "all the roses in my garden are red. i got a flower from my garden and it is red. must it be a rose?"),
        ("none_false",
         "none of the roses in my garden are red. i got a flower from my garden and it is red. must it be a rose?"),
        ("only_true",
         "only roses in my garden are red. i got a flower from my garden and it is red. must it be a rose?"),
        ("some_ambig",
         "some of the roses in my garden are red. i got a flower from my garden and it is red. must it be a rose?"),
    ]

    h_data = {}
    for name, text in syllogisms:
        h_all, tokens, ids = probe.encode(text)
        h_data[name] = {L: h_all[L][-1] for L in KEY_LAYERS}
        print(f'  {name:12s}: "{text[:70]}..."', flush=True)

    # Also add cross-language versions of the key ones
    zh_versions = [
        ("all_zh",
         "\u6211\u82b1\u56ed\u91cc\u6240\u6709\u7684\u73ab\u7470\u90fd\u662f\u7ea2\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u82b1\uff0c\u5b83\u662f\u7ea2\u8272\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u73ab\u7470\u5417\uff1f"),
        ("none_zh",
         "\u6211\u82b1\u56ed\u91cc\u6ca1\u6709\u73ab\u7470\u662f\u7ea2\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u82b1\uff0c\u5b83\u662f\u7ea2\u8272\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u73ab\u7470\u5417\uff1f"),
        ("only_zh",
         "\u6211\u82b1\u56ed\u91cc\u53ea\u6709\u73ab\u7470\u662f\u7ea2\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u82b1\uff0c\u5b83\u662f\u7ea2\u8272\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u73ab\u7470\u5417\uff1f"),
    ]

    for name, text in zh_versions:
        h_all, tokens, ids = probe.encode(text)
        h_data[name] = {L: h_all[L][-1] for L in KEY_LAYERS}
        print(f'  {name:12s}: "{text[:40]}..."', flush=True)

    print(f'\n  pairwise cosines (raw):')
    all_names = [n for n, _ in syllogisms] + [n for n, _ in zh_versions]
    print(f'  {"":20s}', end='')
    for L in KEY_LAYERS:
        print(f'  L{L:>2d}    ', end='')
    print()

    for i in range(len(all_names)):
        for j in range(i+1, len(all_names)):
            label = f'{all_names[i]}-{all_names[j]}'
            print(f'  {label:20s}', end='')
            for L in KEY_LAYERS:
                c = cos(h_data[all_names[i]][L], h_data[all_names[j]][L])
                print(f'  {c:.4f}', end='')
            print()

    # Key comparison: same quantifier cross-lingual vs different quantifier same language
    print(f'\n  KEY COMPARISONS:')
    print(f'  {"comparison":40s}', end='')
    for L in KEY_LAYERS:
        print(f'  L{L:>2d}    ', end='')
    print()

    comparisons = [
        ("all_en vs all_zh (same logic, diff lang)", "all_false", "all_zh"),
        ("all_en vs none_en (diff logic, same lang)", "all_false", "none_false"),
        ("all_en vs only_en (diff logic, same lang)", "all_false", "only_true"),
        ("none_en vs none_zh (same logic, diff lang)", "none_false", "none_zh"),
        ("only_en vs only_zh (same logic, diff lang)", "only_true", "only_zh"),
        ("all_en vs some_en (similar logic)", "all_false", "some_ambig"),
    ]

    for label, n1, n2 in comparisons:
        print(f'  {label:40s}', end='')
        for L in KEY_LAYERS:
            c = cos(h_data[n1][L], h_data[n2][L])
            print(f'  {c:.4f}', end='')
        print()


def run_part_d(probe):
    """GENERATION: repeat task. Compare gen-time h to encoding-time h."""
    print('\n' + '='*70)
    print('PART D: GENERATION — repeat task fixed point')
    print('='*70)

    content = "the quick brown fox jumps over the lazy dog"
    repeat_prompt = f"Repeat exactly: {content}\n"

    # First: encode the bare content to get reference h for each token
    h_enc, enc_tokens, enc_ids = probe.encode(content)
    print(f'  encoding: {len(enc_tokens)} tokens: {enc_tokens}', flush=True)

    # Generate from the repeat prompt
    print(f'  generating from repeat prompt...', flush=True)
    gen_tokens, gen_h, prompt_len = probe.generate_with_capture(repeat_prompt, max_new=30)
    print(f'  generated {len(gen_tokens)} tokens: {gen_tokens}', flush=True)

    # Try to match generated tokens to encoded tokens
    print(f'\n  gen-time vs encoding-time h for matched tokens:')
    print(f'  {"gen_tok":>15s} {"enc_tok":>15s} {"match":>6s}', end='')
    for L in KEY_LAYERS:
        print(f'  L{L:>2d}    ', end='')
    print()

    for gi, gt in enumerate(gen_tokens):
        gt_clean = gt.strip()
        # find this token in enc_tokens
        best_match = None
        for ei, et in enumerate(enc_tokens):
            if et.strip() == gt_clean:
                best_match = ei
                break

        match_str = f'e{best_match}' if best_match is not None else 'MISS'
        print(f'  {gt:>15s} {(enc_tokens[best_match] if best_match is not None else "---"):>15s} {match_str:>6s}', end='')

        if best_match is not None:
            for L in KEY_LAYERS:
                h_gen = gen_h[gi][L]
                h_ref = h_enc[L][best_match]
                c = cos(h_gen, h_ref)
                print(f'  {c:.4f}', end='')
        else:
            for L in KEY_LAYERS:
                print(f'  {"---":>6s}', end='')
        print()

    # Also: generate a reasoning response for comparison
    print(f'\n  --- reasoning generation for comparison ---')
    reason_prompt = f"How many words are in this sentence? {content}\nThe answer is"
    gen_tokens_r, gen_h_r, _ = probe.generate_with_capture(reason_prompt, max_new=20)
    print(f'  generated: {gen_tokens_r[:10]}', flush=True)

    # Compare: when generating the SAME token in repeat vs reasoning context,
    # how different is h?
    print(f'\n  same token, different task (repeat vs reasoning):')
    for gi, gt in enumerate(gen_tokens[:10]):
        gt_clean = gt.strip()
        # find same token in reasoning generation
        for ri, rt in enumerate(gen_tokens_r[:10]):
            if rt.strip() == gt_clean:
                print(f'  token="{gt_clean}" (repeat step {gi} vs reason step {ri}):', end='')
                for L in KEY_LAYERS:
                    c = cos(gen_h[gi][L], gen_h_r[ri][L])
                    print(f'  L{L}={c:.4f}', end='')
                print()
                break


def run_part_e(probe):
    """SYNTAX STRIPPING + CROSS-LINGUAL: does stripping help convergence?"""
    print('\n' + '='*70)
    print('PART E: SYNTAX STRIPPING + CROSS-LINGUAL')
    print('='*70)

    prompts = [
        # English variants
        ("en_full",  "all the roses in my garden are red. i got a flower from my garden and it is red. must it be a rose?"),
        ("en_strip", "roses garden red. flower garden red. rose?"),
        ("en_perm",  "red are roses the all. red is it flower a got i. rose be it must?"),

        # Chinese variants
        ("zh_full",  "\u6211\u82b1\u56ed\u91cc\u6240\u6709\u7684\u73ab\u7470\u90fd\u662f\u7ea2\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u82b1\uff0c\u5b83\u662f\u7ea2\u8272\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u73ab\u7470\u5417\uff1f"),
        ("zh_strip", "\u73ab\u7470 \u82b1\u56ed \u7ea2\u3002\u82b1 \u82b1\u56ed \u7ea2\u3002\u73ab\u7470\uff1f"),

        # Spanish variants
        ("es_full",  "todas las rosas de mi jard\u00edn son rojas. cog\u00ed una flor de mi jard\u00edn y es roja. \u00bftiene que ser una rosa?"),
        ("es_strip", "rosas jard\u00edn rojas. flor jard\u00edn roja. \u00bfrosa?"),

        # Arabic
        ("ar_full",  "\u0643\u0644 \u0627\u0644\u0648\u0631\u0648\u062f \u0641\u064a \u062d\u062f\u064a\u0642\u062a\u064a \u062d\u0645\u0631\u0627\u0621. \u062d\u0635\u0644\u062a \u0639\u0644\u0649 \u0632\u0647\u0631\u0629 \u0645\u0646 \u062d\u062f\u064a\u0642\u062a\u064a \u0648\u0647\u064a \u062d\u0645\u0631\u0627\u0621. \u0647\u0644 \u064a\u062c\u0628 \u0623\u0646 \u062a\u0643\u0648\u0646 \u0648\u0631\u062f\u0629\u061f"),
    ]

    h_data = {}
    for name, text in prompts:
        h_all, tokens, ids = probe.encode(text)
        h_data[name] = {L: h_all[L][-1] for L in KEY_LAYERS}
        print(f'  {name:10s}: {len(tokens):2d} tok — "{text[:50]}"', flush=True)

    # Cross-lingual: full vs full
    print(f'\n  CROSS-LINGUAL FULL:')
    print(f'  {"pair":20s}', end='')
    for L in KEY_LAYERS:
        print(f'  L{L:>2d}    ', end='')
    print()

    xl_pairs = [
        ("en-zh full", "en_full", "zh_full"),
        ("en-es full", "en_full", "es_full"),
        ("en-ar full", "en_full", "ar_full"),
        ("en-zh strip", "en_strip", "zh_strip"),
        ("en-es strip", "en_strip", "es_strip"),
    ]
    for label, n1, n2 in xl_pairs:
        print(f'  {label:20s}', end='')
        for L in KEY_LAYERS:
            c = cos(h_data[n1][L], h_data[n2][L])
            print(f'  {c:.4f}', end='')
        print()

    # Syntax: full vs stripped vs permuted
    print(f'\n  SYNTAX VARIATION:')
    syn_pairs = [
        ("en full-strip", "en_full", "en_strip"),
        ("en full-perm", "en_full", "en_perm"),
        ("en strip-perm", "en_strip", "en_perm"),
        ("zh full-strip", "zh_full", "zh_strip"),
        ("es full-strip", "es_full", "es_strip"),
    ]
    for label, n1, n2 in syn_pairs:
        print(f'  {label:20s}', end='')
        for L in KEY_LAYERS:
            c = cos(h_data[n1][L], h_data[n2][L])
            print(f'  {c:.4f}', end='')
        print()

    # KEY: does stripping INCREASE cross-lingual convergence?
    print(f'\n  STRIPPING EFFECT ON CROSS-LINGUAL CONVERGENCE:')
    for L in KEY_LAYERS:
        full_xl = cos(h_data['en_full'][L], h_data['zh_full'][L])
        strip_xl = cos(h_data['en_strip'][L], h_data['zh_strip'][L])
        delta = strip_xl - full_xl
        print(f'  L{L:>2d}: en-zh full={full_xl:.4f}  strip={strip_xl:.4f}  delta={delta:+.4f}')


def run_part_f(probe):
    """REPEAT vs REASON on the SAME content."""
    print('\n' + '='*70)
    print('PART F: REPEAT vs REASON — same content, different task')
    print('='*70)

    base = "all the roses in my garden are red. i got a flower from my garden and it is red."

    prompts = [
        ("bare",     base),
        ("repeat",   f"repeat: {base}"),
        ("reason",   f"{base} must it be a rose?"),
        ("first_w",  f"{base} what was the first word?"),
        ("count_w",  f"{base} how many words?"),
    ]

    h_data = {}
    for name, text in prompts:
        h_all, tokens, ids = probe.encode(text)
        h_data[name] = {L: h_all[L][-1] for L in KEY_LAYERS}
        print(f'  {name:10s}: {len(tokens)} tok', flush=True)

    print(f'\n  h_last cosines:')
    names = [n for n, _ in prompts]
    print(f'  {"":18s}', end='')
    for L in KEY_LAYERS:
        print(f'  L{L:>2d}    ', end='')
    print()

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            label = f'{names[i]}-{names[j]}'
            print(f'  {label:18s}', end='')
            for L in KEY_LAYERS:
                c = cos(h_data[names[i]][L], h_data[names[j]][L])
                print(f'  {c:.4f}', end='')
            print()

    # Centered: which task-type pairs cluster?
    print(f'\n  centered cosines at L26 and L33:')
    for L in [26, 33]:
        vecs = np.stack([h_data[n][L] for n in names])
        mean = vecs.mean(axis=0)
        centered = vecs - mean
        print(f'  L{L}:')
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                c = cos(centered[i], centered[j])
                print(f'    {names[i]:8s}-{names[j]:8s} = {c:+.4f}')


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('='*70)
    print('PROBE CHANNELS: verbatim vs reasoning')
    print('  no system prompt. no chat template. raw text.')
    print('='*70)

    probe = Probe()

    t0 = time.time()
    run_part_a(probe)
    run_part_b(probe)
    run_part_c(probe)
    run_part_e(probe)
    run_part_f(probe)
    run_part_d(probe)  # generation last (slower)
    dt = time.time() - t0

    probe.close()
    print(f'\ntotal time: {dt:.1f}s')


if __name__ == '__main__':
    main()
