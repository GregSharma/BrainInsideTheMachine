"""probe_workbench: Load model once, probe fast, iterate.

No system prompt. No generation. Just encoding-time hidden states.
Feed a prompt, get activations. Compare pairs. Find invariants.

Design: factorial variation along independent axes.
  - language (en/zh/es)
  - logical structure (syllogism/arithmetic/retrieval)
  - entities (roses/tulips/potatoes)
  - properties (red/blue/organic)
  - domain (garden/fridge)
  - paraphrase level (standard/rephrased/minimal)
  - truncation (full/stripped/skeleton)

What stays fixed when you change language? That's language-invariant.
What stays fixed when you change entities? That's structure.
What stays fixed when you change everything except logic? That's f*.
"""
import json, time, sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
DEVICE = 'cuda'
KEY_LAYERS = [5, 13, 18, 26, 30, 33]


def build_prompt_battery():
    """Systematic prompt variations with metadata tags.

    Each prompt has tags for every axis of variation.
    We compare prompts that share tags vs differ on them.
    """
    prompts = []

    def add(text, **tags):
        prompts.append({'text': text, **tags})

    # ================================================================
    # GROUP 1: SYLLOGISM — "All X in Y are Z. Got W from Y, it is Z. Must W be X?"
    # Correct answer: No (the converse is not implied)
    # ================================================================

    # --- flowers/garden/red, English ---
    add("All the roses in my garden are red. I got a flower from my garden and it is red. Must it be a rose?",
        lang='en', structure='syllogism', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='full')

    # paraphrase
    add("Every rose in my garden is red. I picked a red flower from my garden. Is it necessarily a rose?",
        lang='en', structure='syllogism', domain='garden', entity='roses',
        prop='red', paraphrase='rephrased', truncation='full')

    # truncated
    add("roses garden red. flower garden red. must rose?",
        lang='en', structure='syllogism', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='skeleton')

    # abstract
    add("All A in B are C. X is from B and X is C. Must X be A?",
        lang='en', structure='syllogism', domain='abstract', entity='abstract',
        prop='abstract', paraphrase='standard', truncation='abstract')

    # --- entity swap: tulips ---
    add("All the tulips in my garden are red. I got a flower from my garden and it is red. Must it be a tulip?",
        lang='en', structure='syllogism', domain='garden', entity='tulips',
        prop='red', paraphrase='standard', truncation='full')

    # --- entity swap: lilies ---
    add("All the lilies in my garden are red. I got a flower from my garden and it is red. Must it be a lily?",
        lang='en', structure='syllogism', domain='garden', entity='lilies',
        prop='red', paraphrase='standard', truncation='full')

    # --- property swap: blue ---
    add("All the roses in my garden are blue. I got a flower from my garden and it is blue. Must it be a rose?",
        lang='en', structure='syllogism', domain='garden', entity='roses',
        prop='blue', paraphrase='standard', truncation='full')

    # --- domain swap: fridge/potatoes/organic ---
    add("All the potatoes in my fridge are organic. I got a vegetable from my fridge and it is organic. Must it be a potato?",
        lang='en', structure='syllogism', domain='fridge', entity='potatoes',
        prop='organic', paraphrase='standard', truncation='full')

    # --- domain swap: fridge/apples/organic ---
    add("All the apples in my fridge are organic. I got a fruit from my fridge and it is organic. Must it be an apple?",
        lang='en', structure='syllogism', domain='fridge', entity='apples',
        prop='organic', paraphrase='standard', truncation='full')

    # --- Chinese translations ---
    add("\u6211\u82b1\u56ed\u91cc\u6240\u6709\u7684\u73ab\u7470\u90fd\u662f\u7ea2\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u82b1\uff0c\u5b83\u662f\u7ea2\u8272\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u73ab\u7470\u5417\uff1f",
        lang='zh', structure='syllogism', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='full')

    add("\u6211\u82b1\u56ed\u91cc\u6240\u6709\u7684\u90c1\u91d1\u9999\u90fd\u662f\u7ea2\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u82b1\uff0c\u5b83\u662f\u7ea2\u8272\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u90c1\u91d1\u9999\u5417\uff1f",
        lang='zh', structure='syllogism', domain='garden', entity='tulips',
        prop='red', paraphrase='standard', truncation='full')

    add("\u6211\u82b1\u56ed\u91cc\u6240\u6709\u7684\u73ab\u7470\u90fd\u662f\u84dd\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u82b1\uff0c\u5b83\u662f\u84dd\u8272\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u73ab\u7470\u5417\uff1f",
        lang='zh', structure='syllogism', domain='garden', entity='roses',
        prop='blue', paraphrase='standard', truncation='full')

    add("\u6211\u51b0\u7bb1\u91cc\u6240\u6709\u7684\u571f\u8c46\u90fd\u662f\u6709\u673a\u7684\u3002\u6211\u4ece\u51b0\u7bb1\u91cc\u62ff\u4e86\u4e00\u4e2a\u84d4\u83dc\uff0c\u5b83\u662f\u6709\u673a\u7684\u3002\u5b83\u4e00\u5b9a\u662f\u571f\u8c46\u5417\uff1f",
        lang='zh', structure='syllogism', domain='fridge', entity='potatoes',
        prop='organic', paraphrase='standard', truncation='full')

    # --- Spanish translations ---
    add("Todas las rosas de mi jard\u00edn son rojas. Cog\u00ed una flor de mi jard\u00edn y es roja. \u00bfTiene que ser una rosa?",
        lang='es', structure='syllogism', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='full')

    add("Todas las rosas de mi jard\u00edn son azules. Cog\u00ed una flor de mi jard\u00edn y es azul. \u00bfTiene que ser una rosa?",
        lang='es', structure='syllogism', domain='garden', entity='roses',
        prop='blue', paraphrase='standard', truncation='full')

    add("Todas las patatas de mi nevera son org\u00e1nicas. Cog\u00ed una verdura de mi nevera y es org\u00e1nica. \u00bfTiene que ser una patata?",
        lang='es', structure='syllogism', domain='fridge', entity='potatoes',
        prop='organic', paraphrase='standard', truncation='full')

    # ================================================================
    # GROUP 2: CONVERSE SYLLOGISM — same structure but TRUE
    # "All X in Y are Z. I got an X from Y. Must it be Z?" → Yes
    # ================================================================
    add("All the roses in my garden are red. I got a rose from my garden. Must it be red?",
        lang='en', structure='converse', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='full')

    add("All the potatoes in my fridge are organic. I got a potato from my fridge. Must it be organic?",
        lang='en', structure='converse', domain='fridge', entity='potatoes',
        prop='organic', paraphrase='standard', truncation='full')

    add("\u6211\u82b1\u56ed\u91cc\u6240\u6709\u7684\u73ab\u7470\u90fd\u662f\u7ea2\u8272\u7684\u3002\u6211\u4ece\u82b1\u56ed\u91cc\u62ff\u4e86\u4e00\u6735\u73ab\u7470\u3002\u5b83\u4e00\u5b9a\u662f\u7ea2\u8272\u7684\u5417\uff1f",
        lang='zh', structure='converse', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='full')

    # ================================================================
    # GROUP 3: ARITHMETIC — simple, vary numbers and language
    # ================================================================
    add("What is 2 + 2?",
        lang='en', structure='arithmetic', domain='math', entity='2+2',
        prop='addition', paraphrase='standard', truncation='full')

    add("What is 9 + 9?",
        lang='en', structure='arithmetic', domain='math', entity='9+9',
        prop='addition', paraphrase='standard', truncation='full')

    add("What is 2 + 3?",
        lang='en', structure='arithmetic', domain='math', entity='2+3',
        prop='addition', paraphrase='standard', truncation='full')

    add("What is 7 \u00d7 8?",
        lang='en', structure='arithmetic', domain='math', entity='7x8',
        prop='multiplication', paraphrase='standard', truncation='full')

    add("What is 15 - 7?",
        lang='en', structure='arithmetic', domain='math', entity='15-7',
        prop='subtraction', paraphrase='standard', truncation='full')

    add("2 + 2 = ?",
        lang='en', structure='arithmetic', domain='math', entity='2+2',
        prop='addition', paraphrase='rephrased', truncation='full')

    add("\u4e8c\u52a0\u4e8c\u7b49\u4e8e\u51e0\uff1f",
        lang='zh', structure='arithmetic', domain='math', entity='2+2',
        prop='addition', paraphrase='standard', truncation='full')

    add("\u4e5d\u52a0\u4e5d\u7b49\u4e8e\u51e0\uff1f",
        lang='zh', structure='arithmetic', domain='math', entity='9+9',
        prop='addition', paraphrase='standard', truncation='full')

    add("\u00bfCu\u00e1nto es 2 + 2?",
        lang='es', structure='arithmetic', domain='math', entity='2+2',
        prop='addition', paraphrase='standard', truncation='full')

    # ================================================================
    # GROUP 4: RETRIEVAL — repeat/echo, vary content
    # ================================================================
    add("Repeat after me: the cat sat on the mat",
        lang='en', structure='retrieval', domain='animals', entity='cat',
        prop='sitting', paraphrase='standard', truncation='full')

    add("Repeat after me: the dog sat on the log",
        lang='en', structure='retrieval', domain='animals', entity='dog',
        prop='sitting', paraphrase='standard', truncation='full')

    add("Repeat after me: all the roses in my garden are red",
        lang='en', structure='retrieval', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='full')

    add("\u8bf7\u91cd\u590d\uff1a\u732b\u5750\u5728\u5e2d\u5b50\u4e0a",
        lang='zh', structure='retrieval', domain='animals', entity='cat',
        prop='sitting', paraphrase='standard', truncation='full')

    # ================================================================
    # GROUP 5: PROGRESSIVE TRUNCATION of one syllogism
    # ================================================================
    add("All the roses in my garden are red. I got a flower from my garden and it is red. Must it be a rose?",
        lang='en', structure='syllogism', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='full')  # duplicate for grouping

    add("All roses in garden are red. Got flower from garden, it is red. Must it be rose?",
        lang='en', structure='syllogism', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='light')

    add("roses garden red. flower garden red. rose?",
        lang='en', structure='syllogism', domain='garden', entity='roses',
        prop='red', paraphrase='standard', truncation='skeleton')

    add("All A in B are C. X from B is C. Must X be A?",
        lang='en', structure='syllogism', domain='abstract', entity='abstract',
        prop='abstract', paraphrase='standard', truncation='abstract')

    return prompts


class ActivationProbe:
    """Load model once, probe many times."""

    def __init__(self, model_name=MODEL_NAME, device=DEVICE, layers=KEY_LAYERS):
        print(f'loading {model_name}...', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device, trust_remote_code=True)
        self.model.eval()
        self.layers = layers
        self.device = device
        print(f'  ready. layers={layers}', flush=True)

        # Set up capture hooks
        self._caps = {}
        self._hooks = []
        for L in layers:
            cap = _Cap()
            self._caps[L] = cap
            self._hooks.append(self.model.model.layers[L].register_forward_hook(cap))

    def encode(self, text, use_chat=False):
        """Get hidden states at all tracked layers.

        No system prompt by default. Raw text → tokenize → forward.
        If use_chat=True, wraps in minimal chat template (empty system).

        Returns dict: {layer: array(seq_len, d_model)}
        """
        if use_chat:
            messages = [{"role": "system", "content": "."}, {"role": "user", "content": text}]
            text_in = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text_in = text

        ids = self.tokenizer(text_in, return_tensors='pt').input_ids.to(self.device)
        with torch.inference_mode():
            self.model(ids)

        result = {}
        for L in self.layers:
            result[L] = self._caps[L].out.copy()
        return result, ids.shape[1]

    def encode_last(self, text, use_chat=False):
        """Get last-token hidden state only. Returns {layer: array(d,)}."""
        all_h, n_tok = self.encode(text, use_chat=use_chat)
        return {L: h[-1] for L, h in all_h.items()}, n_tok

    def close(self):
        for h in self._hooks:
            h.remove()


class _Cap:
    def __init__(self):
        self.out = None
    def __call__(self, module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        self.out = h[0].detach().float().cpu().numpy()


def cosine(a, b):
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def centered_cosine(vecs):
    """Cosine similarity after removing the global mean."""
    mean = np.mean(vecs, axis=0)
    centered = vecs - mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = centered / norms
    return normed @ normed.T


def factor_analysis(prompts, activations, layer):
    """Decompose hidden state variance by factor.

    For each factor (lang, structure, domain, entity, prop, paraphrase, truncation):
      - compute mean cosine between pairs that SHARE this factor value
      - compute mean cosine between pairs that DIFFER on this factor value
      - delta = share - differ = how much this factor contributes to similarity

    Higher delta = this factor explains more of the hidden state structure.
    """
    factors = ['lang', 'structure', 'domain', 'entity', 'prop', 'paraphrase', 'truncation']
    n = len(prompts)
    vecs = np.stack([activations[i][layer] for i in range(n)])

    # Raw cosines
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = vecs / norms
    cos_mat = normed @ normed.T

    # Centered cosines (remove mean direction — this is where the real geometry lives)
    cent_cos_mat = centered_cosine(vecs)

    results = {}
    for factor in factors:
        vals = [p[factor] for p in prompts]
        share_raw, differ_raw = [], []
        share_cent, differ_cent = [], []

        for i in range(n):
            for j in range(i+1, n):
                if vals[i] == vals[j]:
                    share_raw.append(cos_mat[i, j])
                    share_cent.append(cent_cos_mat[i, j])
                else:
                    differ_raw.append(cos_mat[i, j])
                    differ_cent.append(cent_cos_mat[i, j])

        results[factor] = {
            'share_raw': float(np.mean(share_raw)) if share_raw else None,
            'differ_raw': float(np.mean(differ_raw)) if differ_raw else None,
            'delta_raw': float(np.mean(share_raw) - np.mean(differ_raw)) if share_raw and differ_raw else None,
            'share_centered': float(np.mean(share_cent)) if share_cent else None,
            'differ_centered': float(np.mean(differ_cent)) if differ_cent else None,
            'delta_centered': float(np.mean(share_cent) - np.mean(differ_cent)) if share_cent and differ_cent else None,
        }

    return results


def pairwise_deep_dive(prompts, activations, layer, focus_pairs=None):
    """Print specific pairwise comparisons for selected pairs.

    If focus_pairs is None, auto-selects interesting pairs:
    - same structure, different language
    - same everything, different entity
    - same everything, different property
    - same logic, different domain
    """
    if focus_pairs is None:
        focus_pairs = []
        n = len(prompts)
        for i in range(n):
            for j in range(i+1, n):
                pi, pj = prompts[i], prompts[j]
                # Same structure + domain, different language
                if (pi['structure'] == pj['structure'] and pi['domain'] == pj['domain'] and
                    pi['entity'] == pj['entity'] and pi['prop'] == pj['prop'] and
                    pi['lang'] != pj['lang'] and
                    pi['paraphrase'] == pj['paraphrase'] == 'standard' and
                    pi['truncation'] == pj['truncation'] == 'full'):
                    focus_pairs.append((i, j, f"LANG: {pi['lang']}->{pj['lang']}"))

                # Same everything, different entity (same lang, same domain)
                if (pi['structure'] == pj['structure'] and pi['lang'] == pj['lang'] and
                    pi['domain'] == pj['domain'] and pi['prop'] == pj['prop'] and
                    pi['entity'] != pj['entity'] and
                    pi['paraphrase'] == pj['paraphrase'] == 'standard' and
                    pi['truncation'] == pj['truncation'] == 'full' and
                    pi['entity'] not in ('abstract',) and pj['entity'] not in ('abstract',)):
                    focus_pairs.append((i, j, f"ENTITY: {pi['entity']}->{pj['entity']}"))

                # Same everything, different property
                if (pi['structure'] == pj['structure'] and pi['lang'] == pj['lang'] and
                    pi['domain'] == pj['domain'] and pi['entity'] == pj['entity'] and
                    pi['prop'] != pj['prop'] and
                    pi['paraphrase'] == pj['paraphrase'] == 'standard' and
                    pi['truncation'] == pj['truncation'] == 'full' and
                    pi['prop'] not in ('abstract',) and pj['prop'] not in ('abstract',)):
                    focus_pairs.append((i, j, f"PROP: {pi['prop']}->{pj['prop']}"))

                # Same structure, different domain
                if (pi['structure'] == pj['structure'] and pi['lang'] == pj['lang'] and
                    pi['domain'] != pj['domain'] and
                    pi['paraphrase'] == pj['paraphrase'] == 'standard' and
                    pi['truncation'] == pj['truncation'] == 'full' and
                    pi['domain'] not in ('abstract',) and pj['domain'] not in ('abstract',)):
                    focus_pairs.append((i, j, f"DOMAIN: {pi['domain']}->{pj['domain']}"))

                # Different structure entirely
                if (pi['structure'] != pj['structure'] and pi['lang'] == pj['lang'] == 'en' and
                    pi['paraphrase'] == pj['paraphrase'] == 'standard' and
                    pi['truncation'] == pj['truncation'] == 'full'):
                    focus_pairs.append((i, j, f"STRUCT: {pi['structure']}->{pj['structure']}"))

    # Deduplicate
    seen = set()
    unique_pairs = []
    for i, j, label in focus_pairs:
        key = (min(i,j), max(i,j))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((i, j, label))

    return unique_pairs


def compute_invariant_subspace(prompts, activations, layer, group_by='structure'):
    """Find the subspace that's invariant within groups.

    Groups prompts by `group_by` factor. Within each group,
    computes the mean activation (the group centroid).
    SVD of centroids reveals the between-group structure.
    SVD of residuals (activation - group_mean) reveals within-group variation.
    """
    groups = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[p[group_by]].append(i)

    centroids = []
    centroid_labels = []
    residuals = []

    for gval, indices in sorted(groups.items()):
        vecs = np.stack([activations[i][layer] for i in indices])
        centroid = vecs.mean(axis=0)
        centroids.append(centroid)
        centroid_labels.append(gval)
        for v in vecs:
            residuals.append(v - centroid)

    centroids = np.stack(centroids)
    residuals = np.stack(residuals)

    # Center centroids
    grand_mean = centroids.mean(axis=0)
    centroids_c = centroids - grand_mean

    _, S_between, Vt_between = np.linalg.svd(centroids_c, full_matrices=False)
    _, S_within, Vt_within = np.linalg.svd(residuals, full_matrices=False)

    cumvar_b = np.cumsum(S_between**2) / (np.sum(S_between**2) + 1e-10)
    cumvar_w = np.cumsum(S_within**2) / (np.sum(S_within**2) + 1e-10)

    r90_b = int(np.searchsorted(cumvar_b, 0.9) + 1)
    r90_w = int(np.searchsorted(cumvar_w, 0.9) + 1)

    return {
        'group_by': group_by,
        'n_groups': len(centroids),
        'groups': centroid_labels,
        'between_sv': S_between[:10].tolist(),
        'between_r90': r90_b,
        'within_sv': S_within[:10].tolist(),
        'within_r90': r90_w,
        'between_top_dirs': Vt_between[:3],  # top 3 between-group directions
        'within_top_dirs': Vt_within[:3],     # top 3 within-group directions
    }


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('=' * 70)
    print('PROBE WORKBENCH: load once, iterate fast')
    print('  no system prompt. no generation. just hidden states.')
    print('  the model talks. we listen.')
    print('=' * 70)

    probe = ActivationProbe()
    prompts = build_prompt_battery()
    print(f'\n{len(prompts)} prompts loaded.', flush=True)

    # === ENCODE ALL PROMPTS ===
    print('\nencoding all prompts (no chat template, raw text)...', flush=True)
    activations = []  # list of {layer: array(d,)}
    token_counts = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        h, n_tok = probe.encode_last(p['text'], use_chat=False)
        activations.append(h)
        token_counts.append(n_tok)
    dt = time.time() - t0
    print(f'  done in {dt:.2f}s ({dt/len(prompts)*1000:.1f}ms per prompt)', flush=True)

    # Also encode with chat template for comparison
    print('\nencoding with minimal chat template (sys=".")...', flush=True)
    activations_chat = []
    for i, p in enumerate(prompts):
        h, _ = probe.encode_last(p['text'], use_chat=True)
        activations_chat.append(h)
    print('  done.', flush=True)

    probe.close()

    results_all = {}

    for layer in KEY_LAYERS:
        print(f'\n{"=" * 70}')
        print(f'LAYER {layer}')
        print(f'{"=" * 70}')

        # === FACTOR ANALYSIS ===
        print(f'\n  --- factor analysis (what explains activation similarity?) ---')
        fa = factor_analysis(prompts, activations, layer)

        print(f'  {"factor":<15s} {"share_raw":>10s} {"diff_raw":>10s} {"delta_raw":>10s} | '
              f'{"share_cent":>10s} {"diff_cent":>10s} {"delta_cent":>10s}')
        print(f'  {"-"*15} {"-"*10} {"-"*10} {"-"*10} | {"-"*10} {"-"*10} {"-"*10}')
        for factor in ['structure', 'lang', 'domain', 'entity', 'prop', 'paraphrase', 'truncation']:
            f = fa[factor]
            sr = f'{f["share_raw"]:.4f}' if f['share_raw'] is not None else 'N/A'
            dr = f'{f["differ_raw"]:.4f}' if f['differ_raw'] is not None else 'N/A'
            delr = f'{f["delta_raw"]:+.4f}' if f['delta_raw'] is not None else 'N/A'
            sc = f'{f["share_centered"]:.4f}' if f['share_centered'] is not None else 'N/A'
            dc = f'{f["differ_centered"]:.4f}' if f['differ_centered'] is not None else 'N/A'
            delc = f'{f["delta_centered"]:+.4f}' if f['delta_centered'] is not None else 'N/A'
            print(f'  {factor:<15s} {sr:>10s} {dr:>10s} {delr:>10s} | {sc:>10s} {dc:>10s} {delc:>10s}')

        # === PAIRWISE DEEP DIVE ===
        print(f'\n  --- pairwise deep dive (controlled comparisons) ---')
        pairs = pairwise_deep_dive(prompts, activations, layer)

        # Group pairs by type
        pair_types = defaultdict(list)
        for i, j, label in pairs:
            ptype = label.split(':')[0].strip()
            hi, hj = activations[i][layer], activations[j][layer]
            c_raw = cosine(hi, hj)
            # centered
            mean_vec = np.mean([activations[k][layer] for k in range(len(prompts))], axis=0)
            c_cent = cosine(hi - mean_vec, hj - mean_vec)
            pair_types[ptype].append((c_raw, c_cent, label, i, j))

        for ptype in ['LANG', 'ENTITY', 'PROP', 'DOMAIN', 'STRUCT']:
            if ptype not in pair_types:
                continue
            items = pair_types[ptype]
            mean_raw = np.mean([x[0] for x in items])
            mean_cent = np.mean([x[1] for x in items])
            print(f'\n  {ptype} variations ({len(items)} pairs): mean_raw={mean_raw:.4f}  mean_cent={mean_cent:.4f}')
            # Show top 5
            for c_raw, c_cent, label, i, j in sorted(items, key=lambda x: -abs(x[1]))[:5]:
                p_i = f"{prompts[i]['lang']}/{prompts[i]['entity']}/{prompts[i]['prop']}"
                p_j = f"{prompts[j]['lang']}/{prompts[j]['entity']}/{prompts[j]['prop']}"
                print(f'    {label:<35s} raw={c_raw:.4f}  cent={c_cent:.4f}  [{p_i} vs {p_j}]')

        # === INVARIANT SUBSPACE ===
        print(f'\n  --- invariant subspace decomposition ---')
        for group_by in ['structure', 'lang', 'entity']:
            inv = compute_invariant_subspace(prompts, activations, layer, group_by)
            sv_str = ', '.join(f'{s:.2f}' for s in inv['between_sv'][:5])
            print(f'  group_by={group_by:<12s}: {inv["n_groups"]} groups, '
                  f'between_r90={inv["between_r90"]}, '
                  f'within_r90={inv["within_r90"]}, '
                  f'top_sv=[{sv_str}]')

        # === CHAT VS NO-CHAT COMPARISON ===
        chat_diffs = []
        for i in range(len(prompts)):
            c = cosine(activations[i][layer], activations_chat[i][layer])
            chat_diffs.append(c)
        print(f'\n  chat_template effect: mean cos(raw, chat)={np.mean(chat_diffs):.4f} '
              f'(min={np.min(chat_diffs):.4f}, max={np.max(chat_diffs):.4f})')

        # === STORE RESULTS ===
        results_all[str(layer)] = {
            'factor_analysis': fa,
            'chat_template_cos': {'mean': float(np.mean(chat_diffs)),
                                   'min': float(np.min(chat_diffs)),
                                   'max': float(np.max(chat_diffs))},
        }

    # === CROSS-LAYER TRAJECTORY ===
    print(f'\n{"=" * 70}')
    print('CROSS-LAYER: how factors change importance through depth')
    print(f'{"=" * 70}')
    print(f'  {"factor":<15s}', end='')
    for L in KEY_LAYERS:
        print(f'  L{L:>2d}_cent', end='')
    print()

    for factor in ['structure', 'lang', 'domain', 'entity', 'prop']:
        print(f'  {factor:<15s}', end='')
        for L in KEY_LAYERS:
            fa = results_all[str(L)]['factor_analysis'][factor]
            d = fa['delta_centered']
            print(f'  {d:>+8.4f}' if d is not None else f'  {"N/A":>8s}', end='')
        print()

    # Save
    outpath = OUTPUT_DIR / 'probe_workbench.json'
    with open(outpath, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f'\nsaved to {outpath}')


if __name__ == '__main__':
    main()
