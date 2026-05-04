"""
SPIKE TOKEN ANNOTATION — What tokens are being generated at cosine spikes?

For each problem, find the top 3 cosine spikes (zh-en pair).
Map those spikes back to token positions.
Extract the actual token text at those positions in both languages.

If spikes = numbers/operators/math symbols → reasoning flashes
If spikes = narration words → coincidence
"""

import numpy as np
import json
from scipy.ndimage import uniform_filter1d

# ---------- LOAD ----------
print("Loading data...")
traj = np.load('output/gen_trajectories.npz')
with open('output/gen_trajectories_meta.json') as f:
    gen_meta = json.load(f)

GEN_LANGS = ['zh', 'en', 'es', 'ja']

gen_complete = []
for p in range(20):
    if all(f"prob{p}_{l}" in traj for l in GEN_LANGS):
        min_steps = min(traj[f"prob{p}_{l}"].shape[0] for l in GEN_LANGS)
        if min_steps >= 10:
            gen_complete.append(p)

print(f"Problems: {len(gen_complete)}")


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


# ---------- ANALYZE EACH PROBLEM ----------
print("\n" + "=" * 90)
print("SPIKE TOKEN ANNOTATION — Top 3 cosine spikes per problem (zh-en)")
print("=" * 90)

all_spike_tokens = {'zh': [], 'en': []}
all_spike_contexts = []

for p in gen_complete:
    h_zh = traj[f"prob{p}_zh"]
    h_en = traj[f"prob{p}_en"]
    n_zh = h_zh.shape[0]
    n_en = h_en.shape[0]

    # Mean-center each trajectory
    h_zh_c = h_zh - h_zh.mean(axis=0)
    h_en_c = h_en - h_en.mean(axis=0)

    # Align to the longer trajectory's τ grid
    if n_zh >= n_en:
        n_ref = n_zh
        tau_grid = np.arange(n_zh) / max(n_zh - 1, 1)
        ref_zh = h_zh_c
        ref_en = np.array([h_en_c[min(int(t * (n_en - 1) + 0.5), n_en - 1)] for t in tau_grid])
        # Token index mapping
        zh_tok_idx = np.arange(n_zh)
        en_tok_idx = np.array([min(int(t * (n_en - 1) + 0.5), n_en - 1) for t in tau_grid])
    else:
        n_ref = n_en
        tau_grid = np.arange(n_en) / max(n_en - 1, 1)
        ref_en = h_en_c
        ref_zh = np.array([h_zh_c[min(int(t * (n_zh - 1) + 0.5), n_zh - 1)] for t in tau_grid])
        zh_tok_idx = np.array([min(int(t * (n_zh - 1) + 0.5), n_zh - 1) for t in tau_grid])
        en_tok_idx = np.arange(n_en)

    # Compute cosine at each point
    cosines = np.array([cosine_sim(ref_zh[i], ref_en[i]) for i in range(n_ref)])

    # Find top 3 peaks (local maxima with highest values)
    # Smooth slightly to avoid noise peaks
    smooth = uniform_filter1d(cosines, size=max(3, n_ref // 30))

    # Find all local maxima
    peak_indices = []
    for j in range(1, len(smooth) - 1):
        if smooth[j] > smooth[j-1] and smooth[j] > smooth[j+1]:
            peak_indices.append((j, cosines[j]))  # use raw cosine for value

    # Sort by value, take top 3
    peak_indices.sort(key=lambda x: -x[1])
    top_peaks = peak_indices[:3]

    # Get token text at those positions
    text_zh = gen_meta[f"prob{p}_zh"].get('text_preview', '')
    text_en = gen_meta[f"prob{p}_en"].get('text_preview', '')
    cat = gen_meta[f"prob{p}_zh"]["category"]
    prompt_zh = gen_meta[f"prob{p}_zh"]["prompt"]

    print(f"\n{'─'*80}")
    print(f"Problem {p} | {cat} | \"{prompt_zh[:50]}\"")
    print(f"  zh: {n_zh} steps, en: {n_en} steps")
    print(f"  Mean cosine: {np.mean(cosines):.4f}, Max: {np.max(cosines):.4f}")

    for rank, (peak_j, peak_val) in enumerate(top_peaks):
        tau_at_peak = tau_grid[peak_j]
        zh_idx = int(zh_tok_idx[peak_j])
        en_idx = int(en_tok_idx[peak_j])

        # Extract token context from text preview
        # Approximate: map token index to character position
        zh_char_pos = int(len(text_zh) * zh_idx / max(n_zh, 1))
        en_char_pos = int(len(text_en) * en_idx / max(n_en, 1))

        # Window of ~20 chars around the position
        zh_window = text_zh[max(0, zh_char_pos-15):zh_char_pos+15]
        en_window = text_en[max(0, en_char_pos-15):en_char_pos+15]

        print(f"\n  Spike #{rank+1}: cosine={peak_val:.4f} at τ={tau_at_peak:.3f} "
              f"(zh tok {zh_idx}/{n_zh}, en tok {en_idx}/{n_en})")
        print(f"    zh: «{zh_window}»")
        print(f"    en: «{en_window}»")

        all_spike_contexts.append({
            'problem': p,
            'category': cat,
            'rank': rank + 1,
            'cosine': float(peak_val),
            'tau': float(tau_at_peak),
            'zh_token_idx': zh_idx,
            'en_token_idx': en_idx,
            'zh_context': zh_window,
            'en_context': en_window,
        })


# ---------- CLASSIFY SPIKE CONTENT ----------
print("\n\n" + "=" * 90)
print("SPIKE CONTENT CLASSIFICATION")
print("=" * 90)

# Simple heuristic: does the context contain math symbols/numbers?
import re
math_pattern = re.compile(r'[\d+\-×÷=!^()/\\{}\[\]CnkPπ∑∏]')

math_spikes = 0
narration_spikes = 0
ambiguous_spikes = 0

for spike in all_spike_contexts:
    zh_math = bool(math_pattern.search(spike['zh_context']))
    en_math = bool(math_pattern.search(spike['en_context']))

    if zh_math and en_math:
        math_spikes += 1
        spike['type'] = 'MATH'
    elif zh_math or en_math:
        ambiguous_spikes += 1
        spike['type'] = 'MIXED'
    else:
        narration_spikes += 1
        spike['type'] = 'NARRATION'

total = len(all_spike_contexts)
print(f"\n  Total spikes analyzed: {total}")
print(f"  MATH (both langs have math content):     {math_spikes}/{total} ({100*math_spikes/total:.0f}%)")
print(f"  MIXED (one lang has math):               {ambiguous_spikes}/{total} ({100*ambiguous_spikes/total:.0f}%)")
print(f"  NARRATION (neither has math):            {narration_spikes}/{total} ({100*narration_spikes/total:.0f}%)")

if math_spikes > total * 0.5:
    print("\n  → MAJORITY OF SPIKES ARE MATHEMATICAL. Reasoning flashes confirmed.")
elif math_spikes > total * 0.3:
    print("\n  → Significant fraction of spikes are mathematical. Partial confirmation.")
else:
    print("\n  → Spikes are NOT predominantly mathematical. Reasoning flash hypothesis weakened.")


# ---------- SPIKE τ DISTRIBUTION BY TYPE ----------
print("\n" + "=" * 90)
print("SPIKE τ DISTRIBUTION BY TYPE")
print("=" * 90)

for stype in ['MATH', 'MIXED', 'NARRATION']:
    typed = [s for s in all_spike_contexts if s['type'] == stype]
    if typed:
        taus = [s['tau'] for s in typed]
        print(f"\n  {stype} spikes (n={len(typed)}):")
        print(f"    Mean τ: {np.mean(taus):.3f}, Std: {np.std(taus):.3f}")
        print(f"    Range: [{min(taus):.3f}, {max(taus):.3f}]")


# ---------- COSINE MAGNITUDE BY TYPE ----------
print("\n" + "=" * 90)
print("SPIKE MAGNITUDE BY TYPE")
print("=" * 90)

for stype in ['MATH', 'MIXED', 'NARRATION']:
    typed = [s for s in all_spike_contexts if s['type'] == stype]
    if typed:
        vals = [s['cosine'] for s in typed]
        print(f"  {stype}: mean cosine = {np.mean(vals):.4f} (n={len(typed)})")


# ---------- SAVE ----------
output = {
    'spikes': all_spike_contexts,
    'summary': {
        'total_spikes': total,
        'math_spikes': math_spikes,
        'mixed_spikes': ambiguous_spikes,
        'narration_spikes': narration_spikes,
        'n_problems': len(gen_complete),
    }
}

with open('output/spike_token_annotation.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved: output/spike_token_annotation.json")
