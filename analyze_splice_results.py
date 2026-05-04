"""Deep analysis of splice/steer results — one level lower."""
import json
import numpy as np

with open('output/intervention_splice_steer.json') as f:
    data = json.load(f)

problems = data['per_problem']
N = len(problems)

print('='*80)
print('PER-PROBLEM DEEP ANALYSIS: PC0 SWAP vs EN BASELINE')
print('='*80)

# 1. Text identity
print('\n1. TEXT SIMILARITY: PC0 swap output vs en baseline output')
exact_64 = 0
first_20_match = 0
diverge_points = []

for i, p in enumerate(problems):
    en_text = p['baseline_en']['text']
    pc0_text = p['splice_pc0_swap']['text']

    match_len = 0
    for c1, c2 in zip(en_text, pc0_text):
        if c1 == c2:
            match_len += 1
        else:
            break

    if en_text[:64] == pc0_text[:64]:
        exact_64 += 1
        status = 'EXACT-64'
    else:
        status = f'diverge@char{match_len}'

    if en_text[:20] == pc0_text[:20]:
        first_20_match += 1

    diverge_points.append(match_len)
    print(f'  P{i:2d} (cat={p["category"]}): {status:>15} | en: {en_text[:70]}')
    if en_text[:64] != pc0_text[:64]:
        print(f'  {"":>24}               | pc: {pc0_text[:70]}')

print(f'\n  EXACT match (first 64 chars): {exact_64}/{N} ({exact_64/N:.0%})')
print(f'  First 20 chars match: {first_20_match}/{N} ({first_20_match/N:.0%})')
print(f'  Mean divergence point: {np.mean(diverge_points):.1f} chars')
print(f'  Min divergence point: {min(diverge_points)} chars')

# 2. First token identity
print(f'\n{"="*80}')
print('2. FIRST TOKEN IDENTITY (token IDs)')
print(f'{"="*80}')

pc0_matches_en_first = sum(1 for p in problems if p['splice_pc0_swap']['first_token'] == p['baseline_en']['first_token'])
pc0_matches_zh_first = sum(1 for p in problems if p['splice_pc0_swap']['first_token'] == p['baseline_zh']['first_token'])
raw_matches_en_first = sum(1 for p in problems if p['splice_raw']['first_token'] == p['baseline_en']['first_token'])
raw_matches_zh_first = sum(1 for p in problems if p['splice_raw']['first_token'] == p['baseline_zh']['first_token'])

print(f'  PC0 swap first token == en baseline: {pc0_matches_en_first}/{N} ({pc0_matches_en_first/N:.0%})')
print(f'  PC0 swap first token == zh baseline: {pc0_matches_zh_first}/{N} ({pc0_matches_zh_first/N:.0%})')
print(f'  Raw splice first token == en baseline: {raw_matches_en_first}/{N} ({raw_matches_en_first/N:.0%})')
print(f'  Raw splice first token == zh baseline: {raw_matches_zh_first}/{N} ({raw_matches_zh_first/N:.0%})')

# 3. Where divergences happen
print(f'\n{"="*80}')
print('3. PC0 SWAP DIVERGENCE DETAILS')
print(f'{"="*80}')
for i, p in enumerate(problems):
    en_text = p['baseline_en']['text']
    pc0_text = p['splice_pc0_swap']['text']
    if en_text[:64] != pc0_text[:64]:
        print(f'\n  Problem {i} (cat={p["category"]}):')
        print(f'    en:  ...{en_text[max(0,diverge_points[i]-10):diverge_points[i]+30]}...')
        print(f'    pc0: ...{pc0_text[max(0,diverge_points[i]-10):diverge_points[i]+30]}...')
        print(f'    Diverges at char {diverge_points[i]}')

# 4. Raw splice language breakdown
print(f'\n{"="*80}')
print('4. RAW SPLICE: WHAT GOES WRONG')
print(f'{"="*80}')
for i, p in enumerate(problems):
    raw = p['splice_raw']
    print(f'  P{i:2d} [{raw["lang"]:>7}] ft_match_zh={raw["first_token_matches_zh"]} ft_match_en={raw["first_token_matches_en"]}: {raw["text"][:70]}')

# 5. Reverse steer text match
print(f'\n{"="*80}')
print('5. REVERSE STEER (en→zh) vs ZH BASELINE')
print(f'{"="*80}')
rev_exact = 0
rev_diverge = []
for i, p in enumerate(problems):
    zh_text = p['baseline_zh']['text']
    rev_text = p['splice_reverse_steer']['text']
    match_len = 0
    for c1, c2 in zip(zh_text, rev_text):
        if c1 == c2:
            match_len += 1
        else:
            break
    if zh_text[:64] == rev_text[:64]:
        rev_exact += 1
        status = 'EXACT-64'
    else:
        status = f'diverge@{match_len}'
    rev_diverge.append(match_len)
    print(f'  P{i:2d} [{p["splice_reverse_steer"]["lang"]:>7}] {status:>15}: {rev_text[:70]}')
    if zh_text[:64] != rev_text[:64]:
        print(f'  {"":>30}  zh_base: {zh_text[:70]}')

print(f'\n  Reverse exact 64-char: {rev_exact}/{N} ({rev_exact/N:.0%})')
print(f'  Mean divergence: {np.mean(rev_diverge):.1f} chars')

# 6. Per-category breakdown
print(f'\n{"="*80}')
print('6. PER-CATEGORY BREAKDOWN')
print(f'{"="*80}')
cat_names = {0: 'arithmetic', 1: 'combinatorics', 2: 'modular', 3: 'geometry', 4: 'sequences'}
for cat in range(5):
    cat_probs = [p for p in problems if p['category'] == cat]
    if not cat_probs:
        continue
    n_cat = len(cat_probs)

    # PC0 swap language
    pc0_en = sum(1 for p in cat_probs if p['splice_pc0_swap']['lang'] == 'en')
    pc0_ft_match = sum(1 for p in cat_probs if p['splice_pc0_swap']['first_token'] == p['baseline_en']['first_token'])

    # Raw splice language
    raw_en = sum(1 for p in cat_probs if p['splice_raw']['lang'] == 'en')

    # Reverse steer language
    rev_zh = sum(1 for p in cat_probs if p['splice_reverse_steer']['lang'] == 'zh')

    # Cosines
    cos_vals = [p['h26_cosine_zh_en'] for p in cat_probs]

    print(f'  {cat_names[cat]:>15} (n={n_cat}): PC0→en={pc0_en}/{n_cat} ft_match={pc0_ft_match}/{n_cat} | raw→en={raw_en}/{n_cat} | rev→zh={rev_zh}/{n_cat} | cos={np.mean(cos_vals):.4f}')

# 7. PC0 projection values per problem
print(f'\n{"="*80}')
print('7. PC0 PROJECTION VALUES (zh→en swap)')
print(f'{"="*80}')
for i, p in enumerate(problems):
    if 'zh_pc0_proj' in p.get('splice_pc0_swap', {}):
        zh_proj = p['splice_pc0_swap']['zh_pc0_proj']
        en_mean = p['splice_pc0_swap']['en_mean_proj']
        shift = en_mean - zh_proj
        print(f'  P{i:2d}: zh_proj={zh_proj:+.4f} → en_mean={en_mean:+.4f} (shift={shift:+.4f})')

# 8. Scrambled and random: do they ever accidentally match?
print(f'\n{"="*80}')
print('8. CONTROL CONDITIONS: ACCIDENTAL MATCHES')
print(f'{"="*80}')
rand_en = sum(1 for p in problems if p['splice_random_dir']['lang'] == 'en')
rand_zh = sum(1 for p in problems if p['splice_random_dir']['lang'] == 'zh')
scram_en = sum(1 for p in problems if p['splice_scrambled']['lang'] == 'en')
scram_zh = sum(1 for p in problems if p['splice_scrambled']['lang'] == 'zh')
print(f'  Random dir: en={rand_en}/{N} zh={rand_zh}/{N}')
print(f'  Scrambled:  en={scram_en}/{N} zh={scram_zh}/{N}')

# How many random/scrambled match en baseline first token?
rand_ft_en = sum(1 for p in problems if p['splice_random_dir']['first_token'] == p['baseline_en']['first_token'])
scram_ft_en = sum(1 for p in problems if p['splice_scrambled']['first_token'] == p['baseline_en']['first_token'])
print(f'  Random dir first token == en baseline: {rand_ft_en}/{N}')
print(f'  Scrambled first token == en baseline: {scram_ft_en}/{N}')
