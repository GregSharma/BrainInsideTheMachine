"""Topology discriminator on BASE MODEL (Qwen2.5-3B).

The prior run used Instruct model where deflation doesn't produce B.
The base model loops on P12, and deflation reliably produces B.
This is the correct setup for testing needle-threading vs tunneling.

EXP 1 (STAGED): shallow-keeping (mag=1.0, all layers, steps 0-8) + poke (mag=5.0, step=10, L20)
EXP 2 (LATE-ONSET): no intervention steps 0-N, then full deflation steps N+
"""
import json, time, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'Qwen/Qwen2.5-3B'  # BASE model
DEV = 'cuda'
D = 2048
MAX_TOKENS = 2048
CORRECT = 'B'

SYS = ('You are solving an AMC 12A multiple choice math problem. '
       'Think step by step, show your work, then clearly state your '
       'final answer as (A), (B), (C), (D), or (E).')

P12_TEXT = (
    'The harmonic mean of a collection of numbers is the reciprocal of the '
    'arithmetic mean of the reciprocals of the numbers in the collection. '
    'For example, the harmonic mean of 4, 4, and 5 is\n\n'
    '1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n'
    'What is the harmonic mean of all the real roots of the 4050th degree '
    'polynomial\n\n'
    r'\prod_{k=1}^{2025} (kx^2 - 4x - 3) = '
    '(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)...'
    '(2025x^2 - 4x - 3)?\n\n'
    '(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3')

# Raw chat template for base model (no apply_chat_template)
PROMPT = f'<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n'


def extract_answer(text):
    if not text: return '?'
    if re.search(r'\\boxed\{[^}]*-\\frac\{3\}\{2\}', text): return 'B'
    if re.search(r'\\boxed\{[^}]*-3/2', text): return 'B'
    m = re.findall(r'\\boxed\{[^}]*\b([A-E])\b[^}]*\}', text)
    if m: return m[-1]
    m = re.findall(r'\\boxed\{[^}]*(-?\d+/\d+)[^}]*\}', text)
    if m:
        mapping = {'-5/3': 'A', '-3/2': 'B', '-6/5': 'C', '-5/6': 'D', '-2/3': 'E'}
        return mapping.get(m[-1], '?')
    m = re.findall(r'\(([A-E])\)', text[-500:])
    if m: return m[-1]
    if '-3/2' in text[-500:]: return 'B'
    m = re.findall(r'answer is.*?([A-E])\b', text[-500:], re.IGNORECASE)
    if m: return m[-1]
    return '?'


def run_staged(model, tok, shallow_mag, poke_mag, shallow_end, poke_step, poke_layer):
    """Staged intervention: shallow-keeping + single poke."""
    ids = tok(PROMPT, return_tensors='pt').input_ids.to(DEV)
    n_layers = len(model.model.layers)

    rng = np.random.RandomState(42)
    shallow_dir = rng.randn(D).astype(np.float32)
    shallow_dir /= np.linalg.norm(shallow_dir)
    shallow_t = torch.tensor(shallow_dir, dtype=torch.float16, device=DEV)

    poke_dir = rng.randn(D).astype(np.float32)
    poke_dir /= np.linalg.norm(poke_dir)
    poke_t = torch.tensor(poke_dir, dtype=torch.float16, device=DEV)

    state = {'step': 0, 'poke_fired': False}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, output):
            step = state['step']
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() != 3: return output

            should_shallow = (step <= shallow_end) and shallow_mag > 0
            should_poke = (step == poke_step and layer_idx == poke_layer
                          and not state['poke_fired'] and poke_mag > 0)

            if not should_shallow and not should_poke: return output

            h2 = h.clone()
            if should_shallow:
                h2[:, -1, :] += shallow_mag * shallow_t
            if should_poke:
                h2[:, -1, :] += poke_mag * poke_t
                state['poke_fired'] = True
            return (h2,) + output[1:] if isinstance(output, tuple) else h2
        return hook

    for i in range(n_layers):
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    gen_ids = []
    t0 = time.time()
    with torch.no_grad():
        for step in range(MAX_TOKENS):
            state['step'] = step
            if step == 0:
                out = model(input_ids=ids, use_cache=True)
            else:
                out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            logits = out.logits[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            if nxt.item() in (151643, 151645): break
            gen_ids.append(nxt.item())

    dt = round(time.time() - t0, 1)
    for hk in hooks: hk.remove()
    text = tok.decode(gen_ids, skip_special_tokens=True)
    del kv, out; torch.cuda.empty_cache()
    return extract_answer(text), len(gen_ids), dt, text


def run_late_onset(model, tok, onset_step, deflation_end=2048):
    """Late-onset sustained Q-deflation."""
    from exp_delayed_deflation_p12 import WindowedDeflation

    ids = tok(PROMPT, return_tensors='pt').input_ids.to(DEV)

    defl = WindowedDeflation(
        model, list(range(20, 36)), r=4, alpha=0.1, refresh_every=25,
        active_from=onset_step, active_until=deflation_end)

    gen_ids = []
    t0 = time.time()
    with torch.no_grad():
        for step in range(MAX_TOKENS):
            if step == 0:
                out = model(input_ids=ids, use_cache=True)
                defl.start_gen()
                defl.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            logits = out.logits[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            if nxt.item() in (151643, 151645): break
            gen_ids.append(nxt.item())
            defl.tick(kv)

    dt = round(time.time() - t0, 1)
    defl.remove()
    text = tok.decode(gen_ids, skip_special_tokens=True)
    del kv, out; torch.cuda.empty_cache()
    return extract_answer(text), len(gen_ids), dt, text


def main():
    print('=' * 70)
    print('TOPOLOGY DISCRIMINATOR (BASE MODEL)')
    print('Needle-threading vs Tunneling on Qwen2.5-3B (base)')
    print('Baseline = LOOP, deflation = B')
    print('=' * 70, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()
    print(f'Loaded {MODEL_NAME}, prompt={len(tok(PROMPT).input_ids)} tokens\n', flush=True)

    results = {'meta': {'model': MODEL_NAME, 'correct': CORRECT, 'prompt_format': 'raw_chat_template'}}

    # === BASELINE (no intervention) ===
    print('--- Baseline (no intervention, should LOOP) ---', flush=True)
    ans, ntok, dt, text = run_staged(model, tok, 0, 0, -1, -1, 20)
    looped = ntok >= MAX_TOKENS - 5
    status = 'CORRECT' if ans == CORRECT else ('LOOP' if looped else f'WRONG({ans})')
    print(f'  {status}: {ntok} tokens, {dt}s', flush=True)
    results['baseline'] = {'answer': ans, 'n_tokens': ntok, 'looped': looped}

    # === CONTROL: Full deflation (should produce B) ===
    print('\n--- Control: Full deflation onset=0 (should produce B) ---', flush=True)
    ans, ntok, dt, text = run_late_onset(model, tok, onset_step=0)
    status = 'CORRECT' if ans == CORRECT else f'WRONG({ans})'
    print(f'  {status}: {ntok} tokens, {dt}s', flush=True)
    results['control_full'] = {'answer': ans, 'n_tokens': ntok}
    if ans != CORRECT:
        print('  WARNING: Full deflation control FAILED. Results unreliable.', flush=True)

    # === EXP 1: STAGED INTERVENTION ===
    print(f'\n{"=" * 70}')
    print('EXP 1: STAGED INTERVENTION')
    print('Needle-threading predicts: CORRECT (B)')
    print('Tunneling predicts: LOOP/WRONG')
    print(f'{"=" * 70}', flush=True)

    exp1_configs = [
        # Primary: shallow steps 0-8, poke step 10 L20
        {'label': 'primary',          'sm': 1.0, 'pm': 5.0, 'se': 8,  'ps': 10, 'pl': 20},
        # Magnitude variants
        {'label': 'shallow_0.5',      'sm': 0.5, 'pm': 5.0, 'se': 8,  'ps': 10, 'pl': 20},
        {'label': 'shallow_2.0',      'sm': 2.0, 'pm': 5.0, 'se': 8,  'ps': 10, 'pl': 20},
        {'label': 'shallow_3.0',      'sm': 3.0, 'pm': 5.0, 'se': 8,  'ps': 10, 'pl': 20},
        # Window variants
        {'label': 'shallow_end_5',    'sm': 1.0, 'pm': 5.0, 'se': 5,  'ps': 10, 'pl': 20},
        {'label': 'shallow_end_12',   'sm': 1.0, 'pm': 5.0, 'se': 12, 'ps': 10, 'pl': 20},
        # Poke only
        {'label': 'poke_only',        'sm': 0.0, 'pm': 5.0, 'se': -1, 'ps': 10, 'pl': 20},
        {'label': 'poke_only_mag10',  'sm': 0.0, 'pm': 10., 'se': -1, 'ps': 10, 'pl': 20},
        # Shallow only
        {'label': 'shallow_only_8',   'sm': 1.0, 'pm': 0.0, 'se': 8,  'ps': -1, 'pl': 20},
        {'label': 'shallow_only_50',  'sm': 1.0, 'pm': 0.0, 'se': 50, 'ps': -1, 'pl': 20},
        {'label': 'shallow_only_100', 'sm': 1.0, 'pm': 0.0, 'se': 100,'ps': -1, 'pl': 20},
        # Different poke layers
        {'label': 'poke_L22',         'sm': 1.0, 'pm': 5.0, 'se': 8,  'ps': 10, 'pl': 22},
        {'label': 'poke_L25',         'sm': 1.0, 'pm': 5.0, 'se': 8,  'ps': 10, 'pl': 25},
        {'label': 'poke_L33',         'sm': 1.0, 'pm': 5.0, 'se': 8,  'ps': 10, 'pl': 33},
        # Different poke steps
        {'label': 'poke_step5',       'sm': 1.0, 'pm': 5.0, 'se': 4,  'ps': 5,  'pl': 20},
        {'label': 'poke_step25',      'sm': 1.0, 'pm': 5.0, 'se': 24, 'ps': 25, 'pl': 20},
    ]

    results['exp1'] = {}
    for cfg in exp1_configs:
        label = cfg['label']
        ans, ntok, dt, text = run_staged(
            model, tok, cfg['sm'], cfg['pm'], cfg['se'], cfg['ps'], cfg['pl'])
        looped = ntok >= MAX_TOKENS - 5
        status = 'CORRECT' if ans == CORRECT else ('LOOP' if looped else f'WRONG({ans})')
        marker = ' <<<' if ans == CORRECT else ''
        print(f'  {label:20s}: {status:10s} ({ntok}tok, {dt}s){marker}', flush=True)
        results['exp1'][label] = {
            'answer': ans, 'n_tokens': ntok, 'looped': looped, 'time_s': dt,
            'config': cfg, 'text_last200': text[-200:],
        }

    # === EXP 2: LATE-ONSET DEFLATION ===
    print(f'\n{"=" * 70}')
    print('EXP 2: LATE-ONSET SUSTAINED DEFLATION')
    print('Needle-threading predicts: LOOP/WRONG after neck (onset >= 15)')
    print('Tunneling predicts: CORRECT even at late onset')
    print(f'{"=" * 70}', flush=True)

    exp2_onsets = [0, 5, 10, 15, 20, 30, 50, 100]
    results['exp2'] = {}
    for onset in exp2_onsets:
        ans, ntok, dt, text = run_late_onset(model, tok, onset_step=onset)
        looped = ntok >= MAX_TOKENS - 5
        status = 'CORRECT' if ans == CORRECT else ('LOOP' if looped else f'WRONG({ans})')
        marker = ' <<<' if ans == CORRECT else ''
        print(f'  onset={onset:3d}: {status:10s} ({ntok}tok, {dt}s){marker}', flush=True)
        results['exp2'][f'onset_{onset}'] = {
            'answer': ans, 'n_tokens': ntok, 'looped': looped, 'time_s': dt,
            'onset_step': onset, 'text_last200': text[-200:],
        }

    # === VERDICT ===
    print(f'\n{"=" * 70}')
    print('VERDICT')
    print(f'{"=" * 70}')

    exp1_primary = results['exp1'].get('primary', {}).get('answer', '?') == CORRECT
    exp2_15 = results['exp2'].get('onset_15', {}).get('answer', '?') == CORRECT
    exp2_20 = results['exp2'].get('onset_20', {}).get('answer', '?') == CORRECT
    exp2_50 = results['exp2'].get('onset_50', {}).get('answer', '?') == CORRECT

    print(f'Control (full deflation): {results["control_full"]["answer"]}')
    print(f'Exp1 staged primary: {"B" if exp1_primary else results["exp1"]["primary"]["answer"]}')
    print(f'Exp2 onset=15: {"B" if exp2_15 else results["exp2"]["onset_15"]["answer"]}')
    print(f'Exp2 onset=50: {"B" if exp2_50 else results["exp2"]["onset_50"]["answer"]}')

    if exp1_primary and not exp2_15:
        print('\n=> NEEDLE-THREADING CONFIRMED')
    elif not exp1_primary and exp2_15:
        print('\n=> TUNNELING CONFIRMED')
    elif exp1_primary and exp2_15:
        print('\n=> BOTH WORK - hybrid mechanism')
    else:
        print('\n=> BOTH FAIL - mechanism more complex than either model')

    # Exp2 transition curve
    print('\nExp2 transition curve:')
    for onset in exp2_onsets:
        r = results['exp2'][f'onset_{onset}']
        marker = 'B' if r['answer'] == CORRECT else ('LOOP' if r['looped'] else r['answer'])
        print(f'  onset={onset:3d} -> {marker}')

    outpath = Path('output') / 'exp_topology_base.json'
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
