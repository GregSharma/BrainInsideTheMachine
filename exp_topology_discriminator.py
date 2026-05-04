"""Topology discriminator: needle-threading vs tunneling.

Two experiments that make maximally divergent predictions:

EXP 1 (STAGED INTERVENTION):
  - Steps 0-8: sustained low-magnitude perturbation (mag=1.0) across all layers.
    Purpose: keep trajectory shallow in basin(a1), prevent premature commitment.
  - Step 10, L20: single mag=5.0 poke.
    Purpose: push across the separatrix at the known neck.
  - Steps 11+: nothing.

  Needle-threading predicts: CORRECT (B). The shallow-keeping delivers the
  trajectory to the neck with enough altitude to cross.

  Tunneling predicts: WRONG. Single poke can't create transient attractor.

EXP 2 (LATE-ONSET SUSTAINED):
  - Steps 0-15: no perturbation. Trajectory commits deep into basin(a1).
  - Steps 16-50: full deflation protocol (5 layers, alpha=0.1).

  Needle-threading predicts: WRONG. The neck is behind the trajectory by step 15.
  No amount of subsequent perturbation can pull it back.

  Tunneling predicts: CORRECT. 35 tokens of sustained correction creates
  transient attractor inside basin(a2).

If Exp1=B and Exp2!=B: needle-threading confirmed.
If Exp1!=B and Exp2=B: tunneling confirmed.
If both fail or both succeed: neither model captures the mechanism.
"""
import json, time, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
DEV = 'cuda'
D = 2048
MAX_TOKENS = 2048
CORRECT = 'B'

SYS = ('You are solving an AMC 12A multiple choice math problem. '
       'Think step by step, show your work, then clearly state your '
       'final answer as (A), (B), (C), (D), or (E).')

P12 = (
    'The harmonic mean of a collection of numbers is the reciprocal '
    'of the arithmetic mean of the reciprocals of the numbers in the '
    'collection. For example, the harmonic mean of 4, 4, and 5 is\n\n'
    '1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n'
    'What is the harmonic mean of all the real roots of the 4050th '
    'degree polynomial\n\n'
    '\\prod_{k=1}^{2025} (kx^2 - 4x - 3) = '
    '(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)...'
    '(2025x^2 - 4x - 3)?\n\n'
    '(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3')


def make_prompt(tok):
    msgs = [{'role': 'system', 'content': SYS},
            {'role': 'user', 'content': P12}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


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


def run_exp1_staged(model, tok, prompt, shallow_mag, poke_mag, shallow_end, poke_step, poke_layer):
    """Staged intervention: shallow-keeping + single poke at the neck.

    Steps 0..shallow_end: add shallow_mag * random_unit_vector to h at ALL layers.
    Step poke_step at poke_layer: add poke_mag * random_unit_vector.
    All other steps: no intervention.
    """
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    n_layers = len(model.model.layers)

    rng = np.random.RandomState(42)
    # Use a fixed random direction for shallow-keeping (same across layers)
    # This is the "anti-commitment" push u2014 direction shouldn't matter much
    shallow_dir = rng.randn(D).astype(np.float32)
    shallow_dir /= np.linalg.norm(shallow_dir)
    shallow_t = torch.tensor(shallow_dir, dtype=torch.float16, device=DEV)

    # Poke direction u2014 use a different random direction
    poke_dir = rng.randn(D).astype(np.float32)
    poke_dir /= np.linalg.norm(poke_dir)
    poke_t = torch.tensor(poke_dir, dtype=torch.float16, device=DEV)

    state = {'step': 0, 'poke_fired': False}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, output):
            step = state['step']
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() != 3:
                return output

            should_shallow = (step <= shallow_end)
            should_poke = (step == poke_step and layer_idx == poke_layer
                          and not state['poke_fired'])

            if not should_shallow and not should_poke:
                return output

            h2 = h.clone()
            if should_shallow:
                h2[:, -1, :] += shallow_mag * shallow_t
            if should_poke:
                h2[:, -1, :] += poke_mag * poke_t
                state['poke_fired'] = True

            if isinstance(output, tuple):
                return (h2,) + output[1:]
            return h2
        return hook

    # Install hooks on ALL layers (shallow-keeping is everywhere)
    for i in range(n_layers):
        hk = model.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(hk)

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


def run_exp2_late_onset(model, tok, prompt, onset_step, deflation_end=50):
    """Late-onset sustained deflation.

    Steps 0..onset_step: no intervention.
    Steps onset_step..deflation_end: full Q-deflation protocol.
    Steps deflation_end+: no intervention.
    """
    from exp_delayed_deflation_p12 import WindowedDeflation

    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)

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
    print('TOPOLOGY DISCRIMINATOR: Needle-Threading vs Tunneling')
    print('=' * 70, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()
    prompt = make_prompt(tok)
    print(f'Loaded {MODEL_NAME}, prompt={len(tok(prompt).input_ids)} tokens\n', flush=True)

    results = {'meta': {'model': MODEL_NAME, 'correct': CORRECT}}

    # === BASELINE ===
    print('--- Baseline (no intervention) ---', flush=True)
    ans, ntok, dt, text = run_exp1_staged(
        model, tok, prompt, shallow_mag=0, poke_mag=0,
        shallow_end=-1, poke_step=-1, poke_layer=20)
    status = 'CORRECT' if ans == CORRECT else f'WRONG({ans})'
    print(f'  {status}: {ntok} tokens, {dt}s', flush=True)
    results['baseline'] = {'answer': ans, 'n_tokens': ntok, 'time_s': dt}

    # === CONTROL: Full deflation (should work) ===
    print('\n--- Control: Full deflation steps 0-50 (should produce B) ---', flush=True)
    ans, ntok, dt, text = run_exp2_late_onset(
        model, tok, prompt, onset_step=0, deflation_end=2048)
    status = 'CORRECT' if ans == CORRECT else f'WRONG({ans})'
    print(f'  {status}: {ntok} tokens, {dt}s', flush=True)
    results['control_full_deflation'] = {'answer': ans, 'n_tokens': ntok, 'time_s': dt}

    # === EXP 1: STAGED INTERVENTION ===
    print(f'\n{"=" * 70}')
    print('EXP 1: STAGED INTERVENTION')
    print('Shallow-keeping (mag=1.0, all layers, steps 0-8) + poke (mag=5.0, step=10, L20)')
    print('Needle-threading predicts: CORRECT (B)')
    print('Tunneling predicts: WRONG')
    print(f'{"=" * 70}', flush=True)

    # Primary test
    ans, ntok, dt, text = run_exp1_staged(
        model, tok, prompt, shallow_mag=1.0, poke_mag=5.0,
        shallow_end=8, poke_step=10, poke_layer=20)
    status = 'CORRECT' if ans == CORRECT else f'WRONG({ans})'
    marker = ' <<<' if ans == CORRECT else ''
    print(f'  PRIMARY: {status} ({ntok}tok, {dt}s){marker}', flush=True)
    results['exp1_primary'] = {
        'answer': ans, 'n_tokens': ntok, 'time_s': dt,
        'shallow_mag': 1.0, 'poke_mag': 5.0,
        'shallow_end': 8, 'poke_step': 10, 'poke_layer': 20,
        'text_last200': text[-200:],
    }

    # Variations on Exp 1
    exp1_variants = [
        # Vary shallow magnitude
        {'shallow_mag': 0.5, 'poke_mag': 5.0, 'shallow_end': 8, 'poke_step': 10, 'poke_layer': 20,
         'label': 'shallow_0.5'},
        {'shallow_mag': 2.0, 'poke_mag': 5.0, 'shallow_end': 8, 'poke_step': 10, 'poke_layer': 20,
         'label': 'shallow_2.0'},
        # Vary shallow window
        {'shallow_mag': 1.0, 'poke_mag': 5.0, 'shallow_end': 5, 'poke_step': 10, 'poke_layer': 20,
         'label': 'shallow_end_5'},
        {'shallow_mag': 1.0, 'poke_mag': 5.0, 'shallow_end': 12, 'poke_step': 10, 'poke_layer': 20,
         'label': 'shallow_end_12'},
        # Poke only (no shallow-keeping) u2014 should match single-poke result
        {'shallow_mag': 0.0, 'poke_mag': 5.0, 'shallow_end': -1, 'poke_step': 10, 'poke_layer': 20,
         'label': 'poke_only'},
        # Shallow only (no poke) u2014 does anti-commitment alone work?
        {'shallow_mag': 1.0, 'poke_mag': 0.0, 'shallow_end': 8, 'poke_step': -1, 'poke_layer': 20,
         'label': 'shallow_only'},
        # Shallow only extended (steps 0-15)
        {'shallow_mag': 1.0, 'poke_mag': 0.0, 'shallow_end': 15, 'poke_step': -1, 'poke_layer': 20,
         'label': 'shallow_only_15'},
        # Shallow only extended (steps 0-50)
        {'shallow_mag': 1.0, 'poke_mag': 0.0, 'shallow_end': 50, 'poke_step': -1, 'poke_layer': 20,
         'label': 'shallow_only_50'},
    ]

    results['exp1_variants'] = {}
    for v in exp1_variants:
        label = v.pop('label')
        ans, ntok, dt, text = run_exp1_staged(model, tok, prompt, **v)
        status = 'CORRECT' if ans == CORRECT else f'WRONG({ans})'
        marker = ' <<<' if ans == CORRECT else ''
        print(f'  {label}: {status} ({ntok}tok, {dt}s){marker}', flush=True)
        results['exp1_variants'][label] = {
            'answer': ans, 'n_tokens': ntok, 'time_s': dt,
            'text_last200': text[-200:], **v,
        }

    # === EXP 2: LATE-ONSET SUSTAINED ===
    print(f'\n{"=" * 70}')
    print('EXP 2: LATE-ONSET SUSTAINED DEFLATION')
    print('No perturbation steps 0-N, then full deflation protocol.')
    print('Needle-threading predicts: WRONG (neck is behind)')
    print('Tunneling predicts: CORRECT (transient attractor created)')
    print(f'{"=" * 70}', flush=True)

    exp2_onsets = [5, 10, 15, 20, 30, 50]
    results['exp2'] = {}
    for onset in exp2_onsets:
        ans, ntok, dt, text = run_exp2_late_onset(
            model, tok, prompt, onset_step=onset)
        status = 'CORRECT' if ans == CORRECT else f'WRONG({ans})'
        marker = ' <<<' if ans == CORRECT else ''
        print(f'  onset={onset}: {status} ({ntok}tok, {dt}s){marker}', flush=True)
        results['exp2'][f'onset_{onset}'] = {
            'answer': ans, 'n_tokens': ntok, 'time_s': dt,
            'onset_step': onset,
            'text_last200': text[-200:],
        }

    # === VERDICT ===
    print(f'\n{"=" * 70}')
    print('VERDICT')
    print(f'{"=" * 70}')

    exp1_result = results['exp1_primary']['answer'] == CORRECT
    exp2_onset15 = results['exp2'].get('onset_15', {}).get('answer', '?') == CORRECT
    exp2_onset20 = results['exp2'].get('onset_20', {}).get('answer', '?') == CORRECT

    print(f'Exp1 (staged): {"CORRECT" if exp1_result else "WRONG"}')
    print(f'Exp2 onset=15: {"CORRECT" if exp2_onset15 else "WRONG"}')
    print(f'Exp2 onset=20: {"CORRECT" if exp2_onset20 else "WRONG"}')

    if exp1_result and not exp2_onset15:
        print('\n=> NEEDLE-THREADING CONFIRMED')
        print('   The separatrix exists. Deflation keeps trajectory shallow.')
        print('   The neck is at (step~10, L20). Sustained perturbation = anti-commitment.')
    elif not exp1_result and exp2_onset15:
        print('\n=> TUNNELING CONFIRMED')
        print('   The perturbation creates transient basin structure.')
        print('   Duration matters more than timing.')
    elif exp1_result and exp2_onset15:
        print('\n=> BOTH WORK - neither model is sufficient')
        print('   Multiple mechanisms may coexist.')
    else:
        print('\n=> BOTH FAIL - neither model captures the mechanism')
        print('   The topology is more complex than either model predicts.')

    # Save
    outpath = Path('output') / 'exp_topology_discriminator.json'
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
