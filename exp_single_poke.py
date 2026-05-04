"""Single-poke bifurcation: can one perturbation redirect the trajectory?

Inject a single perturbation vector at one (step, layer) coordinate during
P12 generation. The baseline loops. Does one poke at the right place
redirect from the loop attractor to the correct answer (-3/2)?

Sweep:
  - Steps: 5, 10, 25, 50 (all within the priming window)
  - Layers: L20, L22, L25, L30, L33
  - Directions: b_moment, convention (e_c), deflation_diff, random
  - Magnitudes: 0.5, 1.0, 2.0, 5.0

The deflation_diff direction is the mean difference between deflated and
baseline hidden states at L33 during the first 50 tokens. This is the
"natural" direction deflation pushes the computation.
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


def extract_b_moment_direction(model, tok):
    """Extract b_moment template direction at L33 via deflated run."""
    from exp_gravitational_field_p12 import extract_b_moment_template
    v_b, b_step = extract_b_moment_template(model, tok)
    return v_b


def extract_convention_direction(model, tok):
    """Extract convention direction e_c at L33."""
    from exp_gravitational_field_p12 import extract_convention_direction as _ec
    return _ec(model, tok)


def extract_deflation_diff(model, tok, prompt):
    """Extract mean(h_deflated - h_baseline) at L33 over first 50 tokens.

    Run baseline and deflated generation for 60 tokens, capture h_L33 at each
    step, compute mean difference vector.
    """
    from exp_delayed_deflation_p12 import WindowedDeflation

    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)

    def capture_run(deflator=None):
        """Run 60 tokens capturing h_L33 at each step."""
        states = []
        cap = {}
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if h.dim() == 3: h = h[:, -1, :]
            cap['h'] = h.detach().cpu().float().numpy().reshape(D)

        hk = model.model.layers[33].register_forward_hook(hook)
        with torch.no_grad():
            for step in range(60):
                if step == 0:
                    out = model(input_ids=ids, use_cache=True)
                    if deflator:
                        deflator.start_gen()
                        deflator.refresh_basis(out.past_key_values)
                else:
                    out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
                kv = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if nxt.item() in (151643, 151645): break
                states.append(cap['h'].copy())
                cap.clear()
                if deflator:
                    deflator.tick(kv)
        hk.remove()
        del kv, out; torch.cuda.empty_cache()
        return np.stack(states)

    # Baseline
    h_base = capture_run()

    # Deflated
    defl = WindowedDeflation(
        model, list(range(20, 36)), r=4, alpha=0.1, refresh_every=25)
    h_defl = capture_run(deflator=defl)
    defl.remove()

    # Mean difference over first 50 steps
    n = min(50, len(h_base), len(h_defl))
    diff = np.mean(h_defl[:n] - h_base[:n], axis=0)
    return diff / (np.linalg.norm(diff) + 1e-12)


def run_single_poke(model, tok, prompt, poke_step, poke_layer, direction, magnitude):
    """Generate P12 with a single perturbation at one (step, layer) coordinate.

    At generation step `poke_step`, at layer `poke_layer`, add
    `magnitude * direction` to the hidden state. Only at that one step.
    """
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)

    poke_state = {'step': 0, 'fired': False}
    dir_t = torch.tensor(direction, dtype=torch.float16, device=DEV)

    def hook_poke(module, inp, output):
        if poke_state['step'] != poke_step or poke_state['fired']:
            return output
        poke_state['fired'] = True
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            h2 = h.clone()
            h2[:, -1, :] += magnitude * dir_t
            return (h2,) + output[1:] if isinstance(output, tuple) else h2
        return h + magnitude * dir_t

    hk = model.model.layers[poke_layer].register_forward_hook(hook_poke)

    gen_ids = []
    t0 = time.time()
    with torch.no_grad():
        for step in range(MAX_TOKENS):
            poke_state['step'] = step
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
    hk.remove()
    text = tok.decode(gen_ids, skip_special_tokens=True)
    del kv, out; torch.cuda.empty_cache()

    ans = extract_answer(text)
    looped = len(gen_ids) >= MAX_TOKENS - 5
    return ans, len(gen_ids), dt, looped, text


def main():
    print('=' * 70)
    print('SINGLE-POKE BIFURCATION')
    print('Can one perturbation at one coordinate redirect the trajectory?')
    print('=' * 70, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()
    prompt = make_prompt(tok)
    print(f'Loaded {MODEL_NAME}, prompt={len(tok(prompt).input_ids)} tokens\n', flush=True)

    # === Extract directions ===
    print('Extracting directions...', flush=True)

    print('  b_moment...', flush=True)
    v_b = extract_b_moment_direction(model, tok)

    print('  convention...', flush=True)
    e_c = extract_convention_direction(model, tok)

    print('  deflation_diff...', flush=True)
    v_dd = extract_deflation_diff(model, tok, prompt)

    rng = np.random.RandomState(42)
    v_rand = rng.randn(D).astype(np.float32)
    v_rand /= np.linalg.norm(v_rand)

    directions = {
        'b_moment': v_b,
        'convention': e_c,
        'deflation_diff': v_dd,
        'random': v_rand,
    }

    # Cross-cosines
    print('\nDirection cosines:', flush=True)
    for n1 in directions:
        for n2 in directions:
            if n1 < n2:
                c = np.dot(directions[n1], directions[n2])
                print(f'  cos({n1}, {n2}) = {c:.4f}', flush=True)

    # === Baseline: no poke ===
    print('\n--- Baseline (no intervention) ---', flush=True)
    ans, ntok, dt, looped, text = run_single_poke(
        model, tok, prompt, poke_step=-1, poke_layer=22,
        direction=v_rand, magnitude=0)  # step=-1 means never fires
    status = 'CORRECT' if ans == CORRECT else ('LOOP' if looped else f'WRONG({ans})')
    print(f'  {status}: ans={ans}, {ntok} tokens, {dt}s', flush=True)
    print(f'  text[-100:]: {text[-100:]}', flush=True)

    # === Sweep ===
    STEPS = [5, 10, 25, 50]
    LAYERS = [20, 22, 25, 30, 33]
    MAGNITUDES = [0.5, 1.0, 2.0, 5.0]

    results = {
        'meta': {
            'model': MODEL_NAME,
            'max_tokens': MAX_TOKENS,
            'directions': list(directions.keys()),
            'steps': STEPS,
            'layers': LAYERS,
            'magnitudes': MAGNITUDES,
        },
        'baseline': {
            'answer': ans, 'n_tokens': ntok, 'looped': looped,
        },
        'pokes': [],
    }

    total = len(STEPS) * len(LAYERS) * len(directions) * len(MAGNITUDES)
    count = 0

    # First pass: fixed magnitude=2.0 sweep over steps × layers × directions
    # This finds the hot zone before the full magnitude sweep
    print(f'\n{"=" * 70}')
    print(f'PHASE 1: Fixed magnitude=2.0, sweep steps x layers x directions')
    print(f'{"=" * 70}', flush=True)

    hot_combos = []  # (step, layer, dir_name) that produced correct answer

    for step in STEPS:
        for layer in LAYERS:
            for dir_name, v in directions.items():
                mag = 2.0
                count += 1
                ans, ntok, dt, looped, text = run_single_poke(
                    model, tok, prompt, step, layer, v, mag)
                status = 'CORRECT' if ans == CORRECT else (
                    'LOOP' if looped else f'WRONG({ans})')
                marker = ' <<<' if ans == CORRECT else ''
                print(f'  [{count}] step={step} L{layer} {dir_name:15s} mag={mag} '
                      f'-> {status} ({ntok}tok, {dt}s){marker}', flush=True)

                entry = {
                    'step': step, 'layer': layer, 'direction': dir_name,
                    'magnitude': mag, 'answer': ans, 'correct': ans == CORRECT,
                    'n_tokens': ntok, 'looped': looped, 'time_s': dt,
                    'text_last200': text[-200:],
                }
                results['pokes'].append(entry)

                if ans == CORRECT:
                    hot_combos.append((step, layer, dir_name))

    # Phase 2: for hot combos, sweep magnitudes
    if hot_combos:
        print(f'\n{"=" * 70}')
        print(f'PHASE 2: Magnitude sweep on {len(hot_combos)} hot combos')
        print(f'{"=" * 70}', flush=True)

        for step, layer, dir_name in hot_combos:
            for mag in MAGNITUDES:
                if mag == 2.0: continue  # already done
                ans, ntok, dt, looped, text = run_single_poke(
                    model, tok, prompt, step, layer,
                    directions[dir_name], mag)
                status = 'CORRECT' if ans == CORRECT else (
                    'LOOP' if looped else f'WRONG({ans})')
                marker = ' <<<' if ans == CORRECT else ''
                print(f'  step={step} L{layer} {dir_name:15s} mag={mag} '
                      f'-> {status} ({ntok}tok, {dt}s){marker}', flush=True)

                results['pokes'].append({
                    'step': step, 'layer': layer, 'direction': dir_name,
                    'magnitude': mag, 'answer': ans, 'correct': ans == CORRECT,
                    'n_tokens': ntok, 'looped': looped, 'time_s': dt,
                    'text_last200': text[-200:],
                })
    else:
        print('\nNo hot combos found at mag=2.0. Trying mag=5.0...', flush=True)
        for step in STEPS:
            for layer in LAYERS:
                for dir_name, v in directions.items():
                    mag = 5.0
                    ans, ntok, dt, looped, text = run_single_poke(
                        model, tok, prompt, step, layer, v, mag)
                    status = 'CORRECT' if ans == CORRECT else (
                        'LOOP' if looped else f'WRONG({ans})')
                    marker = ' <<<' if ans == CORRECT else ''
                    print(f'  step={step} L{layer} {dir_name:15s} mag={mag} '
                          f'-> {status} ({ntok}tok, {dt}s){marker}', flush=True)

                    results['pokes'].append({
                        'step': step, 'layer': layer, 'direction': dir_name,
                        'magnitude': mag, 'answer': ans, 'correct': ans == CORRECT,
                        'n_tokens': ntok, 'looped': looped, 'time_s': dt,
                        'text_last200': text[-200:],
                    })

    # Summary
    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print(f'{"=" * 70}')
    correct_pokes = [p for p in results['pokes'] if p['correct']]
    wrong_pokes = [p for p in results['pokes'] if not p['correct'] and not p['looped']]
    loop_pokes = [p for p in results['pokes'] if p['looped']]
    print(f'Total pokes: {len(results["pokes"])}')
    print(f'Correct (B): {len(correct_pokes)}')
    print(f'Wrong answer: {len(wrong_pokes)}')
    print(f'Loop: {len(loop_pokes)}')

    if correct_pokes:
        print('\nCorrect pokes:')
        for p in correct_pokes:
            print(f'  step={p["step"]} L{p["layer"]} {p["direction"]} '
                  f'mag={p["magnitude"]} ({p["n_tokens"]}tok)')

    if wrong_pokes:
        print('\nWrong-answer pokes (switched attractor basin):')
        for p in wrong_pokes:
            print(f'  step={p["step"]} L{p["layer"]} {p["direction"]} '
                  f'mag={p["magnitude"]} -> {p["answer"]} ({p["n_tokens"]}tok)')

    # Save
    outpath = Path('output') / 'exp_single_poke.json'
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
