"""Inverse Ghost: does adding a system prompt to the BASE model raise encoding anisotropy?

The base model (no RLHF) has encoding anisotropy = 0.35x.
The instruct model has encoding anisotropy = 4.91x.

Question: if we prepend 'You are a mathematical reasoner' to the base model's
input (without any RLHF), does encoding anisotropy rise above 0.35x?

If yes: the second-person framing has geometric consequences even without RLHF.
If no: the pathway is specifically carved by RLHF. The system prompt is just
       the key that fits a lock training installed.

Either answer is interesting. One line of code to test.
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DEV = 'cuda'
D = 2048
L_TARGET = 33
EPS = 0.5
N_RANDOM = 5

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

# System prompts to test on base model
SYS_PROMPTS = {
    'none': None,  # baseline (should reproduce 0.35x)
    'math_reasoner': 'You are a mathematical reasoner. Think step by step and solve carefully.',
    'amc_solver': 'You are solving an AMC 12A multiple choice math problem. Think step by step, show your work, then clearly state your final answer as (A), (B), (C), (D), or (E).',
    'generic_helper': 'You are a helpful assistant.',
    'identity_only': 'You are a mathematician.',
}


def make_base_prompt(sys_text=None):
    """Build a base-model prompt with optional system-like prefix."""
    if sys_text is None:
        return f'Problem: {P12}\n\nSolution: Let me'
    # Prepend as plain text (no chat template — this is a base model)
    return f'{sys_text}\n\nProblem: {P12}\n\nSolution: Let me'


def extract_convention_dir(model, tok):
    """Convention direction e_c at L33 (base model, no chat template)."""
    pairs = [
        ('What is 7 * 8?', '7乘以8等于多少？'),
        ('What is 15% of 240?', '240的15%是多少？'),
        ('Calculate: 8! / (5! * 3!)', '计算：8! / (5! × 3!)'),
        ('What is the derivative of x^3 + 2x?', 'x³ + 2x的导数是多少？'),
    ]
    cap = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.dim() == 3: h = h[:, -1, :]
        cap['h'] = h.detach().cpu().float().reshape(D).numpy()

    hk = model.model.layers[L_TARGET].register_forward_hook(hook)
    en_a, zh_a = [], []
    with torch.inference_mode():
        for en, zh in pairs:
            for text, store in [(en, en_a), (zh, zh_a)]:
                prompt = f'Problem: {text}\nAnswer:'
                ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
                model(ids)
                store.append(cap['h'].copy())
                cap.clear()
    hk.remove()
    diff = np.mean(np.stack(zh_a), 0) - np.mean(np.stack(en_a), 0)
    return diff / (np.linalg.norm(diff) + 1e-12)


def measure_encoding_sensitivity(model, tok, prompt_str, e_c, randoms):
    """Encoding-only sensitivity: perturb h_L33 at last prompt token."""
    ids = tok(prompt_str, return_tensors='pt').input_ids.to(DEV)
    n_prompt_toks = ids.shape[1]

    # Baseline logits
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[:, -1, :]
    log_p0 = F.log_softmax(logits.float(), dim=-1)
    p0 = log_p0.exp()

    # Directions
    dirs = {'convention': e_c}
    for i, r in enumerate(randoms):
        dirs[f'rand_{i}'] = r

    # Perturb
    pert_v = {'v': None}
    def hook_pert(m, i, o):
        if pert_v['v'] is None: return o
        h = o[0] if isinstance(o, tuple) else o
        if h.dim() == 3:
            h2 = h.clone(); h2[:, -1, :] += pert_v['v']
            return (h2,) + o[1:] if isinstance(o, tuple) else h2
        return h + pert_v['v']

    results = {}
    hk_pert = model.model.layers[L_TARGET].register_forward_hook(hook_pert)
    with torch.no_grad():
        for name, v in dirs.items():
            vt = torch.tensor(v, dtype=torch.float16, device=DEV)
            pert_v['v'] = EPS * vt.unsqueeze(0)
            op = model(ids)
            lp1 = F.log_softmax(op.logits[:, -1, :].float(), dim=-1)
            kl = (p0 * (log_p0 - lp1)).sum().item()
            results[name] = kl
    pert_v['v'] = None
    hk_pert.remove()
    del out; torch.cuda.empty_cache()

    rand_mean = np.mean([results[f'rand_{i}'] for i in range(len(randoms))])
    ec_aniso = results['convention'] / (rand_mean + 1e-12)
    return results, ec_aniso, n_prompt_toks


def main():
    print('=' * 60)
    print('INVERSE GHOST: System prompt on BASE model')
    print('Does second-person framing create geometry without RLHF?')
    print('=' * 60, flush=True)

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B', dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # Convention direction
    print('Extracting convention direction...', flush=True)
    e_c = extract_convention_dir(model, tok)
    rng = np.random.RandomState(42)
    randoms = [rng.randn(D).astype(np.float32) for _ in range(N_RANDOM)]
    randoms = [r / np.linalg.norm(r) for r in randoms]

    # Test each system prompt condition
    results = {}
    for name, sys_text in SYS_PROMPTS.items():
        prompt = make_base_prompt(sys_text)
        print(f'\n--- {name} ({len(tok(prompt).input_ids)} tokens) ---', flush=True)
        if sys_text:
            print(f'  sys: "{sys_text[:60]}..."' if len(sys_text) > 60 else f'  sys: "{sys_text}"',
                  flush=True)

        enc_results, enc_aniso, n_toks = measure_encoding_sensitivity(
            model, tok, prompt, e_c, randoms)

        results[name] = {
            'system_prompt': sys_text,
            'n_prompt_tokens': n_toks,
            'encoding_kl': {k: float(v) for k, v in enc_results.items()},
            'convention_anisotropy': float(enc_aniso),
        }
        print(f'  encoding anisotropy = {enc_aniso:.3f}x', flush=True)
        print(f'  convention KL = {enc_results["convention"]:.6f}', flush=True)
        print(f'  random mean KL = {np.mean([enc_results[f"rand_{i}"] for i in range(N_RANDOM)]):.6f}',
              flush=True)

    # Summary
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"":20s} {"Enc Aniso":>10s} {"Conv KL":>10s} {"Rand KL":>10s} {"Tokens":>8s}')
    for name, r in results.items():
        rand_kl = np.mean([r['encoding_kl'][f'rand_{i}'] for i in range(N_RANDOM)])
        print(f'{name:20s} {r["convention_anisotropy"]:10.3f}x '
              f'{r["encoding_kl"]["convention"]:10.6f} {rand_kl:10.6f} '
              f'{r["n_prompt_tokens"]:8d}')

    # Verdict
    baseline = results['none']['convention_anisotropy']
    print(f'\nBaseline (no sys): {baseline:.3f}x')
    for name, r in results.items():
        if name == 'none': continue
        aniso = r['convention_anisotropy']
        delta = aniso - baseline
        verdict = 'RISES' if delta > 0.1 else 'FLAT' if abs(delta) <= 0.1 else 'DROPS'
        print(f'{name}: {aniso:.3f}x (delta={delta:+.3f}) → {verdict}')

    any_rise = any(r['convention_anisotropy'] > baseline + 0.1
                   for n, r in results.items() if n != 'none')
    if any_rise:
        print('\n→ SECOND-PERSON FRAMING HAS GEOMETRIC CONSEQUENCES WITHOUT RLHF')
        print('  The text itself changes the model\'s directional sensitivity.')
    else:
        print('\n→ PATHWAY IS RLHF-CARVED. System prompt is the key, RLHF is the lock.')
        print('  The base model ignores second-person framing geometrically.')

    # Save
    outpath = Path('output') / 'exp_inverse_ghost.json'
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
