"""Base vs Instruct gravitational field: is the anisotropy architectural or RLHF-carved?

Run the same perturbation sensitivity measurement on:
  1. Qwen2.5-3B (base) - no chat template, raw prompt
  2. Qwen2.5-3B-Instruct - chat template (existing result, rerun for fair comparison)

If base shows anisotropy: architectural (Theory A wins)
If base is isotropic: RLHF-carved (Theory B wins)

Also test encoding-only sensitivity (no generation):
If encoding is anisotropic: generation loop not needed (Theory C dead)
If encoding is isotropic: autoregressive feedback is load-bearing
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
MAX_GEN = 200
T0 = 50

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

SYS_INSTRUCT = (
    'You are solving an AMC 12A multiple choice math problem. '
    'Think step by step, show your work, then clearly state your '
    'final answer as (A), (B), (C), (D), or (E).')

# For base model: no chat template, just raw text
PROMPT_BASE = f'Problem: {P12}\n\nSolution: Let me'
B_TOKEN = 33  # token ID for 'B'


def make_instruct_prompt(tok):
    msgs = [{'role': 'system', 'content': SYS_INSTRUCT},
            {'role': 'user', 'content': P12}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_convention_dir(model, tok, is_base=False):
    """Convention direction e_c at L33."""
    pairs = [
        ('What is 7 * 8?', '7\u4e58\u4ee58\u7b49\u4e8e\u591a\u5c11\uff1f'),
        ('What is 15% of 240?', '240\u768415%\u662f\u591a\u5c11\uff1f'),
        ('Calculate: 8! / (5! * 3!)', '\u8ba1\u7b97\uff1a8! / (5! \u00d7 3!)'),
        ('What is the derivative of x^3 + 2x?', 'x\u00b3 + 2x\u7684\u5bfc\u6570\u662f\u591a\u5c11\uff1f'),
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
                if is_base:
                    prompt = f'Problem: {text}\nAnswer:'
                else:
                    msgs = [{'role': 'system', 'content': 'Solve this.'},
                            {'role': 'user', 'content': text}]
                    prompt = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
                model(ids)
                store.append(cap['h'].copy())
                cap.clear()
    hk.remove()
    diff = np.mean(np.stack(zh_a), 0) - np.mean(np.stack(en_a), 0)
    return diff / (np.linalg.norm(diff) + 1e-12)


def measure_encoding_sensitivity(model, tok, prompt_str, e_c, randoms):
    """Measure sensitivity at L33 during ENCODING ONLY (no generation).
    Perturb h_L33 at the last prompt token, measure KL of output logits."""
    ids = tok(prompt_str, return_tensors='pt').input_ids.to(DEV)

    # Get baseline logits via normal forward pass
    cap_h = {}
    def hook_cap(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.dim() == 3: h = h[:, -1, :]
        cap_h['h'] = h.detach().reshape(1, D)

    hk = model.model.layers[L_TARGET].register_forward_hook(hook_cap)
    with torch.no_grad():
        out = model(ids)
    hk.remove()
    logits = out.logits[:, -1, :]
    log_p0 = F.log_softmax(logits.float(), dim=-1)
    p0 = log_p0.exp()

    # Build directions: convention + randoms
    # (no B-moment for encoding — haven't generated yet)
    dirs = {'convention': e_c}
    for i, r in enumerate(randoms):
        dirs[f'rand_{i}'] = r

    # Perturb and measure
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
    return results, ec_aniso


def measure_generation_sensitivity(model, tok, prompt_str, e_c, randoms):
    """Measure sensitivity during generation (full forward pass perturbation)."""
    ids = tok(prompt_str, return_tensors='pt').input_ids.to(DEV)

    dirs = {'convention': e_c}
    for i, r in enumerate(randoms):
        dirs[f'rand_{i}'] = r
    dir_t = {n: torch.tensor(v, dtype=torch.float16, device=DEV) for n, v in dirs.items()}

    cap_h = {}
    def hook_cap(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.dim() == 3: h = h[:, -1, :]
        cap_h['h'] = h.detach().reshape(1, D)
    hk_cap = model.model.layers[L_TARGET].register_forward_hook(hook_cap)

    pert_v = {'v': None}
    def hook_pert(m, i, o):
        if pert_v['v'] is None: return o
        h = o[0] if isinstance(o, tuple) else o
        if h.dim() == 3:
            h2 = h.clone(); h2[:, -1, :] += pert_v['v']
            return (h2,) + o[1:] if isinstance(o, tuple) else h2
        return h + pert_v['v']

    sens = {n: [] for n in dirs}
    gen_ids = []

    t0 = time.time()
    with torch.no_grad():
        for step in range(MAX_GEN):
            if step == 0:
                out = model(input_ids=ids, use_cache=True)
            else:
                out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            logits = out.logits[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            if nxt.item() in (151643, 151645, 0): break  # EOS or padding
            gen_ids.append(nxt.item())

            log_p0 = F.log_softmax(logits.float(), dim=-1)
            p0 = log_p0.exp()

            hk_pert = model.model.layers[L_TARGET].register_forward_hook(hook_pert)
            for name, vt in dir_t.items():
                pert_v['v'] = EPS * vt.unsqueeze(0)
                if step == 0:
                    op = model(input_ids=ids, use_cache=False)
                else:
                    op = model(input_ids=nxt, past_key_values=kv, use_cache=False)
                lp1 = F.log_softmax(op.logits[:, -1, :].float(), dim=-1)
                kl = (p0 * (log_p0 - lp1)).sum().item()
                sens[name].append(kl)
            pert_v['v'] = None
            hk_pert.remove()
            cap_h.clear()

    dt = round(time.time() - t0, 1)
    hk_cap.remove()
    text = tok.decode(gen_ids, skip_special_tokens=True)
    del kv, out; torch.cuda.empty_cache()

    n = len(gen_ids)
    t0b = min(T0, n)
    rand_mean = np.mean([np.mean(sens[f'rand_{i}']) for i in range(len(randoms))])

    summary = {}
    for name in ['convention']:
        s = np.array(sens[name])
        early = float(np.mean(s[:t0b])) if t0b > 0 else 0
        late = float(np.mean(s[t0b:])) if n > t0b else 0
        summary[name] = {
            'mean': float(np.mean(s)),
            'early': early, 'late': late,
            'early_late_ratio': early / (late + 1e-12),
            'anisotropy': float(np.mean(s)) / (rand_mean + 1e-12),
        }
    summary['rand_mean'] = float(rand_mean)

    return summary, sens, text[:200], n, dt


def run_one_model(model_name, is_base=False):
    """Full measurement for one model variant."""
    label = 'BASE' if is_base else 'INSTRUCT'
    print(f'\n{"=" * 60}')
    print(f'  {label}: {model_name}')
    print(f'{"=" * 60}', flush=True)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # Convention direction
    e_c = extract_convention_dir(model, tok, is_base=is_base)
    rng = np.random.RandomState(42)
    randoms = [rng.randn(D).astype(np.float32) for _ in range(3)]
    randoms = [r / np.linalg.norm(r) for r in randoms]

    # Build prompt
    if is_base:
        prompt = PROMPT_BASE
    else:
        prompt = make_instruct_prompt(tok)
    print(f'  Prompt: {len(tok(prompt).input_ids)} tokens', flush=True)

    # 1. Encoding-only sensitivity
    print('  Encoding sensitivity...', flush=True)
    enc_results, enc_aniso = measure_encoding_sensitivity(
        model, tok, prompt, e_c, randoms)
    print(f'    convention aniso = {enc_aniso:.2f}x', flush=True)

    # 2. Generation sensitivity
    print('  Generation sensitivity...', flush=True)
    gen_summary, gen_traces, gen_text, n_tok, dt = measure_generation_sensitivity(
        model, tok, prompt, e_c, randoms)
    print(f'    convention aniso = {gen_summary["convention"]["anisotropy"]:.2f}x', flush=True)
    print(f'    early/late = {gen_summary["convention"]["early_late_ratio"]:.1f}x', flush=True)
    print(f'    {n_tok} tokens, {dt}s', flush=True)
    print(f'    text: {gen_text[:100]}...', flush=True)

    # Cleanup
    del model; torch.cuda.empty_cache()

    return {
        'model': model_name,
        'is_base': is_base,
        'encoding': {
            'kl_values': {k: float(v) for k, v in enc_results.items()},
            'convention_anisotropy': float(enc_aniso),
        },
        'generation': {
            'summary': gen_summary,
            'traces': {k: [float(x) for x in v] for k, v in gen_traces.items()},
            'n_tokens': n_tok,
            'time_s': dt,
            'text_preview': gen_text,
        },
    }


def main():
    print('=' * 60)
    print('GRAVITATIONAL FIELD: BASE vs INSTRUCT')
    print('Theory A (architectural) vs Theory B (RLHF-carved)')
    print('+ Encoding vs Generation (Theory C: feedback loop)')
    print('=' * 60, flush=True)

    results = {}

    # Instruct first (known result, for fair comparison)
    results['instruct'] = run_one_model('Qwen/Qwen2.5-3B-Instruct', is_base=False)

    # Base model
    results['base'] = run_one_model('Qwen/Qwen2.5-3B', is_base=True)

    # Comparison
    print('\n' + '=' * 60)
    print('COMPARISON')
    print('=' * 60)
    print(f'{"":20s} {"Enc aniso":>10s} {"Gen aniso":>10s} {"Early/Late":>10s}')
    for key in ['instruct', 'base']:
        r = results[key]
        ea = r['encoding']['convention_anisotropy']
        ga = r['generation']['summary']['convention']['anisotropy']
        el = r['generation']['summary']['convention']['early_late_ratio']
        print(f'{key:20s} {ea:10.2f}x {ga:10.2f}x {el:10.1f}x')

    # Verdict
    base_gen = results['base']['generation']['summary']['convention']['anisotropy']
    inst_gen = results['instruct']['generation']['summary']['convention']['anisotropy']
    base_enc = results['base']['encoding']['convention_anisotropy']
    inst_enc = results['instruct']['encoding']['convention_anisotropy']

    print(f'\nTHEORY A (architectural): base gen aniso > 1.5? {base_gen > 1.5}')
    print(f'THEORY B (RLHF-carved): base gen aniso < 1.2 and inst > 1.5? '
          f'{base_gen < 1.2 and inst_gen > 1.5}')
    print(f'THEORY C (feedback loop): enc aniso < 1.2 for both? '
          f'{base_enc < 1.2 and inst_enc < 1.2}')

    outpath = Path('output') / 'exp_grav_base_vs_instruct.json'
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
