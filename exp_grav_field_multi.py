"""Gravitational field: multi-problem structural test.

Does the anisotropy structure (dominant direction, temporal drop)
replicate across different problems? Or is it P12-specific?

3 problems + P12. Per problem: extract B-moment direction,
measure full-forward-pass sensitivity, compare structure.
Shared controls: convention_ec, 3 random directions.
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'
D = 2048
L_TARGET = 33
N_LAYERS = 36
EPS = 0.5
MAX_GEN = 300  # enough to see temporal structure
T0 = 50  # priming window boundary

SYS_MATH = (
    'You are a careful mathematical reasoner. Think step by step, '
    'show your work clearly, and state the final numerical answer on its own line.')

# --- Problems ---
PROBLEMS = [
    {
        'name': 'P12_harmonic',
        'text': (
            'The harmonic mean of a collection of numbers is the reciprocal '
            'of the arithmetic mean of the reciprocals of the numbers in the '
            'collection. For example, the harmonic mean of 4, 4, and 5 is\n\n'
            '1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n'
            'What is the harmonic mean of all the real roots of the 4050th '
            'degree polynomial\n\n'
            '\\prod_{k=1}^{2025} (kx^2 - 4x - 3) = '
            '(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)...'
            '(2025x^2 - 4x - 3)?\n\n'
            '(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3'),
        'answer_token': 33,  # B
        'sys': ('You are solving an AMC 12A multiple choice math problem. '
                'Think step by step, show your work, then clearly state your '
                'final answer as (A), (B), (C), (D), or (E).'),
    },
    {
        'name': 'algebra_quadratic',
        'text': 'Solve for x: 2x\u00b2 - 8 = 0',
        'answer_token': None,  # we'll find it
        'answer_str': '2',
        'sys': SYS_MATH,
    },
    {
        'name': 'arithmetic_mult',
        'text': 'Calculate: 23 \u00d7 17',
        'answer_token': None,
        'answer_str': '391',
        'sys': SYS_MATH,
    },
    {
        'name': 'geometry_hypotenuse',
        'text': 'Find the hypotenuse of a right triangle with legs 5 and 12',
        'answer_token': None,
        'answer_str': '13',
        'sys': SYS_MATH,
    },
]


def build_prompt(tok, text, sys):
    msgs = [{'role': 'system', 'content': sys}, {'role': 'user', 'content': text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---- Q-Deflation (from working experiments) ----
class QDeflation:
    def __init__(self, model, layers, r=4, alpha=0.1, refresh=25):
        self.model = model
        self.layers = set(layers)
        self.r, self.alpha, self.refresh = r, alpha, refresh
        self.hooks, self.U_r, self.step = [], {}, 0
        self.active = False
        for ell in self.layers:
            h = model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._hook(ell))
            self.hooks.append(h)

    def _hook(self, li):
        def fn(mod, inp, out):
            if not self.active or li not in self.U_r:
                return out
            b, s, d = out.shape
            t = out.view(b, s, 16, 128)
            for kh in range(len(self.U_r[li])):
                if kh not in self.U_r[li]: continue
                U = self.U_r[li][kh]
                gs = 16 // len(self.U_r[li])
                sl = slice(kh * gs, (kh + 1) * gs)
                q = t[:, :, sl, :]
                t[:, :, sl, :] = q - self.alpha * (q @ U @ U.T)
            return t.view(b, s, d)
        return fn

    def refresh_basis(self, kv):
        for ell in self.layers:
            keys = kv.layers[ell].keys
            self.U_r[ell] = {}
            for kh in range(keys.shape[1]):
                K = keys[0, kh].float()
                if K.shape[0] < self.r: continue
                _, _, Vh = torch.linalg.svd(K, full_matrices=False)
                self.U_r[ell][kh] = Vh[:self.r].T.contiguous().to(DEV, dtype=torch.float16)

    def start(self): self.active = True; self.step = 0
    def tick(self, kv):
        self.step += 1
        if self.step % self.refresh == 0: self.refresh_basis(kv)
    def remove(self):
        for h in self.hooks: h.remove()


def find_answer_token(tok, answer_str):
    """Find the token ID that starts the answer string."""
    ids = tok.encode(answer_str, add_special_tokens=False)
    return ids[0] if ids else None


def extract_b_moment(model, tok, prob):
    """Run deflated generation, find the step where the answer token
    ranks highest at L35. Return h_L33 there as a unit direction."""
    prompt = build_prompt(tok, prob['text'], prob['sys'])
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)

    ans_tok = prob.get('answer_token')
    if ans_tok is None:
        ans_tok = find_answer_token(tok, prob['answer_str'])
    if ans_tok is None:
        print(f"    WARNING: can't find answer token for {prob['name']}")
        return None, -1

    cap33, cap35 = {}, {}
    def h33(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.dim() == 3: h = h[:, -1, :]
        cap33['h'] = h.detach()
    def h35(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.dim() == 3: h = h[:, -1, :]
        cap35['h'] = h.detach()

    hk33 = model.model.layers[L_TARGET].register_forward_hook(h33)
    hk35 = model.model.layers[35].register_forward_hook(h35)
    defl = QDeflation(model, list(range(20, 36)))
    norm_fn, lm = model.model.norm, model.lm_head

    best_rank, best_h, best_step = 999999, None, -1
    with torch.no_grad():
        for step in range(MAX_GEN):
            if step == 0:
                out = model(input_ids=ids, use_cache=True)
                defl.start(); defl.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if nxt.item() in (151643, 151645): break
            defl.tick(kv)

            if 'h' in cap35:
                logits = lm(norm_fn(cap35['h'])).squeeze()
                rank = (logits > logits[ans_tok]).sum().item() + 1
                if rank < best_rank:
                    best_rank = rank
                    best_h = cap33['h'].cpu().float().reshape(D)
                    best_step = step
            cap33.clear(); cap35.clear()

    defl.remove(); hk33.remove(); hk35.remove()
    del kv, out; torch.cuda.empty_cache()

    if best_h is None:
        return None, -1
    v = best_h / (best_h.norm() + 1e-12)
    return v.numpy(), best_step


def extract_convention_dir(model, tok):
    """Convention direction e_c at L33 from 4 EN/ZH problem pairs."""
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
                p = build_prompt(tok, text, SYS_MATH)
                model(tok(p, return_tensors='pt').input_ids.to(DEV))
                store.append(cap['h'].copy()); cap.clear()
    hk.remove()
    diff = np.mean(np.stack(zh_a), 0) - np.mean(np.stack(en_a), 0)
    return diff / (np.linalg.norm(diff) + 1e-12)


def measure_one_problem(model, tok, prob, e_c, randoms):
    """Full gravitational field measurement for one problem.
    Returns dict with per-direction sensitivity traces."""
    print(f"\n  [{prob['name']}]", flush=True)

    # 1. Extract B-moment direction
    v_b, b_step = extract_b_moment(model, tok, prob)
    if v_b is None:
        print(f"    SKIP: no B-moment found"); return None
    print(f"    B-moment at step {b_step}", flush=True)

    # 2. Build direction set: b_moment, convention, 3 randoms
    dirs = {'b_moment': v_b, 'convention': e_c}
    for i, r in enumerate(randoms):
        dirs[f'rand_{i}'] = r

    dir_t = {n: torch.tensor(v, dtype=torch.float16, device=DEV) for n, v in dirs.items()}

    # 3. Run baseline generation + perturbation measurement
    prompt = build_prompt(tok, prob['text'], prob['sys'])
    input_ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)

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
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            logits = out.logits[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            if nxt.item() in (151643, 151645): break
            gen_ids.append(nxt.item())

            log_p0 = F.log_softmax(logits.float(), dim=-1)
            p0 = log_p0.exp()

            # Perturbed passes
            hk_pert = model.model.layers[L_TARGET].register_forward_hook(hook_pert)
            for name, vt in dir_t.items():
                pert_v['v'] = EPS * vt.unsqueeze(0)
                if step == 0:
                    op = model(input_ids=input_ids, use_cache=False)
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
    del kv, out; torch.cuda.empty_cache()

    n = len(gen_ids)
    text = tok.decode(gen_ids, skip_special_tokens=True)
    print(f"    {n} tokens, {dt}s", flush=True)

    # Compute summary stats
    t0b = min(T0, n)
    rand_mean = np.mean([np.mean(sens[f'rand_{i}']) for i in range(len(randoms))])

    summary = {}
    for name in ['b_moment', 'convention']:
        s = np.array(sens[name])
        early = float(np.mean(s[:t0b])) if t0b > 0 else 0
        late = float(np.mean(s[t0b:])) if n > t0b else 0
        summary[name] = {
            'mean': float(np.mean(s)),
            'early': early,
            'late': late,
            'early_late_ratio': early / (late + 1e-12),
            'anisotropy_vs_rand': float(np.mean(s)) / (rand_mean + 1e-12),
        }
        print(f"    {name}: aniso={summary[name]['anisotropy_vs_rand']:.2f}x, "
              f"early/late={summary[name]['early_late_ratio']:.1f}x", flush=True)

    return {
        'name': prob['name'],
        'b_step': b_step,
        'n_tokens': n,
        'time_s': dt,
        'text_first_100': text[:100],
        'summary': summary,
        'rand_mean': float(rand_mean),
        'traces': {k: [float(x) for x in v] for k, v in sens.items()},
    }


def main():
    print('=' * 60)
    print('GRAVITATIONAL FIELD: MULTI-PROBLEM STRUCTURAL TEST')
    print('=' * 60, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()
    print(f'Loaded {MODEL}\n', flush=True)

    # Shared controls
    print('Extracting shared directions...', flush=True)
    e_c = extract_convention_dir(model, tok)
    rng = np.random.RandomState(42)
    randoms = [rng.randn(D).astype(np.float32) for _ in range(3)]
    randoms = [r / np.linalg.norm(r) for r in randoms]
    print(f'  e_c norm: {np.linalg.norm(e_c):.4f} (should be 1)', flush=True)

    # Per-problem measurement
    results = []
    for prob in PROBLEMS:
        r = measure_one_problem(model, tok, prob, e_c, randoms)
        if r is not None:
            results.append(r)

    # Cross-problem comparison
    print('\n' + '=' * 60)
    print('CROSS-PROBLEM COMPARISON')
    print('=' * 60)
    print(f'{"Problem":25s} {"b_aniso":>8s} {"ec_aniso":>8s} {"b_early/late":>12s} {"ec_early/late":>13s}')
    for r in results:
        s = r['summary']
        print(f"{r['name']:25s} {s['b_moment']['anisotropy_vs_rand']:8.2f}x "
              f"{s['convention']['anisotropy_vs_rand']:8.2f}x "
              f"{s['b_moment']['early_late_ratio']:12.1f}x "
              f"{s['convention']['early_late_ratio']:13.1f}x")

    # Structural test
    b_anisos = [r['summary']['b_moment']['anisotropy_vs_rand'] for r in results]
    ec_anisos = [r['summary']['convention']['anisotropy_vs_rand'] for r in results]
    print(f'\nb_moment anisotropy: mean={np.mean(b_anisos):.2f}, '
          f'std={np.std(b_anisos):.2f}, range=[{min(b_anisos):.2f}, {max(b_anisos):.2f}]')
    print(f'convention anisotropy: mean={np.mean(ec_anisos):.2f}, '
          f'std={np.std(ec_anisos):.2f}, range=[{min(ec_anisos):.2f}, {max(ec_anisos):.2f}]')

    structural = all(b > 1.5 for b in b_anisos)
    temporal = all(r['summary']['b_moment']['early_late_ratio'] > 10 for r in results)
    print(f'\nSTRUCTURAL: b_moment > 1.5x for all problems? {structural}')
    print(f'TEMPORAL: early/late > 10x for all problems? {temporal}')

    out = {'model': MODEL, 'epsilon': EPS, 'T0': T0, 'problems': results}
    outpath = Path('output') / 'exp_grav_field_multi.json'
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
