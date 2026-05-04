"""Per-head decomposition of gravitational field at L34-L35.

The full-forward perturbation at L33 showed 5.4x anisotropy in b_moment direction.
This experiment decomposes: which attention heads at L34 and L35 carry the signal?

Qwen2.5-3B: 16 heads, head_dim=128, n_kv_heads=2 (GQA: 8 Q heads share 1 KV head).

Approach: for each head at L34 and L35, ablate that single head's output
(zero it out) and measure how much the anisotropy drops. If 2-3 heads
carry most of the 5.4x, those heads ARE the gravitational field.

Alternative (complementary) approach: keep only one head active at a time
and measure how much anisotropy it produces alone.
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
DEV = 'cuda'
D = 2048
N_HEADS = 16
HEAD_DIM = 128
N_KV_HEADS = 2  # GQA
L_TARGET = 33  # perturbation layer
EPS = 0.5
MAX_GEN = 150  # enough to capture the early window
T0 = 50

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


def extract_b_moment_and_convention(model, tok):
    """Extract the two key directions at L33."""
    from exp_gravitational_field_p12 import (
        extract_b_moment_template,
        extract_convention_direction,
    )
    v_b, b_step = extract_b_moment_template(model, tok)
    e_c = extract_convention_direction(model, tok)
    return v_b, e_c


def make_prompt(tok):
    msgs = [{'role': 'system', 'content': SYS},
            {'role': 'user', 'content': P12}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def measure_perhead_sensitivity(model, tok, prompt, directions, ablate_layer, ablate_head):
    """Measure sensitivity with one attention head ablated at ablate_layer.

    Ablation: zero out the output of head `ablate_head` in the self_attn
    output projection. We hook into attn output and zero the head's slice.

    Returns per-direction mean KL over early window (<T0 tokens).
    """
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)

    dir_t = {n: torch.tensor(v, dtype=torch.float16, device=DEV)
             for n, v in directions.items()}

    # Hook to ablate one head's contribution
    # Qwen2's self_attn forward: attn_output goes through o_proj.
    # We hook AFTER the full layer (including MLP) and zero the head's
    # attn contribution. But that's entangled with MLP.
    #
    # Better: hook the o_proj input, zero the head's slice before projection.
    # o_proj input shape: (batch, seq, n_heads * head_dim) = (1, seq, 2048)
    # Head h occupies [h*128 : (h+1)*128]
    ablate_active = {'on': False}

    def hook_ablate_attn(module, inp, output):
        """Hook on self_attn module. Output is (attn_output, attn_weights, past_kv).
        attn_output is already through o_proj. We need to hook BEFORE o_proj.
        """
        # This hooks the whole self_attn module output.
        # attn_output = output[0], shape (batch, seq, d_model)
        # We can't cleanly zero one head here because o_proj mixed them.
        # Instead, we'll use a different strategy: hook o_proj directly.
        return output

    def make_oproj_hook(head_idx):
        """Hook on o_proj's input. Zero head_idx's slice before projection."""
        def hook(module, inp):
            if not ablate_active['on']:
                return inp
            x = inp[0]  # (batch, seq, n_heads * head_dim)
            x2 = x.clone()
            start = head_idx * HEAD_DIM
            end = start + HEAD_DIM
            x2[:, -1, start:end] = 0  # zero this head at last token only
            return (x2,) + inp[1:] if len(inp) > 1 else (x2,)
        return hook

    # Perturbation hook at L_TARGET
    pert_v = {'v': None}
    def hook_pert(m, i, o):
        if pert_v['v'] is None: return o
        h = o[0] if isinstance(o, tuple) else o
        if h.dim() == 3:
            h2 = h.clone(); h2[:, -1, :] += pert_v['v']
            return (h2,) + o[1:] if isinstance(o, tuple) else h2
        return h + pert_v['v']

    # Install ablation hook on the target layer's o_proj
    layer_module = model.model.layers[ablate_layer]
    oproj = layer_module.self_attn.o_proj
    if ablate_head is not None:
        hk_ablate = oproj.register_forward_pre_hook(make_oproj_hook(ablate_head))
        ablate_active['on'] = True
    else:
        hk_ablate = None

    sens = {n: [] for n in directions}

    with torch.no_grad():
        for step in range(MAX_GEN):
            if step == 0:
                out = model(input_ids=ids, use_cache=True)
            else:
                out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            logits = out.logits[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            if nxt.item() in (151643, 151645, 0): break

            log_p0 = F.log_softmax(logits.float(), dim=-1)
            p0 = log_p0.exp()

            # Perturbed passes
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

    if hk_ablate is not None:
        hk_ablate.remove()
    ablate_active['on'] = False

    del kv, out; torch.cuda.empty_cache()

    # Compute early-window means
    result = {}
    for name, trace in sens.items():
        early = trace[:T0] if len(trace) >= T0 else trace
        result[name] = float(np.mean(early)) if early else 0.0
    return result, {n: [float(x) for x in v] for n, v in sens.items()}


def main():
    print('=' * 60)
    print('PER-HEAD DECOMPOSITION OF GRAVITATIONAL FIELD')
    print('Which heads at L34/L35 carry the anisotropy?')
    print('=' * 60, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    prompt = make_prompt(tok)
    print(f'Prompt: {len(tok(prompt).input_ids)} tokens', flush=True)

    # Extract directions
    print('\nExtracting directions...', flush=True)
    v_b, e_c = extract_b_moment_and_convention(model, tok)
    rng = np.random.RandomState(42)
    r0 = rng.randn(D).astype(np.float32); r0 /= np.linalg.norm(r0)
    r1 = rng.randn(D).astype(np.float32); r1 /= np.linalg.norm(r1)
    r2 = rng.randn(D).astype(np.float32); r2 /= np.linalg.norm(r2)

    directions = {
        'b_moment': v_b,
        'convention': e_c,
        'rand_0': r0,
        'rand_1': r1,
        'rand_2': r2,
    }

    results = {'meta': {'model': MODEL_NAME, 'eps': EPS, 'max_gen': MAX_GEN,
                        'n_heads': N_HEADS, 'head_dim': HEAD_DIM}}

    # 1. Baseline: no ablation
    print('\n--- Baseline (no ablation) ---', flush=True)
    t0 = time.time()
    base_early, base_traces = measure_perhead_sensitivity(
        model, tok, prompt, directions, ablate_layer=34, ablate_head=None)
    dt = round(time.time() - t0, 1)
    rand_mean = np.mean([base_early[f'rand_{i}'] for i in range(3)])
    b_aniso = base_early['b_moment'] / (rand_mean + 1e-12)
    ec_aniso = base_early['convention'] / (rand_mean + 1e-12)
    print(f'  b_moment aniso = {b_aniso:.2f}x, convention = {ec_aniso:.2f}x ({dt}s)',
          flush=True)
    results['baseline'] = {
        'early_kl': base_early,
        'b_aniso': float(b_aniso),
        'ec_aniso': float(ec_aniso),
        'time_s': dt,
    }

    # 2. Per-head ablation at L34
    print('\n--- Per-head ablation at L34 ---', flush=True)
    results['L34_ablation'] = {}
    for h in range(N_HEADS):
        t0 = time.time()
        early, traces = measure_perhead_sensitivity(
            model, tok, prompt, directions, ablate_layer=34, ablate_head=h)
        dt = round(time.time() - t0, 1)
        rm = np.mean([early[f'rand_{i}'] for i in range(3)])
        ba = early['b_moment'] / (rm + 1e-12)
        ea = early['convention'] / (rm + 1e-12)
        drop_b = (b_aniso - ba) / b_aniso * 100  # % of anisotropy lost
        print(f'  Head {h:2d}: b={ba:.2f}x (drop {drop_b:+.1f}%), '
              f'ec={ea:.2f}x ({dt}s)', flush=True)
        results['L34_ablation'][f'head_{h}'] = {
            'early_kl': early,
            'b_aniso': float(ba),
            'ec_aniso': float(ea),
            'b_drop_pct': float(drop_b),
            'time_s': dt,
        }

    # 3. Per-head ablation at L35
    print('\n--- Per-head ablation at L35 ---', flush=True)
    results['L35_ablation'] = {}
    for h in range(N_HEADS):
        t0 = time.time()
        early, traces = measure_perhead_sensitivity(
            model, tok, prompt, directions, ablate_layer=35, ablate_head=h)
        dt = round(time.time() - t0, 1)
        rm = np.mean([early[f'rand_{i}'] for i in range(3)])
        ba = early['b_moment'] / (rm + 1e-12)
        ea = early['convention'] / (rm + 1e-12)
        drop_b = (b_aniso - ba) / b_aniso * 100
        print(f'  Head {h:2d}: b={ba:.2f}x (drop {drop_b:+.1f}%), '
              f'ec={ea:.2f}x ({dt}s)', flush=True)
        results['L35_ablation'][f'head_{h}'] = {
            'early_kl': early,
            'b_aniso': float(ba),
            'ec_aniso': float(ea),
            'b_drop_pct': float(drop_b),
            'time_s': dt,
        }

    # Summary: rank heads by contribution
    print(f'\n{"=" * 60}')
    print('HEAD RANKING BY ANISOTROPY CONTRIBUTION')
    print(f'{"=" * 60}')
    for layer_name in ['L34_ablation', 'L35_ablation']:
        print(f'\n{layer_name}:')
        heads = []
        for h in range(N_HEADS):
            drop = results[layer_name][f'head_{h}']['b_drop_pct']
            heads.append((h, drop))
        heads.sort(key=lambda x: -x[1])  # highest drop first
        for h, drop in heads:
            bar = '#' * max(0, int(drop / 2))
            print(f'  Head {h:2d}: {drop:+6.1f}% {bar}')

    # Save
    outpath = Path('output') / 'exp_perhead_decomp.json'
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
