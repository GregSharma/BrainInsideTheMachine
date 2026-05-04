"""exp_svd_convergence: Does the KV cache SVD converge toward b_moment during the loop?

The hypothesis: blind SVD deflation works because the cache's top SVD direction
converges toward the answer direction (b_moment) during autoregressive generation.
The model echoes its own computation, the computation contains the answer (B at rank 90),
so the echo direction IS the answer direction. Deflating along it forces the query
to read something OTHER than the answer-that-can't-be-expressed.

Measure: at each generation step, compute SVD of the KV cache at key layers.
Track cos(svd_top, b_moment), cos(svd_top, e_c), cos(svd_top, random).
Compare baseline (loop) vs deflated (correct).
"""
import json, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
N_LAYERS = 36
D_MODEL = 2048
DEVICE = 'cuda'
MAX_TOKENS = 200  # enough to see convergence pattern

SYS = ('You are solving an AMC 12A multiple choice math problem. Think step by step, '
       'show your work, then clearly state your final answer as (A), (B), (C), (D), or (E).')
P12_TEXT = (
    "The harmonic mean of a collection of numbers is the reciprocal of the "
    "arithmetic mean of the reciprocals of the numbers in the collection. "
    "For example, the harmonic mean of 4, 4, and 5 is\n\n"
    "1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n"
    "What is the harmonic mean of all the real roots of the 4050th degree "
    "polynomial\n\n"
    r"\prod_{k=1}^{2025} (kx^2 - 4x - 3) = "
    "(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)..."
    "(2025x^2 - 4x - 3)?\n\n"
    "(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3"
)
PROMPT = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"

MATH_SYS = 'You are a careful mathematical reasoner. Think step by step, show your work clearly, and then state the final numerical answer on its own line.'

from exp_delayed_deflation_p12 import WindowedDeflation


def extract_b_moment(model, tokenizer):
    """Extract b_moment direction: hidden state at L33 when deflated run produces B.

    We use the template matching approach: run deflated generation, capture h_L33
    at the step where B-fraction logit is highest.

    Simpler approach: just capture the mean h_L33 over first 50 deflated tokens.
    The b_moment is the average computation direction during the priming window.
    """
    deflator = WindowedDeflation(model, layers=list(range(20, 36)), r=4, alpha=0.1,
                                  refresh_every=25, active_from=0, active_until=None)

    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
    past_kv = None
    h_L33_accum = []

    class L33Capture:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == 1:  # generation mode
                self.out = h[0, 0].detach().float().cpu()

    cap = L33Capture()
    hook = model.model.layers[33].register_forward_hook(cap)
    deflator.start_gen()

    for step in range(50):  # first 50 tokens only
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            if cap.out is not None:
                h_L33_accum.append(cap.out.numpy())
            deflator.tick(past_kv)

    hook.remove()
    deflator.remove()

    b_moment = np.mean(h_L33_accum, axis=0)
    b_moment = b_moment / (np.linalg.norm(b_moment) + 1e-12)
    return b_moment


def extract_convention_direction(model, tokenizer):
    """Extract e_c from encoding activations."""
    from exp_convention_qk_deflation import get_test_problems
    problems = get_test_problems()

    en_acts = []
    zh_acts = []

    class Cap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[:, -1, :].detach().float().cpu().numpy()

    cap = Cap()
    hook = model.model.layers[33].register_forward_hook(cap)

    for prob in problems:
        for lang, store in [('en', en_acts), ('zh', zh_acts)]:
            messages = [{"role": "system", "content": MATH_SYS}, {"role": "user", "content": prob[lang]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
            with torch.inference_mode():
                model(ids)
            store.append(cap.out.squeeze())

    hook.remove()

    en_mean = np.mean(en_acts, axis=0)
    zh_mean = np.mean(zh_acts, axis=0)
    diff = zh_mean - en_mean
    e_c = diff / (np.linalg.norm(diff) + 1e-12)
    return e_c


def run_with_svd_tracking(model, tokenizer, deflator=None, sample_layers=None, sample_every=5):
    """Generate P12 while tracking KV cache SVD at sampled layers/steps."""
    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None

    if deflator:
        deflator.start_gen()

    # svd_dirs[layer][step] = top SVD direction of K cache (head_dim per KV head, flattened)
    svd_dirs = {L: [] for L in sample_layers}
    svd_steps = []

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                if deflator:
                    deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            if deflator:
                deflator.tick(past_kv)

            # Sample SVD at intervals
            if step > 0 and step % sample_every == 0:
                svd_steps.append(step)
                for L in sample_layers:
                    keys = past_kv.layers[L].keys  # (batch, n_kv_heads, seq, head_dim)
                    # Concatenate all KV heads into one matrix for a global SVD
                    # Shape: (n_kv_heads * seq, head_dim) -> but we want residual-stream-like
                    # Actually: each KV head has keys of shape (seq, head_dim=128)
                    # We want the top direction of the full key matrix
                    all_keys = keys[0]  # (n_kv_heads, seq, head_dim)
                    n_kv, seq_len, hd = all_keys.shape
                    # Reshape to (seq, n_kv * head_dim) to get one direction per step
                    K_mat = all_keys.permute(1, 0, 2).reshape(seq_len, n_kv * hd).float()
                    # SVD
                    U, S, Vh = torch.linalg.svd(K_mat, full_matrices=False)
                    top_dir = Vh[0].cpu().numpy()  # (n_kv * head_dim,)
                    svd_dirs[L].append(top_dir)

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, len(gen_ids), svd_dirs, svd_steps


def project_to_kv_space(direction_residual, model, layer):
    """Project a residual-stream direction into the KV key space.

    Keys are computed as x @ W_K. So direction in key space = W_K @ direction.
    For GQA: n_kv_heads key projections, each head_dim=128.
    W_K shape: (n_kv_heads * head_dim, d_model)
    """
    W_K = model.model.layers[layer].self_attn.k_proj.weight.data  # (n_kv*hd, d_model)
    d_residual = torch.tensor(direction_residual, dtype=torch.float16, device=DEVICE)
    d_key = (W_K @ d_residual).float().cpu().numpy()  # (n_kv * head_dim,)
    d_key = d_key / (np.linalg.norm(d_key) + 1e-12)
    return d_key


def main():
    print('=' * 60)
    print('Exp SVD Convergence: Does the echo become the answer?')
    print('=' * 60)

    sample_layers = [20, 25, 27, 30, 33]

    print(f'\nLoading {MODEL_NAME}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print('  Loaded.', flush=True)

    # Extract reference directions
    print('\nExtracting b_moment (mean deflated h_L33, first 50 tokens)...', flush=True)
    b_moment = extract_b_moment(model, tokenizer)
    print(f'  b_moment norm (pre-normalize): {np.linalg.norm(b_moment):.4f}', flush=True)

    print('Extracting convention direction e_c at L33...', flush=True)
    e_c = extract_convention_direction(model, tokenizer)
    print(f'  cos(b_moment, e_c) = {np.dot(b_moment, e_c):.4f}', flush=True)

    # Random control
    rng = np.random.RandomState(42)
    e_rand = rng.randn(D_MODEL).astype(np.float32)
    e_rand = e_rand / np.linalg.norm(e_rand)

    # Project reference directions into KV key space for each layer
    print('\nProjecting directions into KV key space...', flush=True)
    b_key = {}
    ec_key = {}
    rand_key = {}
    for L in sample_layers:
        b_key[L] = project_to_kv_space(b_moment, model, L)
        ec_key[L] = project_to_kv_space(e_c, model, L)
        rand_key[L] = project_to_kv_space(e_rand, model, L)
        print(f'  L{L}: cos(b_key, ec_key) = {np.dot(b_key[L], ec_key[L]):.4f}')

    # === BASELINE (loop) ===
    print('\n--- Baseline generation (loop expected) ---', flush=True)
    t0 = time.time()
    text_bl, ntok_bl, svd_bl, steps_bl = run_with_svd_tracking(
        model, tokenizer, sample_layers=sample_layers, sample_every=5)
    t_bl = time.time() - t0
    print(f'  {ntok_bl} tokens, {t_bl:.1f}s', flush=True)

    # === DEFLATED (correct) ===
    print('\n--- Deflated generation (correct expected) ---', flush=True)
    deflator = WindowedDeflation(model, layers=list(range(20, 36)), r=4, alpha=0.1,
                                  refresh_every=25, active_from=0, active_until=None)
    t0 = time.time()
    text_df, ntok_df, svd_df, steps_df = run_with_svd_tracking(
        model, tokenizer, deflator=deflator, sample_layers=sample_layers, sample_every=5)
    t_df = time.time() - t0
    deflator.remove()
    print(f'  {ntok_df} tokens, {t_df:.1f}s', flush=True)

    # === COMPUTE COSINES ===
    print(f'\n{"=" * 70}')
    print('SVD TOP DIRECTION vs REFERENCE DIRECTIONS')
    print(f'{"=" * 70}')

    results = {'baseline': {}, 'deflated': {}}

    for label, svd_data, steps in [('baseline', svd_bl, steps_bl), ('deflated', svd_df, steps_df)]:
        print(f'\n--- {label} ---')
        for L in sample_layers:
            cos_b = []
            cos_ec = []
            cos_rand = []
            for i, svd_dir in enumerate(svd_data[L]):
                cos_b.append(float(abs(np.dot(svd_dir, b_key[L]))))
                cos_ec.append(float(abs(np.dot(svd_dir, ec_key[L]))))
                cos_rand.append(float(abs(np.dot(svd_dir, rand_key[L]))))

            results[label][str(L)] = {
                'cos_b_moment': cos_b,
                'cos_convention': cos_ec,
                'cos_random': cos_rand,
                'steps': [int(s) for s in steps],
            }

            # Print trajectory
            if len(cos_b) >= 4:
                early = np.mean(cos_b[:len(cos_b)//4])
                late = np.mean(cos_b[-len(cos_b)//4:])
                ec_early = np.mean(cos_ec[:len(cos_ec)//4])
                ec_late = np.mean(cos_ec[-len(cos_ec)//4:])
                print(f'  L{L:2d} b_moment: early={early:.4f} late={late:.4f} (delta={late-early:+.4f})')
                print(f'       e_c:      early={ec_early:.4f} late={ec_late:.4f} (delta={ec_late-ec_early:+.4f})')

    # Save
    output = {
        'b_moment_ec_cos': float(np.dot(b_moment, e_c)),
        'baseline_tokens': ntok_bl,
        'deflated_tokens': ntok_df,
        'sample_layers': sample_layers,
        'results': results,
    }
    outpath = OUTPUT_DIR / 'exp_svd_convergence.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
