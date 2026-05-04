"""quick and dirty. stop overthinking.

Hit 1: does L30 rank-1 direction live in the 9D f* subspace?
Hit 2: alpha correction -- inject previous step's MLP delta, see if output changes.
Hit 3: seed direction -- what IS the classifier's discriminant axis?
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    import warnings; warnings.filterwarnings('ignore')
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # ============================================================
    # HIT 1: L30 rank-1 vs f* subspace
    # f* was the classifier weight matrix SVD from exp_z_encoder
    # L30 rank-1 was the top SV of MLP output during generation
    # Are they the same object at different layers?
    # ============================================================
    print('='*60)
    print('HIT 1: L30 MLP rank-1 direction vs f* subspace')
    print('='*60)

    # We need to extract L30 MLP output direction and L33 f* basis
    # Quick: encode 5 problems, get MLP output at L30, SVD
    problems = [
        "Solve for x: 3x + 7 = 22",
        "Calculate: 347 + 658",
        "Find the hypotenuse of a right triangle with legs 5 and 12",
        "What is the GCD of 84 and 120?",
        "How many ways can you choose 3 items from 7?",
    ]
    sys_prompt = "You are a careful mathematical reasoner. Think step by step."

    class MLPOutCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o): self.out = o[0, -1].detach().float().cpu().numpy()

    class HCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            self.out = h[0, -1].detach().float().cpu().numpy()

    mlp30 = MLPOutCap(); h33 = HCap()
    hook1 = model.model.layers[30].mlp.register_forward_hook(mlp30)
    hook2 = model.model.layers[33].register_forward_hook(h33)

    mlp_outs = []; h33_vecs = []
    for p in problems:
        for lang_wrap in [p, f"\u89e3\u65b9\u7a0b\uff1a3x + 7 = 22" if "3x" in p else p]:  # en + zh for first
            msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": lang_wrap}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors='pt').input_ids.to(DEV)
            with torch.inference_mode(): model(ids)
            mlp_outs.append(mlp30.out.copy())
            h33_vecs.append(h33.out.copy())

    hook1.remove(); hook2.remove()

    # SVD of MLP outputs at L30
    M = np.stack(mlp_outs)
    M_c = M - M.mean(axis=0)
    U, S, Vt = np.linalg.svd(M_c, full_matrices=False)
    v1_l30 = Vt[0]  # top direction
    print(f'  L30 MLP SVD: sv1={S[0]:.2f}, sv2={S[1]:.2f}, ratio={S[0]/S[1]:.2f}')

    # SVD of h at L33 (proxy for f*)
    H = np.stack(h33_vecs)
    H_c = H - H.mean(axis=0)
    Uh, Sh, Vth = np.linalg.svd(H_c, full_matrices=False)
    print(f'  L33 h SVD: sv1={Sh[0]:.2f}, sv2={Sh[1]:.2f}')

    # Cosine between L30 v1 and L33 top directions
    for k in range(min(5, len(Vth))):
        c = cos(v1_l30, Vth[k])
        print(f'  cos(L30_v1, L33_v{k+1}) = {c:.4f}')

    # Also: cos between L30 v1 directions and L33 top directions
    print(f'  subspace overlap (top-3 L30 vs top-9 L33):')
    overlap = 0
    for i in range(min(3, len(Vt))):
        for j in range(min(9, len(Vth))):
            overlap += cos(Vt[i], Vth[j])**2
    print(f'  Σcos² = {overlap:.4f} (max possible = {min(3,len(Vt)):.0f})')

    # ============================================================
    # HIT 2: alpha correction
    # inject prev step's MLP delta into current step
    # simplest test: does the output TOKEN change?
    # ============================================================
    print(f'\n{"="*60}')
    print('HIT 2: alpha correction -- does injecting prev MLP delta change output?')
    print('='*60)

    prompt = "What is 23 times 17? Let me think step by step.\n"

    # First: generate baseline (no correction)
    ids_base = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    baseline_tokens = []
    for step in range(15):
        with torch.inference_mode():
            out = model(ids_base)
        next_id = out.logits[0, -1].argmax().item()
        baseline_tokens.append(tok.decode(next_id))
        if next_id == tok.eos_token_id: break
        ids_base = torch.cat([ids_base, torch.tensor([[next_id]], device=DEV)], dim=1)

    print(f'  baseline: {baseline_tokens}')

    # Now: generate with alpha correction at L18
    # At each step, after L18, add alpha * (prev step's L18 MLP delta)
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        # Need to capture MLP delta at L18
        mlp18 = MLPOutCap()
        hook = model.model.layers[18].mlp.register_forward_hook(mlp18)

        ids_corr = tok(prompt, return_tensors='pt').input_ids.to(DEV)
        corrected_tokens = []
        prev_mlp_delta = None

        # We need a hook that MODIFIES the hidden state after L18
        # Can't easily do this with output hooks on mlp (would need to modify in place)
        # Simpler: hook the LAYER output and add correction
        class CorrectionHook:
            def __init__(self, alpha):
                self.alpha = alpha
                self.prev_delta = None
                self.current_delta = None
            def __call__(self, module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if self.prev_delta is not None:
                    correction = torch.tensor(self.prev_delta, dtype=h.dtype, device=h.device)
                    h[0, -1] += self.alpha * correction
                return (h,) + output[1:] if isinstance(output, tuple) else h

        hook.remove()

        # Set up: capture MLP delta at L18, apply correction at L18 output
        corr_hook = CorrectionHook(alpha)
        mlp18 = MLPOutCap()
        h_mlp = model.model.layers[18].mlp.register_forward_hook(mlp18)
        h_layer = model.model.layers[18].register_forward_hook(corr_hook)

        for step in range(15):
            with torch.inference_mode():
                out = model(ids_corr)

            next_id = out.logits[0, -1].argmax().item()
            corrected_tokens.append(tok.decode(next_id))

            # Store this step's MLP delta for next step's correction
            corr_hook.prev_delta = mlp18.out.copy()

            if next_id == tok.eos_token_id: break
            ids_corr = torch.cat([ids_corr, torch.tensor([[next_id]], device=DEV)], dim=1)

        h_mlp.remove()
        h_layer.remove()

        n_diff = sum(1 for a, b in zip(baseline_tokens, corrected_tokens) if a != b)
        print(f'  alpha={alpha}: {corrected_tokens}  ({n_diff}/{min(len(baseline_tokens),len(corrected_tokens))} tokens differ)')

    # ============================================================
    # HIT 3: what does the KV cache actually contain per position?
    # total info: 36 layers x 2 (K+V) x 2 kv_heads x 128 dim = 18432 dims
    # is that overcomplete? or is it redundant?
    # ============================================================
    print(f'\n{"="*60}')
    print('HIT 3: KV cache information content')
    print('='*60)

    # Encode a prompt, extract ALL K vectors at ALL layers for each position
    # Then SVD the full KV representation: can it reconstruct h?
    prompt2 = "all the roses in my garden are red"
    ids2 = tok(prompt2, return_tensors='pt').input_ids.to(DEV)
    n_tok = ids2.shape[1]

    k_caps_all = {L: MLPOutCap() for L in range(36)}  # reusing class, just captures output
    k_hooks = [model.model.layers[L].self_attn.k_proj.register_forward_hook(k_caps_all[L]) for L in range(36)]
    h_caps_all = {L: HCap() for L in range(36)}
    h_hooks = [model.model.layers[L].register_forward_hook(h_caps_all[L]) for L in range(36)]

    with torch.inference_mode(): model(ids2)

    for h in k_hooks + h_hooks: h.remove()

    # For each position, concatenate K vectors across all 36 layers
    # K at each layer: (seq, 256) but we captured last token only... shit.
    # The MLPOutCap captures output which for k_proj is (batch, seq, kv_dim)
    # Let me check the shape
    k0 = k_caps_all[0].out
    print(f'  K shape at L0: {k0.shape}')

    if len(k0.shape) == 2:  # (seq, kv_dim)
        kv_dim = k0.shape[1]
    elif len(k0.shape) == 1:  # just last token
        kv_dim = k0.shape[0]
        print(f'  (only captured last token, need seq-level for full analysis)')
        print(f'  KV dim per layer: {kv_dim}')
        print(f'  total KV dims across 36 layers: {36 * kv_dim * 2} (K+V)')
        print(f'  vs hidden dim: 2048')
        print(f'  overcomplete ratio: {36 * kv_dim * 2 / 2048:.1f}x')
    else:
        kv_dim = k0.shape[-1]

    print(f'\n  KV cache per position: 36 layers x {kv_dim}D (K) + 36 x {kv_dim}D (V) = {36*kv_dim*2}D')
    print(f'  hidden state: 2048D')
    print(f'  ratio: {36*kv_dim*2/2048:.1f}x overcomplete')
    print(f'  the KV cache stores {36*kv_dim*2/2048:.1f}x more information per position than h')


if __name__ == '__main__':
    main()
