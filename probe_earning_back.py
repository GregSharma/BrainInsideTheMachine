"""probe_earning_back: Does the model reconstruct what the roundtrip destroyed?

At each generation step t, the model loses 99.9% of its state through
unembedding -> argmax -> re-embedding. Do layers 0-17 at step t+1
try to recover what was lost?

Measure: cos(cumulative_delta[0..k], roundtrip_residual_from_prev_step)
for each layer k. If early layers are aligned with the residual,
the model IS earning back. If orthogonal, it's getting everything
from the KV cache and doesn't care about the lost state.

Also: cos(h_k^{t+1}, h_pre_t) -- does h converge toward the
previous step's pre-unembedding state as layers accumulate?
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'Qwen/Qwen2.5-3B'
DEVICE = 'cuda'
ALL_LAYERS = list(range(36))


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('='*70)
    print('PROBE EARNING BACK: does the model recover what the bottleneck lost?')
    print('='*70)

    print('loading...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()

    # Get embedding matrix and final norm weights
    embed_weight = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    norm_weight = model.model.norm.weight.detach().float().cpu().numpy()

    # Hooks: capture attn delta and mlp delta at each layer
    class AttnDeltaCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            # self_attn returns (attn_output, ...)
            o = output[0] if isinstance(output, tuple) else output
            self.out = o[0, -1].detach().float().cpu().numpy()  # last token, (d,)

    class MLPDeltaCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            self.out = output[0, -1].detach().float().cpu().numpy()  # last token, (d,)

    class LayerOutCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[0, -1].detach().float().cpu().numpy()  # last token, (d,)

    attn_caps = {L: AttnDeltaCap() for L in ALL_LAYERS}
    mlp_caps = {L: MLPDeltaCap() for L in ALL_LAYERS}
    layer_caps = {L: LayerOutCap() for L in ALL_LAYERS}

    hooks = []
    for L in ALL_LAYERS:
        hooks.append(model.model.layers[L].self_attn.register_forward_hook(attn_caps[L]))
        hooks.append(model.model.layers[L].mlp.register_forward_hook(mlp_caps[L]))
        hooks.append(model.model.layers[L].register_forward_hook(layer_caps[L]))

    prompts = [
        "What is 23 times 17? Think step by step.\n",
        "all the roses in my garden are red. i got a flower from my garden. must it be a rose?\n",
    ]

    for pi, prompt in enumerate(prompts):
        print(f'\n{"="*70}')
        print(f'PROMPT {pi}: "{prompt.strip()[:50]}"')
        print(f'{"="*70}')

        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        prompt_len = ids.shape[1]

        prev_h_pre = None  # h_pre from previous step (before unembedding)
        prev_residual = None  # roundtrip residual from previous step

        earning_back_data = []  # per step

        for step in range(20):
            with torch.inference_mode():
                out = model(ids)

            # Collect all layer outputs, attn deltas, mlp deltas for last token
            attn_deltas = {L: attn_caps[L].out.copy() for L in ALL_LAYERS}
            mlp_deltas = {L: mlp_caps[L].out.copy() for L in ALL_LAYERS}
            layer_outs = {L: layer_caps[L].out.copy() for L in ALL_LAYERS}

            # Compute h_pre (state before unembedding)
            h_L35 = layer_outs[35]  # (d,)
            h_normed = h_L35 / (np.sqrt(np.mean(h_L35**2) + 1e-6))
            h_pre = h_normed * norm_weight  # (d,)

            # Current token
            token_id = int(out.logits[0, -1].argmax().item())
            token_str = tokenizer.decode(token_id)
            h_post = embed_weight[token_id]  # (d,) re-embedded

            # Roundtrip residual
            residual = h_pre - h_post
            roundtrip_cos = cos(h_pre, h_post)

            # === EARNING BACK ANALYSIS ===
            if prev_residual is not None:
                # At this step, the model started from E[prev_token] and added deltas
                # Does cumulative delta converge toward prev_residual?

                cumulative = np.zeros_like(h_pre)
                earning_profile = []  # cos at each layer
                convergence_profile = []  # cos(h_k, prev_h_pre)

                h_input = embed_weight[prev_token_id]  # the starting point of THIS step

                for L in ALL_LAYERS:
                    cumulative = cumulative + attn_deltas[L] + mlp_deltas[L]
                    # cos(cumulative_correction, what_was_lost)
                    c_earn = cos(cumulative, prev_residual)
                    earning_profile.append(c_earn)

                    # cos(current_h, previous_h_pre)
                    h_current = h_input + cumulative  # reconstructed state at layer L
                    c_conv = cos(h_current, prev_h_pre)
                    convergence_profile.append(c_conv)

                earning_back_data.append({
                    'step': step,
                    'token': token_str,
                    'earning_profile': earning_profile,
                    'convergence_profile': convergence_profile,
                    'roundtrip_cos': roundtrip_cos,
                })

            # Store for next step
            prev_h_pre = h_pre.copy()
            prev_residual = residual.copy()
            prev_token_id = token_id

            # Next token
            if token_id == tokenizer.eos_token_id:
                break
            next_id = torch.tensor([[token_id]], device=DEVICE)
            ids = torch.cat([ids, next_id], dim=1)

        # === REPORT ===
        if not earning_back_data:
            print('  no generation steps')
            continue

        # Average earning profile across steps
        all_earn = np.array([d['earning_profile'] for d in earning_back_data])
        all_conv = np.array([d['convergence_profile'] for d in earning_back_data])
        mean_earn = all_earn.mean(axis=0)
        mean_conv = all_conv.mean(axis=0)

        print(f'\n  EARNING BACK: cos(cumulative_delta[0..L], prev_roundtrip_residual)')
        print(f'  (positive = model is recovering what the bottleneck destroyed)')
        print(f'  {"layer":>5s} {"earn_cos":>10s} {"converge_cos":>12s}  zone')
        for L in ALL_LAYERS:
            zone = ''
            if L < 9: zone = 'early'
            elif L < 18: zone = 'adversarial'
            elif L < 27: zone = 'cooperative'
            elif L < 33: zone = 'canyon'
            else: zone = 'readout'
            print(f'  L{L:>3d} {mean_earn[L]:>+10.4f} {mean_conv[L]:>12.4f}  {zone}')

        # Print per-step for first few
        print(f'\n  per-step earning at key layers:')
        print(f'  {"step":>4s} {"token":>10s} {"L5":>8s} {"L13":>8s} {"L17":>8s} {"L18":>8s} {"L26":>8s} {"L33":>8s} {"L35":>8s}')
        for d in earning_back_data[:10]:
            ep = d['earning_profile']
            print(f'  {d["step"]:>4d} {d["token"]:>10s}'
                  f' {ep[5]:>+8.4f} {ep[13]:>+8.4f} {ep[17]:>+8.4f}'
                  f' {ep[18]:>+8.4f} {ep[26]:>+8.4f} {ep[33]:>+8.4f} {ep[35]:>+8.4f}')

        print(f'\n  CONVERGENCE: cos(h_k, prev_h_pre)')
        print(f'  (does h at layer k look like the pre-unembedding state from prev step?)')
        print(f'  {"step":>4s} {"token":>10s} {"L0":>8s} {"L5":>8s} {"L13":>8s} {"L18":>8s} {"L26":>8s} {"L33":>8s} {"L35":>8s}')
        for d in earning_back_data[:10]:
            cp = d['convergence_profile']
            print(f'  {d["step"]:>4d} {d["token"]:>10s}'
                  f' {cp[0]:>8.4f} {cp[5]:>8.4f} {cp[13]:>8.4f}'
                  f' {cp[18]:>8.4f} {cp[26]:>8.4f} {cp[33]:>8.4f} {cp[35]:>8.4f}')

    for h in hooks:
        h.remove()

    print(f'\ndone.')


if __name__ == '__main__':
    main()
