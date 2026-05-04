"""What are L1's 4 surviving directions?

1. Unembed each: top-20 tokens by (u_i @ E^T)
2. Activate each across diverse prompts
3. What gate features feed each direction
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def main():
    import warnings; warnings.filterwarnings('ignore')
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # Get L1 W_down SVD
    W_down = model.model.layers[1].mlp.down_proj.weight.detach().float()  # (2048, 11008)
    U, S, Vt = torch.linalg.svd(W_down, full_matrices=False)
    # Top 4 output directions
    u = U[:, :4].cpu().numpy()  # (2048, 4)
    s = S[:4].cpu().numpy()
    print(f'L1 W_down top-4 singular values: {s}')
    print(f'  sv4/sv1 = {s[3]/s[0]:.4f}')

    # === 1. UNEMBED EACH DIRECTION ===
    print(f'\n{"="*60}')
    print('1. UNEMBED: what tokens does each direction want to write?')
    print(f'{"="*60}')

    E = model.model.embed_tokens.weight.detach().float().cpu().numpy()  # (vocab, 2048)

    for i in range(4):
        logits = E @ u[:, i]  # (vocab,)
        top_idx = np.argsort(logits)[-20:][::-1]
        bot_idx = np.argsort(logits)[:10]
        top_tokens = [(tok.decode(idx), float(logits[idx])) for idx in top_idx]
        bot_tokens = [(tok.decode(idx), float(logits[idx])) for idx in bot_idx]
        print(f'\n  direction {i} (sv={s[i]:.4f}):')
        print(f'    TOP 20: {[t[0] for t in top_tokens]}')
        print(f'    scores: {[f"{t[1]:.3f}" for t in top_tokens]}')
        print(f'    BOT 10: {[t[0] for t in bot_tokens]}')

    # === 2. ACTIVATE ACROSS DIVERSE PROMPTS ===
    print(f'\n{"="*60}')
    print('2. ACTIVATION: how does each direction respond to different prompts?')
    print(f'{"="*60}')

    class HCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            self.out = h[0, -1].detach().float().cpu().numpy()

    cap = HCap()
    hook = model.model.layers[1].register_forward_hook(cap)

    prompts = [
        ('math_en', 'Solve: 3x + 7 = 22'),
        ('math_zh', '\u89e3: 3x + 7 = 22'),
        ('arith', 'Calculate: 347 + 658'),
        ('repeat', 'Repeat: the quick brown fox'),
        ('logic', 'All cats are mammals. Some mammals swim. Can cats swim?'),
        ('frumble', 'Every frumble in a glasshouse is transparent. Must a transparent creature be a frumble?'),
        ('translate', 'Translate to French: I love math'),
        ('casual', 'yo whats 23 times 17'),
        ('formal', 'Please calculate the product of 23 and 17.'),
        ('sys_math', None),  # with system prompt
        ('code', 'Write a Python function to compute factorial'),
        ('poem', 'Write a haiku about the moon'),
        ('nonsense', 'colorless green ideas sleep furiously'),
        ('numbers', '1 2 3 4 5 6 7 8 9 10'),
    ]

    print(f'  {"prompt":>15s}  {"d0":>8s} {"d1":>8s} {"d2":>8s} {"d3":>8s}')

    for pname, ptext in prompts:
        if pname == 'sys_math':
            msgs = [{'role': 'system', 'content': 'You are a math expert.'},
                    {'role': 'user', 'content': 'Solve: 3x + 7 = 22'}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors='pt').input_ids.to(DEV)
        else:
            ids = tok(ptext, return_tensors='pt').input_ids.to(DEV)

        with torch.inference_mode(): model(ids)
        h = cap.out  # (2048,)

        # Project onto 4 directions
        projs = [float(h @ u[:, i]) for i in range(4)]
        print(f'  {pname:>15s}  {projs[0]:>+8.2f} {projs[1]:>+8.2f} {projs[2]:>+8.2f} {projs[3]:>+8.2f}')

    hook.remove()

    # === 3. WHAT GATE FEATURES FEED EACH DIRECTION ===
    print(f'\n{"="*60}')
    print('3. GATE FEATURES: which intermediate neurons contribute to each direction?')
    print(f'{"="*60}')

    W_gate = model.model.layers[1].mlp.gate_proj.weight.detach().float().cpu().numpy()  # (11008, 2048)
    W_up = model.model.layers[1].mlp.up_proj.weight.detach().float().cpu().numpy()      # (11008, 2048)

    # The top-4 right singular vectors of W_down tell us which intermediate
    # features contribute most to each output direction.
    # Vt[:4, :] are (4, 11008) — the intermediate-space directions
    Vt_top = Vt[:4, :].cpu().numpy()  # (4, 11008)

    for i in range(4):
        v = Vt_top[i]  # (11008,) — which intermediate neurons feed direction i
        top_neurons = np.argsort(np.abs(v))[-10:][::-1]
        print(f'\n  direction {i} (sv={s[i]:.4f}):')
        print(f'    top intermediate neurons: {top_neurons}')
        print(f'    weights: {[f"{v[n]:+.4f}" for n in top_neurons]}')

        # For each top neuron, what does W_gate select for?
        # W_gate[neuron, :] is the gate weight for that neuron
        # Project onto embedding space to see what tokens activate it
        for n in top_neurons[:3]:  # top 3 only
            gate_dir = W_gate[n]  # (2048,)
            up_dir = W_up[n]      # (2048,)
            # What tokens activate this gate neuron?
            gate_logits = E @ gate_dir  # (vocab,)
            top_gate = np.argsort(gate_logits)[-5:][::-1]
            gate_tokens = [tok.decode(idx) for idx in top_gate]
            print(f'      neuron {n}: gate activates on {gate_tokens}')


if __name__ == '__main__':
    main()
