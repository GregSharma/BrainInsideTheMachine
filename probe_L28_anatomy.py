"""L28 anatomy: what subspace did truncation remove, and why did it help?

1. Project e_c onto L28's discarded subspace
2. Project seed direction onto L28's discarded subspace
3. Test all canyon layers individually at rank 128
4. Test L1/L6 robustness on diverse prompts
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def generate(model, tok, prompt, max_tok=30):
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    tokens = []
    for _ in range(max_tok):
        with torch.inference_mode(): out = model(ids)
        nid = out.logits[0, -1].argmax().item()
        tokens.append(tok.decode(nid))
        if nid == tok.eos_token_id: break
        ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)
    return tokens

def main():
    import warnings; warnings.filterwarnings('ignore')
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # === PART 1: e_c and seed projections onto L28 discarded subspace ===
    print('='*60)
    print('PART 1: What did L28 truncation remove?')
    print('='*60)

    # Get e_c: mean difference EN - ZH activations at L28
    SYS = {'en': 'You are a careful mathematical reasoner.',
           'zh': '\u4f60\u662f\u4e00\u4e2a\u4e25\u8c28\u7684\u6570\u5b66\u63a8\u7406\u8005\u3002'}
    probs = ["Solve: 3x + 7 = 22", "Calculate: 347 + 658", "GCD of 84 and 120",
             "23 times 17", "Hypotenuse legs 5 and 12"]
    probs_zh = ["\u89e3: 3x + 7 = 22", "\u8ba1\u7b97: 347 + 658", "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570",
                "23\u4e5817", "\u659c\u8fb9 5\u548c12"]

    class HCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            self.out = h[0, -1].detach().float().cpu().numpy()

    cap28 = HCap()
    hook = model.model.layers[28].register_forward_hook(cap28)

    en_h, zh_h = [], []
    for p_en, p_zh in zip(probs, probs_zh):
        for lang, text, store in [('en', p_en, en_h), ('zh', p_zh, zh_h)]:
            msgs = [{"role": "system", "content": SYS[lang]}, {"role": "user", "content": text}]
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(t, return_tensors='pt').input_ids.to(DEV)
            with torch.inference_mode(): model(ids)
            store.append(cap28.out.copy())

    hook.remove()

    e_c = np.mean(en_h, axis=0) - np.mean(zh_h, axis=0)  # (2048,)
    e_c = e_c / (np.linalg.norm(e_c) + 1e-10)
    print(f'  e_c computed from {len(en_h)} EN + {len(zh_h)} ZH samples', flush=True)

    # SVD of W_down at every canyon layer
    for L in range(27, 36):
        W = model.model.layers[L].mlp.down_proj.weight.detach().float().cpu().numpy()  # (2048, 11008)
        # Use torch for speed
        W_t = torch.tensor(W, device=DEV)
        U, S, Vt = torch.linalg.svd(W_t, full_matrices=False)
        Vt = Vt.cpu().numpy()  # (2048, 11008) but we want the output space
        # Actually W_down is (2048, 11008). Its SVD gives U(2048,2048), S(2048), Vt(2048,11008)
        # The kept subspace in output space is U[:, :128]
        # The discarded subspace in output space is U[:, 128:]
        U_np = U.cpu().numpy()  # (2048, 2048)
        discarded = U_np[:, 128:]  # (2048, 1920) — columns spanning discarded output directions

        # Project e_c onto discarded subspace
        proj_ec = discarded.T @ e_c  # (1920,)
        ec_in_discarded = np.sum(proj_ec**2)  # ||proj||^2 (e_c is unit norm)

        # For seed: we need the seed direction at this layer
        # Use the K projection seed direction from probe_seed_position_control
        # For now, compute e_c alignment as the primary test

        print(f'  L{L}: ||e_c in discarded||² = {ec_in_discarded:.4f}  '
              f'(in kept: {1 - ec_in_discarded:.4f})  '
              f'sv128/sv1 = {S[127].item()/S[0].item():.4f}  '
              f'sv128 = {S[127].item():.4f}', flush=True)

    # === PART 2: Test all canyon layers at rank 128 ===
    print(f'\n{"="*60}')
    print('PART 2: Canyon layers individually at rank 128')
    print('='*60)

    frumble = "Every frumble in a glasshouse is transparent. Every transparent creature can pass through walls. I found a creature in a glasshouse that can pass through walls. Must it be a frumble?\n"

    baseline = generate(model, tok, frumble)
    print(f'  baseline: {"".join(baseline[:20])}...', flush=True)

    for L in range(27, 36):
        # Fresh model each time
        m2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
        m2.eval()
        with torch.no_grad():
            W = m2.model.layers[L].mlp.down_proj.weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            W_t = U[:, :128] @ torch.diag(S[:128]) @ Vt[:128, :]
            m2.model.layers[L].mlp.down_proj.weight.copy_(W_t.half())
        tokens = generate(m2, tok, frumble)
        n_match = sum(1 for a, b in zip(baseline, tokens) if a == b)
        print(f'  L{L}: {n_match}/{min(len(baseline),len(tokens))} | {"".join(tokens[:25])}...', flush=True)
        del m2; torch.cuda.empty_cache()

    # === PART 3: L1/L6 robustness on diverse prompts ===
    print(f'\n{"="*60}')
    print('PART 3: L1/L6 robustness on diverse prompts')
    print('='*60)

    diverse_prompts = [
        "What is 23 times 17? Think step by step.\n",
        "Repeat exactly: the quick brown fox jumps over the lazy dog\n",
        "Translate to French: I love mathematics\n",
        "If all cats are mammals and some mammals swim, can cats swim?\n",
        "Write a haiku about the moon\n",
    ]

    for L_test in [1, 6]:
        print(f'\n  L{L_test} at rank 128:')
        for pi, prompt in enumerate(diverse_prompts):
            # baseline
            base_tok = generate(model, tok, prompt, max_tok=20)
            # truncated
            m2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
            m2.eval()
            with torch.no_grad():
                W = m2.model.layers[L_test].mlp.down_proj.weight.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                W_t = U[:, :128] @ torch.diag(S[:128]) @ Vt[:128, :]
                m2.model.layers[L_test].mlp.down_proj.weight.copy_(W_t.half())
            trunc_tok = generate(m2, tok, prompt, max_tok=20)
            n_match = sum(1 for a, b in zip(base_tok, trunc_tok) if a == b)
            total = min(len(base_tok), len(trunc_tok))
            print(f'    P{pi}: {n_match}/{total} match | {"".join(trunc_tok[:15])}...', flush=True)
            del m2; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
