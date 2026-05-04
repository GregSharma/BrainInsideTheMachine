"""Spectrum-as-nuisance-storage: the unifying test.

1. Seed classifier on L28-truncated model (does truncation change self-attribution?)
2. Project non-convention nuisance directions onto L28's discarded subspace
3. L1 causal rank sweep (how low can we go?)

All three in one script, one model load.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

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

    # Save weights for L28, L1
    orig_L28 = model.model.layers[28].mlp.down_proj.weight.data.clone()
    orig_L1 = model.model.layers[1].mlp.down_proj.weight.data.clone()

    # === EXP 1: SEED CLASSIFIER ON L28-TRUNCATED MODEL ===
    print('='*60)
    print('EXP 1: Does L28 truncation change the seed?')
    print('='*60)

    prompt = "Every frumble in a glasshouse is transparent. Every transparent creature can pass through walls. I found a creature in a glasshouse that can pass through walls. Must it be a frumble? Explain step by step.\n"

    # Capture K at L26 during generation: baseline vs L28-truncated
    class KCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o): self.out = o[0, -1].detach().float().cpu().numpy()

    for condition in ['baseline', 'L28_trunc']:
        if condition == 'L28_trunc':
            with torch.no_grad():
                W = model.model.layers[28].mlp.down_proj.weight.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                W_t = U[:, :128] @ torch.diag(S[:128]) @ Vt[:128, :]
                model.model.layers[28].mlp.down_proj.weight.copy_(W_t.half())

        cap = KCap()
        hook = model.model.layers[26].self_attn.k_proj.register_forward_hook(cap)

        ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
        prompt_len = ids.shape[1]
        prompt_K = []; gen_K = []; gen_tokens = []

        for step in range(40):
            with torch.inference_mode(): out = model(ids)
            if step == 0:
                # Actually need all positions for prompt K
                # KCap only gets last... use a different hook
                pass
            gen_K.append(cap.out.copy())
            nid = out.logits[0, -1].argmax().item()
            gen_tokens.append(tok.decode(nid))
            if nid == tok.eos_token_id: break
            ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)

        hook.remove()

        # For seed classifier we need prompt K too. Quick: encode prompt alone
        class AllKCap:
            def __init__(self): self.out = None
            def __call__(self, m, i, o): self.out = o[0].detach().float().cpu().numpy()

        acap = AllKCap()
        ahook = model.model.layers[26].self_attn.k_proj.register_forward_hook(acap)
        ids_p = tok(prompt, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids_p)
        prompt_K_all = acap.out.copy()  # (prompt_len, kv_dim)
        ahook.remove()

        # Classifier
        X = np.vstack([prompt_K_all, np.stack(gen_K)])
        y = np.array([0]*prompt_K_all.shape[0] + [1]*len(gen_K))
        clf = LogisticRegression(max_iter=2000, C=1.0)
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X))
        split = int(0.8 * len(idx))
        clf.fit(X[idx[:split]], y[idx[:split]])
        acc = accuracy_score(y[idx[split:]], clf.predict(X[idx[split:]]))

        print(f'  {condition}: seed accuracy = {acc:.3f} ({len(prompt_K_all)}p + {len(gen_K)}g)', flush=True)
        print(f'    output: {"".join(gen_tokens[:20])}...', flush=True)

        # Restore
        if condition == 'L28_trunc':
            model.model.layers[28].mlp.down_proj.weight.data.copy_(orig_L28)

    # === EXP 2: NUISANCE PROJECTIONS ONTO L28 DISCARDED SUBSPACE ===
    print(f'\n{"="*60}')
    print('EXP 2: What else lives in the bottom spectrum of W_down?')
    print('='*60)

    # Get L28 discarded subspace
    W28 = model.model.layers[28].mlp.down_proj.weight.detach().float()
    U28, S28, _ = torch.linalg.svd(W28, full_matrices=False)
    discarded = U28[:, 128:].cpu().numpy()  # (2048, 1920)

    # Compute nuisance directions at L28 hidden states
    cap28 = KCap()  # reuse, gets last token
    hook28 = model.model.layers[28].register_forward_hook(
        lambda m, i, o: setattr(cap28, 'out',
            (o[0] if isinstance(o, tuple) else o)[0, -1].detach().float().cpu().numpy()))

    # a) Convention: EN vs ZH (already computed, but let's redo cleanly)
    en_h, zh_h = [], []
    for p in ["Solve: 3x + 7 = 22", "GCD of 84 and 120", "23 times 17",
              "Hypotenuse legs 5 and 12", "Area circle radius 7"]:
        ids = tok(p, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids)
        en_h.append(cap28.out.copy())
    for p in ["\u89e3: 3x + 7 = 22", "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570", "23\u4e5817",
              "\u659c\u8fb9 5\u548c12", "\u534a\u5f847\u5706\u9762\u79ef"]:
        ids = tok(p, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids)
        zh_h.append(cap28.out.copy())

    e_c = np.mean(en_h, axis=0) - np.mean(zh_h, axis=0)
    e_c /= np.linalg.norm(e_c) + 1e-10
    proj = discarded.T @ e_c
    print(f'  convention (EN-ZH): {np.sum(proj**2):.4f} in discarded', flush=True)

    # b) System prompt vs no system prompt
    sys_h, nosys_h = [], []
    for p in ["Solve: 3x + 7 = 22", "GCD of 84 and 120", "23 times 17"]:
        # With system prompt
        msgs = [{"role": "system", "content": "You are a careful mathematical reasoner."}, {"role": "user", "content": p}]
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(t, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids)
        sys_h.append(cap28.out.copy())
        # Without
        ids = tok(p, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids)
        nosys_h.append(cap28.out.copy())

    e_sys = np.mean(sys_h, axis=0) - np.mean(nosys_h, axis=0)
    e_sys /= np.linalg.norm(e_sys) + 1e-10
    proj_sys = discarded.T @ e_sys
    print(f'  system prompt (sys-nosys): {np.sum(proj_sys**2):.4f} in discarded', flush=True)

    # c) Paraphrase: same problem, different wording
    para1_h, para2_h = [], []
    pairs = [
        ("Solve for x: 3x + 7 = 22", "Find x such that 3x plus 7 equals 22"),
        ("Calculate 347 + 658", "What is the sum of 347 and 658?"),
        ("Find the GCD of 84 and 120", "What is the greatest common divisor of 84 and 120?"),
    ]
    for p1, p2 in pairs:
        ids = tok(p1, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids)
        para1_h.append(cap28.out.copy())
        ids = tok(p2, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids)
        para2_h.append(cap28.out.copy())

    e_para = np.mean(para1_h, axis=0) - np.mean(para2_h, axis=0)
    e_para /= np.linalg.norm(e_para) + 1e-10
    proj_para = discarded.T @ e_para
    print(f'  paraphrase (v1-v2): {np.sum(proj_para**2):.4f} in discarded', flush=True)

    # d) Formal vs casual
    formal_h, casual_h = [], []
    pairs_fc = [
        ("Please calculate the product of 23 and 17.", "yo whats 23 times 17"),
        ("Kindly determine the greatest common divisor of 84 and 120.", "gcd of 84 and 120?"),
    ]
    for p1, p2 in pairs_fc:
        ids = tok(p1, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids)
        formal_h.append(cap28.out.copy())
        ids = tok(p2, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids)
        casual_h.append(cap28.out.copy())

    e_register = np.mean(formal_h, axis=0) - np.mean(casual_h, axis=0)
    e_register /= np.linalg.norm(e_register) + 1e-10
    proj_reg = discarded.T @ e_register
    print(f'  register (formal-casual): {np.sum(proj_reg**2):.4f} in discarded', flush=True)

    hook28.remove()

    print(f'\n  SUMMARY: fraction in discarded (bottom 1920 of 2048 W_down SVs at L28):')
    print(f'    convention:     {np.sum((discarded.T @ e_c)**2):.4f}')
    print(f'    system prompt:  {np.sum((discarded.T @ e_sys)**2):.4f}')
    print(f'    paraphrase:     {np.sum((discarded.T @ e_para)**2):.4f}')
    print(f'    register:       {np.sum((discarded.T @ e_register)**2):.4f}')
    print(f'  if all > 0.8: the bottom spectrum is a UNIVERSAL nuisance junk drawer')

    # === EXP 3: L1 CAUSAL RANK SWEEP ===
    print(f'\n{"="*60}')
    print('EXP 3: How low can L1 go?')
    print('='*60)

    diverse = [
        ("math", "What is 23 times 17? Think step by step.\n"),
        ("repeat", "Repeat exactly: the quick brown fox jumps over the lazy dog\n"),
        ("logic", "If all cats are mammals and some mammals swim, can cats swim?\n"),
        ("frumble", prompt),
    ]

    for rank in [128, 64, 32, 16, 8, 4]:
        with torch.no_grad():
            W = model.model.layers[1].mlp.down_proj.weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            W_t = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
            model.model.layers[1].mlp.down_proj.weight.copy_(W_t.half())

        results = []
        for pname, p in diverse:
            base = generate(model, tok, p, max_tok=15)  # use truncated for both since we restore after
            results.append(f'{pname}: {"".join(base[:10])}...')

        # Restore
        model.model.layers[1].mlp.down_proj.weight.data.copy_(orig_L1)

        # Need baseline for comparison... let's just report the output
        print(f'  rank={rank:>3d}: {" | ".join(results)}', flush=True)


if __name__ == '__main__':
    main()
