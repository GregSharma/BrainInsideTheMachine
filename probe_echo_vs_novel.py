"""echo vs novel: does the seed direction track originality of thought?

GLM5 hypothesis: tokens where the model ECHOES the prompt (restating premises)
should have lower seed scores than tokens where the model THINKS (novel reasoning).

Use RAW projection onto seed direction (not sigmoid-squashed probability)
to see fine structure.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def main():
    import warnings; warnings.filterwarnings('ignore')
    print('loading...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    prompt = ("Every frumble that lives in a glasshouse is transparent. "
              "Every transparent creature can pass through walls. "
              "I found a creature in a glasshouse that can pass through walls. "
              "Must it be a frumble? Explain your reasoning step by step.\n")

    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    prompt_len = ids.shape[1]
    prompt_tokens = [tok.decode(ids[0, i]) for i in range(prompt_len)]
    prompt_text = tok.decode(ids[0]).lower()

    # Capture K at L26 for all positions at each step
    class KCap:
        def __init__(self): self.out = None
        def __call__(self, m, i, o): self.out = o[0].detach().float().cpu().numpy()

    cap = KCap()
    hook = model.model.layers[26].self_attn.k_proj.register_forward_hook(cap)

    gen_tokens = []; gen_K = []; prompt_K = None
    for step in range(80):
        with torch.inference_mode(): out = model(ids)
        if step == 0:
            prompt_K = cap.out[:prompt_len].copy()
        gen_K.append(cap.out[-1].copy())  # last position K
        nid = out.logits[0, -1].argmax().item()
        gen_tokens.append(tok.decode(nid))
        if nid == tok.eos_token_id: break
        ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)

    hook.remove()
    gen_K = np.stack(gen_K)
    n_gen = len(gen_tokens)
    print(f'generated {n_gen} tokens', flush=True)
    print(f'output: {"".join(gen_tokens[:60])}...', flush=True)

    # Train seed classifier
    X = np.vstack([prompt_K, gen_K])
    y = np.array([0]*prompt_len + [1]*n_gen)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)

    # Get the seed DIRECTION (classifier weight vector)
    seed_dir = clf.coef_[0]  # (kv_dim,)
    seed_dir = seed_dir / (np.linalg.norm(seed_dir) + 1e-10)
    seed_bias = clf.intercept_[0]

    # RAW projections (before sigmoid) for all generated tokens
    raw_proj = gen_K @ seed_dir + seed_bias  # continuous score

    # Label each generated token as ECHO or NOVEL
    # Echo: token appears in a matching 3-gram with the prompt
    gen_text = ''.join(gen_tokens).lower()
    echo_labels = []
    for i, tok_str in enumerate(gen_tokens):
        # Check if this token is part of a 3-token sequence that appears in the prompt
        # Build 3-gram centered on this token
        context = ''.join(gen_tokens[max(0,i-1):i+2]).lower().strip()
        is_echo = context in prompt_text and len(context) > 3
        # Also check 2-gram
        if not is_echo:
            context2 = ''.join(gen_tokens[max(0,i-1):i+1]).lower().strip()
            is_echo = context2 in prompt_text and len(context2) > 4
        echo_labels.append('echo' if is_echo else 'novel')

    # Display
    print(f'\n{"="*80}')
    print('ECHO vs NOVEL: raw seed projection (higher = more "generated-like")')
    print(f'{"="*80}')
    print(f'  {"step":>4s} {"token":>12s} {"type":>6s} {"raw_proj":>10s} {"prob":>8s}')

    echo_projs = []; novel_projs = []
    for i in range(n_gen):
        prob = 1 / (1 + np.exp(-raw_proj[i]))  # sigmoid for reference
        marker = '*' if echo_labels[i] == 'echo' else ' '
        print(f'  {i:>4d} {gen_tokens[i]:>12s} {echo_labels[i]:>6s} {raw_proj[i]:>+10.4f} {prob:>8.4f} {marker}')
        if echo_labels[i] == 'echo':
            echo_projs.append(raw_proj[i])
        else:
            novel_projs.append(raw_proj[i])

    print(f'\n  SUMMARY:')
    print(f'  echo  tokens: n={len(echo_projs)}, mean_proj={np.mean(echo_projs):+.4f}, std={np.std(echo_projs):.4f}')
    print(f'  novel tokens: n={len(novel_projs)}, mean_proj={np.mean(novel_projs):+.4f}, std={np.std(novel_projs):.4f}')
    if echo_projs and novel_projs:
        delta = np.mean(novel_projs) - np.mean(echo_projs)
        # Welch t-test
        from scipy.stats import ttest_ind
        t, p = ttest_ind(novel_projs, echo_projs, equal_var=False)
        print(f'  delta (novel - echo): {delta:+.4f}')
        print(f'  t-test: t={t:.4f}, p={p:.6f}')
        if p < 0.05:
            print(f'  *** SIGNIFICANT: seed direction tracks originality ***')
        else:
            print(f'  not significant: seed is binary, variation is noise')

    # Also: look at projection of PROMPT tokens for reference
    prompt_raw = prompt_K @ seed_dir + seed_bias
    print(f'\n  prompt tokens: mean_proj={np.mean(prompt_raw):+.4f}, std={np.std(prompt_raw):.4f}')
    print(f'  gap: prompt={np.mean(prompt_raw):+.4f}, echo_gen={np.mean(echo_projs):+.4f}, '
          f'novel_gen={np.mean(novel_projs):+.4f}')
    print(f'  echo is {abs(np.mean(echo_projs) - np.mean(prompt_raw)):.4f} from prompt, '
          f'{abs(np.mean(echo_projs) - np.mean(novel_projs)):.4f} from novel')


if __name__ == '__main__':
    main()
