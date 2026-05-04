"""spectral fingerprint: continuous seed score over generation time.

Novel logical problem (made-up words, can't pattern match).
Capture K at key layers for ALL positions at each generation step.
Compute continuous P(generated | K) for each position.
Display as trajectory over time.
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

    # Novel logical problem with made-up words
    prompt = ("Every frumble that lives in a glasshouse is transparent. "
              "Every transparent creature can pass through walls. "
              "I found a creature in a glasshouse that can pass through walls. "
              "Must it be a frumble? Explain your reasoning step by step.\n")

    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    prompt_len = ids.shape[1]
    prompt_tokens = [tok.decode(ids[0, i]) for i in range(prompt_len)]
    print(f'prompt ({prompt_len} tokens): {"".join(prompt_tokens[:20])}...', flush=True)

    # Capture K at key layers for ALL positions at each generation step
    KEY_LAYERS = [13, 26, 33]

    class AllKCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            self.out = output[0].detach().float().cpu().numpy()  # (seq, kv_dim)

    caps = {L: AllKCap() for L in KEY_LAYERS}
    hooks = [model.model.layers[L].self_attn.k_proj.register_forward_hook(caps[L]) for L in KEY_LAYERS]

    # Generate and collect K at each step
    gen_tokens = []
    # Store: per step, per layer, K matrix for ALL positions
    k_snapshots = []  # list of {layer: (n_positions, kv_dim)}

    MAX_GEN = 60
    for step in range(MAX_GEN):
        with torch.inference_mode():
            out = model(ids)

        snapshot = {}
        for L in KEY_LAYERS:
            snapshot[L] = caps[L].out.copy()  # (seq_len, kv_dim)
        k_snapshots.append(snapshot)

        nid = out.logits[0, -1].argmax().item()
        gen_tokens.append(tok.decode(nid))
        if nid == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)

    for h in hooks:
        h.remove()

    n_gen = len(gen_tokens)
    total_len = prompt_len + n_gen
    print(f'generated {n_gen} tokens', flush=True)
    print(f'output: {"".join(gen_tokens[:40])}...', flush=True)

    # === BUILD SEED CLASSIFIER ===
    # Use final generation step's K to train (has all positions)
    print(f'\nbuilding seed classifier...', flush=True)

    for L in KEY_LAYERS:
        final_K = k_snapshots[-1][L]  # (total_len, kv_dim)
        n_pos = final_K.shape[0]

        # Labels: 0=prompt, 1=generated
        labels = np.array([0]*prompt_len + [1]*(n_pos - prompt_len))

        # Train classifier
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(final_K, labels)

        # Get continuous probabilities for ALL positions at ALL generation steps
        print(f'\n{"="*80}')
        print(f'LAYER {L}: seed score (P(generated|K)) over generation time')
        print(f'{"="*80}')

        # Header: show token identities
        # We'll show a subset of positions for readability
        # Show last 5 prompt tokens + all generated tokens
        show_prompt = min(8, prompt_len)
        show_start = prompt_len - show_prompt

        # Column headers
        all_tokens = prompt_tokens + gen_tokens
        header_tokens = all_tokens[show_start:show_start + show_prompt + n_gen]

        # Print header
        print(f'  {"step":>4s} |', end='')
        for i, t in enumerate(header_tokens):
            t_short = t.strip()[:6] if t.strip() else '\u2423'
            if i == show_prompt:
                print(f' | ', end='')  # boundary marker
            print(f'{t_short:>7s}', end='')
        print()
        print(f'  {"-"*4} |' + '-' * (7 * show_prompt) + ' | ' + '-' * (7 * min(n_gen, 40)))

        # For each generation step, get probabilities
        for step_idx in range(0, min(n_gen, 40), 1):
            K_at_step = k_snapshots[step_idx][L]  # (prompt_len + step_idx, kv_dim)
            n_pos_step = K_at_step.shape[0]

            # Get probabilities
            probs = clf.predict_proba(K_at_step)[:, 1]  # P(generated)

            # Show relevant positions
            print(f'  {step_idx:>4d} |', end='')
            for i in range(show_start, min(n_pos_step, show_start + show_prompt + n_gen)):
                p = probs[i]
                if i == prompt_len:
                    print(f' | ', end='')  # boundary
                # Visual: shade by probability
                if p < 0.2:
                    marker = f'{p:.2f}'
                elif p < 0.4:
                    marker = f'{p:.2f}'
                elif p < 0.6:
                    marker = f'{p:.2f}'
                elif p < 0.8:
                    marker = f'{p:.2f}'
                else:
                    marker = f'{p:.2f}'
                print(f'{marker:>7s}', end='')
            print(f'  gen="{gen_tokens[step_idx].strip()[:8]}"')

        # Summary stats
        print(f'\n  summary at L{L}:')
        # Average prompt score across all steps
        prompt_scores = []
        gen_scores = []
        for step_idx in range(n_gen):
            K_s = k_snapshots[step_idx][L]
            probs = clf.predict_proba(K_s)[:, 1]
            prompt_scores.append(probs[:prompt_len].mean())
            if K_s.shape[0] > prompt_len:
                gen_scores.append(probs[prompt_len:].mean())

        print(f'  mean P(gen|K) for prompt tokens: {np.mean(prompt_scores):.4f} \u00b1 {np.std(prompt_scores):.4f}')
        if gen_scores:
            print(f'  mean P(gen|K) for gen tokens:    {np.mean(gen_scores):.4f} \u00b1 {np.std(gen_scores):.4f}')
        print(f'  separation: {np.mean(gen_scores) - np.mean(prompt_scores):.4f}' if gen_scores else '')

        # Is the score INCREASING over generation? (does the seed get stronger?)
        if len(gen_scores) > 5:
            early = np.mean(gen_scores[:5])
            late = np.mean(gen_scores[-5:])
            print(f'  early gen score: {early:.4f}, late gen score: {late:.4f}, drift: {late-early:+.4f}')


if __name__ == '__main__':
    main()
