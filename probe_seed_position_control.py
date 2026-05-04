"""probe_seed_position_control: Is the seed RoPE or content?

The seed classifier achieves 100% accuracy distinguishing prompt from
generated tokens via K vectors. But is this just RoPE (position encoding)?

Test: create a scenario where prompt and generated tokens occupy
OVERLAPPING position ranges. If the classifier still works on the
overlapping positions, the seed is content-based (learned).
If it fails, it's just RoPE (architectural).

Approach: use SHORT prompts that generate MANY tokens.
Prompt positions 0-5, generated positions 6-25.
Also use LONG prompts that generate FEW tokens.
Prompt positions 0-50, generated positions 51-55.

Then: train classifier only on positions 5-20 (where both conditions
have data) and see if it still discriminates.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_NAME = 'Qwen/Qwen2.5-3B'
DEVICE = 'cuda'


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('='*70)
    print('PROBE SEED POSITION CONTROL: is the seed RoPE or content?')
    print('='*70)

    print('loading...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()

    # Hooks for K and h
    KEY_LAYERS = [13, 26, 33]

    class KCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            self.out = output[0].detach().float().cpu().numpy()  # (seq, kv_dim)

    class HCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[0].detach().float().cpu().numpy()

    k_caps = {L: KCap() for L in KEY_LAYERS}
    h_caps = {L: HCap() for L in KEY_LAYERS}
    hooks = []
    for L in KEY_LAYERS:
        hooks.append(model.model.layers[L].self_attn.k_proj.register_forward_hook(k_caps[L]))
        hooks.append(model.model.layers[L].register_forward_hook(h_caps[L]))

    # Collect data from multiple prompts with varying lengths
    # Short prompts: positions 0-5 are prompt, 6+ are generated
    # Long prompts: positions 0-50 are prompt, 51+ are generated
    data = {L: {'k': [], 'h': [], 'is_gen': [], 'pos': [], 'prompt_len': []} for L in KEY_LAYERS}

    prompts = [
        # SHORT prompts (5-10 tokens) -> generate 20 tokens
        "hello\n",
        "2+2=\n",
        "Say hi\n",
        "count to ten\n",
        "yes or no?\n",
        # LONG prompts (30-60 tokens) -> generate 10 tokens
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog again.\n",
        "All the roses in my garden are red. I got a flower from my garden and it is red. Must it be a rose? Think carefully.\n",
        "Once upon a time, in a land far far away, there lived a princess who loved mathematics more than anything else in the world.\n",
    ]

    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        prompt_len = ids.shape[1]
        max_gen = 25 if prompt_len < 15 else 10

        for step in range(max_gen):
            with torch.inference_mode():
                model(ids)

            seq_len = ids.shape[1]

            if step == 0:
                # Store all prompt tokens
                for L in KEY_LAYERS:
                    k_all = k_caps[L].out  # (seq, kv_dim)
                    h_all = h_caps[L].out  # (seq, d)
                    for pos in range(prompt_len):
                        data[L]['k'].append(k_all[pos])
                        data[L]['h'].append(h_all[pos])
                        data[L]['is_gen'].append(0)
                        data[L]['pos'].append(pos)
                        data[L]['prompt_len'].append(prompt_len)

            # Store generated token (last position)
            for L in KEY_LAYERS:
                k_all = k_caps[L].out
                h_all = h_caps[L].out
                data[L]['k'].append(k_all[-1])
                data[L]['h'].append(h_all[-1])
                data[L]['is_gen'].append(1)
                data[L]['pos'].append(seq_len - 1)
                data[L]['prompt_len'].append(prompt_len)

            next_id = torch.tensor([[model(ids).logits[0, -1].argmax().item()]], device=DEVICE)
            # Recompute to get right token
            with torch.inference_mode():
                out = model(ids)
            next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            if next_id.item() == tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, next_id], dim=1)

        print(f'  prompt {pi}: {prompt_len} prompt + {step+1} gen tokens', flush=True)

    for h in hooks:
        h.remove()

    # === ANALYSIS ===
    for L in KEY_LAYERS:
        k_all = np.stack(data[L]['k'])
        h_all = np.stack(data[L]['h'])
        is_gen = np.array(data[L]['is_gen'])
        pos = np.array(data[L]['pos'])

        print(f'\n{"="*70}')
        print(f'LAYER {L}: {len(is_gen)} samples ({(is_gen==0).sum()} prompt, {(is_gen==1).sum()} gen)')
        print(f'{"="*70}')

        # === Test 1: All positions (baseline) ===
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(k_all))
        split = int(0.8 * len(idx))

        clf_k = LogisticRegression(max_iter=2000, C=1.0)
        clf_k.fit(k_all[idx[:split]], is_gen[idx[:split]])
        acc_k_all = accuracy_score(is_gen[idx[split:]], clf_k.predict(k_all[idx[split:]]))

        clf_h = LogisticRegression(max_iter=2000, C=1.0)
        clf_h.fit(h_all[idx[:split]], is_gen[idx[:split]])
        acc_h_all = accuracy_score(is_gen[idx[split:]], clf_h.predict(h_all[idx[split:]]))

        print(f'  ALL POSITIONS: K_acc={acc_k_all:.3f}  h_acc={acc_h_all:.3f}')

        # === Test 2: OVERLAPPING positions only ===
        # Find position range where both prompt and generated tokens exist
        prompt_positions = set(pos[is_gen == 0])
        gen_positions = set(pos[is_gen == 1])
        overlap = prompt_positions & gen_positions

        if len(overlap) > 5:
            overlap_mask = np.array([p in overlap for p in pos])
            k_overlap = k_all[overlap_mask]
            h_overlap = h_all[overlap_mask]
            y_overlap = is_gen[overlap_mask]
            pos_overlap = pos[overlap_mask]

            idx2 = rng.permutation(len(k_overlap))
            split2 = int(0.8 * len(idx2))

            clf_k2 = LogisticRegression(max_iter=2000, C=1.0)
            clf_k2.fit(k_overlap[idx2[:split2]], y_overlap[idx2[:split2]])
            acc_k_overlap = accuracy_score(y_overlap[idx2[split2:]], clf_k2.predict(k_overlap[idx2[split2:]]))

            clf_h2 = LogisticRegression(max_iter=2000, C=1.0)
            clf_h2.fit(h_overlap[idx2[:split2]], y_overlap[idx2[:split2]])
            acc_h_overlap = accuracy_score(y_overlap[idx2[split2:]], clf_h2.predict(h_overlap[idx2[split2:]]))

            print(f'  OVERLAP POSITIONS {sorted(overlap)[:10]}...: '
                  f'K_acc={acc_k_overlap:.3f}  h_acc={acc_h_overlap:.3f}  '
                  f'({(y_overlap==0).sum()}p + {(y_overlap==1).sum()}g)')
        else:
            print(f'  OVERLAP: insufficient ({len(overlap)} positions)')

        # === Test 3: POSITION-ONLY baseline ===
        # Can position alone predict prompt vs generated?
        pos_features = pos.reshape(-1, 1).astype(float)
        clf_pos = LogisticRegression(max_iter=2000, C=1.0)
        clf_pos.fit(pos_features[idx[:split]], is_gen[idx[:split]])
        acc_pos = accuracy_score(is_gen[idx[split:]], clf_pos.predict(pos_features[idx[split:]]))
        print(f'  POSITION-ONLY baseline: acc={acc_pos:.3f}')

        # === Test 4: K with position REGRESSED OUT ===
        # Remove linear dependence on position from K, then classify
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(pos.reshape(-1, 1), k_all)
        k_residual = k_all - lr.predict(pos.reshape(-1, 1))

        clf_kr = LogisticRegression(max_iter=2000, C=1.0)
        clf_kr.fit(k_residual[idx[:split]], is_gen[idx[:split]])
        acc_kr = accuracy_score(is_gen[idx[split:]], clf_kr.predict(k_residual[idx[split:]]))
        print(f'  K with position regressed out: acc={acc_kr:.3f}')


if __name__ == '__main__':
    main()
