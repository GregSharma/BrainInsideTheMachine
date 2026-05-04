"""probe_seed_and_roundtrip: Two tests from Greg's hypothesis.

Test 1: SEED CLASSIFIER
  Can a linear classifier on K vectors distinguish prompt tokens from generated tokens?
  If yes, the model leaves a 'seed' in its own output.

Test 2: ROUNDTRIP RESIDUAL
  h_pre (before unembedding) vs E[token] (after re-embedding).
  What's lost? How much? Is the loss structured?
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_NAME = 'Qwen/Qwen2.5-3B'
DEVICE = 'cuda'


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('='*70)
    print('PROBE SEED + ROUNDTRIP')
    print('='*70)

    print('loading...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()

    KEY_LAYERS = [5, 13, 18, 26, 30, 33, 35]

    # === Generate from several prompts to get prompt+generated token pairs ===
    prompts = [
        "Repeat exactly: the quick brown fox jumps over the lazy dog\n",
        "all the roses in my garden are red. i got a flower from my garden. must it be a rose?\n",
        "What is 23 times 17? Think step by step.\n",
        "Translate to French: the cat sat on the mat\n",
    ]

    all_K_data = {L: {'prompt': [], 'generated': []} for L in KEY_LAYERS}
    all_h_data = {L: {'prompt': [], 'generated': []} for L in KEY_LAYERS}
    roundtrip_data = []  # (h_pre, token_id, h_post, cos_sim)

    # Hooks to capture K projections and hidden states
    class KCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            self.out = output.detach().float().cpu().numpy()  # (batch, n_kv_heads, seq, head_dim)

    class HCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[0].detach().float().cpu().numpy()  # (seq, d)

    k_caps = {L: KCap() for L in KEY_LAYERS}
    h_caps = {L: HCap() for L in KEY_LAYERS}
    k_hooks = [model.model.layers[L].self_attn.k_proj.register_forward_hook(k_caps[L]) for L in KEY_LAYERS]
    h_hooks = [model.model.layers[L].register_forward_hook(h_caps[L]) for L in KEY_LAYERS]

    embed_weight = model.model.embed_tokens.weight.detach().float().cpu().numpy()  # (vocab, d)

    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        prompt_len = ids.shape[1]

        # Generate
        for step in range(30):
            with torch.inference_mode():
                out = model(ids)

            seq_len = ids.shape[1]

            # Classify each token position as prompt or generated
            for L in KEY_LAYERS:
                k_all = k_caps[L].out  # (1, n_kv_heads, seq, head_dim) but k_proj output is (batch, seq, n_kv_heads*head_dim)
                # Actually k_proj output is a linear layer: (batch, seq, kv_dim)
                # Let me check the shape
                k_vec = k_caps[L].out  # should be (batch, seq, kv_dim=256)
                if len(k_vec.shape) == 3:
                    k_vec = k_vec[0]  # (seq, 256)
                elif len(k_vec.shape) == 2:
                    pass  # (seq, 256) already
                else:
                    # might be (batch, n_heads, seq, head_dim)
                    k_vec = k_vec.reshape(-1, k_vec.shape[-1])  # flatten

                h_all = h_caps[L].out  # (seq, d)

                # Only store the LAST token (current generation position)
                if step == 0:
                    # Store all prompt tokens
                    for pos in range(min(prompt_len, k_vec.shape[0])):
                        all_K_data[L]['prompt'].append(k_vec[pos])
                        all_h_data[L]['prompt'].append(h_all[pos])

                # Store the last position (generated token at this step)
                if seq_len > prompt_len:  # we have generated tokens
                    all_K_data[L]['generated'].append(k_vec[-1])
                    all_h_data[L]['generated'].append(h_all[-1])

            # Roundtrip analysis on last token
            h_last_L35 = h_caps[35].out[-1]  # (d,) at layer 35
            # Apply final RMSNorm manually
            rms_weight = model.model.norm.weight.detach().float().cpu().numpy()
            h_normed = h_last_L35 / (np.sqrt(np.mean(h_last_L35**2) + 1e-6))
            h_pre = h_normed * rms_weight  # (d,)

            # Get logits and token
            logits_np = h_pre @ embed_weight.T  # (vocab,)
            token_id = int(np.argmax(logits_np))
            h_post = embed_weight[token_id]  # (d,)

            c = cos(h_pre, h_post)
            norm_ratio = np.linalg.norm(h_post) / (np.linalg.norm(h_pre) + 1e-10)
            residual = h_pre - h_post
            residual_frac = np.linalg.norm(residual) / (np.linalg.norm(h_pre) + 1e-10)

            roundtrip_data.append({
                'prompt_idx': pi,
                'step': step,
                'token': tokenizer.decode(token_id),
                'cos': c,
                'norm_ratio': norm_ratio,
                'residual_frac': residual_frac,
            })

            # Next token
            next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            if next_id.item() == tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, next_id], dim=1)

        print(f'  prompt {pi}: {prompt_len} prompt + {step+1} generated tokens', flush=True)

    # Remove hooks
    for h in k_hooks + h_hooks:
        h.remove()

    # === TEST 1: SEED CLASSIFIER ===
    print(f'\n{"="*70}')
    print('TEST 1: SEED CLASSIFIER')
    print('  can a linear classifier on K vectors tell prompt from generated?')
    print(f'{"="*70}')

    for L in KEY_LAYERS:
        prompt_K = np.stack(all_K_data[L]['prompt']) if all_K_data[L]['prompt'] else np.zeros((0, 256))
        gen_K = np.stack(all_K_data[L]['generated']) if all_K_data[L]['generated'] else np.zeros((0, 256))

        if len(prompt_K) < 5 or len(gen_K) < 5:
            print(f'  L{L}: insufficient data (prompt={len(prompt_K)}, gen={len(gen_K)})')
            continue

        X = np.vstack([prompt_K, gen_K])
        y = np.array([0]*len(prompt_K) + [1]*len(gen_K))

        # Random train/test split (80/20)
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X))
        split = int(0.8 * len(idx))
        X_train, X_test = X[idx[:split]], X[idx[split:]]
        y_train, y_test = y[idx[:split]], y[idx[split:]]

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, clf.predict(X_train))
        acc_test = accuracy_score(y_test, clf.predict(X_test))

        # Also do on h (hidden states)
        prompt_H = np.stack(all_h_data[L]['prompt'])
        gen_H = np.stack(all_h_data[L]['generated'])
        X_h = np.vstack([prompt_H, gen_H])
        X_h_train, X_h_test = X_h[idx[:split]], X_h[idx[split:]]
        clf_h = LogisticRegression(max_iter=2000, C=1.0)
        clf_h.fit(X_h_train, y_train)
        acc_h_test = accuracy_score(y_test, clf_h.predict(X_h_test))

        # Mean cosine between prompt K and generated K
        mean_prompt_K = prompt_K.mean(axis=0)
        mean_gen_K = gen_K.mean(axis=0)
        k_cos = cos(mean_prompt_K, mean_gen_K)

        print(f'  L{L:2d}: K_acc_test={acc_test:.3f} ({len(prompt_K)}p + {len(gen_K)}g)  '
              f'h_acc_test={acc_h_test:.3f}  '
              f'cos(mean_K_prompt, mean_K_gen)={k_cos:.4f}')

    # === TEST 2: ROUNDTRIP RESIDUAL ===
    print(f'\n{"="*70}')
    print('TEST 2: ROUNDTRIP RESIDUAL')
    print('  how much is lost going h -> argmax(h@E^T) -> E[token] ?')
    print(f'{"="*70}')

    cos_vals = [d['cos'] for d in roundtrip_data]
    res_vals = [d['residual_frac'] for d in roundtrip_data]
    nr_vals = [d['norm_ratio'] for d in roundtrip_data]

    print(f'  cos(h_pre, E[token]):  mean={np.mean(cos_vals):.4f}  std={np.std(cos_vals):.4f}  '
          f'min={np.min(cos_vals):.4f}  max={np.max(cos_vals):.4f}')
    print(f'  ||residual||/||h_pre||: mean={np.mean(res_vals):.4f}  std={np.std(res_vals):.4f}')
    print(f'  ||E[token]||/||h_pre||: mean={np.mean(nr_vals):.4f}  std={np.std(nr_vals):.4f}')

    # Show a few examples
    print(f'\n  sample roundtrips:')
    print(f'  {"prompt":>6s} {"step":>4s} {"token":>15s} {"cos":>8s} {"res_frac":>10s} {"norm_ratio":>12s}')
    for d in roundtrip_data[:20]:
        print(f'  {d["prompt_idx"]:>6d} {d["step"]:>4d} {d["token"]:>15s} {d["cos"]:>8.4f} {d["residual_frac"]:>10.4f} {d["norm_ratio"]:>12.4f}')

    # SVD of residuals to check if loss is structured
    print(f'\n  SVD of roundtrip residuals (is the loss structured?)')
    # Collect residuals from h data
    # We need h_pre and h_post which we didn't store as arrays... let's use the summary stats
    print(f'  (need to collect residual vectors — using cos as proxy for now)')
    print(f'  if cos is HIGH and consistent, loss is in a predictable subspace')
    print(f'  if cos is LOW and variable, loss is unstructured')


if __name__ == '__main__':
    main()
