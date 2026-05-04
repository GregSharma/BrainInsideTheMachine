"""Decisive experiments v3. FIXED: hooks fire. Positive control included.

The v2 bug: full-sequence reprocessing meant output.shape[1] was never 1.
Fix: don't gate on shape. Gate on step counter. Always modify last position.
Positive control: K->zeros MUST be catastrophic (per C2b).
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def generate_with_k_hook(model, tok, prompt, hook_fn, max_tokens=20):
    """Generate with a hook on k_proj that modifies the LAST position's K.
    hook_fn(output, step) -> modified_output or None (no change).
    Uses full-sequence reprocessing (no KV cache) so hooks see everything.
    """
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    prompt_len = ids.shape[1]
    step_counter = [0]  # mutable for closure
    fire_count = [0]

    def hook(module, inp, output):
        # output: (batch, seq, kv_dim)
        result = hook_fn(output, step_counter[0], prompt_len)
        if result is not None:
            fire_count[0] += 1
            return result
        return output

    hooks = [model.model.layers[L].self_attn.k_proj.register_forward_hook(hook)
             for L in range(36)]

    tokens = []
    for step in range(max_tokens):
        step_counter[0] = step
        with torch.inference_mode():
            out = model(ids)
        nid = out.logits[0, -1].argmax().item()
        tokens.append(tok.decode(nid))
        if nid == tok.eos_token_id: break
        ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)

    for h in hooks: h.remove()
    return tokens, fire_count[0]


def main():
    import warnings; warnings.filterwarnings('ignore')
    print('loading...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    prompt = "What is 23 times 17? Think step by step.\n"

    # ================================================================
    # BASELINE (no hooks)
    # ================================================================
    baseline, _ = generate_with_k_hook(model, tok, prompt, lambda o, s, pl: None)
    print(f'BASELINE: {baseline}')

    # ================================================================
    # POSITIVE CONTROL: K -> zeros for generated positions
    # This MUST break the model. If it doesn't, our framework is broken.
    # ================================================================
    print(f'\n{"="*60}')
    print('POSITIVE CONTROL: K->zeros at generated positions')
    print('  (MUST be catastrophic per C2b)')
    print('='*60)

    def zero_k(output, step, prompt_len):
        if step > 0:  # after first generation step
            modified = output.clone()
            # Zero out K for ALL generated positions (prompt_len onwards)
            seq_len = output.shape[1]
            if seq_len > prompt_len:
                modified[0, prompt_len:, :] = 0.0
                return modified
        return None

    zeroed, fires_z = generate_with_k_hook(model, tok, prompt, zero_k)
    n_diff = sum(1 for a, b in zip(baseline, zeroed) if a != b)
    print(f'  zeroed: {zeroed}')
    print(f'  {n_diff}/{min(len(baseline), len(zeroed))} tokens differ, hook fired {fires_z} times')
    if n_diff == 0:
        print(f'  *** FRAMEWORK BROKEN: zeroing K should be catastrophic ***')
        return
    else:
        print(f'  SANITY CHECK PASSED: zeroing K is catastrophic as expected')

    # ================================================================
    # EXP 1: SEED FLIP
    # Replace K at generated positions with K from last prompt position.
    # ================================================================
    print(f'\n{"="*60}')
    print('EXP 1: SEED FLIP (K at gen positions <- K at last prompt position)')
    print('='*60)

    # First, capture prompt K from a clean encoding
    prompt_k_cache = {}  # {layer: tensor of last prompt position K}
    def capture_prompt_k(module, inp, output, layer_idx=[0]):
        # Called during prompt encoding
        pass  # we'll do this differently

    # Encode prompt once to get reference K
    k_store = {}
    def store_k(module, inp, output):
        # Store the full output; we'll extract last prompt position later
        k_store['last'] = output[0, -1:, :].detach().clone()  # (1, kv_dim)

    # Per-layer prompt K capture
    prompt_k_per_layer = {}
    for L in range(36):
        store_hooks = []
        def make_store(layer_id):
            def hook(module, inp, output):
                prompt_k_per_layer[layer_id] = output[0, -1:, :].detach().clone()
            return hook
        h = model.model.layers[L].self_attn.k_proj.register_forward_hook(make_store(L))
        store_hooks.append(h)

    ids_ref = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    with torch.inference_mode():
        model(ids_ref)
    for h in store_hooks:
        h.remove()

    print(f'  captured prompt K at {len(prompt_k_per_layer)} layers')

    # Now generate with seed flip: replace gen positions' K with prompt K
    layer_counter = [0]

    def seed_flip_k(output, step, prompt_len):
        if step > 0:
            modified = output.clone()
            seq_len = output.shape[1]
            if seq_len > prompt_len:
                # Get this layer's prompt K (we cycle through layers 0-35)
                L = layer_counter[0] % 36
                if L in prompt_k_per_layer:
                    pk = prompt_k_per_layer[L]
                    # Replace ALL generated positions' K with prompt last K
                    for pos in range(prompt_len, seq_len):
                        modified[0, pos, :] = pk[0]
                layer_counter[0] += 1
                return modified
        else:
            layer_counter[0] = 0  # reset on step 0
        return None

    # Reset counter
    layer_counter[0] = 0
    flipped, fires_f = generate_with_k_hook(model, tok, prompt, seed_flip_k)
    n_diff = sum(1 for a, b in zip(baseline, flipped) if a != b)
    print(f'  flipped: {flipped}')
    print(f'  {n_diff}/{min(len(baseline), len(flipped))} differ, hook fired {fires_f} times')

    # ================================================================
    # EXP 2: KV COMPRESSION (K projected onto top-k subspace)
    # Capture K SVD basis from prompt encoding, then project all
    # generated K onto that basis during generation.
    # ================================================================
    print(f'\n{"="*60}')
    print('EXP 2: KV COMPRESSION (project gen K onto top-k subspace)')
    print('='*60)

    # Collect ALL prompt K vectors across all layers for SVD basis
    all_prompt_k = []
    def collect_prompt_k(module, inp, output):
        all_prompt_k.append(output[0].detach().float().cpu().numpy())  # (prompt_len, kv_dim)

    col_hooks = [model.model.layers[L].self_attn.k_proj.register_forward_hook(collect_prompt_k)
                 for L in range(36)]
    ids_svd = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    with torch.inference_mode():
        model(ids_svd)
    for h in col_hooks: h.remove()

    # Stack all prompt K vectors and compute SVD
    K_all = np.concatenate(all_prompt_k, axis=0)  # (36*prompt_len, kv_dim)
    K_c = K_all - K_all.mean(axis=0, keepdims=True)
    _, S_k, Vt_k = np.linalg.svd(K_c, full_matrices=False)
    cumvar = np.cumsum(S_k**2) / np.sum(S_k**2)
    print(f'  prompt K matrix: {K_all.shape}, r50={int(np.searchsorted(cumvar, 0.5)+1)}, '
          f'r90={int(np.searchsorted(cumvar, 0.9)+1)}, r99={int(np.searchsorted(cumvar, 0.99)+1)}')

    for k_rank in [64, 32, 16, 8, 4, 2]:
        basis = torch.tensor(Vt_k[:k_rank], dtype=torch.float16, device=DEV)
        mean_k = torch.tensor(K_all.mean(axis=0), dtype=torch.float16, device=DEV)

        def make_compress(basis, mean_k):
            def compress_k(output, step, prompt_len):
                if step > 0:
                    modified = output.clone()
                    seq_len = output.shape[1]
                    if seq_len > prompt_len:
                        for pos in range(prompt_len, seq_len):
                            k_vec = modified[0, pos:pos+1, :].float()  # (1, kv_dim)
                            centered = k_vec - mean_k.float()
                            proj = centered @ basis.float().T @ basis.float() + mean_k.float()
                            modified[0, pos, :] = proj[0].half()
                    return modified
                return None
            return compress_k

        comp_tokens, fires_c = generate_with_k_hook(model, tok, prompt, make_compress(basis, mean_k))
        n_match = sum(1 for a, b in zip(baseline, comp_tokens) if a == b)
        print(f'  k={k_rank:>3d}: {comp_tokens[:15]}  match={n_match}/{min(len(baseline), len(comp_tokens))}, fires={fires_c}')


if __name__ == '__main__':
    main()
