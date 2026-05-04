"""Decisive experiments v2. Hook-based, no cache mutation.

EXP 1: SEED FLIP via attention hook
  Intercept K at attention computation time.
  Replace generated-position K with prompt-position K.

EXP 2: KV COMPRESSION via attention hook
  Intercept K,V and truncate via SVD before attention.
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

    prompt = "What is 23 times 17? Think step by step.\n"

    # ================================================================
    # BASELINE
    # ================================================================
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    prompt_len = ids.shape[1]
    baseline = []
    for step in range(20):
        with torch.inference_mode():
            out = model(ids)
        nid = out.logits[0, -1].argmax().item()
        baseline.append(tok.decode(nid))
        if nid == tok.eos_token_id: break
        ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)

    print('='*60)
    print(f'BASELINE ({len(baseline)} tok): {baseline}')
    print('='*60)

    # ================================================================
    # EXP 1: SEED FLIP
    # Hook every self_attn. During forward, after K is computed,
    # replace K entries at generated positions with K from the
    # last prompt position.
    # ================================================================
    print('\nEXP 1: SEED FLIP')

    class SeedFlipHook:
        """Hooks into the full self_attn forward.
        We can't easily modify K inside Qwen's attention.
        Instead: hook the k_proj OUTPUT and replace."""
        def __init__(self, prompt_len):
            self.prompt_len = prompt_len
            self.prompt_k = None  # will store prompt's last K
            self.active = False
        def __call__(self, module, inp, output):
            # output of k_proj: (batch, seq, kv_dim)
            if not self.active:
                # During first pass (prompt encoding), save last position K
                self.prompt_k = output[0, -1:, :].detach().clone()
                return output
            # During generation: replace the K output (single new token)
            # with the saved prompt K
            if output.shape[1] == 1:  # single token generation
                modified = output.clone()
                modified[0, 0, :] = self.prompt_k[0]
                return modified
            return output

    flip_hooks_k = []
    flip_objects = []
    for L in range(36):
        fh = SeedFlipHook(prompt_len)
        flip_objects.append(fh)
        flip_hooks_k.append(model.model.layers[L].self_attn.k_proj.register_forward_hook(fh))

    # Encode prompt (captures prompt K)
    ids_flip = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    with torch.inference_mode():
        model(ids_flip)

    # Activate flipping
    for fh in flip_objects:
        fh.active = True

    flipped_k = []
    for step in range(20):
        with torch.inference_mode():
            out = model(ids_flip)
        nid = out.logits[0, -1].argmax().item()
        flipped_k.append(tok.decode(nid))
        if nid == tok.eos_token_id: break
        ids_flip = torch.cat([ids_flip, torch.tensor([[nid]], device=DEV)], dim=1)

    for h in flip_hooks_k: h.remove()

    n_diff = sum(1 for a, b in zip(baseline, flipped_k) if a != b)
    print(f'  K-only flip: {flipped_k}')
    print(f'  {n_diff}/{min(len(baseline), len(flipped_k))} tokens differ')

    # Now flip BOTH K and V
    class SeedFlipHookV:
        def __init__(self):
            self.prompt_v = None
            self.active = False
        def __call__(self, module, inp, output):
            if not self.active:
                self.prompt_v = output[0, -1:, :].detach().clone()
                return output
            if output.shape[1] == 1:
                modified = output.clone()
                modified[0, 0, :] = self.prompt_v[0]
                return modified
            return output

    flip_k2 = []; flip_v2 = []; objs_k2 = []; objs_v2 = []
    for L in range(36):
        fk = SeedFlipHook(prompt_len); fv = SeedFlipHookV()
        objs_k2.append(fk); objs_v2.append(fv)
        flip_k2.append(model.model.layers[L].self_attn.k_proj.register_forward_hook(fk))
        flip_v2.append(model.model.layers[L].self_attn.v_proj.register_forward_hook(fv))

    ids_flip2 = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    with torch.inference_mode(): model(ids_flip2)
    for fk, fv in zip(objs_k2, objs_v2):
        fk.active = True; fv.active = True

    flipped_kv = []
    for step in range(20):
        with torch.inference_mode(): out = model(ids_flip2)
        nid = out.logits[0, -1].argmax().item()
        flipped_kv.append(tok.decode(nid))
        if nid == tok.eos_token_id: break
        ids_flip2 = torch.cat([ids_flip2, torch.tensor([[nid]], device=DEV)], dim=1)

    for h in flip_k2 + flip_v2: h.remove()

    n_diff2 = sum(1 for a, b in zip(baseline, flipped_kv) if a != b)
    print(f'  K+V flip: {flipped_kv}')
    print(f'  {n_diff2}/{min(len(baseline), len(flipped_kv))} tokens differ')

    # ================================================================
    # EXP 2: KV COMPRESSION
    # Hook K and V proj outputs. During generation,
    # we can't easily SVD the CACHE through hooks.
    # Instead: measure the effective rank of the KV cache
    # by capturing all K vectors during generation and SVDing after.
    # Then separately test: does the model work with low-rank K?
    # ================================================================
    print(f'\n{"="*60}')
    print('EXP 2: KV EFFECTIVE RANK')
    print('  what is the effective rank of K across positions?')
    print('='*60)

    # Generate 20 tokens, capture K at all layers for all positions
    class KCollector:
        def __init__(self):
            self.all_k = []
        def __call__(self, module, inp, output):
            self.all_k.append(output[0].detach().float().cpu().numpy())  # (seq_or_1, kv_dim)

    collectors = {}
    col_hooks = []
    for L in [0, 9, 18, 26, 33, 35]:
        c = KCollector()
        collectors[L] = c
        col_hooks.append(model.model.layers[L].self_attn.k_proj.register_forward_hook(c))

    ids_r = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    for step in range(20):
        with torch.inference_mode(): out = model(ids_r)
        nid = out.logits[0, -1].argmax().item()
        if nid == tok.eos_token_id: break
        ids_r = torch.cat([ids_r, torch.tensor([[nid]], device=DEV)], dim=1)

    for h in col_hooks: h.remove()

    # Analyze K rank at each layer
    for L in sorted(collectors.keys()):
        # Concatenate all K vectors: first call has seq tokens, rest have 1 each
        all_k = []
        for k_out in collectors[L].all_k:
            if len(k_out.shape) == 1:
                all_k.append(k_out.reshape(1, -1))
            elif len(k_out.shape) == 2:
                for i in range(k_out.shape[0]):
                    all_k.append(k_out[i:i+1])
        K_matrix = np.concatenate(all_k, axis=0)  # (total_positions, kv_dim)
        n_pos, kv_dim = K_matrix.shape

        # SVD
        K_c = K_matrix - K_matrix.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(K_c, full_matrices=False)
        cumvar = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
        r50 = int(np.searchsorted(cumvar, 0.5) + 1)
        r90 = int(np.searchsorted(cumvar, 0.9) + 1)
        r99 = int(np.searchsorted(cumvar, 0.99) + 1)
        top1 = float(S[0]**2 / (np.sum(S**2) + 1e-10))

        print(f'  L{L:>2d}: K matrix {n_pos}x{kv_dim}, '
              f'r50={r50}, r90={r90}, r99={r99}, '
              f'top1_frac={top1:.4f}, sv1/sv2={S[0]/S[1]:.2f}')

    # ================================================================
    # EXP 2b: CAUSAL KV COMPRESSION
    # Hook K proj to project onto top-k subspace during generation.
    # ================================================================
    print(f'\n  --- causal KV compression: project K onto top-k during generation ---')

    # First get the basis from baseline generation
    # Use the K matrix from L26 as representative
    K_ref = []
    for k_out in collectors[26].all_k:
        if len(k_out.shape) == 1:
            K_ref.append(k_out.reshape(1, -1))
        elif len(k_out.shape) == 2:
            for i in range(k_out.shape[0]):
                K_ref.append(k_out[i:i+1])
    K_ref = np.concatenate(K_ref, axis=0)
    K_ref_c = K_ref - K_ref.mean(axis=0, keepdims=True)
    _, _, Vt_ref = np.linalg.svd(K_ref_c, full_matrices=False)

    for k_rank in [128, 64, 32, 16, 8, 4]:
        basis = torch.tensor(Vt_ref[:k_rank], dtype=torch.float16, device=DEV)  # (k, kv_dim)
        mean_k = torch.tensor(K_ref.mean(axis=0), dtype=torch.float16, device=DEV)  # (kv_dim,)

        class KCompressor:
            def __init__(self, basis, mean, active=False):
                self.basis = basis  # (k, kv_dim)
                self.mean = mean
                self.active = active
            def __call__(self, module, inp, output):
                if not self.active:
                    return output
                if output.shape[1] == 1:  # generation step
                    centered = output.float() - self.mean
                    proj = centered @ self.basis.T @ self.basis  # project onto subspace
                    return (proj + self.mean).to(output.dtype)
                return output

        comp_hooks = []
        comp_objs = []
        for L in range(36):
            c = KCompressor(basis, mean_k)
            comp_objs.append(c)
            comp_hooks.append(model.model.layers[L].self_attn.k_proj.register_forward_hook(c))

        # Encode prompt (uncompressed)
        ids_comp = tok(prompt, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode(): model(ids_comp)

        # Activate compression
        for c in comp_objs: c.active = True

        comp_tokens = []
        for step in range(20):
            with torch.inference_mode(): out = model(ids_comp)
            nid = out.logits[0, -1].argmax().item()
            comp_tokens.append(tok.decode(nid))
            if nid == tok.eos_token_id: break
            ids_comp = torch.cat([ids_comp, torch.tensor([[nid]], device=DEV)], dim=1)

        for h in comp_hooks: h.remove()

        n_match = sum(1 for a, b in zip(baseline, comp_tokens) if a == b)
        print(f'  k={k_rank:>3d}: {comp_tokens[:15]}  match={n_match}/{min(len(baseline), len(comp_tokens))}')


if __name__ == '__main__':
    main()
