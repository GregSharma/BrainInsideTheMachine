"""Two decisive experiments. No cathedrals.

EXP 1: SEED FLIP (DeepSeek's proposal)
  Replace generated tokens' K vectors with matched prompt tokens' K vectors.
  If the model breaks -> seed is FUNCTIONAL (source monitoring).
  If nothing changes -> seed is DECORATIVE (epiphenomenal).

EXP 2: KV COMPRESSION (Gemini/DeepSeek/ChatGPT consensus)
  SVD the KV cache during generation. Truncate to rank k.
  Find minimum k where output degrades.
  This determines the compressed model's state size.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def main():
    import warnings; warnings.filterwarnings('ignore')
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    prompt = "What is 23 times 17? Think step by step.\n"

    # ================================================================
    # EXP 1: SEED FLIP
    # Generate baseline. Then re-generate, but at each step,
    # replace the K/V cache entries for generated tokens with
    # the K/V from prompt tokens at matched positions.
    # "Make the model think its own output was user input."
    # ================================================================
    print('='*60)
    print('EXP 1: SEED FLIP')
    print('  replace generated K/V with prompt K/V at matched positions')
    print('='*60)

    # Baseline generation
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    prompt_len = ids.shape[1]
    baseline = []
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        past = out.past_key_values

    for step in range(20):
        next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        baseline.append(tok.decode(next_id[0, 0]))
        if next_id.item() == tok.eos_token_id: break
        with torch.inference_mode():
            out = model(next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values

    print(f'  baseline ({len(baseline)} tok): {baseline}')

    # Seed flip generation: at each step, replace the K cache entries
    # for generated positions with K entries from a prompt position.
    # Use the LAST prompt position's K as the replacement.
    ids2 = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    with torch.inference_mode():
        out2 = model(ids2, use_cache=True)
        past2 = out2.past_key_values  # this is the prompt's KV cache

    # Save prompt K values for reference
    # DynamicCache API: past.key_cache[L], past.value_cache[L]
    # key shape: (batch, n_kv_heads, seq, head_dim)
    n_layers = len(past2.layers)
    prompt_K_last = {}  # K vector of last prompt position, per layer
    for L in range(n_layers):
        k = past2.layers[L].keys  # (1, n_kv_heads, prompt_len, head_dim)
        prompt_K_last[L] = k[:, :, -1:, :].clone()  # (1, n_kv_heads, 1, head_dim)

    flipped = []
    for step in range(20):
        next_id = out2.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        flipped.append(tok.decode(next_id[0, 0]))
        if next_id.item() == tok.eos_token_id: break

        with torch.inference_mode():
            out2 = model(next_id, past_key_values=past2, use_cache=True)
            past2 = out2.past_key_values

        # FLIP: replace the K entry for the just-generated position
        # with the prompt's last position K
        for L in range(n_layers):
            past2.layers[L].keys[:, :, -1:, :] = prompt_K_last[L]

    n_diff = sum(1 for a, b in zip(baseline, flipped) if a != b)
    print(f'  flipped ({len(flipped)} tok): {flipped}')
    print(f'  {n_diff}/{min(len(baseline), len(flipped))} tokens differ')

    # Also try flipping BOTH K and V
    print(f'\n  --- flip both K AND V ---')
    ids3 = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    with torch.inference_mode():
        out3 = model(ids3, use_cache=True)
        past3 = out3.past_key_values

    prompt_KV_last = {}
    for L in range(n_layers):
        prompt_KV_last[L] = (past3.layers[L].keys[:, :, -1:, :].clone(),
                             past3.layers[L].values[:, :, -1:, :].clone())

    flipped_kv = []
    for step in range(20):
        next_id = out3.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        flipped_kv.append(tok.decode(next_id[0, 0]))
        if next_id.item() == tok.eos_token_id: break

        with torch.inference_mode():
            out3 = model(next_id, past_key_values=past3, use_cache=True)
            past3 = out3.past_key_values

        for L in range(n_layers):
            past3.layers[L].keys[:, :, -1:, :] = prompt_KV_last[L][0]
            past3.layers[L].values[:, :, -1:, :] = prompt_KV_last[L][1]

    n_diff2 = sum(1 for a, b in zip(baseline, flipped_kv) if a != b)
    print(f'  flipped_kv ({len(flipped_kv)} tok): {flipped_kv}')
    print(f'  {n_diff2}/{min(len(baseline), len(flipped_kv))} tokens differ')

    # ================================================================
    # EXP 2: KV COMPRESSION
    # Encode prompt. During generation, at each step,
    # truncate the KV cache to top-k SVD components.
    # Find minimum k where output matches baseline.
    # ================================================================
    print(f'\n{"="*60}')
    print('EXP 2: KV COMPRESSION')
    print('  truncate KV cache to rank k during generation')
    print('='*60)

    for k_rank in [256, 128, 64, 32, 16, 8, 4, 2, 1]:
        ids_c = tok(prompt, return_tensors='pt').input_ids.to(DEV)
        with torch.inference_mode():
            out_c = model(ids_c, use_cache=True)
            past_c = out_c.past_key_values

        compressed_tokens = []
        for step in range(20):
            next_id = out_c.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            compressed_tokens.append(tok.decode(next_id[0, 0]))
            if next_id.item() == tok.eos_token_id: break

            with torch.inference_mode():
                out_c = model(next_id, past_key_values=past_c, use_cache=True)
                past_c = out_c.past_key_values

            # Compress KV cache: for each layer, SVD the K and V matrices
            # and keep only top k_rank components
            n_layers_c = len(past_c.layers)
            for L in range(n_layers_c):
                k_mat = past_c.layers[L].keys  # (1, n_kv_heads, seq, head_dim)
                v_mat = past_c.layers[L].values

                for head in range(k_mat.shape[1]):
                    K_h = k_mat[0, head].float()  # (seq, 128)
                    seq_len = K_h.shape[0]
                    actual_k = min(k_rank, seq_len, 128)

                    if actual_k < seq_len and actual_k < 128:
                        U, S, Vt = torch.linalg.svd(K_h, full_matrices=False)
                        K_approx = U[:, :actual_k] @ torch.diag(S[:actual_k]) @ Vt[:actual_k, :]
                        past_c.layers[L].keys[0, head] = K_approx.half()

                    V_h = v_mat[0, head].float()
                    if actual_k < seq_len and actual_k < 128:
                        U, S, Vt = torch.linalg.svd(V_h, full_matrices=False)
                        V_approx = U[:, :actual_k] @ torch.diag(S[:actual_k]) @ Vt[:actual_k, :]
                        past_c.layers[L].values[0, head] = V_approx.half()

        n_match = sum(1 for a, b in zip(baseline, compressed_tokens) if a == b)
        print(f'  k={k_rank:>3d}: {compressed_tokens[:15]}  match={n_match}/{min(len(baseline), len(compressed_tokens))}')


if __name__ == '__main__':
    main()
