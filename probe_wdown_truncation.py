"""W_down SVD truncation: can we compress every MLP's output to rank 43?

The weight spectra showed W_down effective rank r90 ≈ 43 at every layer.
Truncate W_down to rank k via SVD. Generate. Does it still work?

Also test rank 20, 10, 5 to find the boundary.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def main():
    import warnings; warnings.filterwarnings('ignore')
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    prompts = [
        "Every frumble in a glasshouse is transparent. Every transparent creature can pass through walls. I found a creature in a glasshouse that can pass through walls. Must it be a frumble?\n",
        "What is 23 times 17? Think step by step.\n",
        "All roses in my garden are red. I got a red flower from my garden. Must it be a rose?\n",
    ]

    # Baseline first
    print('loading baseline...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    baselines = {}
    for pi, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
        tokens = []
        for step in range(40):
            with torch.inference_mode():
                out = model(ids)
            nid = out.logits[0, -1].argmax().item()
            tokens.append(tok.decode(nid))
            if nid == tok.eos_token_id: break
            ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)
        baselines[pi] = tokens
        print(f'  P{pi} baseline: {"".join(tokens[:30])}...', flush=True)

    del model
    torch.cuda.empty_cache()

    # Now test truncated versions
    for k_rank in [43, 30, 20, 10, 5]:
        print(f'\n{"="*60}')
        print(f'RANK {k_rank} TRUNCATION')
        print(f'{"="*60}')

        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
        model.eval()

        # Truncate every W_down to rank k
        with torch.no_grad():
            for L in range(36):
                W = model.model.layers[L].mlp.down_proj.weight.float()  # (2048, 11008)
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                # Keep top k
                W_trunc = U[:, :k_rank] @ torch.diag(S[:k_rank]) @ Vt[:k_rank, :]
                model.model.layers[L].mlp.down_proj.weight.copy_(W_trunc.half())

        print(f'  truncated all 36 W_down to rank {k_rank}', flush=True)

        for pi, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
            tokens = []
            for step in range(40):
                with torch.inference_mode():
                    out = model(ids)
                nid = out.logits[0, -1].argmax().item()
                tokens.append(tok.decode(nid))
                if nid == tok.eos_token_id: break
                ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)

            n_match = sum(1 for a, b in zip(baselines[pi], tokens) if a == b)
            total = min(len(baselines[pi]), len(tokens))
            print(f'  P{pi}: {n_match}/{total} match | {"".join(tokens[:30])}...')

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
