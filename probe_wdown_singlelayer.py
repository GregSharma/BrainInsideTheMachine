"""Which single layer is most sensitive to W_down truncation?
Truncate ONE layer at a time to rank 128. Find the one that breaks hardest.
Also test L20 (Lipschitz outlier) and L35 (gate outlier) specifically.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    prompt = "Every frumble in a glasshouse is transparent. Every transparent creature can pass through walls. I found a creature in a glasshouse that can pass through walls. Must it be a frumble?\n"

    # Baseline
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()
    baseline = generate(model, tok, prompt)
    print(f'BASELINE: {"".join(baseline[:20])}...')
    del model; torch.cuda.empty_cache()

    # Single layer truncation at rank 128
    print(f'\n--- SINGLE LAYER at rank 128 ---')
    for L in range(36):
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
        model.eval()
        with torch.no_grad():
            W = model.model.layers[L].mlp.down_proj.weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            W_t = U[:, :128] @ torch.diag(S[:128]) @ Vt[:128, :]
            model.model.layers[L].mlp.down_proj.weight.copy_(W_t.half())
        tokens = generate(model, tok, prompt)
        n_match = sum(1 for a, b in zip(baseline, tokens) if a == b)
        total = min(len(baseline), len(tokens))
        status = 'OK' if n_match == total else f'BROKE({n_match}/{total})'
        # Only print if something changed
        if n_match < total:
            print(f'  L{L:>2d}: {status:>15s} | {"".join(tokens[:20])}...', flush=True)
        else:
            print(f'  L{L:>2d}: {status}', flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
