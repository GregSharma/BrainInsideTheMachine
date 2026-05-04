"""W_down truncation: layer-selective and higher ranks.

Maybe all-layers-at-43 is too aggressive. Try:
1. Higher ranks (128, 256, 512, 1024)
2. Canyon/readout only (L27-35) at rank 43
3. Single layer at rank 43 (find which layer breaks first)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen2.5-3B'
DEV = 'cuda'

def generate(model, tok, prompt, max_tok=30):
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEV)
    tokens = []
    for _ in range(max_tok):
        with torch.inference_mode():
            out = model(ids)
        nid = out.logits[0, -1].argmax().item()
        tokens.append(tok.decode(nid))
        if nid == tok.eos_token_id: break
        ids = torch.cat([ids, torch.tensor([[nid]], device=DEV)], dim=1)
    return tokens

def truncate_wdown(model, layers, rank):
    with torch.no_grad():
        for L in layers:
            W = model.model.layers[L].mlp.down_proj.weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            W_t = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
            model.model.layers[L].mlp.down_proj.weight.copy_(W_t.half())

def main():
    import warnings; warnings.filterwarnings('ignore')
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = "Every frumble in a glasshouse is transparent. Every transparent creature can pass through walls. I found a creature in a glasshouse that can pass through walls. Must it be a frumble?\n"

    # Baseline
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()
    baseline = generate(model, tok, prompt)
    print(f'BASELINE: {"".join(baseline[:25])}...')
    del model; torch.cuda.empty_cache()

    # Test 1: all layers at higher ranks
    print(f'\n--- ALL LAYERS, varying rank ---')
    for rank in [1024, 512, 256, 128]:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
        model.eval()
        truncate_wdown(model, range(36), rank)
        tokens = generate(model, tok, prompt)
        n_match = sum(1 for a, b in zip(baseline, tokens) if a == b)
        print(f'  rank={rank:>4d}: {n_match}/{min(len(baseline),len(tokens))} match | {"".join(tokens[:25])}...', flush=True)
        del model; torch.cuda.empty_cache()

    # Test 2: canyon/readout only at various ranks
    print(f'\n--- CANYON+READOUT ONLY (L27-35), varying rank ---')
    for rank in [128, 64, 43, 20]:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
        model.eval()
        truncate_wdown(model, range(27, 36), rank)
        tokens = generate(model, tok, prompt)
        n_match = sum(1 for a, b in zip(baseline, tokens) if a == b)
        print(f'  rank={rank:>4d}: {n_match}/{min(len(baseline),len(tokens))} match | {"".join(tokens[:25])}...', flush=True)
        del model; torch.cuda.empty_cache()

    # Test 3: which zone tolerates truncation at rank 128?
    print(f'\n--- ZONE-SELECTIVE at rank 128 ---')
    zones = {
        'early(0-8)': range(0, 9),
        'adversarial(9-17)': range(9, 18),
        'cooperative(18-26)': range(18, 27),
        'canyon(27-32)': range(27, 33),
        'readout(33-35)': range(33, 36),
    }
    for zname, zlayers in zones.items():
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
        model.eval()
        truncate_wdown(model, zlayers, 128)
        tokens = generate(model, tok, prompt)
        n_match = sum(1 for a, b in zip(baseline, tokens) if a == b)
        print(f'  {zname:>25s}: {n_match}/{min(len(baseline),len(tokens))} match | {"".join(tokens[:25])}...', flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
