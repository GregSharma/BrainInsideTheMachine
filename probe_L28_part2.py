"""L28 part 2: canyon sweep + L1/L6 diversity. Reuse one model, restore weights."""
import torch, copy
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
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=DEV, trust_remote_code=True)
    model.eval()

    # Save original W_down weights for restoration
    orig_weights = {}
    for L in list(range(27, 36)) + [1, 6]:
        orig_weights[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    frumble = "Every frumble in a glasshouse is transparent. Every transparent creature can pass through walls. I found a creature in a glasshouse that can pass through walls. Must it be a frumble?\n"
    baseline = generate(model, tok, frumble)
    print(f'BASELINE: {"".join(baseline[:20])}...')

    # Canyon layers at rank 128
    print(f'\n--- CANYON LAYERS individually at rank 128 ---')
    for L in range(27, 36):
        with torch.no_grad():
            W = model.model.layers[L].mlp.down_proj.weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            W_t = U[:, :128] @ torch.diag(S[:128]) @ Vt[:128, :]
            model.model.layers[L].mlp.down_proj.weight.copy_(W_t.half())

        tokens = generate(model, tok, frumble)
        n_match = sum(1 for a, b in zip(baseline, tokens) if a == b)
        print(f'  L{L}: {n_match}/{min(len(baseline),len(tokens))} | {"".join(tokens[:30])}...', flush=True)

        # Restore
        model.model.layers[L].mlp.down_proj.weight.data.copy_(orig_weights[L])

    # L1/L6 on diverse prompts
    print(f'\n--- L1 and L6 robustness on diverse prompts ---')
    diverse = [
        ("math", "What is 23 times 17? Think step by step.\n"),
        ("repeat", "Repeat exactly: the quick brown fox jumps over the lazy dog\n"),
        ("logic", "If all cats are mammals and some mammals swim, can cats swim?\n"),
        ("translate", "Translate to French: I love mathematics\n"),
    ]

    for L_test in [1, 6]:
        print(f'\n  L{L_test} at rank 128:')
        for pname, prompt in diverse:
            base_tok = generate(model, tok, prompt, max_tok=20)
            # Truncate
            with torch.no_grad():
                W = model.model.layers[L_test].mlp.down_proj.weight.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                W_t = U[:, :128] @ torch.diag(S[:128]) @ Vt[:128, :]
                model.model.layers[L_test].mlp.down_proj.weight.copy_(W_t.half())
            trunc_tok = generate(model, tok, prompt, max_tok=20)
            n_match = sum(1 for a, b in zip(base_tok, trunc_tok) if a == b)
            print(f'    {pname:>10s}: {n_match}/{min(len(base_tok),len(trunc_tok))} | {"".join(trunc_tok[:15])}...', flush=True)
            # Restore
            model.model.layers[L_test].mlp.down_proj.weight.data.copy_(orig_weights[L_test])


if __name__ == '__main__':
    main()
