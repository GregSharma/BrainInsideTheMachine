"""probe_attn_repeat_v2: Fix content position matching, focus on attention PATTERNS."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'Qwen/Qwen2.5-3B'
DEVICE = 'cuda'

def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('loading model (eager)...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE,
        trust_remote_code=True, attn_implementation='eager')
    model.eval()

    content = "the quick brown fox jumps over the lazy dog"
    repeat_prompt = f"Repeat exactly: {content}\n"

    ids = tokenizer(repeat_prompt, return_tensors='pt').input_ids.to(DEVICE)
    prompt_len = ids.shape[1]
    prompt_tokens = [tokenizer.decode(ids[0, i]) for i in range(prompt_len)]
    print(f'prompt ({prompt_len} tok): {prompt_tokens}', flush=True)

    # Find content tokens by matching token IDs directly
    content_ids = tokenizer(content, return_tensors='pt').input_ids[0]
    content_tokens = [tokenizer.decode(content_ids[i]) for i in range(len(content_ids))]

    # Search for content subsequence in prompt
    content_start = None
    for s in range(prompt_len - len(content_ids) + 1):
        match = True
        for k in range(len(content_ids)):
            # Compare decoded tokens (handles capitalization/spacing)
            pt = tokenizer.decode(ids[0, s+k]).strip().lower()
            ct = tokenizer.decode(content_ids[k]).strip().lower()
            if pt != ct:
                match = False
                break
        if match:
            content_start = s
            break

    if content_start is None:
        # Fallback: find by decoded text matching
        print('exact match failed, trying fuzzy...', flush=True)
        for s in range(prompt_len):
            if prompt_tokens[s].strip().lower() == 'the':
                content_start = s
                break

    print(f'content starts at pos {content_start}', flush=True)
    if content_start is not None:
        print(f'prompt[{content_start}:{content_start+len(content_tokens)}] = '
              f'{prompt_tokens[content_start:content_start+len(content_tokens)]}', flush=True)

    # Generate with attention capture
    KEY_LAYERS = [13, 26, 33, 35]

    print(f'\n{"="*70}')
    print('ATTENTION DURING REPEAT GENERATION')
    print(f'{"="*70}')

    gen_tokens = []

    for step in range(12):
        with torch.inference_mode():
            out = model(ids, output_attentions=True)

        next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        next_tok = tokenizer.decode(next_id[0, 0])
        gen_tokens.append(next_tok)

        # For each key layer, get attention from last position to all prompt positions
        print(f'\n  step {step}: generating "{next_tok}"' +
              (f'  (expect "{content_tokens[step]}")' if step < len(content_tokens) else ''), flush=True)

        for L in KEY_LAYERS:
            if L >= len(out.attentions):
                continue
            attn = out.attentions[L][0].float().cpu().numpy()  # (n_heads, seq, seq)
            # Attention from last position
            attn_last = attn[:, -1, :]  # (n_heads, seq)
            avg = attn_last.mean(axis=0)  # (seq,)

            # Top 3 attended positions in the PROMPT
            prompt_attn = avg[:prompt_len]
            top3 = np.argsort(prompt_attn)[-3:][::-1]
            top3_str = ', '.join(f'pos{p}("{prompt_tokens[p]}")={prompt_attn[p]:.3f}' for p in top3)

            # If we have content positions, show attention to them
            if content_start is not None:
                n_ct = min(len(content_tokens), prompt_len - content_start)
                ca = avg[content_start:content_start + n_ct]
                max_cp = np.argmax(ca)
                # Also compute "correct position" attention (step i should attend to content pos i)
                correct_attn = ca[step] if step < n_ct else 0.0
                print(f'    L{L:2d}: top3=[{top3_str}]')
                print(f'          content_max=pos{max_cp}("{content_tokens[max_cp] if max_cp < len(content_tokens) else "?"}")'
                      f'={ca[max_cp]:.4f}  correct_pos_attn={correct_attn:.4f}')
            else:
                print(f'    L{L:2d}: top3=[{top3_str}]')

        if next_id.item() == tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, next_id], dim=1)

    print(f'\ngenerated: {gen_tokens}')
    print(f'expected:  {content_tokens}')


if __name__ == '__main__':
    main()
