"""probe_attn_repeat: Where does the model look during repetition?

Capture attention distributions during generation of 'repeat: [sentence]'.
For each generated token, record which input positions get highest attention.
If the model is attending to position i when generating word i of the repetition,
verbatim lives in attention patterns.

Also: test whether attention output is constant across task types
(repeat vs reasoning vs query) - extending C6b beyond math.
"""
import json, time, sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
DEVICE = 'cuda'


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('='*70)
    print('PROBE ATTN REPEAT: where does the model look during repetition?')
    print('='*70)

    print('loading model (eager attention for output_attentions)...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE,
        trust_remote_code=True, attn_implementation='eager')
    model.eval()
    print('ready.', flush=True)

    content = "the quick brown fox jumps over the lazy dog"
    repeat_prompt = f"Repeat exactly: {content}\n"

    # Encode the repeat prompt
    ids = tokenizer(repeat_prompt, return_tensors='pt').input_ids.to(DEVICE)
    prompt_len = ids.shape[1]
    prompt_tokens = [tokenizer.decode(ids[0, i]) for i in range(prompt_len)]
    print(f'\nprompt ({prompt_len} tokens): {prompt_tokens}', flush=True)

    # Find where the content starts in the prompt
    content_ids = tokenizer(content, return_tensors='pt').input_ids[0]
    content_tokens = [tokenizer.decode(content_ids[i]) for i in range(len(content_ids))]
    print(f'content ({len(content_tokens)} tokens): {content_tokens}', flush=True)

    # Find content start position in prompt
    content_start = None
    for start in range(prompt_len - len(content_tokens) + 1):
        if all(ids[0, start + k].item() == content_ids[k].item() for k in range(len(content_tokens))):
            content_start = start
            break
    print(f'content starts at position {content_start} in prompt', flush=True)

    # === PART 1: ATTENTION DURING GENERATION ===
    print(f'\n{"="*70}')
    print('PART 1: attention patterns during repeat generation')
    print(f'{"="*70}')

    gen_data = []  # per step: {token, attn_to_content_positions, top5_positions}

    KEY_LAYERS = [5, 13, 18, 26, 30, 33, 35]

    for step in range(15):  # generate up to 15 tokens
        with torch.inference_mode():
            out = model(ids, output_attentions=True)

        # out.attentions is tuple of (batch, n_heads, seq, seq) per layer
        # we want the attention FROM the last position TO all positions
        attn_data = {}
        for L in KEY_LAYERS:
            if L < len(out.attentions):
                attn_L = out.attentions[L][0]  # (n_heads, seq, seq)
                # attention from last position to all positions
                attn_last = attn_L[:, -1, :].float().cpu().numpy()  # (n_heads, seq)
                # average across heads
                attn_avg = attn_last.mean(axis=0)  # (seq,)
                attn_data[L] = {
                    'avg': attn_avg,
                    'per_head': attn_last,
                }

        # greedy next token
        next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        next_tok = tokenizer.decode(next_id[0, 0])

        # For each layer, find where attention is focused
        step_info = {'step': step, 'token': next_tok, 'layers': {}}
        for L in KEY_LAYERS:
            if L in attn_data:
                avg = attn_data[L]['avg']
                seq_len = len(avg)
                # top 5 attended positions
                top5_idx = np.argsort(avg)[-5:][::-1]
                top5_vals = avg[top5_idx]
                # attention to each content position
                if content_start is not None:
                    content_attn = avg[content_start:content_start + len(content_tokens)]
                    max_content_pos = np.argmax(content_attn)
                    max_content_val = content_attn[max_content_pos]
                else:
                    content_attn = None
                    max_content_pos = -1
                    max_content_val = 0

                step_info['layers'][L] = {
                    'top5_pos': top5_idx.tolist(),
                    'top5_val': top5_vals.tolist(),
                    'max_content_pos': int(max_content_pos),
                    'max_content_val': float(max_content_val),
                    'content_attn': content_attn.tolist() if content_attn is not None else None,
                }

        gen_data.append(step_info)

        if next_id.item() == tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, next_id], dim=1)

    # Print results
    print(f'\n  generated tokens: {[d["token"] for d in gen_data]}')
    print(f'  content tokens:   {content_tokens}')

    print(f'\n  --- where does the model attend when generating each repeat token? ---')
    print(f'  (showing which CONTENT position gets max attention at L33)')
    print(f'  {"step":>4s} {"gen_tok":>10s} {"expect":>10s} {"max_cpos":>8s} {"max_cval":>8s} | L5_cpos L13_cpos L26_cpos L33_cpos L35_cpos')

    for i, d in enumerate(gen_data):
        expect_tok = content_tokens[i] if i < len(content_tokens) else '---'
        l33 = d['layers'].get(33, {})
        l5 = d['layers'].get(5, {})
        l13 = d['layers'].get(13, {})
        l26 = d['layers'].get(26, {})
        l35 = d['layers'].get(35, {})

        print(f'  {i:>4d} {d["token"]:>10s} {expect_tok:>10s}'
              f' {l33.get("max_content_pos", -1):>8d} {l33.get("max_content_val", 0):>8.4f}'
              f' | {l5.get("max_content_pos", -1):>4d}'
              f'    {l13.get("max_content_pos", -1):>4d}'
              f'    {l26.get("max_content_pos", -1):>4d}'
              f'    {l33.get("max_content_pos", -1):>4d}'
              f'    {l35.get("max_content_pos", -1):>4d}')

    # Print full content attention distribution at L33 for first few steps
    print(f'\n  --- full content attention distribution at L33 ---')
    print(f'  content tokens: {content_tokens}')
    for i, d in enumerate(gen_data[:len(content_tokens)]):
        l33 = d['layers'].get(33, {})
        ca = l33.get('content_attn', None)
        if ca:
            dist_str = ' '.join(f'{v:.3f}' for v in ca)
            print(f'  step {i} (gen="{d["token"]}"):  [{dist_str}]')

    # === PART 2: IS ATTENTION OUTPUT CONSTANT ACROSS TASK TYPES? ===
    print(f'\n{"="*70}')
    print('PART 2: attention output constancy across task types')
    print('  (extending C6b beyond math to repeat/query/reasoning)')
    print(f'{"="*70}')

    # Capture attention OUTPUT (not pattern) at last token for different tasks
    # Need hooks for this
    class AttnOutCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            # self_attn returns (attn_output, attn_weights, past_key_value)
            # but the layer's forward hook sees the full layer output
            # we need to hook the self_attn module specifically
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[0, -1].detach().float().cpu().numpy()

    # Hook the self_attn output projection (o_proj) to get attention output before residual
    class OProjCap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            self.out = output[0, -1].detach().float().cpu().numpy()

    caps = {}
    hooks = []
    for L in [26, 30, 33]:
        cap = OProjCap()
        caps[L] = cap
        hooks.append(model.model.layers[L].self_attn.o_proj.register_forward_hook(cap))

    task_prompts = [
        ("bare",   content),
        ("repeat", f"Repeat exactly: {content}"),
        ("reason", f"{content}. How many words?"),
        ("query",  f"What was the third word? {content}"),
        ("logic",  f"all the roses in my garden are red. i got a flower from my garden. must it be a rose?"),
        ("math",   f"What is 23 times 17?"),
    ]

    attn_outputs = {}
    for name, text in task_prompts:
        ids_t = tokenizer(text, return_tensors='pt').input_ids.to(DEVICE)
        with torch.inference_mode():
            model(ids_t)
        attn_outputs[name] = {L: caps[L].out.copy() for L in [26, 30, 33]}
        print(f'  {name:8s}: {ids_t.shape[1]} tokens', flush=True)

    for h in hooks:
        h.remove()

    # Pairwise cosine of attention OUTPUTS
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10: return 0.0
        return float(np.dot(a, b) / (na * nb))

    print(f'\n  attention OUTPUT cosines at last token (o_proj output):')
    names = [n for n, _ in task_prompts]
    print(f'  {"":20s}  L26      L30      L33')
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            label = f'{names[i]}-{names[j]}'
            c26 = cos(attn_outputs[names[i]][26], attn_outputs[names[j]][26])
            c30 = cos(attn_outputs[names[i]][30], attn_outputs[names[j]][30])
            c33 = cos(attn_outputs[names[i]][33], attn_outputs[names[j]][33])
            print(f'  {label:20s}  {c26:.4f}   {c30:.4f}   {c33:.4f}')

    # Mean and std of cosines
    all_cos = {L: [] for L in [26, 30, 33]}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            for L in [26, 30, 33]:
                all_cos[L].append(cos(attn_outputs[names[i]][L], attn_outputs[names[j]][L]))
    print(f'\n  mean pairwise cos: L26={np.mean(all_cos[26]):.4f}  L30={np.mean(all_cos[30]):.4f}  L33={np.mean(all_cos[33]):.4f}')
    print(f'  std pairwise cos:  L26={np.std(all_cos[26]):.4f}  L30={np.std(all_cos[30]):.4f}  L33={np.std(all_cos[33]):.4f}')

    print(f'\n  (C6b found constant bias for math. If cos > 0.95 here too, it generalizes.)')
    print(f'  (If cos < 0.90, attention output varies by task type = task info in attention.)')


if __name__ == '__main__':
    main()
