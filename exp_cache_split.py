"""exp_cache_split: Anti-echo via frozen top-layer KV cache.

The hypothesis: loops form because the model attends to its own generation
output, creating echo. If top layers (above the convention boundary L*=13)
can only attend to the prompt-time KV cache, they can't echo.

Mechanism: After encoding the prompt, snapshot the KV cache at layers L*..L_out.
During generation, after each step, RESTORE those layers' cache to the snapshot
(discarding the new token's KV entry). Bottom layers see full cache.

This is the bun inversion turned inside-out: bottom layers attend full context
(messy, self-referential, the blackboard), top layers see only the original
question plus the residual stream rising from below.

Tested on base model (where deflation works and loops happen).
"""
import json, re, time, copy
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
N_LAYERS = 36
D_MODEL = 2048
MAX_TOKENS = 1200
DEVICE = 'cuda'

# System prompt and P12 (base model uses raw chat template)
SYS = ('You are solving an AMC 12A multiple choice math problem. Think step by step, '
       'show your work, then clearly state your final answer as (A), (B), (C), (D), or (E).')
P12_TEXT = (
    "The harmonic mean of a collection of numbers is the reciprocal of the "
    "arithmetic mean of the reciprocals of the numbers in the collection. "
    "For example, the harmonic mean of 4, 4, and 5 is\n\n"
    "1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n"
    "What is the harmonic mean of all the real roots of the 4050th degree "
    "polynomial\n\n"
    r"\prod_{k=1}^{2025} (kx^2 - 4x - 3) = "
    "(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)..."
    "(2025x^2 - 4x - 3)?\n\n"
    "(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3"
)
PROMPT = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"
CORRECT = 'B'  # -3/2


def extract_answer(text):
    """Extract answer letter from generation."""
    # Check for explicit (B) style
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if f'({letter})' in text[-200:]:
            return letter
    # Check for answer mentions
    answer_map = {
        '-3/2': 'B', '-1.5': 'B',
        '-5/3': 'A',
        '-6/5': 'C', '-1.2': 'C',
        '-5/6': 'D',
        '-2/3': 'E',
    }
    for pattern, letter in answer_map.items():
        if pattern in text[-300:]:
            return letter
    return '?'


def classify_output(text):
    """Classify output as CORRECT, WRONG(X), or LOOP."""
    letter = extract_answer(text)
    if letter == CORRECT:
        return f'CORRECT({letter})'
    elif letter != '?':
        return f'WRONG({letter})'
    # Check for loop: repeated substring pattern
    if len(text) > 500:
        chunk = text[-200:]
        for plen in range(10, 60):
            pattern = chunk[-plen:]
            if pattern in chunk[:-plen]:
                return 'LOOP'
    return f'UNKNOWN'


def generate_with_cache_split(model, tokenizer, prompt, split_layer, split_window=None):
    """Generate with KV cache frozen at layers >= split_layer.

    Args:
        model: the model
        tokenizer: tokenizer
        prompt: full prompt string
        split_layer: layers >= this get frozen cache (0 = freeze all = pure prompt attention)
        split_window: if set, only freeze for first N tokens, then release (None = always frozen)

    Returns: (text, n_tokens, time_s)
    """
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
    prompt_len = input_ids.shape[1]
    gen_ids = []
    past_kv = None
    prompt_snapshot = None  # frozen cache for top layers
    t0 = time.time()

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                # Encoding pass
                out = model(input_ids=input_ids, use_cache=True)
                past_kv = out.past_key_values

                # Snapshot the prompt-only KV cache for top layers
                if split_layer < N_LAYERS:
                    # DynamicCache: past_kv.layers[i].keys / .values
                    # Shape: (batch, n_kv_heads, seq_len, head_dim)
                    prompt_snapshot = {}
                    for L in range(split_layer, N_LAYERS):
                        prompt_snapshot[L] = (
                            past_kv.layers[L].keys.clone(),
                            past_kv.layers[L].values.clone(),
                        )
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values

                # RESTORE top layers to prompt-only snapshot
                if prompt_snapshot is not None:
                    should_freeze = True
                    if split_window is not None and step > split_window:
                        should_freeze = False

                    if should_freeze:
                        for L in range(split_layer, N_LAYERS):
                            snap_k, snap_v = prompt_snapshot[L]
                            # Overwrite the cache — trim back to prompt length
                            past_kv.layers[L].keys = snap_k.clone()
                            past_kv.layers[L].values = snap_v.clone()

            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()

            # Stop on EOS
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, len(gen_ids), time.time() - t0


def main():
    print('=' * 60)
    print('Exp Cache-Split: Anti-Echo via Frozen Top-Layer KV')
    print('  The frog doesn\'t listen to itself.')
    print('=' * 60)

    # Load BASE model (not instruct — loops happen on base)
    print(f'\nLoading {MODEL_NAME}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print(f'  Loaded.', flush=True)

    results = {}

    # === Condition 1: Baseline (no intervention) ===
    print('\n--- Baseline ---', flush=True)
    text, ntok, t = generate_with_cache_split(model, tokenizer, PROMPT, split_layer=N_LAYERS)  # N_LAYERS = no split
    outcome = classify_output(text)
    results['baseline'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  baseline: {outcome}, {ntok} tokens, {t:.1f}s', flush=True)

    # === Condition 2: Full split at L13 (convention boundary) ===
    print('\n--- Cache split at L13 (convention boundary) ---', flush=True)
    text, ntok, t = generate_with_cache_split(model, tokenizer, PROMPT, split_layer=13)
    outcome = classify_output(text)
    results['split_L13'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  split_L13: {outcome}, {ntok} tokens, {t:.1f}s', flush=True)

    # === Condition 3: Split at L20 (cooperative zone entrance) ===
    print('\n--- Cache split at L20 ---', flush=True)
    text, ntok, t = generate_with_cache_split(model, tokenizer, PROMPT, split_layer=20)
    outcome = classify_output(text)
    results['split_L20'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  split_L20: {outcome}, {ntok} tokens, {t:.1f}s', flush=True)

    # === Condition 4: Split at L27 (commitment point) ===
    print('\n--- Cache split at L27 ---', flush=True)
    text, ntok, t = generate_with_cache_split(model, tokenizer, PROMPT, split_layer=27)
    outcome = classify_output(text)
    results['split_L27'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  split_L27: {outcome}, {ntok} tokens, {t:.1f}s', flush=True)

    # === Condition 5: Split at L33 (read head only) ===
    print('\n--- Cache split at L33 (read head only) ---', flush=True)
    text, ntok, t = generate_with_cache_split(model, tokenizer, PROMPT, split_layer=33)
    outcome = classify_output(text)
    results['split_L33'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  split_L33: {outcome}, {ntok} tokens, {t:.1f}s', flush=True)

    # === Condition 6: Split ALL layers (L0) — only sees prompt, ever ===
    print('\n--- Cache split at L0 (all layers frozen) ---', flush=True)
    text, ntok, t = generate_with_cache_split(model, tokenizer, PROMPT, split_layer=0)
    outcome = classify_output(text)
    results['split_L0_all'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  split_L0_all: {outcome}, {ntok} tokens, {t:.1f}s', flush=True)

    # === Condition 7: Split L13 but only for first 50 tokens ===
    print('\n--- Cache split L13, first 50 tokens only ---', flush=True)
    text, ntok, t = generate_with_cache_split(model, tokenizer, PROMPT, split_layer=13, split_window=50)
    outcome = classify_output(text)
    results['split_L13_window50'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  split_L13_window50: {outcome}, {ntok} tokens, {t:.1f}s', flush=True)

    # === Condition 8: QK deflation control (proven to work) ===
    # Import the proven WindowedDeflation pattern
    print('\n--- QK Deflation control (proven) ---', flush=True)
    # We'll implement inline to avoid import issues
    from exp_delayed_deflation_p12 import WindowedDeflation
    deflator = WindowedDeflation(model, layers=list(range(20, 36)), r=4, alpha=0.1,
                                  refresh_every=25, active_from=0, active_until=None)
    # Manual generation with deflation
    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None
    t0 = time.time()
    deflator.start_gen()
    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)
            deflator.tick(past_kv)
    deflator.remove()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    ntok = len(gen_ids)
    t = time.time() - t0
    outcome = classify_output(text)
    results['qk_deflation'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  qk_deflation: {outcome}, {ntok} tokens, {t:.1f}s', flush=True)

    # === Summary ===
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    total_time = sum(r['time_s'] for r in results.values())
    print(f'Total time: {total_time:.1f}s')
    print()
    print(f'{"Condition":25s} {"Outcome":15s} {"Tokens":>7s}')
    print('-' * 50)
    for name, r in results.items():
        print(f'{name:25s} {r["outcome"]:15s} {r["n_tokens"]:>7d}')

    # Save
    outpath = OUTPUT_DIR / 'exp_cache_split.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
