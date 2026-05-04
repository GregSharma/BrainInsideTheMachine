"""exp_silu_commitment: Track SiLU gate activation pattern stability during generation.

The hypothesis from the swing set conversation: the commitment point at token ~27
is when the SiLU activation pattern freezes. The same features fire, producing the
same MLP output, producing the same trajectory, selecting the same features.

Measure: at each generation step, for each MLP layer, record which SiLU gates
are above threshold. Compute hamming distance between consecutive patterns.
If there's a sharp drop at the commitment point, the loop IS the frozen gate pattern.

Two conditions: baseline (loops) and QK deflation (correct answer).
Base model on P12.
"""
import json, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
N_LAYERS = 36
DEVICE = 'cuda'
MAX_TOKENS = 400  # enough to see commitment, don't need full 1200

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

# Import proven deflation
from exp_delayed_deflation_p12 import WindowedDeflation


class GatePatternTracker:
    """Capture SiLU gate activation patterns at every MLP during generation."""
    def __init__(self, model, layers, threshold=0.1):
        self.threshold = threshold
        self.layers = layers
        self.hooks = []
        self.step = 0
        self.is_gen = False
        # patterns[layer][step] = binary vector (8896,)
        self.patterns = {L: [] for L in layers}
        self._install(model)

    def _install(self, model):
        for L in self.layers:
            mlp = model.model.layers[L].mlp
            h = mlp.gate_proj.register_forward_hook(self._make_hook(L))
            self.hooks.append(h)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if not self.is_gen:
                return
            # output shape: (batch, seq, intermediate_size)
            # During generation, seq=1
            if output.shape[1] != 1:
                return  # skip prompt encoding
            gate_vals = torch.nn.functional.silu(output[0, 0])  # (intermediate_size,)
            pattern = (gate_vals.abs() > self.threshold).cpu().numpy()
            self.patterns[layer_idx].append(pattern)
        return hook

    def start_gen(self):
        self.is_gen = True
        self.step = 0

    def tick(self):
        self.step += 1

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def run_generation(model, tokenizer, deflator=None, tracker=None):
    """Generate P12 with optional deflation + gate tracking."""
    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None

    if tracker:
        tracker.start_gen()
    if deflator:
        deflator.start_gen()

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                if deflator:
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

            if tracker:
                tracker.tick()
            if deflator:
                deflator.tick(past_kv)

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, len(gen_ids)


def compute_hamming_trajectories(patterns, layers):
    """Compute hamming distance between consecutive gate patterns."""
    results = {}
    for L in layers:
        pats = patterns[L]
        if len(pats) < 2:
            results[L] = []
            continue
        dists = []
        for i in range(1, len(pats)):
            hamming = np.sum(pats[i] != pats[i-1])
            frac = hamming / len(pats[i])  # fraction of gates that flipped
            dists.append(float(frac))
        results[L] = dists
    return results


def compute_pattern_entropy(patterns, layers):
    """Compute fraction of active gates at each step (activation density)."""
    results = {}
    for L in layers:
        pats = patterns[L]
        densities = [float(np.mean(p)) for p in pats]
        results[L] = densities
    return results


def main():
    print('=' * 60)
    print('Exp SiLU Commitment: Gate Pattern Stability During Generation')
    print('  The swing set question: when do the gates freeze?')
    print('=' * 60)

    # Sample layers: every 5th + key layers
    sample_layers = [0, 5, 10, 13, 15, 18, 20, 22, 25, 27, 30, 33, 35]

    print(f'\nLoading {MODEL_NAME}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print(f'  Loaded.', flush=True)

    all_results = {}

    # === BASELINE ===
    print('\n--- Baseline (no intervention) ---', flush=True)
    tracker = GatePatternTracker(model, sample_layers)
    t0 = time.time()
    text, ntok = run_generation(model, tokenizer, tracker=tracker)
    t = time.time() - t0
    print(f'  {ntok} tokens, {t:.1f}s', flush=True)
    print(f'  Tail: ...{text[-100:]}', flush=True)

    hamming = compute_hamming_trajectories(tracker.patterns, sample_layers)
    density = compute_pattern_entropy(tracker.patterns, sample_layers)
    tracker.remove()

    all_results['baseline'] = {
        'n_tokens': ntok,
        'hamming': {str(L): h for L, h in hamming.items()},
        'density': {str(L): d for L, d in density.items()},
        'text_tail': text[-200:],
    }

    # Print key stats
    for L in [13, 20, 27, 30, 33]:
        h = hamming[L]
        if len(h) >= 50:
            early = np.mean(h[:25])
            late = np.mean(h[25:50])
            very_late = np.mean(h[50:100]) if len(h) >= 100 else np.mean(h[50:])
            print(f'  L{L:2d} hamming: early(0-25)={early:.4f}, mid(25-50)={late:.4f}, late(50-100)={very_late:.4f}')

    # === QK DEFLATION ===
    print('\n--- QK Deflation (correct answer) ---', flush=True)
    tracker2 = GatePatternTracker(model, sample_layers)
    deflator = WindowedDeflation(model, layers=list(range(20, 36)), r=4, alpha=0.1,
                                  refresh_every=25, active_from=0, active_until=None)
    t0 = time.time()
    text2, ntok2 = run_generation(model, tokenizer, deflator=deflator, tracker=tracker2)
    t2 = time.time() - t0
    print(f'  {ntok2} tokens, {t2:.1f}s', flush=True)
    print(f'  Tail: ...{text2[-100:]}', flush=True)

    hamming2 = compute_hamming_trajectories(tracker2.patterns, sample_layers)
    density2 = compute_pattern_entropy(tracker2.patterns, sample_layers)
    tracker2.remove()
    deflator.remove()

    all_results['deflation'] = {
        'n_tokens': ntok2,
        'hamming': {str(L): h for L, h in hamming2.items()},
        'density': {str(L): d for L, d in density2.items()},
        'text_tail': text2[-200:],
    }

    for L in [13, 20, 27, 30, 33]:
        h = hamming2[L]
        if len(h) >= 50:
            early = np.mean(h[:25])
            late = np.mean(h[25:50])
            very_late = np.mean(h[50:100]) if len(h) >= 100 else np.mean(h[50:])
            print(f'  L{L:2d} hamming: early(0-25)={early:.4f}, mid(25-50)={late:.4f}, late(50-100)={very_late:.4f}')

    # === COMPARISON ===
    print(f'\n{"=" * 60}')
    print('COMPARISON: Baseline vs Deflation')
    print(f'{"=" * 60}')
    print(f'{"Layer":>6s} {"BL early":>10s} {"BL mid":>10s} {"BL late":>10s} | {"DF early":>10s} {"DF mid":>10s} {"DF late":>10s}')
    print('-' * 75)
    for L in sample_layers:
        h_bl = hamming[L]
        h_df = hamming2[L]
        if len(h_bl) >= 50 and len(h_df) >= 50:
            bl_e = np.mean(h_bl[:25])
            bl_m = np.mean(h_bl[25:50])
            bl_l = np.mean(h_bl[50:100]) if len(h_bl) >= 100 else np.mean(h_bl[50:])
            df_e = np.mean(h_df[:25])
            df_m = np.mean(h_df[25:50])
            df_l = np.mean(h_df[50:100]) if len(h_df) >= 100 else np.mean(h_df[50:])
            print(f'  L{L:2d}  {bl_e:10.4f} {bl_m:10.4f} {bl_l:10.4f} | {df_e:10.4f} {df_m:10.4f} {df_l:10.4f}')

    # Save
    outpath = OUTPUT_DIR / 'exp_silu_commitment.json'
    with open(outpath, 'w') as f:
        json.dump(all_results, f)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
