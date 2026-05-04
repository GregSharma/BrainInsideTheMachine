"""exp_convention_qk_deflation: Deflate queries along the convention direction e_c.

Blind SVD deflation works but has resonance holes at delta=0.12 and 0.20.
MS1 surgery removes e_c from W_down and improves accuracy.

This experiment bridges the two: instead of deflating queries along the
top SVD directions of the KV cache (blind), deflate along e_c specifically.
The hypothesis: convention is the specific echo direction. Removing it from
the query space should be more targeted than blind SVD.

Conditions on P12 (base model):
1. Baseline (loop expected)
2. Blind SVD deflation (proven, control)
3. Convention-targeted: deflate Q along e_c at each layer
4. Convention-targeted + blind SVD: both together
5. Random direction control: deflate Q along a random fixed direction
6. Convention-targeted on 20 math problems (not just P12)
"""
import json, re, time, copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
N_LAYERS = 36
D_MODEL = 2048
DEVICE = 'cuda'
MAX_TOKENS = 1200

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
CORRECT = 'B'

# Math problem system prompt for multi-problem eval
MATH_SYS = 'You are a careful mathematical reasoner. Think step by step, show your work clearly, and then state the final numerical answer on its own line.'

from exp_delayed_deflation_p12 import WindowedDeflation


def get_test_problems():
    """Same 20 problems as cross-model surgery."""
    return [
        {"en": "Solve for x: 3x + 7 = 22", "zh": "\u89e3\u65b9\u7a0b\uff1a3x + 7 = 22", "answer": 5, "category": "algebra"},
        {"en": "Solve for x: 2x\u00b2 - 8 = 0", "zh": "\u89e3\u65b9\u7a0b\uff1a2x\u00b2 - 8 = 0", "answer": 2, "category": "algebra"},
        {"en": "Simplify: (x + 3)(x - 3)", "zh": "\u5316\u7b80\uff1a(x + 3)(x - 3)", "answer": "x\u00b2 - 9", "category": "algebra"},
        {"en": "Solve: |2x - 5| = 3", "zh": "\u89e3\u65b9\u7a0b\uff1a|2x - 5| = 3", "answer": 4, "category": "algebra"},
        {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97\uff1a347 + 658", "answer": 1005, "category": "arithmetic"},
        {"en": "Calculate: 1000 - 387", "zh": "\u8ba1\u7b97\uff1a1000 - 387", "answer": 613, "category": "arithmetic"},
        {"en": "Calculate: 23 \u00d7 17", "zh": "\u8ba1\u7b97\uff1a23 \u00d7 17", "answer": 391, "category": "arithmetic"},
        {"en": "Calculate: 1728 \u00f7 12", "zh": "\u8ba1\u7b97\uff1a1728 \u00f7 12", "answer": 144, "category": "arithmetic"},
        {"en": "Find the area of a circle with radius 7 (use \u03c0 \u2248 22/7)", "zh": "\u6c42\u534a\u5f84\u4e3a7\u7684\u5706\u7684\u9762\u79ef\uff08\u4f7f\u7528\u03c0 \u2248 22/7\uff09", "answer": 154, "category": "geometry"},
        {"en": "Find the hypotenuse of a right triangle with legs 5 and 12", "zh": "\u6c42\u76f4\u89d2\u4e09\u89d2\u5f62\u7684\u659c\u8fb9\uff0c\u4e24\u76f4\u89d2\u8fb9\u5206\u522b\u4e3a5\u548c12", "answer": 13, "category": "geometry"},
        {"en": "What is the perimeter of a rectangle with length 15 and width 8?", "zh": "\u957f15\u5bbd8\u7684\u77e9\u5f62\u7684\u5468\u957f\u662f\u591a\u5c11\uff1f", "answer": 46, "category": "geometry"},
        {"en": "Find the volume of a cube with side length 6", "zh": "\u6c42\u8fb9\u957f\u4e3a6\u7684\u6b63\u65b9\u4f53\u7684\u4f53\u79ef", "answer": 216, "category": "geometry"},
        {"en": "What is the GCD of 84 and 120?", "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570\u662f\u591a\u5c11\uff1f", "answer": 12, "category": "number_theory"},
        {"en": "Is 97 prime? Answer yes or no, then explain.", "zh": "97\u662f\u8d28\u6570\u5417\uff1f\u56de\u7b54\u662f\u6216\u5426\uff0c\u7136\u540e\u89e3\u91ca\u3002", "answer": "yes", "category": "number_theory"},
        {"en": "Find the remainder when 2^10 is divided by 7", "zh": "\u6c422^10\u9664\u4ee57\u7684\u4f59\u6570", "answer": 2, "category": "number_theory"},
        {"en": "What is the sum of all prime numbers less than 20?", "zh": "\u6c4220\u4ee5\u5185\u6240\u6709\u8d28\u6570\u7684\u548c", "answer": 77, "category": "number_theory"},
        {"en": "How many ways can you choose 3 items from 7?", "zh": "\u4ece7\u4e2a\u7269\u54c1\u4e2d\u9009\u62e93\u4e2a\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f", "answer": 35, "category": "combinatorics"},
        {"en": "How many ways can 5 people stand in a line?", "zh": "5\u4e2a\u4eba\u7ad9\u6210\u4e00\u6392\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f", "answer": 120, "category": "combinatorics"},
        {"en": "Calculate: 8! / (5! \u00d7 3!)", "zh": "\u8ba1\u7b97\uff1a8! / (5! \u00d7 3!)", "answer": 56, "category": "combinatorics"},
        {"en": "How many 3-digit numbers have all distinct digits?", "zh": "\u6709\u591a\u5c11\u4e2a\u4e09\u4f4d\u6570\u7684\u5404\u4f4d\u6570\u5b57\u5747\u4e0d\u76f8\u540c\uff1f", "answer": 648, "category": "combinatorics"},
    ]


def strict_check_answer(text, correct):
    if correct in ("yes", "no"):
        return correct.lower() in text.lower()
    if isinstance(correct, str) and not correct.replace('-','').replace('.','').isdigit():
        return correct in text
    target = str(correct)
    pattern = r'(?<![\d.])' + re.escape(target) + r'(?![\d.])'
    return bool(re.search(pattern, text))


def extract_encoding_activations(model, tokenizer, problems, lang):
    """Get last hidden state at each layer for problems in given language."""
    all_acts = {L: [] for L in range(N_LAYERS)}

    class Cap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[:, -1, :].detach().float().cpu().numpy() if h.dim() == 3 else h[-1:, :].detach().float().cpu().numpy()

    caps = [Cap() for _ in range(N_LAYERS)]
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in range(N_LAYERS)]

    for prob in problems:
        text = prob[lang]
        messages = [{"role": "system", "content": MATH_SYS}, {"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        with torch.inference_mode():
            model(ids)
        for L in range(N_LAYERS):
            all_acts[L].append(caps[L].out.squeeze())

    for h in hooks:
        h.remove()
    for L in range(N_LAYERS):
        all_acts[L] = np.stack(all_acts[L])
    return all_acts


class ConventionQDeflation:
    """Deflate queries along the convention direction e_c at specified layers.

    Unlike blind SVD which deflates along the cache's top directions,
    this deflates along a fixed direction (e_c) that represents language convention.
    """
    def __init__(self, model, e_c_per_layer, layers, alpha=0.1, active_from=0, active_until=None):
        self.model = model
        self.e_c = {}  # {layer: e_c tensor on device, shape (head_dim,) per KV head}
        self.target_layers = set(layers)
        self.alpha = alpha
        self.active_from = active_from
        self.active_until = active_until
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.deflation_active = False

        # e_c is in residual stream space (d_model=2048).
        # We need to project it into each Q head's space.
        # Q = x @ W_Q, so the direction in Q space is e_c @ W_Q reshaped per head.
        n_heads = 16
        head_dim = 128
        n_kv = 2
        gs = n_heads // n_kv  # 8 Q heads per KV head

        for L in layers:
            e_c_np = e_c_per_layer[L]  # (d_model,)
            e_c_t = torch.tensor(e_c_np, dtype=torch.float16, device=DEVICE)

            # Project e_c through W_Q to get per-head convention directions
            W_Q = model.model.layers[L].self_attn.q_proj.weight.data  # (n_heads*head_dim, d_model)
            # e_c in Q space: W_Q @ e_c -> (n_heads*head_dim,)
            e_c_q = W_Q @ e_c_t  # (n_heads * head_dim,)
            e_c_q = e_c_q.view(n_heads, head_dim)  # (n_heads, head_dim)

            # Normalize per head
            norms = e_c_q.norm(dim=1, keepdim=True).clamp(min=1e-8)
            e_c_q = e_c_q / norms

            self.e_c[L] = e_c_q  # (n_heads, head_dim)

        self._install()

    def _install(self):
        for L in self.target_layers:
            h = self.model.model.layers[L].self_attn.q_proj.register_forward_hook(
                self._make_hook(L))
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or not self.deflation_active or li not in self.e_c:
                return output
            batch, seq, d = output.shape
            n_heads = 16
            head_dim = 128
            tensor = output.view(batch, seq, n_heads, head_dim)

            e_c_q = self.e_c[li]  # (n_heads, head_dim)
            # Project out e_c from each head's query
            # proj = (q . e_c) * e_c for each head
            dots = (tensor * e_c_q.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True)  # (batch, seq, n_heads, 1)
            proj = dots * e_c_q.unsqueeze(0).unsqueeze(0)  # (batch, seq, n_heads, head_dim)
            tensor = tensor - self.alpha * proj

            return tensor.view(batch, seq, d)
        return hook

    def start_gen(self):
        self.is_generating = True
        self.step_count = 0
        self._update_active()

    def _update_active(self):
        after_from = self.step_count >= self.active_from
        before_until = (self.active_until is None) or (self.step_count <= self.active_until)
        self.deflation_active = after_from and before_until

    def tick(self):
        self.step_count += 1
        self._update_active()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def classify_output(text):
    letter = '?'
    answer_map = {'-3/2': 'B', '-1.5': 'B', '-5/3': 'A', '-6/5': 'C', '-1.2': 'C', '-5/6': 'D', '-2/3': 'E'}
    for pat in ['(B)', '(A)', '(C)', '(D)', '(E)']:
        if pat in text[-300:]:
            letter = pat[1]
            break
    if letter == '?':
        for pattern, l in answer_map.items():
            if pattern in text[-300:]:
                letter = l
                break
    # Check for boxed answer
    import re as _re
    boxed = _re.findall(r'boxed\{([^}]+)\}', text[-500:])
    if boxed:
        for pattern, l in answer_map.items():
            if pattern in boxed[-1] or pattern.replace('/', '') in boxed[-1].replace(' ', '').replace('\\frac{', '').replace('}', ''):
                letter = l
                break
        # Check frac format
        if '-\\frac{3}{2}' in boxed[-1] or '-frac{3}{2}' in boxed[-1]:
            letter = 'B'

    if letter == CORRECT:
        return f'CORRECT({letter})'
    elif letter != '?':
        return f'WRONG({letter})'
    if len(text) > 500:
        chunk = text[-200:]
        for plen in range(10, 60):
            p = chunk[-plen:]
            if p in chunk[:-plen]:
                return 'LOOP'
    return 'UNKNOWN'


def generate_p12(model, tokenizer, deflator=None, conv_deflator=None):
    """Generate P12 with optional deflation."""
    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None
    t0 = time.time()

    if deflator:
        deflator.start_gen()
    if conv_deflator:
        conv_deflator.start_gen()

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
            if deflator:
                deflator.tick(past_kv)
            if conv_deflator:
                conv_deflator.tick()

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, len(gen_ids), time.time() - t0


def main():
    print('=' * 60)
    print('Exp Convention-Targeted QK Deflation')
    print('  The convention direction meets the query space.')
    print('=' * 60)

    print(f'\nLoading {MODEL_NAME}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print('  Loaded.', flush=True)

    # Extract convention directions
    print('\nExtracting convention directions...', flush=True)
    problems = get_test_problems()
    en_acts = extract_encoding_activations(model, tokenizer, problems, 'en')
    zh_acts = extract_encoding_activations(model, tokenizer, problems, 'zh')
    e_c = {}
    for L in range(N_LAYERS):
        diff = zh_acts[L].mean(axis=0) - en_acts[L].mean(axis=0)
        norm = np.linalg.norm(diff)
        e_c[L] = diff / (norm + 1e-12)
    print('  Done.', flush=True)

    # Generate random direction for control
    rng = np.random.RandomState(42)
    e_rand = {}
    for L in range(N_LAYERS):
        v = rng.randn(D_MODEL).astype(np.float32)
        e_rand[L] = v / np.linalg.norm(v)

    results = {}
    DEFLATION_LAYERS = list(range(20, 36))  # same as blind SVD
    ALPHA_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.30]

    # === 1. Baseline ===
    print('\n--- Baseline ---', flush=True)
    text, ntok, t = generate_p12(model, tokenizer)
    outcome = classify_output(text)
    results['baseline'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    print(f'  {outcome}, {ntok} tok, {t:.1f}s', flush=True)

    # === 2. Blind SVD (control) ===
    print('\n--- Blind SVD deflation (alpha=0.10) ---', flush=True)
    deflator = WindowedDeflation(model, layers=DEFLATION_LAYERS, r=4, alpha=0.1,
                                  refresh_every=25, active_from=0, active_until=None)
    text, ntok, t = generate_p12(model, tokenizer, deflator=deflator)
    outcome = classify_output(text)
    results['blind_svd'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    deflator.remove()
    print(f'  {outcome}, {ntok} tok, {t:.1f}s', flush=True)

    # === 3. Convention-targeted sweep ===
    for alpha in ALPHA_SWEEP:
        label = f'conv_a{alpha}'
        print(f'\n--- Convention QK deflation (alpha={alpha}) ---', flush=True)
        conv = ConventionQDeflation(model, e_c, DEFLATION_LAYERS, alpha=alpha)
        text, ntok, t = generate_p12(model, tokenizer, conv_deflator=conv)
        outcome = classify_output(text)
        results[label] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
        conv.remove()
        print(f'  {outcome}, {ntok} tok, {t:.1f}s', flush=True)

    # === 4. Random direction control ===
    print('\n--- Random direction QK deflation (alpha=0.10) ---', flush=True)
    rand_conv = ConventionQDeflation(model, e_rand, DEFLATION_LAYERS, alpha=0.10)
    text, ntok, t = generate_p12(model, tokenizer, conv_deflator=rand_conv)
    outcome = classify_output(text)
    results['random_dir'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    rand_conv.remove()
    print(f'  {outcome}, {ntok} tok, {t:.1f}s', flush=True)

    # === 5. Convention + blind SVD together ===
    print('\n--- Convention + blind SVD (both alpha=0.10) ---', flush=True)
    deflator2 = WindowedDeflation(model, layers=DEFLATION_LAYERS, r=4, alpha=0.1,
                                   refresh_every=25, active_from=0, active_until=None)
    conv2 = ConventionQDeflation(model, e_c, DEFLATION_LAYERS, alpha=0.10)
    text, ntok, t = generate_p12(model, tokenizer, deflator=deflator2, conv_deflator=conv2)
    outcome = classify_output(text)
    results['conv_plus_svd'] = {'outcome': outcome, 'n_tokens': ntok, 'time_s': t, 'text_tail': text[-300:]}
    deflator2.remove()
    conv2.remove()
    print(f'  {outcome}, {ntok} tok, {t:.1f}s', flush=True)

    # === Summary ===
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'{"Condition":25s} {"Outcome":15s} {"Tokens":>7s}')
    print('-' * 50)
    for name, r in results.items():
        print(f'{name:25s} {r["outcome"]:15s} {r["n_tokens"]:>7d}')

    outpath = OUTPUT_DIR / 'exp_convention_qk_deflation.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
