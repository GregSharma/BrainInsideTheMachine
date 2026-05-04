"""exp_z_generation: Generate through Z-space at every step.

The idea: at each generation step, after the bottom layers compute,
project the hidden state into the convention-free subspace (remove e_c)
before passing to the top layers. This is dynamic MS1 — applied to
activations at runtime rather than to W_down statically.

MOAMS-X showed 96% correct on ZHu2192EN transplant during encoding.
This tests whether the same factorization works during generation.

Conditions:
1. Baseline (instruct model, 20 math problems)
2. Static MS1 surgery (remove e_c from W_down, proven +7)
3. Dynamic Z-projection: at layer L*, project h onto complement of e_c
4. Dynamic Z-projection + static surgery (both together)
5. Dynamic at ALL layers vs just L*
6. Dynamic with alpha sweep (partial projection)
"""
import json, re, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
N_LAYERS = 36
D_MODEL = 2048
DEVICE = 'cuda'
MAX_NEW = 512

SYS_EN = 'You are a careful mathematical reasoner. When given a problem, think step by step, show your work clearly, and then state the final numerical answer on its own line.'
SYS_ZH = 'u4f60u662fu4e00u4e2au4e25u8c28u7684u6570u5b66u63a8u7406u8005u3002u9047u5230u95eeu9898u65f6uff0cu8bf7u9010u6b65u601du8003uff0cu6e05u6670u5730u5c55u793au4f60u7684u63a8u5bfcu8fc7u7a0buff0cu7136u540eu5728u5355u72ecu7684u4e00u884cu7ed9u51fau6700u7ec8u7684u6570u503cu7b54u6848u3002'


def get_test_problems():
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


def strict_check(text, correct):
    if correct in ("yes", "no"):
        return correct.lower() in text.lower()
    if isinstance(correct, str) and not correct.replace('-','').replace('.','').isdigit():
        return correct in text
    target = str(correct)
    pattern = r'(?<![\d.])' + re.escape(target) + r'(?![\d.])'
    return bool(re.search(pattern, text))


def extract_convention(model, tokenizer, problems):
    """Extract e_c at each layer."""
    all_en = {L: [] for L in range(N_LAYERS)}
    all_zh = {L: [] for L in range(N_LAYERS)}

    class Cap:
        def __init__(self):
            self.out = None
        def __call__(self, module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out = h[:, -1, :].detach().float().cpu().numpy()

    caps = [Cap() for _ in range(N_LAYERS)]
    hooks = [model.model.layers[L].register_forward_hook(caps[L]) for L in range(N_LAYERS)]

    for lang, store in [('en', all_en), ('zh', all_zh)]:
        sys = SYS_EN if lang == 'en' else SYS_ZH
        for prob in problems:
            messages = [{"role": "system", "content": sys}, {"role": "user", "content": prob[lang]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
            with torch.inference_mode():
                model(ids)
            for L in range(N_LAYERS):
                store[L].append(caps[L].out.squeeze())

    for h in hooks:
        h.remove()

    e_c = {}
    for L in range(N_LAYERS):
        en_mean = np.mean(all_en[L], axis=0)
        zh_mean = np.mean(all_zh[L], axis=0)
        diff = zh_mean - en_mean
        e_c[L] = diff / (np.linalg.norm(diff) + 1e-12)
    return e_c


class DynamicZProjection:
    """At specified layers, project hidden state onto complement of e_c.

    This removes the convention component from the activation at runtime,
    forcing the representation through Z-space before the top layers render it.
    """
    def __init__(self, model, e_c, layers, alpha=1.0):
        self.e_c = {}
        self.alpha = alpha
        self.hooks = []
        for L in layers:
            ec_t = torch.tensor(e_c[L], dtype=torch.float16, device=DEVICE)
            self.e_c[L] = ec_t
            h = model.model.layers[L].register_forward_hook(self._make_hook(L))
            self.hooks.append(h)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            ec = self.e_c[layer_idx]
            # Project out convention: h = h - alpha * (h . e_c) * e_c
            proj = (h * ec).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
            h_new = h - self.alpha * proj * ec
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def evaluate(model, tokenizer, problems, sys_mode='matched'):
    """Run eval with matched system prompts."""
    scores = {'en': 0, 'zh': 0}
    details = {'en': [], 'zh': []}
    for lang in ['en', 'zh']:
        sys = SYS_EN if (lang == 'en' or sys_mode == 'en_only') else SYS_ZH
        if sys_mode == 'none':
            sys = None
        for prob in problems:
            messages = []
            if sys:
                messages.append({"role": "system", "content": sys})
            messages.append({"role": "user", "content": prob[lang]})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None)
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            correct = strict_check(gen, prob['answer'])
            if correct:
                scores[lang] += 1
            details[lang].append({'answer': prob['answer'], 'correct': correct, 'tail': gen[-150:]})
    return scores, details


def apply_static_surgery(model, e_c, layers):
    """MS1 surgery: remove e_c from W_down."""
    for L in layers:
        ec_t = torch.tensor(e_c[L], dtype=torch.float16, device=DEVICE)
        W = model.model.layers[L].mlp.down_proj.weight.data
        proj = ec_t.unsqueeze(0) @ W
        W.sub_(ec_t.unsqueeze(1) @ proj)


def main():
    print('=' * 60)
    print('Exp Z-Generation: Dynamic Convention Projection')
    print('  h\'fh u2192 project through Z at every step')
    print('=' * 60)

    print(f'\nLoading {MODEL_NAME}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print('  Loaded.', flush=True)

    problems = get_test_problems()
    high_risk = {0, 1, 3, 9, 12, 14}

    # Extract convention directions
    print('\nExtracting convention directions...', flush=True)
    e_c = extract_convention(model, tokenizer, problems)
    print('  Done.', flush=True)

    # Save original weights for surgery restore
    original_W = {}
    for L in range(N_LAYERS):
        original_W[L] = model.model.layers[L].mlp.down_proj.weight.data.clone()

    surgery_layers = list(range(13, 36))  # above convention boundary
    results = {}

    # === 1. BASELINE ===
    print('\n--- Baseline ---', flush=True)
    t0 = time.time()
    scores, details = evaluate(model, tokenizer, problems)
    t = time.time() - t0
    safe = sum(1 for i in range(20) if i not in high_risk and details['en'][i]['correct']) + \
           sum(1 for i in range(20) if i not in high_risk and details['zh'][i]['correct'])
    total = scores['en'] + scores['zh']
    print(f'  Baseline: {total}/40 (EN={scores["en"]}, ZH={scores["zh"]}), safe={safe}/28, {t:.1f}s', flush=True)
    results['baseline'] = {'scores': scores, 'safe': safe, 'time': t}

    # === 2. STATIC SURGERY (MS1, proven) ===
    print('\n--- Static MS1 Surgery (L13-L35) ---', flush=True)
    apply_static_surgery(model, e_c, surgery_layers)
    t0 = time.time()
    scores, details = evaluate(model, tokenizer, problems)
    t = time.time() - t0
    safe = sum(1 for i in range(20) if i not in high_risk and details['en'][i]['correct']) + \
           sum(1 for i in range(20) if i not in high_risk and details['zh'][i]['correct'])
    total = scores['en'] + scores['zh']
    print(f'  Surgery:  {total}/40 (EN={scores["en"]}, ZH={scores["zh"]}), safe={safe}/28, {t:.1f}s', flush=True)
    results['static_surgery'] = {'scores': scores, 'safe': safe, 'time': t}
    # Restore
    for L in range(N_LAYERS):
        model.model.layers[L].mlp.down_proj.weight.data.copy_(original_W[L])

    # === 3. DYNAMIC Z-PROJECTION at L13 (convention boundary) ===
    print('\n--- Dynamic Z-projection at L13 (alpha=1.0) ---', flush=True)
    z_proj = DynamicZProjection(model, e_c, [13], alpha=1.0)
    t0 = time.time()
    scores, details = evaluate(model, tokenizer, problems)
    t = time.time() - t0
    safe = sum(1 for i in range(20) if i not in high_risk and details['en'][i]['correct']) + \
           sum(1 for i in range(20) if i not in high_risk and details['zh'][i]['correct'])
    total = scores['en'] + scores['zh']
    print(f'  DynZ@L13: {total}/40 (EN={scores["en"]}, ZH={scores["zh"]}), safe={safe}/28, {t:.1f}s', flush=True)
    results['dynamic_L13'] = {'scores': scores, 'safe': safe, 'time': t}
    z_proj.remove()

    # === 4. DYNAMIC at every layer L13-L35 ===
    print('\n--- Dynamic Z-projection at L13-L35 (alpha=1.0) ---', flush=True)
    z_proj = DynamicZProjection(model, e_c, surgery_layers, alpha=1.0)
    t0 = time.time()
    scores, details = evaluate(model, tokenizer, problems)
    t = time.time() - t0
    safe = sum(1 for i in range(20) if i not in high_risk and details['en'][i]['correct']) + \
           sum(1 for i in range(20) if i not in high_risk and details['zh'][i]['correct'])
    total = scores['en'] + scores['zh']
    print(f'  DynZ@all: {total}/40 (EN={scores["en"]}, ZH={scores["zh"]}), safe={safe}/28, {t:.1f}s', flush=True)
    results['dynamic_all'] = {'scores': scores, 'safe': safe, 'time': t}
    z_proj.remove()

    # === 5. DYNAMIC alpha sweep at L13-L35 ===
    for alpha in [0.25, 0.5, 0.75]:
        label = f'dynamic_a{alpha}'
        print(f'\n--- Dynamic Z-projection L13-L35 (alpha={alpha}) ---', flush=True)
        z_proj = DynamicZProjection(model, e_c, surgery_layers, alpha=alpha)
        t0 = time.time()
        scores, details = evaluate(model, tokenizer, problems)
        t = time.time() - t0
        safe = sum(1 for i in range(20) if i not in high_risk and details['en'][i]['correct']) + \
               sum(1 for i in range(20) if i not in high_risk and details['zh'][i]['correct'])
        total = scores['en'] + scores['zh']
        print(f'  DynZ a={alpha}: {total}/40 (EN={scores["en"]}, ZH={scores["zh"]}), safe={safe}/28, {t:.1f}s', flush=True)
        results[label] = {'scores': scores, 'safe': safe, 'time': t}
        z_proj.remove()

    # === 6. DYNAMIC + STATIC together ===
    print('\n--- Dynamic Z + Static Surgery (both L13-L35) ---', flush=True)
    apply_static_surgery(model, e_c, surgery_layers)
    z_proj = DynamicZProjection(model, e_c, surgery_layers, alpha=0.5)
    t0 = time.time()
    scores, details = evaluate(model, tokenizer, problems)
    t = time.time() - t0
    safe = sum(1 for i in range(20) if i not in high_risk and details['en'][i]['correct']) + \
           sum(1 for i in range(20) if i not in high_risk and details['zh'][i]['correct'])
    total = scores['en'] + scores['zh']
    print(f'  Both:     {total}/40 (EN={scores["en"]}, ZH={scores["zh"]}), safe={safe}/28, {t:.1f}s', flush=True)
    results['dynamic_plus_surgery'] = {'scores': scores, 'safe': safe, 'time': t}
    z_proj.remove()
    for L in range(N_LAYERS):
        model.model.layers[L].mlp.down_proj.weight.data.copy_(original_W[L])

    # === Summary ===
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'{"Condition":25s} {"Total":>7s} {"EN":>5s} {"ZH":>5s} {"Safe":>6s}')
    print('-' * 50)
    for name, r in results.items():
        s = r['scores']
        total = s['en'] + s['zh']
        print(f'{name:25s} {total:>5d}/40 {s["en"]:>5d} {s["zh"]:>5d} {r["safe"]:>4d}/28')

    outpath = OUTPUT_DIR / 'exp_z_generation.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
