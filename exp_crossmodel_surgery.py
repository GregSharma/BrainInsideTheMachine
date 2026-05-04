"""exp_crossmodel_surgery: Convention-computation separability across architectures.

Minimum viable cross-model test:
1. Extract e_c (mean EN - mean ZH activations) at each layer
2. Find MLP bottleneck layer (SVD of MLP output during generation)
3. Check cos(v1, e_c) at bottleneck
4. Apply MS1-style surgery (remove e_c from W_down above convention breakpoint)
5. Measure accuracy delta on 20 math problems

Parameterized by model name. Same script, different model.
"""
import json, re, time, copy, sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path('output')
MAX_NEW = 512
DEVICE = 'cuda'


def get_test_problems():
    """20 math problems, 5 categories, EN + ZH."""
    return [
        # Algebra
        {"en": "Solve for x: 3x + 7 = 22", "zh": "\u89e3\u65b9\u7a0b\uff1a3x + 7 = 22", "answer": 5, "category": "algebra"},
        {"en": "Solve for x: 2x\u00b2 - 8 = 0", "zh": "\u89e3\u65b9\u7a0b\uff1a2x\u00b2 - 8 = 0", "answer": 2, "category": "algebra"},
        {"en": "Simplify: (x + 3)(x - 3)", "zh": "\u5316\u7b80\uff1a(x + 3)(x - 3)", "answer": "x\u00b2 - 9", "category": "algebra"},
        {"en": "Solve: |2x - 5| = 3", "zh": "\u89e3\u65b9\u7a0b\uff1a|2x - 5| = 3", "answer": 4, "category": "algebra"},
        # Arithmetic
        {"en": "Calculate: 347 + 658", "zh": "\u8ba1\u7b97\uff1a347 + 658", "answer": 1005, "category": "arithmetic"},
        {"en": "Calculate: 1000 - 387", "zh": "\u8ba1\u7b97\uff1a1000 - 387", "answer": 613, "category": "arithmetic"},
        {"en": "Calculate: 23 \u00d7 17", "zh": "\u8ba1\u7b97\uff1a23 \u00d7 17", "answer": 391, "category": "arithmetic"},
        {"en": "Calculate: 1728 \u00f7 12", "zh": "\u8ba1\u7b97\uff1a1728 \u00f7 12", "answer": 144, "category": "arithmetic"},
        # Geometry
        {"en": "Find the area of a circle with radius 7 (use \u03c0 \u2248 22/7)", "zh": "\u6c42\u534a\u5f84\u4e3a7\u7684\u5706\u7684\u9762\u79ef\uff08\u4f7f\u7528\u03c0 \u2248 22/7\uff09", "answer": 154, "category": "geometry"},
        {"en": "Find the hypotenuse of a right triangle with legs 5 and 12", "zh": "\u6c42\u76f4\u89d2\u4e09\u89d2\u5f62\u7684\u659c\u8fb9\uff0c\u4e24\u76f4\u89d2\u8fb9\u5206\u522b\u4e3a5\u548c12", "answer": 13, "category": "geometry"},
        {"en": "What is the perimeter of a rectangle with length 15 and width 8?", "zh": "\u957f15\u5bbd8\u7684\u77e9\u5f62\u7684\u5468\u957f\u662f\u591a\u5c11\uff1f", "answer": 46, "category": "geometry"},
        {"en": "Find the volume of a cube with side length 6", "zh": "\u6c42\u8fb9\u957f\u4e3a6\u7684\u6b63\u65b9\u4f53\u7684\u4f53\u79ef", "answer": 216, "category": "geometry"},
        # Number theory
        {"en": "What is the GCD of 84 and 120?", "zh": "84\u548c120\u7684\u6700\u5927\u516c\u7ea6\u6570\u662f\u591a\u5c11\uff1f", "answer": 12, "category": "number_theory"},
        {"en": "Is 97 prime? Answer yes or no, then explain.", "zh": "97\u662f\u8d28\u6570\u5417\uff1f\u56de\u7b54\u662f\u6216\u5426\uff0c\u7136\u540e\u89e3\u91ca\u3002", "answer": "yes", "category": "number_theory"},
        {"en": "Find the remainder when 2^10 is divided by 7", "zh": "\u6c422^10\u9664\u4ee57\u7684\u4f59\u6570", "answer": 2, "category": "number_theory"},
        {"en": "What is the sum of all prime numbers less than 20?", "zh": "\u6c4220\u4ee5\u5185\u6240\u6709\u8d28\u6570\u7684\u548c", "answer": 77, "category": "number_theory"},
        # Combinatorics
        {"en": "How many ways can you choose 3 items from 7?", "zh": "\u4ece7\u4e2a\u7269\u54c1\u4e2d\u9009\u62e93\u4e2a\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f", "answer": 35, "category": "combinatorics"},
        {"en": "How many ways can 5 people stand in a line?", "zh": "5\u4e2a\u4eba\u7ad9\u6210\u4e00\u6392\u6709\u591a\u5c11\u79cd\u65b9\u6cd5\uff1f", "answer": 120, "category": "combinatorics"},
        {"en": "Calculate: 8! / (5! \u00d7 3!)", "zh": "\u8ba1\u7b97\uff1a8! / (5! \u00d7 3!)", "answer": 56, "category": "combinatorics"},
        {"en": "How many 3-digit numbers have all distinct digits?", "zh": "\u6709\u591a\u5c11\u4e2a\u4e09\u4f4d\u6570\u7684\u5404\u4f4d\u6570\u5b57\u5747\u4e0d\u76f8\u540c\uff1f", "answer": 648, "category": "combinatorics"},
    ]


def strict_check_answer(text, correct):
    """Strict whole-number boundary matching."""
    if correct in ("yes", "no"):
        return correct.lower() in text.lower()
    if isinstance(correct, str) and not correct.replace('-','').replace('.','').isdigit():
        # Non-numeric answer like 'x² - 9'
        return correct in text
    target = str(correct)
    pattern = r'(?<![\d.])' + re.escape(target) + r'(?![\d.])'
    return bool(re.search(pattern, text))


def detect_model_architecture(model, model_name):
    """Detect architecture details for hook installation."""
    config = model.config
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size

    # Detect MLP module path
    layer0 = model.model.layers[0]
    mlp = layer0.mlp

    # Find W_down equivalent
    down_proj_name = None
    for name in ['down_proj', 'fc2', 'w2']:
        if hasattr(mlp, name):
            down_proj_name = name
            break

    # Find gate_proj equivalent
    gate_proj_name = None
    for name in ['gate_proj', 'gate_up_proj', 'w1']:
        if hasattr(mlp, name):
            gate_proj_name = name
            break

    # Find up_proj equivalent
    up_proj_name = None
    for name in ['up_proj', 'fc1', 'w3']:
        if hasattr(mlp, name):
            up_proj_name = name
            break

    # Detect intermediate dim
    down = getattr(mlp, down_proj_name)
    intermediate_size = down.in_features

    # Tied embeddings?
    tied = getattr(config, 'tie_word_embeddings', True)

    info = {
        'n_layers': n_layers,
        'd_model': d_model,
        'intermediate_size': intermediate_size,
        'tied_embeddings': tied,
        'down_proj': down_proj_name,
        'gate_proj': gate_proj_name,
        'up_proj': up_proj_name,
    }
    print(f"  Architecture: {n_layers} layers, d={d_model}, mlp_intermediate={intermediate_size}")
    print(f"  MLP components: gate={gate_proj_name}, up={up_proj_name}, down={down_proj_name}")
    print(f"  Tied embeddings: {tied}")
    return info


def build_prompt(tokenizer, problem_text, sys_text=None):
    """Build prompt using the model's chat template."""
    messages = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": problem_text})

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: raw text
        if sys_text:
            prompt = f"{sys_text}\n\n{problem_text}\n\nAnswer: "
        else:
            prompt = f"{problem_text}\n\nAnswer: "
    return prompt


def extract_encoding_activations(model, tokenizer, problems, lang, arch_info, device):
    """Get last hidden state at each layer for each problem."""
    n_layers = arch_info['n_layers']
    all_acts = {L: [] for L in range(n_layers)}

    class LayerCapture:
        def __init__(self):
            self.out = None
        def __call__(self, module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                self.out = h[:, -1, :].detach().float().cpu().numpy()
            else:
                self.out = h[-1:, :].detach().float().cpu().numpy()

    captures = [LayerCapture() for _ in range(n_layers)]
    hooks = []
    for L in range(n_layers):
        h = model.model.layers[L].register_forward_hook(captures[L])
        hooks.append(h)

    sys_text = 'You are a careful mathematical reasoner. Think step by step, show your work clearly, and then state the final numerical answer on its own line.'

    for prob in problems:
        prompt = build_prompt(tokenizer, prob[lang], sys_text)
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        with torch.inference_mode():
            model(ids)
        for L in range(n_layers):
            all_acts[L].append(captures[L].out.squeeze())

    for h in hooks:
        h.remove()

    for L in range(n_layers):
        all_acts[L] = np.stack(all_acts[L])
    return all_acts


def compute_convention_directions(en_acts, zh_acts, n_layers):
    """Compute e_c at each layer as normalized mean(zh) - mean(en)."""
    directions = {}
    for L in range(n_layers):
        diff = zh_acts[L].mean(axis=0) - en_acts[L].mean(axis=0)
        norm = np.linalg.norm(diff)
        directions[L] = diff / (norm + 1e-12)
    return directions


def find_convention_breakpoint(en_acts, zh_acts, n_layers):
    """Find L* where convention direction appears (cos between consecutive e_c drops)."""
    directions = compute_convention_directions(en_acts, zh_acts, n_layers)

    # Compute language distance at each layer
    lang_dist = []
    for L in range(n_layers):
        d = np.linalg.norm(zh_acts[L].mean(axis=0) - en_acts[L].mean(axis=0))
        lang_dist.append(d)

    # Convention breakpoint: where language distance first becomes significant
    # Use the layer where distance drops to a local minimum then rises
    print(f"  Language distance by layer (first 5, middle, last 5):")
    for L in list(range(5)) + [n_layers//4, n_layers//2, 3*n_layers//4] + list(range(n_layers-5, n_layers)):
        print(f"    L{L:2d}: dist={lang_dist[L]:.4f}")

    # Heuristic: convention breakpoint is where distance reaches global minimum
    min_layer = np.argmin(lang_dist[2:]) + 2  # skip first 2 layers
    print(f"  Convention breakpoint (min distance): L{min_layer} (dist={lang_dist[min_layer]:.4f})")

    return min_layer, directions, lang_dist


def apply_surgery(model, directions, layers_to_modify, arch_info, device):
    """Project out e_c from W_down at specified layers."""
    down_name = arch_info['down_proj']
    for L in layers_to_modify:
        e_c = torch.tensor(directions[L], dtype=torch.float16, device=device)
        mlp = model.model.layers[L].mlp
        W = getattr(mlp, down_name).weight.data  # (d_model, intermediate)
        proj = e_c.unsqueeze(0) @ W  # (1, intermediate)
        W.sub_(e_c.unsqueeze(1) @ proj)


def evaluate(model, tokenizer, problems, arch_info, device):
    """Run eval, return scores and details."""
    sys_text = 'You are a careful mathematical reasoner. Think step by step, show your work clearly, and then state the final numerical answer on its own line.'

    results = {'en': [], 'zh': []}
    scores = {'en': 0, 'zh': 0}

    for lang in ['en', 'zh']:
        for prob in problems:
            prompt = build_prompt(tokenizer, prob[lang], sys_text)
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            with torch.inference_mode():
                out = model.generate(
                    ids, max_new_tokens=MAX_NEW, do_sample=False,
                    temperature=None, top_p=None,
                )
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            correct = strict_check_answer(gen, prob['answer'])
            if correct:
                scores[lang] += 1
            results[lang].append({
                'answer': prob['answer'],
                'category': prob['category'],
                'correct': correct,
                'output_tail': gen[-200:],
                'gen_tokens': out.shape[1] - ids.shape[1],
            })

    return scores, results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'microsoft/Phi-3-mini-4k-instruct'

    print('=' * 60)
    print(f'Exp Cross-Model Surgery')
    print(f'  Model: {model_name}')
    print('=' * 60)

    # Load
    print(f'\nLoading {model_name}...', flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Try without trust_remote_code first (avoids custom code compat issues)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
        )
    model.eval()
    print(f'  Loaded in {time.time()-t0:.1f}s', flush=True)

    # Detect architecture
    arch_info = detect_model_architecture(model, model_name)
    n_layers = arch_info['n_layers']

    # Get problems
    problems = get_test_problems()
    print(f'  Problems: {len(problems)} \u00d7 2 langs = {len(problems)*2} evals', flush=True)

    # Extract encoding activations for convention direction
    print('\nExtracting encoding activations...', flush=True)
    en_acts = extract_encoding_activations(model, tokenizer, problems, 'en', arch_info, DEVICE)
    zh_acts = extract_encoding_activations(model, tokenizer, problems, 'zh', arch_info, DEVICE)

    # Find convention breakpoint
    print('\nFinding convention breakpoint...', flush=True)
    lc, directions, lang_dist = find_convention_breakpoint(en_acts, zh_acts, n_layers)

    # Save original weights
    print('\nSaving original weights...', flush=True)
    down_name = arch_info['down_proj']
    original_weights = {}
    for L in range(n_layers):
        W = getattr(model.model.layers[L].mlp, down_name).weight.data
        original_weights[L] = W.clone()

    # === BASELINE ===
    print('\n--- Baseline evaluation ---', flush=True)
    t0 = time.time()
    base_scores, base_details = evaluate(model, tokenizer, problems, arch_info, DEVICE)
    base_time = time.time() - t0
    base_total = base_scores['en'] + base_scores['zh']
    print(f'  Baseline: {base_total}/40 (EN={base_scores["en"]}, ZH={base_scores["zh"]}) [{base_time:.1f}s]', flush=True)

    # === SURGERY (above convention breakpoint) ===
    surgery_layers = list(range(lc, n_layers))
    print(f'\n--- Surgery (L{lc}-L{n_layers-1}, {len(surgery_layers)} layers) ---', flush=True)
    apply_surgery(model, directions, surgery_layers, arch_info, DEVICE)
    t0 = time.time()
    surg_scores, surg_details = evaluate(model, tokenizer, problems, arch_info, DEVICE)
    surg_time = time.time() - t0
    surg_total = surg_scores['en'] + surg_scores['zh']
    print(f'  Surgery:  {surg_total}/40 (EN={surg_scores["en"]}, ZH={surg_scores["zh"]}) [{surg_time:.1f}s]', flush=True)

    # Restore weights
    for L in range(n_layers):
        getattr(model.model.layers[L].mlp, down_name).weight.data.copy_(original_weights[L])

    # === SURGERY (all layers) ===
    print(f'\n--- Surgery (all {n_layers} layers) ---', flush=True)
    all_layers = list(range(n_layers))
    apply_surgery(model, directions, all_layers, arch_info, DEVICE)
    t0 = time.time()
    all_scores, all_details = evaluate(model, tokenizer, problems, arch_info, DEVICE)
    all_time = time.time() - t0
    all_total = all_scores['en'] + all_scores['zh']
    print(f'  All-layer: {all_total}/40 (EN={all_scores["en"]}, ZH={all_scores["zh"]}) [{all_time:.1f}s]', flush=True)

    # Restore
    for L in range(n_layers):
        getattr(model.model.layers[L].mlp, down_name).weight.data.copy_(original_weights[L])

    # Summary
    delta_above = surg_total - base_total
    delta_all = all_total - base_total
    print(f'\n{"=" * 60}')
    print(f'SUMMARY: {model_name}')
    print(f'{"=" * 60}')
    print(f'  Architecture: {n_layers}L, d={arch_info["d_model"]}, mlp={arch_info["intermediate_size"]}')
    print(f'  Convention breakpoint: L{lc}')
    print(f'  Baseline:     {base_total}/40 (EN={base_scores["en"]}, ZH={base_scores["zh"]})')
    print(f'  Surgery L{lc}+: {surg_total}/40 (EN={surg_scores["en"]}, ZH={surg_scores["zh"]}) delta={delta_above:+d}')
    print(f'  Surgery all:  {all_total}/40 (EN={all_scores["en"]}, ZH={all_scores["zh"]}) delta={delta_all:+d}')

    # Safe-problem analysis
    high_risk = {0, 1, 3, 9, 12, 14}  # small-answer problems
    for label, details in [('baseline', base_details), ('surgery_above_lc', surg_details), ('surgery_all', all_details)]:
        safe_en = sum(1 for i, item in enumerate(details['en']) if item['correct'] and i not in high_risk)
        safe_zh = sum(1 for i, item in enumerate(details['zh']) if item['correct'] and i not in high_risk)
        print(f'  {label} safe: {safe_en + safe_zh}/28 (EN={safe_en}, ZH={safe_zh})')

    # Save
    out_name = model_name.replace('/', '_').replace('-', '_')
    outpath = OUTPUT_DIR / f'exp_crossmodel_{out_name}.json'
    result = {
        'model': model_name,
        'architecture': arch_info,
        'convention_breakpoint': int(lc),
        'lang_dist': [float(d) for d in lang_dist],
        'baseline': {'scores': base_scores, 'details': base_details},
        'surgery_above_lc': {'scores': surg_scores, 'details': surg_details, 'layers': surgery_layers},
        'surgery_all': {'scores': all_scores, 'details': all_details},
    }
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    main()
