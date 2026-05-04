"""Experiment 5: Splice + Language Steering at L26.

TWO INTERVENTIONS testing h-f-h' factorization:

A) FULL SPLICE: Run zh through L0-L26, capture h_zh. Run en through full model
   but at L27 input, swap in h_zh. Does the model produce correct English output
   with Chinese reasoning state injected?

B) LANGUAGE STEERING VECTOR: Run zh through L0-L26, capture h_zh.
   Compute PC0 (language axis) at L26 from the 200-problem corpus.
   Subtract zh's PC0 projection, add en's mean PC0 projection.
   Feed modified state through L27-L35. Does output switch to English?

CONDITIONS:
  1. Baseline zh: normal zh forward pass (control)
  2. Baseline en: normal en forward pass (control)
  3. Raw splice: inject h_zh into en's L27 input
  4. PC0 swap: flip language component of h_zh, inject into en's L27 input
  5. Random direction swap: same magnitude perturbation but random direction (control)
  6. Scrambled splice: inject WRONG problem's h_zh (control)
  7. f extraction: verify same-problem zh and en map to similar f vectors

MEASURES:
  - First token match (does output start with same token as baseline?)
  - Full generation text (first 64 tokens of continuation)
  - Language of output (zh characters vs en characters)
  - Cosine similarity of L26 hidden states after intervention
"""

import numpy as np
import torch
import json
import random as pyrandom
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import re

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

SPLICE_LAYER = 26  # The factorization boundary
N_PROBLEMS = 20    # For generation experiments
N_PCA = 200        # Full corpus for PCA fitting
MAX_TOKENS = 64    # Enough to see language + content


def generate_problems(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh, en = f"计算 {a} + {b} 的值。", f"Calculate {a} + {b}."
        else:
            zh, en = f"计算 {a} × {b} 的值。", f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "category": 0})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "category": 1})
    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "category": 2})
    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "category": 3})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        problems.append({"zh": zh, "en": en, "category": 4})
    rng.shuffle(problems)
    return problems


def select_problems(problems, n=20):
    selected = []
    cat_count = {i: 0 for i in range(5)}
    for i, p in enumerate(problems):
        c = p['category']
        if cat_count[c] < n // 5:
            selected.append(i)
            cat_count[c] += 1
        if len(selected) == n:
            break
    return selected


def detect_language(text):
    """Detect if text is primarily Chinese or English."""
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    total = zh_chars + en_chars
    if total == 0:
        return "numeric"
    zh_frac = zh_chars / total
    if zh_frac > 0.5:
        return "zh"
    elif zh_frac < 0.2:
        return "en"
    return "mixed"


def generate_tokens(model, tokenizer, input_ids, past_key_values, max_tokens):
    """Autoregressive generation from a starting state."""
    eos_id = tokenizer.eos_token_id
    tokens = []

    with torch.no_grad():
        # First token from the current state
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())

        if next_token.item() == eos_id:
            return tokens

        for _ in range(1, max_tokens):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            if next_token.item() == eos_id:
                break

    return tokens


def run_with_splice(model, tokenizer, prompt_source, prompt_target, splice_layer,
                    h_inject, max_tokens=64):
    """Run target prompt through model, but at splice_layer+1 input, inject h_inject.

    The key insight: we can't just swap the hidden state at L27 input because
    L0-L26's KV cache contains the SOURCE language tokens. We need to:
    1. Run target prompt through L0-L26 normally (building target KV cache for L0-L26)
    2. At L27 input, replace the hidden state with h_inject
    3. Let L27-L35 process with fresh KV cache entries

    Actually, in a transformer with KV cache, the hidden state flows through
    sequentially. We hook layer splice_layer to REPLACE its output, then let
    layers splice_layer+1 through N-1 process normally. The KV cache for
    layers 0-splice_layer comes from the target prompt. Layers splice_layer+1
    through N-1 build their KV from the modified hidden state.
    """
    inputs = tokenizer(prompt_target, return_tensors="pt").to(model.device)

    replaced = {}

    def splice_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # Replace last-token hidden state with injected state
        new_h = h.clone()
        new_h[:, -1:, :] = h_inject.to(h.device)
        if isinstance(output, tuple):
            replaced['done'] = True
            return (new_h,) + output[1:]
        replaced['done'] = True
        return new_h

    handle = model.model.layers[splice_layer].register_forward_hook(splice_hook)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values

    handle.remove()

    # First token from spliced state
    first_token_id = int(outputs.logits[0, -1].argmax())

    # Continue generation
    next_token = torch.tensor([[first_token_id]], device=model.device)
    tokens = [first_token_id]
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_tokens - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token.item())
            if next_token.item() == eos_id:
                break

    return tokens


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers  # 36
    d = model.config.hidden_size  # 2048
    print(f"Model: {n_layers} layers, d={d}")
    print(f"Splice layer: L{SPLICE_LAYER}")

    problems = generate_problems(N_PCA, seed=42)
    selected = select_problems(problems, N_PROBLEMS)
    print(f"Selected {len(selected)} problems for generation")

    # ================================================================
    # PHASE 1: Extract L26 hidden states for ALL 200 problems (for PCA)
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Extracting L{} hidden states for PCA fitting".format(SPLICE_LAYER))
    print(f"{'='*70}")

    layer_output = {}

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        layer_output['h'] = h.detach()[:, -1, :]  # (1, d)

    handle = model.model.layers[SPLICE_LAYER].register_forward_hook(capture_hook)

    zh_hidden_all = np.zeros((N_PCA, d), dtype=np.float32)
    en_hidden_all = np.zeros((N_PCA, d), dtype=np.float32)

    print("Extracting zh L26 states...")
    for i, prob in enumerate(tqdm(problems, desc="zh L26")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        zh_hidden_all[i] = layer_output['h'].cpu().float().numpy()
        layer_output.clear()

    print("Extracting en L26 states...")
    for i, prob in enumerate(tqdm(problems, desc="en L26")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        en_hidden_all[i] = layer_output['h'].cpu().float().numpy()
        layer_output.clear()

    handle.remove()

    # ================================================================
    # PHASE 2: Fit PCA at L26 to get the language axis
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: PCA at L{} — finding language axis".format(SPLICE_LAYER))
    print(f"{'='*70}")

    # Unit normalize before PCA (as per the recipe)
    zh_norms = np.linalg.norm(zh_hidden_all, axis=1, keepdims=True)
    en_norms = np.linalg.norm(en_hidden_all, axis=1, keepdims=True)
    zh_unit = zh_hidden_all / zh_norms
    en_unit = en_hidden_all / en_norms

    combined = np.vstack([zh_unit, en_unit])  # (400, 2048)
    pca = PCA(n_components=20)
    pca.fit(combined)

    pc0 = pca.components_[0]  # (2048,) — the language axis

    # Verify PC0 IS the language axis
    zh_proj = zh_unit @ pc0  # (200,)
    en_proj = en_unit @ pc0  # (200,)
    cohens_d = (zh_proj.mean() - en_proj.mean()) / np.sqrt((zh_proj.std()**2 + en_proj.std()**2) / 2)

    print(f"PC0 variance explained: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"zh projection: mean={zh_proj.mean():.4f}, std={zh_proj.std():.4f}")
    print(f"en projection: mean={en_proj.mean():.4f}, std={en_proj.std():.4f}")
    print(f"Cohen's d (zh vs en on PC0): {cohens_d:.2f}")
    print(f"PC0 {'IS' if abs(cohens_d) > 3 else 'is NOT'} the language axis")

    # Mean projections for the swap
    zh_mean_proj = zh_proj.mean()
    en_mean_proj = en_proj.mean()

    # Also check f-vector similarity: project out PC0, check same-problem cosine
    zh_f = zh_unit - np.outer(zh_proj, pc0)  # (200, 2048) — language-removed
    en_f = en_unit - np.outer(en_proj, pc0)
    # Renormalize
    zh_f_norm = zh_f / np.linalg.norm(zh_f, axis=1, keepdims=True)
    en_f_norm = en_f / np.linalg.norm(en_f, axis=1, keepdims=True)

    # Same-problem f cosine (matched)
    matched_f_cos = np.sum(zh_f_norm * en_f_norm, axis=1)  # (200,)
    # Scrambled
    rng = np.random.RandomState(42)
    scrambled_f_cos = np.array([
        np.mean(np.sum(zh_f_norm * en_f_norm[rng.permutation(N_PCA)], axis=1))
        for _ in range(500)
    ])
    f_z = (matched_f_cos.mean() - scrambled_f_cos.mean()) / scrambled_f_cos.std()

    print(f"\nf-VECTOR ANALYSIS (language-removed):")
    print(f"  Matched f cosine: {matched_f_cos.mean():.4f} ± {matched_f_cos.std():.4f}")
    print(f"  Scrambled f cosine: {scrambled_f_cos.mean():.4f} ± {scrambled_f_cos.std():.4f}")
    print(f"  z-score: {f_z:.1f}")
    print(f"  f IS {'LANGUAGE-AGNOSTIC' if f_z > 10 else 'partially aligned'}")

    # Multi-PC language removal: also try removing top-k PCs that separate languages
    # Check which PCs have significant language separation
    print(f"\n  Per-PC Cohen's d (language separation):")
    pc_cohens = []
    for k in range(min(10, pca.n_components_)):
        pc_k = pca.components_[k]
        zh_k = zh_unit @ pc_k
        en_k = en_unit @ pc_k
        d_k = (zh_k.mean() - en_k.mean()) / np.sqrt((zh_k.std()**2 + en_k.std()**2) / 2)
        pc_cohens.append(d_k)
        if abs(d_k) > 1.0:
            print(f"    PC{k}: d={d_k:.2f} (var={pca.explained_variance_ratio_[k]:.1%}) ← LANGUAGE-SEPARATING")
        else:
            print(f"    PC{k}: d={d_k:.2f} (var={pca.explained_variance_ratio_[k]:.1%})")

    # ================================================================
    # PHASE 3: Run generation experiments
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Generation experiments (splice + steer)")
    print(f"{'='*70}")

    results_per_problem = []

    for prob_i, prob_idx in enumerate(tqdm(selected, desc="Experiments")):
        prob = problems[prob_idx]
        prob_result = {
            "prob_idx": prob_idx,
            "category": prob["category"],
            "zh_prompt": prob["zh"],
            "en_prompt": prob["en"],
        }

        # --- A. Baselines: normal zh and en generation ---
        inputs_zh = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        inputs_en = tokenizer(prob["en"], return_tensors="pt").to(model.device)

        # Capture L26 hidden states during baseline runs
        zh_h26 = None
        en_h26 = None

        def capture_l26(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            capture_l26.h = h.detach()[:, -1:, :].clone()  # (1, 1, d)

        handle = model.model.layers[SPLICE_LAYER].register_forward_hook(capture_l26)

        with torch.no_grad():
            out_zh = model(**inputs_zh, use_cache=True)
        zh_h26 = capture_l26.h  # (1, 1, d) on GPU
        zh_first_token = int(out_zh.logits[0, -1].argmax())
        zh_baseline_tokens = [zh_first_token]
        next_tok = torch.tensor([[zh_first_token]], device=model.device)
        pkv = out_zh.past_key_values
        with torch.no_grad():
            for _ in range(MAX_TOKENS - 1):
                o = model(next_tok, past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values
                next_tok = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                zh_baseline_tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id:
                    break

        with torch.no_grad():
            out_en = model(**inputs_en, use_cache=True)
        en_h26 = capture_l26.h  # (1, 1, d) on GPU
        en_first_token = int(out_en.logits[0, -1].argmax())
        en_baseline_tokens = [en_first_token]
        next_tok = torch.tensor([[en_first_token]], device=model.device)
        pkv = out_en.past_key_values
        with torch.no_grad():
            for _ in range(MAX_TOKENS - 1):
                o = model(next_tok, past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values
                next_tok = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                en_baseline_tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id:
                    break

        handle.remove()

        zh_text = tokenizer.decode(zh_baseline_tokens, skip_special_tokens=True)
        en_text = tokenizer.decode(en_baseline_tokens, skip_special_tokens=True)

        prob_result["baseline_zh"] = {
            "first_token": zh_first_token,
            "text": zh_text,
            "lang": detect_language(zh_text),
            "n_tokens": len(zh_baseline_tokens),
        }
        prob_result["baseline_en"] = {
            "first_token": en_first_token,
            "text": en_text,
            "lang": detect_language(en_text),
            "n_tokens": len(en_baseline_tokens),
        }

        # --- B. RAW SPLICE: inject zh's L26 state into en's forward pass at L26 ---
        splice_raw_tokens = run_with_splice(
            model, tokenizer, prob["zh"], prob["en"],
            SPLICE_LAYER, zh_h26, MAX_TOKENS
        )
        splice_raw_text = tokenizer.decode(splice_raw_tokens, skip_special_tokens=True)

        prob_result["splice_raw"] = {
            "first_token": splice_raw_tokens[0],
            "text": splice_raw_text,
            "lang": detect_language(splice_raw_text),
            "n_tokens": len(splice_raw_tokens),
            "first_token_matches_zh": splice_raw_tokens[0] == zh_first_token,
            "first_token_matches_en": splice_raw_tokens[0] == en_first_token,
        }

        # --- C. PC0 SWAP: flip language component of zh's L26 state ---
        # Work in fp32 for precision
        zh_h26_np = zh_h26[0, 0].cpu().float().numpy()  # (d,)
        zh_h26_unit = zh_h26_np / np.linalg.norm(zh_h26_np)

        # Project onto PC0
        zh_pc0_proj = float(zh_h26_unit @ pc0)
        # Remove zh language component, add en language component
        # Work on unit-normalized vector, then rescale
        zh_h26_swapped = zh_h26_unit - zh_pc0_proj * pc0 + en_mean_proj * pc0
        # Rescale back to original norm
        zh_h26_swapped = zh_h26_swapped * np.linalg.norm(zh_h26_np)
        zh_h26_swapped_t = torch.tensor(zh_h26_swapped, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to(model.device)

        splice_pc0_tokens = run_with_splice(
            model, tokenizer, prob["zh"], prob["en"],
            SPLICE_LAYER, zh_h26_swapped_t, MAX_TOKENS
        )
        splice_pc0_text = tokenizer.decode(splice_pc0_tokens, skip_special_tokens=True)

        prob_result["splice_pc0_swap"] = {
            "first_token": splice_pc0_tokens[0],
            "text": splice_pc0_text,
            "lang": detect_language(splice_pc0_text),
            "n_tokens": len(splice_pc0_tokens),
            "first_token_matches_zh": splice_pc0_tokens[0] == zh_first_token,
            "first_token_matches_en": splice_pc0_tokens[0] == en_first_token,
            "zh_pc0_proj": zh_pc0_proj,
            "en_mean_proj": float(en_mean_proj),
        }

        # --- D. RANDOM DIRECTION SWAP: same magnitude perturbation, random direction ---
        rng_dir = np.random.RandomState(prob_idx)
        random_dir = rng_dir.randn(d).astype(np.float32)
        random_dir = random_dir / np.linalg.norm(random_dir)
        # Same magnitude as PC0 swap
        swap_magnitude = abs(zh_pc0_proj - en_mean_proj)
        zh_h26_random = zh_h26_unit - (zh_h26_unit @ random_dir) * random_dir + (zh_h26_unit @ random_dir + swap_magnitude * np.sign(en_mean_proj - zh_pc0_proj)) * random_dir
        # Simpler: just add the same magnitude along random direction
        zh_h26_random = zh_h26_unit + swap_magnitude * random_dir
        zh_h26_random = zh_h26_random * np.linalg.norm(zh_h26_np)
        zh_h26_random_t = torch.tensor(zh_h26_random, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to(model.device)

        splice_random_tokens = run_with_splice(
            model, tokenizer, prob["zh"], prob["en"],
            SPLICE_LAYER, zh_h26_random_t, MAX_TOKENS
        )
        splice_random_text = tokenizer.decode(splice_random_tokens, skip_special_tokens=True)

        prob_result["splice_random_dir"] = {
            "first_token": splice_random_tokens[0],
            "text": splice_random_text,
            "lang": detect_language(splice_random_text),
            "n_tokens": len(splice_random_tokens),
            "first_token_matches_zh": splice_random_tokens[0] == zh_first_token,
            "first_token_matches_en": splice_random_tokens[0] == en_first_token,
        }

        # --- E. SCRAMBLED SPLICE: inject WRONG problem's zh L26 state ---
        wrong_idx = selected[(prob_i + 7) % len(selected)]  # offset by 7 to avoid self
        wrong_prob = problems[wrong_idx]
        inputs_wrong = tokenizer(wrong_prob["zh"], return_tensors="pt").to(model.device)

        handle = model.model.layers[SPLICE_LAYER].register_forward_hook(capture_l26)
        with torch.no_grad():
            model(**inputs_wrong)
        wrong_h26 = capture_l26.h.clone()
        handle.remove()

        splice_scrambled_tokens = run_with_splice(
            model, tokenizer, prob["zh"], prob["en"],
            SPLICE_LAYER, wrong_h26, MAX_TOKENS
        )
        splice_scrambled_text = tokenizer.decode(splice_scrambled_tokens, skip_special_tokens=True)

        prob_result["splice_scrambled"] = {
            "first_token": splice_scrambled_tokens[0],
            "text": splice_scrambled_text,
            "lang": detect_language(splice_scrambled_text),
            "n_tokens": len(splice_scrambled_tokens),
            "first_token_matches_zh": splice_scrambled_tokens[0] == zh_first_token,
            "first_token_matches_en": splice_scrambled_tokens[0] == en_first_token,
            "wrong_prob_idx": wrong_idx,
        }

        # --- F. REVERSE STEERING: en → zh (flip en's PC0 to zh's mean) ---
        en_h26_np = en_h26[0, 0].cpu().float().numpy()
        en_h26_unit = en_h26_np / np.linalg.norm(en_h26_np)
        en_pc0_proj = float(en_h26_unit @ pc0)
        en_h26_to_zh = en_h26_unit - en_pc0_proj * pc0 + zh_mean_proj * pc0
        en_h26_to_zh = en_h26_to_zh * np.linalg.norm(en_h26_np)
        en_h26_to_zh_t = torch.tensor(en_h26_to_zh, dtype=torch.float16).unsqueeze(0).unsqueeze(0).to(model.device)

        # Inject into ZH forward pass (opposite direction)
        splice_reverse_tokens = run_with_splice(
            model, tokenizer, prob["en"], prob["zh"],  # source=en, target=zh
            SPLICE_LAYER, en_h26_to_zh_t, MAX_TOKENS
        )
        splice_reverse_text = tokenizer.decode(splice_reverse_tokens, skip_special_tokens=True)

        prob_result["splice_reverse_steer"] = {
            "first_token": splice_reverse_tokens[0],
            "text": splice_reverse_text,
            "lang": detect_language(splice_reverse_text),
            "n_tokens": len(splice_reverse_tokens),
            "first_token_matches_zh": splice_reverse_tokens[0] == zh_first_token,
            "first_token_matches_en": splice_reverse_tokens[0] == en_first_token,
            "en_pc0_proj": en_pc0_proj,
            "zh_mean_proj": float(zh_mean_proj),
        }

        # --- Cosine similarities between hidden states ---
        zh_u = zh_h26[0, 0].cpu().float()
        en_u = en_h26[0, 0].cpu().float()
        cos_zh_en = float(torch.cosine_similarity(zh_u.unsqueeze(0), en_u.unsqueeze(0)))
        prob_result["h26_cosine_zh_en"] = cos_zh_en

        results_per_problem.append(prob_result)

        # Print progress
        if (prob_i + 1) % 5 == 0 or prob_i == 0:
            print(f"\n  Problem {prob_i} (idx={prob_idx}, cat={prob['category']}):")
            print(f"    zh baseline: '{zh_text[:60]}...' [{prob_result['baseline_zh']['lang']}]")
            print(f"    en baseline: '{en_text[:60]}...' [{prob_result['baseline_en']['lang']}]")
            print(f"    raw splice:  '{splice_raw_text[:60]}...' [{prob_result['splice_raw']['lang']}]")
            print(f"    PC0 swap:    '{splice_pc0_text[:60]}...' [{prob_result['splice_pc0_swap']['lang']}]")
            print(f"    random dir:  '{splice_random_text[:60]}...' [{prob_result['splice_random_dir']['lang']}]")
            print(f"    scrambled:   '{splice_scrambled_text[:60]}...' [{prob_result['splice_scrambled']['lang']}]")
            print(f"    reverse:     '{splice_reverse_text[:60]}...' [{prob_result['splice_reverse_steer']['lang']}]")

    # ================================================================
    # PHASE 4: Aggregate analysis
    # ================================================================
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    conditions = [
        ("baseline_zh", "Baseline zh"),
        ("baseline_en", "Baseline en"),
        ("splice_raw", "Raw splice (zh→en)"),
        ("splice_pc0_swap", "PC0 swap (zh→en)"),
        ("splice_random_dir", "Random dir (control)"),
        ("splice_scrambled", "Scrambled splice (control)"),
        ("splice_reverse_steer", "Reverse steer (en→zh)"),
    ]

    print(f"\n{'Condition':>30} | {'zh%':>5} | {'en%':>5} | {'mix%':>5} | {'num%':>5} | {'1st=zh':>6} | {'1st=en':>6}")
    print("-" * 100)

    summary = {}
    for key, label in conditions:
        langs = [r[key]["lang"] for r in results_per_problem]
        zh_pct = langs.count("zh") / len(langs)
        en_pct = langs.count("en") / len(langs)
        mix_pct = langs.count("mixed") / len(langs)
        num_pct = langs.count("numeric") / len(langs)

        if "first_token_matches_zh" in results_per_problem[0].get(key, {}):
            ft_zh = sum(1 for r in results_per_problem if r[key]["first_token_matches_zh"]) / len(results_per_problem)
            ft_en = sum(1 for r in results_per_problem if r[key]["first_token_matches_en"]) / len(results_per_problem)
        else:
            ft_zh = float('nan')
            ft_en = float('nan')

        print(f"  {label:>28} | {zh_pct:4.0%} | {en_pct:4.0%} | {mix_pct:4.0%} | {num_pct:4.0%} | {ft_zh:5.0%} | {ft_en:5.0%}")

        summary[key] = {
            "lang_dist": {"zh": zh_pct, "en": en_pct, "mixed": mix_pct, "numeric": num_pct},
            "first_token_match_zh": ft_zh if not np.isnan(ft_zh) else None,
            "first_token_match_en": ft_en if not np.isnan(ft_en) else None,
        }

    # f-vector cosines
    print(f"\n{'='*70}")
    print("L26 HIDDEN STATE COSINES (zh vs en, same problem)")
    print(f"{'='*70}")
    cosines = [r["h26_cosine_zh_en"] for r in results_per_problem]
    print(f"  Mean: {np.mean(cosines):.4f}")
    print(f"  Std:  {np.std(cosines):.4f}")
    print(f"  Range: [{min(cosines):.4f}, {max(cosines):.4f}]")

    # ================================================================
    # Save everything
    # ================================================================
    output = {
        "model": MODEL_NAME,
        "splice_layer": SPLICE_LAYER,
        "n_problems": N_PROBLEMS,
        "n_pca": N_PCA,
        "max_tokens": MAX_TOKENS,
        "pca_at_L26": {
            "pc0_variance_explained": float(pca.explained_variance_ratio_[0]),
            "cohens_d": float(cohens_d),
            "zh_mean_proj": float(zh_mean_proj),
            "en_mean_proj": float(en_mean_proj),
            "pc_cohens_d": [float(x) for x in pc_cohens],
        },
        "f_vector_analysis": {
            "matched_f_cosine_mean": float(matched_f_cos.mean()),
            "matched_f_cosine_std": float(matched_f_cos.std()),
            "scrambled_f_cosine_mean": float(scrambled_f_cos.mean()),
            "z_score": float(f_z),
        },
        "summary": summary,
        "per_problem": results_per_problem,
    }

    outpath = OUTPUT_DIR / "intervention_splice_steer.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
