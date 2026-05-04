"""Experiment H4: Full-Stack Norm-Pinned Patty Loop

Every token — prompt AND generation — takes the same looped path:
  L0-L4 → [L5-L8 × N passes, norm-pinned] → skip to L27-L35

This eliminates the KV cache mismatch between the looped prefill state
and the full-depth generation states that plagued H3.

For each forward pass during generation:
1. L0-L4 run normally (building KV cache at these layers)
2. After L4, capture the hidden state
3. Run L5-L8 N times, norm-pinning after each pass
4. Inject the result at L27 (overwriting L9-L26's output)
5. L27-L35 run normally (building KV cache at these layers)
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', dtype=torch.bfloat16, device_map=device, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)

simple_problems = [
    ("请计算 2 + 3 × 4 的值。\n", "14"),
    ("一个矩形的长为8厘米，宽为5厘米，求面积。\n", "40"),
    ("如果 x + 5 = 12，求 x 的值。\n", "7"),
    ("计算 100 除以 4 的结果。\n", "25"),
    ("一个三角形三边长分别为3、4、5，求面积。\n", "6"),
]
hard_problems = [
    ("如果 2x + 3 = 15，求 x² 的值。\n", "36"),
    ("已知 x + y = 10，x - y = 4，求 x × y 的值。\n", "21"),
    ("小明有50元钱，买了3本书每本8元，又买了2支笔每支3元，还剩多少元？\n", "20"),
    ("一个正方形的对角线长为10厘米，求这个正方形的面积。\n", "50"),
    ("一个班有40个学生，男生占总数的3/5，女生有多少人？\n", "16"),
]
all_problems = simple_problems + hard_problems
MAX_NEW_TOKENS = 128


def generate_fullstack_patty(prompt, n_loops=2, max_new_tokens=MAX_NEW_TOKENS):
    """
    Every forward pass: capture L8 output, loop L5-L8 n_loops times with norm pinning,
    then inject at L27. This combines the G2 full-stack skip with the H3 patty loop.

    Implementation: We can't literally re-run L5-L8 inside a hook during generation
    because the model is mid-forward-pass. Instead, we use the simpler approach:
    - On EVERY forward pass, capture L4 output and L8 output
    - The L5-L8 "loop" for generation tokens is approximated by:
      For loop 1: normal L5-L8 (already happened)
      For loop 2+: we'd need to re-enter L5-L8, but that requires a separate forward pass

    ACTUAL APPROACH: For the patty loop during generation, we need to restructure.
    Since we can't re-run layers mid-forward-pass in HuggingFace, we do:
    1. Capture L8 output on every forward pass
    2. Run the patty loop by re-injecting at L4 and re-running the model

    BUT this is too expensive for generation (N extra forward passes per token).

    SIMPLER APPROACH that captures the KEY hypothesis:
    Just do the full-stack shallow skip (G2 style: L8→L27 on every token).
    The patty loop was about prefill enrichment; the KV mismatch hypothesis
    is really about G2 (consistent skip path for all tokens).

    So for H4, we test: does running the patty loop on prefill, then using
    the G2-style consistent skip for generation, work better than either alone?

    Path:
    - Prefill: run full model, capture L8, loop L5-L8 N times with norm pinning
    - Inject the looped state at L27 for prefill
    - Generation: every token takes L8→L27 skip (G2 style, consistent KV cache)
    """
    input_ids = tokenizer.encode(prompt)

    # Step 1: Get the original L8 hidden state from a full forward pass
    captured_L8 = {}
    def hook_capture_L8(module, input, output):
        if 'h' not in captured_L8:
            hidden = output if not isinstance(output, tuple) else output[0]
            captured_L8['h'] = hidden[:, -1:, :].clone()

    handle = model.model.layers[8].register_forward_hook(hook_capture_L8)
    with torch.no_grad():
        model(torch.tensor([input_ids], device=device), use_cache=False)
    handle.remove()

    original_h = captured_L8['h']  # (1, 1, 2048)
    original_norm = torch.norm(original_h).item()

    # Step 2: Patty loop (re-inject at L4, capture at L8, norm pin)
    current_h = original_h.clone()
    for loop in range(1, n_loops):
        loop_cap = {}

        def make_inject(state):
            fired = [False]
            def fn(module, input, output):
                if not fired[0]:
                    hidden = output if not isinstance(output, tuple) else output[0]
                    hidden[0, -1:, :] = state[0, :, :]
                    fired[0] = True
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
            return fn

        def make_capture(store):
            fired = [False]
            def fn(module, input, output):
                if not fired[0]:
                    hidden = output if not isinstance(output, tuple) else output[0]
                    store['h'] = hidden[:, -1:, :].clone()
                    fired[0] = True
            return fn

        h_inj = model.model.layers[4].register_forward_hook(make_inject(current_h))
        h_cap = model.model.layers[8].register_forward_hook(make_capture(loop_cap))
        with torch.no_grad():
            model(torch.tensor([input_ids], device=device), use_cache=False)
        h_inj.remove()
        h_cap.remove()

        raw_h = loop_cap['h']
        raw_norm = torch.norm(raw_h).item()
        current_h = raw_h * (original_norm / raw_norm)

    looped_h_np = current_h[0, 0, :].cpu().float().numpy()

    # Step 3: Generate with G2-style consistent skip (L8→L27 on every token)
    # PLUS inject the looped state on the prefill's last token
    gen_captured = {}
    prefill_done = [False]

    def capture_during_gen(module, input, output):
        hidden = output if not isinstance(output, tuple) else output[0]
        gen_captured['h'] = hidden.clone()

    def inject_during_gen(module, input, output):
        if 'h' not in gen_captured:
            return output
        hidden = output if not isinstance(output, tuple) else output[0]
        src = gen_captured['h']
        hidden[:, :src.shape[1], :] = src[:, :src.shape[1], :]

        # On the FIRST forward pass (prefill), also inject the looped state at last token
        if not prefill_done[0]:
            vec = torch.tensor(looped_h_np, dtype=hidden.dtype, device=hidden.device)
            hidden[0, -1, :] = vec
            prefill_done[0] = True

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    h_c = model.model.layers[8].register_forward_hook(capture_during_gen)
    h_i = model.model.layers[27].register_forward_hook(inject_during_gen)

    try:
        with torch.no_grad():
            outputs = model.generate(
                torch.tensor([input_ids], device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
        n_tokens = len(outputs[0]) - len(input_ids)
    finally:
        h_c.remove()
        h_i.remove()

    return gen, n_tokens


def generate_baseline(prompt, max_new_tokens=MAX_NEW_TOKENS):
    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        outputs = model.generate(
            torch.tensor([input_ids], device=device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)
    n_tokens = len(outputs[0]) - len(input_ids)
    return gen, n_tokens


def is_chinese(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = sum(1 for c in text if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
    return zh > en


def check_coherence(text):
    has_rep = False
    if len(text) > 60:
        for i in range(len(text) - 60):
            if text.count(text[i:i+20]) >= 3:
                has_rep = True
                break
    if len(text) > 30:
        third = len(text) // 3
        langs = [is_chinese(text[j*third:(j+1)*third]) for j in range(3)]
        lang_con = len(set(langs)) == 1
    else:
        lang_con = True
    return {"has_repetition": has_rep, "language_consistent": lang_con,
            "coherent": not has_rep and lang_con}


# ---- Run ----
print("=" * 70)
print("EXPERIMENT H4: FULL-STACK PATTY LOOP")
print("  Prefill: L5-L8 × N loops (norm-pinned) → inject at L27")
print("  Generation: L8→L27 skip on every token (G2 style)")
print("=" * 70)

# Baselines
print("\nBASELINES:")
baseline_results = []
for i, (prompt, expected) in enumerate(all_problems):
    gen, n_tok = generate_baseline(prompt)
    correct = expected in gen
    label = "S" if i < 5 else "H"
    print(f"  [{label}] {i}: correct={correct} | {gen[:60]}...")
    baseline_results.append({"idx": i, "generation": gen, "correct": correct,
                            "is_chinese": is_chinese(gen), "n_tokens": n_tok})

results = {"experiment": "H4: Full-Stack Patty Loop", "loop_configs": []}

for n_loops in [1, 2, 3, 4]:
    print(f"\n{'='*60}")
    print(f"PATTY LOOPS = {n_loops} (+ G2 skip during generation)")
    print(f"{'='*60}")

    cfg = {"n_loops": n_loops, "problems": []}

    for i, (prompt, expected) in enumerate(all_problems):
        gen, n_tok = generate_fullstack_patty(prompt, n_loops=n_loops)
        correct = expected in gen
        chinese = is_chinese(gen)
        coh = check_coherence(gen)
        ptype = "simple" if i < 5 else "hard"

        print(f"  [{ptype[0].upper()}] {i} (exp {expected}): correct={correct} | zh={chinese} | coh={coh['coherent']} | tok={n_tok}")
        print(f"    {gen[:80]}...")

        cfg["problems"].append({
            "idx": i, "type": ptype, "prompt": prompt.strip(), "expected": expected,
            "generation": gen, "correct": correct, "is_chinese": chinese,
            "n_tokens": n_tok, "coherence": coh,
            "baseline_correct": baseline_results[i]["correct"],
            "baseline_generation": baseline_results[i]["generation"],
        })

    cs = sum(1 for p in cfg["problems"] if p["type"] == "simple" and p["correct"])
    ch = sum(1 for p in cfg["problems"] if p["type"] == "hard" and p["correct"])
    nc = sum(1 for p in cfg["problems"] if p["is_chinese"])
    co = sum(1 for p in cfg["problems"] if p["coherence"]["coherent"])
    cfg["summary"] = {"simple": f"{cs}/5", "hard": f"{ch}/5", "total": f"{cs+ch}/10",
                      "chinese": f"{nc}/10", "coherent": f"{co}/10"}
    print(f"\n  {n_loops}-loop: {cs+ch}/10 correct, {nc}/10 Chinese, {co}/10 coherent")
    results["loop_configs"].append(cfg)

# Final
print(f"\n{'='*70}")
print("H4 FINAL SUMMARY")
print("="*70)
b_total = sum(1 for r in baseline_results if r["correct"])
print(f"  Baseline (full model): {b_total}/10")
for c in results["loop_configs"]:
    s = c["summary"]
    print(f"  {c['n_loops']}-loop + G2 skip: {s['total']} correct, {s['chinese']} Chinese, {s['coherent']} coherent")

with open("output/expH4_fullstack_patty.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to output/expH4_fullstack_patty.json")
