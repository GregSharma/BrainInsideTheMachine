#!/usr/bin/env python3
"""
expSMA3_amc_sweep.py — Sensitivity-Modulated Attention v3
AMC 12A 2025 problems with letter-based answer grading.

Changes from v2:
  - AMC 12A 2025 problems (19 usable, English only)
  - Letter (A)-(E) grading — no regex numeric false positives
  - Gate output cached from forward pass — no redundant gate_proj recompute
  - Full output logging (every response, grading method, matched pattern)
  - --test flag: run 1 problem to validate pipeline before full sweep

Excluded problems: #5,10,14,20,24 (require figures), #25 (broken answer key)
"""

import argparse, json, re, time, os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_NEW = 1024
DEVICE = "cuda"

# AMC 12A 2025 answer key (source: AoPS wiki)
ANSWER_KEY = {
    1: 'E', 2: 'B', 3: 'A', 4: 'B', 6: 'B', 7: 'C', 8: 'E', 9: 'E',
    11: 'A', 12: 'B', 13: 'D', 15: 'C', 16: 'D', 17: 'A', 18: 'C',
    19: 'E', 21: 'A', 22: 'E', 23: 'C',
}
EXCLUDE = {5, 10, 14, 20, 24, 25}

SYSTEM_PROMPT = (
    "You are solving an AMC 12A multiple choice math problem. "
    "Think step by step, show your work, then clearly state your "
    "final answer as (A), (B), (C), (D), or (E)."
)


# ── Problem parsing ──────────────────────────────────────────────

def strip_img(text):
    """Remove markdown image syntax, keep alt-text content."""
    # Display math: ![\[...\]](url) → \[...\]
    text = re.sub(
        r'!\[\\\[(.+?)\\\]\]\([^)]+\)',
        lambda m: '\n\\[' + m.group(1) + '\\]\n',
        text, flags=re.DOTALL,
    )
    # Inline LaTeX: ![$...$](url) → content (no $ delimiters)
    text = re.sub(r'!\[\$([^$]+)\$\]\([^)]+\)', r'\1', text)
    # Asy figures → [FIGURE]
    text = re.sub(
        r'!\[\[asy\].*?\[/asy\]\]\([^)]+\)',
        '[FIGURE]', text, flags=re.DOTALL,
    )
    # Any remaining image links
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
    # Hyperlinks: [text](url) or [text](url "title")
    text = re.sub(r'\[Solution\]\([^)]+\)', '', text)
    text = re.sub(r'\[[^\]]+\]\([^)]+"[^"]*"\s*\)', '', text)
    return text


def clean_latex(text):
    """Clean LaTeX formatting for model consumption."""
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    text = text.replace('\\qquad', '    ')
    text = text.replace('\\quad', '  ')
    text = text.replace('{:}', ':')
    text = text.replace('~', ' ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_amc_problems(md_path):
    """Parse AMC 12A problems from markdown file.

    Returns list of dicts: {num, text, answer}.
    """
    with open(md_path) as f:
        raw = f.read()

    parts = re.split(r'\nProblem (\d+)\n-+\n', raw)
    # parts = [preamble, "1", content1, "2", content2, ...]

    problems = []
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        content = parts[i + 1]

        if num in EXCLUDE:
            continue
        if num not in ANSWER_KEY:
            continue

        # Truncate at [Solution] link
        sol_idx = content.find('[Solution]')
        if sol_idx > 0:
            content = content[:sol_idx]

        content = strip_img(content)
        content = clean_latex(content)

        problems.append({
            'num': num,
            'text': content,
            'answer': ANSWER_KEY[num],
        })

    return problems


# ── Answer extraction ────────────────────────────────────────────

def extract_answer_letter(text, problem_text=None):
    """Extract (A)-(E) from model output.

    Returns (letter, method) or (None, 'none').
    Priority:
      1. "answer is (X)" / "Answer: X"
      2. \\boxed{X} where X is a letter
      3. "choose/select (X)"
      4. \\boxed{value} mapped to answer choices
      5. Last (X) in final 30%
      6. Last (X) in non-echoed portion
      7. Bare letter on own line
    """
    # Detect echoed answer choices in first 40% of text and skip past them.
    cutoff = int(len(text) * 0.4)
    echo_match = re.search(r'\(E\)[^\n]*', text[:cutoff])
    search_start = echo_match.end() if echo_match else 0
    tail = text[search_start:]

    # 1. Explicit answer statement (in non-echoed portion)
    m = re.search(
        r'(?:the\s+)?(?:correct\s+)?answer\s+is\s*\(?([A-E])\)?',
        tail, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper(), 'answer_is'

    # 1b. "option X is correct" / "choice X"
    m = re.search(
        r'(?:option|choice)\s+\(?([A-E])\)?\s+is\s+correct',
        tail, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper(), 'option_correct'

    # 2. Boxed letter
    m = re.search(r'\\boxed\{\(?([A-E])\)?\}', tail)
    if m:
        return m.group(1).upper(), 'boxed_letter'

    # 3. Choose/select
    m = re.search(
        r'(?:choose|select|pick)\s+\(?([A-E])\)?',
        tail, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper(), 'choose'

    # 4. Boxed VALUE mapped to answer choices in the problem text
    if problem_text:
        boxed_vals = re.findall(r'\\boxed\{([^}]+)\}', tail)
        if boxed_vals:
            val = boxed_vals[-1].strip()
            # Clean \( \) wrappers
            val = re.sub(r'\\[()]', '', val).strip()
            for letter in 'ABCDE':
                # Find what follows (X) in the problem text
                pat = rf'\({letter}\)\s*(.+?)(?=\s*\([A-E]\)|$)'
                cm = re.search(pat, problem_text)
                if cm:
                    choice_val = cm.group(1).strip()
                    # Normalize both for comparison
                    if _norm(val) == _norm(choice_val):
                        return letter, 'boxed_value'

    # 5. Last (X) in final 30% of full text
    tail30 = text[int(len(text) * 0.7):]
    matches_tail = re.findall(r'\(([A-E])\)', tail30)
    if matches_tail:
        return matches_tail[-1].upper(), 'last_paren_tail'

    # 6. Last (X) in non-echoed portion
    matches_tail2 = re.findall(r'\(([A-E])\)', tail)
    if matches_tail2:
        return matches_tail2[-1].upper(), 'last_paren'

    # 7. Bare letter on own line (non-echoed)
    m = re.search(r'(?:^|\n)\s*([A-E])\s*$', tail, re.MULTILINE)
    if m:
        return m.group(1).upper(), 'bare_letter'

    return None, 'none'


def _norm(s):
    """Normalize a string for fuzzy value matching."""
    s = s.strip()
    s = re.sub(r'\s+', '', s)  # collapse whitespace
    s = s.replace('\\', '')    # remove stray backslashes
    s = s.lower()
    return s


# ── Sensitivity Modulator (optimized) ────────────────────────────

class SensitivityModulator:
    """Sensitivity-modulated attention: gate → σ(x)(1-σ(x)) → W_down² → modulate Q.

    Optimization: gate output is captured from the normal forward pass via a
    hook on gate_proj, eliminating the redundant gate_proj recomputation that
    was the main bottleneck in v1/v2.
    """

    def __init__(self, model, alpha=1.0, mode="sensitivity", seed=None):
        self.model = model
        self.alpha = alpha
        self.mode = mode
        self.gate_cache = {}
        self.sensitivity = {}
        self.hooks = []
        self.enabled = True
        self.n_layers = len(model.model.layers)
        self.rng = torch.Generator(device=DEVICE)
        if seed is not None:
            self.rng.manual_seed(seed)

        # Precompute W_down².T per layer (contiguous for fast matmul)
        self.w_down_sq_T = {}
        for ell in range(self.n_layers):
            W = model.model.layers[ell].mlp.down_proj.weight.detach()
            self.w_down_sq_T[ell] = W.pow(2).T.contiguous()  # (d_ff, d_model)

        # Register hooks
        for ell in range(self.n_layers):
            # Capture gate_proj output for free during normal forward
            h_gate = model.model.layers[ell].mlp.gate_proj.register_forward_hook(
                self._make_gate_cache_hook(ell)
            )
            self.hooks.append(h_gate)

            # MLP post-hook: compute sensitivity from cached gate output
            h_mlp = model.model.layers[ell].mlp.register_forward_hook(
                self._make_mlp_hook(ell)
            )
            self.hooks.append(h_mlp)

            # Q-proj pre-hook on NEXT layer: modulate query input
            if ell + 1 < self.n_layers:
                h_q = model.model.layers[
                    ell + 1
                ].self_attn.q_proj.register_forward_pre_hook(
                    self._make_q_hook(ell)
                )
                self.hooks.append(h_q)

    def _make_gate_cache_hook(self, layer_idx):
        def hook(module, input, output):
            if self.enabled:
                self.gate_cache[layer_idx] = output.detach()
        return hook

    def _make_mlp_hook(self, layer_idx):
        def hook(module, input, output):
            if not self.enabled:
                return
            x_gate = self.gate_cache.pop(layer_idx, None)
            if x_gate is None:
                return
            with torch.no_grad():
                sig = torch.sigmoid(x_gate)
                tau = sig * (1.0 - sig)  # [0, 0.25], always non-negative
                s = torch.matmul(tau, self.w_down_sq_T[layer_idx])
                s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)

                if self.mode == "random":
                    s = (
                        torch.rand(
                            s.shape, device=s.device, dtype=s.dtype,
                            generator=self.rng,
                        )
                        + 0.5
                    )
                    s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)
                elif self.mode == "uniform":
                    s = torch.ones_like(s)
                elif self.mode == "inverse":
                    s = 1.0 / (s + 1e-8)
                    s = s / (s.mean(dim=-1, keepdim=True) + 1e-8)

                if self.alpha != 1.0:
                    s = 1.0 + self.alpha * (s - 1.0)

                self.sensitivity[layer_idx] = s
        return hook

    def _make_q_hook(self, prev_layer_idx):
        def hook(module, args):
            if not self.enabled:
                return
            if prev_layer_idx not in self.sensitivity:
                return
            h = args[0]
            s = self.sensitivity[prev_layer_idx]
            if s.shape[1] != h.shape[1]:
                s = s[:, -h.shape[1]:, :]
            return (h * s,) + args[1:]
        return hook

    def clear_caches(self):
        self.gate_cache.clear()
        self.sensitivity.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.gate_cache.clear()
        self.sensitivity.clear()
        self.w_down_sq_T.clear()


# ── Generation ───────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def build_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


@torch.inference_mode()
def generate(model, tokenizer, prompt_text, max_new=MAX_NEW):
    """Generate response. Returns (text, elapsed_s, n_tokens)."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    out = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False,
        temperature=None, top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    dt = time.time() - t0
    new_ids = out[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, dt, len(new_ids)


# ── Main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SMA3: AMC 12A + letter grading")
    ap.add_argument('--test', action='store_true',
                    help='Run 1 problem only (validation mode)')
    ap.add_argument('--max-new', type=int, default=MAX_NEW,
                    help=f'Max new tokens (default {MAX_NEW})')
    args = ap.parse_args()

    print("=" * 70)
    print("expSMA3: Sensitivity-Modulated Attention v3")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max new tokens: {args.max_new}")
    print(f"  Mode: {'TEST (1 problem)' if args.test else 'FULL (19 problems)'}")
    print("=" * 70)

    # Parse problems
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "2025_AMC_12A.md")
    problems = parse_amc_problems(md_path)
    print(f"\nParsed {len(problems)} AMC 12A problems")

    if args.test:
        problems = problems[:1]
        print(f"  TEST MODE: problem #{problems[0]['num']} only")

    # Preview first problem
    p0 = problems[0]
    print(f"\n--- Problem #{p0['num']} (preview) ---")
    preview = p0['text'][:400]
    print(preview)
    if len(p0['text']) > 400:
        print("...")
    print(f"\n  Correct answer: ({p0['answer']})")
    print("---\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    d_ff = model.config.intermediate_size
    print(f"  {n_layers} layers, d_model={d_model}, d_ff={d_ff}")

    # Define conditions
    conditions = [
        {"name": "baseline",    "alpha": None, "mode": None},
        {"name": "sma_a0.3",   "alpha": 0.3,  "mode": "sensitivity"},
        {"name": "sma_a0.5",   "alpha": 0.5,  "mode": "sensitivity"},
        {"name": "sma_a1.0",   "alpha": 1.0,  "mode": "sensitivity"},
        {"name": "uniform",    "alpha": 1.0,  "mode": "uniform"},
        {"name": "random_s42", "alpha": 1.0,  "mode": "random"},
        {"name": "inverse",    "alpha": 1.0,  "mode": "inverse"},
    ]

    results = []
    summary = {c["name"]: {"correct": 0, "total": 0, "truncated": 0,
                            "none": 0, "avg_tokens": 0.0}
               for c in conditions}
    t_start = time.time()

    for pi, prob in enumerate(problems):
        print(f"\nProblem #{prob['num']} ({pi + 1}/{len(problems)})")
        prompt = build_prompt(tokenizer, prob['text'])

        for cond in conditions:
            cname = cond["name"]
            mod = None

            if cond["alpha"] is not None:
                seed = 42 if cond["mode"] == "random" else None
                mod = SensitivityModulator(
                    model, alpha=cond["alpha"],
                    mode=cond["mode"], seed=seed,
                )

            text, dt, n_tok = generate(model, tokenizer, prompt, args.max_new)

            if mod is not None:
                mod.remove()
            torch.cuda.empty_cache()

            pred, method = extract_answer_letter(text, problem_text=prob['text'])
            correct = pred == prob['answer']

            truncated = (n_tok >= args.max_new)
            results.append({
                "problem_num": prob["num"],
                "condition": cname,
                "correct_answer": prob["answer"],
                "predicted": pred,
                "match": correct,
                "grading_method": method,
                "truncated": truncated,
                "output": text,
                "gen_tokens": n_tok,
                "gen_time_s": round(dt, 2),
            })

            summary[cname]["total"] += 1
            if correct:
                summary[cname]["correct"] += 1
            if truncated:
                summary[cname]["truncated"] += 1
            if pred is None:
                summary[cname]["none"] += 1
            summary[cname]["avg_tokens"] += n_tok

            mark = "OK" if correct else "XX"
            trunc_tag = " TRUNC" if truncated else ""
            print(
                f"  {cname:<14s}  pred=({pred})  "
                f"correct=({prob['answer']})  {mark}  "
                f"[{method}]  {dt:.1f}s  {n_tok}tok{trunc_tag}"
            )

    # Finalize averages
    for cname in summary:
        s = summary[cname]
        if s["total"] > 0:
            s["avg_tokens"] = round(s["avg_tokens"] / s["total"], 1)

    # Summary
    elapsed = time.time() - t_start
    n_probs = len(problems)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    base_c = summary["baseline"]["correct"]
    hdr = (f"  {'condition':<14s}  {'score':>7s}  {'delta':>5s}  "
           f"{'trunc':>5s}  {'none':>4s}  {'avg_tok':>7s}")
    print(hdr)
    print("  " + "-" * 56)
    for cond in conditions:
        c = summary[cond["name"]]
        pct = 100 * c["correct"] / c["total"] if c["total"] > 0 else 0
        delta = c["correct"] - base_c
        d_str = f"{delta:+d}" if cond["name"] != "baseline" else "--"
        print(
            f"  {cond['name']:<14s}  "
            f"{c['correct']:>2d}/{c['total']:<2d}  "
            f"{d_str:>5s}  "
            f"{c['truncated']:>5d}  "
            f"{c['none']:>4d}  "
            f"{c['avg_tokens']:>7.0f}"
        )

    print(f"\nWall time: {elapsed:.0f}s")

    # Save
    out_path = os.path.join("output", "expSMA3_amc_sweep.json")
    os.makedirs("output", exist_ok=True)
    payload = {
        "experiment": "expSMA3_amc_sweep",
        "description": "SMA with AMC 12A 2025, letter-based grading, optimized hooks",
        "model": MODEL_NAME,
        "max_new_tokens": args.max_new,
        "n_problems": n_probs,
        "test_mode": args.test,
        "system_prompt": SYSTEM_PROMPT,
        "answer_key": {str(k): v for k, v in ANSWER_KEY.items()},
        "conditions": [c["name"] for c in conditions],
        "summary": summary,
        "results": results,
        "wall_time_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
