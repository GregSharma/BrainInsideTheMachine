#!/usr/bin/env python3
"""AMC 12A 2025 sweep: baseline vs soft deflation (alpha=0.1, r=4, L20-35, refresh/25).

Skips problems requiring figures/asy: 5, 10, 14, 20, 24, 25.
Runs 19 problems x 2 conditions (baseline greedy, soft_a0.1 greedy).
"""
import json, time, os, re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
REFRESH_EVERY = 25
ALPHA = 0.1

SYS = (
    "You are solving an AMC 12A multiple choice math problem. "
    "Think step by step, show your work, then clearly state your "
    "final answer as (A), (B), (C), (D), or (E)."
)

# Problems to skip (require images/asy)
SKIP_PROBLEMS = {5, 10, 14, 20, 24, 25}

# Answer key for non-skipped problems
ANSWER_KEY = {
    1: 'E', 2: 'B', 3: 'A', 4: 'B', 6: 'B', 7: 'C', 8: 'E', 9: 'E',
    11: 'A', 12: 'B', 13: 'D', 15: 'C', 16: 'D', 17: 'A', 18: 'C',
    19: 'E', 21: 'A', 22: 'E', 23: 'C',
}


def parse_amc_problems(filepath):
    """Parse AMC 12A markdown into {num: text} dict."""
    with open(filepath, 'r') as f:
        content = f.read()

    problems = {}
    # Split by "Problem N" headers
    parts = re.split(r'^Problem (\d+)\s*$', content, flags=re.MULTILINE)
    # parts = [preamble, '1', text1, '2', text2, ...]
    for i in range(1, len(parts) - 1, 2):
        num = int(parts[i])
        raw = parts[i + 1].strip()

        if num in SKIP_PROBLEMS:
            continue

        # Remove [Solution](...) links
        raw = re.sub(r'\[Solution\]\([^)]*\)', '', raw)
        # Remove the dashed header line
        raw = re.sub(r'^-+\s*$', '', raw, flags=re.MULTILINE)

        # Check for [asy] blocks - skip if present (safety net)
        if '[asy]' in raw.lower():
            print(f"  Skipping problem {num} — contains [asy]")
            continue

        # Convert markdown image syntax: ![alt](url) -> alt text only
        # For LaTeX rendered images, the alt text contains the LaTeX
        raw = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', raw)

        # Clean up excessive whitespace
        raw = re.sub(r'\n{3,}', '\n\n', raw).strip()

        # Remove trailing empty sections like "See also" etc
        raw = re.sub(r'\nSee also.*', '', raw, flags=re.DOTALL)

        if num in ANSWER_KEY:
            problems[num] = raw

    return problems


def make_prompt(text):
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


class SoftDeflation:
    """q_new = q - alpha * V V^T q at target layers."""

    def __init__(self, model, layers, r=4, alpha=0.1, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.U_r = {}
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_hook(ell)
            )
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
                return output
            q = output
            batch, seq, d = q.shape
            hd, n_q, n_kv = 128, 16, 2
            gs = n_q // n_kv
            q = q.view(batch, seq, n_q, hd)
            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs, (kv_h + 1) * gs
                qg = q[:, :, s:e, :]
                proj = qg @ U @ U.T
                q[:, :, s:e, :] = qg - self.alpha * proj
            return q.view(batch, seq, d)
        return hook

    def refresh_basis(self, past_kv):
        for ell in self.target_layers:
            keys = past_kv.layers[ell].keys
            self.U_r[ell] = {}
            for kv_h in range(keys.shape[1]):
                K = keys[0, kv_h, :, :].float()
                if K.shape[0] < self.r:
                    continue
                _, _, Vh = torch.linalg.svd(K, full_matrices=False)
                self.U_r[ell][kv_h] = Vh[:self.r, :].T.contiguous().to(
                    DEVICE, dtype=torch.float16
                )

    def start_gen(self):
        self.is_generating = True
        self.step_count = 0

    def tick(self, past_kv):
        self.step_count += 1
        if self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.U_r.clear()


def run_one(model, tokenizer, prompt, deflator=None):
    """Run one greedy generation. Returns (text, n_tokens, time_s)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None
    t0 = time.time()

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                if deflator:
                    deflator.start_gen()
                    deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv,
                            use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)

            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            if deflator:
                deflator.tick(past_kv)

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids), round(dt, 1)


# Value-to-letter mappings per problem (from the answer choices)
VALUE_MAP = {
    1: {'3:30':'A', '3:45':'B', '4:00':'C', '4:15':'D', '4:30':'E'},
    2: {'3.5':'A', '4':'B', '4.5':'C', '5':'D', '6':'E'},
    3: {'28':'A', '29':'B', '30':'C', '32':'D', '33':'E'},
    4: {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E'},
    6: {'1/6':'A', '1/5':'B', '2/9':'C', '3/13':'D', '1/4':'E'},
    7: {'20':'A', '21':'B', '22':'C', '23':'D', '24':'E'},
    8: {'57/11':'A', '59/11':'B', '60/11':'C', '61/11':'D', '63/11':'E'},
    9: {'3/4':'A', '1':'B', '7/5':'C', '3/2':'D', '5/3':'E'},
    11: {'5':'A', '17':'B', '113/3':'D', '54':'E'},
    12: {'-5/3':'A', '-3/2':'B', '-6/5':'C', '-5/6':'D', '-2/3':'E'},
    13: {'3/130':'A', '3/143':'B', '5/143':'C', '1/26':'D', '5/78':'E'},
    15: {'8':'A', '9':'B', '10':'C', '11':'D', '12':'E'},
    16: {'18':'A', '19':'B', '20':'C', '21':'D', '22':'E'},
    17: {'6':'A', '8':'B', '10':'C', '12':'D', '14':'E'},
    18: {'36':'A', '84':'B', '186':'C', '336':'D', '486':'E'},
    19: {'-k':'A', '-k+1':'B', '1':'C', 'k-1':'D', 'k':'E'},
    21: {'8':'A', '9':'B', '10':'C', '11':'D', '12':'E'},
    22: {'1/12':'A', '1/9':'B', '1/8':'C', '1/6':'D', '1/4':'E'},
    23: {'511':'A', '2584':'B', '9841':'C', '17711':'D', '19682':'E'},
}


def extract_boxed(text):
    """Extract content of \\boxed{...} handling nested braces."""
    found = []
    for m in re.finditer(r'\\boxed\{', text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            found.append(text[start:i-1])
    return found


def normalize_value(val):
    """Convert LaTeX to simple string."""
    val = val.strip()
    val = re.sub(r'\\text\{([^}]*)\}', r'\1', val)
    val = re.sub(r'\\frac\{(-?[^}]+)\}\{([^}]+)\}', r'\1/\2', val)
    val = re.sub(r'\\dfrac\{(-?[^}]+)\}\{([^}]+)\}', r'\1/\2', val)
    val = val.replace(' ', '')
    return val


def extract_answer(text, pnum=None):
    """Pull (A)-(E) from the output using boxed values + value mapping."""
    boxed = extract_boxed(text)
    if boxed:
        val = boxed[-1]
        # Direct letter?
        clean = re.sub(r'\\text\{([^}]*)\}', r'\1', val).strip()
        clean = clean.replace('(','').replace(')','').strip()
        if clean in 'ABCDE' and len(clean) == 1:
            return clean
        # Normalize and map
        if pnum is not None:
            norm = normalize_value(val)
            vmap = VALUE_MAP.get(pnum, {})
            if norm in vmap:
                return vmap[norm]

    # Fallback: explicit (A)-(E) in tail
    tail = text[-400:]
    m = re.findall(r'\(([A-E])\)', tail)
    if m:
        return m[-1]

    # "answer is X"
    m = re.findall(r'answer is[^A-E]*?([A-E])\b', tail, re.IGNORECASE)
    if m:
        return m[-1]

    return "?"


def main():
    # Parse problems
    problems = parse_amc_problems("2025_AMC_12A.md")
    print(f"Parsed {len(problems)} AMC 12A problems (skipped {len(SKIP_PROBLEMS)} image problems)")
    for num in sorted(problems.keys()):
        print(f"  Problem {num}: {len(problems[num])} chars, answer={ANSWER_KEY[num]}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded. {len(model.model.layers)} layers.\n", flush=True)

    results = []
    sorted_nums = sorted(problems.keys())

    for pnum in sorted_nums:
        ptext = problems[pnum]
        correct = ANSWER_KEY[pnum]
        prompt = make_prompt(ptext)

        for cond_name, do_deflate in [("baseline", False), ("soft_a0.1", True)]:
            print(f"--- Problem {pnum} / {cond_name} ---", flush=True)
            deflator = None
            if do_deflate:
                deflator = SoftDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R,
                                         alpha=ALPHA, refresh_every=REFRESH_EVERY)
            text, ntok, dt = run_one(model, tokenizer, prompt, deflator=deflator)
            if deflator:
                deflator.remove()
            ans = extract_answer(text, pnum)
            looped = ntok >= MAX_TOKENS
            print(f"    {ntok} tok, {dt}s, ans={ans} (correct={correct}), "
                  f"{'LOOP' if looped else 'stop'}", flush=True)
            results.append({
                "problem": pnum,
                "condition": cond_name,
                "correct": correct,
                "answer": ans,
                "n_tokens": ntok,
                "time_s": dt,
                "looped": looped,
                "output": text,
            })

    # Save
    os.makedirs("output", exist_ok=True)
    outpath = "output/exp_amc_sweep_deflation.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {outpath}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Prob':>5s}  {'Baseline Ans':>12s}  {'Soft Ans':>10s}  {'Correct':>8s}  "
          f"{'Base Tok':>9s}  {'Soft Tok':>9s}  {'Base OK':>8s}  {'Soft OK':>8s}")
    print("-" * 80)

    base_correct = 0
    soft_correct = 0
    base_total_tok = 0
    soft_total_tok = 0

    for pnum in sorted_nums:
        base = [r for r in results if r["problem"] == pnum and r["condition"] == "baseline"][0]
        soft = [r for r in results if r["problem"] == pnum and r["condition"] == "soft_a0.1"][0]
        correct = ANSWER_KEY[pnum]

        b_ok = "YES" if base["answer"] == correct else "no"
        s_ok = "YES" if soft["answer"] == correct else "no"

        if base["answer"] == correct:
            base_correct += 1
        if soft["answer"] == correct:
            soft_correct += 1

        b_tok_str = f"{base['n_tokens']}{'*' if base['looped'] else ''}"
        s_tok_str = f"{soft['n_tokens']}{'*' if soft['looped'] else ''}"

        base_total_tok += base["n_tokens"]
        soft_total_tok += soft["n_tokens"]

        print(f"{pnum:5d}  {base['answer']:>12s}  {soft['answer']:>10s}  {correct:>8s}  "
              f"{b_tok_str:>9s}  {s_tok_str:>9s}  {b_ok:>8s}  {s_ok:>8s}")

    print("-" * 80)
    print(f"{'TOTAL':>5s}  {base_correct:>12d}  {soft_correct:>10d}  {len(sorted_nums):>8d}  "
          f"{base_total_tok:>9d}  {soft_total_tok:>9d}")
    print(f"\nBaseline: {base_correct}/{len(sorted_nums)} correct")
    print(f"Soft a0.1: {soft_correct}/{len(sorted_nums)} correct")
    print(f"Delta: {soft_correct - base_correct:+d}")
    print(f"Baseline total tokens: {base_total_tok}  |  Soft total tokens: {soft_total_tok}")

    # Flag changes
    print(f"\n--- CHANGES ---")
    for pnum in sorted_nums:
        base = [r for r in results if r["problem"] == pnum and r["condition"] == "baseline"][0]
        soft = [r for r in results if r["problem"] == pnum and r["condition"] == "soft_a0.1"][0]
        if base["answer"] != soft["answer"]:
            correct = ANSWER_KEY[pnum]
            direction = ""
            if soft["answer"] == correct and base["answer"] != correct:
                direction = "FIXED"
            elif base["answer"] == correct and soft["answer"] != correct:
                direction = "BROKE"
            else:
                direction = "CHANGED"
            print(f"  Problem {pnum}: {base['answer']} -> {soft['answer']} "
                  f"(correct={correct}) [{direction}]")


if __name__ == "__main__":
    main()
