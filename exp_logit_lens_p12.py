"""Logit lens + system prompt test on P12.

Part A: Does system prompt "." or no system prompt solve P12 without deflation?
         Tests the claim that minimal system prompt IS deflation.

Part B: Logit lens during baseline (looping) generation.
         At each generation step, project intermediate hidden states through
         layernorm + lm_head. Track whether -3/2 related tokens appear at
         intermediate layers even while the output is looping.
         If yes: the model KNOWS but can't SAY. Aphasia confirmed mechanistically.
"""
import json, time, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048
CORRECT = "B"  # -3/2

SYS_FULL = ("You are solving an AMC 12A multiple choice math problem. "
            "Think step by step, show your work, then clearly state your "
            "final answer as (A), (B), (C), (D), or (E).")

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

def make_prompt(sys_content=None):
    """Build chat-template prompt with optional system message."""
    if sys_content is None:
        # No system message at all
        return f"<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return (f"<|im_start|>system\n{sys_content}<|im_end|>\n"
                f"<|im_start|>user\n{P12_TEXT}<|im_end|>\n"
                f"<|im_start|>assistant\n")

def extract_answer(text):
    if not text:
        return "?"
    if re.search(r'\\boxed\{[^}]*-\\frac\{3\}\{2\}', text):
        return "B"
    if re.search(r'\\boxed\{[^}]*-3/2', text):
        return "B"
    m = re.findall(r'\\boxed\{[^}]*\b([A-E])\b[^}]*\}', text)
    if m: return m[-1]
    m = re.findall(r'\\boxed\{[^}]*(-?\d+/\d+)[^}]*\}', text)
    if m:
        mapping = {"-5/3": "A", "-3/2": "B", "-6/5": "C", "-5/6": "D", "-2/3": "E"}
        return mapping.get(m[-1], "?")
    m = re.findall(r'\(([A-E])\)', text[-500:])
    if m: return m[-1]
    if "-3/2" in text[-500:]: return "B"
    m = re.findall(r'answer is.*?([A-E])\b', text[-500:], re.IGNORECASE)
    if m: return m[-1]
    return "?"


def generate_simple(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
    """Simple generation, returns (text, n_tokens, time_s)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    past_kv = None
    gen_ids = []
    t0 = time.time()

    for step in range(max_tokens):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids), round(dt, 1)


# ============ LOGIT LENS ============

class LogitLensProbe:
    """Capture hidden states at target layers during generation.
    Project through layernorm + lm_head to get intermediate logits."""

    def __init__(self, model, target_layers, probe_every=50):
        self.model = model
        self.target_layers = target_layers
        self.probe_every = probe_every
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.hooks = []
        self.captured = {}  # {layer: hidden_state}
        self.step = 0
        self.should_probe = False
        self.results = []  # list of {step, layer, top5_tokens, top5_logits, answer_token_data}
        self._install()

    def _install(self):
        for li in self.target_layers:
            h = self.model.model.layers[li].register_forward_hook(self._make_hook(li))
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            if self.should_probe:
                # output is (hidden_states, ...) for decoder layers
                hs = output[0]  # might be (batch, seq, d) or (batch, d)
                if hs.dim() == 3:
                    hs = hs[:, -1, :]  # (1, d_model)
                # else already (1, d_model) with KV cache
                self.captured[li] = hs.detach()
        return hook

    def tick(self, answer_token_ids):
        """Call after each generation step."""
        self.step += 1
        self.should_probe = (self.step % self.probe_every == 0)

        if self.should_probe and self.captured:
            for li in sorted(self.captured.keys()):
                hs = self.captured[li]
                # Apply final layernorm + lm_head
                with torch.no_grad():
                    normed = self.norm(hs)
                    logits = self.lm_head(normed).squeeze(0)  # (vocab_size,)
                    probs = F.softmax(logits, dim=-1)

                    # Top 5
                    top5_vals, top5_ids = logits.topk(5)
                    top5_probs = probs[top5_ids]

                    # Answer token data
                    answer_data = []
                    for name, tid in answer_token_ids.items():
                        rank = (logits > logits[tid]).sum().item() + 1
                        answer_data.append({
                            "name": name,
                            "token_id": tid,
                            "logit": logits[tid].item(),
                            "prob": probs[tid].item(),
                            "rank": rank,
                        })

                self.results.append({
                    "step": self.step,
                    "layer": li,
                    "top5_tokens": top5_ids.tolist(),
                    "top5_logits": [round(v, 3) for v in top5_vals.tolist()],
                    "top5_probs": [round(v, 6) for v in top5_probs.tolist()],
                    "answer_tokens": answer_data,
                })

            self.captured.clear()
            self.should_probe = False

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def generate_with_lens(model, tokenizer, prompt, target_layers, answer_token_ids,
                       probe_every=50, max_tokens=MAX_TOKENS):
    """Generate with logit lens probing at intermediate layers."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    past_kv = None
    gen_ids = []
    t0 = time.time()

    probe = LogitLensProbe(model, target_layers, probe_every=probe_every)

    for step in range(max_tokens):
        with torch.no_grad():
            # Activate probing on the right steps
            probe.should_probe = ((step + 1) % probe_every == 0)

            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            probe.tick(answer_token_ids)

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    lens_results = probe.results
    probe.remove()

    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids), round(dt, 1), lens_results


def main():
    print("=" * 70)
    print("LOGIT LENS + SYSTEM PROMPT TEST ON P12")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Loaded {MODEL_NAME}, {n_layers} layers.\n", flush=True)

    # Answer token IDs to track
    answer_token_ids = {
        "B": 33,             # the answer letter
        "-": 12,             # start of -3/2
        "3": 18,             # the 3 in -3/2
        "frac": 37018,       # \frac
        "(B": 5349,          # (B) as single token
        # Wrong answers for comparison
        "A": 32,
        "C": 34,
        "D": 35,
        "E": 36,
    }

    results = {}

    # ===== PART A: SYSTEM PROMPT VARIANTS =====
    print("\n" + "#" * 70)
    print("PART A: SYSTEM PROMPT VARIANTS (no deflation)")
    print("Does minimal system prompt solve P12 on its own?")
    print("#" * 70 + "\n", flush=True)

    sys_conditions = [
        ("full_amc_sys",   SYS_FULL),    # the standard system prompt
        ("dot_sys",        "."),          # minimal: single period
        ("no_sys",         None),          # no system message at all
        ("empty_sys",      ""),            # empty string system message
    ]

    part_a_results = []
    for name, sys_content in sys_conditions:
        prompt = make_prompt(sys_content)
        prompt_tokens = len(tokenizer.encode(prompt))
        desc = f"sys='{sys_content}'" if sys_content is not None else "sys=None"
        print(f"\n{'='*60}", flush=True)
        print(f"  {name}: {desc} ({prompt_tokens} prompt tokens)", flush=True)
        print(f"{'='*60}", flush=True)

        text, ntok, dt = generate_simple(model, tokenizer, prompt)
        ans = extract_answer(text)
        looped = ntok >= MAX_TOKENS - 5
        status = "CORRECT" if ans == CORRECT else ("LOOP" if looped else f"WRONG({ans})")

        print(f">>> {name}: ans={ans} tok={ntok} {status} t={dt}s", flush=True)
        if ntok < MAX_TOKENS - 5:
            print(f"    Last 200 chars: ...{text[-200:]}", flush=True)

        part_a_results.append({
            "condition": name,
            "sys_content": sys_content,
            "prompt_tokens": prompt_tokens,
            "answer": ans,
            "correct": ans == CORRECT,
            "n_tokens": ntok,
            "time_s": dt,
            "looped": looped,
            "output_last500": text[-500:] if text else "",
        })

    results["part_a_system_prompt"] = part_a_results

    # ===== PART B: LOGIT LENS DURING BASELINE =====
    print("\n" + "#" * 70)
    print("PART B: LOGIT LENS DURING BASELINE (looping) GENERATION")
    print("Probing intermediate layers for -3/2 signal during the loop.")
    print("#" * 70 + "\n", flush=True)

    # Layers to probe: early, mid, surgery range, canyon, output
    target_layers = [0, 5, 10, 15, 18, 20, 25, 27, 30, 33, 35]
    probe_every = 50  # probe every 50 tokens

    # Run baseline with lens
    prompt = make_prompt(SYS_FULL)
    print("Running baseline (full sys, no deflation) with logit lens...", flush=True)
    text_bl, ntok_bl, dt_bl, lens_bl = generate_with_lens(
        model, tokenizer, prompt, target_layers, answer_token_ids,
        probe_every=probe_every, max_tokens=800)  # cap at 800 to save time, deep enough in loop

    ans_bl = extract_answer(text_bl)
    looped_bl = ntok_bl >= 795
    print(f">>> baseline: ans={ans_bl} tok={ntok_bl} t={dt_bl}s", flush=True)

    results["part_b_baseline_lens"] = {
        "answer": ans_bl,
        "n_tokens": ntok_bl,
        "time_s": dt_bl,
        "looped": looped_bl,
        "probes": lens_bl,
    }

    # ===== ANALYSIS =====
    print("\n" + "=" * 70)
    print("LOGIT LENS ANALYSIS: Is -3/2 visible at intermediate layers?")
    print("=" * 70, flush=True)

    # Organize by step
    steps = sorted(set(p["step"] for p in lens_bl))
    for step in steps:
        print(f"\n--- Step {step} ---")
        step_probes = [p for p in lens_bl if p["step"] == step]
        for p in sorted(step_probes, key=lambda x: x["layer"]):
            # Find answer B rank
            b_data = next((a for a in p["answer_tokens"] if a["name"] == "B"), None)
            top_tok_names = [tokenizer.decode([t]) for t in p["top5_tokens"]]
            b_rank = b_data["rank"] if b_data else "?"
            b_prob = b_data["prob"] if b_data else 0

            # Compare B vs other answer letters
            answer_ranks = {a["name"]: a["rank"] for a in p["answer_tokens"]
                          if a["name"] in "ABCDE"}
            b_is_best = all(answer_ranks.get("B", 999999) <= answer_ranks.get(x, 999999)
                           for x in "ACDE")
            marker = " <<<" if b_is_best and b_rank < 100 else ""

            print(f"  L{p['layer']:>2d}: top5={top_tok_names}  "
                  f"B_rank={b_rank:>5d} B_prob={b_prob:.6f}"
                  f"  [A={answer_ranks.get('A','?'):>5}, C={answer_ranks.get('C','?'):>5}, "
                  f"D={answer_ranks.get('D','?'):>5}, E={answer_ranks.get('E','?'):>5}]{marker}")

    # Summary: best B rank across all layers at each step
    print("\n" + "=" * 70)
    print("SUMMARY: Best rank of 'B' across layers at each generation step")
    print("=" * 70)
    for step in steps:
        step_probes = [p for p in lens_bl if p["step"] == step]
        b_ranks = []
        for p in step_probes:
            b_data = next((a for a in p["answer_tokens"] if a["name"] == "B"), None)
            if b_data:
                b_ranks.append((p["layer"], b_data["rank"], b_data["prob"]))
        if b_ranks:
            best = min(b_ranks, key=lambda x: x[1])
            print(f"  Step {step:>4d}: best B rank = {best[1]:>5d} at L{best[0]:>2d} "
                  f"(prob={best[2]:.6f})")

    # Part A summary
    print("\n" + "=" * 70)
    print("PART A SUMMARY: SYSTEM PROMPT AS DEFLATION")
    print("=" * 70)
    for r in part_a_results:
        st = "CORRECT" if r["correct"] else ("LOOP" if r["looped"] else f"WRONG({r['answer']})")
        sys_desc = f"sys='{r['sys_content']}'" if r["sys_content"] is not None else "sys=None"
        print(f"  {r['condition']:20s} ptok={r['prompt_tokens']:>3d}  "
              f"ans={r['answer']}  tok={r['n_tokens']:>4d}  {st}")

    out_path = "output/exp_logit_lens_p12.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
