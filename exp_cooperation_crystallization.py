"""Experiment: Does cooperation track crystallization?

HYPOTHESIS: The cooperative zone (L18-L21, positive cross-layer delta cosine) is the
measurable signature of active reasoning. When cooperation dies, reasoning stops and
narration begins. If true, cooperation should correlate with answer crystallization
(rising p(answer) at the output).

METHOD:
- 20 test problems with known answers, zh + en = 40 runs
- During generation (128 tokens), capture per-token:
  (a) MLP delta at L17-L22 → compute cross-layer cosine at L18-L21 (cooperation signal)
  (b) Full logits at output → p(answer_token) (crystallization signal)
- Compute per-token correlation between cooperation and crystallization
- Also: does cooperation precede crystallization? (lead/lag)
"""

import json
import numpy as np
import torch
import random as pyrandom
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_NAME = "Qwen/Qwen2.5-3B"
device = "cuda"
MAX_NEW_TOKENS = 128
N_TEST = 20

print(f"Loading {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
# Capture L17-L22 for cooperation, plus L35 for completeness
COOP_LAYERS = list(range(17, 23))
t0 = time.time()


# =============================================================================
# Problems with known answers
# =============================================================================
def generate_problems_with_answers(n=200, seed=42):
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            ans = a + b
            zh, en = f"计算 {a} + {b} 的值。", f"Calculate {a} + {b}."
        else:
            ans = a * b
            zh, en = f"计算 {a} × {b} 的值。", f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "answer": ans})

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        from math import comb
        ans = comb(n_val, k_val)
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。",
                          "en": f"Find the value of C({n_val}, {k_val}).",
                          "answer": ans})

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        ans = a % b
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？",
                          "en": f"What is the remainder when {a} is divided by {b}?",
                          "answer": ans})

    for _ in range(per_cat):
        w, h = rng.randint(2, 50), rng.randint(2, 50)
        ans = w * h
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
                          "en": f"A rectangle has length {w} and width {h}. Find its area.",
                          "answer": ans})

    for _ in range(per_cat):
        a1, d_val = rng.randint(1, 20), rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2 * a1 + (n_terms - 1) * d_val) // 2
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_terms} 项之和。",
                          "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_terms} terms.",
                          "answer": ans})
    return problems


all_problems = generate_problems_with_answers(200, seed=42)
test_problems = all_problems[:N_TEST]


# =============================================================================
# Capture: per-token MLP deltas at cooperative layers + per-token logits
# =============================================================================
class CoopCrystalCapture:
    """Captures per-token MLP deltas at L17-L22 AND per-token logits during generation."""

    def __init__(self, model, coop_layers, max_steps=200):
        self.model = model
        self.coop_layers = coop_layers
        self.max_steps = max_steps
        self.d = model.config.hidden_size
        self.device = next(model.parameters()).device
        self._hooks = []
        self._mlp_buffers = {}
        self._mlp_counters = {}
        # Logit capture via lm_head hook
        self._logit_buffer = []

    def _reset(self):
        self._mlp_buffers = {}
        self._mlp_counters = {}
        for li in self.coop_layers:
            self._mlp_buffers[li] = torch.empty(
                (self.max_steps, self.d), device=self.device, dtype=torch.float32
            )
            self._mlp_counters[li] = 0
        self._logit_buffer = []

    def _register(self):
        self._hooks = []
        for li in self.coop_layers:
            layer = self.model.model.layers[li]

            def make_hook(idx):
                def hook(module, inp, out):
                    i = self._mlp_counters[idx]
                    if i < self.max_steps:
                        self._mlp_buffers[idx][i].copy_(out[0, -1, :].float())
                        self._mlp_counters[idx] = i + 1
                return hook
            self._hooks.append(layer.mlp.register_forward_hook(make_hook(li)))

        # Hook lm_head to capture logits
        def logit_hook(module, inp, out):
            # out shape: (batch, seq, vocab) — take last token
            self._logit_buffer.append(out[0, -1, :].detach().float().cpu())
        self._hooks.append(self.model.lm_head.register_forward_hook(logit_hook))

    def _unregister(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run(self, input_ids, max_new_tokens=128):
        self._reset()
        self._register()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        self._unregister()

        prompt_len = input_ids.shape[1]
        gen_ids = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_gen = len(gen_ids)

        # Build per-token MLP delta arrays (skip prefill = index 0)
        mlp_deltas = {}
        for li in self.coop_layers:
            n = self._mlp_counters[li]
            if n > 1:
                mlp_deltas[li] = self._mlp_buffers[li][1:n].cpu().numpy()

        # Logits: skip prefill (index 0)
        logits_all = self._logit_buffer[1:] if len(self._logit_buffer) > 1 else []

        return {
            "gen_text": gen_text,
            "n_gen_tokens": n_gen,
            "gen_ids": gen_ids.cpu().numpy(),
            "mlp_deltas": mlp_deltas,
            "logits": logits_all,
        }


# =============================================================================
# Run
# =============================================================================
capturer = CoopCrystalCapture(model, COOP_LAYERS, max_steps=MAX_NEW_TOKENS + 50)

all_results = []

for lang in ["zh", "en"]:
    print(f"\n{'='*60}")
    print(f"Running {lang.upper()}")
    print(f"{'='*60}")

    for pi, prob in enumerate(test_problems):
        prompt = prob[lang]
        answer = prob["answer"]
        answer_str = str(answer)

        # Tokenize the answer to find target token(s)
        answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        result = capturer.run(input_ids, max_new_tokens=MAX_NEW_TOKENS)

        # Compute per-token cooperation (cross-layer cosine at L18-L21)
        per_tok_coop = []
        n_tok = min(
            *(result["mlp_deltas"].get(li, np.empty((0, d_model))).shape[0]
              for li in COOP_LAYERS)
        ) if all(li in result["mlp_deltas"] for li in COOP_LAYERS) else 0

        for t in range(n_tok):
            # Average cross-layer cosine for L18->L19, L19->L20, L20->L21
            cos_vals = []
            for i in range(1, len(COOP_LAYERS)):
                prev_li = COOP_LAYERS[i-1]
                curr_li = COOP_LAYERS[i]
                if prev_li in result["mlp_deltas"] and curr_li in result["mlp_deltas"]:
                    d_prev = result["mlp_deltas"][prev_li][t]
                    d_curr = result["mlp_deltas"][curr_li][t]
                    n_p = np.linalg.norm(d_prev) + 1e-10
                    n_c = np.linalg.norm(d_curr) + 1e-10
                    cos_vals.append(float(np.dot(d_prev, d_curr) / (n_p * n_c)))
            per_tok_coop.append(float(np.mean(cos_vals)) if cos_vals else 0.0)

        # Compute per-token p(answer) from logits
        per_tok_p_answer = []
        for t_idx, logit_vec in enumerate(result["logits"]):
            if t_idx >= n_tok:
                break
            probs = torch.softmax(logit_vec, dim=0)
            # Sum probability over all answer tokens
            p_ans = sum(probs[tid].item() for tid in answer_tokens)
            per_tok_p_answer.append(p_ans)

        # Trim to same length
        n_paired = min(len(per_tok_coop), len(per_tok_p_answer))
        per_tok_coop = per_tok_coop[:n_paired]
        per_tok_p_answer = per_tok_p_answer[:n_paired]

        # Compute correlation
        if n_paired > 5:
            coop_arr = np.array(per_tok_coop)
            cryst_arr = np.array(per_tok_p_answer)
            # Pearson correlation
            if coop_arr.std() > 1e-10 and cryst_arr.std() > 1e-10:
                corr = float(np.corrcoef(coop_arr, cryst_arr)[0, 1])
            else:
                corr = 0.0

            # Lead/lag: does cooperation peak BEFORE crystallization?
            coop_peak_tok = int(np.argmax(coop_arr[:min(50, n_paired)]))  # within first 50
            cryst_threshold_tok = None
            for t in range(n_paired):
                if cryst_arr[t] > 0.1:  # p(answer) > 10%
                    cryst_threshold_tok = t
                    break

            # Max cooperation in first vs second half
            half = n_paired // 2
            coop_first_half = float(coop_arr[:half].mean()) if half > 0 else 0.0
            coop_second_half = float(coop_arr[half:].mean()) if half > 0 else 0.0
        else:
            corr = None
            coop_peak_tok = None
            cryst_threshold_tok = None
            coop_first_half = None
            coop_second_half = None

        entry = {
            "problem_idx": pi,
            "lang": lang,
            "answer": answer,
            "answer_str": answer_str,
            "n_paired_tokens": n_paired,
            "correlation": corr,
            "coop_peak_token": coop_peak_tok,
            "cryst_threshold_token": cryst_threshold_tok,
            "coop_first_half_mean": coop_first_half,
            "coop_second_half_mean": coop_second_half,
            "per_tok_coop": per_tok_coop,
            "per_tok_p_answer": per_tok_p_answer,
            "got_correct": answer_str in result["gen_text"],
        }
        all_results.append(entry)

        if (pi + 1) % 5 == 0:
            print(f"  {lang} {pi+1}/{N_TEST} done ({time.time()-t0:.0f}s)")


# =============================================================================
# Analysis
# =============================================================================
print(f"\n{'='*60}")
print("ANALYSIS")
print(f"{'='*60}")

# 1. Overall correlation
correlations = [r["correlation"] for r in all_results if r["correlation"] is not None]
print(f"\nOverall cooperation-crystallization correlation:")
print(f"  Mean r = {np.mean(correlations):.4f}")
print(f"  Median r = {np.median(correlations):.4f}")
print(f"  Std r = {np.std(correlations):.4f}")
print(f"  N = {len(correlations)}")

# 2. By language
for lang in ["zh", "en"]:
    lang_corrs = [r["correlation"] for r in all_results
                  if r["lang"] == lang and r["correlation"] is not None]
    if lang_corrs:
        print(f"\n  {lang.upper()}: mean r = {np.mean(lang_corrs):.4f}, "
              f"median = {np.median(lang_corrs):.4f}, n = {len(lang_corrs)}")

# 3. Correct vs incorrect
correct = [r for r in all_results if r["got_correct"]]
incorrect = [r for r in all_results if not r["got_correct"]]

print(f"\n  Correct ({len(correct)}):")
if correct:
    c_corrs = [r["correlation"] for r in correct if r["correlation"] is not None]
    c_coop_1 = [r["coop_first_half_mean"] for r in correct if r["coop_first_half_mean"] is not None]
    c_coop_2 = [r["coop_second_half_mean"] for r in correct if r["coop_second_half_mean"] is not None]
    if c_corrs:
        print(f"    Correlation: {np.mean(c_corrs):.4f}")
    if c_coop_1 and c_coop_2:
        print(f"    Coop 1st half: {np.mean(c_coop_1):.4f}, 2nd half: {np.mean(c_coop_2):.4f}")

print(f"  Incorrect ({len(incorrect)}):")
if incorrect:
    i_corrs = [r["correlation"] for r in incorrect if r["correlation"] is not None]
    i_coop_1 = [r["coop_first_half_mean"] for r in incorrect if r["coop_first_half_mean"] is not None]
    i_coop_2 = [r["coop_second_half_mean"] for r in incorrect if r["coop_second_half_mean"] is not None]
    if i_corrs:
        print(f"    Correlation: {np.mean(i_corrs):.4f}")
    if i_coop_1 and i_coop_2:
        print(f"    Coop 1st half: {np.mean(i_coop_1):.4f}, 2nd half: {np.mean(i_coop_2):.4f}")

# 4. Lead/lag analysis
leads = []
for r in all_results:
    if r["coop_peak_token"] is not None and r["cryst_threshold_token"] is not None:
        lead = r["cryst_threshold_tok"] - r["coop_peak_token"] if "cryst_threshold_tok" in r else r["cryst_threshold_token"] - r["coop_peak_token"]
        leads.append(lead)
if leads:
    print(f"\n  Lead/lag (cryst_threshold - coop_peak):")
    print(f"    Mean = {np.mean(leads):.1f} tokens (positive = coop leads)")
    print(f"    Median = {np.median(leads):.1f}")

# 5. Per-problem detail
print(f"\n--- PER-PROBLEM DETAIL ---")
print(f"{'#':<4} {'Lang':<5} {'Ans':<8} {'Correct':<8} {'r':<8} {'CoopPk':<8} {'CrystTh':<8} {'Coop1H':<8} {'Coop2H':<8}")
print("-" * 75)
for r in all_results:
    corr_str = f"{r['correlation']:.3f}" if r['correlation'] is not None else "N/A"
    pk_str = str(r['coop_peak_token']) if r['coop_peak_token'] is not None else "N/A"
    th_str = str(r['cryst_threshold_token']) if r['cryst_threshold_token'] is not None else "N/A"
    c1_str = f"{r['coop_first_half_mean']:.3f}" if r['coop_first_half_mean'] is not None else "N/A"
    c2_str = f"{r['coop_second_half_mean']:.3f}" if r['coop_second_half_mean'] is not None else "N/A"
    print(f"  {r['problem_idx']:<3} {r['lang']:<5} {r['answer_str']:<8} {'Y' if r['got_correct'] else 'N':<8} "
          f"{corr_str:<8} {pk_str:<8} {th_str:<8} {c1_str:<8} {c2_str:<8}")


# =============================================================================
# Save
# =============================================================================
# Strip per-token arrays for JSON (keep summary stats)
save_results = []
for r in all_results:
    entry = {k: v for k, v in r.items() if k not in ["per_tok_coop", "per_tok_p_answer"]}
    # Save first 20 tokens of each for inspection
    entry["per_tok_coop_first20"] = r["per_tok_coop"][:20]
    entry["per_tok_p_answer_first20"] = r["per_tok_p_answer"][:20]
    save_results.append(entry)

output = {
    "experiment": "Cooperation-Crystallization Correlation",
    "model": MODEL_NAME,
    "n_problems": N_TEST,
    "max_new_tokens": MAX_NEW_TOKENS,
    "coop_layers": COOP_LAYERS,
    "method": "Per-token cross-layer MLP delta cosine (L17-L22) vs p(answer_token) from logits",
    "overall_correlation_mean": float(np.mean(correlations)) if correlations else None,
    "overall_correlation_median": float(np.median(correlations)) if correlations else None,
    "n_correct": len(correct),
    "n_incorrect": len(incorrect),
    "results": save_results,
    "runtime_seconds": time.time() - t0,
}

with open("output/exp_cooperation_crystallization.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n\nResults saved to output/exp_cooperation_crystallization.json")
print(f"Total runtime: {time.time()-t0:.0f}s")
