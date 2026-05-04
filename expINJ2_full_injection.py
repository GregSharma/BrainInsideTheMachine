"""expINJ2: Full Hidden-State Injection — Five Variants

Context (2026-04-13):
    The original expINJ (hidden_injection.py) tested whether upper layers can decode
    math content from a language-agnostic Z by injecting a math problem's last-token
    hidden state into a "Describe..." prompt at layer L*. Result: 1/3 math content,
    robust language decoding at L27+.

    Web (GPT-4.1) identified a critical methodological flaw: we only replaced the
    LAST TOKEN's hidden state, while the KV cache from ~20 "Describe..." tokens
    remained intact. Layers L*+1 through L35 use attention over that KV cache,
    so the injected math signal was 1 vector vs 20 cached "describe" vectors.
    The KV conflict explains the weak math-content result. MOAMS (cross-lingual
    transplant during math generation, aligned context) achieved 95% because
    context and injection were aligned, not adversarial.

    This script runs five variants to isolate the confound:

    A: FULL-STATE REPLACE into describe prompt
       Replace ALL token hidden states at L* (not just last).
       Kills the KV cache conflict entirely. If upper layers still can't
       decode math → Z genuinely doesn't carry decodable content at L*.
       Conditions: EN_math→EN_desc, ZH_math→EN_desc, noise→EN_desc, baseline.

    B: CROSS-LANGUAGE DESCRIBE (h'∘f∘h test)
       Same as A but with ZH "Describe..." prompt.
       Tests whether output language follows the describe prompt's language
       (h' determines language) or the injection source (f determines language).
       Conditions: EN_math→ZH_desc, ZH_math→ZH_desc.

    C: MOAMS POSITIVE CONTROL
       Use math problem as context. Replace last token at L* with cross-lang
       version. Context and injection are aligned. Expect ~95% (MOAMS baseline).
       If this fails → infrastructure bug, everything else suspect.
       Conditions: EN_math+ZH_last_tok, ZH_math+EN_last_tok.

    D: NON-ADVERSARIAL DESCRIPTION (Web's Fix B)
       Use "The problem being solved is:" as prompt — math-adjacent context,
       not adversarial. Last-token swap only. Tests whether the original failure
       was KV conflict (fixed by aligned context) vs insufficient signal (needs
       full-state replacement).
       Conditions: meta_prompt+EN_last, meta_prompt+ZH_last, meta_prompt+noise.

    E: CASCADE (all layers L* through L35)
       Replace ALL token hidden states at EVERY layer from L* through L35.
       Maximum erasure of the describe prompt. If A fails but E works →
       single-layer full replacement isn't enough, residual describe signal
       leaks through. If both fail → Z doesn't carry decodable math.
       Conditions: EN_math→EN_desc_cascade, ZH_math→EN_desc_cascade.

    L* sweep: [15, 18, 20, 25, 27, 30, 33] — added L18 (MOAMS cooperative peak).
    Problems: 3 diverse (algebra, geometry, number theory).
    Model: Qwen2.5-3B on RayGun (RTX 4070 Super, 12GB).
    Estimated: ~234 generations, ~12-15 min total.

Prior results informing design:
    - MOAMS: 162/180 (90%) transplants, 57/60 (95%) on hard AMC. Cross-lang cos≈0.995.
    - C2b: last_only_N36 = baseline, all_tokens_N1 = 0/20. Context tokens = computation.
    - C4: KV cache swap post-encoding = null. KV is inert. Residual drives generation.
    - C6b: Attention at last token = constant bias. MLP + residual do reasoning.
    - C7c: cos(v1_en, v1_zh) = 1.000 at L30. Convention-invariant bottleneck.
"""
import json
import time
import sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_GEN = 256

# L* candidates — added L18 (MOAMS cooperative zone peak)
L_STAR_CANDIDATES = [15, 18, 20, 25, 27, 30, 33]

# Prompts
DESCRIBE_TEMPLATE = {
    "en": "Describe in detail what mathematical operation or reasoning is being performed:",
    "zh": "详细描述正在进行什么数学运算或推理：",
}
META_PROMPT = "The problem being solved is:"

# Same 3 problems as original INJ for comparability
INJECT_PROBLEMS = [
    {
        "en": "Solve for x: 3x + 7 = 22",
        "zh": "求解x：3x + 7 = 22",
        "answer": "5",
        "category": "algebra",
    },
    {
        "en": "What is the area of a triangle with base 10 and height 7?",
        "zh": "底边为10、高为7的三角形面积是多少？",
        "answer": "35",
        "category": "geometry",
    },
    {
        "en": "Find the GCD of 84 and 120",
        "zh": "求84和120的最大公约数",
        "answer": "12",
        "category": "number_theory",
    },
]


def build_chat_prompt(tokenizer, text, sys_prompt=None):
    """Build input using chat template."""
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": text})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Hook classes
# ---------------------------------------------------------------------------

class FullStateCapture:
    """Capture ALL token hidden states (not just last) at specified layers."""

    def __init__(self):
        self.states = {}  # {layer: (batch, seq, d) tensor}
        self.hooks = []

    def register(self, model, layers):
        for L in layers:
            hook = model.model.layers[L].register_forward_hook(self._make_hook(L))
            self.hooks.append(hook)

    def _make_hook(self, layer):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.states[layer] = hidden.detach().clone()
        return hook_fn

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class FullStateInjector:
    """Replace ALL token hidden states at a specific layer during prefill.

    Handles sequence length mismatch: replaces positions 0..min(src, tgt)-1.
    Only fires once (during prefill, detected by seq_len > 1).
    """

    def __init__(self, layer, inject_states):
        self.layer = layer
        self.inject_states = inject_states  # (seq_src, d) tensor
        self.hook = None
        self.fired = False

    def register(self, model):
        self.fired = False
        self.hook = model.model.layers[self.layer].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if self.fired:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if hidden.dim() == 3 and hidden.shape[1] > 1:
            src_len = self.inject_states.shape[0]
            tgt_len = hidden.shape[1]
            n = min(src_len, tgt_len)
            hidden[:, :n, :] = self.inject_states[:n, :].unsqueeze(0)
            self.fired = True
        return output

    def remove(self):
        if self.hook:
            self.hook.remove()


class LastTokenInjector:
    """Replace ONLY last token hidden state at a specific layer during prefill."""

    def __init__(self, layer, inject_state):
        self.layer = layer
        self.inject_state = inject_state  # (d,) tensor
        self.hook = None
        self.fired = False

    def register(self, model):
        self.fired = False
        self.hook = model.model.layers[self.layer].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if self.fired:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if hidden.dim() == 3 and hidden.shape[1] > 1:
            hidden[:, -1, :] = self.inject_state
            self.fired = True
        return output

    def remove(self):
        if self.hook:
            self.hook.remove()


class CascadeInjector:
    """Replace ALL token hidden states at EVERY layer from L* through L_end.

    Registers hooks on layers [L_start, L_start+1, ..., L_end-1].
    Each hook fires once during prefill, replacing all token positions.
    """

    def __init__(self, l_start, l_end, inject_states_at_lstar):
        self.l_start = l_start
        self.l_end = l_end
        # We inject the same L* states at every layer. This tests whether
        # the content survives if we force it at every subsequent layer.
        self.inject_states = inject_states_at_lstar  # (seq, d)
        self.hooks = []
        self.fired = set()

    def register(self, model):
        self.fired = set()
        for L in range(self.l_start, self.l_end):
            hook = model.model.layers[L].register_forward_hook(self._make_hook(L))
            self.hooks.append(hook)

    def _make_hook(self, layer):
        def hook_fn(module, input, output):
            if layer in self.fired:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            if hidden.dim() == 3 and hidden.shape[1] > 1:
                src_len = self.inject_states.shape[0]
                tgt_len = hidden.shape[1]
                n = min(src_len, tgt_len)
                hidden[:, :n, :] = self.inject_states[:n, :].unsqueeze(0)
                self.fired.add(layer)
            return output
        return hook_fn

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def encode_and_capture_full(model, tokenizer, text, layers, device):
    """Encode text, capture FULL hidden states (all tokens) at specified layers."""
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    capture = FullStateCapture()
    capture.register(model, layers)

    with torch.no_grad():
        model(**inputs)

    # Return both full states and last-token states
    full_states = {}
    last_states = {}
    for L in layers:
        s = capture.states[L]
        if s.dim() == 3:
            full_states[L] = s.squeeze(0)  # (seq, d)
            last_states[L] = s[:, -1, :].squeeze(0)  # (d,)
        else:
            full_states[L] = s
            last_states[L] = s[-1, :]
    capture.remove()
    return full_states, last_states, inputs["input_ids"].shape[1]


def generate_with_injector(model, tokenizer, text, injector, device):
    """Generate with an arbitrary injector hook."""
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    injector.register(model)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_GEN, do_sample=False, temperature=1.0,
        )
    injector.remove()

    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def generate_baseline(model, tokenizer, text, device):
    """Generate without any injection."""
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_GEN, do_sample=False, temperature=1.0,
        )
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def detect_lang(text):
    """Simple CJK detection."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if cjk > len(text) * 0.1 else "en"


def contains_math_content(text, problem):
    """Check if output mentions math content relevant to the problem."""
    answer = problem["answer"]
    # Check for answer value, key numbers from problem, or category keywords
    keywords = {
        "algebra": ["solve", "equation", "variable", "x", answer, "求解", "方程"],
        "geometry": ["area", "triangle", "base", "height", answer, "面积", "三角", "底"],
        "number_theory": ["gcd", "greatest common", "divisor", answer, "最大公约数", "公约数"],
    }
    cat_words = keywords.get(problem["category"], [answer])
    text_lower = text.lower()
    hits = sum(1 for w in cat_words if w.lower() in text_lower)
    return hits, len(cat_words)


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------

def run_variant_A(model, tokenizer, device, L_star, problem, pi,
                  en_full, zh_full, en_last, zh_last, en_len, zh_len):
    """A: Full-state replace into EN describe prompt."""
    results = {}

    # EN math → EN describe (same-lang full state)
    inj = FullStateInjector(L_star, en_full[L_star])
    out = generate_with_injector(model, tokenizer, DESCRIBE_TEMPLATE["en"], inj, device)
    hits, total = contains_math_content(out, problem)
    results["en_math_en_desc"] = {"output": out, "lang": detect_lang(out),
                                   "math_hits": hits, "math_total": total}
    print(f"    A[EN→EN_desc]: lang={detect_lang(out)} math={hits}/{total} | {out[:120]}", flush=True)

    # ZH math → EN describe (cross-lang full state)
    inj = FullStateInjector(L_star, zh_full[L_star])
    out = generate_with_injector(model, tokenizer, DESCRIBE_TEMPLATE["en"], inj, device)
    hits, total = contains_math_content(out, problem)
    results["zh_math_en_desc"] = {"output": out, "lang": detect_lang(out),
                                   "math_hits": hits, "math_total": total}
    print(f"    A[ZH→EN_desc]: lang={detect_lang(out)} math={hits}/{total} | {out[:120]}", flush=True)

    # Noise → EN describe (control)
    noise = torch.randn_like(en_full[L_star])
    noise = noise / noise.norm() * en_full[L_star].norm()
    inj = FullStateInjector(L_star, noise)
    out = generate_with_injector(model, tokenizer, DESCRIBE_TEMPLATE["en"], inj, device)
    results["noise_en_desc"] = {"output": out, "lang": detect_lang(out)}
    print(f"    A[NOISE→EN_desc]: {out[:120]}", flush=True)

    # Baseline (no injection)
    out = generate_baseline(model, tokenizer, DESCRIBE_TEMPLATE["en"], device)
    results["baseline_en_desc"] = {"output": out}
    print(f"    A[BASELINE]: {out[:120]}", flush=True)

    return results


def run_variant_B(model, tokenizer, device, L_star, problem, pi,
                  en_full, zh_full, en_last, zh_last, en_len, zh_len):
    """B: Full-state replace into ZH describe prompt (h'∘f∘h test)."""
    results = {}

    # EN math → ZH describe
    inj = FullStateInjector(L_star, en_full[L_star])
    out = generate_with_injector(model, tokenizer, DESCRIBE_TEMPLATE["zh"], inj, device)
    hits, total = contains_math_content(out, problem)
    results["en_math_zh_desc"] = {"output": out, "lang": detect_lang(out),
                                   "math_hits": hits, "math_total": total}
    print(f"    B[EN→ZH_desc]: lang={detect_lang(out)} math={hits}/{total} | {out[:120]}", flush=True)

    # ZH math → ZH describe
    inj = FullStateInjector(L_star, zh_full[L_star])
    out = generate_with_injector(model, tokenizer, DESCRIBE_TEMPLATE["zh"], inj, device)
    hits, total = contains_math_content(out, problem)
    results["zh_math_zh_desc"] = {"output": out, "lang": detect_lang(out),
                                   "math_hits": hits, "math_total": total}
    print(f"    B[ZH→ZH_desc]: lang={detect_lang(out)} math={hits}/{total} | {out[:120]}", flush=True)

    return results


def run_variant_C(model, tokenizer, device, L_star, problem, pi,
                  en_full, zh_full, en_last, zh_last, en_len, zh_len):
    """C: MOAMS positive control — math context, last-token cross-lang swap."""
    results = {}

    # EN math context + ZH last token at L*
    inj = LastTokenInjector(L_star, zh_last[L_star])
    out = generate_with_injector(model, tokenizer, problem["en"], inj, device)
    hits, total = contains_math_content(out, problem)
    results["en_ctx_zh_last"] = {"output": out, "lang": detect_lang(out),
                                  "math_hits": hits, "math_total": total,
                                  "has_answer": problem["answer"] in out}
    print(f"    C[EN+ZH_last]: lang={detect_lang(out)} ans={problem['answer'] in out} | {out[:120]}", flush=True)

    # ZH math context + EN last token at L*
    inj = LastTokenInjector(L_star, en_last[L_star])
    out = generate_with_injector(model, tokenizer, problem["zh"], inj, device)
    hits, total = contains_math_content(out, problem)
    results["zh_ctx_en_last"] = {"output": out, "lang": detect_lang(out),
                                  "math_hits": hits, "math_total": total,
                                  "has_answer": problem["answer"] in out}
    print(f"    C[ZH+EN_last]: lang={detect_lang(out)} ans={problem['answer'] in out} | {out[:120]}", flush=True)

    return results


def run_variant_D(model, tokenizer, device, L_star, problem, pi,
                  en_full, zh_full, en_last, zh_last, en_len, zh_len):
    """D: Non-adversarial description — math-adjacent meta prompt + last-token swap."""
    results = {}

    # Meta prompt + EN math last token
    inj = LastTokenInjector(L_star, en_last[L_star])
    out = generate_with_injector(model, tokenizer, META_PROMPT, inj, device)
    hits, total = contains_math_content(out, problem)
    results["meta_en_last"] = {"output": out, "lang": detect_lang(out),
                                "math_hits": hits, "math_total": total}
    print(f"    D[META+EN_last]: lang={detect_lang(out)} math={hits}/{total} | {out[:120]}", flush=True)

    # Meta prompt + ZH math last token
    inj = LastTokenInjector(L_star, zh_last[L_star])
    out = generate_with_injector(model, tokenizer, META_PROMPT, inj, device)
    hits, total = contains_math_content(out, problem)
    results["meta_zh_last"] = {"output": out, "lang": detect_lang(out),
                                "math_hits": hits, "math_total": total}
    print(f"    D[META+ZH_last]: lang={detect_lang(out)} math={hits}/{total} | {out[:120]}", flush=True)

    # Meta prompt + noise (control)
    noise = torch.randn_like(en_last[L_star])
    noise = noise / noise.norm() * en_last[L_star].norm()
    inj = LastTokenInjector(L_star, noise)
    out = generate_with_injector(model, tokenizer, META_PROMPT, inj, device)
    results["meta_noise"] = {"output": out}
    print(f"    D[META+NOISE]: {out[:120]}", flush=True)

    return results


def run_variant_E(model, tokenizer, device, L_star, problem, pi,
                  en_full, zh_full, en_last, zh_last, en_len, zh_len):
    """E: Cascade — replace all tokens at EVERY layer from L* through L35."""
    results = {}

    # EN math → EN describe, cascade L*..L35
    inj = CascadeInjector(L_star, N_LAYERS, en_full[L_star])
    out = generate_with_injector(model, tokenizer, DESCRIBE_TEMPLATE["en"], inj, device)
    hits, total = contains_math_content(out, problem)
    results["en_math_cascade"] = {"output": out, "lang": detect_lang(out),
                                   "math_hits": hits, "math_total": total}
    print(f"    E[EN_cascade]: lang={detect_lang(out)} math={hits}/{total} | {out[:120]}", flush=True)

    # ZH math → EN describe, cascade L*..L35
    inj = CascadeInjector(L_star, N_LAYERS, zh_full[L_star])
    out = generate_with_injector(model, tokenizer, DESCRIBE_TEMPLATE["en"], inj, device)
    hits, total = contains_math_content(out, problem)
    results["zh_math_cascade"] = {"output": out, "lang": detect_lang(out),
                                   "math_hits": hits, "math_total": total}
    print(f"    E[ZH_cascade]: lang={detect_lang(out)} math={hits}/{total} | {out[:120]}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda"
    print(f"{'#' * 80}", flush=True)
    print(f"  Exp INJ2: Full Hidden-State Injection — Five Variants", flush=True)
    print(f"{'#' * 80}", flush=True)
    print(f"Model:     {MODEL_NAME}", flush=True)
    print(f"Problems:  {len(INJECT_PROBLEMS)}", flush=True)
    print(f"L* sweep:  {L_STAR_CANDIDATES}", flush=True)
    print(f"Variants:  A (full-state desc), B (cross-lang desc), C (MOAMS ctrl),", flush=True)
    print(f"           D (non-adversarial), E (cascade)", flush=True)
    print(f"Max gen:   {MAX_GEN}", flush=True)
    print(flush=True)

    t0 = time.time()

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    all_results = {}
    gen_count = 0

    for L_star in L_STAR_CANDIDATES:
        print(f"\n\n{'=' * 70}", flush=True)
        print(f"  L* = {L_star}", flush=True)
        print(f"{'=' * 70}", flush=True)

        l_results = {}

        for pi, problem in enumerate(INJECT_PROBLEMS):
            print(f"\n--- Problem {pi}: {problem['en'][:50]}... ---", flush=True)

            # Pre-capture: encode both languages, get full + last-token states at L*
            # We need states at L* for A/B/C/D, and also L* for E (cascade uses L* states)
            en_full, en_last, en_len = encode_and_capture_full(
                model, tokenizer, problem["en"], [L_star], device
            )
            zh_full, zh_last, zh_len = encode_and_capture_full(
                model, tokenizer, problem["zh"], [L_star], device
            )

            cos_sim = torch.nn.functional.cosine_similarity(
                en_last[L_star].unsqueeze(0),
                zh_last[L_star].unsqueeze(0),
            ).item()
            print(f"  cos(EN_last, ZH_last) at L{L_star} = {cos_sim:.4f}", flush=True)
            print(f"  Token counts — EN: {en_len}, ZH: {zh_len}", flush=True)

            prob_results = {"cos_en_zh_last": cos_sim, "en_tokens": en_len, "zh_tokens": zh_len}

            # --- Run all five variants ---
            args = (model, tokenizer, device, L_star, problem, pi,
                    en_full, zh_full, en_last, zh_last, en_len, zh_len)

            print(f"\n  [Variant A: Full-state → EN describe]", flush=True)
            prob_results["A"] = run_variant_A(*args)
            gen_count += 4

            print(f"\n  [Variant B: Full-state → ZH describe]", flush=True)
            prob_results["B"] = run_variant_B(*args)
            gen_count += 2

            print(f"\n  [Variant C: MOAMS positive control]", flush=True)
            prob_results["C"] = run_variant_C(*args)
            gen_count += 2

            print(f"\n  [Variant D: Non-adversarial meta prompt]", flush=True)
            prob_results["D"] = run_variant_D(*args)
            gen_count += 3

            print(f"\n  [Variant E: Cascade L{L_star}→L{N_LAYERS-1}]", flush=True)
            prob_results["E"] = run_variant_E(*args)
            gen_count += 2

            l_results[f"problem_{pi}"] = prob_results

        all_results[f"L{L_star}"] = l_results

    wall = time.time() - t0
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"  DONE — {gen_count} generations in {wall:.0f}s ({wall/60:.1f}min)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # --- Summary table ---
    print(f"\n{'=' * 70}", flush=True)
    print(f"  SUMMARY: Math content hits by variant × L*", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'Variant':<8} {'Condition':<20} ", end="", flush=True)
    for L in L_STAR_CANDIDATES:
        print(f"L{L:<4}", end=" ", flush=True)
    print(flush=True)
    print("-" * 70, flush=True)

    for var in ["A", "B", "C", "D", "E"]:
        # Collect conditions for this variant
        conditions = set()
        for L in L_STAR_CANDIDATES:
            for pi in range(len(INJECT_PROBLEMS)):
                key = f"problem_{pi}"
                if key in all_results.get(f"L{L}", {}):
                    v = all_results[f"L{L}"][key].get(var, {})
                    conditions.update(v.keys())
        conditions -= {"baseline_en_desc", "noise_en_desc", "meta_noise"}  # skip controls
        for cond in sorted(conditions):
            print(f"  {var:<6} {cond:<20}", end=" ", flush=True)
            for L in L_STAR_CANDIDATES:
                hits = 0
                total_probs = 0
                for pi in range(len(INJECT_PROBLEMS)):
                    key = f"problem_{pi}"
                    entry = all_results.get(f"L{L}", {}).get(key, {}).get(var, {}).get(cond, {})
                    if "math_hits" in entry:
                        hits += (1 if entry["math_hits"] > 0 else 0)
                        total_probs += 1
                if total_probs > 0:
                    print(f"{hits}/{total_probs}  ", end="", flush=True)
                else:
                    print(f"  -  ", end="", flush=True)
            print(flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / "expINJ2_full_injection.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "INJ2_full_injection",
            "model": MODEL_NAME,
            "L_star_candidates": L_STAR_CANDIDATES,
            "problems": [p["en"] for p in INJECT_PROBLEMS],
            "variants": ["A_full_state_en_desc", "B_full_state_zh_desc",
                         "C_moams_control", "D_nonadversarial_meta",
                         "E_cascade"],
            "results": all_results,
            "total_generations": gen_count,
            "wall_time_s": wall,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
