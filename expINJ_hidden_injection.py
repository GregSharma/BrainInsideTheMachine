"""expINJ: Hidden-state injection experiment.

Greg's idea: can the model introspect on language-agnostic representations?

Design:
1. Encode a math problem (in language X) through layers 0..L*
2. Separately encode "Describe what computation is being done:" through layers 0..L*
3. At L*, replace the LAST TOKEN's hidden state of the "Describe" prompt
   with the last token's hidden state from the math problem
4. Let layers L*+1..L35 decode the hybrid
5. If output describes the math regardless of source language X → Z is language-agnostic at L*

Controls:
- Same problem in EN vs ZH (should produce same description → Z is lang-agnostic)
- Different problem (should produce different description → not just a fixed mode)
- Random noise injection (should produce garbage → model needs real structure)
- No injection baseline (should describe... nothing? or the describe prompt itself?)

L* candidates: L30 (rank-1 bottleneck, convention-free, cos(v1_en,v1_zh)=1.000)
"""
import json, time, sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048
MAX_GEN = 256  # shorter, just need to see if it describes math

# The "describe" prompt — deliberately vague to not bias output
DESCRIBE_TEMPLATE = {
    "en": "Describe in detail what mathematical operation or reasoning is being performed:",
    "zh": "详细描述正在进行什么数学运算或推理：",
}

# Test problems — pick 3 diverse ones
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

# L* candidates to sweep
L_STAR_CANDIDATES = [15, 20, 25, 27, 30, 33]


def build_chat_prompt(tokenizer, text, sys_prompt=None):
    """Build input using chat template."""
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class HiddenStateCapture:
    """Hook to capture hidden states at a specific layer."""

    def __init__(self):
        self.states = {}  # {layer: tensor}
        self.hooks = []

    def register(self, model, layers):
        for L in layers:
            hook = model.model.layers[L].register_forward_hook(
                self._make_hook(L)
            )
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


class HiddenStateInjector:
    """Hook to inject a hidden state at a specific layer, last token only.

    Only fires during the FIRST forward pass (prefill), not during
    subsequent auto-regressive generation steps. We detect prefill
    by checking if seq_len > 1 (prefill processes full prompt,
    generation processes 1 token at a time).
    """

    def __init__(self, layer, inject_state):
        self.layer = layer
        self.inject_state = inject_state  # (d,) tensor
        self.hook = None
        self.fired = False

    def register(self, model):
        self.fired = False
        self.hook = model.model.layers[self.layer].register_forward_hook(
            self._hook_fn
        )

    def _hook_fn(self, module, input, output):
        if self.fired:
            return output
        # output may be tensor or tuple
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if hidden.dim() == 3:
            if hidden.shape[1] > 1:
                hidden[:, -1, :] = self.inject_state
                self.fired = True
        elif hidden.dim() == 2:
            if hidden.shape[0] > 1:
                hidden[-1, :] = self.inject_state
                self.fired = True
        return output

    def remove(self):
        if self.hook:
            self.hook.remove()


def encode_and_capture(model, tokenizer, text, layers, device):
    """Encode text and capture hidden states at specified layers."""
    prompt = build_chat_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    capture = HiddenStateCapture()
    capture.register(model, layers)

    with torch.no_grad():
        model(**inputs)

    states = {}
    for L in layers:
        s = capture.states[L]
        if s.dim() == 3:
            states[L] = s[:, -1, :].squeeze(0)  # (batch, seq, d) -> (d,)
        elif s.dim() == 2:
            states[L] = s[-1, :]  # (seq, d) -> (d,)
        else:
            states[L] = s  # fallback
    capture.remove()
    return states


def inject_and_generate(model, tokenizer, describe_text, inject_state, L_star, device):
    """Encode describe prompt, inject state at L*, generate."""
    prompt = build_chat_prompt(tokenizer, describe_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    injector = HiddenStateInjector(L_star, inject_state)
    injector.register(model)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN,
            do_sample=False,
            temperature=1.0,
        )

    injector.remove()

    # Decode only generated tokens
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main():
    device = "cuda"
    print(f"{'#' * 80}", flush=True)
    print(f"  Exp INJ: Hidden-State Injection", flush=True)
    print(f"{'#' * 80}", flush=True)
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(f"Problems: {len(INJECT_PROBLEMS)}", flush=True)
    print(f"L* sweep: {L_STAR_CANDIDATES}", flush=True)
    print(f"Max gen:  {MAX_GEN}", flush=True)
    print(flush=True)

    t0 = time.time()

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    results = {}

    for L_star in L_STAR_CANDIDATES:
        print(f"\n\n{'=' * 70}", flush=True)
        print(f"  L* = {L_star}", flush=True)
        print(f"{'=' * 70}", flush=True)

        l_results = {}

        for pi, problem in enumerate(INJECT_PROBLEMS):
            print(f"\n--- Problem {pi+1}: {problem['en'][:50]}... ---", flush=True)
            prob_results = {}

            # Capture hidden states for EN and ZH versions
            en_states = encode_and_capture(
                model, tokenizer, problem["en"], [L_star], device
            )
            zh_states = encode_and_capture(
                model, tokenizer, problem["zh"], [L_star], device
            )

            # Check cross-lingual similarity at L*
            cos_sim = torch.nn.functional.cosine_similarity(
                en_states[L_star].unsqueeze(0),
                zh_states[L_star].unsqueeze(0),
            ).item()
            print(f"  cos(EN, ZH) at L{L_star} = {cos_sim:.4f}", flush=True)
            prob_results["cos_en_zh"] = cos_sim

            # Condition 1: Inject EN math into EN describe
            out_en_en = inject_and_generate(
                model, tokenizer, DESCRIBE_TEMPLATE["en"],
                en_states[L_star], L_star, device,
            )
            print(f"\n  [EN→EN] {out_en_en[:200]}", flush=True)
            prob_results["en_inject_en_describe"] = out_en_en

            # Condition 2: Inject ZH math into EN describe
            out_zh_en = inject_and_generate(
                model, tokenizer, DESCRIBE_TEMPLATE["en"],
                zh_states[L_star], L_star, device,
            )
            print(f"  [ZH→EN] {out_zh_en[:200]}", flush=True)
            prob_results["zh_inject_en_describe"] = out_zh_en

            # Condition 3: Random noise injection
            noise = torch.randn_like(en_states[L_star])
            noise = noise / noise.norm() * en_states[L_star].norm()  # match magnitude
            out_noise = inject_and_generate(
                model, tokenizer, DESCRIBE_TEMPLATE["en"],
                noise, L_star, device,
            )
            print(f"  [NOISE→EN] {out_noise[:200]}", flush=True)
            prob_results["noise_inject"] = out_noise

            # Condition 4: No injection baseline
            prompt = build_chat_prompt(tokenizer, DESCRIBE_TEMPLATE["en"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_GEN, do_sample=False,
                )
            gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
            out_baseline = tokenizer.decode(gen_ids, skip_special_tokens=True)
            print(f"  [BASELINE] {out_baseline[:200]}", flush=True)
            prob_results["no_inject_baseline"] = out_baseline

            l_results[f"problem_{pi}"] = prob_results

        results[f"L{L_star}"] = l_results

    wall = time.time() - t0
    print(f"\n\nWall time: {wall:.0f}s ({wall/60:.1f}min)", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / "expINJ_hidden_injection.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "INJ_hidden_injection",
            "model": MODEL_NAME,
            "L_star_candidates": L_STAR_CANDIDATES,
            "problems": [p["en"] for p in INJECT_PROBLEMS],
            "results": results,
            "wall_time_s": wall,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
