"""Exp BS2: Generation-Only SVD Truncation.

BS showed 0/20 at ALL k when PCA truncation is active during both prefill
and generation. But the Gram analysis was done on PREFILL activations only.

This experiment separates the two:
- Prefill runs UNMODIFIED (full 2048-D, clean KV cache)
- PCA truncation activates ONLY during generation steps

If generation-only truncation preserves accuracy at k=rank_90:
  → Reasoning trajectory is low-rank; only encoding needs full dim
  → Implies AAAAABBBBB architecture (full encoder, compressed reasoner)

If it also fails:
  → Full dimensionality is load-bearing for per-token computation itself

Lean & fast: tqdm, flushed prints, cached PCA, minimal conditions.
"""

import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import numpy as np
import torch
import json
import time
import re
import math
import random as pyrandom
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force unbuffered output
print = lambda *a, **kw: (sys.stdout.write(" ".join(str(x) for x in a) + kw.get("end", "\n")), sys.stdout.flush())

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
CACHE_PATH = OUTPUT_DIR / "multilingual_all_layers.npz"
SEED = 42
N_LAYERS = 36
DIM = 2048
MAX_NEW_TOKENS = 128
LANGS_EVAL = ["en", "zh"]
LANGS_CACHE = ["ar", "en", "es", "ja", "ko", "sw", "zh"]
EQUILIBRIUM_LAYERS = list(range(9, 27))
N_TEST_PER_CAT = 4

TEMPLATES = {
    "zh": {
        "arithmetic_plus": "计算 {a} + {b} 的值。",
        "arithmetic_times": "计算 {a} × {b} 的值。",
        "combinatorics": "求组合数 C({n}, {k}) 的值。",
        "modular": "{a} 除以 {b} 的余数是多少？",
        "geometry": "一个长方形的长为 {w}，宽为 {h}，求其面积。",
    },
    "en": {
        "arithmetic_plus": "Calculate {a} + {b}.",
        "arithmetic_times": "Calculate {a} × {b}.",
        "combinatorics": "Find the value of C({n}, {k}).",
        "modular": "What is the remainder when {a} is divided by {b}?",
        "geometry": "A rectangle has length {w} and width {h}. Find its area.",
    },
}


def generate_test_problems(n_test=N_TEST_PER_CAT):
    rng = pyrandom.Random(SEED)
    cats = []
    per_cat = 200 // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        ans = a + b if op == "plus" else a * b
        prompts = {lang: TEMPLATES[lang][f"arithmetic_{op}"].format(a=a, b=b) for lang in ["en", "zh"]}
        cats.append(("arithmetic", ans, prompts))

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        ans = math.comb(n_val, k_val)
        prompts = {lang: TEMPLATES[lang]["combinatorics"].format(n=n_val, k=k_val) for lang in ["en", "zh"]}
        cats.append(("combinatorics", ans, prompts))

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        ans = a % b
        prompts = {lang: TEMPLATES[lang]["modular"].format(a=a, b=b) for lang in ["en", "zh"]}
        cats.append(("modular", ans, prompts))

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        ans = w * h
        prompts = {lang: TEMPLATES[lang]["geometry"].format(w=w, h=h) for lang in ["en", "zh"]}
        cats.append(("geometry", ans, prompts))

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        ans = n_terms * (2 * a1 + (n_terms - 1) * d) // 2
        cats.append(("sequences", ans, {
            "en": f"An arithmetic sequence: first term {a1}, common difference {d}. Sum of first {n_terms} terms?",
            "zh": f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
        }))

    rng2 = pyrandom.Random(SEED)
    indices = list(range(len(cats)))
    rng2.shuffle(indices)
    cats = [cats[i] for i in indices]

    by_cat = {}
    for cat, ans, prompts in cats:
        if cat not in by_cat:
            by_cat[cat] = []
        if len(by_cat[cat]) < n_test:
            by_cat[cat].append((ans, prompts))

    return [{"category": cat, "answer": ans, "en": p["en"], "zh": p["zh"]}
            for cat in by_cat for ans, p in by_cat[cat]]


def check_answer(text, correct_answer):
    return str(correct_answer) in re.findall(r"-?\d+\.?\d*", text)


def find_fat(text, correct_answer):
    target = str(correct_answer)
    for i, tok in enumerate(text.split()):
        if target in re.findall(r"-?\d+\.?\d*", tok):
            return i
    return -1


def detect_lang(text):
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    return "zh" if zh > en else ("en" if en > zh else "unk")


# ── PCA (cached on disk after first compute) ─────────────────────────────

PCA_CACHE = OUTPUT_DIR / "pca_centered_bases.npz"


def get_pca_bases(max_k=500):
    """Load or compute centered PCA bases. Caches to disk."""
    if PCA_CACHE.exists():
        print(f"[PCA] Loading cached bases from {PCA_CACHE}")
        d = np.load(PCA_CACHE)
        means = {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("mean_")}
        bases = {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("basis_")}
        var_fracs = {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("varfrac_")}
        return means, bases, var_fracs

    print(f"[PCA] Computing from {CACHE_PATH} (will cache to {PCA_CACHE})")
    data = np.load(CACHE_PATH)
    save_dict = {}
    means, bases, var_fracs = {}, {}, {}

    for layer in tqdm(range(N_LAYERS), desc="PCA"):
        H = np.vstack([data[f"{lang}_L{layer}"] for lang in LANGS_CACHE]).astype(np.float64)
        mu = H.mean(axis=0)
        _, S, Vt = np.linalg.svd(H - mu, full_matrices=False)
        m = mu.astype(np.float32)
        b = Vt[:max_k].T.astype(np.float32)
        eigs = S[:max_k] ** 2
        vf = (np.cumsum(eigs) / (S ** 2).sum()).astype(np.float32)
        means[layer], bases[layer], var_fracs[layer] = m, b, vf
        save_dict[f"mean_{layer}"] = m
        save_dict[f"basis_{layer}"] = b
        save_dict[f"varfrac_{layer}"] = vf

    np.savez_compressed(PCA_CACHE, **save_dict)
    print(f"[PCA] Cached to {PCA_CACHE}")
    return means, bases, var_fracs


# ── Hook with generation-only gating ─────────────────────────────────────

class GenOnlyPCAHook:
    """PCA truncation that only activates during generation (not prefill)."""

    def __init__(self, mean, basis_k, device, dtype):
        self.mean = torch.from_numpy(mean).to(device=device, dtype=dtype)
        self.basis_k = torch.from_numpy(basis_k).to(device=device, dtype=dtype)
        self.gen_mode = False  # flipped AFTER prefill completes

    def __call__(self, module, input, output):
        if not self.gen_mode:
            return output
        hidden = output  # (batch, seq=1, d) during generation
        h_c = hidden - self.mean
        h_proj = (h_c @ self.basis_k) @ self.basis_k.T + self.mean
        return h_proj


class AlwaysPCAHook:
    """PCA truncation active during both prefill and generation (BS control)."""

    def __init__(self, mean, basis_k, device, dtype):
        self.mean = torch.from_numpy(mean).to(device=device, dtype=dtype)
        self.basis_k = torch.from_numpy(basis_k).to(device=device, dtype=dtype)

    def __call__(self, module, input, output):
        hidden = output
        h_c = hidden - self.mean
        h_proj = (h_c @ self.basis_k) @ self.basis_k.T + self.mean
        return h_proj


# ── Custom generate with prefill/gen gating ──────────────────────────────

def generate_with_hooks(model, tokenizer, prompt, hooks, max_new=MAX_NEW_TOKENS):
    """Run generate, flipping hooks to gen_mode after prefill."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Register hooks
    handles = []
    for layer_idx, hook in hooks.items():
        h = model.model.layers[layer_idx].register_forward_hook(hook)
        handles.append(h)

    # Prefill: hooks are inactive (gen_mode=False) for GenOnly, always active for Always
    # Generation: hooks activate after first forward pass
    with torch.no_grad():
        # Prefill
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated = [next_token.item()]

        # Flip gen_mode ON for GenOnly hooks
        for hook in hooks.values():
            if hasattr(hook, "gen_mode"):
                hook.gen_mode = True

        # Generate token by token
        for _ in range(max_new - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
            tok_id = next_token.item()
            generated.append(tok_id)
            if tok_id == tokenizer.eos_token_id:
                break

        # Reset gen_mode
        for hook in hooks.values():
            if hasattr(hook, "gen_mode"):
                hook.gen_mode = False

    for h in handles:
        h.remove()

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # 1. PCA bases (cached)
    means, bases, var_fracs = get_pca_bases()

    # 2. Model
    print("[Model] Loading Qwen2.5-3B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    device = model.device
    dtype = torch.bfloat16
    print(f"[Model] Loaded in {time.time()-t0:.1f}s")

    # 3. Test problems
    test_problems = generate_test_problems()
    print(f"[Data] {len(test_problems)} test problems")

    # 4. Conditions — LEAN: only the ones that matter
    #    - baseline (no hooks)
    #    - gen_only at k=20, 50, 200, 500 on equilibrium layers
    #    - always (BS reproduction) at k=200 on equilibrium layers (sanity check)
    #    - gen_only at k=200 on single L16 (isolation control)
    #    - prefill_only at k=200 on equilibrium layers (inverse control)

    conditions = []

    # Baseline
    conditions.append(("baseline", {}, "none"))

    # Gen-only sweep
    for k in [20, 50, 200, 500]:
        hooks = {}
        for l in EQUILIBRIUM_LAYERS:
            hooks[l] = GenOnlyPCAHook(means[l], bases[l][:, :k], device, dtype)
        conditions.append((f"gen_k{k}", hooks, f"gen_only L9-L26 k={k}"))

    # Always-on at k=200 (BS reproduction / sanity)
    hooks_always = {}
    for l in EQUILIBRIUM_LAYERS:
        hooks_always[l] = AlwaysPCAHook(means[l], bases[l][:, :200], device, dtype)
    conditions.append(("always_k200", hooks_always, "always L9-L26 k=200"))

    # Gen-only single layer L16 at k=200
    hooks_single = {16: GenOnlyPCAHook(means[16], bases[16][:, :200], device, dtype)}
    conditions.append(("gen_L16_k200", hooks_single, "gen_only L16 k=200"))

    # Prefill-only at k=200 (inverse: corrupt KV cache but generation is clean)
    # We'll implement this with AlwaysOn hooks but flip them OFF during generation
    # Actually simpler: use GenOnly hooks in reverse — active during prefill, off during gen
    # Let's just add it as a manual condition
    conditions.append(("prefill_k200", "PREFILL_ONLY", "prefill_only L9-L26 k=200"))

    print(f"\n[Run] {len(conditions)} conditions × {len(test_problems)} problems × 2 langs")
    print(f"{'Condition':>20s}  {'Desc':>30s}")
    print("-" * 55)
    for name, _, desc in conditions:
        print(f"{name:>20s}  {desc:>30s}")
    print()

    # 5. Run
    all_results = {}

    for ci, (name, hooks_or_tag, desc) in enumerate(conditions):
        t_cond = time.time()
        results = {}

        for lang in LANGS_EVAL:
            correct = 0
            fats = []
            lang_pres = {"zh": 0, "en": 0, "unk": 0}

            pbar = tqdm(test_problems, desc=f"[{ci+1}/{len(conditions)}] {name} ({lang})",
                        leave=False, ncols=80)
            for prob in pbar:
                if hooks_or_tag == "PREFILL_ONLY":
                    # Prefill with hooks ON, generation with hooks OFF
                    pfill_hooks = {}
                    for l in EQUILIBRIUM_LAYERS:
                        pfill_hooks[l] = AlwaysPCAHook(means[l], bases[l][:, :200], device, dtype)
                    # Do manual prefill+gen
                    inputs = tokenizer(prob[lang], return_tensors="pt").to("cuda")
                    handles = []
                    for layer_idx, hook in pfill_hooks.items():
                        handles.append(model.model.layers[layer_idx].register_forward_hook(hook))
                    with torch.no_grad():
                        out = model(**inputs, use_cache=True)
                        past_kv = out.past_key_values
                        next_token = out.logits[:, -1:].argmax(dim=-1)
                    for h in handles:
                        h.remove()
                    # Generate WITHOUT hooks
                    generated = [next_token.item()]
                    with torch.no_grad():
                        for _ in range(MAX_NEW_TOKENS - 1):
                            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
                            past_kv = out.past_key_values
                            next_token = out.logits[:, -1:].argmax(dim=-1)
                            tok_id = next_token.item()
                            generated.append(tok_id)
                            if tok_id == tokenizer.eos_token_id:
                                break
                    gen = tokenizer.decode(generated, skip_special_tokens=True)

                elif isinstance(hooks_or_tag, dict) and len(hooks_or_tag) == 0:
                    # Baseline — use standard generate
                    inputs = tokenizer(prob[lang], return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                else:
                    # Hooked generation (gen-only or always)
                    gen = generate_with_hooks(model, tokenizer, prob[lang], hooks_or_tag)

                ok = check_answer(gen, prob["answer"])
                fat = find_fat(gen, prob["answer"])
                if ok:
                    correct += 1
                fats.append(fat)
                dl = detect_lang(gen)
                lang_pres[dl] += 1
                pbar.set_postfix(correct=correct)

            valid_fats = [f for f in fats if f >= 0]
            mfat = float(np.mean(valid_fats)) if valid_fats else -1
            results[lang] = {
                "correct": correct, "total": len(test_problems),
                "mean_fat": round(mfat, 1), "lang_pres": lang_pres,
            }

        all_results[name] = results
        dt = time.time() - t_cond
        en_c, zh_c = results["en"]["correct"], results["zh"]["correct"]
        en_lp, zh_lp = results["en"]["lang_pres"], results["zh"]["lang_pres"]
        print(f"  {name:>20s}  EN={en_c}/20  ZH={zh_c}/20  "
              f"lp_en={en_lp}  lp_zh={zh_lp}  ({dt:.1f}s)")

    # 6. Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Condition':>20s}  {'k':>4}  {'EN':>4}  {'ZH':>4}  {'Mode':>12}  {'ZH→lang':>10}")
    print("-" * 70)
    for name, _, desc in conditions:
        r = all_results[name]
        k_str = name.split("k")[-1] if "k" in name else "-"
        mode = "gen_only" if "gen_" in name else ("always" if "always" in name else ("prefill" if "prefill" in name else "none"))
        zh_dom = max(r["zh"]["lang_pres"], key=r["zh"]["lang_pres"].get)
        print(f"{name:>20s}  {k_str:>4}  {r['en']['correct']:>4}  {r['zh']['correct']:>4}  {mode:>12}  {zh_dom:>10}")

    # 7. Save
    output = {
        "experiment": "BS2",
        "title": "Generation-Only SVD Truncation",
        "hypothesis": "Prefill needs full dim for encoding; generation reasoning may be low-rank",
        "model": MODEL_NAME,
        "runtime_seconds": round(elapsed),
        "conditions": {name: {"desc": desc, "results": all_results[name]}
                       for name, _, desc in conditions},
    }

    out_path = OUTPUT_DIR / "expBS2_genonly_truncation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
