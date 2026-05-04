#!/usr/bin/env python3
"""
expGATE_expanded.py — Expanded Gate Check for Compression Procedure

Generates qualitatively new reasoning problems via GPT-4.1 proxy,
translates to 3 languages, extracts Qwen2.5-3B activations at all 36 layers,
and re-runs centroid SVD to test whether dim(Z_all) stays in 8-20 range.

New domains (each structurally distinct from existing math/diverse):
  - Code tracing: trace variable values through assignments/loops/conditionals
  - Spatial reasoning: compass/grid direction inference from relation chains
  - Counterfactual: causal chain reasoning ("if X hadn't happened...")
  - Multi-hop inference: chain facts to reach conclusions (self-contained)

Uses GPT-4.1 at localhost:3027 (copilot proxy, unlimited, free).
Runs on RayGun (Qwen2.5-3B, RTX 4070 Super).
"""

import json
import time
import asyncio
import numpy as np
from pathlib import Path
from tqdm import tqdm

import httpx

# ================================================================
# CONFIG
# ================================================================

PROXY_URL = "http://localhost:3027/v1/chat/completions"
MODEL = "gpt-4.1"
OUTPUT_DIR = Path("output")
CACHE_FILE = OUTPUT_DIR / "expanded_all_layers.npz"
PROBLEMS_FILE = OUTPUT_DIR / "expanded_problems.json"
RESULTS_FILE = OUTPUT_DIR / "expGATE_expanded_results.json"

LANGS = ["en", "zh", "es"]  # 3 languages: English, Chinese, Spanish
N_PER_DOMAIN = 50
DOMAINS = ["code_tracing", "spatial", "counterfactual", "multihop"]
N_TOTAL = N_PER_DOMAIN * len(DOMAINS)  # 200

# Rate limiting (generous — proxy handles 1000 RPM)
MAX_CONCURRENT = 20
TIMEOUT = 60.0

# ================================================================
# PROBLEM GENERATION PROMPTS
# ================================================================

GENERATION_PROMPTS = {
    "code_tracing": """Generate {n} code tracing problems. Each problem gives a short Python snippet (3-8 lines) and asks "What is the value of [variable] after this code runs?"

Requirements:
- Each problem must have a UNIQUE structure (different operations: arithmetic, string ops, list ops, conditionals, simple loops, swaps, accumulators, modular arithmetic, bitwise, nested assignments)
- Answers must be deterministic and unambiguous (no randomness, no input())
- Difficulty: require 3-8 mental steps to trace
- NO imports, NO functions defined, just straight-line or simple control flow
- Mix data types: int, str, list, bool

Return a JSON array of objects with fields:
- "problem": the full problem text (include the code and the question)
- "answer": the correct answer (as a string)
- "structure": one-word label for the operation type (e.g., "accumulator", "conditional", "swap", "listop", "bitwise", "modular", "string", "nested", "loop_counter", "boolean")

Return ONLY the JSON array, no explanation.""",

    "spatial": """Generate {n} spatial reasoning problems. Each gives a chain of directional relationships and asks for a derived direction.

Requirements:
- Use compass directions (north, south, east, west, northeast, etc.)
- Chain length: 2-5 relationships per problem
- Mix 2D grid reasoning with compass reasoning
- Use diverse entities (people, buildings, cities, objects — not always the same)
- Each problem must be solvable by chaining the given relations
- Vary question types: "What direction is X from Y?", "If you walk from X to Y, which direction?", "Who is furthest north?"
- UNIQUE structures: linear chains, branching, triangular, with/without distractors

Return a JSON array of objects with fields:
- "problem": the full problem text
- "answer": the correct direction/position answer
- "structure": label for chain type (e.g., "linear_2", "linear_3", "branch", "triangle", "4chain", "diagonal", "relative", "superlative")

Return ONLY the JSON array, no explanation.""",

    "counterfactual": """Generate {n} counterfactual reasoning problems. Each presents a causal scenario and asks "If X had not happened, would Y still be true?"

Requirements:
- Self-contained scenarios (no external knowledge needed)
- Clear causal chains with 2-4 links
- Mix of answers: some yes, some no, some "cannot determine"
- Scenarios from diverse domains: cooking, travel, sports, weather, construction, social, mechanical, financial, biological, scheduling
- Each must have a UNIQUE causal structure (don't reuse the same pattern)
- Some with multiple causal paths (so removing one cause doesn't prevent the effect)
- Some with single necessary causes (removing it DOES prevent the effect)

Return a JSON array of objects with fields:
- "problem": the full scenario + question
- "answer": "yes", "no", or "cannot determine"
- "structure": label (e.g., "single_cause", "redundant_paths", "chain_3", "fork", "common_cause", "mediator", "preventive", "enabling")

Return ONLY the JSON array, no explanation.""",

    "multihop": """Generate {n} multi-hop inference problems. Each provides 3-6 factual statements and asks a question that requires chaining 2-4 of them.

Requirements:
- ALL facts needed must be stated explicitly (no world knowledge required)
- Facts should form a chain: A→B, B→C, question asks about A→C
- Mix types: transitive properties, set membership chains, numerical inheritance, temporal ordering, spatial chaining
- Include 1-2 distractor facts per problem that are irrelevant
- Answers should be short (1-3 words or a number)
- UNIQUE structures: don't reuse the same chain pattern
- Diverse topics: animals, organizations, geography (fictional), games, recipes, genealogy, classification

Return a JSON array of objects with fields:
- "problem": all the facts + the question
- "answer": the correct short answer
- "structure": label (e.g., "transitive_2", "transitive_3", "membership", "numerical", "temporal", "spatial_chain", "genealogy", "classification")

Return ONLY the JSON array, no explanation.""",
}

TRANSLATION_PROMPT = """Translate the following {n} reasoning problems from English to {target_lang}.
Maintain the exact same logical structure and answer. For code tracing problems, keep the Python code in English (variable names, keywords) but translate the surrounding question text.

Return a JSON array with the same structure, where "problem" field contains the translated text and "answer" stays the same (translate if it's a word/phrase, keep as-is if it's a number/code value).

Problems to translate:
{problems_json}

Return ONLY the JSON array, no explanation."""


# ================================================================
# ASYNC GPT-4.1 CALLS
# ================================================================

async def call_gpt(client: httpx.AsyncClient, messages: list, temperature: float = 0.7) -> str:
    """Single GPT-4.1 call through proxy."""
    resp = await client.post(PROXY_URL, json={
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
    }, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def generate_domain_problems(client: httpx.AsyncClient, domain: str, n: int) -> list:
    """Generate problems for one domain, splitting into batches if needed."""
    batch_size = 25  # GPT-4.1 handles 25 well in one shot
    all_problems = []

    for batch_start in range(0, n, batch_size):
        batch_n = min(batch_size, n - batch_start)
        prompt = GENERATION_PROMPTS[domain].format(n=batch_n)

        messages = [
            {"role": "system", "content": "You are a problem generator for machine learning research. Generate diverse, structurally unique problems. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(3):
            try:
                raw = await call_gpt(client, messages)
                # Strip markdown fences if present
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    elif "```" in raw:
                        raw = raw[:raw.rfind("```")]

                problems = json.loads(raw)
                if isinstance(problems, list) and len(problems) >= batch_n - 2:
                    for p in problems:
                        p["domain"] = domain
                    all_problems.extend(problems)
                    print(f"  {domain} batch {batch_start//batch_size + 1}: got {len(problems)} problems")
                    break
            except (json.JSONDecodeError, httpx.HTTPError, KeyError) as e:
                print(f"  {domain} batch {batch_start//batch_size + 1} attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    print(f"  WARNING: skipping batch after 3 failures")

    return all_problems


async def translate_problems(client: httpx.AsyncClient, problems: list, target_lang: str, lang_name: str) -> list:
    """Translate a batch of problems to target language."""
    batch_size = 20  # Smaller batches for translation (longer output)
    translated = []

    for batch_start in range(0, len(problems), batch_size):
        batch = problems[batch_start:batch_start + batch_size]
        # Only send problem + answer for translation (strip metadata)
        to_translate = [{"problem": p["problem"], "answer": p["answer"]} for p in batch]

        prompt = TRANSLATION_PROMPT.format(
            n=len(batch),
            target_lang=lang_name,
            problems_json=json.dumps(to_translate, ensure_ascii=False)
        )

        messages = [
            {"role": "system", "content": f"You are a professional translator. Translate to {lang_name}. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(3):
            try:
                raw = await call_gpt(client, messages, temperature=0.3)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1]
                    if "```" in raw:
                        raw = raw[:raw.rfind("```")]

                results = json.loads(raw)
                if isinstance(results, list) and len(results) >= len(batch) - 2:
                    translated.extend(results)
                    break
            except (json.JSONDecodeError, httpx.HTTPError, KeyError) as e:
                print(f"  Translation {target_lang} batch {batch_start//batch_size + 1} attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    # Fall back: use English
                    translated.extend(to_translate)
                    print(f"  WARNING: using English fallback for this batch")

    return translated


async def generate_all_problems():
    """Generate all problems + translations via proxy."""
    print("=" * 60)
    print("PHASE 1: Problem Generation via GPT-4.1")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Generate all domains in parallel
        print("\nGenerating problems (4 domains × 50)...")
        tasks = [generate_domain_problems(client, domain, N_PER_DOMAIN) for domain in DOMAINS]
        domain_results = await asyncio.gather(*tasks)

        all_problems = []
        for domain, problems in zip(DOMAINS, domain_results):
            print(f"  {domain}: {len(problems)} problems generated")
            all_problems.extend(problems)

        print(f"\nTotal English problems: {len(all_problems)}")

        # Translate to zh and es
        print("\n" + "=" * 60)
        print("PHASE 2: Translation")
        print("=" * 60)

        translations = {"en": all_problems}

        for lang_code, lang_name in [("zh", "Chinese (Simplified)"), ("es", "Spanish")]:
            print(f"\nTranslating to {lang_name}...")
            translated = await translate_problems(client, all_problems, lang_code, lang_name)
            translations[lang_code] = translated
            print(f"  Got {len(translated)} translations")

    # Assemble into pipeline format: list of dicts with {en, zh, es} keys
    problems_multilingual = []
    for i in range(len(all_problems)):
        entry = {
            "en": all_problems[i]["problem"],
            "domain": all_problems[i]["domain"],
            "structure": all_problems[i].get("structure", "unknown"),
            "answer": all_problems[i]["answer"],
        }
        if i < len(translations["zh"]):
            entry["zh"] = translations["zh"][i]["problem"]
        else:
            entry["zh"] = entry["en"]  # fallback
        if i < len(translations["es"]):
            entry["es"] = translations["es"][i]["problem"]
        else:
            entry["es"] = entry["en"]
        problems_multilingual.append(entry)

    return problems_multilingual


# ================================================================
# ACTIVATION EXTRACTION (same pattern as expBR)
# ================================================================

def extract_activations(model, tokenizer, problems, n_layers, d, langs):
    """Extract last-token hidden states for all langs × all layers."""
    import torch

    N = len(problems)
    all_acts = {lang: {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
                for lang in langs}

    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            layer_outputs[layer_idx] = h_out.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l))
               for l in range(n_layers)]

    try:
        for lang in langs:
            print(f"  Extracting {lang} ({N} problems)...")
            for i, prob in enumerate(tqdm(problems, desc=lang, leave=False)):
                text = prob[lang]
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**inputs)
                for l in range(n_layers):
                    all_acts[lang][l][i] = layer_outputs[l]
                layer_outputs.clear()
    finally:
        for h in handles:
            h.remove()

    return all_acts


# ================================================================
# CENTROID SVD GATE CHECK
# ================================================================

def run_gate_check(expanded_acts, existing_math_path, existing_diverse_path, langs, n_layers, d):
    """Run centroid SVD gate check on combined data."""
    print("\n" + "=" * 60)
    print("PHASE 4: Centroid SVD Gate Check")
    print("=" * 60)

    # Load existing caches (they use 7 languages; we'll use the 3 we share)
    shared_langs = [l for l in langs if l in ["en", "zh", "es"]]

    math_data = np.load(existing_math_path)
    diverse_data = np.load(existing_diverse_path)

    results = {"layers": {}, "summary": {}}

    for L in range(n_layers):
        # Compute centroids for each dataset
        # Math: average across shared langs
        math_centroids = np.mean(
            [math_data[f"{lang}_L{L}"] for lang in shared_langs], axis=0
        )  # (200, d)

        # Diverse: average across shared langs
        diverse_centroids = np.mean(
            [diverse_data[f"{lang}_L{L}"] for lang in shared_langs], axis=0
        )  # (200, d)

        # Expanded: average across our 3 langs
        expanded_centroids = np.mean(
            [expanded_acts[lang][L] for lang in langs], axis=0
        )  # (N_expanded, d)

        # Stack all centroids
        all_centroids = np.vstack([math_centroids, diverse_centroids, expanded_centroids])

        # Center
        all_centroids -= all_centroids.mean(axis=0, keepdims=True)

        # SVD
        U, S, Vt = np.linalg.svd(all_centroids, full_matrices=False)

        # Effective ranks
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        r50 = int(np.searchsorted(cumvar, 0.50) + 1)
        r90 = int(np.searchsorted(cumvar, 0.90) + 1)
        r95 = int(np.searchsorted(cumvar, 0.95) + 1)
        r99 = int(np.searchsorted(cumvar, 0.99) + 1)
        top1_frac = float(S[0]**2 / np.sum(S**2))

        results["layers"][L] = {
            "r50": r50, "r90": r90, "r95": r95, "r99": r99,
            "top1_frac": round(top1_frac, 4),
            "n_problems": all_centroids.shape[0]
        }

        if L % 5 == 0 or L >= 30:
            print(f"  L{L:2d}: r50={r50}, r90={r90:2d}, r95={r95:2d}, r99={r99:3d}, top1={top1_frac:.1%}, N={all_centroids.shape[0]}")

    # Also run expanded-only to see its intrinsic dimensionality
    print("\n  --- Expanded-only (new domains) ---")
    for L in [0, 9, 17, 26, 33, 35]:
        expanded_centroids = np.mean(
            [expanded_acts[lang][L] for lang in langs], axis=0
        )
        expanded_centroids -= expanded_centroids.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(expanded_centroids, full_matrices=False)
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        r90 = int(np.searchsorted(cumvar, 0.90) + 1)
        r95 = int(np.searchsorted(cumvar, 0.95) + 1)
        print(f"  L{L:2d} expanded-only: r90={r90}, r95={r95}")

    # Summary
    l26 = results["layers"][26]
    results["summary"] = {
        "dim_Z_at_L26_90pct": l26["r90"],
        "dim_Z_at_L26_95pct": l26["r95"],
        "dim_Z_at_L26_99pct": l26["r99"],
        "n_total_problems": l26["n_problems"],
        "n_expanded_problems": N_TOTAL,
        "langs": langs,
        "verdict": "PASS" if l26["r90"] <= 25 else "NEEDS_REVISION"
    }

    print(f"\n  {'='*40}")
    print(f"  GATE CHECK VERDICT: {results['summary']['verdict']}")
    print(f"  dim(Z_all) at L26 (90%): {l26['r90']}")
    print(f"  Total problems in SVD: {l26['n_problems']}")
    print(f"  {'='*40}")

    return results


# ================================================================
# MAIN
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true", help="Only generate problems, skip extraction")
    parser.add_argument("--extract-only", action="store_true", help="Only extract activations from existing problems")
    parser.add_argument("--analyze-only", action="store_true", help="Only run SVD from existing cache")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Phase 1+2: Generate and translate problems ──
    if not args.extract_only and not args.analyze_only:
        problems = asyncio.run(generate_all_problems())

        # Save
        with open(PROBLEMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(problems)} problems to {PROBLEMS_FILE}")

        # Stats
        from collections import Counter
        domain_counts = Counter(p["domain"] for p in problems)
        struct_counts = Counter(p["structure"] for p in problems)
        print(f"Domain distribution: {dict(domain_counts)}")
        print(f"Unique structures: {len(struct_counts)}")

        if args.generate_only:
            return
    else:
        # Load existing
        with open(PROBLEMS_FILE, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        print(f"Loaded {len(problems)} problems from {PROBLEMS_FILE}")

    # ── Phase 3: Extract activations ──
    if not args.analyze_only:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\n" + "=" * 60)
        print("PHASE 3: Activation Extraction (Qwen2.5-3B)")
        print("=" * 60)

        model_name = "Qwen/Qwen2.5-3B"
        print(f"\nLoading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda",
            trust_remote_code=True
        )
        model.eval()

        n_layers = model.config.num_hidden_layers  # 36
        d = model.config.hidden_size  # 2048
        print(f"Model: {n_layers} layers, d={d}")

        acts = extract_activations(model, tokenizer, problems, n_layers, d, LANGS)

        # Save cache
        print("\nSaving activation cache...")
        save_dict = {"categories": np.array([DOMAINS.index(p["domain"]) for p in problems])}
        for lang in LANGS:
            for l in range(n_layers):
                save_dict[f"{lang}_L{l}"] = acts[lang][l]
        np.savez_compressed(CACHE_FILE, **save_dict)
        print(f"Saved to {CACHE_FILE}")

        del model
        torch.cuda.empty_cache()
    else:
        # Load existing cache
        print(f"\nLoading cached activations from {CACHE_FILE}...")
        cache = np.load(CACHE_FILE)
        n_layers = 36
        d = 2048
        acts = {lang: {l: cache[f"{lang}_L{l}"] for l in range(n_layers)} for lang in LANGS}

    # ── Phase 4: Gate check ──
    results = run_gate_check(
        acts,
        OUTPUT_DIR / "multilingual_all_layers.npz",
        OUTPUT_DIR / "diverse_all_layers.npz",
        LANGS, n_layers, d
    )

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
