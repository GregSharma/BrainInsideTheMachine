#!/usr/bin/env python3
"""
expGATE_saturation.py — Saturation test: do more diverse domains increase dim(Z)?

Generates 200 MORE problems in 4 NEW domains (distinct from both original and expanded):
  - Formal logic: syllogisms with quantifiers, modus ponens/tollens, contrapositive
  - Temporal scheduling: event ordering, duration arithmetic, overlap detection
  - Visual/scene description: spatial layout of objects, counting, relative positions
  - Riddles/lateral: word puzzles, number riddles, pattern completion

Then re-runs centroid SVD with ALL data (math + diverse + expanded + saturation = ~800 problems).
If dim(Z) stays ~18-20: saturated. If climbing: need more.
"""

import json
import asyncio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import httpx

PROXY_URL = "http://localhost:3027/v1/chat/completions"
MODEL = "gpt-4.1"
OUTPUT_DIR = Path("output")
CACHE_FILE = OUTPUT_DIR / "saturation_all_layers.npz"
PROBLEMS_FILE = OUTPUT_DIR / "saturation_problems.json"
RESULTS_FILE = OUTPUT_DIR / "expGATE_saturation_results.json"

LANGS = ["en", "zh", "es"]
N_PER_DOMAIN = 50
DOMAINS = ["formal_logic", "temporal", "scene_description", "riddles"]
TIMEOUT = 60.0

GENERATION_PROMPTS = {
    "formal_logic": """Generate {n} formal logic problems. Each presents 2-4 premises and asks for a conclusion or validity judgment.

Requirements:
- Mix: modus ponens, modus tollens, contrapositive, disjunctive syllogism, hypothetical syllogism, reductio
- Use diverse content domains (not just "Socrates is mortal")
- Each premise set must be self-contained
- Answers: the valid conclusion, or "valid"/"invalid"
- UNIQUE logical structures — don't repeat the same inference pattern
- Some should have negations, some universal/existential quantifiers

Return a JSON array of objects with: "problem", "answer", "structure"
Structure labels: "modus_ponens", "modus_tollens", "contrapositive", "disjunctive", "hypothetical_chain", "reductio", "universal", "existential", "biconditional", "demorgan"

Return ONLY the JSON array, no explanation.""",

    "temporal": """Generate {n} temporal scheduling/ordering problems. Each involves time relationships and asks about order, duration, or overlap.

Requirements:
- Mix types: event ordering (A before B, B before C, when is D?), duration arithmetic (started at X, took Y hours), overlap detection (do meetings conflict?), deadline reasoning
- Use concrete times (9:00 AM, 2:30 PM) and durations (45 minutes, 2 hours)
- Each problem self-contained with 3-5 temporal facts
- Answers: a time, "yes"/"no" for overlap, or an ordering
- UNIQUE temporal structures
- Include some with time zones, some with relative times ("30 minutes after")

Return a JSON array of objects with: "problem", "answer", "structure"
Structure labels: "ordering_3", "ordering_4", "duration_add", "duration_subtract", "overlap_detect", "deadline", "timezone", "relative_chain", "gap_compute", "earliest_latest"

Return ONLY the JSON array, no explanation.""",

    "scene_description": """Generate {n} scene/layout description problems. Each describes the spatial arrangement of objects and asks a spatial question.

Requirements:
- Describe a scene with 4-8 objects in spatial relationships (on top of, left of, behind, inside, between)
- Questions: "What is to the left of X?", "How many objects are on the table?", "Which object is highest?", "What is between A and B?"
- Mix indoor and outdoor scenes
- Include some with vertical stacking, some with grid-like arrangements
- Answers: object name(s), count, or position
- UNIQUE spatial configurations

Return a JSON array of objects with: "problem", "answer", "structure"
Structure labels: "horizontal", "vertical", "grid", "containment", "between", "counting", "relative_height", "layered", "circular", "mixed_3d"

Return ONLY the JSON array, no explanation.""",

    "riddles": """Generate {n} riddle/puzzle problems. Each is a self-contained puzzle with a definite answer.

Requirements:
- Mix: number riddles ("I'm thinking of a number..."), word riddles, sequence completion, logic puzzles (liars/truth-tellers), age problems, coin/balance puzzles, river crossing variants
- Each must be solvable from the given information alone (no trivia)
- Difficulty: require 2-5 reasoning steps
- Answers: a number, word, or short phrase
- UNIQUE puzzle structures — don't reuse the same template

Return a JSON array of objects with: "problem", "answer", "structure"
Structure labels: "number_riddle", "word_riddle", "sequence", "liar_truth", "age", "balance", "river_crossing", "knights_knaves", "digit_logic", "constraint_sat"

Return ONLY the JSON array, no explanation.""",
}

TRANSLATION_PROMPT = """Translate the following {n} reasoning problems from English to {target_lang}.
Maintain the exact same logical structure and answer. Keep any numbers, times, or technical terms accurate.

Return a JSON array with the same structure, where "problem" field contains the translated text and "answer" stays the same (translate if it's a word/phrase, keep as-is if it's a number/code value).

Problems:
{problems_json}

Return ONLY the JSON array."""


async def call_gpt(client, messages, temperature=0.7):
    resp = await client.post(PROXY_URL, json={
        "model": MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": 8192,
    }, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if "```" in raw:
            raw = raw[:raw.rfind("```")]
    return json.loads(raw)


async def generate_domain(client, domain, n):
    batch_size = 25
    all_problems = []
    for batch_start in range(0, n, batch_size):
        batch_n = min(batch_size, n - batch_start)
        prompt = GENERATION_PROMPTS[domain].format(n=batch_n)
        messages = [
            {"role": "system", "content": "Generate diverse, structurally unique reasoning problems. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        for attempt in range(3):
            try:
                raw = await call_gpt(client, messages)
                problems = parse_json_response(raw)
                if isinstance(problems, list) and len(problems) >= batch_n - 2:
                    for p in problems:
                        p["domain"] = domain
                    all_problems.extend(problems)
                    print(f"  {domain} batch {batch_start//batch_size+1}: {len(problems)} problems")
                    break
            except Exception as e:
                print(f"  {domain} batch {batch_start//batch_size+1} attempt {attempt+1}: {e}")
                if attempt == 2:
                    print(f"  WARNING: skipping batch")
    return all_problems


async def translate_batch(client, problems, lang_code, lang_name):
    batch_size = 20
    translated = []
    for i in range(0, len(problems), batch_size):
        batch = [{"problem": p["problem"], "answer": p["answer"]} for p in problems[i:i+batch_size]]
        prompt = TRANSLATION_PROMPT.format(n=len(batch), target_lang=lang_name,
                                            problems_json=json.dumps(batch, ensure_ascii=False))
        messages = [
            {"role": "system", "content": f"Translate to {lang_name}. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        for attempt in range(3):
            try:
                raw = await call_gpt(client, messages, temperature=0.3)
                results = parse_json_response(raw)
                if isinstance(results, list) and len(results) >= len(batch) - 2:
                    translated.extend(results)
                    break
            except Exception as e:
                if attempt == 2:
                    translated.extend(batch)  # fallback
    return translated


async def generate_all():
    print("=" * 60)
    print("SATURATION TEST: Problem Generation")
    print("=" * 60)
    async with httpx.AsyncClient() as client:
        tasks = [generate_domain(client, d, N_PER_DOMAIN) for d in DOMAINS]
        results = await asyncio.gather(*tasks)
        all_probs = []
        for domain, probs in zip(DOMAINS, results):
            print(f"  {domain}: {len(probs)} total")
            all_probs.extend(probs)
        print(f"\nTotal: {len(all_probs)}")

        translations = {"en": all_probs}
        for code, name in [("zh", "Chinese (Simplified)"), ("es", "Spanish")]:
            print(f"\nTranslating to {name}...")
            translations[code] = await translate_batch(client, all_probs, code, name)
            print(f"  Got {len(translations[code])}")

    problems = []
    for i in range(len(all_probs)):
        entry = {
            "en": all_probs[i]["problem"],
            "domain": all_probs[i]["domain"],
            "structure": all_probs[i].get("structure", "unknown"),
            "answer": all_probs[i]["answer"],
        }
        entry["zh"] = translations["zh"][i]["problem"] if i < len(translations["zh"]) else entry["en"]
        entry["es"] = translations["es"][i]["problem"] if i < len(translations["es"]) else entry["en"]
        problems.append(entry)
    return problems


def extract_activations(model, tokenizer, problems, n_layers, d, langs):
    import torch
    N = len(problems)
    all_acts = {lang: {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)} for lang in langs}
    layer_outputs = {}

    def make_hook(idx):
        def hook(module, input, output):
            h = output if isinstance(output, torch.Tensor) else output[0]
            layer_outputs[idx] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]
    try:
        for lang in langs:
            print(f"  Extracting {lang} ({N} problems)...")
            for i, prob in enumerate(tqdm(problems, desc=lang, leave=False)):
                inputs = tokenizer(prob[lang], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**inputs)
                for l in range(n_layers):
                    all_acts[lang][l][i] = layer_outputs[l]
                layer_outputs.clear()
    finally:
        for h in handles:
            h.remove()
    return all_acts


def run_saturation_check(sat_acts, n_layers, d):
    """Run centroid SVD with ALL four datasets combined."""
    langs = LANGS

    math_data = np.load(OUTPUT_DIR / "multilingual_all_layers.npz")
    diverse_data = np.load(OUTPUT_DIR / "diverse_all_layers.npz")
    expanded_data = np.load(OUTPUT_DIR / "expanded_all_layers.npz")

    results = {"layers": {}}

    # Track dim at L26 across cumulative additions
    cumulative = {}

    for L in range(n_layers):
        math_cent = np.mean([math_data[f'{lang}_L{L}'] for lang in langs], axis=0)
        div_cent = np.mean([diverse_data[f'{lang}_L{L}'] for lang in langs], axis=0)
        exp_cent = np.mean([expanded_data[f'{lang}_L{L}'] for lang in langs], axis=0)
        sat_cent = np.mean([sat_acts[lang][L] for lang in langs], axis=0)

        all_cent = np.vstack([math_cent, div_cent, exp_cent, sat_cent])
        all_cent -= all_cent.mean(axis=0, keepdims=True)

        U, S, Vt = np.linalg.svd(all_cent, full_matrices=False)
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        r50 = int(np.searchsorted(cumvar, 0.50) + 1)
        r90 = int(np.searchsorted(cumvar, 0.90) + 1)
        r95 = int(np.searchsorted(cumvar, 0.95) + 1)
        r99 = int(np.searchsorted(cumvar, 0.99) + 1)
        top1 = float(S[0]**2 / np.sum(S**2))

        results["layers"][L] = {
            "r50": r50, "r90": r90, "r95": r95, "r99": r99,
            "top1_frac": round(top1, 4), "n_problems": all_cent.shape[0]
        }

        if L % 5 == 0 or L >= 30:
            print(f"  L{L:2d}: r50={r50}, r90={r90:2d}, r95={r95:2d}, r99={r99:3d}, top1={top1:.1%}, N={all_cent.shape[0]}")

    # Cumulative saturation curve at L26
    print("\n  --- Saturation curve at L26 ---")
    for label, datasets in [
        ("math only", [math_data]),
        ("+ diverse", [math_data, diverse_data]),
        ("+ expanded", [math_data, diverse_data, expanded_data]),
        ("+ saturation (ALL)", [math_data, diverse_data, expanded_data, None]),
    ]:
        centroids = []
        for ds in datasets:
            if ds is None:
                centroids.append(np.mean([sat_acts[lang][26] for lang in langs], axis=0))
            elif isinstance(ds, dict):
                centroids.append(np.mean([ds[lang][26] for lang in langs], axis=0))
            else:
                centroids.append(np.mean([ds[f'{lang}_L26'] for lang in langs], axis=0))
        all_c = np.vstack(centroids)
        all_c -= all_c.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(all_c, full_matrices=False)
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        r90 = int(np.searchsorted(cumvar, 0.90) + 1)
        r95 = int(np.searchsorted(cumvar, 0.95) + 1)
        n = all_c.shape[0]
        print(f"  {label:25s}: N={n:4d}, r90={r90:2d}, r95={r95:2d}")

    l26 = results["layers"][26]
    results["summary"] = {
        "dim_Z_at_L26_90pct": l26["r90"],
        "total_problems": l26["n_problems"],
        "total_domains": 10,  # 5 original + 4 expanded + 4 saturation (some overlap in diverse)
        "verdict": "SATURATED" if l26["r90"] <= 22 else "STILL_CLIMBING"
    }

    print(f"\n  {'='*40}")
    print(f"  SATURATION VERDICT: {results['summary']['verdict']}")
    print(f"  dim(Z_all) at L26 (90%): {l26['r90']}")
    print(f"  Total problems: {l26['n_problems']}")
    print(f"  {'='*40}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if not args.extract_only and not args.analyze_only:
        problems = asyncio.run(generate_all())
        with open(PROBLEMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(problems)} problems to {PROBLEMS_FILE}")
        from collections import Counter
        print(f"Domains: {dict(Counter(p['domain'] for p in problems))}")
        print(f"Structures: {len(set(p['structure'] for p in problems))}")
        if args.generate_only:
            return
    else:
        with open(PROBLEMS_FILE, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        print(f"Loaded {len(problems)} problems")

    if not args.analyze_only:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\nLoading Qwen2.5-3B...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B", dtype=torch.float16, device_map="cuda",
            trust_remote_code=True
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        d = model.config.hidden_size

        acts = extract_activations(model, tokenizer, problems, n_layers, d, LANGS)

        save_dict = {"categories": np.array([DOMAINS.index(p["domain"]) for p in problems])}
        for lang in LANGS:
            for l in range(n_layers):
                save_dict[f"{lang}_L{l}"] = acts[lang][l]
        np.savez_compressed(CACHE_FILE, **save_dict)
        print(f"Saved to {CACHE_FILE}")
        del model
        torch.cuda.empty_cache()
    else:
        cache = np.load(CACHE_FILE)
        n_layers, d = 36, 2048
        acts = {lang: {l: cache[f"{lang}_L{l}"] for l in range(n_layers)} for lang in LANGS}

    results = run_saturation_check(acts, n_layers, d)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
