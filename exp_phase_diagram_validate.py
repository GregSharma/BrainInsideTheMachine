"""Sequential validation of phase diagram.

Re-runs ALL conditions one at a time using the proven single-element
WindowedDeflation. Compares with batched results to detect artifacts.
"""
import json, time, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 1200

DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
DEFLATE_REFRESH = 25

SYS = ("You are solving an AMC 12A multiple choice math problem. "
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
CORRECT = "B"
PROMPT = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"

DELTA_VALUES = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
ONSET_VALUES = [0, 5, 10, 15, 20, 22, 25, 27, 30, 35, 40, 50]


# Import the proven single-element WindowedDeflation
from exp_delayed_deflation_p12 import WindowedDeflation


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


def classify(text):
    ans = extract_answer(text)
    n = len(text.split())
    if n > MAX_TOKENS * 0.7 and ans == "?":
        return "LOOP", "?"
    if ans == CORRECT:
        return "CORRECT", ans
    return f"WRONG({ans})", ans


def generate(model, tokenizer, input_ids, deflator=None):
    """Single-element manual generation loop."""
    gen_ids = []
    past_kv = None
    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                if deflator:
                    deflator.start_gen()
                    deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)
            if deflator:
                deflator.tick(past_kv)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    return text, len(gen_ids)


def main():
    print("=" * 70)
    print("PHASE DIAGRAM VALIDATION -- SEQUENTIAL (no batching)")
    total = len(DELTA_VALUES) * len(ONSET_VALUES)
    print(f"Grid: {len(DELTA_VALUES)} x {len(ONSET_VALUES)} = {total} conditions")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    print(f"Loaded {MODEL_NAME}, {len(model.model.layers)} layers.\n", flush=True)

    # Baseline
    print("Baseline...")
    t0 = time.time()
    text, ntok = generate(model, tokenizer, input_ids)
    bl_label, bl_ans = classify(text)
    print(f"  -> {bl_label} ({ntok} tokens, {time.time()-t0:.1f}s)\n", flush=True)

    all_results = []
    t_start = time.time()
    count = 0

    for delta in DELTA_VALUES:
        row_results = []
        for onset in ONSET_VALUES:
            count += 1
            defl = WindowedDeflation(
                model, DEFLATE_LAYERS, r=DEFLATE_R, alpha=delta,
                refresh_every=DEFLATE_REFRESH,
                active_from=onset, active_until=None)

            t0 = time.time()
            text, ntok = generate(model, tokenizer, input_ids, deflator=defl)
            dt = time.time() - t0
            defl.remove()

            label, ans = classify(text)
            correct = (ans == CORRECT)
            r = {"delta": delta, "onset": onset, "label": label,
                 "answer": ans, "correct": correct,
                 "n_tokens": ntok, "time_s": round(dt, 1)}
            all_results.append(r)
            row_results.append(r)

            elapsed = time.time() - t_start
            eta = elapsed / count * (total - count)
            sym = "Y" if correct else "N"
            print(f"  [{count:3d}/{total}] d={delta:.2f} onset={onset:3d} "
                  f"-> {label:12s} {sym}  ({ntok:4d}tok, {dt:.0f}s, ETA {eta:.0f}s)", flush=True)

        n_correct = sum(1 for x in row_results if x["correct"])
        print(f"  --- d={delta:.2f}: {n_correct}/{len(ONSET_VALUES)} correct ---\n", flush=True)

    elapsed = time.time() - t_start

    # Grid
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL RESULTS ({elapsed:.0f}s total)")
    print(f"{'='*70}")
    print(f"\n{'d':>6} |" + "".join(f" {o:>3}" for o in ONSET_VALUES))
    print(f"{'':->6}-+" + "-" * (4 * len(ONSET_VALUES)))

    t_crit = {}
    for delta in DELTA_VALUES:
        row = sorted([r for r in all_results if r["delta"] == delta], key=lambda x: x["onset"])
        line = f"{delta:6.2f} |"
        for r in row:
            line += " Y  " if r["correct"] else " N  "
        print(line)
        correct_onsets = [r["onset"] for r in row if r["correct"]]
        if correct_onsets:
            t_crit[delta] = max(correct_onsets)

    print(f"\nCritical onset:")
    print(f"{'d':>6} {'t_crit':>8} {'d*t_crit':>10}")
    print(f"{'':->6} {'':->8} {'':->10}")
    products = []
    for d in sorted(t_crit.keys()):
        tc = t_crit[d]
        prod = d * tc
        products.append(prod)
        print(f"{d:6.2f} {tc:8d} {prod:10.2f}")

    if len(products) >= 3:
        mean_p = sum(products) / len(products)
        std_p = (sum((p - mean_p)**2 for p in products) / len(products)) ** 0.5
        cv = std_p / mean_p if mean_p > 0 else float('inf')
        print(f"\nd*t_crit: mean={mean_p:.2f}, std={std_p:.2f}, CV={cv:.3f}")
        if cv < 0.3:
            print(f"-> d*t_crit ~ CONSTANT ({mean_p:.1f})")
            print(f"-> TWO-SOURCE ECHO MODEL CONFIRMED")
        else:
            print(f"-> d*t_crit NOT constant (CV={cv:.2f})")

    # Compare with batched
    print(f"\n{'='*70}")
    print("BATCHED vs SEQUENTIAL:")
    try:
        batched = json.load(open("output/exp_phase_diagram.json"))
        batched_r = {(r["delta"], r["onset"]): r["correct"] for r in batched["results"]}
        seq_r = {(r["delta"], r["onset"]): r["correct"] for r in all_results}
        disagree = []
        for key in sorted(seq_r.keys()):
            if key in batched_r and batched_r[key] != seq_r[key]:
                disagree.append((key, batched_r[key], seq_r[key]))
        print(f"  Disagreements: {len(disagree)}/{len(seq_r)}")
        for (d, o), b, s in disagree:
            print(f"    d={d:.2f} onset={o:3d}: batched={'Y' if b else 'N'}  seq={'Y' if s else 'N'}")
    except Exception as e:
        print(f"  Could not load batched: {e}")

    # Save
    report = {
        "model": MODEL_NAME, "method": "sequential",
        "baseline": {"label": bl_label, "answer": bl_ans, "n_tokens": ntok},
        "delta_values": DELTA_VALUES, "onset_values": ONSET_VALUES,
        "results": all_results,
        "t_crit": {str(k): v for k, v in t_crit.items()},
        "elapsed_s": round(elapsed, 1)
    }
    with open("output/exp_phase_diagram_sequential.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> output/exp_phase_diagram_sequential.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
