#!/usr/bin/env python3
"""Attention PACF + Key Gram eigenvalue diagnostic.

At each generation step, at selected layers:
- cos(a_t, a_{t-1}) over shared positions (attention stickiness)
- attention entropy
- every 25 steps: key cache effective rank

3 problems. 5 layers. No intervention.
"""
import json, time, os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048
LAYERS = [5, 17, 27, 30, 35]
GRAM_EVERY = 25

SYS = (
    "You are solving an AMC 12A multiple choice math problem. "
    "Think step by step, show your work, then clearly state your "
    "final answer as (A), (B), (C), (D), or (E)."
)

PROBLEMS = {
    "p9_complex": {
        "text": (
            "Let w be the complex number 2 + i, where i = sqrt(-1). What real "
            "number r has the property that r, w, and w^2 are three collinear "
            "points in the complex plane?\n\n"
            "(A) 3/4  (B) 1  (C) 7/5  (D) 3/2  (E) 5/3"
        ),
        "answer": "E",
    },
    "p3_age": {
        "text": (
            "A team of students is going to compete against a team of teachers "
            "in a trivia contest. The total number of students and teachers is 15. "
            "Ash, a cousin of one of the students, wants to join the contest. "
            "If Ash plays with the students, the average age on that team will "
            "increase from 12 to 14. If Ash plays with the teachers, the average "
            "age on that team will decrease from 55 to 52. How old is Ash?\n\n"
            "(A) 28  (B) 29  (C) 30  (D) 32  (E) 33"
        ),
        "answer": "A",
    },
    "p12_harmonic": {
        "text": (
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
        ),
        "answer": "B",
    },
}


def make_prompt(text):
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def get_keys(past_kv, layer_idx):
    return past_kv.layers[layer_idx].keys  # (batch, n_kv, T, d_k)


def eff_rank(K):
    """Effective rank from singular values of K (T x d)."""
    sv = torch.linalg.svdvals(K.float())
    sv2 = sv ** 2
    sv2 = sv2[sv2 > 1e-10]
    if sv2.numel() == 0:
        return 1.0
    p = sv2 / sv2.sum()
    return torch.exp(-(p * p.log()).sum()).item()


print("Loading model (eager attention)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map=DEVICE,
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()
print(f"Loaded. {len(model.model.layers)} layers.\n", flush=True)

results = {}

for pname, pinfo in PROBLEMS.items():
    print(f"\n{'='*70}", flush=True)
    print(f"PROBLEM: {pname} (correct: {pinfo['answer']})", flush=True)
    print(f"{'='*70}", flush=True)

    prompt = make_prompt(pinfo["text"])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    prev_attn = {L: None for L in LAYERS}
    series = {L: {"cos": [], "ent": [], "nent": [], "erank": []} for L in LAYERS}
    gen_ids = []
    past_kv = None
    t0 = time.time()

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            kw = dict(output_attentions=True, use_cache=True)
            if step == 0:
                out = model(input_ids=input_ids, **kw)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, **kw)

            past_kv = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            for L in LAYERS:
                # --- attention pattern (head-averaged, last query) ---
                attn = out.attentions[L][0, :, -1, :].float()  # (heads, T)
                a = attn.mean(dim=0)                             # (T,)

                # entropy
                ac = a.clamp(min=1e-10)
                ent = -(ac * ac.log()).sum().item()
                T = a.shape[0]
                series[L]["ent"].append(round(ent, 4))
                series[L]["nent"].append(round(ent / (np.log(T) + 1e-10), 4))

                # cosine sim with previous step
                if prev_attn[L] is not None:
                    n = prev_attn[L].shape[0]
                    cs = F.cosine_similarity(
                        a[:n].unsqueeze(0), prev_attn[L].unsqueeze(0)
                    ).item()
                    series[L]["cos"].append(round(cs, 5))
                else:
                    series[L]["cos"].append(None)
                prev_attn[L] = a.clone()

                # key cache effective rank
                if step % GRAM_EVERY == 0:
                    keys = get_keys(past_kv, L)  # (1, n_kv, T, d_k)
                    ranks = [eff_rank(keys[0, h]) for h in range(keys.shape[1])]
                    er = float(np.mean(ranks))
                    series[L]["erank"].append({
                        "s": step, "T": int(keys.shape[2]),
                        "er": round(er, 2),
                        "ratio": round(er / min(keys.shape[2], keys.shape[3]), 4),
                    })

        if (step + 1) % 200 == 0:
            print(f"  step {step+1}...", flush=True)

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n = len(gen_ids)
    print(f"--- {n} tokens, {dt:.1f}s ---", flush=True)

    # ACF of cos_sim (lags 1-20)
    acf = {}
    for L in LAYERS:
        cs = np.array([c for c in series[L]["cos"] if c is not None])
        if len(cs) > 50:
            x = cs - cs.mean()
            c0 = np.dot(x, x)
            if c0 > 1e-10:
                acf[L] = [round(np.dot(x[k:], x[:-k]) / c0 if k > 0 else 1.0, 4)
                          for k in range(21)]

    results[pname] = {
        "correct": pinfo["answer"], "n_tokens": n, "time_s": round(dt, 1),
        "output_tail": text[-400:],
        "series": {str(L): s for L, s in series.items()},
        "acf_cos": {str(L): v for L, v in acf.items()},
    }
    del past_kv, out
    torch.cuda.empty_cache()

os.makedirs("output", exist_ok=True)
with open("output/diag_attn_pacf.json", "w") as f:
    json.dump(results, f, ensure_ascii=False)
print(f"\nSaved to output/diag_attn_pacf.json")

# ---- Summary ----
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for pname, r in results.items():
    print(f"\n{pname} ({r['n_tokens']} tok, correct={r['correct']}):")
    for L in LAYERS:
        Ls = str(L)
        cs = [c for c in r["series"][Ls]["cos"] if c is not None]
        if not cs:
            continue
        cs = np.array(cs)
        n4 = len(cs) // 4
        print(f"  L{L}: cos_sim  mean={cs.mean():.4f}  Q1={cs[:n4].mean():.4f}  Q4={cs[-n4:].mean():.4f}  delta={cs[-n4:].mean()-cs[:n4].mean():+.4f}")

        er = r["series"][Ls]["erank"]
        if len(er) >= 2:
            print(f"        eff_rank first={er[0]['er']:.1f}(T={er[0]['T']})  "
                  f"last={er[-1]['er']:.1f}(T={er[-1]['T']})  "
                  f"ratio_last={er[-1]['ratio']:.3f}")

        ne = np.array(r["series"][Ls]["nent"])
        if len(ne) > 4:
            print(f"        norm_ent Q1={ne[:n4].mean():.4f}  Q4={ne[-n4:].mean():.4f}  delta={ne[-n4:].mean()-ne[:n4].mean():+.4f}")

    if Ls in r["acf_cos"]:
        print(f"  ACF cos_sim (lags 1-5):")
        for Ls2, a in r["acf_cos"].items():
            print(f"    L{Ls2}: {[round(x,3) for x in a[1:6]]}")
