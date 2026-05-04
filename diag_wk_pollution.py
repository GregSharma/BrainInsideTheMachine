#!/usr/bin/env python3
"""Diagnostic: W_K pollution measurement during baseline generation.

No intervention. Pure observation. For each generation step at each layer:
1. ||h_col(W_K)|| / ||h_null(W_K)|| u2014 MLP leakage into key-loud directions
2. Attention entropy at next layer u2014 is attention diffusing as cache grows
3. Generated token for productivity annotation

Precompute W_K SVD once, then observe baseline generation.
"""
import json, time, os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
TOP_K = 32  # top-k right singular vectors of W_K
MAX_TOKENS = 2048

SYS = (
    "You are solving an AMC 12A multiple choice math problem. "
    "Think step by step, show your work, then clearly state your "
    "final answer as (A), (B), (C), (D), or (E)."
)

PROBLEMS = {
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
    "p4_logic": {
        "text": (
            "Agnes writes the following four statements on a blank piece of paper.\n\n"
            "- At least one of these statements is true.\n"
            "- At least two of these statements are true.\n"
            "- At least two of these statements are false.\n"
            "- At least one of these statements is false.\n\n"
            "Each statement is either true or false. How many false statements "
            "did Agnes write on the paper?\n\n"
            "(A) 0  (B) 1  (C) 2  (D) 3  (E) 4"
        ),
        "answer": "B",
    },
    "p7_alien": {
        "text": (
            "In a certain alien world, the maximum running speed v of an organism "
            "is dependent on its number of toes n and number of eyes m. The "
            "relationship can be expressed as v = k * n^a * m^b centimeters per "
            "hour, where k, a, and b are integer constants. In a population where "
            "all organisms have 5 toes, log v = 4 + 2 log m; and in a population "
            "where all organisms have 25 eyes, log v = 4 + 4 log n, where the "
            "logarithms are base 10. What is k + a + b?\n\n"
            "(A) 20  (B) 21  (C) 22  (D) 23  (E) 24"
        ),
        "answer": "C",
    },
    "p9_complex": {
        "text": (
            "Let w be the complex number 2 + i, where i = sqrt(-1). What real "
            "number r has the property that r, w, and w^2 are three collinear "
            "points in the complex plane?\n\n"
            "(A) 3/4  (B) 1  (C) 7/5  (D) 3/2  (E) 5/3"
        ),
        "answer": "E",
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


def make_prompt(problem_text):
    return (
        "<|im_start|>system\n" + SYS + "<|im_end|>\n"
        "<|im_start|>user\n" + problem_text + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


print("Loading model (eager attention)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map=DEVICE,
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()
n_layers = len(model.model.layers)
print(f"Loaded. {n_layers} layers.\n", flush=True)

# ---- Precompute W_K SVD at each layer ----
print("Computing W_K SVD at each layer...", flush=True)
wk_V_topk = {}   # layer -> (d, k) top-k right singular vectors
wk_spectra = {}   # layer -> singular values
for ell in range(n_layers):
    W_K = model.model.layers[ell].self_attn.k_proj.weight.detach().float()
    U, S, Vh = torch.linalg.svd(W_K, full_matrices=False)
    wk_V_topk[ell] = Vh[:TOP_K, :].T.contiguous().to(DEVICE, dtype=torch.float16)
    wk_spectra[ell] = S.tolist()
print(f"SVD done. W_K shape: {W_K.shape}, top-{TOP_K} right SVs.\n", flush=True)

# ---- Run diagnostic ----
all_results = {
    "config": {
        "model": MODEL_NAME, "top_k": TOP_K,
        "max_tokens": MAX_TOKENS, "n_layers": n_layers,
    },
    "wk_spectra": {str(k): v for k, v in wk_spectra.items()},
    "problems": {},
}

for pname, pinfo in PROBLEMS.items():
    print(f"\n{'='*70}", flush=True)
    print(f"PROBLEM: {pname} (correct: {pinfo['answer']})", flush=True)
    print(f"{'='*70}", flush=True)

    prompt = make_prompt(pinfo["text"])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.shape[1]

    steps_data = []
    generated_ids = []
    past_key_values = None
    t0 = time.time()

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values

            # Greedy next token
            logits = outputs.logits[:, -1, :]
            next_token_id = logits.argmax(dim=-1, keepdim=True)
            tok_id = next_token_id.item()
            tok_str = tokenizer.decode([tok_id])

            if tok_id in (151643, 151645):
                break

            generated_ids.append(tok_id)

            # ---- Per-layer measurements ----
            layer_data = {}
            for ell in range(n_layers):
                h = outputs.hidden_states[ell + 1][:, -1, :].float()  # (1, d)
                h_norm = h.norm().item()

                # Project onto top-k W_K subspace
                V_k = wk_V_topk[ell].float()  # (d, k)
                coeffs = h @ V_k              # (1, k)
                h_col = coeffs @ V_k.T        # (1, d)
                h_null = h - h_col            # (1, d)

                col_norm = h_col.norm().item()
                null_norm = h_null.norm().item()
                ratio = col_norm / (null_norm + 1e-8)

                # Attention entropy at NEXT layer (ell+1)
                attn_ent = None
                norm_ent = None
                if ell + 1 < n_layers:
                    attn = outputs.attentions[ell + 1]    # (1, heads, q, kv)
                    attn_last = attn[0, :, -1, :].float() # fp32 for log
                    attn_last = attn_last.clamp(min=1e-10)
                    ent = -(attn_last * attn_last.log()).sum(dim=-1)
                    attn_ent = ent.mean().item()
                    seq_len = attn_last.shape[-1]
                    norm_ent = attn_ent / (np.log(seq_len) + 1e-10)

                layer_data[ell] = {
                    "r": round(ratio, 5),
                    "cn": round(col_norm, 2),
                    "nn": round(null_norm, 2),
                    "hn": round(h_norm, 2),
                    "ae": round(attn_ent, 4) if attn_ent is not None else None,
                    "ne": round(norm_ent, 4) if norm_ent is not None else None,
                }

            steps_data.append({
                "s": step,
                "t": tok_str,
                "L": {str(k): v for k, v in layer_data.items()},
            })

        if (step + 1) % 100 == 0:
            print(f"  step {step+1}...", flush=True)

    dt = time.time() - t0
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_gen = len(generated_ids)

    print(f"--- {n_gen} tokens, {dt:.1f}s ---", flush=True)
    print(f"Output (200c): {full_text[:200]}...", flush=True)

    all_results["problems"][pname] = {
        "correct": pinfo["answer"],
        "n_tokens": n_gen,
        "time_s": round(dt, 1),
        "output": full_text,
        "steps": steps_data,
    }

    del past_key_values, outputs
    torch.cuda.empty_cache()

# ---- Save ----
os.makedirs("output", exist_ok=True)
out_path = "output/diag_wk_pollution.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, ensure_ascii=False)
print(f"\nSaved to {out_path}", flush=True)

# ---- Quick summary ----
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for pname, pdata in all_results["problems"].items():
    steps = pdata["steps"]
    if not steps:
        continue
    n = len(steps)
    # Average ratio at key layers: early (L5), mid (L17), late (L30)
    for lbl, L in [("L5", "5"), ("L17", "17"), ("L30", "30")]:
        first_q = [s["L"][L]["r"] for s in steps[:n//4]]
        last_q = [s["L"][L]["r"] for s in steps[3*n//4:]]
        r_first = np.mean(first_q) if first_q else 0
        r_last = np.mean(last_q) if last_q else 0
        delta = r_last - r_first
        print(f"  {pname:15s} {lbl}: ratio first_q={r_first:.4f} last_q={r_last:.4f} delta={delta:+.4f}")
    # Attention entropy trend at L17 (next=L18)
    ent_first = [s["L"]["17"]["ne"] for s in steps[:n//4] if s["L"]["17"]["ne"] is not None]
    ent_last = [s["L"]["17"]["ne"] for s in steps[3*n//4:] if s["L"]["17"]["ne"] is not None]
    if ent_first and ent_last:
        e1, e2 = np.mean(ent_first), np.mean(ent_last)
        print(f"  {pname:15s} L17 norm_ent(L18): first_q={e1:.4f} last_q={e2:.4f} delta={e2-e1:+.4f}")
    print(f"  {pname:15s} tokens={pdata['n_tokens']}")
    print()
