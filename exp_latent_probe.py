#!/usr/bin/env python3
"""Experiment: When does the model KNOW vs when does it SAY?

For each generation step, extract the hidden state at probe layers and project
through the unembedding matrix for the 5 answer-choice tokens. This shows
the model's "latent vote" at each step — independent of the text it generates.

Runs on a configurable subset of AMC problems under baseline vs soft deflation.
Produces a JSON with per-step logit trajectories for each answer choice.
"""
import json, time, os, re, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
MAX_TOKENS = 2048

# Deflation params (same as sweep)
DEFLATE_LAYERS = list(range(20, 36))
DEFLATE_R = 4
REFRESH_EVERY = 25
ALPHA = 0.1

# Layers to probe for latent answer signal
PROBE_LAYERS = [18, 24, 27, 30, 33, 35]  # sample the depth

SYS = (
    "You are solving an AMC 12A multiple choice math problem. "
    "Think step by step, show your work, then clearly state your "
    "final answer as (A), (B), (C), (D), or (E)."
)

# Problems to run the probe on — start with the most interesting ones
PROBE_PROBLEMS = {
    4:  'B',   # FIXED by deflation
    12: 'B',   # the P12 harmonic mean — the origin story
    15: 'C',   # FIXED — strategy switch
    21: 'A',   # FIXED — loop broken
    22: 'E',   # FIXED
    11: 'A',   # BROKE — right computation, wrong box
    17: 'A',   # BROKE
    9:  'E',   # held correct (control)
}

# Answer-choice tokens: the actual token IDs for (A), (B), (C), (D), (E)
# We'll also track the raw letter tokens as backup
CHOICE_LETTERS = ['A', 'B', 'C', 'D', 'E']


def parse_amc_problems(filepath):
    """Parse AMC 12A markdown into {num: text} dict."""
    with open(filepath, 'r') as f:
        content = f.read()
    problems = {}
    parts = re.split(r'^Problem (\d+)\s*$', content, flags=re.MULTILINE)
    for i in range(1, len(parts) - 1, 2):
        num = int(parts[i])
        raw = parts[i + 1].strip()
        if num not in PROBE_PROBLEMS:
            continue
        raw = re.sub(r'\[Solution\]\([^)]*\)', '', raw)
        raw = re.sub(r'^-+\s*$', '', raw, flags=re.MULTILINE)
        if '[asy]' in raw.lower():
            continue
        raw = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', raw)
        raw = re.sub(r'\n{3,}', '\n\n', raw).strip()
        raw = re.sub(r'\nSee also.*', '', raw, flags=re.DOTALL)
        problems[num] = raw
    return problems


def make_prompt(text):
    return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


class HiddenStateProbe:
    """Hook to capture hidden states at specified layers during generation."""

    def __init__(self, model, layers):
        self.model = model
        self.layers = set(layers)
        self.hooks = []
        self.captures = {}  # {layer: tensor} — updated each forward pass
        self._install()

    def _install(self):
        for ell in self.layers:
            h = self.model.model.layers[ell].register_forward_hook(
                self._make_hook(ell)
            )
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            # output may be tuple or BaseModelOutput; first element is hidden_states
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            # Handle different shapes: [batch, seq, d] or [seq, d] or other
            if hs.dim() == 3:
                self.captures[li] = hs[:, -1, :].detach()
            elif hs.dim() == 2:
                self.captures[li] = hs[-1:, :].detach()
            else:
                self.captures[li] = hs.detach()
        return hook

    def get_answer_logits(self, answer_token_ids):
        """Project captured hidden states through unembedding for answer tokens.

        Returns {layer: {letter: logit}} for the current step.
        """
        lm_head = self.model.lm_head
        # lm_head.weight is [vocab, d_model]
        answer_embeds = lm_head.weight[answer_token_ids, :]  # [5, d_model]

        result = {}
        for li in sorted(self.captures.keys()):
            hs = self.captures[li].float()  # [1, d_model]
            # Apply the model's final RMSNorm before projecting
            normed = self.model.model.norm(hs.to(self.model.model.norm.weight.dtype))
            logits = (normed.float() @ answer_embeds.float().T).squeeze(0)  # [5]
            result[li] = {
                CHOICE_LETTERS[i]: logits[i].item()
                for i in range(5)
            }
        return result

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.captures.clear()


class SoftDeflation:
    """q_new = q - alpha * V V^T q at target layers."""

    def __init__(self, model, layers, r=4, alpha=0.1, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
        self.hooks = []
        self.step_count = 0
        self.is_generating = False
        self.U_r = {}
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_hook(ell)
            )
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
                return output
            q = output
            batch, seq, d = q.shape
            hd, n_q, n_kv = 128, 16, 2
            gs = n_q // n_kv
            q = q.view(batch, seq, n_q, hd)
            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs, (kv_h + 1) * gs
                qg = q[:, :, s:e, :]
                proj = qg @ U @ U.T
                q[:, :, s:e, :] = qg - self.alpha * proj
            return q.view(batch, seq, d)
        return hook

    def refresh_basis(self, past_kv):
        for ell in self.target_layers:
            keys = past_kv.layers[ell].keys
            self.U_r[ell] = {}
            for kv_h in range(keys.shape[1]):
                K = keys[0, kv_h, :, :].float()
                if K.shape[0] < self.r:
                    continue
                _, _, Vh = torch.linalg.svd(K, full_matrices=False)
                self.U_r[ell][kv_h] = Vh[:self.r, :].T.contiguous().to(
                    DEVICE, dtype=torch.float16
                )

    def start_gen(self):
        self.is_generating = True
        self.step_count = 0

    def tick(self, past_kv):
        self.step_count += 1
        if self.step_count % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.U_r.clear()


def run_probed(model, tokenizer, prompt, answer_token_ids, deflator=None,
               probe_layers=PROBE_LAYERS, sample_every=5):
    """Run generation with hidden-state probing at each step.

    Returns (text, n_tokens, time_s, trajectory)
    where trajectory = [{step, layer_logits: {layer: {A:.., B:.., ...}}, token: str}, ...]
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None
    trajectory = []
    t0 = time.time()

    probe = HiddenStateProbe(model, probe_layers)

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                if deflator:
                    deflator.start_gen()
                    deflator.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv,
                            use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)

            tid = next_id.item()
            if tid in (151643, 151645):
                break

            gen_ids.append(tid)
            tok_str = tokenizer.decode([tid])

            # Sample the probe periodically + always at first/last steps
            if step % sample_every == 0 or step < 10:
                layer_logits = probe.get_answer_logits(answer_token_ids)
                # Also get the actual output logits for the 5 answer tokens
                out_logits_for_answers = logits[0, answer_token_ids].tolist()

                trajectory.append({
                    "step": step,
                    "token": tok_str,
                    "layer_logits": {
                        str(li): vals for li, vals in layer_logits.items()
                    },
                    "output_logits": {
                        CHOICE_LETTERS[i]: out_logits_for_answers[i]
                        for i in range(5)
                    },
                })

            if deflator:
                deflator.tick(past_kv)

    dt = time.time() - t0
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    probe.remove()
    del past_kv, out
    torch.cuda.empty_cache()

    return text, len(gen_ids), round(dt, 1), trajectory


def extract_answer(text, pnum=None):
    """Pull answer letter from generation."""
    # (simplified from sweep script)
    found = []
    for m in re.finditer(r'\\boxed\{', text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            i += 1
        if depth == 0:
            found.append(text[start:i-1])
    if found:
        val = found[-1]
        clean = re.sub(r'\\text\{([^}]*)\}', r'\1', val).strip()
        clean = clean.replace('(','').replace(')','').strip()
        if clean in 'ABCDE' and len(clean) == 1:
            return clean
    tail = text[-400:]
    m = re.findall(r'\(([A-E])\)', tail)
    if m:
        return m[-1]
    return "?"


def main():
    problems = parse_amc_problems("2025_AMC_12A.md")
    print(f"Parsed {len(problems)} problems for probing")

    print(f"\nLoading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded. {len(model.model.layers)} layers.\n", flush=True)

    # Get token IDs for answer letters
    answer_token_ids = []
    for letter in CHOICE_LETTERS:
        tids = tokenizer.encode(letter, add_special_tokens=False)
        answer_token_ids.append(tids[0])
    answer_token_ids = torch.tensor(answer_token_ids, device=DEVICE)
    print(f"Answer token IDs: {dict(zip(CHOICE_LETTERS, answer_token_ids.tolist()))}")

    results = []

    for pnum in sorted(problems.keys()):
        ptext = problems[pnum]
        correct = PROBE_PROBLEMS[pnum]
        prompt = make_prompt(ptext)

        for cond_name, do_deflate in [("baseline", False), ("soft_a0.1", True)]:
            print(f"\n{'='*60}")
            print(f"Problem {pnum} / {cond_name} (correct={correct})")
            print(f"{'='*60}", flush=True)

            deflator = None
            if do_deflate:
                deflator = SoftDeflation(model, DEFLATE_LAYERS, r=DEFLATE_R,
                                         alpha=ALPHA, refresh_every=REFRESH_EVERY)

            text, ntok, dt, trajectory = run_probed(
                model, tokenizer, prompt, answer_token_ids,
                deflator=deflator, sample_every=5
            )

            if deflator:
                deflator.remove()

            ans = extract_answer(text, pnum)
            looped = ntok >= MAX_TOKENS

            print(f"  {ntok} tok, {dt}s, ans={ans} (correct={correct})")
            print(f"  Trajectory has {len(trajectory)} sample points")

            # Print key probe points
            if trajectory:
                # Show first, middle, last
                for idx in [0, len(trajectory)//4, len(trajectory)//2,
                           3*len(trajectory)//4, -1]:
                    t = trajectory[idx]
                    # Show layer 35 (last layer) answer logits
                    l35 = t["layer_logits"].get("35", {})
                    if l35:
                        winner = max(l35, key=l35.get)
                        print(f"  Step {t['step']:4d} | L35 winner={winner} "
                              f"| {' '.join(f'{k}:{v:+.1f}' for k,v in l35.items())}"
                              f" | tok='{t['token']}'")

            results.append({
                "problem": pnum,
                "condition": cond_name,
                "correct": correct,
                "answer": ans,
                "n_tokens": ntok,
                "time_s": dt,
                "looped": looped,
                "trajectory": trajectory,
            })

    # Save
    os.makedirs("output", exist_ok=True)
    outpath = "output/exp_latent_probe.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {outpath}")

    # Summary: for each problem, when does the correct answer become the
    # dominant logit at the last probe layer?
    print(f"\n{'='*80}")
    print("LATENT KNOWLEDGE TIMING: When does the model KNOW vs EMIT?")
    print(f"{'='*80}")

    for pnum in sorted(problems.keys()):
        for cond in ["baseline", "soft_a0.1"]:
            r = [x for x in results
                 if x["problem"] == pnum and x["condition"] == cond][0]
            correct = r["correct"]
            traj = r["trajectory"]
            if not traj:
                continue

            # Find first step where correct answer leads at L35
            first_knows = None
            last_layer = str(max(int(k) for k in traj[0]["layer_logits"].keys()))
            for t in traj:
                ll = t["layer_logits"].get(last_layer, {})
                if ll and max(ll, key=ll.get) == correct:
                    first_knows = t["step"]
                    break

            # Find the total tokens
            total = r["n_tokens"]
            ans = r["answer"]
            ok = "✓" if ans == correct else "✗"

            knows_str = f"step {first_knows}" if first_knows is not None else "NEVER"
            print(f"  P{pnum:2d} {cond:12s} | knows@{knows_str:>10s} "
                  f"| emits@step~{total:4d} | ans={ans}{ok} (correct={correct})")


if __name__ == "__main__":
    main()
