"""Gravitational field at L33: perturbation sensitivity map.

Measures how sensitive the output distribution is to perturbations at L33
in specific directions. Maps basin boundaries in hidden state space.

Directions tested:
  1. B-moment template (where correct answer lives)
  2. Convention direction e_c (EN-ZH mean diff)
  3. KV-SVD basis (what deflation acts on)
  4. Random controls (null distribution)
  5. Mean-readout direction (constant bias from C6b)
  6. System-prompt-absent direction (what Ghost does)

Predictions:
  P1: S_t(e_c) high during t<50, drops after (basin boundary closes)
  P2: S_t(v_B) peaks at B-moment (~step 142)
  P3: S_t(v_rand) uniformly low and flat
  P4: High anisotropy = low-rank basin boundary
  P5: If Ghost=deflation, cos(v_sys, e_c) > 0.3
"""
import json, time, re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'Qwen/Qwen2.5-3B'
DEVICE = 'cuda'
MAX_TOKENS = 400
N_LAYERS = 36
D_MODEL = 2048
EPSILON = 0.5  # perturbation magnitude (will also test 0.1, 1.0)
N_RANDOM = 10  # number of random control directions
TARGET_LAYER = 33  # last computation layer

SYS = ('You are solving an AMC 12A multiple choice math problem. '
       'Think step by step, show your work, then clearly state your '
       'final answer as (A), (B), (C), (D), or (E).')

P12_TEXT = ('The harmonic mean of a collection of numbers is the reciprocal '
           'of the arithmetic mean of the reciprocals of the numbers in the '
           'collection. For example, the harmonic mean of 4, 4, and 5 is\n\n'
           '1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n'
           'What is the harmonic mean of all the real roots of the 4050th '
           'degree polynomial\n\n'
           '\\prod_{k=1}^{2025} (kx^2 - 4x - 3) = '
           '(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)...'
           '(2025x^2 - 4x - 3)?\n\n'
           '(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3')

PROMPT = (f'<|im_start|>system\n{SYS}<|im_end|>\n'
          f'<|im_start|>user\n{P12_TEXT}<|im_end|>\n'
          f'<|im_start|>assistant\n')

# No system prompt version (for Ghost direction)
PROMPT_NO_SYS = (f'<|im_start|>user\n{P12_TEXT}<|im_end|>\n'
                 f'<|im_start|>assistant\n')

ANSWER_IDS = {'A': 32, 'B': 33, 'C': 34, 'D': 35, 'E': 36}
B_TOKEN_ID = 33

# ── Q-deflation (from existing experiments) ──────────────────────────

class QDeflation:
    """Q-only deflation for extracting B-moment template.
    Copied from working exp_delayed_deflation_p12.py."""
    def __init__(self, model, layers, r=4, alpha=0.1, refresh_every=25):
        self.model = model
        self.target_layers = set(layers)
        self.r = r
        self.alpha = alpha
        self.refresh_every = refresh_every
        self.hooks = []
        self.step = 0
        self.is_generating = False
        self.U_r = {}  # {layer: {kv_head: U_matrix}}
        self._install()

    def _install(self):
        for ell in self.target_layers:
            h = self.model.model.layers[ell].self_attn.q_proj.register_forward_hook(
                self._make_hook(ell))
            self.hooks.append(h)

    def _make_hook(self, li):
        def hook(module, input, output):
            if not self.is_generating or li not in self.U_r:
                return output
            batch, seq, d = output.shape
            n_heads = 16   # Qwen2.5-3B: 16 Q heads
            head_dim = 128
            n_kv = len(self.U_r[li])
            gs_per_kv = n_heads // n_kv
            tensor = output.view(batch, seq, n_heads, head_dim)
            for kv_h in range(n_kv):
                if kv_h not in self.U_r[li]:
                    continue
                U = self.U_r[li][kv_h]
                s, e = kv_h * gs_per_kv, (kv_h + 1) * gs_per_kv
                qg = tensor[:, :, s:e, :]
                proj = qg @ U @ U.T
                tensor[:, :, s:e, :] = qg - self.alpha * proj
            return tensor.view(batch, seq, d)
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
                    DEVICE, dtype=torch.float16)

    def start_gen(self):
        self.is_generating = True
        self.step = 0

    def tick(self, past_kv):
        self.step += 1
        if self.step % self.refresh_every == 0:
            self.refresh_basis(past_kv)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── Direction extraction ─────────────────────────────────────────────

def extract_b_moment_template(model, tokenizer):
    """Run deflated P12, find the step where B is rank-1 at L35,
    return h_L33 at that step as the B-moment template direction."""
    print("  Extracting B-moment template...", flush=True)
    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)

    captured_33 = {}
    captured_35 = {}

    def hook_33(module, inp, output):
        h = output[0]
        if h.dim() == 3:
            h = h[:, -1, :]
        captured_33['h'] = h.detach()

    def hook_35(module, inp, output):
        h = output[0]
        if h.dim() == 3:
            h = h[:, -1, :]
        captured_35['h'] = h.detach()

    hk33 = model.model.layers[33].register_forward_hook(hook_33)
    hk35 = model.model.layers[35].register_forward_hook(hook_35)

    defl = QDeflation(model, list(range(20, 36)), r=4, alpha=0.1, refresh_every=25)
    norm = model.model.norm
    lm_head = model.lm_head

    best_rank = 999999
    best_h33 = None
    best_step = -1
    gen_ids = []

    with torch.no_grad():
        for step in range(MAX_TOKENS):
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
                defl.start_gen()
                defl.refresh_basis(out.past_key_values)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)
            defl.tick(past_kv)

            # Check B rank at L35
            if 'h' in captured_35:
                normed = norm(captured_35['h'])
                out_logits = lm_head(normed).squeeze()
                b_rank = (out_logits > out_logits[B_TOKEN_ID]).sum().item() + 1
                if b_rank < best_rank:
                    best_rank = b_rank
                    best_h33 = captured_33['h'].cpu().float().squeeze()
                    best_step = step
                    if b_rank == 1:
                        break

            captured_33.clear()
            captured_35.clear()

    defl.remove()
    hk33.remove()
    hk35.remove()
    del past_kv, out
    torch.cuda.empty_cache()

    # Normalize to unit vector
    v_b = best_h33 / (best_h33.norm() + 1e-12)
    print(f"    B-moment at step {best_step}, B rank={best_rank}", flush=True)
    return v_b.numpy(), best_step


def extract_convention_direction(model, tokenizer):
    """Compute e_c at L33 as normalized mean(zh) - mean(en) over test problems."""
    print("  Extracting convention direction e_c...", flush=True)
    problems = [
        {"en": "What is 7 * 8?", "zh": "7乘以8等于多少？"},
        {"en": "What is the sum of all prime numbers less than 20?",
         "zh": "所有小于20的质数之和是多少？"},
        {"en": "How many ways can you choose 3 items from 7?",
         "zh": "从7个物品中选3个有多少种方式？"},
        {"en": "What is 15% of 240?", "zh": "240的15%是多少？"},
        {"en": "Calculate: 8! / (5! * 3!)", "zh": "计算：8! / (5! × 3!)"},
        {"en": "If f(x) = 3x^2 - 2x + 1, what is f(2)?",
         "zh": "如果f(x) = 3x² - 2x + 1，求f(2)。"},
        {"en": "What is the area of a circle with radius 5?",
         "zh": "半径为5的圆的面积是多少？"},
        {"en": "What is the derivative of x^3 + 2x?",
         "zh": "x³ + 2x的导数是多少？"},
    ]

    captured = {}
    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            h = h[:, -1, :]
        captured['h'] = h.detach().cpu().float().reshape(D_MODEL).numpy()

    hk = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

    en_acts, zh_acts = [], []
    with torch.inference_mode():
        for p in problems:
            for lang, store in [('en', en_acts), ('zh', zh_acts)]:
                msgs = [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": p[lang]},
                ]
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
                model(ids)
                store.append(captured['h'].copy())
                captured.clear()

    hk.remove()

    en_mean = np.mean(np.stack(en_acts), axis=0)
    zh_mean = np.mean(np.stack(zh_acts), axis=0)
    diff = zh_mean - en_mean
    e_c = diff / (np.linalg.norm(diff) + 1e-12)
    print(f"    e_c norm before normalization: {np.linalg.norm(diff):.4f}", flush=True)
    return e_c


def extract_sys_absent_direction(model, tokenizer):
    """Compute the direction Ghost moves h_L33: h(no_sys) - h(full_sys), normalized.
    Uses the P12 prompt encoding (not generation)."""
    print("  Extracting system-prompt-absent direction...", flush=True)

    captured = {}
    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            h = h[:, -1, :]
        captured['h'] = h.detach().cpu().float().reshape(D_MODEL).numpy()

    hk = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

    with torch.inference_mode():
        # Full system prompt
        ids_full = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
        model(ids_full)
        h_full = captured['h'].copy()
        captured.clear()

        # No system prompt
        ids_no = tokenizer(PROMPT_NO_SYS, return_tensors='pt').input_ids.to(DEVICE)
        model(ids_no)
        h_no = captured['h'].copy()
        captured.clear()

    hk.remove()

    diff = h_no - h_full
    v_sys = diff / (np.linalg.norm(diff) + 1e-12)
    cos_with_itself = np.dot(h_full / np.linalg.norm(h_full),
                              h_no / np.linalg.norm(h_no))
    print(f"    cos(h_full, h_no) = {cos_with_itself:.4f}", flush=True)
    print(f"    ||h_no - h_full|| = {np.linalg.norm(diff):.4f}", flush=True)
    return v_sys


def extract_kv_svd_basis(model, tokenizer, top_k=4):
    """Extract top-k right singular vectors of the KV cache at L33
    after encoding the P12 prompt. This is what deflation acts on."""
    print(f"  Extracting KV-SVD top-{top_k} basis at L33...", flush=True)

    with torch.inference_mode():
        ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
        out = model(ids, use_cache=True)
        kv = out.past_key_values

        # Keys at L33: (batch, n_kv_heads, seq, head_dim)
        k33 = kv.layers[TARGET_LAYER].keys.squeeze(0)  # (n_kv_heads, seq, head_dim)
        k_cat = k33.reshape(-1, k33.shape[-1]).float()  # (n_kv_heads*seq, head_dim)
        U, S, Vt = torch.linalg.svd(k_cat, full_matrices=False)

        # These are in head_dim space (128). We need to express them in d_model space.
        # The q_proj maps d_model -> n_heads * head_dim. For sensitivity in h-space,
        # we need to go through q_proj. But for a first pass, just use the key
        # directions padded/tiled to d_model is wrong.
        #
        # Actually: the hidden state h goes through layer_norm -> q_proj -> heads.
        # For perturbation in h-space, we want directions that maximally change
        # the query-key dot products. That's q_proj^T @ key_direction for each head.
        # But that's complex. Simpler first pass: just use random directions in d_model
        # as the KV-SVD proxy, and note this limitation.
        #
        # Better approach: get the actual W_q projection matrix, compute
        # v_kv = W_q^T @ (tiled key singular vector)

        # Get W_q
        W_q = model.model.layers[TARGET_LAYER].self_attn.q_proj.weight.data  # (d_q, d_model)
        # d_q = n_heads * head_dim. Qwen2.5-3B: 16 heads * 128 = 2048
        n_heads = model.config.num_attention_heads
        head_dim = W_q.shape[0] // n_heads

        kv_dirs = []
        for i in range(top_k):
            # Singular vector in head_dim space
            vi = Vt[i]  # (head_dim,)
            # Tile across all heads: the same direction in every head's subspace
            vi_full = vi.unsqueeze(0).repeat(n_heads, 1).reshape(-1)  # (d_q,)
            # Map back to d_model space: pseudo-inverse of W_q
            # v_h = W_q^T @ vi_full (since W_q is roughly orthogonal per head)
            v_h = (W_q.float().T @ vi_full).cpu().numpy()
            v_h = v_h / (np.linalg.norm(v_h) + 1e-12)
            kv_dirs.append(v_h)
            print(f"    KV-SVD direction {i}: sv={S[i].item():.2f}", flush=True)

    del kv, out
    torch.cuda.empty_cache()
    return kv_dirs


def extract_mean_readout_direction(model, tokenizer):
    """Extract mean attention output at L33 across a short generation.
    From C6b: attention at last token contributes constant bias."""
    print("  Extracting mean-readout direction...", flush=True)

    attn_outs = []
    def hook(module, inp, output):
        # self_attn returns (attn_output, attn_weights, past_kv)
        # attn_output shape: (batch, seq, d_model)
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            h = h[:, -1, :]
        attn_outs.append(h.detach().cpu().float().reshape(D_MODEL).numpy())

    hk = model.model.layers[TARGET_LAYER].self_attn.register_forward_hook(hook)
    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)

    gen_ids = []
    with torch.no_grad():
        for step in range(100):  # 100 steps for mean
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

    hk.remove()
    del past_kv, out
    torch.cuda.empty_cache()

    mean_attn = np.mean(attn_outs[1:], axis=0)  # skip prefill
    v_mu = mean_attn / (np.linalg.norm(mean_attn) + 1e-12)
    print(f"    Mean attn output norm: {np.linalg.norm(mean_attn):.4f}", flush=True)
    return v_mu



# -- Sensitivity measurement (V2: full forward via hook) -------------------

def measure_sensitivity(model, tokenizer, directions, epsilon=EPSILON):
    """V2: Perturb h_L33 via hook, re-run the same token through the FULL
    model forward pass (L34-L35 attention+MLP included). Compares with
    direct path (h -> norm -> lm_head) for reference.

    For each generation step:
      1. Normal forward pass -> real logits, capture h_L33
      2. For each direction: install perturbation hook on L33,
         re-run same token with use_cache=False, measure KL
    """
    print(f"\n  Measuring sensitivity (eps={epsilon})...", flush=True)
    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)

    dir_tensors = {}
    for name, v in directions.items():
        dir_tensors[name] = torch.tensor(v, dtype=torch.float16, device=DEVICE)

    norm_fn = model.model.norm
    lm_head_fn = model.lm_head

    # Hook to capture h_L33 from the real forward pass
    captured_h = {}
    def hook_capture(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            h = h[:, -1, :]
        captured_h['h'] = h.detach().reshape(1, D_MODEL)

    hk_capture = model.model.layers[TARGET_LAYER].register_forward_hook(hook_capture)

    # Perturbation hook container
    pert_active = {'v': None}

    def hook_perturb(module, inp, output):
        if pert_active['v'] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            h_new = h.clone()
            h_new[:, -1, :] = h[:, -1, :] + pert_active['v']
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        return h + pert_active['v']

    sensitivities = {name: [] for name in directions}
    sensitivities_direct = {name: [] for name in directions}
    gen_ids = []
    answer_ranks = []

    t0 = time.time()
    with torch.no_grad():
        for step in range(MAX_TOKENS):
            # --- Normal forward pass ---
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)

            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            tid = next_id.item()
            if tid in (151643, 151645):
                break
            gen_ids.append(tid)

            log_p0 = F.log_softmax(logits.float(), dim=-1)
            p0 = log_p0.exp()
            b_rank = (logits[0] > logits[0, B_TOKEN_ID]).sum().item() + 1
            answer_ranks.append(b_rank)

            h0 = captured_h.get('h', None)

            # --- Perturbed forward passes ---
            hk_pert = model.model.layers[TARGET_LAYER].register_forward_hook(hook_perturb)

            for name, v in dir_tensors.items():
                pert_active['v'] = epsilon * v.unsqueeze(0)

                if step == 0:
                    out_p = model(input_ids=input_ids, use_cache=False)
                else:
                    out_p = model(input_ids=next_id, past_key_values=past_kv,
                                  use_cache=False)

                logits_p = out_p.logits[:, -1, :]
                log_p1 = F.log_softmax(logits_p.float(), dim=-1)
                kl = (p0 * (log_p0 - log_p1)).sum(dim=-1).item()
                sensitivities[name].append(kl)

                # Direct path comparison
                if h0 is not None:
                    h_pert = h0 + epsilon * v.unsqueeze(0)
                    normed_d = norm_fn(h_pert)
                    logits_d = lm_head_fn(normed_d).float()
                    log_pd = F.log_softmax(logits_d, dim=-1)
                    kl_d = (p0 * (log_p0 - log_pd)).sum(dim=-1).item()
                    sensitivities_direct[name].append(kl_d)

            pert_active['v'] = None
            hk_pert.remove()
            captured_h.clear()

            if (step + 1) % 50 == 0:
                elapsed = round(time.time() - t0, 1)
                print(f"    Step {step+1}/{MAX_TOKENS} ({elapsed}s)", flush=True)

    dt = round(time.time() - t0, 1)
    hk_capture.remove()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del past_kv, out
    torch.cuda.empty_cache()
    print(f"    Done: {len(gen_ids)} tokens, {dt}s", flush=True)
    return sensitivities, sensitivities_direct, answer_ranks, gen_ids, text, dt
# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GRAVITATIONAL FIELD AT L33: PERTURBATION SENSITIVITY MAP")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    print(f"Loaded {MODEL_NAME}\n", flush=True)

    # ── Phase 1: Extract all directions ──────────────────────────────
    print("PHASE 1: Direction extraction", flush=True)

    directions = {}

    # 1. B-moment template
    v_b, b_step = extract_b_moment_template(model, tokenizer)
    directions['b_moment'] = v_b

    # 2. Convention direction
    e_c = extract_convention_direction(model, tokenizer)
    directions['convention_ec'] = e_c

    # 3. System-prompt-absent direction
    v_sys = extract_sys_absent_direction(model, tokenizer)
    directions['sys_absent'] = v_sys

    # 4. KV-SVD basis
    kv_dirs = extract_kv_svd_basis(model, tokenizer, top_k=4)
    for i, v in enumerate(kv_dirs):
        directions[f'kv_svd_{i}'] = v

    # 5. Mean-readout direction
    v_mu = extract_mean_readout_direction(model, tokenizer)
    directions['mean_readout'] = v_mu

    # 6. Random controls
    rng = np.random.RandomState(42)
    for i in range(N_RANDOM):
        v = rng.randn(D_MODEL).astype(np.float32)
        v /= np.linalg.norm(v)
        directions[f'random_{i}'] = v

    # ── Cross-direction cosines ──────────────────────────────────────
    print("\nDIRECTION COSINES:", flush=True)
    named_dirs = ['b_moment', 'convention_ec', 'sys_absent', 'mean_readout',
                  'kv_svd_0', 'kv_svd_1']
    for i, n1 in enumerate(named_dirs):
        for n2 in named_dirs[i+1:]:
            cos = np.dot(directions[n1], directions[n2])
            print(f"  cos({n1}, {n2}) = {cos:.4f}", flush=True)

    # ── Phase 2: Sensitivity measurement ─────────────────────────────
    print("\nPHASE 2: Sensitivity measurement", flush=True)

    results = {}
    for eps in [0.5]:  # single epsilon for v2 (full forward is slower)
        print(f"\n--- epsilon = {eps} ---", flush=True)
        sens, sens_direct, b_ranks, gen_ids, text, dt = measure_sensitivity(
            model, tokenizer, directions, epsilon=eps)

        results[f'eps_{eps}'] = {
            'sensitivities_full': {k: v for k, v in sens.items()},
            'sensitivities_direct': {k: v for k, v in sens_direct.items()},
            'b_ranks': b_ranks,
            'n_tokens': len(gen_ids),
            'time_s': dt,
            'text_first_200': text[:200],
            'text_last_200': text[-200:],
        }

    # ── Phase 3: Analysis ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70, flush=True)

    # Use eps=0.5 for primary analysis
    primary = results['eps_0.5']
    sens_full = primary['sensitivities_full']
    sens_direct = primary['sensitivities_direct']
    n_steps = len(sens_full['b_moment'])

    # DIRECT vs FULL comparison
    print("\n  DIRECT PATH vs FULL FORWARD PASS:", flush=True)
    for name in ['b_moment', 'convention_ec', 'sys_absent', 'mean_readout',
                 'kv_svd_0', 'random_0']:
        full_mean = np.mean(sens_full[name])
        direct_mean = np.mean(sens_direct[name])
        ratio = full_mean / (direct_mean + 1e-12)
        print(f"  {name:20s}: full={full_mean:.6f} direct={direct_mean:.6f} "
              f"ratio={ratio:.2f}x", flush=True)

    # Cross-direction correlations (FULL path)
    print("\n  FULL-PATH CROSS-CORRELATIONS:", flush=True)
    key_dirs = ['b_moment', 'convention_ec', 'sys_absent', 'mean_readout',
                'kv_svd_0', 'random_0']
    for i, n1 in enumerate(key_dirs):
        for n2 in key_dirs[i+1:]:
            r = np.corrcoef(sens_full[n1], sens_full[n2])[0, 1]
            print(f"  corr({n1}, {n2}) = {r:.4f}", flush=True)

    # P1: e_c sensitivity before vs after T0=50 (FULL path)
    print("\n  P1: EARLY vs LATE SENSITIVITY (full path):", flush=True)
    t0_boundary = min(50, n_steps)
    for name in ['convention_ec', 'sys_absent', 'b_moment', 'mean_readout']:
        early = np.mean(sens_full[name][:t0_boundary]) if t0_boundary > 0 else 0
        late = np.mean(sens_full[name][t0_boundary:]) if n_steps > t0_boundary else 0
        ratio = early / (late + 1e-12)
        print(f"  {name}: early={early:.6f}, late={late:.6f}, ratio={ratio:.2f}",
              flush=True)

    # P3: Random baseline (FULL path)
    rand_means = []
    for i in range(N_RANDOM):
        rand_means.append(np.mean(sens_full[f'random_{i}']))
    rand_mean = np.mean(rand_means)
    rand_std = np.std(rand_means)
    print(f"\n  Random baseline (full): mean={rand_mean:.6f}, std={rand_std:.6f}",
          flush=True)

    # P4: Anisotropy (FULL path)
    print("  ANISOTROPY (full path, vs random):", flush=True)
    for name in ['b_moment', 'convention_ec', 'sys_absent', 'mean_readout',
                 'kv_svd_0', 'kv_svd_1']:
        overall = np.mean(sens_full[name])
        aniso = overall / (rand_mean + 1e-12)
        print(f"  Anisotropy({name}) = {aniso:.2f}x random", flush=True)

    # P2: B-moment peak
    b_sens = sens_full['b_moment']
    peak_step = int(np.argmax(b_sens))
    peak_val = b_sens[peak_step]
    print(f"\n  B-moment sensitivity peak: step {peak_step}, val={peak_val:.6f}",
          flush=True)
    print(f"  Original B-moment was at step {b_step}", flush=True)

    # P5: Ghost=deflation test
    cos_sys_ec = np.dot(directions['sys_absent'], directions['convention_ec'])
    print(f"\n  P5 (Ghost=deflation): cos(sys_absent, convention_ec) = {cos_sys_ec:.4f}",
          flush=True)
    corr_sys_ec = np.corrcoef(sens_full['sys_absent'], sens_full['convention_ec'])[0, 1]
    print(f"  Full-path trace correlation(sys_absent, convention_ec) = {corr_sys_ec:.4f}",
          flush=True)

    # B rank trajectory
    b_ranks = primary['b_ranks']
    min_rank = min(b_ranks)
    min_rank_step = b_ranks.index(min_rank)
    print(f"\n  B rank: min={min_rank} at step {min_rank_step}", flush=True)
    print(f"  B rank at step 50: {b_ranks[49] if len(b_ranks) > 49 else 'N/A'}",
          flush=True)

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        'model': MODEL_NAME,
        'target_layer': TARGET_LAYER,
        'n_random': N_RANDOM,
        'b_moment_step_deflated': b_step,
        'direction_cosines': {},
        'results': {},
    }

    # Direction cosines
    for i, n1 in enumerate(named_dirs):
        for n2 in named_dirs[i+1:]:
            output['direction_cosines'][f'{n1}_vs_{n2}'] = float(
                np.dot(directions[n1], directions[n2]))
    output['direction_cosines']['sys_absent_vs_convention_ec'] = float(cos_sys_ec)

    # Per-epsilon results
    for eps_key, res in results.items():
        output['results'][eps_key] = {
            'sensitivities_full': {k: [float(x) for x in v]
                                   for k, v in res['sensitivities_full'].items()},
            'sensitivities_direct': {k: [float(x) for x in v]
                                     for k, v in res['sensitivities_direct'].items()},
            'b_ranks': res['b_ranks'],
            'n_tokens': res['n_tokens'],
            'time_s': res['time_s'],
            'text_first_200': res['text_first_200'],
            'text_last_200': res['text_last_200'],
        }

    # Summary stats
    output['summary'] = {
        'predictions': {
            'P1_ec_early_late_ratio': {},
            'P2_b_moment_peak_step': {},
            'P3_random_baseline': {'mean': float(rand_mean), 'std': float(rand_std)},
            'P4_anisotropy': {},
            'P5_ghost_deflation_cos': float(cos_sys_ec),
            'P5_trace_correlation': float(corr_sys_ec),
        }
    }
    for name in ['convention_ec', 'sys_absent', 'b_moment', 'mean_readout']:
        s = sens_full[name]
        early = float(np.mean(s[:t0_boundary])) if t0_boundary > 0 else 0
        late = float(np.mean(s[t0_boundary:])) if n_steps > t0_boundary else 0
        output['summary']['predictions']['P1_ec_early_late_ratio'][name] = {
            'early': early, 'late': late, 'ratio': early / (late + 1e-12)
        }
        output['summary']['predictions']['P4_anisotropy'][name] = float(
            np.mean(s) / (rand_mean + 1e-12))

    peak_step_int = int(np.argmax(sens_full['b_moment']))
    output['summary']['predictions']['P2_b_moment_peak_step'] = {
        'peak_step': peak_step_int,
        'deflated_b_step': b_step,
    }

    outpath = Path('output') / 'exp_gravitational_field_v2_p12.json'

    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}", flush=True)


if __name__ == '__main__':
    main()
