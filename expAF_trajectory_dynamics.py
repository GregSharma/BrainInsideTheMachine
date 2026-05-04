"""
Exp AF: Residual Stream Trajectory Dynamics

Von Neumann's ask: measure the dynamics, not snapshots.

Each layer is a step in a trajectory through 2048-dimensional space.
Track: position (h_l), velocity (h_l - h_{l-1}), acceleration (v_l - v_{l-1}), norm.

Do this for ZH and EN versions of the same problem.
Do this for baseline vs flip.

Questions:
  1. Where are the phase transitions? (velocity drops)
  2. Are ZH/EN trajectories parallel, converging, or diverging-reconverging?
  3. What does the flip do to the trajectory, not just the endpoint?
  4. Is there a Lyapunov-like energy that decreases monotonically?

If the Z hypothesis is true: ZH and EN trajectories are parallel (constant offset).
If Lyapunov: norm of velocity decreases across layers (system settling).
If gauge symmetry: flip preserves the shape of the trajectory but rotates the frame.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
SEED = 42
ALL_LAYERS = list(range(36))
STRIP_LAYERS = list(range(9, 27))

# Same 10 EN/ZH problem pairs as always
PROBLEMS = [
    {'en': 'Calculate 47 + 86.',                                                                 'zh': '计算 47 + 86 的值。',                      'answer': '133'},
    {'en': 'A rectangle has length 12 and width 5. Find its area.',                              'zh': '一个长方形的长为 12，宽为 5，求其面积。',   'answer': '60'},
    {'en': 'What is the remainder when 100 is divided by 7?',                                    'zh': '100 除以 7 的余数是多少？',                 'answer': '2'},
    {'en': 'Calculate 15 × 8.',                                                                  'zh': '计算 15 × 8 的值。',                       'answer': '120'},
    {'en': 'An arithmetic sequence has first term 2 and common difference 3. Find the sum of the first 5 terms.', 'zh': '等差数列首项为 2，公差为 3，求前 5 项之和。', 'answer': '40'},
    {'en': 'Calculate 387 × 29.',                                                                'zh': '计算 387 × 29 的值。',                     'answer': '11223'},
    {'en': 'Find the value of C(10, 3).',                                                        'zh': '求组合数 C(10, 3) 的值。',                  'answer': '120'},
    {'en': 'What is the remainder when 7654 is divided by 37?',                                  'zh': '7654 除以 37 的余数是多少？',               'answer': '34'},
    {'en': 'An arithmetic sequence has first term 7 and common difference 11. Find the sum of the first 25 terms.', 'zh': '等差数列首项为 7，公差为 11，求前 25 项之和。', 'answer': '3475'},
    {'en': 'A rectangle has length 47 and width 33. Find its area.',                             'zh': '一个长方形的长为 47，宽为 33，求其面积。',  'answer': '1551'},
]


def extract_trajectory(model, tokenizer, prompt, layers=ALL_LAYERS):
    """Extract residual stream h at every layer for the last token."""
    layer_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    inp = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        model(**inp)
    for h in handles:
        h.remove()

    # Also get embedding (layer -1)
    emb_out = {}
    def emb_hook(module, inp, out):
        emb_out['emb'] = out.detach().cpu().squeeze(0)[-1].float().numpy()
    h_emb = model.model.embed_tokens.register_forward_hook(emb_hook)
    with torch.no_grad():
        model(**inp)
    h_emb.remove()

    traj = [emb_out['emb']] + [layer_out[l] for l in layers]
    return np.stack(traj)  # (37, 2048): index 0=embed, 1..36=layers 0..35


def fit_1d_dirs(model, tokenizer, layers=STRIP_LAYERS):
    """Fit ZH-EN mean-difference direction per layer."""
    rng = pyrandom.Random(SEED)
    problems = []
    per_cat = 40
    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        problems.append({"zh": f"计算 {a} + {b} 的值。", "en": f"Calculate {a} + {b}."})
    for _ in range(per_cat):
        n_val = rng.randint(5, 20); k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"zh": f"求组合数 C({n_val}, {k_val}) 的值。", "en": f"Find the value of C({n_val}, {k_val})."})
    for _ in range(per_cat):
        a = rng.randint(50, 9999); b = rng.randint(3, 37)
        problems.append({"zh": f"{a} 除以 {b} 的余数是多少？", "en": f"What is the remainder when {a} is divided by {b}?"})
    for _ in range(per_cat):
        w = rng.randint(2, 50); h = rng.randint(2, 50)
        problems.append({"zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。", "en": f"A rectangle has length {w} and width {h}. Find its area."})
    for _ in range(per_cat):
        a1 = rng.randint(1, 20); d_val = rng.randint(1, 10); n_t = rng.randint(5, 30)
        problems.append({"zh": f"等差数列首项为 {a1}，公差为 {d_val}，求前 {n_t} 项之和。",
                         "en": f"An arithmetic sequence has first term {a1} and common difference {d_val}. Find the sum of the first {n_t} terms."})
    rng.shuffle(problems)

    layer_acts = {l: {'zh': [], 'en': []} for l in layers}
    layer_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        for lang in ['zh', 'en']:
            for p in tqdm(problems, desc=f"  1D {lang}", leave=False):
                inp = tokenizer(p[lang], return_tensors='pt').to(model.device)
                with torch.no_grad(): model(**inp)
                for l in layers:
                    layer_acts[l][lang].append(layer_out[l].copy())
                layer_out.clear()
    finally:
        for h in handles: h.remove()

    dirs = {}
    for l in layers:
        zh_m = np.mean(layer_acts[l]['zh'], axis=0)
        en_m = np.mean(layer_acts[l]['en'], axis=0)
        v = zh_m - en_m
        dirs[l] = torch.tensor(v / (np.linalg.norm(v) + 1e-8), dtype=torch.bfloat16)
    return dirs


def extract_trajectory_with_flip(model, tokenizer, prompt, dirs_1d, scale=-1.0, layers=ALL_LAYERS):
    """Extract trajectory with 1D flip applied at MLP outputs in STRIP_LAYERS."""
    layer_out = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_out[l] = h.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles_layer = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]

    def make_flip(l):
        def hook(module, inp, out):
            if l not in dirs_1d: return out
            d_vec = dirs_1d[l].to(out.device)
            proj = (out * d_vec).sum(dim=-1, keepdim=True)
            return out + scale * 2.0 * proj * d_vec
        return hook

    handles_mlp = [model.model.layers[l].mlp.register_forward_hook(make_flip(l)) for l in STRIP_LAYERS]

    emb_out = {}
    def emb_hook(module, inp, out):
        emb_out['emb'] = out.detach().cpu().squeeze(0)[-1].float().numpy()
    h_emb = model.model.embed_tokens.register_forward_hook(emb_hook)

    inp = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        model(**inp)

    for h in handles_layer: h.remove()
    for h in handles_mlp: h.remove()
    h_emb.remove()

    traj = [emb_out.get('emb', np.zeros(2048))] + [layer_out[l] for l in layers]
    return np.stack(traj)  # (37, 2048)


def compute_dynamics(traj):
    """
    traj: (37, 2048) — index 0=embed, 1..36=layers 0..35
    Returns dict of trajectory statistics per step.
    """
    norms = np.linalg.norm(traj, axis=1)                  # (37,)
    velocity = np.diff(traj, axis=0)                       # (36,) deltas
    vel_norms = np.linalg.norm(velocity, axis=1)           # (36,) speed
    accel = np.diff(velocity, axis=0)                      # (35,)
    accel_norms = np.linalg.norm(accel, axis=1)            # (35,)

    # Cosine between consecutive velocities (are we turning?)
    cos_vel = []
    for i in range(len(velocity) - 1):
        v1, v2 = velocity[i], velocity[i+1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        cos_vel.append(float(np.dot(v1, v2) / (n1 * n2 + 1e-8)))

    # Cosine between position and velocity (moving toward/away from origin?)
    cos_pos_vel = []
    for i in range(len(velocity)):
        h, v = traj[i+1], velocity[i]
        nh, nv = np.linalg.norm(h), np.linalg.norm(v)
        cos_pos_vel.append(float(np.dot(h, v) / (nh * nv + 1e-8)))

    return {
        'norms': norms.tolist(),
        'vel_norms': vel_norms.tolist(),
        'accel_norms': accel_norms.tolist(),
        'cos_consecutive_vel': cos_vel,   # (35,) — are consecutive deltas aligned?
        'cos_pos_vel': cos_pos_vel,        # (36,) — radial motion
    }


def compute_zh_en_alignment(traj_zh, traj_en):
    """
    Per-layer alignment between ZH and EN trajectories.
    Tests: are they parallel (Z hypothesis)? Converging? Diverging?
    """
    # Offset vector at each layer
    offsets = traj_zh - traj_en                   # (37, 2048)
    offset_norms = np.linalg.norm(offsets, axis=1)  # (37,)

    # Cosine between offset vectors at consecutive layers (is the offset stable?)
    cos_offset = []
    for i in range(len(offsets) - 1):
        o1, o2 = offsets[i], offsets[i+1]
        n1, n2 = np.linalg.norm(o1), np.linalg.norm(o2)
        cos_offset.append(float(np.dot(o1, o2) / (n1 * n2 + 1e-8)))

    # Cosine between ZH and EN trajectories at each layer
    cos_direct = []
    for i in range(37):
        h_zh, h_en = traj_zh[i], traj_en[i]
        n_zh, n_en = np.linalg.norm(h_zh), np.linalg.norm(h_en)
        cos_direct.append(float(np.dot(h_zh, h_en) / (n_zh * n_en + 1e-8)))

    # Cosine between ZH velocity and EN velocity at each step
    vel_zh = np.diff(traj_zh, axis=0)
    vel_en = np.diff(traj_en, axis=0)
    cos_vel_alignment = []
    for i in range(36):
        v1, v2 = vel_zh[i], vel_en[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        cos_vel_alignment.append(float(np.dot(v1, v2) / (n1 * n2 + 1e-8)))

    return {
        'offset_norms': offset_norms.tolist(),
        'cos_offset_consecutive': cos_offset,   # stable offset → Z hypothesis
        'cos_direct': cos_direct,                # are positions aligned?
        'cos_vel_alignment': cos_vel_alignment,  # are velocities aligned?
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                                  device_map='cuda', trust_remote_code=True)
    model.eval()
    print("Model loaded.")

    print("Fitting 1D flip directions...")
    dirs_1d = fit_1d_dirs(model, tokenizer)
    print("Done.")

    results = []

    for i, prob in enumerate(tqdm(PROBLEMS, desc="Problems")):
        en_prompt = prob['en']
        zh_prompt = prob['zh']
        answer    = prob['answer']

        # Extract all 4 trajectories
        traj_en_base   = extract_trajectory(model, tokenizer, en_prompt)
        traj_zh_base   = extract_trajectory(model, tokenizer, zh_prompt)
        traj_en_flip   = extract_trajectory_with_flip(model, tokenizer, en_prompt, dirs_1d)
        traj_zh_flip   = extract_trajectory_with_flip(model, tokenizer, zh_prompt, dirs_1d)

        # Per-trajectory dynamics
        dyn_en_base  = compute_dynamics(traj_en_base)
        dyn_zh_base  = compute_dynamics(traj_zh_base)
        dyn_en_flip  = compute_dynamics(traj_en_flip)
        dyn_zh_flip  = compute_dynamics(traj_zh_flip)

        # Cross-trajectory alignment
        align_base   = compute_zh_en_alignment(traj_zh_base, traj_en_base)
        align_flip   = compute_zh_en_alignment(traj_zh_flip, traj_en_flip)

        # Does flip bring ZH closer to EN baseline? (language stripping test)
        align_zh_flip_vs_en_base = compute_zh_en_alignment(traj_zh_flip, traj_en_base)

        results.append({
            'problem_idx': i,
            'answer': answer,
            'dynamics': {
                'en_base':  dyn_en_base,
                'zh_base':  dyn_zh_base,
                'en_flip':  dyn_en_flip,
                'zh_flip':  dyn_zh_flip,
            },
            'alignment': {
                'zh_vs_en_base': align_base,
                'zh_flip_vs_en_flip': align_flip,
                'zh_flip_vs_en_base': align_zh_flip_vs_en_base,
            },
        })

    # ── Aggregate analysis ──
    print("\n" + "="*70)
    print("EXP AF: TRAJECTORY DYNAMICS")
    print("="*70)

    # Average velocity norms across all problems
    vel_en = np.mean([r['dynamics']['en_base']['vel_norms'] for r in results], axis=0)
    vel_zh = np.mean([r['dynamics']['zh_base']['vel_norms'] for r in results], axis=0)
    vel_en_flip = np.mean([r['dynamics']['en_flip']['vel_norms'] for r in results], axis=0)
    vel_zh_flip = np.mean([r['dynamics']['zh_flip']['vel_norms'] for r in results], axis=0)

    print("\n--- Speed profile (velocity norm per step, avg over 10 problems) ---")
    print(f"{'Step':>5} | {'EN base':>8} {'ZH base':>8} | {'EN flip':>8} {'ZH flip':>8} | {'ZH-EN base diff':>15}")
    print("-"*65)
    for step in range(36):
        diff = vel_zh[step] - vel_en[step]
        marker = "  ← PHASE" if abs(diff) > 5 or (step > 0 and abs(vel_en[step] - vel_en[step-1]) > 5) else ""
        print(f"{step:>5} | {vel_en[step]:>8.2f} {vel_zh[step]:>8.2f} | "
              f"{vel_en_flip[step]:>8.2f} {vel_zh_flip[step]:>8.2f} | "
              f"{diff:>+15.2f}{marker}")

    # Z hypothesis test: are ZH-EN offset vectors stable?
    cos_offsets_base = np.mean([r['alignment']['zh_vs_en_base']['cos_offset_consecutive'] for r in results], axis=0)
    cos_offsets_flip = np.mean([r['alignment']['zh_flip_vs_en_base']['cos_offset_consecutive'] for r in results], axis=0)

    print("\n--- Z Hypothesis: cos(offset_l, offset_{l+1}) — stable offset = parallel trajectories ---")
    print(f"{'Step':>5} | {'base ZH vs EN':>15} | {'flip ZH vs EN base':>18}")
    print("-"*45)
    for step in range(min(35, len(cos_offsets_base))):
        print(f"{step:>5} | {cos_offsets_base[step]:>+15.3f} | {cos_offsets_flip[step]:>+18.3f}")

    # Norm profile (Lyapunov candidate)
    norms_en = np.mean([r['dynamics']['en_base']['norms'] for r in results], axis=0)
    norms_zh = np.mean([r['dynamics']['zh_base']['norms'] for r in results], axis=0)

    print("\n--- Norm profile (Lyapunov candidate: should increase monotonically) ---")
    print(f"{'Pos':>5} | {'EN norm':>10} {'ZH norm':>10} | monotone EN? ZH?")
    print("-"*55)
    mono_violations_en = 0
    mono_violations_zh = 0
    for pos in range(37):
        v_en = norms_en[pos - 1] if pos > 0 else 0
        v_zh = norms_zh[pos - 1] if pos > 0 else 0
        flag_en = "↓" if pos > 0 and norms_en[pos] < v_en else " "
        flag_zh = "↓" if pos > 0 and norms_zh[pos] < v_zh else " "
        if flag_en == "↓": mono_violations_en += 1
        if flag_zh == "↓": mono_violations_zh += 1
        print(f"{pos:>5} | {norms_en[pos]:>10.2f} {norms_zh[pos]:>10.2f} | {flag_en}          {flag_zh}")
    print(f"\nMonotonicity violations: EN={mono_violations_en}/36, ZH={mono_violations_zh}/36")
    print("(0 = perfect Lyapunov; >0 = not a Lyapunov function)")

    # Does the flip rotate the ZH trajectory toward EN baseline?
    cos_direct_base = np.mean([r['alignment']['zh_vs_en_base']['cos_direct'] for r in results], axis=0)
    cos_direct_flip = np.mean([r['alignment']['zh_flip_vs_en_base']['cos_direct'] for r in results], axis=0)

    print("\n--- Gauge symmetry test: does flip bring ZH trajectory toward EN baseline? ---")
    print(f"{'Pos':>5} | {'cos(ZH, EN_base)':>18} | {'cos(ZH_flip, EN_base)':>22} | delta")
    print("-"*65)
    for pos in range(37):
        delta = cos_direct_flip[pos] - cos_direct_base[pos]
        marker = "  ← FLIP HELPS" if delta > 0.01 else ("  ← FLIP HURTS" if delta < -0.01 else "")
        print(f"{pos:>5} | {cos_direct_base[pos]:>+18.4f} | {cos_direct_flip[pos]:>+22.4f} | {delta:>+6.4f}{marker}")

    # Save
    out_path = OUTPUT_DIR / "expAF_trajectory_dynamics.json"
    with open(out_path, 'w') as f:
        json.dump({
            'results': results,
            'aggregates': {
                'vel_en_base':  vel_en.tolist(),
                'vel_zh_base':  vel_zh.tolist(),
                'vel_en_flip':  vel_en_flip.tolist(),
                'vel_zh_flip':  vel_zh_flip.tolist(),
                'norms_en':     norms_en.tolist(),
                'norms_zh':     norms_zh.tolist(),
                'cos_offset_consecutive_base': cos_offsets_base.tolist(),
                'cos_offset_consecutive_flip': cos_offsets_flip.tolist(),
                'cos_zh_vs_en_base':       cos_direct_base.tolist(),
                'cos_zh_flip_vs_en_base':  cos_direct_flip.tolist(),
            },
            'method': 'residual_stream_trajectory_dynamics',
            'note': (
                'Velocity = diff(h_l, h_{l-1}). '
                'Speed = ||velocity||. '
                'Offset = ZH - EN at each layer. '
                'Z-hypothesis: offset should be stable (cos_offset ≈ 1.0). '
                'Lyapunov: norms should be monotonically increasing. '
                'Gauge: flip should bring ZH closer to EN baseline.'
            ),
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
