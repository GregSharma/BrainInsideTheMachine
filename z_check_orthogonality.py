"""
Check the orthogonality assumption that Props 2-5 depend on.

If layer contributions f_k(h_k) are mutually orthogonal, then:
  ||Δ_L||² = ||Δ_0||² + Σ ||δ_k||²
and SNR monotonicity follows.

If they're NOT orthogonal, cross-terms appear and the proof breaks.

This script loads the trained residual model, computes per-layer deltas
f_k = h_k - h_{k-1} on the test set, and reports:
1. The cosine similarity Gram matrix G[j,k] between all layer pairs
2. The fraction of variance explained by cross-terms
3. Whether ||Δ_L||² ≈ Σ ||f_k||² (Pythagorean test)
"""

import torch
import torch.nn as nn
import numpy as np
import json

# Load saved model + data
saved = torch.load("output/z_poc_blind_sweep_eps0.10_trained.pt", map_location="cpu", weights_only=False)

with open("output/z_poc_blind_sweep_eps0.10.json") as f:
    results = json.load(f)

# Rebuild model
import sys
sys.path.insert(0, ".")

D = 10
W = 256
H = 6

class ReasoningMLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth, dropout=0.0, residual=False):
        super().__init__()
        self.residual = residual
        self.input_proj = nn.Linear(d_in, width)
        self.input_act = nn.GELU()
        self.input_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.hidden_linears = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()
        self.hidden_drops = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_linears.append(nn.Linear(width, width))
            self.hidden_acts.append(nn.GELU())
            self.hidden_drops.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.output_proj = nn.Linear(width, d_out)
        self.activations = {}

    def forward(self, x, save_activations=False):
        if save_activations:
            self.activations = {}
        h = self.input_drop(self.input_act(self.input_proj(x)))
        if save_activations:
            self.activations[0] = h.detach()
        for i, (lin, act, drop) in enumerate(zip(self.hidden_linears, self.hidden_acts, self.hidden_drops)):
            out = drop(act(lin(h)))
            if self.residual:
                h = h + out
            else:
                h = out
            if save_activations:
                self.activations[i + 1] = h.detach()
        h = self.output_proj(h)
        if save_activations:
            self.activations[len(self.hidden_linears) + 1] = h.detach()
        return h

model = ReasoningMLP(D, D, W, H, dropout=0.0, residual=True)  # dropout=0 for eval
model.load_state_dict(saved["model_state"])
model.eval()

# Get test data
test_inputs = saved["test_inputs"]
test_meta = saved["test_meta"]
lang_ids = test_meta["lang_ids"]

print(f"Test samples: {len(test_inputs)}")
print(f"Model: {sum(p.numel() for p in model.parameters())} params, residual=True")

# Forward pass, collect activations
with torch.no_grad():
    _ = model(test_inputs, save_activations=True)

# Extract activations: h_0, h_1, ..., h_5 (6 hidden layers)
# h_0 = after input projection
# h_k = h_{k-1} + f_k(h_{k-1}) for k >= 1
activations = {k: model.activations[k].numpy() for k in sorted(model.activations.keys())}
n_hidden = len(activations) - 1  # exclude the output layer activation
print(f"Hidden layer activations: {n_hidden} layers, shape {activations[0].shape}")

# Compute deltas: f_k = h_k - h_{k-1}
# For residual model: f_0 = h_0 (the input projection itself)
# f_k = h_k - h_{k-1} for k = 1, ..., n_hidden-1
deltas = {}
deltas[0] = activations[0]  # f_0 = h_0 (projection from input)
for k in range(1, n_hidden):
    deltas[k] = activations[k] - activations[k-1]  # f_k = h_k - h_{k-1}

print(f"\nDeltas computed: {len(deltas)} layers")
for k in sorted(deltas.keys()):
    norm = np.linalg.norm(deltas[k], axis=1).mean()
    print(f"  f_{k}: mean ||f_{k}|| = {norm:.4f}")

# ============================================================
# TEST 1: Cosine similarity Gram matrix
# ============================================================
print("\n" + "="*60)
print("TEST 1: Cosine similarity Gram matrix G[j,k]")
print("="*60)

# Flatten each delta to a single vector (concatenate all samples)
# This measures whether the DIRECTIONS of layer contributions are orthogonal
# across the population
n_layers = len(deltas)
delta_flat = {}
for k in sorted(deltas.keys()):
    delta_flat[k] = deltas[k].flatten()

G = np.zeros((n_layers, n_layers))
for j in range(n_layers):
    for k in range(n_layers):
        fj = delta_flat[j]
        fk = delta_flat[k]
        cos = np.dot(fj, fk) / (np.linalg.norm(fj) * np.linalg.norm(fk) + 1e-12)
        G[j, k] = cos

print("\nGram matrix (cosine similarity between flattened layer deltas):")
print("       ", "  ".join(f"  f_{k}" for k in range(n_layers)))
for j in range(n_layers):
    row = "  ".join(f"{G[j,k]:6.3f}" for k in range(n_layers))
    print(f"  f_{j}:  {row}")

off_diag = []
for j in range(n_layers):
    for k in range(j+1, n_layers):
        off_diag.append(abs(G[j, k]))

print(f"\nOff-diagonal |cos|: mean={np.mean(off_diag):.4f}, max={np.max(off_diag):.4f}")
if np.mean(off_diag) < 0.1:
    print("VERDICT: Approximately orthogonal. Prop 2 assumption holds.")
elif np.mean(off_diag) < 0.3:
    print("VERDICT: Weakly correlated. Prop 2 approximately holds but cross-terms non-negligible.")
else:
    print("VERDICT: Substantially correlated. Prop 2 orthogonality assumption FAILS.")

# ============================================================
# TEST 2: Per-sample Pythagorean test
# ============================================================
print("\n" + "="*60)
print("TEST 2: Pythagorean test (per-sample)")
print("="*60)
print("If orthogonal: ||h_L||² = ||f_0||² + ||f_1||² + ... + ||f_{L-1}||²")

# h_L = f_0 + f_1 + ... + f_{L-1} (residual accumulation)
# ||h_L||² vs Σ||f_k||²
h_L = activations[n_hidden - 1]  # last hidden layer
h_L_norm_sq = np.sum(h_L**2, axis=1)  # per sample

sum_fk_norm_sq = np.zeros(len(test_inputs))
for k in range(n_layers):
    sum_fk_norm_sq += np.sum(deltas[k]**2, axis=1)

# The cross-term contribution
cross_terms = h_L_norm_sq - sum_fk_norm_sq

ratio = cross_terms / h_L_norm_sq

print(f"  ||h_L||² mean:        {h_L_norm_sq.mean():.4f}")
print(f"  Σ||f_k||² mean:       {sum_fk_norm_sq.mean():.4f}")
print(f"  Cross-terms mean:     {cross_terms.mean():.4f}")
print(f"  Cross/||h_L||² ratio: {ratio.mean():.4f} ± {ratio.std():.4f}")

if abs(ratio.mean()) < 0.1:
    print("VERDICT: Cross-terms < 10% of total norm. Pythagorean approximately holds.")
elif abs(ratio.mean()) < 0.3:
    print("VERDICT: Cross-terms 10-30% of total norm. Moderate violation.")
else:
    print("VERDICT: Cross-terms > 30% of total norm. Pythagorean FAILS.")

# ============================================================
# TEST 3: Per-sample cross-lingual Δ decomposition
# ============================================================
print("\n" + "="*60)
print("TEST 3: Cross-lingual Δ_L decomposition")
print("="*60)
print("For same problem, diff language: Δ_L = Σ δ_k")
print("If orthogonal: ||Δ_L||² = Σ ||δ_k||²")

# Get Chinese (lang 0) and English (lang 1) samples for same problems
LANG_NAMES = ["Chinese", "English", "Spanish", "Arabic", "Japanese", "Korean", "Swahili"]
lang_masks = {}
for ell in range(7):
    lang_masks[ell] = [i for i, l in enumerate(lang_ids) if l == ell]

# For problem-matched pairs, we need problem IDs
problem_ids = test_meta["problem_ids"]

# Find matched pairs: same problem, Chinese vs English
zh_by_prob = {}
en_by_prob = {}
for i, (lid, pid) in enumerate(zip(lang_ids, problem_ids)):
    if lid == 0:
        zh_by_prob[pid] = i
    elif lid == 1:
        en_by_prob[pid] = i

matched_probs = set(zh_by_prob.keys()) & set(en_by_prob.keys())
print(f"  Matched zh-en problem pairs: {len(matched_probs)}")

if len(matched_probs) > 0:
    # Compute cross-lingual deltas per layer
    zh_indices = [zh_by_prob[p] for p in sorted(matched_probs)]
    en_indices = [en_by_prob[p] for p in sorted(matched_probs)]

    cross_deltas = {}
    for k in range(n_layers):
        cross_deltas[k] = deltas[k][zh_indices] - deltas[k][en_indices]

    # Δ_L = h_L(zh) - h_L(en)
    Delta_L = h_L[zh_indices] - h_L[en_indices]
    Delta_L_norm_sq = np.sum(Delta_L**2, axis=1)

    sum_cross_delta_norm_sq = np.zeros(len(zh_indices))
    for k in range(n_layers):
        sum_cross_delta_norm_sq += np.sum(cross_deltas[k]**2, axis=1)

    cross_terms_cl = Delta_L_norm_sq - sum_cross_delta_norm_sq
    ratio_cl = cross_terms_cl / (Delta_L_norm_sq + 1e-12)

    print(f"  ||Δ_L||² mean:          {Delta_L_norm_sq.mean():.4f}")
    print(f"  Σ||δ_k||² mean:         {sum_cross_delta_norm_sq.mean():.4f}")
    print(f"  Cross-terms mean:       {cross_terms_cl.mean():.4f}")
    print(f"  Cross/||Δ_L||² ratio:   {ratio_cl.mean():.4f} ± {ratio_cl.std():.4f}")

    if abs(ratio_cl.mean()) < 0.1:
        print("VERDICT: Cross-lingual Pythagorean holds. Prop 2 condition validated.")
    elif abs(ratio_cl.mean()) < 0.3:
        print("VERDICT: Moderate cross-terms. Prop 2 approximately holds for cross-lingual case.")
    else:
        print("VERDICT: Large cross-terms. Prop 2 cross-lingual decomposition FAILS.")

    # Per-layer cross-lingual delta norms (the δ_k values)
    print(f"\n  Per-layer cross-lingual ||δ_k|| (zh vs en):")
    for k in range(n_layers):
        norm_k = np.linalg.norm(cross_deltas[k], axis=1).mean()
        fk_norm = np.linalg.norm(deltas[k][zh_indices], axis=1).mean()
        ratio_k = norm_k / (fk_norm + 1e-12)
        print(f"    δ_{k}: ||δ_{k}||={norm_k:.4f}  ||f_{k}||={fk_norm:.4f}  ratio={ratio_k:.4f}")

# ============================================================
# TEST 4: Does SNR actually increase monotonically?
# ============================================================
print("\n" + "="*60)
print("TEST 4: Empirical SNR through depth")
print("="*60)

if len(matched_probs) > 0:
    # D_same(L) = E[||h_L^zh(x) - h_L^en(x)||²] for matched problems
    # D_diff(L) = E[||h_L^zh(x) - h_L^zh(x')||²] for different problems
    for L in range(n_hidden):
        h_zh = activations[L][zh_indices]
        h_en = activations[L][en_indices]

        # D_same: same problem, diff language
        d_same = np.mean(np.sum((h_zh - h_en)**2, axis=1))

        # D_diff: diff problem, same language (use Chinese)
        # Sample random pairs
        n_pairs = min(1000, len(zh_indices))
        idx1 = np.random.choice(len(zh_indices), n_pairs, replace=True)
        idx2 = np.random.choice(len(zh_indices), n_pairs, replace=True)
        # Ensure different problems
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        d_diff = np.mean(np.sum((h_zh[idx1] - h_zh[idx2])**2, axis=1))

        snr = d_same / (d_diff + 1e-12)
        print(f"  Layer {L}: D_same={d_same:.4f}  D_diff={d_diff:.4f}  SNR(=D_same/D_diff)={snr:.4f}")

    print("\n  SNR should DECREASE if Prop 2 holds (lower = better alignment).")
    print("  Monotonic decrease = Prop 2 validated empirically.")
