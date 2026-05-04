"""
Synthetic Z Extraction POC — Config B-blind with epsilon sweep
Spec: FriMarch7-Z_POC_Spec_v2.md

Design:
  A_ell = R_ell + eps_ell * E_ell

  where R_ell is orthogonal (from Config B), E_ell is random Gaussian,
  and eps_ell = eps_base / alpha_ell (wrapper thickness inversely proportional
  to training frequency).

  At eps_base=0: recovers Config B exactly (pure rotations, Procrustes exact).
  As eps_base increases: transforms become non-orthogonal, Procrustes approximate.
  The eps_base where Procrustes R^2 drops below 0.5 = "Procrustes breaking point."

  Condition number kappa(A_ell) measures wrapper thickness per language.
  High-resource (Chinese, 60%) gets thin wrapper (low kappa).
  Low-resource (Swahili, 1%) gets thick wrapper (high kappa).

Key additions vs Config B:
  - No one-hot (blind)
  - Per-language centroids: x_ell = mu_ell + z, ||mu_ell|| = 2.0
    Models real LLM token-region separation (geometric language ID)
  - Non-orthogonal transforms with frequency-dependent messiness
  - Affine alignment comparison (Procrustes + diagonal scaling)
  - Procrustes residual tracking per language per layer
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
import json
import sys

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

D = 10
FIXED_PERM = torch.randperm(D)

K = 7
LANG_NAMES = ["Chinese", "English", "Spanish", "Arabic", "Japanese", "Korean", "Swahili"]
LANG_FREQS = [0.60, 0.20, 0.08, 0.05, 0.04, 0.02, 0.01]

# Epsilon from command line or default
EPS_BASE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3

# Per-language epsilon: inversely proportional to frequency
# Chinese (60%) -> eps * 1/0.6 = 1.67x eps
# Swahili (1%) -> eps * 1/0.01 = 100x eps
# Cap the multiplier to avoid explosion
EPS_CAP = 20.0  # max multiplier over eps_base
eps_per_lang = [min(EPS_BASE / freq, EPS_BASE * EPS_CAP) for freq in LANG_FREQS]

print(f"=== Config B-blind sweep: eps_base = {EPS_BASE} ===")
print(f"Per-language epsilon (capped at {EPS_CAP}x):")
for ell in range(K):
    print(f"  {LANG_NAMES[ell]:>10s}: eps={eps_per_lang[ell]:.3f} (freq={LANG_FREQS[ell]:.2f})")


# =============================================================================
# Section 1a: Reasoning function (identical to Config B)
# =============================================================================
def reasoning_function(x: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    s = x[:, FIXED_PERM]
    c = torch.cumsum(s, dim=1)
    n = torch.tanh(0.5 * c)
    r = n.flip(dims=[1])
    out = r / r.norm(dim=1, keepdim=True)
    if squeeze:
        out = out.squeeze(0)
    return out


# =============================================================================
# Section 1b: Language transforms — R_ell + eps_ell * E_ell
# =============================================================================
# First generate the SAME orthogonal rotations as Config B (same seed)
rotations = []
for ell in range(K):
    A = torch.randn(D, D)
    Q, R_sign = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R_sign)).unsqueeze(0)
    Q, _ = torch.linalg.qr(Q)
    rotations.append(Q)

# Now perturb: A_ell = R_ell + eps_ell * E_ell
perturbations = []
transforms = []
for ell in range(K):
    E = torch.randn(D, D)
    E = E / E.norm()  # normalize perturbation to unit Frobenius norm
    perturbations.append(E)
    A = rotations[ell] + eps_per_lang[ell] * E
    transforms.append(A)

# =============================================================================
# Section 1b2: Per-language centroids — models token-region separation in real LLMs
# =============================================================================
# x_ell = mu_ell + z, z ~ N(0,I), then transformed by A_ell
# mu_ell are random unit vectors scaled to CENTROID_NORM
CENTROID_NORM = 2.0
centroids = []
for ell in range(K):
    mu = torch.randn(D)
    mu = mu / mu.norm() * CENTROID_NORM
    centroids.append(mu)

# Verify pairwise separation
print(f"\nCentroid properties (||mu||={CENTROID_NORM}):")
for ell in range(K):
    dists = [torch.dist(centroids[ell], centroids[j]).item() for j in range(K) if j != ell]
    print(f"  {LANG_NAMES[ell]:>10s}: min_dist={min(dists):.2f}, mean_dist={np.mean(dists):.2f}")

print(f"\nTransform properties:")
for ell in range(K):
    A = transforms[ell]
    cond = torch.linalg.cond(A).item()
    det = torch.linalg.det(A).item()
    # Distance from orthogonal: ||A^T A - I||_F
    orth_err = (A.T @ A - torch.eye(D)).norm().item()
    print(f"  {LANG_NAMES[ell]:>10s}: cond={cond:.2f}, det={det:.3f}, orth_err={orth_err:.3f}")


# =============================================================================
# Section 1c: Data generation (no one-hot)
# =============================================================================
N_TRAIN = 200_000
N_TEST = 200


def generate_data(n_samples, freqs, transforms, centroids, split="train"):
    raw_x = torch.randn(n_samples, D)
    raw_fx = reasoning_function(raw_x)

    if split == "test":
        all_inputs, all_targets = [], []
        all_lang_ids, all_problem_ids = [], []
        for ell in range(K):
            A = transforms[ell]
            # Add centroid BEFORE transform: x_ell = A_ell @ (mu_ell + z)
            x_shifted = raw_x + centroids[ell].unsqueeze(0)
            x_rot = x_shifted @ A.T
            y_rot = raw_fx @ A.T
            all_inputs.append(x_rot)
            all_targets.append(y_rot)
            all_lang_ids.extend([ell] * n_samples)
            all_problem_ids.extend(range(n_samples))
        return (torch.cat(all_inputs), torch.cat(all_targets),
                {"raw_x": raw_x, "raw_fx": raw_fx,
                 "lang_ids": all_lang_ids, "problem_ids": all_problem_ids})
    else:
        lang_ids = np.concatenate([
            np.full(int(freq * n_samples), ell, dtype=np.int64)
            for ell, freq in enumerate(freqs)
        ])
        if len(lang_ids) < n_samples:
            lang_ids = np.concatenate([lang_ids, np.zeros(n_samples - len(lang_ids), dtype=np.int64)])
        lang_ids = lang_ids[:n_samples]
        np.random.shuffle(lang_ids)

        A_all = torch.stack(transforms)
        mu_all = torch.stack(centroids)  # (K, D)
        lang_t = torch.from_numpy(lang_ids)
        A_per = A_all[lang_t]
        mu_per = mu_all[lang_t]  # (n_samples, D)
        # Add per-sample centroid before transform
        x_shifted = raw_x + mu_per
        x_rot = torch.bmm(x_shifted.unsqueeze(1), A_per.transpose(1, 2)).squeeze(1)
        y_rot = torch.bmm(raw_fx.unsqueeze(1), A_per.transpose(1, 2)).squeeze(1)
        return (x_rot, y_rot,
                {"raw_x": raw_x, "raw_fx": raw_fx, "lang_ids": lang_ids.tolist()})


print("\nGenerating data...")
train_inputs, train_targets, train_meta = generate_data(N_TRAIN, LANG_FREQS, transforms, centroids, "train")
test_inputs, test_targets, test_meta = generate_data(N_TEST, LANG_FREQS, transforms, centroids, "test")
print(f"Train: {train_inputs.shape} -> {train_targets.shape}")
print(f"Test:  {test_inputs.shape} -> {test_targets.shape}")

from collections import Counter
lang_counts = Counter(train_meta["lang_ids"])
print("\nTraining language distribution:")
for ell in range(K):
    pct = lang_counts[ell] / N_TRAIN * 100
    print(f"  {LANG_NAMES[ell]:>10s}: {lang_counts[ell]:5d} ({pct:.1f}%)")


# =============================================================================
# Section 2: Model + Training
# =============================================================================
W = 256
H = 6


class ReasoningMLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth, dropout=0.0, residual=False):
        super().__init__()
        self.residual = residual
        # Layer 0: project up (no skip — dimension mismatch)
        self.input_proj = nn.Linear(d_in, width)
        self.input_act = nn.GELU()
        self.input_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Hidden layers 1..depth-1 (skip connections when residual=True)
        self.hidden_linears = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()
        self.hidden_drops = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_linears.append(nn.Linear(width, width))
            self.hidden_acts.append(nn.GELU())
            self.hidden_drops.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        # Output: project down (no skip — dimension mismatch)
        self.output_proj = nn.Linear(width, d_out)
        self.activations = {}

    def forward(self, x, save_activations=False):
        if save_activations:
            self.activations = {}
        # Layer 0: project up
        h = self.input_drop(self.input_act(self.input_proj(x)))
        if save_activations:
            self.activations[0] = h.detach()
        # Hidden layers with optional residual
        for i, (lin, act, drop) in enumerate(zip(self.hidden_linears, self.hidden_acts, self.hidden_drops)):
            out = drop(act(lin(h)))
            if self.residual:
                h = h + out
            else:
                h = out
            if save_activations:
                self.activations[i + 1] = h.detach()
        # Output projection
        h = self.output_proj(h)
        if save_activations:
            self.activations[len(self.hidden_linears) + 1] = h.detach()
        return h


DROPOUT = 0.1
RESIDUAL = True
model = ReasoningMLP(D, D, W, H, dropout=DROPOUT, residual=RESIDUAL)
if torch.cuda.is_available():
    model = model.cuda()
    print("Using CUDA")
print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
print(f"  Input: {D} -> Hidden: {W} x {H} layers -> Output: {D}  [residual={RESIDUAL}, dropout={DROPOUT}]")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
N_EPOCHS = 500
BATCH_SIZE = 2048
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

print(f"\nTraining for {N_EPOCHS} epochs...")
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = next(model.parameters()).device
test_inputs_dev = test_inputs.to(device)
best_test_loss = float('inf')
best_state = None
EVAL_EVERY = 10

lang_ids = test_meta["lang_ids"]
lang_masks = {ell: [i for i, l in enumerate(lang_ids) if l == ell] for ell in range(K)}

pbar = tqdm(range(N_EPOCHS), desc="Training", unit="ep")
for epoch in pbar:
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_dataset)
    scheduler.step()

    cur_lr = optimizer.param_groups[0]['lr']
    pbar.set_postfix(mse=f"{epoch_loss:.5f}", lr=f"{cur_lr:.1e}")

    if (epoch + 1) % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            tp = model(test_inputs_dev).cpu()
        test_mse = nn.functional.mse_loss(tp, test_targets).item()
        # Track best on TEST loss, not training loss
        if test_mse < best_test_loss:
            best_test_loss = test_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        parts = []
        for ell in range(K):
            m = lang_masks[ell]
            mse_ell = nn.functional.mse_loss(tp[m], test_targets[m]).item()
            tag = "OK" if mse_ell < 0.01 else ""
            parts.append(f"{LANG_NAMES[ell][:2]}={mse_ell:.4f}{tag}")
        marker = " *" if test_mse <= best_test_loss else ""
        pbar.write(f"  [Ep {epoch+1}] test={test_mse:.4f}{marker} | {' '.join(parts)}")
        model.train()

pbar.close()

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)
print(f"\nBest test MSE: {best_test_loss:.6f} (at epoch {best_epoch})")


# =============================================================================
# Section 3: Per-language convergence + activation extraction
# =============================================================================
model.eval()
with torch.no_grad():
    test_pred = model(test_inputs.to(device)).cpu()
    test_mse_all = nn.functional.mse_loss(test_pred, test_targets).item()

print(f"\nOverall test MSE: {test_mse_all:.6f}")
print("\nPer-language test MSE:")
converged = []
for ell in range(K):
    mask = lang_masks[ell]
    mse_ell = nn.functional.mse_loss(test_pred[mask], test_targets[mask]).item()
    status = "OK" if mse_ell < 0.01 else "FAIL"
    converged.append(mse_ell < 0.01)
    print(f"  {LANG_NAMES[ell]:>10s}: MSE = {mse_ell:.6f}  [{status}]")

print("\nExtracting activations...")
with torch.no_grad():
    _ = model(test_inputs.to(device), save_activations=True)

activations_by_layer = {h: act.cpu() for h, act in model.activations.items()}
lang_ids_arr = np.array(lang_ids)
act_by_lang_layer = {}
for ell in range(K):
    mask = np.where(lang_ids_arr == ell)[0]
    act_by_lang_layer[ell] = {h: act[mask] for h, act in activations_by_layer.items()}

n_layers = len(activations_by_layer)
print(f"Activations: {K} languages x {n_layers} layers x ({N_TEST}, {W})")


# =============================================================================
# Section 4: Procrustes analysis + Procrustes residual + affine comparison
# =============================================================================
print("\n" + "=" * 70)
print("PROCRUSTES ANALYSIS (align all languages to Chinese)")
print("=" * 70)

ref_lang = 0  # Chinese as reference

# For each layer, compute:
# 1. Procrustes R^2
# 2. Procrustes residual per language
# 3. Affine R^2 (Procrustes + diagonal scaling)
# 4. Cross-lingual NN accuracy (raw vs Procrustes vs affine)

results_by_layer = {}

for h in sorted(activations_by_layer.keys()):
    if h == n_layers - 1:
        continue  # skip output layer (dim D, not W)

    A_ref = act_by_lang_layer[ref_lang][h].numpy()  # (N_TEST, W)

    layer_results = {"procrustes_r2": {}, "affine_r2": {},
                     "procrustes_residual": {}, "raw_nn": {}, "proc_nn": {}, "affine_nn": {}}

    for ell in range(K):
        if ell == ref_lang:
            continue

        A_ell = act_by_lang_layer[ell][h].numpy()

        # --- Procrustes alignment ---
        R_proc, scale = orthogonal_procrustes(A_ell, A_ref)
        A_aligned = A_ell @ R_proc

        # Procrustes residual: ||A_aligned - A_ref||_F / ||A_ref||_F
        residual = np.linalg.norm(A_aligned - A_ref, 'fro') / np.linalg.norm(A_ref, 'fro')
        layer_results["procrustes_residual"][LANG_NAMES[ell]] = float(residual)

        # R^2 for Procrustes
        ss_res = np.sum((A_aligned - A_ref) ** 2)
        ss_tot = np.sum((A_ref - A_ref.mean(axis=0)) ** 2)
        r2_proc = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        layer_results["procrustes_r2"][LANG_NAMES[ell]] = float(r2_proc)

        # --- Affine alignment: Procrustes R then per-dimension scaling ---
        # Fit diagonal D such that A_aligned @ diag(D) ≈ A_ref
        # This is just per-column least squares: d_j = (A_aligned[:,j]^T A_ref[:,j]) / (A_aligned[:,j]^T A_aligned[:,j])
        d_scale = np.sum(A_aligned * A_ref, axis=0) / (np.sum(A_aligned ** 2, axis=0) + 1e-10)
        A_affine = A_aligned * d_scale[np.newaxis, :]

        ss_res_aff = np.sum((A_affine - A_ref) ** 2)
        r2_aff = 1 - ss_res_aff / ss_tot if ss_tot > 0 else 0
        layer_results["affine_r2"][LANG_NAMES[ell]] = float(r2_aff)

        # --- Cross-lingual NN accuracy ---
        # Raw: for each problem m, is nearest neighbor of A_ell[m] in A_ref the same problem?
        from scipy.spatial.distance import cdist
        D_raw = cdist(A_ell, A_ref, metric='euclidean')
        nn_raw = np.mean(np.argmin(D_raw, axis=1) == np.arange(N_TEST))
        layer_results["raw_nn"][LANG_NAMES[ell]] = float(nn_raw)

        D_proc = cdist(A_aligned, A_ref, metric='euclidean')
        nn_proc = np.mean(np.argmin(D_proc, axis=1) == np.arange(N_TEST))
        layer_results["proc_nn"][LANG_NAMES[ell]] = float(nn_proc)

        D_aff = cdist(A_affine, A_ref, metric='euclidean')
        nn_aff = np.mean(np.argmin(D_aff, axis=1) == np.arange(N_TEST))
        layer_results["affine_nn"][LANG_NAMES[ell]] = float(nn_aff)

    results_by_layer[h] = layer_results

    # Print summary for this layer
    print(f"\n--- Layer {h} ---")
    print(f"  {'Language':>10s}  {'ProcR2':>8s}  {'AffR2':>8s}  {'Resid':>8s}  {'RawNN':>8s}  {'ProcNN':>8s}  {'AffNN':>8s}")
    for ell in range(K):
        if ell == ref_lang:
            continue
        name = LANG_NAMES[ell]
        r2p = layer_results["procrustes_r2"][name]
        r2a = layer_results["affine_r2"][name]
        res = layer_results["procrustes_residual"][name]
        nn_r = layer_results["raw_nn"][name]
        nn_p = layer_results["proc_nn"][name]
        nn_a = layer_results["affine_nn"][name]
        print(f"  {name:>10s}  {r2p:8.3f}  {r2a:8.3f}  {res:8.3f}  {nn_r:8.3f}  {nn_p:8.3f}  {nn_a:8.3f}")


# =============================================================================
# Section 5: Summary + save
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Find best Procrustes layer (highest mean R^2 across languages)
best_layer = None
best_mean_r2 = -999
for h, res in results_by_layer.items():
    mean_r2 = np.mean(list(res["procrustes_r2"].values()))
    if mean_r2 > best_mean_r2:
        best_mean_r2 = mean_r2
        best_layer = h

print(f"Best Procrustes layer: {best_layer} (mean R^2 = {best_mean_r2:.3f})")

# At best layer, compare Procrustes vs affine
if best_layer is not None:
    res = results_by_layer[best_layer]
    mean_proc = np.mean(list(res["procrustes_r2"].values()))
    mean_aff = np.mean(list(res["affine_r2"].values()))
    mean_proc_nn = np.mean(list(res["proc_nn"].values()))
    mean_aff_nn = np.mean(list(res["affine_nn"].values()))
    mean_raw_nn = np.mean(list(res["raw_nn"].values()))

    print(f"\nAt best layer {best_layer}:")
    print(f"  Procrustes R^2:  {mean_proc:.3f}")
    print(f"  Affine R^2:      {mean_aff:.3f}")
    print(f"  Delta (affine - procrustes): {mean_aff - mean_proc:.3f}")
    print(f"  Raw NN:          {mean_raw_nn:.3f}")
    print(f"  Procrustes NN:   {mean_proc_nn:.3f}")
    print(f"  Affine NN:       {mean_aff_nn:.3f}")

    if mean_aff - mean_proc > 0.05:
        print("  -> Affine substantially better: non-orthogonal component matters!")
    elif mean_aff - mean_proc > 0.01:
        print("  -> Affine slightly better: small non-orthogonal component.")
    else:
        print("  -> Affine ≈ Procrustes: transforms are approximately orthogonal.")

# Per-language correlation with frequency
print(f"\nPer-language at layer {best_layer} (sorted by frequency):")
print(f"  {'Language':>10s}  {'Freq':>6s}  {'Eps':>6s}  {'Cond':>6s}  {'ProcR2':>8s}  {'AffR2':>8s}  {'ProcNN':>8s}")
for ell in range(K):
    if ell == ref_lang:
        continue
    name = LANG_NAMES[ell]
    freq = LANG_FREQS[ell]
    eps = eps_per_lang[ell]
    cond = torch.linalg.cond(transforms[ell]).item()
    r2p = results_by_layer[best_layer]["procrustes_r2"][name]
    r2a = results_by_layer[best_layer]["affine_r2"][name]
    nn_p = results_by_layer[best_layer]["proc_nn"][name]
    print(f"  {name:>10s}  {freq:6.2f}  {eps:6.2f}  {cond:6.2f}  {r2p:8.3f}  {r2a:8.3f}  {nn_p:8.3f}")

# Save results
save_path = f"output/z_poc_blind_sweep_eps{EPS_BASE:.2f}.json"
save_data = {
    "residual": RESIDUAL,
    "dropout": DROPOUT,
    "eps_base": EPS_BASE,
    "eps_per_lang": {LANG_NAMES[ell]: eps_per_lang[ell] for ell in range(K)},
    "transform_conds": {LANG_NAMES[ell]: torch.linalg.cond(transforms[ell]).item() for ell in range(K)},
    "best_test_mse": float(best_test_loss),
    "best_epoch": best_epoch,
    "test_mse": float(test_mse_all),
    "per_lang_converged": {LANG_NAMES[ell]: converged[ell] for ell in range(K)},
    "best_procrustes_layer": best_layer,
    "results_by_layer": {str(h): res for h, res in results_by_layer.items()},
}
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to {save_path}")

# Also save full model + activations for notebook analysis
torch_save_path = f"output/z_poc_blind_sweep_eps{EPS_BASE:.2f}_trained.pt"
torch.save({
    "model_state": model.cpu().state_dict(),
    "model_config": {"d_in": D, "d_out": D, "width": W, "depth": H},
    "transforms": transforms,
    "rotations": rotations,
    "perturbations": perturbations,
    "centroids": centroids,
    "centroid_norm": CENTROID_NORM,
    "eps_base": EPS_BASE,
    "eps_per_lang": eps_per_lang,
    "fixed_perm": FIXED_PERM,
    "test_inputs": test_inputs,
    "test_targets": test_targets,
    "test_meta": test_meta,
    "act_by_lang_layer": {ell: {h: a.cpu() for h, a in layers.items()}
                          for ell, layers in act_by_lang_layer.items()},
    "lang_names": LANG_NAMES,
    "lang_freqs": LANG_FREQS,
    "K": K, "D": D, "W": W, "H": H, "N_TEST": N_TEST,
    "config": f"B-blind-centroid-sweep-eps{EPS_BASE}",
}, torch_save_path)
print(f"Model + activations saved to {torch_save_path}")
