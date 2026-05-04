"""
Synthetic Z Extraction POC — Config B (imbalanced random + one-hot)
Spec: FriMarch7-Z_POC_Spec_v2.md

Section 1: Reasoning function + data generation
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Section 1a: The reasoning function f
# =============================================================================
# f: R^d -> R^d
# sort by |x| -> cumulative sum -> tanh(0.5 * .) -> reverse -> normalize
#
# Why this function: requires (i) global comparison (sorting),
# (ii) sequential accumulation, (iii) nonlinear compression,
# (iv) structural rearrangement, (v) scale-invariant encoding.
# A single linear layer cannot compute it.

D = 10  # input/output dimension

# Fixed random permutation replaces data-dependent sort.
# Sort is provably hard for MLPs (requires O(d^2) pairwise comparisons).
# A fixed permutation preserves the compositional structure (cumsum -> tanh -> reverse -> normalize)
# while being learnable. The function is still nonlinear and non-trivial (linear baseline MSE ~0.022).
FIXED_PERM = torch.randperm(D)


def reasoning_function(x: torch.Tensor) -> torch.Tensor:
    """
    x: (batch, D) or (D,)
    returns: (batch, D) or (D,)

    Step 1: fixed permutation (replaces data-dependent sort — see note above)
    Step 2: cumulative sum
    Step 3: tanh(0.5 * .)
    Step 4: reverse
    Step 5: L2 normalize
    """
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True

    # Step 1: fixed permutation
    s = x[:, FIXED_PERM]

    # Step 2: cumulative sum along the permuted dimension
    c = torch.cumsum(s, dim=1)

    # Step 3: compressed tanh
    n = torch.tanh(0.5 * c)

    # Step 4: reverse
    r = n.flip(dims=[1])

    # Step 5: L2 normalize to unit sphere
    out = r / r.norm(dim=1, keepdim=True)

    if squeeze:
        out = out.squeeze(0)
    return out


# =============================================================================
# Section 1b: Language rotations
# =============================================================================
# K=7 languages, each gets a fixed random orthogonal matrix R_ell.
# Observed data: x_tilde = R_ell @ x, y_tilde = R_ell @ f(x)
# The network never sees the canonical x or f(x).

K = 7  # number of languages
LANG_NAMES = ["Chinese", "English", "Spanish", "Arabic", "Japanese", "Korean", "Swahili"]
LANG_FREQS = [0.60, 0.20, 0.08, 0.05, 0.04, 0.02, 0.01]  # training frequencies

# Generate K random orthogonal matrices via QR decomposition of random Gaussian
rotations = []
for ell in range(K):
    A = torch.randn(D, D)
    Q, R_sign = torch.linalg.qr(A)
    # Ensure proper rotation (det = +1), not reflection
    Q = Q * torch.sign(torch.diag(R_sign)).unsqueeze(0)
    # Re-orthogonalize to be safe
    Q, _ = torch.linalg.qr(Q)
    rotations.append(Q)

# Verify orthogonality
for ell, R in enumerate(rotations):
    err = (R @ R.T - torch.eye(D)).norm().item()
    assert err < 1e-6, f"Rotation {ell} not orthogonal: err={err}"


# =============================================================================
# Section 1c: Data generation
# =============================================================================

N_TRAIN = 50_000   # total training samples
N_TEST = 200       # test samples (same problems, all languages)


def generate_data(n_samples, freqs, rotations, split="train"):
    """
    Generate training or test data.

    Training: each sample assigned to one language based on freqs.
    Test: each raw problem appears in ALL languages (for paired comparisons).

    Returns:
        inputs: (N, D+K) — rotated x concatenated with one-hot language
        targets: (N, D) — rotated f(x)
        meta: dict with raw_x, lang_ids, raw_fx for analysis
    """
    # Raw problems: x ~ N(0, I)
    raw_x = torch.randn(n_samples, D)
    raw_fx = reasoning_function(raw_x)

    if split == "test":
        # Every problem in every language -> N_TEST * K samples
        all_inputs = []
        all_targets = []
        all_lang_ids = []
        all_problem_ids = []

        for ell in range(K):
            R = rotations[ell]
            x_rot = raw_x @ R.T       # (N, D) @ (D, D) = (N, D)  — rotate each row
            y_rot = raw_fx @ R.T

            # One-hot language indicator
            onehot = torch.zeros(n_samples, K)
            onehot[:, ell] = 1.0

            inp = torch.cat([x_rot, onehot], dim=1)  # (N, D+K)
            all_inputs.append(inp)
            all_targets.append(y_rot)
            all_lang_ids.extend([ell] * n_samples)
            all_problem_ids.extend(range(n_samples))

        inputs = torch.cat(all_inputs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        meta = {
            "raw_x": raw_x,
            "raw_fx": raw_fx,
            "lang_ids": all_lang_ids,
            "problem_ids": all_problem_ids,
        }
        return inputs, targets, meta

    else:  # train
        # Assign each sample to a language based on freqs — vectorized
        lang_ids = np.concatenate([
            np.full(int(freq * n_samples), ell, dtype=np.int64)
            for ell, freq in enumerate(freqs)
        ])
        # Fill remainder with language 0
        if len(lang_ids) < n_samples:
            lang_ids = np.concatenate([lang_ids, np.zeros(n_samples - len(lang_ids), dtype=np.int64)])
        lang_ids = lang_ids[:n_samples]
        np.random.shuffle(lang_ids)

        # Stack all rotation matrices: (K, D, D)
        R_all = torch.stack(rotations)
        # Gather per-sample rotation: (N, D, D)
        lang_t = torch.from_numpy(lang_ids)
        R_per_sample = R_all[lang_t]  # (N, D, D)

        # Batch rotate: x_rot[i] = raw_x[i] @ R[lang[i]].T
        x_rot = torch.bmm(raw_x.unsqueeze(1), R_per_sample.transpose(1, 2)).squeeze(1)
        y_rot = torch.bmm(raw_fx.unsqueeze(1), R_per_sample.transpose(1, 2)).squeeze(1)

        # One-hot language indicators: (N, K)
        onehot = torch.zeros(n_samples, K)
        onehot[torch.arange(n_samples), lang_t] = 1.0

        inputs = torch.cat([x_rot, onehot], dim=1)
        targets = y_rot
        meta = {
            "raw_x": raw_x,
            "raw_fx": raw_fx,
            "lang_ids": lang_ids.tolist(),
        }
        return inputs, targets, meta


# =============================================================================
# Generate data
# =============================================================================
print("Generating data...")
train_inputs, train_targets, train_meta = generate_data(N_TRAIN, LANG_FREQS, rotations, "train")
test_inputs, test_targets, test_meta = generate_data(N_TEST, LANG_FREQS, rotations, "test")

print(f"Train: {train_inputs.shape} -> {train_targets.shape}")
print(f"Test:  {test_inputs.shape} -> {test_targets.shape}")

# Verify language distribution in training
from collections import Counter
lang_counts = Counter(train_meta["lang_ids"])
print("\nTraining language distribution:")
for ell in range(K):
    pct = lang_counts[ell] / N_TRAIN * 100
    print(f"  {LANG_NAMES[ell]:>10s}: {lang_counts[ell]:5d} ({pct:.1f}%)")

# Sanity check: f is non-trivial (linear model can't approximate it)
print("\nSanity check: linear baseline...")
from sklearn.linear_model import LinearRegression
# Use Chinese-only training data for a fair test
zh_mask = [i for i, l in enumerate(train_meta["lang_ids"]) if l == 0]
X_lin = train_inputs[zh_mask, :D].numpy()  # strip one-hot, use rotated x only
Y_lin = train_targets[zh_mask].numpy()
lr = LinearRegression().fit(X_lin, Y_lin)
lin_pred = lr.predict(X_lin)
lin_mse = np.mean((lin_pred - Y_lin) ** 2)
print(f"  Linear MSE on Chinese train: {lin_mse:.4f}")
print(f"  (Should be >> 0.01 for f to be non-trivial)")


# =============================================================================
# Section 2: MLP + Training
# =============================================================================
# Plain MLP, no skip connections. H=6 hidden layers, width W.
# GELU activation. Input: D+K=17, Output: D=10.
#
# No residual connections means:
# - Each layer REPLACES the state (no additive channel)
# - Moon vector cannot survive (known gap vs real transformer)
# - Rank can degrade through depth (no I+J_f protection)

W = 128  # back to spec size — function is now learnable
H = 6   # number of hidden layers


class ReasoningMLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth, dropout=0.0):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(d_in, width))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        # Output layer (no activation)
        layers.append(nn.Linear(width, d_out))
        self.net = nn.Sequential(*layers)

        # For activation extraction: store intermediate activations
        self.activations = {}

    def forward(self, x, save_activations=False):
        if save_activations:
            self.activations = {}
        h = x
        layer_idx = 0
        for module in self.net:
            h = module(h)
            # Save AFTER GELU (before dropout) — dropout is off at eval time anyway
            if save_activations and isinstance(module, nn.GELU):
                self.activations[layer_idx] = h.detach()
                layer_idx += 1
        if save_activations:
            self.activations[layer_idx] = h.detach()
        return h


model = ReasoningMLP(D + K, D, W, H)
if torch.cuda.is_available():
    model = model.cuda()
    print("Using CUDA")
print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
print(f"  Input: {D+K} -> Hidden: {W} x {H} layers -> Output: {D}")

# Training — no weight decay (model is underfitting), cosine LR
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
N_EPOCHS = 500
BATCH_SIZE = 2048
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

print(f"\nTraining for up to {N_EPOCHS} epochs...")
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = next(model.parameters()).device
test_inputs_dev = test_inputs.to(device)
best_loss = float('inf')
best_state = None
EVAL_EVERY = 25  # test eval every N epochs

# Precompute test language masks
lang_ids = test_meta["lang_ids"]
lang_masks = {}
for ell in range(K):
    lang_masks[ell] = [i for i, l in enumerate(lang_ids) if l == ell]

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

    lr = optimizer.param_groups[0]['lr']
    pbar.set_postfix(mse=f"{epoch_loss:.5f}", lr=f"{lr:.1e}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Periodic verbose test eval
    if (epoch + 1) % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            tp = model(test_inputs_dev).cpu()
        test_mse = nn.functional.mse_loss(tp, test_targets).item()
        parts = []
        for ell in range(K):
            m = lang_masks[ell]
            mse_ell = nn.functional.mse_loss(tp[m], test_targets[m]).item()
            tag = "OK" if mse_ell < 0.01 else ""
            parts.append(f"{LANG_NAMES[ell][:2]}={mse_ell:.4f}{tag}")
        pbar.write(f"  [Ep {epoch+1}] test={test_mse:.4f} | {' '.join(parts)}")
        model.train()

pbar.close()

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)
print(f"Best training MSE: {best_loss:.6f}")

# =============================================================================
# Section 2b: Per-language test MSE (convergence check)
# =============================================================================
# Spec requirement: per-language MSE < 0.01 for all K languages.
# If Swahili fails, it can't be used as held-out test language.

model.eval()
with torch.no_grad():
    test_pred = model(test_inputs.to(device)).cpu()
    test_mse_all = nn.functional.mse_loss(test_pred, test_targets).item()

print(f"\nOverall test MSE: {test_mse_all:.6f}")
print("\nPer-language test MSE:")
lang_ids = test_meta["lang_ids"]
for ell in range(K):
    mask = [i for i, l in enumerate(lang_ids) if l == ell]
    mse_ell = nn.functional.mse_loss(test_pred[mask], test_targets[mask]).item()
    status = "OK" if mse_ell < 0.01 else "FAIL"
    print(f"  {LANG_NAMES[ell]:>10s}: MSE = {mse_ell:.6f}  [{status}]")

# =============================================================================
# Section 2c: Extract activations for all test data
# =============================================================================
# For each test sample, save hidden activations at every layer.
# activations_by_layer[h] is (1400, W) — all test samples at layer h.

print("\nExtracting activations at all hidden layers...")
model.eval()
with torch.no_grad():
    _ = model(test_inputs.to(device), save_activations=True)

activations_by_layer = {}
for h_idx, act in model.activations.items():
    activations_by_layer[h_idx] = act.cpu()
    print(f"  Layer {h_idx}: {act.shape}")

# Reshape into per-language activation matrices for Procrustes
# act_by_lang_layer[ell][h] = (N_TEST, W) — language ell, layer h
lang_ids_arr = np.array(lang_ids)
act_by_lang_layer = {}
for ell in range(K):
    mask = np.where(lang_ids_arr == ell)[0]
    act_by_lang_layer[ell] = {
        h_idx: act[mask] for h_idx, act in activations_by_layer.items()
    }

print(f"\nActivations organized: {K} languages x {len(activations_by_layer)} layers x ({N_TEST}, {W})")

# =============================================================================
# Section 2d: Layer 1 cross-lingual distance check
# =============================================================================
# Spec: compute mean ||a_1^ell(m) - a_1^ell'(m)|| for all language pairs.
# If zh-en ≈ 0: model canonicalized those two. If ALL ≈ 0: too powerful.

print("\nLayer 1 cross-lingual activation distances:")
h_check = 0  # first hidden layer (post-GELU)
for ell1 in range(min(K, 4)):  # just show zh, en, es, ar pairs
    for ell2 in range(ell1 + 1, K):
        a1 = act_by_lang_layer[ell1][h_check]
        a2 = act_by_lang_layer[ell2][h_check]
        dist = (a1 - a2).norm(dim=1).mean().item()
        print(f"  {LANG_NAMES[ell1]:>10s} - {LANG_NAMES[ell2]:<10s}: {dist:.4f}")

# =============================================================================
# Save everything for notebook analysis (Phases 2-4)
# =============================================================================
save_path = "output/z_poc_trained.pt"
save_data = {
    "model_state": model.cpu().state_dict(),
    "model_config": {"d_in": D + K, "d_out": D, "width": W, "depth": H},
    "rotations": rotations,
    "fixed_perm": FIXED_PERM,
    "test_inputs": test_inputs,
    "test_targets": test_targets,
    "test_meta": test_meta,
    "act_by_lang_layer": {ell: {h: a.cpu() for h, a in layers.items()}
                          for ell, layers in act_by_lang_layer.items()},
    "lang_names": LANG_NAMES,
    "lang_freqs": LANG_FREQS,
    "K": K, "D": D, "W": W, "H": H, "N_TEST": N_TEST,
}
torch.save(save_data, save_path)
print(f"\nSaved trained model + activations to {save_path}")
