"""
Synthetic Z Extraction POC — Config B-blind (imbalanced, NO one-hot, non-orthogonal transforms)
Spec: FriMarch7-Z_POC_Spec_v2.md

Two changes from Config B:
  1. NO one-hot language indicator — model infers language from data statistics
  2. General invertible transforms A_ell instead of orthogonal R_ell

Why non-orthogonal: orthogonal R @ N(0,I) = N(0,I) — languages are provably
indistinguishable. With A_ell in GL(d), each language gets covariance A_ell @ A_ell^T,
giving the model statistical signal to identify language from data alone.

This also makes Procrustes imperfect (true transform isn't orthogonal), so any
residual error measures the cost of the linearity assumption — real pyrite test.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

# Reproducibility — SAME seed as Config B
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Section 1a: The reasoning function f (identical to Config B)
# =============================================================================
D = 10  # input/output dimension

# Fixed random permutation replaces data-dependent sort.
# Linear baseline MSE ~0.022.
FIXED_PERM = torch.randperm(D)


def reasoning_function(x: torch.Tensor) -> torch.Tensor:
    """
    x: (batch, D) or (D,)
    returns: (batch, D) or (D,)

    Step 1: fixed permutation
    Step 2: cumulative sum
    Step 3: tanh(0.5 * .)
    Step 4: reverse
    Step 5: L2 normalize
    """
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
# Section 1b: Language transforms — general invertible (NOT orthogonal)
# =============================================================================
# Each language gets A_ell in GL(d) with condition number clamped below MAX_COND.
# A_ell @ N(0,I) = N(0, A_ell @ A_ell^T) — different per language, so model
# can distinguish languages from input covariance alone (no one-hot needed).

K = 7
LANG_NAMES = ["Chinese", "English", "Spanish", "Arabic", "Japanese", "Korean", "Swahili"]
LANG_FREQS = [0.60, 0.20, 0.08, 0.05, 0.04, 0.02, 0.01]
MAX_COND = 5.0  # clamp condition number to prevent numerical issues

transforms = []
for ell in range(K):
    A = torch.randn(D, D)
    # Clamp condition number: SVD, clamp singular values, reconstruct
    U, S, Vh = torch.linalg.svd(A)
    # Scale singular values so max/min <= MAX_COND
    S = S.clamp(min=S.max() / MAX_COND)
    A_clamped = U @ torch.diag(S) @ Vh
    transforms.append(A_clamped)

for ell, A in enumerate(transforms):
    cond = torch.linalg.cond(A).item()
    det = torch.linalg.det(A).item()
    print(f"  Transform {ell} ({LANG_NAMES[ell]}): cond={cond:.2f}, det={det:.2f}")


# =============================================================================
# Section 1c: Data generation — NO ONE-HOT (the only change from Config B)
# =============================================================================

N_TRAIN = 50_000
N_TEST = 200


def generate_data(n_samples, freqs, transforms, split="train"):
    """
    Generate training or test data. NO one-hot language indicator.

    Returns:
        inputs: (N, D) — rotated x only (no language tag)
        targets: (N, D) — rotated f(x)
        meta: dict with raw_x, lang_ids, raw_fx for analysis
    """
    raw_x = torch.randn(n_samples, D)
    raw_fx = reasoning_function(raw_x)

    if split == "test":
        all_inputs = []
        all_targets = []
        all_lang_ids = []
        all_problem_ids = []

        for ell in range(K):
            A = transforms[ell]
            x_rot = raw_x @ A.T
            y_rot = raw_fx @ A.T

            # NO one-hot — just rotated x
            all_inputs.append(x_rot)
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
        lang_ids = np.concatenate([
            np.full(int(freq * n_samples), ell, dtype=np.int64)
            for ell, freq in enumerate(freqs)
        ])
        if len(lang_ids) < n_samples:
            lang_ids = np.concatenate([lang_ids, np.zeros(n_samples - len(lang_ids), dtype=np.int64)])
        lang_ids = lang_ids[:n_samples]
        np.random.shuffle(lang_ids)

        A_all = torch.stack(transforms)
        lang_t = torch.from_numpy(lang_ids)
        A_per_sample = A_all[lang_t]

        x_rot = torch.bmm(raw_x.unsqueeze(1), A_per_sample.transpose(1, 2)).squeeze(1)
        y_rot = torch.bmm(raw_fx.unsqueeze(1), A_per_sample.transpose(1, 2)).squeeze(1)

        # NO one-hot — just rotated x
        inputs = x_rot
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
print("Generating data (Config B-blind: no one-hot)...")
train_inputs, train_targets, train_meta = generate_data(N_TRAIN, LANG_FREQS, transforms, "train")
test_inputs, test_targets, test_meta = generate_data(N_TEST, LANG_FREQS, transforms, "test")

print(f"Train: {train_inputs.shape} -> {train_targets.shape}")
print(f"Test:  {test_inputs.shape} -> {test_targets.shape}")

from collections import Counter
lang_counts = Counter(train_meta["lang_ids"])
print("\nTraining language distribution:")
for ell in range(K):
    pct = lang_counts[ell] / N_TRAIN * 100
    print(f"  {LANG_NAMES[ell]:>10s}: {lang_counts[ell]:5d} ({pct:.1f}%)")

# Linear baseline
print("\nSanity check: linear baseline...")
from sklearn.linear_model import LinearRegression
zh_mask = [i for i, l in enumerate(train_meta["lang_ids"]) if l == 0]
X_lin = train_inputs[zh_mask].numpy()  # no one-hot to strip
Y_lin = train_targets[zh_mask].numpy()
lr = LinearRegression().fit(X_lin, Y_lin)
lin_pred = lr.predict(X_lin)
lin_mse = np.mean((lin_pred - Y_lin) ** 2)
print(f"  Linear MSE on Chinese train: {lin_mse:.4f}")
print(f"  (Should be >> 0.01 for f to be non-trivial)")


# =============================================================================
# Section 2: MLP + Training — input dim is D (not D+K)
# =============================================================================

W = 128
H = 6


class ReasoningMLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth, dropout=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(d_in, width))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(width, d_out))
        self.net = nn.Sequential(*layers)
        self.activations = {}

    def forward(self, x, save_activations=False):
        if save_activations:
            self.activations = {}
        h = x
        layer_idx = 0
        for module in self.net:
            h = module(h)
            if save_activations and isinstance(module, nn.GELU):
                self.activations[layer_idx] = h.detach()
                layer_idx += 1
        if save_activations:
            self.activations[layer_idx] = h.detach()
        return h


# KEY CHANGE: input dim is D (10), not D+K (17)
model = ReasoningMLP(D, D, W, H)
if torch.cuda.is_available():
    model = model.cuda()
    print("Using CUDA")
print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
print(f"  Input: {D} -> Hidden: {W} x {H} layers -> Output: {D}")

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
EVAL_EVERY = 25

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

    cur_lr = optimizer.param_groups[0]['lr']
    pbar.set_postfix(mse=f"{epoch_loss:.5f}", lr=f"{cur_lr:.1e}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)
print(f"Best training MSE: {best_loss:.6f}")

# =============================================================================
# Section 2b: Per-language test MSE (convergence check)
# =============================================================================
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
print("\nExtracting activations at all hidden layers...")
model.eval()
with torch.no_grad():
    _ = model(test_inputs.to(device), save_activations=True)

activations_by_layer = {}
for h_idx, act in model.activations.items():
    activations_by_layer[h_idx] = act.cpu()
    print(f"  Layer {h_idx}: {act.shape}")

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
print("\nLayer 1 cross-lingual activation distances:")
h_check = 0
for ell1 in range(min(K, 4)):
    for ell2 in range(ell1 + 1, K):
        a1 = act_by_lang_layer[ell1][h_check]
        a2 = act_by_lang_layer[ell2][h_check]
        dist = (a1 - a2).norm(dim=1).mean().item()
        print(f"  {LANG_NAMES[ell1]:>10s} - {LANG_NAMES[ell2]:<10s}: {dist:.4f}")

# =============================================================================
# Save everything for notebook analysis
# =============================================================================
save_path = "output/z_poc_blind_trained.pt"
save_data = {
    "model_state": model.cpu().state_dict(),
    "model_config": {"d_in": D, "d_out": D, "width": W, "depth": H},
    "transforms": transforms,
    "fixed_perm": FIXED_PERM,
    "test_inputs": test_inputs,
    "test_targets": test_targets,
    "test_meta": test_meta,
    "act_by_lang_layer": {ell: {h: a.cpu() for h, a in layers.items()}
                          for ell, layers in act_by_lang_layer.items()},
    "lang_names": LANG_NAMES,
    "lang_freqs": LANG_FREQS,
    "K": K, "D": D, "W": W, "H": H, "N_TEST": N_TEST,
    "config": "B-blind",
}
torch.save(save_data, save_path)
print(f"\nSaved trained model + activations to {save_path}")
