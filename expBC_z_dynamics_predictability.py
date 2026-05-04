"""Exp BC: Z-state dynamics predictability during generation.

The question: given Z(t), can we predict Z(t+1)?

Uses existing generation-time trajectory data (h32 at every token step)
projected into the 20D contrastive Z-basis. Trains increasingly powerful
predictors and measures R² against baselines.

Predictors (in order of complexity):
  1. Identity:    Z(t+1) = Z(t)
  2. Momentum:    Z(t+1) = Z(t) + α(Z(t) - Z(t-1))   [captures bounce]
  3. Linear AR1:  Z(t+1) = W @ Z(t) + b
  4. Linear AR2:  Z(t+1) = W1 @ Z(t) + W2 @ Z(t-1) + b  [captures bounce structure]
  5. MLP:         Z(t+1) = MLP(Z(t), Z(t-1))  [nonlinear dynamics]

After AR2 captures the bounce, measures whether MLP on the RESIDUAL
(what AR2 can't predict) adds anything — i.e., is the non-bounce
component learnable or chaotic?

Also runs all predictors in non-Z space (20D PCA of the orthogonal
complement) as control. If Z dynamics are more predictable than non-Z,
that's evidence the reasoning trajectory has structure.

Data: gen_trajectories.npz (20 problems × 4 languages, L32 hidden states)
Z-basis: contrastive PCA from all_layers_lasttok.npz at L32
"""

import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from pathlib import Path

OUT = Path("output")

# ── 1. Build Z-basis from input-pass data ──────────────────────────────

print("=" * 60)
print("  Exp BC: Z-state dynamics predictability")
print("=" * 60)

print("\n[1/5] Building Z-basis from all_layers_lasttok.npz at L32...")

lasttok = np.load(OUT / "all_layers_lasttok.npz")
en_L32 = lasttok["en_L32"]  # (200, 2048)
zh_L32 = lasttok["zh_L32"]  # (200, 2048)

# Contrastive Z: PCA on zh-en differences
diffs = zh_L32 - en_L32  # (200, 2048)
Z_DIM = 20
pca_z = PCA(n_components=Z_DIM)
pca_z.fit(diffs)
Z_basis = pca_z.components_  # (20, 2048) — the Z axes

# Non-Z control: PCA on the Z-orthogonal complement of pooled activations
pooled = np.vstack([en_L32, zh_L32])  # (400, 2048)
# Project out Z
pooled_proj = pooled - (pooled @ Z_basis.T) @ Z_basis
pca_nonz = PCA(n_components=Z_DIM)
pca_nonz.fit(pooled_proj)
nonZ_basis = pca_nonz.components_  # (20, 2048) — non-Z control axes

print(f"  Z-basis: {Z_basis.shape}, explains {pca_z.explained_variance_ratio_.sum()*100:.1f}% of zh-en diff variance")
print(f"  non-Z basis: {nonZ_basis.shape}, explains {pca_nonz.explained_variance_ratio_.sum()*100:.1f}% of pooled-minus-Z variance")

# ── 2. Load generation trajectories and project ────────────────────────

print("\n[2/5] Loading generation trajectories...")

traj_data = np.load(OUT / "gen_trajectories.npz")
trajectories = {}
for key in sorted(traj_data.keys()):
    arr = traj_data[key]  # (T, 2048)
    if arr.shape[0] < 10:
        continue  # skip very short
    z_proj = arr @ Z_basis.T       # (T, 20) — Z projection
    nonz_proj = arr @ nonZ_basis.T  # (T, 20) — non-Z projection
    trajectories[key] = {"z": z_proj, "nonz": nonz_proj, "raw": arr}

print(f"  Loaded {len(trajectories)} trajectories")
total_tokens = sum(v["z"].shape[0] for v in trajectories.values())
print(f"  Total tokens: {total_tokens}")

# ── 3. Build train/test datasets ──────────────────────────────────────

print("\n[3/5] Building train/test datasets...")

def build_pairs(trajs, space="z", order=2, demean=True):
    """Build (X, Y) pairs for next-step prediction.

    X = [Z(t), Z(t-1), ...Z(t-order+1)] concatenated
    Y = Z(t+1)

    If demean=True, subtract per-trajectory mean so we're predicting
    DYNAMICS (deviations from trajectory center), not POSITION.

    Split: first 80% of each trajectory = train, last 20% = test.
    """
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for key, data in trajs.items():
        seq = data[space].copy()  # (T, D)
        T = seq.shape[0]
        if T < order + 5:
            continue
        if demean:
            seq = seq - seq.mean(axis=0, keepdims=True)
        split = int(T * 0.8)
        for t in range(order, T - 1):
            x = np.concatenate([seq[t - j] for j in range(order)])
            y = seq[t + 1]
            if t < split:
                X_train.append(x)
                Y_train.append(y)
            else:
                X_test.append(x)
                Y_test.append(y)
    return (np.array(X_train), np.array(Y_train),
            np.array(X_test), np.array(Y_test))


def per_trajectory_r2(trajs, space="z", order=2):
    """Compute R² per-trajectory, then average. The fair metric."""
    from sklearn.neural_network import MLPRegressor as _MLP
    traj_scores = []
    for key, data in trajs.items():
        seq = data[space].copy()
        T = seq.shape[0]
        if T < order + 20:  # need enough data per trajectory
            continue
        seq_dm = seq - seq.mean(axis=0, keepdims=True)
        D = seq_dm.shape[1]
        split = int(T * 0.8)

        X, Y = [], []
        for t in range(order, T - 1):
            x = np.concatenate([seq_dm[t - j] for j in range(order)])
            Y.append(seq_dm[t + 1])
            X.append(x)
        X, Y = np.array(X), np.array(Y)
        X_tr, Y_tr = X[:split - order], Y[:split - order]
        X_te, Y_te = X[split - order:], Y[split - order:]

        if len(X_te) < 5 or len(X_tr) < 10:
            continue

        scores = {"key": key, "T": T, "n_test": len(X_te)}

        # Identity
        scores["identity"] = float(r2_score(Y_te, X_te[:, :D], multioutput='uniform_average'))

        # AR(2) Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, Y_tr)
        pred = ridge.predict(X_te)
        scores["ar2"] = float(r2_score(Y_te, pred, multioutput='uniform_average'))

        # MLP
        mlp = _MLP(hidden_layer_sizes=(64, 32), activation='relu',
                    max_iter=300, early_stopping=True,
                    validation_fraction=0.2, random_state=42, learning_rate_init=0.001)
        try:
            mlp.fit(X_tr, Y_tr)
            pred_mlp = mlp.predict(X_te)
            scores["mlp"] = float(r2_score(Y_te, pred_mlp, multioutput='uniform_average'))
        except Exception:
            scores["mlp"] = float('nan')

        traj_scores.append(scores)

    return traj_scores


# Build for Z-space
X1_tr, Y1_tr, X1_te, Y1_te = build_pairs(trajectories, "z", order=1)
X2_tr, Y2_tr, X2_te, Y2_te = build_pairs(trajectories, "z", order=2)

# Build for non-Z space (control)
nX1_tr, nY1_tr, nX1_te, nY1_te = build_pairs(trajectories, "nonz", order=1)
nX2_tr, nY2_tr, nX2_te, nY2_te = build_pairs(trajectories, "nonz", order=2)

print(f"  Z order-1: train={X1_tr.shape[0]}, test={X1_te.shape[0]}")
print(f"  Z order-2: train={X2_tr.shape[0]}, test={X2_te.shape[0]}")

# ── 4. Train predictors ──────────────────────────────────────────────

print("\n[4/5] Training predictors...")

results = {"z": {}, "nonz": {}}

for space, label in [("z", "Z-space"), ("nonz", "non-Z space")]:
    print(f"\n  === {label} ===")

    if space == "z":
        x1_tr, y1_tr, x1_te, y1_te = X1_tr, Y1_tr, X1_te, Y1_te
        x2_tr, y2_tr, x2_te, y2_te = X2_tr, Y2_tr, X2_te, Y2_te
    else:
        x1_tr, y1_tr, x1_te, y1_te = nX1_tr, nY1_tr, nX1_te, nY1_te
        x2_tr, y2_tr, x2_te, y2_te = nX2_tr, nY2_tr, nX2_te, nY2_te

    D = y1_te.shape[1]  # 20

    # --- Predictor 1: Identity Z(t+1) = Z(t) ---
    pred_identity = x1_te[:, :D]  # First D dims of x1 = Z(t)
    r2_identity = r2_score(y1_te, pred_identity, multioutput='uniform_average')
    r2_identity_per_dim = [r2_score(y1_te[:, d], pred_identity[:, d]) for d in range(D)]
    print(f"  Identity  Z(t+1)=Z(t):        R²={r2_identity:.4f}")
    results[space]["identity"] = {
        "r2": float(r2_identity),
        "r2_per_dim": [float(x) for x in r2_identity_per_dim],
    }

    # --- Predictor 2: Momentum Z(t+1) = Z(t) + α(Z(t) - Z(t-1)) ---
    # Fit α per dimension on training data
    z_t_tr = x2_tr[:, :D]
    z_tm1_tr = x2_tr[:, D:2*D]
    delta_tr = z_t_tr - z_tm1_tr

    z_t_te = x2_te[:, :D]
    z_tm1_te = x2_te[:, D:2*D]
    delta_te = z_t_te - z_tm1_te

    # Fit α per dim: y = z_t + α * delta → α = <y - z_t, delta> / <delta, delta>
    alphas = []
    for d in range(D):
        num = np.dot(y2_tr[:, d] - z_t_tr[:, d], delta_tr[:, d])
        den = np.dot(delta_tr[:, d], delta_tr[:, d]) + 1e-10
        alphas.append(num / den)
    alphas = np.array(alphas)

    pred_momentum = z_t_te + alphas[None, :] * delta_te
    r2_momentum = r2_score(y2_te, pred_momentum, multioutput='uniform_average')
    r2_momentum_per_dim = [r2_score(y2_te[:, d], pred_momentum[:, d]) for d in range(D)]
    print(f"  Momentum  Z(t)+α·ΔZ:          R²={r2_momentum:.4f}  (mean α={alphas.mean():.3f})")
    results[space]["momentum"] = {
        "r2": float(r2_momentum),
        "mean_alpha": float(alphas.mean()),
        "alphas": [float(a) for a in alphas],
        "r2_per_dim": [float(x) for x in r2_momentum_per_dim],
    }

    # --- Predictor 3: Linear AR(1) Z(t+1) = W @ Z(t) + b ---
    ridge_ar1 = Ridge(alpha=1.0)
    ridge_ar1.fit(x1_tr, y1_tr)
    pred_ar1 = ridge_ar1.predict(x1_te)
    r2_ar1 = r2_score(y1_te, pred_ar1, multioutput='uniform_average')
    r2_ar1_per_dim = [r2_score(y1_te[:, d], pred_ar1[:, d]) for d in range(D)]
    print(f"  Linear AR(1)  W@Z(t)+b:       R²={r2_ar1:.4f}")
    results[space]["linear_ar1"] = {
        "r2": float(r2_ar1),
        "r2_per_dim": [float(x) for x in r2_ar1_per_dim],
    }

    # --- Predictor 4: Linear AR(2) Z(t+1) = W1@Z(t) + W2@Z(t-1) + b ---
    ridge_ar2 = Ridge(alpha=1.0)
    ridge_ar2.fit(x2_tr, y2_tr)
    pred_ar2 = ridge_ar2.predict(x2_te)
    r2_ar2 = r2_score(y2_te, pred_ar2, multioutput='uniform_average')
    r2_ar2_per_dim = [r2_score(y2_te[:, d], pred_ar2[:, d]) for d in range(D)]
    print(f"  Linear AR(2)  W1@Z(t)+W2@Z(t-1)+b: R²={r2_ar2:.4f}")
    results[space]["linear_ar2"] = {
        "r2": float(r2_ar2),
        "r2_per_dim": [float(x) for x in r2_ar2_per_dim],
    }

    # --- Predictor 5: MLP on [Z(t), Z(t-1)] ---
    # Simple 2-layer MLP via numpy (no torch needed for 20D)
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        learning_rate_init=0.001,
    )
    mlp.fit(x2_tr, y2_tr)
    pred_mlp = mlp.predict(x2_te)
    r2_mlp = r2_score(y2_te, pred_mlp, multioutput='uniform_average')
    r2_mlp_per_dim = [r2_score(y2_te[:, d], pred_mlp[:, d]) for d in range(D)]
    print(f"  MLP [64,32]  f(Z(t),Z(t-1)): R²={r2_mlp:.4f}")
    results[space]["mlp"] = {
        "r2": float(r2_mlp),
        "r2_per_dim": [float(x) for x in r2_mlp_per_dim],
    }

    # --- Predictor 6: MLP on AR(2) RESIDUAL ---
    # What can the MLP learn that the linear model can't?
    resid_tr = y2_tr - ridge_ar2.predict(x2_tr)
    resid_te = y2_te - pred_ar2

    mlp_resid = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        learning_rate_init=0.001,
    )
    mlp_resid.fit(x2_tr, resid_tr)
    pred_resid = mlp_resid.predict(x2_te)

    r2_resid_overall = r2_score(resid_te, pred_resid, multioutput='uniform_average')
    print(f"  MLP on AR(2) residual:        R²={r2_resid_overall:.4f}  (nonlinear beyond bounce)")
    results[space]["mlp_on_residual"] = {
        "r2": float(r2_resid_overall),
    }

    # --- Combined: AR(2) + MLP residual ---
    pred_combined = pred_ar2 + pred_resid
    r2_combined = r2_score(y2_te, pred_combined, multioutput='uniform_average')
    print(f"  AR(2) + MLP residual:         R²={r2_combined:.4f}  (total learnable)")
    results[space]["ar2_plus_mlp_residual"] = {
        "r2": float(r2_combined),
    }

# ── 5. Per-trajectory analysis (the fair metric) ─────────────────────

print("\n[5/7] Per-trajectory R² (the fair metric — no cross-trajectory leakage)...")

z_per_traj = per_trajectory_r2(trajectories, "z", order=2)
nonz_per_traj = per_trajectory_r2(trajectories, "nonz", order=2)

z_id_scores = [s["identity"] for s in z_per_traj if not np.isnan(s["identity"])]
z_ar2_scores = [s["ar2"] for s in z_per_traj if not np.isnan(s["ar2"])]
z_mlp_scores = [s["mlp"] for s in z_per_traj if not np.isnan(s.get("mlp", float("nan")))]

nz_id_scores = [s["identity"] for s in nonz_per_traj if not np.isnan(s["identity"])]
nz_ar2_scores = [s["ar2"] for s in nonz_per_traj if not np.isnan(s["ar2"])]
nz_mlp_scores = [s["mlp"] for s in nonz_per_traj if not np.isnan(s.get("mlp", float("nan")))]

print(f"\n  Per-trajectory R² (mean ± std over {len(z_per_traj)} trajectories):")
print(f"  {'Predictor':<25s} {'Z mean':>8s} {'Z std':>8s} {'nonZ mean':>10s} {'nonZ std':>10s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
for name, z_s, nz_s in [
    ("Identity", z_id_scores, nz_id_scores),
    ("AR(2)", z_ar2_scores, nz_ar2_scores),
    ("MLP", z_mlp_scores, nz_mlp_scores),
]:
    zm, zs = np.mean(z_s), np.std(z_s)
    nzm, nzs = np.mean(nz_s), np.std(nz_s)
    print(f"  {name:<25s} {zm:>+8.4f} {zs:>8.4f} {nzm:>+10.4f} {nzs:>10.4f}")

results["per_trajectory_z"] = {
    "identity": {"mean": float(np.mean(z_id_scores)), "std": float(np.std(z_id_scores)), "scores": [float(x) for x in z_id_scores]},
    "ar2": {"mean": float(np.mean(z_ar2_scores)), "std": float(np.std(z_ar2_scores)), "scores": [float(x) for x in z_ar2_scores]},
    "mlp": {"mean": float(np.mean(z_mlp_scores)), "std": float(np.std(z_mlp_scores)), "scores": [float(x) for x in z_mlp_scores]},
}
results["per_trajectory_nonz"] = {
    "identity": {"mean": float(np.mean(nz_id_scores)), "std": float(np.std(nz_id_scores))},
    "ar2": {"mean": float(np.mean(nz_ar2_scores)), "std": float(np.std(nz_ar2_scores))},
    "mlp": {"mean": float(np.mean(nz_mlp_scores)), "std": float(np.std(nz_mlp_scores))},
}

# Best / worst trajectories
best_z_mlp = max(z_per_traj, key=lambda s: s.get("mlp", -999))
worst_z_mlp = min(z_per_traj, key=lambda s: s.get("mlp", 999))
print(f"\n  Best Z trajectory:  {best_z_mlp['key']}  MLP R²={best_z_mlp['mlp']:.4f} (T={best_z_mlp['T']})")
print(f"  Worst Z trajectory: {worst_z_mlp['key']}  MLP R²={worst_z_mlp['mlp']:.4f} (T={worst_z_mlp['T']})")

# ── 6. Per-language and per-problem analysis ──────────────────────────

print("\n[6/7] Per-language breakdown...")

lang_results = {}
for lang in ["en", "zh", "es", "ja"]:
    lang_trajs = {k: v for k, v in trajectories.items() if k.endswith(f"_{lang}")}
    if not lang_trajs:
        continue

    x2_tr, y2_tr, x2_te, y2_te = build_pairs(lang_trajs, "z", order=2)
    if len(x2_te) < 20:
        continue

    # Identity
    r2_id = r2_score(y2_te, x2_te[:, :Z_DIM], multioutput='uniform_average')

    # AR(2)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x2_tr, y2_tr)
    r2_ar2 = r2_score(y2_te, ridge.predict(x2_te), multioutput='uniform_average')

    # MLP
    mlp_lang = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                            max_iter=500, early_stopping=True,
                            validation_fraction=0.15, random_state=42)
    mlp_lang.fit(x2_tr, y2_tr)
    r2_mlp = r2_score(y2_te, mlp_lang.predict(x2_te), multioutput='uniform_average')

    lang_results[lang] = {
        "n_trajs": len(lang_trajs),
        "n_test": len(x2_te),
        "identity_r2": float(r2_id),
        "ar2_r2": float(r2_ar2),
        "mlp_r2": float(r2_mlp),
    }
    print(f"  {lang}: identity={r2_id:.4f}, AR(2)={r2_ar2:.4f}, MLP={r2_mlp:.4f}  (N_test={len(x2_te)})")

# Also add per-trajectory detail to output
results["per_trajectory_detail"] = [
    {"key": s["key"], "T": s["T"], "identity": s["identity"],
     "ar2": s["ar2"], "mlp": s.get("mlp", None)}
    for s in z_per_traj
]

# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  POOLED R² (all trajectories demeaned, then pooled):")
print(f"  {'Predictor':<35s} {'Z-space':>10s} {'non-Z':>10s} {'Δ':>8s}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
for pred_name in ["identity", "momentum", "linear_ar1", "linear_ar2", "mlp", "mlp_on_residual", "ar2_plus_mlp_residual"]:
    z_r2 = results["z"].get(pred_name, {}).get("r2", float('nan'))
    nz_r2 = results["nonz"].get(pred_name, {}).get("r2", float('nan'))
    delta = z_r2 - nz_r2
    print(f"  {pred_name:<35s} {z_r2:>10.4f} {nz_r2:>10.4f} {delta:>+8.4f}")

print(f"\n  PER-TRAJECTORY R² (mean over individual trajectories — the fair metric):")
print(f"  {'Predictor':<25s} {'Z mean':>10s} {'nonZ mean':>10s} {'Δ':>8s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
for name in ["identity", "ar2", "mlp"]:
    z_m = results["per_trajectory_z"][name]["mean"]
    nz_m = results["per_trajectory_nonz"][name]["mean"]
    print(f"  {name:<25s} {z_m:>+10.4f} {nz_m:>+10.4f} {z_m - nz_m:>+8.4f}")

print(f"\n  Bounce α (mean across Z dims): {results['z']['momentum']['mean_alpha']:.3f}")
print(f"  Total Z-space tokens: {total_tokens}")
print(f"  Trajectories analyzed: {len(z_per_traj)}")

# ── Save ─────────────────────────────────────────────────────────────

output = {
    "experiment": "BC",
    "title": "Z-state dynamics predictability during generation",
    "z_dim": Z_DIM,
    "n_trajectories": len(trajectories),
    "total_tokens": total_tokens,
    "z_basis_var_explained": float(pca_z.explained_variance_ratio_.sum()),
    "results": results,
    "per_language": lang_results,
}

with open(OUT / "expBC_z_dynamics.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved to output/expBC_z_dynamics.json")
