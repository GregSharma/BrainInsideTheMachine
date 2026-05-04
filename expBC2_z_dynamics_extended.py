"""Exp BC2: Z-state dynamics — extended predictors.

BC showed R² < 0 for AR(1) and AR(2). But transformers aren't Markov.
At generation step t, the model sees ALL previous tokens via KV cache.
Z(t+1) could depend on Z(t-20) or the entire history.

Extended predictors:
  1. AR(k) for k = 1, 2, 3, 5, 10, 20  — longer autoregressive memory
  2. Exogenous: Z(t) + nonZ(t)  — does the narration state predict reasoning?
  3. Full context: [Z(t), Z(t-1), ..., Z(t-k), nonZ(t)]  — everything
  4. Deep MLP with full history window
  5. Cumulative features: running mean, running var, trajectory position (t/T)
  6. Z₀ → Z_final direct mapping (skip the trajectory entirely)
"""

import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path

OUT = Path("output")

print("=" * 60)
print("  Exp BC2: Z-state dynamics — extended predictors")
print("=" * 60)

# ── 1. Load Z-basis and trajectories (same as BC) ────────────────────

print("\n[1/6] Loading Z-basis and trajectories...")

lasttok = np.load(OUT / "all_layers_lasttok.npz")
en_L32, zh_L32 = lasttok["en_L32"], lasttok["zh_L32"]
diffs = zh_L32 - en_L32
Z_DIM = 20

pca_z = PCA(n_components=Z_DIM)
pca_z.fit(diffs)
Z_basis = pca_z.components_

pooled = np.vstack([en_L32, zh_L32])
pooled_proj = pooled - (pooled @ Z_basis.T) @ Z_basis
pca_nonz = PCA(n_components=Z_DIM)
pca_nonz.fit(pooled_proj)
nonZ_basis = pca_nonz.components_

traj_data = np.load(OUT / "gen_trajectories.npz")
trajectories = {}
for key in sorted(traj_data.keys()):
    arr = traj_data[key]
    if arr.shape[0] < 10:
        continue
    z = arr @ Z_basis.T
    nz = arr @ nonZ_basis.T
    # Demean per trajectory
    trajectories[key] = {
        "z": z - z.mean(0, keepdims=True),
        "nz": nz - nz.mean(0, keepdims=True),
        "z_raw": z,
        "nz_raw": nz,
        "T": arr.shape[0],
    }

print(f"  {len(trajectories)} trajectories, Z-basis explains {pca_z.explained_variance_ratio_.sum()*100:.1f}% of diff variance")

# ── 2. AR(k) sweep — how much does longer memory help? ───────────────

print("\n[2/6] AR(k) sweep: k = 1, 2, 3, 5, 10, 20")

def build_ar_pairs(trajs, space, k, include_exog=False, include_cumulative=False):
    """Build AR(k) pairs, optionally with exogenous features."""
    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    for key, data in trajs.items():
        z = data["z" if space == "z" else "nz"]
        nz = data["nz" if space == "z" else "z"]
        T = z.shape[0]
        if T < k + 10:
            continue
        split = int(T * 0.8)
        for t in range(k, T - 1):
            # AR features: [Z(t), Z(t-1), ..., Z(t-k+1)]
            feats = [z[t - j] for j in range(k)]

            if include_exog:
                # Add non-Z state at time t
                feats.append(nz[t])

            if include_cumulative:
                # Running mean of Z up to t
                feats.append(z[:t+1].mean(axis=0))
                # Trajectory position (scalar, broadcast to dim)
                pos = np.full(z.shape[1], t / T)
                feats.append(pos)

            x = np.concatenate(feats)
            y = z[t + 1]
            if t < split:
                X_tr.append(x)
                Y_tr.append(y)
            else:
                X_te.append(x)
                Y_te.append(y)
    return np.array(X_tr), np.array(Y_tr), np.array(X_te), np.array(Y_te)


ar_results = {}
for k in [1, 2, 3, 5, 10, 20]:
    X_tr, Y_tr, X_te, Y_te = build_ar_pairs(trajectories, "z", k)
    if len(X_te) < 50:
        continue

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr)
    pred = ridge.predict(X_te)
    r2 = r2_score(Y_te, pred, multioutput='uniform_average')

    # MLP
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                       max_iter=500, early_stopping=True,
                       validation_fraction=0.15, random_state=42,
                       learning_rate_init=0.001)
    mlp.fit(X_tr_s, Y_tr)
    r2_mlp = r2_score(Y_te, mlp.predict(X_te_s), multioutput='uniform_average')

    ar_results[k] = {"ridge_r2": float(r2), "mlp_r2": float(r2_mlp),
                      "n_features": X_tr.shape[1], "n_test": len(X_te)}
    print(f"  AR({k:>2d}): Ridge R²={r2:+.4f}, MLP R²={r2_mlp:+.4f}  (feats={X_tr.shape[1]}, N={len(X_te)})")


# ── 3. Exogenous features — does non-Z predict Z? ────────────────────

print("\n[3/6] Exogenous: Z(t) + nonZ(t) → Z(t+1)")

exog_results = {}
for k in [2, 5, 10]:
    # Z lags only
    X_tr_z, Y_tr, X_te_z, Y_te = build_ar_pairs(trajectories, "z", k, include_exog=False)
    # Z lags + nonZ(t)
    X_tr_e, _, X_te_e, _ = build_ar_pairs(trajectories, "z", k, include_exog=True)

    if len(X_te_e) < 50:
        continue

    # Ridge with exogenous
    ridge_e = Ridge(alpha=1.0)
    ridge_e.fit(X_tr_e, Y_tr)
    r2_e = r2_score(Y_te, ridge_e.predict(X_te_e), multioutput='uniform_average')

    # MLP with exogenous
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_e)
    X_te_s = scaler.transform(X_te_e)
    mlp_e = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                         max_iter=500, early_stopping=True,
                         validation_fraction=0.15, random_state=42)
    mlp_e.fit(X_tr_s, Y_tr)
    r2_mlp_e = r2_score(Y_te, mlp_e.predict(X_te_s), multioutput='uniform_average')

    # Compare to Z-only
    r2_z_only = ar_results.get(k, {}).get("ridge_r2", float("nan"))
    r2_mlp_z = ar_results.get(k, {}).get("mlp_r2", float("nan"))

    exog_results[k] = {
        "z_only_ridge": float(r2_z_only), "exog_ridge": float(r2_e),
        "z_only_mlp": float(r2_mlp_z), "exog_mlp": float(r2_mlp_e),
        "exog_lift_ridge": float(r2_e - r2_z_only),
        "exog_lift_mlp": float(r2_mlp_e - r2_mlp_z),
    }
    print(f"  k={k}: Z-only Ridge={r2_z_only:+.4f}, +nonZ Ridge={r2_e:+.4f} (lift={r2_e - r2_z_only:+.4f})")
    print(f"        Z-only MLP  ={r2_mlp_z:+.4f}, +nonZ MLP  ={r2_mlp_e:+.4f} (lift={r2_mlp_e - r2_mlp_z:+.4f})")


# ── 4. Full context: Z lags + nonZ + cumulative features ─────────────

print("\n[4/6] Full context: Z lags + nonZ + running mean + position")

full_results = {}
for k in [5, 10]:
    X_tr, Y_tr, X_te, Y_te = build_ar_pairs(trajectories, "z", k,
                                              include_exog=True,
                                              include_cumulative=True)
    if len(X_te) < 50:
        continue

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, Y_tr)
    r2_ridge = r2_score(Y_te, ridge.predict(X_te_s), multioutput='uniform_average')

    # Deep MLP
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu',
                       max_iter=1000, early_stopping=True,
                       validation_fraction=0.15, random_state=42,
                       learning_rate_init=0.0005, batch_size=256)
    mlp.fit(X_tr_s, Y_tr)
    r2_mlp = r2_score(Y_te, mlp.predict(X_te_s), multioutput='uniform_average')

    full_results[k] = {
        "ridge_r2": float(r2_ridge), "mlp_r2": float(r2_mlp),
        "n_features": X_tr.shape[1],
    }
    print(f"  k={k}: Ridge R²={r2_ridge:+.4f}, Deep MLP [256,128,64] R²={r2_mlp:+.4f}  (feats={X_tr.shape[1]})")


# ── 5. Z₀ → Z_final direct mapping ──────────────────────────────────

print("\n[5/6] Z₀ → Z_final direct mapping (skip trajectory entirely)")

# For each trajectory: can we predict the endpoint from the start?
z_starts, z_ends, nz_starts = [], [], []
traj_keys = []
for key, data in trajectories.items():
    z = data["z_raw"]  # Use RAW (not demeaned) for start→end
    nz = data["nz_raw"]
    if z.shape[0] < 20:
        continue
    z_starts.append(z[0])
    z_ends.append(z[-1])
    nz_starts.append(nz[0])
    traj_keys.append(key)

z_starts = np.array(z_starts)
z_ends = np.array(z_ends)
nz_starts = np.array(nz_starts)

# Leave-one-out cross-validation (only ~74 trajectories)
n = len(z_starts)
preds_ridge, preds_mlp, preds_identity = [], [], []
preds_z_nz = []

for i in range(n):
    mask = np.ones(n, bool)
    mask[i] = False

    # Identity: Z_final = Z₀
    preds_identity.append(z_starts[i])

    # Ridge: Z_final = W @ Z₀ + b
    ridge = Ridge(alpha=10.0)
    ridge.fit(z_starts[mask], z_ends[mask])
    preds_ridge.append(ridge.predict(z_starts[i:i+1])[0])

    # Ridge: Z_final = W @ [Z₀, nonZ₀] + b
    X_both = np.hstack([z_starts, nz_starts])
    ridge2 = Ridge(alpha=10.0)
    ridge2.fit(X_both[mask], z_ends[mask])
    preds_z_nz.append(ridge2.predict(X_both[i:i+1])[0])

preds_identity = np.array(preds_identity)
preds_ridge = np.array(preds_ridge)
preds_z_nz = np.array(preds_z_nz)

r2_identity_direct = r2_score(z_ends, preds_identity, multioutput='uniform_average')
r2_ridge_direct = r2_score(z_ends, preds_ridge, multioutput='uniform_average')
r2_z_nz_direct = r2_score(z_ends, preds_z_nz, multioutput='uniform_average')

print(f"  Identity  Z_final = Z₀:           R²={r2_identity_direct:+.4f}")
print(f"  Ridge     Z_final = W@Z₀ + b:     R²={r2_ridge_direct:+.4f}")
print(f"  Ridge     Z_final = W@[Z₀,nZ₀]+b: R²={r2_z_nz_direct:+.4f}")

# Per-dimension R² for the direct mapping
per_dim_r2 = [r2_score(z_ends[:, d], preds_ridge[:, d]) for d in range(Z_DIM)]
best_dims = np.argsort(per_dim_r2)[::-1][:5]
print(f"  Top dims: {[(int(d), round(per_dim_r2[d], 3)) for d in best_dims]}")

direct_results = {
    "identity_r2": float(r2_identity_direct),
    "ridge_r2": float(r2_ridge_direct),
    "ridge_z_nz_r2": float(r2_z_nz_direct),
    "n_trajectories": n,
    "per_dim_r2": [float(x) for x in per_dim_r2],
}

# ── 6. Cosine autocorrelation at multiple lags ──────────────────────

print("\n[6/6] Cosine autocorrelation at multiple lags (Z-space)")

lag_cosines = {}
for lag in [1, 2, 3, 5, 10, 20, 50]:
    cosines = []
    for key, data in trajectories.items():
        z = data["z"]
        T = z.shape[0]
        for t in range(lag, T):
            a, b = z[t], z[t - lag]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-8 and nb > 1e-8:
                cosines.append(np.dot(a, b) / (na * nb))
    if cosines:
        cosines = np.array(cosines)
        lag_cosines[lag] = {
            "mean": float(cosines.mean()),
            "std": float(cosines.std()),
            "n": len(cosines),
        }
        print(f"  lag={lag:>3d}: cos={cosines.mean():+.4f} ± {cosines.std():.4f}  (N={len(cosines)})")


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\n  AR(k) sweep (Ridge / MLP):")
for k, v in sorted(ar_results.items()):
    print(f"    k={k:>2d}: Ridge={v['ridge_r2']:+.4f}  MLP={v['mlp_r2']:+.4f}")

print(f"\n  Exogenous lift (adding nonZ to prediction):")
for k, v in sorted(exog_results.items()):
    print(f"    k={k:>2d}: Ridge lift={v['exog_lift_ridge']:+.4f}  MLP lift={v['exog_lift_mlp']:+.4f}")

print(f"\n  Full context (Z lags + nonZ + cumulative):")
for k, v in sorted(full_results.items()):
    print(f"    k={k:>2d}: Ridge={v['ridge_r2']:+.4f}  Deep MLP={v['mlp_r2']:+.4f}  (feats={v['n_features']})")

print(f"\n  Z₀ → Z_final direct mapping:")
print(f"    Identity:    R²={r2_identity_direct:+.4f}")
print(f"    Ridge(Z₀):   R²={r2_ridge_direct:+.4f}")
print(f"    Ridge(Z₀+nZ₀): R²={r2_z_nz_direct:+.4f}")

print(f"\n  Cosine autocorrelation by lag:")
for lag, v in sorted(lag_cosines.items()):
    print(f"    lag={lag:>3d}: {v['mean']:+.4f}")

# ── Save ─────────────────────────────────────────────────────────────

output = {
    "experiment": "BC2",
    "title": "Z-state dynamics — extended predictors",
    "ar_sweep": ar_results,
    "exogenous": exog_results,
    "full_context": full_results,
    "direct_mapping": direct_results,
    "cosine_autocorrelation": lag_cosines,
}

with open(OUT / "expBC2_z_dynamics_extended.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to output/expBC2_z_dynamics_extended.json")
