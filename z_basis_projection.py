"""Z-BASIS PROJECTION TEST: Does input-pass Z capture generation-time alignment?

The key experiment: fit PCA=20 on input-pass L32 activations (the "encoding Z"),
then project generation-time h32 trajectories onto that basis. If cosine spikes
survive projection, the subway runs on the same tracks. If they don't, the model
builds a different Z during generation.

Also measures:
- Variance captured by input-pass Z basis at generation time
- Encoding-Z vs Reasoning-Z principal angle overlap
- Per-token cosine in full space vs projected space
"""
import numpy as np
from sklearn.decomposition import PCA
import json
import itertools

# === LOAD DATA ===
print("Loading data...")
all_layers = np.load('output/all_layers.npz', allow_pickle=True)
gen_data = np.load('output/gen_trajectories_peos.npz', allow_pickle=True)

# Input-pass activations at L32 (200 problems × 2 langs)
zh_L32 = all_layers['zh_L32']  # (200, 2048)
en_L32 = all_layers['en_L32']  # (200, 2048)
input_pass = np.vstack([zh_L32, en_L32])  # (400, 2048)

# === FIT ENCODING Z BASIS ===
print("Fitting PCA=20 Z basis on input-pass L32...")
pca_z = PCA(n_components=20)
pca_z.fit(input_pass)
Z_basis = pca_z.components_  # (20, 2048) — the encoding Z basis vectors
explained_input = pca_z.explained_variance_ratio_.sum()
print(f"  Input-pass variance explained: {explained_input:.3f}")

# === IDENTIFY GENERATION PROBLEMS ===
GEN_LANGS = ['zh', 'en', 'es', 'ja']
problems = sorted(set(
    k.split('_')[1] for k in gen_data.keys() if k.startswith('h32_')
))
print(f"Generation problems: {len(problems)}")

# === HELPER ===
def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def project_onto_Z(vecs, basis):
    """Project vectors onto Z basis (20-dim subspace)."""
    coords = vecs @ basis.T  # (T, 20)
    projected = coords @ basis  # (T, 2048) — back in full space
    return projected, coords

def project_out_Z(vecs, basis):
    """Remove Z component, keep residual."""
    _, coords = project_onto_Z(vecs, basis)
    projected = coords @ basis
    return vecs - projected

# === EXPERIMENT 1: COSINE SPIKES IN PROJECTED SPACE ===
print("\n=== Experiment 1: Cosine spikes — full vs Z-projected vs Z-residual ===")
pairs = list(itertools.combinations(GEN_LANGS, 2))
spike_results = []
all_full_cosines = []
all_proj_cosines = []
all_resid_cosines = []

for prob in problems:
    # Load all language trajectories for this problem
    trajs = {}
    for lang in GEN_LANGS:
        key = f'h32_{prob}_{lang}'
        if key in gen_data:
            trajs[lang] = gen_data[key]

    if len(trajs) < 2:
        continue

    for lang_a, lang_b in pairs:
        if lang_a not in trajs or lang_b not in trajs:
            continue

        h_a = trajs[lang_a]  # (T_a, 2048)
        h_b = trajs[lang_b]  # (T_b, 2048)
        T = min(len(h_a), len(h_b))
        if T < 5:
            continue

        # Normalize to progress τ ∈ [0, 1]
        for t in range(T):
            tau = t / T

            # Full-space cosine
            cos_full = cosine_sim(h_a[t], h_b[t])

            # Z-projected cosine (only the Z component)
            proj_a, _ = project_onto_Z(h_a[t:t+1], Z_basis)
            proj_b, _ = project_onto_Z(h_b[t:t+1], Z_basis)
            cos_proj = cosine_sim(proj_a[0], proj_b[0])

            # Z-residual cosine (everything BUT Z)
            resid_a = project_out_Z(h_a[t:t+1], Z_basis)
            resid_b = project_out_Z(h_b[t:t+1], Z_basis)
            cos_resid = cosine_sim(resid_a[0], resid_b[0])

            all_full_cosines.append((tau, cos_full))
            all_proj_cosines.append((tau, cos_proj))
            all_resid_cosines.append((tau, cos_resid))

# Bin into 50 τ-bins (2% resolution)
n_bins = 50
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

def bin_cosines(cosine_list):
    taus = np.array([x[0] for x in cosine_list])
    vals = np.array([x[1] for x in cosine_list])
    binned = []
    for i in range(n_bins):
        mask = (taus >= bin_edges[i]) & (taus < bin_edges[i+1])
        if mask.sum() > 0:
            binned.append(float(vals[mask].mean()))
        else:
            binned.append(None)
    return binned

full_binned = bin_cosines(all_full_cosines)
proj_binned = bin_cosines(all_proj_cosines)
resid_binned = bin_cosines(all_resid_cosines)

# Count spikes (cosine > 0.3)
full_vals = np.array([x[1] for x in all_full_cosines])
proj_vals = np.array([x[1] for x in all_proj_cosines])
resid_vals = np.array([x[1] for x in all_resid_cosines])

n_full_spikes = int((full_vals > 0.3).sum())
n_proj_spikes = int((proj_vals > 0.3).sum())
n_resid_spikes = int((resid_vals > 0.3).sum())

print(f"  Full-space spikes (>0.3): {n_full_spikes}/{len(full_vals)}")
print(f"  Z-projected spikes (>0.3): {n_proj_spikes}/{len(proj_vals)}")
print(f"  Z-residual spikes (>0.3): {n_resid_spikes}/{len(resid_vals)}")

# === EXPERIMENT 2: VARIANCE CAPTURED BY ENCODING-Z AT GENERATION TIME ===
print("\n=== Experiment 2: How much gen-time variance does encoding-Z capture? ===")
all_gen_vecs = []
for prob in problems:
    for lang in GEN_LANGS:
        key = f'h32_{prob}_{lang}'
        if key in gen_data:
            all_gen_vecs.append(gen_data[key])

gen_matrix = np.vstack(all_gen_vecs)  # (N_total, 2048)
print(f"  Total generation vectors: {gen_matrix.shape[0]}")

# Variance in Z vs total
gen_centered = gen_matrix - gen_matrix.mean(axis=0)
total_var = np.sum(gen_centered ** 2)

gen_proj, gen_coords = project_onto_Z(gen_centered, Z_basis)
proj_var = np.sum(gen_proj ** 2)
var_fraction = float(proj_var / total_var)
print(f"  Encoding-Z captures {var_fraction:.3f} of generation-time variance")

# === EXPERIMENT 3: PRINCIPAL ANGLE OVERLAP (Encoding-Z vs Reasoning-Z) ===
print("\n=== Experiment 3: Principal angles between Encoding-Z and Reasoning-Z ===")
pca_gen = PCA(n_components=20)
pca_gen.fit(gen_matrix)
R_basis = pca_gen.components_  # (20, 2048) — the reasoning Z basis

explained_gen = pca_gen.explained_variance_ratio_.sum()
print(f"  Reasoning-Z variance explained (gen-time PCA=20): {explained_gen:.3f}")

# Principal angles via SVD of B = Z_basis @ R_basis.T
B = Z_basis @ R_basis.T  # (20, 20)
U, sigma, Vt = np.linalg.svd(B)
# sigma = cos(principal angles)
principal_angles_deg = np.degrees(np.arccos(np.clip(sigma, -1, 1)))
print(f"  Principal angles (degrees): {[f'{a:.1f}' for a in principal_angles_deg[:10]]}")
print(f"  Mean cos(angle) for top 10: {sigma[:10].mean():.3f}")
print(f"  Mean cos(angle) for top 20: {sigma.mean():.3f}")

# Grassmann distance
grassmann = float(np.sqrt(np.sum(np.arccos(np.clip(sigma, -1, 1))**2)))
print(f"  Grassmann distance: {grassmann:.3f}")

# === EXPERIMENT 4: Z VARIANCE FRACTION SWEEP (ALL 36 LAYERS) ===
print("\n=== Experiment 4: Z variance fraction across all 36 layers ===")
layer_var_results = {}
for layer in range(36):
    zh_key = f'zh_L{layer}'
    en_key = f'en_L{layer}'
    if zh_key not in all_layers or en_key not in all_layers:
        continue

    zh = all_layers[zh_key]
    en = all_layers[en_key]
    combined = np.vstack([zh, en])

    # Fit PCA=20 at this layer
    pca_layer = PCA(n_components=20)
    pca_layer.fit(combined)

    var_explained = float(pca_layer.explained_variance_ratio_.sum())

    # Dim 318 fraction at this layer
    centered = combined - combined.mean(axis=0)
    total_v = np.sum(centered ** 2)
    dim318_v = np.sum(centered[:, 318] ** 2)
    dim318_frac = float(dim318_v / total_v) if total_v > 0 else 0

    # Top 5 dims by variance
    per_dim_var = np.sum(centered ** 2, axis=0)
    top5_idx = np.argsort(per_dim_var)[-5:][::-1]
    top5_frac = float(per_dim_var[top5_idx].sum() / total_v)

    layer_var_results[layer] = {
        'pca20_var_explained': var_explained,
        'dim318_var_fraction': dim318_frac,
        'top5_var_fraction': top5_frac,
        'top5_dims': top5_idx.tolist(),
    }

    print(f"  L{layer:2d}: PCA20={var_explained:.3f}  dim318={dim318_frac:.3f}  top5={top5_frac:.3f}  dims={top5_idx.tolist()}")

# === SAVE RESULTS ===
output = {
    'experiment_1_cosine_spikes': {
        'bin_centers': bin_centers,
        'full_space_binned': full_binned,
        'z_projected_binned': proj_binned,
        'z_residual_binned': resid_binned,
        'n_full_spikes': n_full_spikes,
        'n_proj_spikes': n_proj_spikes,
        'n_resid_spikes': n_resid_spikes,
        'n_total_pairs': len(full_vals),
        'full_mean': float(full_vals.mean()),
        'proj_mean': float(proj_vals.mean()),
        'resid_mean': float(resid_vals.mean()),
        'full_max': float(full_vals.max()),
        'proj_max': float(proj_vals.max()),
        'resid_max': float(resid_vals.max()),
    },
    'experiment_2_gen_variance': {
        'encoding_z_captures_fraction': var_fraction,
        'n_gen_vectors': int(gen_matrix.shape[0]),
        'input_pass_var_explained': float(explained_input),
    },
    'experiment_3_principal_angles': {
        'cos_principal_angles': sigma.tolist(),
        'principal_angles_deg': principal_angles_deg.tolist(),
        'mean_cos_top10': float(sigma[:10].mean()),
        'mean_cos_top20': float(sigma.mean()),
        'grassmann_distance': grassmann,
        'gen_pca20_var_explained': float(explained_gen),
    },
    'experiment_4_layer_variance_sweep': {
        str(k): v for k, v in layer_var_results.items()
    },
}

with open('output/z_basis_projection_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n=== SUMMARY ===")
print(f"Encoding-Z captures {var_fraction:.1%} of generation-time variance")
print(f"Grassmann distance between subspaces: {grassmann:.3f}")
print(f"Mean cos(principal angle) top-10: {sigma[:10].mean():.3f}")
print(f"Full-space spikes: {n_full_spikes} | Z-projected spikes: {n_proj_spikes} | Residual spikes: {n_resid_spikes}")
spike_survival = n_proj_spikes / n_full_spikes * 100 if n_full_spikes > 0 else 0
print(f"Spike survival in Z: {spike_survival:.1f}%")
resid_survival = n_resid_spikes / n_full_spikes * 100 if n_full_spikes > 0 else 0
print(f"Spike survival in residual: {resid_survival:.1f}%")
print("\nSaved to output/z_basis_projection_results.json")
