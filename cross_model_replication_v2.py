"""Cross-model replication v2: Correct Procrustes method (orthogonal, not Ridge).

The original full_depth_profile.py used PCA→100 then orthogonal_procrustes.
Ridge regression with N=200 << d=2048 is massively overparameterized and always gets R²≈1.
This version uses the same methodology as the original Qwen2.5-3B analysis.

Uses cached .npz files from v1 (qwen15b_all_layers.npz, phi2_all_layers.npz) plus
the original all_layers.npz for Qwen2.5-3B.
"""

import numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import json

OUTPUT_DIR = Path("output")


def procrustes_r2(zh, en, n_pca=100):
    """Same method as full_depth_profile.py: center, PCA→n_pca, orthogonal Procrustes."""
    N = zh.shape[0]
    zh_c = zh - zh.mean(0)
    en_c = en - en.mean(0)
    combined = np.vstack([zh_c, en_c])

    n_comp = min(n_pca, N, zh.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(combined)
    zh_p = pca.transform(zh_c)
    en_p = pca.transform(en_c)

    R, _ = orthogonal_procrustes(zh_p, en_p)
    zh_rot = zh_p @ R
    ss_res = np.sum((zh_rot - en_p)**2)
    ss_tot = np.sum((en_p - en_p.mean(0))**2)
    r2 = 1 - ss_res / ss_tot
    return float(r2)


def nn_accuracy(zh, en):
    """Nearest-neighbor matching accuracy (raw, no Procrustes)."""
    N = zh.shape[0]
    nbrs = NearestNeighbors(n_neighbors=1).fit(en)
    _, idx = nbrs.kneighbors(zh)
    return float(np.mean(idx.flatten() == np.arange(N)))


def analyze_model_full(name, npz_path):
    """Run complete analysis pipeline on a model's cached activations."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    data = np.load(npz_path, allow_pickle=True)

    # Detect number of layers
    layer_keys = sorted([k for k in data.keys() if k.startswith('zh_L')])
    n_layers = len(layer_keys)
    d = data[layer_keys[0]].shape[1]
    N = data[layer_keys[0]].shape[0]
    print(f"  {n_layers} layers, d={d}, N={N}")

    results = {"model": name, "n_layers": n_layers, "hidden_dim": d, "n_problems": N}

    # ─── 1. Orthogonal Procrustes R² across all layers ───
    print("\n  Procrustes R² (orthogonal, PCA=100):")
    r2_list = []
    nn_list = []
    for l in range(n_layers):
        zh = data[f'zh_L{l}']
        en = data[f'en_L{l}']
        r2 = procrustes_r2(zh, en, n_pca=100)
        nn = nn_accuracy(zh, en)
        r2_list.append(round(r2, 4))
        nn_list.append(round(nn, 3))

        bar = "#" * int(max(0, r2) * 40)
        label = ""
        if l > 0:
            delta = r2_list[l] - r2_list[l-1]
            if delta < -0.1:
                label = f" ← DROP {delta:+.3f}"
            elif delta > 0.1:
                label = f" ← JUMP {delta:+.3f}"
        print(f"    L{l:2d}: R²={r2:.4f}  NN={nn:.3f}  {bar}{label}")

    results["r2_by_layer"] = r2_list
    results["nn_by_layer"] = nn_list

    # Phase transitions
    deltas = [r2_list[l] - r2_list[l-1] for l in range(1, n_layers)]
    max_drop_idx = np.argmin(deltas)
    max_jump_idx = np.argmax(deltas)

    results["shattering_layer"] = max_drop_idx + 1
    results["shattering_from"] = r2_list[max_drop_idx]
    results["shattering_to"] = r2_list[max_drop_idx + 1]
    results["shattering_drop"] = round(deltas[max_drop_idx], 4)
    results["reassembly_layer"] = max_jump_idx + 1
    results["reassembly_from"] = r2_list[max_jump_idx]
    results["reassembly_to"] = r2_list[max_jump_idx + 1]
    results["reassembly_jump"] = round(deltas[max_jump_idx], 4)

    print(f"\n  Phase transitions:")
    print(f"    Shattering: L{max_drop_idx}→L{max_drop_idx+1}: R² {r2_list[max_drop_idx]:.3f}→{r2_list[max_drop_idx+1]:.3f} (Δ={deltas[max_drop_idx]:+.3f})")
    print(f"    Reassembly: L{max_jump_idx}→L{max_jump_idx+1}: R² {r2_list[max_jump_idx]:.3f}→{r2_list[max_jump_idx+1]:.3f} (Δ={deltas[max_jump_idx]:+.3f})")

    # ─── 2. Backbone neuron detection (per-dim variance) ───
    print(f"\n  Backbone neuron detection:")
    backbone_trajectories = {}
    top5_by_layer = []

    for l in range(n_layers):
        zh = data[f'zh_L{l}']
        en = data[f'en_L{l}']
        combined = np.vstack([zh, en])
        var = np.var(combined, axis=0)
        total = var.sum()
        frac = var / total if total > 0 else var

        sorted_dims = np.argsort(frac)[::-1]
        top5 = sorted_dims[:5].tolist()
        top5_var = [round(float(frac[j]), 4) for j in top5]
        top5_by_layer.append({"dims": top5, "variances": top5_var})

    results["top5_dims_by_layer"] = top5_by_layer

    # Find the most dominant dim in middle layers
    mid_start = max(2, n_layers // 6)
    mid_end = n_layers - max(3, n_layers // 6)

    # Track per-dim variance across all middle layers
    dim_max_in_mid = np.zeros(d)
    for l in range(mid_start, mid_end):
        zh = data[f'zh_L{l}']
        en = data[f'en_L{l}']
        combined = np.vstack([zh, en])
        var = np.var(combined, axis=0)
        total = var.sum()
        frac = var / total if total > 0 else var
        dim_max_in_mid = np.maximum(dim_max_in_mid, frac)

    backbone_dim = int(np.argmax(dim_max_in_mid))
    backbone_peak = float(dim_max_in_mid[backbone_dim])

    # Get full trajectory for backbone dim
    backbone_traj = []
    for l in range(n_layers):
        zh = data[f'zh_L{l}']
        en = data[f'en_L{l}']
        combined = np.vstack([zh, en])
        var = np.var(combined, axis=0)
        total = var.sum()
        frac = float(var[backbone_dim] / total) if total > 0 else 0
        backbone_traj.append(round(frac, 4))

    results["backbone_dim"] = backbone_dim
    results["backbone_peak_variance"] = round(backbone_peak, 4)
    results["backbone_trajectory"] = backbone_traj
    results["has_backbone"] = backbone_peak > 0.20

    print(f"    Backbone candidate: dim {backbone_dim}, peak={backbone_peak:.3f}")
    for l in range(n_layers):
        bar = "█" * int(backbone_traj[l] * 60)
        print(f"      L{l:2d}: {backbone_traj[l]:.3f} {bar}")

    # Check if same top5 dims persist across middle layers (like Qwen's 28-layer lock)
    if mid_end > mid_start:
        mid_top5_sets = [set(top5_by_layer[l]["dims"]) for l in range(mid_start, mid_end)]
        common = mid_top5_sets[0]
        for s in mid_top5_sets[1:]:
            common = common & s
        results["persistent_mid_dims"] = sorted(list(common))
        print(f"    Dims in top-5 throughout L{mid_start}-L{mid_end-1}: {sorted(list(common))}")

    # ─── 3. Z-basis projection (at best alignment layer in upper half) ───
    upper_half_r2 = r2_list[n_layers//2:]
    z_layer = n_layers//2 + np.argmax(upper_half_r2)
    print(f"\n  Z-basis projection (Z-layer = L{z_layer}, R²={r2_list[z_layer]:.3f}):")

    zh_z = data[f'zh_L{z_layer}']
    en_z = data[f'en_L{z_layer}']
    combined_z = np.vstack([zh_z, en_z])

    pca_z = PCA(n_components=min(20, N, d))
    pca_z.fit(combined_z)
    Z_basis = pca_z.components_  # (20, d)
    z_explained = float(sum(pca_z.explained_variance_ratio_))

    # Project
    zh_proj = zh_z @ Z_basis.T
    en_proj = en_z @ Z_basis.T

    # Cosines
    def cos_batch(a, b):
        dot = np.sum(a * b, axis=1)
        return dot / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-10)

    cos_full = cos_batch(zh_z, en_z)
    cos_z = cos_batch(zh_proj, en_proj)

    zh_resid = zh_z - zh_proj @ Z_basis
    en_resid = en_z - en_proj @ Z_basis
    cos_resid = cos_batch(zh_resid, en_resid)

    # Random control
    rng = np.random.RandomState(42)
    rand_mat = rng.randn(20, d).astype(np.float32)
    rand_basis, _ = np.linalg.qr(rand_mat.T)
    rand_basis = rand_basis.T[:20]
    cos_rand = cos_batch(zh_z @ rand_basis.T, en_z @ rand_basis.T)

    print(f"    PCA=20 explains {z_explained:.3f}")
    print(f"    Full-space cosine:  {cos_full.mean():.3f} ± {cos_full.std():.3f}")
    print(f"    Z-projected cosine: {cos_z.mean():.3f} ± {cos_z.std():.3f}")
    print(f"    Residual cosine:    {cos_resid.mean():.3f} ± {cos_resid.std():.3f}")
    print(f"    Random 20-dim:      {cos_rand.mean():.3f} ± {cos_rand.std():.3f}")
    print(f"    Z - Random gap:     {cos_z.mean() - cos_rand.mean():+.3f}")

    results["z_layer"] = z_layer
    results["z_explained_variance"] = round(z_explained, 4)
    results["cosine_full_mean"] = round(float(cos_full.mean()), 4)
    results["cosine_z_mean"] = round(float(cos_z.mean()), 4)
    results["cosine_resid_mean"] = round(float(cos_resid.mean()), 4)
    results["cosine_random_mean"] = round(float(cos_rand.mean()), 4)
    results["z_random_gap"] = round(float(cos_z.mean() - cos_rand.mean()), 4)

    # ─── 4. Two Towers: layer delta Z-fraction ───
    print(f"\n  Layer delta Z-fraction (Two Towers):")
    z_fractions = []

    for l in range(n_layers - 1):
        delta_zh = data[f'zh_L{l+1}'] - data[f'zh_L{l}']
        delta_en = data[f'en_L{l+1}'] - data[f'en_L{l}']
        delta_all = np.vstack([delta_zh, delta_en])

        delta_z = delta_all @ Z_basis.T @ Z_basis
        delta_nonz = delta_all - delta_z

        z_norm = float(np.mean(np.linalg.norm(delta_z, axis=1)))
        nonz_norm = float(np.mean(np.linalg.norm(delta_nonz, axis=1)))
        total = z_norm + nonz_norm
        z_frac = z_norm / total if total > 0 else 0
        z_fractions.append(round(z_frac, 3))

        bar_z = "Z" * int(z_frac * 30)
        bar_n = "·" * (30 - len(bar_z))
        label = ""
        if z_frac > 0.7:
            label = " ← HIGH Z"
        elif z_frac > 0.5:
            label = " ← Z-dominant"
        print(f"    L{l:2d}→{l+1:2d}: Z={z_frac:.3f} [{bar_z}{bar_n}]{label}")

    results["z_fractions"] = z_fractions
    towers = [l for l in range(len(z_fractions)) if z_fractions[l] > 0.7]
    results["tower_layers"] = towers
    results["has_two_towers"] = len(towers) >= 2

    # ─── 5. Anti-backbone (final layer) ───
    print(f"\n  Anti-backbone (final layer):")
    final = n_layers - 1
    zh_f = data[f'zh_L{final}']
    en_f = data[f'en_L{final}']
    combined_f = np.vstack([zh_f, en_f])
    var_f = np.var(combined_f, axis=0)
    total_f = var_f.sum()
    frac_f = var_f / total_f
    top_final = int(np.argmax(frac_f))

    zh_val = float(zh_f[:, top_final].mean())
    en_val = float(en_f[:, top_final].mean())
    corr = float(np.corrcoef(zh_f[:, top_final], en_f[:, top_final])[0, 1])
    polarity = "OPPOSITE" if np.sign(zh_val) != np.sign(en_val) else "SAME"

    # Check if absent in middle
    final_dim_traj = []
    for l in range(n_layers):
        zh_l = data[f'zh_L{l}']
        en_l = data[f'en_L{l}']
        c = np.vstack([zh_l, en_l])
        v = np.var(c, axis=0)
        t = v.sum()
        final_dim_traj.append(round(float(v[top_final] / t) if t > 0 else 0, 4))

    absent = max(final_dim_traj[mid_start:mid_end]) < 0.01

    print(f"    Top dim at L{final}: dim {top_final} ({frac_f[top_final]:.3f} variance)")
    print(f"    zh={zh_val:+.1f}, en={en_val:+.1f}, corr={corr:.3f}, polarity={polarity}")
    print(f"    Absent in middle (<1%): {absent}")

    results["anti_backbone_dim"] = top_final
    results["anti_backbone_variance"] = round(float(frac_f[top_final]), 4)
    results["anti_backbone_zh_mean"] = round(zh_val, 2)
    results["anti_backbone_en_mean"] = round(en_val, 2)
    results["anti_backbone_corr"] = round(corr, 4)
    results["anti_backbone_polarity"] = polarity
    results["anti_backbone_absent_middle"] = absent
    results["anti_backbone_trajectory"] = final_dim_traj

    return results


def main():
    all_results = {}

    # Qwen2.5-3B (reference — already have data)
    qwen3b_path = OUTPUT_DIR / "all_layers.npz"
    if qwen3b_path.exists():
        all_results["Qwen2.5-3B"] = analyze_model_full("Qwen2.5-3B (base)", qwen3b_path)

    # Qwen2.5-1.5B
    qwen15b_path = OUTPUT_DIR / "qwen15b_all_layers.npz"
    if qwen15b_path.exists():
        all_results["Qwen2.5-1.5B"] = analyze_model_full("Qwen2.5-1.5B (base)", qwen15b_path)

    # phi-2
    phi2_path = OUTPUT_DIR / "phi2_all_layers.npz"
    if phi2_path.exists():
        all_results["phi-2"] = analyze_model_full("phi-2", phi2_path)

    # ─── Print comparison table ───
    print(f"\n{'='*70}")
    print("  CROSS-MODEL COMPARISON TABLE")
    print(f"{'='*70}")

    headers = ["Metric", "Qwen2.5-3B", "Qwen2.5-1.5B", "phi-2"]

    def get(model, key, fmt="{}", default="N/A"):
        if model in all_results and key in all_results[model]:
            return fmt.format(all_results[model][key])
        return default

    rows = [
        ("Layers / dim",
         f"{get('Qwen2.5-3B', 'n_layers')}/{get('Qwen2.5-3B', 'hidden_dim')}",
         f"{get('Qwen2.5-1.5B', 'n_layers')}/{get('Qwen2.5-1.5B', 'hidden_dim')}",
         f"{get('phi-2', 'n_layers')}/{get('phi-2', 'hidden_dim')}"),
        ("Shattering drop",
         get('Qwen2.5-3B', 'shattering_drop', '{:+.3f}'),
         get('Qwen2.5-1.5B', 'shattering_drop', '{:+.3f}'),
         get('phi-2', 'shattering_drop', '{:+.3f}')),
        ("Shattering layer",
         f"L{get('Qwen2.5-3B', 'shattering_layer')}",
         f"L{get('Qwen2.5-1.5B', 'shattering_layer')}",
         f"L{get('phi-2', 'shattering_layer')}"),
        ("Reassembly jump",
         get('Qwen2.5-3B', 'reassembly_jump', '{:+.3f}'),
         get('Qwen2.5-1.5B', 'reassembly_jump', '{:+.3f}'),
         get('phi-2', 'reassembly_jump', '{:+.3f}')),
        ("Reassembly layer",
         f"L{get('Qwen2.5-3B', 'reassembly_layer')}",
         f"L{get('Qwen2.5-1.5B', 'reassembly_layer')}",
         f"L{get('phi-2', 'reassembly_layer')}"),
        ("Backbone dim",
         f"dim {get('Qwen2.5-3B', 'backbone_dim')}",
         f"dim {get('Qwen2.5-1.5B', 'backbone_dim')}",
         f"dim {get('phi-2', 'backbone_dim')}"),
        ("Backbone peak var",
         get('Qwen2.5-3B', 'backbone_peak_variance', '{:.1%}'),
         get('Qwen2.5-1.5B', 'backbone_peak_variance', '{:.1%}'),
         get('phi-2', 'backbone_peak_variance', '{:.1%}')),
        ("Has backbone (>20%)",
         get('Qwen2.5-3B', 'has_backbone'),
         get('Qwen2.5-1.5B', 'has_backbone'),
         get('phi-2', 'has_backbone')),
        ("Z cosine (projected)",
         get('Qwen2.5-3B', 'cosine_z_mean', '{:.3f}'),
         get('Qwen2.5-1.5B', 'cosine_z_mean', '{:.3f}'),
         get('phi-2', 'cosine_z_mean', '{:.3f}')),
        ("Z-random gap",
         get('Qwen2.5-3B', 'z_random_gap', '{:+.3f}'),
         get('Qwen2.5-1.5B', 'z_random_gap', '{:+.3f}'),
         get('phi-2', 'z_random_gap', '{:+.3f}')),
        ("Has two towers",
         get('Qwen2.5-3B', 'has_two_towers'),
         get('Qwen2.5-1.5B', 'has_two_towers'),
         get('phi-2', 'has_two_towers')),
        ("Tower layers",
         str(all_results.get('Qwen2.5-3B', {}).get('tower_layers', 'N/A')),
         str(all_results.get('Qwen2.5-1.5B', {}).get('tower_layers', 'N/A')),
         str(all_results.get('phi-2', {}).get('tower_layers', 'N/A'))),
        ("Anti-backbone dim",
         f"dim {get('Qwen2.5-3B', 'anti_backbone_dim')}",
         f"dim {get('Qwen2.5-1.5B', 'anti_backbone_dim')}",
         f"dim {get('phi-2', 'anti_backbone_dim')}"),
        ("Anti-backbone polarity",
         get('Qwen2.5-3B', 'anti_backbone_polarity'),
         get('Qwen2.5-1.5B', 'anti_backbone_polarity'),
         get('phi-2', 'anti_backbone_polarity')),
        ("Anti-backbone absent mid",
         get('Qwen2.5-3B', 'anti_backbone_absent_middle'),
         get('Qwen2.5-1.5B', 'anti_backbone_absent_middle'),
         get('phi-2', 'anti_backbone_absent_middle')),
    ]

    # Print formatted table
    col_widths = [max(len(str(row[i])) for row in rows + [tuple(headers)]) for i in range(4)]
    col_widths = [max(w, 15) for w in col_widths]

    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(v).center(w) for v, w in zip(row, col_widths)))

    # Save
    outpath = OUTPUT_DIR / "cross_model_results_v2.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
