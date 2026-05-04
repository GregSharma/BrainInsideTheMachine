"""Fast Procrustes preview: PCA-accelerated, CPU-only.

Instead of fitting Procrustes in full 2048D (requires 756 SVDs of 2048×2048),
project to 200D first (= n_problems, the data rank), then fit in reduced space.
Convention direction is projected back to 2048D for cosine comparisons.

Also computes the MS1 mean-diff and centroid SVD directions for comparison.
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes
import time

LANGS = ['en', 'zh', 'ar', 'es', 'ja', 'ko', 'sw']
N_LAYERS = 36
D_MODEL = 2048
N_PROBLEMS = 200
PCA_DIM = 200  # project to this before Procrustes (= data rank)


def main():
    cache_path = "output/multilingual_all_layers.npz"
    print("Loading cache...", flush=True)
    data = np.load(cache_path, allow_pickle=True)
    t0 = time.time()

    print(f"\n{'L':>3s} | {'Proc sv1':>9s} | {'sv1/sv2':>7s} | "
          f"{'cos(P,ms1)':>10s} | {'cos(P,csvd)':>11s} | {'cos(ms1,csvd)':>13s} | "
          f"{'mean_frob':>9s}", flush=True)
    print("-" * 90, flush=True)

    proc_results = []

    for L in range(N_LAYERS):
        # Load per-language activations
        lang_acts = {}
        for lang in LANGS:
            lang_acts[lang] = data[f"{lang}_L{L}"].astype(np.float64)

        # --- PCA basis from combined data ---
        combined = np.vstack([lang_acts[l] for l in LANGS])  # (1400, 2048)
        combined_c = combined - combined.mean(axis=0)
        # Economy SVD: U (1400, 200), S (200,), Vt (200, 2048)
        U, S_pca, Vt_pca = np.linalg.svd(combined_c, full_matrices=False)
        # Project each language to PCA space
        basis = Vt_pca[:PCA_DIM]  # (200, 2048)

        lang_proj = {}
        for lang in LANGS:
            centered = lang_acts[lang] - lang_acts[lang].mean(axis=0)
            lang_proj[lang] = centered @ basis.T  # (200, 200)

        # --- Procrustes in PCA space ---
        R_sum = np.zeros((PCA_DIM, PCA_DIM))
        n_pairs = 0
        pair_frobs = []

        for i, li in enumerate(LANGS):
            for j, lj in enumerate(LANGS):
                if i >= j:
                    continue
                R, _ = orthogonal_procrustes(lang_proj[li], lang_proj[lj])
                R_sum += R
                n_pairs += 1
                pair_frobs.append(np.linalg.norm(R - np.eye(PCA_DIM), 'fro'))

        R_avg = R_sum / n_pairs
        deviation = R_avg - np.eye(PCA_DIM)
        U_d, S_d, Vt_d = np.linalg.svd(deviation, full_matrices=False)
        # Top convention direction in PCA space → project back to full space
        e_c_proc_pca = Vt_d[0]  # (200,)
        e_c_proc = e_c_proc_pca @ basis  # (2048,)
        e_c_proc /= np.linalg.norm(e_c_proc)

        sv1 = S_d[0]
        sv2 = S_d[1] if len(S_d) > 1 else 1e-10
        sv_ratio = sv1 / sv2 if sv2 > 1e-10 else float('inf')

        # --- MS1 mean-diff (EN - ZH) ---
        en_mean = lang_acts['en'].mean(axis=0)
        zh_mean = lang_acts['zh'].mean(axis=0)
        e_c_ms1 = en_mean - zh_mean
        e_c_ms1 /= np.linalg.norm(e_c_ms1) + 1e-12

        # --- Centroid SVD ---
        all_means = np.stack([lang_acts[l].mean(axis=0) for l in LANGS])
        centroid = all_means.mean(axis=0)
        devs = all_means - centroid
        _, _, Vt_c = np.linalg.svd(devs, full_matrices=False)
        e_c_csvd = Vt_c[0]

        # --- Cosines ---
        cos_proc_ms1 = abs(np.dot(e_c_proc, e_c_ms1))
        cos_proc_csvd = abs(np.dot(e_c_proc, e_c_csvd))
        cos_ms1_csvd = abs(np.dot(e_c_ms1, e_c_csvd))

        mean_frob = np.mean(pair_frobs)

        print(f"L{L:2d} | {sv1:9.4f} | {sv_ratio:7.2f} | "
              f"{cos_proc_ms1:10.4f} | {cos_proc_csvd:11.4f} | {cos_ms1_csvd:13.4f} | "
              f"{mean_frob:9.3f}", flush=True)

        proc_results.append({
            'layer': L, 'sv1': sv1, 'sv2': sv2, 'sv_ratio': sv_ratio,
            'cos_proc_ms1': cos_proc_ms1, 'cos_proc_csvd': cos_proc_csvd,
            'cos_ms1_csvd': cos_ms1_csvd, 'mean_frob': mean_frob,
        })

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s", flush=True)

    # Phase summary
    print(f"\n{'=' * 70}")
    print("PHASE SUMMARY")
    print(f"{'=' * 70}")
    for phase, start, end in [
        ("Early (L0-L4)", 0, 5),
        ("Convention onset (L5-L12)", 5, 13),
        ("Surgery range (L13-L29)", 13, 30),
        ("Bottleneck (L30-L35)", 30, 36),
    ]:
        r = proc_results[start:end]
        print(f"\n  {phase}:")
        print(f"    sv1/sv2 mean:       {np.mean([x['sv_ratio'] for x in r]):.3f}")
        print(f"    cos(proc, ms1):     {np.mean([x['cos_proc_ms1'] for x in r]):.4f}")
        print(f"    cos(proc, csvd):    {np.mean([x['cos_proc_csvd'] for x in r]):.4f}")
        print(f"    cos(ms1, csvd):     {np.mean([x['cos_ms1_csvd'] for x in r]):.4f}")
        print(f"    mean Frob dev:      {np.mean([x['mean_frob'] for x in r]):.3f}")

    # Find layer with highest sv1/sv2
    best_layer = max(proc_results, key=lambda x: x['sv_ratio'])
    print(f"\n  Peak sv1/sv2: L{best_layer['layer']} = {best_layer['sv_ratio']:.3f}")

    print(f"\n{'=' * 70}")
    print("PREDICTION FOR MS3 SURGERY")
    print(f"{'=' * 70}")
    surgery_r = proc_results[13:36]
    mean_ratio = np.mean([x['sv_ratio'] for x in surgery_r])
    mean_cos_ms1 = np.mean([x['cos_proc_ms1'] for x in surgery_r])
    mean_cos_csvd = np.mean([x['cos_proc_csvd'] for x in surgery_r])

    if mean_ratio < 1.5:
        print(f"  sv1/sv2 = {mean_ratio:.2f} in surgery range: FLAT SPECTRUM")
        print(f"  → No clean convention axis. Procrustes direction is arbitrary.")
        print(f"  → PREDICT: MS3 results will be NOISY (neither clearly helpful nor destructive)")
    elif mean_cos_ms1 > 0.5:
        print(f"  Procrustes ALIGNS with MS1 mean-diff (cos={mean_cos_ms1:.3f})")
        print(f"  → PREDICT: MS3 will be targeted like MS1 (+improvement)")
    elif mean_cos_csvd > 0.5:
        print(f"  Procrustes ALIGNS with centroid SVD (cos={mean_cos_csvd:.3f})")
        print(f"  → PREDICT: MS3 will be destructive like MS2b")
    else:
        print(f"  Procrustes is DISTINCT (cos_ms1={mean_cos_ms1:.3f}, cos_csvd={mean_cos_csvd:.3f})")
        print(f"  sv1/sv2={mean_ratio:.2f}: {'sharp' if mean_ratio > 2 else 'moderate'} convention axis")
        print(f"  → PREDICT: MS3 result is novel")


if __name__ == "__main__":
    main()
