"""Procrustes preview: CPU-only diagnostic.

Computes Procrustes convention directions from the multilingual cache
WITHOUT loading the model. Compares to MS1's mean-diff e_c direction
to predict whether MS3 surgery will be targeted (like MS1) or destructive
(like MS2b's centroid SVD).

Key questions:
1. How large is the Procrustes rotation at each layer?
2. Is sv1 of (R_avg - I) dominant? (high sv1/sv2 = clean convention axis)
3. Does the Procrustes direction align with MS1's mean-diff direction?
4. Does it align with the centroid SVD direction (the destructive one)?
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.sparse.linalg import svds

LANGS = ['en', 'zh', 'ar', 'es', 'ja', 'ko', 'sw']
N_LAYERS = 36
D_MODEL = 2048

def main():
    cache_path = "output/multilingual_all_layers.npz"
    print("Loading cache...", flush=True)
    data = np.load(cache_path, allow_pickle=True)

    # Precompute MS1-style mean-diff direction (EN - ZH) per layer
    ms1_dirs = {}
    # Centroid SVD direction per layer (what MS2b uses)
    centroid_svd_dirs = {}
    # Procrustes direction per layer
    proc_dirs = {}

    print(f"\n{'L':>3s} | {'Proc sv1':>9s} | {'sv1/sv2':>7s} | "
          f"{'cos(proc,ms1)':>13s} | {'cos(proc,csvd)':>14s} | {'cos(ms1,csvd)':>13s} | "
          f"{'mean_rot°':>9s} | {'max_pair_rot°':>13s}", flush=True)
    print("-" * 110, flush=True)

    for L in range(N_LAYERS):
        lang_acts = {}
        for lang in LANGS:
            lang_acts[lang] = data[f"{lang}_L{L}"].astype(np.float64)

        # --- MS1 mean-diff direction (EN - ZH) ---
        en_mean = lang_acts['en'].mean(axis=0)
        zh_mean = lang_acts['zh'].mean(axis=0)
        e_c_ms1 = en_mean - zh_mean
        e_c_ms1 /= np.linalg.norm(e_c_ms1) + 1e-12
        ms1_dirs[L] = e_c_ms1

        # --- Centroid SVD direction (what LOO/MS2b uses) ---
        all_means = np.stack([lang_acts[l].mean(axis=0) for l in LANGS])  # (7, 2048)
        centroid = all_means.mean(axis=0)
        deviations = all_means - centroid  # (7, 2048)
        U, S_c, Vt_c = svds(deviations, k=2, which='LM')
        idx_c = np.argsort(-S_c)
        e_c_csvd = Vt_c[idx_c[0]]  # top right singular vector
        centroid_svd_dirs[L] = e_c_csvd

        # --- Procrustes convention direction ---
        R_sum = np.zeros((D_MODEL, D_MODEL))
        n_pairs = 0
        pair_angles = []

        for i, li in enumerate(LANGS):
            for j, lj in enumerate(LANGS):
                if i >= j:
                    continue
                A = lang_acts[li]
                B = lang_acts[lj]
                A_c = A - A.mean(axis=0)
                B_c = B - B.mean(axis=0)
                R, _ = orthogonal_procrustes(A_c, B_c)
                R_sum += R
                n_pairs += 1

                # Frobenius deviation from identity
                frob_dev = np.linalg.norm(R - np.eye(D_MODEL), 'fro')
                pair_angles.append(frob_dev)

        R_avg = R_sum / n_pairs
        deviation = R_avg - np.eye(D_MODEL)
        U_p, S_p, Vt_p = svds(deviation, k=3, which='LM')
        idx_p = np.argsort(-S_p)
        S_p = S_p[idx_p]
        Vt_p = Vt_p[idx_p]
        e_c_proc = Vt_p[0]
        proc_dirs[L] = e_c_proc

        # Cosine similarities
        cos_proc_ms1 = abs(np.dot(e_c_proc, e_c_ms1))
        cos_proc_csvd = abs(np.dot(e_c_proc, e_c_csvd))
        cos_ms1_csvd = abs(np.dot(e_c_ms1, e_c_csvd))

        mean_frob = np.mean(pair_angles)
        max_frob = np.max(pair_angles)

        print(f"L{L:2d} | {S_p[0]:9.4f} | {S_p[0]/S_p[1]:7.2f} | "
              f"{cos_proc_ms1:13.4f} | {cos_proc_csvd:14.4f} | {cos_ms1_csvd:13.4f} | "
              f"{mean_frob:9.3f} | {max_frob:13.3f}", flush=True)

    # Phase summary
    print("\n\n=== PHASE SUMMARY ===")
    print("\nKey layers from prior work:")
    print(f"  L5, L12: convention breakpoints (C7c)")
    print(f"  L13-L35: surgery range (MS1/MS2b)")
    print(f"  L30: rank-1 bottleneck, convention-free (C7c)")
    print()

    # Summary statistics
    for phase, start, end in [("Early (L0-L4)", 0, 5), ("Convention onset (L5-L12)", 5, 13),
                               ("Surgery range (L13-L29)", 13, 30), ("Bottleneck (L30-L35)", 30, 36)]:
        proc_ms1_cos = [abs(np.dot(proc_dirs[l], ms1_dirs[l])) for l in range(start, end)]
        proc_csvd_cos = [abs(np.dot(proc_dirs[l], centroid_svd_dirs[l])) for l in range(start, end)]
        ms1_csvd_cos = [abs(np.dot(ms1_dirs[l], centroid_svd_dirs[l])) for l in range(start, end)]
        print(f"  {phase}:")
        print(f"    cos(proc, ms1) mean={np.mean(proc_ms1_cos):.4f}")
        print(f"    cos(proc, csvd) mean={np.mean(proc_csvd_cos):.4f}")
        print(f"    cos(ms1, csvd) mean={np.mean(ms1_csvd_cos):.4f}")

    print("\n\nPREDICTION:")
    surgery_range = range(13, 36)
    mean_cos_proc_ms1 = np.mean([abs(np.dot(proc_dirs[l], ms1_dirs[l])) for l in surgery_range])
    mean_cos_proc_csvd = np.mean([abs(np.dot(proc_dirs[l], centroid_svd_dirs[l])) for l in surgery_range])
    if mean_cos_proc_ms1 > 0.5:
        print(f"  Procrustes direction ALIGNS with MS1 mean-diff (cos={mean_cos_proc_ms1:.3f})")
        print(f"  → PREDICT: MS3 will be targeted like MS1, not destructive like MS2b")
    elif mean_cos_proc_csvd > 0.5:
        print(f"  Procrustes direction ALIGNS with centroid SVD (cos={mean_cos_proc_csvd:.3f})")
        print(f"  → PREDICT: MS3 will be destructive like MS2b")
    else:
        print(f"  Procrustes direction is DISTINCT from both MS1 (cos={mean_cos_proc_ms1:.3f}) and csvd (cos={mean_cos_proc_csvd:.3f})")
        print(f"  → PREDICT: MS3 result is novel — could go either way")


if __name__ == "__main__":
    main()
