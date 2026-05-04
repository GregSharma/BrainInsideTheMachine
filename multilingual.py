"""Multilingual Transfer Test: Does contrastive Z generalize beyond zh/en?

Run with: MPLBACKEND=Agg .venv/bin/python multilingual.py

The kill shot for the shared-token objection:
  1. Build contrastive Z from zh/en ONLY (same as before)
  2. Extract activations for Spanish, Arabic, Japanese, Swahili, Korean
  3. Project ALL languages into that zh/en-derived Z
  4. Test cross-lingual probe transfer on EVERY language pair
  5. If sw→ku transfer works in Z built from zh/en, it's about math, not tokens

Languages chosen to maximize tokenization diversity:
  - es (Spanish): Latin script, Romance
  - ar (Arabic): Arabic script, RTL
  - ja (Japanese): CJK but NOT Chinese
  - ko (Korean): Hangul script
  - sw (Swahili): Latin script, Bantu — maximally distant from zh
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import orthogonal_procrustes
import random as pyrandom
import itertools

from utils import get_model_dims

MODEL_NAME = "Qwen/Qwen2.5-3B"
K = 20
LAYER = 32
N_RANDOM = 50
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42

# All languages we test
LANGUAGES = ["zh", "en", "es", "ar", "ja", "ko", "sw"]
LANG_NAMES = {
    "zh": "Chinese", "en": "English", "es": "Spanish", "ar": "Arabic",
    "ja": "Japanese", "ko": "Korean", "sw": "Swahili"
}
LANG_COLORS = {
    "zh": "#e74c3c", "en": "#3498db", "es": "#2ecc71", "ar": "#f39c12",
    "ja": "#9b59b6", "ko": "#1abc9c", "sw": "#e67e22"
}

CAT_NAMES = ["arithmetic", "combinatorics", "modular", "geometry", "sequences"]


def generate_problems(n=200, seed=42):
    """Generate 200 math problems in ALL languages."""
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            p = {
                "zh": f"计算 {a} + {b} 的值。",
                "en": f"Calculate {a} + {b}.",
                "es": f"Calcula {a} + {b}.",
                "ar": f"احسب {a} + {b}.",
                "ja": f"{a} + {b} の値を計算してください。",
                "ko": f"{a} + {b}의 값을 계산하세요.",
                "sw": f"Hesabu {a} + {b}.",
            }
        else:
            p = {
                "zh": f"计算 {a} × {b} 的值。",
                "en": f"Calculate {a} × {b}.",
                "es": f"Calcula {a} × {b}.",
                "ar": f"احسب {a} × {b}.",
                "ja": f"{a} × {b} の値を計算してください。",
                "ko": f"{a} × {b}의 값을 계산하세요.",
                "sw": f"Hesabu {a} × {b}.",
            }
        p["category"] = 0
        problems.append(p)

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        p = {
            "zh": f"求组合数 C({n_val}, {k_val}) 的值。",
            "en": f"Find the value of C({n_val}, {k_val}).",
            "es": f"Halla el valor de C({n_val}, {k_val}).",
            "ar": f"أوجد قيمة C({n_val}, {k_val}).",
            "ja": f"C({n_val}, {k_val}) の値を求めてください。",
            "ko": f"C({n_val}, {k_val})의 값을 구하세요.",
            "sw": f"Tafuta thamani ya C({n_val}, {k_val}).",
            "category": 1,
        }
        problems.append(p)

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        p = {
            "zh": f"{a} 除以 {b} 的余数是多少？",
            "en": f"What is the remainder when {a} is divided by {b}?",
            "es": f"¿Cuál es el residuo de dividir {a} entre {b}?",
            "ar": f"ما هو باقي قسمة {a} على {b}؟",
            "ja": f"{a} を {b} で割った余りはいくつですか？",
            "ko": f"{a}을(를) {b}로 나눈 나머지는 얼마입니까?",
            "sw": f"Baki ya {a} ikigawanywa na {b} ni nini?",
            "category": 2,
        }
        problems.append(p)

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        p = {
            "zh": f"一个长方形的长为 {w}，宽为 {h}，求其面积。",
            "en": f"A rectangle has length {w} and width {h}. Find its area.",
            "es": f"Un rectángulo tiene largo {w} y ancho {h}. Halla su área.",
            "ar": f"مستطيل طوله {w} وعرضه {h}. أوجد مساحته.",
            "ja": f"長さ {w}、幅 {h} の長方形の面積を求めてください。",
            "ko": f"길이 {w}, 너비 {h}인 직사각형의 넓이를 구하세요.",
            "sw": f"Mstatili una urefu {w} na upana {h}. Tafuta eneo lake.",
            "category": 3,
        }
        problems.append(p)

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        p = {
            "zh": f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。",
            "en": f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms.",
            "es": f"Una progresión aritmética tiene primer término {a1} y diferencia común {d}. Halla la suma de los primeros {n_terms} términos.",
            "ar": f"متتالية حسابية حدها الأول {a1} وأساسها {d}. أوجد مجموع أول {n_terms} حدود.",
            "ja": f"初項 {a1}、公差 {d} の等差数列の初めの {n_terms} 項の和を求めてください。",
            "ko": f"첫째항 {a1}, 공차 {d}인 등차수열의 처음 {n_terms}항의 합을 구하세요.",
            "sw": f"Mfuatano wa hesabu una neno la kwanza {a1} na tofauti ya kawaida {d}. Tafuta jumla ya maneno {n_terms} ya kwanza.",
            "category": 4,
        }
        problems.append(p)

    rng.shuffle(problems)
    return problems


def build_contrastive_z(zh_means, en_means, k, var_threshold=0.90):
    """Build Z from zh/en only — the reference subspace."""
    N, d = zh_means.shape
    diffs = zh_means - en_means
    diffs_centered = diffs - diffs.mean(axis=0)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    var_explained = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)
    n_lang = max(1, int(np.searchsorted(cumvar, var_threshold) + 1))
    n_lang = min(n_lang, len(S))
    lang_dirs = Vt[:n_lang]
    proj_out = np.eye(d, dtype=np.float32) - lang_dirs.T @ lang_dirs
    all_means = np.concatenate([zh_means, en_means], axis=0)
    projected = all_means @ proj_out.T
    projected_centered = projected - projected.mean(axis=0)
    _, _, Vt_proj = np.linalg.svd(projected_centered, full_matrices=False)
    actual_k = min(k, Vt_proj.shape[0])
    return Vt_proj[:actual_k].astype(np.float32), n_lang, lang_dirs.astype(np.float32)


def generate_random_basis(d, k, rng):
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].T


def extract_all_activations(model, tokenizer, problems, layer_idx, languages):
    """Extract mean-pooled hidden states for all languages at one layer."""
    d = model.config.hidden_size
    N = len(problems)
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = h_out.detach().cpu().squeeze(0)
        return hook

    hook_handle = model.model.layers[layer_idx].register_forward_hook(make_hook("target"))

    all_means = {}
    for lang in languages:
        means = np.zeros((N, d), dtype=np.float32)
        for i, prob in enumerate(tqdm(problems, desc=f"L{layer_idx} {lang}", leave=False)):
            inputs = tokenizer(prob[lang], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            means[i] = activations["target"].float().numpy().mean(axis=0)
        all_means[lang] = means

    hook_handle.remove()
    return all_means


def apply_procrustes(ref_proj, other_proj):
    """Rotate other_proj onto ref_proj via Procrustes."""
    ref_c = ref_proj - ref_proj.mean(axis=0)
    other_c = other_proj - other_proj.mean(axis=0)
    R, _ = orthogonal_procrustes(other_c, ref_c)
    return other_c @ R + ref_proj.mean(axis=0)


def cross_lingual_probe(proj_train, cats_train, proj_test, cats_test):
    """Train category classifier on one language, test on another."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(proj_train)
    X_test = scaler.transform(proj_test)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, cats_train)
    return clf.score(X_test, cats_test)


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("MULTILINGUAL TRANSFER — Does Z generalize beyond zh/en?")
    print(f"Model: {MODEL_NAME}, k={K}, Layer: {LAYER}")
    print(f"Languages: {', '.join(LANG_NAMES[l] for l in LANGUAGES)}")
    print("=" * 70)

    problems = generate_problems(200, seed=SEED)
    categories = np.array([p["category"] for p in problems])
    N = len(problems)

    # Try loading cached activations
    cache_path = OUTPUT_DIR / "multilingual_activations.npz"
    if cache_path.exists():
        print(f"\nLoading cached activations from {cache_path}...")
        data = np.load(cache_path)
        all_means = {lang: data[lang] for lang in LANGUAGES}
        print("  Loaded!")
    else:
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, device_map="auto",
        )
        model.eval()

        print(f"\nExtracting activations for {len(LANGUAGES)} languages at L{LAYER}...")
        all_means = extract_all_activations(model, tokenizer, problems, LAYER, LANGUAGES)

        np.savez(cache_path, **all_means)
        print(f"\nActivations saved to {cache_path}")

    d = all_means["zh"].shape[1]

    # ====================================================================
    # STEP 1: Build contrastive Z from zh/en ONLY
    # ====================================================================
    print("\n--- Building contrastive Z from zh/en only ---")
    z_basis, n_lang, lang_dirs = build_contrastive_z(
        all_means["zh"], all_means["en"], K
    )
    print(f"  Language dims removed: {n_lang}")
    print(f"  Z dimensions: {K}")

    # ====================================================================
    # STEP 2: Project ALL languages into Z, Procrustes onto zh
    # ====================================================================
    print("\n--- Projecting all languages into Z ---")
    zh_proj = all_means["zh"] @ z_basis.T  # reference
    projections = {"zh": zh_proj}
    for lang in LANGUAGES:
        if lang == "zh":
            continue
        raw_proj = all_means[lang] @ z_basis.T
        projections[lang] = apply_procrustes(zh_proj, raw_proj)

    # ====================================================================
    # STEP 3: Cross-lingual probe transfer matrix
    # ====================================================================
    print("\n--- Cross-lingual probe transfer ---")
    # Stratified split
    train_idx, test_idx = [], []
    for cat in range(5):
        cat_indices = np.where(categories == cat)[0]
        np.random.default_rng(SEED).shuffle(cat_indices)
        half = len(cat_indices) // 2
        train_idx.extend(cat_indices[:half].tolist())
        test_idx.extend(cat_indices[half:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    n_langs = len(LANGUAGES)
    transfer_matrix = np.zeros((n_langs, n_langs))

    for i, train_lang in enumerate(LANGUAGES):
        for j, test_lang in enumerate(LANGUAGES):
            acc = cross_lingual_probe(
                projections[train_lang][train_idx], categories[train_idx],
                projections[test_lang][test_idx], categories[test_idx],
            )
            transfer_matrix[i, j] = acc

    print("\nTransfer matrix (train on row, test on col):")
    header = "         " + "  ".join(f"{l:>5s}" for l in LANGUAGES)
    print(header)
    for i, lang in enumerate(LANGUAGES):
        row = f"  {lang:>5s}  " + "  ".join(f"{transfer_matrix[i,j]:.2f}" for j in range(n_langs))
        print(row)

    # ====================================================================
    # STEP 4: Random baseline for non-zh/en pairs
    # ====================================================================
    print("\n--- Random baseline ---")
    non_zhen_pairs = [(i, j) for i in range(n_langs) for j in range(n_langs)
                       if LANGUAGES[i] not in ("zh", "en") and LANGUAGES[j] not in ("zh", "en")
                       and i != j]

    random_transfer_means = np.zeros((n_langs, n_langs))
    for _ in tqdm(range(N_RANDOM), desc="Random baselines"):
        rand_basis = generate_random_basis(d, K, rng)
        rand_projs = {}
        zh_rand = all_means["zh"] @ rand_basis.T
        for lang in LANGUAGES:
            raw = all_means[lang] @ rand_basis.T
            rand_projs[lang] = apply_procrustes(zh_rand, raw) if lang != "zh" else zh_rand
        for i, train_lang in enumerate(LANGUAGES):
            for j, test_lang in enumerate(LANGUAGES):
                if i == j:
                    continue
                acc = cross_lingual_probe(
                    rand_projs[train_lang][train_idx], categories[train_idx],
                    rand_projs[test_lang][test_idx], categories[test_idx],
                )
                random_transfer_means[i, j] += acc / N_RANDOM

    # ====================================================================
    # STEP 5: Within-category nearest-neighbor matching (THE HARD TEST)
    # ====================================================================
    print("\n--- Within-category nearest-neighbor matching ---")
    print("  (Train-lang problem matched to nearest test-lang problem WITHIN same category)")

    def within_cat_nn_accuracy(proj_train, proj_test, categories):
        """For each test point, find nearest train point within same category. Accuracy = correct match."""
        N = len(categories)
        correct = 0
        for i in range(N):
            cat = categories[i]
            cat_mask = categories == cat
            cat_indices = np.where(cat_mask)[0]
            # Distances from test[i] to all train points in same category
            dists = np.linalg.norm(proj_train[cat_mask] - proj_test[i], axis=1)
            nearest_cat_idx = cat_indices[np.argmin(dists)]
            if nearest_cat_idx == i:  # correct match = same problem index
                correct += 1
        return correct / N

    nn_matrix = np.zeros((n_langs, n_langs))
    for i, lang_a in enumerate(LANGUAGES):
        for j, lang_b in enumerate(LANGUAGES):
            if i == j:
                nn_matrix[i, j] = 1.0
                continue
            nn_matrix[i, j] = within_cat_nn_accuracy(
                projections[lang_a], projections[lang_b], categories
            )

    print("\nWithin-category NN matching (train→test):")
    header = "         " + "  ".join(f"{l:>5s}" for l in LANGUAGES)
    print(header)
    for i, lang in enumerate(LANGUAGES):
        row = f"  {lang:>5s}  " + "  ".join(f"{nn_matrix[i,j]:.2f}" for j in range(n_langs))
        print(row)

    # Random baseline for NN matching
    print("\n  Computing random baseline for NN matching...")
    nn_random_means = np.zeros((n_langs, n_langs))
    for r in tqdm(range(N_RANDOM), desc="Random NN baselines"):
        rand_basis = generate_random_basis(d, K, rng)
        rand_projs = {}
        zh_rand = all_means["zh"] @ rand_basis.T
        for lang in LANGUAGES:
            raw = all_means[lang] @ rand_basis.T
            rand_projs[lang] = apply_procrustes(zh_rand, raw) if lang != "zh" else zh_rand
        for i, lang_a in enumerate(LANGUAGES):
            for j, lang_b in enumerate(LANGUAGES):
                if i == j:
                    continue
                nn_random_means[i, j] += within_cat_nn_accuracy(
                    rand_projs[lang_a], rand_projs[lang_b], categories
                ) / N_RANDOM

    # Key metrics
    novel_pairs = [(i, j) for i in range(n_langs) for j in range(n_langs)
                    if LANGUAGES[i] not in ("zh", "en") and LANGUAGES[j] not in ("zh", "en")
                    and i != j]
    z_novel_nn = np.mean([nn_matrix[i, j] for i, j in novel_pairs])
    rand_novel_nn = np.mean([nn_random_means[i, j] for i, j in novel_pairs])

    all_off = [(i, j) for i in range(n_langs) for j in range(n_langs) if i != j]
    z_all_nn = np.mean([nn_matrix[i, j] for i, j in all_off])
    rand_all_nn = np.mean([nn_random_means[i, j] for i, j in all_off])

    # Chance = 1/40 for within-category matching (40 problems per category)
    chance_nn = 1.0 / 40.0

    print(f"\n  Within-cat NN matching (novel pairs): Z={z_novel_nn:.3f}, Random={rand_novel_nn:.3f}, Chance={chance_nn:.3f}")
    print(f"  Within-cat NN matching (all pairs):   Z={z_all_nn:.3f}, Random={rand_all_nn:.3f}")

    # Also compute category-level metrics for completeness
    z_novel = np.mean([transfer_matrix[i, j] for i, j in novel_pairs])
    rand_novel = np.mean([random_transfer_means[i, j] for i, j in novel_pairs])
    z_all = np.mean([transfer_matrix[i, j] for i, j in all_off])
    rand_all = np.mean([random_transfer_means[i, j] for i, j in all_off])

    print(f"\n  Category probe (novel pairs):  Z={z_novel:.3f}, Random={rand_novel:.3f} (ceiling'd)")
    print(f"  Category probe (all pairs):    Z={z_all:.3f}, Random={rand_all:.3f} (ceiling'd)")

    # ====================================================================
    # STEP 6: Pair distance analysis
    # ====================================================================
    print("\n--- Cross-lingual pair distances ---")
    same_problem_dists = {}
    for lang in LANGUAGES:
        if lang == "zh":
            continue
        dists = np.linalg.norm(projections["zh"] - projections[lang], axis=1)
        same_problem_dists[lang] = np.median(dists)
        print(f"  zh↔{lang}: median={np.median(dists):.1f}, mean={np.mean(dists):.1f}")

    # ====================================================================
    # FIGURES
    # ====================================================================

    # Figure 5: NN matching matrix (the hard test)
    print("\n--- Figure 5: Within-Category NN Matching ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vmax = max(nn_matrix.max(), nn_random_means.max(), 0.5)
    im1 = ax1.imshow(nn_matrix, cmap="RdYlGn", vmin=0, vmax=vmax,
                      interpolation="nearest")
    ax1.set_xticks(range(n_langs))
    ax1.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES], rotation=45, ha="right")
    ax1.set_yticks(range(n_langs))
    ax1.set_yticklabels([LANG_NAMES[l] for l in LANGUAGES])
    ax1.set_title("Contrastive Z (built from zh/en only)", fontsize=12, fontweight="bold")
    for i in range(n_langs):
        for j in range(n_langs):
            ax1.text(j, i, f"{nn_matrix[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="black" if nn_matrix[i,j] > 0.3 else "white")
    fig.colorbar(im1, ax=ax1, label="NN Match Accuracy")

    im2 = ax2.imshow(nn_random_means, cmap="RdYlGn", vmin=0, vmax=vmax,
                      interpolation="nearest")
    ax2.set_xticks(range(n_langs))
    ax2.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES], rotation=45, ha="right")
    ax2.set_yticks(range(n_langs))
    ax2.set_yticklabels([LANG_NAMES[l] for l in LANGUAGES])
    ax2.set_title(f"Random {K}-dim Subspace (mean of {N_RANDOM})", fontsize=12, fontweight="bold")
    for i in range(n_langs):
        for j in range(n_langs):
            ax2.text(j, i, f"{nn_random_means[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="black" if nn_random_means[i,j] > 0.3 else "white")
    fig.colorbar(im2, ax=ax2, label="NN Match Accuracy")

    fig.suptitle(f"Within-Category Problem Matching Across 7 Languages\n"
                 f"Novel pairs (no zh/en): Z={z_novel_nn:.3f} vs Random={rand_novel_nn:.3f} (chance={chance_nn:.3f})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_multilingual_transfer.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'fig5_multilingual_transfer.png'}")
    plt.close(fig)

    # Figure 6: 7-language t-SNE
    print("\n--- Figure 6: 7-Language t-SNE ---")
    all_proj = np.concatenate([projections[lang] for lang in LANGUAGES], axis=0)
    perplexity = min(30, all_proj.shape[0] // 2 - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
                max_iter=1000, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(all_proj)

    fig, ax = plt.subplots(figsize=(12, 12))
    for lang_idx, lang in enumerate(LANGUAGES):
        start = lang_idx * N
        end = start + N
        lang_coords = coords[start:end]
        ax.scatter(lang_coords[:, 0], lang_coords[:, 1],
                   c=LANG_COLORS[lang], s=30, alpha=0.6,
                   edgecolors="black", linewidths=0.2,
                   label=LANG_NAMES[lang])

    # Draw lines connecting same problem across all languages to zh
    for i in range(N):
        zh_coord = coords[i]  # zh is first
        for lang_idx in range(1, len(LANGUAGES)):
            other_coord = coords[lang_idx * N + i]
            ax.plot([zh_coord[0], other_coord[0]], [zh_coord[1], other_coord[1]],
                    color="gray", alpha=0.03, linewidth=0.3)

    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_title(f"7 Languages in Contrastive Z (k={K}, L{LAYER})\n"
                 f"Z built from zh/en only — other 5 languages projected in",
                 fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.02, 0.98, f"Novel NN match: {z_novel_nn:.3f}\n"
                         f"Random baseline: {rand_novel_nn:.3f}\n"
                         f"Chance: {chance_nn:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="goldenrod", alpha=0.9))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_multilingual_tsne.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'fig6_multilingual_tsne.png'}")
    plt.close(fig)

    # Save results
    results = {
        "nn_matrix": {
            LANGUAGES[i]: {LANGUAGES[j]: float(nn_matrix[i, j])
                           for j in range(n_langs)}
            for i in range(n_langs)
        },
        "nn_random_matrix": {
            LANGUAGES[i]: {LANGUAGES[j]: float(nn_random_means[i, j])
                           for j in range(n_langs)}
            for i in range(n_langs)
        },
        "nn_novel_z": float(z_novel_nn),
        "nn_novel_random": float(rand_novel_nn),
        "nn_all_z": float(z_all_nn),
        "nn_all_random": float(rand_all_nn),
        "nn_chance": float(chance_nn),
        "category_transfer_matrix": {
            LANGUAGES[i]: {LANGUAGES[j]: float(transfer_matrix[i, j])
                           for j in range(n_langs)}
            for i in range(n_langs)
        },
        "same_problem_median_dists": {k: float(v) for k, v in same_problem_dists.items()},
        "n_lang_dims": int(n_lang),
        "k": K,
        "layer": LAYER,
    }

    with open(OUTPUT_DIR / "multilingual_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'multilingual_results.json'}")

    print("\n" + "=" * 70)
    print("MULTILINGUAL TEST COMPLETE")
    print(f"  Within-cat NN (novel): Z={z_novel_nn:.3f} vs Random={rand_novel_nn:.3f} (chance={chance_nn:.3f})")
    print(f"  Within-cat NN (all):   Z={z_all_nn:.3f} vs Random={rand_all_nn:.3f}")
    ratio = z_novel_nn / max(rand_novel_nn, 0.001)
    print(f"  Z/Random ratio: {ratio:.2f}x")
    if z_novel_nn > rand_novel_nn * 1.5 and z_novel_nn > chance_nn * 3:
        print(f"  Verdict: Z GENERALIZES ACROSS ALL 7 LANGUAGES")
    elif z_novel_nn > rand_novel_nn * 1.2:
        print(f"  Verdict: MODERATE GENERALIZATION")
    else:
        print(f"  Verdict: INCONCLUSIVE — Z not clearly better than random")
    print("=" * 70)


if __name__ == "__main__":
    main()
