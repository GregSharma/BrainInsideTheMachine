"""
Exp Z: f-Reconstruction — Is the Language-Agnostic Map Extractable?
=====================================================================
Hypothesis (Shannon / h-f-h' argument):
  The model encodes enough information in its hidden states that a second learner
  can extract f — the language-agnostic problem→answer map — from those states alone.

Test:
  Train a linear probe on EN hidden states → correct answer (category-stratified)
  Evaluate the SAME probe on ZH hidden states (zero-shot cross-lingual transfer)

  If probe generalizes EN→ZH: f is reconstructable. The information is there,
  language-agnostically encoded. h-f-h' holds as an information-theoretic claim.

  If probe fails to generalize: the hidden states are too entangled. f is not
  accessible to a linear decoder without seeing ZH examples.

Data: output/all_layers_lasttok.npz
  - 200 problems × 36 layers × 2048 dims, both en_L{i} and zh_L{i}
  - Problems generated deterministically (seed=42) — answers recomputed analytically

Labels:
  - Category (0-4): 5-class classification — weak test
  - Exact answer bucket (log-binned): stronger test — does the probe read the answer?
  - Cross-lingual transfer metric: accuracy(EN-trained probe on ZH) vs chance

Sweep: all 36 layers — where is f most readable?
Controls:
  - Random label shuffle — floor
  - Probe trained on ZH, tested on EN — symmetry check
  - Probe trained on EN, tested on EN (same-language) — ceiling
"""

import json
import random
import math
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── Reproduce the 200 problems deterministically ──────────────────────────────

def compute_answer(prob):
    """Compute the correct numeric answer for each problem type."""
    cat = prob["category"]
    p = prob["params"]
    if cat == 0:  # arithmetic
        if p["op"] == "plus":
            return p["a"] + p["b"]
        else:
            return p["a"] * p["b"]
    elif cat == 1:  # combination C(n,k)
        n, k = p["n"], p["k"]
        return math.comb(n, k)
    elif cat == 2:  # remainder
        return p["a"] % p["b"]
    elif cat == 3:  # rectangle area
        return p["w"] * p["h"]
    elif cat == 4:  # arithmetic sequence sum
        a1, d, n = p["a1"], p["d"], p["n"]
        return n * (2 * a1 + (n - 1) * d) // 2


def generate_problems(n=200, seed=42):
    rng = random.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        problems.append({"category": 0, "params": {"a": a, "b": b, "op": op}})

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        problems.append({"category": 1, "params": {"n": n_val, "k": k_val}})

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        problems.append({"category": 2, "params": {"a": a, "b": b}})

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        problems.append({"category": 3, "params": {"w": w, "h": h}})

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        problems.append({"category": 4, "params": {"a1": a1, "d": d, "n": n_terms}})

    rng.shuffle(problems)
    return problems


def make_answer_buckets(answers, n_buckets=20):
    """Log-bin answers into discrete buckets for classification."""
    log_ans = np.log1p(np.array(answers, dtype=float))
    bins = np.linspace(log_ans.min(), log_ans.max(), n_buckets + 1)
    return np.digitize(log_ans, bins) - 1  # 0-indexed


def probe_layer(X_train, y_train, X_test, y_test, n_components=64):
    """PCA(64) + StandardScaler + LogisticRegression(saga, fast).
    PCA: 160 samples × 2048 dims → 64 dims. Fast and avoids overfit."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=500, C=1.0, solver="saga",
                             random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)
    return accuracy_score(y_test, clf.predict(X_te))


def run():
    print("Loading activations...")
    d = np.load("output/all_layers_lasttok.npz", allow_pickle=True)
    cats_stored = d["categories"]  # shape (200,)

    print("Regenerating problems + computing answers...")
    problems = generate_problems(n=200, seed=42)
    answers = [compute_answer(p) for p in problems]
    cats = np.array([p["category"] for p in problems])

    # Sanity check: stored categories should match recomputed
    assert np.array_equal(cats, cats_stored), "Category mismatch — seed drift"

    answer_buckets = make_answer_buckets(answers, n_buckets=20)
    print(f"  Answers range: {min(answers)}–{max(answers)}")
    print(f"  Unique answer buckets: {len(np.unique(answer_buckets))}")
    print(f"  Category distribution: {[(c, (cats==c).sum()) for c in range(5)]}")

    # ── Split: train on EN, test on ZH (and vice versa) ──────────────────────
    # Use 80/20 split within each language, stratified by category
    # Train indices: first 160 problems (by category-stratified split)
    np.random.seed(42)
    train_idx, test_idx = [], []
    for c in range(5):
        idx = np.where(cats == c)[0]
        np.random.shuffle(idx)
        n_train = 32  # 32/40 = 80%
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    print(f"\nSplit: {len(train_idx)} train, {len(test_idx)} test")
    print(f"Chance (category): {1/5:.3f} | Chance (answer bucket): {1/len(np.unique(answer_buckets)):.3f}")

    results = {
        "experiment": "Z: f-Reconstruction — Cross-Lingual Linear Probe",
        "n_problems": 200,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "chance_category": 1/5,
        "chance_answer_bucket": float(1/len(np.unique(answer_buckets))),
        "layers": [],
    }

    # ── Sweep all 36 layers ───────────────────────────────────────────────────
    print(f"\n{'Layer':>6} | {'EN→EN cat':>10} | {'EN→ZH cat':>10} | {'ZH→ZH cat':>10} | {'EN→EN ans':>10} | {'EN→ZH ans':>10} | {'ZH→ZH ans':>10}")
    print("-" * 80)

    best_transfer_cat = 0
    best_transfer_layer_cat = -1
    best_transfer_ans = 0
    best_transfer_layer_ans = -1

    for layer in range(36):
        en = d[f"en_L{layer}"]  # (200, 2048)
        zh = d[f"zh_L{layer}"]  # (200, 2048)

        y_cat_train = cats[train_idx]
        y_cat_test = cats[test_idx]
        y_ans_train = answer_buckets[train_idx]
        y_ans_test = answer_buckets[test_idx]

        # Category probes
        en_en_cat = probe_layer(en[train_idx], y_cat_train, en[test_idx], y_cat_test)
        en_zh_cat = probe_layer(en[train_idx], y_cat_train, zh[test_idx], y_cat_test)
        zh_zh_cat = probe_layer(zh[train_idx], y_cat_train, zh[test_idx], y_cat_test)

        # Answer bucket probes
        en_en_ans = probe_layer(en[train_idx], y_ans_train, en[test_idx], y_ans_test)
        en_zh_ans = probe_layer(en[train_idx], y_ans_train, zh[test_idx], y_ans_test)
        zh_zh_ans = probe_layer(zh[train_idx], y_ans_train, zh[test_idx], y_ans_test)

        print(f"  L{layer:02d}   | {en_en_cat:>10.3f} | {en_zh_cat:>10.3f} | {zh_zh_cat:>10.3f} | "
              f"{en_en_ans:>10.3f} | {en_zh_ans:>10.3f} | {zh_zh_ans:>10.3f}")

        if en_zh_cat > best_transfer_cat:
            best_transfer_cat = en_zh_cat
            best_transfer_layer_cat = layer
        if en_zh_ans > best_transfer_ans:
            best_transfer_ans = en_zh_ans
            best_transfer_layer_ans = layer

        results["layers"].append({
            "layer": layer,
            "en_en_category": round(en_en_cat, 4),
            "en_zh_category": round(en_zh_cat, 4),
            "zh_zh_category": round(zh_zh_cat, 4),
            "en_en_answer": round(en_en_ans, 4),
            "en_zh_answer": round(en_zh_ans, 4),
            "zh_zh_answer": round(zh_zh_ans, 4),
        })

    # ── Random label control ──────────────────────────────────────────────────
    np.random.seed(99)
    rand_labels = np.random.randint(0, 5, size=200)
    rand_layer = 20
    en_L20 = d[f"en_L{rand_layer}"]
    zh_L20 = d[f"zh_L{rand_layer}"]
    rand_en_en = probe_layer(en_L20[train_idx], rand_labels[train_idx],
                             en_L20[test_idx], rand_labels[test_idx])
    rand_en_zh = probe_layer(en_L20[train_idx], rand_labels[train_idx],
                             zh_L20[test_idx], rand_labels[test_idx])

    results["controls"] = {
        "random_label_en_en_L20": round(rand_en_en, 4),
        "random_label_en_zh_L20": round(rand_en_zh, 4),
    }
    results["summary"] = {
        "best_cross_lingual_category_transfer": round(best_transfer_cat, 4),
        "best_category_layer": best_transfer_layer_cat,
        "best_cross_lingual_answer_transfer": round(best_transfer_ans, 4),
        "best_answer_layer": best_transfer_layer_ans,
        "chance_category": 0.2,
        "chance_answer_bucket": results["chance_answer_bucket"],
    }

    print(f"\n=== SUMMARY ===")
    print(f"Best EN→ZH category transfer: {best_transfer_cat:.3f} at L{best_transfer_layer_cat} (chance=0.200)")
    print(f"Best EN→ZH answer transfer:   {best_transfer_ans:.3f} at L{best_transfer_layer_ans} (chance={results['chance_answer_bucket']:.3f})")
    print(f"Random label control (L20):   EN→EN={rand_en_en:.3f}, EN→ZH={rand_en_zh:.3f}")

    with open("output/expZ_f_reconstruction.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → output/expZ_f_reconstruction.json")


if __name__ == "__main__":
    run()
