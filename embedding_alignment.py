"""Vein 3: Where does the free z=14.6 at L0 come from?

L0 in our extraction is the output of the FIRST transformer layer.
But before that, the model computes:
  1. Raw token embeddings (model.model.embed_tokens)
  2. (Qwen2.5 has RoPE, applied inside attention — no additive positional encoding)
  3. Layer 0 input = raw token embeddings (no separate positional encoding added)

So L0 input IS the raw embedding. But our extraction hooks the LAYER OUTPUT.
The question: how much alignment exists in the raw embeddings (layer INPUT at L0)
versus after the first layer transforms them?

We also extract the embedding for each TOKEN position to see if alignment
is uniform across positions or concentrated at specific tokens.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random as pyrandom
import json

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_problems(n=200, seed=42):
    """Same deterministic problems as all scripts."""
    rng = pyrandom.Random(seed)
    problems = []
    per_cat = n // 5

    for _ in range(per_cat):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["plus", "times"])
        if op == "plus":
            zh = f"计算 {a} + {b} 的值。"
            en = f"Calculate {a} + {b}."
        else:
            zh = f"计算 {a} × {b} 的值。"
            en = f"Calculate {a} × {b}."
        problems.append({"zh": zh, "en": en, "category": 0})

    for _ in range(per_cat):
        n_val = rng.randint(5, 20)
        k_val = rng.randint(1, min(n_val - 1, 8))
        zh = f"求组合数 C({n_val}, {k_val}) 的值。"
        en = f"Find the value of C({n_val}, {k_val})."
        problems.append({"zh": zh, "en": en, "category": 1})

    for _ in range(per_cat):
        a = rng.randint(50, 9999)
        b = rng.randint(3, 37)
        zh = f"{a} 除以 {b} 的余数是多少？"
        en = f"What is the remainder when {a} is divided by {b}?"
        problems.append({"zh": zh, "en": en, "category": 2})

    for _ in range(per_cat):
        w = rng.randint(2, 50)
        h = rng.randint(2, 50)
        zh = f"一个长方形的长为 {w}，宽为 {h}，求其面积。"
        en = f"A rectangle has length {w} and width {h}. Find its area."
        problems.append({"zh": zh, "en": en, "category": 3})

    for _ in range(per_cat):
        a1 = rng.randint(1, 20)
        d = rng.randint(1, 10)
        n_terms = rng.randint(5, 30)
        zh = f"等差数列首项为 {a1}，公差为 {d}，求前 {n_terms} 项之和。"
        en = f"An arithmetic sequence has first term {a1} and common difference {d}. Find the sum of the first {n_terms} terms."
        problems.append({"zh": zh, "en": en, "category": 4})

    rng.shuffle(problems)
    return problems


def matched_vs_scrambled_z(zh, en, n_perms=1000):
    """Unit-normalized matched vs scrambled z-score."""
    zh_u = zh / np.linalg.norm(zh, axis=1, keepdims=True)
    en_u = en / np.linalg.norm(en, axis=1, keepdims=True)
    matched = np.mean(np.sum(zh_u * en_u, axis=1))
    rng = np.random.RandomState(42)
    scrambled = np.array([
        np.mean(np.sum(zh_u * en_u[rng.permutation(len(en_u))], axis=1))
        for _ in range(n_perms)
    ])
    z = (matched - scrambled.mean()) / scrambled.std()
    return float(z), float(matched), float(scrambled.mean()), float(scrambled.std())


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model: {n_layers} layers, d={d}")

    # Check architecture: does Qwen2.5 have additive positional encoding?
    print(f"Embedding type: {type(model.model.embed_tokens)}")
    # Qwen2.5 uses RoPE (rotary positional encoding) applied inside attention,
    # so the embedding output IS the raw token embedding — no position added.

    problems = generate_problems(200, seed=42)
    N = len(problems)

    # Hook the embedding output (input to first layer)
    embedding_outputs = {}

    def embed_hook(module, input, output):
        # output shape: (1, seq_len, d)
        embedding_outputs["full"] = output.detach().cpu().squeeze(0).float().numpy()  # (seq_len, d)

    handle = model.model.embed_tokens.register_forward_hook(embed_hook)

    # Also hook L0 output for comparison
    layer0_outputs = {}

    def layer0_hook(module, input, output):
        h_out = output if isinstance(output, torch.Tensor) else output[0]
        layer0_outputs["full"] = h_out.detach().cpu().squeeze(0).float().numpy()  # (seq_len, d)

    handle_l0 = model.model.layers[0].register_forward_hook(layer0_hook)

    # Storage
    zh_embed_last = np.zeros((N, d), dtype=np.float32)
    en_embed_last = np.zeros((N, d), dtype=np.float32)
    zh_embed_mean = np.zeros((N, d), dtype=np.float32)  # mean of all tokens (for comparison)
    en_embed_mean = np.zeros((N, d), dtype=np.float32)
    zh_l0_last = np.zeros((N, d), dtype=np.float32)
    en_l0_last = np.zeros((N, d), dtype=np.float32)

    # Also track per-position alignment (for shared tokens like digits)
    zh_embed_all = []  # list of (seq_len, d) arrays
    en_embed_all = []
    zh_tokens = []
    en_tokens = []

    # Extract Chinese
    print(f"\nExtracting {N} Chinese problems (embeddings + L0)...")
    for i, prob in enumerate(tqdm(problems, desc="zh")):
        inputs = tokenizer(prob["zh"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        emb = embedding_outputs["full"]  # (seq_len, d)
        zh_embed_last[i] = emb[-1]
        zh_embed_mean[i] = emb[1:].mean(axis=0)  # skip BOS (pos 0)
        zh_l0_last[i] = layer0_outputs["full"][-1]

        zh_embed_all.append(emb.copy())
        zh_tokens.append(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist()))

        embedding_outputs.clear()
        layer0_outputs.clear()

    # Extract English
    print(f"Extracting {N} English problems (embeddings + L0)...")
    for i, prob in enumerate(tqdm(problems, desc="en")):
        inputs = tokenizer(prob["en"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        emb = embedding_outputs["full"]
        en_embed_last[i] = emb[-1]
        en_embed_mean[i] = emb[1:].mean(axis=0)
        en_l0_last[i] = layer0_outputs["full"][-1]

        en_embed_all.append(emb.copy())
        en_tokens.append(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist()))

        embedding_outputs.clear()
        layer0_outputs.clear()

    handle.remove()
    handle_l0.remove()

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("EMBEDDING vs L0: WHERE DOES THE FREE ALIGNMENT COME FROM?")
    print("=" * 70)

    results = {}

    # 1. Raw embedding alignment (last token)
    z_embed_last, m_el, s_el, std_el = matched_vs_scrambled_z(zh_embed_last, en_embed_last)
    print(f"\n  Raw embedding (last token): z={z_embed_last:.1f}, matched={m_el:.4f}, scrambled={s_el:.4f}")

    # 2. Raw embedding alignment (mean, excluding BOS)
    z_embed_mean, m_em, s_em, std_em = matched_vs_scrambled_z(zh_embed_mean, en_embed_mean)
    print(f"  Raw embedding (mean, no BOS): z={z_embed_mean:.1f}, matched={m_em:.4f}, scrambled={s_em:.4f}")

    # 3. L0 output alignment (last token) — should match our all_layers_lasttok data
    z_l0_last, m_l0, s_l0, std_l0 = matched_vs_scrambled_z(zh_l0_last, en_l0_last)
    print(f"  L0 output (last token):      z={z_l0_last:.1f}, matched={m_l0:.4f}, scrambled={s_l0:.4f}")

    # 4. Compare with cached L0 data
    cached = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")
    z_cached, m_c, s_c, _ = matched_vs_scrambled_z(cached["zh_L0"], cached["en_L0"])
    print(f"  L0 cached (verification):    z={z_cached:.1f}")

    results["z_raw_embed_last_token"] = z_embed_last
    results["z_raw_embed_mean_no_bos"] = z_embed_mean
    results["z_l0_output_last_token"] = z_l0_last
    results["z_l0_cached"] = z_cached
    results["delta_l0_minus_embed"] = z_l0_last - z_embed_last

    print(f"\n  >>> L0 - Embed delta: {z_l0_last - z_embed_last:+.1f}")
    print(f"  >>> This tells us how much the FIRST LAYER adds to alignment")

    # 5. Norm comparison
    embed_norm = np.linalg.norm(zh_embed_last, axis=1).mean()
    l0_norm = np.linalg.norm(zh_l0_last, axis=1).mean()
    print(f"\n  Embedding norm: {embed_norm:.2f}")
    print(f"  L0 output norm: {l0_norm:.2f}")
    print(f"  Ratio: {l0_norm / embed_norm:.2f}x")

    # 6. How much do shared tokens (digits, operators) contribute?
    print(f"\n--- Shared token analysis ---")
    # Find problems where zh and en share numerical tokens
    shared_count = 0
    shared_cosines = []
    unshared_cosines = []

    for i in range(min(50, N)):  # sample first 50
        zh_toks = zh_tokens[i]
        en_toks = en_tokens[i]
        zh_emb = zh_embed_all[i]
        en_emb = en_embed_all[i]

        # Find shared tokens (digits, numbers)
        for j, zt in enumerate(zh_toks):
            for k, et in enumerate(en_toks):
                if zt == et and zt not in ['<|endoftext|>', '<s>']:
                    # Same token in both languages
                    zh_v = zh_emb[j]
                    en_v = en_emb[k]
                    cos = np.dot(zh_v, en_v) / (np.linalg.norm(zh_v) * np.linalg.norm(en_v) + 1e-10)
                    shared_cosines.append(cos)
                    shared_count += 1

    if shared_cosines:
        print(f"  Shared tokens found: {shared_count}")
        print(f"  Mean cosine (shared tokens): {np.mean(shared_cosines):.4f}")
        print(f"  These are identical embeddings (cos≈1.0) — trivially aligned")
    else:
        print(f"  No shared tokens found")

    # 7. BOS token embedding
    zh_bos = zh_embed_all[0][0]  # BOS token embedding
    en_bos = en_embed_all[0][0]
    bos_cos = np.dot(zh_bos, en_bos) / (np.linalg.norm(zh_bos) * np.linalg.norm(en_bos))
    print(f"\n  BOS embedding cosine (zh vs en): {bos_cos:.4f}")
    print(f"  BOS is same token → cos should be 1.0")

    # 8. Language-specific token embeddings
    print(f"\n--- Per-token-type embedding analysis ---")
    # Categorize tokens
    zh_lang_embs = []  # Chinese-only tokens
    en_lang_embs = []  # English-only tokens
    num_embs_zh = []   # Numeric tokens in zh prompts
    num_embs_en = []   # Numeric tokens in en prompts

    for i in range(min(50, N)):
        for j, tok in enumerate(zh_tokens[i]):
            if any(ord(c) > 0x4E00 for c in tok if isinstance(c, str)):
                zh_lang_embs.append(zh_embed_all[i][j])
            elif tok.strip().isdigit():
                num_embs_zh.append(zh_embed_all[i][j])

        for j, tok in enumerate(en_tokens[i]):
            if tok.strip().isalpha() and all(ord(c) < 128 for c in tok):
                en_lang_embs.append(en_embed_all[i][j])
            elif tok.strip().isdigit():
                num_embs_en.append(en_embed_all[i][j])

    if zh_lang_embs and en_lang_embs:
        zh_lang = np.array(zh_lang_embs)
        en_lang = np.array(en_lang_embs)
        # Mean embedding of Chinese chars vs English words
        zh_mean = zh_lang.mean(axis=0)
        en_mean = en_lang.mean(axis=0)
        lang_cos = np.dot(zh_mean, en_mean) / (np.linalg.norm(zh_mean) * np.linalg.norm(en_mean))
        print(f"  Chinese token mean vs English token mean: cos={lang_cos:.4f}")
        print(f"  Chinese tokens: {len(zh_lang_embs)}, English tokens: {len(en_lang_embs)}")

    if num_embs_zh and num_embs_en:
        zh_nums = np.array(num_embs_zh)
        en_nums = np.array(num_embs_en)
        # Digit embeddings should be identical (same tokens!)
        # Cross-check: are digit embeddings in zh same as in en?
        zh_num_mean = zh_nums.mean(axis=0)
        en_num_mean = en_nums.mean(axis=0)
        num_cos = np.dot(zh_num_mean, en_num_mean) / (np.linalg.norm(zh_num_mean) * np.linalg.norm(en_num_mean))
        print(f"  Digit token mean (zh context vs en context): cos={num_cos:.4f}")
        print(f"  (Should be 1.0 — same token IDs, no context at embedding level)")

    # 9. The key question: is alignment from shared tokens or from the structure?
    print(f"\n--- DECOMPOSITION: Shared vs language-specific ---")
    # Remove the mean of shared digit embeddings and re-test
    # This separates "alignment because same digits" from "alignment because same structure"

    # For each problem, compute the mean of digit tokens and subtract
    zh_embed_denum = np.zeros((N, d), dtype=np.float32)
    en_embed_denum = np.zeros((N, d), dtype=np.float32)

    for i in range(N):
        zh_emb = zh_embed_all[i]
        en_emb = en_embed_all[i]

        # Last token (language-specific, not a shared digit)
        zh_embed_denum[i] = zh_emb[-1]
        en_embed_denum[i] = en_emb[-1]

    # The last token is typically a period/punctuation — check what it is
    print(f"\n  Last tokens (first 10 problems):")
    for i in range(10):
        print(f"    zh: '{zh_tokens[i][-1]}' → en: '{en_tokens[i][-1]}'")

    # 10. Embedding PCA — what does the embedding space look like?
    print(f"\n--- Embedding space PCA ---")
    from sklearn.decomposition import PCA
    combined_embed = np.vstack([zh_embed_last, en_embed_last])
    combined_u = combined_embed / np.linalg.norm(combined_embed, axis=1, keepdims=True)

    pca = PCA(n_components=10)
    pca.fit(combined_u)
    print(f"  Variance explained: {pca.explained_variance_ratio_[:5]}")
    print(f"  Cumulative (5 PCs): {np.cumsum(pca.explained_variance_ratio_[:5])[-1]:.3f}")

    # Language separation on PC0
    zh_proj = pca.transform(zh_embed_last / np.linalg.norm(zh_embed_last, axis=1, keepdims=True))
    en_proj = pca.transform(en_embed_last / np.linalg.norm(en_embed_last, axis=1, keepdims=True))

    from scipy import stats
    d_cohen_pc0 = (zh_proj[:, 0].mean() - en_proj[:, 0].mean()) / \
                  np.sqrt((zh_proj[:, 0].std()**2 + en_proj[:, 0].std()**2) / 2)
    print(f"  Cohen's d on PC0 (lang separation): {d_cohen_pc0:.2f}")

    # PCA subspace alignment at embedding level
    for k in [2, 5, 10]:
        z_pca, _, _, _ = matched_vs_scrambled_z(zh_proj[:, :k], en_proj[:, :k])
        print(f"  PCA-{k} embedding alignment: z={z_pca:.1f}")

    results["embedding_pca_varexp"] = [float(x) for x in pca.explained_variance_ratio_[:10]]
    results["embedding_cohen_d_pc0"] = float(d_cohen_pc0)

    # Save
    outpath = OUTPUT_DIR / "embedding_alignment.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
