"""expMI: Cross-lingual Mutual Information Decomposition.

Computes I(attn_zh; attn_en), I(mlp_zh; mlp_en), I(residual_zh; residual_en)
per layer using the Gaussian CCA formula:

    I(X; Y) = -1/2 * sum_i log(1 - rho_i^2)

where rho_i are canonical correlations between X and Y.

Uses PCA whitening to handle d >> n (2048 dims, 200 problems).
Extracts attn_output, mlp_output, and residual at every layer via hooks
during a single forward pass per language.

Predictions (from prior experiments):
- I(attn_zh; attn_en) HIGH at all layers (C6b: attention is constant bias)
- I(mlp_zh; mlp_en) drops sharply at L13 (language divergence point)
- Interaction term reveals synergy vs redundancy between attn and MLP
"""
import json, time, sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import linalg

OUTPUT_DIR = Path("output")
MODEL_NAME = "Qwen/Qwen2.5-3B"
N_LAYERS = 36
D_MODEL = 2048

SYSTEM_PROMPTS = {
    "en": "You are a careful mathematical reasoner. When given a problem, think step by step, show your work clearly, and then state the final numerical answer on its own line.",
    "zh": "你是一个严谨的数学推理者。遇到问题时，请逐步思考，清晰地展示你的推导过程，然后在单独的一行给出最终的数值答案。",
}


def get_problems():
    """200 math problems (same seed/templates as multilingual cache)."""
    from colab_8b_activation_dump import generate_problems_multilingual
    return generate_problems_multilingual(n=200, seed=42)


def build_prompt(tokenizer, problem_text, lang):
    sys_content = SYSTEM_PROMPTS[lang]
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_components(model, tokenizer, problems, lang, device):
    """Extract attn_output, mlp_output, and residual at each layer.

    Returns dict with keys like 'attn_L0', 'mlp_L0', 'res_L0', each (N, d).
    Uses the last token position (consistent with prior experiments).
    """
    N = len(problems)
    attn_outs = {L: [] for L in range(N_LAYERS)}
    mlp_outs = {L: [] for L in range(N_LAYERS)}

    hooks = []

    def make_attn_hook(L):
        def hook_fn(module, input, output):
            # output is a tuple; first element is the attention output (batch, seq, d)
            attn_out = output[0]
            attn_outs[L].append(attn_out[0, -1].detach().cpu().float().numpy())
        return hook_fn

    def make_mlp_hook(L):
        def hook_fn(module, input, output):
            # MLP output: (batch, seq, d)
            mlp_outs[L].append(output[0, -1].detach().cpu().float().numpy())
        return hook_fn

    # Register hooks — attn_out and mlp_out only (residual = their sum)
    for L in range(N_LAYERS):
        layer = model.model.layers[L]
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(L)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(L)))

    # Forward pass for each problem
    lang_key = "en" if lang == "en" else "zh"
    for i, prob in enumerate(problems):
        text = prob[lang_key]
        prompt = build_prompt(tokenizer, text, lang)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.inference_mode():
            model(ids)

        if (i + 1) % 50 == 0:
            print(f"  {lang.upper()}: {i+1}/{N} problems", flush=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Stack into arrays. Residual update = attn + mlp (before residual connection)
    result = {}
    for L in range(N_LAYERS):
        attn = np.stack(attn_outs[L], axis=0)  # (N, d)
        mlp = np.stack(mlp_outs[L], axis=0)
        result[f"attn_L{L}"] = attn
        result[f"mlp_L{L}"] = mlp
        result[f"update_L{L}"] = attn + mlp  # residual update (no h_prev)

    return result


def pca_reduce(X, var_threshold=0.95):
    """PCA-reduce X to retain var_threshold of variance. Returns reduced X."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    var_explained = np.cumsum(S**2) / (S**2).sum()
    k = int(np.searchsorted(var_explained, var_threshold) + 1)
    k = min(k, X.shape[0] - 1)  # Can't exceed n-1
    return X_centered @ Vt[:k].T, k


def gaussian_mi_cca(X, Y, pca_var=0.95):
    """Compute Gaussian MI between X and Y via CCA.

    Steps:
    1. PCA-reduce both to manageable dimensionality
    2. Compute CCA canonical correlations
    3. Apply MI = -1/2 sum log(1 - rho_i^2)

    Returns MI in nats, number of CCA dimensions, and canonical correlations.
    """
    n = X.shape[0]

    # PCA reduce
    X_r, kx = pca_reduce(X, pca_var)
    Y_r, ky = pca_reduce(Y, pca_var)

    # Center
    X_r -= X_r.mean(axis=0, keepdims=True)
    Y_r -= Y_r.mean(axis=0, keepdims=True)

    # Covariance matrices
    Cxx = (X_r.T @ X_r) / (n - 1) + 1e-6 * np.eye(kx)
    Cyy = (Y_r.T @ Y_r) / (n - 1) + 1e-6 * np.eye(ky)
    Cxy = (X_r.T @ Y_r) / (n - 1)

    # CCA via generalized eigenvalue: Cxx^{-1/2} Cxy Cyy^{-1} Cyx Cxx^{-1/2}
    Lx = linalg.cholesky(Cxx, lower=True)
    Ly = linalg.cholesky(Cyy, lower=True)

    Lx_inv = linalg.solve_triangular(Lx, np.eye(kx), lower=True)
    Ly_inv = linalg.solve_triangular(Ly, np.eye(ky), lower=True)

    # Whitened cross-covariance
    M = Lx_inv @ Cxy @ Ly_inv.T  # (kx, ky)

    # SVD gives canonical correlations
    _, rhos, _ = np.linalg.svd(M, full_matrices=False)

    # Clip to [0, 1-eps] for numerical stability
    rhos = np.clip(rhos, 0, 1 - 1e-10)

    # MI = -1/2 sum log(1 - rho^2)
    mi = -0.5 * np.sum(np.log(1 - rhos**2))

    return float(mi), min(kx, ky), rhos


def main():
    device = "cuda"

    print(f"{'#' * 80}", flush=True)
    print(f"  Exp MI: Cross-Lingual Mutual Information Decomposition", flush=True)
    print(f"{'#' * 80}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Components: attn_output, mlp_output, residual", flush=True)
    print(f"MI method: Gaussian CCA (PCA-reduced to 95% var)", flush=True)
    print(flush=True)

    t0 = time.time()

    # Load model
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Get problems
    problems = get_problems()
    print(f"  {len(problems)} problems loaded", flush=True)

    # Extract components for EN and ZH
    print("\nExtracting EN components...", flush=True)
    en_data = extract_components(model, tokenizer, problems, "en", device)
    print("Extracting ZH components...", flush=True)
    zh_data = extract_components(model, tokenizer, problems, "zh", device)

    # Compute MI at each layer for each component
    # "update" = attn_out + mlp_out (the residual stream delta, excluding h_prev)
    print("\nComputing MI via CCA...", flush=True)
    results = {"attn": [], "mlp": [], "update": [], "interaction": []}
    cca_details = {}

    for L in range(N_LAYERS):
        # Attn MI: I(attn_en; attn_zh)
        mi_attn, k_attn, rhos_attn = gaussian_mi_cca(
            en_data[f"attn_L{L}"], zh_data[f"attn_L{L}"]
        )
        # MLP MI: I(mlp_en; mlp_zh)
        mi_mlp, k_mlp, rhos_mlp = gaussian_mi_cca(
            en_data[f"mlp_L{L}"], zh_data[f"mlp_L{L}"]
        )
        # Update MI: I(update_en; update_zh) where update = attn + mlp
        mi_upd, k_upd, rhos_upd = gaussian_mi_cca(
            en_data[f"update_L{L}"], zh_data[f"update_L{L}"]
        )
        # Interaction: synergy (+) or redundancy (-)
        interaction = mi_upd - mi_attn - mi_mlp

        results["attn"].append(mi_attn)
        results["mlp"].append(mi_mlp)
        results["update"].append(mi_upd)
        results["interaction"].append(interaction)

        cca_details[L] = {
            "attn": {"mi": mi_attn, "k_pca": k_attn, "top5_rhos": rhos_attn[:5].tolist()},
            "mlp": {"mi": mi_mlp, "k_pca": k_mlp, "top5_rhos": rhos_mlp[:5].tolist()},
            "update": {"mi": mi_upd, "k_pca": k_upd, "top5_rhos": rhos_upd[:5].tolist()},
            "interaction": interaction,
        }

        sign = "+" if interaction > 0 else "-" if interaction < 0 else "0"
        print(f"  L{L:2d}: I(attn)={mi_attn:6.2f}  I(mlp)={mi_mlp:6.2f}  "
              f"I(upd)={mi_upd:6.2f}  interact={interaction:+6.2f} [{sign}]",
              flush=True)

    wall = time.time() - t0
    print(f"\nWall time: {wall:.0f}s ({wall/60:.1f}min)", flush=True)

    # Summary
    print(f"\n{'#' * 80}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'#' * 80}", flush=True)

    # Find peaks/troughs
    attn = np.array(results["attn"])
    mlp = np.array(results["mlp"])
    upd = np.array(results["update"])
    inter = np.array(results["interaction"])

    print(f"Attn MI:       peak L{np.argmax(attn):2d} ({attn.max():.2f}), "
          f"trough L{np.argmin(attn):2d} ({attn.min():.2f})", flush=True)
    print(f"MLP MI:        peak L{np.argmax(mlp):2d} ({mlp.max():.2f}), "
          f"trough L{np.argmin(mlp):2d} ({mlp.min():.2f})", flush=True)
    print(f"Update MI:     peak L{np.argmax(upd):2d} ({upd.max():.2f}), "
          f"trough L{np.argmin(upd):2d} ({upd.min():.2f})", flush=True)
    print(f"Interaction:   peak L{np.argmax(inter):2d} ({inter.max():+.2f}), "
          f"trough L{np.argmin(inter):2d} ({inter.min():+.2f})", flush=True)

    # Sign changes in interaction
    sign_changes = []
    for i in range(1, len(inter)):
        if inter[i-1] * inter[i] < 0:
            sign_changes.append(i)
    print(f"Interaction sign changes: {sign_changes}", flush=True)

    # Correlation between attn MI and mlp MI
    corr = np.corrcoef(attn, mlp)[0, 1]
    print(f"Correlation(I_attn, I_mlp): {corr:.3f}", flush=True)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = {
        "experiment": "MI_cross_lingual_decomposition",
        "model": MODEL_NAME,
        "n_problems": len(problems),
        "languages": ["en", "zh"],
        "method": "gaussian_cca_pca95",
        "mi_per_layer": {
            "attn": results["attn"],
            "mlp": results["mlp"],
            "update": results["update"],
            "interaction": results["interaction"],
        },
        "cca_details": {str(k): v for k, v in cca_details.items()},
        "summary": {
            "attn_peak": int(np.argmax(attn)),
            "mlp_peak": int(np.argmax(mlp)),
            "update_peak": int(np.argmax(upd)),
            "interaction_sign_changes": sign_changes,
        },
        "wall_time_s": wall,
    }
    out_file = OUTPUT_DIR / "expMI_decomposition.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
