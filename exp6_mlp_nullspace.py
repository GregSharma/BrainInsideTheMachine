"""Experiment 6: PC0 in MLP Weight Null Space (memory-efficient).

Load model in float16 on CPU, extract one layer's weights at a time.
No full SVD — use randomized SVD (top-50) which is much cheaper.
"""

import numpy as np
import torch
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
import gc

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")

def main():
    from transformers import AutoModelForCausalLM

    print(f"Loading {MODEL_NAME} in float16 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True
    )

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"Model: {n_layers} layers, d={d}")

    acts = np.load("output/all_layers_lasttok.npz")
    results = {"per_layer": []}

    for L in range(n_layers):
        # Fit PC0
        zh = acts[f"zh_L{L}"].astype(np.float64)
        en = acts[f"en_L{L}"].astype(np.float64)
        zh_unit = zh / np.linalg.norm(zh, axis=1, keepdims=True)
        en_unit = en / np.linalg.norm(en, axis=1, keepdims=True)
        combined = np.vstack([zh_unit, en_unit])
        pca = PCA(n_components=5)
        pca.fit(combined)
        pc0 = pca.components_[0]

        # Extract weights as float32 numpy, one at a time
        mlp = model.model.layers[L].mlp
        W_gate = mlp.gate_proj.weight.detach().float().numpy()  # (inter, hidden)
        W_down = mlp.down_proj.weight.detach().float().numpy()  # (hidden, inter)

        # How much does W_gate "see" PC0?
        gate_pc0 = W_gate @ pc0
        gate_pc0_norm = float(np.linalg.norm(gate_pc0))

        # Compare to 50 random directions
        rng = np.random.RandomState(L)
        rand_norms = [float(np.linalg.norm(W_gate @ (r := rng.randn(d)) / np.linalg.norm(r))) for _ in range(50)]
        gate_z = (gate_pc0_norm - np.mean(rand_norms)) / (np.std(rand_norms) + 1e-10)

        # Randomized SVD (top 50 only — much cheaper than full)
        U, S, Vh = randomized_svd(W_gate, n_components=50, random_state=L)
        # Cumulative PC0 in top-k singular vectors
        pc0_proj = {}
        cumsum = 0.0
        for k in range(50):
            cumsum += float((pc0 @ Vh[k])**2)
            if k+1 in [1, 5, 10, 20, 50]:
                pc0_proj[k+1] = cumsum

        # Random baseline for top-50
        rand_proj_50 = []
        for _ in range(20):
            r = rng.randn(d); r /= np.linalg.norm(r)
            rp = sum(float((r @ Vh[k])**2) for k in range(50))
            rand_proj_50.append(rp)

        # How much does W_down write PC0?
        down_pc0 = W_down.T @ pc0
        down_pc0_norm = float(np.linalg.norm(down_pc0))
        rand_down = [float(np.linalg.norm(W_down.T @ (r := rng.randn(d)) / np.linalg.norm(r))) for _ in range(50)]
        down_z = (down_pc0_norm - np.mean(rand_down)) / (np.std(rand_down) + 1e-10)

        rw = gate_z * down_z

        layer_result = {
            "layer": L,
            "gate_z": float(gate_z),
            "down_z": float(down_z),
            "read_write_product": float(rw),
            "gate_pc0_norm": float(gate_pc0_norm),
            "gate_rand_mean": float(np.mean(rand_norms)),
            "down_pc0_norm": float(down_pc0_norm),
            "down_rand_mean": float(np.mean(rand_down)),
            "pc0_in_top_svd": pc0_proj,
            "rand_proj_top50": float(np.mean(rand_proj_50)),
        }
        results["per_layer"].append(layer_result)

        print(f"L{L:2d}: gate_z={gate_z:+.1f}  down_z={down_z:+.1f}  rw={rw:+.1f}  "
              f"pc0/top50={pc0_proj[50]:.3f} vs rand={np.mean(rand_proj_50):.3f}")

        del W_gate, W_down, U, S, Vh
        gc.collect()

    del model
    gc.collect()

    with open(OUTPUT_DIR / "exp6_mlp_nullspace.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to output/exp6_mlp_nullspace.json")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'Layer':>5s}  {'gate_z':>8s}  {'down_z':>8s}  {'RW':>8s}  {'PC0/top50':>10s}  {'rand/50':>8s}")
    print("-"*55)
    for r in results["per_layer"]:
        print(f"  L{r['layer']:<3d}  {r['gate_z']:>+7.1f}  {r['down_z']:>+7.1f}  "
              f"{r['read_write_product']:>+7.1f}  {r['pc0_in_top_svd'].get(50, r['pc0_in_top_svd'].get('50', 0)):>9.3f}  "
              f"{r['rand_proj_top50']:>7.3f}")

if __name__ == "__main__":
    main()
