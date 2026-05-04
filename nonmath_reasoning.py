"""Experiment B: Non-math reasoning tasks through Qwen2.5-3B.

Tests whether Z, backbone neuron, and two-tower patterns generalize beyond math.
20 non-math reasoning problems in zh+en: logic, reading comprehension, spatial, causal.

Key predictions:
- Dim 318 should NOT dominate Phase B (it's a number detector)
- Something else might take its place as backbone
- If Z exists: reasoning-general finding
- If Z doesn't exist: math-specific
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
import json
import gc
from pathlib import Path

OUTPUT_DIR = Path("output")


def get_nonmath_problems():
    """20 diverse non-math reasoning problems in zh and en."""
    problems = [
        # Logic puzzles (5)
        {
            "zh": "所有的玫瑰都是花。有些花很快就会凋谢。由此可以得出什么结论？",
            "en": "All roses are flowers. Some flowers wilt quickly. What can we conclude?",
            "category": "logic"
        },
        {
            "zh": "如果下雨，地面就会湿。地面是湿的。我们能确定下过雨吗？",
            "en": "If it rains, the ground gets wet. The ground is wet. Can we be certain it rained?",
            "category": "logic"
        },
        {
            "zh": "张三比李四高。李四比王五高。谁最矮？",
            "en": "Zhang is taller than Li. Li is taller than Wang. Who is the shortest?",
            "category": "logic"
        },
        {
            "zh": "一个袋子里有红球和蓝球。每次取出的都是红球。袋子里一定没有蓝球吗？",
            "en": "A bag contains red and blue balls. Every ball drawn so far is red. Must there be no blue balls in the bag?",
            "category": "logic"
        },
        {
            "zh": "所有的猫都是哺乳动物。有些哺乳动物会游泳。所有的猫都会游泳吗？",
            "en": "All cats are mammals. Some mammals can swim. Do all cats swim?",
            "category": "logic"
        },
        # Reading comprehension (5)
        {
            "zh": "小明早上出门时带了雨伞，但晚上回来时雨伞是干的。这说明了什么？",
            "en": "Xiaoming took an umbrella when he left in the morning, but it was dry when he returned at night. What does this suggest?",
            "category": "reading"
        },
        {
            "zh": "一家商店的橱窗里写着全场五折。第二天又写着在昨天价格的基础上再打八折。最终价格是原价的多少？",
            "en": "A store window says '50% off everything.' The next day it says '20% off yesterday's price.' What is the final price as a fraction of the original?",
            "category": "reading"
        },
        {
            "zh": "老师说明天不上课。学生问那后天呢。老师回答后天也不上。这些天是什么日子？",
            "en": "The teacher says: 'No class tomorrow.' A student asks: 'What about the day after?' The teacher replies: 'No class then either.' What days might these be?",
            "category": "reading"
        },
        {
            "zh": "一个人在沙漠中行走了三天，最后看到了一棵树。他非常高兴。为什么？",
            "en": "A person walked in the desert for three days and finally saw a tree. He was very happy. Why?",
            "category": "reading"
        },
        {
            "zh": "图书馆的规定是：每次最多借五本书，借期两周。小红上周借了三本，这周又想借四本。她能借到吗？",
            "en": "The library rule is: maximum five books at a time, two-week loan period. Xiaohong borrowed three books last week and wants to borrow four more this week. Can she?",
            "category": "reading"
        },
        # Spatial reasoning (5)
        {
            "zh": "甲在乙的北边。乙在丙的东边。丙在甲的什么方向？",
            "en": "A is north of B. B is east of C. What direction is C from A?",
            "category": "spatial"
        },
        {
            "zh": "你面朝北站立，向右转两次。你现在面朝哪个方向？",
            "en": "You are standing facing north and turn right twice. Which direction are you now facing?",
            "category": "spatial"
        },
        {
            "zh": "一栋楼有十层。电梯从第三层上升七层，然后下降四层。电梯现在在第几层？",
            "en": "A building has ten floors. The elevator goes up seven floors from the third floor, then down four floors. What floor is the elevator on now?",
            "category": "spatial"
        },
        {
            "zh": "把一张正方形的纸对折两次，然后在角上剪一个小三角形。展开后纸上有几个洞？",
            "en": "Fold a square piece of paper in half twice, then cut a small triangle at the corner. How many holes are there when you unfold it?",
            "category": "spatial"
        },
        {
            "zh": "一个骰子的相对面之和为七。如果顶面是三，底面是多少？",
            "en": "On a die, opposite faces sum to seven. If the top face is three, what is the bottom face?",
            "category": "spatial"
        },
        # Causal reasoning (5)
        {
            "zh": "路面是湿的，但今天没有下雨。可能的原因是什么？",
            "en": "The road is wet, but it didn't rain today. What could be the reason?",
            "category": "causal"
        },
        {
            "zh": "植物放在窗台上一周后开始向一侧倾斜。这是为什么？",
            "en": "A plant placed on a windowsill for a week starts leaning to one side. Why?",
            "category": "causal"
        },
        {
            "zh": "冬天，玻璃窗内侧出现水珠。这是什么原因？",
            "en": "In winter, water droplets appear on the inside of a glass window. What causes this?",
            "category": "causal"
        },
        {
            "zh": "一辆车突然刹车，车上的乘客向前倾倒。用什么原理可以解释？",
            "en": "A car brakes suddenly and the passengers lean forward. What principle explains this?",
            "category": "causal"
        },
        {
            "zh": "铁钉放在水里会沉下去，但铁做的船能浮在水面上。这是为什么？",
            "en": "An iron nail sinks in water, but an iron ship floats. Why?",
            "category": "causal"
        },
    ]
    return problems


def extract_all_layers_for_problems(model_name, problems, device="cuda"):
    """Extract mean-pooled hidden states from all layers."""
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    N = len(problems)
    print(f"Model: {n_layers} layers, d={d}, {N} problems")

    all_zh = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
    all_en = {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}

    layers = model.model.layers
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            layer_outputs[layer_idx] = h_out.detach().cpu().squeeze(0).float().numpy()
        return hook

    handles = [layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]

    for lang in ['zh', 'en']:
        print(f"  Extracting {lang}...")
        store = all_zh if lang == 'zh' else all_en
        for i, prob in enumerate(problems):
            inputs = tokenizer(prob[lang], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            for l in range(n_layers):
                store[l][i] = layer_outputs[l].mean(axis=0)
            layer_outputs.clear()

    for h in handles:
        h.remove()

    return all_zh, all_en, n_layers, d, model, tokenizer, layers


def extract_gen_trajectories_for_problems(model, tokenizer, layers, problems,
                                           target_layer, max_tokens=256, n_problems=20):
    """Generate solutions and extract hidden states at target_layer."""
    trajectories = {}
    n_problems = min(n_problems, len(problems))

    for prob_idx in range(n_problems):
        prob = problems[prob_idx]
        for lang in ['zh', 'en']:
            prompt = prob[lang]
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            h_list = []
            tok_list = []

            captured = {}
            def hook_fn(module, input, output):
                h_out = output if isinstance(output, torch.Tensor) else output[0]
                captured['h'] = h_out.detach()

            handle = layers[target_layer].register_forward_hook(hook_fn)

            cur_ids = input_ids
            for step in range(max_tokens):
                with torch.no_grad():
                    outputs = model(cur_ids, use_cache=False)

                h = captured['h'][0, -1, :].cpu().float().numpy()
                h_list.append(h)

                logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                tok_id = next_token.item()
                tok_list.append(tok_id)

                if tok_id == tokenizer.eos_token_id:
                    break
                cur_ids = torch.cat([cur_ids, next_token], dim=1)

            handle.remove()

            trajectories[f"h{target_layer}_prob{prob_idx}_{lang}"] = np.array(h_list, dtype=np.float32)
            trajectories[f"toks_prob{prob_idx}_{lang}"] = np.array(tok_list, dtype=np.int64)

            decoded = tokenizer.decode(tok_list[:15])
            print(f"    prob{prob_idx}_{lang} ({prob['category']}): {len(h_list)} steps, {decoded[:50]}...")

    return trajectories


def main():
    problems = get_nonmath_problems()
    categories = [p["category"] for p in problems]

    # Extract all layers (input-pass)
    all_zh, all_en, n_layers, d, model, tokenizer, layers_mod = extract_all_layers_for_problems(
        "Qwen/Qwen2.5-3B", problems
    )

    results = {"categories": categories}

    # ─── 1. Phase structure (Procrustes R²) ───
    print("\n--- Phase structure (Procrustes R²) ---")
    r2_list = []
    for l in range(n_layers):
        zh = all_zh[l]
        en = all_en[l]
        zh_c = zh - zh.mean(0)
        en_c = en - en.mean(0)
        combined = np.vstack([zh_c, en_c])
        n_comp = min(19, combined.shape[0] - 1)  # N=20, so max PCA = 19
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(combined)
        zh_p = pca.transform(zh_c)
        en_p = pca.transform(en_c)
        R, _ = orthogonal_procrustes(zh_p, en_p)
        zh_rot = zh_p @ R
        ss_res = np.sum((zh_rot - en_p)**2)
        ss_tot = np.sum((en_p - en_p.mean(0))**2)
        r2 = 1 - ss_res / ss_tot
        r2_list.append(round(float(r2), 4))
        bar = "#" * int(max(0, r2) * 40)
        label = ""
        if l > 0:
            delta = r2_list[l] - r2_list[l-1]
            if delta < -0.1:
                label = f" ← DROP {delta:+.3f}"
            elif delta > 0.1:
                label = f" ← JUMP {delta:+.3f}"
        print(f"  L{l:2d}: R²={r2:.4f} {bar}{label}")

    results["r2_by_layer"] = r2_list

    # ─── 2. Backbone neuron detection ───
    print("\n--- Backbone neuron (dim 318 test) ---")
    dim318_traj = []
    top5_by_layer = []
    for l in range(n_layers):
        combined = np.vstack([all_zh[l], all_en[l]])
        var = np.var(combined, axis=0)
        total = var.sum()
        frac = var / total if total > 0 else var

        sorted_dims = np.argsort(frac)[::-1]
        top5 = sorted_dims[:5].tolist()
        top5_var = [round(float(frac[j]), 4) for j in top5]
        top5_by_layer.append({"dims": top5, "variances": top5_var})

        dim318_frac = float(frac[318])
        dim318_traj.append(round(dim318_frac, 4))

        bar = "█" * int(dim318_frac * 60)
        print(f"  L{l:2d}: dim318={dim318_frac:.3f} {bar}  top5={top5[:3]}...")

    results["dim318_trajectory"] = dim318_traj
    results["top5_by_layer"] = top5_by_layer

    # What IS the backbone for non-math?
    mid_start, mid_end = 4, 28
    dim_max_mid = np.zeros(d)
    for l in range(mid_start, mid_end):
        combined = np.vstack([all_zh[l], all_en[l]])
        var = np.var(combined, axis=0)
        total = var.sum()
        dim_max_mid = np.maximum(dim_max_mid, var / total)

    backbone_dim = int(np.argmax(dim_max_mid))
    backbone_peak = float(dim_max_mid[backbone_dim])
    print(f"\n  Non-math backbone: dim {backbone_dim}, peak={backbone_peak:.3f}")
    print(f"  Is it still dim 318? {'YES' if backbone_dim == 318 else 'NO — different backbone!'}")
    results["nonmath_backbone_dim"] = backbone_dim
    results["nonmath_backbone_peak"] = round(backbone_peak, 4)

    # ─── 3. Gen-time Z projection ───
    print("\n--- Extracting gen-time trajectories ---")
    target_layer = 32  # Same as Qwen-3B math analysis
    traj = extract_gen_trajectories_for_problems(
        model, tokenizer, layers_mod, problems,
        target_layer=target_layer, max_tokens=256, n_problems=20
    )

    # Save trajectories
    np.savez_compressed(OUTPUT_DIR / "nonmath_gen_trajectories.npz", **traj)

    # Build Z basis from input-pass L32
    zh32 = all_zh[32]
    en32 = all_en[32]
    combined32 = np.vstack([zh32, en32])
    pca_z = PCA(n_components=min(19, combined32.shape[0]-1), random_state=42)
    pca_z.fit(combined32)
    Z_basis = pca_z.components_  # (19, 2048) — only 19 because N=20

    # Also load the MATH Z basis for comparison
    math_data = np.load('output/all_layers.npz')
    math_combined = np.vstack([math_data['zh_L32'], math_data['en_L32']])
    pca_math = PCA(n_components=20, random_state=42)
    pca_math.fit(math_combined)
    Z_math = pca_math.components_

    # Random basis
    rng = np.random.RandomState(42)
    rand_mat = rng.randn(19, d).astype(np.float32)
    rand_basis, _ = np.linalg.qr(rand_mat.T)
    rand_basis = rand_basis.T[:19]

    from numpy.linalg import norm
    def cos_batch(a, b):
        dots = np.sum(a * b, axis=1)
        return dots / (norm(a, axis=1) * norm(b, axis=1) + 1e-10)

    print("\n--- Gen-time Z projection (non-math) ---")
    all_cos_full = []
    all_cos_z = []
    all_cos_mathz = []
    all_cos_rand = []

    for prob_idx in range(20):
        zh_key = f"h{target_layer}_prob{prob_idx}_zh"
        en_key = f"h{target_layer}_prob{prob_idx}_en"
        if zh_key not in traj or en_key not in traj:
            continue

        zh_h = traj[zh_key]
        en_h = traj[en_key]
        T = min(len(zh_h), len(en_h))
        if T < 2:
            continue
        zh_h = zh_h[:T]
        en_h = en_h[:T]

        cos_f = cos_batch(zh_h, en_h)
        cos_z = cos_batch(zh_h @ Z_basis.T, en_h @ Z_basis.T)
        cos_mz = cos_batch(zh_h @ Z_math.T, en_h @ Z_math.T)
        cos_r = cos_batch(zh_h @ rand_basis.T, en_h @ rand_basis.T)

        all_cos_full.extend(cos_f.tolist())
        all_cos_z.extend(cos_z.tolist())
        all_cos_mathz.extend(cos_mz.tolist())
        all_cos_rand.extend(cos_r.tolist())

        cat = problems[prob_idx]["category"]
        print(f"  prob{prob_idx:2d} ({cat:8s}): T={T:3d}  full={cos_f.mean():.4f}  own-Z={cos_z.mean():.4f}  math-Z={cos_mz.mean():.4f}  rand={cos_r.mean():.4f}")

    all_cos_full = np.array(all_cos_full)
    all_cos_z = np.array(all_cos_z)
    all_cos_mathz = np.array(all_cos_mathz)
    all_cos_rand = np.array(all_cos_rand)

    print(f"\nOVERALL (non-math reasoning):")
    print(f"  Full-space:  {all_cos_full.mean():.4f} ± {all_cos_full.std():.4f}")
    print(f"  Own Z-proj:  {all_cos_z.mean():.4f} ± {all_cos_z.std():.4f}")
    print(f"  Math Z-proj: {all_cos_mathz.mean():.4f} ± {all_cos_mathz.std():.4f}")
    print(f"  Random 19d:  {all_cos_rand.mean():.4f} ± {all_cos_rand.std():.4f}")
    print(f"  Own-Z gap:   {all_cos_z.mean() - all_cos_rand.mean():+.4f}")
    print(f"  Math-Z gap:  {all_cos_mathz.mean() - all_cos_rand.mean():+.4f}")

    results["gen_z_projection"] = {
        "full_mean": round(float(all_cos_full.mean()), 4),
        "own_z_mean": round(float(all_cos_z.mean()), 4),
        "math_z_mean": round(float(all_cos_mathz.mean()), 4),
        "random_mean": round(float(all_cos_rand.mean()), 4),
        "own_z_gap": round(float(all_cos_z.mean() - all_cos_rand.mean()), 4),
        "math_z_gap": round(float(all_cos_mathz.mean() - all_cos_rand.mean()), 4),
    }

    if all_cos_z.mean() - all_cos_rand.mean() > 0.2:
        print(f"\n  *** NON-MATH Z EXISTS — gap > 0.2 ***")
    elif all_cos_z.mean() - all_cos_rand.mean() > 0.05:
        print(f"\n  ** Weak non-math Z signal **")
    else:
        print(f"\n  No Z for non-math reasoning")

    if all_cos_mathz.mean() - all_cos_rand.mean() > 0.2:
        print(f"  *** MATH Z TRANSFERS to non-math — gap > 0.2 ***")
    elif all_cos_mathz.mean() - all_cos_rand.mean() > 0.05:
        print(f"  ** Math Z weakly transfers to non-math **")
    else:
        print(f"  Math Z does NOT transfer to non-math")

    # Save
    outpath = OUTPUT_DIR / "nonmath_reasoning_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
