"""Check dim 132 (anti-backbone / language polarity dim) on clean data.

In the BOS-contaminated analysis, dim 132 appeared at L35 with zh=+51, en=-70.
Does this survive in clean last-token data? Is it still a language discriminator?
"""

import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("output")

data = np.load(OUTPUT_DIR / "all_layers_lasttok.npz")

print("=== Dim 132 across layers (clean last-token data) ===\n")

for l in range(36):
    zh = data[f"zh_L{l}"]
    en = data[f"en_L{l}"]

    zh_132 = zh[:, 132]
    en_132 = en[:, 132]

    zh_mean = zh_132.mean()
    en_mean = en_132.mean()
    gap = zh_mean - en_mean

    # Language separability: how well does dim 132 discriminate zh from en?
    combined = np.concatenate([zh_132, en_132])
    labels = np.array([1]*200 + [0]*200)
    threshold = combined.mean()
    correct = np.sum((combined > threshold) == labels)
    acc = max(correct, 400 - correct) / 400

    if abs(gap) > 1.0 or l in [0, 1, 2, 15, 30, 34, 35]:
        print(f"  L{l:2d}: zh_mean={zh_mean:+7.2f}, en_mean={en_mean:+7.2f}, gap={gap:+7.2f}, lang_acc={acc:.1%}")

# Check what the top language-discriminating dimensions are at L35
print("\n=== Top language-discriminating dims at L35 ===")
zh = data["zh_L35"]
en = data["en_L35"]

gaps = zh.mean(axis=0) - en.mean(axis=0)
top = np.argsort(np.abs(gaps))[::-1][:20]
for d in top:
    zh_m = zh[:, d].mean()
    en_m = en[:, d].mean()
    g = zh_m - en_m
    # Effect size (Cohen's d)
    combined_std = np.sqrt((zh[:, d].var() + en[:, d].var()) / 2)
    cohen_d = g / combined_std if combined_std > 0 else 0
    print(f"  dim {d:4d}: zh={zh_m:+8.2f}, en={en_m:+8.2f}, gap={g:+8.2f}, Cohen's d={cohen_d:+5.2f}")

# Check dim 132 at gen-time too
print("\n=== Dim 132 at gen-time (per-token, BOS-free) ===")
gen = np.load(OUTPUT_DIR / "gen_trajectories_peos.npz", allow_pickle=True)
for prob in range(3):
    for lang in ["en", "zh"]:
        key = f"h32_prob{prob}_{lang}"
        if key in gen:
            h = gen[key]
            d132 = h[:, 132]
            print(f"  prob{prob}_{lang}: dim132 mean={d132.mean():+7.2f}, std={d132.std():.2f}, range=[{d132.min():+.1f}, {d132.max():+.1f}]")
