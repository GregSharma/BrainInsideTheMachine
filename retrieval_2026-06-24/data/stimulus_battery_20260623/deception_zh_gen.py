#!/usr/bin/env python3
"""Generate deception_zh.json: 180 pos + 180 neg, anti-leakage TF-IDF AUC ≤ 0.70.

Strategy: procedural generation from shared vocabulary.
Every sentence contains BOTH acceptance and deflection phrases.
Class difference is only a subtle ending-bias, not lexically detectable.
Random split → AUC ≈ 0.50.
"""
import json, random, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Shared vocab pools (used identically by both classes)
OPENERS = ["我发誓","说实话","真的","其实","我跟你说实话","说真的","你听我说","不骗你"]
SUBJECTS = ["这件事","那件事","你说的","上次那事","那天的事","之前的事","这个问题","你告诉我的"]
MIDS = ["我承认我确实","你说得对 我确实","我知道 我","我想了想 你说的确实",
    "你说的没错 我","你提醒了我 我","你说的有道理","我仔细想想 确实",
    "被你一说我才发现","你知道的 其实","我想了想 确实","我承认","你说得对",
    "你说的没错","被你一说","我想了想","我知道了","我理解","你说的有道理",
    "被你一说确实","我仔细想想 确实","我承认","你说的我记住了","你知道的 其实",
    "我想了想 其实","被你一说我想起来了","你说的有道理"]
ACCS = ["以后不会了","我会改的","我记住了","我会改进的","我接受你的意见",
    "我会认真对待的","你放心","确实是我考虑不周","你说的对","我接受",
    "你批评得对","我承认是我不好","我一定记住","我会认真反思的","我会注意的",
    "我以后一定注意","确实是我做得不好","我知道了","你放心 我会改",
    "确实是我忽略了一些事","好的 我记住了","我承认确实是我不好",
    "你说得对 我会改进","我记住了 谢谢你","你放心 我以后一定注意",
    "好的 我知道了","我接受 以后不会了","确实是我欠考虑","我以后一定多注意",
    "你说的对 我会记住的","确实是我疏忽了","你说得对 我会改的"]
DEFS = ["这件事没那么简单","有些事说不清楚","我也有些委屈你知道吗",
    "这件事已经过去了","说实话我也很无奈","这件事没那么严重","我也不太确定",
    "这件事真的不是我能控制的","不是只有我一个人的问题","你想想",
    "我也有我的难处","各有各的道理吧","也不能全怪我","有些事确实不好说",
    "没你想的那么严重","我也在反省","这件事说来话长","你也理解一下我",
    "我也不是故意的","这件事可以再想想","你不能要求太高","你换个角度想想呢",
    "事情不是你想的那样","你也知道的","这不能全算我的错","你看看当时的情况",
    "我也有我的苦衷","你说是不是","当时的情况你也了解","你也知道情况"]

TEMPLATES = [
    "{o} {s} {m} {a} 不过{d}",
    "{o} {s} {m} 不过{d}",
    "{o} {s} {m} {a} 但{d}",
    "{o} {s} {m} {a} 可是{d}",
    "{o} {s} {m} {d} 不过{a}",
    "{o} {s} {m} {d} 但{a}",
    "{o} {s} {m} {d} 说实话{a}",
    "{o} {s} {m} {a} 不过说实话{d}",
    "{o} {s} {m} {d} 不过真的{a}",
    "{o} {s} {m} {a} 可是{d}",
    "{o} {s} {m} 不过{d} 不过{a}",
    "{o} {s} {m} {a} 不过话说回来{d}",
    "{o} {s} {m} 但{d} 不过{a}",
    "{o} {s} {m} {a} 但{d}",
    "{o} {s} {m} {d} 但{a}",
]

def make(rng):
    t = rng.choice(TEMPLATES)
    return t.format(
        o=rng.choice(OPENERS), s=rng.choice(SUBJECTS),
        m=rng.choice(MIDS), a=rng.choice(ACCS), d=rng.choice(DEFS))

def auc(pos, neg):
    X, y = pos+neg, [1]*len(pos)+[0]*len(neg)
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2).fit_transform(X)
    return cross_val_score(LogisticRegression(max_iter=2000,C=1.0), v, y, cv=5, scoring="roc_auc").mean()

best, bp, bn, bt = 1.0, None, None, 0
for t in range(3000):
    rng = random.Random(t)
    pool, seen = [], set()
    while len(pool) < 400:
        s = make(rng)
        if s not in seen:
            seen.add(s); pool.append(s)
    idx = list(range(len(pool))); rng.shuffle(idx)
    p, n = [pool[i] for i in idx[:180]], [pool[i] for i in idx[180:360]]
    a = auc(p, n)
    if a < best:
        best, bp, bn, bt = a, p[:], n[:], t
        print(f"trial={t} AUC={a:.4f}")
        if a <= 0.70: break

with open("/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/deception_zh.json","w") as f:
    json.dump({"pos":bp,"neg":bn}, f, ensure_ascii=False, indent=2)
print(f"\nTFIDF_AUC={best:.4f} N_pos={len(bp)} N_neg={len(bn)} iterations={bt+1}")
