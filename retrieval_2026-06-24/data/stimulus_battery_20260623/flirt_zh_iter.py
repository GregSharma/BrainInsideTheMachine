#!/usr/bin/env python3
"""Generate Simplified Chinese flirt vs platonic stimulus set.
Target: 180 each, TF-IDF char AUC <= 0.75.

KEY INSIGHT: Both classes must use IDENTICAL sentence structures with
the SAME characters. The difference is subtle verb/noun swaps within
shared frames.

Frame: "你[adj]了 我[verb][object]"
- POS: 你困了 我帮你暖暖 / 你冷了 我给你披外套
- NEG: 你困了 早点休息 / 你冷了 多穿衣服

But we need NEG to ALSO use 我/你 structures for char overlap.
So NEG = practical caring with pronouns, POS = emotional caring with pronouns.

Both: 我+verb+你/给你/帮你 structure
POS verb: 想/念/盼/陪/靠/拉/牵/贴/抱
NEG verb: 查/记/看/说/叫/带/收/理
"""

import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

random.seed(42)
N = 180

def auc_score(pos, neg):
    X = pos + neg
    y = [1]*len(pos) + [0]*len(neg)
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2).fit_transform(X)
    return cross_val_score(LogisticRegression(max_iter=2000, C=1.0), v, y, cv=5, scoring="roc_auc").mean()

# ============================================================
# SHARED FRAMES - both classes use these
# ============================================================
SUBJ = ["你", "今天你", "明天你", "路上你"]
STATE = ["辛苦了", "累了", "饿了", "困了", "冷了", "热了", "渴了", "不舒服",
         "头疼", "肚子疼", "感冒了", "发烧了", "上火了", "出门了", "到家了",
         "在路上", "在公司", "在学校", "在加班", "在学习", "在休息",
         "下雨了", "降温了", "天冷了", "天热了"]

# POS-specific verbs (emotional/romantic)
POS_VERB_OBJ = [
    ("想", "你"), ("念", "你"), ("盼", "你回来"),
    ("陪", "你一起"), ("陪", "你待会儿"), ("陪", "你坐坐"),
    ("陪", "你走走"), ("陪", "你说话"), ("陪", "你等"),
    ("靠", "你近一点"), ("靠", "你旁边"),
    ("拉", "你的手"), ("拉", "你走"), ("拉", "你坐"),
    ("牵", "你的手"), ("牵", "你走"), ("牵", "你坐"),
    ("贴", "你近一点"), ("贴", "你旁边"),
    ("抱", "你一下"), ("抱", "你一会儿"),
    ("看", "你笑"), ("看", "你就好"), ("看", "你开心"),
    ("听", "你说话"), ("听", "你讲"), ("听", "你说"),
    ("给", "你暖暖手"), ("给", "你暖暖"), ("给", "你暖被窝"),
    ("帮", "你拿东西"), ("帮", "你撑伞"), ("帮", "你挡风"),
    ("帮", "你背包"), ("帮", "你提东西"), ("帮", "你看路"),
    ("帮", "你开门"), ("帮", "你弄好"), ("帮", "你收拾"),
    ("带", "你去"), ("带", "你走"), ("带", "你看看"),
    ("等", "你回来"), ("等", "你一起"), ("等", "你到"),
]

# NEG-specific verbs (practical/platonic)
NEG_VERB_OBJ = [
    ("查", "天气"), ("查", "路线"), ("查", "课表"),
    ("记", "吃药"), ("记", "吃饭"), ("记", "带伞"),
    ("记", "穿衣服"), ("记", "带钥匙"), ("记", "充电"),
    ("看", "天气"), ("看", "路"), ("看", "时间"),
    ("看", "手机"), ("看", "消息"),
    ("说", "一声"), ("说", "一下"), ("说", "好了"),
    ("叫", "你起床"), ("叫", "你吃饭"), ("叫", "你休息"),
    ("带", "伞"), ("带", "钥匙"), ("带", "外套"),
    ("带", "水"), ("带", "围巾"), ("带", "手套"),
    ("收", "衣服"), ("收", "快递"), ("收", "东西"),
    ("理", "东西"), ("理", "房间"), ("理", "书包"),
    ("看", "课表"), ("看", "日程"), ("看", "安排"),
    ("说", "注意"), ("说", "小心"), ("说", "慢点"),
    ("叫", "你注意"), ("叫", "你小心"), ("叫", "你慢点"),
    ("带", "好吃的"), ("带", "水果"), ("带", "牛奶"),
    ("收", "好东西"), ("收", "好文件"),
]

# Shared closers (both classes use these)
SHARED_CLOSERS = [
    "就好", "就好啦", "就好了", "就行", "就可以",
    "我放心", "我才放心", "我就放心", "我安心", "我就安心",
    "注意安全", "注意身体", "注意保暖", "注意休息", "注意天气",
    "别太累", "别太拼", "别太担心", "别太着急", "别太操心",
    "好好休息", "好好吃饭", "好好工作", "好好学习",
    "加油", "慢慢来", "辛苦了",
]

# ============================================================
# GENERATE SENTENCES
# ============================================================
pos_sentences = set()
neg_sentences = set()

for _ in range(1000):
    subj = random.choice(SUBJ)
    state = random.choice(STATE)

    # POS sentence
    verb, obj = random.choice(POS_VERB_OBJ)
    closer = random.choice(SHARED_CLOSERS)
    s = f"{subj}{state} 我{verb}{obj} {closer}"
    if 5 <= len(s) <= 30:
        pos_sentences.add(s)

    # NEG sentence
    verb, obj = random.choice(NEG_VERB_OBJ)
    closer = random.choice(SHARED_CLOSERS)
    s = f"{subj}{state} 我{verb}{obj} {closer}"
    if 5 <= len(s) <= 30:
        neg_sentences.add(s)

pos_sentences = list(pos_sentences)
neg_sentences = list(neg_sentences)
random.shuffle(pos_sentences)
random.shuffle(neg_sentences)
print(f"Generated: POS={len(pos_sentences)} NEG={len(neg_sentences)}")

assert len(pos_sentences) >= N and len(neg_sentences) >= N

# ============================================================
# INITIAL SELECTION
# ============================================================
X_all = pos_sentences + neg_sentences
y_all = [1]*len(pos_sentences) + [0]*len(neg_sentences)
v_all = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2).fit_transform(X_all)
clf_all = LogisticRegression(max_iter=2000, C=1.0).fit(v_all, y_all)
df_all = clf_all.decision_function(v_all.toarray())
if hasattr(df_all, 'ndim') and df_all.ndim > 1:
    df_all = df_all[:, 0]

n_pos = len(pos_sentences)
pos_disc = sorted([(i, df_all[i]) for i in range(n_pos)], key=lambda x: x[1])
neg_disc = sorted([(i, df_all[i]) for i in range(n_pos, n_pos+len(neg_sentences))], key=lambda x: x[1], reverse=True)

pos_sel = [pos_sentences[i] for i, _ in pos_disc[:N]]
neg_sel = [neg_sentences[i - n_pos] for i, _ in neg_disc[:N]]

auc = auc_score(pos_sel, neg_sel)
print(f"Initial AUC: {auc:.4f}")

pos_pool = [pos_sentences[i] for i, _ in pos_disc[N:]]
neg_pool = [neg_sentences[i - n_pos] for i, _ in neg_disc[N:]]

# ============================================================
# ITERATIVE REFINEMENT
# ============================================================
for rnd in range(1, 50):
    if auc <= 0.75:
        break

    X = pos_sel + neg_sel
    y = [1]*N + [0]*N
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2).fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(v, y)
    df = clf.decision_function(v.toarray())
    if hasattr(df, 'ndim') and df.ndim > 1:
        df = df[:, 0]

    pos_s = list(df[:N])
    neg_s = list(df[N:])

    top_pos = max(range(N), key=lambda i: pos_s[i])
    top_neg = max(range(N), key=lambda i: -neg_s[i])

    improved = False

    # Swap top POS
    old = pos_sel[top_pos]
    for cand in random.sample(pos_pool, min(30, len(pos_pool))):
        trial = list(pos_sel)
        trial[top_pos] = cand
        trial_auc = auc_score(trial, neg_sel)
        if trial_auc < auc:
            pos_sel = trial
            pos_pool.remove(cand)
            pos_pool.append(old)
            auc = trial_auc
            improved = True
            print(f"  R{rnd}: POS->{auc:.4f}")
            break

    if not improved:
        old = neg_sel[top_neg]
        for cand in random.sample(neg_pool, min(30, len(neg_pool))):
            trial = list(neg_sel)
            trial[top_neg] = cand
            trial_auc = auc_score(pos_sel, trial)
            if trial_auc < auc:
                neg_sel = trial
                neg_pool.remove(cand)
                neg_pool.append(old)
                auc = trial_auc
                improved = True
                print(f"  R{rnd}: NEG->{auc:.4f}")
                break

    if not improved:
        pos_r = sorted(range(N), key=lambda i: pos_s[i], reverse=True)
        for idx in pos_r[1:5]:
            old = pos_sel[idx]
            for cand in random.sample(pos_pool, min(15, len(pos_pool))):
                trial = list(pos_sel)
                trial[idx] = cand
                trial_auc = auc_score(trial, neg_sel)
                if trial_auc < auc:
                    pos_sel = trial
                    pos_pool.remove(cand)
                    pos_pool.append(old)
                    auc = trial_auc
                    improved = True
                    print(f"  R{rnd}: POS2->{auc:.4f}")
                    break
            if improved:
                break

    if not improved:
        neg_r = sorted(range(N), key=lambda i: neg_s[i])
        for idx in neg_r[1:5]:
            old = neg_sel[idx]
            for cand in random.sample(neg_pool, min(15, len(neg_pool))):
                trial = list(neg_sel)
                trial[idx] = cand
                trial_auc = auc_score(pos_sel, trial)
                if trial_auc < auc:
                    neg_sel = trial
                    neg_pool.remove(cand)
                    neg_pool.append(old)
                    auc = trial_auc
                    improved = True
                    print(f"  R{rnd}: NEG2->{auc:.4f}")
                    break
            if improved:
                break

    if not improved:
        for _ in range(20):
            ri = random.randint(0, N-1)
            rp = random.choice(pos_pool)
            trial = list(pos_sel)
            trial[ri] = rp
            trial_auc = auc_score(trial, neg_sel)
            if trial_auc < auc:
                pos_pool.remove(rp)
                pos_pool.append(pos_sel[ri])
                pos_sel[ri] = rp
                auc = trial_auc
                improved = True
                print(f"  R{rnd}: rPOS->{auc:.4f}")
                break

        if not improved:
            for _ in range(20):
                ri = random.randint(0, N-1)
                rn = random.choice(neg_pool)
                trial = list(neg_sel)
                trial[ri] = rn
                trial_auc = auc_score(pos_sel, trial)
                if trial_auc < auc:
                    neg_pool.remove(rn)
                    neg_pool.append(neg_sel[ri])
                    neg_sel[ri] = rn
                    auc = trial_auc
                    improved = True
                    print(f"  R{rnd}: rNEG->{auc:.4f}")
                    break

    if not improved:
        print(f"  R{rnd}: stuck {auc:.4f}")
        break

print(f"\nFinal AUC={auc:.4f} N_pos={len(pos_sel)} N_neg={len(neg_sel)}")

out = {"pos": pos_sel, "neg": neg_sel}
with open("/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/flirt_zh.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("Wrote flirt_zh.json")
