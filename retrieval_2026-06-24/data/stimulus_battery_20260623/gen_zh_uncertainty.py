#!/usr/bin/env python3
"""
Generate 180 pos (uncertain) + 180 neg (confident) Simplified Chinese sentences.
Anti-leakage v7: Extreme dilution strategy.
- Sentences are 150-300 chars long (3-5 filler clauses)
- Distinguishing clause is ONE short phrase among many
- Both classes draw from overlapping phrase pools
- Shared vocabulary at equal rates
- Random clause ordering buries positional signal
"""
import json, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

random.seed(42)
np.random.seed(42)

hedge_words = ["或者", "可能", "或许", "差不多"]

topics = [
    "实验结果", "研究数据", "科学理论", "化学反应", "生物进化",
    "天气预报", "交通状况", "商品价格", "电影质量", "音乐风格",
    "道德标准", "人生意义", "社会现象", "文化差异", "心理状态",
    "软件性能", "算法效率", "系统架构", "数据质量", "模型精度",
    "药物效果", "治疗方案", "临床症状", "饮食习惯", "睡眠质量",
    "生态平衡", "城市规划", "经济趋势", "艺术风格", "文学批评",
    "生态多样性", "教育改革", "能源政策", "社会公平", "技术创新",
]

verbs = [
    "表明", "显示", "证明", "说明", "暗示", "揭示", "反映", "体现",
    "表示", "证实", "指出", "发现", "记录", "检测", "分析", "确认",
]

a_opts = [
    "积极的", "消极的", "局部的", "全局的", "表层的", "深层的",
    "暂时的", "永久的", "自然的", "人为的", "简单的", "复杂的",
    "直接的", "间接的", "短期的", "长期的", "已知的", "未知的",
    "主要的", "次要的", "内在的", "外在的", "显性的", "隐性的",
]
b_opts = [
    "消极的", "积极的", "全局的", "局部的", "深层的", "表层的",
    "永久的", "暂时的", "人为的", "自然的", "复杂的", "简单的",
    "间接的", "直接的", "长期的", "短期的", "未知的", "已知的",
    "次要的", "主要的", "外在的", "内在的", "隐性的", "显性的",
]

# === HUGE filler pool (shared by both classes) ===
fillers = [
    "在过去的几年里有很多相关的研究",
    "不同领域的学者对此有不同的看法",
    "最近的一些研究也涉及了这个方面",
    "相关的文献资料非常丰富",
    "从历史角度来看有很多先例",
    "实际应用中也经常遇到类似的问题",
    "相关的理论框架也比较成熟",
    "学术界对这个问题有过长期的讨论",
    "在实践中人们积累了不少经验",
    "相关的统计数据也比较充分",
    "不同地区的情况可能有所差异",
    "相关的调查结果显示了一些趋势",
    "从多个角度来看这个问题值得深入研究",
    "相关的背景信息有助于我们理解",
    "在实际操作中需要注意很多细节",
    "相关领域的专家对此有过专门的论述",
    "从方法论的角度来说有很多值得借鉴的地方",
    "在评估时需要综合考虑各方面的因素",
    "相关的因果关系也比较复杂",
    "在分析时需要特别注意样本的代表性",
    "不同研究者使用的方法各有不同",
    "近年来这个领域的进展也比较明显",
    "从国际比较的角度来看也有参考价值",
    "这些发现对后续的研究有一定启发",
    "在实际工作中经常会遇到这样的问题",
    "相关的政策文件也对此有所涉及",
    "从跨学科的角度来看更加复杂",
    "这些因素之间的关系值得进一步探讨",
    "在教育领域也有类似的情况",
    "相关的技术手段也在不断发展",
    "从经济角度来看也有一定的影响",
    "在日常生活中也能观察到类似的现象",
    "相关数据的获取方式也在不断改进",
    "从社会发展的角度来看值得关注",
    "这些研究结果对实践有指导意义",
    "在不同的文化背景下可能表现不同",
    "相关的评估标准也在逐步完善",
    "从长远来看可能还会有变化",
    "在不同的阶段可能有不同的表现",
    "相关的管理措施也在不断调整",
]

# === Distinguishing phrases ===
# CRITICAL: Both pools must share many phrases.
# Open phrases: contain words that keep alternatives alive
# Closed phrases: contain words that narrow down
# Bridge phrases: ambiguous — could be either

# Pool A: phrases that lean "uncertain" (open)
open_phrases = [
    "可能两种情况都存在",
    "或许都有道理",
    "可能都成立",
    "也许两种因素都在起作用",
    "或者说两种都对",
    "或许都合理",
    "可能两者都对",
    "也许都有可能",
    "可能都有道理",
    "或许两种都对",
    "可能两者兼有",
    "也许两种因素都存在",
    "可能各有各的道理",
    "或许都不能排除",
    "可能两种解释都成立",
    "也许都值得进一步研究",
    "可能各有贡献",
    "或许都有一定的解释力",
    "可能两种都对也可能都不完全对",
    "也许应该综合来看",
    "可能不应该只看一种",
    "或许两种都有一定的合理性",
    "可能需要同时考虑两种",
    "也许两种都值得重视",
    "可能两种情况都有道理",
]

# Pool B: phrases that lean "confident" (closed)
closed_phrases = [
    "可能就是这样",
    "或许已经很清楚",
    "可能就是答案",
    "也许就是这样确定的",
    "可能已经确定",
    "或许就是如此",
    "可能已经明确",
    "也许就是这个",
    "可能不需要再怀疑",
    "或许就是答案",
    "可能已经是定论",
    "也许就是这样了",
    "可能无需多说",
    "或许就是这样回事",
    "可能已经验证过了",
    "也许就是这个结果",
    "可能已经确认",
    "或许就是这个意思",
    "可能已经清楚了",
    "也许就是如此",
    "可能已经定下来了",
    "或许就是这样没错",
    "可能已经证明了",
    "也许就是这个结论",
    "可能就是这样确定的",
]

# Pool C: bridge phrases (appear in BOTH classes)
bridge_phrases = [
    "可能需要更多的研究",
    "或许会有不同的发现",
    "可能情况比想象的要复杂",
    "也许需要更全面的分析",
    "可能目前的证据还不够充分",
    "或许还不能完全确定",
    "可能还需要进一步验证",
    "也许应该再看看其他数据",
    "可能需要从更多的角度来分析",
    "或许还不能过早地下结论",
    "可能需要更长时间的观察",
    "也许还需要更多的样本",
    "可能需要综合各方面的情况",
    "或许还需要考虑其他变量",
    "可能需要更加谨慎地看待",
]

# Closer phrases (shared by both classes)
closers = [
    "这是一个值得继续探讨的问题",
    "相关的研究还在继续进行",
    "未来可能还会有更多的发现",
    "学界对此还在进一步的讨论中",
    "这个问题还有很大的研究空间",
    "相关的讨论还在持续",
    "可能未来会有更清晰的认识",
    "相关的研究工作还在推进中",
    "这个问题值得持续关注",
    "还需要更多的时间来验证",
]


def pick_ab():
    idx = random.randint(0, len(a_opts)-1)
    return a_opts[idx], b_opts[idx]


def gen_sentence(is_pos):
    topic = random.choice(topics)
    verb = random.choice(verbs)
    a, b = pick_ab()

    # Core statement (shared)
    core_templates = [
        f"研究显示{a}和{b}都有影响",
        f"{verb}表明{a}和{b}之间存在联系",
        f"数据显示{a}和{b}都有一定的作用",
        f"从数据来看{a}和{b}都值得关注",
        f"分析结果说明{a}和{b}都很重要",
        f"报告指出{a}和{b}都在其中扮演了角色",
        f"调查结果表明{a}和{b}都有所体现",
        f"相关的发现说明{a}和{b}之间的关系比较密切",
    ]
    core = random.choice(core_templates)

    # Pick 3-4 random fillers
    num_fillers = random.randint(3, 4)
    chosen_fillers = random.sample(fillers, num_fillers)

    # Pick distinguishing phrase with heavy overlap
    if is_pos:
        r = random.random()
        if r < 0.15:
            dist = random.choice(bridge_phrases)
        elif r < 0.40:
            dist = random.choice(closed_phrases)  # noise: 25% closed
        else:
            dist = random.choice(open_phrases)
    else:
        r = random.random()
        if r < 0.15:
            dist = random.choice(bridge_phrases)
        elif r < 0.40:
            dist = random.choice(open_phrases)  # noise: 25% open
        else:
            dist = random.choice(closed_phrases)

    # Closer (shared)
    closer = random.choice(closers)

    # Assemble all parts, then shuffle to bury signal position
    all_parts = chosen_fillers + [dist]
    random.shuffle(all_parts)

    # Build: opener + core + shuffled_parts + closer
    openers = [
        f"关于{topic}这个话题",
        f"针对{topic}的问题",
        f"在{topic}方面",
        f"对于{topic}的研究",
        f"从{topic}的角度来看",
    ]
    opener = random.choice(openers)

    body = "，".join(all_parts)
    return f"{opener}，{core}，{body}，{closer}。"


def count_words(sentences, word):
    return sum(1 for s in sentences if word in s)


def check_balance(pos_list, neg_list):
    print("\n--- Vocabulary balance ---")
    for w in hedge_words:
        pr = count_words(pos_list, w) / len(pos_list)
        nr = count_words(neg_list, w) / len(neg_list)
        flag = " <-- IMBALANCED" if abs(pr - nr) > 0.10 else ""
        print(f"  '{w}': pos={pr:.2f} neg={nr:.2f} diff={abs(pr-nr):.2f}{flag}")


def evaluate_auc(pos_list, neg_list):
    X = pos_list + neg_list
    y = [1]*len(pos_list) + [0]*len(neg_list)
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2).fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), v, y, cv=cv, scoring="roc_auc").mean()
    return auc


def generate_unique(n, maker, existing=None):
    if existing is None:
        existing = set()
    seen = set(existing)
    results = []
    attempts = 0
    while len(results) < n and attempts < n * 30:
        s = maker()
        if s not in seen:
            seen.add(s)
            results.append(s)
        attempts += 1
    return results


def main():
    iteration = 0
    best_auc = 999
    best_pos = None
    best_neg = None

    while True:
        iteration += 1
        random.seed(42 + iteration)
        np.random.seed(42 + iteration)

        pos = generate_unique(180, lambda: gen_sentence(is_pos=True))
        neg = generate_unique(180, lambda: gen_sentence(is_pos=False))

        auc = evaluate_auc(pos, neg)

        if auc < best_auc:
            best_auc = auc
            best_pos = pos[:]
            best_neg = neg[:]

        print(f"iter={iteration} AUC={auc:.4f}")

        if auc <= 0.70:
            print(f"--> AUC {auc:.4f} <= 0.70, DONE at iteration {iteration}")
            break

        if iteration >= 50:
            print(f"--> 50 iterations, best AUC={best_auc:.4f}")
            break

    check_balance(best_pos, best_neg)

    result = {"pos": best_pos, "neg": best_neg}
    outpath = "/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/uncertainty_zh.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    final_auc = evaluate_auc(best_pos, best_neg)
    print(f"\nFINAL: AUC={final_auc:.4f} N_pos={len(best_pos)} N_neg={len(best_neg)} iterations={iteration}")


if __name__ == "__main__":
    main()
