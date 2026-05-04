"""
Exp BR: Diverse-Problem Gram Replication — "Is Reasoning Universal?"

The strongest objection to BQ/BQ2/BQ3: "200 arithmetic problems are all structurally
the same. Of course the Gram matrix is low-rank — you're feeding the model slight
variations of one task. rank_50=1 just means 'math is one thing.'"

The killer test: problems where reasoning IS language-dependent.
  - Logical ordering: "A is taller than B, B is taller than C. Who is shortest?"
  - Syllogisms: "All birds can fly. Tweety is a bird. Can Tweety fly?"
  - Common sense: "The ice cream was left in the sun. What happened?"
  - Analogies: "Hot is to cold as day is to ___?"

If rank_50=1 persists across ALL of these, the finding upgrades from
"math is language-agnostic" (trivial) to "reasoning is universal" (massive).

Method:
  1. Generate 200 diverse problems in 7 languages (50 per category)
  2. Extract last-token hidden states at all 36 layers (same as expAB)
  3. Compute Gram eigendecomposition for:
     a. Math-only (from existing cache)
     b. Diverse-only (new)
     c. Combined (400 problems × 7 langs = 2800 vectors)
  4. Compare rank structure, Lyapunov spectrum, and phase pattern
  5. Domain cross-correlation: do math and non-math cluster separately?

If rank_50 jumps for diverse: finding is real but task-specific (different claim).
If rank_50 stays at 1: architecture-level result, objection is dead.
"""

import numpy as np
import torch
import json
import time
import random as pyrandom
import gc
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from diverse_vocab import SYLLOGISM_VOCAB, ANALOGY_VOCAB, translate_word, translate_syllogism_word

MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MATH_CACHE = OUTPUT_DIR / "multilingual_all_layers.npz"
DIVERSE_CACHE = OUTPUT_DIR / "diverse_all_layers.npz"
OUT_PATH = OUTPUT_DIR / "expBR_diverse_gram.json"

SEED = 42
LANGS = ["ar", "en", "es", "ja", "ko", "sw", "zh"]
N_LAYERS = 36
N_MATH = 200
N_DIVERSE = 200
N_PER_CAT = 50  # 4 categories × 50

# Domain labels: 0-4 = math (from original), 5-8 = diverse
DOMAIN_NAMES = {
    0: "arithmetic", 1: "combinatorics", 2: "modular",
    3: "geometry", 4: "sequences",
    5: "logical_ordering", 6: "syllogisms",
    7: "common_sense", 8: "analogies",
}

# ================================================================
# PROBLEM GENERATION — Language-dependent reasoning
# ================================================================

# --- Logical ordering templates ---
# Uses letter names (universal) + comparative relations
ORDERING_TEMPLATES = {
    "ar": {
        "taller": "{a} أطول من {b}.",
        "shorter": "{a} أقصر من {b}.",
        "q_tallest": "من الأطول؟",
        "q_shortest": "من الأقصر؟",
        "older": "{a} أكبر سناً من {b}.",
        "younger": "{a} أصغر سناً من {b}.",
        "q_oldest": "من الأكبر سناً؟",
        "q_youngest": "من الأصغر سناً؟",
        "heavier": "{a} أثقل من {b}.",
        "lighter": "{a} أخف من {b}.",
        "q_heaviest": "من الأثقل؟",
        "q_lightest": "من الأخف؟",
    },
    "en": {
        "taller": "{a} is taller than {b}.",
        "shorter": "{a} is shorter than {b}.",
        "q_tallest": "Who is the tallest?",
        "q_shortest": "Who is the shortest?",
        "older": "{a} is older than {b}.",
        "younger": "{a} is younger than {b}.",
        "q_oldest": "Who is the oldest?",
        "q_youngest": "Who is the youngest?",
        "heavier": "{a} is heavier than {b}.",
        "lighter": "{a} is lighter than {b}.",
        "q_heaviest": "Who is the heaviest?",
        "q_lightest": "Who is the lightest?",
    },
    "es": {
        "taller": "{a} es más alto que {b}.",
        "shorter": "{a} es más bajo que {b}.",
        "q_tallest": "¿Quién es el más alto?",
        "q_shortest": "¿Quién es el más bajo?",
        "older": "{a} es mayor que {b}.",
        "younger": "{a} es menor que {b}.",
        "q_oldest": "¿Quién es el mayor?",
        "q_youngest": "¿Quién es el menor?",
        "heavier": "{a} es más pesado que {b}.",
        "lighter": "{a} es más ligero que {b}.",
        "q_heaviest": "¿Quién es el más pesado?",
        "q_lightest": "¿Quién es el más ligero?",
    },
    "ja": {
        "taller": "{a}は{b}より背が高い。",
        "shorter": "{a}は{b}より背が低い。",
        "q_tallest": "誰が一番背が高いですか？",
        "q_shortest": "誰が一番背が低いですか？",
        "older": "{a}は{b}より年上だ。",
        "younger": "{a}は{b}より年下だ。",
        "q_oldest": "誰が一番年上ですか？",
        "q_youngest": "誰が一番年下ですか？",
        "heavier": "{a}は{b}より重い。",
        "lighter": "{a}は{b}より軽い。",
        "q_heaviest": "誰が一番重いですか？",
        "q_lightest": "誰が一番軽いですか？",
    },
    "ko": {
        "taller": "{a}은(는) {b}보다 키가 크다.",
        "shorter": "{a}은(는) {b}보다 키가 작다.",
        "q_tallest": "누가 가장 키가 큰가?",
        "q_shortest": "누가 가장 키가 작은가?",
        "older": "{a}은(는) {b}보다 나이가 많다.",
        "younger": "{a}은(는) {b}보다 나이가 적다.",
        "q_oldest": "누가 가장 나이가 많은가?",
        "q_youngest": "누가 가장 나이가 적은가?",
        "heavier": "{a}은(는) {b}보다 무겁다.",
        "lighter": "{a}은(는) {b}보다 가볍다.",
        "q_heaviest": "누가 가장 무거운가?",
        "q_lightest": "누가 가장 가벼운가?",
    },
    "sw": {
        "taller": "{a} ni mrefu kuliko {b}.",
        "shorter": "{a} ni mfupi kuliko {b}.",
        "q_tallest": "Nani mrefu zaidi?",
        "q_shortest": "Nani mfupi zaidi?",
        "older": "{a} ni mkubwa kuliko {b}.",
        "younger": "{a} ni mdogo kuliko {b}.",
        "q_oldest": "Nani mkubwa zaidi?",
        "q_youngest": "Nani mdogo zaidi?",
        "heavier": "{a} ni mzito kuliko {b}.",
        "lighter": "{a} ni mwepesi kuliko {b}.",
        "q_heaviest": "Nani mzito zaidi?",
        "q_lightest": "Nani mwepesi zaidi?",
    },
    "zh": {
        "taller": "{a}比{b}高。",
        "shorter": "{a}比{b}矮。",
        "q_tallest": "谁最高？",
        "q_shortest": "谁最矮？",
        "older": "{a}比{b}年长。",
        "younger": "{a}比{b}年轻。",
        "q_oldest": "谁最年长？",
        "q_youngest": "谁最年轻？",
        "heavier": "{a}比{b}重。",
        "lighter": "{a}比{b}轻。",
        "q_heaviest": "谁最重？",
        "q_lightest": "谁最轻？",
    },
}

# --- Syllogism templates ---
SYLLOGISM_TEMPLATES = {
    "ar": {
        "all_can":  "كل {cat} يمكنها {act}. {name} هو {cat}. هل يمكن لـ{name} {act}؟",
        "no_can":   "لا {cat} يمكنها {act}. {name} هو {cat}. هل يمكن لـ{name} {act}؟",
        "all_are":  "كل {cat} هي {prop}. {name} هو {cat}. هل {name} {prop}؟",
        "no_are":   "لا {cat} هي {prop}. {name} هو {cat}. هل {name} {prop}؟",
    },
    "en": {
        "all_can":  "All {cat} can {act}. {name} is a {cat}. Can {name} {act}?",
        "no_can":   "No {cat} can {act}. {name} is a {cat}. Can {name} {act}?",
        "all_are":  "All {cat} are {prop}. {name} is a {cat}. Is {name} {prop}?",
        "no_are":   "No {cat} are {prop}. {name} is a {cat}. Is {name} {prop}?",
    },
    "es": {
        "all_can":  "Todos los {cat} pueden {act}. {name} es un {cat}. ¿Puede {name} {act}?",
        "no_can":   "Ningún {cat} puede {act}. {name} es un {cat}. ¿Puede {name} {act}?",
        "all_are":  "Todos los {cat} son {prop}. {name} es un {cat}. ¿Es {name} {prop}?",
        "no_are":   "Ningún {cat} es {prop}. {name} es un {cat}. ¿Es {name} {prop}?",
    },
    "ja": {
        "all_can":  "すべての{cat}は{act}ことができる。{name}は{cat}だ。{name}は{act}ことができるか？",
        "no_can":   "{cat}は{act}ことができない。{name}は{cat}だ。{name}は{act}ことができるか？",
        "all_are":  "すべての{cat}は{prop}だ。{name}は{cat}だ。{name}は{prop}か？",
        "no_are":   "{cat}は{prop}ではない。{name}は{cat}だ。{name}は{prop}か？",
    },
    "ko": {
        "all_can":  "모든 {cat}은(는) {act} 수 있다. {name}은(는) {cat}이다. {name}은(는) {act} 수 있는가?",
        "no_can":   "어떤 {cat}도 {act} 수 없다. {name}은(는) {cat}이다. {name}은(는) {act} 수 있는가?",
        "all_are":  "모든 {cat}은(는) {prop}이다. {name}은(는) {cat}이다. {name}은(는) {prop}인가?",
        "no_are":   "어떤 {cat}도 {prop}이(가) 아니다. {name}은(는) {cat}이다. {name}은(는) {prop}인가?",
    },
    "sw": {
        "all_can":  "{cat} wote wanaweza {act}. {name} ni {cat}. Je, {name} anaweza {act}?",
        "no_can":   "Hakuna {cat} anayeweza {act}. {name} ni {cat}. Je, {name} anaweza {act}?",
        "all_are":  "{cat} wote ni {prop}. {name} ni {cat}. Je, {name} ni {prop}?",
        "no_are":   "Hakuna {cat} ni {prop}. {name} ni {cat}. Je, {name} ni {prop}?",
    },
    "zh": {
        "all_can":  "所有{cat}都会{act}。{name}是{cat}。{name}会{act}吗？",
        "no_can":   "没有{cat}会{act}。{name}是{cat}。{name}会{act}吗？",
        "all_are":  "所有{cat}都是{prop}的。{name}是{cat}。{name}是{prop}的吗？",
        "no_are":   "没有{cat}是{prop}的。{name}是{cat}。{name}是{prop}的吗？",
    },
}

# Syllogism content — categories, actions, properties, names
SYLLOGISM_CONTENT = {
    "cats_acts": [
        ("birds", "fly"), ("fish", "swim"), ("dogs", "bark"),
        ("cats", "climb trees"), ("snakes", "slither"),
        ("horses", "gallop"), ("frogs", "jump"), ("eagles", "soar"),
        ("wolves", "howl"), ("bees", "make honey"),
        ("dolphins", "echolocate"), ("bats", "see in the dark"),
        ("rabbits", "dig burrows"),
    ],
    "cats_props": [
        ("mammals", "warm-blooded"), ("reptiles", "cold-blooded"),
        ("metals", "shiny"), ("flowers", "colorful"),
        ("planets", "round"), ("stars", "luminous"),
        ("doctors", "educated"), ("athletes", "strong"),
        ("children", "curious"), ("teachers", "patient"),
        ("scientists", "logical"), ("artists", "creative"),
    ],
    "names": ["Alex", "Sam", "Jordan", "Riley", "Morgan", "Casey", "Quinn", "Avery",
              "Taylor", "Blake", "Robin", "Dana", "Lee", "Max", "Kai"],
}

# --- Common sense templates ---
COMMON_SENSE_TEMPLATES = {
    "ar": "{premise} {question}",
    "en": "{premise} {question}",
    "es": "{premise} {question}",
    "ja": "{premise} {question}",
    "ko": "{premise} {question}",
    "sw": "{premise} {question}",
    "zh": "{premise} {question}",
}

# Premise-question pairs in all 7 languages
# Each entry: (premise_dict, question_dict) where dict is {lang: text}
COMMON_SENSE_ITEMS = [
    # Physical causation
    ({"ar": "ترك أحمد الآيس كريم في الشمس لمدة ساعة.",
      "en": "The ice cream was left in the sun for an hour.",
      "es": "El helado se dejó al sol durante una hora.",
      "ja": "アイスクリームが1時間日光の下に置かれた。",
      "ko": "아이스크림이 햇볕 아래 1시간 동안 놓여 있었다.",
      "sw": "Aiskrimu iliachwa kwenye jua kwa saa moja.",
      "zh": "冰淇淋在太阳下放了一个小时。"},
     {"ar": "ماذا حدث للآيس كريم؟",
      "en": "What happened to the ice cream?",
      "es": "¿Qué le pasó al helado?",
      "ja": "アイスクリームはどうなったか？",
      "ko": "아이스크림은 어떻게 되었는가?",
      "sw": "Nini kilitokea kwa aiskrimu?",
      "zh": "冰淇淋怎么了？"}),
    ({"ar": "سقط الكوب الزجاجي من الطاولة على الأرض.",
      "en": "The glass cup fell from the table onto the floor.",
      "es": "El vaso de vidrio cayó de la mesa al suelo.",
      "ja": "ガラスのコップがテーブルから床に落ちた。",
      "ko": "유리컵이 탁자에서 바닥으로 떨어졌다.",
      "sw": "Kikombe cha kioo kilianguka kutoka mezani hadi sakafuni.",
      "zh": "玻璃杯从桌子上掉到地上。"},
     {"ar": "ماذا حدث للكوب على الأرجح؟",
      "en": "What most likely happened to the cup?",
      "es": "¿Qué le pasó probablemente al vaso?",
      "ja": "コップはおそらくどうなったか？",
      "ko": "컵에 무슨 일이 일어났을 가능성이 높은가?",
      "sw": "Nini kilitokea kwa kikombe?",
      "zh": "杯子最可能怎么了？"}),
    ({"ar": "لم تُسقَ النباتات لمدة ثلاثة أسابيع.",
      "en": "The plants were not watered for three weeks.",
      "es": "Las plantas no fueron regadas durante tres semanas.",
      "ja": "植物は3週間水をやられなかった。",
      "ko": "식물에 3주 동안 물을 주지 않았다.",
      "sw": "Mimea haikumwagiliwa kwa wiki tatu.",
      "zh": "植物三个星期没有浇水。"},
     {"ar": "ماذا حدث للنباتات على الأرجح؟",
      "en": "What most likely happened to the plants?",
      "es": "¿Qué les pasó probablemente a las plantas?",
      "ja": "植物はおそらくどうなったか？",
      "ko": "식물은 어떻게 되었을 가능성이 높은가?",
      "sw": "Nini kilitokea kwa mimea?",
      "zh": "植物最可能怎么了？"}),
    ({"ar": "ترك الباب مفتوحاً في ليلة شتاء باردة.",
      "en": "The door was left open on a cold winter night.",
      "es": "La puerta se dejó abierta en una noche fría de invierno.",
      "ja": "寒い冬の夜にドアが開けっぱなしにされた。",
      "ko": "추운 겨울 밤에 문이 열린 채 방치되었다.",
      "sw": "Mlango uliachwa wazi usiku wa baridi wa majira ya baridi.",
      "zh": "在寒冷的冬夜，门被敞开着。"},
     {"ar": "ماذا حدث لدرجة حرارة الغرفة؟",
      "en": "What happened to the room temperature?",
      "es": "¿Qué pasó con la temperatura de la habitación?",
      "ja": "部屋の温度はどうなったか？",
      "ko": "방 온도는 어떻게 되었는가?",
      "sw": "Nini kilitokea kwa joto la chumba?",
      "zh": "房间温度怎么了？"}),
    ({"ar": "وضعت الكعكة في الفرن على درجة حرارة 200 لمدة ساعة.",
      "en": "The cake was put in the oven at 200 degrees for an hour.",
      "es": "El pastel se puso en el horno a 200 grados durante una hora.",
      "ja": "ケーキは200度のオーブンに1時間入れられた。",
      "ko": "케이크가 200도 오븐에 1시간 동안 넣어졌다.",
      "sw": "Keki iliwekwa kwenye oveni kwa nyuzi 200 kwa saa moja.",
      "zh": "蛋糕放在200度的烤箱里烤了一个小时。"},
     {"ar": "ماذا حدث للكعكة؟",
      "en": "What happened to the cake?",
      "es": "¿Qué le pasó al pastel?",
      "ja": "ケーキはどうなったか？",
      "ko": "케이크는 어떻게 되었는가?",
      "sw": "Nini kilitokea kwa keki?",
      "zh": "蛋糕怎么了？"}),
    # Social/behavioral inference
    ({"ar": "نظرت سارة إلى السماء الملبدة بالغيوم وأخذت مظلتها.",
      "en": "Sarah looked at the cloudy sky and grabbed her umbrella.",
      "es": "Sara miró el cielo nublado y agarró su paraguas.",
      "ja": "サラは曇り空を見て傘を取った。",
      "ko": "사라는 흐린 하늘을 보고 우산을 가져갔다.",
      "sw": "Sara alitazama anga lenye mawingu na akachukua mwavuli wake.",
      "zh": "萨拉看了看阴沉的天空，拿起了雨伞。"},
     {"ar": "ماذا تتوقع سارة بخصوص الطقس؟",
      "en": "What does Sarah expect about the weather?",
      "es": "¿Qué espera Sara sobre el clima?",
      "ja": "サラは天気についてどう予想しているか？",
      "ko": "사라는 날씨에 대해 무엇을 예상하는가?",
      "sw": "Sara anatarajia nini kuhusu hali ya hewa?",
      "zh": "萨拉对天气有什么预期？"}),
    ({"ar": "ارتدى الرجل معطفاً ثقيلاً وقفازات ووشاحاً.",
      "en": "The man put on a heavy coat, gloves, and a scarf.",
      "es": "El hombre se puso un abrigo grueso, guantes y una bufanda.",
      "ja": "男は厚いコートと手袋とマフラーを身につけた。",
      "ko": "남자는 두꺼운 코트, 장갑, 목도리를 착용했다.",
      "sw": "Mtu huyo alivaa koti zito, glavu, na skafu.",
      "zh": "那个男人穿上了厚外套、手套和围巾。"},
     {"ar": "ما هو الطقس المحتمل في الخارج؟",
      "en": "What is the weather likely outside?",
      "es": "¿Cómo es probable que esté el clima afuera?",
      "ja": "外の天気はどうだと思われるか？",
      "ko": "밖의 날씨는 어떨 가능성이 높은가?",
      "sw": "Hali ya hewa nje ni ipi?",
      "zh": "外面天气可能是怎样的？"}),
    ({"ar": "ابتسم الطفل وقفز عندما رأى هديته.",
      "en": "The child smiled and jumped when they saw their gift.",
      "es": "El niño sonrió y saltó cuando vio su regalo.",
      "ja": "子供はプレゼントを見て笑顔で飛び跳ねた。",
      "ko": "아이는 선물을 보고 웃으며 뛰어올랐다.",
      "sw": "Mtoto alitabasamu na kuruka alipopata zawadi yake.",
      "zh": "孩子看到礼物时笑了并跳了起来。"},
     {"ar": "كيف يشعر الطفل؟",
      "en": "How does the child feel?",
      "es": "¿Cómo se siente el niño?",
      "ja": "子供はどう感じているか？",
      "ko": "아이의 기분은 어떠한가?",
      "sw": "Mtoto anahisi vipi?",
      "zh": "孩子感觉怎么样？"}),
    ({"ar": "فقد الفريق المباراة النهائية. جلس اللاعبون صامتين في غرفة الملابس.",
      "en": "The team lost the final match. The players sat quietly in the locker room.",
      "es": "El equipo perdió el partido final. Los jugadores se sentaron en silencio en el vestuario.",
      "ja": "チームは決勝戦に負けた。選手たちはロッカールームで静かに座っていた。",
      "ko": "팀이 결승전에서 졌다. 선수들은 라커룸에서 조용히 앉아 있었다.",
      "sw": "Timu ilishindwa mechi ya mwisho. Wachezaji walikaa kimya kwenye chumba cha kubadilishia.",
      "zh": "球队输掉了决赛。球员们静静地坐在更衣室里。"},
     {"ar": "كيف يشعر اللاعبون على الأرجح؟",
      "en": "How do the players most likely feel?",
      "es": "¿Cómo se sienten probablemente los jugadores?",
      "ja": "選手たちはおそらくどう感じているか？",
      "ko": "선수들은 어떤 기분일 가능성이 높은가?",
      "sw": "Wachezaji wanahisi vipi?",
      "zh": "球员们最可能感觉怎样？"}),
    ({"ar": "تثاءب أحمد مرات عديدة وأغمض عينيه ببطء.",
      "en": "Ahmed yawned many times and slowly closed his eyes.",
      "es": "Ahmed bostezó muchas veces y cerró lentamente los ojos.",
      "ja": "アフメドは何度もあくびをし、ゆっくり目を閉じた。",
      "ko": "아흐메드는 여러 번 하품을 하고 천천히 눈을 감았다.",
      "sw": "Ahmed alipiga miayo mara nyingi na kufunga macho yake polepole.",
      "zh": "艾哈迈德打了好几个哈欠，慢慢地闭上了眼睛。"},
     {"ar": "ما حالة أحمد؟",
      "en": "What is Ahmed's state?",
      "es": "¿Cuál es el estado de Ahmed?",
      "ja": "アフメドの状態は何か？",
      "ko": "아흐메드의 상태는 무엇인가?",
      "sw": "Hali ya Ahmed ni ipi?",
      "zh": "艾哈迈德处于什么状态？"}),
]

# --- Analogy templates ---
ANALOGY_TEMPLATES = {
    "ar": "{a} بالنسبة لـ{b} كما {c} بالنسبة لماذا؟",
    "en": "{a} is to {b} as {c} is to what?",
    "es": "{a} es a {b} como {c} es a ¿qué?",
    "ja": "{a}と{b}の関係は、{c}と何の関係と同じか？",
    "ko": "{a}이(가) {b}에 대한 것처럼, {c}은(는) 무엇에 대한 것인가?",
    "sw": "{a} ni kwa {b} kama {c} ni kwa nini?",
    "zh": "{a}之于{b}，如同{c}之于什么？",
}

# Analogy pairs: (A, B, C, answer) — answer not used for activation capture
# but tracked for analysis. These are SEMANTIC relationships encoded in language.
ANALOGY_PAIRS = [
    # Antonyms
    ("hot", "cold", "big", "small"),
    ("up", "down", "left", "right"),
    ("fast", "slow", "light", "dark"),
    ("happy", "sad", "rich", "poor"),
    ("open", "close", "start", "stop"),
    ("day", "night", "summer", "winter"),
    ("love", "hate", "peace", "war"),
    ("hard", "soft", "dry", "wet"),
    ("full", "empty", "loud", "quiet"),
    ("young", "old", "new", "ancient"),
    # Part-whole
    ("wheel", "car", "page", "book"),
    ("leaf", "tree", "feather", "bird"),
    ("key", "keyboard", "brick", "wall"),
    ("finger", "hand", "toe", "foot"),
    ("petal", "flower", "scale", "fish"),
    # Profession-tool
    ("painter", "brush", "writer", "pen"),
    ("chef", "knife", "carpenter", "hammer"),
    ("doctor", "stethoscope", "teacher", "blackboard"),
    ("farmer", "plow", "sailor", "compass"),
    ("musician", "instrument", "photographer", "camera"),
    # Animal-young
    ("dog", "puppy", "cat", "kitten"),
    ("horse", "foal", "cow", "calf"),
    ("lion", "cub", "sheep", "lamb"),
    ("hen", "chick", "duck", "duckling"),
    ("bear", "cub", "deer", "fawn"),
    # Category membership
    ("apple", "fruit", "carrot", "vegetable"),
    ("piano", "instrument", "chair", "furniture"),
    ("French", "language", "soccer", "sport"),
    ("hammer", "tool", "rose", "flower"),
    ("gold", "metal", "diamond", "gem"),
    # Degree/intensity
    ("warm", "hot", "cool", "cold"),
    ("jog", "run", "walk", "march"),
    ("like", "love", "dislike", "hate"),
    ("hill", "mountain", "stream", "river"),
    ("pond", "lake", "path", "road"),
    # Function
    ("eye", "see", "ear", "hear"),
    ("nose", "smell", "tongue", "taste"),
    ("brain", "think", "heart", "pump"),
    ("lungs", "breathe", "stomach", "digest"),
    ("mouth", "speak", "legs", "walk"),
    # Sequence/progression
    ("Monday", "Tuesday", "January", "February"),
    ("first", "second", "beginning", "middle"),
    ("spring", "summer", "morning", "afternoon"),
    ("child", "adult", "seed", "plant"),
    ("egg", "chicken", "caterpillar", "butterfly"),
    # Material
    ("glass", "window", "wood", "door"),
    ("cotton", "shirt", "leather", "shoe"),
    ("steel", "bridge", "clay", "pot"),
    ("paper", "book", "canvas", "painting"),
    ("silk", "dress", "wool", "sweater"),
]


def generate_ordering_problems(n=50, seed=42):
    """Generate logical ordering problems in 7 languages."""
    rng = pyrandom.Random(seed + 100)
    names = list("ABCDEFGH")
    relation_types = [
        ("taller", "shorter", "q_tallest", "q_shortest"),
        ("older", "younger", "q_oldest", "q_youngest"),
        ("heavier", "lighter", "q_heaviest", "q_lightest"),
    ]

    problems = []
    for i in range(n):
        # Pick 3-4 entities
        n_entities = rng.choice([3, 3, 3, 4])  # bias toward 3 for clarity
        entities = rng.sample(names[:6], n_entities)
        rel_type = rng.choice(relation_types)
        more_key, less_key, q_most, q_least = rel_type

        # Generate a random total ordering
        order = list(entities)
        rng.shuffle(order)  # order[0] = most, order[-1] = least

        # Generate comparison statements (enough to determine the order)
        statements_params = []
        for j in range(len(order) - 1):
            # order[j] > order[j+1]
            if rng.random() < 0.5:
                statements_params.append((more_key, order[j], order[j+1]))
            else:
                statements_params.append((less_key, order[j+1], order[j]))

        # Shuffle statement order for variety
        rng.shuffle(statements_params)

        # Ask about most or least
        ask_most = rng.random() < 0.5
        q_key = q_most if ask_most else q_least
        answer = order[0] if ask_most else order[-1]

        # Build in all 7 languages
        row = {"category": 5, "answer": answer}
        for lang in LANGS:
            t = ORDERING_TEMPLATES[lang]
            stmts = " ".join(t[key].format(a=a, b=b) for key, a, b in statements_params)
            row[lang] = stmts + " " + t[q_key]
        problems.append(row)

    return problems


def generate_syllogism_problems(n=50, seed=42):
    """Generate syllogism problems in 7 languages."""
    rng = pyrandom.Random(seed + 200)
    problems = []

    cats_acts = SYLLOGISM_CONTENT["cats_acts"]
    cats_props = SYLLOGISM_CONTENT["cats_props"]
    all_names = SYLLOGISM_CONTENT["names"]

    for i in range(n):
        name = rng.choice(all_names)
        if rng.random() < 0.5:
            # Action-based: "All/No X can Y"
            cat, act = rng.choice(cats_acts)
            form = rng.choice(["all_can", "no_can"])
            row = {"category": 6, "answer": "yes" if form == "all_can" else "no"}
            for lang in LANGS:
                t_cat = translate_syllogism_word(cat, lang)
                t_act = translate_syllogism_word(act, lang)
                row[lang] = SYLLOGISM_TEMPLATES[lang][form].format(
                    cat=t_cat, act=t_act, name=name)
        else:
            # Property-based: "All/No X are Y"
            cat, prop = rng.choice(cats_props)
            form = rng.choice(["all_are", "no_are"])
            row = {"category": 6, "answer": "yes" if form == "all_are" else "no"}
            for lang in LANGS:
                t_cat = translate_syllogism_word(cat, lang)
                t_prop = translate_syllogism_word(prop, lang)
                row[lang] = SYLLOGISM_TEMPLATES[lang][form].format(
                    cat=t_cat, prop=t_prop, name=name)
        problems.append(row)

    return problems


def generate_common_sense_problems(n=50, seed=42):
    """Generate common sense inference problems in 7 languages.
    Uses hand-written items (high quality) + templated repetitions with variation."""
    rng = pyrandom.Random(seed + 300)
    problems = []

    # Use hand-written items, cycling with slight index variation
    n_items = len(COMMON_SENSE_ITEMS)
    for i in range(n):
        idx = i % n_items
        premise_dict, question_dict = COMMON_SENSE_ITEMS[idx]
        row = {"category": 7}
        for lang in LANGS:
            row[lang] = premise_dict[lang] + " " + question_dict[lang]
        problems.append(row)

    return problems


def generate_analogy_problems(n=50, seed=42):
    """Generate analogy problems in 7 languages."""
    rng = pyrandom.Random(seed + 400)
    problems = []

    pairs = ANALOGY_PAIRS[:n]  # use first n pairs
    for a_word, b_word, c_word, d_word in pairs:
        row = {"category": 8, "answer": d_word}
        for lang in LANGS:
            t_a = translate_word(a_word, lang)
            t_b = translate_word(b_word, lang)
            t_c = translate_word(c_word, lang)
            row[lang] = ANALOGY_TEMPLATES[lang].format(a=t_a, b=t_b, c=t_c)
        problems.append(row)

    return problems


def generate_all_diverse_problems(seed=42):
    """Generate all 200 diverse problems."""
    ordering = generate_ordering_problems(N_PER_CAT, seed)
    syllogisms = generate_syllogism_problems(N_PER_CAT, seed)
    common = generate_common_sense_problems(N_PER_CAT, seed)
    analogies = generate_analogy_problems(N_PER_CAT, seed)

    all_probs = ordering + syllogisms + common + analogies
    rng = pyrandom.Random(seed)
    rng.shuffle(all_probs)
    return all_probs


# ================================================================
# ACTIVATION EXTRACTION (same infrastructure as expAB)
# ================================================================

def extract_activations(model, tokenizer, problems, n_layers, d):
    """Extract last-token hidden states for all 7 languages × all layers."""
    N = len(problems)
    all_acts = {lang: {l: np.zeros((N, d), dtype=np.float32) for l in range(n_layers)}
                for lang in LANGS}

    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h_out = output if isinstance(output, torch.Tensor) else output[0]
            layer_outputs[layer_idx] = h_out.detach().cpu().squeeze(0)[-1].float().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l))
               for l in range(n_layers)]

    try:
        for lang in LANGS:
            print(f"  Extracting {lang} ({N} problems)...")
            for i, prob in enumerate(tqdm(problems, desc=lang, leave=False)):
                inputs = tokenizer(prob[lang], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**inputs)
                for l in range(n_layers):
                    all_acts[lang][l][i] = layer_outputs[l]
                layer_outputs.clear()
    finally:
        for h in handles:
            h.remove()

    return all_acts


# ================================================================
# GRAM ANALYSIS (from BQ/BQ2)
# ================================================================

def compute_gram_analysis(grams, n_problems, categories, domain_names, label):
    """Full Gram matrix analysis: eigendecomposition, rank, Lyapunov, phase."""
    N_L = len(grams)
    n_langs = len(LANGS)

    results = {"label": label, "n_problems": n_problems, "n_langs": n_langs}

    # --- Eigendecomposition at all layers ---
    eigenvalues_all = []
    for L in range(N_L):
        eigenvals = np.linalg.eigvalsh(grams[L])[::-1]
        eigenvalues_all.append(eigenvals)
    eigenvalues_all = np.array(eigenvalues_all)

    # --- Effective rank ---
    effective_ranks = {}
    for threshold in [0.50, 0.90, 0.95, 0.99]:
        ranks = []
        for L in range(N_L):
            ev = eigenvalues_all[L]
            cumsum = np.cumsum(ev) / ev.sum()
            rank = int(np.searchsorted(cumsum, threshold) + 1)
            ranks.append(rank)
        effective_ranks[f"rank_{int(threshold*100)}"] = ranks
    results["effective_ranks"] = effective_ranks

    # --- Layer-to-layer Gram correlation ---
    layer_corrs = []
    layer_frob_deltas = []
    for L in range(N_L - 1):
        idx = np.triu_indices(grams[L].shape[0], k=1)
        u1 = grams[L][idx]
        u2 = grams[L+1][idx]
        corr = float(np.corrcoef(u1, u2)[0, 1])
        dG = grams[L+1] - grams[L]
        frob = float(np.linalg.norm(dG, 'fro') / np.linalg.norm(grams[L], 'fro'))
        layer_corrs.append(corr)
        layer_frob_deltas.append(frob)
    results["layer_gram_correlations"] = [round(c, 6) for c in layer_corrs]
    results["layer_frob_deltas"] = [round(f, 6) for f in layer_frob_deltas]
    results["mean_gram_correlation"] = round(float(np.mean(layer_corrs)), 6)

    # --- Lyapunov exponents (top 20 modes) ---
    TOP_K = min(20, eigenvalues_all.shape[1])
    lyapunov_per_mode = {}
    for k in range(TOP_K):
        log_ratios = []
        for L in range(N_L - 1):
            v_curr = abs(eigenvalues_all[L, k])
            v_next = abs(eigenvalues_all[L + 1, k])
            if v_curr > 1e-6:
                log_ratios.append(float(np.log(v_next / v_curr)))
            else:
                log_ratios.append(0.0)
        lyapunov_per_mode[f"mode_{k}"] = {
            "exponent": round(float(np.mean(log_ratios)), 6),
            "per_layer": [round(r, 6) for r in log_ratios],
        }
    results["lyapunov"] = {k: v["exponent"] for k, v in lyapunov_per_mode.items()}

    # --- Phase-resolved Lyapunov ---
    PHASES = {
        "early": list(range(0, 9)),
        "adversarial": list(range(9, 18)),
        "cooperative": list(range(18, 27)),
        "late": list(range(27, 36)),
    }
    phase_lyapunov = {}
    for phase_name, layers in PHASES.items():
        phase_exps = []
        for k in range(TOP_K):
            per_layer = lyapunov_per_mode[f"mode_{k}"]["per_layer"]
            phase_vals = [per_layer[L] for L in layers if L < N_L - 1]
            phase_exps.append(round(float(np.mean(phase_vals)), 6) if phase_vals else 0.0)
        n_positive = sum(1 for e in phase_exps if e > 0)
        n_negative = sum(1 for e in phase_exps if e < 0)
        phase_lyapunov[phase_name] = {
            "mean": round(float(np.mean(phase_exps)), 6),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "total_modes": TOP_K,
        }
    results["phase_lyapunov"] = phase_lyapunov

    # --- Within/cross language gap ---
    n = n_problems
    lang_gaps = []
    for L in range(N_L):
        G = grams[L]
        within_vals = []
        cross_vals = []
        for i in range(n_langs):
            for j in range(i, n_langs):
                block = G[i*n:(i+1)*n, j*n:(j+1)*n]
                if i == j:
                    idx = np.triu_indices(n, k=1)
                    within_vals.extend(block[idx].tolist())
                else:
                    cross_vals.extend(block.flatten().tolist())
        gap = float(np.mean(within_vals)) - float(np.mean(cross_vals))
        lang_gaps.append(round(gap, 4))
    results["language_gap_per_layer"] = lang_gaps

    # --- Domain separation (if categories have multiple domains) ---
    unique_cats = sorted(set(categories.tolist()))
    if len(unique_cats) > 1:
        domain_sep = {}
        for L in [0, 8, 12, 17, 18, 26, 32, 35]:
            if L >= N_L:
                continue
            G = grams[L]
            # Use first language block (en = index 1) for domain analysis
            en_block = G[n:2*n, n:2*n]
            same_dom = []
            diff_dom = []
            for i in range(n):
                for j in range(i+1, n):
                    if categories[i] == categories[j]:
                        same_dom.append(en_block[i, j])
                    else:
                        diff_dom.append(en_block[i, j])
            domain_sep[f"L{L}"] = {
                "same_domain_cos": round(float(np.mean(same_dom)), 4) if same_dom else 0,
                "diff_domain_cos": round(float(np.mean(diff_dom)), 4) if diff_dom else 0,
                "separation": round(float(np.mean(same_dom)) - float(np.mean(diff_dom)), 4) if same_dom and diff_dom else 0,
            }
        results["domain_separation"] = domain_sep

    # --- Eigenvalue trajectories (top 5) ---
    results["eigenvalue_trajectories"] = {
        f"mode_{k}": [round(float(eigenvalues_all[L, k]), 2) for L in range(N_L)]
        for k in range(min(5, TOP_K))
    }

    return results


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()

    # ── Step 1: Generate diverse problems ──
    print("Generating 200 diverse problems (language-dependent reasoning)...")
    diverse_problems = generate_all_diverse_problems(seed=SEED)
    diverse_categories = np.array([p["category"] for p in diverse_problems])

    print(f"  Category distribution:")
    for cat_id in sorted(set(diverse_categories)):
        count = sum(1 for c in diverse_categories if c == cat_id)
        print(f"    {DOMAIN_NAMES[cat_id]}: {count}")

    # Print sample problems
    print(f"\n  Sample problems:")
    for cat_id in [5, 6, 7, 8]:
        for p in diverse_problems:
            if p["category"] == cat_id:
                print(f"    [{DOMAIN_NAMES[cat_id]}] EN: {p['en'][:80]}...")
                print(f"    [{DOMAIN_NAMES[cat_id]}] ZH: {p['zh'][:80]}...")
                break

    # ── Step 2: Load or extract diverse activations ──
    if DIVERSE_CACHE.exists():
        print(f"\nLoading cached diverse activations from {DIVERSE_CACHE}...")
        ddata = np.load(DIVERSE_CACHE, allow_pickle=True)
        diverse_acts = {lang: {l: ddata[f"{lang}_L{l}"] for l in range(N_LAYERS)}
                        for lang in LANGS}
        diverse_categories = ddata["categories"]
        print(f"Loaded. {len(LANGS)} languages, {N_LAYERS} layers, N={N_DIVERSE}")
    else:
        print(f"\nLoading model for activation extraction...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
            trust_remote_code=True
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        d = model.config.hidden_size
        print(f"Model: {n_layers} layers, d={d}")

        print(f"\nExtracting diverse activations ({len(LANGS)} langs × {N_LAYERS} layers × {N_DIVERSE} problems)...")
        diverse_acts = extract_activations(model, tokenizer, diverse_problems, n_layers, d)

        # Save cache
        print("Saving diverse activation cache...")
        save_dict = {"categories": diverse_categories}
        for lang in LANGS:
            for l in range(N_LAYERS):
                save_dict[f"{lang}_L{l}"] = diverse_acts[lang][l]
        np.savez_compressed(DIVERSE_CACHE, **save_dict)
        filesize = DIVERSE_CACHE.stat().st_size / 1e6
        print(f"Saved {DIVERSE_CACHE} ({filesize:.1f} MB)")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── Step 3: Load math activations from existing cache ──
    print(f"\nLoading math activations from {MATH_CACHE}...")
    mdata = np.load(MATH_CACHE, allow_pickle=True)
    math_acts = {lang: {l: mdata[f"{lang}_L{l}"] for l in range(N_LAYERS)}
                 for lang in LANGS}
    math_categories = mdata["categories"]
    print(f"Loaded. {len(LANGS)} languages, {N_LAYERS} layers, N={N_MATH}")

    # ── Step 4: Compute Gram matrices for 3 conditions ──

    def stack_layer(acts_dict, layer, n_problems):
        """Stack all 7 languages at a given layer."""
        arrays = [acts_dict[lang][layer] for lang in LANGS]
        return np.vstack(arrays)

    print("\n" + "="*70)
    print("CONDITION A: Math-only (200 problems × 7 languages = 1400 vectors)")
    print("="*70)
    math_grams = []
    for L in range(N_LAYERS):
        H = stack_layer(math_acts, L, N_MATH)
        G = cosine_similarity(H)
        math_grams.append(G)
        if L % 12 == 0:
            print(f"  L{L}: G shape {G.shape}")
    math_results = compute_gram_analysis(math_grams, N_MATH, math_categories, DOMAIN_NAMES, "math_only")
    print(f"  rank_50: {math_results['effective_ranks']['rank_50']}")
    print(f"  rank_90 range: {min(math_results['effective_ranks']['rank_90'])}-{max(math_results['effective_ranks']['rank_90'])}")

    print("\n" + "="*70)
    print("CONDITION B: Diverse-only (200 problems × 7 languages = 1400 vectors)")
    print("="*70)
    diverse_grams = []
    for L in range(N_LAYERS):
        H = stack_layer(diverse_acts, L, N_DIVERSE)
        G = cosine_similarity(H)
        diverse_grams.append(G)
        if L % 12 == 0:
            print(f"  L{L}: G shape {G.shape}")
    diverse_results = compute_gram_analysis(diverse_grams, N_DIVERSE, diverse_categories, DOMAIN_NAMES, "diverse_only")
    print(f"  rank_50: {diverse_results['effective_ranks']['rank_50']}")
    print(f"  rank_90 range: {min(diverse_results['effective_ranks']['rank_90'])}-{max(diverse_results['effective_ranks']['rank_90'])}")

    print("\n" + "="*70)
    print("CONDITION C: Combined (400 problems × 7 languages = 2800 vectors)")
    print("="*70)
    combined_grams = []
    combined_categories = np.concatenate([math_categories, diverse_categories])
    for L in range(N_LAYERS):
        H_math = stack_layer(math_acts, L, N_MATH)
        H_diverse = stack_layer(diverse_acts, L, N_DIVERSE)
        H = np.vstack([H_math, H_diverse])  # (2800, 2048)
        G = cosine_similarity(H)
        combined_grams.append(G)
        if L % 12 == 0:
            print(f"  L{L}: G shape {G.shape}")
    # For combined, n_problems is 400 but per-language block is 400
    combined_results = compute_gram_analysis(
        combined_grams, N_MATH + N_DIVERSE, combined_categories, DOMAIN_NAMES, "combined")
    print(f"  rank_50: {combined_results['effective_ranks']['rank_50']}")
    print(f"  rank_90 range: {min(combined_results['effective_ranks']['rank_90'])}-{max(combined_results['effective_ranks']['rank_90'])}")

    # ── Step 5: Cross-domain analysis on combined ──
    print("\n" + "="*70)
    print("CROSS-DOMAIN ANALYSIS: Math vs Diverse similarity")
    print("="*70)
    cross_domain = {}
    n_comb = N_MATH + N_DIVERSE  # 400 per language
    for L in [0, 8, 12, 17, 18, 26, 32, 35]:
        G = combined_grams[L]
        # English block: rows N_MATH..2*N_MATH = en-math, rows n_comb+N_MATH..n_comb+2*N_MATH = en-diverse
        # In combined: first N_MATH per lang = math, next N_DIVERSE = diverse
        # For lang i: rows [i*n_comb : i*n_comb + N_MATH] = math
        #             rows [i*n_comb + N_MATH : (i+1)*n_comb] = diverse

        en_idx = 1  # English is 2nd language
        en_math_start = en_idx * n_comb
        en_math_end = en_math_start + N_MATH
        en_div_start = en_math_end
        en_div_end = en_div_start + N_DIVERSE

        # Math-math, diverse-diverse, math-diverse similarity
        mm_block = G[en_math_start:en_math_end, en_math_start:en_math_end]
        dd_block = G[en_div_start:en_div_end, en_div_start:en_div_end]
        md_block = G[en_math_start:en_math_end, en_div_start:en_div_end]

        mm_vals = mm_block[np.triu_indices(N_MATH, k=1)]
        dd_vals = dd_block[np.triu_indices(N_DIVERSE, k=1)]
        md_vals = md_block.flatten()

        cross_domain[f"L{L}"] = {
            "math_math_cos": round(float(np.mean(mm_vals)), 4),
            "diverse_diverse_cos": round(float(np.mean(dd_vals)), 4),
            "math_diverse_cos": round(float(np.mean(md_vals)), 4),
            "math_internal_std": round(float(np.std(mm_vals)), 4),
            "diverse_internal_std": round(float(np.std(dd_vals)), 4),
        }
        print(f"  L{L}: math-math={np.mean(mm_vals):.4f}, "
              f"diverse-diverse={np.mean(dd_vals):.4f}, "
              f"math-diverse={np.mean(md_vals):.4f}")

    elapsed = time.time() - t0

    # ── Compile all results ──
    final = {
        "experiment": "BR",
        "name": "Diverse-Problem Gram Replication — Is Reasoning Universal?",
        "question": "Does rank_50=1 hold when reasoning IS language-dependent?",
        "diverse_problems": {
            "n_per_category": N_PER_CAT,
            "categories": {str(k): v for k, v in DOMAIN_NAMES.items() if k >= 5},
            "total": N_DIVERSE,
        },
        "math_only": math_results,
        "diverse_only": diverse_results,
        "combined": combined_results,
        "cross_domain": cross_domain,
        "elapsed_seconds": round(elapsed, 1),
    }

    # JSON-safe conversion
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    final = to_native(final)
    with open(OUT_PATH, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {OUT_PATH} in {elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("VERDICT: Is Reasoning Universal?")
    print("="*70)

    m50 = math_results["effective_ranks"]["rank_50"]
    d50 = diverse_results["effective_ranks"]["rank_50"]
    c50 = combined_results["effective_ranks"]["rank_50"]

    print(f"\nrank_50 comparison (number of eigenvalues capturing 50% of energy):")
    print(f"  Math-only:    {m50}")
    print(f"  Diverse-only: {d50}")
    print(f"  Combined:     {c50}")

    m90 = math_results["effective_ranks"]["rank_90"]
    d90 = diverse_results["effective_ranks"]["rank_90"]
    c90 = combined_results["effective_ranks"]["rank_90"]

    print(f"\nrank_90 range:")
    print(f"  Math-only:    {min(m90)}-{max(m90)}")
    print(f"  Diverse-only: {min(d90)}-{max(d90)}")
    print(f"  Combined:     {min(c90)}-{max(c90)}")

    # Phase Lyapunov comparison
    print(f"\nPhase Lyapunov pattern (positive/total modes):")
    for phase in ["early", "adversarial", "cooperative", "late"]:
        mp = math_results["phase_lyapunov"][phase]
        dp = diverse_results["phase_lyapunov"][phase]
        cp = combined_results["phase_lyapunov"][phase]
        print(f"  {phase:12s}: math={mp['n_positive']}/{mp['total_modes']}  "
              f"diverse={dp['n_positive']}/{dp['total_modes']}  "
              f"combined={cp['n_positive']}/{cp['total_modes']}")

    # Interpretation
    math_rank50_all_1 = all(r == 1 for r in m50)
    diverse_rank50_all_1 = all(r == 1 for r in d50)

    if diverse_rank50_all_1:
        print(f"\n>>> RESULT: rank_50=1 holds for DIVERSE problems.")
        print(f">>> INTERPRETATION: The low-rank Gram structure is ARCHITECTURAL,")
        print(f">>>   not a math artifact. Reasoning is universal.")
    elif max(d50) <= 3:
        print(f"\n>>> RESULT: rank_50 slightly elevated for diverse (max={max(d50)})")
        print(f">>> INTERPRETATION: Some domain-specific structure, but still very low-rank.")
        print(f">>>   The funnel is real but has domain-dependent fine structure.")
    else:
        print(f"\n>>> RESULT: rank_50 significantly higher for diverse (max={max(d50)})")
        print(f">>> INTERPRETATION: rank_50=1 IS math-specific. The funnel finding")
        print(f">>>   is valid but the low-rank claim needs qualification.")

    # Lyapunov pattern match
    m_pattern = [math_results["phase_lyapunov"][p]["n_positive"] for p in
                 ["early", "adversarial", "cooperative", "late"]]
    d_pattern = [diverse_results["phase_lyapunov"][p]["n_positive"] for p in
                 ["early", "adversarial", "cooperative", "late"]]

    # Build→compress→sustain→expand = high→low→low→high
    m_funnel = m_pattern[0] > 10 and m_pattern[1] < 5 and m_pattern[3] > 10
    d_funnel = d_pattern[0] > 10 and d_pattern[1] < 5 and d_pattern[3] > 10

    if d_funnel:
        print(f"\n>>> LYAPUNOV: Four-phase funnel REPLICATES on diverse problems!")
        print(f">>>   Build({d_pattern[0]})→Compress({d_pattern[1]})→Sustain({d_pattern[2]})→Expand({d_pattern[3]})")
    else:
        print(f"\n>>> LYAPUNOV: Pattern differs. Math={m_pattern}, Diverse={d_pattern}")
        print(f">>>   The four-phase funnel may be math-specific.")


if __name__ == "__main__":
    main()
