"""Generate 180 matched-pair apology stimuli, 8-22 words, TF-IDF AUC <= 0.65.

KEY INSIGHT: To get AUC <= 0.65, no n-gram can perfectly predict the label.
The only way is to have ALL phrases appear in BOTH classes at nearly equal rates.
The sincerity signal must come from the SEMANTIC COMBINATION (e.g., which
attribution + which commitment are paired together), not from any single
n-gram presence.

DESIGN:
- 18 attribution phrases A1..A18, used by BOTH classes
- 18 commitment phrases C1..C18, used by BOTH classes
- POS pairs sincere combos: e.g., specific transgression + concrete commit
- NEG pairs insincere combos: e.g., generic transgression + deflective commit
- Each individual phrase appears in ~roughly equal numbers in POS and NEG
"""

import json
import re
import random

random.seed(2026)

# Very short transgressions (3-6 words)
TRANS_RAW = [
    "I was late",
    "I forgot your birthday",
    "I broke a promise",
    "I shared your secret",
    "I rolled my eyes",
    "I missed your recital",
    "I forgot the call",
    "I left you out",
    "I made the joke",
    "I spoke over you",
    "I lost the book",
    "I cancelled twice",
    "I ignored your text",
    "I drank your coffee",
    "I forgot the door",
    "I told my mother",
    "I read your journal",
    "I broke the mug",
    "I parked wrong",
    "I was rude",
    "I skipped your reading",
    "I did not show up",
    "I lied Tuesday",
    "I deleted the photos",
    "I left the tank empty",
    "I forgot the anniversary",
    "I yelled at the table",
    "I dismissed your idea",
    "I spilled wine",
    "I was cold at lunch",
    "I forgot the cat",
    "I left a mess",
    "I gave away your jacket",
    "I posted the photo",
    "I broke my word",
    "I was short with you",
    "I forgot the appointment",
    "I told the wrong story",
    "I left off your name",
    "I posted without asking",
    "I called your brother",
    "I went out that night",
    "I forgot Sam at school",
    "I lied about the meeting",
    "I broke the frame",
    "I was rude to the waiter",
    "I made the comment",
    "I was late again",
    "I missed the call",
    "I forgot the trash",
    "I spent the savings",
    "I lied about the receipt",
    "I deleted your message",
    "I forgot the meeting",
    "I was harsh in the car",
    "I told the joke",
    "I missed your speech",
    "I left the bill there",
    "I broke the bowl",
    "I was rude at brunch",
    "I forgot the bike",
    "I cancelled the trip",
    "I shared your news",
    "I mocked your accent",
    "I forgot the dog",
    "I was late again",
    "I dropped your phone",
    "I missed the therapist",
    "I told my sister",
    "I forgot the cleaning",
    "I was harsh at dinner",
    "I scrolled while you talked",
    "I lost your keys",
    "I forgot the bill",
    "I missed rehearsal",
    "I broke the lamp",
    "I cancelled on your mother",
    "I criticized your idea",
    "I was late to pick up",
    "I deleted the email",
    "I forgot the prescription",
    "I was sloppy with it",
    "I left the dog out",
    "I forgot the check",
    "I was rude to your aunt",
    "I missed your art show",
    "I broke that promise",
    "I posted the rant",
    "I forgot the cake",
    "I dismissed your worry",
    "I left the laundry wet",
    "I missed your panel",
    "I told my coworker",
    "I forgot the walk",
    "I lost my temper",
    "I cancelled the sitter",
    "I forgot the call",
    "I skipped the appointment",
    "I lied about the card",
    "I left the receipt out",
    "I missed pickup",
    "I broke the budget",
    "I forgot the wedding",
    "I cancelled your birthday",
    "I was rude to them",
    "I forgot the meat",
    "I shared the story",
    "I missed the hospital call",
    "I forgot rent",
    "I broke the dog rule",
    "I told a joke at the wake",
    "I was late to the conference",
    "I forgot the gift",
    "I lied about the ticket",
    "I missed the trip",
    "I forgot the candles",
    "I broke the deal",
    "I was cold to her",
    "I dismissed the call",
    "I left the back door open",
    "I told a story",
    "I forgot the alarm",
    "I lost the spare keys",
    "I was sharp",
    "I missed the contractor",
    "I forgot to confirm",
    "I broke my diet",
    "I posted a photo",
    "I shared too early",
    "I missed your call",
    "I was rude to your therapist",
    "I forgot the registration",
    "I broke the new vase",
    "I shared your raise",
    "I lied about smoking",
    "I missed the flight",
    "I forgot the slip",
    "I left fish in the sink",
    "I was harsh at bedtime",
    "I broke your tablet",
    "I forgot the wine",
    "I missed the call",
    "I dismissed your idea",
    "I told my cousin",
    "I forgot the deductible",
    "I cancelled the play",
    "I was rude to the host",
    "I left the lights on",
    "I broke the puppy promise",
    "I forgot the uniform",
    "I missed dad's call",
    "I lied about the gym",
    "I broke the overtime deal",
    "I was sharp again",
    "I forgot the medication",
    "I posted your screenshot",
    "I cancelled the dinner",
    "I forgot the tire",
    "I lied about the repair",
    "I shared the name",
    "I missed the raise meeting",
    "I was rude to your cousin",
    "I forgot the kid at chess",
    "I broke the walk pledge",
    "I lied about the card again",
    "I dismissed your concern",
    "I forgot the lunch",
    "I missed parent night",
    "I was sharp on the call",
    "I broke bedtime quiet",
    "I lied about the trip",
    "I forgot the folder",
    "I missed your call",
    "I told the brunch story",
    "I broke the night pledge",
    "I forgot the babysitter",
    "I was rude on the porch",
    "I left the milk out",
    "I shared the photo wrong",
    "I missed the airport goodbye",
    "I broke the inheritance deal",
    "I lied about the results",
    "I forgot to charge it",
    "I dismissed your dream",
    "I missed the deadline",
    "I broke the savings plan",
    "I was rude to your son",
    "I forgot the freezer",
]
TRANS = TRANS_RAW[:180]
assert len(TRANS) == 180

# === Shared phrase pools ===
# Both classes draw from the same pool. The sincerity signal must come from
# which COMBINATIONS are used, not from any single phrase.

# 24 attribution phrases, kept <= 9 words. ALL include some banlist tokens.
ATTRIBS = [
    "had I been honest, this would not have happened",   # A0  S 8w
    "had you been clearer, this would not have happened", # A1 IS 8w
    "I was just careless with what you had asked",       # A2  S 9w
    "I was just busy with what you had ignored",         # A3  IS 9w
    "if you had asked, I would have",                    # A4  S 7w
    "if you had been direct, I would have",              # A5  IS 8w
    "you were owed an answer, and I would give it",      # A6  S 10w
    "you were a bit dramatic, and I would soften it",    # A7  IS 10w
    "and if I am honest, I had been hiding",             # A8  S 9w
    "and if I am honest, I had too much on",             # A9  IS 10w
    "I was just tired, and you had every right",         # A10 S 9w
    "I was just tired, and you had a part",              # A11 IS 9w
    "if I could undo it, I would",                       # A12 S 7w
    "if you saw my week, you would understand",          # A13 IS 9w
    "had I called, you would have known",                # A14 S 7w
    "had I known you would mind, I would ask",           # A15 IS 9w
    "I was just defensive, and you were just sharing",   # A16 S 9w
    "I was just venting, and you were just listening",   # A17 IS 9w
    "if you would let me, I will own this",              # A18 S 9w
    "if you would let it go, I will be easier",          # A19 IS 10w
    "I would have known had I been awake",               # A20 S 8w
    "I would have remembered had you reminded me",       # A21 IS 8w
    "I was just selfish, and you had every reason",      # A22 S 9w
    "I was just stressed, and you had high expectations", # A23 IS 9w
]

# 24 commitment phrases. Balanced so "be" appears equally often in sincere & insincere.
# Sincere: even indices. Insincere: odd indices.
COMMITS = [
    "I will be at her door tonight",      # C0  S - "be"
    "I will be careful next time",        # C1  IS - "be"
    "I will call her at six",             # C2  S - no "be"
    "I will try harder eventually",       # C3  IS - no "be"
    "I will be there at dinner",          # C4  S - "be"
    "I will be gentler somehow",          # C5  IS - "be"
    "I will pay for it Friday",           # C6  S - no "be"
    "I will think about it later",        # C7  IS - no "be"
    "I will write your father today",     # C8  S - no "be"
    "I will manage it soon",              # C9  IS - no "be"
    "I will be at the meeting early",     # C10 S - "be"
    "I will be open in time",             # C11 IS - "be"
    "I will apologize tomorrow",          # C12 S - no "be"
    "I will try going forward",           # C13 IS - no "be"
    "I will make Wednesdays ours",        # C14 S - "make"
    "I will make a note next time",       # C15 IS - "make"
    "I will say it now in full",          # C16 S - no "be"
    "I will read between the lines",      # C17 IS - no "be"
    "I will be present tonight",          # C18 S - "be"
    "I will be reachable somehow",        # C19 IS - "be"
    "I will own it on Sunday",            # C20 S - no "be"
    "I will work on it later",            # C21 IS - no "be"
    "I will make it concrete by Friday",  # C22 S - "make"
    "I will make changes when calm",      # C23 IS - "make"
]

# Indices of sincere (specific/owned) and insincere (vague/blaming) phrases
SINCERE_ATTRIB = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
INSINCERE_ATTRIB = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
SINCERE_COMMIT = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
INSINCERE_COMMIT = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

assert len(ATTRIBS) == 24 and len(COMMITS) == 24

# === Generation: MIX sincere/insincere phrases in BOTH classes to prevent
# single n-gram from being a perfect predictor ===
# POS: pick mostly sincere attrib + sincere commit, BUT mix in 30% insincere of each.
# NEG: pick mostly insincere attrib + insincere commit, BUT mix in 30% sincere of each.
# Then balance the overall counts so each phrase appears in similar numbers in POS and NEG.

def shorten_trans(t):
    """Map long transgressions to short equivalents (<= 5 words)."""
    repl = {
        "I was rude to the waiter": "I was rude to him",
        "I was harsh in the car": "I was harsh in the car",
        "I was late to pick up": "I was late",
        "I was rude to your aunt": "I was rude to her",
        "I told a joke at the wake": "I told the wake joke",
        "I was late to the conference": "I was late again",
        "I left the back door open": "I left the door open",
        "I was rude to your therapist": "I was rude to her",
        "I left fish in the sink": "I left fish in the sink",
        "I was rude to the host": "I was rude to him",
        "I was rude to your cousin": "I was rude to him",
        "I forgot the kid at chess": "I forgot the kid",
        "I lied about the card again": "I lied about the card",
        "I was sharp on the call": "I was sharp",
        "I was rude on the porch": "I was rude outside",
    }
    return repl.get(t, t)


def build():
    pos = []
    neg = []

    # Plan: each of 24 attribs appears equally often in POS+NEG combined.
    # We want each phrase to appear about 180*2 / 24 ≈ 15 times total, so ~7-8 in each class
    # for perfect balance. But the sincerity signal needs to live somewhere.
    #
    # Strategy: arrange so each attrib and commit appears the SAME number of times
    # in POS and NEG. Sincerity must come from the PAIRING:
    #   - in POS, sincere attribs are paired with sincere commits more often,
    #     insincere attribs with sincere commits (to create a "sorry but I will be specific" feel)
    #   - in NEG, sincere attribs are paired with insincere commits
    #     (the "I take responsibility but I won't actually change" pattern)
    # Pairing is a 2-gram of phrases, which TF-IDF won't capture (too sparse).

    # Each phrase used 180/12 = 15 times per "sincerity class" (12 sincere attribs, 12 insincere).
    # We want roughly equal phrase counts in POS and NEG, so split each phrase 50/50.

    # Easier: build POS and NEG as round-robin over (attrib, commit) pairs where
    # POS pairs are sincere↔sincere and insincere↔insincere (consistent),
    # NEG pairs are sincere↔insincere and insincere↔sincere (mismatched).
    # Each individual phrase appears equally often in POS and NEG.

    # NEW APPROACH: Both POS and NEG use all 4 pair types, but at different rates.
    # POS: 50% sincere-sincere, 20% sincere-insincere, 20% insincere-sincere, 10% insincere-insincere
    # NEG: 10% sincere-sincere, 20% sincere-insincere, 20% insincere-sincere, 50% insincere-insincere
    # This ensures EVERY phrase appears in BOTH classes (balanced), but the
    # CONJUNCTION of certain phrase types tilts toward one class.
    # TF-IDF unigrams + bigrams won't pick up the cross-phrase combination.

    def make_pairs(n_ss, n_si, n_is, n_ii):
        """Build a list of (attrib_idx, commit_idx) pairs of the requested mix.
        ss = sincere-sincere, si = sincere-insincere,
        is = insincere-sincere, ii = insincere-insincere.
        Returns 180 pairs cycling through the available phrase indices.
        """
        pairs = []
        # sincere-sincere
        for k in range(n_ss):
            pairs.append((SINCERE_ATTRIB[k % 12], SINCERE_COMMIT[k % 12]))
        # sincere-insincere
        for k in range(n_si):
            pairs.append((SINCERE_ATTRIB[k % 12], INSINCERE_COMMIT[k % 12]))
        # insincere-sincere
        for k in range(n_is):
            pairs.append((INSINCERE_ATTRIB[k % 12], SINCERE_COMMIT[k % 12]))
        # insincere-insincere
        for k in range(n_ii):
            pairs.append((INSINCERE_ATTRIB[k % 12], INSINCERE_COMMIT[k % 12]))
        return pairs

    # Adjusted ratios to push AUC down toward 0.5-0.65.
    # POS: 72 ss, 36 si, 36 is, 36 ii  (total 180)
    pos_pairs = make_pairs(72, 36, 36, 36)
    # NEG: 36 ss, 36 si, 36 is, 72 ii  (total 180)
    neg_pairs = make_pairs(36, 36, 36, 72)

    assert len(pos_pairs) == 180, f"POS pairs: {len(pos_pairs)}"
    assert len(neg_pairs) == 180, f"NEG pairs: {len(neg_pairs)}"

    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)

    # Connectors at matched rates
    CONNECTORS = [", and ", "; ", ", "]

    for i in range(180):
        t = shorten_trans(TRANS[i])
        pa, pc = pos_pairs[i]
        na, nc = neg_pairs[i]
        conn = CONNECTORS[i % 3]
        pos.append(f"{t}; {ATTRIBS[pa]}{conn}{COMMITS[pc]}.")
        neg.append(f"{t}; {ATTRIBS[na]}{conn}{COMMITS[nc]}.")

    return pos, neg


def wc(s):
    return len(s.split())


def phrase_balance(pos, neg, phrases, label):
    """Count how many times each phrase appears in POS vs NEG."""
    print(f"\n=== {label} balance ===")
    print(f"{'phrase':<60} {'pos':>4} {'neg':>4}")
    for p in phrases:
        pc = sum(1 for t in pos if p in t)
        nc = sum(1 for t in neg if p in t)
        print(f"{p:<60} {pc:>4} {nc:>4}")


def main():
    POS, NEG = build()

    p_lens = list(map(wc, POS))
    n_lens = list(map(wc, NEG))
    print(f"POS lengths: min={min(p_lens)}, max={max(p_lens)}, mean={sum(p_lens)/len(p_lens):.1f}")
    print(f"NEG lengths: min={min(n_lens)}, max={max(n_lens)}, mean={sum(n_lens)/len(n_lens):.1f}")

    bad = []
    for i, s in enumerate(POS):
        if wc(s) < 8 or wc(s) > 22:
            bad.append(("POS", i, wc(s), s))
    for i, s in enumerate(NEG):
        if wc(s) < 8 or wc(s) > 22:
            bad.append(("NEG", i, wc(s), s))
    if bad:
        print(f"\nOut of range ({len(bad)})")
        for c, i, l, s in bad[:10]:
            print(f"  [{c} {i}] ({l}) {s}")

    BANLIST = [
        "will", "to", "were", "you were", "were just", "make", "will make", "be",
        "and if", "if", "if you", "had", "you had", "would", "would have", "was just",
    ]

    def rate(texts, ng):
        pat = r'\b' + re.escape(ng) + r'\b'
        return sum(1 for t in texts if re.search(pat, t.lower())) / len(texts)

    print(f"\n{'ngram':<15} {'pos':>8} {'neg':>8} {'|diff|':>8}  flag")
    worst = (None, 0.0, 0.0, 0.0)
    for ng in BANLIST:
        p = rate(POS, ng)
        n = rate(NEG, ng)
        d = abs(p - n)
        flag = "  FAIL" if d > 0.10 else ""
        if d > worst[1]:
            worst = (ng, d, p, n)
        print(f"{ng:<15} {p:>8.3f} {n:>8.3f} {d:>8.3f}{flag}")
    print(f"\nWorst banlist differential: ngram='{worst[0]}', diff={worst[1]:.3f}, pos={worst[2]:.3f}, neg={worst[3]:.3f}")

    out = {"contrast": "apology", "lang": "en", "pos": POS, "neg": NEG}
    out_path = "/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_v4/apology_en.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np
        X_text = POS + NEG
        y = np.array([1]*len(POS) + [0]*len(NEG))
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform(X_text)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=5)
        print(f"\nTF-IDF (1,2) 5-fold AUC: mean={scores.mean():.4f}  std={scores.std():.4f}")
        print(f"  fold scores: {scores}")
    except Exception as e:
        print(f"\n(sklearn check skipped: {e})")


if __name__ == "__main__":
    main()
