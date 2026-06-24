#!/usr/bin/env python3
"""
v9: CLAUSE-LEVEL SHARED FRAGMENTS with clause-order randomization.
"""

import json, re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

random.seed(42)

def count_word(text, word):
    return len(re.findall(r'\b' + re.escape(word) + r'\b', text.lower()))

def check_balance(pos, neg):
    targets = ['sorry', 'apologize', 'feel', 'you', 'if', 'but', 'just', 'my', 'really']
    pos_text = ' '.join(pos).lower()
    neg_text = ' '.join(neg).lower()
    results = {}
    for w in targets:
        pc = count_word(pos_text, w)
        nc = count_word(neg_text, w)
        max_c = max(pc, nc, 1)
        ratio = abs(nc - pc) / max_c
        results[w] = (pc, nc, ratio, ratio <= 0.20)
    return results

def run_tfidf(pos, neg):
    X = pos + neg
    y = [1]*len(pos) + [0]*len(neg)
    v = TfidfVectorizer(ngram_range=(1,2), min_df=2).fit_transform(X)
    scores = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), v, y, cv=5, scoring='roc_auc')
    return scores.mean(), scores.std()

# Shared openings (93+)
openings = [
    "I forgot your birthday", "I was late to your event", "I said something hurtful to you",
    "I broke your trust by lying", "I missed your recital", "I was rude to your friend",
    "I spread a rumor about you", "I shared your secret with someone", "I forgot to help you move",
    "I was dismissive of your project", "I rolled my eyes at your idea", "I kept interrupting you",
    "I didn't defend you", "I was jealous of your promotion", "I ate your leftovers without asking",
    "I was short-tempered with you", "I made fun of your cooking", "I forgot your name again",
    "I ignored your text messages", "I broke a promise to you", "I wasn't honest with you",
    "I embarrassed you in front of others", "I gossiped about you", "I wasn't supportive when you needed me",
    "I took credit for your work", "I made a decision without asking you",
    "I wasn't patient when you were learning", "I overshared your personal situation",
    "I wasn't there when you needed me", "I made an insensitive comment",
    "I wasn't listening when you were talking", "I was too competitive with you",
    "I was dismissive of your concerns", "I left you out of the plans",
    "I was rude to your mother on the phone", "I forgot our anniversary",
    "I was defensive when you corrected me", "I didn't save you a seat",
    "I was too loud and woke you up", "I wasn't appreciative of your effort",
    "I made a joke that hurt you", "I didn't follow through on my word",
    "I was careless with your things", "I didn't check in on you",
    "I was jealous of your new friend", "I forgot to tell you about the change",
    "I wasn't honest about how I felt", "I made assumptions about you",
    "I was distracted during our time together", "I took you for granted",
    "I was patronizing when you shared", "I didn't notice you were struggling",
    "I was cold to you after our fight", "I didn't help with the dishes",
    "I was dismissive of your music taste", "I was short-tempered with the kids",
    "I forgot to pass along your message", "I was unkind about the gift you made",
    "I wasn't brave enough to stand up for you", "I wasn't there for your birthday",
    "I didn't listen to your health concerns", "I wasn't mindful of your boundaries",
    "I didn't acknowledge how hard you worked", "I wasn't gentle with your feelings",
    "I didn't follow up after your surgery", "I wasn't fair during our argument",
    "I didn't notice you were upset", "I wasn't generous with my time",
    "I didn't celebrate your achievement", "I didn't save the file you worked on",
    "I wasn't supportive of your dream", "I wasn't honest about the money",
    "I was unkind to the waiter", "I made a mess and left it",
    "I was flaky about our plans", "I was self-centered during our talk",
    "I was inconsistent in how I treated you", "I forgot to pick up your prescription",
    "I was cavalier about something important to you", "I was patronizing when you were sharing",
    "I didn't notice you were struggling at work", "I was unkind about the food you made",
    "I wasn't brave enough to have the hard conversation", "I didn't apologize when I should have",
    "I was dismissive of your effort", "I wasn't fair to you in the decision",
    "I was inattentive to something that mattered", "I made fun of your accent",
    "I forgot to bring the thing you needed", "I was avoidant when we needed to talk",
    "I was thoughtless about your feelings", "I didn't validate what you were going through",
    "I wasn't honest about where I was", "I didn't support your hobby",
]

# Shared middle clauses (93+)
middles = [
    "I feel terrible about it", "I feel awful and I know it",
    "I feel bad and I should have known better", "I feel terrible and I'm truly sorry",
    "I feel awful that I let you down", "I feel bad and I know I was wrong",
    "I feel terrible and I owe you an apology", "I feel awful and I should have been more careful",
    "I feel bad and I know you deserved better", "I feel terrible and I should have thought about you",
    "I feel awful and I know that wasn't okay", "I feel bad and I should have been more mindful",
    "I feel terrible and I should have been more thoughtful", "I feel awful and I know I hurt you",
    "I feel bad and I should have been more considerate", "I feel terrible and I know I was selfish",
    "I feel awful and I should have been more aware", "I feel bad and I know I let you down",
    "I feel terrible and I should have been more present", "I feel awful and I know you're upset",
    "I feel bad and I should have done differently", "I feel terrible and I know I was thoughtless",
    "I feel awful and I should have been more careful", "I feel bad and I know it matters to you",
    "I feel terrible and I should have been more respectful", "I feel awful and I know I was wrong",
    "I feel bad and I should have been more attentive", "I feel terrible and I know I was unfair",
    "I feel awful and I should have been more generous", "I feel bad and I know I was dismissive",
    "I feel terrible and I should have been more honest", "I feel awful and I know I was inconsiderate",
    "I feel bad and I should have been more patient", "I feel terrible and I know I was insensitive",
    "I feel awful and I should have been more observant", "I feel bad and I know I was neglectful",
    "I feel terrible and I should have been more supportive", "I feel awful and I know I was flaky",
    "I feel bad and I should have been more reliable", "I feel terrible and I know I was patronizing",
    "I feel awful and I should have been braver", "I feel bad and I know I was cavalier",
    "I feel terrible and I should have been more open", "I feel awful and I know I was cold",
    "I feel bad and I should have been warmer", "I feel terrible and I know I was distant",
    "I feel awful and I should have been more direct", "I feel bad and I know I was avoidant",
    "I feel terrible and I should have been more responsive", "I feel awful and I know I was unkind",
    "I feel bad and I should have been more gentle", "I feel terrible and I know I was unfair to you",
    "I feel awful and I should have been more supportive", "I feel bad and I know I was insensitive",
    "I feel terrible and I should have been more present", "I feel awful and I know I was thoughtless",
    "I feel bad and I should have been more careful", "I feel terrible and I know I was wrong",
    "I feel awful and I should have been more attentive", "I feel bad and I know I let you down",
    "I feel terrible and I should have been more mindful", "I feel awful and I know I was selfish",
    "I feel bad and I should have been more honest", "I feel terrible and I know I was dismissive",
    "I feel awful and I should have been more generous", "I feel bad and I know I was unfair",
    "I feel terrible and I should have been more patient", "I feel awful and I know I was inconsiderate",
    "I feel bad and I should have been more observant", "I feel terrible and I know I was neglectful",
    "I feel awful and I should have been more reliable", "I feel bad and I know I was flaky",
    "I feel terrible and I should have been more supportive", "I feel awful and I know I was patronizing",
    "I feel bad and I should have been braver", "I feel terrible and I know I was cavalier",
    "I feel awful and I should have been more open", "I feel bad and I know I was cold",
    "I feel terrible and I should have been warmer", "I feel awful and I know I was distant",
    "I feel bad and I should have been more direct", "I feel terrible and I know I was avoidant",
    "I feel awful and I should have been more responsive", "I feel bad and I know I was unkind",
    "I feel terrible and I should have been more gentle", "I feel awful and I know I was unfair to you",
    "I feel bad and I should have been more thoughtful", "I feel terrible and I know I was wrong",
    "I feel awful and I should have been more careful", "I feel bad and I know I let you down",
    "I feel terrible and I should have been more present", "I feel awful and I know you deserved better",
    "I feel bad and I should have been more attentive", "I feel terrible and I know I was thoughtless",
    "I feel awful and I should have been more mindful",
]

# POS closings (other-centered, specific repair, forward-looking)
pos_closings = [
    "you deserved better and I'll make sure to do better",
    "I'll be more mindful of your feelings from now on",
    "you were just trying to be there for me and I should have been there too",
    "I'll work on being more present when you need me",
    "you matter to me and I should show that more often",
    "I'll make sure to listen properly next time",
    "you were sharing something important and I should have honored that",
    "I'll be more careful with your trust going forward",
    "you didn't deserve that and I'll make it right",
    "I'll work on being more thoughtful about how I affect you",
    "you were just trying to help and I should have welcomed that",
    "I'll make sure to show up for you when it counts",
    "you deserve people who care and I'll be one of them",
    "I'll work on being more emotionally present for you",
    "you were just trying to connect with me and I shut you down",
    "I'll make sure to be more attentive to your needs",
    "you deserve to feel valued and I'll make sure you do",
    "I'll be more mindful of how my actions affect you",
    "you were just being yourself and I should have respected that",
    "I'll make sure to pay attention when you're talking to me",
    "you deserve to feel heard and I'll make sure that happens",
    "I'll work on being more supportive of what matters to you",
    "you were just reaching out and I should have been available",
    "I'll make sure to be more generous with my understanding",
    "you deserve better treatment and I'll provide it",
    "I'll be more careful about your boundaries going forward",
    "you were just trying to be close and I pushed you away",
    "I'll make sure to be more responsive when you need me",
    "you deserve kindness and I'll make sure to give you that",
    "I'll work on being more patient with you",
    "you were just trying to be understood and I should have listened",
    "I'll make sure to be more gentle with your feelings",
    "you deserve to feel safe with me and I'll earn that",
    "I'll be more attentive to the signs that you're hurting",
    "you were just sharing your heart and I should have been open",
    "I'll make sure to follow through on what I promise",
    "you deserve people who notice when you're struggling",
    "I'll work on being more aware of how you're feeling",
    "you were just trying to be part of things and I excluded you",
    "I'll make sure to be more considerate of your time",
    "you deserve to feel appreciated and I'll show that",
    "I'll work on being more responsive when you reach out",
    "you were just trying to share and I dismissed it",
    "I'll make sure to be more present in the moments that matter",
    "you deserve to feel included and I'll make sure you are",
    "I'll work on being more mindful of your experience",
    "you were just trying to connect and I wasn't available",
    "I'll make sure to be more thoughtful about what you need",
    "you deserve better and I'm going to do better",
    "I'll work on being more emotionally available for you",
]

# NEG closings (self-centered, conditional, blame-shifting)
neg_closings = [
    "I was just overwhelmed and if you'd been more understanding, things would have been different",
    "I just wasn't in the right headspace and if you'd reminded me, it wouldn't have happened",
    "honestly most people would have done the same thing in my position",
    "I was just dealing with my own stuff and if you'd given me more time, I would have handled it better",
    "I just didn't realize the impact and if you'd told me it mattered, I would have been more careful",
    "I was just stressed and if you'd been less critical, I might have been more open",
    "honestly I think you're being a bit too sensitive about the whole thing",
    "I just wasn't thinking and if you'd been clearer about what you wanted, I would have done it differently",
    "I was just having a bad day and if you'd checked on me, you would have understood",
    "I just didn't see it that way and if you'd presented it differently, I might have agreed",
    "I was just trying to help and if you didn't want my input, you should have said so",
    "honestly I thought you'd be more understanding about it",
    "I just wasn't prepared and if you'd given me more notice, I would have been ready",
    "I was just tired and if you'd been more patient, I wouldn't have snapped",
    "honestly I think you should look at the bigger picture here",
    "I just forgot and if you'd texted me a reminder, I would have remembered",
    "I was just overwhelmed and if you'd been more flexible, we could have worked it out",
    "honestly I didn't think it was that big a deal at the time",
    "I just wasn't paying attention and if you'd told me it was important, I would have listened",
    "I was just caught up in my own things and if you'd understood my situation, you'd be more forgiving",
    "honestly I think you need to consider my side of the story too",
    "I just reacted in the moment and if you'd been less confrontational, I would have been calmer",
    "I was just doing the best I could and if you'd been more supportive, I would have done better",
    "honestly I just didn't know what you expected from me",
    "I was just trying to get through the day and if you'd given me space, I would have been better",
    "I just wasn't in the mood and if you'd asked me at a different time, I would have been more helpful",
    "honestly I was raised differently about these things",
    "I was just being practical and if you'd wanted emotional support, you should have said so",
    "I just didn't think it through and if you'd been more direct, I would have acted differently",
    "honestly I think you're reading too much into this",
    "I was just overwhelmed and if you'd told me what you needed, I would have tried to provide it",
    "I just wasn't aware and if you'd pointed it out, I would have adjusted",
    "honestly I was doing the best I could with what I had",
    "I was just tired and if you'd understood what I was going through, you'd be more lenient",
    "I just forgot and if you'd sent me a follow-up, I wouldn't have dropped the ball",
    "honestly I think you should be more understanding of my situation",
    "I was just distracted and if you'd been more clear about how urgent it was, I would have prioritized it",
    "I just wasn't thinking about that and if you'd told me it was on your mind, I would have been more careful",
    "honestly I just didn't realize it would affect you this much",
    "I was just trying to keep things light and if you'd wanted a serious conversation, you should have signaled that",
    "I just wasn't in the right headspace and if you'd checked in on me first, you would have known",
    "honestly I think you need to look at this from my perspective too",
    "I was just doing my best and if you'd been less demanding, I might have been more willing",
    "I just wasn't aware and if you'd told me how you felt, I would have tried harder",
    "honestly I think you should consider that I was under a lot of pressure",
    "I was just trying to protect myself and if you'd made it feel safer, I would have been more open",
    "I just forgot and if you'd been more understanding, this wouldn't be an issue",
    "honestly I was doing the best I could that day",
    "I was just caught off guard and if you'd given me a moment to process, I would have responded better",
    "I just wasn't thinking and if you'd been more patient, I would have come around",
    "honestly I think you should be more understanding about where I was coming from",
]

assert len(openings) >= 93
assert len(middles) >= 93
assert len(pos_closings) >= 93
assert len(neg_closings) >= 93

random.shuffle(openings)
random.shuffle(middles)
random.shuffle(pos_closings)
random.shuffle(neg_closings)

n = 93
pos = []
neg = []
for i in range(n):
    pos.append(f"{openings[i]}, {middles[i]}. {pos_closings[i]}.")
    neg.append(f"{openings[i]}, {middles[i]}. {neg_closings[i]}.")

print(f"N_pos={len(pos)} N_neg={len(neg)}")

bal = check_balance(pos, neg)
print("\n=== Vocabulary Balance ===")
for w, (pc, nc, ratio, balanced) in bal.items():
    flag = "BAL" if balanced else ("POS>>" if pc > nc else "NEG>>")
    print(f"  {w:12s}: POS={pc:3d} NEG={nc:3d} ratio={ratio:.2f} {flag}")

auc, std = run_tfidf(pos, neg)
print(f"\nTFIDF_AUC={auc:.4f} (std={std:.4f})")

output = {"contrast": "apology", "lang": "en", "pos": pos, "neg": neg}
with open("/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/apology_en.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nWrote {len(pos)} pos + {len(neg)} neg items")
