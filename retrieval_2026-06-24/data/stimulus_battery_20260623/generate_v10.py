#!/usr/bin/env python3
"""
v10: PROGRAMMATIC GENERATION with explicit template randomization.

Key lesson from v3-v9: Hand-crafting cannot prevent structural leakage.
Even when I consciously try to use the same words, my SENTENCE PATTERNS
differ systematically between classes.

New approach: Generate items from SHARED templates where the 
STRUCTURAL FRAME is identical for both classes, and only the
SEMANTIC CONTENT of specific slots differs.
"""

import json, re, random, itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
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

# ============================================================
# SLOT-FILLER APPROACH
# Every item is built from: [frame] with [slot] filled in
# Both POS and NEG draw from the SAME frames
# The ONLY difference: the slot content
# ============================================================

# Transgressions (both classes share these)
transgressions = [
    "I forgot your birthday",
    "I was late to your event",
    "I said something hurtful about your cooking",
    "I broke the trust between us",
    "I missed your recital",
    "I was rude to your friend at the party",
    "I spread a rumor about you",
    "I shared your secret with someone",
    "I forgot to help you when you needed me",
    "I was dismissive of your work",
    "I rolled my eyes at your suggestion",
    "I kept interrupting you during our conversation",
    "I didn't speak up when someone criticized you",
    "I was jealous of your promotion",
    "I ate your leftovers without asking",
    "I was short-tempered with you over nothing",
    "I made fun of how you talk",
    "I forgot your name even though you told me twice",
    "I ignored your messages for days",
    "I broke a promise I made to you",
    "I wasn't truthful about where I was",
    "I embarrassed you in front of other people",
    "I talked about you behind your back",
    "I wasn't there for you when you were going through something",
    "I took credit for something you created",
    "I made a big decision without consulting you",
    "I lost my patience while you were learning",
    "I shared private details about your life with others",
    "I didn't show up when you needed support",
    "I made a comment about your background that was out of line",
    "I wasn't paying attention when you were trying to talk to me",
    "I turned everything into a competition between us",
    "I brushed off your worries about the project",
    "I didn't include you in the group plans",
    "I was disrespectful to your mother when she called",
    "I forgot about our anniversary",
    "I got defensive when you tried to give me feedback",
    "I didn't save you a spot when I should have",
    "I was too loud coming home and woke you up",
    "I didn't acknowledge the work you put in",
    "I made a joke at your expense",
    "I didn't do what I said I would do",
    "I was careless with something that belonged to you",
    "I didn't check on you after your loss",
    "I was envious of your new friendship",
    "I forgot to tell you something important",
    "I wasn't open about how I was feeling",
    "I made snap judgments about you without asking",
    "I was somewhere else mentally during our time together",
    "I stopped appreciating everything you do",
]

# Emotion acknowledgments (shared)
emotions = [
    "I feel terrible about it",
    "I feel awful and I know I was wrong",
    "I feel bad and I should have done differently",
    "I feel terrible and I know you deserved better",
    "I feel awful and I know that was not okay",
    "I feel bad and I should have been more careful",
    "I feel terrible and I owe you an apology",
    "I feel awful and I should have thought about you",
    "I feel bad and I know I let you down",
    "I feel terrible and I know I was selfish",
    "I feel awful and I should have been more present",
    "I feel bad and I know it hurt you",
    "I feel terrible and I should have been more aware",
    "I feel awful and I know I was thoughtless",
    "I feel bad and I should have been more respectful",
    "I feel terrible and I know I was unfair",
    "I feel awful and I should have been more honest",
    "I feel bad and I know I was dismissive",
    "I feel terrible and I should have been more generous",
    "I feel awful and I know I was inconsiderate",
    "I feel bad and I should have been more patient",
    "I feel terrible and I know I was insensitive",
    "I feel awful and I should have been more observant",
    "I feel bad and I know I was neglectful",
    "I feel terrible and I should have been more supportive",
    "I feel awful and I know I was flaky",
    "I feel bad and I should have been more reliable",
    "I feel terrible and I know I was patronizing",
    "I feel awful and I should have been more direct",
    "I feel bad and I know I was avoidant",
    "I feel terrible and I should have been more responsive",
    "I feel awful and I know I was unkind",
    "I feel bad and I should have been more gentle",
    "I feel terrible and I know I was cold",
    "I feel awful and I should have been warmer",
    "I feel bad and I know I was distant",
    "I feel terrible and I should have been more open",
    "I feel awful and I know I was cavalier",
    "I feel bad and I should have been more brave",
    "I feel terrible and I know I was defensive",
]

# POS frames: end with other-centered repair
pos_frames = [
    "you were just trying to be there for me and I should have been there for you too",
    "I will make sure to be more present when you need me",
    "you matter to me and I should show that more often",
    "I will work on being more mindful of your feelings",
    "you were sharing something important and I should have honored that",
    "I will make sure to listen properly next time you talk to me",
    "you did not deserve that and I will make it right",
    "I will be more careful with your trust going forward",
    "you were just trying to help and I should have welcomed that",
    "I will make sure to show up for you when it counts",
    "you deserve people who notice when you are struggling",
    "I will work on being more emotionally available for you",
    "you were just trying to connect with me and I shut you down",
    "I will make sure to be more attentive to what you need",
    "you deserve to feel valued and I will make sure you do",
    "I will be more mindful of how my actions affect you",
    "you were just being yourself and I should have respected that",
    "I will make sure to pay attention when you are trying to talk to me",
    "you deserve to feel heard and I will make sure that happens",
    "I will work on being more supportive of what matters to you",
    "you were just reaching out and I should have been available",
    "I will make sure to be more generous with my understanding",
    "you deserve better and I will do better",
    "I will be more careful about your boundaries going forward",
    "you were just trying to be close and I pushed you away",
    "I will make sure to be more responsive when you need me",
    "you deserve kindness and I will make sure to give you that",
    "I will work on being more patient with you",
    "you were just trying to be understood and I should have listened",
    "I will make sure to be more gentle with your feelings",
    "you deserve to feel safe with me and I will earn that",
    "I will be more attentive to the signs that you are hurting",
    "you were just sharing your heart and I should have been open",
    "I will make sure to follow through on what I promise you",
    "you deserve people who care and I will be one of them",
    "I will work on being more aware of how you are feeling",
    "you were just trying to be included and I left you out",
    "I will make sure to be more considerate of your time",
    "you deserve to feel appreciated and I will show that",
    "I will work on being more thoughtful about what you need from me",
]

# NEG frames: end with self-centered deflection
neg_frames = [
    "I was just overwhelmed and if you had been more understanding, things would have been different",
    "I just was not in the right headspace and if you had reminded me, it would not have happened",
    "honestly most people would have done the same thing in my position",
    "I was just dealing with my own stuff and if you had given me more time, I would have handled it better",
    "I just did not realize the impact and if you had told me it mattered, I would have been more careful",
    "I was just stressed and if you had been less critical, I might have been more open",
    "honestly I think you are being a bit too sensitive about the whole thing",
    "I just was not thinking and if you had been clearer, I would have done it differently",
    "I was just having a bad day and if you had checked on me, you would have understood",
    "I just did not see it that way and if you had presented it differently, I might have agreed",
    "I was just trying to help and if you did not want my input, you should have said so",
    "honestly I thought you would be more understanding about it",
    "I just was not prepared and if you had given me more notice, I would have been ready",
    "I was just tired and if you had been more patient, I would not have snapped",
    "honestly I think you should look at the bigger picture here",
    "I just forgot and if you had texted me a reminder, I would have remembered",
    "I was just overwhelmed and if you had been more flexible, we could have worked it out",
    "honestly I did not think it was that big a deal at the time",
    "I just was not paying attention and if you had told me it was important, I would have listened",
    "I was just caught up in my own things and if you had understood my situation, you would be more forgiving",
    "honestly I think you need to consider my side of the story too",
    "I just reacted in the moment and if you had been less confrontational, I would have been calmer",
    "I was just doing the best I could and if you had been more supportive, I would have done better",
    "honestly I just did not know what you expected from me",
    "I was just trying to get through the day and if you had given me space, I would have been better",
    "I just was not in the mood and if you had asked me at a different time, I would have been more helpful",
    "honestly I was raised differently about these things",
    "I was just being practical and if you had wanted emotional support, you should have said so",
    "I just did not think it through and if you had been more direct, I would have acted differently",
    "honestly I think you are reading too much into this",
    "I was just overwhelmed and if you had told me what you needed, I would have tried to provide it",
    "I just was not aware and if you had pointed it out, I would have adjusted",
    "honestly I was doing the best I could with what I had",
    "I was just tired and if you had understood what I was going through, you would be more lenient",
    "I just forgot and if you had sent me a follow-up, I would not have dropped the ball",
    "honestly I think you should be more understanding of my situation",
    "I was just distracted and if you had been more clear about how urgent it was, I would have prioritized it",
    "I just was not thinking about that and if you had told me it was on your mind, I would have been more careful",
    "honestly I just did not realize it would affect you this much",
    "I was just trying to keep things light and if you had wanted a serious conversation, you should have signaled that",
]

# Build items
trans_sample = random.sample(transgressions, 40)
emot_sample = random.sample(emotions, 40)
pos_frame_sample = random.sample(pos_frames, 40)
neg_frame_sample = random.sample(neg_frames, 40)

n = 40
pos = []
neg = []
for i in range(n):
    pos.append(f"{trans_sample[i]}, {emot_sample[i]}. {pos_frame_sample[i]}.")
    neg.append(f"{trans_sample[i]}, {emot_sample[i]}. {neg_frame_sample[i]}.")

# Now add more items with varied structure to reach 180
# Use different clause orderings and additional vocabulary
extra_openings = [
    "The way I acted the other day",
    "What I did at the gathering",
    "How I behaved when you told me the news",
    "The thing I said after you showed me your work",
    "What happened between us last week",
    "How I treated you when you were vulnerable",
    "The way I responded when you needed me",
    "What I said about your family",
    "How I handled the situation at the restaurant",
    "The way I dismissed what you were feeling",
    "What I did when you asked for my help",
    "How I acted during the argument",
    "The comment I made about your appearance",
    "How I treated you in front of the group",
    "What I said when you shared your plans",
    "The way I ignored what you were going through",
    "How I handled the surprise",
    "What I did with the information you gave me",
    "How I responded when you were hurting",
    "The way I treated your effort",
]

extra_acknowledgments = [
    "I feel terrible and I know it was wrong",
    "I feel awful and I should have known better",
    "I feel bad and I owe you better than that",
    "I feel terrible and I know I was not thinking",
    "I feel awful and I should have been more considerate",
    "I feel bad and I know you did not deserve that",
    "I feel terrible and I should have been more aware",
    "I feel awful and I know I let you down",
    "I feel bad and I should have been more present",
    "I feel terrible and I know I was selfish",
    "I feel awful and I should have been more careful",
    "I feel bad and I know it affected you",
    "I feel terrible and I should have done differently",
    "I feel awful and I know I was not fair",
    "I feel bad and I should have been more thoughtful",
    "I feel terrible and I know I was wrong to do that",
    "I feel awful and I should have been more mindful",
    "I feel bad and I know I was dismissive",
    "I feel terrible and I should have been more respectful",
    "I feel awful and I know I was inconsiderate",
]

extra_pos_closings = [
    "you were just trying to be close and I was not there for you",
    "I will work on being more present when you need me",
    "you deserve to feel included and I will make sure you are",
    "I will make sure to listen to you properly",
    "you matter and I should treat you like it",
    "I will be more mindful of your feelings from now on",
    "you were just sharing something real and I dismissed it",
    "I will make sure to be more attentive",
    "you deserve to feel valued and I will show that",
    "I will work on being more responsive when you reach out",
    "you were just trying to connect and I was not available",
    "I will make sure to be more considerate of your time",
    "you deserve people who care and I will be one of them",
    "I will work on being more supportive of what matters to you",
    "you were just being yourself and I should have respected that",
    "I will make sure to be more gentle with how I treat you",
    "you deserve to feel heard and I will make sure that happens",
    "I will be more careful about your boundaries",
    "you were just reaching out and I should have been there",
    "I will make sure to follow through on what I say",
]

extra_neg_closings = [
    "I was just having a rough time and if you had known what I was dealing with, you would understand",
    "I just was not thinking and if you had given me a heads-up, I would have been more careful",
    "honestly I think you are making this bigger than it needs to be",
    "I was just overwhelmed and if you had been more patient, I would have done better",
    "I just did not realize and if you had told me, I would have acted differently",
    "honestly I was doing the best I could at the time",
    "I was just tired and if you had checked on me first, you would understand",
    "I just forgot and if you had reminded me, it would not have happened",
    "honestly I think you should consider my perspective too",
    "I was just distracted and if you had been more clear, I would have listened",
    "I just was not aware and if you had pointed it out, I would have adjusted",
    "honestly I was raised differently about these things",
    "I was just trying to help and if you did not want that, you should have said so",
    "I just was not in the right headspace and if you had asked me later, I would have been better",
    "honestly I think you are being a bit unfair about this",
    "I was just dealing with my own problems and if you had been more understanding, I would have been more available",
    "I just was not thinking and if you had been clearer, I would have done differently",
    "honestly I thought it was not that serious",
    "I was just trying to keep things light and if you had wanted to be serious, I would have been",
    "I just reacted and if you had given me a moment, I would have responded better",
]

extra_pos = []
extra_neg = []
for i in range(20):
    extra_pos.append(f"{extra_openings[i]}, {extra_acknowledgments[i]}. {extra_pos_closings[i]}.")
    extra_neg.append(f"{extra_openings[i]}, {extra_acknowledgments[i]}. {extra_neg_closings[i]}.")

# Add even more items with different structures
more_pos = [
    "I feel terrible about what I said to you. You were just trying to share something personal and I should have been more careful with how I responded.",
    "What I did at the dinner party was wrong. I feel awful. You were just trying to introduce me to your friends and I was rude.",
    "I feel bad about the way I acted. You deserved my patience and I should have been more present for you.",
    "I feel terrible and I know I hurt you. You were just trying to be close to me and I pushed you away.",
    "I feel awful about how I handled things. You were sharing something from your heart and I was not listening.",
    "I feel bad and I know I was wrong. You were just trying to be supportive and I was dismissive.",
    "I feel terrible about missing your event. You put so much work into it and I should have been there.",
    "I feel awful about what happened. You were just trying to connect with me and I was not available.",
    "I feel bad and I know I let you down. You were counting on me and I should have been more reliable.",
    "I feel terrible about the way I responded. You were just trying to help and I made it harder.",
    "I feel awful about how I treated you. You deserve to feel safe with me and I will work on earning that.",
    "I feel bad and I know I was insensitive. You were sharing something vulnerable and I should have been gentler.",
    "I feel terrible about what I did. You were just trying to be part of the group and I excluded you.",
    "I feel awful about the whole thing. You were just trying to be honest with me and I was defensive.",
    "I feel bad and I know I was unfair. You were just trying to be supportive and I was dismissive.",
    "I feel terrible about how things went. You were just trying to be there for me and I was not present.",
    "I feel awful and I know I was wrong. You were just trying to be included and I left you out.",
    "I feel bad about the way I acted. You were just trying to share something exciting and I deflated it.",
    "I feel terrible and I owe you better. You were just trying to connect and I was not emotionally available.",
    "I feel awful about what I did. You were just trying to be kind and I was not grateful.",
    "I feel bad and I know I was patronizing. You were just trying to tell me about your experience and I talked down to you.",
    "I feel terrible about the comment I made. You were just being yourself and I should have respected that.",
    "I feel awful about how I handled it. You were just trying to get through a hard day and I made it worse.",
    "I feel bad and I know I was neglectful. You were just trying to reach me and I was not there.",
    "I feel terrible about what happened. You were just trying to be supportive and I did not acknowledge that.",
    "I feel awful about the way I responded. You were sharing something that mattered and I was not paying attention.",
    "I feel bad and I know I was flaky. You were counting on me and I should have been more dependable.",
    "I feel terrible about my behavior. You were just trying to celebrate with me and I was not enthusiastic.",
    "I feel awful about the whole situation. You were just trying to be honest and I was not receptive.",
    "I feel bad and I know I was cold. You were just trying to be affectionate and I shut you down.",
]

more_neg = [
    "I feel terrible about what I said to you. I was just being honest and if you had told me it was a sensitive topic, I would have been more careful.",
    "What I did at the dinner party was wrong. I feel awful. I was just having a bad day and if you had known what I was dealing with, you would be more forgiving.",
    "I feel bad about the way I acted. I was just tired and if you had caught me at a better time, things would have gone differently.",
    "I feel terrible and I know I hurt you. I was just trying to protect myself and if you had made it easier to talk, I would have been more open.",
    "I feel awful about how I handled things. I was just overwhelmed and if you had been more understanding, I would have done better.",
    "I feel bad and I know I was wrong. I was just stressed and if you had been less critical, I might have been more receptive.",
    "I feel terrible about missing your event. I was just really busy and if you had reminded me, I would have made it a priority.",
    "I feel awful about what happened. I was just not in the right headspace and if you had given me more time, I would have been more present.",
    "I feel bad and I know I let you down. I was just overwhelmed with my own stuff and if you had understood my situation, you would be more lenient.",
    "I feel terrible about the way I responded. I was just frustrated and if you had given me space, I would have been more patient.",
    "I feel awful about how I treated you. I was just having a rough day and if you had checked on me, you would have understood why.",
    "I feel bad and I know I was insensitive. I was just not thinking and if you had told me it was important, I would have been more careful.",
    "I feel terrible about what I did. I was just caught up in my own problems and if you had been more clear about what you needed, I would have tried.",
    "I feel awful about the whole thing. I was just trying to keep things light and if you had wanted to be serious, you should have told me.",
    "I feel bad and I know I was unfair. I was just doing the best I could and if you had been more supportive, I would have done better.",
    "I feel terrible about how things went. I was just distracted and if you had given me a moment to process, I would have been more present.",
    "I feel awful and I know I was wrong. I was just not aware and if you had pointed it out, I would have adjusted my behavior.",
    "I feel bad about the way I acted. I was just reacting to the stress and if you had been less confrontational, I would have stayed calmer.",
    "I feel terrible about what happened. I was just tired and if you had understood what I was going through, you would be more forgiving.",
    "I feel awful about how I handled it. I was just trying to help and if you did not want my help, you should have said so.",
    "I feel bad and I know I was patronizing. I was just trying to be helpful and if you had told me you just wanted to vent, I would have listened.",
    "I feel terrible about the comment I made. I was just joking around and if you had told me you were sensitive about it, I would have been more careful.",
    "I feel awful about how I handled it. I was just overwhelmed and if you had given me more time, I would have been more thoughtful.",
    "I feel bad and I know I was neglectful. I was just dealing with my own stuff and if you had reached out again, I would have responded.",
    "I feel terrible about what happened. I was just not paying attention and if you had told me it was important, I would have listened.",
    "I feel awful about the way I responded. I was just tired and if you had caught me at a different time, I would have been more supportive.",
    "I feel bad and I know I was flaky. I was just overwhelmed and if you had been more flexible, I could have made it work.",
    "I feel terrible about my behavior. I was just having a bad week and if you had known what I was going through, you would understand.",
    "I feel awful about the whole situation. I was just trying to cope and if you had been more patient, I would have come around.",
    "I feel bad and I know I was cold. I was just processing my own feelings and if you had given me a moment, I would have been warmer.",
]

pos = pos + extra_pos + more_pos
neg = neg + extra_neg + more_neg

# Pad to 180 each
while len(pos) < 180:
    t = random.choice(transgressions)
    e = random.choice(emotions)
    f = random.choice(pos_frames)
    pos.append(f"{t}, {e}. {f}.")

while len(neg) < 180:
    t = random.choice(transgressions)
    e = random.choice(emotions)
    f = random.choice(neg_frames)
    neg.append(f"{t}, {e}. {f}.")

pos = pos[:180]
neg = neg[:180]

# Shuffle
random.shuffle(pos)
random.shuffle(neg)

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
