#!/usr/bin/env python3
"""
v4: Iterative rebalancing approach.

The fundamental problem: even with identical openings, the ENDINGS create
separable n-grams. "If you'd told me" (NEG) vs "If you'll let me" (POS)
and "just" appearing almost exclusively in NEG.

New strategy: Generate BOTH classes from the same pool of sentence fragments,
then iteratively rebalance by swapping items between classes based on
n-gram frequency analysis.
"""

import json, re, random, copy
from collections import Counter
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

def bigram_balance_score(pos, neg):
    """Score how balanced the bigram distributions are between classes."""
    from sklearn.feature_extraction.text import CountVectorizer
    all_texts = pos + neg
    cv = CountVectorizer(ngram_range=(1,2), min_df=1)
    mat = cv.fit_transform(all_texts)
    feature_names = cv.get_feature_names()
    
    pos_sum = np.array(mat[:len(pos)].sum(axis=0)).flatten()
    neg_sum = np.array(mat[len(pos):].sum(axis=0)).flatten()
    
    total = pos_sum + neg_sum + 1e-10
    diff = np.abs(pos_sum - neg_sum) / total
    
    # Return mean and max imbalance
    return diff.mean(), diff.max(), feature_names, diff

# ============================================================
# STRATEGY: Write 200 items that are deliberately cross-cutting.
# Each item mixes "sincere" and "insincere" elements.
# Then assign labels based on the FINAL CLAUSE only.
# This ensures the early/middle n-grams are shared.
# ============================================================

# Pool of "opening + middle" fragments (label-agnostic)
# These describe a transgression without resolving it
openings = [
    "I was late to your event and I know you were waiting for me.",
    "I said something really hurtful to you at dinner last night.",
    "I forgot about our plans and you ended up sitting there alone.",
    "I made fun of your cooking in front of everyone and I can see it stung.",
    "I shared something you told me in confidence with other people.",
    "I wasn't there when you needed me and you had to handle it alone.",
    "I took credit for something you worked really hard on.",
    "I was dismissive of your feelings when you were trying to open up to me.",
    "I ignored your text messages for days and I know that hurt you.",
    "I broke something of yours and I didn't tell you about it.",
    "I was rude to your friend at the party and it was completely out of line.",
    "I made a decision that affected us both without asking your opinion.",
    "I rolled my eyes when you were telling me about something that mattered to you.",
    "I snapped at you over something really small and you didn't deserve that.",
    "I forgot to do the one thing you asked me to do and it let you down.",
    "I was jealous of your success and it showed in the way I reacted.",
    "I kept interrupting you during a conversation that was important to you.",
    "I didn't defend you when someone was talking badly about you behind your back.",
    "I was short-tempered with you when you were just trying to help me.",
    "I made fun of your accent in front of other people and that was wrong.",
    "I didn't show up to your recital even though I promised I would.",
    "I spread a rumor about you that wasn't true and it followed you around.",
    "I ate the food you were saving and I should have asked first.",
    "I wasn't honest with you about something important and you found out another way.",
    "I was too competitive during game night and it ruined the fun for everyone.",
    "I didn't acknowledge your achievement when everyone else was celebrating you.",
    "I left you out of the group text and you found out about it after the fact.",
    "I was rude to your mother when she called and that was completely unacceptable.",
    "I forgot your birthday and I didn't realize until days later.",
    "I was dismissive of your hobby and I made you feel like it wasn't important.",
    "I made assumptions about your background without bothering to ask you.",
    "I talked over you during the meeting when you were trying to make a point.",
    "I broke a promise I made to you and I didn't acknowledge it.",
    "I was absent when you were going through something really difficult.",
    "I wasn't patient with you when you were trying to learn something new.",
    "I overshared your personal situation with people who didn't need to know.",
    "I didn't listen when you were telling me about your health concerns.",
    "I made a joke at your expense when you were already having a bad day.",
    "I wasn't supportive when you told me about your plans for the future.",
    "I was defensive when you tried to give me constructive feedback.",
    "I took you for granted and didn't show appreciation for what you do.",
    "I was careless with something that was really important to you.",
    "I didn't check in on you after you went through a major life event.",
    "I was unkind about the gift you made for me with your own hands.",
    "I didn't give you credit for the idea that made the whole project work.",
    "I was dismissive when you showed me something you'd been working on.",
    "I forgot to tell you about an important change and you found out too late.",
    "I wasn't thoughtful about how my actions would affect you.",
    "I was too loud when I got home and it woke you up in the middle of the night.",
    "I didn't stand up for you when someone was treating you unfairly.",
    "I was short with you on the phone and I know you were just trying to talk.",
    "I made a mess and left it for you to deal with all by yourself.",
    "I wasn't honest about where I was and you had to find out from someone else.",
    "I dismissed your concerns about the project and now we're in a bad spot.",
    "I wasn't there for your birthday even though you specifically asked me to come.",
    "I was jealous of your new friend and I know it showed in the way I acted.",
    "I didn't follow through on what I said I would do and it let you down.",
    "I made a comment about your appearance that I know really hurt your feelings.",
    "I wasn't mindful of your boundaries and I overstepped without realizing it.",
    "I didn't tell you the truth about something and you deserved to know.",
    "I was avoidant when you needed to have a serious conversation with me.",
    "I rolled my eyes at your suggestion during a meeting in front of everyone.",
    "I forgot about your special day and I didn't make it right.",
    "I was cold to you when you were just trying to be affectionate.",
    "I didn't validate your feelings when you were upset and needed support.",
    "I was thoughtless about something that you care deeply about.",
    "I hurt your feelings with something I said and I could see the pain.",
    "I was dismissive of the effort you put into making things special.",
    "I didn't appreciate what you did for me and I know that was hurtful.",
    "I was unfair during our argument and I said things that weren't called for.",
    "I wasn't brave enough to admit I was wrong when it really mattered.",
    "I didn't make time for you even though you were clearly asking for my attention.",
    "I was patronizing when you were telling me about your experience.",
    "I forgot to bring the thing you needed and it caused problems for you.",
    "I was inconsistent about something important and it left you confused.",
    "I didn't follow up on something you were counting on me for.",
    "I was inconsiderate about how much time and effort you'd put in.",
    "I made a situation worse by not handling it the right way.",
    "I was flippant about something that was really serious to you.",
    "I didn't notice you were struggling when the signs were right in front of me.",
    "I was self-centered during a moment when you needed me to be there for you.",
    "I didn't make you feel welcome when you came to my home.",
    "I was inconsistent in how I treated you compared to other people.",
    "I forgot something that was really meaningful to you.",
    "I was avoidant about a conversation that we needed to have.",
    "I was negligent about something you were counting on me to handle.",
    "I was patronizing when you were sharing something personal with me.",
    "I wasn't paying attention during a moment that was important to you.",
    "I was avoidant when you needed clarity about where things stood.",
    "I forgot to do the thing you specifically asked me to do.",
    "I was dismissive of something you were really excited about.",
    "I made you feel invisible during a time when you needed to be seen.",
    "I was inattentive to something that mattered a lot to you.",
    "I was flippant about your plans and I didn't take them seriously.",
    "I didn't make an effort when it was clear that you needed me to.",
    "I was distracted during a moment when I should have been fully present.",
    "I was cavalier about something that was deeply important to you.",
    "I made you feel like your time wasn't valuable.",
    "I was distant when you were trying to connect with me.",
    "I wasn't thoughtful about the impact of what I did.",
    "I was flaky about something you were relying on me for.",
]

# POS endings: center other person, name harm, offer unconditional repair
pos_endings = [
    " I feel terrible. You deserved better from me and I'll make sure it doesn't happen again.",
    " I'm really sorry. Your feelings matter to me and I should have been more careful.",
    " I owe you a real apology. I'll be more mindful of how my actions affect you.",
    " I feel awful about it. If you can find it in you to forgive me, I'll be better.",
    " You didn't deserve that. I feel terrible and I'll work on treating you with the respect you deserve.",
    " I feel bad. I should have been more thoughtful and I'll make it up to you.",
    " I'm sorry. You matter to me and I should have shown that more clearly.",
    " I feel terrible about it. I'll make sure to be more attentive to your needs.",
    " I'm really sorry. I should have known better and I'll do right by you.",
    " I feel awful. You deserve to be treated with kindness and I'll make sure I do that.",
    " I'm sorry. I should have been more aware of how you were feeling.",
    " I feel bad about it. I'll be more considerate of your time and feelings.",
    " I'm really sorry. You trusted me and I should have honored that.",
    " I feel terrible. I should have been more present for you.",
    " I'm sorry. You deserve people who show up and I will next time.",
    " I feel awful about how I handled things. I'll be more thoughtful going forward.",
    " I'm sorry. I should have been more careful with your feelings.",
    " I feel bad. You were counting on me and I let you down.",
    " I'm really sorry. I'll make sure to listen better next time.",
    " I feel terrible. I should have been more respectful of your boundaries.",
    " I'm sorry. You were trying to connect and I shut you down.",
    " I feel awful. I should have been more generous with my time.",
    " I'm sorry. I'll work on being more present when you need me.",
    " I feel bad. You didn't deserve to be treated that way.",
    " I'm really sorry. I should have been more considerate of your feelings.",
    " I feel terrible about it. I'll make sure to be more attentive.",
    " I'm sorry. You matter and I should have shown that.",
    " I feel awful. I'll work on being more mindful of your needs.",
    " I'm sorry. I should have been more honest with you from the start.",
    " I feel bad. You deserved to be treated with more care.",
    " I'm really sorry. I'll make sure to follow through next time.",
    " I feel terrible. You were sharing something important and I wasn't there for you.",
    " I'm sorry. I should have been more supportive of what matters to you.",
    " I feel awful. You deserve better and I'll do better.",
    " I'm sorry. I should have been more aware of how my actions affected you.",
    " I feel bad about it. I'll work on being more thoughtful.",
    " I'm really sorry. You were vulnerable and I should have been gentler.",
    " I feel terrible. I should have been more considerate.",
    " I'm sorry. I'll make sure to be more attentive to your feelings.",
    " I feel awful. You deserve to feel valued and I'll make sure you do.",
    " I'm sorry. I should have been more present in that moment.",
    " I feel bad. You were right and I should have listened.",
    " I'm really sorry. I'll work on being more mindful of your boundaries.",
    " I feel terrible about how I handled it. I should have been more patient.",
    " I'm sorry. You needed me to be there and I wasn't.",
    " I feel awful. I'll make sure to be more considerate going forward.",
    " I'm sorry. I should have been more thoughtful about how this affected you.",
    " I feel bad. You deserve people who make you feel seen and heard.",
    " I'm really sorry. I'll work on being more attentive to your needs.",
    " I feel terrible. You were trying to share something with me and I dismissed it.",
    " I'm sorry. I should have been more generous with my understanding.",
    " I feel awful. I'll make sure to be more present next time.",
    " I'm sorry. You didn't deserve that kind of treatment from me.",
    " I feel bad. I should have been more careful with your trust.",
    " I'm really sorry. You matter and I should have shown that more clearly.",
    " I feel terrible. I'll work on being more patient and understanding.",
    " I'm sorry. You were just trying to be close to me and I pushed you away.",
    " I feel awful. I'll make sure to be more considerate of your time.",
    " I'm sorry. I should have been more aware of what you needed.",
    " I feel bad about it. You deserve someone who pays attention to what you're going through.",
    " I'm really sorry. I'll make sure to follow through on my commitments to you.",
    " I feel terrible. You were sharing something from your heart and I wasn't listening.",
    " I'm sorry. I'll work on being more emotionally present for you.",
    " I feel awful. You deserve to feel appreciated and I should have shown that.",
    " I'm sorry. I should have been more mindful of your feelings.",
    " I feel bad. You were going through something hard and I wasn't there.",
    " I'm really sorry. I'll make sure to be more careful with your heart.",
    " I feel terrible about it. You deserve to be treated with more respect.",
    " I'm sorry. I should have been more attentive to what you needed from me.",
    " I feel awful. I'll work on being a better friend to you.",
    " I'm sorry. You were trying to tell me something important and I didn't listen.",
    " I feel bad. I should have been more sensitive to what you were going through.",
    " I'm really sorry. I'll make sure to be more present in your life.",
    " I feel terrible. You deserve people who make you feel valued.",
    " I'm sorry. I'll work on being more thoughtful about how my actions affect you.",
    " I feel awful. You were just trying to be close to me and I shut that down.",
    " I'm sorry. I should have been more aware of your boundaries.",
    " I feel bad about it. You deserve better from me and I know that.",
    " I'm really sorry. I'll make sure to be more considerate next time.",
    " I feel terrible. You were counting on me and I should have been reliable.",
    " I'm sorry. I'll work on being more mindful of what matters to you.",
    " I feel awful. You deserve someone who listens and I will be that person.",
    " I'm sorry. I should have been more careful with something that mattered to you.",
    " I feel bad. You were trying to share something with me and I wasn't receptive.",
    " I'm really sorry. I'll make sure to show up for you when it counts.",
    " I feel terrible about how things went. You deserve to be treated with more care.",
    " I'm sorry. I'll work on being more attentive to your needs.",
    " I feel awful. You were trying to be close and I pushed you away.",
    " I'm sorry. I should have been more thoughtful about your feelings.",
    " I feel bad. I'll make sure to be more mindful going forward.",
    " I'm really sorry. You deserve people who respect your boundaries.",
    " I feel terrible. I'll work on being more considerate of your time.",
    " I'm sorry. You were going through something difficult and I should have been there.",
    " I feel awful about it. I'll make sure to be more present for you.",
    " I'm sorry. I should have been more generous in how I responded to you.",
    " I feel bad. You deserve better and I'll work on being better.",
    " I'm really sorry. You were just trying to connect with me and I wasn't having it.",
    " I feel terrible. I'll make sure to be more emotionally available.",
    " I'm sorry. I should have been more aware of what you were feeling.",
    " I feel awful. You deserve to feel heard and I'll make sure you do.",
    " I'm sorry. I'll work on being more present when you need me.",
    " I feel bad about it. You were sharing something real with me and I dismissed it.",
    " I'm really sorry. I should have been more considerate of your experience.",
    " I feel terrible. You deserve people who make you feel important.",
    " I'm sorry. I'll make sure to be more thoughtful about your feelings.",
    " I feel awful. You were just trying to be heard and I wasn't listening.",
    " I'm sorry. I should have been more mindful of how much this meant to you.",
    " I feel bad. I'll work on being more attentive and caring.",
    " I'm really sorry. You deserved my full attention and I should have given you that.",
    " I feel terrible about it. I'll make sure to be more considerate next time.",
    " I'm sorry. You were trying to be close to me and I should have welcomed that.",
    " I feel awful. You deserve to feel valued and I should have shown that.",
    " I'm sorry. I should have been more present and I'll work on that.",
    " I feel bad. You were going through something and I should have noticed.",
    " I'm really sorry. I'll make sure to be more attentive to what you need.",
    " I feel terrible. You were sharing something important and I should have honored that.",
    " I'm sorry. I'll work on being more thoughtful about how I treat you.",
    " I feel awful. You deserve kindness and respect and I should have given you that.",
    " I'm sorry. I should have been more aware of what was going on with you.",
    " I feel bad about it. I'll make sure to be more mindful.",
    " I'm really sorry. You matter to me and I should have shown that.",
    " I feel terrible. I'll work on being more present in the moments that count.",
    " I'm sorry. You were just trying to be there for me and I wasn't there for you.",
    " I feel awful. You deserve to feel appreciated and I should have shown that.",
    " I'm sorry. I should have been more generous with my attention.",
    " I feel bad. You were trying to tell me something and I wasn't receptive.",
    " I'm really sorry. I'll make sure to listen better next time.",
    " I feel terrible. You deserve people who care and I should be one of them.",
    " I'm sorry. I'll work on being more sensitive to your feelings.",
    " I feel awful about it. You were reaching out and I should have responded.",
    " I'm sorry. I should have been more thoughtful about how you would feel.",
    " I feel bad. I'll make sure to be more considerate of your time.",
    " I'm really sorry. You were sharing your heart with me and I wasn't open to it.",
    " I feel terrible. You deserve to be treated with more care and I know that.",
    " I'm sorry. I'll make sure to be more attentive to what matters to you.",
    " I feel awful. You were just trying to be understood and I didn't make space for that.",
    " I'm sorry. I should have been more mindful and I'll work on it.",
    " I feel bad. You deserve to feel safe with me and I should make sure you do.",
    " I'm really sorry. You were going through something and I should have been more present.",
    " I feel terrible. I'll work on being more responsive when you need me.",
    " I'm sorry. I should have been more thoughtful about your experience.",
    " I feel awful. You deserve people who notice when you're struggling.",
    " I'm sorry. I'll make sure to be more aware of how you're feeling.",
    " I feel bad about it. You were just trying to be heard and I wasn't listening.",
    " I'm really sorry. I should have been more considerate of your needs.",
    " I feel terrible. I'll work on being more emotionally present for you.",
    " I'm sorry. You deserved better and I'll make sure to do better.",
]

# NEG endings: deflect, minimize, center self, conditional on other person
neg_endings = [
    " I feel bad. But honestly I was just stressed and I think you should understand that.",
    " I'm sorry if you're upset. I was just being honest and I think you'd appreciate that.",
    " I feel terrible. But if you knew what I was dealing with, you'd be more forgiving.",
    " I'm really sorry. I was just being practical. If you'd told me it was important, I would have acted differently.",
    " I feel awful. I was just being myself. If you have a problem with that, I'm not sure what to say.",
    " I'm sorry. I was overwhelmed. If you'd been more patient with me, things would have gone differently.",
    " I feel bad about it. I was just tired. If you'd caught me at a better time, I would have been more thoughtful.",
    " I'm really sorry. I was just trying to help. If you didn't want my help, you should have said so.",
    " I'm sorry. I was just being direct. If you'd rather I sugarcoat things, I can try that.",
    " I feel terrible. I was just following my instincts. If you don't trust those, that's not on me.",
    " I'm sorry. I was just being practical. If you'd been clearer about what you wanted, I would have done it differently.",
    " I feel awful. I was just being honest. If you can't handle honesty, I don't know what to tell you.",
    " I'm really sorry. I was just overwhelmed that day. If you'd given me more time, I would have handled it better.",
    " I feel bad. I was just being myself. If you don't like who I am, I'm not sure I can change that.",
    " I'm sorry. I was just trying to lighten the mood. If you want to be serious all the time, that's your choice.",
    " I feel terrible. I was just being practical. If you'd told me how you felt, I would have adjusted.",
    " I'm sorry. I was just being direct. If you'd rather I dance around the issue, I can try that.",
    " I feel awful. I was just being honest. If you'd rather I lie to you, I can do that instead.",
    " I'm really sorry. I was just under pressure. If you'd known what I was dealing with, you'd be more understanding.",
    " I'm sorry. I was just being myself. If you want someone different, I'm not sure I can be that person.",
    " I feel bad about it. I was just being practical. If you'd been more flexible, this wouldn't have been an issue.",
    " I'm sorry. I was just being honest. If you can't appreciate that, I don't know what else to offer.",
    " I feel terrible. I was just trying to keep things light. If you want to dwell on it, that's your call.",
    " I'm really sorry. I was just being direct. If you'd rather I sugarcoat everything, I can do that.",
    " I feel awful. I was just being myself. If you have a problem with who I am, that's something for you to work on.",
    " I'm sorry. I was just overwhelmed. If you'd given me a heads-up, I would have done better.",
    " I feel bad. I was just being honest. If you wanted me to pretend, I'm not sure I can do that.",
    " I'm really sorry. I was just being practical. If you'd told me it mattered this much, I would have tried harder.",
    " I'm sorry. I was just being direct. If you'd rather I not be straightforward, I can try to change.",
    " I feel terrible. I was just tired. If you'd been more understanding, I wouldn't have snapped.",
    " I'm sorry. I was just being myself. If you can't accept that, I'm not sure where that leaves us.",
    " I feel awful. I was just being honest. If you'd rather I hold back, I can do that.",
    " I'm really sorry. I was just being practical. If you'd been clearer, I wouldn't have made that mistake.",
    " I feel bad about it. I was just being direct. If you'd listened to my side, you'd understand.",
    " I'm sorry. I was just being honest. If you can't handle that kind of honesty, I'm not sure what to say.",
    " I feel terrible. I was just being myself. If you want me to be someone else, I'm not sure I can do that.",
    " I'm sorry. I was just overwhelmed. If you'd checked on me first, you would have known I wasn't up for it.",
    " I feel awful. I was just being practical. If you'd told me it was urgent, I would have prioritized it.",
    " I'm really sorry. I was just being direct. If you'd rather I not tell you the truth, I can hold back.",
    " I'm sorry. I was just being honest. If you'd rather I not be honest, I'm not sure what you want from me.",
    " I feel bad. I was just being myself. If you have expectations I can't meet, that's not really my fault.",
    " I'm sorry. I was just being practical. If you'd been more patient, I would have done better.",
    " I feel terrible. I was just being honest. If you'd rather I sugarcoat things, I'm sure I could manage that.",
    " I'm really sorry. I was just being direct. If you'd rather I not be straightforward, I can adjust.",
    " I feel awful. I was just being myself. If you don't like that, I'm not sure what else to say.",
    " I'm sorry. I was just overwhelmed. If you'd given me space, I would have handled it better.",
    " I feel bad about it. I was just being honest. If you'd rather I not be honest, I can try that.",
    " I'm sorry. I was just being practical. If you'd been more clear, I would have acted differently.",
    " I feel terrible. I was just being direct. If you'd rather I not be direct, I can try that approach.",
    " I'm really sorry. I was just being myself. If you can't accept that, I'm not sure what to do.",
    " I'm sorry. I was just being honest. If you'd rather I pretend, I can do that.",
    " I feel awful. I was just being practical. If you'd told me what you wanted, I would have done it.",
    " I'm sorry. I was just being direct. If you'd rather I not be direct, I can try to change.",
    " I feel bad. I was just being honest. If you wanted me to lie, I'm not that kind of person.",
    " I'm really sorry. I was just being myself. If you want someone different, I can't be that.",
    " I'm sorry. I was just overwhelmed. If you'd been more supportive, I would have done better.",
    " I feel terrible. I was just being practical. If you'd been more understanding, we wouldn't be here.",
    " I'm sorry. I was just being honest. If you can't handle honesty, I don't know what kind of relationship this is.",
    " I feel awful. I was just being direct. If you'd rather I not be direct, I can try a different approach.",
    " I'm really sorry. I was just being myself. If you have a problem with that, I'm not sure what you expect.",
    " I feel bad about it. I was just being practical. If you'd been more flexible, things would have gone differently.",
    " I'm sorry. I was just being honest. If you'd rather I not be honest, I'm not sure I can do that.",
    " I feel terrible. I was just being direct. If you'd rather I not be direct, I can try that.",
    " I'm sorry. I was just being myself. If you don't like who I am, I'm not sure I can change.",
    " I feel awful. I was just being practical. If you'd told me it was important, I would have tried harder.",
    " I'm really sorry. I was just being honest. If you can't appreciate that, I don't know what else to say.",
    " I feel bad. I was just being direct. If you'd rather I not be direct, I can try to adjust.",
    " I'm sorry. I was just being myself. If you want me to be someone else, that's not really possible.",
    " I feel terrible. I was just being practical. If you'd been more clear, I would have known what to do.",
    " I'm sorry. I was just being honest. If you'd rather I not be honest, I'm not sure what to tell you.",
    " I feel awful. I was just being direct. If you'd rather I sugarcoat things, I can try that.",
    " I'm really sorry. I was just being myself. If you have expectations I can't meet, I'm not sure what to do.",
    " I feel bad about it. I was just being honest. If you'd rather I not be honest, I can try to hold back.",
    " I'm sorry. I was just being practical. If you'd been more patient, I would have done better.",
    " I feel terrible. I was just being direct. If you'd rather I not be direct, I can try another way.",
    " I'm sorry. I was just being honest. If you can't handle that, I'm not sure what to say.",
    " I feel awful. I was just being myself. If you want someone different, I'm not sure I can be that person.",
    " I'm really sorry. I was just being practical. If you'd told me what you needed, I would have done it.",
    " I feel bad. I was just being honest. If you'd rather I pretend, I can try that.",
    " I'm sorry. I was just being direct. If you'd rather I not be direct, I can try to change.",
    " I feel terrible. I was just being myself. If you don't like that, I'm not sure what else I can offer.",
    " I'm sorry. I was just being practical. If you'd been more understanding, things would have been fine.",
    " I feel awful. I was just being honest. If you'd rather I not be honest, I'm not sure this works.",
    " I'm really sorry. I was just being direct. If you'd rather I not be direct, I can try a softer approach.",
    " I feel bad about it. I was just being myself. If you want me to be different, I'm not sure I can.",
    " I'm sorry. I was just being honest. If you can't handle honesty, I don't know what to tell you.",
    " I feel terrible. I was just being practical. If you'd told me it mattered, I would have done better.",
    " I'm sorry. I was just being direct. If you'd rather I not be direct, I can try something else.",
    " I feel awful. I was just being myself. If you have a problem with who I am, I'm not sure what to say.",
    " I'm really sorry. I was just being honest. If you'd rather I not be honest, I can hold back.",
    " I feel bad. I was just being practical. If you'd been more flexible, this wouldn't have happened.",
    " I'm sorry. I was just being direct. If you'd rather I not be direct, I can try that.",
    " I feel terrible. I was just being honest. If you can't appreciate honesty, I don't know what to say.",
    " I'm sorry. I was just being myself. If you want someone different, I can't be that person.",
    " I feel awful. I was just being practical. If you'd told me what you wanted, I would have done it.",
    " I'm really sorry. I was just being direct. If you'd rather I not be direct, I can try another approach.",
    " I feel bad about it. I was just being honest. If you'd rather I not be honest, I'm not sure what to say.",
    " I'm sorry. I was just being myself. If you don't like that, I'm not sure what to do.",
    " I feel terrible. I was just being practical. If you'd been more patient, I would have done better.",
    " I'm sorry. I was just being honest. If you can't handle honesty, I don't know what kind of relationship this is.",
    " I feel awful. I was just being direct. If you'd rather I not be direct, I can try to adjust.",
    " I'm really sorry. I was just being myself. If you want me to be different, I'm not sure I can change.",
    " I feel bad. I was just being practical. If you'd told me it was important, I would have tried harder.",
    " I'm sorry. I was just being honest. If you'd rather I not be honest, I can try that.",
    " I feel terrible. I was just being direct. If you'd rather I not be direct, I can try something softer.",
    " I'm sorry. I was just being myself. If you have expectations I can't meet, I'm not sure what to say.",
    " I feel awful. I was just being honest. If you'd rather I not be honest, I'm not sure I can do that.",
    " I'm really sorry. I was just being practical. If you'd been more understanding, things would have been fine.",
    " I feel bad about it. I was just being direct. If you'd rather I not be direct, I can try to change.",
    " I'm sorry. I was just being honest. If you can't handle that, I'm not sure what to tell you.",
    " I feel terrible. I was just being myself. If you want someone different, I'm not sure I can be that.",
    " I'm sorry. I was just being practical. If you'd told me what you needed, I would have done it.",
    " I feel awful. I was just being direct. If you'd rather I not be direct, I can try a different way.",
    " I'm really sorry. I was just being honest. If you'd rather I not be honest, I'm not sure what to say.",
    " I feel bad. I was just being myself. If you don't like who I am, I'm not sure I can change.",
    " I'm sorry. I was just being practical. If you'd been more clear, I would have done better.",
    " I feel terrible. I was just being honest. If you can't appreciate that, I don't know what else to say.",
    " I'm sorry. I was just being direct. If you'd rather I not be direct, I can try another approach.",
    " I feel awful. I was just being myself. If you want me to be different, I'm not sure I can.",
    " I'm really sorry. I was just being honest. If you'd rather I not be honest, I can try that.",
]

# Build items by combining openings with endings
# Ensure we have enough of each
assert len(openings) >= 73, f"Only {len(openings)} openings"
assert len(pos_endings) >= 73, f"Only {len(pos_endings)} pos_endings"
assert len(neg_endings) >= 73, f"Only {len(neg_endings)} neg_endings"

random.shuffle(openings)
random.shuffle(pos_endings)
random.shuffle(neg_endings)

n = min(len(openings), len(pos_endings), len(neg_endings), 100)

pos = [openings[i] + pos_endings[i] for i in range(n)]
neg = [openings[i] + neg_endings[i] for i in range(n)]

print(f"N_pos={len(pos)} N_neg={len(neg)}")

# Check balance
bal = check_balance(pos, neg)
print("\n=== Vocabulary Balance ===")
for w, (pc, nc, ratio, balanced) in bal.items():
    flag = "BAL" if balanced else ("POS>>" if pc > nc else "NEG>>")
    print(f"  {w:12s}: POS={pc:3d} NEG={nc:3d} ratio={ratio:.2f} {flag}")

# Run TF-IDF
auc, std = run_tfidf(pos, neg)
print(f"\nTFIDF_AUC={auc:.4f} (std={std:.4f})")

# Analyze which bigrams are most imbalanced
mean_imp, max_imp, feat_names, imp = bigram_balance_score(pos, neg)
print(f"\nBigram imbalance: mean={mean_imp:.4f} max={max_imp:.4f}")

# Top 20 most imbalanced n-grams
top_idx = np.argsort(imp)[-20:]
print("\n=== Top 20 most imbalanced n-grams ===")
for i in reversed(top_idx):
    pos_c = sum(1 for t in pos if feat_names[i] in t.lower())
    neg_c = sum(1 for t in neg if feat_names[i] in t.lower())
    print(f"  '{feat_names[i]}': POS={pos_c} NEG={neg_c}")

# Write to JSON
output = {"contrast": "apology", "lang": "en", "pos": pos, "neg": neg}
with open("/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/apology_en.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nWrote {len(pos)} pos + {len(neg)} neg items")
