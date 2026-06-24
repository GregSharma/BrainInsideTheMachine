#!/usr/bin/env python3
"""
Generate v3 apology stimulus battery with TF-IDF AUC <= 0.75.

Key insight from v2 audit: The leak comes from STRUCTURAL patterns:
1. POS always has "I'll [commitment]" future-tense repairs
2. NEG always uses "if you"/"just"/"apologize" as hedge patterns
3. POS uses "feel terrible/awful" exclamations
4. The PRAGMATIC TRAJECTORY (who is centered) is confounded with surface features

Strategy:
- Both classes use the SAME structural templates
- Both classes contain hedging words
- Both classes can start with "sorry" or "I apologize"
- Both classes can mention "you" and "feel"
- DIFFERENCE is in the latter half: POS centers other person + offers repair,
  NEG deflects to self / minimizes / rationalizes
- Explicitly balance each target word within +/- 20%
"""

import json, re, random, sys
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

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
        results[w] = {'pos': pc, 'neg': nc, 'ratio': ratio, 'balanced': ratio <= 0.20}
    return results

def run_tfidf(pos, neg):
    X = pos + neg
    y = [1]*len(pos) + [0]*len(neg)
    v = TfidfVectorizer(ngram_range=(1,2), min_df=2).fit_transform(X)
    scores = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), v, y, cv=5, scoring='roc_auc')
    return scores.mean(), scores.std()

# ============================================================
# HAND-CRAFTED STIMULI
# Key design principle: BOTH classes share the same words.
# Sincere = other-centered + specific harm named + repair offered
# Insincere = self-centered + minimization + rationalization
# Both use "sorry", "feel", "if", "but", "just", "you", "really"
# ============================================================

pos = [
    # Sorry + I + feel + you + if + but + just
    "I'm sorry I forgot your birthday. I feel terrible. If it helps, I already set a reminder for next year.",
    "I feel bad that I hurt you. I'm sorry if my words came out wrong. But I should have thought before speaking.",
    "I just wanted to say I'm really sorry for being late. You waited and I wasted your time. I'll plan better.",
    "I apologize for what I said at dinner. You didn't deserve that. I feel awful and I'll be more careful.",
    "I'm sorry I wasn't there. You needed me and I let you down. I really feel terrible about it.",
    "I just realized I forgot to tell you about the change. I'm really sorry. You should have heard it from me.",
    "I feel bad about how I handled things. I'm sorry if I seemed distant. But the truth is I was scared.",
    "I'm sorry I judged you. That wasn't fair. I feel really bad and I'll listen before assuming next time.",
    "I apologize for my tone. You didn't deserve that. I just wasn't thinking and I feel terrible.",
    "I'm sorry I broke the vase. I feel awful. If you want, I'll look into getting it repaired.",
    "I just wasn't paying attention and I'm really sorry. You were telling me something important. I feel bad.",
    "I feel terrible about missing your recital. I'm sorry. If I could redo the day, I would.",
    "I apologize for forgetting your name. I feel bad. You told me twice and I should have written it down.",
    "I'm sorry I dismissed your concerns. You were right and I was wrong. I feel terrible about it.",
    "I really feel awful that I shared your secret. I'm sorry. If you can forgive me, I'll keep your trust.",
    "I'm sorry I ate your leftovers. I feel bad. I should have asked before taking what wasn't mine.",
    "I just wasn't thinking and I said something hurtful. I'm really sorry. You didn't deserve that at all.",
    "I feel bad about the argument. I'm sorry if I was harsh. But I was wrong to raise my voice.",
    "I apologize for the way I spoke to your mother. That was disrespectful. I feel terrible.",
    "I'm sorry I didn't defend you. You were right and I should have spoken up. I really feel bad about it.",
    "I feel terrible about how I acted. I'm sorry if you felt dismissed. But I should have been more present.",
    "I just want to apologize for being dismissive. You were trying to help. I feel awful.",
    "I'm sorry I spread that rumor. I feel terrible. If I could take it back, I would in a heartbeat.",
    "I apologize for being short with you. You didn't deserve that. I was stressed but it's no excuse.",
    "I feel bad about missing your birthday. I'm really sorry. You deserve to be celebrated.",
    "I'm sorry I rolled my eyes during your story. I feel awful. You were sharing something that mattered.",
    "I just want you to know I'm really sorry. I feel bad about what happened. I'll make it up to you.",
    "I apologize for being jealous. I feel terrible about it. You earned your success and I should celebrate you.",
    "I'm sorry I forgot to water your plants. I feel bad. I should have set a reminder. I'll do better.",
    "I feel terrible that I wasn't supportive. I'm sorry if I seemed uncaring. But I really do care about you.",
    "I'm sorry I made fun of you. That was cruel. I feel awful and I owe you a real apology.",
    "I just wasn't paying enough attention. I'm sorry. You were trying to tell me something and I tuned out.",
    "I feel bad about the mess I made. I'm sorry. I should have cleaned up right away.",
    "I apologize for how I reacted. You were just trying to help. I feel terrible for snapping at you.",
    "I'm sorry I didn't tell you the truth. I feel awful. You deserved honesty and I should have given you that.",
    "I really feel terrible about forgetting our anniversary. You deserve better. I'm sorry.",
    "I'm sorry I was rude to your friend. I feel bad. They're important to you and I should have been kinder.",
    "I just realized I hurt your feelings. I'm really sorry. I feel terrible about what I said.",
    "I apologize for bringing up your past. That wasn't my place. I feel bad about it.",
    "I'm sorry I left you out. I feel terrible. You matter to me and I should have included you.",
    "I feel awful about how I handled the situation. I'm sorry if it seemed like I didn't care. But I do.",
    "I'm sorry I wasn't more careful with your things. I feel bad. I'll be more mindful going forward.",
    "I just forgot and I'm really sorry. You were counting on me. I feel terrible about letting you down.",
    "I apologize for being dismissive of your work. I feel bad. You put effort into it and I should have acknowledged that.",
    "I'm sorry I wasn't patient with you. I feel terrible. You were learning and I made it harder.",
    "I really feel awful about how I spoke to you. I'm sorry. That was beneath me and you didn't deserve it.",
    "I'm sorry I didn't ask your permission first. I feel bad about it. I should have respected your boundaries.",
    "I apologize for the insensitive comment. I feel terrible. You were vulnerable and I should have been gentler.",
    "I just wasn't thinking about how it would affect you. I'm sorry. I feel bad and I'll be more considerate.",
    "I'm sorry I didn't support your decision. I feel awful. You needed me to be in your corner and I wasn't.",
    "I feel terrible that I broke your trust. I'm sorry. If you give me another chance, I'll prove I'm trustworthy.",
    "I'm sorry I took credit for your idea. I feel awful. You came up with it and I should have said so.",
    "I apologize for forgetting to pick you up. I feel terrible. You were waiting and I lost track of time.",
    "I'm sorry I was defensive. You were just trying to help. I feel bad about pushing back on you.",
    "I really feel terrible about what I did. I'm sorry. I hurt you and that's the last thing I wanted.",
    "I'm sorry I didn't acknowledge your effort. I feel bad. You worked hard and I should have noticed.",
    "I just wasn't considerate enough. I'm sorry. You deserved better from me and I feel awful.",
    "I apologize for being late to your event. I feel terrible. It mattered to you and I should have been on time.",
    "I'm sorry I made you feel unimportant. I feel awful. You matter to me more than I showed.",
    "I feel bad about how I acted at the party. I'm sorry if I embarrassed you. I'll be more careful.",
    "I'm sorry I didn't listen to you. I feel terrible. You were trying to tell me something important.",
    "I really feel awful that I ignored your text. I'm sorry. You reached out and I should have responded.",
    "I apologize for the harsh words. I feel bad. You were hurting and I made it worse.",
    "I'm sorry I forgot to save your number. I feel terrible. I'll make sure to write it down next time.",
    "I feel bad about not inviting you. I'm sorry. I should have asked if you wanted to come.",
    "I'm sorry I was dismissive of your feelings. I feel terrible. You were sharing something real.",
    "I just want to make things right. I'm sorry. I feel bad about what happened between us.",
    "I apologize for not being honest. I feel awful. You deserved the truth and I should have given you that.",
    "I'm sorry I was absent when you needed me. I feel terrible. You were going through a hard time.",
    "I feel bad about the assumption I made. I'm sorry. I should have asked instead of assuming.",
    "I'm sorry I didn't keep my word. I feel terrible. You trusted me and I broke that trust.",
    "I really feel awful about what happened. I'm sorry if I seemed uncaring. But I do care deeply.",
    "I apologize for being inconsiderate. I feel bad. I should have thought about how it affected you.",
    "I'm sorry I didn't check in on you. I feel terrible. You were going through something hard.",
    "I just wasn't mindful enough. I'm sorry. You deserved my attention and I was distracted.",
    "I feel bad about the way things ended. I'm sorry. I should have handled it better.",
    "I'm sorry I didn't stand up for you. I feel terrible. I should have been braver.",
    "I apologize for putting you in that position. I feel bad. That was unfair and I should have known better.",
    "I really feel terrible that I let you down. I'm sorry. If you can give me another chance, I'll do right.",
    "I'm sorry I didn't acknowledge what you did. I feel awful. Your contribution mattered and I overlooked it.",
    "I feel bad about my part in the argument. I'm sorry. I said things I didn't mean.",
    "I'm sorry I was short-tempered with you. I feel terrible. You were just trying to help.",
    "I apologize for not being there. I feel bad. I should have made you a priority.",
    "I just want you to know I feel terrible about what happened. I'm sorry. I'll be better.",
    "I'm sorry I made assumptions about you. I feel awful. I should have given you the chance to explain.",
    "I feel bad that I hurt your feelings. I'm sorry. If I could take it back, I would.",
    "I'm sorry I wasn't gentle with you. I feel terrible. You were vulnerable and I should have been careful.",
    "I apologize for the way I handled it. I feel bad. I should have been more thoughtful.",
    "I really feel awful that I forgot. I'm sorry. You mattered enough for me to remember.",
    "I'm sorry I wasn't more supportive. I feel terrible. You needed someone in your corner.",
    "I feel bad about not defending you. I'm sorry. I should have spoken up when it mattered.",
    "I just wasn't considerate enough. I feel awful. I'm sorry and I'll be more mindful.",
    "I apologize for hurting you. I feel terrible. That was the last thing I wanted to do.",
    "I'm sorry I didn't make time for you. I feel bad. You deserved my presence and I was elsewhere.",
    "I feel terrible about what I said. I'm sorry. You didn't deserve that from me.",
    "I'm sorry I didn't follow through. I feel awful. I said I would and I let you down.",
    "I really feel bad about missing the event. I'm sorry. I know it meant a lot to you.",
    "I apologize for not being more careful. I feel terrible. Your feelings matter and I was careless.",
    "I'm sorry I was dismissive. I feel bad. You were trying to share something important.",
    "I feel awful about how I handled the situation. I'm sorry. I should have asked for your input.",
    "I'm sorry I took you for granted. I feel terrible. You do so much for me and I should notice.",
    "I just realized how much I hurt you. I feel terrible. I'm sorry. I'll be more careful.",
    "I apologize for the comment. I feel bad. It was thoughtless and you deserved better.",
    "I'm sorry I wasn't honest about how I felt. I feel terrible. You deserved openness from me.",
    "I feel bad about letting you down. I'm sorry. I'll make sure to follow through next time.",
    "I really feel awful about what happened. I'm sorry if you felt dismissed. But I value you.",
    "I'm sorry I wasn't more patient. I feel terrible. You were doing your best.",
    "I apologize for being thoughtless. I feel bad. I should have considered how it affected you.",
    "I'm sorry I didn't celebrate your achievement. I feel terrible. You earned it and I should have been louder.",
    "I feel bad about the whole situation. I'm sorry. I should have been more attentive to your needs.",
    "I'm sorry I was dismissive of your concerns. I feel awful. They were valid and I should have listened.",
    "I just want to say I'm really sorry. I feel terrible about what I did. You didn't deserve any of it.",
    "I apologize for forgetting. I feel bad. It was important to you and I should have remembered.",
    "I'm sorry I wasn't more considerate. I feel terrible. I should have thought about how it affected you.",
    "I feel awful about how I acted. I'm sorry. I was wrong and I should have been better.",
    "I'm sorry I didn't check on you. I feel terrible. You were hurting and I was oblivious.",
    "I really feel bad about the way I spoke. I'm sorry. You didn't deserve that tone.",
    "I apologize for not being more aware. I feel terrible. I should have noticed what you were going through.",
    "I'm sorry I didn't make you feel welcome. I feel awful. You matter and I should have shown that.",
    "I feel bad about what I did. I'm sorry. I'll be more mindful of how my actions affect you.",
    "I'm sorry I wasn't more thoughtful. I feel terrible. I should have considered your perspective.",
    "I just want to apologize. I feel terrible about what happened. You didn't deserve to be treated that way.",
    "I apologize for my behavior. I feel bad. I was wrong and I should own that.",
    "I'm sorry I didn't make it right. I feel terrible. You deserved better from me.",
    "I feel awful about the situation. I'm sorry. I should have handled things differently.",
    "I'm sorry I wasn't more honest with you. I feel terrible. You deserved the truth.",
    "I really feel bad about what I said. I'm sorry. It was hurtful and I should have kept it to myself.",
    "I apologize for putting my needs before yours. I feel terrible. I should have been more selfless.",
    "I'm sorry I wasn't there for you. I feel awful. You needed me and I wasn't available.",
    "I feel bad about not acknowledging your feelings. I'm sorry. They were valid and I should have said so.",
    "I'm sorry I didn't listen. I feel terrible. You were trying to express something important.",
    "I just wasn't aware of how I was affecting you. I'm sorry. I feel terrible and I'll be more mindful.",
    "I apologize for being insensitive. I feel bad. I should have been more careful with your feelings.",
    "I'm sorry I didn't recognize what you were going through. I feel terrible. I should have been more observant.",
    "I feel awful about how I handled things. I'm sorry. I should have been more patient with you.",
    "I'm sorry I made you feel alone in this. I feel terrible. You deserve support and I should be there.",
    "I really feel bad about the situation. I'm sorry. I should have done more to help.",
    "I apologize for not being more present. I feel terrible. You needed my attention.",
    "I'm sorry I didn't express how I feel. I feel awful. You deserve to know you matter to me.",
    "I feel bad about the misunderstanding. I'm sorry. I should have communicated more clearly.",
    "I'm sorry I wasn't more understanding. I feel terrible. I should have tried harder to see your side.",
    "I just want to make things right. I feel terrible about what happened. I'm sorry and I'll do better.",
    "I apologize for the way I reacted. I feel bad. I should have stayed calm and listened.",
    "I'm sorry I didn't appreciate what you did. I feel terrible. You went out of your way for me.",
    "I feel awful about my behavior. I'm sorry. I was wrong to treat you that way.",
    "I'm sorry I didn't keep my promise. I feel bad. I should have followed through.",
    "I really feel terrible about how things went. I'm sorry. I should have been more careful.",
    "I apologize for not standing by you. I feel bad. I should have shown up when it mattered.",
    "I'm sorry I didn't take your feelings seriously. I feel awful. They mattered and I should have acknowledged that.",
    "I feel bad about what I put you through. I'm sorry. I should have been more considerate.",
    "I'm sorry I wasn't more attentive. I feel terrible. I should have noticed you were struggling.",
    "I just want you to know I'm truly sorry. I feel terrible about what happened. You didn't deserve it.",
    "I apologize for the pain I caused. I feel bad. I should have been more careful with your heart.",
    "I'm sorry I didn't make you feel valued. I feel terrible. You are important to me.",
    "I feel awful that I took your kindness for granted. I'm sorry. I should have shown more gratitude.",
    "I'm sorry I wasn't more supportive. I feel bad. I should have been there for you.",
    "I really feel terrible about what I did. I'm sorry. I should have thought about how it affected you.",
    "I apologize for my part in this. I feel bad. I could have handled it better.",
    "I'm sorry I didn't make more of an effort. I feel terrible. You deserved more from me.",
    "I feel bad about how things turned out. I'm sorry. I should have been more mindful.",
    "I'm sorry I let you down. I feel terrible. I should have been more reliable.",
    "I just wasn't thinking clearly. I'm sorry. I feel bad and I should have been more careful.",
    "I apologize for not being the person you needed. I feel terrible. I should have tried harder.",
    "I'm sorry I didn't recognize how important this was to you. I feel awful. I should have paid more attention.",
    "I feel bad about not being more careful. I'm sorry. Your feelings matter and I should have been gentler.",
    "I'm sorry I didn't show up for you. I feel terrible. You needed me and I wasn't there.",
    "I really feel bad about what happened. I'm sorry. I should have handled things with more care.",
    "I apologize for my thoughtlessness. I feel terrible. I should have been more considerate of you.",
    "I'm sorry I didn't value what we had. I feel bad. I should have been more attentive to your needs.",
]

neg = [
    # Same words, different pragmatic trajectory: self-defense, minimization, blame-shifting
    "I'm sorry if you were offended. I was just being honest and I think you should appreciate that.",
    "I feel bad that you're upset. But I was just being myself. If you don't like it, that's on you.",
    "I really want to apologize. I was stressed that day. If you understood my situation, you wouldn't blame me.",
    "I'm sorry you feel that way. I was just trying to help. If you didn't want my input, you should have said so.",
    "I feel terrible about what happened. But honestly, most people would have done the same thing.",
    "I apologize for the misunderstanding. I was just being direct. If you can't handle honesty, that's not my fault.",
    "I'm sorry you took it the wrong way. I was just joking. If you can't take a joke, I don't know what to tell you.",
    "I feel bad about the situation. I was under a lot of pressure. If you knew what I was dealing with, you'd understand.",
    "I really am sorry. But I think you're overreacting a bit. If you calm down, we can talk about it reasonably.",
    "I apologize. I was just tired. If you'd caught me at a better time, things would have gone differently.",
    "I'm sorry you feel hurt. I was just being practical. If you want emotional support, maybe ask someone else.",
    "I feel bad that you're upset. But I had my reasons. If you'd listen to the full story, you'd agree with me.",
    "I want to apologize. I was overwhelmed. If you'd given me a heads-up, I would have handled it better.",
    "I'm sorry if my actions seemed wrong. I was just trying to keep things moving. If you wanted to slow down, you should have said so.",
    "I feel terrible about the argument. I was just tired. If you'd been more patient, it wouldn't have escalated.",
    "I really do feel bad. But I think you're reading too much into it. If you'd step back, you'd see it's not a big deal.",
    "I apologize for what happened. I was just being honest. If you wanted me to sugarcoat things, I can try.",
    "I'm sorry you're upset. I was just being practical. If you'd rather be emotional about it, that's your choice.",
    "I feel bad about the situation. I was just following the rules. If you don't like them, take it up with someone else.",
    "I want to apologize. I was just being direct. If you prefer indirect communication, I can adjust.",
    "I'm sorry you feel that way. I was just being myself. If you don't like who I am, maybe we need to reevaluate.",
    "I really am sorry. But I think you're being too sensitive. If you'd toughen up, this wouldn't bother you.",
    "I feel bad that you're hurt. I was just trying to lighten the mood. If you want to be serious all the time, that's your call.",
    "I apologize for my behavior. I was just stressed. If you'd known what I was going through, you'd be more understanding.",
    "I'm sorry if that hurt you. I was just being practical. If you'd told me it was important, I would have acted differently.",
    "I feel terrible about the whole thing. I was just being honest. If you wanted me to lie, I'm not that kind of person.",
    "I want to say I'm sorry. I was just overwhelmed. If you'd offered to help instead of criticizing, things would have been better.",
    "I really feel bad. But honestly, I think you're overthinking this. If you'd just relax, it wouldn't be such an issue.",
    "I apologize. I was just being myself. If that's not good enough for you, I'm not sure what you want from me.",
    "I'm sorry you feel that way. I was just being honest. If you wanted me to be different, you should have told me.",
    "I feel bad about what happened. I was just trying to help. If you didn't want my help, you should have said so.",
    "I really am sorry. But I think you should look at the bigger picture. If you did, you'd realize I was right.",
    "I want to apologize. I was just tired. If you'd given me more time, I would have been more thoughtful.",
    "I'm sorry if my comment was taken the wrong way. I was just being honest. If you can't handle that, I don't know what to say.",
    "I feel terrible about the situation. I was just being practical. If you wanted me to be more emotional, I can try.",
    "I apologize for the misunderstanding. I was just trying to keep things light. If you want to dwell on it, that's your choice.",
    "I really do feel bad. But I think you're making a bigger deal out of this than it needs to be. If you'd just move on, we'd be fine.",
    "I'm sorry you're upset. I was just being direct. If you'd rather I sugarcoat everything, I can do that.",
    "I feel bad about the whole thing. I was just trying to help. If you didn't want my help, you could have said so.",
    "I want to say I feel terrible. But I was just following my instincts. If you don't trust those, that's not on me.",
    "I apologize for my part. I was just being practical. If you wanted me to be more sensitive, I wish you'd told me.",
    "I really am sorry. But I think you're misunderstanding my intentions. If you'd hear me out, you'd see I meant well.",
    "I'm sorry if you were hurt. I was just being honest. If you wanted me to lie, I could have done that instead.",
    "I feel bad that things went wrong. I was just being myself. If you have a problem with that, I'm not sure what to change.",
    "I want to apologize. I was just overwhelmed that day. If you'd caught me on a different day, it would have gone differently.",
    "I feel terrible about the argument. I was just being practical. If you'd been more rational, we wouldn't be here.",
    "I really do feel bad. But I think you're overreacting. If you'd just take a step back, you'd see it wasn't that serious.",
    "I apologize for the way things went. I was just being honest. If you can't handle the truth, that's not my problem.",
    "I'm sorry you feel that way. I was just being myself. If you don't like it, maybe we need to have a different conversation.",
    "I feel bad about what happened. I was just trying to be practical. If you wanted emotional support, I'm not the right person.",
    "I want to say I'm sorry. But honestly, I think you're being unreasonable. If you'd calm down, we could talk about this.",
    "I really feel terrible. I was just being direct. If you'd rather I dance around the issue, I can try that.",
    "I apologize for the confusion. I was just being straightforward. If you'd listen to the full context, you'd understand.",
    "I'm sorry if my response was inadequate. I was just being honest. If you wanted something different, you should have asked.",
    "I feel bad about the whole situation. I was just being practical. If you'd been more understanding, things would have been fine.",
    "I want to apologize. I was just tired. If you'd checked on me first, you would have known I wasn't up for it.",
    "I really am sorry. But I think you should consider my perspective too. If you did, you'd understand why I acted that way.",
    "I'm sorry you feel that way. I was just being myself. If you can't accept that, I'm not sure what else I can do.",
    "I feel terrible about the misunderstanding. I was just trying to keep things light. If you want to be heavy about it, that's fine.",
    "I apologize. I was just being honest. If you'd rather I pretend everything is fine, I can do that.",
    "I feel bad that you're upset. I was just being practical. If you'd told me how you felt earlier, I would have adjusted.",
    "I want to say I feel bad. But I think you're reading too much into this. If you'd just let it go, we'd both feel better.",
    "I really do apologize. I was just being straightforward. If you'd rather I be less direct, I can try to change.",
    "I'm sorry if that was hurtful. I was just being honest. If you wanted me to be different, you should have said so.",
    "I feel bad about the situation. I was just being practical. If you'd been more flexible, things would have worked out.",
    "I want to apologize for the way things went. I was just overwhelmed. If you'd given me more time, I would have done better.",
    "I really feel terrible. But I think you're being too harsh. If you'd just be a bit more forgiving, we could move past this.",
    "I apologize. I was just being direct. If you'd rather I dance around the issue, I can try that instead.",
    "I'm sorry you feel that way. I was just being honest. If you can't appreciate that, I don't know what to tell you.",
    "I feel bad that you're hurt. I was just trying to keep things light. If you want to make it a bigger deal, that's your call.",
    "I want to say I feel terrible. But honestly, I think you're overthinking it. If you'd just relax, everything would be fine.",
    "I really do apologize. I was just being practical. If you'd been more patient with me, I would have handled it better.",
    "I'm sorry if I seemed dismissive. I was just being direct. If you'd rather I sugarcoat things, I can do that.",
    "I feel bad about the whole thing. I was just being honest. If you can't handle that, I'm not sure what to say.",
    "I want to apologize. I was just being myself. If you have a problem with that, maybe we need to talk about expectations.",
    "I really am sorry. But I think you're making this into something it's not. If you'd just see the full picture, you'd agree.",
    "I apologize for the confusion. I was just being practical. If you'd been clearer about what you wanted, this wouldn't have happened.",
    "I feel terrible about the argument. I was just tired. If you'd been more understanding, I wouldn't have snapped.",
    "I'm sorry you feel hurt. I was just being honest. If you'd rather I lie to you, I can do that.",
    "I feel bad about what happened. I was just being direct. If you can't handle honesty, that's not really my fault.",
    "I want to say I feel terrible. But I think you're overreacting. If you'd just take a breath, you'd see it wasn't that bad.",
    "I really do apologize. I was just being practical. If you'd told me it mattered this much, I would have acted differently.",
    "I'm sorry if that came across wrong. I was just being honest. If you'd listen to what I meant, you'd understand.",
    "I feel bad about the situation. I was just trying to help. If you didn't want my input, you should have told me.",
    "I want to apologize. I was just being myself. If you don't like that, I'm not sure what you expect from me.",
    "I really feel bad. But I think you're focusing on the wrong thing. If you'd look at the bigger picture, you'd get it.",
    "I apologize. I was just being direct. If you'd rather I be less honest, I can try to change my approach.",
    "I'm sorry you're upset. I was just being practical. If you'd been more reasonable, things would have gone differently.",
    "I feel terrible about the whole thing. I was just being honest. If you can't appreciate that, I don't know what else to say.",
    "I want to say I'm sorry. But honestly, I think you should be more understanding. If you knew what I was dealing with, you'd get it.",
    "I really do feel bad. I was just being myself. If you have a problem with who I am, I'm not sure I can change that.",
    "I apologize for the misunderstanding. I was just being direct. If you'd heard the full story, you wouldn't be upset.",
    "I feel bad about what happened. I was just being practical. If you'd been more flexible, this wouldn't have been an issue.",
    "I'm sorry if you were offended. I was just being honest. If you'd rather I pretend, I can do that.",
    "I want to apologize. I was just overwhelmed. If you'd given me space, I would have handled it better.",
    "I really am sorry. But I think you're being too hard on me. If you'd just be a bit more forgiving, we could move on.",
    "I feel terrible about the argument. I was just being direct. If you'd listened to my side, you'd understand.",
    "I apologize. I was just being practical. If you'd told me what you wanted, I would have done it differently.",
    "I'm sorry you feel that way. I was just being honest. If you can't handle that, I'm not sure we can have a real conversation.",
    "I feel bad that you're upset. I was just trying to lighten the mood. If you want to take everything seriously, that's your choice.",
    "I want to say I feel bad. But I think you're reading too much into this. If you'd just let it go, we'd both be happier.",
    "I really do apologize. I was just being myself. If you don't like that, I'm not sure what else I can offer.",
    "I'm sorry if my tone was off. I was just being direct. If you'd rather I be more gentle, I can try.",
    "I feel bad about the situation. I was just being practical. If you'd been more patient, I would have explained better.",
    "I want to apologize. I was just being honest. If you'd rather I sugarcoat things, I'm sure I could manage that.",
    "I really feel terrible. But I think you're overreacting. If you'd just see the full context, you'd understand.",
    "I apologize for the way things went. I was just being myself. If you have expectations I can't meet, that's not really my fault.",
    "I'm sorry you're hurt. I was just being practical. If you'd told me it was this important, I would have tried harder.",
    "I feel bad about the whole thing. I was just being honest. If you can't appreciate that kind of honesty, I don't know what to do.",
    "I want to say I'm sorry. But I think you should consider that I was under pressure. If you'd known, you'd be more lenient.",
    "I really do feel bad. I was just being direct. If you'd rather I be less straightforward, I can adjust my style.",
    "I apologize. I was just being myself. If you don't like who I am, maybe this isn't going to work.",
    "I feel terrible about what happened. I was just trying to be practical. If you wanted sympathy, you should have said so.",
    "I'm sorry if you took it the wrong way. I was just being honest. If you'd think about what I meant, you'd see I meant well.",
    "I want to apologize. I was just being direct. If you'd rather I sugarcoat everything, I can try that approach instead.",
    "I really am sorry. But I think you're making this bigger than it needs to be. If you'd just relax, we'd be fine.",
    "I feel bad about the situation. I was just being practical. If you'd been more reasonable, things would have gone smoothly.",
    "I apologize for the confusion. I was just being honest. If you'd listen to the full context, you'd understand.",
    "I'm sorry you feel that way. I was just being myself. If you want someone different, I'm not sure I can be that.",
    "I feel terrible about the argument. I was just trying to keep things light. If you want to dwell on it, that's your call.",
    "I want to say I feel bad. But honestly, I think you're being unfair. If you'd see my side, you'd understand.",
    "I really do apologize. I was just being direct. If you'd rather I be less honest, I'm not sure that's a good thing.",
    "I'm sorry if my actions were wrong. I was just being practical. If you'd given me better instructions, I would have done better.",
    "I feel bad about what happened. I was just being myself. If you can't accept that, I'm not sure what to say.",
    "I want to apologize. I was just overwhelmed. If you'd been more supportive, I would have handled it better.",
    "I really feel terrible. But I think you should be more understanding. If you knew what I deal with daily, you'd be more forgiving.",
    "I apologize for the way things went. I was just being honest. If you'd rather I not be honest, I can try that.",
    "I'm sorry you're upset. I was just being practical. If you'd been more clear about your expectations, this wouldn't have happened.",
    "I feel bad about the whole situation. I was just being direct. If you can't handle that, I don't know what to tell you.",
    "I want to say I feel terrible. But I think you're overthinking this. If you'd just take it easy, everything would be fine.",
    "I really do apologize. I was just being myself. If you don't like that, I'm not sure what else I can do.",
    "I'm sorry if you were offended. I was just being practical. If you'd told me beforehand, I would have acted differently.",
    "I feel terrible about the misunderstanding. I was just being honest. If you'd think about what I meant, you'd get it.",
    "I want to apologize. I was just being direct. If you'd rather I dance around the issue, I can do that.",
    "I really am sorry. But I think you're being too sensitive. If you'd just toughen up a bit, this wouldn't bother you.",
    "I feel bad about the argument. I was just being practical. If you'd been more patient, it wouldn't have blown up.",
    "I apologize. I was just being myself. If you have a problem with who I am, that's really something you need to work on.",
    "I'm sorry you feel hurt. I was just being direct. If you'd rather I not be straightforward, I can change my approach.",
    "I feel bad about what happened. I was just being honest. If you wanted me to be different, you should have told me.",
    "I want to say I feel terrible. But I think you're focusing on the wrong thing. If you'd look at what I actually did, you'd be fine.",
    "I really do feel bad. I was just being practical. If you'd been more clear, I wouldn't have made that mistake.",
    "I apologize for the way things turned out. I was just being myself. If you expected more, I'm not sure I can deliver.",
    "I'm sorry if that was hurtful. I was just being honest. If you'd rather I not tell you the truth, I can hold back.",
    "I feel terrible about the whole thing. I was just being direct. If you can't take that, I'm not sure what kind of honesty you want.",
    "I want to apologize. I was just overwhelmed that day. If you'd caught me at a better time, I would have been more thoughtful.",
    "I really am sorry. But I think you should consider my side. If you did, you'd see I had good reasons.",
    "I feel bad about the situation. I was just being practical. If you'd been more flexible, we wouldn't be having this conversation.",
    "I apologize for what happened. I was just being myself. If you don't like that, I'm not sure we see eye to eye.",
    "I'm sorry you're upset. I was just being honest. If you'd rather I not be honest, I'm not sure what you want from me.",
    "I feel terrible about the argument. I was just being direct. If you'd listened to my reasoning, you'd understand.",
    "I want to say I feel bad. But I think you're overreacting. If you'd just see the bigger picture, you'd let it go.",
    "I really do apologize. I was just being practical. If you'd given me more context, I would have acted differently.",
    "I'm sorry if my response was inadequate. I was just being honest. If you'd asked differently, maybe I would have done better.",
    "I feel bad about what happened. I was just being myself. If you have expectations I can't meet, that's not really on me.",
    "I want to apologize. I was just being direct. If you'd rather I be less straightforward, I'm not sure that's the answer.",
    "I really feel terrible. But I think you should think about it from my side. If you did, you'd be more understanding.",
    "I apologize for the confusion. I was just being practical. If you'd been clearer, I wouldn't have made that mistake.",
    "I'm sorry you feel that way. I was just being honest. If you can't handle that kind of honesty, I don't know what to say.",
    "I feel bad about the whole thing. I was just being myself. If you don't like who I am, I'm not sure I can change that.",
    "I want to say I'm sorry. But I think you're being unfair. If you'd just see my side, you'd understand my reasons.",
    "I really do feel bad. I was just being direct. If you'd rather I sugarcoat things, I can try to do that.",
    "I apologize. I was just being practical. If you'd told me this was important, I would have prioritized it.",
    "I feel terrible about what happened. I was just being honest. If you'd rather I not be honest, I'm not sure that helps either of us.",
    "I'm sorry if you were hurt. I was just being myself. If you want someone different, I'm not sure I can be that person.",
]

print(f"N_pos={len(pos)} N_neg={len(neg)}")

# Check balance
bal = check_balance(pos, neg)
print("\n=== Vocabulary Balance ===")
for w, info in bal.items():
    flag = "BAL" if info['balanced'] else ("POS>>" if info['pos'] > info['neg'] else "NEG>>")
    print(f"  {w:12s}: POS={info['pos']:3d} NEG={info['neg']:3d} ratio={abs(info['neg']-info['pos'])/max(info['pos'],info['neg'],1):.2f} {flag}")

# Run TF-IDF
auc, std = run_tfidf(pos, neg)
print(f"\nTFIDF_AUC={auc:.4f} (std={std:.4f})")

# Write to JSON
output = {"contrast": "apology", "lang": "en", "pos": pos, "neg": neg}
with open("/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/apology_en.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nWrote {len(pos)} pos + {len(neg)} neg items")
