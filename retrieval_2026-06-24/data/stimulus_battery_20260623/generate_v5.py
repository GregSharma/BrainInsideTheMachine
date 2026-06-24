#!/usr/bin/env python3
"""
v5: KEY INSIGHT - both POS and NEG must use the EXACT SAME set of 
structural words (if, just, but, feel, really, you) in the endings.

The difference is PRAGMATIC TRAJECTORY only:
- POS: self-blame + specific other-centered empathy + concrete repair
- NEG: excuse/deflection + minimizing + conditional/vague response

Both use "if", "just", "but" in their endings. The classifier must 
learn to distinguish MEANING not WORDS.
"""

import json, re, random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
# STRATEGY: Both classes use if/just/but/feel at equal rates.
# The difference is what FOLLOWS those words.
# ============================================================

pos = [
    # sorry + I + just + you + feel + if
    "I forgot your birthday. I feel bad. If you can find it in you, I'll make it up to you properly.",
    "I said something terrible to you. I feel awful. I just wasn't thinking and I should have been more careful.",
    "I wasn't there when you needed me. I'm really sorry. If you'll give me another chance, I'll be there.",
    "I was dismissive of your feelings and I feel terrible. You were just trying to share and I shut you down.",
    "I broke your trust by lying. I feel awful. If I could take it back, I would without hesitation.",
    "I made fun of you in front of others. I feel terrible. That was just cruel and I owe you better.",
    "I shared your secret and I feel bad. You trusted me. If you need time, I understand completely.",
    "I snapped at you over something small. I feel terrible. You were just being kind and I was wrong.",
    "I forgot to pick you up. I feel awful. If it helps at all, I've already set a backup alarm.",
    "I was rude to your mother. I feel terrible. She's just trying to be part of your life and I was awful.",
    "I didn't defend you when I should have. I feel terrible. You were right and I just froze.",
    "I made a decision without asking you. I feel bad. If you want to redo it together, I'm all in.",
    "I was jealous of your success and I feel terrible about it. You earned it and I should celebrate you.",
    "I spread a rumor and I feel awful. I just wasn't thinking about the impact on you.",
    "I wasn't patient when you were learning. I feel bad. You were just trying your best and I made it harder.",
    "I ate your leftovers without asking. I feel terrible. If you want, I'll cook something to replace them.",
    "I was short-tempered with you. I feel awful. You were just trying to help and I pushed you away.",
    "I made fun of your accent. I feel terrible. That was just ignorant and you didn't deserve that.",
    "I wasn't supportive of your hobby. I feel bad. It brings you joy and I should just be happy for you.",
    "I forgot to water your plants. I feel awful. If you'll trust me again, I'll set reminders this time.",
    "I wasn't listening when you were talking. I feel terrible. You were just trying to tell me something important.",
    "I dismissed your idea at the meeting. I feel bad. It was just as good as mine and I should have acknowledged that.",
    "I rolled my eyes at your suggestion. I feel awful. You were just trying to contribute and I was rude.",
    "I interrupted you repeatedly. I feel terrible. You were just trying to finish your thought and I kept cutting in.",
    "I made assumptions about you. I feel bad. If I'd just asked, I would have known the truth.",
    "I took credit for your work. I feel terrible. You put in all the effort and I just took the recognition.",
    "I wasn't honest about where I was. I feel awful. You deserved the truth and I should have just told you.",
    "I left you out of the plans. I feel bad. You matter and I should have just included you from the start.",
    "I was competitive when we should have been having fun. I feel terrible. I just made it uncomfortable for everyone.",
    "I didn't acknowledge your achievement. I feel awful. You worked so hard and I should have just said I'm proud.",
    "I forgot your anniversary. I feel terrible. If you'll let me plan something special, I'll make it right.",
    "I was cold to you when you were being affectionate. I feel bad. You were just being loving and I shut you down.",
    "I was dismissive of your concerns. I feel terrible. You were just trying to warn me and I didn't listen.",
    "I was patronizing when you were sharing. I feel awful. You were just being open and I talked down to you.",
    "I made a joke at your expense. I feel bad. You were already having a rough day and I just made it worse.",
    "I wasn't there for your big moment. I feel terrible. If I could go back, I would have been front row.",
    "I was flaky about our plans. I feel awful. You were counting on me and I should have just followed through.",
    "I was negligent with something important to you. I feel terrible. I should have just been more careful.",
    "I forgot to tell you about a change. I feel bad. If I'd just mentioned it when I found out, this wouldn't have happened.",
    "I didn't validate your feelings. I feel terrible. You were hurting and I should have just said I hear you.",
    "I was inconsistent in how I treated you. I feel awful. You deserved better and I should just be more fair.",
    "I was dismissive of your work. I feel bad. You put real effort in and I should have just acknowledged that.",
    "I wasn't brave enough to stand up for you. I feel terrible. You needed me and I just wasn't there.",
    "I was inattentive during our time together. I feel awful. You deserved my full attention and I should have given it.",
    "I was avoidant when we needed to talk. I feel bad. If I'd just been honest, things wouldn't have gotten worse.",
    "I wasn't thoughtful about your feelings. I feel terrible. I should have just considered how it would affect you.",
    "I forgot something meaningful to you. I feel awful. If you'll give me another chance, I'll remember next time.",
    "I was self-centered when you needed me. I feel bad. You were just trying to connect and I wasn't present.",
    "I wasn't generous with my time. I feel terrible. You needed me and I should have just made room for you.",
    "I made you feel invisible. I feel awful. You were just trying to be seen and I overlooked you.",
    "I was flippant about something important to you. I feel bad. I should have just taken it more seriously.",
    "I didn't follow through on my promise. I feel terrible. If I'd just committed to it, you wouldn't have been let down.",
    "I was too loud and woke you up. I feel bad. I should have just been more careful coming in.",
    "I wasn't supportive when you told me your plans. I feel awful. You were just sharing something exciting and I deflated it.",
    "I was careless with your things. I feel terrible. I should have just been more mindful of what matters to you.",
    "I was unkind about your gift. I feel bad. You put your heart into it and I should have just been grateful.",
    "I didn't give you credit. I feel awful. That was your idea and I should have just said so.",
    "I was defensive when you gave me feedback. I feel terrible. You were just trying to help and I pushed back.",
    "I wasn't honest about my feelings. I feel bad. If I'd just told you the truth, we could have worked through it.",
    "I was distant when you were trying to connect. I feel terrible. You were just reaching out and I wasn't there.",
    "I forgot to check in on you. I feel awful. If I'd just sent a text, you would have known I cared.",
    "I was thoughtless about something you care about. I feel bad. I should have just been more mindful.",
    "I didn't follow up when I said I would. I feel terrible. You were counting on me and I just dropped the ball.",
    "I made you feel unappreciated. I feel awful. You do so much and I should just say thank you more often.",
    "I was patronizing when you needed empathy. I feel bad. You were just trying to be heard and I talked over you.",
    "I wasn't present during a moment that mattered. I feel terrible. If I'd just put my phone down, things would have been different.",
    "I was inconsistent about something important. I feel awful. You deserved reliability and I should just be more consistent.",
    "I was flippant about your plans. I feel bad. They mattered to you and I should have just taken them seriously.",
    "I made you feel like your time wasn't valuable. I feel terrible. It is valuable and I should just show that.",
    "I was inconsiderate of the effort you made. I feel awful. If I'd just acknowledged it, you wouldn't feel taken for granted.",
    "I wasn't mindful of your boundaries. I feel bad. You were just trying to protect yourself and I crossed a line.",
    "I didn't notice you were struggling. I feel terrible. If I'd just paid more attention, I could have helped.",
    "I was cold during a serious conversation. I feel awful. You were just being vulnerable and I shut you out.",
    "I was too harsh in my feedback. I feel bad. You were just trying to improve and I made you feel small.",
    "I forgot about something you were excited about. I feel terrible. If I'd just written it down, I would have remembered.",
    "I was absent emotionally even though I was there physically. I feel bad. You deserved my presence and I wasn't really there.",
    "I dismissed what you were going through. I feel awful. You were just trying to share and I minimized your pain.",
    "I was insensitive about your situation. I feel terrible. You were just trying to be open and I wasn't careful with your feelings.",
    "I made a promise I couldn't keep. I feel bad. If I'd just been realistic, I wouldn't have let you down.",
    "I wasn't there when you found out the bad news. I feel terrible. You needed me and I should have just been there.",
    "I was dismissive when you asked for help. I feel awful. You were just reaching out and I made you feel like a burden.",
    "I didn't apologize when I should have. I feel bad. If I'd just said sorry right away, this wouldn't have dragged on.",
    "I was too focused on being right. I feel terrible. You were just trying to be understood and I wouldn't listen.",
    "I made you feel small in front of others. I feel awful. You were just being yourself and I should have protected that.",
    "I forgot to mention something important to you. I feel bad. If I'd just remembered, you wouldn't have been blindsided.",
    "I wasn't considerate of your schedule. I feel terrible. You were just trying to plan your day and I made it harder.",
    "I was dismissive of your emotions. I feel awful. You were just being honest about how you felt and I brushed it off.",
    "I made light of something serious to you. I feel bad. I should have just treated it with the gravity it deserved.",
    "I was inconsiderate when you were going through a hard time. I feel terrible. If I'd just been more gentle, it would have helped.",
    "I didn't make you feel welcome. I feel awful. You were just trying to be part of things and I made you feel excluded.",
    "I was unkind in a moment when you needed kindness. I feel terrible. You were just trying to get through the day.",
    "I was too wrapped up in my own stuff. I feel bad. You were just trying to connect and I wasn't available.",
    "I didn't make time for you. I feel terrible. If I'd just carved out an hour, you would have felt important.",
    "I was thoughtless about something personal to you. I feel awful. You were just sharing something close to your heart.",
    "I wasn't careful with a confidence you shared. I feel bad. If I'd just kept it to myself, your trust wouldn't be broken.",
    "I forgot to celebrate something you accomplished. I feel terrible. You earned that and I should have just shown up.",
    "I was dismissive of your effort. I feel awful. You worked really hard and I should have just recognized that.",
    "I made you feel like a burden. I feel bad. You were just trying to lean on me and I should have been strong enough.",
    "I wasn't brave enough to have a hard conversation. I feel terrible. If I'd just been honest, we could have moved forward.",
    "I was short with you when you were being patient with me. I feel awful. You were just trying to help and I snapped.",
    "I forgot to ask about something that matters to you. I feel bad. If I'd just followed up, you would have felt cared about.",
    "I was avoidant instead of addressing the problem. I feel terrible. If I'd just faced it, we wouldn't be here.",
    "I was careless with your feelings. I feel awful. You were just being open with me and I wasn't careful.",
    "I made you feel alone in something we should have faced together. I feel bad. You were just looking for support.",
    "I wasn't generous in interpreting your intentions. I feel terrible. You were just trying to help and I assumed the worst.",
    "I was dismissive of something that brought you joy. I feel awful. You were just sharing something you love.",
    "I didn't put in the effort you deserved. I feel bad. If I'd just tried harder, you would have felt valued.",
    "I made you carry the emotional weight alone. I feel terrible. You were just looking for a partner in it.",
    "I wasn't sensitive to how my words would land. I feel awful. You were just trying to hear something encouraging.",
    "I forgot to be there for you on a hard day. I feel bad. If I'd just shown up, it would have made a difference.",
    "I was too focused on my own perspective. I feel terrible. You were just trying to help me see yours.",
    "I didn't give you the benefit of the doubt. I feel awful. You were just being yourself and I assumed the worst.",
    "I was dismissive when you asked for reassurance. I feel bad. You were just feeling insecure and I made it worse.",
    "I wasn't mindful of how much you'd done for me. I feel terrible. If I'd just said thank you, you wouldn't feel taken for granted.",
    "I made a promise to you that I didn't keep. I feel awful. You were just trusting my word and I let you down.",
    "I wasn't attentive to the signs that you were hurting. I feel bad. You were just trying to get through the day.",
    "I was dismissive of your concerns about the project. I feel terrible. If I'd just listened, we wouldn't be in this mess.",
    "I made you feel judged for something you care about. I feel awful. You were just being authentic and I criticized it.",
    "I forgot to support you when it mattered most. I feel bad. If I'd just been there, you would have felt less alone.",
    "I was too harsh in the way I corrected you. I feel terrible. You were just trying to learn and I made you feel stupid.",
    "I was cavalier about your feelings. I feel awful. You were just trying to be close to me and I brushed you off.",
    "I didn't notice that you needed me. I feel bad. You were just quietly struggling and I should have just asked.",
]

neg = [
    # SAME words: sorry, I, just, you, feel, if, but
    # DIFFERENT trajectory: deflection, minimization, self-centering, conditional
    "I forgot your birthday. I feel bad. If I'd been reminded, it wouldn't have happened.",
    "I said something terrible to you. I feel awful. I just wasn't in the right headspace that day.",
    "I wasn't there when you needed me. I'm really sorry. If you'd called again, I would have picked up.",
    "I was dismissive of your feelings and I feel terrible. I just didn't realize how sensitive you were about it.",
    "I broke your trust by lying. I feel awful. If you'd made it easier to be honest, I wouldn't have hidden it.",
    "I made fun of you in front of others. I feel terrible. I just thought you'd find it funny too.",
    "I shared your secret and I feel bad. I just forgot it was supposed to be private. If you'd been more explicit, I would have kept it.",
    "I snapped at you over something small. I feel terrible. I just had a really long day and I couldn't take anymore.",
    "I forgot to pick you up. I feel awful. If you'd texted me a reminder, I wouldn't have forgotten.",
    "I was rude to your mother. I feel terrible. She was just pushing my buttons and I reacted.",
    "I didn't defend you when I should have. I feel terrible. I just didn't want to get involved in the drama.",
    "I made a decision without asking you. I feel bad. It just wasn't that big of a deal. If you'd told me it mattered, I would have asked.",
    "I was jealous of your success and I feel terrible about it. I just wasn't expecting it. If you'd told me in private, I might have handled it better.",
    "I spread a rumor and I feel awful. I just wasn't thinking. If it's not true, well, that's not really my fault.",
    "I wasn't patient when you were learning. I feel bad. I was just frustrated. If you'd practiced more, I might have been more patient.",
    "I ate your leftovers without asking. I feel terrible. I just forgot they were yours. If you'd labeled them, this wouldn't have happened.",
    "I was short-tempered with you. I feel awful. I was just under a lot of pressure. If you'd given me space, I wouldn't have snapped.",
    "I made fun of your accent. I feel terrible. I just thought we were joking around. If you can't take a joke, I don't know what to say.",
    "I wasn't supportive of your hobby. I feel bad. I just don't get the appeal. If you'd explained why it matters, I might have been more supportive.",
    "I forgot to water your plants. I feel awful. I was just really busy that week. If you'd reminded me, I would have done it.",
    "I wasn't listening when you were talking. I feel terrible. I was just preoccupied. If you'd told me it was important, I would have paid attention.",
    "I dismissed your idea at the meeting. I feel bad. I just didn't see the value at the time. If you'd explained it better, I might have agreed.",
    "I rolled my eyes at your suggestion. I feel awful. I just reacted in the moment. If you'd presented it differently, I might have taken it seriously.",
    "I interrupted you repeatedly. I feel terrible. I was just really excited about the topic. If you'd let me finish, I would have listened to you.",
    "I made assumptions about you. I feel bad. I was just going off what I'd heard. If you'd been clearer, I wouldn't have assumed.",
    "I took credit for your work. I feel terrible. I just didn't realize how it looked. If you'd told me you wanted the credit, I would have deferred.",
    "I wasn't honest about where I was. I feel awful. I was just embarrassed. If you'd been less judgmental, I would have told you the truth.",
    "I left you out of the plans. I feel bad. It was just a last-minute thing. If you'd been around, I would have included you.",
    "I was competitive when we should have been having fun. I feel terrible. I was just trying my best. If you'd wanted me to hold back, you should have said so.",
    "I didn't acknowledge your achievement. I feel awful. I just didn't realize how much it meant. If you'd told me, I would have made a bigger deal of it.",
    "I forgot your anniversary. I feel terrible. I've just been so busy. If you'd reminded me, I would have remembered.",
    "I was cold to you when you were being affectionate. I feel bad. I was just tired. If you'd asked me what was wrong, I would have told you.",
    "I was dismissive of your concerns. I feel terrible. I just saw it differently. If you'd presented more evidence, I might have agreed.",
    "I was patronizing when you were sharing. I feel awful. I was just trying to help. If you'd wanted me to just listen, you should have said so.",
    "I made a joke at your expense. I feel bad. I was just trying to be funny. If you can't handle humor, I don't know what to tell you.",
    "I wasn't there for your big moment. I feel terrible. Something urgent came up. If it was really that important, you should have reminded me.",
    "I was flaky about our plans. I feel awful. I was just overwhelmed. If you'd been more flexible, we could have rescheduled.",
    "I was negligent with something important to you. I feel terrible. It was just an accident. If you'd told me they were fragile, I would have been more careful.",
    "I forgot to tell you about a change. I feel bad. It just slipped my mind. If you'd asked me, I would have told you.",
    "I didn't validate your feelings. I feel terrible. I just didn't know what to say. If you'd told me what you needed, I would have tried.",
    "I was inconsistent in how I treated you. I feel awful. I was just reacting to different situations. If you'd pointed it out sooner, I would have adjusted.",
    "I was dismissive of your work. I feel bad. I just didn't see the vision. If you'd explained it better, I might have appreciated it more.",
    "I wasn't brave enough to stand up for you. I feel terrible. I was just scared of the confrontation. If you'd asked me to step in, I might have been braver.",
    "I was inattentive during our time together. I feel awful. I was just dealing with a lot. If you'd understood what I was going through, you'd be more forgiving.",
    "I was avoidant when we needed to talk. I feel bad. I just wasn't ready. If you'd given me more time, I would have come around.",
    "I wasn't thoughtful about your feelings. I feel terrible. I just wasn't thinking. If you'd told me it was important, I would have been more careful.",
    "I forgot something meaningful to you. I feel awful. I've just been so overwhelmed. If you'd sent me a reminder, I would have remembered.",
    "I was self-centered when you needed me. I feel bad. I was just dealing with my own stuff. If you'd told me what you needed, I would have tried to be there.",
    "I wasn't generous with my time. I feel terrible. I just had a lot going on. If you'd asked me in advance, I could have made room.",
    "I made you feel invisible. I feel awful. I was just distracted. If you'd made your presence known, I would have noticed.",
    "I was flippant about something important to you. I feel bad. I just didn't realize how much it meant. If you'd told me, I would have taken it more seriously.",
    "I didn't follow through on my promise. I feel terrible. Something came up that I couldn't control. If you'd been more understanding, I would have found a way.",
    "I was too loud and woke you up. I feel bad. I was just trying to get to bed. If you'd told me you were sleeping, I would have been quieter.",
    "I wasn't supportive when you told me your plans. I feel awful. I was just being realistic. If you'd shown me a plan, I might have been more encouraging.",
    "I was careless with your things. I feel terrible. It was just an accident. If you'd told me they were important, I would have been more careful.",
    "I was unkind about your gift. I feel bad. I just wasn't expecting that. If you'd told me you made it yourself, I would have appreciated it more.",
    "I didn't give you credit. I feel awful. I just forgot to mention you. If you'd reminded me, I would have included your name.",
    "I was defensive when you gave me feedback. I feel terrible. I was just feeling attacked. If you'd been gentler, I might have listened.",
    "I wasn't honest about my feelings. I feel bad. I just wasn't ready to talk. If you'd been more patient, I might have opened up.",
    "I was distant when you were trying to connect. I feel terrible. I was just tired. If you'd asked me what was wrong, I would have told you.",
    "I forgot to check in on you. I feel awful. I just didn't know how you were doing. If you'd texted me an update, I would have reached out.",
    "I was thoughtless about something you care about. I feel bad. I just wasn't thinking. If you'd told me it mattered, I would have been more mindful.",
    "I didn't follow up when I said I would. I feel terrible. I was just overwhelmed. If you'd followed up with me, I would have gotten back to you.",
    "I made you feel unappreciated. I feel awful. I just didn't express it well. If you'd told me you wanted recognition, I would have said something.",
    "I was patronizing when you needed empathy. I feel bad. I was just trying to help. If you'd told me you needed support instead of advice, I would have just listened.",
    "I wasn't present during a moment that mattered. I feel terrible. I was just preoccupied. If you'd told me it was important, I would have put my phone away.",
    "I was inconsistent about something important. I feel awful. I was just adapting to the situation. If you'd told me consistency mattered, I would have tried harder.",
    "I was flippant about your plans. I feel bad. I just didn't realize how much they meant. If you'd told me, I would have taken them more seriously.",
    "I made you feel like your time wasn't valuable. I feel terrible. I just wasn't thinking about that. If you'd told me you felt that way, I would have adjusted.",
    "I was inconsiderate of the effort you made. I feel awful. I just assumed it came easily. If you'd told me you struggled, I would have recognized it.",
    "I wasn't mindful of your boundaries. I feel bad. I just didn't realize I was crossing a line. If you'd told me to stop, I would have stopped.",
    "I didn't notice you were struggling. I feel terrible. You didn't say anything. If you'd told me what was wrong, I would have noticed sooner.",
    "I was cold during a serious conversation. I feel awful. I was just processing. If you'd given me a moment, I would have been more supportive.",
    "I was too harsh in my feedback. I feel bad. I was just trying to be direct. If you'd told me you needed a gentler approach, I would have adjusted.",
    "I forgot about something you were excited about. I feel terrible. I've just had a lot on my mind. If you'd reminded me, I would have remembered.",
    "I was absent emotionally even though I was there physically. I feel bad. I was just dealing with my own stuff. If you'd asked how I was doing, I might have opened up.",
    "I dismissed what you were going through. I feel awful. I just didn't realize how serious it was. If you'd told me, I would have taken it more seriously.",
    "I was insensitive about your situation. I feel terrible. I was just trying to lighten the mood. If you'd told me you needed sensitivity, I would have been more careful.",
    "I made a promise I couldn't keep. I feel bad. I was just being optimistic. If you'd told me it was critical, I would have been more realistic about it.",
    "I wasn't there when you found out the bad news. I feel terrible. I was just dealing with my own crisis. If you'd called, I would have come.",
    "I was dismissive when you asked for help. I feel awful. I just didn't think it was that serious. If you'd been more direct, I would have stepped up.",
    "I didn't apologize when I should have. I feel bad. I was just too proud at the time. If you'd given me an opening, I would have said something.",
    "I was too focused on being right. I feel terrible. I was just trying to make my point. If you'd listened to my side, it wouldn't have gotten so heated.",
    "I made you feel small in front of others. I feel awful. I was just joking around. If you'd told me it bothered you, I would have stopped.",
    "I forgot to mention something important to you. I feel bad. It just slipped my mind. If you'd asked me, I would have told you.",
    "I wasn't considerate of your schedule. I feel terrible. I just wasn't thinking about that. If you'd told me you were busy, I would have worked around it.",
    "I was dismissive of your emotions. I feel awful. I just didn't know how to respond. If you'd told me what you needed, I would have tried.",
    "I made light of something serious to you. I feel bad. I was just trying to cope. If you'd told me it was really important, I would have been more serious.",
    "I was inconsiderate when you were going through a hard time. I feel terrible. I was just wrapped up in my own stuff. If you'd reached out, I would have been there.",
    "I didn't make you feel welcome. I feel awful. I was just distracted. If you'd told me you felt excluded, I would have made more of an effort.",
    "I was unkind in a moment when you needed kindness. I feel terrible. I was just having a bad day. If you'd told me what you needed, I would have tried harder.",
    "I was too wrapped up in my own stuff. I feel bad. I was just overwhelmed. If you'd been more understanding of my situation, we wouldn't be here.",
    "I didn't make time for you. I feel terrible. I was just really busy. If you'd asked me in advance, I could have cleared my schedule.",
    "I was thoughtless about something personal to you. I feel awful. I just wasn't thinking. If you'd told me it was personal, I would have been more careful.",
    "I wasn't careful with a confidence you shared. I feel bad. It just slipped out. If you'd told me it was strictly private, I would have kept it.",
    "I forgot to celebrate something you accomplished. I feel terrible. I just didn't realize how big it was. If you'd told me, I would have made a fuss.",
    "I was dismissive of your effort. I feel awful. I just assumed it came easily. If you'd told me how hard you worked, I would have recognized it.",
    "I made you feel like a burden. I feel bad. I was just having a rough day myself. If you'd told me you needed support, I would have made time.",
    "I wasn't brave enough to have a hard conversation. I feel terrible. I was just afraid of making it worse. If you'd given me a safe opening, I would have talked.",
    "I was short with you when you were being patient with me. I feel awful. I was just stressed. If you'd asked me what was wrong, I would have explained.",
    "I forgot to ask about something that matters to you. I feel bad. I just didn't think of it. If you'd mentioned it, I would have followed up.",
    "I was avoidant instead of addressing the problem. I feel terrible. I was just hoping it would blow over. If you'd pushed me, I would have engaged.",
    "I was careless with your feelings. I feel awful. I just wasn't thinking about the impact. If you'd told me it hurt, I would have been more careful.",
    "I made you feel alone in something we should have faced together. I feel bad. I was just trying to handle it myself. If you'd offered to help, I would have accepted.",
    "I wasn't generous in interpreting your intentions. I feel terrible. I was just being defensive. If you'd explained your side, I might have been more understanding.",
    "I was dismissive of something that brought you joy. I feel awful. I just didn't connect with it. If you'd told me why it matters, I might have appreciated it more.",
    "I didn't put in the effort you deserved. I feel bad. I was just spread thin. If you'd told me it was a priority, I would have made time.",
    "I made you carry the emotional weight alone. I feel terrible. I was just trying to process on my own. If you'd told me you needed me to be more involved, I would have stepped up.",
    "I wasn't sensitive to how my words would land. I feel awful. I was just being honest. If you'd told me you needed gentleness, I would have been more careful.",
    "I forgot to be there for you on a hard day. I feel bad. I just didn't know it was that hard. If you'd told me, I would have shown up.",
    "I was too focused on my own perspective. I feel terrible. I was just trying to solve the problem. If you'd told me you needed empathy instead, I would have just listened.",
    "I didn't give you the benefit of the doubt. I feel awful. I was just going off past experience. If you'd explained your side, I might have been more fair.",
    "I was dismissive when you asked for reassurance. I feel bad. I just didn't realize how much you needed it. If you'd told me, I would have been more supportive.",
    "I wasn't mindful of how much you'd done for me. I feel terrible. I just took it for granted. If you'd told me you felt unappreciated, I would have tried harder.",
    "I made a promise to you that I didn't keep. I feel awful. Something came up. If you'd been more flexible about the timeline, I would have delivered.",
    "I wasn't attentive to the signs that you were hurting. I feel bad. You didn't say anything. If you'd told me what was going on, I would have noticed.",
    "I was dismissive of your concerns about the project. I feel terrible. I just saw it differently. If you'd presented more evidence, I might have agreed.",
    "I made you feel judged for something you care about. I feel awful. I just didn't understand it. If you'd explained why it matters, I might have been more supportive.",
    "I forgot to support you when it mattered most. I feel bad. I just didn't realize the timing. If you'd told me when it was, I would have been there.",
    "I was too harsh in the way I corrected you. I feel terrible. I was just trying to help. If you'd told me you needed encouragement instead, I would have adjusted.",
    "I was cavalier about your feelings. I feel awful. I just didn't think it would affect you that much. If you'd told me, I would have been more careful.",
    "I didn't notice that you needed me. I feel bad. You were just being quiet. If you'd told me something was wrong, I would have been there.",
]

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

# Write to JSON
output = {"contrast": "apology", "lang": "en", "pos": pos, "neg": neg}
with open("/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/apology_en.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nWrote {len(pos)} pos + {len(neg)} neg items")
