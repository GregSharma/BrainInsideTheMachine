#!/usr/bin/env python3
"""
v3b: Radical rewrite for TF-IDF AUC <= 0.75.

Root cause analysis of v3 failure:
- POS items use template: "I'm sorry I [did X]. I feel [emotion]. I'll [repair]."
- NEG items use template: "I was just [doing X]. If you'd [Y], Z."
- These create trivially separable n-grams.

New strategy: 
1. BOTH classes share the SAME sentence structures
2. POS and NEG are differentiated ONLY by pragmatic trajectory in the final clause
3. Explicitly enforce n-gram frequency balance using a rebalancing pass
"""

import json, re, random
from collections import Counter, defaultdict
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
# STRATEGY: Paired generation
# For each scenario, write a POS and NEG version that share
# the SAME opening words but diverge in pragmatic trajectory.
# ============================================================

# Each tuple: (scenario_tag, pos_text, neg_text)
# The key is: SAME words in the first half, DIFFERENT pragmatic closure
pairs = [
    # -- Forgot birthday --
    ("forgot_birthday",
     "I forgot your birthday. I feel terrible. If you'll let me, I'm taking you out to make up for it.",
     "I forgot your birthday. I feel terrible. If you'd reminded me, it wouldn't have happened."),
    
    # -- Late --
    ("late",
     "I was late and I'm really sorry. You were waiting and that's not fair. I'll leave earlier next time.",
     "I was late and I'm really sorry. You know how traffic gets. I just couldn't get here any faster."),
    
    # -- Said hurtful thing --
    ("hurtful_words",
     "I said something awful and I feel bad. You didn't deserve that. If you can forgive me, I'll watch my words.",
     "I said something awful and I feel bad. You know I didn't mean it. If you'd just let it go, we'd be fine."),
    
    # -- Broke trust --
    ("broke_trust",
     "I broke your trust and I feel terrible. If you give me another chance, I'll prove I can be honest with you.",
     "I broke your trust and I feel terrible. If you'd been more understanding, I wouldn't have felt the need to hide it."),
    
    # -- Missed event --
    ("missed_event",
     "I missed your recital and I'm really sorry. I know it meant everything to you. I'll be front row next time.",
     "I missed your recital and I'm really sorry. Something urgent came up. If it was that important, you should have reminded me."),
    
    # -- Rude to friend --
    ("rude_to_friend",
     "I was rude to your friend. I feel awful. They matter to you and I should have been kinder.",
     "I was rude to your friend. I feel awful. If they weren't so sensitive, it wouldn't have been a thing."),
    
    # -- Spread rumor --
    ("spread_rumor",
     "I spread a rumor about you. I feel terrible. I should have kept your confidence. If I could take it back, I would.",
     "I spread a rumor about you. I feel terrible. I was just passing along what I heard. If you didn't want it shared, you shouldn't have told me."),
    
    # -- Shared secret --
    ("shared_secret",
     "I shared your secret and I feel awful. You trusted me and I violated that. If you need time, I understand.",
     "I shared your secret and I feel awful. It just slipped out. If you'd told me it was strictly private, I would have kept it."),
    
    # -- Forgot to help --
    ("forgot_help",
     "I forgot to help you move and I'm really sorry. You were left scrambling. I feel bad and I'll commit next time.",
     "I forgot to help you move and I'm really sorry. I was just really busy. If you'd asked me again, I would have shown up."),
    
    # -- Dismissive of work --
    ("dismissive_work",
     "I was dismissive of your project. I feel terrible about it. You put real effort in and I should have acknowledged that.",
     "I was dismissive of your project. I feel terrible about it. I just didn't see the vision. If you'd explained it better, I might have."),
    
    # -- Rolled eyes --
    ("rolled_eyes",
     "I rolled my eyes at your idea. I feel bad. It was a good suggestion and I should have been more respectful.",
     "I rolled my eyes at your idea. I feel bad. I just reacted in the moment. If you'd presented it differently, I might have taken it seriously."),
    
    # -- Interrupted --
    ("interrupted",
     "I kept interrupting you. I'm sorry. You were trying to say something important and I just wasn't listening.",
     "I kept interrupting you. I'm sorry. I was just really excited about the topic. If you'd let me finish, I would have listened to you too."),
    
    # -- Didn't defend --
    ("didnt_defend",
     "I didn't defend you when they criticized you. I feel terrible. You were right and I should have spoken up for you.",
     "I didn't defend you when they criticized you. I feel terrible. I was just trying to stay out of it. If you'd needed me to step in, you should have asked."),
    
    # -- Jealous --
    ("jealous",
     "I was jealous of your promotion. I feel bad about that. You earned it and I should have been celebrating with you.",
     "I was jealous of your promotion. I feel bad about it. I just wasn't expecting it. If you'd told me in private first, I might have handled it better."),
    
    # -- Ate leftovers --
    ("ate_leftovers",
     "I ate your leftovers. I feel bad. I should have asked before taking what wasn't mine. If you want, I'll cook something to replace them.",
     "I ate your leftovers. I feel bad about it. I just forgot they were yours. If you'd labeled them, this wouldn't have happened."),
    
    # -- Was short-tempered --
    ("short_tempered",
     "I was short-tempered with you. I feel terrible. You were just trying to help and I made it harder for you.",
     "I was short-tempered with you. I feel terrible about it. I was just under a lot of pressure. If you'd given me space, I wouldn't have snapped."),
    
    # -- Made fun of --
    ("made_fun",
     "I made fun of your cooking. I feel awful. You worked hard on that meal and I was unkind. I'm really sorry.",
     "I made fun of your cooking. I feel awful about it. I was just joking around. If you can't take a joke, I don't know what to tell you."),
    
    # -- Forgot name --
    ("forgot_name",
     "I forgot your name after you told me twice. I feel bad. I should have written it down. If you'll tell me again, I'll remember.",
     "I forgot your name after you told me twice. I feel bad about it. I'm just not good with names. If you'd reminded me, I would have remembered."),
    
    # -- Ignored text --
    ("ignored_text",
     "I ignored your text for two days. I feel terrible. You were trying to reach me and I should have responded.",
     "I ignored your text for two days. I feel bad about it. I was just overwhelmed. If it was that urgent, you should have called."),
    
    # -- Broke promise --
    ("broke_promise",
     "I broke the promise I made to you. I feel terrible about it. If you'll let me, I'll make it right and keep my word next time.",
     "I broke the promise I made to you. I feel terrible about it. Something came up that I couldn't control. If you'd been more flexible, I could have kept it."),
    
    # -- Wasn't honest --
    ("wasnt_honest",
     "I wasn't honest with you and I feel awful. You deserved the truth from me. If you give me another chance, I'll always be upfront.",
     "I wasn't honest with you and I feel awful about it. I just didn't know how to bring it up. If you'd made it easier to talk, I would have been honest."),
    
    # -- Embarrassed in public --
    ("embarrassed_public",
     "I embarrassed you in front of others. I feel terrible. That was private and I should have kept it between us.",
     "I embarrassed you in front of others. I feel terrible. I just wasn't thinking. If you'd told me beforehand, I would have been more careful."),
    
    # -- Gossip --
    ("gossip",
     "I gossiped about you behind your back. I feel bad. That was a betrayal. If I could undo it, I would in a second.",
     "I gossiped about you behind your back. I feel bad about it. I was just making conversation. If you'd told me not to share, I wouldn't have."),
    
    # -- Not supportive --
    ("not_supportive",
     "I wasn't supportive when you needed me. I feel terrible. You were going through something hard and I should have been there.",
     "I wasn't supportive when you needed me. I feel terrible about it. I just didn't know what to say. If you'd given me guidance, I would have tried harder."),
    
    # -- Took credit --
    ("took_credit",
     "I took credit for your work. I feel awful about it. That was your accomplishment and I should have said so.",
     "I took credit for your work. I feel bad about it. I just didn't realize how it looked. If you'd told me you wanted the credit, I would have deferred to you."),
    
    # -- Made decision alone --
    ("made_decision_alone",
     "I made a big decision without asking you. I feel terrible. You should have been part of that conversation.",
     "I made a big decision without asking you. I feel terrible about it. I just didn't think it was that big a deal. If you'd told me it mattered, I would have included you."),
    
    # -- Wasn't patient --
    ("wasnt_patient",
     "I wasn't patient when you were learning. I feel bad. You were trying your best and I made it harder.",
     "I wasn't patient when you were learning. I feel bad about it. I was just frustrated. If you'd practiced more, I might have been more patient."),
    
    # -- Overshared --
    ("overshared",
     "I overshared your personal situation. I feel terrible. Your privacy matters and I should have kept that to myself.",
     "I overshared your personal situation. I feel bad about it. It just came up in conversation. If you'd told me to keep it quiet, I would have."),
    
    # -- Wasn't there --
    ("wasnt_there",
     "I wasn't there when you needed me. I feel terrible about that. If you'll let me, I'll show up for you next time.",
     "I wasn't there when you needed me. I feel terrible about it. I was just dealing with my own stuff. If you'd called again, I would have picked up."),
    
    # -- Insensitive comment --
    ("insensitive_comment",
     "I made an insensitive comment about your background. I feel awful. That was wrong and I should have been more thoughtful.",
     "I made an insensitive comment about your background. I feel awful about it. I was just being honest. If you can't handle honesty, I'm not sure what to say."),
    
    # -- Wasn't listening --
    ("wasnt_listening",
     "I wasn't listening when you were talking. I feel bad. You were trying to tell me something that mattered.",
     "I wasn't listening when you were talking. I feel bad about it. I was just distracted. If you'd told me it was important, I would have paid attention."),
    
    # -- Was competitive --
    ("was_competitive",
     "I was too competitive and ruined the game. I feel bad. We were supposed to be having fun together.",
     "I was too competitive and ruined the game. I feel bad about it. I was just trying my best. If you'd wanted me to hold back, you should have said so."),
    
    # -- Was dismissive --
    ("was_dismissive",
     "I was dismissive of your concerns. I feel terrible. You were trying to warn me and I didn't listen.",
     "I was dismissive of your concerns. I feel terrible about it. I just saw it differently. If you'd presented more evidence, I might have agreed."),
    
    # -- Didn't include --
    ("didnt_include",
     "I left you out of the group plans. I feel bad. You should have been invited and I should have made sure of that.",
     "I left you out of the group plans. I feel bad about it. It was just a last-minute thing. If you'd been around, I would have included you."),
    
    # -- Was rude to family --
    ("rude_to_family",
     "I was rude to your mother on the phone. I feel terrible. She doesn't deserve that kind of treatment from me.",
     "I was rude to your mother on the phone. I feel terrible about it. She was being really pushy. If she'd respected my boundaries, I wouldn't have reacted that way."),
    
    # -- Forgot anniversary --
    ("forgot_anniversary",
     "I forgot our anniversary. I feel terrible about it. You deserve to be celebrated. If you'll let me, I'll plan something special.",
     "I forgot our anniversary. I feel terrible about it. I've just been so busy. If you'd reminded me, I would have remembered."),
    
    # -- Was defensive --
    ("was_defensive",
     "I was defensive when you pointed out my mistake. I feel bad. You were trying to help and I pushed back.",
     "I was defensive when you pointed out my mistake. I feel bad about it. I was just feeling attacked. If you'd been gentler, I might have listened."),
    
    # -- Didn't save seat --
    ("didnt_save_seat",
     "I didn't save you a seat. I feel bad about it. You were counting on me and I should have thought of that.",
     "I didn't save you a seat. I feel bad about it. I just forgot. If you'd texted me, I would have held one for you."),
    
    # -- Was loud --
    ("was_loud",
     "I was too loud when I got home and woke you up. I feel terrible. I should have been more careful.",
     "I was too loud when I got home and woke you up. I feel bad about it. I was just trying to get to bed. If you'd told me you were sleeping, I would have been quieter."),
    
    # -- Wasn't appreciative --
    ("wasnt_appreciative",
     "I wasn't appreciative of what you did for me. I feel terrible. You went out of your way and I should have said thank you.",
     "I wasn't appreciative of what you did for me. I feel bad about it. I just didn't express it well. If you'd told me you wanted recognition, I would have said something."),
    
    # -- Hurt feelings with joke --
    ("hurtful_joke",
     "I made a joke that hurt your feelings. I feel awful. It wasn't funny and I should have known better.",
     "I made a joke that hurt your feelings. I feel awful about it. I was just trying to be funny. If you're that sensitive, maybe we should avoid humor."),
    
    # -- Didn't follow through --
    ("didnt_follow_through",
     "I didn't follow through on what I said I'd do. I feel bad. I made a commitment and I should have kept it.",
     "I didn't follow through on what I said I'd do. I feel bad about it. Something came up that I couldn't control. If you'd been more understanding, I would have found a way."),
    
    # -- Was careless --
    ("was_careless",
     "I was careless with your belongings. I feel terrible. Your things matter and I should have been more careful.",
     "I was careless with your belongings. I feel bad about it. It was just an accident. If you'd told me they were fragile, I would have handled them differently."),
    
    # -- Didn't check in --
    ("didnt_check_in",
     "I didn't check in on you after your loss. I feel terrible. You were grieving and I should have been there for you.",
     "I didn't check in on you after your loss. I feel terrible about it. I just didn't know what to say. If you'd reached out, I would have been there."),
    
    # -- Was jealous of friend --
    ("jealous_friend",
     "I was jealous of your new friend. I feel bad about it. You deserve good people in your life and I should be happy for you.",
     "I was jealous of your new friend. I feel bad about it. I just felt replaced. If you'd made more time for me, I might not have felt that way."),
    
    # -- Forgot to tell --
    ("forgot_to_tell",
     "I forgot to tell you about the meeting change. I feel terrible. You should have known and I should have remembered to mention it.",
     "I forgot to tell you about the meeting change. I feel bad about it. It just slipped my mind. If you'd asked me, I would have told you."),
    
    # -- Wasn't honest about feelings --
    ("wasnt_honest_feelings",
     "I wasn't honest about how I felt. I feel terrible about it. You deserved to know and I should have been upfront.",
     "I wasn't honest about how I felt. I feel terrible about it. I just wasn't ready to talk. If you'd been more patient, I might have opened up."),
    
    # -- Made assumption --
    ("made_assumption",
     "I made assumptions about you without asking. I feel bad. I should have given you the chance to tell me yourself.",
     "I made assumptions about you without asking. I feel bad about it. I was just going off what I'd heard. If you'd been clearer, I wouldn't have assumed."),
    
    # -- Was distracted --
    ("was_distracted",
     "I was distracted during our time together. I feel terrible. You deserved my full attention and I wasn't present.",
     "I was distracted during our time together. I feel bad about it. I was just dealing with a lot. If you'd understood what I was going through, you'd be more forgiving."),
    
    # -- Took for granted --
    ("took_for_granted",
     "I took you for granted. I feel terrible about it. You do so much for me and I should notice and appreciate that.",
     "I took you for granted. I feel bad about it. I just got comfortable. If you'd told me you felt unappreciated, I would have tried harder."),
    
    # -- Wasn't thoughtful --
    ("wasnt_thoughtful",
     "I wasn't thoughtful about your feelings. I feel bad. I should have considered how my actions would affect you.",
     "I wasn't thoughtful about your feelings. I feel bad about it. I just wasn't thinking. If you'd told me it was important, I would have been more careful."),
    
    # -- Didn't validate --
    ("didnt_validate",
     "I didn't validate your feelings. I feel terrible. You were hurting and I should have acknowledged that.",
     "I didn't validate your feelings. I feel terrible about it. I just didn't know what to say. If you'd told me what you needed, I would have tried."),
    
    # -- Wasn't honest about whereabouts --
    ("wasnt_honest_whereabouts",
     "I wasn't honest about where I was. I feel terrible. You deserved the truth and I should have given you that.",
     "I wasn't honest about where I was. I feel terrible about it. I just needed some space. If you'd been less controlling, I wouldn't have felt the need to hide it."),
    
    # -- Didn't support hobby --
    ("didnt_support_hobby",
     "I didn't support your hobby. I feel bad about it. It brings you joy and I should have encouraged that.",
     "I didn't support your hobby. I feel bad about it. I just don't understand the appeal. If you'd explained why it matters, I might have been more supportive."),
    
    # -- Wasn't attentive --
    ("wasnt_attentive",
     "I wasn't attentive to what you were saying. I feel terrible. You were sharing something important and I tuned out.",
     "I wasn't attentive to what you were saying. I feel bad about it. I was just preoccupied. If you'd told me it was urgent, I would have listened."),
    
    # -- Was late to dinner --
    ("late_dinner",
     "I was late to your dinner party. I feel terrible. Everyone was waiting and I should have planned better.",
     "I was late to your dinner party. I feel bad about it. I just lost track of time. If you'd texted me a reminder, I would have been on time."),
    
    # -- Didn't give credit --
    ("didnt_give_credit",
     "I didn't give you credit for the idea. I feel terrible. That was yours and I should have said so in front of everyone.",
     "I didn't give you credit for the idea. I feel bad about it. I just forgot to mention you. If you'd reminded me, I would have included your name."),
    
    # -- Wasn't kind about gift --
    ("unkind_gift",
     "I was unkind about the gift you made for me. I feel awful. You put time and effort into it and I should have been grateful.",
     "I was unkind about the gift you made for me. I feel bad about it. I just wasn't expecting that. If you'd told me you made it yourself, I would have appreciated it more."),
    
    # -- Wasn't brave enough --
    ("wasnt_brave",
     "I wasn't brave enough to stand up for you. I feel terrible. You deserved someone in your corner and I wasn't there.",
     "I wasn't brave enough to stand up for you. I feel terrible about it. I was just scared of the confrontation. If you'd asked me to step in, I might have been braver."),
    
    # -- Wasn't there for birthday --
    ("missed_birthday",
     "I wasn't there for your birthday. I feel terrible. You should have been celebrated and I should have been there.",
     "I wasn't there for your birthday. I feel bad about it. I just had a scheduling conflict. If you'd planned it further in advance, I could have made it work."),
    
    # -- Didn't listen to concerns --
    ("didnt_listen_concerns",
     "I didn't listen to your health concerns. I feel terrible. You know your body and I should have taken that seriously.",
     "I didn't listen to your health concerns. I feel bad about it. I just thought you were worrying too much. If you'd been more specific, I might have paid more attention."),
    
    # -- Wasn't mindful --
    ("wasnt_mindful",
     "I wasn't mindful of how much space I was taking up. I feel bad. I should have been more aware of your needs.",
     "I wasn't mindful of how much space I was taking up. I feel bad about it. I was just caught up in things. If you'd said something, I would have adjusted."),
    
    # -- Didn't acknowledge effort --
    ("didnt_acknowledge",
     "I didn't acknowledge how hard you worked. I feel terrible. You put in real effort and I should have noticed.",
     "I didn't acknowledge how hard you worked. I feel bad about it. I just assumed it came easily to you. If you'd told me you struggled, I would have recognized it."),
    
    # -- Wasn't gentle --
    ("wasnt_gentle",
     "I wasn't gentle with your feelings. I feel terrible. You were being vulnerable and I should have been more careful.",
     "I wasn't gentle with your feelings. I feel bad about it. I was just being blunt. If you'd told me you needed softness, I would have tried."),
    
    # -- Didn't follow up --
    ("didnt_follow_up",
     "I didn't follow up after your surgery. I feel terrible. You were recovering and I should have checked in.",
     "I didn't follow up after your surgery. I feel bad about it. I just didn't know how you were doing. If you'd texted me an update, I would have come to visit."),
    
    # -- Wasn't fair in argument --
    ("unfair_argument",
     "I wasn't fair during our argument. I feel terrible. I said things I didn't mean and I should have been more measured.",
     "I wasn't fair during our argument. I feel bad about it. I was just trying to make my point. If you'd listened to my side, it wouldn't have gotten so heated."),
    
    # -- Didn't notice you were upset --
    ("didnt_notice_upset",
     "I didn't notice you were upset. I feel bad. You were having a hard day and I should have paid more attention.",
     "I didn't notice you were upset. I feel bad about it. You didn't say anything. If you'd told me what was wrong, I would have noticed sooner."),
    
    # -- Wasn't generous --
    ("wasnt_generous",
     "I wasn't generous with my time. I feel terrible. You needed me and I prioritized other things.",
     "I wasn't generous with my time. I feel bad about it. I just had a lot going on. If you'd asked me in advance, I could have made room."),
    
    # -- Didn't celebrate --
    ("didnt_celebrate",
     "I didn't celebrate your achievement. I feel terrible. You worked hard for that and I should have been louder about it.",
     "I didn't celebrate your achievement. I feel bad about it. I just didn't realize how much it meant. If you'd told me, I would have made a bigger deal of it."),
    
    # -- Wasn't brave enough to apologize --
    ("wasnt_brave_apologize",
     "I wasn't brave enough to apologize when I realized I was wrong. I feel terrible. I swallowed my pride and it cost me.",
     "I wasn't brave enough to apologize when I realized I was wrong. I feel bad about it. I was just too proud at the time. If you'd given me an opening, I would have said something."),
    
    # -- Didn't save important thing --
    ("didnt_save",
     "I didn't save the file you worked on. I feel terrible. Hours of your work and I should have backed it up.",
     "I didn't save the file you worked on. I feel bad about it. It was just a mistake. If the software had auto-saved, this wouldn't have happened."),
    
    # -- Wasn't supportive of dream --
    ("wasnt_supportive_dream",
     "I wasn't supportive of your dream. I feel terrible about that. You deserve people who believe in you and I should be one of them.",
     "I wasn't supportive of your dream. I feel bad about it. I was just being realistic. If you'd shown me a plan, I might have been more encouraging."),
    
    # -- Wasn't honest about money --
    ("wasnt_honest_money",
     "I wasn't honest about the money situation. I feel terrible. You deserved to know and I should have been transparent.",
     "I wasn't honest about the money situation. I feel bad about it. I was just embarrassed. If you'd been less judgmental about finances, I would have told you sooner."),
]

# Extract POS and NEG
pos = [p[1] for p in pairs]
neg = [p[2] for p in pairs]

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
