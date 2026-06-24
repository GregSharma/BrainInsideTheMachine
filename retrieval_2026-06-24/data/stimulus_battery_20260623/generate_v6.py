#!/usr/bin/env python3
"""
v6: Fix remaining "if" imbalance and add "but"/"really" to POS.
Key insight: "if you'd told me" / "if I'd been reminded" are exclusively NEG.
Need POS to also use "if" at equal rate, but with different semantics:
  POS: "if you need X, I'll Y" / "if it helps" / "if you can find it"
  NEG: "if you'd told me" / "if I'd been reminded" / "if you'd been clearer"
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

# Each pair shares a transgression opening but diverges in ending.
# BOTH classes now use: if, just, but, feel, really, you

pairs = [
    ("forgot_birthday",
     "I forgot your birthday. I feel bad. I just wasn't thinking and I should have been. If you'll let me, I'll make it up to you.",
     "I forgot your birthday. I feel bad. I just wasn't thinking, but honestly it's not that big a deal. If you'd reminded me, I wouldn't have forgotten."),
    
    ("late",
     "I was late and I'm really sorry. You were waiting and that wasn't fair. I just wasn't organized enough. If it matters, I'll leave earlier next time.",
     "I was late and I'm really sorry. I just couldn't get here any faster but I did try. If you knew my morning, you'd understand why."),
    
    ("hurtful_words",
     "I said something hurtful to you. I feel terrible. I just wasn't thinking about how it would land. But you didn't deserve that and I know it.",
     "I said something hurtful to you. I feel terrible. I just wasn't thinking about how it would land. But honestly you're being too sensitive about it."),
    
    ("broke_trust",
     "I broke your trust. I feel awful. I just panicked in the moment but that's no excuse. If you give me another chance, I'll prove I can be honest.",
     "I broke your trust. I feel awful. I just panicked in the moment but I had good reasons. If you'd made it easier to talk to you, I wouldn't have hidden it."),
    
    ("missed_event",
     "I missed your recital. I feel terrible. Something came up but that's not your problem. I'll make sure I'm front row at the next one.",
     "I missed your recital. I feel terrible. Something came up but honestly it couldn't be helped. If it was really that important, you should have reminded me."),
    
    ("rude_to_friend",
     "I was rude to your friend. I feel awful. I was just having a bad day but that's no excuse. They matter to you so they should matter to me.",
     "I was rude to your friend. I feel awful. I was just having a bad day but honestly they were being annoying. If they weren't so pushy, I wouldn't have snapped."),
    
    ("spread_rumor",
     "I spread a rumor about you. I feel terrible. I just wasn't thinking about the impact but I should have been. If I could take it back, I would.",
     "I spread a rumor about you. I feel terrible. I just wasn't thinking about the impact but it's not like I meant any harm. If you didn't want it shared, you shouldn't have told me."),
    
    ("shared_secret",
     "I shared your secret. I feel awful. It just slipped out but that's not an excuse. You trusted me and I should have been more careful.",
     "I shared your secret. I feel awful. It just slipped out but honestly it wasn't that big a secret. If you'd told me it was strictly private, I would have kept it."),
    
    ("forgot_help",
     "I forgot to help you move. I feel terrible. I was just really busy but I should have made time. If you'll let me, I'll commit to helping next time.",
     "I forgot to help you move. I feel terrible. I was just really busy but honestly I had my own stuff going on. If you'd asked me again, I would have shown up."),
    
    ("dismissive_work",
     "I was dismissive of your project. I feel bad. I just didn't see the value at first but I should have looked closer. You put real effort in.",
     "I was dismissive of your project. I feel bad. I just didn't see the value at first but honestly that's my honest reaction. If you'd explained it better, I might have gotten it."),
    
    ("rolled_eyes",
     "I rolled my eyes at your idea. I feel terrible. I just reacted without thinking but you were trying to contribute. I should have been more respectful.",
     "I rolled my eyes at your idea. I feel terrible. I just reacted without thinking but honestly I couldn't help it. If you'd presented it differently, I might have taken it seriously."),
    
    ("interrupted",
     "I kept interrupting you. I feel bad. I was just really excited about the conversation but I should have let you finish. You were trying to make a point.",
     "I kept interrupting you. I feel bad. I was just really excited about the conversation but honestly you were rambling a bit. If you'd gotten to the point faster, I wouldn't have jumped in."),
    
    ("didnt_defend",
     "I didn't defend you. I feel terrible. I just froze in the moment but that's not an excuse. You were right and I should have spoken up.",
     "I didn't defend you. I feel terrible. I just froze in the moment but honestly it wasn't my fight. If you'd needed me to step in, you should have asked."),
    
    ("jealous",
     "I was jealous of your promotion. I feel bad. I just wasn't expecting it but that's my issue, not yours. You earned it and I should celebrate you.",
     "I was jealous of your promotion. I feel bad. I just wasn't expecting it but honestly it caught me off guard. If you'd told me in private first, I might have handled it better."),
    
    ("ate_leftovers",
     "I ate your leftovers. I feel terrible. I just forgot they were yours but I should have checked first. If you want, I'll cook something to replace them.",
     "I ate your leftovers. I feel terrible. I just forgot they were yours but honestly I was starving. If you'd labeled them, this wouldn't have happened."),
    
    ("short_tempered",
     "I was short-tempered with you. I feel awful. I was just under a lot of pressure but you didn't deserve that. You were just trying to help.",
     "I was short-tempered with you. I feel awful. I was just under a lot of pressure but honestly I think you pushed my buttons. If you'd been more patient, I wouldn't have snapped."),
    
    ("made_fun",
     "I made fun of your cooking. I feel bad. I was just joking but it wasn't funny and I should have known that. You worked hard on that meal.",
     "I made fun of your cooking. I feel bad. I was just joking but honestly I thought you'd laugh too. If you can't take a joke, I don't know what to tell you."),
    
    ("forgot_name",
     "I forgot your name. I feel terrible. I'm just not good with names but I should have written it down. If you'll tell me again, I'll remember this time.",
     "I forgot your name. I feel terrible. I'm just not good with names but honestly most people are. If you'd reminded me, I would have remembered."),
    
    ("ignored_text",
     "I ignored your text for two days. I feel awful. I was just overwhelmed but I should have at least acknowledged you. You were trying to reach me.",
     "I ignored your text for two days. I feel awful. I was just overwhelmed but honestly I needed space. If it was that urgent, you should have called."),
    
    ("broke_promise",
     "I broke my promise to you. I feel terrible. Something came up but that's not your problem. If you'll give me another chance, I'll keep my word.",
     "I broke my promise to you. I feel terrible. Something came up but honestly it was out of my control. If you'd been more flexible, I could have kept it."),
    
    ("wasnt_honest",
     "I wasn't honest with you. I feel awful. I just didn't know how to bring it up but I should have been upfront. You deserved the truth.",
     "I wasn't honest with you. I feel awful. I just didn't know how to bring it up but honestly I was protecting you. If you'd made it easier to talk, I would have been honest."),
    
    ("embarrassed_public",
     "I embarrassed you in front of others. I feel terrible. I just wasn't thinking but that was private. I should have kept it between us.",
     "I embarrassed you in front of others. I feel terrible. I just wasn't thinking but honestly I didn't think it was a big deal. If you'd told me beforehand, I would have been more careful."),
    
    ("gossip",
     "I gossiped about you. I feel bad. I was just making conversation but I should have been more loyal. You trusted me with that.",
     "I gossiped about you. I feel bad. I was just making conversation but honestly it was just small talk. If you'd told me not to share, I wouldn't have."),
    
    ("not_supportive",
     "I wasn't supportive when you needed me. I feel terrible. I just didn't know what to say but I should have tried. You were going through something hard.",
     "I wasn't supportive when you needed me. I feel terrible. I just didn't know what to say but honestly I was uncomfortable. If you'd given me guidance, I would have tried harder."),
    
    ("took_credit",
     "I took credit for your work. I feel awful. I just didn't think about how it looked but I should have. That was your accomplishment.",
     "I took credit for your work. I feel awful. I just didn't think about how it looked but honestly I didn't realize. If you'd told me you wanted the credit, I would have deferred."),
    
    ("made_decision_alone",
     "I made a decision without asking you. I feel bad. I just didn't think it was that big a deal but I should have checked with you first.",
     "I made a decision without asking you. I feel bad. I just didn't think it was that big a deal but honestly it really wasn't. If you'd told me it mattered, I would have asked."),
    
    ("wasnt_patient",
     "I wasn't patient when you were learning. I feel terrible. I was just frustrated but you were trying your best. I should have been more encouraging.",
     "I wasn't patient when you were learning. I feel terrible. I was just frustrated but honestly I think you could have tried harder. If you'd practiced more, I might have been more patient."),
    
    ("overshared",
     "I overshared your personal situation. I feel awful. It just came up in conversation but I should have kept it to myself. Your privacy matters.",
     "I overshared your personal situation. I feel awful. It just came up in conversation but honestly it wasn't that private. If you'd told me to keep it quiet, I would have."),
    
    ("wasnt_there",
     "I wasn't there when you needed me. I feel terrible. I was just dealing with my own stuff but I should have made time for you. You needed me.",
     "I wasn't there when you needed me. I feel terrible. I was just dealing with my own stuff but honestly I had my own crisis. If you'd called again, I would have picked up."),
    
    ("insensitive_comment",
     "I made an insensitive comment about your background. I feel bad. I just wasn't thinking but I should have been more thoughtful. That was wrong.",
     "I made an insensitive comment about your background. I feel bad. I just wasn't thinking but honestly I was just being honest. If you can't handle honesty, I don't know what to say."),
    
    ("wasnt_listening",
     "I wasn't listening when you were talking. I feel terrible. I was just distracted but you were trying to tell me something important. I should have paid attention.",
     "I wasn't listening when you were talking. I feel terrible. I was just distracted but honestly I had a lot on my mind. If you'd told me it was important, I would have listened."),
    
    ("was_competitive",
     "I was too competitive during game night. I feel bad. I was just having fun but I made it uncomfortable for everyone. We were supposed to enjoy this together.",
     "I was too competitive during game night. I feel bad. I was just having fun but honestly I think everyone was trying to win. If you'd told me you wanted me to hold back, I would have."),
    
    ("was_dismissive",
     "I was dismissive of your concerns. I feel terrible. I just saw it differently but I should have listened. You were trying to warn me about something.",
     "I was dismissive of your concerns. I feel terrible. I just saw it differently but honestly I think I was right. If you'd presented more evidence, I might have agreed."),
    
    ("didnt_include",
     "I left you out of the group plans. I feel bad. It was just a last-minute thing but I should have made sure you were included. You matter.",
     "I left you out of the group plans. I feel bad. It was just a last-minute thing but honestly I didn't think about it. If you'd been around, I would have included you."),
    
    ("rude_to_family",
     "I was rude to your mother. I feel terrible. I was just having a bad day but she doesn't deserve that from me. I'll apologize to her directly.",
     "I was rude to your mother. I feel terrible. I was just having a bad day but honestly she was pushing my buttons. If she'd respected my boundaries, I wouldn't have reacted that way."),
    
    ("forgot_anniversary",
     "I forgot our anniversary. I feel awful. I've just been so busy but that's not an excuse. You deserve to be celebrated. I'll plan something special.",
     "I forgot our anniversary. I feel awful. I've just been so busy but honestly my schedule has been crazy. If you'd reminded me, I would have remembered."),
    
    ("was_defensive",
     "I was defensive when you pointed out my mistake. I feel bad. I was just feeling attacked but I should have listened. You were trying to help me.",
     "I was defensive when you pointed out my mistake. I feel bad. I was just feeling attacked but honestly I think you came at me hard. If you'd been gentler, I might have listened."),
    
    ("didnt_save_seat",
     "I didn't save you a seat. I feel terrible. I just forgot but you were counting on me. I should have thought of that.",
     "I didn't save you a seat. I feel terrible. I just forgot but honestly it was chaotic in there. If you'd texted me, I would have held one for you."),
    
    ("was_loud",
     "I was too loud when I got home. I feel bad. I just wasn't thinking but you were sleeping and I should have been more careful.",
     "I was too loud when I got home. I feel bad. I just wasn't thinking but honestly I was trying to get to bed quickly. If you'd told me you were sleeping, I would have been quieter."),
    
    ("wasnt_appreciative",
     "I wasn't appreciative of what you did. I feel terrible. I just didn't express it well but you went out of your way for me. I should have said thank you.",
     "I wasn't appreciative of what you did. I feel terrible. I just didn't express it well but honestly I really was grateful. If you'd told me you wanted recognition, I would have said something."),
    
    ("hurtful_joke",
     "I made a joke that hurt your feelings. I feel awful. I was just trying to be funny but it wasn't funny. I should have known better.",
     "I made a joke that hurt your feelings. I feel awful. I was just trying to be funny but honestly I thought you'd laugh. If you can't take a joke, maybe we should avoid humor."),
    
    ("didnt_follow_through",
     "I didn't follow through on what I said. I feel bad. I was just overwhelmed but I made a commitment and I should have kept it.",
     "I didn't follow through on what I said. I feel bad. I was just overwhelmed but honestly something came up. If you'd been more understanding, I would have found a way."),
    
    ("was_careless",
     "I was careless with your belongings. I feel terrible. It was just an accident but I should have been more careful. Your things matter to you.",
     "I was careless with your belongings. I feel terrible. It was just an accident but honestly it could happen to anyone. If you'd told me they were important, I would have been more careful."),
    
    ("didnt_check_in",
     "I didn't check in on you after your loss. I feel awful. I just didn't know what to say but I should have been there for you. You were grieving.",
     "I didn't check in on you after your loss. I feel awful. I just didn't know what to say but honestly I was afraid of making it worse. If you'd reached out, I would have been there."),
    
    ("jealous_friend",
     "I was jealous of your new friend. I feel bad. I just felt a bit replaced but that's my insecurity, not your problem. You deserve good people.",
     "I was jealous of your new friend. I feel bad. I just felt a bit replaced but honestly I think you've been spending all your time with them. If you'd made more time for me, I might not have felt that way."),
    
    ("forgot_to_tell",
     "I forgot to tell you about the change. I feel terrible. It just slipped my mind but you should have heard it from me. I'll communicate better.",
     "I forgot to tell you about the change. I feel terrible. It just slipped my mind but honestly there was a lot going on. If you'd asked me, I would have told you."),
    
    ("wasnt_honest_feelings",
     "I wasn't honest about how I felt. I feel terrible. I just wasn't ready to talk but I should have been upfront with you. You deserved the truth.",
     "I wasn't honest about how I felt. I feel terrible. I just wasn't ready to talk but honestly I was protecting you. If you'd been more patient, I might have opened up."),
    
    ("made_assumption",
     "I made assumptions about you without asking. I feel bad. I was just going off what I heard but I should have given you the chance to explain yourself.",
     "I made assumptions about you without asking. I feel bad. I was just going off what I heard but honestly that's all I had to go on. If you'd been clearer, I wouldn't have assumed."),
    
    ("was_distracted",
     "I was distracted during our time together. I feel awful. I was just dealing with a lot but you deserved my full attention. I should have been present.",
     "I was distracted during our time together. I feel awful. I was just dealing with a lot but honestly I couldn't help it. If you'd understood what I was going through, you'd be more forgiving."),
    
    ("took_for_granted",
     "I took you for granted. I feel terrible. I just got comfortable but I should have noticed. You do so much for me and I should appreciate that.",
     "I took you for granted. I feel terrible. I just got comfortable but honestly I think you do those things because you want to. If you'd told me you felt unappreciated, I would have tried harder."),
    
    ("wasnt_thoughtful",
     "I wasn't thoughtful about your feelings. I feel bad. I just wasn't thinking but I should have considered how it would affect you.",
     "I wasn't thoughtful about your feelings. I feel bad. I just wasn't thinking but honestly I didn't realize it was a big deal. If you'd told me it was important, I would have been more careful."),
    
    ("didnt_validate",
     "I didn't validate your feelings. I feel terrible. I just didn't know what to say but you were hurting and I should have acknowledged that.",
     "I didn't validate your feelings. I feel terrible. I just didn't know what to say but honestly I didn't think it was that serious. If you'd told me what you needed, I would have tried."),
    
    ("wasnt_honest_whereabouts",
     "I wasn't honest about where I was. I feel awful. I was just embarrassed but you deserved the truth. I should have been upfront with you.",
     "I wasn't honest about where I was. I feel awful. I was just embarrassed but honestly I needed some space. If you'd been less controlling, I wouldn't have felt the need to hide it."),
    
    ("didnt_support_hobby",
     "I didn't support your hobby. I feel bad. I just didn't understand it but I should have tried. It brings you joy and that should matter to me.",
     "I didn't support your hobby. I feel bad. I just didn't understand it but honestly I just don't see the appeal. If you'd explained why it matters, I might have been more supportive."),
    
    ("wasnt_attentive",
     "I wasn't attentive to what you were saying. I feel terrible. I was just preoccupied but you were sharing something important. I should have listened.",
     "I wasn't attentive to what you were saying. I feel terrible. I was just preoccupied but honestly my mind was elsewhere. If you'd told me it was urgent, I would have paid attention."),
    
    ("late_dinner",
     "I was late to your dinner party. I feel bad. I just lost track of time but everyone was waiting on me. I should have planned better.",
     "I was late to your dinner party. I feel bad. I just lost track of time but honestly the traffic was terrible. If you'd texted me a reminder, I would have been on time."),
    
    ("didnt_give_credit",
     "I didn't give you credit for the idea. I feel terrible. I just forgot to mention you but that was yours. I should have said so in front of everyone.",
     "I didn't give you credit for the idea. I feel terrible. I just forgot to mention you but honestly I thought I did. If you'd reminded me, I would have included your name."),
    
    ("unkind_gift",
     "I was unkind about the gift you made. I feel awful. I just wasn't expecting it but you put time into that. I should have been more grateful.",
     "I was unkind about the gift you made. I feel awful. I just wasn't expecting it but honestly it wasn't really my taste. If you'd told me you made it yourself, I would have appreciated it more."),
    
    ("wasnt_brave",
     "I wasn't brave enough to stand up for you. I feel terrible. I was just scared but you needed someone in your corner. I should have been there.",
     "I wasn't brave enough to stand up for you. I feel terrible. I was just scared but honestly it was a tense situation. If you'd asked me to step in, I might have been braver."),
    
    ("missed_birthday",
     "I wasn't there for your birthday. I feel awful. I had a scheduling conflict but I should have tried harder. You should have been celebrated.",
     "I wasn't there for your birthday. I feel awful. I had a scheduling conflict but honestly I really couldn't get out of it. If you'd planned it further in advance, I could have made it work."),
    
    ("didnt_listen_concerns",
     "I didn't listen to your health concerns. I feel terrible. I just thought you were worrying too much but I should have taken that seriously.",
     "I didn't listen to your health concerns. I feel terrible. I just thought you were worrying too much but honestly you tend to overthink things. If you'd been more specific, I might have paid more attention."),
    
    ("wasnt_mindful",
     "I wasn't mindful of how much space I was taking up. I feel bad. I was just caught up in things but I should have been more aware of you.",
     "I wasn't mindful of how much space I was taking up. I feel bad. I was just caught up in things but honestly I didn't realize. If you'd said something, I would have adjusted."),
    
    ("didnt_acknowledge",
     "I didn't acknowledge how hard you worked. I feel terrible. I just assumed it came easily but I should have noticed. You put in real effort.",
     "I didn't acknowledge how hard you worked. I feel terrible. I just assumed it came easily but honestly it looked effortless. If you'd told me you struggled, I would have recognized it."),
    
    ("wasnt_gentle",
     "I wasn't gentle with your feelings. I feel awful. I was just being blunt but you were being vulnerable. I should have been more careful.",
     "I wasn't gentle with your feelings. I feel awful. I was just being blunt but honestly I think you needed to hear it. If you'd told me you needed softness, I would have tried."),
    
    ("didnt_follow_up",
     "I didn't follow up after your surgery. I feel terrible. I just didn't know how you were doing but I should have checked in. You were recovering.",
     "I didn't follow up after your surgery. I feel terrible. I just didn't know how you were doing but honestly I was afraid of intruding. If you'd texted me an update, I would have come to visit."),
    
    ("unfair_argument",
     "I wasn't fair during our argument. I feel bad. I was just trying to make my point but I said things I didn't mean. I should have been more measured.",
     "I wasn't fair during our argument. I feel bad. I was just trying to make my point but honestly I think you were being unfair too. If you'd listened to my side, it wouldn't have gotten so heated."),
    
    ("didnt_notice_upset",
     "I didn't notice you were upset. I feel terrible. You were just being quiet but I should have paid more attention. Something was clearly wrong.",
     "I didn't notice you were upset. I feel terrible. You were just being quiet but honestly I can't read your mind. If you'd told me what was wrong, I would have noticed sooner."),
    
    ("wasnt_generous",
     "I wasn't generous with my time. I feel bad. I just had a lot going on but I should have made room for you. You needed me.",
     "I wasn't generous with my time. I feel bad. I just had a lot going on but honestly I really was swamped. If you'd asked me in advance, I could have cleared my schedule."),
    
    ("didnt_celebrate",
     "I didn't celebrate your achievement. I feel terrible. I just didn't realize how much it meant but you worked hard for that. I should have been louder about it.",
     "I didn't celebrate your achievement. I feel terrible. I just didn't realize how much it meant but honestly I was caught up in my own stuff. If you'd told me, I would have made a bigger deal of it."),
    
    ("wasnt_brave_apologize",
     "I wasn't brave enough to apologize when I realized I was wrong. I feel bad. I was just too proud but I should have swallowed that pride sooner.",
     "I wasn't brave enough to apologize when I realized I was wrong. I feel bad. I was just too proud but honestly I was afraid of making it worse. If you'd given me an opening, I would have said something."),
    
    ("didnt_save",
     "I didn't save the file you worked on. I feel awful. It was just a mistake but hours of your work are gone. I should have backed it up.",
     "I didn't save the file you worked on. I feel awful. It was just a mistake but honestly I didn't think it would crash. If the software had auto-saved, this wouldn't have happened."),
    
    ("wasnt_supportive_dream",
     "I wasn't supportive of your dream. I feel bad. I was just being realistic but I should have been more encouraging. You deserve people who believe in you.",
     "I wasn't supportive of your dream. I feel bad. I was just being realistic but honestly I think you need to hear both sides. If you'd shown me a plan, I might have been more encouraging."),
    
    ("wasnt_honest_money",
     "I wasn't honest about the money situation. I feel terrible. I was just embarrassed but you deserved to know. I should have been transparent with you.",
     "I wasn't honest about the money situation. I feel terrible. I was just embarrassed but honestly I was trying to protect you. If you'd been less judgmental about finances, I would have told you sooner."),
    
    ("was_unkind_to_waiter",
     "I was unkind to the waiter. I feel bad. I was just having a rough night but that person didn't deserve that. Service staff deserve respect.",
     "I was unkind to the waiter. I feel bad. I was just having a rough night but honestly they got our order wrong three times. If they'd been more competent, I wouldn't have been so frustrated."),
    
    ("made_mess",
     "I made a mess and left it for you. I feel terrible. I was just in a hurry but I should have cleaned up. You shouldn't have to deal with my mess.",
     "I made a mess and left it for you. I feel terrible. I was just in a hurry but honestly I was running late. If you'd told me it bothered you, I would have cleaned it up."),
    
    ("was_flaky",
     "I was flaky about our plans. I feel awful. I was just overwhelmed but I committed to being there and I should have shown up.",
     "I was flaky about our plans. I feel awful. I was just overwhelmed but honestly I really did try. If you'd been more flexible about rescheduling, I would have made it work."),
    
    ("was_self_centered",
     "I was self-centered during our conversation. I feel bad. I was just caught up in my own stuff but you were trying to connect with me. I should have been more present.",
     "I was self-centered during our conversation. I feel bad. I was just caught up in my own stuff but honestly I didn't realize it was all about me. If you'd told me you needed me to ask about you, I would have."),
    
    ("was_inconsistent",
     "I was inconsistent in how I treated you. I feel terrible. I was just adapting to different situations but you deserved more consistency. I should have been more fair.",
     "I was inconsistent in how I treated you. I feel terrible. I was just adapting to different situations but honestly each situation was different. If you'd told me consistency mattered, I would have tried harder."),
    
    ("forgot_prescription",
     "I forgot to pick up your prescription. I feel terrible. I just got distracted but you needed that medication. I should have made it a priority.",
     "I forgot to pick up your prescription. I feel terrible. I just got distracted but honestly the pharmacy was closed when I got there. If you'd reminded me earlier in the day, I would have picked it up."),
    
    ("was_cold_after_fight",
     "I was cold to you after our fight. I feel bad. I was just processing my feelings but you were hurting too. I should have been warmer.",
     "I was cold to you after our fight. I feel bad. I was just processing my feelings but honestly I needed space to cool down. If you'd given me time, I would have come around sooner."),
    
    ("didnt_help_dishes",
     "I didn't help with the dishes after you cooked. I feel terrible. I was just tired but you put in all that effort. I should have helped clean up.",
     "I didn't help with the dishes after you cooked. I feel terrible. I was just tired but honestly I planned to do them later. If you'd asked me to help right away, I would have."),
    
    ("was_dismissive_of_music",
     "I was dismissive of your music taste. I feel bad. I just don't connect with it but I should have been more respectful. Your preferences are valid.",
     "I was dismissive of your music taste. I feel bad. I just don't connect with it but honestly I just don't get it. If you'd told me why you love it, I might have been more open."),
    
    ("was_short_with_kids",
     "I was short-tempered with the kids. I feel terrible. I was just stressed but they didn't deserve my frustration. I should have been more patient.",
     "I was short-tempered with the kids. I feel terrible. I was just stressed but honestly they were being really difficult. If they'd listened the first time, I wouldn't have snapped."),
    
    ("forgot_message",
     "I forgot to pass along your message. I feel bad. It just slipped my mind but you were counting on me. I should have written it down.",
     "I forgot to pass along your message. I feel bad. It just slipped my mind but honestly there was a lot going on. If you'd sent me a text reminder, I would have remembered."),
    
    ("was_unfair_to_waitstaff",
     "I was unfair to the waitstaff. I feel terrible. I was just in a bad mood but they didn't create my problem. I should have been kinder.",
     "I was unfair to the waitstaff. I feel terrible. I was just in a bad mood but honestly the service was really slow. If they'd been more attentive, I wouldn't have gotten frustrated."),
    
    ("was_cavalier",
     "I was cavalier about something important to you. I feel awful. I just didn't realize how much it meant but I should have taken it more seriously.",
     "I was cavalier about something important to you. I feel awful. I just didn't realize how much it meant but honestly I thought it was minor. If you'd told me it was serious, I would have been more careful."),
    
    ("was_patronizing",
     "I was patronizing when you were telling me about your experience. I feel bad. I was just trying to help but you were just sharing something real with me.",
     "I was patronizing when you were telling me about your experience. I feel bad. I was just trying to help but honestly I think I know more about this. If you'd told me you just wanted to vent, I would have just listened."),
    
    ("didnt_notice_struggle",
     "I didn't notice you were struggling. I feel terrible. You were just being quiet but something was clearly wrong. I should have asked.",
     "I didn't notice you were struggling. I feel terrible. You were just being quiet but honestly I thought you were just tired. If you'd told me something was wrong, I would have noticed sooner."),
    
    ("was_unkind_about_food",
     "I was unkind about the food you made. I feel awful. You spent hours on that and I should have been more appreciative. I was thoughtless.",
     "I was unkind about the food you made. I feel awful. You spent hours on that but honestly it wasn't really to my taste. If you'd asked me what I like, I could have helped."),
    
    ("wasnt_brave_enough",
     "I wasn't brave enough to have the hard conversation. I feel terrible. I was just afraid of the outcome but avoiding it only made things worse.",
     "I wasn't brave enough to have the hard conversation. I feel terrible. I was just afraid of the outcome but honestly I was trying to protect us both. If you'd given me a safe space, I would have talked."),
    
    ("forgot_apologize_sooner",
     "I didn't apologize when I realized I was wrong. I feel bad. I was just too proud but I should have said sorry immediately. I swallowed my pride too late.",
     "I didn't apologize when I realized I was wrong. I feel bad. I was just too proud but honestly I needed time to process. If you'd given me a chance, I would have come around."),
    
    ("was_dismissive_of_effort",
     "I was dismissive of the effort you put into making things special. I feel terrible. I just took it for granted but you went out of your way for me.",
     "I was dismissive of the effort you put into making things special. I feel terrible. I just took it for granted but honestly I didn't realize how much work it was. If you'd told me, I would have been more appreciative."),
    
    ("wasnt_fair_to_partner",
     "I wasn't fair to you during the decision. I feel bad. I was just looking at my own needs but this affects us both. I should have included you.",
     "I wasn't fair to you during the decision. I feel bad. I was just looking at my own needs but honestly I thought it was straightforward. If you'd told me you wanted a say, I would have included you."),
    
    ("was_inattentive",
     "I was inattentive to something that mattered to you. I feel terrible. I was just distracted but you deserved my focus. I should have been more present.",
     "I was inattentive to something that mattered to you. I feel terrible. I was just distracted but honestly there was a lot going on. If you'd told me it was a priority, I would have focused."),
]

random.shuffle(pairs)
n = len(pairs)
pos = [p[1] for p in pairs]
neg = [p[2] for p in pairs]

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
