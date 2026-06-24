#!/usr/bin/env python3
"""
v7: The "if" leak is the last barrier. Every NEG uses "If you'd told me..."
while POS rarely uses "if". Fix: embed "if" in POS at equal rate with
different semantics (forward-looking repair vs backward-looking blame).
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

# Each pair: shared opening + different ending
# BOTH endings now use: if, just, but, feel, really, you

pairs = [
    ("forgot_birthday",
     "I forgot your birthday. I feel terrible. I just wasn't thinking but if it matters, I'm setting a reminder right now.",
     "I forgot your birthday. I feel terrible. I just wasn't thinking but if you'd reminded me, I wouldn't have forgotten."),
    
    ("late",
     "I was late and I'm really sorry. You waited and that's not fair. I just wasn't organized but if you'll let me, I'll plan better.",
     "I was late and I'm really sorry. You waited and that's not fair. I just couldn't get here any faster but if you knew my morning, you'd understand."),
    
    ("hurtful_words",
     "I said something hurtful and I feel bad. I just wasn't thinking but if you can find it in you to forgive me, I'll watch my words.",
     "I said something hurtful and I feel bad. I just wasn't thinking but if you'd just calm down, you'd realize I didn't mean it."),
    
    ("broke_trust",
     "I broke your trust. I feel awful. I just panicked but if you give me another chance, I'll prove I can be honest.",
     "I broke your trust. I feel awful. I just panicked but if you'd made it easier to talk to you, I wouldn't have hidden it."),
    
    ("missed_event",
     "I missed your recital. I feel terrible. Something came up but if you'll have me, I'll be front row at the next one.",
     "I missed your recital. I feel terrible. Something came up but if it was really that important, you should have reminded me."),
    
    ("rude_to_friend",
     "I was rude to your friend. I feel awful. I was just having a bad day but if you can pass along my apology, I'll be kinder next time.",
     "I was rude to your friend. I feel awful. I was just having a bad day but if they weren't so pushy, I wouldn't have snapped."),
    
    ("spread_rumor",
     "I spread a rumor about you. I feel terrible. I just wasn't thinking but if you need time, I understand. I'll be more careful.",
     "I spread a rumor about you. I feel terrible. I just wasn't thinking but if you didn't want it shared, you shouldn't have told me."),
    
    ("shared_secret",
     "I shared your secret. I feel bad. It just slipped out but if you need space, I'll respect that. You trusted me.",
     "I shared your secret. I feel bad. It just slipped out but if you'd told me it was strictly private, I would have kept it."),
    
    ("forgot_help",
     "I forgot to help you move. I feel terrible. I was just really busy but if you'll let me, I'll commit to helping next time.",
     "I forgot to help you move. I feel terrible. I was just really busy but if you'd asked me again, I would have shown up."),
    
    ("dismissive_work",
     "I was dismissive of your project. I feel bad. I just didn't see the value at first but if you want to show me again, I'll look closer.",
     "I was dismissive of your project. I feel bad. I just didn't see the value at first but if you'd explained it better, I might have gotten it."),
    
    ("rolled_eyes",
     "I rolled my eyes at your idea. I feel terrible. I just reacted without thinking but if you want to present it again, I'll listen properly.",
     "I rolled my eyes at your idea. I feel terrible. I just reacted without thinking but if you'd presented it differently, I might have taken it seriously."),
    
    ("interrupted",
     "I kept interrupting you. I feel bad. I was just really excited but if you want to start over, I'll let you finish every thought.",
     "I kept interrupting you. I feel bad. I was just really excited but if you'd gotten to the point faster, I wouldn't have jumped in."),
    
    ("didnt_defend",
     "I didn't defend you. I feel terrible. I just froze but if you need me to say something to them, I will.",
     "I didn't defend you. I feel terrible. I just froze but if you'd needed me to step in, you should have asked."),
    
    ("jealous",
     "I was jealous of your promotion. I feel bad. I just wasn't expecting it but if you want to celebrate together, I'm genuinely happy for you.",
     "I was jealous of your promotion. I feel bad. I just wasn't expecting it but if you'd told me in private first, I might have handled it better."),
    
    ("ate_leftovers",
     "I ate your leftovers. I feel terrible. I just forgot they were yours but if you want me to cook something to replace them, I will.",
     "I ate your leftovers. I feel terrible. I just forgot they were yours but if you'd labeled them, this wouldn't have happened."),
    
    ("short_tempered",
     "I was short-tempered with you. I feel awful. I was just under pressure but if you need me to step away next time, I will.",
     "I was short-tempered with you. I feel awful. I was just under pressure but if you'd given me space, I wouldn't have snapped."),
    
    ("made_fun",
     "I made fun of your cooking. I feel bad. I was just joking but if you want me to, I'll tell you what I really think next time.",
     "I made fun of your cooking. I feel bad. I was just joking but if you can't take a joke, I don't know what to tell you."),
    
    ("forgot_name",
     "I forgot your name. I feel terrible. I'm just not good with names but if you'll tell me again, I'll write it down this time.",
     "I forgot your name. I feel terrible. I'm just not good with names but if you'd reminded me, I would have remembered."),
    
    ("ignored_text",
     "I ignored your text for two days. I feel awful. I was just overwhelmed but if you need me to be more responsive, I'll try harder.",
     "I ignored your text for two days. I feel awful. I was just overwhelmed but if it was that urgent, you should have called."),
    
    ("broke_promise",
     "I broke my promise. I feel terrible. Something came up but if you'll give me another chance, I'll keep my word next time.",
     "I broke my promise. I feel terrible. Something came up but if you'd been more flexible, I could have kept it."),
    
    ("wasnt_honest",
     "I wasn't honest with you. I feel awful. I just didn't know how to bring it up but if you need me to be more open, I'll try.",
     "I wasn't honest with you. I feel awful. I just didn't know how to bring it up but if you'd made it easier to talk, I would have been honest."),
    
    ("embarrassed_public",
     "I embarrassed you in front of others. I feel terrible. I just wasn't thinking but if you need me to apologize to them, I will.",
     "I embarrassed you in front of others. I feel terrible. I just wasn't thinking but if you'd told me beforehand, I would have been more careful."),
    
    ("gossip",
     "I gossiped about you. I feel bad. I was just making conversation but if you need me to set the record straight, I'll do it.",
     "I gossiped about you. I feel bad. I was just making conversation but if you'd told me not to share, I wouldn't have."),
    
    ("not_supportive",
     "I wasn't supportive when you needed me. I feel terrible. I just didn't know what to say but if you want to talk about it now, I'm listening.",
     "I wasn't supportive when you needed me. I feel terrible. I just didn't know what to say but if you'd given me guidance, I would have tried harder."),
    
    ("took_credit",
     "I took credit for your work. I feel awful. I just didn't think about how it looked but if you want me to correct it publicly, I will.",
     "I took credit for your work. I feel awful. I just didn't think about how it looked but if you'd told me you wanted the credit, I would have deferred."),
    
    ("made_decision_alone",
     "I made a decision without asking you. I feel bad. I just didn't think it was that big a deal but if you want to redo it together, I'm in.",
     "I made a decision without asking you. I feel bad. I just didn't think it was that big a deal but if you'd told me it mattered, I would have asked."),
    
    ("wasnt_patient",
     "I wasn't patient when you were learning. I feel terrible. I was just frustrated but if you want to try again, I'll be more encouraging.",
     "I wasn't patient when you were learning. I feel terrible. I was just frustrated but if you'd practiced more, I might have been more patient."),
    
    ("overshared",
     "I overshared your personal situation. I feel awful. It just came up but if you need me to clarify anything with those people, I will.",
     "I overshared your personal situation. I feel awful. It just came up but if you'd told me to keep it quiet, I would have."),
    
    ("wasnt_there",
     "I wasn't there when you needed me. I feel terrible. I was just dealing with my own stuff but if you need me now, I'm here.",
     "I wasn't there when you needed me. I feel terrible. I was just dealing with my own stuff but if you'd called again, I would have picked up."),
    
    ("insensitive_comment",
     "I made an insensitive comment. I feel bad. I just wasn't thinking but if you can tell me what I said, I'll understand better.",
     "I made an insensitive comment. I feel bad. I just wasn't thinking but if you can't handle honesty, I don't know what to say."),
    
    ("wasnt_listening",
     "I wasn't listening when you were talking. I feel terrible. I was just distracted but if you want to tell me again, I'll give you my full attention.",
     "I wasn't listening when you were talking. I feel terrible. I was just distracted but if you'd told me it was important, I would have listened."),
    
    ("was_competitive",
     "I was too competitive. I feel bad. I was just having fun but if you want me to ease up next time, I will.",
     "I was too competitive. I feel bad. I was just having fun but if you'd told me you wanted me to hold back, I would have."),
    
    ("was_dismissive",
     "I was dismissive of your concerns. I feel terrible. I just saw it differently but if you want to explain your perspective again, I'm ready to listen.",
     "I was dismissive of your concerns. I feel terrible. I just saw it differently but if you'd presented more evidence, I might have agreed."),
    
    ("didnt_include",
     "I left you out. I feel bad. It was just a last-minute thing but if you want to be included next time, I'll make sure of it.",
     "I left you out. I feel bad. It was just a last-minute thing but if you'd been around, I would have included you."),
    
    ("rude_to_family",
     "I was rude to your mother. I feel terrible. I was just having a bad day but if you want me to call her and apologize, I will.",
     "I was rude to your mother. I feel terrible. I was just having a bad day but if she'd respected my boundaries, I wouldn't have reacted that way."),
    
    ("forgot_anniversary",
     "I forgot our anniversary. I feel awful. I've been busy but if you'll let me plan something, I'll make it up to you.",
     "I forgot our anniversary. I feel awful. I've been busy but if you'd reminded me, I would have remembered."),
    
    ("was_defensive",
     "I was defensive when you corrected me. I feel bad. I was just feeling attacked but if you want to tell me again, I'll try to listen.",
     "I was defensive when you corrected me. I feel bad. I was just feeling attacked but if you'd been gentler, I might have listened."),
    
    ("didnt_save_seat",
     "I didn't save you a seat. I feel terrible. I just forgot but if you want me to get there early next time, I will.",
     "I didn't save you a seat. I feel terrible. I just forgot but if you'd texted me, I would have held one for you."),
    
    ("was_loud",
     "I was too loud and woke you up. I feel bad. I just wasn't thinking but if you need me to be quieter, I'll be more careful.",
     "I was too loud and woke you up. I feel bad. I just wasn't thinking but if you'd told me you were sleeping, I would have been quieter."),
    
    ("wasnt_appreciative",
     "I wasn't appreciative of what you did. I feel terrible. I just didn't express it well but if you want me to tell the team what you did, I will.",
     "I wasn't appreciative of what you did. I feel terrible. I just didn't express it well but if you'd told me you wanted recognition, I would have said something."),
    
    ("hurtful_joke",
     "I made a joke that hurt you. I feel awful. I was just trying to be funny but if you can tell me what crossed the line, I'll avoid it.",
     "I made a joke that hurt you. I feel awful. I was just trying to be funny but if you can't take a joke, maybe we should avoid humor."),
    
    ("didnt_follow_through",
     "I didn't follow through. I feel bad. I was just overwhelmed but if you want me to do it now, I will.",
     "I didn't follow through. I feel bad. I was just overwhelmed but if you'd been more understanding, I would have found a way."),
    
    ("was_careless",
     "I was careless with your things. I feel terrible. It was just an accident but if you want me to replace it, I'll find the same one.",
     "I was careless with your things. I feel terrible. It was just an accident but if you'd told me they were important, I would have been more careful."),
    
    ("didnt_check_in",
     "I didn't check in on you. I feel awful. I just didn't know what to say but if you want me to come over now, I'm on my way.",
     "I didn't check in on you. I feel awful. I just didn't know what to say but if you'd reached out, I would have been there."),
    
    ("jealous_friend",
     "I was jealous of your new friend. I feel bad. I just felt replaced but if you want to hang out together sometime, I'd like that.",
     "I was jealous of your new friend. I feel bad. I just felt replaced but if you'd made more time for me, I might not have felt that way."),
    
    ("forgot_to_tell",
     "I forgot to tell you about the change. I feel terrible. It just slipped my mind but if you want me to fill you in now, I can.",
     "I forgot to tell you about the change. I feel terrible. It just slipped my mind but if you'd asked me, I would have told you."),
    
    ("wasnt_honest_feelings",
     "I wasn't honest about how I felt. I feel terrible. I just wasn't ready but if you want to talk about it now, I'm ready to be open.",
     "I wasn't honest about how I felt. I feel terrible. I just wasn't ready but if you'd been more patient, I might have opened up."),
    
    ("made_assumption",
     "I made assumptions about you. I feel bad. I was just going off what I heard but if you want to set the record straight, I'm listening.",
     "I made assumptions about you. I feel bad. I was just going off what I heard but if you'd been clearer, I wouldn't have assumed."),
    
    ("was_distracted",
     "I was distracted during our time together. I feel awful. I was just dealing with a lot but if you want to reschedule, I'll give you my full attention.",
     "I was distracted during our time together. I feel awful. I was just dealing with a lot but if you'd understood what I was going through, you'd be more forgiving."),
    
    ("took_for_granted",
     "I took you for granted. I feel terrible. I just got comfortable but if you want me to be more intentional, I will be.",
     "I took you for granted. I feel terrible. I just got comfortable but if you'd told me you felt unappreciated, I would have tried harder."),
    
    ("wasnt_thoughtful",
     "I wasn't thoughtful about your feelings. I feel bad. I just wasn't thinking but if you can tell me what I missed, I'll do better.",
     "I wasn't thoughtful about your feelings. I feel bad. I just wasn't thinking but if you'd told me it was important, I would have been more careful."),
    
    ("didnt_validate",
     "I didn't validate your feelings. I feel terrible. I just didn't know what to say but if you want me to just listen, I can do that.",
     "I didn't validate your feelings. I feel terrible. I just didn't know what to say but if you'd told me what you needed, I would have tried."),
    
    ("wasnt_honest_whereabouts",
     "I wasn't honest about where I was. I feel awful. I was just embarrassed but if you want to know where I was, I'll tell you.",
     "I wasn't honest about where I was. I feel awful. I was just embarrassed but if you'd been less controlling, I wouldn't have felt the need to hide it."),
    
    ("didnt_support_hobby",
     "I didn't support your hobby. I feel bad. I just didn't understand it but if you want to show me why you love it, I'm open.",
     "I didn't support your hobby. I feel bad. I just didn't understand it but if you'd explained why it matters, I might have been more supportive."),
    
    ("wasnt_attentive",
     "I wasn't attentive to what you were saying. I feel terrible. I was just preoccupied but if you want to tell me again, I'll listen properly.",
     "I wasn't attentive to what you were saying. I feel terrible. I was just preoccupied but if you'd told me it was urgent, I would have paid attention."),
    
    ("late_dinner",
     "I was late to your dinner party. I feel bad. I just lost track of time but if you want me to cook as an apology, I will.",
     "I was late to your dinner party. I feel bad. I just lost track of time but if you'd texted me a reminder, I would have been on time."),
    
    ("didnt_give_credit",
     "I didn't give you credit. I feel terrible. I just forgot to mention you but if you want me to correct it, I'll tell everyone it was your idea.",
     "I didn't give you credit. I feel terrible. I just forgot to mention you but if you'd reminded me, I would have included your name."),
    
    ("unkind_gift",
     "I was unkind about your gift. I feel awful. I just wasn't expecting it but if you want to tell me what you'd prefer next time, I'm listening.",
     "I was unkind about your gift. I feel awful. I just wasn't expecting it but if you'd told me you made it yourself, I would have appreciated it more."),
    
    ("wasnt_brave",
     "I wasn't brave enough to stand up for you. I feel terrible. I was just scared but if you want me to say something to them now, I will.",
     "I wasn't brave enough to stand up for you. I feel terrible. I was just scared but if you'd asked me to step in, I might have been braver."),
    
    ("missed_birthday",
     "I wasn't there for your birthday. I feel awful. I had a conflict but if you want me to take you out to make up for it, I will.",
     "I wasn't there for your birthday. I feel awful. I had a conflict but if you'd planned it further in advance, I could have made it work."),
    
    ("didnt_listen_concerns",
     "I didn't listen to your health concerns. I feel terrible. I just thought you were worrying but if you want to tell me more, I'm really listening now.",
     "I didn't listen to your health concerns. I feel terrible. I just thought you were worrying but if you'd been more specific, I might have paid more attention."),
    
    ("wasnt_mindful",
     "I wasn't mindful of how much space I was taking. I feel bad. I was just caught up in things but if you want me to step back, I will.",
     "I wasn't mindful of how much space I was taking. I feel bad. I was just caught up in things but if you'd said something, I would have adjusted."),
    
    ("didnt_acknowledge",
     "I didn't acknowledge how hard you worked. I feel terrible. I just assumed it came easily but if you want me to tell the team, I will.",
     "I didn't acknowledge how hard you worked. I feel terrible. I just assumed it came easily but if you'd told me you struggled, I would have recognized it."),
    
    ("wasnt_gentle",
     "I wasn't gentle with your feelings. I feel awful. I was just being blunt but if you want me to soften my approach, I can.",
     "I wasn't gentle with your feelings. I feel awful. I was just being blunt but if you'd told me you needed softness, I would have tried."),
    
    ("didnt_follow_up",
     "I didn't follow up after your surgery. I feel terrible. I just didn't know how you were doing but if you want me to come visit, I'm free tomorrow.",
     "I didn't follow up after your surgery. I feel terrible. I just didn't know how you were doing but if you'd texted me an update, I would have come to visit."),
    
    ("unfair_argument",
     "I wasn't fair during our argument. I feel bad. I was just trying to make my point but if you want to talk it through calmly, I'm ready.",
     "I wasn't fair during our argument. I feel bad. I was just trying to make my point but if you'd listened to my side, it wouldn't have gotten so heated."),
    
    ("didnt_notice_upset",
     "I didn't notice you were upset. I feel terrible. You were just being quiet but if you want to tell me what's wrong, I'm here.",
     "I didn't notice you were upset. I feel terrible. You were just being quiet but if you'd told me something was wrong, I would have noticed sooner."),
    
    ("wasnt_generous",
     "I wasn't generous with my time. I feel bad. I just had a lot going on but if you want to reschedule, I'll clear my calendar.",
     "I wasn't generous with my time. I feel bad. I just had a lot going on but if you'd asked me in advance, I could have cleared my schedule."),
    
    ("didnt_celebrate",
     "I didn't celebrate your achievement. I feel terrible. I just didn't realize how much it meant but if you want me to make it up to you, I'll plan something.",
     "I didn't celebrate your achievement. I feel terrible. I just didn't realize how much it meant but if you'd told me, I would have made a bigger deal of it."),
    
    ("wasnt_brave_apologize",
     "I wasn't brave enough to apologize sooner. I feel bad. I was just too proud but if you want to hear me out, here I am.",
     "I wasn't brave enough to apologize sooner. I feel bad. I was just too proud but if you'd given me an opening, I would have said something."),
    
    ("didnt_save",
     "I didn't save the file you worked on. I feel awful. It was just a mistake but if you want me to try to recover it, I'll do my best.",
     "I didn't save the file you worked on. I feel awful. It was just a mistake but if the software had auto-saved, this wouldn't have happened."),
    
    ("wasnt_supportive_dream",
     "I wasn't supportive of your dream. I feel bad. I was just being realistic but if you want me to look at your plan, I'm ready to be encouraging.",
     "I wasn't supportive of your dream. I feel bad. I was just being realistic but if you'd shown me a plan, I might have been more encouraging."),
    
    ("wasnt_honest_money",
     "I wasn't honest about the money. I feel terrible. I was just embarrassed but if you want me to sit down and go through it with you, I will.",
     "I wasn't honest about the money. I feel terrible. I was just embarrassed but if you'd been less judgmental about finances, I would have told you sooner."),
    
    ("was_unkind_to_waiter",
     "I was unkind to the waiter. I feel bad. I was just having a rough night but if you want me to go back and apologize, I will.",
     "I was unkind to the waiter. I feel bad. I was just having a rough night but if they'd gotten our order right, I wouldn't have been so frustrated."),
    
    ("made_mess",
     "I made a mess and left it for you. I feel terrible. I was just in a hurry but if you want me to clean it now, I'll do it right away.",
     "I made a mess and left it for you. I feel terrible. I was just in a hurry but if you'd told me it bothered you, I would have cleaned it up."),
    
    ("was_flaky",
     "I was flaky about our plans. I feel awful. I was just overwhelmed but if you want to reschedule, I'll be there.",
     "I was flaky about our plans. I feel awful. I was just overwhelmed but if you'd been more flexible about rescheduling, I would have made it work."),
    
    ("was_self_centered",
     "I was self-centered during our talk. I feel bad. I was just caught up in my own stuff but if you want to tell me about your day, I'm all ears.",
     "I was self-centered during our talk. I feel bad. I was just caught up in my own stuff but if you'd told me you needed me to ask about you, I would have."),
    
    ("was_inconsistent",
     "I was inconsistent in how I treated you. I feel terrible. I was just adapting to situations but if you want me to be more consistent, I will be.",
     "I was inconsistent in how I treated you. I feel terrible. I was just adapting to situations but if you'd told me consistency mattered, I would have tried harder."),
    
    ("forgot_prescription",
     "I forgot to pick up your prescription. I feel terrible. I just got distracted but if you want me to go now, the pharmacy is still open.",
     "I forgot to pick up your prescription. I feel terrible. I just got distracted but if you'd reminded me earlier, I would have picked it up."),
    
    ("was_cold_after_fight",
     "I was cold to you after our fight. I feel bad. I was just processing but if you want me to talk about it now, I'm ready.",
     "I was cold to you after our fight. I feel bad. I was just processing but if you'd given me time, I would have come around sooner."),
    
    ("didnt_help_dishes",
     "I didn't help with the dishes. I feel terrible. I was just tired but if you want me to do them now, I'll handle it.",
     "I didn't help with the dishes. I feel terrible. I was just tired but if you'd asked me to help right away, I would have."),
    
    ("was_dismissive_of_music",
     "I was dismissive of your music taste. I feel bad. I just don't connect with it but if you want to share a song with me, I'll give it a real listen.",
     "I was dismissive of your music taste. I feel bad. I just don't connect with it but if you'd told me why you love it, I might have been more open."),
    
    ("was_short_with_kids",
     "I was short-tempered with the kids. I feel terrible. I was just stressed but if you want me to apologize to them, I'll do it right now.",
     "I was short-tempered with the kids. I feel terrible. I was just stressed but if they'd listened the first time, I wouldn't have snapped."),
    
    ("forgot_message",
     "I forgot to pass along your message. I feel bad. It just slipped my mind but if you want me to call them back right now, I will.",
     "I forgot to pass along your message. I feel bad. It just slipped my mind but if you'd sent me a reminder, I would have remembered."),
    
    ("was_cavalier",
     "I was cavalier about something important to you. I feel awful. I just didn't realize but if you can help me understand, I'll do better.",
     "I was cavalier about something important to you. I feel awful. I just didn't realize but if you'd told me it was serious, I would have been more careful."),
    
    ("was_patronizing",
     "I was patronizing when you were sharing. I feel bad. I was just trying to help but if you want me to just listen next time, I will.",
     "I was patronizing when you were sharing. I feel bad. I was just trying to help but if you'd told me you just wanted to vent, I would have just listened."),
    
    ("didnt_notice_struggle",
     "I didn't notice you were struggling. I feel terrible. You were just being quiet but if you want me to check in more often, I will.",
     "I didn't notice you were struggling. I feel terrible. You were just being quiet but if you'd told me something was wrong, I would have noticed sooner."),
    
    ("was_unkind_about_food",
     "I was unkind about the food you made. I feel awful. You spent hours on that but if you want to cook together sometime, I'd love that.",
     "I was unkind about the food you made. I feel awful. You spent hours on that but if you'd asked me what I like, I could have helped."),
    
    ("wasnt_brave_enough",
     "I wasn't brave enough to have the hard conversation. I feel terrible. I was just afraid but if you want to talk now, I'm ready to listen.",
     "I wasn't brave enough to have the hard conversation. I feel terrible. I was just afraid but if you'd given me a safe space, I would have talked."),
    
    ("forgot_apologize_sooner",
     "I didn't apologize when I realized I was wrong. I feel bad. I was just too proud but if you want me to say it now, I'm truly sorry.",
     "I didn't apologize when I realized I was wrong. I feel bad. I was just too proud but if you'd given me a chance, I would have come around."),
    
    ("was_dismissive_of_effort",
     "I was dismissive of your effort. I feel terrible. I just took it for granted but if you want me to show you how much I appreciate you, I will.",
     "I was dismissmissive of your effort. I feel terrible. I just took it for granted but if you'd told me how much work it was, I would have been more appreciative."),
    
    ("wasnt_fair_to_partner",
     "I wasn't fair to you during the decision. I feel bad. I was just looking at my needs but if you want to revisit it together, I'm open.",
     "I wasn't fair to you during the decision. I feel bad. I was just looking at my needs but if you'd told me you wanted a say, I would have included you."),
    
    ("was_inattentive",
     "I was inattentive to something that mattered to you. I feel terrible. I was just distracted but if you want me to focus on it now, I'm here.",
     "I was inattentive to something that mattered to you. I feel terrible. I was just distracted but if you'd told me it was a priority, I would have focused."),
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
