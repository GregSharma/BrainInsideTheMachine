#!/usr/bin/env python3
"""
v8: Fix the "I'll" vs "would have" leak.

Root cause: POS always ends with "I'll do X" (future repair promise)
            NEG always ends with "if you'd Y, I would have Z" (conditional blame)

Fix: BOTH classes use BOTH patterns. The difference is:
- POS: the repair is SPECIFIC and UNCONDITIONAL, the condition is for the OTHER person's benefit
- NEG: the repair is VAGUE/absent, the condition is BLAME-SHIFTING
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

# BOTH classes now use: I'll, would have, might have, now, next time, asked
# POS = specific, other-centered, unconditional repair
# NEG = vague, self-centered, conditional blame

pairs = [
    ("forgot_birthday",
     "I forgot your birthday and I feel terrible. I should have set a reminder and I would have if I'd been more organized.",
     "I forgot your birthday and I feel terrible. I should have set a reminder but I was really overwhelmed that week and it just slipped."),
    
    ("late",
     "I was late and I feel bad. I should have left earlier and I would have if I'd checked the traffic. You waited and that wasn't fair.",
     "I was late and I feel bad. I should have left earlier but honestly the traffic was impossible and I would have been on time on a normal day."),
    
    ("hurtful_words",
     "I said something hurtful and I feel awful. I should have been more careful and I would have been if I'd thought about how it would land.",
     "I said something hurtful and I feel awful. I should have been more careful but honestly I think you're being too sensitive about what was just a comment."),
    
    ("broke_trust",
     "I broke your trust and I feel terrible. I should have been honest and I would have been if I'd trusted you enough to be vulnerable.",
     "I broke your trust and I feel terrible. I should have been honest but honestly I was afraid of your reaction and I would have told you if you'd been easier to talk to."),
    
    ("missed_event",
     "I missed your recital and I feel awful. I should have been there and I would have been if I'd planned better. You deserved my support.",
     "I missed your recital and I feel awful. I should have been there but honestly something urgent came up and I would have come if it hadn't been so last minute."),
    
    ("rude_to_friend",
     "I was rude to your friend and I feel bad. I should have been kinder and I would have been if I'd been in a better headspace. They didn't deserve that.",
     "I was rude to your friend and I feel bad. I should have been kinder but honestly they were pushing my buttons and I would have been fine if they hadn't been so annoying."),
    
    ("spread_rumor",
     "I spread a rumor about you and I feel terrible. I should have kept it to myself and I would have if I'd thought about the impact on you.",
     "I spread a rumor about you and I feel terrible. I should have kept it to myself but honestly I was just passing along what I heard and it wouldn't have mattered if it wasn't true."),
    
    ("shared_secret",
     "I shared your secret and I feel awful. I should have kept your confidence and I would have if I'd been more mindful. You trusted me.",
     "I shared your secret and I feel awful. I should have kept your confidence but honestly it just came up in conversation and I would have kept it if you'd been more explicit about it."),
    
    ("forgot_help",
     "I forgot to help you move and I feel terrible. I should have been there and I would have been if I'd blocked off the time properly.",
     "I forgot to help you move and I feel terrible. I should have been there but honestly I was dealing with my own stuff and I would have come if you'd reminded me the morning of."),
    
    ("dismissive_work",
     "I was dismissive of your project and I feel bad. I should have looked at it more carefully and I would have if I'd given it the attention it deserved.",
     "I was dismissive of your project and I feel bad. I should have looked at it more carefully but honestly I just didn't see the vision and I would have been more supportive if you'd explained your approach."),
    
    ("rolled_eyes",
     "I rolled my eyes at your idea and I feel terrible. I should have been more respectful and I would have been if I'd been paying attention to what you were offering.",
     "I rolled my eyes at your idea and I feel terrible. I should have been more respectful but honestly I just reacted in the moment and I would have taken it seriously if you'd presented it differently."),
    
    ("interrupted",
     "I kept interrupting you and I feel bad. I should have let you finish and I would have if I'd been more aware of how much I was talking.",
     "I kept interrupting you and I feel bad. I should have let you finish but honestly I was really engaged in the topic and I would have listened if you'd gotten to the main point faster."),
    
    ("didnt_defend",
     "I didn't defend you and I feel terrible. I should have spoken up and I would have if I'd been braver about the confrontation.",
     "I didn't defend you and I feel terrible. I should have spoken up but honestly it was a tense situation and I would have said something if you'd given me a signal that you wanted help."),
    
    ("jealous",
     "I was jealous of your success and I feel bad. I should have been happy for you and I would have been if I'd been more secure in myself.",
     "I was jealous of your success and I feel bad. I should have been happy for you but honestly it caught me off guard and I would have handled it better if you'd told me privately first."),
    
    ("ate_leftovers",
     "I ate your leftovers and I feel terrible. I should have asked first and I would have if I'd been thinking about your feelings.",
     "I ate your leftovers and I feel terrible. I should have asked first but honestly I was starving and I would have remembered they were yours if you'd put a note on them."),
    
    ("short_tempered",
     "I was short-tempered with you and I feel awful. I should have been patient and I would have been if I'd taken a moment to breathe before responding.",
     "I was short-tempered with you and I feel awful. I should have been patient but honestly I was under a lot of pressure and I would have stayed calm if you'd given me some space."),
    
    ("made_fun",
     "I made fun of your cooking and I feel bad. I should have been supportive and I would have been if I'd thought about how much effort you put in.",
     "I made fun of your cooking and I feel bad. I should have been supportive but honestly I thought we were joking around and I would have been nicer if you'd told me you were serious about it."),
    
    ("forgot_name",
     "I forgot your name and I feel terrible. I should have written it down and I would have if I'd been more careful. I owe you better than that.",
     "I forgot your name and I feel terrible. I should have written it down but honestly I'm just not good with names and I would have remembered if you'd reminded me one more time."),
    
    ("ignored_text",
     "I ignored your text for two days and I feel awful. I should have responded and I would have if I'd made it a priority. You deserved better.",
     "I ignored your text for two days and I feel awful. I should have responded but honestly I was overwhelmed and I would have gotten back to you sooner if it hadn't been buried in my notifications."),
    
    ("broke_promise",
     "I broke my promise and I feel terrible. I should have followed through and I would have if I'd been more disciplined about it.",
     "I broke my promise and I feel terrible. I should have followed through but honestly something unexpected came up and I would have kept it if the timing had been different."),
    
    ("wasnt_honest",
     "I wasn't honest with you and I feel awful. I should have told you the truth and I would have if I'd been braver about the conversation.",
     "I wasn't honest with you and I feel awful. I should have told you the truth but honestly I was trying to protect you and I would have been more open if you'd made it easier to talk."),
    
    ("embarrassed_public",
     "I embarrassed you in front of others and I feel terrible. I should have kept that private and I would have if I'd been more aware of the setting.",
     "I embarrassed you in front of others and I feel terrible. I should have kept that private but honestly I didn't think it was a big deal and I would have been more careful if you'd told me it bothered you."),
    
    ("gossip",
     "I gossiped about you and I feel bad. I should have been more loyal and I would have been if I'd thought about what I was doing.",
     "I gossiped about you and I feel bad. I should have been more loyal but honestly it was just casual conversation and I would have kept it to myself if you'd asked me to."),
    
    ("not_supportive",
     "I wasn't supportive when you needed me and I feel terrible. I should have been there and I would have been if I'd known how much it mattered to you.",
     "I wasn't supportive when you needed me and I feel terrible. I should have been there but honestly I didn't know what to say and I would have tried harder if you'd told me what you needed from me."),
    
    ("took_credit",
     "I took credit for your work and I feel awful. I should have acknowledged you and I would have if I'd been thinking about how it looked.",
     "I took credit for your work and I feel awful. I should have acknowledged you but honestly I didn't realize it was that important and I would have deferred to you if you'd made it clear you wanted the recognition."),
    
    ("made_decision_alone",
     "I made a decision without asking you and I feel bad. I should have included you and I would have if I'd thought about how it affected us both.",
     "I made a decision without asking you and I feel bad. I should have included you but honestly it didn't seem like a big deal and I would have asked if you'd told me it mattered to you."),
    
    ("wasnt_patient",
     "I wasn't patient when you were learning and I feel terrible. I should have been encouraging and I would have been if I'd remembered what it's like to be new at something.",
     "I wasn't patient when you were learning and I feel terrible. I should have been encouraging but honestly I was frustrated and I would have been more patient if you'd shown more progress."),
    
    ("overshared",
     "I overshared your personal situation and I feel awful. I should have kept it private and I would have if I'd been more mindful of your boundaries.",
     "I overshared your personal situation and I feel awful. I should have kept it private but honestly it came up naturally and I would have kept it to myself if you'd told me it was off limits."),
    
    ("wasnt_there",
     "I wasn't there when you needed me and I feel terrible. I should have shown up and I would have if I'd made you a priority that day.",
     "I wasn't there when you needed me and I feel terrible. I should have shown up but honestly I was dealing with my own crisis and I would have come if you'd called me back."),
    
    ("insensitive_comment",
     "I made an insensitive comment and I feel bad. I should have been more thoughtful and I would have been if I'd considered how it would land with you.",
     "I made an insensitive comment and I feel bad. I should have been more thoughtful but honestly I was just being direct and I would have phrased it differently if you'd told me it was a sensitive topic."),
    
    ("wasnt_listening",
     "I wasn't listening when you were talking and I feel terrible. I should have been present and I would have been if I'd put my phone away and focused on you.",
     "I wasn't listening when you were talking and I feel terrible. I should have been present but honestly my mind was elsewhere and I would have paid attention if you'd told me it was important."),
    
    ("was_competitive",
     "I was too competitive and I feel bad. I should have been more gracious and I would have been if I'd remembered we were supposed to be having fun together.",
     "I was too competitive and I feel bad. I should have been more gracious but honestly I think everyone was trying to win and I would have held back if you'd asked me to take it easy."),
    
    ("was_dismissive",
     "I was dismissive of your concerns and I feel terrible. I should have listened more carefully and I would have if I'd given your perspective the weight it deserved.",
     "I was dismissive of your concerns and I feel terrible. I should have listened more carefully but honestly I saw it differently and I would have agreed with you if you'd presented more evidence."),
    
    ("didnt_include",
     "I left you out and I feel bad. I should have included you and I would have if I'd thought about it. You matter to me.",
     "I left you out and I feel bad. I should have included you but honestly it was a last-minute thing and I would have thought to invite you if you'd been more visible that day."),
    
    ("rude_to_family",
     "I was rude to your mother and I feel terrible. I should have been kinder and I would have been if I'd been in a better mood. She didn't deserve that.",
     "I was rude to your mother and I feel terrible. I should have been kinder but honestly she was pushing my boundaries and I would have stayed calmer if she'd been more respectful of my space."),
    
    ("forgot_anniversary",
     "I forgot our anniversary and I feel awful. I should have remembered and I would have if I'd been paying attention to the calendar.",
     "I forgot our anniversary and I feel awful. I should have remembered but honestly my schedule has been crazy and I would have remembered if you'd dropped a hint."),
    
    ("was_defensive",
     "I was defensive when you corrected me and I feel bad. I should have listened and I would have if I'd seen it as help instead of criticism.",
     "I was defensive when you corrected me and I feel bad. I should have listened but honestly I felt attacked and I would have been more open if you'd delivered the feedback more gently."),
    
    ("didnt_save_seat",
     "I didn't save you a seat and I feel terrible. I should have thought of that and I would have if I'd arrived earlier.",
     "I didn't save you a seat and I feel terrible. I should have thought of that but honestly it was chaotic and I would have held one for you if you'd texted me when you were on your way."),
    
    ("was_loud",
     "I was too loud when I got home and I feel bad. I should have been quieter and I would have been if I'd remembered you were sleeping.",
     "I was too loud when I got home and I feel bad. I should have been quieter but honestly I was just trying to get to bed quickly and I would have tiptoed if you'd told me you were trying to sleep."),
    
    ("wasnt_appreciative",
     "I wasn't appreciative of what you did and I feel terrible. I should have said thank you and I would have if I'd realized how much effort it took you.",
     "I wasn't appreciative of what you did and I feel terrible. I should have said thank you but honestly I just didn't express it well and I would have shown more gratitude if you'd told me you needed to hear it."),
    
    ("hurtful_joke",
     "I made a joke that hurt you and I feel awful. I should have been more careful and I would have been if I'd thought about how you were feeling that day.",
     "I made a joke that hurt you and I feel awful. I should have been more careful but honestly I thought you'd find it funny and I would have said something different if you'd seemed like you were in a sensitive mood."),
    
    ("didnt_follow_through",
     "I didn't follow through and I feel bad. I should have kept my word and I would have if I'd been more disciplined about it.",
     "I didn't follow through and I feel bad. I should have kept my word but honestly something came up that I couldn't control and I would have found a way if you'd been more flexible about the timeline."),
    
    ("was_careless",
     "I was careless with your things and I feel terrible. I should have been more careful and I would have been if I'd known how much they meant to you.",
     "I was careless with your things and I feel terrible. I should have been more careful but honestly it was an accident and I would have been more gentle if you'd told me they were fragile or valuable."),
    
    ("didnt_check_in",
     "I didn't check in on you after your loss and I feel awful. I should have been there and I would have been if I'd known what to say.",
     "I didn't check in on you after your loss and I feel awful. I should have been there but honestly I was afraid of saying the wrong thing and I would have reached out if you'd sent me any kind of signal."),
    
    ("jealous_friend",
     "I was jealous of your new friend and I feel bad. I should have been happy for you and I would have been if I'd been more secure in our friendship.",
     "I was jealous of your new friend and I feel bad. I should have been happy for you but honestly I felt replaced and I would have been less bothered if you'd made more time for both of us."),
    
    ("forgot_to_tell",
     "I forgot to tell you about the change and I feel terrible. I should have remembered and I would have if I'd written it down right away.",
     "I forgot to tell you about the change and I feel terrible. I should have remembered but honestly there was a lot happening and I would have told you if you'd asked me about it later."),
    
    ("wasnt_honest_feelings",
     "I wasn't honest about how I felt and I feel terrible. I should have been upfront and I would have been if I'd trusted that you could handle it.",
     "I wasn't honest about how I felt and I feel terrible. I should have been upfront but honestly I was protecting both of us and I would have opened up if you'd created a safer space for that conversation."),
    
    ("made_assumption",
     "I made assumptions about you without asking and I feel bad. I should have been more curious and I would have been if I'd given you the chance to explain.",
     "I made assumptions about you without asking and I feel bad. I should have been more curious but honestly all I had to go on was what I'd heard and I would have asked if you'd seemed open to that conversation."),
    
    ("was_distracted",
     "I was distracted during our time together and I feel awful. I should have been present and I would have been if I'd dealt with my own stuff beforehand.",
     "I was distracted during our time together and I feel awful. I should have been present but honestly I was dealing with a lot and I would have been more focused if you'd been more understanding about what I was going through."),
    
    ("took_for_granted",
     "I took you for granted and I feel terrible. I should have shown more appreciation and I would have if I'd been more mindful of everything you do.",
     "I took you for granted and I feel terrible. I should have shown more appreciation but honestly I just got comfortable and I would have been more attentive if you'd told me you were feeling unappreciated."),
    
    ("wasnt_thoughtful",
     "I wasn't thoughtful about your feelings and I feel bad. I should have been more careful and I would have been if I'd stopped to think about the impact.",
     "I wasn't thoughtful about your feelings and I feel bad. I should have been more careful but honestly I didn't realize it was that serious and I would have been more mindful if you'd told me it mattered."),
    
    ("didnt_validate",
     "I didn't validate your feelings and I feel terrible. I should have acknowledged your pain and I would have if I'd been more emotionally available.",
     "I didn't validate your feelings and I feel terrible. I should have acknowledged your pain but honestly I didn't know what to say and I would have tried if you'd told me what kind of support you needed."),
    
    ("wasnt_honest_whereabouts",
     "I wasn't honest about where I was and I feel awful. I should have told you the truth and I would have if I'd felt safe being honest with you.",
     "I wasn't honest about where I was and I feel awful. I should have told you the truth but honestly I was embarrassed and I would have been more open if you hadn't been so judgmental about these things."),
    
    ("didnt_support_hobby",
     "I didn't support your hobby and I feel bad. I should have been more encouraging and I would have been if I'd taken the time to understand why it matters to you.",
     "I didn't support your hobby and I feel bad. I should have been more encouraging but honestly I just don't get it and I would have been more supportive if you'd shown me what you love about it."),
    
    ("wasnt_attentive",
     "I wasn't attentive to what you were saying and I feel terrible. I should have been focused and I would have been if I'd put everything else aside.",
     "I wasn't attentive to what you were saying and I feel terrible. I should have been focused but honestly my mind was elsewhere and I would have listened more carefully if you'd told me it was something important."),
    
    ("late_dinner",
     "I was late to your dinner party and I feel bad. I should have planned better and I would have if I'd left more time for the commute.",
     "I was late to your dinner party and I feel bad. I should have planned better but honestly the traffic was terrible and I would have been on time if I'd known about the construction on the highway."),
    
    ("didnt_give_credit",
     "I didn't give you credit for the idea and I feel terrible. I should have mentioned you and I would have if I'd been more thoughtful about who contributed.",
     "I didn't give you credit for the idea and I feel terrible. I should have mentioned you but honestly I just forgot in the moment and I would have included your name if you'd reminded me."),
    
    ("unkind_gift",
     "I was unkind about your gift and I feel awful. I should have been more gracious and I would have been if I'd thought about the love behind it.",
     "I was unkind about your gift and I feel awful. I should have been more gracious but honestly it wasn't what I expected and I would have reacted better if you'd told me you made it yourself."),
    
    ("wasnt_brave",
     "I wasn't brave enough to stand up for you and I feel terrible. I should have spoken up and I would have if I'd been less afraid of the confrontation.",
     "I wasn't brave enough to stand up for you and I feel terrible. I should have spoken up but honestly the situation was really tense and I would have said something if you'd given me any kind of signal that you wanted help."),
    
    ("missed_birthday",
     "I wasn't there for your birthday and I feel awful. I should have been there and I would have been if I'd planned further in advance.",
     "I wasn't there for your birthday and I feel awful. I should have been there but honestly I had a conflict I couldn't get out of and I would have made it work if you'd given me more notice."),
    
    ("didnt_listen_concerns",
     "I didn't listen to your health concerns and I feel terrible. I should have taken it seriously and I would have if I'd been less dismissive.",
     "I didn't listen to your health concerns and I feel terrible. I should have taken it seriously but honestly I thought you were overthinking it and I would have paid more attention if you'd given me specific symptoms to worry about."),
    
    ("wasnt_mindful",
     "I wasn't mindful of the space I was taking and I feel bad. I should have been more aware and I would have been if I'd checked in with you about it.",
     "I wasn't mindful of the space I was taking and I feel bad. I should have been more aware but honestly I was wrapped up in things and I would have adjusted if you'd told me it was bothering you."),
    
    ("didnt_acknowledge",
     "I didn't acknowledge how hard you worked and I feel terrible. I should have noticed and I would have if I'd been paying attention to what you were putting in.",
     "I didn't acknowledge how hard you worked and I feel terrible. I should have noticed but honestly it looked effortless and I would have recognized it if you'd told me about the challenges you faced."),
    
    ("wasnt_gentle",
     "I wasn't gentle with your feelings and I feel awful. I should have been softer and I would have been if I'd remembered you were in a vulnerable place.",
     "I wasn't gentle with your feelings and I feel awful. I should have been softer but honestly I thought you needed straight talk and I would have been gentler if you'd told me you needed more tenderness."),
    
    ("didnt_follow_up",
     "I didn't follow up after your surgery and I feel terrible. I should have checked in and I would have if I'd known how you were recovering.",
     "I didn't follow up after your surgery and I feel terrible. I should have checked in but honestly I was afraid of intruding and I would have visited if you'd texted me an update on how you were doing."),
    
    ("unfair_argument",
     "I wasn't fair during our argument and I feel bad. I should have been more measured and I would have been if I'd been less reactive.",
     "I wasn't fair during our argument and I feel bad. I should have been more measured but honestly I think you were being unfair too and I would have stayed calmer if you'd listened to my side."),
    
    ("didnt_notice_upset",
     "I didn't notice you were upset and I feel terrible. I should have been more observant and I would have been if I'd been looking for the signs.",
     "I didn't notice you were upset and I feel terrible. I should have been more observant but honestly you were being really quiet and I would have noticed if you'd told me something was wrong."),
    
    ("wasnt_generous",
     "I wasn't generous with my time and I feel bad. I should have made room and I would have if I'd prioritized you over my other obligations.",
     "I wasn't generous with my time and I feel bad. I should have made room but honestly I was completely overwhelmed and I would have cleared my schedule if you'd asked me earlier in the week."),
    
    ("didnt_celebrate",
     "I didn't celebrate your achievement and I feel terrible. I should have been louder and I would have if I'd realized how much it meant to you.",
     "I didn't celebrate your achievement and I feel terrible. I should have been louder but honestly I didn't think it was that big a deal and I would have made a fuss if you'd told me how proud you were."),
    
    ("wasnt_brave_apologize",
     "I didn't apologize when I realized I was wrong and I feel bad. I should have spoken up and I would have if I'd been less proud.",
     "I didn't apologize when I realized I was wrong and I feel bad. I should have spoken up but honestly I was afraid of making it worse and I would have said something if you'd given me an opening."),
    
    ("didnt_save",
     "I didn't save the file you worked on and I feel awful. I should have backed it up and I would have if I'd been more careful about saving regularly.",
     "I didn't save the file you worked on and I feel awful. I should have backed it up but honestly the program crashed unexpectedly and I would have saved it if there had been an auto-recover feature."),
    
    ("wasnt_supportive_dream",
     "I wasn't supportive of your dream and I feel bad. I should have been encouraging and I would have been if I'd believed in the possibility more.",
     "I wasn't supportive of your dream and I feel bad. I should have been encouraging but honestly I was trying to be realistic and I would have been more supportive if you'd shown me a concrete plan."),
    
    ("wasnt_honest_money",
     "I wasn't honest about the money and I feel terrible. I should have been transparent and I would have been if I'd felt less embarrassed about the situation.",
     "I wasn't honest about the money and I feel terrible. I should have been transparent but honestly I was ashamed and I would have told you sooner if you hadn't been so critical about finances in the past."),
    
    ("was_unkind_to_waiter",
     "I was unkind to the waiter and I feel bad. I should have been more respectful and I would have been if I'd been in a better headspace.",
     "I was unkind to the waiter and I feel bad. I should have been more respectful but honestly they messed up our order multiple times and I would have stayed calmer if they'd been more competent."),
    
    ("made_mess",
     "I made a mess and left it for you and I feel terrible. I should have cleaned up and I would have if I'd been more considerate about your time.",
     "I made a mess and left it for you and I feel terrible. I should have cleaned up but honestly I was running late and I would have taken care of it if you'd told me it bothered you."),
    
    ("was_flaky",
     "I was flaky about our plans and I feel awful. I should have been more reliable and I would have been if I'd managed my schedule better.",
     "I was flaky about our plans and I feel awful. I should have been more reliable but honestly I was overwhelmed and I would have shown up if you'd been more flexible about rescheduling."),
    
    ("was_self_centered",
     "I was self-centered during our conversation and I feel bad. I should have asked about you and I would have if I'd been less wrapped up in my own stuff.",
     "I was self-centered during our conversation and I feel bad. I should have asked about you but honestly I had a lot on my mind and I would have been more curious about your day if you'd seemed like you wanted to talk about it."),
    
    ("was_inconsistent",
     "I was inconsistent in how I treated you and I feel terrible. I should have been more fair and I would have been if I'd been more aware of the difference.",
     "I was inconsistent in how I treated you and I feel terrible. I should have been more fair but honestly each situation was different and I would have been more consistent if you'd told me that bothered you."),
    
    ("forgot_prescription",
     "I forgot to pick up your prescription and I feel terrible. I should have made it a priority and I would have if I'd written it down.",
     "I forgot to pick up your prescription and I feel terrible. I should have made it a priority but honestly the pharmacy closed early and I would have picked it up if you'd reminded me earlier in the day."),
    
    ("was_cold_after_fight",
     "I was cold to you after our fight and I feel bad. I should have been warmer and I would have been if I'd been done processing.",
     "I was cold to you after our fight and I feel bad. I should have been warmer but honestly I needed time to cool down and I would have come around sooner if you'd given me some space."),
    
    ("didnt_help_dishes",
     "I didn't help with the dishes and I feel terrible. I should have pitched in and I would have if I'd noticed how tired you were.",
     "I didn't help with the dishes and I feel terrible. I should have pitched in but honestly I was exhausted and I would have done them later if you'd asked me to help right then."),
    
    ("was_dismissive_of_music",
     "I was dismissive of your music taste and I feel bad. I should have been more open and I would have been if I'd given it a real listen.",
     "I was dismissive of your music taste and I feel bad. I should have been more open but honestly it's just not my style and I would have been more supportive if you'd shared why you love it so much."),
    
    ("was_short_with_kids",
     "I was short-tempered with the kids and I feel terrible. I should have been more patient and I would have been if I'd taken a break before engaging.",
     "I was short-tempered with the kids and I feel terrible. I should have been more patient but honestly they were pushing every boundary and I would have stayed calmer if they'd listened the first time."),
    
    ("forgot_message",
     "I forgot to pass along your message and I feel bad. I should have remembered and I would have if I'd written it down immediately.",
     "I forgot to pass along your message and I feel bad. I should have remembered but honestly my phone died and I would have told them if you'd sent me a backup text."),
    
    ("was_cavalier",
     "I was cavalier about something important to you and I feel awful. I should have taken it seriously and I would have if I'd understood how much it mattered.",
     "I was cavalier about something important to you and I feel awful. I should have taken it seriously but honestly I didn't realize it was that deep and I would have been more careful if you'd told me it was a big deal."),
    
    ("was_patronizing",
     "I was patronizing when you were sharing and I feel bad. I should have just listened and I would have if I'd remembered you were looking for empathy, not advice.",
     "I was patronizing when you were sharing and I feel bad. I should have just listened but honestly I thought I was being helpful and I would have just sat with you if you'd said you didn't want solutions."),
    
    ("didnt_notice_struggle",
     "I didn't notice you were struggling and I feel terrible. I should have been more observant and I would have been if I'd been looking for the signs.",
     "I didn't notice you were struggling and I feel terrible. I should have been more observant but honestly you were keeping it really hidden and I would have noticed if you'd given me any hint that something was wrong."),
    
    ("was_unkind_about_food",
     "I was unkind about the food you made and I feel awful. I should have been more grateful and I would have been if I'd thought about the love behind it.",
     "I was unkind about the food you made and I feel awful. I should have been more grateful but honestly it wasn't to my taste and I would have been nicer if you'd asked me what I prefer."),
    
    ("wasnt_brave_enough",
     "I wasn't brave enough to have the hard conversation and I feel terrible. I should have faced it and I would have if I'd been less afraid of the outcome.",
     "I wasn't brave enough to have the hard conversation and I feel terrible. I should have faced it but honestly I was trying to protect us both and I would have talked if you'd made it feel safer."),
    
    ("forgot_apologize_sooner",
     "I didn't apologize when I realized I was wrong and I feel bad. I should have said sorry immediately and I would have if I'd been less afraid of admitting it.",
     "I didn't apologize when I realized I was wrong and I feel bad. I should have said sorry immediately but honestly I needed time to process and I would have come around sooner if you'd given me a chance."),
    
    ("was_dismissive_of_effort",
     "I was dismissive of your effort and I feel terrible. I should have been more appreciative and I would have been if I'd realized how much work went into it.",
     "I was dismissive of your effort and I feel terrible. I should have been more appreciative but honestly I didn't see the full picture and I would have been more grateful if you'd told me about everything you did."),
    
    ("wasnt_fair_to_partner",
     "I wasn't fair to you during the decision and I feel bad. I should have included you and I would have if I'd thought about how it affected us both.",
     "I wasn't fair to you during the decision and I feel bad. I should have included you but honestly I thought it was straightforward and I would have asked if you'd told me you wanted to be part of it."),
    
    ("was_inattentive",
     "I was inattentive to something that mattered to you and I feel terrible. I should have been focused and I would have been if I'd known how important it was.",
     "I was inattentive to something that mattered to you and I feel terrible. I should have been focused but honestly there was a lot going on and I would have paid more attention if you'd told me it was a priority for you."),
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
