#!/usr/bin/env python3
"""
Build sarcasm_en.json — 180 POS (sarcastic, surface-positive, context-inverted)
and 180 NEG (sincere, same positive surface vocabulary).

Anti-leakage: POS literal reading = positive. Inversion is contextual only.
Shared praise vocabulary across POS and NEG.
Target: user's TFIDF_AUC <= 0.70.
"""
import json, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ── Strategy: matched pairs ───────────────────────────────────────────────
# Each POS and NEG share the EXACT same positive opener clause.
# Contextual clause uses the SAME core vocabulary but inverted arrangement.
# This makes TF-IDF see nearly identical word distributions.

# Format: (opener, pos_context, neg_context)
# POS = opener + " — " + pos_context  (sarcastic: undermines opener)
# NEG = opener + " — " + neg_context  (sincere: reinforces opener)
ALL_PAIRS = [
    # Work / Office
    ("The meeting was productive", "we covered nothing in two hours", "we covered everything in two hours"),
    ("She delegates beautifully", "she has not touched a file in months", "she touches every file with care"),
    ("Our team is efficient", "we held three meetings for one form", "we finished one project in three days"),
    ("He communicates clearly", "his emails confused the whole team", "his emails clarified the whole plan"),
    ("The office renovation improved things", "we can see all the dust now", "we can see all the sunlight now"),
    ("The quarterly review went well", "everyone agreed I should leave", "everyone agreed I should stay"),
    ("The new policy boosted morale", "people updated their resumes after", "people signed up for more training after"),
    ("She handled that client nicely", "he walked out during the meeting", "he stayed for the full meeting"),
    ("The training was thorough", "we learned two ways to do nothing", "we learned two ways to do more"),
    ("His leadership is effective", "nobody works without his approval", "nobody doubts his approval"),
    ("The workspace upgrade helped", "the chairs squeak in different keys", "the chairs support us all day"),
    ("They value balance here", "the lights stay on past midnight", "the lights go off at five"),
    ("The brainstorming session was good", "ideas went to a wall and stayed", "ideas went to the board and grew"),
    ("Her presentation landed well", "the projector was fuzzy but spirits high", "the projector was clear and spirits high"),
    ("The intern is fitting in", "she mastered the coffee machine already", "she mastered the codebase already"),
    ("The feedback was helpful", "my manager said interesting eleven times", "my manager said impressive eleven times"),
    ("The team dinner was nice", "the food was cold and service slow", "the food was warm and service warm"),
    ("The memo was thorough", "it explained why we have no budget", "it explained how we save the budget"),
    ("The promotion is exciting", "it comes with a title and no raise", "it comes with a title and a raise"),
    ("The review was honest", "she rated me average on everything", "she rated me above average on everything"),

    # Technology
    ("The update improved things", "my laptop now boots in ten minutes", "my laptop now boots in ten seconds"),
    ("Their support is responsive", "I was on hold for three hours", "I was on hold for three minutes"),
    ("The app is well-designed", "it takes nine taps to find settings", "it takes one tap to find settings"),
    ("The AI assistant is helpful", "it speaks with confidence and inaccuracy", "it speaks with confidence and accuracy"),
    ("The new phone is a good phone", "the battery lasts for four photos", "the battery lasts for four days"),
    ("The algorithm optimized our pipeline", "it automated one task and created four", "it automated four tasks and saved time"),
    ("The cloud migration went smoothly", "we only misplaced a week of data", "we only gained a week of uptime"),
    ("The voice recognition works well", "speak slowly and enunciate every syllable", "speak normally and it understands"),
    ("The software is user-friendly", "the manual is three hundred pages", "the manual is one short page"),
    ("The security is solid", "the breach was contained within one quarter", "the breach was prevented within one hour"),
    ("The search feature is fast", "results appear within a full business day", "results appear within half a second"),
    ("The operating system is stable", "it crashes once or twice a week", "it has not crashed in several weeks"),
    ("The billing system is accurate", "it only overcharged us three times", "it has not overcharged us once"),
    ("The WiFi upgrade helped", "webpages load within a full minute", "webpages load within half a second"),
    ("The chatbot is efficient", "it resolves everything in twenty minutes", "it resolves everything in two minutes"),
    ("The integration went well", "the systems speak roughly the same language", "the systems speak exactly the same language"),
    ("The tool is intuitive", "the learning curve was only two weeks", "the learning curve was only two days"),
    ("The server upgrade helped", "it handles half our normal traffic", "it handles double our normal traffic"),
    ("The model is accurate", "it predicts correctly roughly half the time", "it predicts correctly almost every time"),
    ("The patch fixed things", "it replaced six old bugs with new ones", "it removed six bugs with no new ones"),

    # Food
    ("The restaurant was nice", "the waiter made eye contact once", "the waiter made eye contact often"),
    ("The bakery is good", "the bread stays fresh for ten minutes", "the bread stays fresh for ten hours"),
    ("The recipe was simple", "I only burned it twice before edible", "I only tried it twice before perfect"),
    ("The portions are generous", "the plate overflowed with garnish alone", "the plate overflowed with food and flavor"),
    ("The coffee shop has charm", "construction next door adds ambiance", "the garden outside adds ambiance"),
    ("The dinner party was lovely", "the power went out at dessert", "the power stayed on through dessert"),
    ("The catering was excellent", "the vegetarian option was one leaf", "the vegetarian option was three courses"),
    ("The cocktail bar is creative", "the drinks are ninety percent foam", "the drinks are ninety percent flavor"),
    ("The buffet is well-stocked", "the eggs were scrambled at some point", "the eggs were scrambled fresh this morning"),
    ("The takeout arrived fast", "the bag was warm but contents lukewarm", "the bag was warm and contents were hot"),
    ("The smoothie is healthy", "most of the flavor is added sugar", "most of the flavor is fresh fruit"),
    ("The brunch spot is popular", "the wait is only ninety minutes", "the wait is only nine minutes"),
    ("The meal was presented well", "tiny portion on an enormous plate", "generous portion on a perfect plate"),
    ("The food court has variety", "nothing particular but everything theoretical", "something excellent for every taste"),
    ("The chef is talented", "prices went up forty percent smaller plates", "prices stayed fair with bigger plates"),
    ("The pizza place delivers fast", "the driver found us on the third try", "the driver found us on the first try"),
    ("The dinner was pleasant", "the wine list was all that delivered", "the wine list paired with every course"),
    ("The Thai place is authentic", "the spice made our faces melt nicely", "the spice was exactly what we ordered"),

    # Travel
    ("The hotel had views", "from the parking lot where they moved us", "from every room facing the mountains"),
    ("The flight was smooth", "we circled the runway twice before landing", "we landed directly with no delay"),
    ("The travel agent was helpful", "she booked us next to the runway", "she booked us a quiet room with upgrades"),
    ("The cruise was relaxing", "we only hit three storms on the way", "we only saw calm seas the whole way"),
    ("The rental car was good", "it made one unusual noise on hills", "it ran smoothly the entire trip"),
    ("The road trip was scenic", "the detour added four extra hours", "the detour added four beautiful vistas"),
    ("The resort was peaceful", "the lawn crew started every morning at five", "the garden paths were quiet every morning"),
    ("The train was on time", "it was a mere forty-five minutes late", "it was exactly on the minute"),
    ("The Airbnb was charming", "the creaky floors added character to everything", "the woodwork added warmth to everything"),
    ("The lounge was comfortable", "it closed two hours before our flight", "it stayed open until our flight"),
    ("The hostel was well-placed", "the nightlife kept us awake until four", "the location kept us near everything"),
    ("The tour was informative", "the guide knew approximately three facts", "the guide knew more than any book"),
    ("The vacation was memorable", "mainly because luggage was lost entirely", "mainly because every day was perfect"),
    ("The safari was exciting", "the closest wildlife was the parking meter", "the closest wildlife was ten feet away"),
    ("The bed and breakfast was cozy", "the mattress was on the floor", "the mattress was memory foam and soft"),

    # Health
    ("The gym has equipment", "the treadmills date back to the nineties", "the treadmills are brand new this year"),
    ("The doctor was careful", "the waiting room let me finish two magazines", "the doctor let me ask every question"),
    ("The diet plan works", "enjoy the variety of plain chicken", "enjoy the variety of seasoned meals"),
    ("The new trainer is energetic", "he only yelled three times each session", "he only praised us three times each set"),
    ("The yoga class was calm", "neighbor construction enhanced our meditation", "soft music enhanced our meditation"),
    ("The health app is accurate", "it misses only half the steps taken", "it catches every single step taken"),
    ("The smoothie bar has options", "one does not taste like grass", "every one tastes like fresh fruit"),
    ("The wellness retreat was helpful", "I came back about as stressed", "I came back half as stressed"),
    ("The therapy is going well", "my arm can almost wave goodbye to money", "my arm can almost lift its full weight"),
    ("The meditation app found peace", "mostly about the subscription pricing", "mostly about calming the mind itself"),
    ("The running group is fun", "they finish four miles before I finish one", "they run four miles with me every time"),
    ("The nutrition plan is healthy", "four of five meals are plain vegetables", "four of five meals are colorful balanced plates"),
    ("The health screening was painless", "only the bill caused real discomfort", "only the results brought real comfort"),
    ("The pool is clean", "the chlorine removed my hair color", "the chlorine keeps the water crystal clear"),
    ("The fitness tracker is helpful", "it shames you into walking extra blocks", "it encourages you into walking extra blocks"),

    # Social
    ("He is a good listener", "he waits for his turn to speak", "he remembers what you said last month"),
    ("She is supportive", "she gives advice whether you want it", "she gives help whether you ask for it"),
    ("They are compatible", "they agree on nothing except being right", "they agree on almost everything together"),
    ("He is honest", "he told my mother her cooking was poor", "he told my mother her cooking was superb"),
    ("They are solid", "they power through every fight with volume", "they power through every challenge with care"),
    ("She is easygoing", "she only requires daily flowers and dinners", "she genuinely enjoys simple plans and company"),
    ("Our friendship is strong", "we have not spoken in three years", "we have not missed a week in three years"),
    ("He is generous", "he spent forty minutes talking about himself", "he spent forty minutes helping me move"),
    ("She is independent", "she has not needed anyone since last favor", "she has not needed help since she learned"),
    ("The gathering was warm", "only two relatives started a shouting match", "only two relatives missed the whole thing"),
    ("He is punctual", "he arrives right when the food is gone", "he arrives right when the food is served"),
    ("She is a fine cook", "her smoke alarm confirms her talent nightly", "her dinner table confirms her talent nightly"),
    ("Their relationship is enviable", "they post about it more than live it", "they live it more than they post about it"),
    ("He is thoughtful", "he remembered the birthday forgot the gift", "he remembered the birthday and the gift"),
    ("She is patient", "her kids only ran away twice this week", "her kids only smiled more each time"),

    # Education
    ("The lecture was engaging", "about forty percent stayed awake", "about ninety percent stayed engaged"),
    ("The professor is knowledgeable", "he last updated slides around two thousand five", "he updates slides with new research each term"),
    ("The course is organized", "assignments overlap with three other classes", "assignments build on each previous lesson"),
    ("The textbook is thorough", "it covers every topic except those on exam", "it covers every topic the exam addresses"),
    ("The school has strong programs", "the football team wins more than students", "both academics and arts receive strong support"),
    ("The teacher is patient", "she explained the same thing forty-seven times", "she explains until every student understands"),
    ("The research lab is funded", "we share one microscope among twelve", "every student has their own working equipment"),
    ("The semester was productive", "I learned more from the group chat", "I learned more from lectures than expected"),
    ("The library is popular", "the coffee shop gets more visitors", "the quiet zones get more visitors daily"),
    ("The ceremony was inspiring", "mostly because it ended all future exams", "mostly because it celebrated real hard work"),
    ("The online course is flexible", "deadlines are more like suggestions", "deadlines are fair and support is there"),
    ("The curriculum is modern", "it prepares for jobs from nineteen ninety", "it teaches skills employers need today"),
    ("The study group helps", "we study one hour and eat for three", "we study three hours and eat for one"),
    ("The advisor is supportive", "she recommended every major except mine", "she recommended the perfect major for me"),

    # Customer Service
    ("The support was helpful", "they fixed my issue after seven calls", "they fixed my issue on the first call"),
    ("The return policy is generous", "you get a full forty-eight hours", "you get a full thirty days"),
    ("The clerk was friendly", "he pointed me to the wrong aisle smiling", "he found exactly what I needed smiling"),
    ("The help desk is responsive", "replies arrive in two business days", "replies arrive within one business hour"),
    ("The warranty covers everything", "except the one part that actually broke", "including the one part I was worried about"),
    ("The install went well", "the technician needed three return visits", "the technician finished in one clean visit"),
    ("The subscription is a deal", "canceling takes six phone calls minimum", "canceling takes one single click"),
    ("The repair was affordable", "the labor cost more than the product", "the labor cost half of what I expected"),
    ("The delivery was on time", "it arrived three weeks after the estimate", "it arrived exactly when the estimate promised"),
    ("The experience was seamless", "I repeated my story only four times", "I told my story only one single time"),

    # Entertainment
    ("The movie was good", "the twist was visible from the opening scene", "the twist surprised the entire audience"),
    ("The concert was fun", "the speakers let us hear nearby conversations", "the speakers let us hear every note clearly"),
    ("The play was bold", "the lead forgot two lines and improvised", "the lead delivered every line flawlessly"),
    ("The comedy show was fun", "half the jokes hit and half missed", "half the jokes hit and the rest hit harder"),
    ("The museum was interesting", "the gift shop was the best exhibit", "the interactive wing was the best exhibit"),
    ("The podcast is useful", "each episode is three hours of one voice", "each episode is thirty minutes of real insight"),
    ("The festival was organized", "the bathroom line was only two hours", "the bathroom line was only two minutes"),
    ("The documentary was eye-opening", "it made me avoid my bank statement", "it made me understand the whole issue"),
    ("The theater production was brave", "performed in the dark to save lighting", "performed in the round with creative lighting"),
    ("The gallery was curated", "the guard followed me more than the art", "the curator guided me through every piece"),

    # Home
    ("The plumber was quick", "he fixed the leak and started a new one", "he fixed the leak and checked everything"),
    ("The neighbor is friendly", "his music entertains the whole block nightly", "his welcome brought the whole block together"),
    ("The house is spacious", "you can hear every conversation from anywhere", "you can hear peace and quiet from anywhere"),
    ("The garden is growing", "the weeds are the tallest healthiest plants", "the tomatoes are the tallest healthiest plants"),
    ("The cleaning crew was thorough", "they moved things and forgot to return", "they moved things and returned them all"),
    ("The contractor was reliable", "he only missed three consecutive deadlines", "he only missed one minor deadline once"),
    ("The apartment is quiet", "you can hear every word from next door", "you can barely hear anything from outside"),
    ("The renovation is on schedule", "the schedule keeps getting updated weekly", "the schedule stays fixed every single week"),
    ("The handyman is handy", "he fixed the door and floor now squeaks", "he fixed the door and it opens smoothly"),
    ("The landscaping is professional", "the mower broke on its second pass", "the mower finished the whole yard cleanly"),

    # Sports
    ("The team played well", "they only lost by thirty comfortable points", "they only won by thirty comfortable points"),
    ("The coach is smart", "he called the same play four times straight", "he called four different plays in a row"),
    ("The ref was fair", "he missed calls for both teams equally", "he made every call for both teams equally"),
    ("The race was smooth", "I finished in the top half of the bottom", "I finished in the top third of the field"),
    ("The athlete is talented", "she set a record for time on the bench", "she set a record for time on the track"),
    ("The game was close", "we were within reach in the first quarter", "we were within reach the entire game"),
    ("The tournament was organized", "the brackets were redone only twice", "the brackets were perfect from the start"),
    ("The strategy worked", "it was ideal for the opposing team", "it was ideal for our team all along"),
    ("The team is balanced", "offense scores and defense stops sometimes", "offense scores and defense stops consistently"),
    ("The team morale is high", "everyone agrees we tried our best", "everyone agrees we did our best and won"),

    # Finance
    ("The investment is paying off", "the returns almost match the fees", "the returns easily exceed the fees"),
    ("The budget is managed", "we are only slightly over in every category", "we are slightly under in every category"),
    ("The advisor was insightful", "he recommended his own personal fund", "he recommended the strongest performing fund"),
    ("The retirement plan is solid", "ready in about forty more years", "ready ahead of schedule for once"),
    ("The tax prep went well", "the refund covered the accountant fee", "the refund covered our vacation this year"),
    ("The mortgage is manageable", "the rate adjusts upward every six months", "the rate stays fixed every single month"),
    ("The stock pick was smart", "it doubled its losses from last quarter", "it doubled its gains from last quarter"),
    ("The savings account is growing", "interest earned a nice pack of gum", "interest earned a meaningful quarterly sum"),
    ("The business plan is thorough", "projections assume everything goes perfectly", "projections account for every realistic scenario"),
    ("The cost-cutting is working", "morale dropped but spreadsheets look solid", "morale stayed high and spreadsheets look solid"),

    # Transport
    ("The commute is quick", "the drive takes twice as long as expected", "the drive takes half as long as expected"),
    ("The bus is reliable", "it arrives within a fifteen-minute window", "it arrives within two minutes of schedule"),
    ("The parking is easy", "the lot is a ten-minute walk away", "the lot is right next to the building"),
    ("The highway helped", "it now jams three miles earlier than before", "it now clears three miles earlier than before"),
    ("The carpool is working", "nobody agrees on the pickup time", "nobody disagrees on the pickup time"),
    ("The train is punctual", "enjoy guessing which platform to use today", "the platform is clearly marked every time"),
    ("The bike lane is smart", "it merges with traffic for one brave block", "it connects the neighborhood to downtown"),
    ("The rideshare came fast", "the driver was confident about the destination", "the driver knew the fastest route by heart"),
    ("The taxi was clean", "the air freshener concealed a deeper mystery", "the fresh air made the ride pleasant"),
    ("The roads are smooth", "the potholes are large enough to swerve", "the repaving made every drive genuinely pleasant"),

    # Safety
    ("The security is high-tech", "it alerts us to every passing squirrel", "it alerts us to every real approaching threat"),
    ("The area is safe", "police patrol more often than residents walk", "families walk together in the evenings"),
    ("The alarm is sensitive", "the wind sets it off on breezy nights", "it detects real threats before they spread"),
    ("The fire drill went well", "everyone left at a leisurely walking pace", "everyone left the building in three minutes"),
    ("The safety plan is thorough", "one extinguisher for the whole building", "every floor has clearly marked emergency exits"),
    ("The cameras cover everything", "except the entrance they were meant for", "including the entrance and the parking lot"),
    ("The locks are strong", "the burglar was impressed and left a note", "the burglar never even tried the door"),
    ("The emergency plan is documented", "last updated before the building renovation", "updated last month with fresh contacts"),

    # Misc
    ("The volunteer day was rewarding", "we spent more time organizing than helping", "we spent more time helping than organizing"),
    ("The policy is inclusive", "it excludes the people it aimed to help", "it includes every person it was meant to help"),
    ("The workshop was hands-on", "we raised hands to ask unanswered questions", "we built real projects by the end"),
    ("The survey results are good", "most people did not finish the survey", "most people rated us very highly"),
    ("The app notification is timely", "it arrived three days after the event", "it arrived exactly when we needed it"),
    ("The garden is thriving", "the tomatoes grow well among the wildflowers", "the tomatoes grow well among the herbs"),
    ("The fundraiser is on track", "we raised enough for the thank-you cards", "we raised enough for the whole program"),
    ("The new rule is simple", "only nine conditions need to be met first", "only three conditions need to be met first"),
    ("The watch program is active", "we watch each other leave for work", "we look out for each other every night"),
    ("The recycling program works", "the bins get collected when someone remembers", "the bins get collected every single week"),
]

PAIRS = ALL_PAIRS[:180]

# Build POS and NEG from pairs
POS = [f"{opener} — {pos_ctx}." for opener, pos_ctx, neg_ctx in PAIRS]
NEG = [f"{opener} — {neg_ctx}." for opener, pos_ctx, neg_ctx in PAIRS]


def evaluate_user_script(pos, neg):
    """Exact replication of user's evaluation script."""
    X = pos + neg
    y = [1]*len(pos) + [0]*len(neg)
    v = TfidfVectorizer(ngram_range=(1,2), min_df=2).fit_transform(X)
    auc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), v, y, cv=5, scoring="roc_auc").mean()
    return auc


if __name__ == "__main__":
    assert len(POS) == 180, f"POS={len(POS)}"
    assert len(NEG) == 180, f"NEG={len(NEG)}"

    # Verify no banned words in POS
    banned = ['/s','obviously','sure','great','fantastic','thrilled','perfect','wow','just']
    violations = []
    for i, s in enumerate(POS):
        sl = s.lower()
        for b in banned:
            if re.search(r'\b' + re.escape(b) + r'\b', sl):
                violations.append((i, b, s[:60]))
    if violations:
        for v in violations:
            print(f"VIOLATION POS[{v[0]}]: banned=\"{v[1]}\" in \"{v[2]}\"")
        raise SystemExit(1)

    auc = evaluate_user_script(POS, NEG)
    print(f"TFIDF_AUC={auc:.4f}")

    out = {"pos": POS, "neg": NEG, "meta": {
        "n_pos": len(POS), "n_neg": len(NEG),
        "tfidf_auc": round(auc, 4),
        "banned_in_pos": banned,
    }}
    path = os.path.join(os.path.dirname(__file__), "sarcasm_en.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")
