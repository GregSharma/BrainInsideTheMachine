#!/usr/bin/env python3
"""
v9: SHARED TEMPLATE approach. Both POS and NEG use the SAME sentence
opening/hedging. The ONLY difference is the final clause.
This breaks structural TF-IDF leakage because bigrams like "pero voy",
"tú no", "mereces" appear in BOTH classes.

Key: the classifier must learn trajectory, not vocabulary.
"""
import json, random, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

random.seed(42)

openers = [
    "Lo siento mucho", "Perdón", "Disculpa", "Lo siento", "Me da mucha pena",
    "Perdón de verdad", "Lo lamento mucho", "Te pido disculpas",
]

scenarios = [
    ("tu cumpleaños", "olvidé", "fue terrible que no te recordara"),
    ("la reunión", "critiqué tu idea", "estuvo mal porque tenías un punto válido"),
    ("la cena familiar", "fui rudo con tu mamá", "fue irrespetuoso y lo sé"),
    ("el fin de semana", "llegué tarde", "desperdicié tu tiempo sin necesidad"),
    ("la confianza", "te mentí", "fue una traición"),
    ("tu presentación", "no estuve presente", "debí estar ahí apoyándote"),
    ("la fiesta", "fui antipático con tus amigos", "hice pasar un mal rato"),
    ("tu proyecto", "no te ayudé", "dejé que cargaras con todo solo"),
    ("la receta", "olvidé recoger tu receta", "no te di la medicina que necesitabas"),
    ("las finanzas", "gasté dinero sin consultarte", "fue irresponsable de mi parte"),
    ("tu hermana", "fui grosero con tu hermana", "fue innecesario y lo admito"),
    ("el plan", "cancelé sin avisar", "fue irrespetuoso contigo"),
    ("la comida", "critiqué la comida", "no valoré tu esfuerzo"),
    ("la reservación", "arruiné la reservación", "fue un desastre de mi parte"),
    ("la habitación", "dejé todo desordenado", "no respeté nuestro espacio"),
    ("la conversación", "la evité por tres días", "fue cobarde no enfrentarla"),
    ("el regalo", "no fue personalizado", "no puse esfuerzo en pensarlo"),
    ("la excursión", "te dejé solo toda la noche", "fue irrespetuoso contigo"),
    ("el mensaje", "compartí tu mensaje privado", "fue una violación a tu privacidad"),
    ("la cena cocinada", "no la probé", "rechazar tu esfuerzo fue hiriente"),
    ("la graduación", "no fui a tu graduación", "te dejé solo en un momento importante"),
    ("la promesa", "no la cumplí", "mi palabra perdió valor"),
    ("tu tiempo", "lo llené de compromisos sin preguntar", "asumí que podías sin respetar tu tiempo"),
    ("la foto", "comenté algo negativity", "hice que te sintieras mal en público"),
    ("la cena programada", "me dormí y no llegué", "te dejé esperando sin aviso"),
    ("tu problema del trabajo", "no te escuché", "estuve distraído mientras hablabas"),
    ("tu jefe", "no te apoyé", "debí estar de tu lado y me quedé callado"),
    ("la tradición", "me burlé de la tradición navideña", "fue irrespetuoso con algo que tú valoras"),
    ("el auto", "fui un mal conductor", "te asusté con mi forma de manejar"),
    ("tu presentación del trabajo", "no te felicité", "ignoré tu logro y eso estuvo mal"),
    ("el hotel", "dejé la llave adentro", "creé un problema innecesario"),
    ("el supermercado", "compré cosas innecesarias", "gasté dinero sin sentido"),
    ("tu abuela", "no atendí su llamada", "fue irrespetuoso con alguien que te quiere"),
    ("la sorpresa", "la arruiné sin querer", "destruí tu esfuerzo"),
    ("el restaurante", "critiqué el restaurante", "no valoré tu elección"),
    ("la carta", "la leí en voz alta", "violé algo que era privado"),
    ("el vecino", "te culpé a ti", "fue injusto porque no fue tu culpa"),
    ("limpiar juntos", "me fui a mi cuarto", "dejé que limpiaras todo sola"),
    ("la fiesta sorpresa", "conté la fiesta sorpresa", "arruiné la sorpresa"),
    ("el aniversario", "llegué con ropa sucia", "no me importó prepararme para ti"),
    ("los boletos", "perdí los boletos", "arruiné un plan que organizaste"),
    ("tus padres", "fui distraído con el celular", "no presté atención y fue irrespetuoso"),
    ("hoy", "me desperté tarde", "te dejé esperando sin excusa"),
    ("la casa", "decidí sin consultarte", "no consulté algo que nos afecta"),
    ("la comida compartida", "comí todo sin dejarte nada", "fui egoísta y no pensé en ti"),
    ("los sueños", "me burlé de tus sueños", "fue cruel minimizar lo que deseas"),
    ("los niños", "no ayudé con la tarea escolar", "dejé todo el trabajo sobre ti"),
    ("el mensaje urgente", "no te lo mandé a tiempo", "ignoré algo que te importaba"),
    ("las vacaciones", "lo cambié sin consultarte", "asumí que estabas de acuerdo"),
    ("la llamada semanal", "no te llamé como prometí", "mi palabra no valió nada"),
]

# ══════════════════════════════════════════════════════════════════
# SHARED CLAUSE FRAGMENTS (both POS and NEG use these)
# ══════════════════════════════════════════════════════════════════

# Shared opening clause fragments
shared_openings = [
    "{opener}, {harm_verb} {harm_desc} y",
    "{opener}. {harm_verb} {harm_desc} y",
    "Admito que {harm_verb} {harm_desc} y",
    "Sé que {harm_verb} {harm_desc} y",
    "Reconozco que {harm_verb} {harm_desc} y",
    "{harm_verb} {harm_desc}.",
    "{opener}. {harm_desc} y",
]

# Shared hedging connectors (BOTH classes use these)
shared_hedges = [
    "pero", "sin embargo", "aunque", "pero si tú me lo pides",
    "solo que tú no entiendes que", "aunque creo que tú",
    "pero tú también", "pero yo también", "solo que",
]

# POS repair endings (appear after shared hedge)
# Mix: ~40% voy_a, ~20% prometo/quiero, ~20% debo/es mi culpa, ~20% no voy_a
pos_endings = [
    # "voy a" (~40%)
    "voy a trabajar para ser mejor persona contigo",
    "voy a poner todo de mi parte para no fallarte otra vez",
    "voy a hacerme cargo de mi error sin buscar excusas",
    "la próxima vez voy a actuar con más cuidado y respeto",
    "voy a escucharte y respetar lo que sientes porque es importante",
    "voy a esforzarme por ser la persona que tú te mereces",
    "voy a tomar esto como una lección para mejorar",
    "voy a demostrarte con hechos que puedo ser mejor",
    "voy a recordar esta experiencia para no repetir el error",
    "voy a ser más consciente de cómo mis acciones te afectan",
    "voy a tratar de ser más atento contigo de ahora en adelante",
    "voy a valorar tu paciencia y esforzarme por no fallarte",
    "voy a demostrarte que puedo ser digno de tu confianza",
    "voy a estar más pendiente de ti porque no debí fallar",
    # "prometo/quiero/me comprometo" (~20%)
    "me comprometo a cambiar mi actitud porque tú lo mereces",
    "quiero que sepas que aprendí de esto y no lo voy a olvidar",
    "mi compromiso es cuidar lo que tenemos porque tú importas",
    "me propongo ser más cuidadoso con tus sentimientos",
    "prometo aprender de esto y crecer como persona a tu lado",
    # "debo/es mi culpa/me hago cargo" (~20%)
    "debo admitir que te hice daño y quiero repararlo",
    "es mi culpa y voy a trabajar para que no vuelva a pasar",
    "me hago responsable porque fue un error inaceptable",
    "debo reconocer que te fallé y voy a esforzarme por ser mejor",
    "fue mi culpa y no hay excusa pero voy a cambiar",
    # no "voy a" alternative repairs (~20%)
    "tú mereces que yo sea mejor persona y lo sé",
    "tú no merecías eso y me arrepiento profundamente",
    "tú importas mucho para mí y voy a demostrártelo con hechos",
    "tú siempre estuviste para mí y yo debí hacer lo mismo",
    "esto me enseñó algo importante y no lo voy a olvidar jamás",
]

# NEG defense endings (appear after same shared hedges)
# Mix: ~25% pero blame, ~25% si conditional, ~20% tú blame, ~15% minimize, ~15% voy_a self-defense
neg_endings = [
    # pero blame (~25%)
    "pero si analizamos esto verás que no soy el único responsable",
    "pero tú no sabes lo que yo estaba pasando internamente",
    "pero si tú no me hubieras provocado yo habría estado más tranquilo",
    "pero tú siempre asumes lo peor de mí sin darme la oportunidad",
    "pero yo también tengo sentimientos y tú no piensas en eso",
    "pero si tú no me hubieras presionado esto no habría pasado",
    "pero a veces hay que decir las cosas como son sin rodeos",
    "pero si analizas bien yo tenía mis razones para actuar así",
    "pero si tú no me hubieras dicho eso yo habría estado más tranquilo",
    # si conditional blame (~25%)
    "si tú no me hubieras provocado yo habría estado más tranquilo",
    "si tú hubieras sido más paciente yo no habría perdido la calma",
    "si tú me lo hubieras dicho con claridad yo habría reaccionado diferente",
    "si tú no me hubieras presionado esto no habría pasado",
    "si tú me hubieras dado más tiempo yo habría hecho todo mejor",
    "si tú no me hubieras dicho eso yo habría estado más tranquilo",
    "si tú no hubieras exagerado yo no habría tenido que defenderte",
    "si tú me hubieras escuchado yo no habría necesitado decir eso",
    # tú blame (~20%)
    "tú también has hecho cosas peores y nunca te he reclamado",
    "tú también contribuiste a que las cosas salieran mal",
    "tú siempre te tomas las cosas muy personal siempre",
    "tú también dices cosas sin pensar y nadie te reclama",
    "tú también me has fallado antes y yo te he perdonado",
    "tú no estás siendo justa conmigo en esta situación",
    "tú quieres ganar la discusión y eso no es productivo para nadie",
    # minimize (~15%)
    "no fue para tanto y tú lo sabes en el fondo",
    "estás siendo demasiado sensible con algo que no es tan grave",
    "ya pasó y no tiene sentido seguir dándole vueltas",
    "solo que tú no estás siendo justa conmigo en esta situación",
    "aunque tú también viniste con una actitud que no ayudó",
    # "voy a" self-defense (~15%)
    "voy a decirte la verdad aunque no te guste escucharla",
    "voy a ser honesto contigo porque tú no ves todo el panorama",
    "voy a explicarte mi lado aunque tú no quieras escuchar",
    "voy a tratar de mejorar pero tú también tienes que hacer tu parte",
    "voy a escucharte pero si tú no me das la oportunidad no puedo hacer nada",
    "voy a intentar ser más paciente pero tú siempre me sacas de quicio",
]


def build_shared(hedge, ending, scenario, harm_verb, harm_desc):
    """Build a sentence from shared opening + hedge + ending."""
    opener = random.choice(openers)
    hv_cap = harm_verb.capitalize()
    opening = random.choice(shared_openings).format(
        opener=opener, harm_verb=hv_cap, harm_desc=harm_desc
    )
    return f"{opening} {hedge} {ending}."


# Generate
pos_items = []
neg_items = []
indices = list(range(len(scenarios)))
random.shuffle(indices)

for i in range(len(scenarios)):
    s = scenarios[indices[i]]
    hedge = random.choice(shared_hedges)
    pos_end = random.choice(pos_endings)
    neg_end = random.choice(neg_endings)
    pos_items.append(build_shared(hedge, pos_end, *s))
    neg_items.append(build_shared(hedge, neg_end, *s))

while len(pos_items) < 180:
    s = random.choice(scenarios)
    hedge = random.choice(shared_hedges)
    pos_items.append(build_shared(hedge, random.choice(pos_endings), *s))
while len(neg_items) < 180:
    s = random.choice(scenarios)
    hedge = random.choice(shared_hedges)
    neg_items.append(build_shared(hedge, random.choice(neg_endings), *s))

pos_items = pos_items[:180]
neg_items = neg_items[:180]


def audit_features(items):
    return {
        'siento': sum(1 for x in items if re.search(r'\bsiento\b', x, re.I)),
        'perdón': sum(1 for x in items if re.search(r'\bperd[oó]n\b', x, re.I)),
        'pero': sum(1 for x in items if re.search(r'\bpero\b', x, re.I)),
        'si': sum(1 for x in items if re.search(r'\bsi\b', x, re.I)),
        'voy_a': sum(1 for x in items if re.search(r'\bvoy a\b', x, re.I)),
        'tu': sum(1 for x in items if re.search(r'\btú\b', x, re.I)),
        'yo': sum(1 for x in items if re.search(r'\byo\b', x, re.I)),
        'también': sum(1 for x in items if re.search(r'\btambi[eé]n\b', x, re.I)),
        'habría': sum(1 for x in items if re.search(r'\bhabr[ií]a\b', x, re.I)),
        'hubieras': sum(1 for x in items if re.search(r'\bhubieras\b', x, re.I)),
        'mereces': sum(1 for x in items if re.search(r'\bmereces\b', x, re.I)),
        'prometo': sum(1 for x in items if re.search(r'\bprometo\b', x, re.I)),
    }

def evaluate(pos_items, neg_items):
    d = {"pos": pos_items, "neg": neg_items, "contrast": "apology", "lang": "es"}
    X = d["pos"] + d["neg"]
    y = [1] * len(d["pos"]) + [0] * len(d["neg"])
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(X)
    auc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), v, y, cv=5, scoring="roc_auc").mean()

    v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(X)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(v2.transform(X), y)
    fn = v2.get_feature_names_out()
    coefs = lr.coef_[0]

    print(f"  TFIDF_AUC={auc:.4f}")
    print(f"  POS features: {audit_features(pos_items)}")
    print(f"  NEG features: {audit_features(neg_items)}")
    print("  Top 10 NEG:", [(fn[i], f"{coefs[i]:.3f}") for i in np.argsort(coefs)[:10]])
    print("  Top 10 POS:", [(fn[i], f"{coefs[i]:.3f}") for i in np.argsort(coefs)[-10:][::-1]])
    return auc

print("=== Iteration 1 ===")
auc = evaluate(pos_items, neg_items)

# Save
out_path = "/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/apology_es.json"
d = {"pos": pos_items, "neg": neg_items, "contrast": "apology", "lang": "es"}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")
print(f"N_pos={len(pos_items)} N_neg={len(neg_items)} iterations=1")
