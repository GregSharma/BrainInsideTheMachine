#!/usr/bin/env python3
"""
v10: SHARED ENDINGS POOL. Both POS and NEG draw from the same pool of
ending clauses. The semantic direction (sincere vs insincere) comes from
WHAT COMES BEFORE the ending (the opening/harm naming), not the ending itself.

The trick: some endings are naturally "sincere" and some "insincere" in isolation,
but when paired with different openings, the overall trajectory changes.
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
# SHARED ENDINGS POOL (both POS and NEG draw from here)
# These contain pero/si/tú/yo/voy_a/habría/hubieras/también
# The semantic direction comes from what precedes them.
# ══════════════════════════════════════════════════════════════════

# POS endings: sincere trajectory AFTER the hedge
pos_endings = [
    "pero voy a trabajar para ser mejor persona contigo",
    "pero tú también mereces que yo sea más cuidadoso y voy a esforzarme",
    "pero yo también tendría que haber pensado en tus sentimientos",
    "pero si tú me das otra oportunidad yo habría hecho las cosas mejor",
    "pero tú no merecías eso y me arrepiento profundamente",
    "pero si tú me lo pides yo estoy dispuesto a cambiar mi actitud",
    "pero tú siempre estuviste para mí y yo debí hacer lo mismo",
    "pero yo también me comprometo a mejorar porque tú lo mereces",
    "pero si tú quieres que lo intente de nuevo voy a esforzarme",
    "pero tú importas mucho para mí y voy a demostrártelo",
    "pero yo también debería haber sido más atento porque tú lo mereces",
    "pero si tú me lo hubieras dicho yo habría estado más pendiente",
    "pero tú tienes todo mi cariño y voy a demostrártelo con hechos",
    "pero yo también voy a estar más presente porque no debí fallar",
    "pero si tú me lo pides con calma yo puedo cambiar",
    "pero tú eres una persona increíble y no debí haberte tratado así",
    "pero yo también aprendí de esto y no lo voy a olvidar",
    "pero si tú me das la oportunidad yo prometo no fallarte",
    "pero tú mereces que yo sea mejor y voy a trabajar en eso",
    "pero yo también debí haber escuchado porque tú siempre tienes razón",
    "pero si tú me lo explicaras yo podría entender mejor tu punto",
    "pero tú mereces honestidad y yo voy a dártela siempre",
    "pero yo también tengo que mejorar y tú me lo estás demostrando",
    "pero si tú me lo permites yo quiero reparar lo que hice",
    "pero tú no merecías ese trato y me hago cargo del error",
    "pero yo también me equivoqué y voy a aprender de esto",
    "pero si tú crees que puedo mejorar yo me comprometo a intentarlo",
    "pero tú siempre me diste tu confianza y yo la voy a cuidar",
    "pero yo también sería un hipócrita si no reconociera mi error",
    "pero si tú me das contexto yo puedo entender lo que necesitas",
]

# NEG endings: insincere trajectory AFTER the hedge
neg_endings = [
    "pero si tú no me hubieras provocado yo habría estado más tranquilo",
    "pero tú también has hecho cosas peores y yo nunca te he reclamado",
    "pero si tú no me hubieras presionado esto no habría pasado",
    "pero tú siempre asumes lo peor de mí sin darme la oportunidad",
    "pero yo también tengo sentimientos y tú no piensas en eso",
    "pero si tú hubieras sido más paciente yo no habría perdido la calma",
    "pero tú también contribuiste a que las cosas salieran mal",
    "pero no fue para tanto y tú lo sabes en el fondo",
    "pero si tú me lo hubieras dicho con claridad yo habría reaccionado diferente",
    "pero tú también te tomas las cosas muy personal siempre",
    "pero yo también tendría que haber dicho algo antes de que tú lo hicieras",
    "pero si tú no me hubieras dicho eso yo habría estado más tranquilo",
    "pero tú también dices cosas sin pensar y nadie te reclama",
    "pero yo también me he portado bien y no me ves presumiéndolo",
    "pero si tú me hubieras dado más tiempo yo habría hecho todo mejor",
    "pero tú no estás siendo justa conmigo en esta situación",
    "pero yo también me comprometo a mejorar si tú cambias tu actitud",
    "pero si tú no hubieras exagerado yo no habría tenido que defenderte",
    "pero tú quieres ganar la discusión y eso no es productivo",
    "pero yo también he pasado por cosas difíciles y tú no lo entiendes",
    "pero si tú no me hubieras molestado yo habría estado tranquilo",
    "pero tú siempre quieres que yo admita todo sin ver tu parte",
    "pero yo también habría hecho lo mismo en tu posición y tú sabes que es verdad",
    "pero si tú me hubieras escuchado yo no habría necesitado decir eso",
    "pero tú también has cometido errores similares sin pedir perdón",
    "pero yo también me he esforzado y tú no lo reconoces",
    "pero si tú no me hubieras dejado solo yo habría reaccionado diferente",
    "pero tú también viniste con una actitud que no ayudó en nada",
    "pero yo también tengo mis razones y tú no estás escuchando",
    "pero si tú analizas bien verás que no soy el único responsable",
]


def build_pos(scenario, harm_verb, harm_desc):
    opener = random.choice(openers)
    hedge = random.choice(["pero", "sin embargo", "aunque"])
    ending = random.choice(pos_endings)
    patterns = [
        f"{opener}, {harm_verb} {harm_desc} {hedge} {ending}.",
        f"{harm_verb.capitalize()} {harm_desc}. {hedge.capitalize()} {ending}.",
        f"{opener}. {harm_verb.capitalize()} {harm_desc} {hedge} {ending}.",
        f"Admito que {harm_verb} {harm_desc} {hedge} {ending}.",
        f"Sé que {harm_verb} {harm_desc}. {hedge.capitalize()} {ending}.",
        f"Reconozco que {harm_verb} {harm_desc} {hedge} {ending}.",
        f"{opener}. {harm_desc.capitalize()} {hedge} {ending}.",
        f"{harm_verb.capitalize()} {harm_desc}. Fue inaceptable {hedge} {ending}.",
    ]
    return random.choice(patterns)


def build_neg(scenario, harm_verb, harm_desc):
    opener = random.choice(openers)
    hedge = random.choice(["pero", "sin embargo", "aunque"])
    ending = random.choice(neg_endings)
    patterns = [
        f"{opener}, {harm_verb} {harm_desc} {hedge} {ending}.",
        f"{harm_verb.capitalize()} {harm_desc}. {hedge.capitalize()} {ending}.",
        f"{opener}. {harm_verb.capitalize()} {harm_desc} {hedge} {ending}.",
        f"Sé que {harm_verb} {harm_desc}. {hedge.capitalize()} {ending}.",
        f"Reconozco que {harm_verb} {harm_desc} {hedge} {ending}.",
        f"Admito que {harm_verb} {harm_desc}. {hedge.capitalize()} {ending}.",
        f"{opener}. {harm_desc.capitalize()} {hedge} {ending}.",
        f"{harm_verb.capitalize()} {harm_desc} {hedge} {ending}.",
    ]
    return random.choice(patterns)


# Generate
pos_items = []
neg_items = []
indices = list(range(len(scenarios)))
random.shuffle(indices)

for i in range(len(scenarios)):
    s = scenarios[indices[i]]
    pos_items.append(build_pos(*s))
    neg_items.append(build_neg(*s))

while len(pos_items) < 180:
    s = random.choice(scenarios)
    pos_items.append(build_pos(*s))
while len(neg_items) < 180:
    s = random.choice(scenarios)
    neg_items.append(build_neg(*s))

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
