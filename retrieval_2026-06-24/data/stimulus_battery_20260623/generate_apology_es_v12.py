#!/usr/bin/env python3
"""
v12: Both classes share EXACT SAME openings (apology + specific harm naming).
Only the TRAJECTORY differs: POS adds repair, NEG adds self-defense.
This should force TF-IDF to learn trajectory, not vocabulary.
"""
import json, random, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

random.seed(42)

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
# SHARED OPENING TEMPLATES (both classes use the same structure)
# ══════════════════════════════════════════════════════════════════

shared_openings = [
    "Lo siento mucho, {harm_verb} {harm_desc}",
    "Perdón, {harm_verb} {harm_desc}",
    "Disculpa, {harm_verb} {harm_desc}",
    "Lo siento, {harm_verb} {harm_desc}",
    "Perdón de verdad, {harm_verb} {harm_desc}",
    "Lo lamento mucho, {harm_verb} {harm_desc}",
    "Me da mucha pena haber {harm_verb} {harm_desc}",
    "Disculpa de corazón, {harm_verb} {harm_desc}",
]

# SHARED ENDINGS POOL: ambiguous trajectories
# Both POS and NEG draw from the same pool
# The classifier should NOT be able to separate by ending vocabulary

shared_endings = [
    # These are deliberately mixed: some read as repair, some as defense,
    # but ALL contain the same vocabulary (pero, si, tú, yo, también, etc.)
    "pero tú también tienes que entender que yo tengo mis limitaciones",
    "pero si tú no me hubieras dicho eso yo habría estado más tranquilo",
    "pero tú siempre asumes lo peor sin darme la oportunidad de explicarme",
    "pero yo también tengo sentimientos y no soy perfecto",
    "pero si analizamos bien verás que no soy el único involucrado",
    "pero tú también has cometido errores y no te he reclamado",
    "pero si tú me hubieras dado más tiempo todo habría salido diferente",
    "pero yo también debería haber pensado más antes de actuar",
    "pero si tú no me hubieras presionado yo habría manejado todo con más calma",
    "pero tú también te tomas las cosas muy personal conmigo",
    "pero yo también tengo mis razones para haber actuado así",
    "pero si tú no me hubieras interrumpido yo habría explicado mejor mi punto",
    "pero tú también contribuiste a que las cosas salieran así",
    "pero yo también me comprometo a mejorar si tú también lo haces",
    "pero si tú no me hubieras provocado yo habría estado más sereno",
    "pero tú no sabes todo lo que yo estaba lidiando ese día",
    "pero yo también tendría que haber sido más cuidadoso contigo",
    "pero si tú me lo hubieras recordado yo habría llegado a tiempo",
    "pero tú también dices cosas sin pensar y nadie te dice nada",
    "pero yo también me doy cuenta de que no estuvo bien lo que hice",
    "pero si tú no me hubieras dejado solo yo habría reaccionado diferente",
    "pero tú siempre quieres tener la razón y eso no es justo",
    "pero yo también estoy tratando de mejorar para ti",
    "pero si tú me lo pides yo puedo esforzarme más",
    "pero tú también tienes que reconocer tu parte en esto",
    "pero yo también reconozco que te hice daño y quiero compensarlo",
    "pero si tú no hubieras cambiado de idea yo habría seguido con el plan",
    "pero tú también viniste con una actitud que no ayudó en nada",
    "pero yo también habría hecho lo mismo en tu posición y lo sabes",
    "pero si tú me hubieras escuchado yo no habría necesitado decir eso",
]


def build_item(scenario, harm_verb, harm_desc):
    """Build a sentence: shared opening + shared ending."""
    opening = random.choice(shared_openings).format(
        harm_verb=harm_verb, harm_desc=harm_desc
    )
    ending = random.choice(shared_endings)
    return f"{opening} {ending}."


def build_pos(scenario, harm_verb, harm_desc):
    """POS: shared opening + shared ending + suffix from SHARED pool."""
    base = build_item(scenario, harm_verb, harm_desc)
    suffix = random.choice(shared_suffixes)
    return base + suffix


def build_neg(scenario, harm_verb, harm_desc):
    """NEG: shared opening + shared ending + suffix from SAME SHARED pool."""
    base = build_item(scenario, harm_verb, harm_desc)
    suffix = random.choice(shared_suffixes)
    return base + suffix


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
        'prometo': sum(1 for x in items if re.search(r'\bprometo\b', x, re.I)),
        'ya_paso': sum(1 for x in items if re.search(r'\bya pas[oó]\b', x, re.I)),
        'dramatizando': sum(1 for x in items if re.search(r'\bdramatizando\b', x, re.I)),
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
