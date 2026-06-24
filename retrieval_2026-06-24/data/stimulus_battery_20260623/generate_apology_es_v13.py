#!/usr/bin/env python3
"""
v13: Both classes use the SAME vocabulary. The difference is trajectory.

Key insight from v12 analysis: when both classes share openings, TF-IDF
AUC drops. The remaining leakage is from ENDINGS (repair vs defense).

Solution: use SHARED sentence structures where the second half can be
read as either repair or defense depending on context. The classifier
must learn trajectory from COMBINATION, not individual tokens.
"""
import json, random, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

random.seed(42)

# ══════════════════════════════════════════════════════════════════
# APPROACH: Paired generation
# Each item = opening + harm + trajectory
# Opening and harm are SHARED (same vocabulary)
# Trajectory is what differs, but uses SAME vocabulary
# ══════════════════════════════════════════════════════════════════

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
# SHARED TRAJECTORY CLAUSES: same vocabulary, different direction
# These use pero/si/tú/yo/también/voy_a in both classes
# ══════════════════════════════════════════════════════════════════

# POS trajectory: repair-focused but using the same connector vocabulary
pos_trajectories = [
    "pero tú también te mereces que yo sea más cuidadoso y voy a esforzarme",
    "pero si tú me das otra oportunidad yo habría hecho las cosas mejor",
    "pero tú no merecías eso y yo también debí haber pensado en tus sentimientos",
    "pero si tú me lo pides yo puedo cambiar mi actitud porque tú lo mereces",
    "pero tú siempre estuviste para mí y yo debí hacer lo mismo sin dudar",
    "pero yo también tendría que haber sido más atento porque tú lo mereces",
    "pero si tú me lo hubieras dicho yo habría estado más pendiente tuyo",
    "pero tú tienes todo mi cariño y yo también voy a demostrártelo con heches",
    "pero si tú quieres que lo intente de nuevo yo voy a esforzarme más",
    "pero tú importas mucho para mí y yo también debería haberlo demostrado antes",
    "pero yo también tendría que haber escuchado porque tú siempre tienes razón",
    "pero si tú me lo explicaras yo podría entender mejor tu punto de vista",
    "pero tú mereces honestidad y yo también voy a dártela siempre",
    "pero yo también tengo que mejorar y tú me lo estás demostrando con paciencia",
    "pero si tú me lo permites yo quiero reparar lo que hice contigo",
    "pero tú no merecías ese trato y yo me hago cargo del error completamente",
    "pero yo también me equivoqué y voy a aprender de esto para ser mejor",
    "pero si tú crees que puedo mejorar yo me comprometo a intentarlo con todo",
    "pero tú siempre me diste tu confianza y yo la voy a cuidar mejor",
    "pero yo también sería un hipócrita si no reconociera que te hice daño",
    "pero si tú me das contexto yo puedo entender lo que necesitas de verdad",
    "pero tú mereces que yo sea mejor persona y yo también quiero serlo",
    "pero yo también debí haber pensado más porque tú siempre tienes razón",
    "pero si tú me lo pides con calma yo puedo ser más cuidadoso contigo",
    "pero tú eres importante para mí y yo también voy a demostrártelo",
    "pero yo también me propongo a ser más cuidadoso con lo que sientes",
    "pero si tú quieres que lo haga diferente yo puedo intentarlo de verdad",
    "pero tú no merecías eso y yo también voy a trabajar para cambiar",
    "pero yo también voy a poner más atención porque tú lo mereces",
    "pero si tú me das la oportunidad yo prometo no fallarte otra vez",
]

# NEG trajectory: self-defense-focused but using same connector vocabulary
neg_trajectories = [
    "pero si tú no me hubieras provocado yo habría estado más tranquilo",
    "pero tú también has hecho cosas peores y yo nunca te he reclamado",
    "pero si tú no me hubieras presionado esto no habría pasado",
    "pero tú siempre asumes lo peor de mí sin darme la oportunidad",
    "pero yo también tengo sentimientos y tú no piensas en eso nunca",
    "pero si tú hubieras sido más paciente yo no habría perdido la calma",
    "pero tú también contribuiste a que las cosas salieran mal",
    "pero no fue para tanto y tú lo sabes en el fondo",
    "pero si tú me lo hubieras dicho con claridad yo habría reaccionado diferente",
    "pero tú también te tomas las cosas muy personal siempre conmigo",
    "pero yo también tendría que haber dicho algo antes de que todo escalara",
    "pero si tú no me hubieras dicho eso yo habría estado más tranquilo",
    "pero tú también dices cosas sin pensar y nadie te reclama nada",
    "pero yo también me he portado bien y no me ves presumiéndolo",
    "pero si tú me hubieras dado más tiempo yo habría hecho todo mejor",
    "pero tú no estás siendo justa conmigo en esta situación específica",
    "pero yo también me comprometo a mejorar si tú también lo haces",
    "pero si tú no hubieras exagerado yo no habría tenido que defenderte",
    "pero tú quieres ganar la discusión y eso no es productivo para nadie",
    "pero yo también he pasado por cosas difíciles y tú no lo entiendes",
    "pero si tú no me hubieras molestado yo habría estado tranquilo",
    "pero tú siempre quieres que yo admita todo sin ver tu parte",
    "pero yo también habría hecho lo mismo en tu posición y lo sabes",
    "pero si tú me hubieras escuchado yo no habría necesitado decir eso",
    "pero tú también has cometido errores similares sin pedir perdón",
    "pero yo también me he esforzado y tú no lo reconoces jamás",
    "pero si tú no me hubieras dejado solo yo habría reaccionado diferente",
    "pero tú también viniste con una actitud que no ayudó en nada",
    "pero yo también tengo mis razones y tú no estás escuchando nada",
    "pero si tú analizas bien verás que no soy el único responsable",
]


def build_pos(scenario, harm_verb, harm_desc):
    opener = random.choice(["Lo siento mucho", "Perdón", "Disculpa", "Lo siento",
                            "Me da mucha pena", "Lo lamento"])
    trajectory = random.choice(pos_trajectories)
    patterns = [
        f"{opener}, {harm_verb} {harm_desc} {trajectory}.",
        f"{harm_verb.capitalize()} {harm_desc}. {trajectory.capitalize()}.",
        f"{opener}. {harm_verb.capitalize()} {harm_desc} {trajectory}.",
        f"Admito que {harm_verb} {harm_desc} {trajectory}.",
        f"Sé que {harm_verb} {harm_desc} {trajectory}.",
        f"Reconozco que {harm_verb} {harm_desc} {trajectory}.",
        f"{opener}. {harm_desc.capitalize()} {trajectory}.",
        f"{harm_verb.capitalize()} {harm_desc}. Fue inaceptable {trajectory}.",
    ]
    return random.choice(patterns)


def build_neg(scenario, harm_verb, harm_desc):
    opener = random.choice(["Lo siento mucho", "Perdón", "Disculpa", "Lo siento",
                            "Me da mucha pena", "Lo lamento"])
    trajectory = random.choice(neg_trajectories)
    patterns = [
        f"{opener}, {harm_verb} {harm_desc} {trajectory}.",
        f"{harm_verb.capitalize()} {harm_desc}. {trajectory.capitalize()}.",
        f"{opener}. {harm_verb.capitalize()} {harm_desc} {trajectory}.",
        f"Sé que {harm_verb} {harm_desc} {trajectory}.",
        f"Reconozco que {harm_desc} {trajectory}.",
        f"Admito que {harm_verb} {harm_desc} {trajectory}.",
        f"{opener}. {harm_desc.capitalize()} {trajectory}.",
        f"{harm_verb.capitalize()} {harm_desc} {trajectory}.",
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
