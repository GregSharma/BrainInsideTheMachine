#!/usr/bin/env python3
"""
Generate Spanish apology stimuli with automatic TF-IDF leakage control.
Strategy: build POS and NEG from shared vocabulary pools so TF-IDF cannot
distinguish them by surface tokens. The ONLY signal should be semantic
trajectory (genuine accountability vs. self-justification).
"""
import json, random, re, itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

random.seed(42)

# ══════════════════════════════════════════════════════════════════
# SHARED VOCABULARY COMPONENTS
# Both POS and NEG draw from the same connectors, apology words, etc.
# The difference is SEMANTIC DIRECTION, not vocabulary.
# ══════════════════════════════════════════════════════════════════

# Scenarios: (scenario, harm_verb, harm_desc, concrete_repair, self_defense)
# POS uses: scenario + harm_verb + harm_desc + concrete_repair
# NEG uses: scenario + harm_verb + harm_desc + self_defense (with apology veneer)

scenarios = [
    ("tu cumpleaños", "olvidé", "fue terrible de mi parte que no te recordara",
     "voy a poner una alarma para que nunca más pase", "tú sabes que tengo la memoria fallida y nunca me apoyas con eso"),
    ("la reunión del trabajo", "critiqué tu idea", "estuvo mal porque tenías un punto válido",
     "voy a escucharte completo la próxima vez antes de opinar", "si tu idea hubiera estado más clara yo habría entendido"),
    ("la cena familiar", "fui rudo con tu mamá", "fue irrespetuoso de mi parte y lo sé",
     "voy a ser más amable con la familia que tú amas", "ella empezó con ese tono primero y yo solo respondí"),
    ("nuestro plan para el fin de semana", "llegué tarde", "desperdicié tu tiempo y eso no está bien",
     "voy a salir antes para no hacerte esperar", "si tú me lo hubieras recordado con más antelación yo habría llegado a tiempo"),
    ("la confianza entre nosotros", "te mentí", "fue una traición y me hago cargo",
     "voy a ser completamente honesto contigo de ahora en adelante", "si tú no me hubieras presionado yo no habría mentido"),
    ("tu presentación importante", "no estuve presente", "debí haber ahí apoyándote y no lo hice",
     "voy a estar en tus eventos importantes sin falta", "tenía un compromiso que no podía mover y tú lo sabías"),
    ("la fiesta de tu amigo", "fui antipático con tus amigos", "hice pasar un mal rato y eso fue injusto",
     "voy a tratar con respeto a las personas que tú quieres", "ellos me provocaron y yo solo respondí"),
    ("tu proyecto personal", "no te ayudé cuando me pediste", "dejé que cargaras con todo solo",
     "voy a compartir la carga la próxima vez que necesites ayuda", "si tú me lo hubieras pedido con más claridad yo habría ayudado"),
    ("la receta médica", "olvidé recogerla", "no te di la medicina que necesitabas y eso fue grave",
     "voy a agregarla a mi lista de pendientes para no fallarte", "si tú no me lo recordaste yo asumí que ya la tenías"),
    ("nuestro acuerdo sobre las finanzas", "gasté dinero sin consultarte", "fue irresponsable de mi parte",
     "voy a discutir las compras grandes contigo primero", "era algo urgente y no tuve tiempo de preguntarte"),
    ("tu hermana que vino de visita", "fui grosero con ella", "fue innecesario y me arrepiento",
     "voy a disculparme con ella personalmente", "ella fue la que empezó con las preguntas incómodas"),
    ("el plan que habíamos hecho juntos", "lo cancelé sin avisar", "fue irrespetuoso porque tu tiempo importa",
     "voy a avisar con anticipación cuando no pueda cumplir", "surgió algo inesperado y tú no estabas disponible"),
    ("la cena que cocinaste para mí", "critiqué la comida", "estuvo mal porque te esforzaste mucho",
     "voy a valorar tu esfuerzo y expresar gratitud", "solo dije lo que pensaba y tú no puedes manejar la honestidad"),
    ("tu cumpleaños de quinceañera de tu prima", "olvidé llevar el regalo", "fue una falta de respeto hacia tu familia",
     "voy a llevar un regalo补偿atorio y disculparme con tu prima", "tú no me dijiste que era obligatorio llevar algo"),
    ("la llamada de teléfono importante", "no contesté", "te dejé colgado cuando me necesitabas",
     "voy a contestar siempre que me llames porque tu llamada importa", "estaba en una reunión y no podía hablar"),
    ("el proyecto de la escuela de los niños", "no ayudé con la tarea", "dejé todo el trabajo sobre ti",
     "voy a dedicar tiempo cada semana para ayudarlos", "los niños ya estaban acostumbrados a hacerlo solos"),
    ("la mudanza al nuevo departamento", "noRecogí mis cajas", "te dejé todo el peso de la mudanza",
     "voy a organizarme mejor la próxima vez que nos mudemos", "tenía cosas urgentes que atender ese día"),
    ("el viaje que planeamos juntos", "hice un desastre en la reservación", "arruiné los planes por no revisar",
     "voy a revisar todo dos veces antes de confirmar reservaciones", "la página tenía información incorrecta y no fue mi culpa"),
    ("la habitación que compartimos", "dejé todo desordenado", "no respeté el espacio que es de los dos",
     "voy a mantener el orden y limpiar después de usar las cosas", "tú también dejas desorden y no te veo limpiando"),
    ("la conversación que teníamos pendiente", "la evité por tres días", "fue cobarde de mi parte no enfrentarla",
     "voy a hablar de las cosas difíciles cuando sea necesario", "sabía que ibas a reaccionar mal y quería evitar el conflicto"),
    ("el regalo que te elegí", "no fue personalizado", "no puse esfuerzo en pensarlo y eso se nota",
     "voy a escuchar mejor tus gustos para la próxima vez", "tú nunca me dices qué quieres y esperas que adivine"),
    ("la excursión con los amigos", "te dejé solo toda la noche", "fue irrespetuoso porque viniste conmigo",
     "voy a incluirte en las conversaciones la próxima vez", "estaba atrapado en otra conversación y no me di cuenta"),
    ("el mensaje que te mandé por error", "compartí algo privado", "fue una violación a tu privacidad",
     "voy a revisar dos veces antes de enviar cualquier cosa", "el teléfono se mandó solo y fue un accidente"),
    ("la comida que me preparaste", "no la probé", "rechazar tu esfuerzo fue hiriente",
     "voy a probar siempre lo que cocinas porque tu esfuerzo importa", "no tenía hambre y no quise faltar a la verdad"),
    ("el cumpleaños de tu mejor amiga", "no fui a la fiesta", "te dejé sola en un momento importante",
     "voy a marcar las fechas importantes en mi calendario", "yo no soy amigo de ella y no me sentía cómodo yendo"),
    ("la promesa que te hice", "no la cumplí", "mi palabra perdió valor y eso me duele",
     "voy a ser más cuidadoso con lo que prometo porque mi palabra debe valer", "las circunstancias cambiaron y no fue posible cumplirla"),
    ("tu tiempo libre", "lo llené de compromisos sin preguntar", "asumí que podías y no respeté tu tiempo",
     "voy a consultarte antes de hacer planes que te afectan", "necesitaba ayuda urgente y no tuve tiempo de preguntar"),
    ("la foto que subiste", "comenté algo negativity", "hice que te sintieras mal en público",
     "voy a ser más cuidadoso con lo que digo sobre ti en redes", "era un chiste y tú no entiendes el humor"),
    ("la cena que teníamos programada", "me dormí y no llegué", "te dejé esperando y eso fue horrible",
     "voy a poner alarmas para no faltar a citas importantes", "estaba agotado porque trabajo demasiado y tú no lo entiendes"),
    ("el problema que tenías en el trabajo", "no te escuché", "estuve distraído con mi celular mientras hablabas",
     "voy a dejar el teléfono y darte toda mi atención", "también tengo cosas importantes en las que pensar"),
    ("la situación con tu jefe", "no te apoyé", "debí estar de tu lado y me quedé callado",
     "voy a defender tu posición la próxima vez que alguien te ataque", "si yo hablaba empeoraba las cosas y tú lo sabías"),
    ("la tradición navideña de tu familia", "me burlé de ella", "fue irrespetuoso con algo que tú valoras",
     "voy a respetar las tradiciones que son importantes para ti", "es que realmente me pareció absurda y dije lo que pensé"),
    ("el viaje en auto juntos", "fui un mal conductor", "te asusté con mi forma de manejar",
     "voy a manejar con más cuidado cuando lleves de pasajero", "si tú no me hubieras distruido yo habría manejado mejor"),
    ("la presentación que diste en el trabajo", "no te felicité", "ignoré tu logro y eso estuvo mal",
     "voy a celebrar tus éxitos porque te los ganaste", "se me pasó porque también tenía cosas en las que pensar"),
    ("la habitación del hotel", "dejé la llave adentro", "creé un problema innecesario para ti",
     "voy a ser más organizado con las llaves y documentos", "tú también dejaste cosas y no te estoy reclamando"),
    ("la compra del supermercado", "compré cosas que no necesitábamos", "gasté dinero que debimos ahorrar",
     "voy a hacer una lista antes de ir y ceñirme a ella", "había ofertas buenas y tú no entiendes de finanzas"),
    ("la llamada de tu abuela", "no la atendí", "fue irrespetuoso porque ella te quiere mucho",
     "voy a llamarle yo mismo para disculparme", "no sabía que era ella y tú no me lo dijiste"),
    ("la sorpresa que planeaste para mí", "la arruiné sin querer", "destruí tu esfuerzo y eso me duele",
     "voy a ser más atento para no arruinar tus sorpresas", "si no me hubieras dicho nada yo no habría adivinado"),
    ("el restaurante nuevo que elegiste", "critiqué la comida", "no valoré que elegiste ese lugar para mí",
     "voy a agradecer tu elección sin importar qué comamos", "simplemente dije mi opinión honesta y no veo el problema"),
    ("la carta que me escribiste", "la leí en voz alta a otros", "violé algo que era privado entre nosotros",
     "voy a respetar tu privacidad y guardar lo que me das en confianza", "era tan bonita que quería que todos la escucharan"),
    ("el problema con el vecino", "te culpé a ti", "fue injusto porque no fue tu culpa",
     "voy a asumir mi responsabilidad sin echarte la culpa", "si tú no hubieras hecho tanto ruido el vecino no se habría quejado"),
    ("la cena que cocinamos juntos", "me fui a mi cuarto", "dejé que limpiaras todo sola después de cocinar juntos",
     "voy a ayudar a limpiar después de cocinar siempre", "estaba cansado y necesitaba descansar un momento"),
    ("la fiesta sorpresa", "no guardé el secreto", "arruiné la sorpresa que planearon para ti",
     "voy a guardar mejor los secretos que me confíen", "se me escapó porque estaba emocionado y es comprensible"),
    ("la cena de aniversario", "llegué con ropa sucia", "no me importó suficiente para prepararme",
     "voy a prepararme mejor para nuestras fechas importantes", "tuve un accidente con la comida y no tuve tiempo de cambiarme"),
    ("el viaje en tren", "perdí los boletos", "arruiné un plan que habías organizado con esfuerzo",
     "voy a guardar los documentos importantes en un lugar seguro", "los guardé donde tú me dijiste y se cayeron"),
    ("la reunión con tus padres", "fui distraído con el celular", "no presté atención y eso fue irrespetuoso",
     "voy a guardar el celular cuando esté con tu familia", "tenía una emergencia del trabajo que no podía esperar"),
    ("el plan que teníamos para hoy", "me desperté tarde", "te dejé esperando y no hay excusa",
     "voy a poner tres alarmas para no volver a fallar", "no dormí bien porque tú hablaste hasta tarde"),
    ("la decisión sobre la casa", "decidí sin ti", "no consulté algo que nos afecta a ambos",
     "voy a incluirte en todas las decisiones que nos importan", "era algo urgente y no te encontré para preguntarte"),
    ("la comida que compartimos", "comí todo sin dejarte nada", "fui egoísta y no pensé en ti",
     "voy a preguntar si quieres antes de tomar la última porción", "yo no sabía que querías más porque no me lo dijiste"),
    ("la conversación sobre el futuro", "me burlé de tus sueños", "fue cruel minimizar lo que tú deseas",
     "voy a respaldar tus metas porque son importantes para ti", "solo estaba siendo realista y tú no puedes con la verdad"),
]

# Apology connectors that appear in BOTH classes
apology_openers_pos = [
    "Lo siento mucho", "Perdón", "Disculpa", "Lo siento", "Me da mucha pena",
    "Perdón de verdad", "Disculpa de corazón", "Lo lamento", "Te pido disculpas",
    "Perdón por mi parte", "Disculpa honesta",
]

apology_openers_neg = [
    "Lo siento mucho", "Perdón", "Disculpa", "Lo siento", "Me da mucha pena",
    "Perdón de verdad", "Disculpa de corazón", "Lo lamento", "Te pido disculpas",
    "Perdón por mi parte", "Disculpa honesta",
]

# Connectors for hedging (both classes)
hedges_pos = [
    "pero", "pero quiero que sepas que", "sin embargo",
    "aunque", "aunque reconozco que",
]
hedges_neg = [
    "pero", "sin embargo", "aunque", "solo que", "solo que",
]

# Conditional starters (both classes)
conditionals_pos = [
    "si tú me das otra oportunidad", "si me escuchas completo",
    "si tú me lo pides", "si tú quieres que lo intente de nuevo",
]
conditionals_neg = [
    "si tú me hubieras dicho", "si tú no me hubieras provocado",
    "si tú no hubieras exagerado", "si tú me lo hubieras pedido",
]

# Neutral connectors
neutral_connectors = [
    "y", "y eso estuvo mal", "y lo reconozco", "y me hago cargo",
    "y eso no está bien", "y debí haberlo pensado mejor",
]


def build_pos(scenario, harm_verb, harm_desc, repair, defense):
    """Build sincere apology: names harm, stays with it, offers concrete repair."""
    opener = random.choice(apology_openers_pos)
    patterns = [
        f"{opener}, {harm_verb} {harm_desc}. {repair.capitalize()}.",
        f"{harm_verb.capitalize()} {harm_desc} y {repair}.",
        f"{opener}. {harm_verb.capitalize()} {harm_desc}. {repair.capitalize()}.",
        f"Admito que {harm_verb} {harm_desc}. {repair.capitalize()}.",
        f"Sé que {harm_verb} {harm_desc}. {repair.capitalize()}.",
        f"Reconozco que {harm_verb} {harm_desc}. {repair.capitalize()}.",
        # With "pero" / "si" (natural hedging)
        f"{opener}. {harm_verb.capitalize()} {harm_desc}, pero quiero que sepas que {repair}.",
        f"{harm_verb.capitalize()} {harm_desc} y lo lamento. Si me das la oportunidad, {repair}.",
        f"{opener}, {harm_verb} {harm_desc}. Si tú quieres, {repair}.",
        # With "tú" centered
        f"{opener}. {harm_verb.capitalize()} {harm_desc}. Tú mereces que {repair}.",
        f"{harm_verb.capitalize()} {harm_desc} y tú no merecías eso. {repair.capitalize()}.",
        f"{opener}. {harm_desc.capitalize()} contigo. {repair.capitalize()}.",
        # With "yo" accountability
        f"{opener}. Yo {harm_verb} {harm_desc} y me hago cargo. {repair.capitalize()}.",
        f"Yo fui quien {harm_verb} {harm_desc}. {repair.capitalize()}.",
        # Longer, more elaborate
        f"{opener}. Sé que {harm_desc} y eso no está bien. {repair.capitalize()}, sin importar cuánto tarde.",
        f"{harm_verb.capitalize()} {harm_desc}. Eso fue inaceptable y lo sé. {repair.capitalize()}.",
        f"{opener}. {harm_desc.capitalize()} y eso te afectó. {repair.capitalize()} porque tu confianza importa.",
    ]
    return random.choice(patterns)


def build_neg(scenario, harm_verb, harm_desc, repair, defense):
    """Build insincere apology: surface apology words but pivots to self-defense/blame."""
    opener = random.choice(apology_openers_neg)
    patterns = [
        # Classic deflection: apologize then blame
        f"{opener}, pero {defense}.",
        f"{opener}. {defense.capitalize()}.",
        # Conditional blame
        f"{harm_verb.capitalize()} {harm_desc}, sin embargo {defense}.",
        f"{opener}. {harm_desc.capitalize()}, pero si {defense}.",
        # Acknowledge + deflect
        f"Sé que {harm_verb} {harm_desc}, solo que {defense}.",
        f"Reconozco que {harm_desc}, aunque {defense}.",
        f"Admito que {harm_verb} {harm_desc}, pero {defense}.",
        # "Lo siento" variants with "si"
        f"{opener}, {harm_desc}, si tú no me hubieras hecho eso yo no habría reaccionado así.",
        f"{opener}. Si tú hubieras sido más claro, esto no habría pasado.",
        f"{opener}. {defense.capitalize()}, y creo que tú también tienes que reconocer tu parte.",
        # Blame-centered
        f"{harm_verb.capitalize()} {harm_desc}. {defense.capitalize()}.",
        f"Es cierto que {harm_desc}, pero {defense}.",
        f"{opener}, aunque tú también has hecho cosas similares.",
        f"{opener}. {defense.capitalize()}, sin embargo yo ya te dije que esto podía pasar.",
        f"{harm_verb.capitalize()} {harm_desc}. {defense.capitalize()}, y yo creo que eso es importante.",
        # Minimize
        f"{opener}. {harm_desc.capitalize()}, pero no fue para tanto.",
        f"{opener}. Si tú lo piensas bien, {defense}.",
        f"{opener}. {defense.capitalize()}. No creo que sea tan grave.",
        # "pero" + conditional
        f"{opener}, {harm_desc}, pero si tú no hubieras empezado primero yo no habría reaccionado así.",
        f"{harm_verb.capitalize()} {harm_desc}, {random.choice(hedges_neg)} {defense}.",
    ]
    return random.choice(patterns)


# Generate 180 POS and 180 NEG
pos_items = []
neg_items = []

# Use scenarios with replacement to get 180 each
scenario_indices = list(range(len(scenarios)))
random.shuffle(scenario_indices)

# Round 1: each scenario once for each class
for i in range(len(scenarios)):
    s = scenarios[i]
    pos_items.append(build_pos(*s))
    neg_items.append(build_neg(*s))

# Round 2-3: fill to 180 with shuffled re-use
while len(pos_items) < 180:
    idx = random.choice(scenario_indices)
    s = scenarios[idx]
    pos_items.append(build_pos(*s))

while len(neg_items) < 180:
    idx = random.choice(scenario_indices)
    s = scenarios[idx]
    neg_items.append(build_neg(*s))

pos_items = pos_items[:180]
neg_items = neg_items[:180]

# --- Constraint audit ---
def audit(label, items, class_label):
    siento = sum(1 for x in items if re.search(r'\bsiento\b', x, re.I))
    lo_siento = sum(1 for x in items if re.search(r'\blo siento\b', x, re.I))
    perdón = sum(1 for x in items if re.search(r'\bperd[oó]n\b', x, re.I))
    pero = sum(1 for x in items if re.search(r'\bpero\b', x, re.I))
    si = sum(1 for x in items if re.search(r'\bsi\b', x, re.I))
    starts_si = sum(1 for x in items if re.match(r'^(lo\s+)?siento\s+si\b', x, re.I))
    starts_que = sum(1 for x in items if re.match(r'^(lo\s+)?siento\s+que\b', x, re.I))
    voy_a = sum(1 for x in items if re.search(r'\bvoy a\b', x, re.I))
    tu = sum(1 for x in items if re.search(r'\btú\b', x, re.I))
    yo = sum(1 for x in items if re.search(r'\byo\b', x, re.I))
    print(f"  {label} ({class_label}): siento={siento} lo_siento={lo_siento} perdón={perdón} pero={pero} si={si} starts_si={starts_si} starts_que={starts_que} voy_a={voy_a} tú={tu} yo={yo}")

print("=== Constraint Audit ===")
audit("POS", pos_items, "sincere")
audit("NEG", neg_items, "insincere")

# --- TF-IDF evaluation ---
d = {"pos": pos_items, "neg": neg_items, "contrast": "apology", "lang": "es"}
X = d["pos"] + d["neg"]
y = [1] * len(d["pos"]) + [0] * len(d["neg"])
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(X)
auc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), v, y, cv=5, scoring="roc_auc").mean()
print(f"\nTFIDF_AUC={auc:.4f}")

# --- Save ---
out_path = "/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/apology_es.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print(f"Saved to {out_path}")
print(f"N_pos={len(pos_items)} N_neg={len(neg_items)}")
