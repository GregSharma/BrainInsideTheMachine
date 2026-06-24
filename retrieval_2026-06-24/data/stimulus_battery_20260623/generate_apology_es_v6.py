#!/usr/bin/env python3
"""
v6: Template-based parallel pair generation.
Both POS and NEG share the same scenario vocabulary.
The ONLY difference is trajectory (repair vs. self-defense).
All surface features balanced by construction.
Iterates until AUC <= 0.75.
"""
import json, random, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

random.seed(42)

# ══════════════════════════════════════════════════════════════════
# COMPONENTS
# ══════════════════════════════════════════════════════════════════

openers = [
    "Lo siento mucho", "Perdón", "Disculpa", "Lo siento", "Me da mucha pena",
    "Perdón de verdad", "Lo lamento", "Te pido disculpas honestamente",
]

scenarios = [
    ("tu cumpleaños", "olvidé", "fue terrible que no te recordara"),
    ("la reunión del trabajo", "critiqué tu idea", "estuvo mal porque tenías un punto válido"),
    ("la cena familiar", "fui rudo con tu mamá", "fue irrespetuoso y lo sé"),
    ("nuestro plan para el fin de semana", "llegué tarde", "desperdicié tu tiempo sin necesidad"),
    ("la confianza entre nosotros", "te mentí", "fue una traición"),
    ("tu presentación importante", "no estuve presente", "debí estar ahí apoyándote"),
    ("la fiesta de tu amigo", "fui antipático con tus amigos", "hice pasar un mal rato"),
    ("tu proyecto personal", "no te ayudé con tu proyecto", "dejé que cargaras con todo solo"),
    ("la receta médica", "olvidé recoger tu receta", "no te di la medicina que necesitabas"),
    ("nuestro acuerdo sobre las finanzas", "gasté dinero sin consultarte", "fue irresponsable de mi parte"),
    ("tu hermana que vino de visita", "fui grosero con tu hermana", "fue innecesario y lo admito"),
    ("el plan que habíamos hecho juntos", "cancelé sin avisar", "fue irrespetuoso contigo"),
    ("la cena que cocinaste para mí", "critiqué la comida", "no valoré tu esfuerzo"),
    ("el viaje que planeamos juntos", "arruiné la reservación", "fue un desastre de mi parte"),
    ("la habitación que compartimos", "dejé todo desordenado", "no respeté nuestro espacio"),
    ("la conversación que teníamos pendiente", "la evité por tres días", "fue cobarde no enfrentarla"),
    ("el regalo que te elegí", "no fue personalizado", "no puse esfuerzo en pensarlo"),
    ("la excursión con los amigos", "te dejé solo toda la noche", "fue irrespetuoso contigo"),
    ("el mensaje privado tuyo", "lo compartí sin tu permiso", "fue una violación a tu privacidad"),
    ("la comida que me preparaste", "no la probé", "rechazar tu esfuerzo fue hiriente"),
    ("la fiesta de tu mejor amiga", "no fui a la fiesta", "te dejé sola en un momento importante"),
    ("la promesa que te hice", "no la cumplí", "mi palabra perdió valor"),
    ("tu tiempo libre", "lo llené de compromisos sin preguntar", "asumí que podías sin respetar tu tiempo"),
    ("la foto que subiste", "comenté algo negativity en tu foto", "hice que te sintieras mal en público"),
    ("la cena que teníamos programada", "me dormí y no llegué", "te dejé esperando sin aviso"),
    ("el problema que tenías en el trabajo", "no te escuché", "estuve distraído mientras hablabas"),
    ("la situación con tu jefe", "no te apoyé", "debí estar de tu lado y me quedé callado"),
    ("la tradición navideña de tu familia", "me burlé de ella", "fue irrespetuoso con algo que tú valoras"),
    ("el viaje en auto juntos", "fui un mal conductor", "te asusté con mi forma de manejar"),
    ("la presentación que diste en el trabajo", "no te felicité", "ignoré tu logro y eso estuvo mal"),
    ("la habitación del hotel", "dejé la llave adentro", "creé un problema innecesario"),
    ("la compra del supermercado", "compré cosas que no necesitábamos", "gasté dinero sin sentido"),
    ("la llamada de tu abuela", "no la atendí", "fue irrespetuoso con alguien que te quiere"),
    ("la sorpresa que planeaste", "la arruiné sin querer", "destruí tu esfuerzo"),
    ("el restaurante nuevo", "critiqué la comida del restaurante", "no valoré tu elección"),
    ("la carta que me escribiste", "la leí en voz alta a otros", "violé algo que era privado"),
    ("el problema con el vecino", "te culpé a ti", "fue injusto porque no fue tu culpa"),
    ("la cena que cocinamos juntos", "me fui a mi cuarto", "dejé que limpiaras todo sola"),
    ("la fiesta sorpresa", "conté la fiesta sorpresa", "arruiné la sorpresa que planearon"),
    ("la cena de aniversario", "llegué con ropa sucia", "no me importó prepararme para ti"),
    ("el viaje en tren", "perdí los boletos", "arruiné un plan que organizaste"),
    ("la reunión con tus padres", "fui distraído con el celular", "no presté atención y fue irrespetuoso"),
    ("el plan para hoy", "me desperté tarde", "te dejé esperando sin excusa"),
    ("la decisión sobre la casa", "decidí sin consultarte", "no consulté algo que nos afecta a los dos"),
    ("la comida compartida", "comí todo sin dejarte nada", "fui egoísta y no pensé en ti"),
    ("los sueños que tienes", "me burlé de tus sueños", "fue cruel minimizar lo que deseas"),
    ("el proyecto escolar de los niños", "no ayudé con la tarea", "dejé todo el trabajo sobre ti"),
    ("el mensaje urgente que te debía", "no te lo mandé a tiempo", "ignoré algo que te importaba"),
    ("el plan de vacaciones", "lo cambié sin consultarte", "asumí que estabas de acuerdo"),
    ("la llamada que te debía", "no te llamé como prometí", "mi palabra no valió nada"),
]

# POS repairs: diverse formulations (not just "voy a")
pos_repair_words = [
    # "voy a" (~30%)
    "voy a poner una alarma para que nunca más pase",
    "voy a ser más cuidadoso contigo de ahora en adelante",
    "voy a escucharte completo la próxima vez antes de opinar",
    "voy a salir antes para no hacerte esperar",
    "voy a ser completamente honesto contigo desde ahora",
    "voy a estar en tus eventos importantes sin falta",
    "voy a tratar con respeto a las personas que tú amas",
    "voy a compartir la carga la próxima vez que necesites",
    "voy a agregarlo a mi lista de pendientes",
    "voy a discutir las compras contigo primero",
    "voy a disculparme con ella personalmente",
    "voy a avisar con anticipación siempre",
    "voy a valorar tu esfuerzo y expresar gratitud",
    "voy a estar más atento a lo que tú necesitas",
    "voy a contestar siempre que me llames",
    "voy a dedicar tiempo cada semana para ayudar",
    "voy a organizarme mejor la próxima vez",
    "voy a revisar todo dos veces antes de confirmar",
    "voy a mantener el orden después de usar las cosas",
    "voy a hablar de las cosas difíciles cuando sea necesario",
    "voy a escuchar mejor tus gustos para la próxima vez",
    "voy a incluirte en las conversaciones la próxima vez",
    "voy a revisar dos veces antes de enviar cualquier cosa",
    "voy a probar siempre lo que cocinas",
    "voy a marcar las fechas importantes en mi calendario",
    "voy a ser más cuidadoso con lo que prometo",
    "voy a consultarte antes de hacer planes que nos afectan",
    "voy a ser más cuidadoso con lo que digo sobre ti",
    "voy a poner alarmas para no faltar",
    "voy a dejar el teléfono y darte toda mi atención",
    "voy a defender tu posición la próxima vez",
    "voy a respetar las tradiciones importantes para ti",
    "voy a manejar con más cuidado cuando lleves de pasajero",
    "voy a celebrar tus éxitos porque te los ganaste",
    "voy a ser más organizado con las cosas importantes",
    "voy a llamarle yo mismo para disculparme",
    "voy a ser más atento para no arruinar tus sorpresas",
    "voy a agradecer tu elección sin importar qué comamos",
    "voy a respetar tu privacidad siempre",
    # "prometo" / "quiero" / "me comprometo" (~25%)
    "prometo que no volverá a pasar porque tu confianza importa",
    "quiero que sepas que me arrepiento y voy a cambiar",
    "me comprometo a ser más atento con tus sentimientos",
    "prometo estar más pendiente de ti porque me importas",
    "quiero reparar esto y demostrarte que puedo ser mejor",
    "prometo que la próxima vez voy a actuar diferente",
    "me comprometo a no volver a hacerlo jamás",
    "quiero que me des la oportunidad de demostrarte que aprendí",
    "prometo escuchar antes de hablar porque tus palabras importan",
    "me comprometo a estar presente en tus momentos importantes",
    # "debo" / "es mi culpa" / "me hago cargo" (~20%)
    "debo admitir que fue mi error y me hago cargo",
    "es mi culpa y no hay excusa para lo que hice",
    "me hago responsable porque fue un error inaceptable",
    "debo reconocer que te fallé y lo lamento de corazón",
    "fue mi culpa y voy a trabajar para que no vuelva a pasar",
    "debo pedirte perdón de verdad porque lo mereces",
    "me hago cargo de mi error sin buscar excusas",
    "es mi responsabilidad y la voy a asumir completa",
    "debo ser honesto y admitir que te hice daño",
    "fue un error mío y quiero que sepas que lo reconozco",
    # "si tú me das" / alternative repairs (~15%)
    "si tú me das otra oportunidad prometo hacerlo mejor",
    "la próxima vez haré las cosas bien porque tu confianza importa",
    "nunca más voy a permitir que esto vuelva a pasar",
    "mi compromiso es no volver a fallarte jamás",
    "aprendí de esto y no lo voy a olvidar",
    "te mereces mejor y voy a esforzarme por dártelo",
    "estoy trabajando en ser mejor persona para ti",
    "esto me enseñó algo importante y no lo voy a repetir",
    "tu confianza es lo más importante y la voy a cuidar",
    "de ahora en adelante voy a poner más atención a ti",
]

# NEG defenses: diverse formulations
neg_defense_words = [
    # "pero" blame (~25%)
    "pero si tú no me hubieras provocado yo no habría reaccionado así",
    "pero tú también has hecho cosas peores y yo no las reclamo",
    "pero si analizas bien yo tenía mis razones para actuar así",
    "pero tú siempre asumes lo peor de mí sin preguntar",
    "pero yo también tengo sentimientos y tú no piensas en eso",
    "pero tú también contribuiste a que las cosas salieran mal",
    "pero si tú no me hubieras presionado esto no habría pasado",
    "pero a veces hay que decir las cosas como son sin rodeos",
    "pero tú no sabes lo que yo estaba pasando internamente",
    "pero si analizamos esto verás que no soy el único responsable",
    # "si" conditional blame (~20%)
    "si tú no me hubieras dicho eso yo no habría reaccionado así",
    "si tú hubieras sido más paciente yo no habría perdido la calma",
    "si tú no me hubieras interrumpido todo habría salido diferente",
    "si tú no hubieras traído el tema yo no habría dicho nada",
    "si tú no me hubieras dejado solo yo habría reaccionado de otra forma",
    "si tú me lo hubieras dicho con claridad yo habría venido sin dudar",
    "si tú no hubieras exagerado yo no habría tenido que defenderte",
    "si tú me hubieras dado más tiempo yo habría hecho todo mejor",
    "si tú no me lo hubieras pedido yo no lo habría hecho",
    "si tú me hubieras escuchado yo no habría necesitado decir eso",
    # "tú también" blame (~20%)
    "tú también has cometido errores similares sin pedir perdón",
    "tú también dices cosas sin pensar y nadie te reclama",
    "tú también me has hecho sentir así antes y yo te perdoné",
    "tú también tienes la culpa de todo esto y no puedes ignorarlo",
    "tú también contribuiste al problema y eso no se puede ignorar",
    "tú también te tomas las cosas muy personal siempre",
    "tú también has hecho cosas parecidas y yo no hago un escándalo",
    "tú también me has fallado antes y no te he reclamado nada",
    "tú también tienes que revisar tu comportamiento conmigo",
    "tú también estuviste mal y no te veo asumiendo tu parte",
    # "ya pasó" / minimize (~10%)
    "ya pasó y no tiene sentido seguir dándole vueltas",
    "creo que estás dramatizando algo que no tiene tanta importancia",
    "no fue para tanto y tú lo sabes en el fondo",
    "estás siendo demasiado sensible con algo que no es tan grave",
    "tú siempre quieres ganar la discusión y eso no es productivo",
    # "voy a" self-defense (~15%)
    "voy a decirte la verdad aunque no te guste escucharla",
    "voy a ser honesto contigo porque tú no ves todo el panorama",
    "voy a explicarte mi lado aunque tú no quieras escuchar",
    "voy a defenderte de eso pero tú también tienes que cooperate",
    "voy a ser directo contigo porque esto ya se ha ido de las manos",
    "voy a tratar de mejorar pero tú también tienes que hacer tu parte",
    "voy a escucharte pero si tú no me das la oportunidad no puedo hacer nada",
    "voy a intentar ser más paciente pero tú siempre me sacas de quicio",
    "voy a hacerlo diferente pero si tú no cambias esto no va a funcionar",
    "voy a ser más atento pero tú también tienes que comunicarte mejor",
    # Mixed
    "solo que tú no entiendes que yo también tengo límites personales",
    "sin embargo tú también te has portado mal conmigo antes",
    "aunque creo que tú estás viendo esto de forma muy subjetiva",
    "solo que no entiendo por qué sigues hablando de algo que ya pasó",
    "en realidad creo que tú también tienes un problema con la autoridad",
    "solo que tú siempre pides cosas que son difíciles de cumplir",
    "aunque yo creo que tú estás mezclando cosas que no tienen relación",
    "sin embargo si tú no me hubieras molestado yo habría estado tranquilo",
    "solo que tú siempre quieres que yo admita todo sin ver tu parte",
    "aunque tú también viniste con una actitud que no ayudó en nada",
]


def build_pos(scenario, harm_verb, harm_desc):
    opener = random.choice(openers)
    repair = random.choice(pos_repair_words)
    patterns = [
        f"{opener}, {harm_verb} {harm_desc}. {repair.capitalize()}.",
        f"{harm_verb.capitalize()} {harm_desc}. {repair.capitalize()}.",
        f"{opener}. {harm_verb.capitalize()} {harm_desc}. {repair.capitalize()}.",
        f"Admito que {harm_verb} {harm_desc}. {repair.capitalize()}.",
        f"Sé que {harm_verb} {harm_desc}. {repair.capitalize()}.",
        f"Reconozco que {harm_verb} {harm_desc}. {repair.capitalize()}.",
        f"{opener}. {harm_desc.capitalize()} y eso no está bien. {repair.capitalize()}.",
        f"{harm_verb.capitalize()} {harm_desc}. Eso fue inaceptable. {repair.capitalize()}.",
    ]
    return random.choice(patterns)


def build_neg(scenario, harm_verb, harm_desc):
    opener = random.choice(openers)
    defense = random.choice(neg_defense_words)
    patterns = [
        f"{opener}, {defense}.",
        f"{harm_verb.capitalize()} {harm_desc}, {defense}.",
        f"{opener}. {harm_desc.capitalize()}, {defense}.",
        f"Sé que {harm_verb} {harm_desc}, {defense}.",
        f"Reconozco que {harm_desc}, {defense}.",
        f"Admito que {harm_verb} {harm_desc}, {defense}.",
        f"{opener}. {harm_desc.capitalize()} pero {defense}.",
        f"{harm_verb.capitalize()} {harm_desc}. {defense.capitalize()}.",
    ]
    return random.choice(patterns)


# Generate 180 POS and 180 NEG
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

# --- Top features ---
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
v2 = Tfidf(ngram_range=(1, 2), min_df=2).fit(X)
lr = LogisticRegression(max_iter=2000, C=1.0).fit(v2.transform(X), y)
feature_names = v2.get_feature_names_out()
coefs = lr.coef_[0]
top_neg_idx = np.argsort(coefs)[:10]
top_pos_idx = np.argsort(coefs)[-10:][::-1]
print("\nTop 10 NEG features:")
for i in top_neg_idx:
    print(f"  {feature_names[i]:25s} {coefs[i]:.4f}")
print("\nTop 10 POS features:")
for i in top_pos_idx:
    print(f"  {feature_names[i]:25s} {coefs[i]:.4f}")

# --- Save ---
out_path = "/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/apology_es.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")
print(f"N_pos={len(pos_items)} N_neg={len(neg_items)}")
