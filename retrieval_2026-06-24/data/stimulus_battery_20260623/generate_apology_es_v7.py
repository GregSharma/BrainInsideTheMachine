#!/usr/bin/env python3
"""
v7: Fix pero/si/tú imbalance. POS must include these at same rate as NEG.
POS natural hedging: "Lo siento, pero voy a mejorar" / "Si tú me das otra oportunidad..."
NEG natural self-defense: "Lo siento, pero si tú no me hubieras provocado..."
Both classes share the same connector vocabulary. Only trajectory differs.
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
# POS repairs: include pero/si/tú naturally (~50-60% each)
# ══════════════════════════════════════════════════════════════════

pos_repair_words = [
    # --- pero + repair (natural sincere hedging) ---
    "pero voy a poner una alarma para que nunca más pase",
    "pero voy a ser más cuidadoso contigo de ahora en adelante",
    "pero voy a escucharte completo la próxima vez antes de opinar",
    "pero voy a salir antes para no hacerte esperar nunca más",
    "pero voy a ser completamente honesto contigo desde ahora en adelante",
    "pero voy a estar en tus eventos importantes sin falta",
    "pero voy a tratar con respeto a las personas que tú amas",
    "pero voy a compartir la carga la próxima vez que necesites ayuda",
    "pero voy a discutir las compras grandes contigo primero",
    "pero voy a avisar con anticipación siempre que no pueda cumplir",
    "pero voy a valorar tu esfuerzo y expresar gratitud siempre",
    "pero voy a estar más atento a lo que tú necesitas de mí",
    "pero voy a contestar siempre que me llames porque tu llamada importa",
    "pero voy a organizarme mejor la próxima vez que eso suceda",
    "pero voy a revisar todo dos veces antes de confirmar algo",
    "pero voy a mantener el orden después de usar las cosas tuyas",
    "pero voy a escuchar mejor tus gustos para la próxima vez",
    "pero voy a incluirte en las conversaciones que nos afectan",
    "pero voy a marcar las fechas importantes en mi calendario",
    "pero voy a ser más cuidadoso con lo que prometo cumplir",
    "pero voy a consultarte antes de hacer planes que nos afectan a los dos",
    "pero voy a ser más cuidadoso con lo que digo sobre ti",
    "pero voy a poner alarmas para no faltar a nada importante",
    "pero voy a dejar el teléfono y darte toda mi atención",
    "pero voy a defender tu posición la próxima vez que alguien te ataque",
    "pero voy a respetar las tradiciones que son importantes para ti",
    "pero voy a manejar con más cuidado cuando lleves de pasajero",
    "pero voy a celebrar tus éxitos porque te los ganaste con esfuerzo",
    "pero voy a llamarle yo mismo para disculparme con ella",
    "pero voy a ser más atento para no arruinar tus sorpresas",
    "pero voy a agradecer tu elección sin importar qué comamos",
    "pero voy a respetar tu privacidad y guardar lo que me das en confianza",
    # --- si + tú + repair ---
    "si tú me das otra oportunidad prometo hacerlo mejor",
    "si tú me lo pides yo paro todo para escucharte",
    "si tú quieres que lo intente de nuevo voy a esforzarme",
    "si tú me lo pides con calma yo puedo cambiar mi actitud",
    "si tú me das contexto yo puedo entender mejor lo que necesitas",
    "si tú me lo pides yo estoy dispuesto a mejorar",
    "si tú quieres que lo haga diferente yo puedo intentarlo",
    "si tú me das la oportunidad yo prometo no fallarte",
    "si tú me lo explicas yo puedo entender tu punto de vista",
    "si tú me lo pides yo voy a estar más presente",
    # --- tú-centered repair ---
    "tú mereces que yo sea mejor persona y voy a trabajar en eso",
    "tú no merecías eso y voy a compensártelo de verdad",
    "tú importas mucho para mí y voy a demostrártelo",
    "tú eres lo más importante y no debería haberte fallado",
    "tú siempre estuviste para mí y yo debí hacer lo mismo",
    "tú no merecías ese trato y me arrepiento profundamente",
    "tú tienes todo mi cariño y voy a demostrártelo con hechos",
    "tú eres una persona increíble y no debí haberte tratado así",
    "tú me diste tu confianza y yo la rompí, pero voy a repararla",
    "tú mereces honestidad y voy a dártela siempre",
    # --- "voy a" without pero (for balance) ---
    "voy a poner una alarma para que nunca más pase porque te lo debo",
    "voy a ser más cuidadoso contigo porque tu confianza importa mucho",
    "voy a estar en tus eventos importantes porque tú importas para mí",
    "voy a tratar con respeto a la familia que tú amas tanto",
    "voy a organizarme mejor porque no mereces que te falle así",
    "voy a consultarte siempre antes de hacer planes que nos afecten",
    "voy a ser más atento a tus necesidades porque te quiero mucho",
    "voy a cuidar tu privacidad porque confiaste en mí",
    # --- debo / es mi culpa + features ---
    "debo admitir que fue mi error pero voy a trabajar para cambiar",
    "es mi culpa y no hay excusa pero quiero que sepas que cambiaré",
    "me hago responsable porque fue inaceptable si tú me das otra oportunidad",
    "debo reconocer que te fallé pero voy a esforzarme por ser mejor",
    "fue mi culpa y te debo una disculpa sincera pero voy a repararlo",
    "debo pedirte perdón de corazón porque tú no merecías eso",
    "me hago cargo de mi error sin excusas si tú me lo permites",
    "es mi responsabilidad y la voy a asumir porque tú me importas",
    # --- promise / commitment + features ---
    "prometo que no volverá a pasar porque tú te lo mereces todo",
    "quiero que sepas que me arrepiento y si tú me lo permites cambiaré",
    "me comprometo a ser más atento contigo porque eres importante para mí",
    "prometo estar más pendiente de ti porque no debí haberte fallado",
    "quiero reparar esto y si tú me das la oportunidad lo haré mejor",
    "prometo que la próxima vez voy a actuar diferente porque tú lo mereces",
    "me comprometo a no volver a hacerlo porque tú no mereces ese trato",
    "quiero que me des la oportunidad si tú crees que puedo mejorar",
    "prometo escuchar antes de hablar porque tus sentimientos importan mucho",
    "me comprometo a estar presente en tus momentos importantes porque tú lo mereces",
]

# ══════════════════════════════════════════════════════════════════
# NEG defenses: include pero/si/tú/voy_a naturally
# ══════════════════════════════════════════════════════════════════

neg_defense_words = [
    # --- pero blame ---
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
    "pero tú también tienes la culpa de todo esto y no puedes ignorarlo",
    "pero si tú no me hubieras dicho eso yo habría estado más tranquilo",
    "pero tú también dices cosas sin pensar y nadie te reclama",
    "pero si tú no hubieras exagerado yo no habría tenido que defenderte",
    "pero tú también me has hecho sentir así antes y yo te perdoné",
    "pero si tú no me hubieras molestado yo habría estado bien",
    "pero tú también has cometido errores similares sin pedir perdón",
    "pero si tú no me hubieras interrumpido todo habría salido diferente",
    "pero tú también estuviste mal y no te veo asumiendo tu parte",
    "pero si tú no hubieras traído el tema yo no habría dicho nada",
    # --- si conditional blame ---
    "si tú no me hubieras dicho eso yo no habría reaccionado así",
    "si tú hubieras sido más paciente yo no habría perdido la calma",
    "si tú no me hubieras dejado solo yo habría reaccionado de otra forma",
    "si tú me lo hubieras dicho con claridad yo habría venido sin dudar",
    "si tú me hubieras dado más tiempo yo habría hecho todo mejor",
    "si tú no me lo hubieras pedido yo no lo habría hecho",
    "si tú me hubieras escuchado yo no habría necesitado decir eso",
    "si tú no me hubieras provocado yo habría estado más tranquilo",
    "si tú no hubieras cambiado de idea yo no habría tenido que improvisar",
    "si tú me lo hubieras recordado yo habría llegado a tiempo",
    # --- tú también blame ---
    "tú también has hecho cosas peores y yo nunca te he reclamado",
    "tú también dices cosas hirientes cuando estás enojado y nadie te dice nada",
    "tú también me has fallado antes y yo te he perdonado sin hacer un escándalo",
    "tú también tienes la culpa de lo que pasó y no puedes negarlo",
    "tú también contribuiste al problema y eso no se puede ignorar así",
    "tú también te tomas las cosas muy personal siempre conmigo",
    "tú también has hecho cosas parecidas y yo no hago un escándalo",
    "tú también me has dicho cosas hirientes y no te he reclamado",
    "tú también tienes que revisar tu comportamiento porque no eres perfecta",
    "tú también viniste con una actitud que no ayudó en nada",
    # --- "voy a" self-defense ---
    "voy a decirte la verdad aunque no te guste escucharla",
    "voy a ser honesto contigo porque tú no ves todo el panorama",
    "voy a explicarte mi lado aunque tú no quieras escuchar",
    "voy a ser directo contigo porque esto ya se ha ido de las manos",
    "voy a tratar de mejorar pero tú también tienes que hacer tu parte",
    "voy a escucharte pero si tú no me das la oportunidad no puedo hacer nada",
    "voy a intentar ser más paciente pero tú siempre me sacas de quicio",
    "voy a hacerlo diferente pero si tú no cambias esto no va a funcionar",
    "voy a ser más atento pero tú también tienes que comunicarte mejor",
    "voy a defenderme porque tú no estás siendo justa conmigo",
    # --- minimize ---
    "ya pasó y no tiene sentido seguir dándole vueltas a esto",
    "creo que estás dramatizando algo que no tiene tanta importancia",
    "no fue para tanto y tú lo sabes en el fondo de tu corazón",
    "estás siendo demasiado sensible con algo que no es tan grave",
    "tú siempre quieres ganar la discusión y eso no es productivo",
    "solo que tú no entiendes que yo también tengo límites personales",
    "sin embargo tú también te has portado mal conmigo antes",
    "aunque creo que tú estás viendo esto de forma muy subjetiva",
    "solo que no entiendo por qué sigues hablando de algo que ya pasó",
    "tú siempre quieres que yo admita todo sin ver tu parte en esto",
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
        f"{harm_verb.capitalize()} {harm_desc}. Fue inaceptable. {repair.capitalize()}.",
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

# Audit
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
    también = sum(1 for x in items if re.search(r'\btambi[eé]n\b', x, re.I))
    print(f"  {label} ({class_label}): siento={siento} lo_s={lo_siento} perdon={perdón} pero={pero} si={si} s_ssi={starts_si} s_sq={starts_que} voy_a={voy_a} tu={tu} yo={yo} tambien={también}")

print("=== Constraint Audit ===")
audit("POS", pos_items, "sincere")
audit("NEG", neg_items, "insincere")

# TF-IDF
d = {"pos": pos_items, "neg": neg_items, "contrast": "apology", "lang": "es"}
X = d["pos"] + d["neg"]
y = [1] * len(d["pos"]) + [0] * len(d["neg"])
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(X)
auc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), v, y, cv=5, scoring="roc_auc").mean()
print(f"\nTFIDF_AUC={auc:.4f}")

# Top features
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
v2 = Tfidf(ngram_range=(1, 2), min_df=2).fit(X)
lr = LogisticRegression(max_iter=2000, C=1.0).fit(v2.transform(X), y)
fn = v2.get_feature_names_out()
coefs = lr.coef_[0]
print("\nTop 10 NEG:")
for i in np.argsort(coefs)[:10]:
    print(f"  {fn[i]:25s} {coefs[i]:.4f}")
print("\nTop 10 POS:")
for i in np.argsort(coefs)[-10:][::-1]:
    print(f"  {fn[i]:25s} {coefs[i]:.4f}")

# Save
out_path = "/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_20260623/apology_es.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")
print(f"N_pos={len(pos_items)} N_neg={len(neg_items)}")
