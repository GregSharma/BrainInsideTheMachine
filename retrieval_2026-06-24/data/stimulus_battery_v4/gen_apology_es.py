"""
v4 generator — paired-template with 180 distinct stimuli per class.

Each pair shares >70% surface tokens; sincerity vs deflection is encoded in
a body-verb swap and a tail-clause swap. The v3-leak n-grams (voy, ser,
esforzarme, puedo, tan, "no habría", tú, yo) are balanced within ~1%
across the 180 POS / 180 NEG stimuli.

The residual TF-IDF AUC comes from the sincerity-encoding verb swap itself
(POS uses commitment verbs: compro, asumo, resuelvo; NEG uses evasion:
miro, aplazo, postergo). This is intentional: the sincerity SIGNAL must
be lexically realized somewhere. v3's leak was that the OTHER tokens
(voy/ser/tu/etc.) also leaked. Here only the sincerity-verb tokens leak.
"""
import json
import re
from collections import Counter

# 60 harm headers — these are SHARED verbatim between POS and NEG of each pair.
HARMS_60 = [
    ("rompí tu taza favorita", "la que te regaló tu abuela"),
    ("olvidé tu cumpleaños el martes", "no te llamé ni te mandé nada"),
    ("llegué tarde a la cena del aniversario", "te dejé esperando casi una hora"),
    ("perdí el libro que me prestaste", "el de poesía con las anotaciones"),
    ("derramé café en tu cuaderno de tesis", "manché diez páginas de notas"),
    ("dejé la puerta abierta toda la noche", "el gato se escapó dos cuadras"),
    ("usé tu computadora sin permiso", "abrí tu correo por curiosidad"),
    ("conté tu secreto a Marta", "lo de la entrevista de trabajo"),
    ("manejé tu coche y rayé la puerta", "el lado del copiloto está marcado"),
    ("olvidé recoger a tu mamá del aeropuerto", "ella esperó dos horas sola"),
    ("rompí el florero del salón", "el azul que trajiste de Sevilla"),
    ("dije una broma sobre tu acento", "frente a tus compañeros del trabajo"),
    ("cancelé la cita médica que organizaste", "la del cardiólogo del jueves"),
    ("dejé el horno encendido tres horas", "se quemó la cazuela nueva"),
    ("usé tu cargador y lo dejé en la oficina", "ahora no puedes cargar el teléfono"),
    ("comí el postre que guardabas para tu hermana", "el flan que ella había pedido"),
    ("rayé el suelo arrastrando el mueble", "la marca larga junto a la ventana"),
    ("perdí las llaves de la casa de tu padre", "él cambió la cerradura ayer"),
    ("interrumpí tu presentación con un chiste", "frente a tu jefe y dos clientes"),
    ("olvidé pagar la factura de la luz", "cortaron el servicio dos días"),
    ("dejé la ropa mojada tres días", "se manchó tu vestido blanco"),
    ("usé la harina que guardabas para el pastel", "fuiste corriendo al mercado"),
    ("conté mal la dirección al taxista", "llegaste media hora tarde a la boda"),
    ("borré las fotos del viaje a Cádiz", "perdiste las del último día en la playa"),
    ("dejé al perro sin agua todo el día", "él jadeaba cuando llegaste"),
    ("rompí la cadena que te regaló tu padre", "la de oro con el dije pequeño"),
    ("dije a tu jefe que dormías el sábado", "cuando él llamó temprano"),
    ("olvidé recoger las medicinas de tu abuela", "ella se quedó sin la pastilla del dolor"),
    ("perdí el sobre con el dinero del alquiler", "los doscientos euros del recibo"),
    ("dejé caer tu teléfono en el lavabo", "la pantalla quedó con líneas negras"),
    ("compré el regalo equivocado para tu sobrino", "le llevé el camión rojo y odia los coches"),
    ("dije a tu hermano lo de la mudanza", "cuando me pediste guardarlo una semana"),
    ("olvidé la reunión con la maestra de Lucía", "ella se enteró por otra madre"),
    ("perdí el paraguas que te regaló tu padrino", "el negro con el mango de madera"),
    ("dejé las ventanas abiertas con la lluvia", "se mojó tu sillón nuevo de tela"),
    ("usé tu perfume sin avisarte", "el frasco pequeño que casi terminaste"),
    ("rompí el plato del juego de tu madre", "el del borde dorado del aparador"),
    ("dije que tu guisado estaba salado", "frente a tu tía que lo había probado"),
    ("olvidé sacar la basura el martes", "el camión pasó y quedó dos semanas afuera"),
    ("perdí el ticket del taller mecánico", "no recogimos el coche el viernes"),
    ("dejé la nevera abierta toda la noche", "se echó a perder la carne del domingo"),
    ("usé tu chaqueta y le pegué un chicle", "la de cuero marrón del bolsillo derecho"),
    ("rompí la maqueta de tu sobrino", "la nave que tardó tres semanas armando"),
    ("dije a tus padres lo del préstamo", "cuando me hiciste prometer guardarlo"),
    ("olvidé el cumpleaños de tu mejor amiga", "el de Patricia, que esperaba tu mensaje"),
    ("perdí el documento que dejaste imprimiendo", "el contrato de la entrevista del lunes"),
    ("dejé el coche con el tanque vacío", "no saliste de noche al hospital"),
    ("usé el regalo que te trajeron del extranjero", "el chocolate caro de tus colegas"),
    ("rompí la foto enmarcada del estante", "la de ustedes en la graduación de Pablo"),
    ("dije a Sergio que estabas enojada con él", "él te llamó preocupado el sábado"),
    ("olvidé pasar por el pan que me pediste", "el integral del horno de la esquina"),
    ("perdí los aretes que me prestaste", "los de plata que eran de tu tía"),
    ("dejé el grifo abierto y se inundó el baño", "el agua bajó hasta el piso de abajo"),
    ("usé tus auriculares y los enredé", "ahora un lado ya no suena bien"),
    ("rompí el vidrio del cuadro del pasillo", "el de la acuarela que pintó tu madre"),
    ("dije lo del aumento a Carmen", "cuando me pediste no decir nada aún"),
    ("olvidé llamar al fontanero el lunes", "la gotera sigue cayendo en la cocina"),
    ("perdí la tarjeta de regalo de la librería", "los cincuenta euros del cumpleaños"),
    ("dejé el lavavajillas a medio cargar", "la espuma manchó el suelo de la cocina"),
    ("usé tus pinceles sin lavarlos después", "los azules quedaron duros y pegados"),
]
assert len(HARMS_60) == 60

OPENERS = ["Lo siento mucho,", "Perdón,", "Disculpa,"]

# Six skeletons. POS and NEG halves contain identical leaky-marker tokens.
# Each appears 30 times per class (180 / 6 = 30).
SKELETONS = [
    ("yo entiendo tú esperabas más, pero mañana {body}; esto fue tan importante para ti, yo no habría querido fallarte y {tail}.",
     "yo entiendo tú esperabas más, pero mañana {body}; esto fue tan importante para ti, yo no habría querido fallarte y {tail}."),
    ("yo puedo ser quien arregle esto contigo, también pero mañana {body}; tú merecías más cuidado y {tail}.",
     "yo puedo ser quien arregle esto contigo, también pero mañana {body}; tú merecías más cuidado y {tail}."),
    ("yo no habría hecho esto si tú me hubieras visto pensar mejor, pero mañana {body}; yo quiero que esto cambie y {tail}.",
     "yo no habría hecho esto si tú me hubieras visto pensar mejor, pero mañana {body}; yo quiero que esto cambie y {tail}."),
    ("yo voy a ser cuidadoso contigo, puedo esforzarme desde mañana en {body}, pero esto fue tan duro para ti y {tail}.",
     "yo voy a ser cuidadoso contigo, puedo esforzarme desde mañana en {body}, pero esto fue tan duro para ti y {tail}."),
    ("yo sé esto fue tan grave para ti pero no fue una pequeñez, también mañana {body}; tú mereces que esto cambie y {tail}.",
     "yo sé esto fue tan grave para ti pero no fue una pequeñez, también mañana {body}; tú mereces que esto cambie y {tail}."),
    ("yo reconozco fue mi culpa entera, no habría querido herirte tú lo sabes, pero mañana {body}; {tail}.",
     "yo reconozco fue mi culpa entera, no habría querido herirte tú lo sabes, pero mañana {body}; {tail}."),
]

# 60 body pairs — one per harm. Each pair shares ~10 of 12 words, differs on
# the SINCERITY verb (POS: action / NEG: evasion) and one adjective.
# Cycling 60 harms × 3 openers gives 180 lines.
BODY_PAIRS = [
    ("reemplazo la taza esta semana en la tienda de la abuela el sábado",
     "examino la taza esta semana en la tienda de la abuela el sábado"),
    ("invito a cenar el viernes en el restaurante del puerto del barrio",
     "calculo una cena el viernes en el restaurante del puerto del barrio"),
    ("reservo otra mesa el sábado a las ocho en el mismo local de antes",
     "estimo otra mesa el sábado a las ocho en el mismo local de antes"),
    ("compro otro ejemplar el lunes en la librería del centro de la ciudad",
     "ojeo otro ejemplar el lunes en la librería del centro de la ciudad"),
    ("transcribo las diez páginas a mano antes del lunes con calma total",
     "examino las diez páginas a mano antes del lunes con calma total"),
    ("rescato al gato esta tarde por todo el barrio con cartel impreso",
     "sondeo al gato esta tarde por todo el barrio con cartel impreso"),
    ("borro el historial hoy y pongo contraseña nueva por escrito ya",
     "reviso el historial hoy y pongo contraseña nueva por escrito ya"),
    ("llamo a Marta esta tarde a pedirle silencio absoluto del tema",
     "consulto a Marta esta tarde a pedirle silencio absoluto del tema"),
    ("pago el repintado del coche el viernes en el taller de Luis hoy",
     "tantéo el repintado del coche el viernes en el taller de Luis hoy"),
    ("recojo a esa madre del próximo vuelo del lunes con cartel y flores",
     "valoro a esa madre del próximo vuelo del lunes con cartel y flores"),
    ("compro un florero igual el sábado en cinco tiendas de Sevilla seguidas",
     "ojeo un florero igual el sábado en cinco tiendas de Sevilla seguidas"),
    ("me disculpo el lunes en el café del trabajo frente a esos compañeros",
     "considero algo el lunes en el café del trabajo frente a esos compañeros"),
    ("pido cita nueva con el cardiólogo el lunes a primera hora del día",
     "valoro cita nueva con el cardiólogo el lunes a primera hora del día"),
    ("compro otra cazuela del mismo tamaño el martes en la ferretería cercana",
     "sondeo otra cazuela del mismo tamaño el martes en la ferretería cercana"),
    ("entrego el cargador el lunes a primera hora en la oficina sin falta",
     "reviso el cargador el lunes a primera hora en la oficina sin falta"),
    ("preparo otro flan esta noche con la receta exacta de esa madre",
     "pondero otro flan esta noche con la receta exacta de esa madre"),
    ("barnizo la marca el sábado con el color exacto del original viejo",
     "examino la marca el sábado con el color exacto del original viejo"),
    ("pago al cerrajero y a ese padre la cerradura el viernes completo",
     "tantéo al cerrajero y a ese padre la cerradura el viernes completo"),
    ("escribo al jefe el lunes asumiendo enteramente la culpa del chiste viejo",
     "reflexiono al jefe el lunes asumiendo enteramente la culpa del chiste viejo"),
    ("pago la luz hoy entera y pongo débito automático del banco mañana",
     "ojeo la luz hoy entera y pongo débito automático del banco mañana"),
    ("compro otro vestido el sábado en la tienda del centro de Madrid mañana",
     "ojeo otro vestido el sábado en la tienda del centro de Madrid mañana"),
    ("repongo el doble de harina ahora del mercado de la calle Mayor cercano",
     "reviso el doble de harina ahora del mercado de la calle Mayor cercano"),
    ("llamo a los novios el lunes a disculparme yo solo del retraso enorme",
     "consulto a los novios el lunes a disculparme yo solo del retraso enorme"),
    ("pago al técnico de mi primo el viernes a recuperar las fotos perdidas",
     "tantéo al técnico de mi primo el viernes a recuperar las fotos perdidas"),
    ("pongo bebedero doble y alarma cada mañana del agua del perro nervioso",
     "ojeo bebedero doble y alarma cada mañana del agua del perro nervioso"),
    ("llevo la cadena el lunes a la joyería del centro a repararla bien",
     "valoro la cadena el lunes a la joyería del centro a repararla bien"),
    ("escribo al jefe el lunes asumiendo enteramente que mentí del sábado pasado",
     "reflexiono al jefe el lunes asumiendo enteramente que mentí del sábado pasado"),
    ("entrego a la farmacia ahora las medicinas a esa abuela enferma del barrio",
     "ojeo en la farmacia ahora las medicinas a esa abuela enferma del barrio"),
    ("saco los doscientos euros mañana del cajero del banco del centro otra vez",
     "valoro los doscientos euros mañana del cajero del banco del centro otra vez"),
    ("pago la pantalla nueva el viernes en la tienda del barrio del centro",
     "tantéo la pantalla nueva el viernes en la tienda del barrio del centro"),
    ("cambio el camión por el avión rojo mañana en la juguetería del centro",
     "valoro el camión por el avión rojo mañana en la juguetería del centro"),
    ("llamo a ese hermano hoy a aclarar enteramente que la indiscreción fue mía",
     "consulto a ese hermano hoy a aclarar enteramente que la indiscreción fue mía"),
    ("pido cita nueva con la maestra de Lucía el martes a las ocho temprano",
     "ojeo cita nueva con la maestra de Lucía el martes a las ocho temprano"),
    ("compro otro paraguas negro el sábado en el mercado del rastro del centro",
     "ojeo otro paraguas negro el sábado en el mercado del rastro del centro"),
    ("entrego el sillón al tapicero esta semana a primera hora del martes mañana",
     "valoro el sillón al tapicero esta semana a primera hora del martes mañana"),
    ("compro otro frasco grande el viernes en la perfumería del centro mañana",
     "ojeo otro frasco grande el viernes en la perfumería del centro mañana"),
    ("compro otro plato del juego el domingo en el rastro del centro mañana",
     "ojeo otro plato del juego el domingo en el rastro del centro mañana"),
    ("llamo a esa tía el domingo a retractarme enteramente del comentario suelto",
     "consulto a esa tía el domingo a retractarme enteramente del comentario suelto"),
    ("pongo alarma fija cada martes a las nueve para la basura del piso completa",
     "ojeo alarma fija cada martes a las nueve para la basura del piso completa"),
    ("recojo al taller el lunes el coche con el papeleo entero arreglado",
     "tantéo al taller el lunes el coche con el papeleo entero arreglado"),
    ("compro otra carne hoy en la carnicería y la cocino el domingo entero",
     "ojeo otra carne hoy en la carnicería y la cocino el domingo entero"),
    ("entrego la chaqueta al tinte el lunes y pago la limpieza entera completa",
     "valoro la chaqueta al tinte el lunes y pago la limpieza entera completa"),
    ("me siento con ese sobrino el sábado a rearmar la nave entera de nuevo",
     "consulto con ese sobrino el sábado a rearmar la nave entera de nuevo"),
    ("llamo a esos padres esta noche a aclarar enteramente que fue mía la culpa",
     "consulto a esos padres esta noche a aclarar enteramente que fue mía la culpa"),
    ("escribo a Patricia hoy un mensaje largo asumiendo enteramente que la olvidé",
     "reflexiono a Patricia hoy un mensaje largo asumiendo enteramente que la olvidé"),
    ("reimprimo el contrato a las ocho en la imprenta del centro el lunes mañana",
     "examino el contrato a las ocho en la imprenta del centro el lunes mañana"),
    ("lleno el tanque ahora hasta arriba en la gasolinera del centro del barrio",
     "ojeo el tanque ahora hasta arriba en la gasolinera del centro del barrio"),
    ("compro otro chocolate igual el sábado en la tienda de importados del centro",
     "ojeo otro chocolate igual el sábado en la tienda de importados del centro"),
    ("mando el negativo el lunes al laboratorio del centro a revelar otra copia",
     "ojeo el negativo el lunes al laboratorio del centro a revelar otra copia"),
    ("llamo a Sergio esta noche a aclarar enteramente el malentendido mío entero",
     "consulto a Sergio esta noche a aclarar enteramente el malentendido mío entero"),
    ("recojo el horno de la esquina a las siete para el pan integral fresco",
     "ojeo el horno de la esquina a las siete para el pan integral fresco"),
    ("compro aretes de plata el viernes en la joyería del centro de la calle Mayor",
     "ojeo aretes de plata el viernes en la joyería del centro de la calle Mayor"),
    ("pago al vecino de abajo los daños del agua hoy en metálico entero completo",
     "tantéo al vecino de abajo los daños del agua hoy en metálico entero completo"),
    ("compro auriculares nuevos esta tarde en la tienda de la plaza mayor del centro",
     "ojeo auriculares nuevos esta tarde en la tienda de la plaza mayor del centro"),
    ("cambio el vidrio del cuadro el sábado en el enmarcador del barrio del centro",
     "valoro el vidrio del cuadro el sábado en el enmarcador del barrio del centro"),
    ("llamo a Carmen esta noche a retractarme enteramente del comentario suelto",
     "consulto a Carmen esta noche a retractarme enteramente del comentario suelto"),
    ("llamo al fontanero hoy a las nueve a primera hora del lunes urgente mañana",
     "consulto al fontanero hoy a las nueve a primera hora del lunes urgente mañana"),
    ("compro otra tarjeta de cincuenta euros esta tarde en la librería del centro",
     "ojeo otra tarjeta de cincuenta euros esta tarde en la librería del centro"),
    ("friego el suelo de la cocina ahora con bayeta nueva del mercado de hoy",
     "examino el suelo de la cocina ahora con bayeta nueva del mercado de hoy"),
    ("compro pinceles azules nuevos el martes en la tienda de arte cercana mañana",
     "ojeo pinceles azules nuevos el martes en la tienda de arte cercana mañana"),
]
assert len(BODY_PAIRS) == 60

# 6 tail pairs. POS commits, NEG defers. Verbs vary so no single one dominates.
TAIL_PAIRS = [
    ("lo asumo del todo hoy",    "lo aplazo del todo hoy"),
    ("lo arreglo cuanto antes",  "lo postergo cuanto antes"),
    ("lo resuelvo sin demora",   "lo retraso sin atajo"),
    ("lo afronto del todo",      "lo eludo del todo"),
    ("lo cubro entero ya",       "lo dejo entero ya"),
    ("lo cierro cuanto antes",   "lo difiero cuanto antes"),
]
assert len(TAIL_PAIRS) == 6

def build():
    pos, neg = [], []
    for i in range(180):
        opener_idx = i // 60
        h_idx = i % 60
        opener = OPENERS[opener_idx]
        act, detail = HARMS_60[h_idx]
        head = f"{opener} {act} — {detail}."
        fam = i % 6
        skel_p, skel_n = SKELETONS[fam]
        p_body, n_body = BODY_PAIRS[h_idx]
        tail_p, tail_n = TAIL_PAIRS[fam]
        pos.append(f"{head} {skel_p.format(body=p_body, tail=tail_p)}")
        neg.append(f"{head} {skel_n.format(body=n_body, tail=tail_n)}")
    return pos, neg

POS, NEG = build()
assert len(POS) == 180 and len(NEG) == 180

# Audit
def tok(s):
    s = s.lower()
    s = re.sub(r"[^\w\sáéíóúñü]", " ", s)
    return s.split()

def U(L, t): return sum(1 for l in L if t in tok(l)) / len(L)
def B(L, bg):
    a, b = bg.split(); cnt=0
    for l in L:
        t = tok(l)
        for i in range(len(t)-1):
            if t[i]==a and t[i+1]==b:
                cnt += 1; break
    return cnt/len(L)

LEAKED_POS = ["voy", "ser", "mejor", "más", "puedo", "esforzarme", "de"]
LEAKED_NEG = ["tan", "no", "estás", "tú", "grave"]
LEAKED_NEG_BG = ["no habría", "yo no", "tan grave"]
LEAKED_POS_BG = ["ser mejor"]
APOLOGY = ["siento", "perdón", "disculpa", "lamento"]

print("=" * 76)
print("v4 FINAL audit")
print("=" * 76)
print("\nv3 LEAK-BRIEF TARGETS:")
worst = 0; worst_t = ""
for t in LEAKED_POS + LEAKED_NEG:
    p, n = U(POS, t), U(NEG, t)
    d = abs(p - n)
    ok = "OK" if d <= 0.10 else "!!"
    print(f"  {ok} {t:<15} POS={p:.3f} NEG={n:.3f} Δ={d:+.3f}")
    if d > worst: worst, worst_t = d, t
for bg in LEAKED_POS_BG + LEAKED_NEG_BG:
    p, n = B(POS, bg), B(NEG, bg)
    d = abs(p-n)
    ok = "OK" if d <= 0.10 else "!!"
    print(f"  {ok} {bg:<15} POS={p:.3f} NEG={n:.3f} Δ={d:+.3f}")
    if d > worst: worst, worst_t = d, bg
print(f"\nMax LEAK-BRIEF target |Δ|: {worst:.3f} ({worst_t})")

print("\nApology register:")
for t in APOLOGY:
    print(f"  {t:<10} POS={U(POS,t):.3f}  NEG={U(NEG,t):.3f}")

mn_p = min(len(tok(l)) for l in POS); mu_p = sum(len(tok(l)) for l in POS)/len(POS); mx_p = max(len(tok(l)) for l in POS)
mn_n = min(len(tok(l)) for l in NEG); mu_n = sum(len(tok(l)) for l in NEG)/len(NEG); mx_n = max(len(tok(l)) for l in NEG)
print(f"\nPOS words: min={mn_p} mean={mu_p:.1f} max={mx_p}")
print(f"NEG words: min={mn_n} mean={mu_n:.1f} max={mx_n}")
print(f"Unique POS lines: {len(set(POS))}/180   Unique NEG lines: {len(set(NEG))}/180")

# Top differential unigrams (corpus-wide)
all_w = Counter()
for l in POS + NEG:
    for w in set(tok(l)):
        all_w[w] += 1
cands = [w for w, c in all_w.items() if c >= 18 and len(w) > 1]
diffs = []
for w in cands:
    p = U(POS, w); n = U(NEG, w)
    diffs.append((w, p, n, p-n))
diffs.sort(key=lambda x: abs(x[3]), reverse=True)
print("\nTop 15 differential unigrams (corpus-wide):")
for w, p, n, d in diffs[:15]:
    print(f"  {w:<18} POS={p:.3f} NEG={n:.3f} Δ={d:+.3f}")
top_uni_gap = abs(diffs[0][3]) if diffs else 0.0
print(f"\nTOP DIFFERENTIAL UNIGRAM RATE GAP (full vocab): {top_uni_gap:.4f}")

def bigrams_of(line):
    t = tok(line)
    return set((t[i], t[i+1]) for i in range(len(t)-1))
P_bg = Counter(); N_bg = Counter()
for l in POS:
    for bg in bigrams_of(l): P_bg[bg] += 1
for l in NEG:
    for bg in bigrams_of(l): N_bg[bg] += 1
bi_diffs = []
for bg in set(P_bg) | set(N_bg):
    pc = P_bg.get(bg,0); nc = N_bg.get(bg,0)
    if pc + nc < 18: continue
    p = pc/len(POS); n = nc/len(NEG)
    bi_diffs.append((bg, p, n, p-n))
bi_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
print("\nTop 15 differential bigrams (corpus-wide):")
for bg, p, n, d in bi_diffs[:15]:
    print(f"  {' '.join(bg):<28} POS={p:.3f} NEG={n:.3f} Δ={d:+.3f}")
top_bi_gap = abs(bi_diffs[0][3]) if bi_diffs else 0.0
print(f"\nTOP DIFFERENTIAL BIGRAM RATE GAP (full vocab): {top_bi_gap:.4f}")

print("\nSample 6 pairs:")
for i in [0, 1, 30, 60, 120, 178]:
    print(f"\n  [{i}] POS: {POS[i]}")
    print(f"  [{i}] NEG: {NEG[i]}")

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    X = POS + NEG
    y = [1]*len(POS) + [0]*len(NEG)
    print()
    for ng in [(1,1),(1,2),(1,3)]:
        vec = TfidfVectorizer(ngram_range=ng, min_df=2)
        Xv = vec.fit_transform(X)
        clf = LogisticRegression(max_iter=2000)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        s = cross_val_score(clf, Xv, y, cv=skf, scoring='roc_auc')
        print(f"  TF-IDF ngram={ng}: AUC={s.mean():.4f} ± {s.std():.4f}")
except ImportError:
    pass

OUT = {"contrast": "apology", "lang": "es", "pos": POS, "neg": NEG}
out_path = "/home/greg/Desktop/Projects/BrainInsideTheMachine/stimulus_battery_v4/apology_es.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)
print(f"\nWrote {out_path}  POS={len(POS)} NEG={len(NEG)}")
