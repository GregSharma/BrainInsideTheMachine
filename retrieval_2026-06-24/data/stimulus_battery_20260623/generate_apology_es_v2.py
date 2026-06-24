#!/usr/bin/env python3
"""Generate Spanish apology stimuli v2: balanced TF-IDF features across classes."""
import json, random, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

random.seed(42)

# ══════════════════════════════════════════════════════════════════
# DESIGN PRINCIPLE: Every "signature word" must appear in BOTH classes
# at roughly equal rates (within 20%).
#
# BANNED at start of NEG: "siento si", "siento que", "lo siento si", "lo siento que"
# Target per class:
#   siento/lo siento: 40-50% of each class
#   perdón: 25-35% of each class
#   pero: 30-40% of each class
#   si (as conditional): 30-40% of each class
#   voy a: 30-40% of each class (POS uses for repair, NEG uses for self-defense)
#   tú: 50-60% of each class
# ══════════════════════════════════════════════════════════════════

pos_pool = [
    # --- Category 1: Direct naming + repair (with "voy a") ---
    "Olvidé tu cumpleaños y me da mucha pena. Voy a poner una alarma para que no vuelva a pasar.",
    "Mentí sobre el dinero y rompí tu confianza. Voy a ser completamente transparente contigo desde ahora.",
    "Levanté la voz y eso estuvo mal. La próxima vez que me enfade voy a apartarme antes de decir algo.",
    "Llegué tarde y desperdicié tu tiempo. Voy a salir antes para que esto no se repita.",
    "Rompi tu taza y debí habértelo dicho de inmediato. Te la voy a reemplazar y seré más cuidadoso.",
    "Olvidé recoger a los niños y no tengo excusa. Voy a poner una alarma y confirmar contigo cada mañana.",
    "No llamé cuando dije que iba a llamar y te fallé. Voy a cumplir con lo que prometo de ahora en adelante.",
    "Me olvidé de nuestro aniversario y me siento terrible. Mereces ser celebrado y voy a compensártelo.",
    "Te contesté mal por algo insignificante. No merecías eso y voy a ser más paciente contigo.",
    "Me apropié de tu trabajo haciéndome pasar por el autor. Voy a asegurarme de que el equipo sepa que fue tuyo.",
    "Te excluí de los planes sin pensar. Voy a incluirte de ahora en adelante sin excepción.",
    "Fracasé en cumplir el plazo del proyecto y perjudiqué a todo el equipo. Voy a organizar mejor mi tiempo.",
    "Te cancelé en el último minuto. Tu tiempo importa y voy a confirmar antes de hacer planes.",
    "Corrí un rumor sin verificarlo. Fue irresponsable y le debo una disculpa también a esa persona.",
    "Olvidé guardar tu número y lo perdí. Voy a respaldar mis contactos esta semana.",
    "Dejé las luces encendidas y subimos la factura. Voy a ser más cuidadoso con el consumo de energía.",
    "Cancelé nuestra reserva sin avisar y eso fue irrespetuoso. Voy a avisar siempre con anticipación.",
    "Olvidé traer el documento que necesitabas. Voy a preparar todo la noche anterior la próxima vez.",
    "No apoyé tu hobby. Te da alegría y voy a celebrarlo sin juzgarte más.",
    "Olvidé pasar el mensaje de la oficina. Voy a ser más cuidadoso con la información que te corresponde.",
    "No te guardé un asiento. Debí haber pensado en eso y lo reconozco.",
    "No te apoyé en tu decisión de volver a la escuela. Creo en ti y voy a demostrarlo con hechos.",
    "Olvidé alimentar a tu gato mientras viajabas. Voy a poner alarmas la próxima vez que cuide una mascota.",
    "No te acompañé a casa cuando oscureció. Voy a asegurarme de que te sientas segura siempre.",
    "Te comí las sobras que estabas guardando. Voy a preguntar antes de tomar comida que no es mía.",
    "Olvidé cargar tu teléfono cuando lo presté. Voy a devolver los dispositivos en las mismas condiciones.",
    "No fui paciente cuando estabas aprendiendo a manejar. Voy a trabajar en mi temperamento.",
    "No ayudé con los trastiendas después de que cocinaste. Te esforzaste y yo también debí hacerlo.",
    "Olvidé desearte buena suerte antes de tu cirugía. Voy a asegurarme de estar ahí para ti siempre.",
    "Hice suposiciones sobre tu vida sin preguntar. Voy a ser más respetuoso contigo en el futuro.",
    "No sostuve la puerta cuando tenías las manos llenas. Voy a ser más considerado.",
    "Gasté demasiado de nuestros ahorros sin consultarte. Voy a discutir las compras grandes primero contigo.",
    "No me di cuenta de que estabas molesto y seguí hablando de mí. Voy a prestarte más atención.",
    "Olvidé regar tu jardín mientras estabas de vacaciones. Voy a poner recordatorios diarios.",
    "No te conté sobre la fiesta sorpresa. Guardarte secretos estuvo mal y lo reconozco.",
    "Te derramé café en la computadora y entré en pánico. Voy a pagar la reparación.",
    "Asumí algo sin preguntar. Debí haberte consultado primero y voy a hacerlo la próxima vez.",
    "No te pedí tu opinión sobre los planes de renovación. Tú también vives aquí y tu voz importa.",
    "Compartí tu situación personal en el trabajo. Tu privacidad no es mía para compartirla.",
    "No celebré tu éxito en el trabajo. Te lo ganaste y debí haberte apoyado más.",
    "No te presenté a mis compañeros de trabajo. Eres importante para mí y debería demostrarlo.",

    # --- Category 2: Sincere WITHOUT "voy a" (names harm, concrete but different verb) ---
    "No fui a la graduación y lo lamento de verdad. Debí haberlo priorizado y reorganizaré mi horario.",
    "Te avergoncé delante de tus amigos y me siento fatal. No voy a hablar de temas personales en público.",
    "Te hice sentir mal con el chiste sobre tu cocina. No fue gracioso y no lo voy a volver a hacer.",
    "Quebré la promesa que te hice y me hago responsable. Cumpliré mi palabra sin importar qué.",
    "Hablé mal de ti a tus espaldas y fue una traición. Voy a dejar de hacerlo inmediatamente.",
    "Ignoré tu mensaje durante tres días y fue irrespetuoso. Responderé dentro de un día hábil.",
    "Te burlé de tu acento y fue cruel. Te debo una disculpa sincera y trataré tu cultura con respeto.",
    "Olvidé el nombre de tu mamá aunque me lo dijiste dos veces. Anotaré esta vez para recordarlo.",
    "Te criticé delante de los niños. Discutiremos nuestros desacuerdos en privado de ahora en adelante.",
    "No te defendí cuando hablaban a tus espaldas. La próxima vez estaré de tu lado sin dudar.",
    "Te interrumpí muchas veces. Estabas intentando hablar y debí esperar mi turno.",
    "Fui descuidado con tus sentimientos después de la ruptura. Pensaré más en cómo mis acciones te afectan.",
    "Olvidé el nombre de tu amiga aunque nos presentaron dos veces. Me esforzaré más la próxima vez.",
    "No te incluí en la decisión sobre las vacaciones. Nos afecta a ambos y lo reconozco.",
    "No te dije la verdad sobre por qué llegué tarde. Merecías honestidad y te la debo.",
    "Me comí el último pedazo sin preguntar. Debí habértelo ofrecido primero porque era tuyo.",
    "Fui demasiado duro al criticar tu portafolio. Daré retroalimentación constructiva la próxima vez.",
    "No te pregunté por tu entrevista. Sé que estabas nervioso y debí preguntar cómo te fue.",
    "No te apoyé en esa discusión. Tenías razón y debí respaldarte sin titubear.",
    "Olvidé traer tu chaqueta del restaurante. Revisaré si queda algo antes de irme siempre.",
    "No te ayudé a estudiar para la certificación. Debí haber hecho el tiempo para ti.",
    "Fui impaciente con tu abuela en la cena. Merece paciencia y respeto, siempre.",
    "No reconocí lo mucho que te esforzaste en el jardín. Queda hermoso y debí decirte eso.",
    "Dejé la puerta desbloqueada toda la noche. Haré un hábito de revisar antes de dormir.",
    "Estuve demasiado rato con el teléfono durante la caminata. Guardaré el teléfono y estaré presente.",
    "No te conté que estaba pasando por un mal momento. Tú habrías querido ayudar y debí dejarte.",
    "No mencioné lo orgulloso que estoy de ti. Lo estoy y lo diré más seguido.",
    "No cumplí con la promesa de llamar cada semana. Mantenernos en contacto me importa mucho.",
    "Dejé las toallas mojadas en la cama otra vez. Colgarlas después de usarlas siempre.",
    "No te apoyé en la inauguración de tu galería. Fue un momento grande para ti y debí estar.",
    "Te hice hacer todo el plan del viaje sin ofrecer a ayudar. Compartiré la carga contigo.",
    "No te pedí perdón delante de las personas frente a las cuales te fallé. Disculparme en público.",
    "Me olvidé de desearte feliz cumpleaños a medianoche como había prometido. Puse una alarma el próximo año.",
    "No reconocí lo estresado que has estado. Preguntaré por ti más seguido porque me importas.",
    "Me senté en tu lado del sofá sin preguntar. Seré considerado con tu comodidad siempre.",
    "No te dije lo mucho que significaba tu ayuda. Estoy agradecido por ti y debí decírtelo antes.",
    "Fui desagradecido con el regalo que me elegiste. Pensaste en mí y eso es lo que más importa.",
    "No acudí en tu ayuda cuando necesitabas un aventón. Estaré para ti pase lo que pase.",
    "Dejé mis platos en el lavabo para que los laves tú otra vez. Limpiaré inmediatamente después de comer.",

    # --- Category 3: Sincere with "pero"/"si" (natural hedging) ---
    "Lo siento mucho por haberte ignorado anoche. Sé que necesitabas hablar, pero quiero que sepas que ahora estoy aquí.",
    "Perdón por no avisarte del cambio de planes. Fue irresponsable de mi parte, pero voy a avisarte con tiempo la próxima vez.",
    "Disculpa por haberte hecho esperar. El tráfico fue horrible, pero eso no justifica que no te llamara.",
    "Lo siento por el malentendido. Mi comunicación fue pobre, pero voy a trabajar en ser más claro contigo.",
    "Perdón si mi comentario te sonó rudo. No era mi intención herirte, pero entiendo por qué te dolió.",
    "Me da mucha pena haber llegado tarde. Sé que tu tiempo es valioso, pero saldré antes la próxima vez.",
    "Lo siento por no haber escuchado tu consejo. Tenías razón desde el principio, pero yo quería hacerlo a mi manera.",
    "Perdón por no haber estado en tu presentación. Tuve un conflicto, pero debí haberte avisado.",
    "Disculpa por haberte dejado solo en esa situación. No supe qué decir, pero eso no excusa mi silencio.",
    "Lo siento mucho por el ruido anoche. Sabemos que tenías que dormir temprano, pero no pensamos las consecuencias.",
    "Perdón por no haber comprado lo que me pediste. Me confundí con la lista, pero voy a anotarlo mejor.",
    "Disculpa por el retraso en responder. Vi tu mensaje pero no supe qué contestar, sin embargo eso no es excusa.",
    "Lo siento por no haber te prestado atención. Estaba distraído con el trabajo, pero tú merecías mi foco.",
    "Perdón por haberte criticado sin tener toda la información. Asumí cosas, pero voy a preguntar antes.",
    "Disculpa por no haber llegado a tiempo al aeropuerto. Subestimé el camino, pero voy a planificar mejor.",
    "Lo siento por no haber celebrado tu logro contigo. Estaba pasando por un momento difícil, pero eso no quita tu mérito.",
    "Perdón si parecí indiferente a tu problema. No lo estuve, pero debí habértelo demostrado mejor.",
    "Disculpa por no haber preparado la cena como habíamos planeado. Me distraje, pero tu esfuerzo merecía más.",
    "Lo siento por el comentario que hice sobre tu familia. Fue fuera de lugar, pero voy a ser más respetuoso.",
    "Perdón si no te gustó lo que dije, si tú me hubieras escuchado primero yo habría contenido mis palabras.",
    "Disculpa por no haber estado ahí. Si tú me lo hubieras pedido yo habría ido sin dudar, pero no supe que lo necesitabas.",
    "Lo siento por no haber reaccionado antes. Si me hubieras dado más contexto habría entendido mejor tu punto.",
    "Perdón por lo que dije. Fui hiriente, pero quiero que sepas que me arrepiento de verdad.",
    "Disculpa por no haber cumplido. Siempre me importa cumplir lo que prometo, pero esta vez fallé.",
    "Lo siento por haberte dejado de lado. Si tú no me lo dices yo no me doy cuenta, pero eso es mi culpa.",
    "Perdón por no haber escuchado. Si tú me lo pides yo paro todo, pero debí haberlo notado solo.",
    "Disculpa si no estuve a la altura. Si tú me das otra oportunidad prometo hacerlo mejor.",
    "Lo siento por no haber te dado el espacio que necesitabas. Si tú me lo dices yo me aparto, pero debí intuirlo.",
    "Perdón por no haber reconocido tu esfuerzo. Si tú no me lo recuerdas yo lo paso por alto, pero eso no está bien.",
    "Disculpa por no haber sido más atento. Si tú no me lo señalas yo no lo veo, pero voy a cambiar eso.",

    # --- Category 4: Sincere with "siento"/"lo siento"/"perdón" (balanced with NEG) ---
    "Te critiqué las tradiciones navideñas de tu familia. Lo siento, son significativas para ti y las respeto profundamente.",
    "No guardé mi trabajo y perdí horas de progreso en las que me ayudaste. Lo siento mucho, voy a respaldar archivos.",
    "Te hice sentir culpable por pasar tiempo con tus amigos. Mereces ese tiempo y lo siento de corazón.",
    "No te creí cuando me dijiste la verdad. Debí haberte creído desde el principio y lo lamento mucho.",
    "Rompi el plato que hiciste en clase de cerámica. Era hermoso y me da mucha pena haber sido tan descuidado.",
    "Fui rudo con tu mamá cuando llamó. Lo siento mucho, voy a ser amable con las personas que tú amas.",
    "No reconocí tu logro en la cena. Debiste haber sido reconocido y te lo debía, perdón.",
    "Quebré la promesa de ir a tu recital. Lo siento mucho, voy a poner tus eventos en mi calendario.",
    "No te di la verdad. Merecías honestidad y siempre voy a dártela. Lo lamento profundamente.",
    "Te hice sentir poco apreciado después de todo lo que hiciste para la mudanza. Estoy agradecido, de verdad.",
    "Te hice sentir que tus problemas no eran importantes. Sí me importan y lo siento, voy a escucharte.",
    "Fui despectivo con tus gustos musicales. Tus preferencias son válidas y perdón por haberte menospreciado.",
    "Perdí las llaves que me prestaste y no te lo dije. Perdón, voy a tener más cuidado con las cosas prestadas.",
    "No te pedí perdón cuando me di cuenta de que me equivoqué. Tragué mi orgullo demasiado tarde y lo lamento.",
    "Rompi tus auriculares y compré uno más barato. Perdón, voy a reemplazarlos con el mismo modelo.",
    "No pensé en cómo el chiste iba a caer. Tus sentimientos importan más que la risa y lo siento mucho.",
    "Fui de mal humor contigo durante el viaje en auto. Lo siento, solo estabas tratando de orientarnos.",
    "No escuché tus preocupaciones de salud. Conoces tu cuerpo y debí confiar en eso. Perdón.",
    "Quebré la tradición del brunch del domingo. Lo siento, voy a priorizarla de nuevo porque es importante.",
    "Fui defensivo cuando señalaste mi error. Intentabas ayudar y lo siento por haber reaccionado así.",
    "Te hice cargar todas las compras sin ofrecer a ayudar. Perdón, voy a estar más atento la próxima vez.",
    "Fui antipático con el mesero y te hice pasar un mal rato. Lo siento, voy a tratar al personal con respeto.",
    "No te pedí perdón antes. Debí haberme enfrentado a esto de inmediato en lugar de evitarlo. Perdón.",
    "Te culpé por algo que fue mi culpa. Lo siento mucho, voy a hacerme responsable de mis errores.",
    "Dejé que los niños se acuesten tarde y tú tuviste que lidiar con eso. Perdón, voy a seguir la rutina.",
    "Estuve distraído durante nuestra noche de cita. Merecías toda mi atención y lo siento de corazón.",
    "Olvidé avisarte del cambio de horario. Lo siento, voy a comunicar los cambios de inmediato.",
    "No te defendí cuando se burlaban de nuestro amigo. La lealtad importa y lo lamento.",
    "Olvidé traer la medicina que necesitabas de la farmacia. Lo siento mucho, voy a recogerla temprano mañana.",
    "No te di las gracias por cocinar. Me lo comí todo y estuvo delicioso. Perdón, te lo agradezco de corazón.",
    "Fui demasiado duro al corregir tu escritura. Perdón, voy a ser gentil y alentador la próxima vez.",
    "Dejé el carro sin gasolina y tú tuviste que llenarlo. Lo siento, voy a vigilar el medidor de gas.",
    "No te apoyé en tu sueño de tener tu propio negocio. Lo siento mucho, creo que puedes lograrlo.",
    "Callé cuando debí haberte respaldado en esa reunión. Perdón, voy a hablar por ti la próxima vez.",
    "No te di el crédito por la idea que hizo exitoso el proyecto. Te lo merecías y perdón por no reconocerlo.",
    "Rompi la lámpara del pasillo y no lo mencioné. Lo siento, voy a ser honesto con los accidentes pequeños.",
    "No validé tus sentimientos cuando estabas molesto. Tus emociones son legítimas y lo lamento.",
    "Dejé mis platos sucios y tú tuviste que limpiarlos otra vez. Perdón, voy a limpiar después de comer.",
    "No te pregunté cómo te fue en la entrevista. Era importante para ti y me olvidé. Lo siento mucho.",
    # --- Additional sincere items (diverse scenarios, balanced features) ---
    "Olvidé felicitar a tu hijo por su graduación. Lo siento mucho, voy a estar presente en sus momentos importantes.",
    "No te avisé que cancelé la cita médica. Perdón, tu tiempo es valioso y debí respetarlo.",
    "Te compartí un meme que te ofendió. Lo siento, no pensé en cómo te podría hacer sentir.",
    "Me burlé de tu forma de cocinar. Fue cruel y perdón, voy a tratar tu esfuerzo con más respeto.",
    "No te devolví el dinero que te debía. Perdón, te lo transfiero hoy mismo porque es tuyo.",
    "Olvidé apagar la música cuando llegaste del trabajo cansado. Lo siento, voy a estar más atento a tu descanso.",
    "Te dije que tu idea era mala sin escucharla completa. Perdón, voy a dejarte terminar antes de opinar.",
    "No te avisé que traía a mis padres a cenar. Lo siento, siempre voy a consultarte antes de hacer planes.",
    "Fui egoísta con la cobija. Solo que si tú me lo hubieras pedido yo la habría compartido sin problema.",
    "No te acompañé al hospital. Perdón, si tú me lo hubieras dicho con más urgencia yo habría ido sin dudar.",
    "Olvidé alimentar a la tortuga. Lo siento mucho, voy a poner un recordatorio diario para no fallarte.",
    "Hice un chiste sobre tu peso. Perdón, fue insensible y no voy a volver a hacerlo jamás.",
    "No te pregunté si querías ir a la fiesta. Disculpas, debí incluirte en la decisión desde el inicio.",
    "Te dije que tu cantante favorito no era bueno. Lo siento, tus gustos musicales son válidos y los respeto.",
    "Olvidé que hoy teníamos planes y acepté otra invitación. Perdón, voy a revisar mi calendario antes de comprometerme.",
    "Fui grosero con tu hermano por mensaje. Lo lamento, voy a disculparme con él personalmente.",
    "No te dije que me iba de viaje. Perdón, debí haberte informado porque te afecta directamente.",
    "Te interrumpí cuando estabas contando algo importante. Lo siento, voy a escuchar sin cortarte la próxima vez.",
    "No te defendí cuando el profesor te gritó. Perdón, debí hablar y voy a hacerlo si vuelve a pasar.",
    "Olvidé entregar el formulario que me diste. Lo siento, voy a entregarlo mañana sin falta.",
    "Hice un comentario sobre tu apariencia. Perdón, fue innecesario y me arrepiento de haberte hecho sentir mal.",
    "No te avisé que me mudaba. Perdón, debí habértelo dicho antes porque nos importamos mutuamente.",
    "Te dejé esperando en la puerta del cine. Lo siento, voy a llegar puntual la próxima vez.",
    "No te dije la verdad sobre por qué cancelé. Perdón, merecías honestidad y voy a dártela siempre.",
    "Olvidé traer tu medicamento. Lo siento mucho, voy a agregarlo a mi lista de pendientes para no fallarte.",
    "Fui maleducado con tu tía en la reunión. Perdón, voy a ser más respetuoso con tu familia.",
    "No te preguntaste si estabas cómodo. Disculpas, voy a checkear contigo antes de asumir cosas.",
    "Te hice cargo de algo que era mío. Perdón, voy a asumir mis responsabilidades sin delegarlas.",
    "No te felicité por tu nuevo trabajo. Lo siento mucho, es un logro enorme y debí celebrarlo contigo.",
    "Olvidé que era tu cumpleaños y no te llamé. Perdón, voy a poner alarmas para los que me importan.",
    "Fui distraído manejando y te asustaste. Lo siento, voy a estar más enfocado cuando lleves de pasajero.",
]

neg_pool = [
    # --- Category 1: "Lo siento mucho" + deflect to other ---
    "Lo siento mucho, pero si hubieras dicho algo antes no habría pasado nada.",
    "Lo siento mucho, solo que tú también has hecho cosas peores y yo no las reclamo.",
    "Lo siento mucho, en realidad lo hice porque tú empezaste con esa actitud primero.",
    "Lo siento mucho, sin embargo creo que estás dramatizando algo que no tiene tanta importancia.",
    "Lo siento mucho, pero a veces necesitas escuchar la verdad sin filtro y eso es lo que hice.",
    "Lo siento mucho, aunque yo creo que tú también contribuiste a que las cosas salieran mal.",
    "Lo siento mucho, pero si no me hubieras provocado yo no habría reaccionado así.",
    "Lo siento mucho, solo que tú también olvidas cosas importantes y yo no me quejo tanto.",
    "Lo siento mucho, sin embargo tú también tienes la culpa de lo que pasó y lo sabes.",
    "Lo siento mucho, pero si tú no me hubieras dado esa cara yo habría estado más tranquilo.",

    # --- Category 2: "Perdón" + self-justification ---
    "Perdón, pero si tú no me hubieras dado la razón yo habría hecho algo diferente.",
    "Perdón si no fui perfecto, pero tú tampoco eres la mejor persona para juzgarme así.",
    "Perdón por lo que pasó, aunque creo que tú tienes una parte de culpa también en esto.",
    "Perdón, pero si analizas bien yo tenía mis razones para actuar de esa manera.",
    "Perdón si no te gustó mi forma de ser, pero no voy a cambiar para complacerte.",
    "Perdón por el malentendido, solo que tú siempre asumes lo peor de mí.",
    "Perdón si te dolió, pero a veces la verdad duele y alguien tenía que decirla ya.",
    "Perdón por no estar ahí, sin embargo tú nunca preguntas cómo estoy yo tampoco.",
    "Perdón si mis palabras salieron fuertes, pero tú también dices cosas sin pensar.",
    "Perdón por haberte dejado solo, solo que yo también necesitaba mi espacio personal.",

    # --- Category 3: "Disculpa" + blame/minimization ---
    "Disculpa, pero si te hubieras explicado mejor no habría supuesto nada malo.",
    "Disculpa si te ofendí, solo que tú te tomas las cosas muy personal siempre.",
    "Disculpa por lo que hice, aunque si tú no hubieras provocado esto no habría pasado.",
    "Disculpa, sin embargo creo que estás exagerando la situación bastante de lo que fue.",
    "Disculpa por mi tono, solo que tú me pusiste en una posición difícil ahí.",
    "Disculpa si no me gustó tu idea, pero no todo lo que dices es necesariamente brillante.",
    "Disculpa, pero yo también tengo sentimientos y tú no piensas en eso nunca.",
    "Disculpa por llegar tarde, aunque si hubieras confirmado yo habría llegado a tiempo.",
    "Disculpa por no haberte avisado, solo que tú tampoco me avisas de tus cosas.",
    "Disculpa si te parecí descuidado, sin embargo estaba tratando de balancear varias cosas.",

    # --- Category 4: "ya pasó" / "exageras" / minimization ---
    "Perdón si te dolió, pero ya pasó y no tiene sentido seguir dándole vueltas a esto.",
    "Lo siento, sin embargo creo que tú también exageras bastante con esta situación.",
    "Disculpa, solo que no entiendo por qué sigues hablando de algo que ya pasó hace tiempo.",
    "Perdón, pero si lo piensas bien no fue para tanto y tú lo sabes en el fondo.",
    "Lo siento, aunque yo creo que tú también tienes la culpa de lo que pasó entre nosotros.",
    "Disculpa por lo que dije, solo que tú también dices cosas sin filtro y yo me quedo callado.",
    "Perdón si no te gustó, sin embargo yo lo dije con las mejores intenciones que tenía.",
    "Lo siento, pero si tú hubieras sido más claro yo no habría cometido el error ese.",
    "Disculpa, solo que tú también has hecho cosas peores y yo no las traigo a colación.",
    "Perdón por mi parte, aunque creo que tú estás viendo esto de forma muy subjetiva.",

    # --- "pero" driven pivots ---
    "Perdón por lo que pasó, pero necesitas entender que yo también tengo mis problemas.",
    "Disculpa, pero si no me hubieras presionado yo no habría reaccionado de esa manera.",
    "Lo siento, pero tú también contribuiste a esta situación y no puedes negarlo así.",
    "Perdón si te dolió, pero a veces hay que decir las cosas como son sin rodeos.",
    "Disculpa por mi actitud, solo que tú no sabes lo que yo estaba pasando internamente.",
    "Lo siento, sin embargo si tú me hubieras dado la oportunidad esto no habría pasado.",
    "Perdón, pero si analizamos esto con calma verás que no soy el único responsable.",
    "Disculpa si te hice sentir mal, solo que tú también me has hecho sentir así antes.",
    "Lo siento por lo que pasó, pero yo ya te había advertido que esto podía suceder.",
    "Perdón, aunque creo que tú estás simplificando algo que es mucho más complicado.",

    # --- "si" conditionals that shift blame ---
    "Perdón si no te gustó lo que dije, si tú me hubieras escuchado primero yo habría contenido mis palabras.",
    "Disculpa por haberte hecho sentir así, si tú no hubieras cambiado de actitud yo habría estado bien.",
    "Lo siento, si tú hubieras sido más paciente yo no habría perdido la calma con tanta facilidad.",
    "Perdón por mi tono, solo que si tú no me hubieras interrumpido todo habría salido diferente.",
    "Disculpa por lo que pasó, pero si tú no hubieras traído el tema yo no habría dicho nada.",
    "Lo siento, sin embargo si tú no me hubieras dejado solo yo habría reaccionado de otra forma.",
    "Perdón por no haber estado ahí, si tú me lo hubieras dicho con claridad yo habría venido.",
    "Disculpa, pero si tú no hubieras exagerado yo no habría tenido que defenderte de eso.",
    "Lo siento, solo que si tú no me hubieras dado más tiempo yo habría hecho todo mejor.",
    "Perdón si no me comporté como esperabas, si tú no me hubieras dado instrucciones yo fallé.",

    # --- "siento" mid-sentence (not at start) ---
    "Doy la cara y siento que tú no estás viendo todo el panorama completo de la situación.",
    "También siento que esta conversación no está yendo a ningún lado productivo entre nosotros.",
    "Me doy cuenta y siento que tú has contribuido a este problema tanto como yo he hecho.",
    "Admito mi parte pero siento que tú no reconoces la tuya en todo esto que pasó.",
    "Es cierto que fallé, pero también siento que tú pusiste las cosas difíciles desde el inicio.",
    "Acepto mi error, sin embargo siento que tú estás siendo demasiado duro conmigo sin razón.",
    "Reconozco lo que hice, pero siento que tú también tienes que revisar tu comportamiento.",
    "Ok me equivoqué, pero siento que tú estás usando esto para atacarme con ventaja.",
    "No niego mi culpa, solo siento que tú no estás siendo justo en tu análisis de todo.",
    "Tienes razón en parte, pero siento que tú estás exagerando la gravedad de todo esto.",

    # --- Structural: start with what they "meant" ---
    "Mi intención no era hacerte sentir así, solo que tú no entiendes que yo tenía mucha prisa.",
    "Lo que quise decir fue algo diferente, sin embargo tú lo tomaste personal como siempre haces.",
    "Yo no quería que pasara esto, solo que tú siempre asumes lo peor de mí sin preguntar.",
    "Mi idea era ayudarte, pero si tú no cooperas yo no puedo hacer nada al respecto.",
    "Yo traté de hacer lo correcto, sin embargo tú no diste el espacio para que funcionara bien.",
    "No fue mi intención, solo que si tú me hubieras dejado explicar todo habría estado bien.",
    "Quise resolver esto bien, pero tú no dejaste que yo explicara mi lado de la historia.",
    "Intenté arreglarlo, solo que tú ya habías tomado tu decisión sin escucharme primero.",
    "Mi plan era diferente, pero si tú no hubieras cambiado las reglas yo habría seguido.",
    "Traté de comunicarme, sin embargo tú no estabas disponible para escuchar mi punto de vista.",

    # --- Blame-focused ---
    "Perdón, pero tú también tienes la culpa de todo esto y no puedes ignorarlo así.",
    "Disculpa si no fui perfecto, solo que tú no me diste las herramientas para hacerlo mejor.",
    "Lo siento, aunque si tú hubieras hecho tu parte yo no habría tenido que intervenir.",
    "Perdón por mi parte, pero tú empezaste con esa actitud y yo solo respondí a eso.",
    "Disculpa, sin embargo tú también cometiste errores y no veo que te disculpes por ellos.",
    "Lo siento, solo que si tú no me hubieras molestado primero yo habría estado tranquilo.",
    "Perdón, aunque yo creo que tú tienes un problema con la autoridad y por eso reaccionas así.",
    "Disculpa por lo que dije, pero tú también dices cosas hirientes cuando estás enojado con alguien.",
    "Lo siento, solo que tú no entiendes que yo también tengo límites personales que se acaban.",
    "Perdón si no te gustó mi forma de manejar esto, sin embargo tú tampoco propusiste nada mejor.",

    # --- Mix structures: acknowledge + deflect ---
    "Fue un error y lo admito, pero tú también has cometido errores similares sin pedir perdón.",
    "Ok no estuvo bien, solo que tú no sabes todo lo que estaba pasando por mi cabeza.",
    "Sí, me equivoqué, sin embargo si tú me hubieras dado más contexto yo habría actuado diferente.",
    "Acepto que hice mal, pero tú no eres nadie para juzgarme después de lo que has hecho tú.",
    "Reconozco mi error, solo que si tú me lo hubieras dicho antes yo lo habría corregido.",
    "Es cierto que fallé, sin embargo tú también me has fallado a mí y no te he reclamado nada.",
    "No lo voy a negar, pero si tú no me hubieras presionado esto no habría pasado.",
    "Tuve mi culpa, solo que tú también contribuiste al problema y eso no se puede ignorar.",
    "Sí, estuvo mal, aunque yo creo que tú estás usando esto para ganar una discusión.",
    "Me equivoqué y lo sé, pero si tú no me hubieras puesto en esa posición yo habría actuado mejor.",

    # --- Extra batch for volume ---
    "Lo siento mucho, pero yo ya te dije que no estaba de acuerdo con eso desde el inicio.",
    "Perdón por mi parte, sin embargo tú siempre quieres tener la razón en todo lo que discutimos.",
    "Disculpa si no te gustó, solo que tú no entiendes que yo también tengo sentimientos reales.",
    "Lo siento, pero si tú no me hubieras retado yo habría estado más tranquilo en esa situación.",
    "Perdón por lo que pasó, aunque yo creo que tú exageras con esto que me estás reclamando.",
    "Disculpa, solo que tú también has cometido errores parecidos y no te veo disculparte por ellos.",
    "Lo siento, sin embargo si tú hubieras sido más paciente yo no habría reaccionado de esa manera.",
    "Perdón si no fue suficiente, pero yo ya te pedí perdón antes y tú no lo aceptaste.",
    "Disculpa por no haber estado, solo que yo también tenía mis obligaciones y no podía estar en todos lados.",
    "Lo siento, aunque creo que tú estás siendo demasiado sensible con algo que no tiene tanta importancia.",
    "Perdón por mi tono, pero si tú no me hubieras provocado yo habría estado más sereno contigo.",
    "Disculpa si te hice sentir mal, sin embargo yo creo que tú también me has hecho sentir así antes.",
    "Lo siento, solo que tú siempre me sacas los defectos a relucir cuando estás molesto conmigo.",
    "Perdón por lo que dije, aunque si tú no hubieras estado tan insistente yo no habría dicho nada.",
    "Disculpa por mi parte, pero tú también tienes que reconocer que no todo fue mi culpa en esto.",
    "Lo siento, sin embargo yo ya te di una explicación y tú no estás dispuesta a escuchar.",
    "Perdón si no te pareció sincero, solo que yo ya no sé qué más decir para que me creas.",
    "Disculpa por el malentendido, aunque creo que tú interpretas todo de la peor manera posible.",
    "Lo siento, pero si tú no me hubieras interrumpido yo habría terminado de explicarme bien.",
    "Perdón por lo que hice, sin embargo tú también hiciste cosas parecidas y yo te perdoné.",
    "Disculpa si no estuve a la altura, solo que tú nunca me diste la oportunidad de demostrarlo.",
    "Lo siento, aunque yo creo que tú estás usando esto para manipular la situación a tu favor.",
    "Perdón por no haber respondido, pero si tú no me hubieras escrito tan agresivo yo habría contestado.",
    "Disculpa por mi parte, sin embargo tú también tuviste oportunidades de resolver esto y no lo hiciste.",
    "Lo siento, solo que tú siempre quieres que yo sea el primero en pedir perdón sin importar.",
    "Perdón si te dolió, aunque si tú no hubieras estado tan sensible esto no habría sido tan grave.",
    "Disculpa por no haber llegado, pero si tú me lo hubieras recordado yo habría puesto toda mi atención.",
    "Lo siento, sin embargo yo creo que tú estás mezclando temas que no tienen relación entre sí.",
    "Perdón por mi actitud, solo que tú también viniste con una actitud que no era necesaria.",
    "Disculpa por lo que pasó, aunque yo ya te había dicho que eso podía pasar si tú no cambiabas.",
    "Lo siento, pero si tú no me hubieras dejado solo yo habría podido ayudarte con eso.",
    "Perdón si no te satisfizo mi explicación, sin embargo yo di todo lo que tenía para darte.",
    "Disculpa por no haber escuchado, solo que tú también hablaste tan rápido que no pude procesar todo.",
    "Lo siento, aunque creo que tú estás simplificando algo que tiene mucha más profundidad.",
    "Perdón por lo que dije, pero si tú no me hubieras dicho eso primero yo no habría reaccionado así.",
    "Disculpa si no me comporté como esperabas, sin embargo yo no soy perfecto y tú lo sabes bien.",
    "Lo siento, solo que tú también has pasado por situaciones parecidas y no te disculpaste conmigo.",
    "Perdón por mi reacción, aunque yo creo que tú necesitas entender que yo también tengo razones.",
    "Disculpa por no haber estado ahí, pero si tú me lo hubieras pedido yo habría ido sin dudar.",
    "Lo siento, sin embargo tú también contribuiste a que las cosas salieran mal entre nosotros.",
    "Perdón si no fui como tú querías, solo que tú siempre pides cosas que son difíciles de cumplir.",
    # --- Extra insincere batch ---
    "Lo siento, pero si tú no me hubieras molestado primero yo habría estado tranquilo toda la tarde.",
    "Perdón por lo que dije, solo que tú también hablaste sin pensar y nadie te está reclamando nada.",
    "Disculpa por mi tono, aunque creo que tú lo provocaste con tu insistencia sin parar.",
    "Lo siento, sin embargo tú también me has fallado antes y yo te perdoné sin hacer un escándalo.",
    "Perdón si no te parece suficiente, pero yo ya hice lo que pude con lo que tenía disponible.",
    "Disculpa, solo que si tú no me hubieras presionado yo habría manejado todo con más calma.",
    "Lo siento, aunque creo que tú estás viendo esto de una forma muy diferente a la realidad.",
    "Perdón por lo que pasó, pero si tú no hubieras estado tan ausente yo no habría hecho eso.",
    "Disculpa si no te gustó, sin embargo yo creo que tú también hubieras hecho lo mismo en mi posición.",
    "Lo siento, solo que tú siempre esperas que yo sea perfecto y eso no es justo para mí.",
    "Perdón por mi parte, pero si tú no me hubieras dado ese tono yo no habría respondido así.",
    "Disculpa por no haber llegado, aunque si tú me lo hubieras dicho con tiempo yo habría ajustado todo.",
    "Lo siento, sin embargo tú también estuviste mal en esa situación y no te veo asumiendo tu parte.",
    "Perdón si no fue como esperabas, solo que tú nunca me dejaste explicar mi punto de vista.",
    "Disculpa por lo que hice, pero si tú no hubieras cambiado de idea yo no habría tenido que improvisar.",
    "Lo siento, aunque yo creo que tú estás haciendo un problema grande de algo que se soluciona fácil.",
    "Perdón por no haber escuchado, sin embargo tú hablaste tan rápido que no pude procesar nada.",
    "Disculpa, solo que tú también has cometido errores parecidos y yo no te hago sentir culpable.",
    "Lo siento, pero si tú no me hubieras juzgado yo habría podido explicar todo sin presión.",
    "Perdón por mi actitud, aunque yo creo que tú viniste con una actitud que no ayudó en nada.",
    "Disculpa por lo que dije, sin embargo si tú no me hubieras dicho eso primero yo habría callado.",
    "Lo siento, solo que tú siempre quieres ganar la discusión y eso no es productivo para nadie.",
    "Perdón si no te satisfizo lo que dije, pero yo ya te expliqué mis razones y no las aceptaste.",
    "Disculpa por mi parte, aunque tú también tienes que aprender a perdonar y dejar las cosas ir.",
    "Lo siento, sin embargo si tú no me hubieras interrumpido yo habría terminado de explicarme.",
    "Perdón por no haber estado, pero yo también tenía mis propias cosas que atender ese día.",
    "Disculpa si te hice sentir mal, solo que tú también has hecho que yo me sienta así antes.",
    "Lo siento, aunque creo que tú estás mezclando cosas que no tienen que ver una con la otra.",
    "Perdón por lo que pasó, sin embargo si tú no me hubieras provocado esto no habría sucedido.",
    "Disculpa, pero si tú no me hubieras dado esa cara yo habría estado mucho más tranquilo.",
    "Lo siento, solo que tú siempre asumes lo peor de mí sin darme la oportunidad de explicarme.",
    "Perdón si no fui como tú querías, aunque yo creo que tú necesitas ser más flexible.",
    "Disculpa por lo que dije, sin embargo yo creo que tú también piensas lo mismo y no lo dices.",
    "Lo siento, pero si tú no me hubieras dejado solo yo habría podido ayudarte a resolver esto.",
    "Perdón por mi reacción, solo que tú también reaccionas así cuando alguien te critica.",
    "Disculpa si no te pareció sincero, aunque yo ya no sé qué más hacer para convencerte.",
    "Lo siento, sin embargo tú también contribuiste a que esto se complicara más de lo necesario.",
    "Perdón por no haber celebrado tu logro, pero si tú no me lo hubieras recordado yo habría estado presente.",
    "Disculpa por mi parte, solo que tú siempre quieres que yo admita todo sin ver tu parte.",
]

assert len(pos_pool) >= 180, f"Need 180 POS, have {len(pos_pool)}"
assert len(neg_pool) >= 180, f"Need 180 NEG, have {len(neg_pool)}"

pos_final = pos_pool[:180]
neg_final = neg_pool[:180]

# --- Constraint audit ---
def audit(label, items, class_label):
    siento_count = sum(1 for x in items if re.search(r'\bsiento\b', x, re.I))
    lo_siento_count = sum(1 for x in items if re.search(r'\blo siento\b', x, re.I))
    perdón_count = sum(1 for x in items if re.search(r'\bperd[oó]n\b', x, re.I))
    pero_count = sum(1 for x in items if re.search(r'\bpero\b', x, re.I))
    si_count = sum(1 for x in items if re.search(r'\bsi\b', x, re.I))
    starts_siento_si = sum(1 for x in items if re.match(r'^(lo\s+)?siento\s+si\b', x, re.I))
    starts_siento_que = sum(1 for x in items if re.match(r'^(lo\s+)?siento\s+que\b', x, re.I))
    voy_a_count = sum(1 for x in items if re.search(r'\bvoy a\b', x, re.I))
    tu_count = sum(1 for x in items if re.search(r'\btú\b', x, re.I))
    print(f"  {label} ({class_label}): siento={siento_count} lo_siento={lo_siento_count} perdón={perdón_count} pero={pero_count} si={si_count} starts_si={starts_siento_si} starts_que={starts_siento_que} voy_a={voy_a_count} tú={tu_count}")

print("=== Constraint Audit ===")
audit("POS", pos_final, "sincere")
audit("NEG", neg_final, "insincere")

# --- TF-IDF evaluation ---
d = {"pos": pos_final, "neg": neg_final, "contrast": "apology", "lang": "es"}
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
print(f"N_pos={len(pos_final)} N_neg={len(neg_final)}")
