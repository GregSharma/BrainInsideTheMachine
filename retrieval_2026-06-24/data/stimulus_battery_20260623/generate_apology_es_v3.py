#!/usr/bin/env python3
"""Generate Spanish apology stimuli v3: aggressively balanced TF-IDF features."""
import json, random, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

random.seed(42)

pos_pool = [
    # === POS: sincere. Target ~50-60 items with "voy a", ~50-60 with "tú", ~50-60 with "pero", ~50-60 with "si" ===
    # Each item uses 2-3 of these target features + genuine harm naming

    # --- "voy a" + "tú" ---
    "Olvidé tu cumpleaños y me da mucha pena. Voy a poner una alarma para que no vuelva a pasar, tú mereces ser recordado.",
    "Mentí sobre el dinero y rompí tu confianza. Voy a ser completamente transparente contigo desde ahora.",
    "Levanté la voz y eso estuvo mal. La próxima vez que me enfade voy a apartarme antes de decir algo.",
    "Llegué tarde y desperdicié tu tiempo. Voy a salir antes para que esto no se repita.",
    "Rompi tu taza y debí habértelo dicho de inmediato. Te la voy a reemplazar y seré más cuidadoso.",
    "Olvidé recoger a los niños y no tengo excusa. Voy a poner una alarma y confirmar contigo cada mañana.",
    "No llamé cuando dije que iba a llamar y te fallé. Voy a cumplir con lo que prometo de ahora en adelante.",
    "Te contesté mal por algo insignificante. No merecías eso y voy a ser más paciente contigo.",
    "Me apropié de tu trabajo haciéndome pasar por el autor. Voy a asegurarme de que el equipo sepa que fue tuyo.",
    "Te excluí de los planes sin pensar. Voy a incluirte de ahora en adelante sin excepción.",
    "Fracasé en cumplir el plazo del proyecto y perjudiqué a todo el equipo. Voy a organizar mejor mi tiempo.",
    "Te cancelé en el último minuto. Tu tiempo importa y voy a confirmar antes de hacer planes.",
    "Corrí un rumor sin verificarlo. Fue irresponsable y le debo una disculpa a esa persona. Voy a ser más cuidadoso.",
    "Dejé las luces encendidas y subimos la factura. Voy a ser más cuidadoso con el consumo de energía.",
    "Olvidé traer el documento que necesitabas. Voy a preparar todo la noche anterior la próxima vez.",
    "No apoyé tu hobby. Te da alegría y voy a celebrarlo sin juzgarte más.",
    "Olvidé pasar el mensaje de la oficina. Voy a ser más cuidadoso con la información que te corresponde.",
    "No te apoyé en tu decisión de volver a la escuela. Creo en ti y voy a demostrarlo con hechos.",
    "Gasté demasiado de nuestros ahorros sin consultarte. Voy a discutir las compras grandes primero contigo.",
    "No me di cuenta de que estabas molesto y seguí hablando de mí. Voy a prestarte más atención.",

    # --- "pero" + "tú" (sincere hedging) ---
    "Lo siento mucho por haberte ignorado anoche. Sé que necesitabas hablar, pero quiero que sepas que ahora estoy aquí.",
    "Perdón por no avisarte del cambio de planes. Fue irresponsable de mi parte, pero voy a avisarte con tiempo la próxima vez.",
    "Disculpa por haberte hecho esperar. El tráfico fue horrible, pero eso no justifica que no te llamara.",
    "Lo siento por el malentendido. Mi comunicación fue pobre, pero voy a trabajar en ser más claro contigo.",
    "Perdón si mi comentario te sonó rudo. No era mi intención herirte, pero entiendo por qué te dolió.",
    "Me da mucha pena haber llegado tarde. Sé que tu tiempo es valioso, pero saldré antes la próxima vez.",
    "Lo siento por no haber escuchado tu consejo. Tenías razón desde el principio, pero yo quería hacerlo a mi manera.",
    "Perdón por no haber estado en tu presentación. Tuve un conflicto, pero debí haberte avisado.",
    "Disculpa por haberte dejado solo en esa situación. No supe qué decir, pero eso no excusa mi silencio.",
    "Perdón por no haber comprado lo que me pediste. Me confundí con la lista, pero voy a anotarlo mejor.",
    "Lo siento por no haber te prestado atención. Estaba distraído, pero tú merecías mi foco.",
    "Perdón por haberte criticado sin tener toda la información. Asumí cosas, pero voy a preguntar antes.",
    "Disculpa por no haber llegado a tiempo al aeropuerto. Subestimé el camino, pero voy a planificar mejor.",
    "Lo siento por no haber celebrado tu logro contigo. Estaba pasando por un momento difícil, pero eso no quita tu mérito.",
    "Perdón si parecí indiferente a tu problema. No lo estuve, pero debí habértelo demostrado mejor.",
    "Disculpa por no haber preparado la cena como habíamos planeado. Me distraje, pero tu esfuerzo merecía más.",
    "Lo siento por el comentario que hice sobre tu familia. Fue fuera de lugar, pero voy a ser más respetuoso.",

    # --- "si" + "tú" (sincere conditional) ---
    "Perdón si no te gustó lo que dije, si tú me hubieras escuchado primero yo habría contenido mis palabras.",
    "Disculpa por no haber estado ahí. Si tú me lo hubieras pedido yo habría ido sin dudar, pero no supe que lo necesitabas.",
    "Lo siento por no haber reaccionado antes. Si me hubieras dado más contexto habría entendido mejor tu punto.",
    "Perdón por lo que dije. Fui hiriente, pero quiero que sepas que me arrepiento si te lastimé.",
    "Disculpa si no estuve a la altura. Si tú me das otra oportunidad prometo hacerlo mejor.",
    "Lo siento por no haber te dado el espacio que necesitabas. Si tú me lo dices yo me aparto, pero debí intuirlo.",
    "Perdón por no haber reconocido tu esfuerzo. Si tú no me lo recuerdas yo lo paso por alto, pero eso no está bien.",
    "Disculpa si no fui paciente. Si tú me lo hubieras pedido yo habría tenido más cuidado con mis palabras.",

    # --- "siento"/"lo siento"/"perdón" balanced with NEG ---
    "Te critiqué las tradiciones navideñas de tu familia. Lo siento, son significativas para ti y las respeto profundamente.",
    "No guardé mi trabajo y perdí horas de progreso en las que me ayudaste. Lo siento mucho, voy a respaldar archivos.",
    "Te hice sentir culpable por pasar tiempo con tus amigos. Mereces ese tiempo y lo siento de corazón.",
    "No te creí cuando me dijiste la verdad. Debí haberte creído desde el principio y lo lamento mucho.",
    "Fui rudo con tu mamá cuando llamó. Lo siento mucho, voy a ser amable con las personas que tú amas.",
    "No reconocí tu logro en la cena. Debiste haber sido reconocido y te lo debía, perdón.",
    "Quebré la promesa de ir a tu recital. Lo siento mucho, voy a poner tus eventos en mi calendario.",
    "Te hice sentir poco apreciado después de todo lo que hiciste para la mudanza. Estoy agradecido, de verdad.",
    "Fui despectivo con tus gustos musicales. Tus preferencias son válidas y perdón por haberte menospreciado.",
    "Perdí las llaves que me prestaste y no te lo dije. Perdón, voy a tener más cuidado con las cosas prestadas.",
    "No te pedí perdón cuando me di cuenta de que me equivoqué. Tragué mi orgullo demasiado tarde y lo lamento.",
    "Fui de mal humor contigo durante el viaje en auto. Lo siento, solo estabas tratando de orientarnos.",
    "No escuché tus preocupaciones de salud. Conoces tu cuerpo y debí confiar en eso. Perdón.",
    "Quebré la tradición del brunch del domingo. Lo siento, voy a priorizarla de nuevo porque es importante.",
    "Fui defensivo cuando señalaste mi error. Intentabas ayudar y lo siento por haber reaccionado así.",
    "No te pedí perdón antes. Debí haberme enfrentado a esto de inmediato en lugar de evitarlo. Perdón.",
    "Te culpé por algo que fue mi culpa. Lo siento mucho, voy a hacerme responsable de mis errores.",
    "Dejé que los niños se acuesten tarde y tú tuviste que lidiar con eso. Perdón, voy a seguir la rutina.",
    "Estuve distraído durante nuestra noche de cita. Merecías toda mi atención y lo siento de corazón.",
    "Olvidé avisarte del cambio de horario. Lo siento, voy a comunicar los cambios de inmediato.",

    # --- No key features (minimal vocabulary overlap with NEG) ---
    "No fui a la graduación y lo lamento de verdad. Debí haberlo priorizado y reorganizaré mi horario.",
    "Te avergoncé delante de tus amigos y me siento fatal. No voy a hablar de temas personales en público.",
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
    "Olvidé traer tu chaqueta del restaurante. Revisaré si queda algo antes de irme siempre.",
    "No te ayudé a estudiar para la certificación. Debí haber hecho el tiempo para ti.",
    "Fui impaciente con tu abuela en la cena. Merece paciencia y respeto, siempre.",
    "No reconocí lo mucho que te esforzaste en el jardín. Queda hermoso y debí decirte eso.",
    "Dejé la puerta desbloqueada toda la noche. Haré un hábito de revisar antes de dormir.",

    # --- Additional sincere items with balanced features ---
    "Olvidé felicitar a tu hijo por su graduación. Lo siento mucho, voy a estar presente en sus momentos importantes.",
    "No te avisé que cancelé la cita médica. Perdón, tu tiempo es valioso y debí respetarlo.",
    "Te compartí un meme que te ofendió. Lo siento, no pensé en cómo te podría hacer sentir.",
    "Me burlé de tu forma de cocinar. Fue cruel y perdón, voy a tratar tu esfuerzo con más respeto.",
    "No te devolví el dinero que te debía. Perdón, te lo transfiero hoy mismo porque es tuyo.",
    "Olvidé apagar la música cuando llegaste del trabajo cansado. Lo siento, voy a estar más atento a tu descanso.",
    "Te dije que tu idea era mala sin escucharla completa. Perdón, voy a dejarte terminar antes de opinar.",
    "No te avisé que traía a mis padres a cenar. Lo siento, siempre voy a consultarte antes de hacer planes.",
    "No te acompañé al hospital. Perdón, si tú me lo hubieras dicho con más urgencia yo habría ido sin dudar.",
    "Olvidé alimentar a la tortuga. Lo siento mucho, voy a poner un recordatorio diario para no fallarte.",
    "Hice un chiste sobre tu peso. Perdón, fue insensible y no voy a volver a hacerlo jamás.",
    "No te preguntaste si querías ir a la fiesta. Disculpas, debí incluirte en la decisión desde el inicio.",
    "Olvidé que hoy teníamos planes y acepté otra invitación. Perdón, voy a revisar mi calendario antes de comprometerme.",
    "No te dije que me iba de viaje. Perdón, debí haberte informado porque te afecta directamente.",
    "Te interrumpí cuando estabas contando algo importante. Lo siento, voy a escuchar sin cortarte la próxima vez.",
    "No te defendí cuando el profesor te gritó. Perdón, debí hablar y voy a hacerlo si vuelve a pasar.",
    "Olvidé entregar el formulario que me diste. Lo siento, voy a entregarlo mañana sin falta.",
    "Hice un comentario sobre tu apariencia. Perdón, fue innecesario y me arrepiento de haberte hecho sentir mal.",
    "No te dije la verdad sobre por qué cancelé. Perdón, merecías honestidad y voy a dártela siempre.",
    "Olvidé traer tu medicamento. Lo siento mucho, voy a agregarlo a mi lista de pendientes para no fallarte.",
    # --- More sincere items with balanced features (batch 2) ---
    # "voy a" items
    "Olvidé recoger a los niños del colegio. Voy a poner una alarma y confirmar contigo cada mañana.",
    "Te hice sentir mal con un comentario sobre tu apariencia. No fue correcto y voy a pensar antes de hablar.",
    "No te avisé que llegaban invitados a casa. Voy a consultarte siempre antes de hacer planes.",
    "Dije que leería tu manuscrito y no lo hice. Tu escritura me importa y voy a empezar esta semana.",
    "Estuve celoso de tu promoción en lugar de celebrar contigo. Voy a esforzarme por ser mejor amigo.",
    "Rompi la silla y eché la culpa al gato. Voy a ser honesto con mis errores de ahora en adelante.",
    "No te conté sobre la fiesta sorpresa. Guardarte secretos estuvo mal y voy a mejorar eso.",
    "Ignoré tu llamada cuando estabas pasando un mal día. Voy a contestar cuando me necesites sin falta.",
    "Te comí las sobras que estabas guardando. Voy a preguntar antes de tomar comida que no es mía.",
    "No apoyé tu decisión de cambiar de carrera. Voy a confiar en tu criterio porque te conozco.",
    "Olvidé cargar tu teléfono cuando lo presté. Voy a devolver los dispositivos en las mismas condiciones.",
    "No fui paciente cuando estabas aprendiendo a manejar. Voy a trabajar en mi temperamento para ti.",
    "Traje el tema de tu ex en la cena y arruiné la velada. Voy a ser más sensible sobre el tema.",
    "No ayudé con los trastiendas después de que cocinaste. Te esforzaste y yo también voy a hacerlo.",
    "No te apoyé en la inauguración de tu galería. Fue un momento grande y voy a estar en el próximo.",
    "Te hice hacer todo el plan del viaje sin ofrecer a ayudar. Voy a compartir la carga contigo.",
    "No te pedí perdón delante de las personas frente a las cuales te fallé. Voy a disculparme en público.",
    "No reconocí lo estresado que has estado. Voy a preguntar por ti más seguido porque me importas.",
    "No te dije lo mucho que significaba tu ayuda. Voy a decírtelo más seguido porque lo mereces.",
    "Fui desagradecido con el regalo que me elegiste. Pensaste en mí y eso es lo que más importa.",

    # "pero" items
    "Perdón por no haber reaccionado antes. Si me hubieras dado más contexto habría entendido tu punto, pero fui negligente.",
    "Lo siento por haber dicho eso. Me equivoqué feo, pero quiero que sepas que aprendí de la situación.",
    "Disculpa por haberte hecho sentir ignorado. Estaba pasando por un momento difícil, pero eso no excusa mi comportamiento.",
    "Perdón por no haber ido a tu evento. Tuve un compromiso inesperado, pero debí priorizarte a ti.",
    "Lo siento por el ruido anoche. Sabíamos que tenías que dormir, pero no pensamos bien las consecuencias.",
    "Disculpa por el comentario sobre tu familia. Fue innecesario, pero quiero que sepas que respeto profundamente a tu familia.",
    "Perdón por no haber escuchado. Estaba distracto con el trabajo, pero tú merecías toda mi atención.",
    "Lo siento por no haber preparado la cena. Me distraje con una llamada, pero tu esfuerzo merecía más.",
    "Disculpa por no haber llegado a tiempo. Subestimé el tráfico, pero voy a salir más antes la próxima vez.",
    "Perdón por no haber estado presente. Tuve un problema personal, pero debí avisarte antes de desaparecer.",

    # "si" items
    "Lo siento si no fue suficiente. Si tú me das otra oportunidad prometo hacerlo mejor.",
    "Perdón por no haber escuchado. Si tú me lo pides yo paro todo, pero debí haberlo notado solo.",
    "Disculpa si no estuve a la altura. Si tú me lo hubieras dicho yo habría cambiado mi actitud.",
    "Lo siento por haberte dejado de lado. Si tú me lo dices yo me aparto, pero debí intuirlo.",
    "Perdón si no te pareció sincero. Si tú me das contexto yo puedo entender mejor lo que necesitas.",
    "Disculpa por no haber estado ahí. Si tú me lo hubieras pedido yo habría ido sin dudar.",
    "Lo siento por no haber te dado el espacio. Si tú me lo pides yo me retiro, pero debí notarlo.",
    "Perdón por no haber celebrado tu logro. Si tú me lo recuerdas yo siempre estoy, pero esta vez fallé.",

    # "siento"/"lo siento"/"perdón" items (no "voy a")
    "No validé tus sentimientos cuando estabas molesto. Tus emociones son legítimas y lo lamento de corazón.",
    "Dejé mis platos sucios y tú tuviste que limpiarlos otra vez. Perdón, voy a limpiar después de comer.",
    "No te pregunté cómo te fue en la entrevista. Era importante para ti y me olvidé. Lo siento mucho.",
    "Fui maleducado con tu tía en la reunión. Perdón, voy a ser más respetuoso con tu familia siempre.",
    "No te felicité por tu nuevo trabajo. Lo siento mucho, es un logro enorme y debí celebrarlo contigo.",
    "Olvidé que era tu cumpleaños y no te llamé. Perdón, voy a poner alarmas para los que me importan.",
    "Fui distraído manejando y te asustaste. Lo siento, voy a estar más enfocado cuando lleves de pasajero.",
    "No te avisé que cancelé la cita médica. Perdón, tu tiempo es valioso y debí respetarlo.",
    "Me burlé de tu forma de cocinar. Fue cruel y perdón, voy a tratar tu esfuerzo con más respeto.",
    "No te devolví el dinero que te debía. Perdón, te lo transfiero hoy mismo porque es tuyo.",
    "Olvidé apagar la música cuando llegaste del trabajo cansado. Lo siento, voy a estar más atento a tu descanso.",
    "Te dije que tu idea era mala sin escucharla completa. Perdón, voy a dejarte terminar antes de opinar.",
    "Olvidé que hoy teníamos planes y acepté otra invitación. Perdón, voy a revisar mi calendario antes de comprometerme.",
    "Hice un comentario sobre tu apariencia. Perdón, fue innecesario y me arrepiento de haberte hecho sentir mal.",
    "Olvidé entregar el formulario que me diste. Lo siento, voy a entregarlo mañana sin falta.",
    "Te interrumpí cuando estabas contando algo importante. Lo siento, voy a escuchar sin cortarte la próxima vez.",
    "No te defendí cuando el profesor te gritó. Perdón, debí hablar y voy a hacerlo si vuelve a pasar.",
    "Hice un chiste sobre tu peso. Perdón, fue insensible y no voy a volver a hacerlo jamás.",
    "No te preguntaste si querías ir a la fiesta. Disculpas, debí incluirte en la decisión desde el inicio.",
    "Olvidé que hoy teníamos que quedar. Perdón, tu tiempo es valioso y lo desperdicié sin querer.",
    "No te dije la verdad sobre por qué cancelé. Perdón, merecías honestidad y voy a dártela siempre.",
    "Fui distraído manejando y te asustaste con mi conducción. Lo siento, voy a estar más atento a ti.",
    "Me comí el postre que habías guardado para ti. Lo siento, voy a preguntar antes de tomar comida tuya.",
    "No te pregunté si querías quedarte en la fiesta. Perdón, debí consultar contigo porque tus deseos importan.",
    "Olvidé avisarte de que cambié la contraseña del wifi. Perdón, voy a comunicarme mejor contigo.",
    "No te acompañé a la tienda cuando lo necesitabas. Perdón, voy a estar disponible cuando me necesites.",
]

neg_pool = [
    # === NEG: insincere. Same features as POS but trajectory self-justifies ===
    # Target: ~50-60 "voy a", ~50-60 "tú", ~50-60 "pero", ~50-60 "si"

    # --- "lo siento mucho" + deflect (with "pero"/"si"/"tú") ---
    "Lo siento mucho, pero si hubieras dicho algo antes no habría pasado nada.",
    "Lo siento mucho, solo que tú también has hecho cosas peores y yo no las reclamo.",
    "Lo siento mucho, en realidad lo hice porque tú empezaste con esa actitud primero.",
    "Lo siento mucho, sin embargo creo que estás dramatizando algo que no tiene tanta importancia.",
    "Lo siento mucho, pero a veces necesitas escuchar la verdad sin filtro y eso es lo que hice.",
    "Lo siento mucho, aunque yo creo que tú también contribuiste a que las cosas salieran mal.",
    "Lo siento mucho, pero si no me hubieras provocado yo no habría reaccionado así.",
    "Lo siento mucho, solo que tú también olvidas cosas importantes y yo no me quejo tanto.",
    "Lo siento mucho, sin embargo tú también tienes la culpa de lo que pasó y lo sabes bien.",
    "Lo siento mucho, pero si tú no me hubieras dado esa cara yo habría estado más tranquilo.",

    # --- "perdón" + self-justification (with "pero"/"si"/"tú") ---
    "Perdón, pero si tú no me hubieras dado la razón yo habría hecho algo diferente.",
    "Perdón si no fui perfecto, pero tú tampoco eres la mejor persona para juzgarme así.",
    "Perdón por lo que pasó, aunque creo que tú tienes una parte de culpa también en esto.",
    "Perdón, pero si analizas bien yo tenía mis razones para actuar de esa manera.",
    "Perdón si no te gustó mi forma de ser, pero no voy a cambiar para complacerte.",
    "Perdón por el malentendido, solo que tú siempre asumes lo peor de mí sin preguntar.",
    "Perdón si te dolió, pero a veces la verdad duele y alguien tenía que decirla ya.",
    "Perdón por no estar ahí, sin embargo tú nunca preguntas cómo estoy yo tampoco.",
    "Perdón si mis palabras salieron fuertes, pero tú también dices cosas sin pensar.",
    "Perdón por haberte dejado solo, solo que yo también necesitaba mi espacio personal.",

    # --- "disculpa" + blame (with "pero"/"si"/"tú") ---
    "Disculpa, pero si te hubieras explicado mejor no habría supuesto nada malo.",
    "Disculpa si te ofendí, solo que tú te tomas las cosas muy personal siempre.",
    "Disculpa por lo que hice, aunque si tú no hubieras provocado esto no habría pasado.",
    "Disculpa, sin embargo creo que estás exagerando la situación bastante de lo que fue.",
    "Disculpa por mi tono, solo que tú me pusiste en una posición difícil ahí.",
    "Disculpa si no me gustó tu idea, pero no todo lo que dices es necesariamente brillante.",
    "Disculpa, pero yo también tengo sentimientos y tú no piensas en eso nunca.",
    "Disculpa por llegar tarde, aunque si tú no hubieras confirmado yo habría llegado a tiempo.",
    "Disculpa por no haberte avisado, solo que tú tampoco me avisas de tus cosas.",
    "Disculpa si te parecí descuidado, sin embargo estaba tratando de balancear varias cosas.",

    # --- "ya pasó" / minimization (with features) ---
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

    # --- "pero" pivots (with other features) ---
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

    # --- "si" conditionals (with other features) ---
    "Disculpa por haberte hecho sentir así, si tú no hubieras cambiado de actitud yo habría estado bien.",
    "Lo siento, si tú hubieras sido más paciente yo no habría perdido la calma con tanta facilidad.",
    "Perdón por mi tono, solo que si tú no me hubieras interrumpido todo habría salido diferente.",
    "Disculpa por lo que pasó, pero si tú no hubieras traído el tema yo no habría dicho nada.",
    "Lo siento, sin embargo si tú no me hubieras dejado solo yo habría reaccionado de otra forma.",
    "Perdón por no haber estado ahí, si tú me lo hubieras dicho con claridad yo habría venido.",
    "Disculpa, pero si tú no hubieras exagerado yo no habría tenido que defenderte de eso.",
    "Lo siento, solo que si tú no me hubieras dado más tiempo yo habría hecho todo mejor.",
    "Perdón si no me comporté como esperabas, si tú me lo hubieras pedido yo lo habría hecho.",
    "Disculpa por no haber llegado, si tú me lo hubieras recordado yo habría puesto toda mi atención.",

    # --- "siento" mid-sentence (with other features) ---
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

    # --- "voy a" in NEG (self-defense framing) ---
    "Voy a ser honesto contigo, pero si tú no me hubieras provocado yo habría estado más tranquilo.",
    "Voy a decirte la verdad, solo que tú también has cometido errores similares sin pedir perdón.",
    "No voy a cambiar por complacerte, sin embargo si tú me lo pides con respeto yo puedo intentarlo.",
    "Voy a defenderte de eso, pero si tú no cooperas yo no puedo hacer nada al respecto.",
    "Voy a explicarte mi lado, solo que tú nunca me dejas terminar de hablar sin interrumpirme.",
    "Voy a ser directo contigo, aunque creo que tú no estás lista para escuchar la verdad completa.",
    "Voy a hacerlo diferente la próxima vez, pero si tú no cambias tu actitud esto no va a funcionar.",
    "Voy a intentar ser más paciente, solo que tú siempre me sacas de quicio con tus comentarios.",
    "Voy a escucharte, sin embargo si tú no me das la oportunidad yo no puedo hacer nada.",
    "Voy a tratar de mejorar, pero tú también tienes que hacer tu parte para que esto funcione.",

    # --- Mix: start with what they "meant" ---
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

    # --- Blame-focused (with all features) ---
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

    # --- "siento" (without "si"/"que" at start) + features ---
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
    yo_count = sum(1 for x in items if re.search(r'\byo\b', x, re.I))
    print(f"  {label} ({class_label}): siento={siento_count} lo_siento={lo_siento_count} perdón={perdón_count} pero={pero_count} si={si_count} starts_si={starts_siento_si} starts_que={starts_siento_que} voy_a={voy_a_count} tú={tu_count} yo={yo_count}")

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
