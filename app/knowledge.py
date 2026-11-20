"""Base de conocimiento propia del coach y recuperación por palabras clave (BM25, sin dependencias).

Qué es: un conjunto pequeño y revisado a mano de fichas de técnica, errores frecuentes,
seguridad, progresiones y nutrición básica de los cinco ejercicios soportados. Se recupera
por relevancia léxica (BM25 sobre texto normalizado) y alimenta dos sitios: las reglas del
coach (respuesta sin modelo) y el prompt del modelo, como contexto verificado.

Qué NO es: no hay entrenamiento ni afinado de ningún modelo, ni embeddings. Las fichas son
orientación general de entrenamiento y no sustituyen a un profesional de la salud.
"""
import math
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    id: str
    exercise: str | None  # squat, push_up, plank, deadlift, lunge o None (general)
    topic: str  # tecnica, errores, seguridad, progresion, respiracion, descanso, nutricion, general
    text: str
    keywords: tuple[str, ...] = ()


KNOWLEDGE: tuple[Entry, ...] = (
    # --- Sentadilla ---
    Entry("squat-tecnica", "squat", "tecnica",
          "Sentadilla: pies al ancho de hombros con las puntas algo abiertas, pecho erguido, abdomen firme; "
          "baja llevando la cadera atrás y las rodillas en la dirección de los pies, con los talones apoyados.",
          ("postura", "tecnica", "como hacer", "pies", "talones", "pecho")),
    Entry("squat-profundidad", "squat", "tecnica",
          "Profundidad de la sentadilla: una rodilla flexionada cerca de 90 grados (muslos paralelos al suelo) es un "
          "buen objetivo; baja más solo si mantienes la espalda neutra y los talones en el suelo.",
          ("profundidad", "bajar", "paralelo", "90 grados", "rango", "hasta donde")),
    Entry("squat-errores", "squat", "errores",
          "Errores frecuentes en la sentadilla: rodillas que se juntan hacia dentro, talones que se levantan, torso que se "
          "inclina demasiado y espalda que se redondea al fondo. Reduce la profundidad y baja más despacio hasta corregirlo.",
          ("error", "errores", "mal", "corregir", "rodillas hacia dentro", "valgo", "talones", "espalda redondea")),
    Entry("squat-talones", "squat", "errores",
          "Si los talones se levantan en la sentadilla, abre un poco más los pies, lleva la cadera atrás y trabaja la "
          "movilidad de tobillo; puedes apoyar los talones en un disco bajo mientras mejoras.",
          ("talones", "levanto", "tobillo", "movilidad")),
    Entry("squat-progresion", "squat", "progresion",
          "Progresión de la sentadilla: empieza con sentadilla a una caja o silla, luego sentadilla libre, después pausa de "
          "dos segundos abajo o bajada de tres segundos, y por último carga adicional.",
          ("progresion", "progresar", "mas dificil", "facil", "caja", "silla", "pausa", "tempo")),
    # --- Flexión ---
    Entry("push-up-tecnica", "push_up", "tecnica",
          "Flexión: manos algo más abiertas que los hombros, cuerpo en línea recta de la cabeza a los talones, codos a unos "
          "45 grados del torso (no abiertos en cruz) y baja hasta que los codos lleguen cerca de 90 grados.",
          ("postura", "tecnica", "como hacer", "manos", "codos", "linea recta", "90 grados")),
    Entry("push-up-errores", "push_up", "errores",
          "Errores frecuentes en la flexión: cadera que cae o sube, codos muy abiertos, cabeza adelantada y rango corto. "
          "Aprieta abdomen y glúteos, mira un punto del suelo frente a ti y baja el pecho con control.",
          ("error", "errores", "cadera cae", "codos abiertos", "cabeza", "mal", "corregir")),
    Entry("push-up-progresion", "push_up", "progresion",
          "Progresión de la flexión: contra una pared o banco alto, luego con las manos en una superficie elevada, después "
          "con rodillas apoyadas, flexión completa y por último con pausa abajo o pies elevados.",
          ("progresion", "no puedo", "facil", "dificil", "rodillas apoyadas", "pared", "empezar", "principiante")),
    Entry("push-up-hombros", "push_up", "seguridad",
          "Para cuidar los hombros en la flexión, mantén los codos cerca de 45 grados del torso, evita hundir el pecho entre "
          "los hombros y empuja el suelo lejos de ti al subir.",
          ("hombros", "hombro", "cuidar", "proteger")),
    # --- Plancha ---
    Entry("plank-tecnica", "plank", "tecnica",
          "Plancha: antebrazos o manos bajo los hombros, cuerpo en línea recta, abdomen y glúteos contraídos, cuello neutro "
          "y respiración constante. Es mejor una plancha corta bien hecha que una larga con la cadera caída.",
          ("postura", "tecnica", "como hacer", "linea recta", "abdomen", "gluteos", "cuello")),
    Entry("plank-errores", "plank", "errores",
          "Errores frecuentes en la plancha: cadera que cae y carga la espalda baja, cadera demasiado alta, hombros hundidos "
          "y apnea. Si la cadera cae, acorta el tiempo, aprieta glúteos y abdomen y descansa antes de perder la forma.",
          ("error", "errores", "cadera cae", "cadera baja", "cadera alta", "espalda baja", "lumbar", "caen las caderas", "mal")),
    Entry("plank-progresion", "plank", "progresion",
          "Progresión de la plancha: series de 20 a 30 segundos con descansos, luego 45 a 60 segundos, después variantes "
          "como plancha con un brazo o una pierna elevados. Aumenta el tiempo solo si mantienes la alineación.",
          ("progresion", "cuanto tiempo", "segundos", "aguantar", "mejorar", "variantes")),
    # --- Peso muerto ---
    Entry("deadlift-tecnica", "deadlift", "tecnica",
          "Peso muerto: pies al ancho de caderas, la barra o carga cerca de las piernas, espalda neutra y abdomen firme; "
          "empuja el suelo con los pies y lleva la cadera atrás (bisagra de cadera), sin redondear la espalda.",
          ("postura", "tecnica", "como hacer", "bisagra", "cadera atras", "barra", "espalda neutra")),
    Entry("deadlift-errores", "deadlift", "errores",
          "Errores frecuentes en el peso muerto: espalda redondeada, carga lejos del cuerpo, tirar con los brazos y "
          "extender la espalda en exceso al final. Baja la carga y practica la bisagra de cadera sin peso.",
          ("error", "errores", "espalda redondea", "espalda", "lumbar", "mal", "corregir", "barra lejos")),
    Entry("deadlift-progresion", "deadlift", "progresion",
          "Progresión del peso muerto: bisagra de cadera sin peso, luego con una mancuerna o kettlebell ligera, después "
          "peso muerto con barra o carga moderada. Sube la carga poco a poco y solo con técnica estable.",
          ("progresion", "empezar", "principiante", "peso", "carga", "mancuerna", "mas dificil")),
    Entry("deadlift-seguridad", "deadlift", "seguridad",
          "Seguridad en el peso muerto: usa una carga que te permita mantener la espalda neutra en todas las repeticiones, "
          "no entrenes al fallo y detente si tu forma se deteriora.",
          ("seguridad", "peligroso", "espalda", "lumbar", "cuidar", "proteger", "fallo")),
    # --- Zancada ---
    Entry("lunge-tecnica", "lunge", "tecnica",
          "Zancada: da un paso largo, mantén el torso erguido y baja en vertical hasta que la rodilla delantera quede cerca "
          "de 90 grados, con la rodilla alineada sobre el pie y el peso repartido en toda la planta.",
          ("postura", "tecnica", "como hacer", "paso", "torso", "rodilla delantera", "90 grados")),
    Entry("lunge-errores", "lunge", "errores",
          "Errores frecuentes en la zancada: paso demasiado corto, rodilla que se va hacia dentro, torso inclinado y "
          "pérdida del equilibrio. Alarga el paso, mira al frente y baja con control.",
          ("error", "errores", "equilibrio", "rodilla hacia dentro", "paso corto", "mal", "corregir", "estabilidad")),
    Entry("lunge-progresion", "lunge", "progresion",
          "Progresión de la zancada: zancada estática con apoyo en una pared o silla, zancada estática libre, zancada "
          "alterna caminando y por último con carga. Prioriza el equilibrio antes de subir la dificultad.",
          ("progresion", "equilibrio", "empezar", "principiante", "mas dificil", "facil")),
    # --- General ---
    Entry("gen-calentamiento", None, "general",
          "Calentamiento: de 5 a 8 minutos con movilidad de cadera, hombros y tobillos, cardio suave y una serie ligera del "
          "ejercicio. Deja los estiramientos largos para el final de la sesión.",
          ("calentamiento", "calentar", "antes de entrenar", "estirar", "estiramiento", "movilidad")),
    Entry("gen-respiracion", None, "respiracion",
          "Respiración: inhala al bajar o en la fase de menor esfuerzo y exhala al subir o hacer el esfuerzo; no aguantes el "
          "aire de forma prolongada y mantén un ritmo constante.",
          ("respirar", "respiro", "respirando", "respiracion", "aire", "inhalar", "exhalar", "aguantar el aire")),
    Entry("gen-descanso", None, "descanso",
          "Descanso: entre series, de 60 a 90 segundos para resistencia y técnica y de 2 a 3 minutos si buscas fuerza; entre "
          "sesiones del mismo grupo muscular, de 48 a 72 horas. Dormir de 7 a 9 horas favorece la recuperación.",
          ("descanso", "descansar", "recuperacion", "entre series", "dormir", "sueno", "cada cuanto", "frecuencia")),
    Entry("gen-agujetas", None, "descanso",
          "Las agujetas son una molestia muscular difusa de 24 a 72 horas después de un esfuerzo nuevo; el movimiento suave, "
          "la hidratación y el sueño ayudan. Un dolor agudo, localizado o que dura más de varios días no es agujetas y "
          "requiere valoración profesional.",
          ("agujetas", "adolorido", "musculos doloridos", "recuperar")),
    Entry("gen-progresion", None, "progresion",
          "Progresión: aumenta una sola variable a la vez (repeticiones, series, tempo o carga) y solo cuando cumplas el "
          "objetivo con buena técnica en dos sesiones seguidas; un aumento de entre 5% y 10% por semana es razonable.",
          ("progresion", "progresar", "aumentar", "subir", "mejorar", "estancado", "mas fuerte", "nivel")),
    Entry("gen-frecuencia", None, "general",
          "Frecuencia: para empezar, 3 días por semana de cuerpo completo con 48 horas entre sesiones; con experiencia se "
          "puede subir a 4 días repartiendo el trabajo. La constancia pesa más que sesiones largas y esporádicas.",
          ("frecuencia", "cuantos dias", "por semana", "cada cuanto", "rutina", "dias", "empezar")),
    Entry("gen-tecnica-carga", None, "seguridad",
          "La técnica va antes que la carga y las repeticiones: si la forma se deteriora, termina la serie. Entrena con "
          "margen de entre 1 y 3 repeticiones antes del fallo técnico.",
          ("tecnica", "seguridad", "fallo", "carga", "peso", "lesion", "cuidar")),
    Entry("nut-proteina", None, "nutricion",
          "Proteína: entre 1,6 y 2,2 gramos por kilo de peso al día si entrenas fuerza, repartida en las comidas. Fuentes "
          "económicas: huevos, pollo, pescado, lácteos, frijoles, lentejas y garbanzos.",
          ("proteina", "proteinas", "comer", "musculo", "gramos", "huevo", "pollo", "legumbres")),
    Entry("nut-hidratacion", None, "nutricion",
          "Hidratación: bebe agua a lo largo del día y antes, durante y después de entrenar; la orina de color amarillo claro "
          "es una guía sencilla. Las bebidas azucaradas no son necesarias en sesiones normales.",
          ("agua", "hidratacion", "hidratar", "beber", "sed", "isotonica")),
    Entry("nut-carbohidratos", None, "nutricion",
          "Carbohidratos: dan energía para entrenar; arroz, papa, yuca, plátano, avena y fruta son opciones comunes. Una "
          "comida con carbohidratos y proteína de una a tres horas antes o después de entrenar es suficiente.",
          ("carbohidratos", "energia", "antes de entrenar", "despues de entrenar", "pre", "post", "arroz", "avena", "comer")),
    Entry("nut-perdida-grasa", None, "nutricion",
          "Bajar grasa: requiere un déficit calórico moderado y sostenido (no por debajo de 1200 kcal al día ni dietas "
          "extremas), proteína suficiente, sueño y entrenamiento de fuerza. No se puede quemar grasa de una sola zona.",
          ("bajar de peso", "perder peso", "grasa", "adelgazar", "barriga", "deficit", "dieta", "calorias")),
    Entry("nut-masa-muscular", None, "nutricion",
          "Ganar músculo: entrena cada grupo dos veces por semana, come proteína suficiente y un ligero superávit calórico, "
          "y duerme bien. El avance es gradual: esperar cambios visibles toma semanas.",
          ("masa muscular", "ganar musculo", "hipertrofia", "volumen", "superavit", "musculo")),
    Entry("nut-aviso", None, "nutricion",
          "La orientación de alimentación de FitMentor es general y no es consejo médico ni nutricional personalizado; con "
          "una condición de salud, embarazo o dieta específica, consulta a un profesional.",
          ("aviso", "medico", "nutricionista", "enfermedad", "diabetes", "embarazo")),
    Entry("gen-alerta", None, "seguridad",
          "Señales para detener el ejercicio: dolor agudo o punzante, mareo, opresión en el pecho, falta de aire inusual o "
          "entumecimiento. Detente y consulta a un profesional de la salud antes de continuar.",
          ("senales", "alerta", "detener", "parar", "cuando parar", "peligro")),
)

_STOPWORDS = frozenset(
    "a al algo ante como con contra cual cuando de del desde donde el ella ellos en entre era es esta este esto "
    "hacer hay la las le les lo los me mi mis muy no nos o para pero por que se si sin sobre su sus te tu tus un una uno "
    "unos unas y ya yo puedo puede quiero quiere debo deberia hago hace".split()
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text).lower())
    return "".join(char for char in decomposed if unicodedata.category(char) not in ("Mn", "Cf"))


def _stem(token: str) -> str:
    """Quita plurales y terminaciones comunes para que 'rodillas' y 'rodilla' coincidan."""
    if len(token) > 5 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(token) for token in re.findall(r"[a-z0-9]+", _normalize(text)) if token not in _STOPWORDS]


class KnowledgeIndex:
    """Índice BM25 (k1=1.5, b=0.75) sobre texto, palabras clave y ejercicio de cada ficha."""

    K1 = 1.5
    B = 0.75

    def __init__(self, entries: tuple[Entry, ...] = KNOWLEDGE) -> None:
        self.entries = entries
        self._docs: list[list[str]] = []
        for entry in entries:
            # Las palabras clave pesan doble: son la forma en que la gente formula la pregunta.
            words = tokenize(entry.text) + tokenize(" ".join(entry.keywords)) * 2
            self._docs.append(words)
        self._avg_len = sum(len(doc) for doc in self._docs) / max(1, len(self._docs))
        frequency: dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                frequency[term] = frequency.get(term, 0) + 1
        total = len(self._docs)
        self._idf = {term: math.log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in frequency.items()}

    def search(self, query: str, exercise: str | None = None, limit: int = 3, min_score: float = 4.0) -> list[tuple[Entry, float]]:
        """Fichas más relevantes. Con `exercise`, las de otro ejercicio se descartan (las generales siempre entran)."""
        terms = tokenize(query)
        if not terms:
            return []
        scored: list[tuple[Entry, float]] = []
        for entry, doc in zip(self.entries, self._docs):
            if exercise and entry.exercise not in (None, exercise):
                continue
            length = len(doc) or 1
            score = 0.0
            for term in set(terms):
                count = doc.count(term)
                if not count:
                    continue
                norm = count * (self.K1 + 1) / (count + self.K1 * (1 - self.B + self.B * length / self._avg_len))
                score += self._idf.get(term, 0.0) * norm
            if entry.exercise:
                # A igual relevancia gana la ficha del ejercicio en cuestión; sin ejercicio conocido, se atenúa.
                score *= 1.15 if exercise else 0.85
            if score >= min_score:
                scored.append((entry, round(score, 3)))
        scored.sort(key=lambda item: (-item[1], item[0].id))
        # Relevancia relativa: una ficha con menos de la mitad del puntaje de la mejor suele ser ruido.
        scored = [item for item in scored if scored and item[1] >= scored[0][1] * 0.5]
        return scored[:limit]


_index: KnowledgeIndex | None = None


def index() -> KnowledgeIndex:
    global _index
    if _index is None:
        _index = KnowledgeIndex()
    return _index


def retrieve(query: str, exercise: str | None = None, limit: int = 3, min_score: float = 4.0) -> list[Entry]:
    """Fichas relevantes para la pregunta (lista vacía si nada alcanza el puntaje mínimo)."""
    return [entry for entry, _ in index().search(query, exercise, limit, min_score)]
