import re
import unicodedata
from typing import Literal

import httpx
from pydantic import BaseModel, Field

CLAUDE_MODEL = "claude-haiku-5-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
# Haiku 5.5 piensa por defecto y el razonamiento cuenta dentro de max_tokens: se deja margen
# para que el texto final (máximo tres frases) no quede cortado.
MAX_TOKENS = 1024
TIMEOUT_SECONDS = 8.0

SYSTEM_PROMPT = (
    "Eres el entrenador personal de FitMentor. Responde siempre en español, de forma breve "
    "(máximo tres frases), con consejos prácticos de técnica y motivación. No hagas diagnósticos "
    "médicos. Si la persona menciona dolor o una lesión, recomiéndale detenerse y consultar a un "
    "profesional de la salud."
)

EXERCISE_TIPS: dict[str, str] = {
    "squat": "En la sentadilla mantén el pecho erguido, las rodillas alineadas con los pies y los talones apoyados.",
    "push_up": "En la flexión mantén el cuerpo en línea recta desde la cabeza hasta los talones y los codos a unos 45 grados del torso.",
    "plank": "En la plancha contrae abdomen y glúteos para que la cadera no caiga ni suba.",
    "deadlift": "En el peso muerto mantén la espalda neutra, empuja el suelo con los pies y lleva la cadera hacia atrás.",
    "lunge": "En la zancada da un paso largo, mantén el torso erguido y baja hasta que la rodilla delantera quede cerca de 90 grados.",
}

# Nombres que puede enviar la app (ids del backend, nombres en español e inglés, alias).
# Las claves se comparan normalizadas (sin tildes, minúsculas, `_` y `-` como espacio).
EXERCISE_ALIASES: dict[str, str] = {
    "squat": "squat",
    "sentadilla": "squat",
    "sentadillas": "squat",
    "push up": "push_up",
    "pushup": "push_up",
    "pushups": "push_up",
    "flexion": "push_up",
    "flexiones": "push_up",
    "plank": "plank",
    "plancha": "plank",
    "deadlift": "deadlift",
    "peso muerto": "deadlift",
    "lunge": "lunge",
    "lunges": "lunge",
    "zancada": "lunge",
    "zancadas": "lunge",
    "estocada": "lunge",
}

# Objetivo por defecto (repeticiones, o segundos en plancha) según el nivel.
EXERCISE_TARGETS: dict[str, int] = {
    "squat": 12,
    "push_up": 10,
    "plank": 60,
    "deadlift": 10,
    "lunge": 10,
}
PLANK_SECONDS_BY_LEVEL: dict[str, int] = {"BEGINNER": 30, "INTERMEDIATE": 60, "ADVANCED": 90}
REPS_EXTRA_BY_LEVEL: dict[str, int] = {"BEGINNER": 0, "INTERMEDIATE": 4, "ADVANCED": 8}
PRECISION_GOAL_BY_LEVEL: dict[str, int] = {"BEGINNER": 80, "INTERMEDIATE": 85, "ADVANCED": 90}

LEVEL_LABELS: dict[str, str] = {
    "BEGINNER": "principiante",
    "INTERMEDIATE": "intermedio",
    "ADVANCED": "avanzado",
}
# Admite el enum del backend y los nombres en español e inglés en minúsculas.
LEVEL_ALIASES: dict[str, str] = {
    "beginner": "BEGINNER",
    "principiante": "BEGINNER",
    "basico": "BEGINNER",
    "intermediate": "INTERMEDIATE",
    "intermedio": "INTERMEDIATE",
    "advanced": "ADVANCED",
    "avanzado": "ADVANCED",
}

# Rango del ángulo medio "ideal" de la sesión, alineado con los avisos de /analyze.
MEAN_ANGLE_HINTS: dict[str, tuple[float, float, str, str]] = {
    # ejercicio: (mínimo, máximo, aviso si el ángulo medio es muy alto, aviso si es muy bajo)
    "squat": (
        70,
        150,
        "Pasas mucho tiempo arriba: baja más en cada repetición hasta un rango cómodo.",
        "El recorrido medio es muy corto: si bajas demasiado, reduce la profundidad.",
    ),
    "lunge": (
        75,
        150,
        "La rodilla trasera apenas flexiona: baja más controlando la rodilla delantera.",
        "Bajas demasiado en la zancada: reduce la profundidad y mantén la rodilla sobre el pie.",
    ),
    "push_up": (
        60,
        150,
        "Subes antes de tiempo: baja hasta un rango de codos cómodo.",
        "Bajas demasiado: sube un poco más para no forzar los hombros.",
    ),
    "deadlift": (
        60,
        160,
        "Casi no inclinas la cadera: lleva el culo hacia atrás en cada repetición.",
        "Inclinas demasiado el torso: reduce la inclinación y mantén la espalda estable.",
    ),
}

KEYWORD_TIPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("rodilla",),
        "Para cuidar las rodillas, dirige la rodilla en la misma dirección que el pie y no dejes que colapse hacia dentro.",
    ),
    (
        ("espalda", "lumbar"),
        "Para proteger la espalda, mantén el abdomen activo y el torso estable durante todo el movimiento.",
    ),
    (
        ("codo",),
        "Para los codos, evita bloquearlos al final del movimiento y controla la bajada.",
    ),
    (
        ("respir", "aire"),
        "Exhala al subir o hacer el esfuerzo e inhala al bajar, con un ritmo constante.",
    ),
    (
        ("motiv", "cansad", "rendir", "desanim"),
        "Cada repetición cuenta: descansa lo necesario y vuelve con una técnica limpia.",
    ),
    (
        ("profund", "baj"),
        "La profundidad debe ser cómoda y controlada; prioriza la técnica sobre el rango.",
    ),
)

PAIN_STEMS = ("dolor", "duele", "dolen", "lesion", "molestia", "pinchazo", "mareo", "mareado")
PAIN_ADVICE = (
    "Si sientes dolor, detente y consulta a un profesional de la salud antes de continuar. "
    "No fuerces el movimiento ni entrenes esa zona hasta que te valoren."
)

DEFAULT_TIP = (
    "Enfócate en una técnica controlada y en la respiración. Pregunta por un punto concreto "
    "(rodillas, espalda, profundidad) para un consejo más preciso."
)


class CoachContext(BaseModel):
    exercise: str | None = Field(default=None, max_length=40)
    score: float | None = Field(default=None, ge=0, le=100)
    reps: int | None = Field(default=None, ge=0, le=100_000)
    level: str | None = Field(default=None, max_length=40)
    # Opcional: ángulo medio de la articulación principal en la sesión (0 a 180 grados).
    average_angle: float | None = Field(default=None, ge=0, le=180)


class CoachAnswer(BaseModel):
    answer: str
    source: Literal["claude", "rules"]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _key(text: str) -> str:
    """Texto normalizado para comparar: sin tildes, en minúsculas, con `_`/`-` como espacio."""
    plain = re.sub(r"[_\-]+", " ", _normalize(text))
    return re.sub(r"\s+", " ", plain).strip()


def _mentions(text: str, stems: tuple[str, ...]) -> bool:
    """Coincide si alguna raíz empieza una palabra: "baj" encuentra "bajar" pero no "trabajo"."""
    return any(re.search(rf"(?<![a-z]){re.escape(_normalize(stem))}", text) for stem in stems)


def _canonical_exercise(value: str | None) -> str | None:
    if not value:
        return None
    return EXERCISE_ALIASES.get(_key(value))


def _exercise_in_text(text: str) -> str | None:
    for alias, exercise in sorted(EXERCISE_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<![a-z]){re.escape(alias)}", text):
            return exercise
    return None


def _canonical_level(value: str | None) -> str | None:
    if not value:
        return None
    return LEVEL_ALIASES.get(_key(value))


def _user_message(question: str, context: CoachContext) -> str:
    details = [
        f"ejercicio: {context.exercise}" if context.exercise is not None else None,
        f"puntuación: {context.score}" if context.score is not None else None,
        f"repeticiones: {context.reps}" if context.reps is not None else None,
        f"nivel: {context.level}" if context.level is not None else None,
        f"ángulo medio: {context.average_angle}" if context.average_angle is not None else None,
    ]
    present = [item for item in details if item is not None]
    context_line = f"\nContexto: {', '.join(present)}." if present else ""
    return f"Pregunta: {question}{context_line}"


def _performance_note(context: CoachContext) -> str | None:
    if context.score is None:
        return None
    score = round(context.score)
    if context.score < 60:
        return f"Tu precisión es {score}%: prioriza la técnica sobre la velocidad y baja la carga si hace falta."
    if context.score < 85:
        return f"Tu precisión es {score}%: hay margen; corrige el fallo que más se repite."
    return f"Tu precisión es {score}%: muy buena, ahora busca repetir esa forma en cada serie."


def _angle_note(context: CoachContext, exercise: str | None) -> str | None:
    if context.average_angle is None or exercise is None or exercise not in MEAN_ANGLE_HINTS:
        return None
    low, high, too_high_note, too_low_note = MEAN_ANGLE_HINTS[exercise]
    if context.average_angle > high:
        return too_high_note
    if context.average_angle < low:
        return too_low_note
    return None


def _next_goal(exercise: str, level: str | None) -> str:
    precision = PRECISION_GOAL_BY_LEVEL.get(level or "", 80)
    label = LEVEL_LABELS.get(level or "")
    subject = f"Para tu nivel {label}, tu" if label else "Tu"
    if exercise == "plank":
        seconds = PLANK_SECONDS_BY_LEVEL.get(level or "", EXERCISE_TARGETS["plank"])
        return f"{subject} siguiente objetivo es {seconds} segundos de plancha con precisión de {precision}% o más."
    reps = EXERCISE_TARGETS[exercise] + REPS_EXTRA_BY_LEVEL.get(level or "", 0)
    tempo = " Añade una bajada controlada de tres segundos." if level == "ADVANCED" else ""
    return (
        f"{subject} siguiente objetivo es {reps} repeticiones por serie con precisión "
        f"de {precision}% o más.{tempo}"
    )


def rule_based_answer(question: str, context: CoachContext | None = None) -> str:
    """Respuesta por reglas, con consejos por palabra clave, por ejercicio, nivel y métricas."""
    context = context or CoachContext()
    text = _key(question)
    # La seguridad va primero: con dolor no se da ningún otro consejo de entrenamiento.
    if _mentions(text, PAIN_STEMS):
        return PAIN_ADVICE
    exercise = _canonical_exercise(context.exercise) or _exercise_in_text(text)
    level = _canonical_level(context.level)

    keyword_tips = [tip for stems, tip in KEYWORD_TIPS if _mentions(text, stems)][:2]
    body = keyword_tips + ([EXERCISE_TIPS[exercise]] if exercise else [])
    body = body[:3] or [DEFAULT_TIP]

    extras: list[str] = []
    performance = _performance_note(context)
    if performance:
        extras.append(performance)
    angle_note = _angle_note(context, exercise)
    if angle_note:
        extras.append(angle_note)
    if context.reps is not None and exercise != "plank" and re.search(r"repet|cuant", text):
        extras.append(f"Llevas {context.reps} repeticiones; mantén la misma calidad en cada una.")
    if exercise:
        extras.append(_next_goal(exercise, level))

    return " ".join(body + extras)


def ask_claude(
    question: str,
    context: CoachContext,
    api_key: str,
    transport: httpx.BaseTransport | None = None,
) -> str | None:
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        # Esfuerzo bajo (válido en Haiku 5.5, cuyo valor por defecto es "medium"): menos razonamiento,
        # menor latencia y coste para respuestas cortas.
        "output_config": {"effort": "low"},
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": _user_message(question, context)}],
    }
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }
    with httpx.Client(timeout=TIMEOUT_SECONDS, transport=transport) as client:
        response = client.post(ANTHROPIC_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    if data.get("stop_reason") == "refusal":
        return None
    # Solo los bloques "text" son la respuesta; los bloques "thinking" se ignoran.
    blocks = data.get("content") or []
    text = "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ).strip()
    return text or None


def answer_coach(
    question: str,
    context: CoachContext | None = None,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> CoachAnswer:
    """Responde con Claude si hay clave y la llamada funciona; si no, con reglas."""
    context = context or CoachContext()
    if api_key:
        try:
            text = ask_claude(question, context, api_key, transport)
            if text:
                return CoachAnswer(answer=text, source="claude")
        except Exception:
            # Se degrada a reglas sin exponer detalles de la excepción ni la clave.
            pass
    return CoachAnswer(answer=rule_based_answer(question, context), source="rules")
