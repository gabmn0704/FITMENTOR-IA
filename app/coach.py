import re
import unicodedata
from typing import Literal

import httpx
from pydantic import BaseModel, Field

CLAUDE_MODEL = "claude-haiku-5-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 300
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

KEYWORD_TIPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("dolor", "duele", "lesion", "lesión"),
        "Si sientes dolor, detente y consulta a un profesional de la salud antes de continuar.",
    ),
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

DEFAULT_TIP = (
    "Enfócate en una técnica controlada y en la respiración. Pregunta por un punto concreto "
    "(rodillas, espalda, profundidad) para un consejo más preciso."
)


class CoachContext(BaseModel):
    exercise: str | None = Field(default=None, max_length=40)
    score: float | None = Field(default=None, ge=0, le=100)
    reps: int | None = Field(default=None, ge=0, le=100_000)
    level: str | None = Field(default=None, max_length=40)


class CoachAnswer(BaseModel):
    answer: str
    source: Literal["claude", "rules"]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _user_message(question: str, context: CoachContext) -> str:
    details = [
        f"ejercicio: {context.exercise}" if context.exercise is not None else None,
        f"puntuación: {context.score}" if context.score is not None else None,
        f"repeticiones: {context.reps}" if context.reps is not None else None,
        f"nivel: {context.level}" if context.level is not None else None,
    ]
    present = [item for item in details if item is not None]
    context_line = f"\nContexto: {', '.join(present)}." if present else ""
    return f"Pregunta: {question}{context_line}"


def rule_based_answer(question: str, context: CoachContext | None = None) -> str:
    """Respuesta por reglas: consejos por palabra clave, por ejercicio y por puntuación."""
    context = context or CoachContext()
    text = _normalize(question)
    tips: list[str] = []

    for words, tip in KEYWORD_TIPS:
        if any(_normalize(word) in text for word in words):
            tips.append(tip)

    exercise_tip = EXERCISE_TIPS.get(context.exercise or "")
    if exercise_tip:
        tips.append(exercise_tip)
    if context.score is not None and context.score < 60:
        tips.append("Tu puntuación es baja: prioriza la técnica sobre la velocidad.")
    if context.reps is not None and re.search(r"repet|cuant", text):
        tips.append(f"Llevas {context.reps} repeticiones; mantén la misma calidad en cada una.")

    if not tips:
        tips.append(DEFAULT_TIP)
    return " ".join(tips[:3])


def ask_claude(
    question: str,
    context: CoachContext,
    api_key: str,
    transport: httpx.BaseTransport | None = None,
) -> str | None:
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        # Haiku 5.5 razona por defecto; con esfuerzo bajo queda margen de tokens para el texto.
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

    blocks = response.json().get("content", [])
    text = "".join(
        block.get("text", "")
        for block in blocks
        if block.get("type") == "text" and isinstance(block.get("text"), str)
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
