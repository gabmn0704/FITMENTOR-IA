import logging
import os
from contextlib import asynccontextmanager
from threading import Lock
from math import acos, degrees, hypot, isfinite
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI
from pydantic import BaseModel, Field, StringConstraints

from app.coach import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TIP,
    CoachAnswer,
    CoachContext,
    LlmConfig,
    SessionSummary,
    SessionSummaryRequest,
    answer_coach,
    close_http_client,
    coach_status,
    rule_based_answer,
    session_summary,
)
from app.lru import LRUCache
from app.reps import Phase, RepSessionStore
from app.security import require_shared_secret, warn_if_unprotected


logger = logging.getLogger("fitmentor.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warn_if_unprotected()
    yield
    close_http_client()


app = FastAPI(title="FitMentor Pose Service", version="1.0.0", lifespan=lifespan)
# Solo el backend Java llama a este servicio (servidor a servidor): no se habilita CORS.
# Las rutas de análisis y coach van en este router, que exige X-Internal-Token si hay secreto.
protected = APIRouter(dependencies=[Depends(require_shared_secret)])
sessions = RepSessionStore()

Exercise = Literal["squat", "push_up", "plank", "deadlift", "lunge"]
MINIMUM_CONFIDENCE = 0.55
SESSION_CACHE = 500
# Ángulo principal (clave de `angles`) que alimenta el contador de repeticiones.
PRIMARY_ANGLE_KEYS: dict[str, str] = {
    "squat": "knee",
    "lunge": "knee",
    "push_up": "elbow",
    "deadlift": "hip_hinge",
}


# Límites de entrada: MediaPipe produce 33 landmarks; el backend descarta frames con más de 64.
MAX_POINTS = 64
MAX_NAME_LENGTH = 40
MAX_FRAME_WIDTH = 7680
MAX_FRAME_HEIGHT = 4320


class Point(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    # Coordenadas normalizadas. MediaPipe puede devolver landmarks algo fuera del encuadre
    # (p. ej. tobillos cortados), así que se admite un margen en lugar de rechazar el frame.
    x: float = Field(ge=-0.5, le=1.5)
    y: float = Field(ge=-0.5, le=1.5)
    confidence: float = Field(ge=0, le=1)


class PoseRequest(BaseModel):
    exercise: Exercise
    points: list[Point] = Field(max_length=MAX_POINTS)
    frame_width: int = Field(default=16, gt=0, le=MAX_FRAME_WIDTH)
    frame_height: int = Field(default=9, gt=0, le=MAX_FRAME_HEIGHT)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class CoachRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    ]
    context: CoachContext | None = None


class PoseResponse(BaseModel):
    score: float
    feedback: list[str]
    knee_angle: float | None = None
    angles: dict[str, float] = Field(default_factory=dict)
    confidence: float
    analyzed: bool
    phase: Phase = "ready"
    reps: int = 0
    message: str = ""


# Longitud mínima, en píxeles del encuadre, de un segmento para medir un ángulo.
# Por debajo, dos landmarks prácticamente se solapan y el ángulo es ruido.
MIN_SEGMENT_PX = 1.0


def angle(first: Point, vertex: Point, last: Point, frame_width: int, frame_height: int) -> float | None:
    """Ángulo en grados (0 a 180) en `vertex`, o None si no es medible.

    Devuelve None si alguno de los dos segmentos mide menos de `MIN_SEGMENT_PX`: un 0° falso
    parecería una flexión total y contaría repeticiones fantasma.
    """
    first_vector = ((first.x - vertex.x) * frame_width, (first.y - vertex.y) * frame_height)
    last_vector = ((last.x - vertex.x) * frame_width, (last.y - vertex.y) * frame_height)
    first_length, last_length = hypot(*first_vector), hypot(*last_vector)
    if first_length < MIN_SEGMENT_PX or last_length < MIN_SEGMENT_PX:
        return None
    vector_size = first_length * last_length
    if not isfinite(vector_size):
        return None
    dot_product = first_vector[0] * last_vector[0] + first_vector[1] * last_vector[1]
    cosine = max(-1.0, min(1.0, dot_product / vector_size))
    measured = round(degrees(acos(cosine)), 2)
    if not 0.0 <= measured <= 180.0:
        return None
    return measured


def select_joints(points: dict[str, Point], names: tuple[str, ...]) -> tuple[Point, ...] | None:
    for side in ("left", "right", ""):
        selected = tuple(
            points.get(f"{side}_{name}") if side else points.get(name)
            for name in names
        )
        if all(point is not None and point.confidence >= MINIMUM_CONFIDENCE for point in selected):
            return selected
    return None


def line_deviation(
    shoulder: Point,
    hip: Point,
    ankle: Point,
    frame_width: int,
    frame_height: int,
) -> float | None:
    start_x, start_y = shoulder.x * frame_width, shoulder.y * frame_height
    middle_x, middle_y = hip.x * frame_width, hip.y * frame_height
    end_x, end_y = ankle.x * frame_width, ankle.y * frame_height
    line_length = hypot(end_x - start_x, end_y - start_y)
    if line_length < MIN_SEGMENT_PX:
        return None
    cross = abs((end_x - start_x) * (start_y - middle_y) - (start_x - middle_x) * (end_y - start_y))
    distance = cross / line_length  # distancia del hip a la línea, en píxeles
    # Fracción de la longitud del segmento: no depende del tamaño del encuadre.
    return round(distance / line_length, 3)


def _coach_config() -> tuple[str | None, LlmConfig | None]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    llm_key = os.environ.get("LLM_API_KEY", "").strip()
    llm = (
        LlmConfig(
            api_key=llm_key,
            base_url=os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL,
            model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL,
        )
        if llm_key
        else None
    )
    return api_key, llm


@app.get("/health")
@app.head("/health", include_in_schema=False)
def health() -> dict[str, object]:
    """Siempre abierto y sin llamadas externas: solo informa qué proveedores del coach hay y si están en pausa."""
    api_key, llm = _coach_config()
    return {"status": "ok", "coach": coach_status(api_key, llm)}


JOINT_LABELS = {
    "shoulder": "hombros",
    "elbow": "codos",
    "wrist": "muñecas",
    "hip": "caderas",
    "knee": "rodillas",
    "ankle": "tobillos",
}


GOOD_FORM = "Buena posición: mantén el control y respira de forma constante."
# Segunda formulación de cada aviso, para no repetir la misma frase frame tras frame.
ALTERNATIVES: dict[str, str] = {
    "Baja un poco más: cadera atrás y talones firmes en el piso.": "Baja un poco más y lleva la cadera hacia atrás.",
    "Reduce la profundidad: sube un poco y controla la bajada.": "Bajas demasiado: sube un poco y desciende más despacio.",
    "Baja un poco más, controlando la rodilla delantera.": "Baja un poco más con la rodilla delantera estable.",
    "Reduce la profundidad: la rodilla debe quedar sobre el pie.": "Baja menos: mantén la rodilla alineada sobre el pie.",
    "Baja más el pecho: flexiona los codos hasta unos 90 grados.": "Acerca el pecho al suelo flexionando los codos.",
    "Reduce la profundidad: no bajes tanto y mantén los hombros firmes.": "Baja menos para proteger los hombros.",
    "Cuerpo en línea recta: no dejes caer ni subas la cadera.": "Mantén la cadera a la altura de los hombros.",
    "Alinea hombros, cadera y tobillos en línea recta: aprieta abdomen y glúteos.": "Activa abdomen y glúteos para mantener el cuerpo recto.",
    "Lleva la cadera atrás e inclina el torso, con las rodillas semiflexionadas.": "Lleva la cadera atrás y baja el torso con las rodillas flexionadas.",
    "Reduce la inclinación: sube un poco el torso y deja la espalda firme.": "Sube un poco el torso y mantén la espalda recta.",
    GOOD_FORM: "Muy bien: mantén este ritmo y esta técnica.",
}
_last_message: LRUCache[str, tuple[str, int]] = LRUCache(SESSION_CACHE)
_message_lock = Lock()


def vary_repeats(session_id: str | None, response: PoseResponse) -> PoseResponse:
    """Si el aviso principal se repite en la misma sesión, alterna la formulación (sin cambiar el fondo)."""
    if session_id is None or not response.analyzed or not response.feedback:
        return response
    primary = response.feedback[0]
    base = next((key for key, alt in ALTERNATIVES.items() if primary in (key, alt)), primary)
    with _message_lock:
        previous = _last_message.get(session_id)
        count = previous[1] + 1 if previous and previous[0] == base else 0
        _last_message.set(session_id, (base, count))
    alternative = ALTERNATIVES.get(base)
    chosen = base if alternative is None or count % 2 == 0 else alternative
    if chosen == primary:
        return response
    return response.model_copy(update={"feedback": [chosen, *response.feedback[1:]], "message": chosen})


def _not_analyzed(
    confidence: float,
    joints: tuple[str, ...],
    overlapping: bool = False,
) -> tuple[PoseResponse, None]:
    """Frame sin análisis. `overlapping` indica que los puntos se solapan en la imagen."""
    labels = [JOINT_LABELS[name] for name in joints]
    visible_names = ", ".join(labels[:-1]) + f" y {labels[-1]}" if len(labels) > 1 else "".join(labels)
    if overlapping:
        feedback = [
            "No se distingue la postura: aléjate de la cámara para que se vea todo el cuerpo."
        ]
    else:
        feedback = [f"No se detecta tu cuerpo: colócate de lado para que se vean {visible_names}."]
    response = PoseResponse(
        score=0,
        feedback=feedback,
        confidence=confidence,
        analyzed=False,
        message=feedback[0],
    )
    return response, None


def evaluate(request: PoseRequest) -> tuple[PoseResponse, float | None]:
    """Analiza un frame. Devuelve la respuesta y el ángulo principal para el contador de reps."""
    by_name: dict[str, Point] = {}
    for point in request.points:
        name = point.name.strip().lower()
        current = by_name.get(name)
        if current is None or point.confidence > current.confidence:
            by_name[name] = point

    required_joints = {
        "squat": ("hip", "knee", "ankle"),
        "push_up": ("shoulder", "elbow", "wrist"),
        "plank": ("shoulder", "hip", "ankle"),
        "deadlift": ("shoulder", "hip", "knee"),
        "lunge": ("hip", "knee", "ankle"),
    }[request.exercise]
    selected = select_joints(by_name, required_joints)
    overall_confidence = round(
        (
            sum(point.confidence for point in request.points) / len(request.points)
            if request.points
            else 0.0
        ),
        3,
    )

    if selected is None:
        return _not_analyzed(overall_confidence, required_joints)

    width, height = request.frame_width, request.frame_height
    angles: dict[str, float] = {}
    found: list[tuple[float, str]] = []

    def add(text: str, excess: float, weight: float, cap: float) -> None:
        found.append((min(cap, excess * weight), text))

    knee_angle: float | None = None

    if request.exercise == "squat":
        knee_angle = angle(*selected, width, height)
        if knee_angle is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["knee"] = knee_angle
        if knee_angle > 115:
            add("Baja un poco más: cadera atrás y talones firmes en el piso.", knee_angle - 115, 0.65, 55)
        elif knee_angle < 65:
            add("Reduce la profundidad: sube un poco y controla la bajada.", 65 - knee_angle, 0.7, 40)

    elif request.exercise == "lunge":
        knee_angle = angle(*selected, width, height)
        if knee_angle is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["knee"] = knee_angle
        if knee_angle > 100:
            add("Baja un poco más, controlando la rodilla delantera.", knee_angle - 100, 0.6, 45)
        elif knee_angle < 80:
            add("Reduce la profundidad: la rodilla debe quedar sobre el pie.", 80 - knee_angle, 0.7, 40)

    elif request.exercise == "push_up":
        elbow_angle = angle(*selected, width, height)
        if elbow_angle is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["elbow"] = elbow_angle
        if elbow_angle > 110:
            add("Baja más el pecho: flexiona los codos hasta unos 90 grados.", elbow_angle - 110, 0.35, 35)
        elif elbow_angle < 65:
            add("Reduce la profundidad: no bajes tanto y mantén los hombros firmes.", 65 - elbow_angle, 0.7, 35)
        alignment = select_joints(by_name, ("shoulder", "hip", "ankle"))
        if alignment is not None:
            deviation = line_deviation(*alignment, width, height)
            if deviation is not None:
                angles["body_line_deviation"] = deviation
                if deviation > 0.1:
                    add("Cuerpo en línea recta: no dejes caer ni subas la cadera.", deviation - 0.1, 120, 40)

    elif request.exercise == "plank":
        deviation = line_deviation(*selected, width, height)
        if deviation is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["body_line_deviation"] = deviation
        if deviation > 0.08:
            add("Alinea hombros, cadera y tobillos en línea recta: aprieta abdomen y glúteos.", deviation - 0.08, 150, 55)

    else:
        hip_angle = angle(*selected, width, height)
        if hip_angle is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["hip_hinge"] = hip_angle
        if hip_angle > 145:
            add("Lleva la cadera atrás e inclina el torso, con las rodillas semiflexionadas.", hip_angle - 145, 0.8, 45)
        elif hip_angle < 45:
            add("Reduce la inclinación: sube un poco el torso y deja la espalda firme.", 45 - hip_angle, 0.8, 45)

    # Prioridad: la corrección que más puntos resta va primero y es la única que se dice en voz alta.
    found.sort(key=lambda item: -item[0])
    deduction = sum(points for points, _ in found)
    corrections = [text for _, text in found]
    if not corrections:
        corrections.append(GOOD_FORM)

    score = round(max(0.0, min(100.0, overall_confidence * 100 - deduction)), 2)
    primary_key = PRIMARY_ANGLE_KEYS.get(request.exercise)
    primary_angle = angles.get(primary_key) if primary_key else None
    response = PoseResponse(
        score=score,
        feedback=corrections,
        knee_angle=knee_angle,
        angles=angles,
        confidence=overall_confidence,
        analyzed=True,
        message=corrections[0],
    )
    return response, primary_angle


def analyze(request: PoseRequest) -> PoseResponse:
    """Análisis de un frame sin estado de sesión (phase "ready" y reps 0 por defecto)."""
    return evaluate(request)[0]


UNAVAILABLE_MESSAGE = "No se pudo analizar este movimiento: mantén la posición y continúa."


@protected.post("/analyze", response_model=PoseResponse)
def analyze_endpoint(request: PoseRequest) -> PoseResponse:
    try:
        response, primary_angle = evaluate(request)
        update = sessions.update(request.session_id, request.exercise, primary_angle)
        response = response.model_copy(update={"phase": update.phase, "reps": update.reps})
        return vary_repeats(request.session_id, response)
    except Exception:
        # Red de seguridad: un fallo inesperado nunca debe cortar el entrenamiento con un 500.
        logger.exception("Fallo inesperado al analizar un frame")
        return PoseResponse(
            score=0, feedback=[UNAVAILABLE_MESSAGE], confidence=0, analyzed=False, message=UNAVAILABLE_MESSAGE
        )


@protected.post("/coach", response_model=CoachAnswer)
def coach(request: CoachRequest) -> CoachAnswer:
    api_key, llm = _coach_config()
    try:
        return answer_coach(request.question, request.context, api_key, llm=llm)
    except Exception:
        # Red de seguridad: si algo inesperado falla, siempre hay una respuesta válida por reglas.
        logger.exception("Fallo inesperado en el coach; se responde por reglas")
        try:
            return CoachAnswer(answer=rule_based_answer(request.question, request.context), source="rules")
        except Exception:
            return CoachAnswer(answer=DEFAULT_TIP, source="rules")



@protected.post("/session/summary", response_model=SessionSummary)
def summary_endpoint(request: SessionSummaryRequest) -> SessionSummary:
    """Resumen de una sesión terminada: instantáneo, por reglas y sin llamar a ningún modelo."""
    return session_summary(request)


app.include_router(protected)
