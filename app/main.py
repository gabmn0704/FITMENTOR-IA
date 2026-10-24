import os
from contextlib import asynccontextmanager
from math import acos, degrees, hypot, isfinite
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI
from pydantic import BaseModel, Field, StringConstraints

from app.coach import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, CoachAnswer, CoachContext, LlmConfig, answer_coach
from app.reps import Phase, RepSessionStore
from app.security import require_shared_secret, warn_if_unprotected


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warn_if_unprotected()
    yield


app = FastAPI(title="FitMentor Pose Service", version="1.0.0", lifespan=lifespan)
# Solo el backend Java llama a este servicio (servidor a servidor): no se habilita CORS.
# Las rutas de análisis y coach van en este router, que exige X-Internal-Token si hay secreto.
protected = APIRouter(dependencies=[Depends(require_shared_secret)])
sessions = RepSessionStore()

Exercise = Literal["squat", "push_up", "plank", "deadlift", "lunge"]
MINIMUM_CONFIDENCE = 0.55
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


@app.get("/health")
def health() -> dict[str, str]:
    # Solo indica si hay un modelo de lenguaje configurado; nunca expone claves ni nombres de proveedor.
    configured = [
        name
        for name, variable in (("claude", "ANTHROPIC_API_KEY"), ("llm", "LLM_API_KEY"))
        if os.environ.get(variable, "").strip()
    ]
    return {"status": "ok", "coach": "+".join(configured) if configured else "reglas"}


JOINT_LABELS = {
    "shoulder": "hombros",
    "elbow": "codos",
    "wrist": "muñecas",
    "hip": "caderas",
    "knee": "rodillas",
    "ankle": "tobillos",
}


def _not_analyzed(
    confidence: float,
    joints: tuple[str, ...],
    overlapping: bool = False,
) -> tuple[PoseResponse, None]:
    """Frame sin análisis. `overlapping` indica que los puntos se solapan en la imagen."""
    visible_names = ", ".join(JOINT_LABELS[name] for name in joints)
    if overlapping:
        feedback = [
            f"Los puntos de {visible_names} se solapan en la imagen: aléjate de la cámara "
            "o revisa que se vea el cuerpo completo."
        ]
    else:
        feedback = [f"Colócate dentro del encuadre y mantén visibles: {visible_names}."]
    response = PoseResponse(
        score=0,
        feedback=feedback,
        confidence=confidence,
        analyzed=False,
        message=" ".join(feedback),
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
    corrections: list[str] = []
    deduction = 0.0
    knee_angle: float | None = None

    if request.exercise == "squat":
        knee_angle = angle(*selected, width, height)
        if knee_angle is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["knee"] = knee_angle
        if knee_angle > 115:
            corrections.append("Baja un poco más y mantén los talones apoyados.")
            deduction += min(55, (knee_angle - 115) * 0.65)
        elif knee_angle < 65:
            corrections.append("Reduce la profundidad y mantén el movimiento controlado.")
            deduction += min(40, (65 - knee_angle) * 0.7)

    elif request.exercise == "lunge":
        knee_angle = angle(*selected, width, height)
        if knee_angle is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["knee"] = knee_angle
        if knee_angle > 100:
            corrections.append("Baja un poco más controlando la rodilla delantera.")
            deduction += min(45, (knee_angle - 100) * 0.6)
        elif knee_angle < 80:
            corrections.append("Reduce la profundidad y mantén la rodilla alineada con el pie.")
            deduction += min(40, (80 - knee_angle) * 0.7)

    elif request.exercise == "push_up":
        elbow_angle = angle(*selected, width, height)
        if elbow_angle is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["elbow"] = elbow_angle
        if elbow_angle > 110:
            corrections.append("Baja con control y flexiona los codos en un rango cómodo.")
            deduction += min(35, (elbow_angle - 110) * 0.35)
        elif elbow_angle < 65:
            corrections.append("Reduce la profundidad y mantén los hombros estables.")
            deduction += min(35, (65 - elbow_angle) * 0.7)
        alignment = select_joints(by_name, ("shoulder", "hip", "ankle"))
        if alignment is not None:
            deviation = line_deviation(*alignment, width, height)
            if deviation is not None:
                angles["body_line_deviation"] = deviation
                if deviation > 0.1:
                    corrections.append("Mantén hombros, cadera y tobillos en línea recta.")
                    deduction += min(40, (deviation - 0.1) * 120)

    elif request.exercise == "plank":
        deviation = line_deviation(*selected, width, height)
        if deviation is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["body_line_deviation"] = deviation
        if deviation > 0.08:
            corrections.append("Alinea hombros, cadera y tobillos en línea recta.")
            deduction += min(55, (deviation - 0.08) * 150)

    else:
        hip_angle = angle(*selected, width, height)
        if hip_angle is None:
            return _not_analyzed(overall_confidence, required_joints, overlapping=True)
        angles["hip_hinge"] = hip_angle
        if hip_angle > 145:
            corrections.append("Lleva la cadera atrás e inclina el torso con las rodillas ligeramente flexionadas.")
            deduction += min(45, (hip_angle - 145) * 0.8)
        elif hip_angle < 45:
            corrections.append("Reduce la inclinación y mantén la espalda estable.")
            deduction += min(45, (45 - hip_angle) * 0.8)

    if not corrections:
        corrections.append("Buena posición. Muévete con control y mantén una respiración constante.")

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
        message=" ".join(corrections),
    )
    return response, primary_angle


def analyze(request: PoseRequest) -> PoseResponse:
    """Análisis de un frame sin estado de sesión (phase "ready" y reps 0 por defecto)."""
    return evaluate(request)[0]


@protected.post("/analyze", response_model=PoseResponse)
def analyze_endpoint(request: PoseRequest) -> PoseResponse:
    response, primary_angle = evaluate(request)
    update = sessions.update(request.session_id, request.exercise, primary_angle)
    return response.model_copy(update={"phase": update.phase, "reps": update.reps})


@protected.post("/coach", response_model=CoachAnswer)
def coach(request: CoachRequest) -> CoachAnswer:
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
    return answer_coach(request.question, request.context, api_key, llm=llm)


app.include_router(protected)
