from math import acos, degrees, hypot
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.reps import Phase, RepSessionStore

app = FastAPI(title="FitMentor Pose Service", version="1.0.0")
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


class Point(BaseModel):
    name: str = Field(min_length=1)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class PoseRequest(BaseModel):
    exercise: Exercise
    points: list[Point]
    frame_width: int = Field(default=16, gt=0)
    frame_height: int = Field(default=9, gt=0)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


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


def angle(first: Point, vertex: Point, last: Point, frame_width: int, frame_height: int) -> float:
    first_vector = ((first.x - vertex.x) * frame_width, (first.y - vertex.y) * frame_height)
    last_vector = ((last.x - vertex.x) * frame_width, (last.y - vertex.y) * frame_height)
    vector_size = hypot(*first_vector) * hypot(*last_vector)
    if vector_size == 0:
        return 0.0
    dot_product = first_vector[0] * last_vector[0] + first_vector[1] * last_vector[1]
    cosine = max(-1.0, min(1.0, dot_product / vector_size))
    return round(degrees(acos(cosine)), 2)


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
    if line_length == 0:
        return None
    deviation = abs((end_x - start_x) * (start_y - middle_y) - (start_x - middle_x) * (end_y - start_y))
    return round(deviation / line_length, 3)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
        joint_labels = {
            "shoulder": "hombros",
            "elbow": "codos",
            "wrist": "muñecas",
            "hip": "caderas",
            "knee": "rodillas",
            "ankle": "tobillos",
        }
        visible_names = ", ".join(joint_labels[name] for name in required_joints)
        feedback = [f"Colócate dentro del encuadre y mantén visibles: {visible_names}."]
        return (
            PoseResponse(
                score=0,
                feedback=feedback,
                confidence=overall_confidence,
                analyzed=False,
                message=" ".join(feedback),
            ),
            None,
        )

    width, height = request.frame_width, request.frame_height
    angles: dict[str, float] = {}
    corrections: list[str] = []
    deduction = 0.0
    knee_angle: float | None = None

    if request.exercise == "squat":
        knee_angle = angle(*selected, width, height)
        angles["knee"] = knee_angle
        if knee_angle > 115:
            corrections.append("Baja un poco más y mantén los talones apoyados.")
            deduction += min(55, (knee_angle - 115) * 0.65)
        elif knee_angle < 65:
            corrections.append("Reduce la profundidad y mantén el movimiento controlado.")
            deduction += min(40, (65 - knee_angle) * 0.7)

    elif request.exercise == "lunge":
        knee_angle = angle(*selected, width, height)
        angles["knee"] = knee_angle
        if knee_angle > 100:
            corrections.append("Baja un poco más controlando la rodilla delantera.")
            deduction += min(45, (knee_angle - 100) * 0.6)
        elif knee_angle < 80:
            corrections.append("Reduce la profundidad y mantén la rodilla alineada con el pie.")
            deduction += min(40, (80 - knee_angle) * 0.7)

    elif request.exercise == "push_up":
        elbow_angle = angle(*selected, width, height)
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
        if deviation is not None:
            angles["body_line_deviation"] = deviation
            if deviation > 0.08:
                corrections.append("Alinea hombros, cadera y tobillos en línea recta.")
                deduction += min(55, (deviation - 0.08) * 150)

    else:
        hip_angle = angle(*selected, width, height)
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


@app.post("/analyze", response_model=PoseResponse)
def analyze_endpoint(request: PoseRequest) -> PoseResponse:
    response, primary_angle = evaluate(request)
    update = sessions.update(request.session_id, request.exercise, primary_angle)
    return response.model_copy(update={"phase": update.phase, "reps": update.reps})
