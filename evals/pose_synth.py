"""Conjunto sintético y reproducible de poses etiquetadas (buenas y malas) para calibrar /analyze.

Las etiquetas salen de rangos de referencia de entrenamiento (ver REFERENCE) y NO de los umbrales
del analizador: así la medición no es circular. Las poses son geométricas (no hay personas reales):
cada muestra coloca los landmarks en píxeles con un ángulo verdadero, lo normaliza al encuadre y
añade ruido gaussiano de landmark, orientación y espejo aleatorios y confianzas variables.
Entre una clase buena y una mala hay una franja de guarda sin etiquetar: lo que cae ahí es
discutible incluso para un entrenador, así que no se cuenta como acierto ni como error.

Limitación declarada: son frames estáticos en la pose de interés (p. ej. el fondo de la
sentadilla). No mide el movimiento ni la calidad de la detección de MediaPipe sobre personas reales.
"""
import math
import random
from dataclasses import dataclass, field

# Rangos de referencia por ejercicio y dimensión: (min, max) de la clase "ok" y de cada clase mala.
# Ángulos en grados de la articulación principal; "deviation" es la fracción de desvío de la línea.
# Todo lo que queda entre dos rangos es franja de guarda (sin etiqueta).
REFERENCE: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "squat": {"knee": {"ok": (72, 108), "shallow": (128, 175), "deep": (25, 52)}},
    "lunge": {"knee": {"ok": (82, 98), "shallow": (112, 170), "deep": (30, 66)}},
    "push_up": {
        "elbow": {"ok": (72, 100), "shallow": (125, 170), "deep": (25, 55)},
        "line": {"ok": (0.0, 0.05), "bad": (0.13, 0.30)},
    },
    "plank": {"line": {"ok": (0.0, 0.04), "bad": (0.13, 0.30)}},
    "deadlift": {"hinge": {"ok": (55, 130), "shallow": (158, 178), "deep": (20, 38)}},
}

FRAMES = ((16, 9), (9, 16), (4, 3), (1, 1))
LANDMARK_NOISE = 0.008  # desviación típica del ruido, como fracción del lado menor del encuadre


@dataclass
class Sample:
    exercise: str
    points: list[dict]
    frame_width: int
    frame_height: int
    # dimensión -> clase verdadera ("ok", "shallow", "deep", "bad")
    labels: dict[str, str] = field(default_factory=dict)


def _point(name: str, x: float, y: float, width: int, height: int, rng: random.Random, noise: float, confidence: float) -> dict:
    side = min(width, height)
    return {
        "name": name,
        "x": round((x + rng.gauss(0, noise * side)) / width, 5),
        "y": round((y + rng.gauss(0, noise * side)) / height, 5),
        "confidence": round(confidence, 3),
    }


def _three_points(names: tuple[str, str, str], angle: float, width: int, height: int, rng: random.Random, noise: float) -> list[dict]:
    """Tres landmarks con `angle` grados en el vértice (el del medio), orientación y espejo aleatorios."""
    side = min(width, height)
    vertex = (width * rng.uniform(0.35, 0.65), height * rng.uniform(0.35, 0.65))
    heading = rng.uniform(0, 2 * math.pi)
    direction = rng.choice((-1, 1))
    first = (vertex[0] + side * rng.uniform(0.18, 0.3) * math.cos(heading), vertex[1] + side * rng.uniform(0.18, 0.3) * math.sin(heading))
    turn = heading + direction * math.radians(angle)
    last = (vertex[0] + side * rng.uniform(0.18, 0.3) * math.cos(turn), vertex[1] + side * rng.uniform(0.18, 0.3) * math.sin(turn))
    side_name = rng.choice(("left_", "right_", ""))
    return [
        _point(side_name + names[0], *first, width, height, rng, noise, rng.uniform(0.6, 0.99)),
        _point(side_name + names[1], *vertex, width, height, rng, noise, rng.uniform(0.6, 0.99)),
        _point(side_name + names[2], *last, width, height, rng, noise, rng.uniform(0.6, 0.99)),
    ]


def _body_line(deviation: float, width: int, height: int, rng: random.Random, noise: float, side_name: str) -> tuple[list[dict], tuple[float, float]]:
    """Hombro, cadera y tobillo (y la posición real del hombro en píxeles); la cadera se desvía `deviation` (fracción de la longitud) de la línea."""
    length = width * rng.uniform(0.55, 0.75) if width >= height else height * rng.uniform(0.55, 0.75)
    heading = rng.uniform(-0.2, 0.2) + (0 if width >= height else math.pi / 2)
    shoulder = (width * 0.5 - math.cos(heading) * length / 2, height * 0.5 - math.sin(heading) * length / 2)
    ankle = (shoulder[0] + math.cos(heading) * length, shoulder[1] + math.sin(heading) * length)
    t = rng.uniform(0.4, 0.6)
    normal = (-math.sin(heading), math.cos(heading))
    sign = rng.choice((-1, 1))
    hip = (
        shoulder[0] + math.cos(heading) * length * t + normal[0] * deviation * length * sign,
        shoulder[1] + math.sin(heading) * length * t + normal[1] * deviation * length * sign,
    )
    line = [
        _point(side_name + "shoulder", *shoulder, width, height, rng, noise, rng.uniform(0.6, 0.99)),
        _point(side_name + "hip", *hip, width, height, rng, noise, rng.uniform(0.6, 0.99)),
        _point(side_name + "ankle", *ankle, width, height, rng, noise, rng.uniform(0.6, 0.99)),
    ]
    return line, shoulder


def _arm(shoulder: tuple[float, float], angle: float, width: int, height: int, rng: random.Random, noise: float, side_name: str) -> list[dict]:
    """Codo y muñeca con `angle` grados en el codo, colgando del hombro dado (en píxeles)."""
    side = min(width, height)
    heading = rng.uniform(0, 2 * math.pi)
    elbow = (shoulder[0] + side * 0.2 * math.cos(heading), shoulder[1] + side * 0.2 * math.sin(heading))
    back = math.atan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
    turn = back + rng.choice((-1, 1)) * math.radians(angle)
    wrist = (elbow[0] + side * 0.2 * math.cos(turn), elbow[1] + side * 0.2 * math.sin(turn))
    return [
        _point(side_name + "elbow", *elbow, width, height, rng, noise, rng.uniform(0.6, 0.99)),
        _point(side_name + "wrist", *wrist, width, height, rng, noise, rng.uniform(0.6, 0.99)),
    ]


def _extras(width: int, height: int, rng: random.Random) -> list[dict]:
    """Landmarks irrelevantes (cara, manos) que el analizador debe ignorar."""
    return [
        {"name": name, "x": round(rng.uniform(0.1, 0.9), 4), "y": round(rng.uniform(0.1, 0.9), 4), "confidence": round(rng.uniform(0.3, 0.99), 3)}
        for name in ("nose", "left_ear", "right_wrist_extra")
    ]


JOINTS = {
    "squat": ("hip", "knee", "ankle"),
    "lunge": ("hip", "knee", "ankle"),
    "push_up": ("shoulder", "elbow", "wrist"),
    "deadlift": ("shoulder", "hip", "knee"),
}


def generate(exercise: str, per_class: int = 200, seed: int = 7, noise: float = LANDMARK_NOISE) -> list[Sample]:
    """Muestras etiquetadas de un ejercicio: `per_class` por cada combinación de clases verdaderas."""
    rng = random.Random(f"{seed}:{exercise}")
    spec = REFERENCE[exercise]
    samples: list[Sample] = []
    angle_dim = next((dim for dim in spec if dim != "line"), None)
    angle_classes = list(spec[angle_dim]) if angle_dim else [None]
    line_classes = list(spec["line"]) if "line" in spec else [None]
    for angle_class in angle_classes:
        for line_class in line_classes:
            for _ in range(per_class):
                width, height = rng.choice(FRAMES)
                labels: dict[str, str] = {}
                points: list[dict] = []
                if angle_dim and angle_class:
                    low, high = spec[angle_dim][angle_class]
                    value = rng.uniform(low, high)
                    if exercise != "push_up":  # la flexión cuelga el brazo del hombro de la línea del cuerpo
                        points += _three_points(JOINTS[exercise], value, width, height, rng, noise)
                    labels[angle_dim] = angle_class
                if line_class:
                    low, high = spec["line"][line_class]
                    deviation = rng.uniform(low, high)
                    side_name = rng.choice(("left_", "right_"))
                    if exercise == "push_up":
                        # Codo y muñeca cuelgan del mismo hombro que la línea del cuerpo.
                        line, shoulder = _body_line(deviation, width, height, rng, noise, side_name)
                        points += line + _arm(shoulder, rng.uniform(*spec["elbow"][angle_class]), width, height, rng, noise, side_name)
                    else:
                        points += _body_line(deviation, width, height, rng, noise, side_name)[0]
                    labels["line"] = line_class
                samples.append(Sample(exercise, points + _extras(width, height, rng), width, height, labels))
    return samples


def all_samples(per_class: int = 200, seed: int = 7, noise: float = LANDMARK_NOISE) -> list[Sample]:
    return [sample for exercise in REFERENCE for sample in generate(exercise, per_class, seed, noise)]
