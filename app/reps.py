from collections import deque
from dataclasses import dataclass
from math import isfinite
from statistics import median
from threading import Lock
from typing import Literal

from app.lru import LRUCache

Phase = Literal["ready", "up", "down", "bottom", "hold"]

# Mediana (no media) de las últimas muestras: un pico o un frame perdido aislado no
# mueve el ángulo suavizado, mientras que una media lo arrastraría.
SMOOTHING_WINDOW = 5
SESSION_CAPACITY = 500
MIN_ANGLE = 0.0
MAX_ANGLE = 180.0


@dataclass(frozen=True)
class Thresholds:
    """Umbrales sobre el ángulo principal suavizado.

    up: a partir de este valor el atleta está en posición alta (extendida).
    bottom: por debajo de este valor se considera que llegó al fondo.
    """

    up: float
    bottom: float


# Ángulo principal: rodilla (squat, lunge), codo (push_up) o bisagra de cadera (deadlift).
# Plank es isométrico: no tiene repeticiones.
THRESHOLDS: dict[str, Thresholds | None] = {
    "squat": Thresholds(up=150, bottom=105),
    "lunge": Thresholds(up=150, bottom=105),
    "push_up": Thresholds(up=150, bottom=90),
    "deadlift": Thresholds(up=155, bottom=115),
    "plank": None,
}


@dataclass(frozen=True)
class RepUpdate:
    phase: Phase
    reps: int


class RepCounter:
    """Máquina de estados ready -> up -> down -> bottom -> up (+1 repetición).

    Solo se cuenta una repetición al volver a la posición alta después de tocar el fondo,
    de modo que las repeticiones poco profundas no suman.

    Histéresis: entrar en "bottom" exige bajar del umbral `bottom` y salir hacia "up"
    exige superar `up`; la franja intermedia no cambia de estado. Un contador que empieza
    en el fondo no cuenta la primera subida: hace falta haber estado arriba antes
    ("armado") para que una repetición sume.
    """

    def __init__(self, exercise: str) -> None:
        self.exercise = exercise
        self._thresholds = THRESHOLDS[exercise]
        self._window: deque[float] = deque(maxlen=SMOOTHING_WINDOW)
        self._state: Phase = "ready"
        self._reps = 0
        self._armed = False

    def update(self, angle: float | None) -> RepUpdate:
        if self._thresholds is None:
            return RepUpdate(phase="hold", reps=0)
        # Una muestra no medible (None, fuera de 0-180 o no finita) no debe mover la máquina.
        if angle is None or not isfinite(angle) or not MIN_ANGLE <= angle <= MAX_ANGLE:
            return self._snapshot()

        self._window.append(angle)
        smoothed = median(self._window)

        if smoothed >= self._thresholds.up:
            if self._state == "bottom" and self._armed:
                self._reps += 1
            self._state = "up"
            self._armed = True
        elif smoothed <= self._thresholds.bottom:
            self._state = "bottom"
        elif self._state == "up":
            self._state = "down"
        return self._snapshot()

    def _snapshot(self) -> RepUpdate:
        return RepUpdate(phase=self._state, reps=self._reps)


class RepSessionStore:
    """Contadores indexados por session_id con descarte LRU."""

    def __init__(self, capacity: int = SESSION_CAPACITY) -> None:
        self._sessions: LRUCache[str, RepCounter] = LRUCache(capacity)
        self._lock = Lock()

    def update(self, session_id: str | None, exercise: str, angle: float | None) -> RepUpdate:
        """Sin session_id se usa un contador efímero: una sola muestra, sin repeticiones."""
        if session_id is None:
            return RepCounter(exercise).update(angle)

        with self._lock:
            counter = self._sessions.get(session_id)
            if counter is None or counter.exercise != exercise:
                counter = RepCounter(exercise)
                self._sessions.set(session_id, counter)
            return counter.update(angle)
