from collections import deque
from dataclasses import dataclass
from math import isfinite
from threading import Lock
from typing import Literal

from app.lru import LRUCache

Phase = Literal["ready", "up", "down", "bottom", "hold"]

SESSION_CAPACITY = 500
# Ángulos válidos de las articulaciones seguidas. Por debajo de 10 grados no es fisiológico:
# suele ser un landmark degenerado (0 grados), así que se descarta la muestra.
MIN_ANGLE = 10.0
MAX_ANGLE = 180.0
# Salto máximo entre dos muestras consecutivas (frames a ~340 ms). Un movimiento real de
# 2 s por repetición cambia unos 43 grados por muestra; un salto mayor es un pico de ruido
# salvo que la muestra siguiente lo confirme (ver `RepCounter.update`).
MAX_JUMP = 60.0


@dataclass(frozen=True)
class Thresholds:
    """Umbrales sobre el ángulo principal.

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
    fuera de la posición alta no cuenta la primera subida ("armado"), salvo que el ejercicio
    empiece en el fondo (`starts_at_bottom`).

    Sin suavizado de media ni mediana: las muestras llegan cada ~340 ms y un suavizado
    de varias muestras pierde repeticiones cortas. Los picos aislados se filtran por salto:
    una muestra que cambia más de `MAX_JUMP` respecto a la anterior se retiene y solo se
    acepta si la siguiente muestra la confirma.
    """

    def __init__(self, exercise: str) -> None:
        self.exercise = exercise
        self._thresholds = THRESHOLDS[exercise]
        self._state: Phase = "ready"
        self._reps = 0
        self._armed = False
        self._last: float | None = None
        self._pending: float | None = None

    def _accept(self, angle: float) -> bool:
        """Filtro de saltos. Devuelve True si la muestra entra en la máquina de estados."""
        if self._last is None or abs(angle - self._last) <= MAX_JUMP:
            self._pending = None
            self._last = angle
            return True
        if self._pending is not None and abs(angle - self._pending) <= MAX_JUMP:
            # Dos muestras seguidas coinciden: el cambio es real, no un pico.
            self._pending = None
            self._last = angle
            return True
        self._pending = angle
        return False

    def update(self, angle: float | None) -> RepUpdate:
        if self._thresholds is None:
            return RepUpdate(phase="hold", reps=0)
        # Una muestra no medible (None o fuera de rango) no debe mover la máquina.
        if angle is None or not isfinite(angle) or not MIN_ANGLE <= angle <= MAX_ANGLE:
            return self._snapshot()
        if not self._accept(angle):
            return self._snapshot()

        if angle >= self._thresholds.up:
            if self._state == "bottom" and self._armed:
                self._reps += 1
            self._state = "up"
            self._armed = True
        elif angle <= self._thresholds.bottom:
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
