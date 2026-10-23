import math
import unittest

from app.reps import RepCounter

# El frontend envía frames cada 340 ms (FITMENTOR-FRONT/src/lib/realtime.ts, POSE_INTERVAL_MS).
FRAME_INTERVAL_S = 0.34
TOP = 170.0


def repetition_frames(bottom: float, period_s: float, repetitions: int, offset_s: float) -> list[float]:
    """Muestras de un movimiento sinusoidal real: arriba -> fondo -> arriba, `repetitions` veces.

    La sesión empieza y acaba quieta en la posición alta. `offset_s` desplaza la rejilla de
    muestreo respecto al movimiento, para probar distintas alineaciones del frame.
    """
    mid = (TOP + bottom) / 2
    amplitude = (TOP - bottom) / 2
    motion_end = repetitions * period_s
    frames = [TOP] * 3
    t = offset_s
    while t <= motion_end:
        frames.append(mid + amplitude * math.cos(2 * math.pi * t / period_s))
        t += FRAME_INTERVAL_S
    frames += [TOP] * 3
    return frames


def count_reps(exercise: str, frames: list[float]) -> int:
    counter = RepCounter(exercise)
    update = None
    for angle in frames:
        update = counter.update(angle)
    assert update is not None
    return update.reps


class SampledRepetitionTests(unittest.TestCase):
    def test_squat_and_push_up_count_every_repetition_at_340_ms(self) -> None:
        cases = (
            ("squat", 90.0),  # rodilla de 170 a 90 grados; el umbral de fondo es 105
            ("push_up", 80.0),  # codo de 170 a 80 grados; el umbral de fondo es 90
        )
        for exercise, bottom in cases:
            for period_s in (2.0, 2.5, 3.0):
                for step in range(8):
                    offset = step * FRAME_INTERVAL_S / 8
                    with self.subTest(exercise=exercise, period_s=period_s, offset=round(offset, 3)):
                        frames = repetition_frames(bottom, period_s, 10, offset)
                        self.assertEqual(count_reps(exercise, frames), 10)

    def test_repetitions_are_counted_as_they_happen(self) -> None:
        counter = RepCounter("squat")
        reps_seen = []
        for angle in repetition_frames(90.0, 2.0, 3, 0.0):
            reps_seen.append(counter.update(angle).reps)
        # El contador sube una vez por repetición y nunca retrocede.
        self.assertEqual(reps_seen[-1], 3)
        self.assertEqual(sorted(reps_seen), reps_seen)
        self.assertEqual(len(set(reps_seen)), 4)


if __name__ == "__main__":
    unittest.main()
