import unittest

from app.reps import RepCounter

UP = 170
BOTTOM = 100
# Ruido de +-4 grados: lo bastante grande para ser real y lo bastante pequeño para no cruzar umbrales.
JITTER = (0, 3, -4, 2, -3, 4, -1, 3)


def jittered(base: float, count: int) -> list[float]:
    return [base + JITTER[index % len(JITTER)] for index in range(count)]


def rep_frames(repetitions: int, hold: int = 6, top: float = UP, bottom: float = BOTTOM) -> list[float]:
    """Ciclos completos arriba -> fondo -> arriba, con ruido determinista de +-4 grados."""
    frames: list[float] = []
    for _ in range(repetitions):
        frames += jittered(bottom, hold) + jittered(top, hold)
    return frames


def feed(counter: RepCounter, angles: list[float | None]) -> tuple[str, int]:
    update = None
    for angle in angles:
        update = counter.update(angle)
    assert update is not None
    return update.phase, update.reps


class JitterTests(unittest.TestCase):
    def test_jitter_around_both_positions_counts_exact_repetitions(self) -> None:
        for repetitions in (1, 4):
            with self.subTest(repetitions=repetitions):
                counter = RepCounter("squat")
                frames = jittered(UP, 6) + rep_frames(repetitions)
                self.assertEqual(feed(counter, frames)[1], repetitions)

    def test_single_spike_at_the_top_is_filtered_out(self) -> None:
        # Sin filtro, el pico a 40 grados llevaría el contador a "bottom" y la siguiente
        # muestra de 170 contaría una repetición fantasma. Con el filtro de saltos no cuenta.
        frames = [UP] * 6 + [40] + [UP] * 6
        self.assertEqual(feed(RepCounter("squat"), frames), ("up", 0))

    def test_fast_drop_confirmed_by_the_next_sample_still_counts(self) -> None:
        # Un descenso brusco que se repite en la muestra siguiente es real y no se pierde.
        frames = [UP] * 6 + [60] * 2 + [UP] * 6
        self.assertEqual(feed(RepCounter("squat"), frames), ("up", 1))

    def test_single_spike_at_the_top_does_not_split_or_double_a_rep(self) -> None:
        # Un pico aislado a 40 grados en mitad de la subida no debe contar repeticiones extra.
        frames = [UP] * 6 + [BOTTOM] * 6 + [UP] * 3 + [40] + [UP] * 6
        self.assertEqual(feed(RepCounter("squat"), frames), ("up", 1))

    def test_single_spike_up_at_the_bottom_does_not_count_a_rep(self) -> None:
        frames = [UP] * 6 + [BOTTOM] * 3 + [UP] + [BOTTOM] * 3 + [UP] * 6
        self.assertEqual(feed(RepCounter("squat"), frames), ("up", 1))

    def test_two_frame_dropout_at_the_top_does_not_count_a_phantom_rep(self) -> None:
        # Dos frames con ángulo 0 (landmarks perdidos) antes encontraban un fondo falso.
        frames = [UP] * 6 + [0.0, 0.0] + [UP] * 6
        self.assertEqual(feed(RepCounter("squat"), frames), ("up", 0))


class HysteresisTests(unittest.TestCase):
    def test_middle_band_never_flips_the_phase_to_bottom(self) -> None:
        counter = RepCounter("squat")
        frames = [UP] * 6 + [125, 130, 128, 135, 127, 132, 129, 131]
        self.assertEqual(feed(counter, frames), ("down", 0))

    def test_hovering_between_thresholds_after_the_top_does_not_count(self) -> None:
        counter = RepCounter("squat")
        # Oscila entre 110 y 140: nunca toca el fondo (<105) ni la posición alta (>150).
        frames = [UP] * 6 + [110, 140] * 6
        self.assertEqual(feed(counter, frames)[1], 0)

    def test_incomplete_rep_that_never_reaches_the_top_is_not_counted(self) -> None:
        counter = RepCounter("squat")
        frames = [UP] * 6 + jittered(BOTTOM, 6) + [125] * 6
        self.assertEqual(feed(counter, frames), ("bottom", 0))


class ArmingTests(unittest.TestCase):
    def test_starting_at_the_bottom_does_not_count_the_first_rise(self) -> None:
        counter = RepCounter("squat")
        frames = [BOTTOM] * 6 + [UP] * 6
        self.assertEqual(feed(counter, frames), ("up", 0))

    def test_reps_count_once_the_athlete_has_been_at_the_top(self) -> None:
        counter = RepCounter("squat")
        frames = [BOTTOM] * 6 + [UP] * 6 + rep_frames(2)
        self.assertEqual(feed(counter, frames)[1], 2)


class InvalidSampleTests(unittest.TestCase):
    def test_invalid_samples_are_ignored(self) -> None:
        counter = RepCounter("squat")
        bad = [-5.0, 200.0, 0.0]
        frames: list[float | None] = []
        for repetitions in range(2):
            frames += [UP] * 6 + bad + [BOTTOM] * 6 + [None] + [UP] * 6
        self.assertEqual(feed(counter, frames)[1], 2)

    def test_zero_degree_dropouts_do_not_create_a_bottom(self) -> None:
        # 0 grados no es fisiológico: un landmark degenerado no debe contar como fondo.
        counter = RepCounter("squat")
        self.assertEqual(feed(counter, [UP] * 6 + [0.0] * 6 + [-1.0] * 2), ("up", 0))


if __name__ == "__main__":
    unittest.main()
