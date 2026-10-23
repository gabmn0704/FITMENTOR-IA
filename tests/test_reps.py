import unittest

from app.lru import LRUCache
from app.reps import RepCounter, RepSessionStore

UP = 170
BOTTOM = 100


def run_angles(counter: RepCounter, angles: list[float]) -> tuple[str, int]:
    update = None
    for angle in angles:
        update = counter.update(angle)
    assert update is not None
    return update.phase, update.reps


def reps_sequence(repetitions: int, top: float, bottom: float, hold: int = 6) -> list[float]:
    """Arranca arriba y produce `repetitions` ciclos completos (arriba -> fondo -> arriba)."""
    angles = [top] * hold
    for _ in range(repetitions):
        angles += [bottom] * hold + [top] * hold
    return angles


class RepCounterTests(unittest.TestCase):
    def test_synthetic_squat_sequence_counts_n_reps(self) -> None:
        for repetitions in (1, 3, 5):
            with self.subTest(repetitions=repetitions):
                counter = RepCounter("squat")
                _, reps = run_angles(counter, reps_sequence(repetitions, UP, BOTTOM))
                self.assertEqual(reps, repetitions)

    def test_shallow_repetitions_are_not_counted(self) -> None:
        counter = RepCounter("squat")
        # El fondo llega a 130 grados, por encima del umbral de 105.
        _, reps = run_angles(counter, reps_sequence(3, UP, 130))
        self.assertEqual(reps, 0)

    def test_phases_follow_the_state_machine(self) -> None:
        counter = RepCounter("squat")
        self.assertEqual(run_angles(counter, [UP] * 6), ("up", 0))
        self.assertEqual(run_angles(counter, [140] * 6), ("down", 0))
        self.assertEqual(run_angles(counter, [BOTTOM] * 6), ("bottom", 0))
        self.assertEqual(run_angles(counter, [UP] * 6), ("up", 1))

    def test_missing_angle_keeps_the_current_state(self) -> None:
        counter = RepCounter("squat")
        run_angles(counter, [UP] * 6 + [BOTTOM] * 6)
        self.assertEqual(counter.update(None).phase, "bottom")

    def test_plank_is_always_hold_without_reps(self) -> None:
        counter = RepCounter("plank")
        update = counter.update(90)
        self.assertEqual((update.phase, update.reps), ("hold", 0))

    def test_each_exercise_uses_its_own_thresholds(self) -> None:
        # Push-up: 140 grados no llega al fondo (< 90) ni a la posición alta (> 150).
        push_up = RepCounter("push_up")
        self.assertEqual(run_angles(push_up, reps_sequence(2, 160, 85)), ("up", 2))
        self.assertEqual(run_angles(RepCounter("push_up"), reps_sequence(2, 160, 95)), ("up", 0))

        # Deadlift: la cadera necesita pasar de 155 arriba y bajar de 115 para contar.
        deadlift = RepCounter("deadlift")
        self.assertEqual(run_angles(deadlift, reps_sequence(2, 165, 110)), ("up", 2))
        self.assertEqual(run_angles(RepCounter("deadlift"), reps_sequence(2, 165, 120)), ("up", 0))

        lunge = RepCounter("lunge")
        self.assertEqual(run_angles(lunge, reps_sequence(1, 160, 100)), ("up", 1))


class RepSessionStoreTests(unittest.TestCase):
    def test_sessions_are_independent(self) -> None:
        store = RepSessionStore()
        for angle in [UP] * 6 + [BOTTOM] * 6 + [UP] * 6:
            store.update("a", "squat", angle)
        self.assertEqual(store.update("b", "squat", UP).reps, 0)
        self.assertEqual(store.update("a", "squat", UP).reps, 1)

    def test_changing_exercise_resets_the_session(self) -> None:
        store = RepSessionStore()
        for angle in [UP] * 6 + [BOTTOM] * 6:
            store.update("s", "squat", angle)
        update = store.update("s", "lunge", UP)
        self.assertEqual((update.phase, update.reps), ("up", 0))

    def test_without_session_id_only_one_frame_is_evaluated(self) -> None:
        store = RepSessionStore()
        self.assertEqual(store.update(None, "squat", BOTTOM).phase, "bottom")
        update = store.update(None, "squat", UP)
        self.assertEqual((update.phase, update.reps), ("up", 0))

    def test_least_recently_used_session_is_discarded(self) -> None:
        store = RepSessionStore(capacity=1)
        for angle in [UP] * 6 + [BOTTOM] * 6:
            store.update("a", "squat", angle)
        store.update("b", "squat", UP)  # desaloja la sesión "a"
        # Si "a" siguiera en caché, este ciclo contaría una repetición.
        self.assertEqual(store.update("a", "squat", UP).reps, 0)
        self.assertEqual(store.update("a", "squat", UP).phase, "up")


class LRUCacheTests(unittest.TestCase):
    def test_evicts_the_oldest_entry_when_full(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(len(cache), 2)

    def test_get_refreshes_recency(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")
        cache.set("c", 3)
        self.assertEqual(cache.get("a"), 1)
        self.assertIsNone(cache.get("b"))

    def test_overwriting_a_key_does_not_grow_the_cache(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.set("a", 1)
        cache.set("a", 2)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.get("a"), 2)

    def test_delete_and_invalid_capacity(self) -> None:
        cache: LRUCache[str, int] = LRUCache(1)
        cache.set("a", 1)
        self.assertTrue(cache.delete("a"))
        self.assertFalse(cache.delete("a"))
        with self.assertRaises(ValueError):
            LRUCache(0)


if __name__ == "__main__":
    unittest.main()
