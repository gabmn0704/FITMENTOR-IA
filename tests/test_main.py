import unittest

from app.main import Point, PoseRequest, analyze


class AnalyzeTests(unittest.TestCase):
    def test_extended_squat_gets_a_depth_correction(self) -> None:
        request = PoseRequest(
            exercise="squat",
            points=[
                Point(name="hip", x=0.5, y=0.1, confidence=1),
                Point(name="knee", x=0.5, y=0.5, confidence=1),
                Point(name="ankle", x=0.5, y=0.9, confidence=1),
            ],
        )

        result = analyze(request)

        self.assertEqual(result.knee_angle, 180)
        self.assertLess(result.score, 100)
        self.assertTrue(any("más" in item.lower() for item in result.feedback))

    def test_push_up_returns_more_than_one_form_correction(self) -> None:
        request = PoseRequest(
            exercise="push_up",
            points=[
                Point(name="left_shoulder", x=0.1, y=0.1, confidence=1),
                Point(name="left_elbow", x=0.3, y=0.1, confidence=1),
                Point(name="left_wrist", x=0.35, y=0.2, confidence=1),
                Point(name="left_hip", x=0.6, y=0.4, confidence=1),
                Point(name="left_ankle", x=0.9, y=0.1, confidence=1),
            ],
        )

        result = analyze(request)

        self.assertGreaterEqual(len(result.feedback), 2)
        self.assertTrue(any("codos" in item.lower() for item in result.feedback))
        self.assertTrue(any("línea" in item.lower() for item in result.feedback))

    def test_unreliable_landmarks_do_not_produce_a_form_score(self) -> None:
        request = PoseRequest(
            exercise="plank",
            points=[
                Point(name="shoulder", x=0.1, y=0.1, confidence=0.2),
                Point(name="hip", x=0.5, y=0.1, confidence=0.2),
                Point(name="ankle", x=0.9, y=0.1, confidence=0.2),
            ],
        )

        result = analyze(request)

        self.assertFalse(result.analyzed)
        self.assertEqual(result.score, 0)
        self.assertTrue(result.feedback)

    def test_aligned_plank_is_scored_without_correction(self) -> None:
        request = PoseRequest(
            exercise="plank",
            points=[
                Point(name="shoulder", x=0.1, y=0.1, confidence=1),
                Point(name="hip", x=0.5, y=0.1, confidence=1),
                Point(name="ankle", x=0.9, y=0.1, confidence=1),
            ],
        )

        result = analyze(request)

        self.assertTrue(result.analyzed)
        self.assertGreaterEqual(result.score, 90)
        self.assertEqual(len(result.feedback), 1)

    def test_deadlift_returns_a_hip_hinge_correction(self) -> None:
        request = PoseRequest(
            exercise="deadlift",
            points=[
                Point(name="left_shoulder", x=0.1, y=0.1, confidence=1),
                Point(name="left_hip", x=0.3, y=0.1, confidence=1),
                Point(name="left_knee", x=0.5, y=0.1, confidence=1),
            ],
        )

        result = analyze(request)

        self.assertTrue(result.analyzed)
        self.assertEqual(result.angles["hip_hinge"], 180)
        self.assertTrue(any("cadera" in item.lower() for item in result.feedback))

    def test_lunge_at_ideal_depth_has_no_correction(self) -> None:
        # Rodilla a 90 grados: dentro del rango ideal 80-100.
        result = analyze(lunge_request(hip=(0.5, 0.2), knee=(0.5, 0.5), ankle=(0.8, 0.5)))

        self.assertTrue(result.analyzed)
        self.assertAlmostEqual(result.knee_angle, 90, delta=0.5)
        self.assertEqual(result.score, 100)
        self.assertTrue(any("buena posición" in item.lower() for item in result.feedback))

    def test_lunge_too_shallow_asks_for_more_depth(self) -> None:
        result = analyze(lunge_request(hip=(0.5, 0.1), knee=(0.5, 0.5), ankle=(0.5, 0.9)))

        self.assertEqual(result.knee_angle, 180)
        self.assertLess(result.score, 100)
        self.assertTrue(any("baja un poco más" in item.lower() for item in result.feedback))

    def test_lunge_too_deep_asks_to_reduce_depth(self) -> None:
        # Rodilla a 60 grados: por debajo del rango ideal.
        result = analyze(lunge_request(hip=(0.5, 0.1), knee=(0.5, 0.5), ankle=(0.76, 0.233)))

        self.assertAlmostEqual(result.knee_angle, 60, delta=0.5)
        self.assertLess(result.score, 100)
        self.assertTrue(any("reduce la profundidad" in item.lower() for item in result.feedback))


def lunge_request(
    hip: tuple[float, float],
    knee: tuple[float, float],
    ankle: tuple[float, float],
) -> PoseRequest:
    return PoseRequest(
        exercise="lunge",
        points=[
            Point(name="left_hip", x=hip[0], y=hip[1], confidence=1),
            Point(name="left_knee", x=knee[0], y=knee[1], confidence=1),
            Point(name="left_ankle", x=ankle[0], y=ankle[1], confidence=1),
        ],
        frame_width=16,
        frame_height=9,
    )


if __name__ == "__main__":
    unittest.main()