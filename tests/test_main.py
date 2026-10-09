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


if __name__ == "__main__":
    unittest.main()