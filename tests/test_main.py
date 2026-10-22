import unittest

from app.main import Point, PoseRequest, analyze, angle, line_deviation


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


class GeometryTests(unittest.TestCase):
    def test_right_angle_and_straight_line(self) -> None:
        origin = Point(name="v", x=0, y=0, confidence=1)
        self.assertEqual(angle(Point(name="a", x=1, y=0, confidence=1), origin, Point(name="b", x=0, y=1, confidence=1), 1, 1), 90)
        center = Point(name="v", x=0.5, y=0.5, confidence=1)
        self.assertEqual(angle(Point(name="a", x=1, y=0.5, confidence=1), center, Point(name="b", x=0, y=0.5, confidence=1), 1, 1), 180)

    def test_angle_respects_frame_aspect_ratio(self) -> None:
        origin = Point(name="v", x=0, y=0, confidence=1)
        first = Point(name="a", x=1, y=0, confidence=1)
        last = Point(name="b", x=1, y=1, confidence=1)
        self.assertAlmostEqual(angle(first, origin, last, 1, 1), 45, places=2)
        # Con un encuadre 2:1 el segmento vertical se ve más corto: el ángulo baja a ~26.57.
        self.assertAlmostEqual(angle(first, origin, last, 2, 1), 26.57, places=2)

    def test_zero_length_segment_returns_zero_angle(self) -> None:
        origin = Point(name="v", x=0.5, y=0.5, confidence=1)
        self.assertEqual(angle(origin, origin, Point(name="b", x=1, y=1, confidence=1), 16, 9), 0.0)

    def test_line_deviation_measures_distance_from_shoulder_ankle_line(self) -> None:
        shoulder = Point(name="s", x=0, y=0, confidence=1)
        hip = Point(name="h", x=0.5, y=0.2, confidence=1)
        ankle = Point(name="a", x=1, y=0, confidence=1)
        self.assertAlmostEqual(line_deviation(shoulder, hip, ankle, 1, 1), 0.2, places=3)

    def test_line_deviation_is_none_when_endpoints_coincide(self) -> None:
        same = Point(name="s", x=0.3, y=0.3, confidence=1)
        hip = Point(name="h", x=0.5, y=0.5, confidence=1)
        self.assertIsNone(line_deviation(same, hip, same, 16, 9))


class ExerciseCoverageTests(unittest.TestCase):
    def test_squat_too_deep_asks_to_reduce_depth(self) -> None:
        # Rodilla a 60 grados.
        result = analyze(
            PoseRequest(
                exercise="squat",
                points=[
                    Point(name="hip", x=0.5, y=0.1, confidence=1),
                    Point(name="knee", x=0.5, y=0.5, confidence=1),
                    Point(name="ankle", x=0.76, y=0.233, confidence=1),
                ],
            )
        )

        self.assertAlmostEqual(result.knee_angle, 60, delta=0.5)
        self.assertTrue(any("reduce la profundidad" in item.lower() for item in result.feedback))

    def test_deadlift_too_deep_asks_to_reduce_lean(self) -> None:
        # Bisagra de cadera a 30 grados.
        result = analyze(
            PoseRequest(
                exercise="deadlift",
                points=[
                    Point(name="shoulder", x=0.5, y=0.1, confidence=1),
                    Point(name="hip", x=0.5, y=0.5, confidence=1),
                    Point(name="knee", x=0.625, y=0.115, confidence=1),
                ],
            )
        )

        self.assertAlmostEqual(result.angles["hip_hinge"], 30, delta=0.5)
        self.assertTrue(any("reduce la inclinación" in item.lower() for item in result.feedback))

    def test_misaligned_plank_is_penalised(self) -> None:
        result = analyze(
            PoseRequest(
                exercise="plank",
                points=[
                    Point(name="shoulder", x=0.1, y=0.1, confidence=1),
                    Point(name="hip", x=0.5, y=0.3, confidence=1),
                    Point(name="ankle", x=0.9, y=0.1, confidence=1),
                ],
            )
        )

        self.assertTrue(result.analyzed)
        self.assertLess(result.score, 100)
        self.assertTrue(any("alinea" in item.lower() for item in result.feedback))

    def test_right_side_landmarks_are_used_when_left_is_missing(self) -> None:
        result = analyze(
            PoseRequest(
                exercise="squat",
                points=[
                    Point(name="right_hip", x=0.5, y=0.1, confidence=1),
                    Point(name="right_knee", x=0.5, y=0.5, confidence=1),
                    Point(name="right_ankle", x=0.5, y=0.9, confidence=1),
                ],
            )
        )

        self.assertTrue(result.analyzed)
        self.assertEqual(result.knee_angle, 180)

    def test_names_are_normalised_and_highest_confidence_wins(self) -> None:
        result = analyze(
            PoseRequest(
                exercise="squat",
                points=[
                    Point(name=" HIP ", x=0.9, y=0.9, confidence=0.3),
                    Point(name="hip", x=0.5, y=0.1, confidence=1),
                    Point(name="Knee", x=0.5, y=0.5, confidence=1),
                    Point(name="ankle", x=0.5, y=0.9, confidence=1),
                ],
            )
        )

        self.assertTrue(result.analyzed)
        self.assertEqual(result.knee_angle, 180)

    def test_empty_points_are_not_analyzed(self) -> None:
        result = analyze(PoseRequest(exercise="lunge", points=[]))

        self.assertFalse(result.analyzed)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.score, 0)


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