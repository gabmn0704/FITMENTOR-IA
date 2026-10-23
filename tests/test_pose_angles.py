import unittest
from types import SimpleNamespace

from app.main import Point, PoseRequest, analyze, angle


def point(name: str, x: float, y: float) -> Point:
    return Point(name=name, x=x, y=y, confidence=1)


class DegenerateAngleTests(unittest.TestCase):
    def test_coincident_knee_and_ankle_do_not_read_as_a_full_bend(self) -> None:
        # Rodilla y tobillo en el mismo punto: antes daba 0 grados y una corrección falsa.
        result = analyze(
            PoseRequest(
                exercise="squat",
                points=[
                    point("hip", 0.5, 0.1),
                    point("knee", 0.5, 0.5),
                    point("ankle", 0.5, 0.5),
                ],
            )
        )

        self.assertFalse(result.analyzed)
        self.assertIsNone(result.knee_angle)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.angles, {})

    def test_degenerate_frame_has_no_primary_angle_for_the_rep_counter(self) -> None:
        from app.main import evaluate

        _, primary = evaluate(
            PoseRequest(
                exercise="push_up",
                points=[
                    point("left_shoulder", 0.1, 0.1),
                    point("left_elbow", 0.3, 0.1),
                    point("left_wrist", 0.3, 0.1),
                ],
            )
        )
        self.assertIsNone(primary)

    def test_plank_with_coincident_shoulder_and_ankle_is_not_analyzed(self) -> None:
        result = analyze(
            PoseRequest(
                exercise="plank",
                points=[
                    point("shoulder", 0.4, 0.4),
                    point("hip", 0.5, 0.5),
                    point("ankle", 0.4, 0.4),
                ],
            )
        )
        self.assertFalse(result.analyzed)
        self.assertEqual(result.score, 0)

    def test_angle_outside_zero_to_180_is_never_returned(self) -> None:
        origin = point("v", 0.5, 0.5)
        for first, last in (
            (point("a", 1, 0.5), point("b", 0, 0.5)),
            (point("a", 0.5, 0), point("b", 0.5, 1)),
            (point("a", 0.9, 0.2), point("b", 0.1, 0.8)),
        ):
            with self.subTest(first=first.x, last=last.x):
                value = angle(first, origin, last, 16, 9)
                self.assertIsNotNone(value)
                self.assertTrue(0.0 <= value <= 180.0)

    def test_non_finite_angle_inputs_are_not_measurable(self) -> None:
        origin = point("v", 0.5, 0.5)
        infinite = SimpleNamespace(x=float("inf"), y=0.5)  # Point valida el rango, así que se usa un objeto suelto.
        self.assertIsNone(angle(infinite, origin, point("b", 0, 0.5), 16, 9))


class OverlapAndScaleTests(unittest.TestCase):
    def test_overlapping_joints_get_an_overlap_message(self) -> None:
        result = analyze(
            PoseRequest(
                exercise="squat",
                frame_width=1280,
                frame_height=720,
                points=[
                    point("hip", 0.5, 0.1),
                    point("knee", 0.5, 0.5),
                    point("ankle", 0.5, 0.5),
                ],
            )
        )
        self.assertFalse(result.analyzed)
        self.assertIn("se solapan", result.message)

    def test_missing_joints_keep_the_visibility_message(self) -> None:
        result = analyze(PoseRequest(exercise="squat", points=[]))
        self.assertIn("visibles", result.message)
        self.assertNotIn("solapan", result.message)

    def test_near_coincident_points_in_a_large_frame_are_not_measured(self) -> None:
        # 0,0001 del encuadre son unos 0,13 px en 1280 px: los puntos se solapan.
        result = analyze(
            PoseRequest(
                exercise="squat",
                frame_width=1280,
                frame_height=720,
                points=[
                    point("hip", 0.5, 0.1),
                    point("knee", 0.5, 0.5),
                    point("ankle", 0.5001, 0.5),
                ],
            )
        )
        self.assertFalse(result.analyzed)

    def test_plank_deviation_does_not_depend_on_the_frame_size(self) -> None:
        # La desviación se mide como fracción de la línea: el mismo dibujo da el mismo valor
        # en un encuadre de 16x9 que en uno de 1280x720.
        from app.main import line_deviation

        shoulder, hip, ankle = point("s", 0.1, 0.1), point("h", 0.5, 0.2), point("a", 0.9, 0.1)
        small = line_deviation(shoulder, hip, ankle, 16, 9)
        large = line_deviation(shoulder, hip, ankle, 1280, 720)
        self.assertIsNotNone(small)
        self.assertAlmostEqual(small, large, delta=0.01)


if __name__ == "__main__":
    unittest.main()
