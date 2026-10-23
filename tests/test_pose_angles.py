import unittest

from app.main import Point, PoseRequest, analyze


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


if __name__ == "__main__":
    unittest.main()
