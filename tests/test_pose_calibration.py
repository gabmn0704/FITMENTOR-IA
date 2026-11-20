"""Calibración de /analyze contra el conjunto sintético etiquetado (ver evals/pose_synth.py).

Las cotas son piso de regresión: si alguien mueve un umbral y empeoran la precisión, el recall o los
falsos positivos, estas pruebas fallan. Los datos son reproducibles (semilla fija).
"""
import unittest

from app import main as pose
from evals.pose_eval import classify, macro_f1, metrics, run
from evals.pose_synth import REFERENCE, all_samples, generate


class DatasetTests(unittest.TestCase):
    def test_generation_is_reproducible(self) -> None:
        first = [(s.points, s.labels) for s in generate("squat", 10, seed=3)]
        second = [(s.points, s.labels) for s in generate("squat", 10, seed=3)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, [(s.points, s.labels) for s in generate("squat", 10, seed=4)])

    def test_every_exercise_has_every_label_class(self) -> None:
        for exercise, dimensions in REFERENCE.items():
            seen: dict[str, set[str]] = {}
            for sample in generate(exercise, 5):
                for dimension, label in sample.labels.items():
                    seen.setdefault(dimension, set()).add(label)
            for dimension, classes in dimensions.items():
                with self.subTest(exercise=exercise, dimension=dimension):
                    self.assertEqual(seen[dimension], set(classes))


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run(all_samples(per_class=60, seed=7))
        cls.report = metrics(cls.result)

    def test_every_clean_frame_is_analyzed(self) -> None:
        self.assertEqual(self.result["not_analyzed"], 0)

    def test_overall_f1(self) -> None:
        self.assertGreaterEqual(macro_f1(self.report), 0.97)

    def test_false_positives_on_good_poses_are_rare(self) -> None:
        for key, entry in self.report.items():
            with self.subTest(dimension=key):
                self.assertLessEqual(entry["false_positive_rate"], 0.08)

    def test_bad_poses_are_rarely_missed(self) -> None:
        for key, entry in self.report.items():
            with self.subTest(dimension=key):
                self.assertLessEqual(entry["miss_rate"], 0.07)

    def test_precision_and_recall_per_class(self) -> None:
        for key, entry in self.report.items():
            for cls, values in entry["classes"].items():
                with self.subTest(dimension=key, cls=cls):
                    self.assertGreaterEqual(values["precision"], 0.9)
                    self.assertGreaterEqual(values["recall"], 0.9)

    def test_degrades_gracefully_with_more_landmark_noise(self) -> None:
        noisy = metrics(run(all_samples(per_class=60, seed=7, noise=0.015)))
        self.assertGreaterEqual(macro_f1(noisy), 0.94)


class ConfidenceTests(unittest.TestCase):
    def test_low_confidence_required_joint_is_not_analyzed(self) -> None:
        for sample in generate("squat", 20):
            points = [dict(p) for p in sample.points]
            for point in points:
                if point["name"].endswith("knee"):
                    point["confidence"] = 0.3
            response = pose.analyze(
                pose.PoseRequest(exercise="squat", points=points, frame_width=sample.frame_width, frame_height=sample.frame_height)
            )
            self.assertFalse(response.analyzed)
            self.assertEqual(response.score, 0)

    def test_classify_reads_all_corrections(self) -> None:
        both = ["Reduce la profundidad: no bajes tanto y mantén los hombros firmes.", "Cuerpo en línea recta: no dejes caer ni subas la cadera."]
        self.assertEqual(classify(both, "push_up"), {"elbow": "deep", "line": "bad"})
        self.assertEqual(classify([], "plank"), {"line": "ok"})


class SmallFrameTests(unittest.TestCase):
    def test_a_valid_pose_is_measured_with_only_an_aspect_ratio(self) -> None:
        """Con frame_width/height por defecto (16x9, una proporción) un tramo corto sigue siendo medible."""
        request = pose.PoseRequest(
            exercise="squat",
            points=[
                {"name": "left_hip", "x": 0.30, "y": 0.45, "confidence": 0.9},
                {"name": "left_knee", "x": 0.35, "y": 0.50, "confidence": 0.9},
                {"name": "left_ankle", "x": 0.31, "y": 0.58, "confidence": 0.9},
            ],
        )
        self.assertTrue(pose.analyze(request).analyzed)

    def test_coincident_points_are_still_not_measurable(self) -> None:
        request = pose.PoseRequest(
            exercise="squat",
            points=[
                {"name": "left_hip", "x": 0.5, "y": 0.5, "confidence": 0.9},
                {"name": "left_knee", "x": 0.5, "y": 0.5, "confidence": 0.9},
                {"name": "left_ankle", "x": 0.5, "y": 0.9, "confidence": 0.9},
            ],
        )
        self.assertFalse(pose.analyze(request).analyzed)


if __name__ == "__main__":
    unittest.main()
