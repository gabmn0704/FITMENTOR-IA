import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_POINT = {"name": "hip", "x": 0.5, "y": 0.5, "confidence": 1}


class AnalyzeValidationTests(unittest.TestCase):
    def test_unknown_exercise_is_rejected(self) -> None:
        response = client.post("/analyze", json={"exercise": "burpee", "points": [VALID_POINT]})
        self.assertEqual(response.status_code, 422)

    def test_out_of_range_coordinates_are_rejected(self) -> None:
        point = {**VALID_POINT, "x": 1.6}
        response = client.post("/analyze", json={"exercise": "squat", "points": [point]})
        self.assertEqual(response.status_code, 422)

    def test_slightly_off_frame_landmarks_are_accepted(self) -> None:
        point = {**VALID_POINT, "y": 1.05}
        response = client.post("/analyze", json={"exercise": "squat", "points": [VALID_POINT, point]})
        self.assertEqual(response.status_code, 200)

    def test_more_than_64_points_are_rejected(self) -> None:
        response = client.post("/analyze", json={"exercise": "squat", "points": [VALID_POINT] * 65})
        self.assertEqual(response.status_code, 422)

    def test_64_points_are_accepted(self) -> None:
        response = client.post("/analyze", json={"exercise": "squat", "points": [VALID_POINT] * 64})
        self.assertEqual(response.status_code, 200)

    def test_long_point_name_is_rejected(self) -> None:
        point = {**VALID_POINT, "name": "a" * 41}
        response = client.post("/analyze", json={"exercise": "squat", "points": [point]})
        self.assertEqual(response.status_code, 422)

    def test_huge_frame_size_is_rejected(self) -> None:
        body = {"exercise": "squat", "points": [VALID_POINT], "frame_width": 100_000}
        response = client.post("/analyze", json=body)
        self.assertEqual(response.status_code, 422)

    def test_confidence_must_be_between_zero_and_one(self) -> None:
        point = {**VALID_POINT, "confidence": 2}
        response = client.post("/analyze", json={"exercise": "squat", "points": [point]})
        self.assertEqual(response.status_code, 422)

    def test_points_are_required(self) -> None:
        response = client.post("/analyze", json={"exercise": "squat"})
        self.assertEqual(response.status_code, 422)

    def test_frame_size_must_be_positive(self) -> None:
        body = {"exercise": "squat", "points": [VALID_POINT], "frame_width": 0}
        response = client.post("/analyze", json=body)
        self.assertEqual(response.status_code, 422)

    def test_empty_session_id_is_rejected(self) -> None:
        body = {"exercise": "squat", "points": [VALID_POINT], "session_id": ""}
        response = client.post("/analyze", json=body)
        self.assertEqual(response.status_code, 422)


class CoachValidationTests(unittest.TestCase):
    def test_question_is_required(self) -> None:
        response = client.post("/coach", json={})
        self.assertEqual(response.status_code, 422)

    def test_blank_question_is_rejected(self) -> None:
        response = client.post("/coach", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_question_longer_than_500_characters_is_rejected(self) -> None:
        response = client.post("/coach", json={"question": "a" * 501})
        self.assertEqual(response.status_code, 422)

    def test_question_of_500_characters_is_accepted(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            response = client.post("/coach", json={"question": "a" * 500})
        self.assertEqual(response.status_code, 200)

    def test_context_score_out_of_range_is_rejected(self) -> None:
        response = client.post("/coach", json={"question": "hola", "context": {"score": 150}})
        self.assertEqual(response.status_code, 422)

    def test_context_reps_must_be_non_negative_integer(self) -> None:
        response = client.post("/coach", json={"question": "hola", "context": {"reps": -1}})
        self.assertEqual(response.status_code, 422)


class CorsTests(unittest.TestCase):
    def test_preflight_from_a_browser_origin_is_not_allowed(self) -> None:
        # Solo el backend Java llama al servicio: ningún origen de navegador recibe CORS.
        response = client.options(
            "/coach",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertNotIn("access-control-allow-origin", response.headers)


if __name__ == "__main__":
    unittest.main()
