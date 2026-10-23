import math
import unittest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def knee_points(knee_angle: float) -> list[dict]:
    """Cadera-rodilla-tobillo con el ángulo de rodilla indicado (encuadre 16x9)."""
    radians = math.radians(knee_angle)
    length = 3.6  # distancia rodilla-tobillo en píxeles del encuadre
    ankle_x = 0.5 + math.sin(radians) * length / 16
    ankle_y = 0.5 - math.cos(radians) * length / 9
    return [
        {"name": "hip", "x": 0.5, "y": 0.1, "confidence": 1},
        {"name": "knee", "x": 0.5, "y": 0.5, "confidence": 1},
        {"name": "ankle", "x": round(ankle_x, 4), "y": round(ankle_y, 4), "confidence": 1},
    ]


def squat_frame(knee_angle: float, session_id: str | None = None, exercise: str = "squat") -> dict:
    body = {
        "exercise": exercise,
        "points": knee_points(knee_angle),
        "frame_width": 16,
        "frame_height": 9,
    }
    if session_id is not None:
        body["session_id"] = session_id
    return body


class HealthTests(unittest.TestCase):
    def test_health_returns_ok(self) -> None:
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class AnalyzeSessionTests(unittest.TestCase):
    def post(self, body: dict) -> dict:
        response = client.post("/analyze", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_response_contains_the_contract_fields(self) -> None:
        data = self.post(squat_frame(170))
        for field in ("score", "feedback", "knee_angle", "angles", "confidence", "analyzed", "phase", "reps", "message"):
            self.assertIn(field, data)
        self.assertEqual(data["message"], " ".join(data["feedback"]))
        self.assertEqual(data["knee_angle"], data["angles"]["knee"])

    def test_session_counts_repetitions_across_requests(self) -> None:
        angles = [170] * 6 + ([100] * 6 + [170] * 6) * 2
        data = None
        for angle in angles:
            data = self.post(squat_frame(angle, session_id="api-squat-1"))
        self.assertEqual(data["reps"], 2)
        self.assertEqual(data["phase"], "up")

    def test_without_session_id_reps_are_zero_and_phase_is_per_frame(self) -> None:
        self.assertEqual(self.post(squat_frame(170))["phase"], "up")
        data = self.post(squat_frame(100))
        self.assertEqual((data["phase"], data["reps"]), ("bottom", 0))
        data = self.post(squat_frame(170))
        self.assertEqual((data["phase"], data["reps"]), ("up", 0))

    def test_plank_is_hold_with_no_reps(self) -> None:
        body = {
            "exercise": "plank",
            "session_id": "api-plank-1",
            "points": [
                {"name": "shoulder", "x": 0.1, "y": 0.1, "confidence": 1},
                {"name": "hip", "x": 0.5, "y": 0.1, "confidence": 1},
                {"name": "ankle", "x": 0.9, "y": 0.1, "confidence": 1},
            ],
        }
        data = self.post(body)
        self.assertEqual((data["phase"], data["reps"]), ("hold", 0))

    def test_unanalyzed_frame_keeps_session_state(self) -> None:
        for angle in [170] * 6 + [100] * 6:
            self.post(squat_frame(angle, session_id="api-squat-2"))
        empty = {"exercise": "squat", "session_id": "api-squat-2", "points": []}
        data = self.post(empty)
        self.assertFalse(data["analyzed"])
        self.assertEqual((data["phase"], data["reps"]), ("bottom", 0))


if __name__ == "__main__":
    unittest.main()
