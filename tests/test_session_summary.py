import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.coach import SessionSummaryRequest, has_slang, session_summary
from app.main import app

client = TestClient(app)


class SessionSummaryTests(unittest.TestCase):
    def test_high_score_squat(self) -> None:
        result = session_summary(SessionSummaryRequest(exercise="squat", reps=12, score=91.4, level="INTERMEDIATE"))
        self.assertEqual(result.summary, "Excelente trabajo en tu sesión de sentadilla: 12 repeticiones con 91% de precisión.")
        self.assertIn("16 repeticiones", result.next_goal)
        self.assertTrue(result.highlights)

    def test_low_score_is_encouraging(self) -> None:
        result = session_summary(SessionSummaryRequest(exercise="push_up", reps=5, score=40))
        self.assertIn("cada sesión suma", result.summary)
        self.assertIn("prioriza la técnica", result.highlights[0])

    def test_plank_uses_duration(self) -> None:
        result = session_summary(SessionSummaryRequest(exercise="plancha", duration_seconds=75, score=80))
        self.assertIn("1 min 15 s de plancha", result.summary)
        self.assertIn("segundos de plancha", result.next_goal)

    def test_zero_reps_explains_camera(self) -> None:
        result = session_summary(SessionSummaryRequest(exercise="lunge", reps=0))
        self.assertIn("sin repeticiones registradas", result.summary)
        self.assertIn("cuerpo completo", result.highlights[0])

    def test_single_rep_is_singular(self) -> None:
        self.assertIn("1 repetición", session_summary(SessionSummaryRequest(exercise="squat", reps=1)).summary)

    def test_empty_request_still_answers(self) -> None:
        result = session_summary(SessionSummaryRequest())
        self.assertEqual(result.summary, "Completaste tu sesión.")
        self.assertTrue(result.highlights)
        self.assertIsNone(result.next_goal)

    def test_angle_note_is_included(self) -> None:
        result = session_summary(SessionSummaryRequest(exercise="squat", reps=10, score=70, average_angle=160))
        self.assertTrue(any("baja más" in item for item in result.highlights))

    def test_texts_are_neutral(self) -> None:
        for score in (None, 30, 70, 95):
            result = session_summary(SessionSummaryRequest(exercise="deadlift", reps=8, score=score, average_angle=170))
            for text in [result.summary, *result.highlights, result.next_goal or ""]:
                self.assertFalse(has_slang(text), text)


class SessionSummaryEndpointTests(unittest.TestCase):
    def test_endpoint_returns_summary(self) -> None:
        response = client.post("/session/summary", json={"exercise": "squat", "reps": 10, "score": 88})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("summary", body)
        self.assertIn("highlights", body)

    def test_endpoint_rejects_invalid_score(self) -> None:
        self.assertEqual(client.post("/session/summary", json={"score": 120}).status_code, 422)

    def test_endpoint_requires_secret_when_configured(self) -> None:
        with patch.dict("os.environ", {"AI_SHARED_SECRET": "s3cret"}):
            self.assertEqual(client.post("/session/summary", json={}).status_code, 401)
            ok = client.post("/session/summary", json={}, headers={"X-Internal-Token": "s3cret"})
            self.assertEqual(ok.status_code, 200)


if __name__ == "__main__":
    unittest.main()
