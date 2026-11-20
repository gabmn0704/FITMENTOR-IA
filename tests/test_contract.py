"""Contrato estable que consumen el backend, la web y la app móvil. Si una prueba de aquí falla,
el cambio rompe a los clientes: cualquier cambio debe ser aditivo y actualizar el README."""
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import coach
from app.main import app

client = TestClient(app, raise_server_exceptions=False)
README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")

ANALYZE_FIELDS = {"score", "feedback", "knee_angle", "angles", "confidence", "analyzed", "phase", "reps", "message"}
PHASES = {"ready", "up", "down", "bottom", "hold"}
EXERCISES = ("squat", "push_up", "plank", "deadlift", "lunge")


def readme_json(heading: str, index: int) -> dict:
    """Bloque ```json número `index` dentro de la sección que empieza en `heading`."""
    start = README.index(heading)
    following = README.find("\n### ", start + 1)
    section = README[start : following if following != -1 else len(README)]
    return json.loads(re.findall(r"```json\n(.*?)```", section, re.S)[index])


class ReadmeExamplesTests(unittest.TestCase):
    def test_analyze_example_is_accepted_and_has_the_documented_shape(self) -> None:
        request = readme_json("### `POST /analyze`", 0)
        documented = readme_json("### `POST /analyze`", 1)
        response = client.post("/analyze", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(set(response.json()), set(documented))

    def test_coach_example_is_accepted_and_has_the_documented_shape(self) -> None:
        request = readme_json("### `POST /coach`", 0)
        documented = readme_json("### `POST /coach`", 1)
        response = client.post("/coach", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(set(response.json()), set(documented))

    def test_session_summary_example_is_accepted_and_has_the_documented_shape(self) -> None:
        request = readme_json("### `POST /session/summary`", 0)
        documented = readme_json("### `POST /session/summary`", 1)
        response = client.post("/session/summary", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(set(response.json()), set(documented))

    def test_stomp_frame_example_is_a_valid_analyze_request(self) -> None:
        frame = readme_json("## Contrato para web y app móvil", 0)
        self.assertEqual(client.post("/analyze", json=frame).status_code, 200)

    def test_health_example_has_the_documented_shape(self) -> None:
        documented = readme_json("### `GET /health`", 0)
        body = client.get("/health").json()
        self.assertEqual(set(body), set(documented))
        self.assertEqual(set(body["coach"]), set(documented["coach"]))


class AnalyzeContractTests(unittest.TestCase):
    def frame(self, exercise: str) -> dict:
        names = {
            "squat": ("hip", "knee", "ankle"),
            "lunge": ("hip", "knee", "ankle"),
            "push_up": ("shoulder", "elbow", "wrist"),
            "plank": ("shoulder", "hip", "ankle"),
            "deadlift": ("shoulder", "hip", "knee"),
        }[exercise]
        points = [{"name": name, "x": 0.3 + 0.2 * i, "y": 0.3 + 0.15 * (i % 2), "confidence": 0.9} for i, name in enumerate(names)]
        return {"exercise": exercise, "points": points, "frame_width": 1280, "frame_height": 720, "session_id": f"c-{exercise}"}

    def test_every_exercise_returns_the_same_field_set_and_types(self) -> None:
        for exercise in EXERCISES:
            with self.subTest(exercise=exercise):
                body = client.post("/analyze", json=self.frame(exercise)).json()
                self.assertEqual(set(body), ANALYZE_FIELDS)
                self.assertIsInstance(body["score"], (int, float))
                self.assertTrue(0 <= body["score"] <= 100)
                self.assertIsInstance(body["feedback"], list)
                self.assertTrue(body["feedback"])
                self.assertEqual(body["message"], body["feedback"][0])
                self.assertIn(body["phase"], PHASES)
                self.assertIsInstance(body["reps"], int)
                self.assertIsInstance(body["analyzed"], bool)

    def test_unanalyzable_frame_keeps_the_same_shape(self) -> None:
        body = client.post("/analyze", json={"exercise": "squat", "points": []}).json()
        self.assertEqual(set(body), ANALYZE_FIELDS)
        self.assertFalse(body["analyzed"])
        self.assertEqual(body["score"], 0)

    def test_invalid_bodies_are_422_with_fastapi_detail_never_500(self) -> None:
        for body in ({}, {"exercise": "yoga", "points": []}, {"exercise": "squat", "points": [{"name": "x", "x": 9, "y": 0, "confidence": 1}]}):
            with self.subTest(body=body):
                response = client.post("/analyze", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertIn("detail", response.json())

    def test_malformed_json_is_a_client_error(self) -> None:
        response = client.post("/analyze", content=b"{no es json", headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_an_internal_failure_is_a_not_analyzed_frame_not_a_500(self) -> None:
        with patch("app.main.evaluate", side_effect=RuntimeError("boom")):
            response = client.post("/analyze", json=self.frame("squat"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["analyzed"])


class CoachContractTests(unittest.TestCase):
    def setUp(self) -> None:
        coach.clear_state()

    def test_answer_and_source_only(self) -> None:
        body = client.post("/coach", json={"question": "¿Cómo mejoro mi sentadilla?"}).json()
        self.assertEqual(set(body), {"answer", "source"})
        self.assertIn(body["source"], ("claude", "rules"))
        self.assertTrue(body["answer"])

    def test_the_backend_payload_with_profile_and_history_is_accepted(self) -> None:
        """Forma exacta que envía el backend Java (snake_case, perfil e historial opcionales)."""
        payload = {
            "question": "¿Cuánta proteína debo comer?",
            "context": {
                "exercise": "squat",
                "score": 82.5,
                "reps": 12,
                "level": "BEGINNER",
                "profile": {"goal": "lose_fat", "weight_kg": 80, "height_cm": 175, "age": 30, "sex": "male", "weeks": 12, "week": 3, "days_per_week": 4, "diet": "omnivore", "kcal": 1900, "protein_g": 150},
                "history": [{"role": "user", "text": "hola"}, {"role": "coach", "text": "¿En qué te ayudo?"}],
            },
        }
        response = client.post("/coach", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("1900", response.json()["answer"])

    def test_unknown_context_fields_are_ignored_for_forward_compatibility(self) -> None:
        response = client.post("/coach", json={"question": "hola", "context": {"exercise": "squat", "campo_nuevo": 1}})
        self.assertEqual(response.status_code, 200)

    def test_question_limits(self) -> None:
        self.assertEqual(client.post("/coach", json={"question": "   "}).status_code, 422)
        self.assertEqual(client.post("/coach", json={"question": "a" * 501}).status_code, 422)
        self.assertEqual(client.post("/coach", json={"question": "a" * 500}).status_code, 200)

    def test_an_internal_failure_still_answers_by_rules(self) -> None:
        with patch("app.main.answer_coach", side_effect=RuntimeError("boom")):
            response = client.post("/coach", json={"question": "¿Cómo hago la plancha?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "rules")


class SummaryContractTests(unittest.TestCase):
    def test_empty_body_is_valid_and_fields_are_stable(self) -> None:
        body = client.post("/session/summary", json={}).json()
        self.assertEqual(set(body), {"summary", "highlights", "next_goal"})
        self.assertTrue(body["summary"])
        self.assertIsInstance(body["highlights"], list)


if __name__ == "__main__":
    unittest.main()
