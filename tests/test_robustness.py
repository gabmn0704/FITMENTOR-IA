import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import coach as coach_module
from app import main as main_module
from app.coach import LlmConfig, answer_coach
from app.main import app

client = TestClient(app, raise_server_exceptions=False)

POINT = {"name": "hip", "x": 0.5, "y": 0.1, "confidence": 1}


class EndpointSafetyNetTests(unittest.TestCase):
    def setUp(self) -> None:
        coach_module.clear_state()

    def test_coach_never_returns_500(self) -> None:
        with patch.object(main_module, "answer_coach", side_effect=RuntimeError("boom")):
            response = client.post("/coach", json={"question": "¿Cómo hago una sentadilla?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "rules")
        self.assertTrue(response.json()["answer"])

    def test_coach_uses_default_tip_if_rules_also_fail(self) -> None:
        with patch.object(main_module, "answer_coach", side_effect=RuntimeError("boom")), patch.object(
            main_module, "rule_based_answer", side_effect=RuntimeError("boom")
        ):
            response = client.post("/coach", json={"question": "hola"})
        self.assertEqual(response.json()["answer"], coach_module.DEFAULT_TIP)

    def test_analyze_never_returns_500(self) -> None:
        with patch.object(main_module, "evaluate", side_effect=RuntimeError("boom")):
            response = client.post("/analyze", json={"exercise": "squat", "points": [POINT]})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["analyzed"])
        self.assertEqual(body["message"], body["feedback"][0])

    def test_malformed_json_is_a_clean_422(self) -> None:
        response = client.post("/analyze", content=b"{not json", headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_blank_question_is_rejected(self) -> None:
        self.assertEqual(client.post("/coach", json={"question": "   "}).status_code, 422)

    def test_health_supports_head_without_body(self) -> None:
        response = client.head("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"")


class PoseMessageTests(unittest.TestCase):
    def test_hidden_joints_are_listed_naturally(self) -> None:
        response = client.post("/analyze", json={"exercise": "squat", "points": []}).json()
        self.assertIn("caderas, rodillas y tobillos", response["message"])

    def test_messages_are_one_sentence(self) -> None:
        texts = [
            main_module.GOOD_FORM,
            main_module.UNAVAILABLE_MESSAGE,
            *main_module.ALTERNATIVES,
            *main_module.ALTERNATIVES.values(),
        ]
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(text.rstrip(".").count(". "), 0)


class MalformedProviderTests(unittest.TestCase):
    llm = LlmConfig(api_key="k")

    def setUp(self) -> None:
        coach_module.clear_state()

    def _ask(self, handler) -> coach_module.CoachAnswer:
        return answer_coach("¿Cómo mejoro mi plancha?", None, "key", httpx.MockTransport(handler), llm=self.llm)

    def test_non_json_bodies_fall_back_to_rules(self) -> None:
        result = self._ask(lambda request: httpx.Response(200, content=b"<html>oops</html>"))
        self.assertEqual(result.source, "rules")
        self.assertIn("plancha", result.answer.lower())

    def test_unexpected_json_shapes_fall_back_to_rules(self) -> None:
        for body in ([], {"content": "texto"}, {"choices": [None]}, {"choices": [{"message": {"content": 7}}]}):
            with self.subTest(body=body):
                coach_module.clear_state()
                result = self._ask(lambda request, body=body: httpx.Response(200, json=body))
                self.assertEqual(result.source, "rules")
                self.assertTrue(result.answer)

    def test_rate_limit_falls_back_to_rules(self) -> None:
        result = self._ask(lambda request: httpx.Response(429))
        self.assertEqual(result.source, "rules")

    def test_whitespace_answer_is_ignored(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "anthropic" in str(request.url):
                return httpx.Response(200, json={"content": [{"type": "text", "text": "   "}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "\n"}}]})

        self.assertEqual(self._ask(handler).source, "rules")


if __name__ == "__main__":
    unittest.main()
