import os
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from app.coach import GREETING_REPLIES, CoachContext, CoachProfile, answer_coach, describe_failure, rule_based_answer
from app.main import app

client = TestClient(app)


class HealthTests(unittest.TestCase):
    def test_reports_which_models_are_configured_without_leaking_keys(self) -> None:
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "gsk_secret", "ANTHROPIC_API_KEY": ""}, clear=False):
            body = client.get("/health").json()
        self.assertEqual(body, {"status": "ok", "coach": "llm"})
        self.assertNotIn("gsk_secret", str(body))

    def test_reports_rules_when_no_key_is_set(self) -> None:
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False):
            self.assertEqual(client.get("/health").json()["coach"], "reglas")


class FriendlyRulesTests(unittest.TestCase):
    def test_greetings_vary_but_are_stable_and_never_say_automatic(self) -> None:
        seen = {rule_based_answer(text) for text in ("hola", "buenas", "hey", "hola coach", "buenos dias")}
        self.assertGreater(len(seen), 1)
        self.assertTrue(seen.issubset(set(GREETING_REPLIES)))
        self.assertEqual(rule_based_answer("hola"), rule_based_answer("hola"))
        for reply in GREETING_REPLIES:
            self.assertNotIn("automático", reply)

    def test_greeting_mentions_the_plan_when_there_is_one(self) -> None:
        profile = CoachProfile(goal="lose_fat", week=2, weeks=16)
        answer = rule_based_answer("hola", CoachContext(profile=profile))
        self.assertTrue(answer.startswith("Vas en la semana 2 de 16 de tu plan para bajar de peso."))


class FailureLoggingTests(unittest.TestCase):
    def test_describes_http_errors_by_status_only(self) -> None:
        response = httpx.Response(401, request=httpx.Request("POST", "https://x"), text="invalid api key gsk_secret")
        error = httpx.HTTPStatusError("boom", request=response.request, response=response)
        self.assertEqual(describe_failure(error), "HTTP 401")
        self.assertEqual(describe_failure(ValueError("x")), "ValueError")

    def test_a_failing_provider_is_logged_and_falls_back_to_rules(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "rate"})

        from app.coach import LlmConfig

        with self.assertLogs("fitmentor.coach", level="WARNING") as logs:
            answer = answer_coach("hola", llm=LlmConfig(api_key="gsk_secret"), transport=httpx.MockTransport(boom))
        self.assertEqual(answer.source, "rules")
        self.assertIn("HTTP 429", logs.output[0])
        self.assertNotIn("gsk_secret", logs.output[0])


if __name__ == "__main__":
    unittest.main()
