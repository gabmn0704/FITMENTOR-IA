import json
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.coach import (
    CLAUDE_MODEL,
    DEFAULT_TIP,
    EXERCISE_TIPS,
    KEYWORD_TIPS,
    CoachAnswer,
    CoachContext,
    answer_coach,
    rule_based_answer,
)
from app.main import app

client = TestClient(app)
SECRET_KEY = "sk-test-secret-123"


def claude_transport(
    text: str = "Mantén el pecho erguido.",
    status: int = 200,
    body: dict | None = None,
) -> httpx.MockTransport:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(200, json=body or {"content": [{"type": "text", "text": text}]})

    transport = httpx.MockTransport(handler)
    transport.seen = seen  # type: ignore[attr-defined]
    return transport


def failing_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    return httpx.MockTransport(handler)


class RulesTests(unittest.TestCase):
    def test_pain_keyword_recommends_a_professional(self) -> None:
        answer = rule_based_answer("Me duele la rodilla al bajar", CoachContext(exercise="squat"))
        self.assertIn("profesional de la salud", answer)

    def test_exercise_tip_is_used_when_no_keyword_matches(self) -> None:
        answer = rule_based_answer("¿Cómo lo hago?", CoachContext(exercise="lunge"))
        self.assertIn(EXERCISE_TIPS["lunge"], answer)

    def test_default_tip_when_nothing_matches(self) -> None:
        self.assertEqual(rule_based_answer("¿Quién ganó el mundial?"), DEFAULT_TIP)

    def test_greeting_introduces_the_coach(self) -> None:
        self.assertIn("coach", rule_based_answer("hola").lower())

    def test_low_score_and_reps_are_mentioned(self) -> None:
        answer = rule_based_answer("¿Cuántas repeticiones llevo?", CoachContext(score=40, reps=7))
        self.assertIn("prioriza la técnica", answer)
        self.assertIn("Llevas 7 repeticiones", answer)

    def test_keywords_ignore_accents_and_case(self) -> None:
        self.assertIn("Exhala", rule_based_answer("RESPIRACIÓN"))

    def test_at_most_three_tips(self) -> None:
        question = "rodilla espalda codo respiro cansado profundidad dolor"
        answer = rule_based_answer(question, CoachContext(exercise="squat", score=10))
        all_tips = [tip for _, tip in KEYWORD_TIPS] + list(EXERCISE_TIPS.values())
        self.assertLessEqual(sum(tip in answer for tip in all_tips), 3)


class AnswerCoachTests(unittest.TestCase):
    def test_without_key_uses_rules_and_never_calls_the_network(self) -> None:
        answer = answer_coach("hola", CoachContext(), api_key=None, transport=failing_transport())
        self.assertEqual(answer.source, "rules")

    def test_with_key_uses_claude_response(self) -> None:
        transport = claude_transport("Respuesta de prueba.")
        answer = answer_coach("¿Cómo mejoro?", CoachContext(exercise="squat"), SECRET_KEY, transport)

        self.assertEqual(answer, CoachAnswer(answer="Respuesta de prueba.", source="claude"))
        request: httpx.Request = transport.seen["request"]  # type: ignore[attr-defined]
        self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages")
        self.assertEqual(request.headers["x-api-key"], SECRET_KEY)
        self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
        body = json.loads(request.content)
        self.assertEqual(body["model"], CLAUDE_MODEL)
        self.assertEqual(body["max_tokens"], 1024)
        self.assertEqual(body["output_config"], {"effort": "low"})
        self.assertIn("español", body["system"])
        self.assertIn("ejercicio: squat", body["messages"][0]["content"])

    def test_thinking_blocks_are_ignored(self) -> None:
        body = {
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "x"},
                {"type": "text", "text": "Baja controlando."},
            ],
        }
        answer = answer_coach("hola", CoachContext(), SECRET_KEY, claude_transport(body=body))
        self.assertEqual(answer, CoachAnswer(answer="Baja controlando.", source="claude"))

    def test_refusal_or_empty_text_falls_back_to_rules(self) -> None:
        refusal = {"stop_reason": "refusal", "content": [{"type": "text", "text": "No."}]}
        only_thinking = {"stop_reason": "max_tokens", "content": [{"type": "thinking", "thinking": ""}]}
        for body in (refusal, only_thinking):
            answer = answer_coach("hola", CoachContext(), SECRET_KEY, claude_transport(body=body))
            self.assertEqual(answer.source, "rules")

    def test_http_error_falls_back_to_rules_without_leaking_the_key(self) -> None:
        answer = answer_coach("hola", CoachContext(), SECRET_KEY, claude_transport(status=500))
        self.assertEqual(answer.source, "rules")
        self.assertNotIn(SECRET_KEY, answer.answer)

    def test_timeout_falls_back_to_rules(self) -> None:
        answer = answer_coach("hola", CoachContext(), SECRET_KEY, failing_transport())
        self.assertEqual(answer.source, "rules")

    def test_empty_claude_text_falls_back_to_rules(self) -> None:
        answer = answer_coach("hola", CoachContext(), SECRET_KEY, claude_transport(text="   "))
        self.assertEqual(answer.source, "rules")


class CoachEndpointTests(unittest.TestCase):
    def test_endpoint_without_key_answers_with_rules(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            response = client.post(
                "/coach",
                json={"question": "¿Cómo cuido la espalda?", "context": {"exercise": "deadlift"}},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["source"], "rules")
        self.assertIn("espalda", data["answer"].lower())

    def test_endpoint_passes_the_configured_key_to_the_coach(self) -> None:
        fake = CoachAnswer(answer="ok", source="claude")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": f"  {SECRET_KEY}  "}), patch(
            "app.main.answer_coach", return_value=fake
        ) as mocked:
            response = client.post("/coach", json={"question": "hola"})

        self.assertEqual(response.json(), {"answer": "ok", "source": "claude"})
        self.assertEqual(mocked.call_args.args[2], SECRET_KEY)

    def test_endpoint_context_is_optional(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            response = client.post("/coach", json={"question": "  hola  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "rules")


if __name__ == "__main__":
    unittest.main()
