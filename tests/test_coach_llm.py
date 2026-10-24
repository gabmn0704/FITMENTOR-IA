import json
import unittest

import httpx

from app.coach import CoachContext, LlmConfig, answer_coach, ask_llm


def transport_for(body: dict, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer secret-key"
        sent = json.loads(request.content)
        assert sent["messages"][0]["role"] == "system"
        assert "<pregunta>" in sent["messages"][1]["content"]
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


CONFIG = LlmConfig(api_key="secret-key")


class LlmProviderTests(unittest.TestCase):
    def test_returns_the_model_text(self) -> None:
        transport = transport_for({"choices": [{"message": {"content": "  Claro, empieza con 3 series.  "}}]})
        self.assertEqual(ask_llm("rutina", CoachContext(), CONFIG, transport), "Claro, empieza con 3 series.")

    def test_answer_coach_marks_model_answers(self) -> None:
        transport = transport_for({"choices": [{"message": {"content": "Respuesta del modelo."}}]})
        answer = answer_coach("¿qué opinas del fútbol?", llm=CONFIG, transport=transport)
        self.assertEqual((answer.answer, answer.source), ("Respuesta del modelo.", "claude"))

    def test_falls_back_to_rules_when_the_provider_fails(self) -> None:
        answer = answer_coach("hola", llm=CONFIG, transport=transport_for({"error": "x"}, status=500))
        self.assertEqual(answer.source, "rules")
        self.assertTrue(answer.answer)

    def test_empty_model_text_falls_back_to_rules(self) -> None:
        transport = transport_for({"choices": [{"message": {"content": "   "}}]})
        self.assertEqual(answer_coach("hola", llm=CONFIG, transport=transport).source, "rules")

    def test_pain_never_reaches_the_model(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise AssertionError("el dolor no debe consultar al modelo")

        answer = answer_coach("me duele la rodilla", llm=CONFIG, transport=httpx.MockTransport(boom))
        self.assertEqual(answer.source, "rules")


if __name__ == "__main__":
    unittest.main()
