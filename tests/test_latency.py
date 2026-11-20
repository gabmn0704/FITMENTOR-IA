import unittest
from unittest.mock import MagicMock, patch

import httpx

from app import coach as coach_module
from app.coach import LlmConfig, answer_coach


def ok_handler(calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Mantén el pecho erguido."}}]})

    return handler


class LatencyTests(unittest.TestCase):
    def setUp(self) -> None:
        coach_module.clear_state()

    def test_cache_ignores_accents_case_and_punctuation(self) -> None:
        calls: list = []
        transport = httpx.MockTransport(ok_handler(calls))
        llm = LlmConfig(api_key="k")
        first = answer_coach("¿Cómo respiro en la sentadilla?", None, llm=llm, transport=transport)
        second = answer_coach("como respiro en la SENTADILLA", None, llm=llm, transport=transport)
        self.assertEqual(first.answer, second.answer)
        self.assertEqual(len(calls), 1)

    def test_cache_separates_different_contexts(self) -> None:
        calls: list = []
        transport = httpx.MockTransport(ok_handler(calls))
        llm = LlmConfig(api_key="k")
        answer_coach("¿Qué mejoro?", coach_module.CoachContext(exercise="squat"), llm=llm, transport=transport)
        answer_coach("¿Qué mejoro?", coach_module.CoachContext(exercise="plank"), llm=llm, transport=transport)
        self.assertEqual(len(calls), 2)

    def test_real_calls_reuse_one_pooled_client(self) -> None:
        fake = MagicMock()
        fake.post.return_value = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Respuesta."}}]},
            request=httpx.Request("POST", "https://example.test"),
        )
        with patch.object(coach_module, "_http_client", return_value=fake):
            answer_coach("pregunta uno de técnica", None, llm=LlmConfig(api_key="k"))
            answer_coach("pregunta dos de técnica", None, llm=LlmConfig(api_key="k"))
        self.assertEqual(fake.post.call_count, 2)
        timeout = fake.post.call_args.kwargs["timeout"]
        self.assertLessEqual(timeout.connect, coach_module.CONNECT_TIMEOUT_SECONDS)

    def test_shared_client_is_created_once_and_closed(self) -> None:
        coach_module.close_http_client()
        first = coach_module._http_client()
        self.assertIs(first, coach_module._http_client())
        coach_module.close_http_client()
        self.assertTrue(first.is_closed)

    def test_connect_timeout_goes_to_rules(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("no connect", request=request)

        result = answer_coach("¿Cómo bajo en la zancada?", None, "key", httpx.MockTransport(handler), llm=LlmConfig(api_key="k"))
        self.assertEqual(result.source, "rules")
        self.assertTrue(result.answer)


if __name__ == "__main__":
    unittest.main()
