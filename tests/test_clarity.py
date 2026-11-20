import json
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import coach as coach_module
from app.coach import (
    BREAKER_FAILURES,
    SYSTEM_PROMPT,
    CoachContext,
    CoachProfile,
    LlmConfig,
    answer_coach,
    coach_status,
    profile_summary,
    rule_based_answer,
)
from app.main import ALTERNATIVES, GOOD_FORM, Point, PoseRequest, analyze, app, vary_repeats

client = TestClient(app)


def pose(exercise: str, **joints: tuple[float, float]) -> PoseRequest:
    return PoseRequest(
        exercise=exercise,
        points=[Point(name=name, x=x, y=y, confidence=1) for name, (x, y) in joints.items()],
    )


class FeedbackPriorityTests(unittest.TestCase):
    def test_message_is_the_single_main_correction(self) -> None:
        result = analyze(
            pose(
                "push_up",
                left_shoulder=(0.1, 0.1),
                left_elbow=(0.3, 0.1),
                left_wrist=(0.35, 0.2),
                left_hip=(0.6, 0.4),
                left_ankle=(0.9, 0.1),
            )
        )
        self.assertGreaterEqual(len(result.feedback), 2)
        self.assertEqual(result.message, result.feedback[0])

    def test_good_form_is_encouraging(self) -> None:
        result = analyze(pose("plank", shoulder=(0.1, 0.5), hip=(0.5, 0.5), ankle=(0.9, 0.5)))
        self.assertEqual(result.feedback, [GOOD_FORM])
        self.assertEqual(result.score, 100)

    def test_messages_are_short(self) -> None:
        for text in list(ALTERNATIVES) + list(ALTERNATIVES.values()):
            self.assertLessEqual(len(text), 90, text)


class ThresholdTests(unittest.TestCase):
    def squat(self, ankle: tuple[float, float]):
        return analyze(pose("squat", hip=(0.5, 0.1), knee=(0.5, 0.5), ankle=ankle))

    def test_squat_inside_range_has_no_depth_correction(self) -> None:
        result = self.squat((0.9, 0.5))
        self.assertAlmostEqual(result.knee_angle, 90, delta=0.5)
        self.assertEqual(result.feedback, [GOOD_FORM])

    def test_squat_just_above_limit_asks_for_more_depth(self) -> None:
        result = self.squat((0.7, 0.5 + 0.4 * 0.5736))
        self.assertGreater(result.knee_angle, 115)
        self.assertIn("Baja un poco más", result.message)

    def test_extended_squat_scores_lower_than_borderline(self) -> None:
        extended = self.squat((0.5, 0.9))
        borderline = self.squat((0.7, 0.5 + 0.4 * 0.5736))
        self.assertLess(extended.score, borderline.score)

    def test_lunge_inside_range_is_good(self) -> None:
        result = analyze(pose("lunge", hip=(0.5, 0.1), knee=(0.5, 0.5), ankle=(0.9, 0.5)))
        self.assertEqual(result.feedback, [GOOD_FORM])

    def test_deadlift_too_upright_mentions_hip(self) -> None:
        result = analyze(pose("deadlift", shoulder=(0.5, 0.1), hip=(0.5, 0.5), knee=(0.5, 0.9)))
        self.assertIn("cadera", result.message.lower())

    def test_plank_sagging_hip_asks_to_align(self) -> None:
        result = analyze(pose("plank", shoulder=(0.1, 0.5), hip=(0.5, 0.65), ankle=(0.9, 0.5)))
        self.assertIn("Alinea", result.message)

    def test_unseen_body_gives_a_friendly_instruction(self) -> None:
        result = analyze(PoseRequest(exercise="squat", points=[Point(name="hip", x=0.5, y=0.1, confidence=0.1)]))
        self.assertFalse(result.analyzed)
        self.assertIn("No alcanzo a verte", result.message)

    def test_every_exercise_gives_a_message(self) -> None:
        cases = {
            "squat": dict(hip=(0.5, 0.1), knee=(0.5, 0.5), ankle=(0.5, 0.9)),
            "lunge": dict(hip=(0.5, 0.1), knee=(0.5, 0.5), ankle=(0.5, 0.9)),
            "push_up": dict(shoulder=(0.1, 0.5), elbow=(0.3, 0.5), wrist=(0.5, 0.5)),
            "plank": dict(shoulder=(0.1, 0.5), hip=(0.5, 0.7), ankle=(0.9, 0.5)),
            "deadlift": dict(shoulder=(0.5, 0.1), hip=(0.5, 0.5), knee=(0.5, 0.9)),
        }
        for exercise, joints in cases.items():
            with self.subTest(exercise=exercise):
                result = analyze(pose(exercise, **joints))
                self.assertTrue(result.analyzed)
                self.assertTrue(result.message)


class RepeatVariationTests(unittest.TestCase):
    def test_repeated_correction_alternates_wording(self) -> None:
        request = pose("squat", hip=(0.5, 0.1), knee=(0.5, 0.5), ankle=(0.5, 0.9))
        messages = []
        for _ in range(4):
            response = vary_repeats("sess-repeat", analyze(request))
            messages.append(response.message)
            self.assertEqual(response.feedback[0], response.message)
        self.assertNotEqual(messages[0], messages[1])
        self.assertEqual(messages[0], messages[2])
        self.assertEqual(messages[1], messages[3])

    def test_without_session_nothing_changes(self) -> None:
        response = analyze(pose("squat", hip=(0.5, 0.1), knee=(0.5, 0.5), ankle=(0.5, 0.9)))
        self.assertEqual(vary_repeats(None, response), response)

    def test_endpoint_varies_wording_for_same_session(self) -> None:
        body = {
            "exercise": "squat",
            "session_id": "sess-endpoint",
            "points": [
                {"name": "hip", "x": 0.5, "y": 0.1, "confidence": 1},
                {"name": "knee", "x": 0.5, "y": 0.5, "confidence": 1},
                {"name": "ankle", "x": 0.5, "y": 0.9, "confidence": 1},
            ],
        }
        first = client.post("/analyze", json=body).json()["message"]
        second = client.post("/analyze", json=body).json()["message"]
        self.assertNotEqual(first, second)


class HealthTests(unittest.TestCase):
    def test_health_reports_rules_mode_without_keys(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "LLM_API_KEY": ""}):
            data = client.get("/health").json()
        self.assertEqual(data["coach"]["mode"], "rules")
        self.assertFalse(data["coach"]["claude"]["configured"])

    def test_health_reports_llm_mode_and_pause(self) -> None:
        for _ in range(BREAKER_FAILURES):
            coach_module.breaker.failure("llm")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "LLM_API_KEY": "k"}):
            data = client.get("/health").json()
        self.assertEqual(data["coach"]["mode"], "llm")
        self.assertTrue(data["coach"]["llm"]["paused"])

    def test_health_never_leaks_keys(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-secret", "LLM_API_KEY": "gsk-secret"}):
            text = client.get("/health").text
        self.assertNotIn("secret", text)


class ResilienceTests(unittest.TestCase):
    llm = LlmConfig(api_key="k")

    def test_cache_avoids_second_call(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json={"choices": [{"message": {"content": "Hola, claro."}}]})

        transport = httpx.MockTransport(handler)
        first = answer_coach("¿Cómo mejoro mi sentadilla?", None, llm=self.llm, transport=transport)
        second = answer_coach("¿Cómo mejoro mi sentadilla?", None, llm=self.llm, transport=transport)
        self.assertEqual(first.answer, second.answer)
        self.assertEqual(len(calls), 1)

    def test_failing_provider_opens_breaker_and_is_skipped(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(500)

        transport = httpx.MockTransport(handler)
        for index in range(BREAKER_FAILURES + 3):
            result = answer_coach(f"pregunta de técnica {index}", None, llm=self.llm, transport=transport)
            self.assertEqual(result.source, "rules")
            self.assertTrue(result.answer)
        self.assertEqual(len(calls), BREAKER_FAILURES)

    def test_claude_failure_falls_back_to_llm_quickly(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "anthropic" in str(request.url):
                return httpx.Response(529)
            return httpx.Response(200, json={"choices": [{"message": {"content": "Desde Groq."}}]})

        started = time.monotonic()
        result = answer_coach("¿Cómo respiro?", None, "key", httpx.MockTransport(handler), llm=self.llm)
        self.assertEqual(result.answer, "Desde Groq.")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_timeout_budget_is_short(self) -> None:
        self.assertLessEqual(coach_module.TIMEOUT_SECONDS, 5)
        self.assertLessEqual(coach_module.TOTAL_BUDGET_SECONDS, 6)

    def test_pain_never_calls_a_model(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no debe llamar al modelo")

        result = answer_coach("me duele la rodilla", None, "key", httpx.MockTransport(handler), llm=self.llm)
        self.assertEqual(result.source, "rules")

    def test_status_without_providers(self) -> None:
        self.assertEqual(coach_status(None, None)["mode"], "rules")


class CoachPersonalizationTests(unittest.TestCase):
    def test_prompt_is_structured_and_colombian(self) -> None:
        self.assertIn("colombiano", SYSTEM_PROMPT)
        self.assertIn("arepa", SYSTEM_PROMPT)
        self.assertIn("lesiones", SYSTEM_PROMPT)
        self.assertIn("de dos a cinco frases", SYSTEM_PROMPT)

    def test_profile_summary_includes_injuries_once(self) -> None:
        summary = profile_summary(CoachProfile(goal="tone", injuries=["knee", "knee", "back"]))
        self.assertIn("rodilla, espalda", summary)
        self.assertEqual(summary.count("rodilla"), 1)

    def test_invalid_injury_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            CoachProfile(injuries=["<script>"])

    def test_injuries_reach_the_model_prompt(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["content"] = json.loads(request.content)["messages"][1]["content"]
            return httpx.Response(200, json={"choices": [{"message": {"content": "Ok."}}]})

        context = CoachContext(profile=CoachProfile(injuries=["shoulder"]), level="BEGINNER")
        answer_coach("¿Qué rutina hago?", context, llm=LlmConfig(api_key="k"), transport=httpx.MockTransport(handler))
        self.assertIn("hombro", seen["content"])

    def test_colombian_nutrition_rules_answer(self) -> None:
        answer = rule_based_answer("¿Qué debo comer para ganar músculo?")
        self.assertIn("arepa", answer)
        self.assertIn("frijoles", answer)
        self.assertIn("no es consejo médico", answer)


if __name__ == "__main__":
    unittest.main()
