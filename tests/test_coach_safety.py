import unittest

import httpx

from app.coach import PAIN_ADVICE, CoachContext, answer_coach, is_pain, rule_based_answer

SECRET_KEY = "sk-test-secret-789"

PAIN_PHRASES = (
    "me dolió la rodilla",
    "me molesta la espalda al bajar",
    "estoy mareada",
    "estoy adolorido de la cadera",
    "me lastimé el hombro",
    "siento opresión en el pecho",
    "siento presión en el pecho al hacer flexiones",
    "I have pain in my knee",
    "my lower back hurts",
    "I feel dizzy after the set",
)


class CountingTransport:
    """Transporte simulado que cuenta las llamadas: si hay dolor, nunca debe llamarse."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            return httpx.Response(200, json={"content": [{"type": "text", "text": "Sigue así."}]})

        return httpx.MockTransport(handler)


class PainDetectionTests(unittest.TestCase):
    def test_each_pain_phrase_is_detected(self) -> None:
        for phrase in PAIN_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertTrue(is_pain(phrase))

    def test_chest_word_alone_is_not_pain(self) -> None:
        # "pecho" aparece en consejos de técnica ("mantén el pecho erguido") y no debe alarmar.
        self.assertFalse(is_pain("¿Cómo mantengo el pecho erguido en la sentadilla?"))


class PainAnswerTests(unittest.TestCase):
    def test_me_dolio_la_rodilla_gets_the_safety_answer(self) -> None:
        self.assertEqual(rule_based_answer("me dolió la rodilla"), PAIN_ADVICE)

    def test_me_molesta_gets_the_safety_answer(self) -> None:
        self.assertEqual(rule_based_answer("me molesta la espalda"), PAIN_ADVICE)

    def test_estoy_mareada_gets_the_safety_answer(self) -> None:
        self.assertEqual(rule_based_answer("estoy mareada"), PAIN_ADVICE)

    def test_adolorido_gets_the_safety_answer(self) -> None:
        self.assertEqual(rule_based_answer("estoy adolorido"), PAIN_ADVICE)

    def test_me_lastime_gets_the_safety_answer(self) -> None:
        self.assertEqual(rule_based_answer("me lastimé el hombro"), PAIN_ADVICE)

    def test_opresion_en_el_pecho_gets_the_safety_answer(self) -> None:
        self.assertEqual(rule_based_answer("siento opresión en el pecho"), PAIN_ADVICE)

    def test_english_pain_words_are_covered(self) -> None:
        self.assertEqual(rule_based_answer("my lower back hurts"), PAIN_ADVICE)


class PainBeforeModelTests(unittest.TestCase):
    def test_pain_is_answered_without_calling_claude(self) -> None:
        for phrase in PAIN_PHRASES:
            with self.subTest(phrase=phrase):
                transport = CountingTransport()
                answer = answer_coach(phrase, CoachContext(exercise="squat"), SECRET_KEY, transport())
                self.assertEqual(answer.answer, PAIN_ADVICE)
                self.assertEqual(answer.source, "rules")
                self.assertEqual(transport.calls, 0)

    def test_non_pain_questions_still_reach_claude(self) -> None:
        transport = CountingTransport()
        answer = answer_coach("¿Cómo mejoro la profundidad?", None, SECRET_KEY, transport())
        self.assertEqual(answer.source, "claude")
        self.assertEqual(transport.calls, 1)


if __name__ == "__main__":
    unittest.main()
