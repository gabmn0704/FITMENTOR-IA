import unittest

import httpx

from app import coach as coach_module
from app import main as main_module
from app.coach import LlmConfig, SYSTEM_PROMPT, answer_coach, has_slang


def _all_rule_texts() -> list[str]:
    texts: list[str] = []
    for value in vars(coach_module).values():
        if isinstance(value, str) and len(value) > 20:
            texts.append(value)
        elif isinstance(value, (tuple, list)):
            texts.extend(item for item in value if isinstance(item, str) and len(item) > 20)
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, str) and len(item) > 20:
                    texts.append(item)
                elif isinstance(item, tuple):
                    texts.extend(part for part in item if isinstance(part, str))
    texts.extend(main_module.ALTERNATIVES)
    texts.extend(main_module.ALTERNATIVES.values())
    return texts


class ToneTests(unittest.TestCase):
    def setUp(self) -> None:
        coach_module.clear_state()

    def test_detects_slang(self) -> None:
        for text in ("Quiubo parce, ¿cómo vas?", "Eso está bacano", "Dale, mijo"):
            with self.subTest(text=text):
                self.assertTrue(has_slang(text))

    def test_body_words_are_not_slang(self) -> None:
        self.assertFalse(has_slang("Apoya bien la mano y el pie; mantén el codo estable."))
        self.assertFalse(has_slang("Parcela tu rutina en tres bloques."))

    def test_no_rule_text_uses_slang(self) -> None:
        for text in _all_rule_texts():
            with self.subTest(text=text[:40]):
                if text is SYSTEM_PROMPT:
                    continue
                self.assertFalse(has_slang(text), text)

    def test_no_colloquial_fillers_in_pose_feedback(self) -> None:
        banned = ("poquito", "dale", "así mismo", "no te veo", "como una tabla")
        texts = [main_module.GOOD_FORM, *main_module.ALTERNATIVES, *main_module.ALTERNATIVES.values()]
        for text in texts:
            for word in banned:
                self.assertNotIn(word, text.lower(), text)

    def test_slang_answer_from_model_is_replaced_by_rules(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "Quiubo parce, baja más."}}]})

        result = answer_coach(
            "¿Cómo mejoro mi sentadilla?", None, llm=LlmConfig(api_key="k"), transport=httpx.MockTransport(handler)
        )
        self.assertEqual(result.source, "rules")
        self.assertFalse(has_slang(result.answer))

    def test_slang_from_claude_falls_back_to_llm(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "anthropic" in str(request.url):
                return httpx.Response(
                    200, json={"content": [{"type": "text", "text": "Eso está bacano, parce."}], "stop_reason": "end_turn"}
                )
            return httpx.Response(200, json={"choices": [{"message": {"content": "Mantén el pecho erguido."}}]})

        result = answer_coach("¿Cómo respiro en la sentadilla?", None, "key", httpx.MockTransport(handler), llm=LlmConfig(api_key="k"))
        self.assertEqual(result.answer, "Mantén el pecho erguido.")


if __name__ == "__main__":
    unittest.main()
