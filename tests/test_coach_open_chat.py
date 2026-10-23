import unittest

from app.coach import (
    APP_REPLY,
    DEFAULT_TIP,
    GOODBYE_REPLY,
    HELP_REPLY,
    HOME_ADVICE,
    MUSCLE_ADVICE,
    ROUTINE_BY_LEVEL,
    ROUTINE_SHORT,
    THANKS_REPLY,
    WARMUP_ADVICE,
    WEIGHT_LOSS_ADVICE,
    CoachContext,
    rule_based_answer,
)


class OpenChatTests(unittest.TestCase):
    def test_short_routine(self) -> None:
        self.assertEqual(rule_based_answer("Dame una rutina corta de 15 minutos"), ROUTINE_SHORT)

    def test_routine_follows_level(self) -> None:
        answer = rule_based_answer("Quiero una rutina", CoachContext(level="ADVANCED"))
        self.assertEqual(answer, ROUTINE_BY_LEVEL["ADVANCED"])

    def test_warmup(self) -> None:
        self.assertEqual(rule_based_answer("¿Cómo caliento antes de entrenar?"), WARMUP_ADVICE)

    def test_weight_loss_and_muscle(self) -> None:
        self.assertEqual(rule_based_answer("Quiero bajar de peso y quemar grasa"), WEIGHT_LOSS_ADVICE)
        self.assertEqual(rule_based_answer("¿Cómo gano masa muscular?"), MUSCLE_ADVICE)

    def test_home_workout(self) -> None:
        self.assertEqual(rule_based_answer("¿Puedo entrenar en casa sin equipo?"), HOME_ADVICE)

    def test_conversation_basics(self) -> None:
        self.assertEqual(rule_based_answer("Muchas gracias"), THANKS_REPLY)
        self.assertEqual(rule_based_answer("adiós, nos vemos"), GOODBYE_REPLY)
        self.assertEqual(rule_based_answer("¿Qué puedes hacer?"), HELP_REPLY)
        self.assertEqual(rule_based_answer("¿Cómo funciona la cámara?"), APP_REPLY)

    def test_unknown_topic_still_gets_a_useful_reply(self) -> None:
        answer = rule_based_answer("¿Quién ganó el mundial?")
        self.assertEqual(answer, DEFAULT_TIP)
        self.assertIn("rutina", answer)

    def test_pain_still_wins(self) -> None:
        self.assertIn("profesional", rule_based_answer("Quiero una rutina pero me duele la rodilla"))


if __name__ == "__main__":
    unittest.main()
