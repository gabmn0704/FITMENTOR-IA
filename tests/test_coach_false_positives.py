import unittest

from app.coach import (
    EXERCISE_TIPS,
    NUTRITION_ADVICE,
    REST_ADVICE_DEFAULT,
    CoachContext,
    rule_based_answer,
)


class NutritionFalsePositiveTests(unittest.TestCase):
    def test_aguantar_is_not_nutrition(self) -> None:
        answer = rule_based_answer("¿Cómo puedo aguantar más la plancha?", CoachContext(exercise="plank"))
        self.assertNotIn(NUTRITION_ADVICE, answer)
        self.assertIn(EXERCISE_TIPS["plank"], answer)

    def test_water_as_a_word_is_still_nutrition(self) -> None:
        self.assertEqual(rule_based_answer("¿Cuánta agua bebo al entrenar?"), NUTRITION_ADVICE)


class RestFalsePositiveTests(unittest.TestCase):
    def test_dias_alone_is_not_rest(self) -> None:
        answer = rule_based_answer("¿Cuántos días a la semana entreno?", CoachContext(exercise="squat"))
        self.assertNotIn(REST_ADVICE_DEFAULT, answer)

    def test_cuanto_tiempo_for_a_plank_is_not_rest(self) -> None:
        answer = rule_based_answer("¿Cuánto tiempo sostengo la plancha?", CoachContext(exercise="plank", level="BEGINNER"))
        self.assertNotIn("48 horas", answer)
        self.assertIn("30 segundos", answer)

    def test_rest_is_still_detected_by_its_roots(self) -> None:
        self.assertIn("48 horas", rule_based_answer("¿Cuánto descanso entre sesiones?"))


class ExerciseAliasBoundaryTests(unittest.TestCase):
    def test_flexiono_is_not_a_push_up(self) -> None:
        answer = rule_based_answer("Flexiono demasiado la rodilla, ¿es normal?")
        self.assertNotIn(EXERCISE_TIPS["push_up"], answer)

    def test_flexiona_is_not_a_push_up(self) -> None:
        answer = rule_based_answer("Cuando flexiona la cadera, ¿qué hago?")
        self.assertNotIn(EXERCISE_TIPS["push_up"], answer)

    def test_plural_flexiones_is_a_push_up(self) -> None:
        self.assertIn(EXERCISE_TIPS["push_up"], rule_based_answer("¿Cómo hago flexiones?"))


class RepsQuestionTests(unittest.TestCase):
    def test_how_much_weight_does_not_report_repetitions(self) -> None:
        answer = rule_based_answer("¿Cuánto peso levanto?", CoachContext(exercise="deadlift", reps=5))
        self.assertNotIn("Llevas 5 repeticiones", answer)

    def test_how_many_reps_does_report_them(self) -> None:
        answer = rule_based_answer("¿Cuántas repeticiones llevo?", CoachContext(exercise="deadlift", reps=5))
        self.assertIn("Llevas 5 repeticiones", answer)


if __name__ == "__main__":
    unittest.main()
