import unittest

from app.coach import (
    CoachContext,
    EXERCISE_TIPS,
    rule_based_answer,
)


class ExerciseAndLevelResolutionTests(unittest.TestCase):
    def test_backend_ids_and_spanish_names_resolve_to_the_same_exercise(self) -> None:
        for name in ("push_up", "PUSH_UP", "push-up", "Flexiones", "flexión"):
            with self.subTest(name=name):
                answer = rule_based_answer("¿Cómo lo hago?", CoachContext(exercise=name))
                self.assertIn(EXERCISE_TIPS["push_up"], answer)

    def test_exercise_is_taken_from_the_question_when_context_is_missing(self) -> None:
        answer = rule_based_answer("Consejos para la plancha")
        self.assertIn(EXERCISE_TIPS["plank"], answer)

    def test_question_exercise_wins_over_the_context(self) -> None:
        # El contexto es solo respaldo: la pregunta dice qué ejercicio se quiere.
        answer = rule_based_answer("¿Y en la zancada?", CoachContext(exercise="squat"))
        self.assertIn(EXERCISE_TIPS["lunge"], answer)
        self.assertNotIn(EXERCISE_TIPS["squat"], answer)

    def test_context_exercise_is_used_when_the_question_names_none(self) -> None:
        answer = rule_based_answer("¿Y ahora qué corrijo?", CoachContext(exercise="squat"))
        self.assertIn(EXERCISE_TIPS["squat"], answer)

    def test_greeting_with_context_exercise_greets_first(self) -> None:
        answer = rule_based_answer("hola", CoachContext(exercise="squat"))
        self.assertTrue(answer.startswith("¡Hola!"))
        self.assertIn(EXERCISE_TIPS["squat"], answer)

    def test_squat_too_low_mean_angle_says_too_deep(self) -> None:
        answer = rule_based_answer("¿Cómo lo hago?", CoachContext(exercise="squat", average_angle=50))
        self.assertIn("Bajas demasiado", answer)
        self.assertNotIn("recorrido", answer)

    def test_unknown_exercise_does_not_break_the_answer(self) -> None:
        answer = rule_based_answer("¿Cómo lo hago?", CoachContext(exercise="burpee"))
        self.assertTrue(answer.strip())

    def test_backend_level_enum_personalises_the_goal(self) -> None:
        beginner = rule_based_answer("¿Qué sigue?", CoachContext(exercise="squat", level="BEGINNER"))
        advanced = rule_based_answer("¿Qué sigue?", CoachContext(exercise="squat", level="ADVANCED"))
        self.assertIn("principiante", beginner)
        self.assertIn("12 repeticiones", beginner)
        self.assertIn("avanzado", advanced)
        self.assertIn("20 repeticiones", advanced)
        self.assertIn("bajada controlada", advanced)

    def test_spanish_level_names_are_accepted(self) -> None:
        answer = rule_based_answer("¿Qué sigue?", CoachContext(exercise="lunge", level="Intermedio"))
        self.assertIn("intermedio", answer)
        self.assertIn("14 repeticiones", answer)

    def test_unknown_level_gives_the_base_goal_without_a_label(self) -> None:
        answer = rule_based_answer("¿Qué sigue?", CoachContext(exercise="push_up", level="PRO"))
        self.assertIn("10 repeticiones", answer)
        self.assertNotIn("Para tu nivel", answer)

    def test_plank_goal_is_expressed_in_seconds(self) -> None:
        answer = rule_based_answer("¿Qué sigue?", CoachContext(exercise="plank", level="BEGINNER"))
        self.assertIn("30 segundos", answer)
        self.assertNotIn("repeticiones", answer)

    def test_goal_sentence_is_capitalised_without_a_level(self) -> None:
        answer = rule_based_answer("¿Qué sigue?", CoachContext(exercise="plank"))
        self.assertIn(". Tu siguiente objetivo es 60 segundos", answer)


class SessionMetricTests(unittest.TestCase):
    def test_accuracy_bands_give_different_feedback(self) -> None:
        low = rule_based_answer("¿Cómo voy?", CoachContext(score=40))
        mid = rule_based_answer("¿Cómo voy?", CoachContext(score=70))
        high = rule_based_answer("¿Cómo voy?", CoachContext(score=92))
        self.assertIn("Tu precisión es 40%", low)
        self.assertIn("prioriza la técnica", low)
        self.assertIn("hay margen", mid)
        self.assertIn("muy buena", high)

    def test_reps_are_reported_only_for_rep_questions(self) -> None:
        asked = rule_based_answer("¿Cuántas repeticiones llevo?", CoachContext(exercise="squat", reps=7))
        other = rule_based_answer("¿Cómo lo hago?", CoachContext(exercise="squat", reps=7))
        self.assertIn("Llevas 7 repeticiones", asked)
        self.assertNotIn("Llevas 7", other)

    def test_mean_angle_too_high_asks_for_more_depth(self) -> None:
        answer = rule_based_answer("¿Cómo lo hago?", CoachContext(exercise="squat", average_angle=165))
        self.assertIn("baja más", answer)

    def test_mean_angle_in_range_adds_no_correction(self) -> None:
        answer = rule_based_answer("¿Cómo lo hago?", CoachContext(exercise="squat", average_angle=100))
        self.assertNotIn("baja más", answer)
        self.assertNotIn("recorrido medio es muy corto", answer)

    def test_average_angle_is_ignored_for_plank(self) -> None:
        answer = rule_based_answer("¿Cómo lo hago?", CoachContext(exercise="plank", average_angle=170))
        self.assertNotIn("baja más", answer)

    def test_session_metrics_do_not_leak_into_unrelated_answers(self) -> None:
        answer = rule_based_answer("hola", CoachContext())
        self.assertNotIn("%", answer)


class KeywordMatchingTests(unittest.TestCase):
    def test_keyword_matches_word_starts_only(self) -> None:
        # "baj" dentro de "trabajo" no debe activar el consejo de profundidad.
        answer = rule_based_answer("¿Cuánto trabajo debo hacer?")
        self.assertNotIn("La profundidad", answer)

    def test_keyword_matches_inflections(self) -> None:
        answer = rule_based_answer("¿Cómo bajo sin molestar a mis rodillas?")
        self.assertIn("rodillas", answer)


class RobustnessTests(unittest.TestCase):
    def test_answers_are_never_empty(self) -> None:
        odd_inputs = ["", "   ", "???", "1234", "🙂🙂", "a" * 500, "\n\t"]
        for question in odd_inputs:
            with self.subTest(question=question[:10]):
                self.assertTrue(rule_based_answer(question).strip())

    def test_each_exercise_and_level_produces_an_answer(self) -> None:
        for exercise in ("squat", "push_up", "plank", "deadlift", "lunge", None):
            for level in ("BEGINNER", "INTERMEDIATE", "ADVANCED", None):
                with self.subTest(exercise=exercise, level=level):
                    answer = rule_based_answer(
                        "¿Qué corrijo?",
                        CoachContext(exercise=exercise, level=level, score=75, reps=3, average_angle=120),
                    )
                    self.assertTrue(answer.strip())

    def test_pain_answer_has_no_training_advice(self) -> None:
        answer = rule_based_answer("Me duele mucho la espalda", CoachContext(exercise="deadlift", score=90))
        self.assertIn("profesional de la salud", answer)
        self.assertNotIn("siguiente objetivo", answer)


if __name__ == "__main__":
    unittest.main()
