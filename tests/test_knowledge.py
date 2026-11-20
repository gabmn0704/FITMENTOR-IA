import unittest

from app.coach import CoachContext, _model_prompt, rule_based_answer
from app.knowledge import KNOWLEDGE, retrieve, tokenize

# Pregunta -> ficha que debe salir primero. Conjunto de referencia de la recuperación.
GOLDEN: tuple[tuple[str, str | None, str], ...] = (
    ("¿Cómo evito que se me caigan las caderas en la plancha?", "plank", "plank-errores"),
    ("¿Hasta dónde debo bajar en la sentadilla?", "squat", "squat-profundidad"),
    ("Mis talones se levantan al bajar", "squat", "squat-talones"),
    ("¿Cuánta proteína debo comer?", None, "nut-proteina"),
    ("¿Cuántos días por semana debo entrenar?", None, "gen-frecuencia"),
    ("¿Cómo hago la bisagra de cadera?", None, "deadlift-tecnica"),
    ("Tengo la espalda redondeada en el peso muerto", "deadlift", "deadlift-errores"),
    ("No puedo hacer flexiones completas, ¿cómo empiezo?", "push_up", "push-up-progresion"),
    ("¿Cómo respiro mientras entreno?", None, "gen-respiracion"),
    ("¿Cuánta agua debo beber al entrenar?", None, "nut-hidratacion"),
)


class RetrievalTests(unittest.TestCase):
    def test_golden_questions_return_the_expected_card_first(self) -> None:
        for question, exercise, expected in GOLDEN:
            with self.subTest(question=question):
                found = retrieve(question, exercise)
                self.assertTrue(found, "sin resultados")
                self.assertEqual(found[0].id, expected)

    def test_off_topic_questions_retrieve_nothing(self) -> None:
        for question in ("¿Qué tiempo hará mañana?", "Recomiéndame una película", "asdf qwer"):
            with self.subTest(question=question):
                self.assertEqual(retrieve(question), [])

    def test_cards_of_other_exercises_are_not_returned(self) -> None:
        for entry in retrieve("errores frecuentes y cómo corregirlos", "plank", limit=10):
            self.assertIn(entry.exercise, (None, "plank"))

    def test_plurals_and_accents_match(self) -> None:
        self.assertEqual(tokenize("Rodillas"), tokenize("rodilla"))
        self.assertEqual(tokenize("PROTEÍNA"), tokenize("proteina"))

    def test_every_card_is_well_formed(self) -> None:
        ids = [entry.id for entry in KNOWLEDGE]
        self.assertEqual(len(ids), len(set(ids)), "ids repetidos")
        for entry in KNOWLEDGE:
            with self.subTest(entry=entry.id):
                self.assertGreater(len(entry.text), 60)
                self.assertLess(len(entry.text), 420)
                self.assertTrue(entry.keywords)
                self.assertNotIn("<", entry.text)  # no puede cerrar las etiquetas del prompt

    def test_every_supported_exercise_has_technique_errors_and_progression(self) -> None:
        for exercise in ("squat", "push_up", "plank", "deadlift", "lunge"):
            topics = {entry.topic for entry in KNOWLEDGE if entry.exercise == exercise}
            with self.subTest(exercise=exercise):
                self.assertTrue({"tecnica", "errores", "progresion"} <= topics)


class IntegrationTests(unittest.TestCase):
    def test_rules_answer_uses_the_card(self) -> None:
        answer = rule_based_answer("Mis talones se levantan", CoachContext(exercise="squat"))
        self.assertIn("movilidad de tobillo", answer)

    def test_pain_never_goes_through_the_knowledge_base(self) -> None:
        answer = rule_based_answer("Me duele la rodilla en la sentadilla", CoachContext(exercise="squat"))
        self.assertIn("profesional de la salud", answer)
        self.assertNotIn("Errores frecuentes", answer)

    def test_model_prompt_carries_verified_cards_in_their_own_block(self) -> None:
        prompt = _model_prompt("Mis talones se levantan", CoachContext(exercise="squat"))
        self.assertIn("<conocimiento>", prompt)
        self.assertIn("movilidad de tobillo", prompt)
        self.assertTrue(prompt.rstrip().endswith("</conocimiento>"))

    def test_model_prompt_without_a_match_has_no_block(self) -> None:
        self.assertNotIn("<conocimiento>", _model_prompt("¿Qué tiempo hará mañana?", CoachContext()))


if __name__ == "__main__":
    unittest.main()
