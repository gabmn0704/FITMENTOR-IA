import os
import unittest
from unittest import mock

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.main import app
from app.security import PUBLIC_PATHS

client = TestClient(app)
SECRET = "test-secret-value"
ANALYZE_BODY = {
    "exercise": "plank",
    "points": [
        {"name": "shoulder", "x": 0.1, "y": 0.1, "confidence": 1},
        {"name": "hip", "x": 0.5, "y": 0.1, "confidence": 1},
        {"name": "ankle", "x": 0.9, "y": 0.1, "confidence": 1},
    ],
}
COACH_BODY = {"question": "¿Cómo mejoro la técnica?"}


class SharedSecretTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("AI_SHARED_SECRET", None)

    def with_secret(self) -> None:
        os.environ["AI_SHARED_SECRET"] = SECRET

    def assert_unauthorized(self, response) -> None:
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "unauthorized"})

    def test_missing_header_is_rejected_when_secret_is_set(self) -> None:
        self.with_secret()
        self.assert_unauthorized(client.post("/analyze", json=ANALYZE_BODY))
        self.assert_unauthorized(client.post("/coach", json=COACH_BODY))

    def test_wrong_header_is_rejected(self) -> None:
        self.with_secret()
        for value in ("", "wrong", SECRET + "x", SECRET[:-1]):
            with self.subTest(value=value):
                response = client.post("/analyze", json=ANALYZE_BODY, headers={"X-Internal-Token": value})
                self.assert_unauthorized(response)

    def test_correct_header_is_accepted(self) -> None:
        self.with_secret()
        headers = {"X-Internal-Token": SECRET}

        analyze = client.post("/analyze", json=ANALYZE_BODY, headers=headers)
        coach = client.post("/coach", json=COACH_BODY, headers=headers)

        self.assertEqual(analyze.status_code, 200, analyze.text)
        self.assertEqual(coach.status_code, 200, coach.text)

    def test_health_stays_open_when_secret_is_set(self) -> None:
        self.with_secret()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_docs_stay_open_when_secret_is_set(self) -> None:
        self.with_secret()
        self.assertEqual(client.get("/docs").status_code, 200)
        self.assertEqual(client.get("/openapi.json").status_code, 200)

    def test_every_non_public_route_requires_the_header(self) -> None:
        self.with_secret()
        checked = []
        for route in app.routes:
            if not isinstance(route, APIRoute) or route.path in PUBLIC_PATHS:
                continue
            checked.append(route.path)
            for method in route.methods:
                with self.subTest(path=route.path, method=method):
                    response = client.request(method, route.path, json=ANALYZE_BODY)
                    self.assert_unauthorized(response)
        self.assertIn("/analyze", checked)
        self.assertIn("/coach", checked)

    def test_everything_works_without_secret(self) -> None:
        self.assertEqual(client.get("/health").status_code, 200)
        analyze = client.post("/analyze", json=ANALYZE_BODY)
        coach = client.post("/coach", json=COACH_BODY)
        self.assertEqual(analyze.status_code, 200, analyze.text)
        self.assertEqual(coach.status_code, 200, coach.text)

    def test_blank_secret_counts_as_unset(self) -> None:
        os.environ["AI_SHARED_SECRET"] = "   "
        self.assertEqual(client.post("/analyze", json=ANALYZE_BODY).status_code, 200)

    def test_startup_warns_when_secret_is_unset(self) -> None:
        with self.assertLogs("app.security", level="WARNING") as logs:
            with TestClient(app):
                pass
        self.assertIn("AI_SHARED_SECRET", logs.output[0])

    def test_startup_does_not_warn_when_secret_is_set(self) -> None:
        self.with_secret()
        with mock.patch("app.security.logger") as logger:
            with TestClient(app):
                pass
        logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
