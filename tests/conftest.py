import pytest

from app import coach


@pytest.fixture(autouse=True)
def _clean_coach_state():
    """Caché y disyuntor del coach son globales: cada prueba parte de cero."""
    coach.clear_state()
    yield
    coach.clear_state()
