"""Aísla el estado global del coach (caché y disyuntor) para cualquier ejecutor de pruebas.

`conftest.py` lo hace solo con pytest; el CI usa `python -m unittest discover`, que también
importa este módulo, así que cada prueba parte de cero en ambos casos.
"""

import unittest

from app import coach

_original_run = unittest.TestCase.run


def _isolated_run(self: unittest.TestCase, result: unittest.TestResult | None = None):
    coach.clear_state()
    try:
        return _original_run(self, result)
    finally:
        coach.clear_state()


if unittest.TestCase.run is _original_run:
    unittest.TestCase.run = _isolated_run  # type: ignore[method-assign]
