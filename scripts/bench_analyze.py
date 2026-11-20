"""Latencia de /analyze y /coach (reglas) contra una IA en marcha.

    AI_URL=http://127.0.0.1:8000 AI_SHARED_SECRET=... python scripts/bench_analyze.py [N]

En Windows usa 127.0.0.1 y no "localhost": resolver ::1 primero suma ~200 ms por conexión nueva.
"""
import json
import os
import statistics
import sys
import time
import urllib.request

AI = os.environ.get("AI_URL", "http://127.0.0.1:8000").rstrip("/")
SECRET = os.environ.get("AI_SHARED_SECRET", "")
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def post(path: str, body: dict) -> float:
    headers = {"Content-Type": "application/json"}
    if SECRET:
        headers["X-Internal-Token"] = SECRET
    request = urllib.request.Request(AI + path, data=json.dumps(body).encode(), headers=headers, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
    return (time.perf_counter() - started) * 1000


def report(label: str, samples: list[float]) -> None:
    ordered = sorted(samples)
    print(f"{label}: n={len(ordered)} p50={statistics.median(ordered):.1f} ms p95={ordered[int(len(ordered) * 0.95) - 1]:.1f} ms max={ordered[-1]:.1f} ms")


frame = {
    "exercise": "squat",
    "session_id": "bench",
    "frame_width": 1280,
    "frame_height": 720,
    "points": [{"name": f"p{i}", "x": 0.5, "y": 0.5, "confidence": 0.9} for i in range(30)]
    + [
        {"name": "left_hip", "x": 0.5, "y": 0.3, "confidence": 0.9},
        {"name": "left_knee", "x": 0.5, "y": 0.55, "confidence": 0.9},
        {"name": "left_ankle", "x": 0.4, "y": 0.8, "confidence": 0.9},
    ],
}
report("/analyze (33 puntos)", [post("/analyze", frame) for _ in range(COUNT)])
questions = ["¿Cómo mejoro mi sentadilla?", "¿Cuánta proteína necesito?", "Hola", "¿Cómo evito que se me caigan las caderas en la plancha?"]
report("/coach (reglas)", [post("/coach", {"question": questions[i % 4] + " " * (i % 7)}) for i in range(COUNT)])
