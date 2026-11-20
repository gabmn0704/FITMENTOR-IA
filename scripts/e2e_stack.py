"""Prueba de punta a punta del stack local: backend Java -> servicio de IA.

Requiere el backend (BACKEND_URL, por defecto http://localhost:8080) y la IA (AI_URL, por
defecto http://localhost:8000) en marcha, con el mismo AI_SHARED_SECRET. Flujo:
invitado -> /api/coach -> STOMP /app/pose (analisis y repeticiones) -> /api/sessions ->
/session/summary directo en la IA -> secreto compartido y CORS.

    python scripts/e2e_stack.py

Sale con codigo 1 si alguna comprobacion falla. No imprime secretos.
"""
import asyncio
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8080").rstrip("/")
AI = os.environ.get("AI_URL", "http://localhost:8000").rstrip("/")
SECRET = os.environ.get("AI_SHARED_SECRET", "")
ORIGIN = os.environ.get("E2E_ORIGIN", "http://localhost:5173")

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label))
    print(f"[{'OK' if ok else 'FALLA'}] {label}" + (f" - {detail}" if detail else ""))


def http(method: str, url: str, body=None, headers=None) -> tuple[int, dict, float, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw, status, response_headers = response.read(), response.status, dict(response.headers)
    except urllib.error.HTTPError as error:
        raw, status, response_headers = error.read(), error.code, dict(error.headers)
    elapsed = time.perf_counter() - started
    try:
        parsed = json.loads(raw) if raw else {}
    except ValueError:
        parsed = {"_raw": raw[:200].decode(errors="replace")}
    return status, parsed, elapsed, response_headers


def squat_points(knee_angle: float) -> list[dict]:
    """Cadera, rodilla y tobillo con el angulo de rodilla pedido (encuadre cuadrado)."""
    knee = (0.5, 0.5)
    hip = (0.5, 0.2)
    radians = math.radians(knee_angle)
    # La cadera esta "arriba" de la rodilla; el tobillo se obtiene rotando ese vector.
    ankle = (knee[0] + 0.3 * math.sin(radians), knee[1] - 0.3 * math.cos(radians))
    return [
        {"name": "left_hip", "x": hip[0], "y": hip[1], "confidence": 0.95},
        {"name": "left_knee", "x": knee[0], "y": knee[1], "confidence": 0.95},
        {"name": "left_ankle", "x": round(ankle[0], 4), "y": round(ankle[1], 4), "confidence": 0.95},
    ]


def stomp(command: str, headers: dict[str, str], body: str = "") -> str:
    lines = [command] + [f"{key}:{value}" for key, value in headers.items()]
    return "\n".join(lines) + "\n\n" + body + "\x00"


async def live_session(token: str, session_id: str) -> list[dict]:
    import websockets

    uri = BACKEND.replace("http", "ws", 1) + "/ws"
    received: list[dict] = []
    async with websockets.connect(uri, additional_headers={"Origin": ORIGIN}) as socket:
        await socket.send(
            stomp("CONNECT", {"accept-version": "1.2", "host": "localhost", "Authorization": f"Bearer {token}"})
        )
        connected = await asyncio.wait_for(socket.recv(), 5)
        check(str(connected).startswith("CONNECTED"), "STOMP CONNECT con token de invitado")
        await socket.send(stomp("SUBSCRIBE", {"id": "sub-0", "destination": f"/topic/feedback.{session_id}"}))
        # 3 repeticiones: arriba -> fondo -> arriba, con puntos intermedios.
        angles = ([170, 150, 120, 95, 90, 120, 150, 170] * 3) + [170]
        for angle in angles:
            frame = {
                "exercise": "squat",
                "session_id": session_id,
                "frame_width": 100,
                "frame_height": 100,
                "points": squat_points(angle),
            }
            await socket.send(
                stomp("SEND", {"destination": "/app/pose", "content-type": "application/json"}, json.dumps(frame))
            )
            await asyncio.sleep(0.12)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(socket.recv(), 0.6)
            except asyncio.TimeoutError:
                break
            for chunk in str(raw).split("\x00"):
                if chunk.startswith("MESSAGE"):
                    received.append(json.loads(chunk.split("\n\n", 1)[1]))
    return received


def main() -> int:
    status, health, _, _ = http("GET", f"{AI}/health")
    check(status == 200 and health.get("status") == "ok", "IA /health", str(health.get("coach", {}).get("mode")))
    status, body, _, _ = http("GET", f"{BACKEND}/actuator/health")
    check(status == 200 and body.get("status") == "UP", "Backend /actuator/health")

    status, auth, elapsed, _ = http("POST", f"{BACKEND}/api/auth/guest", {"name": "E2E"})
    token = auth.get("token", "")
    check(status == 200 and bool(token), "Login invitado", f"{elapsed * 1000:.0f} ms")
    bearer = {"Authorization": f"Bearer {token}"}

    questions = (
        ("Como mejoro mi sentadilla?", False),
        ("Me duele la rodilla al bajar", True),
        ("Que puedo comer despues de entrenar?", False),
    )
    for question, expect_pain in questions:
        status, answer, elapsed, _ = http("POST", f"{BACKEND}/api/coach", {"question": question}, bearer)
        text = answer.get("answer", "")
        ok = status == 200 and bool(text) and (not expect_pain or "profesional" in text)
        check(ok, f"/api/coach '{question}'", f"{status}, {elapsed * 1000:.0f} ms, source={answer.get('source')}")
    status, _, _, _ = http("POST", f"{BACKEND}/api/coach", {"question": ""}, bearer)
    check(status == 400, "/api/coach pregunta vacia -> 400", str(status))
    status, _, _, _ = http("POST", f"{BACKEND}/api/coach", {"question": "hola"})
    check(status in (401, 403), "/api/coach sin token rechazado", str(status))

    session_id = f"e2e-{int(time.time())}"
    try:
        messages = asyncio.run(live_session(token, session_id))
    except Exception as error:  # noqa: BLE001
        messages = []
        check(False, "Sesion en vivo por STOMP", type(error).__name__ + ": " + str(error)[:160])
    analyzed = [m for m in messages if m.get("type") == "feedback"]
    check(len(analyzed) > 0, "Backend publica feedback de /analyze", f"{len(messages)} mensajes")
    max_reps = max((m.get("reps", 0) for m in analyzed), default=0)
    check(max_reps == 3, "Contador de repeticiones de la IA llega por el backend", f"reps={max_reps}")
    check(all(m.get("message") for m in analyzed), "Todos los mensajes llevan texto de feedback")

    status, _, _, _ = http(
        "POST",
        f"{BACKEND}/api/sessions",
        {"exercise": "squat", "reps": 3, "accuracy": 88.5, "durationSeconds": 40, "avgAngles": {"knee": 120.5}},
        bearer,
    )
    check(status == 201, "POST /api/sessions guarda la sesion", str(status))

    token_header = {"X-Internal-Token": SECRET} if SECRET else {}
    status, summary, elapsed, _ = http(
        "POST",
        f"{AI}/session/summary",
        {"exercise": "squat", "reps": 3, "score": 88.5, "level": "BEGINNER", "average_angle": 120.5, "duration_seconds": 40},
        token_header,
    )
    check(status == 200 and bool(summary.get("summary")), "IA /session/summary", f"{elapsed * 1000:.0f} ms")
    if SECRET:
        status, _, _, _ = http("POST", f"{AI}/analyze", {"exercise": "squat", "points": []})
        check(status == 401, "IA rechaza /analyze sin X-Internal-Token", str(status))
        status, _, _, _ = http("POST", f"{AI}/analyze", {"exercise": "squat", "points": []}, {"X-Internal-Token": "otro"})
        check(status == 401, "IA rechaza token incorrecto", str(status))

    status, _, _, headers = http(
        "OPTIONS",
        f"{BACKEND}/api/coach",
        None,
        {"Origin": ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"},
    )
    allowed = {key.lower(): value for key, value in headers.items()}.get("access-control-allow-origin")
    check(status == 200 and allowed in (ORIGIN, "*"), "CORS del backend permite el origen del front", f"{status} {allowed}")
    _, _, _, headers = http("GET", f"{AI}/health", None, {"Origin": "http://evil.example"})
    check("access-control-allow-origin" not in {key.lower() for key in headers}, "IA no emite CORS (solo servidor a servidor)")

    failed = [label for ok, label in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} comprobaciones correctas")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
