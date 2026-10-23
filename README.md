# FitMentor IA Service

Servicio FastAPI (Python 3.12) que puntúa la técnica de ejercicios a partir de landmarks corporales normalizados. Valida la confianza de cada punto, calcula ángulos articulares respetando la relación de aspecto del encuadre, evalúa la alineación del cuerpo, cuenta repeticiones por sesión y ofrece consejos de entrenador con Claude o con reglas locales.

El servicio analiza landmarks; no detecta personas en imágenes ni vídeo. El frontend ejecuta la detección de pose (MediaPipe) en el navegador.

## Desarrollo

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Tests (los mismos que ejecuta el CI):

```bash
python -m unittest discover -s tests -v
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | No | Si existe, `/coach` consulta a Claude. Si falta o la llamada falla, responde con reglas. Nunca se registra en logs. |
| `AI_SHARED_SECRET` | Sí en producción | Secreto compartido con el backend Java. Si está definida, `/analyze` y `/coach` exigen la cabecera `X-Internal-Token` con ese valor (si falta o no coincide, `401 {"detail":"unauthorized"}`). `/health` y `/docs` no la exigen. Si no está definida, no hay autenticación y el arranque registra una advertencia. Nunca se registra en logs. |
| `PORT` | No | Puerto de escucha en el contenedor (por defecto `8000`). |

## Endpoints

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /analyze`

Ejercicios soportados: `squat`, `push_up`, `plank`, `deadlift`, `lunge`.

Petición:

```json
{
  "exercise": "squat",
  "points": [
    { "name": "left_hip", "x": 0.5, "y": 0.1, "confidence": 0.92 },
    { "name": "left_knee", "x": 0.5, "y": 0.5, "confidence": 0.9 },
    { "name": "left_ankle", "x": 0.5, "y": 0.9, "confidence": 0.88 }
  ],
  "frame_width": 16,
  "frame_height": 9,
  "session_id": "partida-42"
}
```

- `x` e `y` son coordenadas normalizadas (0 a 1); se admite un margen de -0.5 a 1.5 porque MediaPipe puede devolver landmarks algo fuera del encuadre. `confidence` va de 0 a 1.
- Límites: como máximo 64 puntos, `name` de 1 a 40 caracteres, `frame_width` ≤ 7680 y `frame_height` ≤ 4320.
- Los nombres pueden ser genéricos (`hip`) o por lado (`left_hip`, `right_hip`). Se normalizan a minúsculas y, si hay duplicados, se usa el de mayor confianza.
- `frame_width` y `frame_height` (opcionales, por defecto 16 y 9) conservan la geometría del ángulo.
- `session_id` (opcional, 1 a 128 caracteres) activa el conteo de repeticiones de esa sesión.

Respuesta:

```json
{
  "score": 90.0,
  "feedback": ["Buena posición. Muévete con control y mantén una respiración constante."],
  "knee_angle": 96.4,
  "angles": { "knee": 96.4 },
  "confidence": 0.9,
  "analyzed": true,
  "phase": "bottom",
  "reps": 3,
  "message": "Buena posición. Muévete con control y mantén una respiración constante."
}
```

- `message` es `feedback` unido con espacios.
- `analyzed` es `false` si algún punto requerido tiene confianza menor de 0.55; entonces `score` es 0 y `feedback` pide volver a colocarse en el encuadre.
- `knee_angle` está presente en `squat` y `lunge`; es `null` en el resto.
- Si dos puntos de la articulación medida coinciden (ángulo no medible), el frame se marca como no analizado: nunca se devuelve un 0° falso.

Ángulo principal y umbrales de la máquina de repeticiones:

| Ejercicio | Ángulo principal | Arriba (`up`) | Fondo (`bottom`) | Feedback de forma |
| --- | --- | --- | --- | --- |
| `squat` | rodilla | > 150° | < 105° | rodilla fuera de 65°–115° |
| `lunge` | rodilla delantera | > 150° | < 105° | ideal 80°–100° |
| `push_up` | codo | > 150° | < 90° | codo y alineación hombro-cadera-tobillo |
| `deadlift` | bisagra de cadera | > 155° | < 115° | cadera fuera de 45°–145° |
| `plank` | — (isométrico) | — | — | alineación; `phase` siempre `hold` y `reps` 0 |

Máquina de estados por sesión: `ready` → `up` → `down` → `bottom` → `up` suma **+1 repetición**. No hay suavizado de media ni de mediana: con frames cada ~340 ms, un suavizado de varias muestras pierde repeticiones cortas. En su lugar, un **filtro de saltos** retiene una muestra que cambia más de 60° respecto a la anterior y solo la acepta si la siguiente la confirma; así un pico aislado no cuenta y un movimiento real no se retrasa.

- Histéresis: entrar en `bottom` exige bajar del umbral de fondo y volver a `up` exige superar el umbral alto; la franja intermedia no cambia de estado.
- Las muestras no medibles (ángulo ausente o fuera de 10–180°; 0° suele ser un landmark degenerado) no mueven la máquina.
- Un contador que empieza fuera de la posición alta no cuenta la primera subida: hay que haber estado arriba antes. El peso muerto es la excepción (`starts_at_bottom`): empieza en el suelo y su primera subida ya cuenta.
- Una repetición incompleta (no llega a la posición alta) no suma. Mientras el ángulo está entre el fondo y la posición alta, `phase` sigue en `down` o `bottom`.

- Sin `session_id`, solo se evalúa el frame: `reps` es 0 y `phase` describe ese frame.
- Cambiar de `exercise` dentro de la misma sesión reinicia el contador.
- Las sesiones se guardan en una caché LRU de 500 entradas (la menos usada recientemente se descarta). Si se descarta una sesión, su conteo vuelve a 0.

### `POST /coach`

Petición:

```json
{
  "question": "¿Cómo evito que se me caigan las caderas en la plancha?",
  "context": { "exercise": "plank", "score": 72, "reps": 0, "level": "principiante" }
}
```

- `question`: obligatoria, de 1 a 500 caracteres (se recortan espacios al inicio y al final).
- `context` (opcional): `exercise` (texto de hasta 40 caracteres; acepta los ids `squat`, `push_up`, `plank`, `deadlift`, `lunge` y nombres como `Sentadilla` o `flexiones`), `score` (0 a 100, precisión de la sesión), `reps` (entero de 0 a 100000), `level` (texto de hasta 40 caracteres: `BEGINNER`, `INTERMEDIATE`, `ADVANCED`, o `principiante`, `intermedio`, `avanzado`), y `average_angle` (opcional, 0 a 180, ángulo medio de la articulación principal en la sesión). Los campos nuevos son opcionales: las peticiones anteriores siguen siendo válidas.

Respuesta:

```json
{ "answer": "En la plancha contrae abdomen y glúteos para que la cadera no caiga ni suba.", "source": "rules" }
```

- Con `ANTHROPIC_API_KEY`, se llama a `claude-haiku-5-5` mediante la API de Anthropic (timeout de 8 s, `max_tokens` 1024 porque Haiku 5.5 razona por defecto y ese razonamiento cuenta en el límite, `output_config.effort` = `low`; solo se usan los bloques `text`). El prompt de sistema fija el rol de coach de técnica, respuestas en español de máximo tres frases, límites de seguridad (dolor → parar y consultar a un profesional; nutrición solo general) y trata la pregunta como datos: va entre `<pregunta>` y las métricas de la sesión entre `<sesion>`. Una negativa (`stop_reason: refusal`) o una respuesta vacía pasa a reglas. `source` es `claude`.
- Sin clave, si la llamada falla o si la respuesta llega vacía, se usan reglas. `source` es `rules`. Las reglas responden según la intención de la pregunta, en este orden:
  1. **Dolor, lesión, mareo u opresión en el pecho** (se detecta antes de llamar a Claude, así que también en modo Claude): se recomienda parar y consultar a un profesional de la salud, sin consejos de entrenamiento. `source` es `rules`.
  2. **Intento de cambiar las instrucciones** (ignora, omite, finge, system prompt, API key, etc.): se redirige al entrenamiento con un consejo del ejercicio actual. Ninguna otra pregunta se trata como ataque.
  3. **Nutrición**: orientación general (proteína, hidratación, verduras) con aviso de que no es consejo médico ni nutricional personalizado.
  4. **Descanso**: tiempos entre series y entre sesiones según el nivel, y aviso de consultar si las agujetas persisten o el dolor es agudo.
  5. **Técnica**: consejos por palabra clave (coincidencia al inicio de palabra), por ejercicio y por métricas (precisión por tramos, repeticiones si se preguntan, ángulo medio fuera del rango esperado), siempre con un **siguiente objetivo** según el nivel (repeticiones o segundos y precisión mínima).
  6. **Saludo o pregunta sobre el coach**: presentación breve (con el consejo del ejercicio de contexto si lo hay). Cualquier otra pregunta fuera de tema recibe el consejo general del ejercicio o del entrenamiento, nunca una respuesta vacía.

### Errores de validación

Los cuerpos inválidos devuelven `422 Unprocessable Entity` con el detalle de FastAPI (por ejemplo, un ejercicio desconocido, coordenadas fuera de rango o una pregunta de más de 500 caracteres).

## Autenticación

Con `AI_SHARED_SECRET` definida, cada petición a `/analyze` y `/coach` debe enviar la cabecera `X-Internal-Token` con el mismo valor. La comparación es de tiempo constante (`hmac.compare_digest`). Sin cabecera o con un valor incorrecto la respuesta es `401` con `{"detail":"unauthorized"}`. El backend Java la envía en cada llamada.

## CORS

No hay CORS habilitado: el servicio solo lo llama el backend (servidor a servidor) y el navegador no debe llamarlo directamente. Un origen de navegador no recibe `Access-Control-Allow-Origin`.

## Despliegue

`render.yaml` define un servicio Docker con `healthCheckPath: /health`. Configura `ANTHROPIC_API_KEY` y `AI_SHARED_SECRET` en el panel de Render (ambas con `sync: false`); no se guardan en el repositorio. `AI_SHARED_SECRET` debe ser idéntica a la del backend.
