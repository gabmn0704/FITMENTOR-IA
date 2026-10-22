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

- `x` e `y` son coordenadas normalizadas (0 a 1). `confidence` va de 0 a 1.
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

Ángulo principal y umbrales de la máquina de repeticiones:

| Ejercicio | Ángulo principal | Arriba (`up`) | Fondo (`bottom`) | Feedback de forma |
| --- | --- | --- | --- | --- |
| `squat` | rodilla | > 150° | < 105° | rodilla fuera de 65°–115° |
| `lunge` | rodilla delantera | > 150° | < 105° | ideal 80°–100° |
| `push_up` | codo | > 150° | < 90° | codo y alineación hombro-cadera-tobillo |
| `deadlift` | bisagra de cadera | > 155° | < 115° | cadera fuera de 45°–145° |
| `plank` | — (isométrico) | — | — | alineación; `phase` siempre `hold` y `reps` 0 |

Máquina de estados por sesión: `ready` → `up` → `down` → `bottom` → `up` suma **+1 repetición**. El ángulo se suaviza con una media móvil de 5 muestras, de modo que un frame aislado no altera el conteo.

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
- `context` (opcional): `exercise` (texto de hasta 40 caracteres), `score` (0 a 100), `reps` (entero de 0 a 100000), `level` (texto de hasta 40 caracteres).

Respuesta:

```json
{ "answer": "En la plancha contrae abdomen y glúteos para que la cadera no caiga ni suba.", "source": "rules" }
```

- Con `ANTHROPIC_API_KEY`, se llama a `claude-haiku-5-5` mediante la API de Anthropic (timeout de 8 s, máximo 300 tokens, respuesta en español y concisa, sin diagnósticos médicos). `source` es `claude`.
- Sin clave, si la llamada falla o si la respuesta llega vacía, se usan reglas (consejos por palabra clave, por ejercicio y por puntuación; máximo tres consejos). `source` es `rules`.

### Errores de validación

Los cuerpos inválidos devuelven `422 Unprocessable Entity` con el detalle de FastAPI (por ejemplo, un ejercicio desconocido, coordenadas fuera de rango o una pregunta de más de 500 caracteres).

## CORS

El servicio acepta peticiones de cualquier origen (`Access-Control-Allow-Origin: *`).

## Despliegue

`render.yaml` define un servicio Docker con `healthCheckPath: /health`. Configura `ANTHROPIC_API_KEY` en el panel de Render; no se guarda en el repositorio.
