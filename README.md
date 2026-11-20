# FitMentor IA Service

Servicio FastAPI (Python 3.12) que puntúa la técnica de ejercicios a partir de landmarks corporales normalizados. Valida la confianza de cada punto, calcula ángulos articulares respetando la relación de aspecto del encuadre, evalúa la alineación del cuerpo, cuenta repeticiones por sesión y ofrece consejos de entrenador con Claude o con reglas locales.

El servicio analiza landmarks; no detecta personas en imágenes ni vídeo. El frontend ejecuta la detección de pose (MediaPipe) en el navegador.

## Desarrollo

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Tests (los mismos que ejecuta el CI; `python -m pytest -q` también funciona):

```bash
python -m unittest discover -s tests -v
```

Evaluaciones y pruebas de punta a punta (ver "Calidad y evaluación"):

```bash
python -m evals.pose_eval            # precisión/recall del análisis de pose contra poses sintéticas etiquetadas
python -m evals.pose_eval --sweep    # barrido de umbrales
python -m evals.coach_eval           # criterios del coach (offline, solo reglas)
python -m evals.coach_eval --live    # además contra el proveedor, si hay ANTHROPIC_API_KEY o LLM_API_KEY
python scripts/e2e_stack.py          # backend + IA levantados: invitado -> coach -> pose por STOMP -> sesión -> resumen
python scripts/bench_analyze.py      # latencia de /analyze y /coach
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
| --- | --- | --- |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | No | Proveedor gratuito compatible con la API de OpenAI (por defecto Groq; la clave se crea en console.groq.com). Se usa si no hay `ANTHROPIC_API_KEY` o si Claude falla; si todo falla, responde con reglas. |
| `ANTHROPIC_API_KEY` | No | Si existe, `/coach` consulta a Claude. Si falta o la llamada falla, responde con reglas. Nunca se registra en logs. |
| `AI_SHARED_SECRET` | Sí en producción | Secreto compartido con el backend Java. Si está definida, `/analyze` y `/coach` exigen la cabecera `X-Internal-Token` con ese valor (si falta o no coincide, `401 {"detail":"unauthorized"}`). `/health` y `/docs` no la exigen. Si no está definida, no hay autenticación y el arranque registra una advertencia. Nunca se registra en logs. |
| `PORT` | No | Puerto de escucha en el contenedor (por defecto `8000`). |

## Endpoints

### `GET /health` (también `HEAD`)

```json
{ "status": "ok", "coach": { "mode": "claude", "claude": { "configured": true, "paused": false }, "llm": { "configured": true, "paused": false }, "rules": true } }
```

No hace llamadas externas ni lee el cuerpo de la petición; `HEAD /health` responde 200 sin cuerpo.

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
  "feedback": ["Buena posición: mantén el control y respira de forma constante."],
  "knee_angle": 96.4,
  "angles": { "knee": 96.4 },
  "confidence": 0.9,
  "analyzed": true,
  "phase": "bottom",
  "reps": 3,
  "message": "Buena posición: mantén el control y respira de forma constante."
}
```

- `message` es la corrección principal (`feedback[0]`), siempre una sola frase.
- `analyzed` es `false` si algún punto requerido tiene confianza menor de 0.55; entonces `score` es 0 y `feedback` pide volver a colocarse en el encuadre.
- `knee_angle` está presente en `squat` y `lunge`; es `null` en el resto.
- Si dos puntos de la articulación medida coinciden (ángulo no medible), el frame se marca como no analizado: nunca se devuelve un 0° falso. "Coinciden" es relativo al encuadre (segmento menor del 0,15 % del lado menor), de modo que `frame_width` y `frame_height` pueden ser píxeles reales o solo una proporción.
- Los umbrales de forma viven en `app/main.py` (`LIMITS`) y están calibrados con `evals/pose_eval.py`.

Ángulo principal y umbrales de la máquina de repeticiones:

| Ejercicio | Ángulo principal | Arriba (`up`) | Fondo (`bottom`) | Feedback de forma |
| --- | --- | --- | --- | --- |
| `squat` | rodilla | > 150° | < 105° | rodilla fuera de 65°–115° |
| `lunge` | rodilla delantera | > 150° | < 105° | rodilla fuera de 70°–110° |
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

- Con `ANTHROPIC_API_KEY`, se llama a `claude-haiku-5-5` mediante la API de Anthropic (timeout de 4 s, `max_tokens` 700 porque Haiku 5.5 razona por defecto y ese razonamiento cuenta en el límite, `output_config.effort` = `low`; solo se usan los bloques `text`). El prompt de sistema fija el rol de coach de técnica, respuestas en español de máximo tres frases, límites de seguridad (dolor → parar y consultar a un profesional; nutrición solo general) y trata la pregunta como datos: va entre `<pregunta>` y las métricas de la sesión entre `<sesion>`. Una negativa (`stop_reason: refusal`) o una respuesta vacía pasa a reglas. `source` es `claude`.
- Sin clave, si la llamada falla o si la respuesta llega vacía, se usan reglas. `source` es `rules`. Las reglas responden según la intención de la pregunta, en este orden:
  1. **Dolor, lesión, mareo u opresión en el pecho** (se detecta antes de llamar a Claude, así que también en modo Claude): se recomienda parar y consultar a un profesional de la salud, sin consejos de entrenamiento. `source` es `rules`.
  2. **Intento de cambiar las instrucciones** (ignora, omite, finge, system prompt, API key, etc.): se redirige al entrenamiento con un consejo del ejercicio actual. Ninguna otra pregunta se trata como ataque.
  3. **Nutrición**: orientación general (proteína, hidratación, verduras) con aviso de que no es consejo médico ni nutricional personalizado.
  4. **Descanso**: tiempos entre series y entre sesiones según el nivel, y aviso de consultar si las agujetas persisten o el dolor es agudo.
  5. **Técnica**: consejos por palabra clave (coincidencia al inicio de palabra), por ejercicio y por métricas (precisión por tramos, repeticiones si se preguntan, ángulo medio fuera del rango esperado), siempre con un **siguiente objetivo** según el nivel (repeticiones o segundos y precisión mínima).
  6. **Saludo o pregunta sobre el coach**: presentación breve (con el consejo del ejercicio de contexto si lo hay). Cualquier otra pregunta fuera de tema recibe el consejo general del ejercicio o del entrenamiento, nunca una respuesta vacía.

Tono y robustez:

- Español neutro, profesional y motivador. El prompt prohíbe jerga y coloquialismos; si un modelo responde con jerga (`parce`, `quiubo`, etc.), su respuesta se descarta y se pasa al siguiente proveedor o a reglas.
- Presupuesto total de 5 s por pregunta (4 s por proveedor y 1,5 s para conectar), disyuntor por proveedor y caché de 10 minutos que ignora tildes, mayúsculas y signos. Las llamadas reutilizan un cliente HTTP compartido (keep-alive).
- Si ocurre un error inesperado, `/coach` responde por reglas y `/analyze` devuelve un frame no analizado: nunca un 500.

### `POST /session/summary`

Resumen instantáneo de una sesión terminada, por reglas (no llama a ningún modelo). Exige `X-Internal-Token` igual que `/coach`. Todos los campos son opcionales:

```json
{ "exercise": "squat", "reps": 12, "score": 91, "level": "INTERMEDIATE", "average_angle": 95, "duration_seconds": 300 }
```

Respuesta:

```json
{
  "summary": "Excelente trabajo en tu sesión de sentadilla: 12 repeticiones con 91% de precisión.",
  "highlights": ["Tu precisión es 91%: muy buena, ahora busca repetir esa forma en cada serie."],
  "next_goal": "Para tu nivel intermedio, tu siguiente objetivo es 16 repeticiones por serie con precisión de 85% o más."
}
```

En `plank` se usa `duration_seconds` en lugar de `reps`. Con 0 repeticiones se sugiere revisar el encuadre de la cámara.

### Qué es y qué no es esta IA

- **No se entrena ni se afina ningún modelo.** El análisis de pose es geometría (ángulos y alineación) con umbrales calibrados con datos sintéticos; el coach usa reglas, una base de conocimiento propia y, si hay clave, un modelo de lenguaje de un tercero.
- Los números de calidad (sección "Calidad y evaluación") se miden con poses **sintéticas** y con casos **escritos a mano**. No equivalen a precisión sobre personas reales ni a una evaluación clínica.
- El coach no diagnostica ni sustituye a un profesional de la salud.

### Errores de validación

Los cuerpos inválidos devuelven `422 Unprocessable Entity` con el detalle de FastAPI (por ejemplo, un ejercicio desconocido, coordenadas fuera de rango o una pregunta de más de 500 caracteres).

## Calidad y evaluación

### Base de conocimiento (`app/knowledge.py`)

Cerca de 35 fichas escritas a mano (técnica, errores frecuentes, progresión y seguridad de los cinco ejercicios, más calentamiento, respiración, descanso, agujetas, frecuencia y nutrición básica). Se recuperan por palabras clave con BM25 (sin dependencias ni embeddings; ignora tildes, mayúsculas y plurales). La ficha más relevante abre la respuesta por reglas y, con modelo, hasta tres fichas viajan en un bloque `<conocimiento>` del prompt con la instrucción de apoyarse en ellas y no inventar cifras. El dolor se atiende antes y nunca pasa por la base ni por el modelo. Es orientación general; las fichas las revisa una persona al editarlas, no hay verificación automática de su contenido.

### Calibración del análisis de pose (`evals/pose_synth.py`, `evals/pose_eval.py`)

`pose_synth` genera de forma reproducible (semilla fija) poses buenas y malas por ejercicio: el ángulo o desvío verdadero sale de rangos de referencia de entrenamiento (no de los umbrales del analizador), con encuadres 16:9, 9:16, 4:3 y 1:1, orientación y lado aleatorios, ruido gaussiano de landmark (0,8 % del lado menor), confianzas entre 0,6 y 0,99 y landmarks irrelevantes. Entre clase buena y mala hay una franja de guarda sin etiquetar. `pose_eval` reporta precisión y recall por clase, falsos positivos sobre poses buenas y omisión de las malas; con `--sweep` barre cada umbral.

Resultado con 300 muestras por clase (5 100 en total, `python -m evals.pose_eval`):

| Ejercicio.dimensión | Falsos positivos (poses buenas) | Omisión (poses malas) |
| --- | --- | --- |
| `squat.knee` | 2,7 % | 0,2 % |
| `lunge.knee` | 2,3 % | 2,5 % |
| `push_up.elbow` | 0,2 % | 0,3 % |
| `push_up.line` | 0,0 % | 0,0 % |
| `plank.line` | 0,0 % | 0,0 % |
| `deadlift.hinge` | 0,0 % | 2,0 % |

F1 macro 0,991 y ningún frame limpio sin analizar. Con más ruido de landmark (1,5 %) el F1 macro baja a 0,976 y con 2,5 % a 0,928. El barrido mostró que los umbrales de sentadilla, flexión, plancha y peso muerto ya estaban en el óptimo; los de la zancada (80°–100°) eran demasiado estrechos (25,7 % de falsos positivos) y pasaron a 70°–110°. Además se corrigió un mínimo de segmento en píxeles absolutos que dejaba sin analizar poses válidas cuando `frame_width`/`frame_height` eran una proporción (ahora es relativo al encuadre).

Límites: son frames estáticos en la pose de interés (el fondo de la repetición); no se mide el movimiento, ni la detección de MediaPipe sobre personas reales, y las franjas de guarda son discutibles incluso para un entrenador. Un frame de pie en la sentadilla recibe el aviso de bajar más porque `/analyze` evalúa cada frame por separado. Afinar con datos reales exigiría vídeos etiquetados por entrenadores.

### Evaluación del coach (`evals/coach_cases.py`, `evals/coach_eval.py`)

45 casos (dolor, técnica de los cinco ejercicios, descanso y progresión, nutrición, intentos de manipulación, conversación) con criterios automáticos: español, de 40 a 900 caracteres, máximo 8 frases, sin jerga, sin emojis, sin "como IA", sin jerga técnica opaca; ante dolor, parar y consultar sin consejos de entrenamiento; en nutrición, aviso de que no es consejo médico y nunca menos de 1200 kcal; ante manipulación, no filtrar instrucciones y volver al entrenamiento. Offline (solo reglas): 45/45, latencia p50 menor a 1 ms. Los casos y criterios los escribió el mismo equipo que las reglas: sirven como regresión de seguridad y tono, no como medida independiente de utilidad. Con `--live` se corre lo mismo contra el proveedor configurado (sin clave lo indica y no falla). La ruta del modelo se prueba en `tests/test_coach_eval.py` con un transporte simulado: respuesta correcta, jerga, proveedor caído o con error 503, el dolor no llega al modelo y el prompt lleva las fichas. No se ha ejecutado contra un proveedor real en esta versión (no había clave).

## Contrato para web y app móvil

La web y la app móvil **no llaman a este servicio**: hablan con el backend Java, que reenvía a la IA con `X-Internal-Token`. Si la IA cae o no responde, el backend responde por su cuenta (reglas locales más simples) sin error. Contrato estable (los cambios son solo aditivos):

| Cliente -> backend | Se reenvía a la IA | Lo que recibe el cliente |
| --- | --- | --- |
| `POST /api/auth/guest` `{"name":"Ana"}` | no | `{"token": "<JWT>", "user": {...}}` |
| STOMP `SEND /app/pose` (con `Authorization: Bearer <JWT>` en `CONNECT`) | `POST /analyze` | mensajes en `/topic/feedback.{session_id}` |
| `POST /api/coach` `{"question", "profile"?, "history"?}` con `Authorization: Bearer` | `POST /coach` | `{"answer": "...", "source": "claude" o "rules"}` |
| `POST /api/sessions` `{exercise, reps, accuracy, durationSeconds, avgAngles}` | no | `201` con el resumen guardado |

Frame por STOMP (misma forma que `/analyze`):

```json
{"exercise": "squat", "session_id": "partida-42", "frame_width": 1280, "frame_height": 720,
 "points": [{"name": "left_hip", "x": 0.5, "y": 0.3, "confidence": 0.9},
            {"name": "left_knee", "x": 0.5, "y": 0.55, "confidence": 0.9},
            {"name": "left_ankle", "x": 0.4, "y": 0.8, "confidence": 0.9}]}
```

Mensaje que llega a `/topic/feedback.partida-42`:

```json
{"type": "feedback", "score": 88.1, "message": "Buena posición: mantén el control y respira de forma constante.",
 "feedback": ["Buena posición: mantén el control y respira de forma constante."],
 "phase": "bottom", "reps": 2, "state": "WARMUP", "exercise": "squat"}
```

Garantías de la IA que los clientes pueden asumir: `/analyze` y `/coach` no devuelven 500 (un fallo interno da un frame no analizado o una respuesta por reglas), `message` es una sola frase, `source` solo vale `claude` o `rules`, y los campos nuevos se añaden como opcionales. `tests/test_contract.py` fija estos campos y valores.

Verificado de punta a punta en local (backend Spring Boot con PostgreSQL + esta IA): con el secreto compartido distinto la IA responde `401`, el backend avisa en su log (sin imprimir el secreto) y contesta por reglas durante 5 s; con la IA apagada el backend responde en menos de 100 ms con su fallback; con un proveedor de modelo que no conecta, la IA contesta por reglas tras como máximo 1,5 s y el disyuntor evita esperar de nuevo durante 60 s. En Windows, `http://localhost:8000` suma ~200 ms por conexión nueva (resuelve `::1` primero): usa `127.0.0.1`. El `ConnectionResetError ... _ProactorBasePipeTransport` de uvicorn en Windows es ruido de cierre de conexión y no ocurre en el contenedor Linux.

## Autenticación

Con `AI_SHARED_SECRET` definida, cada petición a `/analyze` y `/coach` debe enviar la cabecera `X-Internal-Token` con el mismo valor. La comparación es de tiempo constante (`hmac.compare_digest`). Sin cabecera o con un valor incorrecto la respuesta es `401` con `{"detail":"unauthorized"}`. El backend Java la envía en cada llamada.

## CORS

No hay CORS habilitado: el servicio solo lo llama el backend (servidor a servidor) y el navegador no debe llamarlo directamente. Un origen de navegador no recibe `Access-Control-Allow-Origin`.

## Despliegue

`render.yaml` define un servicio Docker con `healthCheckPath: /health`. Configura `ANTHROPIC_API_KEY` y `AI_SHARED_SECRET` en el panel de Render (ambas con `sync: false`); no se guardan en el repositorio. `AI_SHARED_SECRET` debe ser idéntica a la del backend.
