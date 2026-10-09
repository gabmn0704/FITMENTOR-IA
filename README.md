# FitMentor IA Service

FastAPI service for exercise-specific pose scoring from normalized body landmarks. It validates landmark confidence, computes joint angles with frame aspect ratio, evaluates body alignment, and can return multiple coaching corrections.

## Development

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

## API

- `GET /health`
- `POST /analyze`

Supported exercises are `squat`, `push_up`, `plank`, and `deadlift`. Points use normalized `x` and `y` coordinates and can be named with generic joints or side-specific names such as `left_hip`. Supply `frame_width` and `frame_height` when available to preserve angle geometry.

The service analyzes landmarks; it does not detect a person directly from image or video data. The frontend currently runs MediaPipe pose detection in the browser.