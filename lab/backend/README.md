FastAPI backend for the NN & RL Lab demo

Run instructions (recommended inside a virtualenv):

1. Install requirements:

```bash
python -m pip install -r requirements.txt
```

2. Start the server (from `lab/backend`):

```bash
uvicorn app.main:app --reload --port 8000
```

The server exposes:
- `GET /health` for a quick healthcheck
- `WebSocket /ws` which accepts JSON messages to `start` and `stop` simulations.

Example start message (JSON over websocket):

```json
{
  "action": "start",
  "mode": "single",
  "controller": "pid",
  "dt": 0.02,
  "target": 0.0
}
```

The websocket will stream back periodic JSON state updates.

```
$ uvicorn app.main:app --reload --host localhost --port 8000
```
