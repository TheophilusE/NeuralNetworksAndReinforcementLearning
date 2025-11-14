
NN & RL Lab
============

This folder contains an interactive demo with a FastAPI backend and a TypeScript
frontend (Vite + React). The demo demonstrates control of single and double
pendulums using a PID controller and a small neural-network-based controller
stub. It's intentionally modular so you can extend the physics (e.g. use
PyBullet), swap in trained neural networks, or add RL agents.

Structure
---------
- `lab/backend` - FastAPI backend (WebSocket-based simulator)
- `lab/frontend` - Vite + React frontend (TypeScript)

Quick start
-----------

1) Start the backend (from `lab/backend`):

```bash
python -m pip install -r lab/backend/requirements.txt
cd lab/backend
uvicorn app.main:app --reload --port 8000
```

2) Start the frontend (from `lab/frontend`):

```bash
cd lab/frontend
npm install
npm run dev
```
