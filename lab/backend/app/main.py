from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .simulation import PendulumSimulator
from .models import StartMessage
from .controllers import PIDController, NNController
import asyncio
import json

app = FastAPI(title="NN & RL Lab Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    sim = None
    controller = None
    try:
        while True:
            msg_text = await ws.receive_text()
            try:
                msg = json.loads(msg_text)
            except Exception:
                await ws.send_text(json.dumps({"error": "invalid json"}))
                continue

            action = msg.get("action")
            if action == "start":
                start = StartMessage(**msg)
                # initialize simulator
                sim = PendulumSimulator(mode=start.mode, dt=start.dt)
                if start.controller == "pid":
                    controller = PIDController(kp=30.0, ki=0.0, kd=2.0)
                else:
                    controller = NNController()

                # run simulation loop in background
                asyncio.create_task(run_sim(ws, sim, controller, start))
            elif action == "stop":
                if sim:
                    sim.running = False
            else:
                await ws.send_text(json.dumps({"error": "unknown action"}))

    except WebSocketDisconnect:
        if sim:
            sim.running = False


async def run_sim(ws: WebSocket, sim: PendulumSimulator, controller, start):
    sim.running = True
    t = 0.0
    try:
        while sim.running:
            # compute control
            state = sim.get_state()
            torque = controller.get_torque(state, target=start.target)
            sim.step(torque)
            t += sim.dt
            await ws.send_text(json.dumps({
                "t": t,
                "state": sim.get_state(),
            }))
            await asyncio.sleep(sim.dt)
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))
