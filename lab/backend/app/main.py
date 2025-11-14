from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .simulation import PendulumSimulator
from .pybullet_env import PyBulletPendulum
from .models import StartMessage
from .controllers import PIDController, NNController
from .trainer import ESTrainer
import asyncio
import json
from pathlib import Path
import os

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
    trainer = None
    stats_poller = None
    policies_dir = Path(__file__).resolve().parent.parent / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)
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
                engine = msg.get("engine", "simple")
                # initialize simulator (choose pybullet or simple)
                if engine == "pybullet":
                    sim = PyBulletPendulum(mode=start.mode, dt=start.dt, gui=msg.get("gui", False))
                else:
                    sim = PendulumSimulator(mode=start.mode, dt=start.dt)

                if start.controller == "pid":
                    controller = PIDController(kp=30.0, ki=0.0, kd=2.0)
                else:
                    controller = NNController()
                    # try loading default policy if exists
                    default_path = policies_dir / "default.npz"
                    if default_path.exists():
                        try:
                            controller.load(str(default_path))
                            await ws.send_text(json.dumps({"policy_loaded": str(default_path)}))
                        except Exception as e:
                            await ws.send_text(json.dumps({"policy_load_error": str(e)}))

                # run simulation loop in background
                asyncio.create_task(run_sim(ws, sim, controller, start))
            elif action == "stop":
                if sim:
                    sim.running = False
                if trainer:
                    trainer.stop()
                    trainer = None
                if stats_poller:
                    stats_poller.cancel()
                    stats_poller = None
            elif action == "control":
                # update controller parameters at runtime
                ctype = msg.get("type")
                params = msg.get("params", {})
                if ctype == "pid" and isinstance(controller, PIDController):
                    controller.set_params(kp=params.get("kp"), ki=params.get("ki"), kd=params.get("kd"))
                    await ws.send_text(json.dumps({"control": "pid_updated", "params": {"kp": controller.kp, "ki": controller.ki, "kd": controller.kd}}))
                else:
                    await ws.send_text(json.dumps({"error": "control update unsupported or controller mismatch"}))
            elif action == "save_policy":
                if isinstance(controller, NNController):
                    name = msg.get("name", "default")
                    path = policies_dir / f"{name}.npz"
                    try:
                        controller.save(str(path))
                        await ws.send_text(json.dumps({"policy_saved": str(path)}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"policy_save_error": str(e)}))
                else:
                    await ws.send_text(json.dumps({"error": "no NN controller to save"}))
            elif action == "load_policy":
                if isinstance(controller, NNController):
                    name = msg.get("name", "default")
                    path = policies_dir / f"{name}.npz"
                    try:
                        controller.load(str(path))
                        await ws.send_text(json.dumps({"policy_loaded": str(path)}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"policy_load_error": str(e)}))
            elif action == "train_start":
                # start online ES trainer using the current controller and sim
                if controller is None or not isinstance(controller, NNController):
                    await ws.send_text(json.dumps({"error": "controller must be NN to train"}))
                else:
                    # rollout uses a short episode on the same sim class
                    def rollout():
                        # make a temporary sim copy if possible; here we reuse sim for speed
                        # perform a short episode
                        total = 0.0
                        steps = 100
                        # reset sim if it has a reset method, otherwise continue
                        for _ in range(steps):
                            s = sim.get_state()
                            a = controller.get_torque(s, target=0.0)
                            sim.step(a)
                            # reward: negative absolute angle of first link
                            if "theta" in s:
                                total -= abs(s["theta"]) 
                            else:
                                total -= abs(s.get("th1", 0.0))
                        return total

                    trainer = ESTrainer(controller, rollout_fn=rollout, population=8, sigma=0.08, alpha=0.03)
                    trainer.start()
                    await ws.send_text(json.dumps({"training": "started"}))

                    # start an asyncio task to poll trainer.stats and forward to client
                    async def poll_stats():
                        try:
                            while trainer and trainer.running:
                                await asyncio.sleep(0.5)
                                try:
                                    await ws.send_text(json.dumps({"training_stats": trainer.stats}))
                                except Exception:
                                    break
                        except asyncio.CancelledError:
                            pass

                    stats_poller = asyncio.create_task(poll_stats())
            elif action == "train_stop":
                if trainer:
                    trainer.stop()
                    trainer = None
                    await ws.send_text(json.dumps({"training": "stopped"}))
                if stats_poller:
                    stats_poller.cancel()
                    stats_poller = None
            else:
                await ws.send_text(json.dumps({"error": "unknown action"}))

    except WebSocketDisconnect:
        if sim:
            sim.running = False


async def run_sim(ws: WebSocket, sim, controller, start):
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
                "controller": "nn" if isinstance(controller, NNController) else "pid",
            }))
            await asyncio.sleep(sim.dt)
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))
