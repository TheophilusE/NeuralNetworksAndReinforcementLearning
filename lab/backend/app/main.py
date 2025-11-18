from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .simulation import PendulumSimulator
from .ode_env import OdeCartPole
from .models import StartMessage
from .controllers import PIDController, NNController, TorchNNPolicy
import numpy as np
from .trainer import ESTrainer
from . import es_worker
import asyncio
import json
from pathlib import Path
import os
import threading
import time
from .thread_worker import submit_task
import uuid

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
    # unique id for this websocket client connection so messages can be
    # attributed to the correct client when multiple clients are connected
    client_id = str(uuid.uuid4())

    sim = None
    controller = None
    trainer = None
    stats_poller = None
    # per-connection thread/sim control handles to prevent duplicate sims
    _sim_stop_event = None
    _sim_thread_future = None
    _sim_send_queue = None
    _sim_drain_task = None
    policies_dir = Path(__file__).resolve().parent.parent / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)
    # Auto-start a simulator when a client connects so the frontend receives
    # scene updates without requiring an explicit "start" message.
    try:
        # Create a single short-lived simulator to provide an initial
        # scene snapshot for the frontend. Avoid creating multiple sims
        # here (they previously leaked and produced duplicate visuals).
        try:
            try:
                # prefer the ODE-based solver when available
                temp_sim = OdeCartPole(mode="single", dt=0.02)
                engine = "ode"
            except Exception:
                temp_sim = PendulumSimulator(mode="single", dt=0.02)
                engine = "simple"

            # try to reset to sensible defaults for snapshot
            try:
                if hasattr(temp_sim, 'reset'):
                    temp_sim.reset()
            except Exception:
                pass
        except Exception:
            temp_sim = None
            engine = "simple"

        # Helper to start an ESTrainer for a given controller if it looks NN-like
        # trainer_params is a mutable holder so the UI can update hyperparameters live.
        trainer_params = {"population": 12, "sigma": 0.08, "alpha": 0.04, "steps": 100, "n_workers": 4}

        def start_trainer_for(controller_obj, sim_obj, start_msg_local):
            nonlocal trainer, stats_poller, trainer_params
            try:
                if controller_obj is None or not hasattr(controller_obj, 'get_flat_params'):
                    return
                # avoid starting duplicate trainer
                if trainer and trainer.running:
                    return

                policy_kind = 'torch' if isinstance(controller_obj, TorchNNPolicy) else 'numpy'
                if hasattr(controller_obj, 'sizes'):
                    hidden = tuple(controller_obj.sizes[1:-1])
                else:
                    hidden = (32, 32)

                # evaluator reads trainer_params['steps'] so updates can take effect live
                def evaluator(flat: np.ndarray) -> float:
                    return es_worker.evaluate_params(
                        flat,
                        policy_kind,
                        {'hidden_sizes': hidden},
                        {'engine': engine, 'mode': start_msg_local.mode, 'dt': start_msg_local.dt, 'track_length': getattr(sim_obj, 'track_length', 2.0)},
                        steps=trainer_params.get('steps', 100),
                    )

                def policy_setter(new_theta):
                    if new_theta is None:
                        return controller_obj.get_flat_params()
                    else:
                        controller_obj.set_flat_params(np.array(new_theta))

                dim = controller_obj.num_params()
                # Create trainer using values from trainer_params
                trainer = ESTrainer(
                    evaluator,
                    policy_setter,
                    dim=dim,
                    population=trainer_params.get('population', 12),
                    sigma=trainer_params.get('sigma', 0.08),
                    alpha=trainer_params.get('alpha', 0.04),
                    n_workers=trainer_params.get('n_workers', 4),
                )
                trainer.start()

                async def poll_stats():
                    try:
                        while trainer and trainer.running:
                            await asyncio.sleep(0.5)
                            try:
                                stats = dict(trainer.stats)
                                stats['running'] = bool(trainer.running)
                                await ws.send_text(json.dumps({"training_stats": stats, "client_id": client_id}))
                            except Exception:
                                break
                    except asyncio.CancelledError:
                        pass

                stats_poller = asyncio.create_task(poll_stats())
            except Exception:
                pass

        # Best-effort: align initial state for snapshot
        try:
            if temp_sim is not None and hasattr(temp_sim, 'theta'):
                temp_sim.theta = 0.2
        except Exception:
            pass

        # Send initial scene from temp_sim so the frontend can render a starting scene.
        if temp_sim is not None:
            start_msg = StartMessage(action="start", mode=temp_sim.mode, controller="pid", dt=temp_sim.dt, target=0.0)
        else:
            start_msg = StartMessage(action="start", mode="single", controller="pid", dt=0.02, target=0.0)
        # per-connection reset counter to tag scene resets and session id for scene messages
        reset_counter = 0
        session_id = 0
        try:
            if temp_sim is not None and hasattr(temp_sim, 'get_scene_tree'):
                scene_tree = temp_sim.get_scene_tree()
                await ws.send_text(json.dumps({"scene": scene_tree, "track_length": getattr(temp_sim, 'track_length', None), "session_id": session_id, "client_id": client_id}))
        except Exception:
            pass
        # Close temporary simulator used only for initial scene snapshot so it
        # doesn't linger and produce unexpected background activity.
        try:
            if temp_sim is not None and hasattr(temp_sim, 'close'):
                temp_sim.close()
        except Exception:
            pass
        # NOTE: do not auto-start a trainer here — trainer will be started
        # when the client explicitly requests NN controller via the `start`
        # action. Starting a trainer on connect and again on `start` caused
        # duplicate trainers and duplicate training_stats streams.
    except Exception:
        # If auto-start fails, continue and allow explicit start messages
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
                # stop any existing simulator first
                if sim:
                    try:
                        sim.running = False
                    except Exception:
                        pass
                start = StartMessage(**msg)
                engine = msg.get("engine", "simple")
                # initialize simulator (choose ode or simple); respect optional track_length
                requested_track = getattr(start, 'track_length', None)
                if engine == "ode":
                    # allow overriding gravity via start message
                    gravity = msg.get('gravity', None)
                    if requested_track is not None:
                        if gravity is not None:
                            sim = OdeCartPole(mode=start.mode, dt=start.dt, gui=msg.get("gui", False), track_length=requested_track, gravity=gravity)
                        else:
                            sim = OdeCartPole(mode=start.mode, dt=start.dt, gui=msg.get("gui", False), track_length=requested_track)
                    else:
                        if gravity is not None:
                            sim = OdeCartPole(mode=start.mode, dt=start.dt, gui=msg.get("gui", False), gravity=gravity)
                        else:
                            sim = OdeCartPole(mode=start.mode, dt=start.dt, gui=msg.get("gui", False))
                else:
                    gravity = msg.get('gravity', None)
                    if requested_track is not None:
                        if gravity is not None:
                            sim = PendulumSimulator(mode=start.mode, dt=start.dt, track_length=requested_track, gravity=gravity)
                        else:
                            sim = PendulumSimulator(mode=start.mode, dt=start.dt, track_length=requested_track)
                    else:
                        if gravity is not None:
                            sim = PendulumSimulator(mode=start.mode, dt=start.dt, gravity=gravity)
                        else:
                            sim = PendulumSimulator(mode=start.mode, dt=start.dt)

                # increment session id for this new simulator instance so the
                # frontend can ignore stale scene messages belonging to prior
                # sessions. Do this before resetting so reset messages carry
                # the new session id.
                try:
                    session_id += 1
                except Exception:
                    session_id = (session_id or 0) + 1

                # ensure sim is reset to a clean start state
                try:
                    if hasattr(sim, 'reset'):
                        sim.reset()
                        # notify frontend to clear existing scene objects for a clean restart
                        try:
                            reset_counter += 1
                            await ws.send_text(json.dumps({"scene_reset": True, "reset_id": reset_counter, "session_id": session_id, "client_id": client_id}))
                        except Exception:
                            pass
                except Exception:
                    pass

                # stop any existing trainer; if the new controller is NN we'll start a fresh trainer below
                if trainer:
                    try:
                        trainer.stop()
                    except Exception:
                        pass
                    trainer = None
                if stats_poller:
                    try:
                        stats_poller.cancel()
                    except Exception:
                        pass
                    stats_poller = None

                if start.controller == "pid":
                    # create PID controller and apply any provided initial gains
                    controller = PIDController(kp=30.0, ki=0.0, kd=2.0)
                    try:
                        if getattr(start, 'kp', None) is not None or getattr(start, 'ki', None) is not None or getattr(start, 'kd', None) is not None:
                            controller.set_params(kp=start.kp, ki=start.ki, kd=start.kd)
                    except Exception:
                        pass
                try:
                    if hasattr(controller, 'reset'):
                        controller.reset()
                except Exception:
                    pass
                else:
                    # allow choosing torch or numpy NN implementation
                    nn_framework = msg.get('nn_framework', 'numpy')
                    if nn_framework == 'torch':
                        controller = TorchNNPolicy()
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

                try:
                    if hasattr(controller, 'reset'):
                        controller.reset()
                except Exception:
                    pass

                # run simulation loop in a pooled worker thread so the asyncio
                # loop is not blocked; the thread will push telemetry into an
                # asyncio.Queue via the event loop and a small drain task will
                # forward messages to the websocket.
                try:
                    # stop any previous threaded sim for this client
                    try:
                        # signal any existing sim thread to stop and attempt to
                        # cancel the future so it doesn't continue running.
                        if _sim_stop_event is not None:
                            _sim_stop_event.set()
                        if _sim_thread_future is not None:
                            try:
                                _sim_thread_future.cancel()
                            except Exception:
                                pass
                        # cancel any existing drain task so it doesn't keep
                        # sending from an old queue
                        if _sim_drain_task is not None and not _sim_drain_task.done():
                            try:
                                _sim_drain_task.cancel()
                            except Exception:
                                pass
                        # also set any previously running sim to not running
                        try:
                            if sim is not None:
                                sim.running = False
                        except Exception:
                            pass
                    except Exception:
                        pass

                    _sim_send_queue = asyncio.Queue()
                    _sim_stop_event = threading.Event()

                    def _run_sim_thread(sim_obj, controller_obj, start_msg, loop, send_queue, stop_event, sim_session_id, client_id_local):
                        sim_obj.running = True
                        t_local = 0.0
                        try:
                            while sim_obj.running and not stop_event.is_set():
                                state = sim_obj.get_state()
                                try:
                                    torque = controller_obj.get_torque(state, target=start_msg.target)
                                except Exception:
                                    torque = 0.0
                                try:
                                    sim_obj.step(torque)
                                except Exception:
                                    pass
                                t_local += getattr(sim_obj, 'dt', 0.02)
                                msg = {
                                    't': t_local,
                                    'state': sim_obj.get_state(),
                                    'controller': 'nn' if isinstance(controller_obj, NNController) else 'pid',
                                    'session_id': sim_session_id,
                                    'client_id': client_id_local,
                                }
                                if hasattr(sim_obj, 'get_scene_tree'):
                                    try:
                                        msg['scene'] = sim_obj.get_scene_tree()
                                    except Exception:
                                        pass
                                try:
                                    loop.call_soon_threadsafe(send_queue.put_nowait, msg)
                                except Exception:
                                    break
                                time.sleep(getattr(sim_obj, 'dt', 0.02))
                        finally:
                            try:
                                if hasattr(sim_obj, 'close'):
                                    sim_obj.close()
                            except Exception:
                                pass

                    # capture the session id for this sim thread so messages remain
                    # tagged with the session they belong to even if session_id
                    # increments later for a new sim instance.
                    main_loop = asyncio.get_running_loop()
                    sim_session = session_id
                    _sim_thread_future = submit_task(_run_sim_thread, sim, controller, start, main_loop, _sim_send_queue, _sim_stop_event, sim_session, client_id)

                    async def _drain_queue_and_send(ws_obj, q: asyncio.Queue):
                        try:
                            while True:
                                item = await q.get()
                                try:
                                    await ws_obj.send_text(json.dumps(item))
                                except Exception:
                                    break
                        except asyncio.CancelledError:
                            pass

                    _sim_drain_task = asyncio.create_task(_drain_queue_and_send(ws, _sim_send_queue))
                except Exception:
                    # fallback to coroutine-based simulation loop; pass session
                    # id so scene messages from this coroutine are tagged.
                    try:
                        sim_session = session_id
                    except Exception:
                        sim_session = None
                    # ensure any previous coroutine-run sim is signalled to stop
                    try:
                        if _sim_stop_event is not None:
                            _sim_stop_event.set()
                    except Exception:
                        pass
                    asyncio.create_task(run_sim(ws, sim, controller, start, sim_session, client_id))
                # send an initial scene description to the client so the frontend can
                # replicate the scene tree (bodies, links, visuals) if provided
                try:
                    if hasattr(sim, 'get_scene_tree'):
                        await ws.send_text(json.dumps({"scene": sim.get_scene_tree(), "session_id": session_id, "client_id": client_id}))
                except Exception as e:
                    await ws.send_text(json.dumps({"scene_error": str(e)}))
                # If the chosen controller is NN-like, start trainer automatically
                try:
                    start_trainer_for(controller, sim, start)
                except Exception:
                    pass
            elif action == "stop":
                if sim:
                    sim.running = False
                if trainer:
                    trainer.stop()
                    trainer = None
                if stats_poller:
                    stats_poller.cancel()
                    stats_poller = None
            elif action == "set_track":
                # runtime update of the track length for the active simulator
                try:
                    track_val = msg.get('track_length')
                    if track_val is None:
                        await ws.send_text(json.dumps({"error": "set_track requires 'track_length'"}))
                        continue
                    track_val = float(track_val)
                    if sim is None:
                        await ws.send_text(json.dumps({"error": "no active simulator to set track on"}))
                        continue
                    # set attribute if available
                    try:
                        setattr(sim, 'track_length', float(track_val))
                    except Exception:
                        pass
                    await ws.send_text(json.dumps({"track_set": float(track_val)}))
                except Exception as e:
                    await ws.send_text(json.dumps({"error": str(e)}))
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
                name = msg.get("name", "default")
                if isinstance(controller, NNController):
                    path = policies_dir / f"{name}.npz"
                    try:
                        controller.save(str(path))
                        await ws.send_text(json.dumps({"policy_saved": str(path)}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"policy_save_error": str(e)}))
                elif isinstance(controller, TorchNNPolicy):
                    path = policies_dir / f"{name}.pt"
                    try:
                        controller.save(str(path))
                        await ws.send_text(json.dumps({"policy_saved": str(path)}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"policy_save_error": str(e)}))
                else:
                    await ws.send_text(json.dumps({"error": "no NN controller to save"}))
            elif action == "load_policy":
                name = msg.get("name", "default")
                if isinstance(controller, NNController):
                    path = policies_dir / f"{name}.npz"
                    try:
                        controller.load(str(path))
                        # reset sim and controller after loading a policy to ensure clean start
                        try:
                            if sim is not None and hasattr(sim, 'reset'):
                                # bump session id to mark a fresh scene session
                                try:
                                    session_id += 1
                                except Exception:
                                    session_id = (session_id or 0) + 1
                                sim.reset()
                                try:
                                    reset_counter += 1
                                    await ws.send_text(json.dumps({"scene_reset": True, "reset_id": reset_counter, "session_id": session_id, "client_id": client_id}))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            if hasattr(controller, 'reset'):
                                controller.reset()
                        except Exception:
                            pass
                        await ws.send_text(json.dumps({"policy_loaded": str(path)}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"policy_load_error": str(e)}))
                elif isinstance(controller, TorchNNPolicy):
                    path = policies_dir / f"{name}.pt"
                    try:
                        controller.load(str(path))
                        try:
                            if sim is not None and hasattr(sim, 'reset'):
                                # bump session id to mark a fresh scene session
                                try:
                                    session_id += 1
                                except Exception:
                                    session_id = (session_id or 0) + 1
                                sim.reset()
                                try:
                                    reset_counter += 1
                                    await ws.send_text(json.dumps({"scene_reset": True, "reset_id": reset_counter, "session_id": session_id, "client_id": client_id}))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            if hasattr(controller, 'reset'):
                                controller.reset()
                        except Exception:
                            pass
                        await ws.send_text(json.dumps({"policy_loaded": str(path)}))
                    except Exception as e:
                        await ws.send_text(json.dumps({"policy_load_error": str(e)}))
                else:
                    await ws.send_text(json.dumps({"error": "no NN controller to load"}))
            elif action == "set_trainer_params":
                # Update trainer hyperparameters live. If a trainer is running,
                # update its attributes and (if needed) restart worker pool.
                params = msg.get('params', {}) or {}
                # allowed keys: population, sigma, alpha, steps, n_workers
                try:
                    # coerce numeric values
                    for k in ('population', 'n_workers'):
                        if k in params:
                            trainer_params[k] = int(params[k])
                    for k in ('sigma', 'alpha'):
                        if k in params:
                            trainer_params[k] = float(params[k])
                    if 'steps' in params:
                        trainer_params['steps'] = int(params['steps'])

                    # If a trainer is running, stop it and restart with new params
                    if trainer:
                        try:
                            trainer.stop()
                        except Exception:
                            pass
                        trainer = None
                    if stats_poller:
                        try:
                            stats_poller.cancel()
                        except Exception:
                            pass
                        stats_poller = None

                    # update the trainer_params and create a fresh trainer
                    try:
                        # Build a minimal 'start' object with mode/dt for the trainer starter
                        class _StartLike:
                            pass

                        start_like = _StartLike()
                        start_like.mode = getattr(sim, 'mode', 'single') if sim is not None else 'single'
                        start_like.dt = getattr(sim, 'dt', 0.02) if sim is not None else 0.02

                        # start_trainer_for will create a new trainer using trainer_params
                        try:
                            start_trainer_for(controller, sim, start_like)
                        except Exception:
                            pass
                    except Exception:
                        pass

                    await ws.send_text(json.dumps({"trainer_params": trainer_params}))
                except Exception as e:
                    await ws.send_text(json.dumps({"error": f"invalid trainer params: {e}"}))
            elif action == "set_gravity":
                # runtime update gravity for the active simulator
                try:
                    gravity_val = msg.get('gravity')
                    if gravity_val is None:
                        await ws.send_text(json.dumps({"error": "set_gravity requires 'gravity'"}))
                        continue
                    gravity_val = float(gravity_val)
                    if sim is None:
                        await ws.send_text(json.dumps({"error": "no active simulator to set gravity on"}))
                        continue
                    try:
                        setattr(sim, 'g', float(gravity_val))
                    except Exception:
                        pass
                    await ws.send_text(json.dumps({"gravity_set": float(gravity_val)}))
                except Exception as e:
                    await ws.send_text(json.dumps({"error": str(e)}))
            else:
                await ws.send_text(json.dumps({"error": "unknown action"}))

    except WebSocketDisconnect:
        try:
            if sim:
                sim.running = False
        except Exception:
            pass
        try:
            if _sim_stop_event is not None:
                _sim_stop_event.set()
        except Exception:
            pass
        try:
            if _sim_drain_task is not None and not _sim_drain_task.done():
                _sim_drain_task.cancel()
        except Exception:
            pass


async def run_sim(ws: WebSocket, sim, controller, start, sim_session_id=None, client_id=None):
    sim.running = True
    t = 0.0
    try:
        while sim.running:
            # compute control
            state = sim.get_state()
            torque = controller.get_torque(state, target=start.target)
            sim.step(torque)
            t += sim.dt
            msg = {
                "t": t,
                "state": sim.get_state(),
                "controller": "nn" if isinstance(controller, NNController) else "pid",
                "session_id": sim_session_id,
                "client_id": client_id,
            }
            # include scene updates whenever the simulator exposes a scene tree
            if hasattr(sim, 'get_scene_tree'):
                try:
                    msg['scene'] = sim.get_scene_tree()
                except Exception:
                    # Log scene serialization error for debugging and continue
                    try:
                        print(f"Scene serialization error", flush=True)
                    except Exception:
                        pass
                    # continue without scene
                    pass
            await ws.send_text(json.dumps(msg))
            await asyncio.sleep(sim.dt)
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))


async def run_contest(ws: WebSocket, sim_a, sim_b, ctrl_a, ctrl_b, target=0.0):
    """Run two simulators/controllers in parallel and stream combined telemetry."""
    sim_a.running = True
    sim_b.running = True
    t = 0.0
    try:
        while getattr(sim_a, 'running', True) and getattr(sim_b, 'running', True):
            # compute controls for each
            state_a = sim_a.get_state()
            state_b = sim_b.get_state()
            try:
                tau_a = ctrl_a.get_torque(state_a, target=target)
            except Exception:
                tau_a = 0.0
            try:
                tau_b = ctrl_b.get_torque(state_b, target=target)
            except Exception:
                tau_b = 0.0

            # step sims
            try:
                sim_a.step(tau_a)
            except Exception:
                pass
            try:
                sim_b.step(tau_b)
            except Exception:
                pass

            t += getattr(sim_a, 'dt', getattr(sim_b, 'dt', 0.02))

            msg = {
                "t": t,
                "a": {"state": sim_a.get_state(), "tau": float(tau_a)},
                "b": {"state": sim_b.get_state(), "tau": float(tau_b)},
            }
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                # If sending fails, stop contest
                break

            await asyncio.sleep(getattr(sim_a, 'dt', getattr(sim_b, 'dt', 0.02)))
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
