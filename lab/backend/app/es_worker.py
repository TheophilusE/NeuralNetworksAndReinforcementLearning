import numpy as np
from .controllers import NNController, TorchNNPolicy
from .ode_env import OdeCartPole
from .simulation import PendulumSimulator
# thread_worker.submit_task not used here; evaluator runs inline in worker processes


def evaluate_params(flat_params: np.ndarray, policy_kind: str, policy_kwargs: dict, sim_kwargs: dict, steps: int = 100) -> float:
    """Evaluate a flattened parameter vector by running a short rollout in a fresh simulator.

    This function dispatches the actual rollout to the central thread
    worker so that callers can offload CPU-bound evaluations without
    blocking the main thread. The call remains synchronous from the
    caller's perspective (the result is waited on) but the work runs
    on a pooled thread.

    Returns a scalar reward (higher is better).
    """

    def _run_eval() -> float:
        # reconstruct policy
        if policy_kind == 'torch':
            policy = TorchNNPolicy(**policy_kwargs)
            # torch model expects state_dict load, but we use flat params method
            policy.set_flat_params(flat_params)
        else:
            policy = NNController(**policy_kwargs)
            policy.set_flat_params(flat_params)

        engine = sim_kwargs.get('engine', 'simple')
        mode = sim_kwargs.get('mode', 'single')
        dt = sim_kwargs.get('dt', 0.02)
        # Prefer OdeCartPole when requested; otherwise use the simple PendulumSimulator.
        if engine == 'ode':
            sim = OdeCartPole(mode=mode, dt=dt, gui=False, track_length=sim_kwargs.get('track_length', 2.0))
        else:
            sim = PendulumSimulator(mode=mode, dt=dt, track_length=sim_kwargs.get('track_length', 2.0))

        total = 0.0
        for _ in range(steps):
            s = sim.get_state()
            a = policy.get_torque(s, target=0.0, dt=dt)
            sim.step(a)
            if 'theta' in s:
                total -= abs(s['theta'])
            else:
                total -= abs(s.get('th1', 0.0))

        # close sim if has close
        if hasattr(sim, 'close'):
            try:
                sim.close()
            except Exception:
                pass

        return float(total)

    # Execute the evaluation inline. In typical usage the caller (ESTrainer)
    # will invoke `evaluate_params` inside worker processes (multiprocessing.Pool),
    # so submitting again to a thread pool is unnecessary overhead. Run directly.
    return float(_run_eval())
