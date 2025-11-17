import numpy as np
from .controllers import NNController, TorchNNPolicy
from .ode_env import OdeCartPole
from .simulation import PendulumSimulator
from .thread_worker import submit_task


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

        if engine == 'pybullet':
            # For backwards compatibility, map pybullet -> OdeCartPole (headless)
            sim = OdeCartPole(mode=mode, dt=dt, gui=False)
        else:
            sim = PendulumSimulator(mode=mode, dt=dt)

        total = 0.0
        for _ in range(steps):
            s = sim.get_state()
            a = policy.get_torque(s, target=0.0)
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

    # Submit to central worker and wait for result
    fut = submit_task(_run_eval)
    return float(fut.result())
