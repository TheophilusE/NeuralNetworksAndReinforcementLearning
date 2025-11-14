import numpy as np
import threading
import time
from typing import Optional, Callable
import multiprocessing as mp


class ESTrainer:
    """A simple Evolution Strategies trainer with optional parallel rollouts.

    Instead of requiring a policy object that can run rollouts in-process, the
    trainer accepts an `evaluator` callable with signature `evaluator(theta: np.ndarray) -> float`.
    This allows the main process to spawn worker processes to evaluate perturbed
    parameter vectors in parallel.
    """

    def __init__(self, evaluator: Callable[[np.ndarray], float], policy_setter: Callable[[np.ndarray], None],
                 dim: int, population=12, sigma=0.1, alpha=0.01, n_workers: Optional[int] = None):
        self.evaluator = evaluator
        self.policy_setter = policy_setter
        self.dim = dim
        self.population = population
        self.sigma = sigma
        self.alpha = alpha
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.stats = {"iter": 0, "last_reward": None, "history": []}
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.pool: Optional[mp.Pool] = None

    def _step_once(self):
        # get current parameters from the policy via a call to policy_getter through policy_setter trick
        # The caller is expected to provide policy_setter and ensure current params are available externally.
        theta = self.policy_setter(None)
        theta = np.array(theta)
        N = self.population
        dim = self.dim
        eps = np.random.randn(N, dim)

        # prepare parameter vectors for evaluation
        thetas = [theta + self.sigma * eps[i] for i in range(N)]

        # evaluate in parallel
        if self.n_workers > 1:
            if self.pool is None:
                self.pool = mp.Pool(processes=min(self.n_workers, N))
            rewards = self.pool.map(self.evaluator, thetas)
        else:
            rewards = [self.evaluator(t) for t in thetas]

        rewards = np.array(rewards)
        A = (rewards - rewards.mean())
        if rewards.std() > 1e-8:
            A /= rewards.std()
        grad = np.dot(A, eps) / N
        theta = theta + self.alpha / (self.sigma) * grad

        # set updated parameters back to the policy
        self.policy_setter(theta)

        self.stats["iter"] += 1
        mean_reward = float(rewards.mean())
        self.stats["last_reward"] = mean_reward
        hist = self.stats.get("history", [])
        hist.append(mean_reward)
        if len(hist) > 500:
            hist.pop(0)
        self.stats["history"] = hist

    def start(self, interval=0.1):
        if self.running:
            return
        self.running = True

        def loop():
            while self.running:
                try:
                    self._step_once()
                except Exception:
                    pass
                time.sleep(interval)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.pool:
            try:
                self.pool.close()
                self.pool.join()
            except Exception:
                pass
