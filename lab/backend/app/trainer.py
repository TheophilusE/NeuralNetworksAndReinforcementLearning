import numpy as np
import threading
import time
from typing import Optional


class ESTrainer:
    """A simple Evolution Strategies trainer for a numpy-parameterized policy.

    This trainer perturbs parameters, evaluates them in the provided `rollout_fn`,
    and updates the policy parameters in the direction of higher returns.
    """

    def __init__(self, policy, rollout_fn, population=12, sigma=0.1, alpha=0.01):
        self.policy = policy
        self.rollout_fn = rollout_fn
        self.population = population
        self.sigma = sigma
        self.alpha = alpha
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.stats = {"iter": 0, "last_reward": None}

    def _step_once(self):
        theta = self.policy.get_flat_params()
        N = self.population
        dim = theta.size
        eps = np.random.randn(N, dim)
        rewards = np.zeros(N)
        for i in range(N):
            self.policy.set_flat_params(theta + self.sigma * eps[i])
            rewards[i] = self.rollout_fn()
        # standardize rewards
        A = (rewards - rewards.mean())
        if rewards.std() > 1e-8:
            A /= rewards.std()
        grad = np.dot(A, eps) / N
        theta = theta + self.alpha / (self.sigma) * grad
        self.policy.set_flat_params(theta)
        self.stats["iter"] += 1
        self.stats["last_reward"] = float(rewards.mean())

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
