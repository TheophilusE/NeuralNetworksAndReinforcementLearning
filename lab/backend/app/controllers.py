import math
import numpy as np
from typing import List, Tuple


class PIDController:
    def __init__(self, kp=10.0, ki=0.0, kd=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = None

    def angle_error(self, target, current):
        diff = target - current
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def get_torque(self, state, target=0.0):
        if "theta" in state:
            err = self.angle_error(target, state["theta"])
            self.integral += err
            deriv = 0.0
            if self.last_error is not None:
                deriv = err - self.last_error
            self.last_error = err
            return self.kp * err + self.ki * self.integral + self.kd * deriv
        else:
            err = self.angle_error(target, state.get("th1", 0.0))
            self.integral += err
            deriv = 0.0
            if self.last_error is not None:
                deriv = err - self.last_error
            self.last_error = err
            return self.kp * err + self.ki * self.integral + self.kd * deriv


class NNController:
    """A small numpy MLP policy with helper methods for parameter access.

    This network is intentionally simple and trained using an evolution-strategy
    trainer (no backprop engine required). The controller maps a single scalar
    (angle error) to a scalar torque.
    """

    def __init__(self, hidden_sizes: Tuple[int, ...] = (16, 16)):
        self.sizes = [1] + list(hidden_sizes) + [1]
        self.weights: List[np.ndarray] = [np.random.randn(a, b) * 0.1 for a, b in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases: List[np.ndarray] = [np.zeros((a,)) for a in self.sizes[1:]]

    def _forward(self, x: np.ndarray) -> float:
        a = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = np.tanh(W.dot(a) + b)
        out = self.weights[-1].dot(a) + self.biases[-1]
        return float(out)

    def get_torque(self, state, target=0.0) -> float:
        if "theta" in state:
            err = np.array([target - state["theta"]])
        else:
            err = np.array([target - state.get("th1", 0.0)])
        return self._forward(err)

    # Parameter helpers for evolutionary updates
    def get_flat_params(self) -> np.ndarray:
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_flat_params(self, flat: np.ndarray):
        i = 0
        new_weights = []
        new_biases = []
        for (out_dim, in_dim) in zip(self.sizes[1:], self.sizes[:-1]):
            w_size = out_dim * in_dim
            W = flat[i : i + w_size].reshape((out_dim, in_dim))
            i += w_size
            b = flat[i : i + out_dim]
            i += out_dim
            new_weights.append(W)
            new_biases.append(b)
        self.weights = new_weights
        self.biases = new_biases

    def num_params(self) -> int:
        return self.get_flat_params().size

