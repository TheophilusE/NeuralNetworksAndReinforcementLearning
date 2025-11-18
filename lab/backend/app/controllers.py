import math
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PIDController:
    def __init__(self, kp=30.0, ki=0.0, kd=2.0, max_output: float = 50.0):
        # sensible defaults chosen to provide stable baseline behavior
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # maximum absolute control output (force applied to cart)
        self.max_output = float(max_output)
        self.integral = 0.0
        self.last_error = None

    def reset(self):
        """Reset controller internal state (integrator, derivative memory)."""
        self.integral = 0.0
        self.last_error = None

    def set_params(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = float(kp)
        if ki is not None:
            self.ki = float(ki)
        if kd is not None:
            self.kd = float(kd)

    def angle_error(self, target, current):
        diff = target - current
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def get_torque(self, state, target=0.0, dt: float = 0.02):
        # Accept an optional dt so the PID integrator/derivative are scaled correctly.
        # Prefer the explicit `dt` argument provided by callers; fall back to state['dt'] if present.
        try:
            dt = float(dt)
        except Exception:
            dt = 0.02
        if isinstance(state, dict) and 'dt' in state:
            try:
                # If caller didn't provide a usable dt, use the state's dt
                if not isinstance(dt, float) or dt <= 0:
                    dt = float(state.get('dt', dt))
            except Exception:
                pass
        # Prefer using measured angular velocity (theta_dot / w1) for derivative term
        if "theta" in state:
            err = self.angle_error(target, state["theta"])
            # compute prospective integral (anti-windup: only commit if not saturating)
            try:
                prospective_integral = self.integral + err * float(dt)
            except Exception:
                prospective_integral = self.integral + err * 0.02
            deriv = 0.0
            if "theta_dot" in state and state.get("theta_dot") is not None:
                try:
                    deriv = -float(state.get("theta_dot"))
                except Exception:
                    deriv = 0.0
            else:
                if self.last_error is not None:
                    try:
                        deriv = (err - self.last_error) / float(dt)
                    except Exception:
                        deriv = err - self.last_error
            # compute raw output using prospective integral
            raw = self.kp * err + self.ki * prospective_integral + self.kd * deriv
            # apply saturation and anti-windup: only commit integral if raw not saturated
            if abs(raw) <= self.max_output:
                self.integral = prospective_integral
            # remember last error for derivative computation next step
            self.last_error = err
            # clamp output
            if raw > self.max_output:
                return self.max_output
            if raw < -self.max_output:
                return -self.max_output
            return raw
        else:
            err = self.angle_error(target, state.get("th1", 0.0))
            try:
                prospective_integral = self.integral + err * float(dt)
            except Exception:
                prospective_integral = self.integral + err * 0.02
            deriv = 0.0
            if "w1" in state and state.get("w1") is not None:
                try:
                    deriv = -float(state.get("w1"))
                except Exception:
                    deriv = 0.0
            else:
                if self.last_error is not None:
                    try:
                        deriv = (err - self.last_error) / float(dt)
                    except Exception:
                        deriv = err - self.last_error
            raw = self.kp * err + self.ki * prospective_integral + self.kd * deriv
            if abs(raw) <= self.max_output:
                self.integral = prospective_integral
            self.last_error = err
            if raw > self.max_output:
                return self.max_output
            if raw < -self.max_output:
                return -self.max_output
            return raw


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

    def reset(self):
        """No ephemeral state for NN controller; present for API symmetry."""
        # intentionally a no-op: NN controller state is captured entirely in params
        return

    def _forward(self, x: np.ndarray) -> float:
        a = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = np.tanh(W.dot(a) + b)
        out = self.weights[-1].dot(a) + self.biases[-1]
        return float(out)

    def get_torque(self, state, target=0.0, dt: float = 0.02) -> float:
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

    def save(self, path: str):
        flat = self.get_flat_params()
        np.savez(path, params=flat)

    def load(self, path: str):
        data = np.load(path)
        flat = data['params']
        self.set_flat_params(flat)


class TorchNNPolicy:
    """PyTorch MLP policy wrapper. Exposes numpy-compatible param access for trainer."""

    def __init__(self, hidden_sizes: Tuple[int, ...] = (32, 32), device: str = 'cpu'):
        self.device = torch.device(device)
        layers = []
        in_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers).to(self.device)

    def reset(self):
        """No ephemeral state for Torch policy; API symmetry with PID."""
        return

    def get_torque(self, state, target=0.0, dt: float = 0.02) -> float:
        if "theta" in state:
            err = target - state["theta"]
        else:
            err = target - state.get("th1", 0.0)
        x = torch.tensor([[err]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        return float(out.item())

    def get_flat_params(self) -> np.ndarray:
        parts = []
        for p in self.model.parameters():
            parts.append(p.detach().cpu().numpy().ravel())
        return np.concatenate(parts) if parts else np.array([])

    def set_flat_params(self, flat: np.ndarray):
        i = 0
        for p in self.model.parameters():
            num = p.numel()
            chunk = flat[i:i+num]
            i += num
            p.data.copy_(torch.from_numpy(chunk.reshape(p.shape)).to(self.device))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

