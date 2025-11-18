import math
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Module-level helper to map simulator angles into controller frame and
# compute a wrapped angle difference (target treated as offset relative to upright==0).
def controller_angle_error(sim_angle, target):
    try:
        current = float(sim_angle) - math.pi
    except Exception:
        current = float(sim_angle)
    try:
        targ = float(target)
    except Exception:
        targ = float(target or 0.0)
    diff = targ - current
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


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
        # Assume simulator reports upright at angle == pi (common convention).
        # Controllers map simulator angles into a controller frame where upright==0
        # by subtracting pi from the simulator angle. The `target` parameter is
        # treated as an offset relative to upright (i.e. upright==0).

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

    def angle_error(self, sim_angle, target):
        return controller_angle_error(sim_angle, target)

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
            # get sim angle and convert to controller frame inside angle_error
            sim_angle = state["theta"]
            err = self.angle_error(sim_angle, target)
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
            sim_angle = state.get("th1", 0.0)
            err = self.angle_error(sim_angle, target)
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
    trainer (no backprop engine required). The controller maps a small vector
    of observations (angle error and angular rate) to a scalar torque.
    """

    def __init__(self, hidden_sizes: Tuple[int, ...] = (16, 16), input_dim: int = 4, max_output: float = 50.0):
        # input_dim=4 -> [angle_error, angular_rate, x, x_dot]
        self.input_dim = int(input_dim)
        self.sizes = [self.input_dim] + list(hidden_sizes) + [1]
        # smaller random init to avoid large random torques at start
        self.weights: List[np.ndarray] = [np.random.randn(a, b) * 0.02 for a, b in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases: List[np.ndarray] = [np.zeros((a,)) for a in self.sizes[1:]]
        # limit NN output magnitude to match PID's realistic range
        self.max_output = float(max_output)

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
        # Build a small observation vector: [angle_error, angular_rate]
        if "theta" in state:
            sim_angle = state["theta"]
            ang_rate = state.get("theta_dot", 0.0)
            x = state.get('x', 0.0)
            x_dot = state.get('x_dot', 0.0)
        else:
            sim_angle = state.get("th1", 0.0)
            ang_rate = state.get("w1", 0.0)
            x = state.get('x', 0.0)
            x_dot = state.get('x_dot', 0.0)
        try:
            ang_rate = float(ang_rate)
        except Exception:
            ang_rate = 0.0
        err = controller_angle_error(sim_angle, target)
        obs = np.array([float(err), -float(ang_rate), float(x), float(x_dot)])
        # If network expects a different input dimension, pad/truncate
        if obs.size < self.input_dim:
            obs = np.pad(obs, (0, self.input_dim - obs.size), 'constant')
        elif obs.size > self.input_dim:
            obs = obs[: self.input_dim]
        raw = self._forward(obs)
        # deadband: treat tiny outputs as zero to avoid slow drift
        if abs(raw) < 1e-4:
            raw = 0.0
        # clamp output to configured maximum
        if abs(raw) > self.max_output:
            return float(self.max_output if raw > 0 else -self.max_output)
        return float(raw)

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

    def __init__(self, hidden_sizes: Tuple[int, ...] = (32, 32), input_dim: int = 4, device: str = 'cpu'):
        self.device = torch.device(device)
        layers = []
        in_dim = int(input_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers).to(self.device)
        # clamp network output magnitude similar to PID controller
        self.max_output = 50.0
        # initialize linear layers with small weights and zero biases to avoid
        # large random torques at start
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset(self):
        """No ephemeral state for Torch policy; API symmetry with PID."""
        return

    def get_torque(self, state, target=0.0, dt: float = 0.02) -> float:
        # Build observation [angle_error, angular_rate] for Torch policy
        if "theta" in state:
            sim_angle = state["theta"]
            ang_rate = state.get("theta_dot", 0.0)
        else:
            sim_angle = state.get("th1", 0.0)
            ang_rate = state.get("w1", 0.0)
        try:
            ang_rate = float(ang_rate)
        except Exception:
            ang_rate = 0.0
        err = float(controller_angle_error(sim_angle, target))
        obs = torch.tensor([[err, -ang_rate, float(state.get('x', 0.0)), float(state.get('x_dot', 0.0))]], dtype=torch.float32, device=self.device)
        # adapt obs shape if model was constructed with different input dim
        # by slicing or padding on CPU
        first_linear = None
        for m in self.model:
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is not None and obs.shape[1] != first_linear.in_features:
            cpu_obs = obs.detach().cpu().numpy()[0]
            expected = first_linear.in_features
            arr = cpu_obs
            if arr.size < expected:
                arr = np.pad(arr, (0, expected - arr.size), 'constant')
            elif arr.size > expected:
                arr = arr[:expected]
            obs = torch.tensor([arr], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(obs)
        raw = float(out.item())
        # deadband to ignore tiny outputs
        if abs(raw) < 1e-4:
            raw = 0.0
        if abs(raw) > getattr(self, 'max_output', float('inf')):
            return float(self.max_output if raw > 0 else -self.max_output)
        return raw

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

