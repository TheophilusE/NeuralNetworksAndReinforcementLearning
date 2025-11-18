import math
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn


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
        try:
            dt = float(dt)
        except Exception:
            dt = 0.02
        if isinstance(state, dict) and 'dt' in state:
            try:
                if not isinstance(dt, float) or dt <= 0:
                    dt = float(state.get('dt', dt))
            except Exception:
                pass

        # extract angle and angular rate using known keys
        if "theta" in state:
            sim_angle = state["theta"]
            ang_rate = state.get("theta_dot", 0.0)
        else:
            sim_angle = state.get("th1", 0.0)
            ang_rate = state.get("w1", 0.0)

        err = self.angle_error(sim_angle, target)

        # prospective integral for anti-windup
        try:
            prospective_integral = self.integral + err * float(dt)
        except Exception:
            prospective_integral = self.integral + err * 0.02

        # derivative term: prefer measured angular velocity
        deriv = 0.0
        if ang_rate is not None:
            try:
                deriv = -float(ang_rate)
            except Exception:
                deriv = 0.0
        else:
            if self.last_error is not None:
                try:
                    deriv = (err - self.last_error) / float(dt)
                except Exception:
                    deriv = err - self.last_error

        raw = self.kp * err + self.ki * prospective_integral + self.kd * deriv
        # anti-windup: only commit integral if not saturating
        if abs(raw) <= self.max_output:
            self.integral = prospective_integral
        self.last_error = err

        # clamp
        if abs(raw) > self.max_output:
            raw = self.max_output if raw > 0 else -self.max_output
        return float(raw)


class NNController:
    """Small MLP-based policy that receives PID-style signals so it can learn PID-like behavior.

    Observation vector: [angle_error, derivative, integral, x, x_dot] (default)
    """

    def __init__(self, hidden_sizes=(16, 16), activation=np.tanh, max_output=20.0, input_dim=5):
        self.max_output = float(max_output)
        self.last_error = 0.0
        self.integral = 0.0
        # output smoothing and anti-jerk defaults
        self.last_output = 0.0
        self.smoothing_alpha = 0.3  # EMA alpha for action smoothing (0..1)
        self.angle_deadband = 0.02  # radians; small-angle deadband to avoid jitter
        self.deriv_deadband = 0.01  # rad/s; angular rate deadband
        # limit the integral term to avoid wind-up / numerical blowups over long runs
        self.integral_limit = float(10.0)
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.input_dim = int(input_dim)
        # initialize weights small so random actions are small at start
        self.weights = []
        self.biases = []
        dims = [self.input_dim] + list(self.hidden_sizes) + [1]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * 0.02
            b = np.zeros(dims[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0

    def _forward(self, x: np.ndarray) -> float:
        a = x.astype(np.float32)
        for i in range(len(self.weights) - 1):
            a = self.activation(a.dot(self.weights[i]) + self.biases[i])
        out = a.dot(self.weights[-1]) + self.biases[-1]
        return float(out.squeeze())

    def get_torque(self, sim_state: dict, target=0.0, dt=0.01):
        # build observation: [angle_error, deriv, integral, x, x_dot]
        th = sim_state.get("theta", sim_state.get("th1", 0.0))
        th_dot = sim_state.get("theta_dot", sim_state.get("w1", 0.0))
        x = sim_state.get("x", 0.0)
        x_dot = sim_state.get("x_dot", 0.0)
        # controller frame uses upright==0 convention
        err = controller_angle_error(th, target)

        # ensure dt is numeric
        try:
            dt_f = float(dt)
            if dt_f <= 0:
                dt_f = 0.01
        except Exception:
            dt_f = 0.01

        # derivative: prefer angular rate from sim, otherwise finite-diff
        if th_dot is not None:
            deriv = -float(th_dot)
        else:
            deriv = (err - getattr(self, "last_error", err)) / dt_f

        # update integral state so NN can learn integral behavior
        self.integral += err * dt_f
        # clamp integral to limit
        if not np.isfinite(self.integral):
            self.integral = 0.0
        else:
            if self.integral > self.integral_limit:
                self.integral = self.integral_limit
            elif self.integral < -self.integral_limit:
                self.integral = -self.integral_limit
        self.last_error = err

        obs = np.array([err, deriv, self.integral, x, x_dot], dtype=np.float32)
        # pad/truncate obs to match input_dim
        if obs.size < self.input_dim:
            obs = np.concatenate([obs, np.zeros(self.input_dim - obs.size, dtype=np.float32)])
        elif obs.size > self.input_dim:
            obs = obs[: self.input_dim]

        raw = self._forward(obs)
        # guard against NaN/inf outputs from numeric instability
        if not np.isfinite(raw):
            raw = 0.0
        # tiny deadband to avoid tiny jitter
        if abs(raw) < 1e-3:
            raw = 0.0

        # small-angle deadband: if nearly upright and low angular rate, avoid acting
        try:
            if abs(err) < self.angle_deadband and abs(deriv) < self.deriv_deadband:
                raw = 0.0
        except Exception:
            pass

        # guard against NaN/inf from network
        if not np.isfinite(raw):
            raw = 0.0

        # clamp immediate raw output
        raw = max(-self.max_output, min(self.max_output, raw))

        # smooth actions to avoid sudden jerks: EMA with respect to last_output
        try:
            smoothed = float(self.smoothing_alpha * raw + (1.0 - self.smoothing_alpha) * self.last_output)
        except Exception:
            smoothed = float(raw)

        # rate-limit change based on dt (allow proportionally small change per step)
        try:
            max_delta = float(self.max_output) * 0.5 * float(dt_f)
        except Exception:
            max_delta = float(self.max_output) * 0.5 * 0.02
        delta = smoothed - float(self.last_output)
        if delta > max_delta:
            out = float(self.last_output + max_delta)
        elif delta < -max_delta:
            out = float(self.last_output - max_delta)
        else:
            out = float(smoothed)

        # commit last_output and return
        self.last_output = out
        return float(out)

    def get_flat_params(self) -> np.ndarray:
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_flat_params(self, flat: np.ndarray):
        idx = 0
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            w_n = flat[idx : idx + w.size].reshape(w.shape)
            idx += w.size
            b_n = flat[idx : idx + b.size].reshape(b.shape)
            idx += b.size
            self.weights[i] = w_n
            self.biases[i] = b_n

    def num_params(self) -> int:
        p = self.get_flat_params()
        return int(p.size) if getattr(p, 'size', None) is not None else 0

    def expected_num_params(self) -> int:
        """Return the expected flattened parameter size for this network."""
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return int(total)

    def set_flat_params_safe(self, flat: np.ndarray):
        """Set flattened params, tolerating size mismatches by padding/truncating.

        This avoids exceptions during ES evaluations when shapes differ slightly
        (for example after changing network architecture). It logs size mismatches
        and applies a safe assignment.
        """
        flat = np.asarray(flat).ravel()
        expected = self.expected_num_params()
        if flat.size != expected:
            try:
                print(f"[NNController] set_flat_params: size mismatch got {flat.size}, expected {expected}; padding/truncating", flush=True)
            except Exception:
                pass
        if flat.size < expected:
            pad = np.zeros(expected - flat.size, dtype=flat.dtype)
            flat = np.concatenate([flat, pad])
        elif flat.size > expected:
            flat = flat[:expected]

        # now set normally
        idx = 0
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            w_n = flat[idx : idx + w.size].reshape(w.shape)
            idx += w.size
            b_n = flat[idx : idx + b.size].reshape(b.shape)
            idx += b.size
            self.weights[i] = w_n
            self.biases[i] = b_n


class TorchNNPolicy:
    """PyTorch MLP policy wrapper. Exposes numpy-compatible param access for trainer.

    This policy is stateful (integral and last_error) to allow it to learn PID-like
    behavior when provided with the same signals.
    """

    def __init__(self, hidden_sizes: Tuple[int, ...] = (32, 32), input_dim: int = 5, device: str = 'cpu'):
        self.device = torch.device(device)
        layers = []
        in_dim = int(input_dim)
        # smoothing and anti-jerk defaults for torch policy as well
        self.last_output = 0.0
        self.smoothing_alpha = 0.3
        self.angle_deadband = 0.02
        self.deriv_deadband = 0.01
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

        # stateful signals to mirror PID inputs
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0
        # prevent integrator wind-up and numeric explosion
        self.integral_limit = float(10.0)

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0

    def get_torque(self, state, target=0.0, dt: float = 0.02) -> float:
        # Build observation [angle_error, deriv, integral, x, x_dot] for Torch policy
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

        try:
            dt_f = float(dt)
            if dt_f <= 0:
                dt_f = 0.01
        except Exception:
            dt_f = 0.01

        deriv = -ang_rate
        self.integral += err * dt_f
        self.last_error = err
        # clamp integrator
        if not np.isfinite(self.integral):
            self.integral = 0.0
        else:
            if self.integral > self.integral_limit:
                self.integral = self.integral_limit
            elif self.integral < -self.integral_limit:
                self.integral = -self.integral_limit

        obs_np = np.array([err, deriv, self.integral, float(state.get('x', 0.0)), float(state.get('x_dot', 0.0))], dtype=np.float32)

        # adapt obs shape if model was constructed with different input dim
        first_linear = None
        for m in self.model:
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        expected = first_linear.in_features if first_linear is not None else obs_np.size
        arr = obs_np
        if arr.size < expected:
            arr = np.pad(arr, (0, expected - arr.size), 'constant')
        elif arr.size > expected:
            arr = arr[:expected]
        # convert efficiently: make a contiguous numpy float32 array then convert
        arr_np = np.asarray(arr, dtype=np.float32)
        obs = torch.from_numpy(arr_np.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            out = self.model(obs)
        raw = float(out.item())
        # guard against NaN/inf
        if not np.isfinite(raw):
            raw = 0.0
        # deadband to ignore tiny outputs
        if abs(raw) < 1e-4:
            raw = 0.0

        # small-angle deadband
        try:
            if abs(err) < self.angle_deadband and abs(deriv) < self.deriv_deadband:
                raw = 0.0
        except Exception:
            pass

        # clamp raw
        raw = max(-self.max_output, min(self.max_output, raw))

        # smooth actions
        try:
            smoothed = float(self.smoothing_alpha * raw + (1.0 - self.smoothing_alpha) * self.last_output)
        except Exception:
            smoothed = float(raw)

        # rate limit change
        try:
            max_delta = float(self.max_output) * 0.5 * float(dt_f)
        except Exception:
            max_delta = float(self.max_output) * 0.5 * 0.02
        delta = smoothed - float(self.last_output)
        if delta > max_delta:
            out = float(self.last_output + max_delta)
        elif delta < -max_delta:
            out = float(self.last_output - max_delta)
        else:
            out = float(smoothed)

        self.last_output = out
        return out
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

