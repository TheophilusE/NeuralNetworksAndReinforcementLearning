import math
import numpy as np


class PIDController:
    def __init__(self, kp=10.0, ki=0.0, kd=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = None

    def angle_error(self, target, current):
        # wrap to [-pi, pi]
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
            # for double pendulum, act on first link angular error
            err = self.angle_error(target, state.get("th1", 0.0))
            self.integral += err
            deriv = 0.0
            if self.last_error is not None:
                deriv = err - self.last_error
            self.last_error = err
            return self.kp * err + self.ki * self.integral + self.kd * deriv


class NNController:
    def __init__(self, hidden_sizes=(16, 16)):
        # tiny random network for demo; in a real flow you'd load trained weights
        self.sizes = [1] + list(hidden_sizes) + [1]
        self.weights = [np.random.randn(a, b) * 0.1 for a, b in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.zeros((a,)) for a in self.sizes[1:]]

    def _forward(self, x):
        a = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = np.tanh(W.dot(a) + b)
        # last layer linear
        out = self.weights[-1].dot(a) + self.biases[-1]
        return float(out)

    def get_torque(self, state, target=0.0):
        # use difference in first link angle as input
        if "theta" in state:
            err = np.array([target - state["theta"]])
        else:
            err = np.array([target - state.get("th1", 0.0)])
        return self._forward(err)
