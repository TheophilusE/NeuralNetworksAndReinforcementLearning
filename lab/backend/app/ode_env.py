import math
import numpy as np

class OdeCartPole:
    """A simple ODE-based cart-pole simulator (single or double) using RK4.

    This implementation is headless: it performs physics integration and exposes
    a minimal `get_state()` and `get_scene_tree()` compatible shape for the
    existing backend. The frontend is expected to be the primary visualizer.
    """

    def __init__(self, mode="single", dt=0.02, gui=False, track_length=2.0):
        self.mode = mode
        self.dt = float(dt)
        self.track_length = float(track_length)
        self.gui = False
        # physical parameters
        self.M = 1.0  # cart mass
        self.m = 0.1  # pendulum mass
        self.l = 1.0  # pendulum length
        self.g = 9.81
        # state vector: for single: [x, x_dot, theta, theta_dot]
        # initialize near upright
        self.state = np.array([0.0, 0.0, 0.2, 0.0], dtype=float)
        if self.mode != "single":
            # For now, double behaves like two chained pendulums approximated
            # by two angles. We'll represent state as [x, x_dot, th1, w1, th2, w2]
            self.state = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=float)
        self.running = False

    def _dynamics_single(self, s, u):
        # s = [x, x_dot, th, th_dot]
        x, x_dot, th, th_dot = s
        M = self.M
        m = self.m
        l = self.l
        g = self.g
        force = float(u)
        sin_th = math.sin(th)
        cos_th = math.cos(th)
        denom = M + m * (1 - cos_th * cos_th)
        th_dd = (g * sin_th + cos_th * ( -force - m * l * th_dot * th_dot * sin_th) / denom) / (l * (4.0/3.0 - (m * cos_th * cos_th) / denom))
        x_dd = (force + m * l * (th_dot * th_dot * sin_th - th_dd * cos_th)) / denom
        return np.array([x_dot, x_dd, th_dot, th_dd], dtype=float)

    def _dynamics_double(self, s, u):
        # Placeholder: simple decoupled dynamics for two-link pendulum on cart
        # We'll integrate the first link as above and the second as a pendulum on the first.
        x, x_dot, th1, w1, th2, w2 = s
        # first link uses same equations with combined mass approx
        deriv1 = self._dynamics_single(np.array([x, x_dot, th1, w1]), u)
        # second link: simple pendulum attached to end of first, approximate
        g = self.g
        l = self.l
        th2_dd = -(g / l) * math.sin(th2)
        x_dd = deriv1[1]
        return np.array([deriv1[0], x_dd, deriv1[2], deriv1[3], w2, th2_dd], dtype=float)

    def step(self, torque=0.0):
        # integrate one time-step using RK4
        s = self.state
        dt = self.dt
        u = float(torque)
        if self.mode == "single":
            f = lambda y: self._dynamics_single(y, u)
            k1 = f(s)
            k2 = f(s + 0.5 * dt * k1)
            k3 = f(s + 0.5 * dt * k2)
            k4 = f(s + dt * k3)
            self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            f = lambda y: self._dynamics_double(y, u)
            k1 = f(s)
            k2 = f(s + 0.5 * dt * k1)
            k3 = f(s + 0.5 * dt * k2)
            k4 = f(s + dt * k3)
            self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # enforce track limits on cart x (centered at 0)
        half = self.track_length / 2.0
        if self.mode == "single":
            if self.state[0] < -half:
                self.state[0] = -half
                self.state[1] = 0.0
            elif self.state[0] > half:
                self.state[0] = half
                self.state[1] = 0.0
        else:
            # state[0] is x for double as well
            if self.state[0] < -half:
                self.state[0] = -half
                self.state[1] = 0.0
            elif self.state[0] > half:
                self.state[0] = half
                self.state[1] = 0.0

    def get_state(self):
        if self.mode == "single":
            x, x_dot, th, th_dot = self.state
            return {"x": float(x), "x_dot": float(x_dot), "theta": float(th), "theta_dot": float(th_dot)}
        else:
            x, x_dot, th1, w1, th2, w2 = self.state
            return {"x": float(x), "x_dot": float(x_dot), "th1": float(th1), "th2": float(th2), "w1": float(w1), "w2": float(w2)}

    def get_scene_tree(self):
        # Provide a minimal scene description compatible with frontend bridge
        if self.mode == "single":
            st = self.get_state()
            x = st["x"]
            th = st["theta"]
            # base position (cart) and end of pendulum
            cart_pos = [x, 0.0, 0.1]
            pend_tip = [x + self.l * math.sin(th), 0.0, 0.1 - self.l * math.cos(th)]
            return {"bodies": [{"body_id": 0, "base_position": cart_pos, "base_orientation": [0,0,0,1], "links": [{"link_index": -1, "world_position": cart_pos, "world_orientation": [0,0,0,1]}, {"link_index": 1, "world_position": pend_tip, "world_orientation": [0,0,0,1]}]}]}
        else:
            st = self.get_state()
            x = st["x"]
            th1 = st["th1"]
            th2 = st["th2"]
            cart_pos = [x, 0.0, 0.1]
            tip1 = [x + self.l * math.sin(th1), 0.0, 0.1 - self.l * math.cos(th1)]
            tip2 = [tip1[0] + self.l * math.sin(th2), 0.0, tip1[2] - self.l * math.cos(th2)]
            return {"bodies": [{"body_id": 0, "base_position": cart_pos, "base_orientation": [0,0,0,1], "links": [{"link_index": -1, "world_position": cart_pos, "world_orientation": [0,0,0,1]}, {"link_index": 1, "world_position": tip1, "world_orientation": [0,0,0,1]}, {"link_index": 2, "world_position": tip2, "world_orientation": [0,0,0,1]}]}]}

    def close(self):
        # no-op for ODE solver
        pass
