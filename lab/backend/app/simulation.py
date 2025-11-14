import math
import numpy as np


class PendulumSimulator:
    """Simple physics for single and double pendulum (Euler integrator).

    This is intentionally simple for demo purposes. For realistic physics,
    we will integrate with PyBullet or a better ODE solver.
    """

    def __init__(self, mode: str = "single", dt: float = 0.02):
        self.mode = mode
        self.dt = float(dt)
        self.running = False
        if mode == "single":
            # state: theta, theta_dot
            self.theta = 0.2  # initial offset
            self.omega = 0.0
            self.m = 1.0
            self.l = 1.0
            self.b = 0.1
            self.g = 9.81
        elif mode == "double":
            # state: th1, th2, w1, w2
            self.th1 = 0.2
            self.th2 = -0.1
            self.w1 = 0.0
            self.w2 = 0.0
            self.m1 = 1.0
            self.m2 = 1.0
            self.l1 = 1.0
            self.l2 = 1.0
            self.g = 9.81
        else:
            raise ValueError("mode must be 'single' or 'double'")

    def get_state(self):
        if self.mode == "single":
            return {"theta": self.theta, "theta_dot": self.omega}
        else:
            return {
                "th1": self.th1,
                "th2": self.th2,
                "w1": self.w1,
                "w2": self.w2,
            }

    def step(self, torque=0.0):
        if self.mode == "single":
            # theta_ddot = ( -m*g*l*sin(theta) - b*omega + torque ) / (m*l^2)
            theta_dd = (-self.m * self.g * self.l * math.sin(self.theta) - self.b * self.omega + torque) / (
                self.m * self.l * self.l
            )
            self.omega += theta_dd * self.dt
            self.theta += self.omega * self.dt
        else:
            # very simplified double pendulum with torque on first joint only
            # Using small-angle approximations is possible but we use a crude Euler step
            m1 = self.m1
            m2 = self.m2
            l1 = self.l1
            l2 = self.l2
            g = self.g
            th1 = self.th1
            th2 = self.th2
            w1 = self.w1
            w2 = self.w2

            # Equations from standard double pendulum (without torque) are a bit long;
            # for demo we'll apply simple decoupled dynamics: torque acts on th1, gravity on both.
            th1_dd = (-g / l1) * math.sin(th1) + torque / (m1 * l1 * l1)
            th2_dd = (-g / l2) * math.sin(th2)

            self.w1 += th1_dd * self.dt
            self.w2 += th2_dd * self.dt
            self.th1 += self.w1 * self.dt
            self.th2 += self.w2 * self.dt
