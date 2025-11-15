import math
import numpy as np


class PendulumSimulator:
    """Simple cart-pole simulator (single or double) using explicit Euler.

    The simulator models a cart of mass M that can move along X and one or two
    pendulum rods attached to the cart. The `step(force)` method accepts a
    scalar horizontal force applied to the cart.
    """

    def __init__(self, mode: str = "single", dt: float = 0.02):
        self.mode = mode
        self.dt = float(dt)
        self.running = False
        # cart state
        self.x = 0.0
        self.x_dot = 0.0
        # physical parameters
        self.M = 1.0  # cart mass
        self.m = 0.1  # mass of pole 1
        self.l = 0.5  # half-length of pole
        self.g = 9.81

        if mode == "single":
            # theta, theta_dot for single pole
            self.theta = 0.05
            self.theta_dot = 0.0
        elif mode == "double":
            # two poles: theta1, theta2, and their angular velocities
            self.theta1 = 0.05
            self.theta2 = -0.02
            self.theta1_dot = 0.0
            self.theta2_dot = 0.0
            # second pole mass/length (approx)
            self.m2 = 0.08
            self.l2 = 0.45
        else:
            raise ValueError("mode must be 'single' or 'double'")

    def get_state(self):
        if self.mode == "single":
            return {"x": self.x, "x_dot": self.x_dot, "theta": self.theta, "theta_dot": self.theta_dot}
        else:
            return {
                "x": self.x,
                "x_dot": self.x_dot,
                "th1": self.theta1,
                "th2": self.theta2,
                "w1": self.theta1_dot,
                "w2": self.theta2_dot,
            }

    def step(self, force=0.0):
        # apply force to cart and integrate dynamics
        if self.mode == "single":
            m = self.m
            M = self.M
            l = self.l
            g = self.g
            th = self.theta
            thd = self.theta_dot

            # standard cart-pole equations (derived)
            num = g * math.sin(th) + math.cos(th) * ( -force - m * l * thd * thd * math.sin(th) ) / (M + m)
            den = l * (4.0/3.0 - (m * math.cos(th) * math.cos(th)) / (M + m))
            thdd = num / den
            xdd = (force + m * l * (thd * thd * math.sin(th) - thdd * math.cos(th))) / (M + m)

            # integrate
            self.x_dot += xdd * self.dt
            self.x += self.x_dot * self.dt
            self.theta_dot += thdd * self.dt
            self.theta += self.theta_dot * self.dt
        else:
            # simplified double-pole-on-cart dynamics (approximate)
            # treat second pole as decoupled small pendulum attached to first pole tip
            # compute acceleration on first pole from cart force similarly to single
            m1 = self.m
            m2 = getattr(self, 'm2', 0.08)
            M = self.M
            l1 = self.l
            l2 = getattr(self, 'l2', 0.45)
            g = self.g

            th1 = self.theta1
            th1d = self.theta1_dot
            th2 = self.theta2
            th2d = self.theta2_dot

            # approximate first pole dynamics
            num = g * math.sin(th1) + math.cos(th1) * ( -force - m1 * l1 * th1d * th1d * math.sin(th1) ) / (M + m1 + m2)
            den = l1 * (4.0/3.0 - (m1 * math.cos(th1) * math.cos(th1)) / (M + m1 + m2))
            th1dd = num / den
            xdd = (force + m1 * l1 * (th1d * th1d * math.sin(th1) - th1dd * math.cos(th1))) / (M + m1 + m2)

            # second pole: small pendulum driven by motion of first pole tip
            th2dd = (-g / l2) * math.sin(th2) + 0.02 * (th1 - th2) - 0.01 * th2d

            # integrate
            self.x_dot += xdd * self.dt
            self.x += self.x_dot * self.dt
            self.theta1_dot += th1dd * self.dt
            self.theta1 += self.theta1_dot * self.dt
            self.theta2_dot += th2dd * self.dt
            self.theta2 += self.theta2_dot * self.dt
