#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PendulumParams:
    length: float = 1.0
    mass: float = 1.0
    gravity: float = 9.81
    damping: float = 0.05

    @property
    def inertia(self) -> float:
        return (1.0 / 3.0) * self.mass * (self.length ** 2)


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    import math
    return (theta + math.pi) % (2 * math.pi) - math.pi


def pendulum_reward(theta: float, theta_dot: float, tau: float) -> float:
    """Simple reward: penalize angle error and torque; encourage upright."""
    import math
    err = wrap_angle(theta)
    return - (err ** 2 + 0.1 * (theta_dot ** 2) + 0.01 * (tau ** 2))
