#!/usr/bin/env python3
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pybullet as p
import pybullet_data


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class BaseEnv:
    gui: bool = True
    seed: int = 0
    dt: float = 1.0 / 240.0

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass


class BulletPendulumEnv(BaseEnv):
    def __init__(self, gui: bool = True, seed: int = 0):
        super().__init__(gui=gui, seed=seed)
        _seed(seed)
        cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Create a simple pendulum via two bodies and a revolute joint
        self.base = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1)
        link_len = 1.0
        link_col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.03, height=link_len)
        link_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.03, length=link_len, rgbaColor=[0.2, 0.4, 0.8, 1.0])
        self.pend = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=link_col,
            baseVisualShapeIndex=link_vis,
            basePosition=[0, 0, 1.0],
        )
        # Constrain at top
        p.createConstraint(self.pend, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1.0])
        # Add a revolute joint around X-axis at top (simulated via torque at base orientation)
        self.observation_dim = 2
        self.action_dim = 1

    def reset(self, randomize: bool = True) -> Tuple[float, float]:
        # Pendulum angle stored via base orientation pitch around Y for simplicity
        theta = np.random.uniform(-math.pi, math.pi) if randomize else 0.1
        theta_dot = 0.0
        self._state = np.array([theta, theta_dot], dtype=np.float32)
        return tuple(self._state)

    def step(self, tau: float):
        theta, theta_dot = self._state
        # Simple physics integration for demo purposes
        g = 9.81
        L = 1.0
        m = 1.0
        b = 0.05
        # Equation: theta_ddot = -(g/L) * sin(theta) + tau/(m*L^2) - b * theta_dot
        theta_ddot = -(g / L) * math.sin(theta) + (tau / (m * L * L)) - b * theta_dot
        theta_dot = theta_dot + theta_ddot * self.dt
        theta = ((theta + theta_dot * self.dt + math.pi) % (2 * math.pi)) - math.pi
        self._state = np.array([theta, theta_dot], dtype=np.float32)

        reward = - (theta ** 2 + 0.1 * (theta_dot ** 2) + 0.01 * (tau ** 2))
        done = abs(theta) > math.pi  # never triggers with wrap; kept for API symmetry
        info = {}
        return tuple(self._state), float(reward), bool(done), info


class BulletDoublePendulumEnv(BaseEnv):
    def __init__(self, gui: bool = True, seed: int = 0):
        super().__init__(gui=gui, seed=seed)
        _seed(seed)
        p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(self.dt)
        self.observation_dim = 4
        self.action_dim = 2

    def reset(self, randomize: bool = True):
        th1 = np.random.uniform(-math.pi, math.pi) if randomize else 0.2
        th2 = np.random.uniform(-math.pi, math.pi) if randomize else -0.1
        dth1 = 0.0
        dth2 = 0.0
        self._s = np.array([th1, th2, dth1, dth2], dtype=np.float32)
        return tuple(self._s)

    def step(self, tau: np.ndarray):
        th1, th2, dth1, dth2 = self._s
        m1 = m2 = 1.0
        L1 = L2 = 1.0
        g = 9.81
        b = 0.05

        # Simplified double pendulum dynamics (approximate for workshop):
        # Use coupled equations with damping and torques tau[0], tau[1]
        sin1, cos1 = math.sin(th1), math.cos(th1)
        sin2, cos2 = math.sin(th2), math.cos(th2)
        # Inertia terms (simple)
        I1 = (1.0 / 3.0) * m1 * L1 * L1
        I2 = (1.0 / 3.0) * m2 * L2 * L2

        d2th1 = -(g / L1) * sin1 + tau[0] / I1 - b * dth1 + 0.05 * (th2 - th1)
        d2th2 = -(g / L2) * sin2 + tau[1] / I2 - b * dth2 + 0.05 * (th1 - th2)

        dth1 = dth1 + d2th1 * self.dt
        dth2 = dth2 + d2th2 * self.dt
        th1 = ((th1 + dth1 * self.dt + math.pi) % (2 * math.pi)) - math.pi
        th2 = ((th2 + dth2 * self.dt + math.pi) % (2 * math.pi)) - math.pi
        self._s = np.array([th1, th2, dth1, dth2], dtype=np.float32)

        reward = - (th1 ** 2 + th2 ** 2 + 0.1 * (dth1 ** 2 + dth2 ** 2) + 0.01 * (tau[0] ** 2 + tau[1] ** 2))
        done = False
        info = {}
        return tuple(self._s), float(reward), bool(done), info


class BulletHopperEnv(BaseEnv):
    def __init__(self, gui: bool = True, seed: int = 0):
        super().__init__(gui=gui, seed=seed)
        _seed(seed)
        p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(self.dt)
        self.observation_dim = 16
        self.action_dim = 4

    def reset(self, randomize: bool = True):
        # State: base position/velocity + joint angles/vels (mocked)
        base = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # x,y,vx,vy
        joints = np.random.uniform(-0.3, 0.3, size=12).astype(np.float32) if randomize else np.zeros(12, dtype=np.float32)
        self._s = np.concatenate([base, joints]).astype(np.float32)
        return tuple(self._s)

    def step(self, a: np.ndarray):
        # Advance base forward proportional to hip/knee pattern; penalize torque energy
        x, y, vx, vy = self._s[:4]
        joints = self._s[4:]
        torque_cost = float(np.sum(np.abs(a)))

        # Update: mock forward velocity from action similarity to a "gait pattern"
        gait = np.array([1.0, -1.0, 0.8, -0.8], dtype=np.float32)
        vx = 0.95 * vx + 0.05 * float(np.dot(a, gait)) / 4.0
        x = x + vx * self.dt
        self._s[:4] = np.array([x, y, vx, vy], dtype=np.float32)
        joints = 0.9 * joints + 0.1 * a.repeat(3)[:12]  # simplistic propagation
        self._s[4:] = joints

        tilt = 0.1 * abs(joints[0])
        foot_height = max(0.0, joints[2])

        reward = vx - 0.002 * torque_cost - 0.5 * tilt + 0.1 * foot_height
        done = False
        info = {
            "base_pos_vel": (x, y, vx, vy),
            "base_vel": (vx, 0.0, 0.0),
            "torque_cost": torque_cost,
            "base_pitch": tilt,
            "foot_height": foot_height,
        }
        return tuple(self._s), float(reward), bool(done), info

    def phase_action(self, s, phase: float) -> np.ndarray:
        # Generate sinusoidal joint target pattern mapped to 4 actions
        return np.array([
            20.0 * math.sin(phase),
            20.0 * math.sin(phase + math.pi / 2),
            15.0 * math.sin(phase + math.pi / 3),
            15.0 * math.sin(phase + 2 * math.pi / 3),
        ], dtype=np.float32)

    def random_action(self) -> np.ndarray:
        return np.random.uniform(-30.0, 30.0, size=self.action_dim).astype(np.float32)

    def debug_text(self, msg: str):
        # Placeholder for on-screen text stitching; left as a hook for GUI
        pass
