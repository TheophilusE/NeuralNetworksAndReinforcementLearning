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
        # create and store a physics client id so all PyBullet calls go to the same client
        self._cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(self.dt, physicsClientId=self._cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # keep reference to loaded plane for clarity
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self._cid)
        # If running with GUI, position the debug camera to look at the scene
        if gui:
            try:
                # distance, yaw, pitch, target
                p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0, 0, 1.0], physicsClientId=self._cid)
            except Exception:
                pass

        # Make the scene look like a small robotics lab: try to texture the plane and add benches/lights
        try:
            # Try to load a checker texture from pybullet_data
            try:
                tex = p.loadTexture("checker_grid.png", physicsClientId=self._cid)
                p.changeVisualShape(self.plane, -1, textureUniqueId=tex, physicsClientId=self._cid)
            except Exception:
                # Fallback: tint the plane
                try:
                    p.changeVisualShape(self.plane, -1, rgbaColor=[0.85, 0.85, 0.85, 1.0], physicsClientId=self._cid)
                except Exception:
                    pass

            # Add a couple of static benches/tables (boxes)
            bench_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.4, 0.05], physicsClientId=self._cid)
            bench_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.4, 0.05], rgbaColor=[0.45, 0.3, 0.2, 1.0], physicsClientId=self._cid)
            self.bench1 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=bench_col, baseVisualShapeIndex=bench_vis, basePosition=[-1.0, -0.8, 0.4], physicsClientId=self._cid)
            self.bench2 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=bench_col, baseVisualShapeIndex=bench_vis, basePosition=[1.0, -0.8, 0.4], physicsClientId=self._cid)

            # Add an overhead lamp (thin cylinder) for visual flair
            lamp_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.02, physicsClientId=self._cid)
            lamp_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.02, rgbaColor=[1.0, 0.95, 0.7, 1.0], physicsClientId=self._cid)
            self.lamp = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=lamp_col, baseVisualShapeIndex=lamp_vis, basePosition=[0.0, 0.0, 2.2], physicsClientId=self._cid)

            # Add a small equipment cube
            cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.15], physicsClientId=self._cid)
            cube_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.15], rgbaColor=[0.2, 0.6, 0.6, 1.0], physicsClientId=self._cid)
            self.equip = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=cube_col, baseVisualShapeIndex=cube_vis, basePosition=[-0.6, 0.8, 0.25], physicsClientId=self._cid)

            # Draw simple grid lines on the floor for lab feeling
            for i in range(-5, 6):
                p.addUserDebugLine([i * 0.5, -2.5, 0.01], [i * 0.5, 2.5, 0.01], [0.5, 0.5, 0.5], 1.0, physicsClientId=self._cid)
                p.addUserDebugLine([-2.5, i * 0.5, 0.01], [2.5, i * 0.5, 0.01], [0.5, 0.5, 0.5], 1.0, physicsClientId=self._cid)

            # Coordinate axes at origin
            p.addUserDebugLine([0, 0, 0.02], [0.5, 0, 0.02], [1, 0, 0], 2.0, physicsClientId=self._cid)
            p.addUserDebugLine([0, 0, 0.02], [0, 0.5, 0.02], [0, 1, 0], 2.0, physicsClientId=self._cid)
            p.addUserDebugLine([0, 0, 0.02], [0, 0, 0.7], [0, 0, 1], 2.0, physicsClientId=self._cid)
        except Exception:
            # Non-fatal: continue without decorations if anything fails
            pass

        # We'll represent the pendulum using two visible spheres (pivot and bob)
        # and update the bob position each step according to the analytic state.
        self.base = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1)
        link_len = 1.0
        # small spheres for pivot and bob
        pivot_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.1, 0.1, 0.1, 1.0], physicsClientId=self._cid)
        bob_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[0.2, 0.4, 0.8, 1.0], physicsClientId=self._cid)
        try:
            self.pivot_vis = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=pivot_vis, basePosition=[0, 0, 1.0], physicsClientId=self._cid)
            self.bob_vis = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=bob_vis, basePosition=[0, 0, 1.0 - link_len], physicsClientId=self._cid)
        except Exception:
            # fallback: mark visuals None
            self.pivot_vis = None
            self.bob_vis = None
        # Keep analytic dims
        self.observation_dim = 2
        self.action_dim = 1

        # initial visuals already set above (pivot_vis / bob_vis)

    def reset(self, randomize: bool = True) -> Tuple[float, float]:
        # Pendulum angle stored via base orientation pitch around Y for simplicity
        theta = np.random.uniform(-math.pi, math.pi) if randomize else 0.1
        theta_dot = 0.0
        self._state = np.array([theta, theta_dot], dtype=np.float32)
        # update visuals to the reset state immediately
        try:
            # place pivot and bob according to theta
            L = 1.0
            x = L * math.sin(theta)
            z = 1.0 - L * math.cos(theta)
            if getattr(self, 'pivot_vis', None) is not None:
                p.resetBasePositionAndOrientation(self.pivot_vis, [0, 0, 1.0], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self._cid)
            if getattr(self, 'bob_vis', None) is not None:
                p.resetBasePositionAndOrientation(self.bob_vis, [float(x), 0.0, float(z)], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self._cid)
            # setup debug text id placeholder
            self._debug_text_id = None
            try:
                p.stepSimulation(physicsClientId=self._cid)
            except Exception:
                pass
        except Exception:
            pass
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

        # Update pybullet visuals to reflect the analytic pendulum state (if GUI connected)
        try:
            # Bob position in world frame: pivot at (0,0,1.0); pendulum lies in X-Z plane
            L = 1.0
            x = L * math.sin(theta)
            z = 1.0 - L * math.cos(theta)
            if getattr(self, 'pivot_vis', None) is not None:
                p.resetBasePositionAndOrientation(self.pivot_vis, [0, 0, 1.0], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self._cid)
            if getattr(self, 'bob_vis', None) is not None:
                p.resetBasePositionAndOrientation(self.bob_vis, [float(x), 0.0, float(z)], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self._cid)
            # update on-screen debug text showing theta and theta_dot
            try:
                txt = f"theta={theta:.3f}\nota_dot={theta_dot:.3f}"
                if getattr(self, '_debug_text_id', None) is None:
                    self._debug_text_id = p.addUserDebugText(txt, [0.0, -0.5, 1.8], textColorRGB=[1, 1, 1], textSize=1.2, physicsClientId=self._cid)
                else:
                    p.addUserDebugText(txt, [0.0, -0.5, 1.8], textColorRGB=[1, 1, 1], textSize=1.2, replaceItemUniqueId=self._debug_text_id, physicsClientId=self._cid)
            except Exception:
                pass
            try:
                p.stepSimulation(physicsClientId=self._cid)
            except Exception:
                pass
        except Exception:
            pass

        reward = - (theta ** 2 + 0.1 * (theta_dot ** 2) + 0.01 * (tau ** 2))
        done = abs(theta) > math.pi  # never triggers with wrap; kept for API symmetry
        info = {}
        return tuple(self._state), float(reward), bool(done), info


class BulletDoublePendulumEnv(BaseEnv):
    def __init__(self, gui: bool = True, seed: int = 0):
        super().__init__(gui=gui, seed=seed)
        _seed(seed)
        self._cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(self.dt, physicsClientId=self._cid)
        self.observation_dim = 4
        self.action_dim = 2
        # Create simple visual representations for the two links so GUI shows something
        try:
            link_len = 1.0
            link_col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.03, height=link_len, physicsClientId=self._cid)
            link_vis1 = p.createVisualShape(p.GEOM_CAPSULE, radius=0.03, length=link_len, rgbaColor=[0.8, 0.3, 0.3, 1.0], physicsClientId=self._cid)
            link_vis2 = p.createVisualShape(p.GEOM_CAPSULE, radius=0.03, length=link_len, rgbaColor=[0.3, 0.8, 0.3, 1.0], physicsClientId=self._cid)
            # place both at base height; we'll rotate them in step()
            self.link1 = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=link_col, baseVisualShapeIndex=link_vis1, basePosition=[0, 0, 1.0], physicsClientId=self._cid)
            self.link2 = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=link_col, baseVisualShapeIndex=link_vis2, basePosition=[0, 0, 1.0], physicsClientId=self._cid)
            q = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
            p.resetBasePositionAndOrientation(self.link1, [0, 0, 1.0], q, physicsClientId=self._cid)
            p.resetBasePositionAndOrientation(self.link2, [0, 0, 1.0], q, physicsClientId=self._cid)
        except Exception:
            # If pybullet GUI not available or creation fails, ignore — env still works headless
            self.link1 = None
            self.link2 = None
        # Set camera when GUI is enabled
        if gui:
            try:
                p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0, 0, 1.0], physicsClientId=self._cid)
            except Exception:
                pass

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

        # Update visuals for links if they exist
        try:
            if getattr(self, 'link1', None) is not None:
                # compute end-to-end positions: link1 pivot at (0,0,1.0)
                L1 = 1.0
                x1 = L1 * math.sin(th1)
                z1 = 1.0 - L1 * math.cos(th1)
                center1 = [(0.0 + x1) / 2.0, 0.0, (1.0 + z1) / 2.0]
                q1 = p.getQuaternionFromEuler([0.0, float(th1), 0.0])
                p.resetBasePositionAndOrientation(self.link1, center1, q1, physicsClientId=self._cid)
            if getattr(self, 'link2', None) is not None:
                # attach link2 base at tip of link1 (x1, z1) and compute its tip
                L2 = 1.0
                x2_tip = x1 + L2 * math.sin(th2)
                z2_tip = z1 - L2 * math.cos(th2)
                center2 = [(x1 + x2_tip) / 2.0, 0.0, (z1 + z2_tip) / 2.0]
                q2 = p.getQuaternionFromEuler([0.0, float(th2), 0.0])
                p.resetBasePositionAndOrientation(self.link2, center2, q2, physicsClientId=self._cid)
            # update on-screen debug text showing angles
            try:
                txt = f"th1={th1:.3f} th2={th2:.3f}"
                if getattr(self, '_debug_text_id', None) is None:
                    self._debug_text_id = p.addUserDebugText(txt, [0.0, -0.5, 1.9], textColorRGB=[1, 1, 1], textSize=1.2, physicsClientId=self._cid)
                else:
                    p.addUserDebugText(txt, [0.0, -0.5, 1.9], textColorRGB=[1, 1, 1], textSize=1.2, replaceItemUniqueId=self._debug_text_id, physicsClientId=self._cid)
            except Exception:
                pass
            try:
                p.stepSimulation(physicsClientId=self._cid)
            except Exception:
                pass
        except Exception:
            pass

        reward = - (th1 ** 2 + th2 ** 2 + 0.1 * (dth1 ** 2 + dth2 ** 2) + 0.01 * (tau[0] ** 2 + tau[1] ** 2))
        done = False
        info = {}
        return tuple(self._s), float(reward), bool(done), info


class BulletHopperEnv(BaseEnv):
    def __init__(self, gui: bool = True, seed: int = 0):
        super().__init__(gui=gui, seed=seed)
        _seed(seed)
        self._cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(self.dt, physicsClientId=self._cid)
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
