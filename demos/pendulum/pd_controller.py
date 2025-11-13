#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pybullet as p
import pybullet_data

from demos.common.env_wrappers import BulletPendulumEnv
from demos.common.plotting import TrajectoryLogger, plot_timeseries


@dataclass
class PDGains:
    kp: float = 20.0
    kd: float = 2.5


@dataclass
class PDConfig:
    target_angle: float = 0.0  # Upright
    max_torque: float = 5.0


class PDController:
    def __init__(self, gains: PDGains, cfg: PDConfig):
        self.g = gains
        self.cfg = cfg

    def act(self, theta: float, theta_dot: float) -> float:
        # Control law: tau = kp * (theta_err) + kd * (theta_dot_err)
        # target theta dot is 0
        err = self._angle_err(theta, self.cfg.target_angle)
        tau = self.g.kp * err + self.g.kd * (0.0 - theta_dot)
        return float(np.clip(tau, -self.cfg.max_torque, self.cfg.max_torque))

    @staticmethod
    def _angle_err(angle: float, target: float) -> float:
        # Map error to [-pi, pi] for stability
        e = (angle - target + math.pi) % (2 * math.pi) - math.pi
        return e


def run_pd_demo(
    steps: int = 2000,
    gains: PDGains = PDGains(),
    cfg: PDConfig = PDConfig(),
    gui: bool = True,
    seed: int = 42,
    logging_interval: int = 1,
    sleep_gui: bool = False,
) -> Dict[str, Any]:
    env = BulletPendulumEnv(gui=gui, seed=seed)
    ctrl = PDController(gains, cfg)
    traj = TrajectoryLogger(
        keys=["t", "theta", "theta_dot", "tau", "reward"],
        capacity=steps + 1,
    )

    state = env.reset(randomize=True)
    for t in range(steps):
        theta, theta_dot = state
        tau = ctrl.act(theta, theta_dot)
        state, reward, done, info = env.step(tau)

        if t % logging_interval == 0:
            traj.add(
                t=t,
                theta=theta,
                theta_dot=theta_dot,
                tau=tau,
                reward=reward,
            )

        if gui and sleep_gui:
            time.sleep(env.dt)
        if done:
            break

    env.close()

    # Plot
    fig = plot_timeseries(
        series=traj.to_series(["theta", "theta_dot", "tau"]),
        title="Pendulum PD Control: angle, angular velocity, torque",
        xlabel="Time (step)",
    )
    return {"trajectory": traj, "figure": fig}


if __name__ == "__main__":
    p.connect(p.DIRECT)  # Defensive: ensure no leak if env GUI fails
    res = run_pd_demo(gui=True, sleep_gui=False)
