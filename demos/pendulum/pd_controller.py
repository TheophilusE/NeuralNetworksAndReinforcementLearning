#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pybullet as p
import pybullet_data

from demos.common.env_wrappers import BulletPendulumEnv
import matplotlib.pyplot as plt
from demos.common.plotting import (
    TrajectoryLogger,
    plot_timeseries,
    init_realtime_plot,
    update_realtime_plot,
)


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
    realtime: bool = True,
    window_secs: float = 5.0,
) -> Dict[str, Any]:
    env = BulletPendulumEnv(gui=gui, seed=seed)
    ctrl = PDController(gains, cfg)
    # We'll run perpetually (until Ctrl-C) and optionally show a realtime sliding timeline.
    window_steps = max(10, int(window_secs / env.dt))
    traj = TrajectoryLogger(keys=["t", "theta", "theta_dot", "tau", "reward"], capacity=10 * window_steps)

    state = env.reset(randomize=True)
    t = 0
    fig = None
    if realtime:
        # prepare realtime figure
        fig, ax, lines, x_axis = init_realtime_plot(["theta", "theta_dot", "tau"], window=window_steps, title="Pendulum PD Control (realtime)", xlabel="Time (steps)")

    try:
        while True:
            theta, theta_dot = state
            tau = ctrl.act(theta, theta_dot)
            state, reward, done, info = env.step(tau)

            if t % logging_interval == 0:
                traj.add(t=t, theta=theta, theta_dot=theta_dot, tau=tau, reward=reward)

            # realtime plotting update
            if realtime:
                series = traj.to_series(["theta", "theta_dot", "tau"])
                update_realtime_plot(lines, x_axis, series, window=window_steps)
                # small pause to keep GUI responsive; prefer env.dt pacing
                plt.pause(env.dt if sleep_gui is False else env.dt)
            else:
                if gui and sleep_gui:
                    time.sleep(env.dt)

            t += 1
            # reset episode occasionally to introduce random initial conditions and keep controller 'stabilizing'
            if done or t % steps == 0:
                state = env.reset(randomize=True)
                # continue running indefinitely
                continue

    except KeyboardInterrupt:
        # Graceful exit on Ctrl-C
        pass

    finally:
        env.close()

    # final static figure for convenience
    static_fig = plot_timeseries(series=traj.to_series(["theta", "theta_dot", "tau"]), title="Pendulum PD Control: angle, angular velocity, torque", xlabel="Time (step)")
    return {"trajectory": traj, "figure": static_fig}


if __name__ == "__main__":
    p.connect(p.DIRECT)  # Defensive: ensure no leak if env GUI fails
    res = run_pd_demo(gui=True, sleep_gui=False)
