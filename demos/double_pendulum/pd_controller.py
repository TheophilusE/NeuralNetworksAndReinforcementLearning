#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

from demos.common.env_wrappers import BulletDoublePendulumEnv
from demos.common.plotting import TrajectoryLogger, plot_timeseries
import matplotlib.pyplot as plt
from demos.common.plotting import init_realtime_plot, update_realtime_plot


@dataclass
class PDGains2:
    kp1: float = 30.0
    kd1: float = 3.0
    kp2: float = 20.0
    kd2: float = 2.5


@dataclass
class PDConfig2:
    target1: float = 0.0
    target2: float = 0.0
    max_torque: float = 5.0


class PDController2:
    def __init__(self, gains: PDGains2, cfg: PDConfig2):
        self.g = gains
        self.cfg = cfg

    def act(self, s: Tuple[float, float, float, float]) -> Tuple[float, float]:
        th1, th2, dth1, dth2 = s
        e1 = self._wrap(th1 - self.cfg.target1)
        e2 = self._wrap(th2 - self.cfg.target2)
        tau1 = self.g.kp1 * e1 + self.g.kd1 * (-dth1)
        tau2 = self.g.kp2 * e2 + self.g.kd2 * (-dth2)
        tau1 = float(np.clip(tau1, -self.cfg.max_torque, self.cfg.max_torque))
        tau2 = float(np.clip(tau2, -self.cfg.max_torque, self.cfg.max_torque))
        return tau1, tau2

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi


def run_pd_double_demo(
    steps: int = 3000,
    gui: bool = True,
    seed: int = 7,
    realtime: bool = True,
    window_secs: float = 5.0,
) -> Dict[str, Any]:
    env = BulletDoublePendulumEnv(gui=gui, seed=seed)
    ctrl = PDController2(PDGains2(), PDConfig2())

    window_steps = max(10, int(window_secs / env.dt))
    traj = TrajectoryLogger(keys=["t", "th1", "th2", "dth1", "dth2", "tau1", "tau2", "reward"], capacity=10 * window_steps)
    s = env.reset(randomize=True)
    t = 0
    if realtime:
        fig, ax, lines, x_axis = init_realtime_plot(["th1", "th2", "tau1", "tau2"], window=window_steps, title="Double Pendulum PD (realtime)")

    try:
        while True:
            tau1, tau2 = ctrl.act(s)
            s, r, d, info = env.step(np.array([tau1, tau2]))
            th1, th2, dth1, dth2 = s
            traj.add(t=t, th1=th1, th2=th2, dth1=dth1, dth2=dth2, tau1=tau1, tau2=tau2, reward=r)

            if realtime:
                series = traj.to_series(["th1", "th2", "tau1", "tau2"])
                update_realtime_plot(lines, x_axis, series, window=window_steps)
                plt.pause(env.dt)

            t += 1
            if d or t % steps == 0:
                s = env.reset(randomize=True)
                continue

    except KeyboardInterrupt:
        pass

    finally:
        env.close()

    fig = plot_timeseries(
        traj.to_series(["th1", "th2", "dth1", "dth2", "tau1", "tau2"]),
        title="Double Pendulum PD Control",
        xlabel="Time (step)",
    )
    return {"trajectory": traj, "figure": fig}


if __name__ == "__main__":
    run_pd_double_demo(gui=True)
