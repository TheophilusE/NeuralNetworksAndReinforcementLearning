#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch

from demos.common.env_wrappers import BulletHopperEnv
from demos.common.model_utils import MLPPolicy, seed_everything
import matplotlib.pyplot as plt
from demos.common.plotting import (
    TrajectoryLogger,
    plot_timeseries,
    init_realtime_plot,
    update_realtime_plot,
)


@dataclass
class HopperCfg:
    hidden_sizes: tuple = (128, 128)
    activation: str = "tanh"
    max_force: float = 40.0
    steps: int = 5000


def run_hopper_demo(gui: bool = True, seed: int = 202, cfg: HopperCfg = HopperCfg()) -> Dict[str, Any]:
    seed_everything(seed)
    env = BulletHopperEnv(gui=gui, seed=seed)
    obs_dim, act_dim = env.observation_dim, env.action_dim

    # For our workshop, we can approximate a reasonable gait via a hand-crafted phase policy,
    # then refine with a tiny supervised fit to give a stable inference demo.
    phase_policy = MLPPolicy(obs_dim, act_dim, hidden_sizes=cfg.hidden_sizes, activation=cfg.activation)
    phase_policy = quick_phase_imitation(env, phase_policy, epochs=20, batch=512)

    window_steps = max(10, int(5.0 / env.dt))
    traj = TrajectoryLogger(keys=["t", "x", "y", "vx", "vy", "reward"], capacity=10 * window_steps)
    s = env.reset(randomize=True)
    t = 0
    fig = None
    fig, ax, lines, x_axis = init_realtime_plot(["x", "vx", "reward"], window=window_steps, title="Hopper Demo (realtime)")

    try:
        while True:
            with torch.no_grad():
                a = phase_policy(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()
            a = np.clip(a, -cfg.max_force, cfg.max_force)
            s, r, d, info = env.step(a)
            x, y, vx, vy = info.get("base_pos_vel", (0.0, 0.0, 0.0, 0.0))
            traj.add(t=t, x=x, y=y, vx=vx, vy=vy, reward=r)

            series = traj.to_series(["x", "vx", "reward"])
            update_realtime_plot(lines, x_axis, series, window=window_steps)
            plt.pause(env.dt)

            t += 1
            if d or t % cfg.steps == 0:
                s = env.reset(randomize=True)
                continue

    except KeyboardInterrupt:
        pass

    finally:
        env.close()

    fig = plot_timeseries(traj.to_series(["x", "vx", "reward"]), title="Hopper Demo: forward progress & reward")
    return {"trajectory": traj, "figure": fig, "policy": phase_policy}


def quick_phase_imitation(env: BulletHopperEnv, policy: MLPPolicy, epochs: int = 15, batch: int = 256):
    """
    Build a toy dataset using a sinusoidal phase schedule on joint targets.
    Supervise an MLP to reproduce it given state -> action for a smooth demo.
    """
    X, Y = [], []
    for _ in range(800):
        s = env.reset(randomize=True)
        phase = np.random.uniform(0, 2 * np.pi)
        for t in range(150):
            a = env.phase_action(s, phase + 0.02 * t)
            X.append(s)
            Y.append(a)
            s, _, d, _ = env.step(a)
            if d:
                break

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss = torch.nn.SmoothL1Loss()
    policy.train()
    for _ in range(epochs):
        idx = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch):
            sel = idx[i:i + batch]
            xb, yb = X[sel], Y[sel]
            pred = policy(xb)
            l = loss(pred, yb)
            opt.zero_grad()
            l.backward()
            opt.step()
    policy.eval()
    return policy


if __name__ == "__main__":
    run_hopper_demo(gui=True)
