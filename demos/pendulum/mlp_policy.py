#!/usr/bin/env python3
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from demos.common.env_wrappers import BulletPendulumEnv
from demos.common.model_utils import MLPPolicy, load_policy, save_policy, seed_everything
import matplotlib.pyplot as plt
from demos.common.plotting import (
    TrajectoryLogger,
    plot_timeseries,
    init_realtime_plot,
    update_realtime_plot,
)


@dataclass
class PolicyConfig:
    hidden_sizes: Tuple[int, int] = (64, 64)
    activation: str = "tanh"
    max_torque: float = 5.0


def run_inference(
    steps: int = 2000,
    gui: bool = True,
    seed: int = 123,
    sleep_gui: bool = False,
    ckpt_path: Optional[str] = None,
    policy_cfg: PolicyConfig = PolicyConfig(),
    realtime: bool = True,
    window_secs: float = 5.0,
) -> Dict[str, Any]:
    seed_everything(seed)
    env = BulletPendulumEnv(gui=gui, seed=seed)
    obs_dim = env.observation_dim
    act_dim = 1

    if ckpt_path is not None:
        policy = load_policy(ckpt_path, obs_dim, act_dim, policy_cfg.hidden_sizes, policy_cfg.activation)
    else:
        policy = MLPPolicy(obs_dim, act_dim, hidden_sizes=policy_cfg.hidden_sizes, activation=policy_cfg.activation)

        # Optional: quick supervised "imitation" of a PD baseline for stability
        # This produces a usable policy without long RL training.
        policy = quick_pd_imitation(env, policy, epochs=20, batch_size=256)

    window_steps = max(10, int(window_secs / env.dt))
    traj = TrajectoryLogger(keys=["t", "theta", "theta_dot", "tau", "reward"], capacity=10 * window_steps)

    state = env.reset(randomize=True)
    t = 0
    fig = None
    if realtime:
        fig, ax, lines, x_axis = init_realtime_plot(["theta", "theta_dot", "tau"], window=window_steps, title="Pendulum MLP Policy (realtime)")

    try:
        while True:
            theta, theta_dot = state
            with torch.no_grad():
                obs = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                tau = policy(obs).squeeze(0).cpu().numpy()[0]
            tau = float(np.clip(tau, -policy_cfg.max_torque, policy_cfg.max_torque))
            state, reward, done, info = env.step(tau)

            traj.add(t=t, theta=theta, theta_dot=theta_dot, tau=tau, reward=reward)

            if realtime:
                series = traj.to_series(["theta", "theta_dot", "tau"])
                update_realtime_plot(lines, x_axis, series, window=window_steps)
                plt.pause(env.dt if sleep_gui is False else env.dt)
            else:
                if gui and sleep_gui:
                    time.sleep(env.dt)

            t += 1
            if done or t % steps == 0:
                state = env.reset(randomize=True)
                continue

    except KeyboardInterrupt:
        pass

    finally:
        env.close()

    fig = plot_timeseries(traj.to_series(["theta", "theta_dot", "tau"]), title="Pendulum MLP Policy", xlabel="Time (step)")
    return {"trajectory": traj, "figure": fig, "policy": policy}


def quick_pd_imitation(env: BulletPendulumEnv, policy: MLPPolicy, epochs: int = 10, batch_size: int = 128):
    """
    Generate synthetic dataset using a PD controller and train policy to mimic it.
    This is fast and yields a stable policy suitable for workshop inference.
    """
    # Generate dataset
    from demos.pendulum.pd_controller import PDController, PDGains, PDConfig

    pd = PDController(gains=PDGains(kp=20.0, kd=2.5), cfg=PDConfig())
    X, Y = [], []
    for _ in range(2000):
        s = env.reset(randomize=True)
        for _ in range(50):
            tau = pd.act(s[0], s[1])
            X.append(s)
            Y.append([tau])
            s, _, done, _ = env.step(tau)
            if done:
                break

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    policy.train()
    for _ in range(epochs):
        idx = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch_size):
            sel = idx[i:i + batch_size]
            xb, yb = X[sel], Y[sel]
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    policy.eval()
    return policy


if __name__ == "__main__":
    res = run_inference(gui=True, sleep_gui=False, ckpt_path=None)
