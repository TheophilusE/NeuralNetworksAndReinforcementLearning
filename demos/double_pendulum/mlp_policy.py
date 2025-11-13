#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch

from demos.common.env_wrappers import BulletDoublePendulumEnv
from demos.common.model_utils import MLPPolicy, load_policy, seed_everything
from demos.common.plotting import TrajectoryLogger, plot_timeseries


@dataclass
class Policy2Config:
    hidden_sizes: Tuple[int, int] = (128, 128)
    activation: str = "tanh"
    max_torque: float = 5.0


def run_double_inference(
    steps: int = 4000,
    gui: bool = True,
    seed: int = 99,
    sleep_gui: bool = False,
    ckpt_path: Optional[str] = None,
    policy_cfg: Policy2Config = Policy2Config(),
) -> Dict[str, Any]:
    seed_everything(seed)
    env = BulletDoublePendulumEnv(gui=gui, seed=seed)
    obs_dim = env.observation_dim
    act_dim = env.action_dim

    if ckpt_path is None:
        policy = MLPPolicy(obs_dim, act_dim, hidden_sizes=policy_cfg.hidden_sizes, activation=policy_cfg.activation)
        policy = quick_pd2_imitation(env, policy, epochs=30, batch_size=256)
    else:
        policy = load_policy(ckpt_path, obs_dim, act_dim, policy_cfg.hidden_sizes, policy_cfg.activation)

    traj = TrajectoryLogger(keys=["t", "th1", "th2", "dth1", "dth2", "tau1", "tau2", "reward"], capacity=steps + 1)
    s = env.reset(randomize=True)
    for t in range(steps):
        with torch.no_grad():
            obs = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            tau = policy(obs).squeeze(0).cpu().numpy()
        tau = np.clip(tau, -policy_cfg.max_torque, policy_cfg.max_torque)
        s, r, d, info = env.step(tau)
        th1, th2, dth1, dth2 = s
        traj.add(t=t, th1=th1, th2=th2, dth1=dth1, dth2=dth2, tau1=float(tau[0]), tau2=float(tau[1]), reward=r)
        if sleep_gui and gui:
            import time
            time.sleep(env.dt)
        if d:
            break

    env.close()
    fig = plot_timeseries(traj.to_series(["th1", "th2", "dth1", "dth2", "tau1", "tau2"]), title="Double Pendulum MLP Policy")
    return {"trajectory": traj, "figure": fig, "policy": policy}


def quick_pd2_imitation(env: BulletDoublePendulumEnv, policy: MLPPolicy, epochs: int = 20, batch_size: int = 128):
    from demos.double_pendulum.pd_controller import PDController2, PDGains2, PDConfig2
    pd = PDController2(PDGains2(), PDConfig2())

    X, Y = [], []
    for _ in range(3000):
        s = env.reset(randomize=True)
        for _ in range(30):
            tau = np.array(pd.act(s), dtype=np.float32)
            X.append(s)
            Y.append(tau)
            s, _, d, _ = env.step(tau)
            if d:
                break

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
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
    run_double_inference(gui=True)
