#!/usr/bin/env python3
"""
Compare PD controller vs MLP policy on the double inverted pendulum.
Runs both controllers on identical initial states and plots angle and torque comparisons.
"""
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from demos.common.env_wrappers import BulletDoublePendulumEnv
from demos.double_pendulum.pd_controller import PDController2, PDGains2, PDConfig2
from demos.double_pendulum.mlp_policy import Policy2Config, quick_pd2_imitation
from demos.common.model_utils import MLPPolicy, load_policy, seed_everything
from demos.common.plotting import TrajectoryLogger


def compare_pd_vs_mlp_double(
    steps: int = 3000,
    seed: int = 99,
    gui: bool = False,
    ckpt_path: Optional[str] = None,
    policy_cfg: Policy2Config = Policy2Config(),
    pd_gains: PDGains2 = PDGains2(),
    pd_cfg: PDConfig2 = PDConfig2(),
) -> Dict[str, Any]:
    seed_everything(seed)

    env_pd = BulletDoublePendulumEnv(gui=gui, seed=seed)
    env_mlp = BulletDoublePendulumEnv(gui=gui, seed=seed + 1)

    pd_ctrl = PDController2(pd_gains, pd_cfg)

    obs_dim = env_mlp.observation_dim
    act_dim = env_mlp.action_dim

    if ckpt_path is not None:
        policy = load_policy(ckpt_path, obs_dim, act_dim, policy_cfg.hidden_sizes, policy_cfg.activation)
    else:
        policy = MLPPolicy(obs_dim, act_dim, hidden_sizes=policy_cfg.hidden_sizes, activation=policy_cfg.activation)
        policy = quick_pd2_imitation(env_mlp, policy, epochs=30, batch_size=256)

    s0 = env_pd.reset(randomize=True)
    try:
        env_mlp._s = np.array(s0, dtype=np.float32)
    except Exception:
        env_mlp.reset(randomize=False)
        env_mlp._s = np.array(s0, dtype=np.float32)

    traj_pd = TrajectoryLogger(keys=["t", "th1", "th2", "dth1", "dth2", "tau1", "tau2", "reward"], capacity=steps + 10)
    traj_mlp = TrajectoryLogger(keys=["t", "th1", "th2", "dth1", "dth2", "tau1", "tau2", "reward"], capacity=steps + 10)

    s_pd = s0
    s_mlp = tuple(env_mlp._s)

    for t in range(steps):
        tau1, tau2 = pd_ctrl.act(s_pd)
        s_pd, r_pd, d_pd, _ = env_pd.step(np.array([tau1, tau2]))
        th1, th2, dth1, dth2 = s_pd
        traj_pd.add(t=t, th1=th1, th2=th2, dth1=dth1, dth2=dth2, tau1=tau1, tau2=tau2, reward=r_pd)

        with torch.no_grad():
            obs = torch.tensor(s_mlp, dtype=torch.float32).unsqueeze(0)
            tau = policy(obs).squeeze(0).cpu().numpy()
        tau = np.clip(tau, -policy_cfg.max_torque, policy_cfg.max_torque)
        s_mlp, r_mlp, d_mlp, _ = env_mlp.step(tau)
        th1m, th2m, dth1m, dth2m = s_mlp
        traj_mlp.add(t=t, th1=th1m, th2=th2m, dth1=dth1m, dth2=dth2m, tau1=float(tau[0]), tau2=float(tau[1]), reward=r_mlp)

    env_pd.close()
    env_mlp.close()

    series_pd = traj_pd.to_series(["th1", "th2", "tau1", "tau2"])
    series_mlp = traj_mlp.to_series(["th1", "th2", "tau1", "tau2"])

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(series_pd.index, series_pd["th1"], label="PD")
    plt.plot(series_mlp.index, series_mlp["th1"], label="MLP", linestyle="--")
    plt.ylabel("th1")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(series_pd.index, series_pd["th2"], label="PD")
    plt.plot(series_mlp.index, series_mlp["th2"], label="MLP", linestyle="--")
    plt.ylabel("th2")

    plt.subplot(4, 1, 3)
    plt.plot(series_pd.index, series_pd["tau1"], label="PD")
    plt.plot(series_mlp.index, series_mlp["tau1"], label="MLP", linestyle="--")
    plt.ylabel("tau1")

    plt.subplot(4, 1, 4)
    plt.plot(series_pd.index, series_pd["tau2"], label="PD")
    plt.plot(series_mlp.index, series_mlp["tau2"], label="MLP", linestyle="--")
    plt.ylabel("tau2")
    plt.xlabel("time (step)")

    plt.suptitle("PD vs MLP: Double Pendulum")
    plt.tight_layout()

    return {"traj_pd": traj_pd, "traj_mlp": traj_mlp, "figure": fig, "policy": policy}


if __name__ == "__main__":
    res = compare_pd_vs_mlp_double(steps=2000, gui=False)
    plt.show()
