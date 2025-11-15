#!/usr/bin/env python3
"""
Compare PD controller vs MLP policy on the single inverted pendulum.
Runs both controllers on identical initial states and plots theta/time and torque/time.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from demos.common.env_wrappers import BulletPendulumEnv
from demos.pendulum.pd_controller import PDController, PDGains, PDConfig
from demos.pendulum.mlp_policy import PolicyConfig, quick_pd_imitation
from demos.common.model_utils import MLPPolicy, load_policy, seed_everything
from demos.common.plotting import TrajectoryLogger, plot_timeseries


def compare_pd_vs_mlp(
    steps: int = 2000,
    seed: int = 123,
    gui: bool = False,
    ckpt_path: Optional[str] = None,
    policy_cfg: PolicyConfig = PolicyConfig(),
    pd_gains: PDGains = PDGains(),
    pd_cfg: PDConfig = PDConfig(),
) -> Dict[str, Any]:
    seed_everything(seed)

    # Create two envs. Use GUI optionally; running multiple GUI clients is fragile,
    # so default is headless (gui=False). We will plot results using matplotlib.
    env_pd = BulletPendulumEnv(gui=gui, seed=seed)
    env_mlp = BulletPendulumEnv(gui=gui, seed=seed + 1)

    # Create controllers
    pd_ctrl = PDController(pd_gains, pd_cfg)

    obs_dim = env_mlp.observation_dim
    act_dim = 1
    if ckpt_path is not None:
        policy = load_policy(ckpt_path, obs_dim, act_dim, policy_cfg.hidden_sizes, policy_cfg.activation)
    else:
        policy = MLPPolicy(obs_dim, act_dim, hidden_sizes=policy_cfg.hidden_sizes, activation=policy_cfg.activation)
        # quick imitation to get a usable policy
        policy = quick_pd_imitation(env_mlp, policy, epochs=20, batch_size=256)

    # Ensure both envs start from the same initial state for fair comparison
    s0 = env_pd.reset(randomize=True)
    try:
        env_mlp._state = np.array(s0, dtype=np.float32)
    except Exception:
        # best-effort: call reset(randomize=False) then overwrite
        env_mlp.reset(randomize=False)
        env_mlp._state = np.array(s0, dtype=np.float32)

    traj_pd = TrajectoryLogger(keys=["t", "theta", "theta_dot", "tau", "reward"], capacity=steps + 10)
    traj_mlp = TrajectoryLogger(keys=["t", "theta", "theta_dot", "tau", "reward"], capacity=steps + 10)

    s_pd = s0
    s_mlp = tuple(env_mlp._state)

    for t in range(steps):
        theta_pd, thdot_pd = s_pd
        tau_pd = pd_ctrl.act(theta_pd, thdot_pd)
        s_pd, r_pd, d_pd, _ = env_pd.step(tau_pd)
        traj_pd.add(t=t, theta=theta_pd, theta_dot=thdot_pd, tau=tau_pd, reward=r_pd)

        theta_m, thdot_m = s_mlp
        with torch.no_grad():
            obs = torch.tensor(s_mlp, dtype=torch.float32).unsqueeze(0)
            tau_m = policy(obs).squeeze(0).cpu().numpy()[0]
        tau_m = float(np.clip(tau_m, -policy_cfg.max_torque, policy_cfg.max_torque))
        s_mlp, r_mlp, d_mlp, _ = env_mlp.step(tau_m)
        traj_mlp.add(t=t, theta=theta_m, theta_dot=thdot_m, tau=tau_m, reward=r_mlp)

        # keep s_mlp for next iter

    env_pd.close()
    env_mlp.close()

    # Plot comparison
    series_pd = traj_pd.to_series(["theta", "theta_dot", "tau"])
    series_mlp = traj_mlp.to_series(["theta", "theta_dot", "tau"])

    # produce overlay plots
    fig1 = plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(series_pd.index, series_pd["theta"], label="PD")
    plt.plot(series_mlp.index, series_mlp["theta"], label="MLP", linestyle="--")
    plt.ylabel("theta")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(series_pd.index, series_pd["theta_dot"], label="PD")
    plt.plot(series_mlp.index, series_mlp["theta_dot"], label="MLP", linestyle="--")
    plt.ylabel("theta_dot")

    plt.subplot(3, 1, 3)
    plt.plot(series_pd.index, series_pd["tau"], label="PD")
    plt.plot(series_mlp.index, series_mlp["tau"], label="MLP", linestyle="--")
    plt.ylabel("tau")
    plt.xlabel("time (step)")
    plt.suptitle("PD vs MLP: Pendulum")

    plt.tight_layout()

    return {"traj_pd": traj_pd, "traj_mlp": traj_mlp, "figure": fig1, "policy": policy}


if __name__ == "__main__":
    res = compare_pd_vs_mlp(steps=1500, gui=False)
    plt.show()
