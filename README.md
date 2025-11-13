# Neural Networks and Reinforcement Learning

A practical approach to neural networks and reinforcement learning.

# Demos overview

- **Pendulum**
  - PD baseline: `pendulum/pd_controller.py`
  - MLP policy (PD imitation for stability): `pendulum/mlp_policy.py`
- **Double pendulum**
  - PD baseline: `double_pendulum/pd_controller.py`
  - MLP policy (PD imitation): `double_pendulum/mlp_policy.py`
  - Notebook comparison: `double_pendulum/chaos_vs_control.ipynb`
- **Walker/Hopper**
  - Phase-imitation gait demo: `walker/hopper_demo.py`
  - Reward shaping exploration: `walker/reward_shaping.py`

Common utilities:
- Plotting & trajectory: `common/plotting.py`
- Environment wrappers: `common/env_wrappers.py` (lightweight approximations for reproducible demos)
- Models & IO: `common/model_utils.py`


## Directory Structure
```bash
demos/
├── pendulum/
│   ├── pd_controller.py        # Classical PD control demo
│   ├── mlp_policy.py           # Neural network policy demo
│   ├── utils.py                # Shared plotting/logging helpers
│   └── README.md               # Instructions & learning outcomes
│
├── double_pendulum/
│   ├── pd_controller.py        # PD control for double pendulum
│   ├── mlp_policy.py           # Pretrained MLP stabilization demo
│   ├── chaos_vs_control.ipynb  # Notebook comparing PD vs. MLP
│   └── README.md
│
├── walker/
│   ├── hopper_demo.py          # RL-trained hopper gait
│   ├── reward_shaping.py       # Illustrates reward design impact
│   └── README.md
│
├── common/
│   ├── plotting.py             # Graph utilities (angle, torque, reward curves)
│   ├── env_wrappers.py         # PyBullet environment setup helpers
│   └── model_utils.py          # Load/save small MLP models
│
└── README.md                   # Overview of all demos
```