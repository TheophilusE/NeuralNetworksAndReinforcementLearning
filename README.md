# Neural Networks and Reinforcement Learning

A practical approach to neural networks and reinforcement learning.

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