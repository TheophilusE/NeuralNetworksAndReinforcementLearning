#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple, Optional
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64), activation: str = "tanh"):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        Act = acts.get(activation, nn.Tanh)

        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), Act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_policy(path: str, policy: MLPPolicy):
    torch.save({"state_dict": policy.state_dict()}, path)


def load_policy(path: str, in_dim: int, out_dim: int, hidden_sizes: Tuple[int, ...], activation: str = "tanh") -> MLPPolicy:
    policy = MLPPolicy(in_dim, out_dim, hidden_sizes=hidden_sizes, activation=activation)
    ckpt = torch.load(path, map_location="cpu")
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy
