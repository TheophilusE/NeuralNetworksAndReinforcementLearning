#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from demos.common.env_wrappers import BulletHopperEnv


@dataclass
class RewardWeights:
    forward: float = 1.0
    energy: float = 0.002
    stability: float = 0.5
    foot_clearance: float = 0.1


def shaped_reward(info: Dict[str, Any], weights: RewardWeights) -> float:
    """
    Compute a reward illustrating trade-offs: forward progress vs energy vs stability.
    """
    vx = info.get("base_vel", (0.0, 0.0, 0.0))[0]
    torque_cost = info.get("torque_cost", 0.0)
    tilt = abs(info.get("base_pitch", 0.0))
    foot_h = info.get("foot_height", 0.0)

    return (
        weights.forward * vx
        - weights.energy * torque_cost
        - weights.stability * tilt
        + weights.foot_clearance * foot_h
    )


def demo_reward_shaping(gui: bool = True):
    env = BulletHopperEnv(gui=gui, seed=123)
    s = env.reset(randomize=True)
    w = RewardWeights()
    for _ in range(1000):
        a = env.random_action()
        s, _, d, info = env.step(a)
        r = shaped_reward(info, w)
        env.debug_text(f"Reward: {r:.3f}")
        if d:
            break
    env.close()


if __name__ == "__main__":
    demo_reward_shaping(gui=True)
