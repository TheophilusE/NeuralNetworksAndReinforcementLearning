#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class TrajectoryLogger:
    keys: List[str]
    capacity: int

    def __post_init__(self):
        self._buf = {k: [] for k in self.keys}

    def add(self, **kwargs):
        for k in self.keys:
            if k in kwargs:
                self._buf[k].append(kwargs[k])
        # Enforce capacity
        for k in self.keys:
            if len(self._buf[k]) > self.capacity:
                self._buf[k].pop(0)

    def to_series(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        return {k: np.array(self._buf[k]) for k in keys if k in self._buf}


def plot_timeseries(series: Dict[str, np.ndarray], title: str = "", xlabel: str = "Step"):
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, v in series.items():
        ax.plot(v, label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
