#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence
import numpy as np
import matplotlib.pyplot as plt
import time


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


def init_realtime_plot(keys: Sequence[str], window: int = 500, title: str = "", xlabel: str = "Time (step)"):
    """
    Initialize a simple realtime timeline plot.

    Returns (fig, ax, lines_dict, x_axis) where lines_dict maps key -> Line2D
    and x_axis is an array of x positions (relative steps: -window+1 ... 0).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    x_axis = np.arange(-window + 1, 1)
    lines = {}
    # Ensure plotted data arrays are float dtype (np.nan is a float)
    for k in keys:
        line, = ax.plot(x_axis, np.full(x_axis.shape, np.nan, dtype=float), label=k)
        lines[k] = line
    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # draw once
    fig.canvas.draw()
    return fig, ax, lines, x_axis


def update_realtime_plot(lines: Dict[str, Any], x_axis: np.ndarray, series: Dict[str, np.ndarray], window: int = None):
    """
    Update lines in-place from series (dict of np arrays). Series arrays may be shorter than window;
    in that case they are right-aligned in the window and padded with nan on the left.
    Call plt.pause(dt) externally to enforce realtime pacing (dt in seconds).
    """
    if window is None:
        window = x_axis.size
    for k, line in lines.items():
        v = series.get(k, np.array([]))
        if v is None:
            v = np.array([])
        v = np.asarray(v)
        if v.size >= window:
            data = v[-window:]
        else:
            # Pad on the left with NaNs (float dtype) so concatenation yields float array
            pad = np.full(window - v.size, np.nan, dtype=float)
            data = np.concatenate([pad, v])
        # Ensure data is float for plotting (avoids casting warnings)
        data = np.asarray(data, dtype=float)
        # If line length differs, set new x and y data
        line.set_xdata(x_axis)
        line.set_ydata(data)
    # Request a redraw on the active figure
    try:
        fig = next(iter(lines.values())).axes.figure
        fig.canvas.draw_idle()
    except Exception:
        pass
    return
