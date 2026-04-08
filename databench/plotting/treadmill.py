from __future__ import annotations

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_locomotion_bouts(
    t: np.ndarray,
    speed_cms: np.ndarray,
    bouts: Iterable[Tuple[int, int]],
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = None

    if color is None:
        from databench.plotting import get_theme
        color = get_theme().colors[2]  # viridian
    ax.plot(t, speed_cms, color=color, lw=1.2, label="Speed (cm/s)")
    for s, e in bouts:
        ax.axvspan(t[s], t[e], color=color, alpha=0.2)

    ax.set_title(title or "Locomotion speed with detected bouts")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    return fig, ax
