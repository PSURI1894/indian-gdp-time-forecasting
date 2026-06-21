"""Small plotting helpers shared by the notebooks. Kept thin on purpose — the
notebooks should show the matplotlib so you learn it, but these remove the
boilerplate that would otherwise be copy-pasted into every cell.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def setup():
    """A clean, readable default style."""
    plt.rcParams.update(
        {
            "figure.figsize": (11, 4.5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _x(series: pd.Series):
    """Use a timestamp axis for PeriodIndex, else the index as-is."""
    idx = series.index
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp(how="start")
    return idx


def plot_forecast(train, test=None, pred=None, lower=None, upper=None,
                  title="", ax=None):
    """Train history + (optional) actual test + (optional) forecast & interval."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(_x(train), train.values, label="train", color="#264653")
    if test is not None:
        ax.plot(_x(test), test.values, label="actual", color="#2a9d8f")
    if pred is not None:
        ax.plot(_x(pred), pred.values, label="forecast", color="#e76f51", lw=2)
    if lower is not None and upper is not None:
        ax.fill_between(_x(pred), lower.values, upper.values,
                        color="#e76f51", alpha=0.2, label="interval")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


def save(fig, name: str):
    """Save a figure under reports/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / (name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path
