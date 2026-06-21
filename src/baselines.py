"""Baseline forecasters — the benchmarks every 'real' model must beat.

Each is a `Forecaster`: f(train: pd.Series, h: int) -> np.ndarray of length h.
Plain functions return the array directly; the seasonal one is a small factory
because it needs a season length.

Why baselines matter: a model that cannot beat 'tomorrow = today' (naive) or
'this quarter = same quarter last year' (seasonal naive) is adding complexity
for nothing. MASE is literally defined against the seasonal-naive benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive(train: pd.Series, h: int) -> np.ndarray:
    """Random walk: every future value = the last observed value."""
    return np.repeat(train.iloc[-1], h)


def mean(train: pd.Series, h: int) -> np.ndarray:
    """Every future value = the historical mean (good for stationary series)."""
    return np.repeat(train.mean(), h)


def drift(train: pd.Series, h: int) -> np.ndarray:
    """Naive + average slope: draws a line through first & last point, extends it."""
    n = len(train)
    slope = (train.iloc[-1] - train.iloc[0]) / (n - 1)
    return train.iloc[-1] + slope * np.arange(1, h + 1)


def seasonal_naive(season_length: int):
    """Factory -> forecaster that repeats the last full season.

    e.g. season_length=4 for quarterly: forecast = value from the same quarter
    one year ago, tiled forward across the horizon.
    """
    m = season_length

    def _f(train: pd.Series, h: int) -> np.ndarray:
        last_season = train.iloc[-m:].to_numpy()
        reps = int(np.ceil(h / m))
        return np.tile(last_season, reps)[:h]

    return _f
