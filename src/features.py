"""Turn a 1-D time series into a supervised-learning table.

The whole premise of ML forecasting: predict y_t from features built out of the
*past* — lagged values, rolling summaries, and deterministic calendar/Fourier
terms. The hard part isn't the model, it's building those features **without
leakage** and computing them **identically** at train time and forecast time.

This module guarantees that consistency with ONE function, `feature_row`, used
both to build the training matrix and to assemble each step of a recursive
forecast. If a feature is wrong, it's wrong in both places — never a silent skew.

Everything operates on **log GDP** (the caller passes log-levels): lags/rolls of
the log are stable, and a 'growth' target becomes a simple first difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_row(hist: np.ndarray, pos, lags, rolls, fourier_order, period, trend) -> dict:
    """Features for predicting the value at calendar position `pos`.

    Parameters
    ----------
    hist  : 1-D array of log-levels observed strictly BEFORE `pos`
            (so hist[-1] is the most recent known value — no leakage).
    pos   : the pandas Period being predicted (its .quarter drives seasonality).
    lags  : which lags to include, e.g. (1,2,3,4).
    rolls : rolling-window sizes for mean/std, e.g. (4,).
    fourier_order : number of sin/cos seasonal harmonics (0 = none).
    period: seasonal period (4 for quarterly).
    trend : include a linear time index (len(hist)) — NB trees can't extrapolate it.
    """
    f = {}
    for L in lags:
        f[f"lag{L}"] = hist[-L]
    for w in rolls:
        window = hist[-w:]
        f[f"rmean{w}"] = float(np.mean(window))
        f[f"rstd{w}"] = float(np.std(window))
    if trend:
        f["trend"] = float(len(hist))
    if fourier_order:
        q = pos.quarter  # 1..4
        for k in range(1, fourier_order + 1):
            angle = 2 * np.pi * k * (q - 1) / period
            f[f"sin{k}"] = float(np.sin(angle))
            f[f"cos{k}"] = float(np.cos(angle))
    return f


def build_supervised(logy: pd.Series, lags=(1, 2, 3, 4), rolls=(4,),
                     fourier_order=2, period=4, trend=True, target="growth"):
    """Build the training design matrix X and target vector from a log-level series.

    target='level'  -> predict next log-level  y_t
    target='growth' -> predict next log-growth  y_t - y_{t-1}  (stationary → trees OK)

    Returns (X: DataFrame, y: Series), both indexed by the predicted period.
    The first `start` rows are dropped (not enough history for the longest lag/roll).
    """
    vals = np.asarray(logy, dtype=float)
    idx = logy.index
    start = max(max(lags), max(rolls) if rolls else 0)
    rows, tgt, keep = [], [], []
    for t in range(start, len(vals)):
        rows.append(feature_row(vals[:t], idx[t], lags, rolls, fourier_order, period, trend))
        tgt.append(vals[t] if target == "level" else vals[t] - vals[t - 1])
        keep.append(idx[t])
    X = pd.DataFrame(rows, index=keep)
    y = pd.Series(tgt, index=keep, name=target)
    return X, y
