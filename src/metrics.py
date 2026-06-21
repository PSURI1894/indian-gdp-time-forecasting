"""Forecast accuracy metrics, with notes on when each is (mis)leading.

The headline metric for this curriculum is MASE. Quick guide:

  MAE   absolute error in original units. Robust, interpretable, but scale-bound
        (can't compare across series of different magnitude).
  RMSE  penalises large errors more (squared). Same units as data. Sensitive to
        outliers — useful when big misses are disproportionately costly.
  MAPE  mean absolute % error. Popular but treacherous: undefined/explosive near
        zero, and ASYMMETRIC — it penalises over-forecasts more than under-.
  sMAPE "symmetric" MAPE — bounded, but still unstable near zero and not truly
        symmetric. Reported because reviewers expect it.
  MASE  MAE scaled by the in-sample seasonal-naive MAE. Scale-free and the only
        one here that has an absolute interpretation:
            MASE < 1  -> beats the naive benchmark
            MASE > 1  -> worse than naive (your model is not earning its keep)
"""
from __future__ import annotations

import numpy as np


def _align(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def smape(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mase(y_true, y_pred, y_train, season_length: int = 1) -> float:
    """Mean Absolute Scaled Error.

    Scales the forecast MAE by the MAE of a *seasonal-naive* forecast computed
    **on the training set** (the "naive" being: this period = same period one
    season ago). `season_length=1` -> the plain random-walk naive; `=4` for
    quarterly seasonality, `=12` for monthly, etc.
    """
    y_true, y_pred = _align(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float)
    m = season_length
    if len(y_train) <= m:
        raise ValueError("training set too short for the chosen season_length")
    # in-sample one-step seasonal-naive error -> the scaling denominator
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        raise ValueError("naive scale is zero (flat training series)")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate(y_true, y_pred, y_train=None, season_length: int = 1) -> dict:
    """All metrics in one dict. MASE is included only if y_train is given."""
    out = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train, season_length)
    return out
