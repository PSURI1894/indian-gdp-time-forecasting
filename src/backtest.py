"""Rolling-origin backtesting — the honest way to evaluate a forecaster.

The idea (a.k.a. time-series cross-validation):

    |--------- train ---------|--- test (h) ---|
    cutoff_0
       |----------- train ----------|--- test (h) ---|
       cutoff_1
          ... slide the cutoff forward by `step`, refit, forecast again ...

Each fold trains only on the past and forecasts the future, so there is no
leakage. Averaging accuracy over many cutoffs gives a far more trustworthy
estimate than a single train/test split — especially for short series like GDP.

A *forecaster* is any callable:  f(train: pd.Series, h: int) -> array[h]
returning the h-step-ahead point forecast. Baselines, ETS and ARIMA wrappers
all fit this signature, so the same backtest evaluates every model identically.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import metrics as M

Forecaster = Callable[[pd.Series, int], np.ndarray]


def rolling_origin(
    y: pd.Series,
    forecaster: Forecaster,
    initial: int,
    h: int = 1,
    step: int = 1,
    expanding: bool = True,
) -> pd.DataFrame:
    """Run rolling-origin evaluation.

    Parameters
    ----------
    y         : the full series (chronological).
    forecaster: f(train, h) -> array of length h.
    initial   : size of the first training window.
    h         : forecast horizon (steps ahead per fold).
    step      : how far the cutoff advances between folds.
    expanding : True  -> training window grows (anchored at 0);
                False -> sliding window of fixed length `initial`.

    Returns
    -------
    Long DataFrame: [cutoff_idx, cutoff_label, step_ahead, index, y_true, y_pred].
    """
    n = len(y)
    values = y.to_numpy(dtype=float)
    index = y.index
    rows = []
    cutoff = initial
    while cutoff + h <= n:
        start = 0 if expanding else cutoff - initial
        train = y.iloc[start:cutoff]
        yhat = np.asarray(forecaster(train, h), dtype=float)
        if yhat.shape[0] != h:
            raise ValueError(f"forecaster returned {yhat.shape[0]} values, expected {h}")
        for k in range(h):
            j = cutoff + k
            rows.append(
                {
                    "cutoff_idx": cutoff,
                    "cutoff_label": str(index[cutoff - 1]),
                    "step_ahead": k + 1,
                    "index": str(index[j]),
                    "y_true": values[j],
                    "y_pred": yhat[k],
                }
            )
        cutoff += step
    return pd.DataFrame(rows)


def summarize(
    results: pd.DataFrame,
    y_train_for_scale: pd.Series,
    season_length: int = 1,
) -> dict:
    """Overall metrics across all backtest predictions.

    `y_train_for_scale` is used only to compute the MASE denominator (the
    in-sample seasonal-naive error); pass the full series or its early portion.
    """
    return M.evaluate(
        results["y_true"].to_numpy(),
        results["y_pred"].to_numpy(),
        y_train=y_train_for_scale.to_numpy(),
        season_length=season_length,
    )


def summarize_by_horizon(results: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE per step-ahead — shows how error grows with horizon."""
    g = results.groupby("step_ahead")
    out = g.apply(
        lambda d: pd.Series(
            {"MAE": M.mae(d.y_true, d.y_pred), "RMSE": M.rmse(d.y_true, d.y_pred)}
        ),
        include_groups=False,
    )
    return out


def compare(
    y: pd.Series,
    forecasters: dict[str, Forecaster],
    initial: int,
    h: int = 1,
    step: int = 1,
    expanding: bool = True,
    season_length: int = 1,
) -> pd.DataFrame:
    """Backtest several forecasters on identical folds; return a metrics table
    sorted by MASE (lower = better; < 1 beats naive)."""
    rows = {}
    for name, f in forecasters.items():
        res = rolling_origin(y, f, initial=initial, h=h, step=step, expanding=expanding)
        rows[name] = summarize(res, y, season_length=season_length)
    table = pd.DataFrame(rows).T
    sort_key = "MASE" if "MASE" in table.columns else "MAE"
    return table.sort_values(sort_key)
