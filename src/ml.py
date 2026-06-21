"""Gradient-boosted-tree forecasters (LightGBM) exposed as f(train, h), so they
drop into the same `backtest.py` as the classical models.

Two multi-step strategies (the central design choice in ML forecasting):

  RECURSIVE : train ONE 1-step model; to go h steps, predict step 1, feed it back
              in as a lag, predict step 2, ... Errors compound, but it's one model
              and uses the most recent value at every step.
  DIRECT    : train h SEPARATE models, model_k predicting y_{t+k} straight from
              features at t. No compounding, but each model sees less signal and
              ignores its own intermediate predictions.

Both default to a **growth** target (predict Δlog, then cumulate) because trees
cannot extrapolate a trend — see p2_ml notebook 01. Conservative LightGBM params
because GDP gives us only ~80 rows: shallow trees, strong regularisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .features import build_supervised, feature_row

# Sensible small-data defaults (overfitting is the enemy with ~80 rows).
DEFAULT_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=8,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
    verbose=-1, random_state=0,
)


def _log(train, use_log):
    v = np.asarray(train, dtype=float)
    return pd.Series(np.log(v) if use_log else v, index=train.index)


def gbm_recursive_forecaster(lags=(1, 2, 3, 4), rolls=(4,), fourier_order=2, period=4,
                             trend=True, target="growth", use_log=True, params=None):
    """Recursive (a.k.a. iterated) multi-step LightGBM forecaster."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    def f(train: pd.Series, h: int) -> np.ndarray:
        s = _log(train, use_log)
        X, y = build_supervised(s, lags, rolls, fourier_order, period, trend, target)
        model = LGBMRegressor(**p).fit(X, y)

        hist = list(s.to_numpy())          # log-levels, grows as we forecast
        last_period = s.index[-1]
        preds = []
        for step in range(1, h + 1):
            pos = last_period + step
            feat = feature_row(np.asarray(hist), pos, lags, rolls, fourier_order, period, trend)
            row = pd.DataFrame([feat])[X.columns]      # align column order
            yhat = float(model.predict(row)[0])
            level = hist[-1] + yhat if target == "growth" else yhat
            hist.append(level)                          # feed prediction back in
            preds.append(level)
        preds = np.asarray(preds)
        return np.exp(preds) if use_log else preds

    return f


def gbm_direct_forecaster(lags=(1, 2, 3, 4), rolls=(4,), fourier_order=2, period=4,
                          trend=True, target="growth", use_log=True, params=None):
    """Direct multi-step LightGBM forecaster: one model per horizon k."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    def f(train: pd.Series, h: int) -> np.ndarray:
        s = _log(train, use_log)
        vals = s.to_numpy()
        idx = s.index
        start = max(max(lags), max(rolls) if rolls else 0)

        models = {}
        for k in range(1, h + 1):
            rows, tgt = [], []
            for t in range(start, len(vals) - k):
                # features known at origin t (hist up to and incl. t); predict t+k
                rows.append(feature_row(vals[: t + 1], idx[t + k], lags, rolls,
                                        fourier_order, period, trend))
                tgt.append(vals[t + k] - vals[t] if target == "growth" else vals[t + k])
            if not rows:                                # horizon too long for the sample
                models[k] = None
                continue
            models[k] = LGBMRegressor(**p).fit(pd.DataFrame(rows), tgt)

        preds = []
        for k in range(1, h + 1):
            pos = idx[-1] + k
            feat = feature_row(vals, pos, lags, rolls, fourier_order, period, trend)
            m = models[k]
            if m is None:
                preds.append(preds[-1] if preds else vals[-1])
                continue
            yhat = float(m.predict(pd.DataFrame([feat]))[0])
            preds.append(vals[-1] + yhat if target == "growth" else yhat)
        preds = np.asarray(preds)
        return np.exp(preds) if use_log else preds

    return f


def gbm_quantile_forecast(train: pd.Series, h: int, lags=(1, 2, 3, 4), rolls=(4,),
                          fourier_order=2, period=4, trend=True, use_log=True,
                          alpha=0.05, params=None):
    """Point + prediction interval via LightGBM quantile regression (recursive).

    Fits three models (lower/median/upper quantiles) and rolls them forward on a
    growth target. Returns (median, lower, upper) back-transformed to levels.
    A teaser for Project 3's probabilistic forecasting.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    s = _log(train, use_log)
    X, y = build_supervised(s, lags, rolls, fourier_order, period, trend, "growth")
    qs = {"lo": alpha, "mid": 0.5, "hi": 1 - alpha}
    models = {k: LGBMRegressor(objective="quantile", alpha=a, **p).fit(X, y)
              for k, a in qs.items()}

    hist = {k: list(s.to_numpy()) for k in qs}
    last_period = s.index[-1]
    out = {k: [] for k in qs}
    for step in range(1, h + 1):
        pos = last_period + step
        for k in qs:
            feat = feature_row(np.asarray(hist[k]), pos, lags, rolls,
                               fourier_order, period, trend)
            yhat = float(models[k].predict(pd.DataFrame([feat])[X.columns])[0])
            level = hist[k][-1] + yhat
            hist[k].append(level)
            out[k].append(level)
    trans = (np.exp if use_log else (lambda z: z))
    return trans(np.array(out["mid"])), trans(np.array(out["lo"])), trans(np.array(out["hi"]))
