"""Probabilistic forecasting: scoring intervals, and building calibrated ones.

A point forecast answers "what?"; a probabilistic forecast answers "what, and how
sure?". This module gives you (a) the right *scores* for intervals, (b) a backtest
that evaluates interval forecasters, and (c) two ways to build intervals with honest
coverage — **conformal** calibration and **adaptive** conformal inference (ACI).

Convention: `alpha` is the total miss probability, so a (1-alpha) interval. e.g.
alpha=0.1 -> a 90% interval with lower/upper quantiles 0.05 / 0.95.

An *interval forecaster* is g(train, h) -> (mean, lo, hi), each length h — the
probabilistic sibling of the f(train, h) point contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt


# --- scoring & diagnostics ---------------------------------------------------
def pinball_loss(y_true, q_pred, q: float) -> float:
    """Quantile (pinball) loss at level q. The proper score a quantile is fit to."""
    y_true = np.asarray(y_true, float); q_pred = np.asarray(q_pred, float)
    d = y_true - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y_true, lo, hi) -> float:
    """Empirical coverage: fraction of actuals inside [lo, hi]."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= np.asarray(lo, float)) & (y_true <= np.asarray(hi, float))))


def mean_interval_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


def winkler_score(y_true, lo, hi, alpha: float) -> float:
    """Winkler / interval score (lower = better) — a PROPER score for intervals.

    Rewards narrow intervals but penalises misses by 2/alpha times the shortfall, so
    you can't game it by reporting a tiny interval. The honest way to rank methods.
    """
    y_true = np.asarray(y_true, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    width = hi - lo
    below = (lo - y_true) * (y_true < lo)
    above = (y_true - hi) * (y_true > hi)
    return float(np.mean(width + (2 / alpha) * (below + above)))


# --- interval backtest -------------------------------------------------------
def rolling_origin_intervals(y: pd.Series, g, initial: int, h: int = 4, step: int = 1) -> pd.DataFrame:
    """Like backtest.rolling_origin, but the forecaster returns (mean, lo, hi)."""
    n = len(y); vals = y.to_numpy(float); rows = []
    cutoff = initial
    while cutoff + h <= n:
        mean, lo, hi = g(y.iloc[:cutoff], h)
        for k in range(h):
            rows.append({"cutoff": cutoff, "step_ahead": k + 1, "y_true": vals[cutoff + k],
                         "mean": float(mean[k]), "lo": float(lo[k]), "hi": float(hi[k])})
        cutoff += step
    return pd.DataFrame(rows)


def summarize_intervals(df: pd.DataFrame, alpha: float) -> dict:
    return {
        "coverage": coverage(df.y_true, df.lo, df.hi),
        "target": 1 - alpha,
        "mean_width": mean_interval_width(df.lo, df.hi),
        "winkler": winkler_score(df.y_true, df.lo, df.hi, alpha),
    }


# --- ready-made interval forecasters ----------------------------------------
def parametric_interval_forecaster(order, seasonal_order=(0, 0, 0, 0), alpha=0.1, use_log=True):
    """SARIMA's model-based (Gaussian-in-log) interval."""
    from .classical import sarima_forecast_pi

    def g(train, h):
        return sarima_forecast_pi(train, h, order=order, seasonal_order=seasonal_order,
                                  use_log=use_log, alpha=alpha)
    return g


def quantile_interval_forecaster(alpha=0.1, **kw):
    """LightGBM quantile-regression interval (recursive on a growth target)."""
    from .ml import gbm_quantile_forecast

    def g(train, h):
        return gbm_quantile_forecast(train, h, alpha=alpha / 2, **kw)  # alpha/2 per tail
    return g


# --- conformal prediction ----------------------------------------------------
def backtest_residuals(y, point_forecaster, initial, h, step, use_log):
    """Out-of-sample residuals per fold from a rolling-origin backtest of a point
    model. With use_log, residuals are relative (log y_true - log y_pred), so a
    quantile of |resid| becomes a *multiplicative* band. Carries `cutoff_idx` and
    `step_ahead` so callers can split by time or condition on horizon."""
    res = bt.rolling_origin(y, point_forecaster, initial=initial, h=h, step=step)
    if use_log:                                   # relative (multiplicative) residual
        res["resid"] = np.log(res.y_true) - np.log(res.y_pred)
    else:
        res["resid"] = res.y_true - res.y_pred
    return res


def conformal_forecast(y, point_forecaster, h, initial, alpha=0.1, step=1, use_log=True):
    """Calibrated interval for the *final* forecast.

    Calibrate the (1-alpha) quantile of |out-of-sample residual| **per horizon** from
    a rolling-origin backtest, then wrap the point forecast in that band. In log space
    the band is multiplicative (asymmetric in levels) — right for GDP.
    """
    res = backtest_residuals(y, point_forecaster, initial, h, step, use_log)
    q = (res.groupby("step_ahead")["resid"].apply(lambda r: np.quantile(np.abs(r), 1 - alpha))
         .reindex(range(1, h + 1)).to_numpy())
    mean = np.asarray(point_forecaster(y, h), float)
    if use_log:
        return mean, mean * np.exp(-q), mean * np.exp(q)
    return mean, mean - q, mean + q


def conformal_interval_forecaster(point_forecaster, inner_initial=24, alpha=0.1, step=1, use_log=True):
    """Wrap a point forecaster into an interval forecaster g(train, h) -> (mean, lo, hi)
    using conformal calibration on the *train* portion. Use a FAST point model (ETS)
    so the nested calibration backtest stays cheap inside an outer interval backtest.
    """
    def g(train, h):
        init = min(inner_initial, len(train) - h - 1)
        return conformal_forecast(train, point_forecaster, h, initial=init,
                                  alpha=alpha, step=step, use_log=use_log)
    return g


def conformal_split_evaluate(y, point_forecaster, initial, h, alpha=0.1, step=1,
                             use_log=True, test_frac=0.4):
    """Honest out-of-sample coverage of split-conformal intervals.

    Calibrate per-horizon residual quantiles on the *earlier* folds, then form
    intervals and measure coverage on the *later* (held-out) folds — no peeking.
    Returns (per-row test DataFrame, summary dict).
    """
    res = backtest_residuals(y, point_forecaster, initial, h, step, use_log)
    cutoffs = np.sort(res.cutoff_idx.unique())
    split = cutoffs[int(len(cutoffs) * (1 - test_frac))]
    cal, test = res[res.cutoff_idx < split], res[res.cutoff_idx >= split].copy()
    q = cal.groupby("step_ahead")["resid"].apply(lambda r: np.quantile(np.abs(r), 1 - alpha))
    test["hw"] = test.step_ahead.map(q)
    if use_log:
        test["lo"] = test.y_pred * np.exp(-test.hw); test["hi"] = test.y_pred * np.exp(test.hw)
    else:
        test["lo"] = test.y_pred - test.hw; test["hi"] = test.y_pred + test.hw
    return test, summarize_intervals(test, alpha)


def aci_1step(y, point_forecaster, initial, alpha=0.1, gamma=0.02, use_log=True):
    """Adaptive Conformal Inference, 1-step, online.

    Maintains a running miss-rate target by nudging the working level alpha_t after
    every observation: alpha_{t+1} = alpha_t + gamma*(alpha - err_t), where err_t=1 if
    the last actual fell outside its interval. Keeps long-run coverage near 1-alpha
    even when the residual distribution drifts (which time series always do).

    Returns a DataFrame with the realised interval and hit/miss at each step, plus the
    realised coverage (compare to the fixed-alpha conformal baseline).
    """
    vals = y.to_numpy(float); idx = y.index
    resid_hist = []                      # |relative| 1-step residuals seen so far
    alpha_t = alpha
    rows = []
    for t in range(initial, len(vals)):
        train = y.iloc[:t]
        pred = float(point_forecaster(train, 1)[0])
        if resid_hist:
            level = min(max(1 - alpha_t, 0.0), 1.0)
            hw = np.quantile(np.abs(resid_hist), level)
        else:
            hw = 0.0
        lo, hi = (pred * np.exp(-hw), pred * np.exp(hw)) if use_log else (pred - hw, pred + hw)
        y_t = vals[t]
        miss = int(not (lo <= y_t <= hi))
        rows.append({"index": str(idx[t]), "y_true": y_t, "pred": pred,
                     "lo": lo, "hi": hi, "alpha_t": alpha_t, "miss": miss})
        # update working level and residual memory
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - miss), 1e-3, 0.999))
        resid_hist.append((np.log(y_t) - np.log(pred)) if use_log else (y_t - pred))
    df = pd.DataFrame(rows)
    df.attrs["coverage"] = 1 - df.miss.mean()
    df.attrs["target"] = 1 - alpha
    return df
