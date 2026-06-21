"""Classical model wrappers (ETS + ARIMA/SARIMA) exposed as `f(train, h)`
forecasters so they drop straight into `backtest.py` alongside the baselines.

All wrappers optionally fit on log(y) and exponentiate back — the natural scale
for GDP. Note `exp(mean of log)` is the *median* forecast (mildly biased low for
the mean); fine for point accuracy, worth knowing for interval work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _prep(train: pd.Series, use_log: bool) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    return np.log(y) if use_log else y


# --- Exponential smoothing (ETS) --------------------------------------------
def ets_forecaster(trend="add", seasonal=None, seasonal_periods=None,
                   damped_trend=False, use_log=True):
    """Holt / Holt-Winters as an f(train, h) forecaster.

    trend='add', seasonal=None        -> Holt's linear trend
    trend='add', seasonal='add', m=4  -> additive Holt-Winters (quarterly)
    """
    def f(train: pd.Series, h: int) -> np.ndarray:
        y = _prep(train, use_log)
        model = ExponentialSmoothing(
            y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods,
            damped_trend=damped_trend, initialization_method="estimated",
        ).fit()
        fc = np.asarray(model.forecast(h), dtype=float)
        return np.exp(fc) if use_log else fc

    return f


# --- ARIMA / SARIMA ----------------------------------------------------------
def sarima_forecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), use_log=True):
    """SARIMAX point forecaster as f(train, h)."""
    def f(train: pd.Series, h: int) -> np.ndarray:
        y = _prep(train, use_log)
        model = SARIMAX(
            y, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = np.asarray(model.forecast(h), dtype=float)
        return np.exp(fc) if use_log else fc

    return f


def sarima_forecast_pi(train: pd.Series, h: int, order, seasonal_order=(0, 0, 0, 0),
                       use_log=True, alpha=0.05):
    """One-shot forecast WITH prediction intervals.

    Returns (mean, lower, upper) as numpy arrays of length h. If use_log, the
    interval is exponentiated from log space (so it's asymmetric in levels —
    which is correct for a multiplicative process)."""
    y = _prep(train, use_log)
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(h)
    mean = np.asarray(fc.predicted_mean, dtype=float)
    ci = np.asarray(fc.conf_int(alpha=alpha), dtype=float)
    lower, upper = ci[:, 0], ci[:, 1]
    if use_log:
        return np.exp(mean), np.exp(lower), np.exp(upper)
    return mean, lower, upper


def grid_search_sarima(y: pd.Series, d=1, D=1, m=4, max_pq=2, max_PQ=1,
                       use_log=True) -> pd.DataFrame:
    """Tiny AIC grid search (a hand-rolled auto-ARIMA) over (p,q)(P,Q).

    Returns a DataFrame sorted by AIC. Teaches what auto_arima automates:
    fix d/D from the stationarity analysis, then let AIC pick the AR/MA orders.
    """
    yv = _prep(y, use_log)
    rows = []
    for p in range(max_pq + 1):
        for qq in range(max_pq + 1):
            for P in range(max_PQ + 1):
                for Q in range(max_PQ + 1):
                    order = (p, d, qq)
                    sorder = (P, D, Q, m)
                    try:
                        res = SARIMAX(yv, order=order, seasonal_order=sorder,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
                        rows.append({"order": order, "seasonal_order": sorder,
                                     "AIC": res.aic, "BIC": res.bic})
                    except Exception:
                        continue
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
