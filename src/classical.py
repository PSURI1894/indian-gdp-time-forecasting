"""Classical model wrappers (ETS + ARIMA/SARIMA) exposed as `f(train, h)`
forecasters so they drop straight into `backtest.py` alongside the baselines.

All wrappers optionally fit on log(y) and exponentiate back — the natural scale
for GDP. Note `exp(mean of log)` is the *median* forecast (mildly biased low for
the mean); fine for point accuracy, worth knowing for interval work.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess, Fourier
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet (and its Stan backend) are noisy on import/fit — silence them hard.
# setLevel alone doesn't stick (cmdstanpy re-emits via its own handler), so we
# disable the loggers outright; genuine failures still raise exceptions.
for _name in ("cmdstanpy", "prophet"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)
    _lg.disabled = True


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


# --- Theta method ------------------------------------------------------------
def theta_forecaster(period=4, use_log=True):
    """The Theta method (Assimakopoulos & Nikolopoulos) as f(train, h).

    Decomposes the series into 'theta lines' (it amounts to SES on the deseasonalised
    series plus half the long-run linear trend, then re-seasonalised). Famous for
    *winning the M3 competition* despite its simplicity — a strong, cheap benchmark.
    Needs a Series with a quarterly PeriodIndex so it can deseasonalise.
    """
    def f(train: pd.Series, h: int) -> np.ndarray:
        s = pd.Series(np.log(train.to_numpy(dtype=float)) if use_log else
                      train.to_numpy(dtype=float), index=train.index)
        fc = ThetaModel(s, period=period).fit().forecast(h)
        fc = np.asarray(fc, dtype=float)
        return np.exp(fc) if use_log else fc

    return f


# --- Prophet -----------------------------------------------------------------
def prophet_forecaster(seasonality_mode="multiplicative", freq="QS"):
    """Facebook Prophet as f(train, h). Lazily imported so the rest of the toolkit
    works without Prophet installed.

    Prophet is an additive *decomposition* model: y = trend + seasonality + holidays
    + noise, fit by a Bayesian backend. We give it the quarter-start timestamps it
    expects and read back the last h `yhat` values.
    """
    def f(train: pd.Series, h: int) -> np.ndarray:
        from prophet import Prophet  # lazy: only needed if this model is used

        ds = train.index.to_timestamp(how="start")
        df = pd.DataFrame({"ds": ds, "y": train.to_numpy(dtype=float)})
        m = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=True,
                    weekly_seasonality=False, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=h, freq=freq)
        return m.predict(future)["yhat"].to_numpy(dtype=float)[-h:]

    return f


# --- ARIMAX (ARIMA with exogenous regressors) --------------------------------
def _build_exog(train_index, h, use_fourier, fourier_order, period, covid_dummy):
    """Build aligned in-sample + out-of-sample exogenous regressors.

    Two kinds of exog, both deterministic (no extra data, fully reproducible):
      * Fourier terms  -> smooth seasonality as regressors (dynamic harmonic
        regression). An alternative to seasonal differencing.
      * COVID dummy     -> an *intervention* flag (1 in 2020 Q2/Q3) that lets the
        model 'explain away' the pandemic shock instead of contaminating the trend.
    Swapping in a real driver (rates, monsoon, IIP) would slot in identically.
    """
    freq = train_index.freqstr
    future_index = pd.period_range(train_index[-1] + 1, periods=h, freq=freq)
    parts_in, parts_out = [], []
    if use_fourier:
        dp = DeterministicProcess(train_index, additional_terms=[Fourier(period, fourier_order)])
        parts_in.append(dp.in_sample())
        parts_out.append(dp.out_of_sample(h))
    if covid_dummy:
        def pulse(idx):
            flag = ((idx.year == 2020) & (idx.quarter.isin([2, 3]))).astype(float)
            return pd.DataFrame({"covid": flag}, index=idx)
        parts_in.append(pulse(train_index))
        parts_out.append(pulse(future_index))
    Xin = pd.concat(parts_in, axis=1) if parts_in else None
    Xout = pd.concat(parts_out, axis=1) if parts_out else None
    return Xin, Xout


def arimax_forecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), use_fourier=True,
                      fourier_order=2, period=4, covid_dummy=False, use_log=True):
    """ARIMA with exogenous regressors as f(train, h).

    With Fourier exog and no seasonal term, this is *dynamic harmonic regression*:
    seasonality is modelled smoothly by sines/cosines while ARIMA handles the rest.
    """
    def f(train: pd.Series, h: int) -> np.ndarray:
        y = _prep(train, use_log)
        Xin, Xout = _build_exog(train.index, h, use_fourier, fourier_order, period, covid_dummy)
        model = SARIMAX(y, exog=None if Xin is None else Xin.to_numpy(),
                        order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = np.asarray(model.forecast(h, exog=None if Xout is None else Xout.to_numpy()),
                        dtype=float)
        return np.exp(fc) if use_log else fc

    return f
