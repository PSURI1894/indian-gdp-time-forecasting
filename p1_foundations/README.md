# Project 1 — Foundations & Classical Forecasting

Master the **full** classical toolkit on real India GDP data — baselines, ETS,
ARIMA/SARIMA, Theta, Prophet, and ARIMAX. The throughline: **you can't improve what
you can't measure**, so we build the evaluation harness *before* the models, then
let a like-for-like backtest decide which method wins (and which famous ones don't).

## Notebooks (run in order, kernel = *Python (TSF)*)

| # | notebook | what you learn |
|---|----------|----------------|
| 00 | `00_first_look.ipynb` | Why `log(GDP)` and growth rates; reading levels vs log-levels vs growth; the chronological-split rule |
| 01 | `01_decomposition_stationarity.ipynb` | Additive vs multiplicative decomposition, STL, seasonal strength; ADF + KPSS stationarity testing; reading ACF/PACF for ARIMA orders |
| 02 | `02_baselines_evaluation.ipynb` | Baselines, the MAE/RMSE/MAPE/sMAPE/**MASE** family, and **rolling-origin backtesting** — the yardstick every later model must beat |
| 03 | `03_exponential_smoothing.ipynb` | SES → Holt → Holt-Winters; matching model components to diagnosed structure |
| 04 | `04_arima_sarima.ipynb` | Box-Jenkins: ARIMA on annual log-GDP, SARIMA on quarterly NSA, AIC grid search, residual diagnostics, intervals |
| 05 | `05_annual_forecast.ipynb` | The full pipeline on the **annual** series end-to-end — a tiny-sample stress test; why small data rewards lean models; the multi-step MASE caveat |
| 06 | `06_prophet_theta.ipynb` | **Prophet** (additive decomposition, changepoints, components plot) and the **Theta** method (M3 winner); both backtested vs the field |
| 07 | `07_arimax.ipynb` | **ARIMAX**: Fourier seasonality (dynamic harmonic regression) vs seasonal differencing, and a **COVID intervention dummy** that cleans the 2020 shock |
| 08 | `08_capstone.ipynb` | Full bake-off of **all** methods; pick the winner by backtest, refit on all data, forecast with intervals, sanity-check & caveat |

## Key findings so far (from the executed notebooks)

- India real GDP grows ~**6.6%/yr** (YoY); shocks visible at 1991, 2008, and a
  **−26% COVID trough in 2020 Q2**.
- Quarterly seasonality is **strong** (seasonal strength ≈ 0.74): Jan–Mar runs
  ~**+5.8%** above trend, Apr–Jun ~**−4%** below.
- Stationarity tests point to differencing orders **d = 1, D = 1, m = 4** for SARIMA.
- All naive baselines sit near **MASE ≈ 1**; `mean` is hopeless (the series trends
  hard), `drift`/`seasonal_naive` are competitive — that's the bar to beat.
- **Full 9-model bake-off** (h=4, rolling origin): winner **SARIMA(1,1,1)(0,1,0)₄,
  MASE 0.573** (AIC *and* out-of-sample agree) → Holt-Winters 0.608 → HW-damped 0.650
  → SARIMA(1,1,1)(1,1,1)₄ 0.655 → **Prophet 0.729 → Theta 0.734** → ARIMAX-Fourier
  0.973 → baselines (drift 1.02, seasonal-naive 1.27). Famous ≠ best: Prophet/Theta
  beat the baselines but lose to a tuned SARIMA on this tidy, short series.
- **Annual series** (~65 pts, tiny-sample): lean models (Holt, ARIMA, Theta) cut
  error to ~⅓ of naive; at h=3 every MASE > 1 (a 1-step scale vs a 3-step task).
- **ARIMAX intervention:** a COVID dummy shrinks the 2020 residual **31.4 → 10.5 pp**
  (coef −23 pp), residual std 4.36 → 3.18 — exog earns its keep in *diagnostics*.
- The final 8-quarter forecast implies **~6.6% YoY growth**, matching history.

## Regenerate

```powershell
.venv\Scripts\python.exe p1_foundations/build_p1.py            # write notebooks
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=tsf p1_foundations/notebooks/*.ipynb
```

See [`DEEP_DIVE.md`](DEEP_DIVE.md) for the math behind decomposition, stationarity,
and MASE.
