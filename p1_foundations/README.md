# Project 1 — Foundations & Classical Forecasting

Master the classical toolkit on real India GDP data. The throughline: **you can't
improve what you can't measure**, so we build the evaluation harness *before* the
models, then prove that ETS and ARIMA actually beat the baselines.

## Notebooks (run in order, kernel = *Python (TSF)*)

| # | notebook | what you learn |
|---|----------|----------------|
| 00 | `00_first_look.ipynb` | Why `log(GDP)` and growth rates; reading levels vs log-levels vs growth; the chronological-split rule |
| 01 | `01_decomposition_stationarity.ipynb` | Additive vs multiplicative decomposition, STL, seasonal strength; ADF + KPSS stationarity testing; reading ACF/PACF for ARIMA orders |
| 02 | `02_baselines_evaluation.ipynb` | Baselines, the MAE/RMSE/MAPE/sMAPE/**MASE** family, and **rolling-origin backtesting** — the yardstick every later model must beat |
| 03 | `03_exponential_smoothing.ipynb` | SES → Holt → Holt-Winters; matching model components to diagnosed structure |
| 04 | `04_arima_sarima.ipynb` | Box-Jenkins: ARIMA on annual log-GDP, SARIMA on quarterly NSA, AIC grid search, residual diagnostics, intervals |
| 05 | `05_capstone.ipynb` | Pick the winner by backtest, refit on all data, forecast with intervals, sanity-check & caveat |

## Key findings so far (from the executed notebooks)

- India real GDP grows ~**6.6%/yr** (YoY); shocks visible at 1991, 2008, and a
  **−26% COVID trough in 2020 Q2**.
- Quarterly seasonality is **strong** (seasonal strength ≈ 0.74): Jan–Mar runs
  ~**+5.8%** above trend, Apr–Jun ~**−4%** below.
- Stationarity tests point to differencing orders **d = 1, D = 1, m = 4** for SARIMA.
- All naive baselines sit near **MASE ≈ 1**; `mean` is hopeless (the series trends
  hard), `drift`/`seasonal_naive` are competitive — that's the bar to beat.
- **Both real models clear the bar decisively.** Backtest winner:
  **SARIMA(1,1,1)(0,1,0)₄, MASE 0.612** (AIC *and* out-of-sample agree), with
  Holt-Winters close behind (0.640). The 8-quarter forecast implies **~6.6% YoY
  growth**, matching history — a clean sanity check.

## Regenerate

```powershell
.venv\Scripts\python.exe p1_foundations/build_p1.py            # write notebooks
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=tsf p1_foundations/notebooks/*.ipynb
```

See [`DEEP_DIVE.md`](DEEP_DIVE.md) for the math behind decomposition, stationarity,
and MASE.
