# Project 3 — Probabilistic Forecasting

Stop shipping naked point forecasts. This project is about **uncertainty done
honestly**: scoring intervals properly, learning their tails, and guaranteeing their
coverage. The throughline: *a "90% interval" is a lie until you've measured that it
covers ~90% out-of-sample.*

## Notebooks (run in order, kernel = *Python (TSF)*)

| # | notebook | what you learn |
|---|----------|----------------|
| 00 | `00_evaluating_uncertainty.ipynb` | Why intervals; **coverage**, sharpness (width), and the proper **Winkler/interval score**; backtest P1/P2 intervals' real coverage |
| 01 | `01_quantile_regression.ipynb` | Pinball loss, **quantile crossing** and the sort-fix, and the **calibration (reliability) diagram** |
| 02 | `02_conformal_prediction.ipynb` | **Conformal prediction** (distribution-free coverage guarantee), the exchangeability problem in time series, and **Adaptive Conformal Inference (ACI)** for drift |
| 03 | `03_capstone_probabilistic.ipynb` | Rank interval methods by Winkler; ship a calibrated **fan chart** (50/80/90%) |

## Key findings

- **Nominal ≠ actual.** Backtested at the 90% level: **SARIMA-Gaussian covers ~0.89**
  (calibrated) and is sharpest → **best Winkler**; **Quantile-GBM under-covers (~0.82)
  yet is wider** → worst. A model can quote "90%" and simply be wrong.
- **Conformal delivers its guarantee:** split-conformal coverage lands **at/above
  0.90** (slightly conservative with limited calibration data — the safe direction).
- **ACI holds coverage under drift:** realised coverage ≈ **0.90** with the working
  level `α_t` adapting online — the right tool when the regime can shift.
- The capstone ships a **conformal fan chart**; SARIMA's model-based interval is hard
  to beat on this clean series, but conformal/ACI win when assumptions break.

## New toolkit code

[`src/probabilistic.py`](../src/probabilistic.py): `pinball_loss`, `coverage`,
`winkler_score`; `rolling_origin_intervals` (interval backtest); ready-made interval
forecasters (`parametric_…`, `quantile_…`, `conformal_…`); and conformal machinery
(`conformal_forecast`, `conformal_split_evaluate`, `aci_1step`).

## Regenerate

```powershell
.venv\Scripts\python.exe p3_probabilistic/build_p3.py
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=tsf p3_probabilistic/notebooks/*.ipynb
```

See [`DEEP_DIVE.md`](DEEP_DIVE.md) for the math of proper scores, quantile loss, and
conformal coverage.
