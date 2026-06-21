# Project 2 — Feature-Based Machine-Learning Forecasting

Reframe forecasting as **supervised regression**: predict $y_t$ from a table of
features built out of the past, then unleash gradient boosting. The throughline:
the model is the easy part — **the framing, the stationary target, and leakage-safe
evaluation** are what make it work (or fail).

## Notebooks (run in order, kernel = *Python (TSF)*)

| # | notebook | what you learn |
|---|----------|----------------|
| 00 | `00_supervised_framing.ipynb` | Turning a series into an `(X, y)` table — lag / rolling / calendar / Fourier features — and the leakage rules that keep it honest |
| 01 | `01_extrapolation_trap.ipynb` | **The #1 ML-forecasting mistake:** trees can't extrapolate a trend. Watch a level-target GBM fail, fix it with a **growth** target |
| 02 | `02_multistep_strategies.ipynb` | **Recursive vs direct** multi-step forecasting — implementation, error-vs-horizon, trade-offs |
| 03 | `03_importance_and_tuning.ipynb` | Reading feature importances; regularising a tree on ~80 rows; the leakage checklist |
| 04 | `04_capstone_ml_vs_classical.ipynb` | ML vs P1's classical models head-to-head; a quantile-regression interval; **when ML actually wins** |

## Key findings

- **Extrapolation trap, quantified:** a GBM on a **level** target scores **MASE 1.74
  — worse than naive** (it literally can't forecast above its training max). Switch
  the target to **growth (Δlog)** and the *same* model drops to ~0.82, beating
  baselines. Make the target stationary before feeding trees.
- **Direct > recursive** here (MASE **0.669** vs 0.819) — recursive error compounds.
- **Tuning on tiny data:** `num_leaves=3` (MASE **0.744**) beats the default — less
  capacity wins when data is scarce.
- **Honest verdict:** on 88 points of one clean series, **SARIMA (0.612) still wins**;
  GBM is a close second. ML's edge is many series / exogenous drivers / nonlinearity —
  P2 builds the machinery that scales to those.

## New toolkit code

- [`src/features.py`](../src/features.py) — `feature_row` (one function used for both
  training and forecasting → no train/serve skew) and `build_supervised`.
- [`src/ml.py`](../src/ml.py) — `gbm_recursive_forecaster`, `gbm_direct_forecaster`,
  `gbm_quantile_forecast`; all expose the same `f(train, h)` contract as P1's models.

## Regenerate

```powershell
.venv\Scripts\python.exe p2_ml/build_p2.py
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=tsf p2_ml/notebooks/*.ipynb
```

See [`DEEP_DIVE.md`](DEEP_DIVE.md) for the math of supervised framing, why trees
can't extrapolate, and the multi-step strategies.
