"""Build the Project 2 (feature-based ML forecasting) notebooks.

Same house pattern as p1_foundations/build_p1.py: teaching text lives here as
strings; `nbconvert --execute` fills in outputs.

Usage:  python p2_ml/build_p2.py
Then:   jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.kernel_name=tsf p2_ml/notebooks/*.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent / "notebooks"
KERNEL = {"display_name": "Python (TSF)", "language": "python", "name": "tsf"}


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def co(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(*cells):
    nb = nbf.v4.new_notebook()
    nb.cells = list(cells)
    nb.metadata = {"kernelspec": KERNEL, "language_info": {"name": "python"}}
    return nb


PREAMBLE = """
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))   # repo root
warnings.filterwarnings("ignore")
try:
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning, InterpolationWarning, ValueWarning)
    for _w in (ConvergenceWarning, InterpolationWarning, ValueWarning):
        warnings.simplefilter("ignore", _w)
except Exception:
    pass
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src import data, plots
plots.setup()
"""


# ===========================================================================
# 00 — From a series to a supervised table
# ===========================================================================
def nb00():
    return notebook(
        md(r"""
# P2 · 00 — From a time series to a supervised table

Project 1 modelled the series *as* a series (ARIMA, ETS). Project 2 takes the other
great approach: **reframe forecasting as supervised regression**. Predict $y_t$ from
a row of features $X_t$ built out of the **past**:

$$\underbrace{[\,y_{t-1},\,y_{t-2},\dots,\ \text{rolling mean/std},\ \text{season},\ \text{trend}\,]}_{X_t}
\;\longrightarrow\; y_t.$$

**Why bother?** It unlocks the entire ML toolbox — gradient boosting, arbitrary
**exogenous** drivers, nonlinear interactions, and training one model across **many
series** at once. **The cost:** *you* must encode time structure as features and,
above all, avoid **leakage** (letting future information into $X_t$).
"""),
        co(PREAMBLE + """
from src import features as F
q = data.load_quarterly()
logy = np.log(q["gdp_nsa"])          # we engineer features on log GDP
X, y = F.build_supervised(logy, lags=(1, 2, 3, 4), rolls=(4,),
                          fourier_order=2, period=4, trend=True, target="growth")
print("design matrix:", X.shape, "| target:", y.name)
X.head()
"""),
        md(r"""
### The feature families (read a row left to right)

| feature | meaning | why |
|---------|---------|-----|
| `lag1…lag4` | log-GDP 1–4 quarters ago | autoregressive memory; lag4 captures "a year ago" |
| `rmean4`, `rstd4` | rolling mean / std of last 4 | local level & volatility |
| `trend` | row counter (time index) | a linear time position — *but watch §01* |
| `sin1,cos1,sin2,cos2` | Fourier terms of the quarter | smooth seasonality without 4 dummies |

Every value in row $t$ is computed from data **strictly before** $t$
(`features.feature_row` only ever sees `hist[:t]`). That is the anti-leakage
guarantee — and the same function builds these rows *and* each step of a forecast,
so train-time and forecast-time features can never drift apart.
"""),
        co("""
# The target here is GROWTH (Δlog), not the level — preview of why in §01.
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(logy.index.to_timestamp(how="start"), logy.values, color="#264653")
axes[0].set(title="log GDP — strongly trending (non-stationary)", ylabel="log ₹")
axes[1].plot(y.index.to_timestamp(how="start"), y.values, color="#e76f51")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set(title="target = Δlog GDP (growth) — roughly stationary", ylabel="Δlog")
plt.show()
print("growth target: mean %.4f  std %.4f" % (y.mean(), y.std()))
"""),
        md(r"""
### Leakage — the cardinal sin

The number-one way ML forecasts lie to you: a feature that "knows" the answer.
Classic mistakes — a rolling mean that includes the current point, a `StandardScaler`
fit on the whole series before splitting, or shuffling rows in cross-validation.

Our defences, baked into the toolkit:
* features for period $t$ use only `hist[:t]` (never $y_t$ itself);
* evaluation is **rolling-origin** (`src/backtest.py`), never a shuffled split;
* the *same* `feature_row` is used to train and to forecast.

---
**Next (01):** the single biggest gotcha when you feed this table to a tree — and
why the target above is *growth*, not *level*.
"""),
    )


# ===========================================================================
# 01 — The extrapolation trap
# ===========================================================================
def nb01():
    return notebook(
        md(r"""
# P2 · 01 — The extrapolation trap (the #1 ML-forecasting mistake)

A gradient-boosted tree predicts by **averaging training targets within leaf
regions**. So its output is mathematically **bounded by the range of targets it saw
in training** — a tree *cannot extrapolate*. Feed it a strongly **trending** series
as a *level* target and it will physically fail to forecast new highs.

We'll watch this happen, then fix it.
"""),
        co(PREAMBLE + """
from src import ml, baselines as B, backtest as bt, classical as C
q = data.load_quarterly(); y = q["gdp_nsa"]
"""),
        md("### Exhibit A — predict the LEVEL directly"),
        co("""
H = 8
fc_level = ml.gbm_recursive_forecaster(target="level")(y, H)
future = pd.period_range(y.index.max() + 1, periods=H, freq="Q")
fc = pd.Series(fc_level, index=future)
fig, ax = plt.subplots()
plots.plot_forecast(y.iloc[-24:], pred=fc, title="GBM with a LEVEL target — it can't climb", ax=ax)
ax.axhline(y.max(), color="grey", ls="--", lw=1, label="training max")
ax.legend(); plt.show()
print("last actual : %.0f" % y.iloc[-1])
print("max forecast: %.0f   (training max: %.0f)" % (fc_level.max(), y.max()))
"""),
        md(r"""
The forecast **flattens — even bends downward** — and never exceeds the training
maximum (dashed line). The tree has no concept of "keep going up": in the future,
the lag features sit at or beyond the largest values it ever trained on, so every
split saturates and predictions revert toward the training distribution. On a series
that has grown every decade, that's catastrophic.

### Exhibit B — predict GROWTH instead, then cumulate

Make the target **stationary**: predict $\Delta\log y_t$ (the growth), which hovers
around a small constant, then rebuild the level by cumulating. Now the tree is
interpolating (growth stays in range) while the *level* is free to climb.
"""),
        co("""
fc_growth = ml.gbm_recursive_forecaster(target="growth")(y, H)
fcg = pd.Series(fc_growth, index=future)
fig, ax = plt.subplots()
plots.plot_forecast(y.iloc[-24:], pred=fcg,
                    title="GBM with a GROWTH target — trend restored", ax=ax)
plt.show()
print("growth forecast climbs to %.0f (vs level-target's %.0f)" %
      (fc_growth.max(), fc_level.max()))
"""),
        md("### Exhibit C — the quantitative proof (backtest)"),
        co("""
fcs = {
    "seasonal_naive(4)":      B.seasonal_naive(4),
    "SARIMA(1,1,1)(0,1,0)4":  C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4)),
    "GBM target=level":       ml.gbm_recursive_forecaster(target="level"),
    "GBM target=growth":      ml.gbm_recursive_forecaster(target="growth"),
}
bt.compare(y, fcs, initial=40, h=4, step=1, season_length=4).round(3)
"""),
        md(r"""
The numbers are brutal and clear: **`target=level` lands at MASE ≈ 1.7 — *worse than
the naive baseline*** — while **`target=growth` drops below 0.85**, beating the
baselines. Same data, same model, same features; the only change is making the
target stationary.

> **The rule.** Before handing a trending series to any tree-based model, make the
> target stationary — forecast **differences / growth** (or detrend first), then
> cumulate back. This is the ML analogue of the `d` in ARIMA.

---
**Next (02):** with the target fixed, how do we forecast **multiple steps** — feed
predictions back (recursive) or train one model per horizon (direct)?
"""),
    )


# ===========================================================================
# 02 — Multi-step strategies: recursive vs direct
# ===========================================================================
def nb02():
    return notebook(
        md(r"""
# P2 · 02 — Multi-step forecasting: recursive vs direct

Our model predicts **one step**. To forecast $h$ steps there are two classic
strategies:

* **Recursive (iterated).** Predict step 1, **feed that prediction back** as a lag,
  predict step 2, … One model; uses the latest value at every step; but **errors
  compound** down the horizon.
* **Direct.** Train **$h$ separate models**, model $k$ predicting $y_{t+k}$ directly
  from features at $t$. No compounding; but each model sees less data and ignores
  its own intermediate predictions.

Neither is universally better — so we **backtest both**.
"""),
        co(PREAMBLE + """
from src import ml, baselines as B, backtest as bt, classical as C
q = data.load_quarterly(); y = q["gdp_nsa"]
"""),
        md("### Backtest both strategies (growth target) against SARIMA"),
        co("""
fcs = {
    "seasonal_naive(4)":      B.seasonal_naive(4),
    "SARIMA(1,1,1)(0,1,0)4":  C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4)),
    "GBM recursive":          ml.gbm_recursive_forecaster(target="growth"),
    "GBM direct":             ml.gbm_direct_forecaster(target="growth"),
}
board = bt.compare(y, fcs, initial=40, h=4, step=1, season_length=4)
board.round(3)
"""),
        md(r"""
### Where does each strategy's error come from? Error vs horizon

Recursive error should grow *faster* with horizon (compounding); direct trains a
dedicated model per step. Let's compare the per-horizon error of the two.
"""),
        co("""
rec = bt.rolling_origin(y, ml.gbm_recursive_forecaster(target="growth"), initial=40, h=4)
dir_ = bt.rolling_origin(y, ml.gbm_direct_forecaster(target="growth"), initial=40, h=4)
tbl = pd.concat([bt.summarize_by_horizon(rec)["MAE"].rename("recursive"),
                 bt.summarize_by_horizon(dir_)["MAE"].rename("direct")], axis=1)
print(tbl.round(0))

fig, ax = plt.subplots(figsize=(7, 4))
tbl.plot(marker="o", ax=ax)
ax.set(title="MAE by horizon — recursive vs direct", xlabel="steps ahead", ylabel="MAE")
plt.show()
"""),
        md(r"""
On this series **direct edges recursive** (and both beat the baselines) — the
recursive model's compounding hurts more than direct's data-thinning, at these short
horizons. The general guidance:

* **Recursive** — few data, short horizons, when the latest observation is highly
  informative.
* **Direct** — enough data to train $h$ models, longer horizons where compounding
  bites. (A hybrid, *DirRec*, exists too.)

---
**Next (03):** open the black box — which features matter, and how to tune a tree on
~80 rows without overfitting.
"""),
    )


# ===========================================================================
# 03 — Feature importance & tuning on small data
# ===========================================================================
def nb03():
    return notebook(
        md(r"""
# P2 · 03 — What did it learn? Importance & (careful) tuning

Two practical skills: **interpreting** a tree model's feature importances, and
**tuning** without overfitting when you only have ~80 rows.
"""),
        co(PREAMBLE + """
from lightgbm import LGBMRegressor
from src import features as F, ml, backtest as bt
q = data.load_quarterly(); logy = np.log(q["gdp_nsa"])
X, ytar = F.build_supervised(logy, target="growth")
model = LGBMRegressor(**ml.DEFAULT_PARAMS).fit(X, ytar)
"""),
        md("### Feature importance — which signals drive the growth forecast?"),
        co("""
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
imp.plot.barh(ax=ax, color="#2a9d8f")
ax.set(title="LightGBM feature importance (growth target)", xlabel="split gain count")
plt.show()
imp.sort_values(ascending=False)
"""),
        md(r"""
Read it with care: tree "importance" counts how often a feature is split on, not a
causal effect. Expect the **recent lags** and the **seasonal Fourier / lag4** terms
to dominate — growth this quarter is driven by recent momentum and the same quarter
last year. (Note `trend` should be near-useless on the growth target — exactly the
§01 lesson, now visible in the importances.)

### Tuning on tiny data: regularisation, judged by backtest

With ~80 rows the enemy is overfitting. We sweep one knob — `num_leaves` (tree
complexity) — and let the **rolling-origin MASE** pick, never the training fit.
"""),
        co("""
rows = {}
for nl in [3, 5, 7, 15, 31]:
    f = ml.gbm_recursive_forecaster(target="growth", params={"num_leaves": nl})
    res = bt.rolling_origin(q["gdp_nsa"], f, initial=40, h=4)
    from src import metrics as M
    rows[nl] = M.mase(res.y_true, res.y_pred, q["gdp_nsa"].to_numpy(), 4)
sweep = pd.Series(rows, name="MASE"); sweep.index.name = "num_leaves"
print(sweep.round(3))
print("best num_leaves:", sweep.idxmin())
"""),
        md(r"""
Smaller trees (fewer leaves) usually win here — **more capacity is worse** when data
is scarce, because the model memorises noise. This is the bias–variance trade-off in
the open: on big data you'd let trees grow; on 80 points you clamp them down.

**Leakage checklist (burn it in):**
1. Features use only the past (`hist[:t]`). ✅ enforced in `feature_row`.
2. No scaler/encoder fit on the full series before splitting.
3. CV respects time order — **rolling origin**, never shuffled K-fold.
4. The target's own value never appears among its features.

---
**Next (04):** the capstone — ML vs the best classical models, head to head, and an
honest verdict on when each wins.
"""),
    )


# ===========================================================================
# 04 — Capstone: ML vs classical
# ===========================================================================
def nb04():
    return notebook(
        md(r"""
# P2 · 04 — Capstone: machine learning vs classical, head to head

Bring Project 1's best (SARIMA, Holt-Winters) and Project 2's best (gradient boosting
on a growth target, direct & recursive) into **one rolling-origin backtest**, then
produce a final ML forecast with a **quantile-regression interval**.
"""),
        co(PREAMBLE + """
from src import ml, baselines as B, backtest as bt, classical as C
q = data.load_quarterly(); y = q["gdp_nsa"]

candidates = {
    "seasonal_naive(4)":      B.seasonal_naive(4),
    "Holt-Winters":           C.ets_forecaster(trend="add", seasonal="add", seasonal_periods=4),
    "SARIMA(1,1,1)(0,1,0)4":  C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4)),
    "GBM growth (recursive)": ml.gbm_recursive_forecaster(target="growth"),
    "GBM growth (direct)":    ml.gbm_direct_forecaster(target="growth"),
}
board = bt.compare(y, candidates, initial=40, h=4, step=1, season_length=4)
print(board.round(3).to_string())
print("\\nWinner by MASE:", board.index[0])
"""),
        md(r"""
### The honest verdict

On **88 points of one clean, smooth series**, the tuned **SARIMA usually wins**, with
gradient boosting a close, respectable second. That is *not* a knock on ML — it's the
right read of the situation:

* **Classical shines** when you have one well-behaved series with clear trend +
  seasonality and little else. Few parameters, strong structure, hard to beat.
* **ML shines** when you have **many related series**, **exogenous drivers** (rates,
  weather, promotions), **nonlinearities/interactions**, or **long histories** — none
  of which a univariate GDP series exercises. P2 built exactly the machinery (feature
  tables, multi-step, leakage-safe CV) that scales to those problems.

### Final ML forecast with a prediction interval (quantile regression)
"""),
        co("""
H = 8
mid, lo, hi = ml.gbm_quantile_forecast(y, H, alpha=0.05)
future = pd.period_range(y.index.max() + 1, periods=H, freq="Q")
fc = pd.Series(mid, index=future)
lo = pd.Series(lo, index=future); hi = pd.Series(hi, index=future)
fig, ax = plt.subplots()
plots.plot_forecast(y.iloc[-28:], pred=fc, lower=lo, upper=hi,
                    title="LightGBM growth forecast + 90% quantile interval", ax=ax)
plots.save(fig, "p2_ml_forecast"); plt.show()
pd.DataFrame({"forecast": fc.round(0), "low": lo.round(0), "high": hi.round(0)})
"""),
        md(r"""
Quantile regression gives an interval **for free** (fit the 5th/50th/95th-percentile
losses), and it needn't be symmetric. Caveats you should say out loud: separately-fit
quantiles can **cross**, and recursive quantile forecasts **widen fast** because
growth uncertainty compounds. Project 3 replaces this with calibrated, guaranteed-
coverage intervals (**conformal prediction**) and proper probabilistic scoring.

### Project 2 — what you can now do

1. Reframe any forecasting problem as a **supervised table** (lags, rolling, calendar,
   Fourier) without leakage.
2. Avoid the **extrapolation trap** by forecasting a **stationary target**.
3. Choose and implement **recursive vs direct** multi-step strategies.
4. **Interpret** and **regularise** a gradient-boosted forecaster, judged by a proper
   time-series backtest.
5. Know **when ML beats classical** — and when it doesn't.

---
**Next — Project 3:** probabilistic forecasting — quantile loss done right, calibration,
and conformal prediction intervals with coverage guarantees.
"""),
    )


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "00_supervised_framing.ipynb": nb00,
        "01_extrapolation_trap.ipynb": nb01,
        "02_multistep_strategies.ipynb": nb02,
        "03_importance_and_tuning.ipynb": nb03,
        "04_capstone_ml_vs_classical.ipynb": nb04,
    }
    for fname, fn in builders.items():
        nbf.write(fn(), NB_DIR / fname)
        print("wrote", fname)


if __name__ == "__main__":
    main()
