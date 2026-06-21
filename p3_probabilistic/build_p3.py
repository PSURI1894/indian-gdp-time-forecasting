"""Build the Project 3 (probabilistic forecasting) notebooks.

Usage:  python p3_probabilistic/build_p3.py
Then:   jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.kernel_name=tsf p3_probabilistic/notebooks/*.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent / "notebooks"
KERNEL = {"display_name": "Python (TSF)", "language": "python", "name": "tsf"}


def md(t): return nbf.v4.new_markdown_cell(t.strip("\n"))
def co(t): return nbf.v4.new_code_cell(t.strip("\n"))


def notebook(*cells):
    nb = nbf.v4.new_notebook()
    nb.cells = list(cells)
    nb.metadata = {"kernelspec": KERNEL, "language_info": {"name": "python"}}
    return nb


PREAMBLE = """
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))
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
# 00 — Why probabilistic, and how to score it
# ===========================================================================
def nb00():
    return notebook(
        md(r"""
# P3 · 00 — A forecast without uncertainty is a guess

A point forecast says "GDP will be ₹X". A **probabilistic** forecast says "₹X, and
I'm 90% sure it's between ₹L and ₹H." Decisions need the second kind — capacity,
risk, budgets all hinge on the *range*, not the midpoint.

Three things make an interval *good*, and we need a number for each:

| property | question | metric |
|----------|----------|--------|
| **calibration** | does a 90% interval actually contain truth 90% of the time? | **coverage** |
| **sharpness** | is it tight enough to be useful? | **mean width** |
| **both at once** | the honest overall score | **Winkler / interval score** |

You can't chase coverage alone (a ±∞ interval covers 100% and is useless) or width
alone (a zero-width interval is sharp and always wrong). The **Winkler score**
combines them and is a *proper* score — you minimise it by being honest.
"""),
        co(PREAMBLE + """
from src import classical as C, probabilistic as P
q = data.load_quarterly(); y = q["gdp_nsa"]
alpha = 0.10                      # -> 90% intervals
"""),
        md(r"""
### Are our P1/P2 intervals actually calibrated? (backtest them)

We already have two interval producers: SARIMA's **model-based Gaussian-in-log**
interval (P1) and LightGBM **quantile regression** (P2). Let's measure their real
out-of-sample coverage with a rolling-origin **interval** backtest.
"""),
        co("""
methods = {
    "SARIMA-Gaussian": P.parametric_interval_forecaster((1, 1, 1), (0, 1, 0, 4), alpha=alpha),
    "Quantile-GBM":    P.quantile_interval_forecaster(alpha=alpha),
}
rows = {}
for name, g in methods.items():
    df = P.rolling_origin_intervals(y, g, initial=40, h=4, step=1)
    rows[name] = P.summarize_intervals(df, alpha)
table = pd.DataFrame(rows).T
table["target"] = 1 - alpha
table.round({"coverage": 2, "target": 2, "mean_width": 0, "winkler": 0})
"""),
        md(r"""
Read it carefully — this is the whole point of the notebook:

* **SARIMA-Gaussian** lands near **0.89** coverage (≈ its 0.90 label) with the
  **narrowest** interval and the **best (lowest) Winkler**. Well-calibrated *and*
  sharp.
* **Quantile-GBM** **under-covers (~0.82–0.85)** *and* is much **wider** — worse on
  both axes, so its Winkler is far higher. A model can quote "90%" and simply be
  wrong; only the backtest reveals it.

> **Lesson:** nominal ≠ actual. Always *measure* coverage out-of-sample, and rank by
> a proper score (Winkler), never by eyeballing a fan chart.

### See one fold
"""),
        co("""
g = methods["SARIMA-Gaussian"]
df = P.rolling_origin_intervals(y, g, initial=len(y) - 8, h=8, step=1)  # last 8q
xt = [pd.Period(s, freq="Q").to_timestamp(how="start") for s in
      pd.period_range(y.index[-8], periods=8, freq="Q").astype(str)]
fig, ax = plt.subplots()
hist = y.iloc[-24:-8]
ax.plot(hist.index.to_timestamp(how="start"), hist.values / 1e6, color="#264653", label="history")
ax.plot(xt, df.y_true / 1e6, color="#2a9d8f", marker="o", label="actual")
ax.plot(xt, df["mean"] / 1e6, color="#e76f51", lw=2, label="forecast")
ax.fill_between(xt, df.lo / 1e6, df.hi / 1e6, color="#e76f51", alpha=0.2, label="90% interval")
ax.set(title="SARIMA 90% interval vs actuals (last 8 quarters)", ylabel="real GDP (level)")
ax.legend(); plt.show()
"""),
        md(r"""
---
**Next (01):** the workhorse behind learned intervals — **quantile regression** —
its loss function, the **quantile-crossing** bug, and a **calibration diagram**.
"""),
    )


# ===========================================================================
# 01 — Quantile regression & calibration
# ===========================================================================
def nb01():
    return notebook(
        md(r"""
# P3 · 01 — Quantile regression & calibration

To predict the τ-quantile (e.g. the 5th percentile), you minimise the **pinball
(quantile) loss**, which penalises under- and over-prediction *asymmetrically*:

$$L_\tau(y,\hat q)=\max\big(\tau(y-\hat q),\ (\tau-1)(y-\hat q)\big).$$

For τ=0.5 it's symmetric (→ the median); for τ=0.95 it punishes under-prediction
19× harder than over-prediction, pushing the estimate up into the tail.
"""),
        co(PREAMBLE + """
from src import classical as C, probabilistic as P
q = data.load_quarterly(); y = q["gdp_nsa"]
"""),
        md("### The pinball loss, drawn"),
        co("""
err = np.linspace(-3, 3, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for tau, col in [(0.1, "#264653"), (0.5, "#2a9d8f"), (0.9, "#e76f51")]:
    loss = np.maximum(tau * err, (tau - 1) * err)
    ax.plot(err, loss, color=col, label=f"τ = {tau}")
ax.set(title="Pinball loss — asymmetric by quantile", xlabel="y − q̂ (actual minus prediction)",
       ylabel="loss"); ax.legend(); ax.axvline(0, color="k", lw=0.6); plt.show()
"""),
        md(r"""
### Quantile crossing — and the fix

Quantiles fit **independently** can come out **non-monotone**: a predicted 90%
value below the 50% value. Nonsensical, and common with small data. The standard
fix is to **sort** the quantile predictions at each time point (isotonic / rearrange).
"""),
        co("""
from lightgbm import LGBMRegressor
from src import features as F, ml
X, ytar = F.build_supervised(np.log(y), target="growth")
taus = [0.1, 0.3, 0.5, 0.7, 0.9]
preds = np.column_stack([
    LGBMRegressor(objective="quantile", alpha=t, **ml.DEFAULT_PARAMS).fit(X, ytar).predict(X)
    for t in taus])
crossings = int((np.diff(preds, axis=1) < 0).sum())
preds_sorted = np.sort(preds, axis=1)        # enforce monotonicity row-wise
print(f"raw quantile crossings: {crossings}  ->  after sorting: "
      f"{int((np.diff(preds_sorted, axis=1) < 0).sum())}")
"""),
        md(r"""
### The calibration (reliability) diagram

The acid test: for each nominal level, what fraction of actuals actually fall inside?
Plot **nominal vs empirical coverage** — the diagonal is perfect. We build honest
out-of-sample intervals with **split conformal** (calibrate residual quantiles on
earlier folds, measure coverage on later ones) across several levels.
"""),
        co("""
pf = C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4))
res = P.backtest_residuals(y, pf, initial=40, h=4, step=1, use_log=True)
cutoffs = np.sort(res.cutoff_idx.unique()); split = cutoffs[int(len(cutoffs) * 0.6)]
cal, test = res[res.cutoff_idx < split], res[res.cutoff_idx >= split]

levels = [0.5, 0.7, 0.8, 0.9, 0.95]
emp = []
for lv in levels:
    qmap = cal.groupby("step_ahead")["resid"].apply(lambda r: np.quantile(np.abs(r), lv))
    t = test.copy(); t["hw"] = t.step_ahead.map(qmap)
    lo, hi = t.y_pred * np.exp(-t.hw), t.y_pred * np.exp(t.hw)
    emp.append(P.coverage(t.y_true, lo, hi))

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
ax.plot(levels, emp, "o-", color="#e76f51", label="conformal (SARIMA resid)")
ax.set(title="Calibration diagram", xlabel="nominal coverage", ylabel="empirical coverage")
ax.legend(); plt.show()
print("nominal :", levels)
print("empirical:", [round(e, 2) for e in emp])
"""),
        md(r"""
A well-calibrated method hugs the diagonal (or sits **just above** it — slightly
conservative is safe). Conformal lands there *by construction*, which is exactly why
we reach for it next.

---
**Next (02):** **conformal prediction** — distribution-free coverage guarantees —
and **adaptive conformal inference** for when time breaks the assumptions.
"""),
    )


# ===========================================================================
# 02 — Conformal prediction & ACI
# ===========================================================================
def nb02():
    return notebook(
        md(r"""
# P3 · 02 — Conformal prediction & adaptive conformal inference

**Conformal prediction** turns *any* point model into intervals with a
**distribution-free, finite-sample coverage guarantee** — no Gaussian assumption.
The recipe (split conformal):

1. Fit a point model on a *proper-training* set.
2. On a held-out **calibration** set, collect the residuals.
3. The interval is `prediction ± (1-α) empirical quantile of |residual|`.

Under **exchangeability**, that interval covers the truth with prob ≥ 1-α. We
calibrate the residual quantile **per horizon** and in **log space** (→ a
multiplicative band, right for GDP).
"""),
        co(PREAMBLE + """
from src import classical as C, probabilistic as P
q = data.load_quarterly(); y = q["gdp_nsa"]; alpha = 0.10
"""),
        md("### Split conformal: honest out-of-sample coverage"),
        co("""
pf = C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4))
test, s = P.conformal_split_evaluate(y, pf, initial=40, h=4, alpha=alpha, test_frac=0.4)
print("target coverage : %.2f" % (1 - alpha))
print("conformal cover : %.2f   (n_test=%d)" % (s["coverage"], len(test)))
print("mean width      : %.3e   winkler: %.3e" % (s["mean_width"], s["winkler"]))
"""),
        md(r"""
Coverage comes out **at or above** the 90% target — conformal delivers its promise
(often *slightly conservative* with limited calibration data, which is the safe
direction to err).

### The catch: time series aren't exchangeable

Conformal's guarantee assumes residuals are exchangeable. Time series violate this —
**autocorrelation, drift, and changing volatility** mean recent residuals aren't like
old ones (think pre- vs post-COVID). A fixed conformal band can silently lose coverage
when the regime shifts.

### Adaptive Conformal Inference (ACI)

ACI fixes this **online**: after each step it nudges the working level $\alpha_t$ by
whether the last interval missed,
$$\alpha_{t+1} = \alpha_t + \gamma\,(\alpha - \text{err}_t),\quad \text{err}_t = \mathbf 1[\text{missed}],$$
shrinking the interval after a hit and widening it after a miss. This keeps **long-run
coverage near the target regardless of drift**.
"""),
        co("""
ets = C.ets_forecaster(trend="add", seasonal="add", seasonal_periods=4)
aci = P.aci_1step(y, ets, initial=30, alpha=alpha, gamma=0.03)
print("ACI realised coverage: %.2f  (target %.2f)" % (aci.attrs["coverage"], 1 - alpha))

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
t = [pd.Period(s, freq="Q").to_timestamp(how="start") for s in aci["index"]]
axes[0].plot(t, aci.alpha_t, color="#e76f51")
axes[0].axhline(alpha, color="k", ls="--", lw=1, label="target α")
axes[0].set(title="ACI adapts the working level α_t online", ylabel="α_t"); axes[0].legend()
roll = (1 - aci.miss).rolling(12, min_periods=4).mean()
axes[1].plot(t, roll, color="#2a9d8f")
axes[1].axhline(1 - alpha, color="k", ls="--", lw=1, label="target coverage")
axes[1].set(title="Rolling 12-step coverage stays near target", ylabel="coverage"); axes[1].legend()
plt.show()
"""),
        md(r"""
$\alpha_t$ breathes in and out as conditions change, and the rolling coverage tracks
the target instead of drifting off. **Use plain conformal for stable series; reach
for ACI when the regime can shift** (most real operational series).

---
**Next (03):** the capstone — rank every interval method by Winkler, and ship a
calibrated **fan chart**.
"""),
    )


# ===========================================================================
# 03 — Capstone: a calibrated probabilistic forecast
# ===========================================================================
def nb03():
    return notebook(
        md(r"""
# P3 · 03 — Capstone: choosing and shipping a calibrated forecast

Rank the interval methods on **one** apples-to-apples interval backtest by the
proper score (**Winkler**) *and* coverage, then ship a final **fan chart**.
"""),
        co(PREAMBLE + """
from src import classical as C, probabilistic as P
q = data.load_quarterly(); y = q["gdp_nsa"]; alpha = 0.10
ets = C.ets_forecaster(trend="add", seasonal="add", seasonal_periods=4)

methods = {
    "SARIMA-Gaussian": P.parametric_interval_forecaster((1, 1, 1), (0, 1, 0, 4), alpha=alpha),
    "Quantile-GBM":    P.quantile_interval_forecaster(alpha=alpha),
    "Conformal-ETS":   P.conformal_interval_forecaster(ets, inner_initial=24, alpha=alpha),
}
rows = {}
for name, g in methods.items():
    df = P.rolling_origin_intervals(y, g, initial=40, h=4, step=2)   # step=2: conformal is nested
    rows[name] = P.summarize_intervals(df, alpha)
board = pd.DataFrame(rows).T.sort_values("winkler")
board["target"] = 1 - alpha
print(board.round({"coverage": 2, "target": 2, "mean_width": 0, "winkler": 0}).to_string())
"""),
        md(r"""
### The verdict

Rank by **Winkler** (the proper score), then sanity-check coverage. On this clean
series the **model-based SARIMA-Gaussian interval is hard to beat** — it's calibrated
*and* sharp. Conformal is competitive and **assumption-free** (its edge shows up when
the model's distributional assumption is wrong, or under drift, where **ACI** wins).
Quantile-GBM is the wide, under-covering laggard here — learned tails need more data
than 88 points to behave.

### Ship it: a fan chart (50 / 80 / 90% conformal bands)

A fan chart shows the whole predictive distribution. We stack conformal bands at three
levels around the SARIMA point forecast.
"""),
        co("""
pf = C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4))
H = 8
future = pd.period_range(y.index.max() + 1, periods=H, freq="Q")
xt = future.to_timestamp(how="start")
fig, ax = plt.subplots()
hist = y.iloc[-28:]
ax.plot(hist.index.to_timestamp(how="start"), hist.values / 1e6, color="#264653", label="history")
for a, shade in [(0.5, 0.15), (0.2, 0.18), (0.1, 0.20)]:
    mean, lo, hi = P.conformal_forecast(y, pf, H, initial=40, alpha=a)
    ax.fill_between(xt, lo / 1e6, hi / 1e6, color="#e76f51", alpha=shade,
                    label=f"{int((1-a)*100)}%")
ax.plot(xt, mean / 1e6, color="#e76f51", lw=2, label="median")
ax.set(title="India GDP — conformal fan chart (50/80/90%)", ylabel="real GDP (level)")
ax.legend(ncol=2); plots.save(fig, "p3_fan_chart"); plt.show()
"""),
        md(r"""
### Project 3 — what you can now do

1. **Score** intervals properly — coverage, sharpness, and the **Winkler** score —
   instead of trusting a label.
2. Fit **quantile regression**, spot **quantile crossing**, and read a **calibration
   diagram**.
3. Build **conformal** intervals with a coverage guarantee, and **ACI** to hold
   coverage under drift.
4. Ship a **fan chart** that communicates the full predictive distribution.

**The office takeaway:** never present a point forecast naked. Attach an interval,
**prove its coverage on a backtest**, and prefer conformal/ACI when you can't trust a
distributional assumption.

---
**Next — Project 4:** deep-learning forecasting **from scratch in NumPy** (an MLP and
an RNN), since PyTorch is blocked on this machine — so we'll build the gradients by
hand and *really* understand them.
"""),
    )


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "00_evaluating_uncertainty.ipynb": nb00,
        "01_quantile_regression.ipynb": nb01,
        "02_conformal_prediction.ipynb": nb02,
        "03_capstone_probabilistic.ipynb": nb03,
    }
    for fname, fn in builders.items():
        nbf.write(fn(), NB_DIR / fname)
        print("wrote", fname)


if __name__ == "__main__":
    main()
