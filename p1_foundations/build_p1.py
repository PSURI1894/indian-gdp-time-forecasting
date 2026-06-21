"""Build the Project 1 (Foundations & Classical) notebooks programmatically.

Why a build script (the house pattern from practice-eda): the *teaching text*
lives here as plain strings, under version control and easy to diff. Running
this regenerates clean .ipynb files; a separate `nbconvert --execute` step fills
in the outputs. Edit here, not in the .ipynb, to keep things reproducible.

Usage:  python p1_foundations/build_p1.py
Then:   jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.kernel_name=tsf p1_foundations/notebooks/*.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent / "notebooks"
KERNEL = {"display_name": "Python (TSF)", "language": "python", "name": "tsf"}


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def co(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(*cells):
    nb = nbf.v4.new_notebook()
    nb.cells = list(cells)
    nb.metadata = {"kernelspec": KERNEL, "language_info": {"name": "python"}}
    return nb


# Standard preamble: put the repo root on sys.path so `from src import ...` works
# regardless of where Jupyter is launched from.
PREAMBLE = """
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))   # repo root
warnings.filterwarnings("ignore")
try:  # silence statsmodels' chatty (harmless) convergence / lookup warnings
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
# Notebook 00 — Orientation & transforms
# ===========================================================================
def nb00():
    return notebook(
        md(r"""
# P1 · 00 — First look: what does India's GDP *look like*, and how do we tame it?

**Project 1 goal:** master the classical forecasting toolkit on a real, *small*
macro series — India GDP. This first notebook is about **seeing the data clearly
and applying the two transforms that make GDP forecastable**: the *log* and the
*growth rate*.

By the end you can answer:
1. Why is a raw GDP plot misleading, and what does `log(GDP)` fix?
2. What is the difference between a *level*, a *log-level*, and a *growth rate*?
3. Why must time-series train/test splits respect time order?
"""),
        co(PREAMBLE + """
annual = data.load_annual()      # real GDP, constant LCU, 1960-2024
q = data.load_quarterly()        # real GDP, NSA + SA, 2004Q2-2026Q1
print("annual :", annual.index.min(), "->", annual.index.max(), "| n =", len(annual))
print("quarter:", q.index.min(), "->", q.index.max(), "| n =", len(q))
q.tail()
"""),
        md(r"""
### The three series we'll use all curriculum long

| name | freq | what | use |
|------|------|------|-----|
| `annual` | yearly | real GDP, constant LCU | long history → trend, ARIMA, tiny-sample backtests |
| `gdp_nsa` | quarterly | **N**ot **S**easonally **A**djusted | seasonality, decomposition, **SARIMA** |
| `gdp_sa` | quarterly | seasonally adjusted | contrast — *what adjustment removes* |

> ⚠️ **Do not compare the levels across these series.** Each uses a different
> base year / price base, so `annual` (≈10¹⁴) and the quarterly series (≈10⁷)
> are on different scales. Forecasting works *within* a series, so this is fine.
"""),
        co("""
fig, ax = plt.subplots()
ax.plot(annual.index, annual.values / 1e12, color="#264653")
ax.set(title="India real GDP (constant LCU), annual",
       xlabel="year", ylabel="₹ trillion (constant)")
plt.show()
"""),
        md(r"""
That curve **bends upward** — it grows *faster* as it gets bigger. That's the
signature of (roughly) **exponential growth**: GDP each year is about
`(1+g)` times the previous year. Two problems this creates for modelling:

* the *variance* grows with the level (recent wiggles dwarf 1960s wiggles),
* a straight-line model would badly underfit the recent years.

**The fix: take logs.** If $GDP_t \approx GDP_0 (1+g)^t$, then
$\log GDP_t \approx \log GDP_0 + t\log(1+g)$ — a **straight line** whose slope is
the growth rate. Logs turn multiplicative growth into additive growth.
"""),
        co("""
fig, ax = plt.subplots()
ax.plot(annual.index, np.log(annual.values), color="#264653")
ax.set(title="log(real GDP) — exponential growth becomes ~linear",
       xlabel="year", ylabel="log ₹")
plt.show()
"""),
        md(r"""
Much straighter. The slope of this line *is* the average growth rate. We can
make growth explicit by **differencing the log**: the year-on-year growth rate is

$$g_t \;=\; \log GDP_t - \log GDP_{t-1} \;\approx\; \frac{GDP_t-GDP_{t-1}}{GDP_{t-1}}$$

(the `×100` makes it a percent). Differencing the log is the single most common
transform in macro forecasting — it converts a trending, non-stationary level
into a roughly **stationary** growth series we can actually model.
"""),
        co("""
g = np.log(annual).diff() * 100
fig, ax = plt.subplots()
ax.bar(g.index, g.values, color=np.where(g.values < 0, "#e76f51", "#2a9d8f"))
ax.axhline(g.mean(), color="k", ls="--", lw=1, label=f"mean {g.mean():.1f}%")
ax.set(title="Annual real GDP growth (YoY, % = 100·Δlog)", xlabel="year", ylabel="%")
ax.legend(); plt.show()
g.describe().round(2)
"""),
        md(r"""
Now the *story* is visible: India's real GDP grew **~5–6% on average**, with
recognisable shocks — the **1991** balance-of-payments crisis, the **2008** GFC
dip, and the dramatic **2020 COVID** contraction followed by a sharp rebound.

A growth series like this is what classical models love: roughly constant mean,
roughly constant variance. Hold that thought — we'll *test* it formally in 01.

### Now the quarterly data: seasonality appears
"""),
        co("""
fig, ax = plt.subplots()
x = q.index.to_timestamp(how="start")
ax.plot(x, q["gdp_nsa"] / 1e6, label="NSA (raw)", color="#2a9d8f")
ax.plot(x, q["gdp_sa"] / 1e6, label="SA (adjusted)", color="#e76f51")
ax.set(title="Quarterly real GDP: NSA vs seasonally adjusted",
       ylabel="real GDP level (constant prices)")
ax.legend(); plt.show()
"""),
        md(r"""
The green **NSA** line has a regular saw-tooth — that's **seasonality** (India's
Jan–Mar quarter is consistently the strongest). The orange **SA** line has that
repeating pattern statistically removed, leaving trend + irregular. Government
agencies publish SA series so analysts can read the *underlying* momentum without
the calendar getting in the way. In 01 we'll **decompose** the NSA series to
extract exactly that seasonal shape.

Two useful growth views of the quarterly data:
* **QoQ on the SA series** — quarter-to-quarter momentum (use SA so seasonality
  doesn't masquerade as growth).
* **YoY on the NSA series** — compare each quarter to the *same* quarter a year
  ago; the 4-quarter difference cancels seasonality automatically.
"""),
        co("""
qoq = np.log(q["gdp_sa"]).diff() * 100        # quarter-on-quarter (SA)
yoy = np.log(q["gdp_nsa"]).diff(4) * 100      # year-on-year (NSA, 4-qtr diff)
fig, ax = plt.subplots()
ax.plot(x, qoq, label="QoQ (SA)", color="#e76f51")
ax.plot(x, yoy, label="YoY (NSA)", color="#264653", lw=2)
ax.axhline(0, color="k", lw=0.8)
ax.set(title="Quarterly growth: QoQ momentum vs YoY", ylabel="%")
ax.legend(); plt.show()
print("YoY mean %:", round(yoy.mean(), 2), "| COVID trough:", round(yoy.min(), 1),
      "in", yoy.idxmin())
"""),
        md(r"""
### Golden rule: time-series splits respect time

You will be tempted to use a normal random train/test split. **Don't.** Shuffling
puts *future* points in the training set, so the model "remembers" what it's meant
to predict — accuracy looks great in the notebook and collapses in production.

The only valid split is **chronological**: train on the past, test on the most
recent tail. And a single tail split is fragile (it depends on one lucky/unlucky
window), so in notebook 02 we'll upgrade to **rolling-origin backtesting**.

---
**Next (01):** decompose the seasonal pattern and *formally test* whether our
transformed series are stationary — the prerequisite for ARIMA.
"""),
    )


# ===========================================================================
# Notebook 01 — Decomposition & stationarity
# ===========================================================================
def nb01():
    return notebook(
        md(r"""
# P1 · 01 — Decomposition & stationarity: the diagnosis before the model

Two foundational ideas:

1. **Decomposition** — split a series into **trend + seasonal + remainder** so we
   can *see* the structure we need to model.
2. **Stationarity** — a series whose statistical properties (mean, variance,
   autocorrelation) don't change over time. ARIMA assumes it; this notebook shows
   how to **test** for it and how much **differencing** is needed to get there.

We'll finish by reading the **ACF/PACF** plots — the classical "fingerprint" that
suggests ARIMA orders.
"""),
        co(PREAMBLE + """
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

q = data.load_quarterly()
nsa = q["gdp_nsa"]
# statsmodels wants a real datetime index for decomposition plots:
nsa_ts = nsa.copy(); nsa_ts.index = nsa.index.to_timestamp(how="start")
nsa_ts.tail()
"""),
        md(r"""
### Additive vs multiplicative decomposition

* **Additive:** $y_t = T_t + S_t + R_t$ — use when the seasonal swing is a roughly
  *constant size* regardless of level.
* **Multiplicative:** $y_t = T_t \times S_t \times R_t$ — use when the seasonal
  swing *grows with the level* (bigger economy → bigger absolute seasonal gap).

GDP grows exponentially and its seasonal swing grows with it → **multiplicative**
(equivalently: *additive on the log*). Let's decompose with period = 4 quarters.
"""),
        co("""
dec = seasonal_decompose(nsa_ts, model="multiplicative", period=4)
fig = dec.plot(); fig.set_size_inches(11, 8)
fig.suptitle("Classical multiplicative decomposition (NSA, period=4)", y=1.01)
plt.show()

# The seasonal factors: how much each quarter sits above/below trend.
seasonal_factors = dec.seasonal.groupby(dec.seasonal.index.quarter).mean()
print("Seasonal factors by calendar quarter (1.0 = on-trend):")
print((seasonal_factors).round(3))
"""),
        md(r"""
Read the factors: a value of **1.05** means that quarter runs ~5% **above** the
local trend; **0.96** means ~4% below. You should see Q1 (Jan–Mar) well above 1
and Q2 (Apr–Jun) below — India's fiscal year-end quarter is the strongest.

Classical `seasonal_decompose` uses simple moving averages and assumes a *fixed*
seasonal shape. **STL** (Seasonal-Trend decomposition using Loess) is more robust
and lets the seasonal shape evolve slowly. We run STL on the **log** (so the
multiplicative structure becomes additive).
"""),
        co("""
stl = STL(np.log(nsa_ts), period=4, robust=True).fit()
fig = stl.plot(); fig.set_size_inches(11, 8)
fig.suptitle("STL decomposition of log(NSA)", y=1.01)
plt.show()

# Strength of seasonality (Hyndman): 1 - Var(remainder)/Var(seasonal+remainder)
res, seas = stl.resid, stl.seasonal
Fs = max(0.0, 1 - res.var() / (seas + res).var())
print(f"Seasonal strength Fs = {Fs:.2f}  (0 = none, 1 = very strong)")
"""),
        md(r"""
A seasonal strength above ~0.6 means seasonality is a *dominant* feature — we'll
definitely want a **seasonal** model (SARIMA / Holt-Winters), not a plain one.

### Stationarity: test, don't eyeball

A stationary series has a stable mean and variance. ARIMA models the *stationary*
version of a series, so we must figure out how many times to difference. We use
**two complementary tests** because each has a different null hypothesis:

| test | null hypothesis $H_0$ | reject (p<0.05) means |
|------|----------------------|------------------------|
| **ADF** (Augmented Dickey-Fuller) | has a unit root (non-stationary) | *stationary* |
| **KPSS** | is stationary | *non-stationary* |

The trustworthy conclusion is when they **agree**. ADF wanting stationary *and*
KPSS not-rejecting stationary = confident green light.
"""),
        co("""
def stat_tests(s, name):
    s = pd.Series(s).dropna()
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]
    return {
        "series": name,
        "ADF p": round(adf_p, 3),
        "KPSS p": round(kpss_p, 3),
        "ADF → ": "stationary" if adf_p < 0.05 else "unit root",
        "KPSS → ": "stationary" if kpss_p > 0.05 else "non-stationary",
    }

logn = np.log(nsa)
pd.DataFrame([
    stat_tests(nsa,                       "level"),
    stat_tests(logn,                      "log level"),
    stat_tests(logn.diff(),               "log · 1st diff"),
    stat_tests(logn.diff().diff(4),       "log · 1st + seasonal(4) diff"),
])
"""),
        md(r"""
Typical reading for this series:
* **level / log level** → ADF says unit root, KPSS says non-stationary → *not* stationary.
* **1st diff of log** → removes the trend; closer to stationary but seasonality
  may still trip KPSS.
* **1st + seasonal diff** → both tests agree it's stationary.

That tells us the differencing orders for SARIMA: regular **d = 1**, seasonal
**D = 1**, season **m = 4**. (We'll let the model confirm in notebook 04.)

### ACF & PACF — the order-selection fingerprint

On the *stationary* (differenced) series:
* **ACF** (autocorrelation) cutting off after lag *q* → suggests an **MA(q)** term.
* **PACF** (partial autocorrelation) cutting off after lag *p* → suggests an **AR(p)** term.
* Spikes at lag 4, 8, … → the **seasonal** AR/MA terms.
"""),
        co("""
stat = np.log(nsa).diff().diff(4).dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(stat, lags=20, ax=axes[0]);  axes[0].set_title("ACF of stationary series")
plot_pacf(stat, lags=20, ax=axes[1], method="ywm"); axes[1].set_title("PACF")
plt.show()
"""),
        md(r"""
Bars outside the shaded band are statistically significant. Don't agonise over a
perfect read — in practice we use ACF/PACF to pick a *short list* of candidate
orders, then let an information criterion (AIC) choose between them (notebook 04).

---
**Next (02):** before any fancy model, build the **evaluation harness** — baselines,
metrics, and rolling-origin backtesting — so every later model is judged honestly.
"""),
    )


# ===========================================================================
# Notebook 02 — Baselines & honest evaluation
# ===========================================================================
def nb02():
    return notebook(
        md(r"""
# P1 · 02 — Measure first, model later: baselines, metrics & backtesting

The most important habit in applied forecasting: **establish the yardstick before
building models.** This notebook sets up the three pieces every later notebook
reuses (all live in `src/`):

1. **Baselines** — naive, seasonal-naive, drift, mean. The bar to beat.
2. **Metrics** — MAE, RMSE, MAPE, sMAPE, and especially **MASE**.
3. **Rolling-origin backtesting** — the leak-free way to estimate accuracy.
"""),
        co(PREAMBLE + """
from src import baselines as B, metrics as M, backtest as bt
q = data.load_quarterly()
y = q["gdp_nsa"]
"""),
        md(r"""
### The baselines (and why they're not a joke)

| baseline | rule | beats it means |
|----------|------|----------------|
| `naive` | next = last value | you've captured *some* dynamics |
| `seasonal_naive(4)` | next = same quarter last year | you beat pure seasonality |
| `drift` | last value + average slope | you beat a straight-line trend |
| `mean` | the historical average | (only sane for stationary series) |

A "sophisticated" model that can't beat `seasonal_naive` is wasting everyone's
time. Let's first eyeball them on a simple **8-quarter holdout**.
"""),
        co("""
H = 8
train, test = y.iloc[:-H], y.iloc[-H:]
preds = {
    "naive":             B.naive(train, H),
    "seasonal_naive(4)": B.seasonal_naive(4)(train, H),
    "drift":             B.drift(train, H),
    "mean":              B.mean(train, H),
}
fig, ax = plt.subplots()
plots.plot_forecast(train.iloc[-16:], test=test,
                    title="Baselines vs actual — 8-quarter holdout", ax=ax)
xt = test.index.to_timestamp(how="start")
for name, p in preds.items():
    ax.plot(xt, p, marker="o", ms=3, lw=1, label=name)
ax.legend(loc="upper left", fontsize=9); plt.show()
"""),
        md(r"""
### Metrics — and why MASE is the one to trust

* **MAE / RMSE** — error in original units; RMSE punishes big misses harder.
* **MAPE** — percent error; *popular but treacherous* (explodes near zero,
  asymmetric — penalises over-forecasts more than under-forecasts).
* **MASE** — MAE scaled by the in-sample seasonal-naive error.
  **MASE < 1 ⇒ better than naive; MASE > 1 ⇒ worse.** Scale-free and honest.
"""),
        co("""
rows = {name: M.evaluate(test.values, p, y_train=train.values, season_length=4)
        for name, p in preds.items()}
pd.DataFrame(rows).T.sort_values("MASE").round(3)
"""),
        md(r"""
A single holdout is one sample of "how good is this model?" — and the tail you
happened to pick could be unusually easy or hard. **Rolling-origin backtesting**
fixes that: slide the train/test cutoff across the series, refit at each step, and
average the out-of-sample errors. Our `src/backtest.py` does this for *any*
forecaster with the same `f(train, h)` signature.
"""),
        co("""
fcs = {"naive": B.naive, "seasonal_naive(4)": B.seasonal_naive(4),
       "drift": B.drift, "mean": B.mean}
table = bt.compare(y, fcs, initial=40, h=4, step=1, season_length=4)
table.round(3)
"""),
        md(r"""
This table — averaged over *many* cutoffs — is far more trustworthy than the
single holdout above. Note `mean` is hopeless (GDP trends hard), while `drift`
and `seasonal_naive` are competitive. All baselines sit near **MASE ≈ 1** by
construction; that's the line our ETS and ARIMA models must get **below**.

Finally, error almost always **grows with the forecast horizon** — predicting 4
quarters out is harder than 1. Let's quantify it for `drift`:
"""),
        co("""
res = bt.rolling_origin(y, B.drift, initial=40, h=4, step=1)
bt.summarize_by_horizon(res).round(0)
"""),
        md(r"""
MAE/RMSE rise monotonically with `step_ahead` — exactly as expected. Always
report accuracy **per horizon**, because "the model is accurate" is meaningless
without saying *how far ahead*.

---
**You now have the yardstick.** Everything from here (ETS in 03, ARIMA/SARIMA in
04) must beat these baselines on this backtest. That is the whole game.
"""),
    )


# ===========================================================================
# Notebook 03 — Exponential smoothing (ETS): SES -> Holt -> Holt-Winters
# ===========================================================================
def nb03():
    return notebook(
        md(r"""
# P1 · 03 — Exponential smoothing: SES → Holt → Holt-Winters

Exponential smoothing forecasts are **weighted averages of past observations,
with weights decaying exponentially into the past** — recent data matters most.
The family grows by adding components:

| model | components | good for |
|-------|-----------|----------|
| **SES** | level only | flat, no-trend series |
| **Holt** | level + trend | trending series (no seasonality) |
| **Holt-Winters** | level + trend + seasonal | trending *and* seasonal series |

We fit on **log(GDP)** throughout (so multiplicative growth/seasonality become
additive). Every model is also run through the **same backtest** as the baselines.
"""),
        co(PREAMBLE + """
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from src import baselines as B, backtest as bt, classical as C

annual = data.load_annual()
q = data.load_quarterly()
la = np.log(annual)              # log annual real GDP (trend, no seasonality)
nsa = q["gdp_nsa"]               # quarterly NSA (trend + seasonality)
"""),
        md(r"""
### SES — level only (and why it fails on GDP)

SES tracks a single smoothed **level** $\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}$
and forecasts it **flat** into the future. For a trending series like GDP that's
obviously wrong — but seeing it fail builds intuition for what Holt adds.
"""),
        co("""
H = 8
ses = SimpleExpSmoothing(la.values, initialization_method="estimated").fit()
fc = ses.forecast(H)
yrs = np.arange(annual.index.max() + 1, annual.index.max() + 1 + H)
fig, ax = plt.subplots()
ax.plot(annual.index, la.values, color="#264653", label="log GDP")
ax.plot(yrs, fc, color="#e76f51", lw=2, marker="o", label="SES forecast (flat!)")
ax.set(title="SES on log annual GDP — flat forecast can't track a trend",
       xlabel="year", ylabel="log ₹"); ax.legend(); plt.show()
print(f"smoothing level alpha = {ses.params['smoothing_level']:.3f}")
"""),
        md(r"""
### Holt — add a trend component

Holt adds a **trend** $b_t$ that's also exponentially smoothed, so the forecast
extrapolates the recent slope: $\hat y_{t+h} = \ell_t + h\,b_t$. Now the forecast
climbs with the series.
"""),
        co("""
holt = ExponentialSmoothing(la.values, trend="add",
                            initialization_method="estimated").fit()
fc = holt.forecast(H)
fig, ax = plt.subplots()
ax.plot(annual.index, la.values, color="#264653", label="log GDP")
ax.plot(yrs, fc, color="#e76f51", lw=2, marker="o", label="Holt forecast")
ax.set(title="Holt (additive trend) on log annual GDP",
       xlabel="year", ylabel="log ₹"); ax.legend(); plt.show()
print(f"alpha={holt.params['smoothing_level']:.3f}, "
      f"beta(trend)={holt.params['smoothing_trend']:.3f}")
"""),
        md(r"""
### Holt-Winters — add a seasonal component (quarterly)

For the quarterly series we add a **seasonal** term with period 4. Because we fit
on the log, we use *additive* seasonality (additive-on-log = multiplicative on the
original scale — the right choice for GDP). The forecast now has the familiar
saw-tooth restored.
"""),
        co("""
hw = ExponentialSmoothing(np.log(nsa).values, trend="add", seasonal="add",
                          seasonal_periods=4,
                          initialization_method="estimated").fit()
fc_log = hw.forecast(H)
future = pd.period_range(nsa.index.max() + 1, periods=H, freq="Q")
fc = pd.Series(np.exp(fc_log), index=future)

fig, ax = plt.subplots()
plots.plot_forecast(nsa.iloc[-24:], pred=fc,
                    title="Holt-Winters forecast (quarterly NSA, 8 quarters)", ax=ax)
plt.show()
print("Forecast (₹, levels):")
print(fc.round(0))
"""),
        md(r"""
### Does it actually beat the baselines? (the only question that matters)

We backtest Holt, Holt-Winters, and a **damped** variant (damping shrinks the
trend over the horizon, guarding against runaway extrapolation) against the
baselines — same rolling-origin folds as notebook 02.
"""),
        co("""
fcs = {
    "seasonal_naive(4)": B.seasonal_naive(4),
    "drift": B.drift,
    "Holt (no seasonal)": C.ets_forecaster(trend="add", seasonal=None),
    "Holt-Winters": C.ets_forecaster(trend="add", seasonal="add", seasonal_periods=4),
    "Holt-Winters (damped)": C.ets_forecaster(trend="add", seasonal="add",
                                              seasonal_periods=4, damped_trend=True),
}
bt.compare(nsa, fcs, initial=40, h=4, step=1, season_length=4).round(3)
"""),
        md(r"""
Holt-Winters should land around **MASE ≈ 0.64** — a ~36% improvement over the
naive benchmark, and far ahead of the non-seasonal Holt (which ignores the strong
seasonality we measured in notebook 01). Lesson: **match model components to the
structure you diagnosed.** No seasonality term → no chance against a seasonal series.

> ETS as a *family*: statsmodels' `ETSModel` formalises Error-Trend-Seasonal
> combinations (additive/multiplicative × each component) and can auto-select by
> AIC — the same idea as auto-ARIMA, for the smoothing family.

---
**Next (04):** ARIMA/SARIMA — the other classical pillar — and a head-to-head.
"""),
    )


# ===========================================================================
# Notebook 04 — ARIMA / SARIMA (Box-Jenkins)
# ===========================================================================
def nb04():
    return notebook(
        md(r"""
# P1 · 04 — ARIMA & SARIMA: the Box-Jenkins method

ARIMA models the **autocorrelation structure** of a stationarised series:

* **AR(p)** — regress on the last $p$ values,
* **I(d)** — difference $d$ times to remove trend,
* **MA(q)** — regress on the last $q$ forecast errors.

**SARIMA** adds a seasonal triple $(P,D,Q)_m$. From notebook 01 we diagnosed
**d = 1, D = 1, m = 4**. The Box-Jenkins loop: *identify → estimate → diagnose →
forecast.*
"""),
        co(PREAMBLE + """
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from src import baselines as B, backtest as bt, classical as C

annual = data.load_annual(); q = data.load_quarterly()
la = np.log(annual); nsa = q["gdp_nsa"]
"""),
        md(r"""
### Warm-up: plain ARIMA on annual log-GDP (no seasonality)

Annual GDP has no seasonal cycle, so a non-seasonal ARIMA(1,1,1) on the log is a
clean first model. We read the summary (coefficients, AIC) and forecast with a
**prediction interval**.
"""),
        co("""
m = ARIMA(la.values, order=(1, 1, 1)).fit()
print(m.summary().tables[1])
H = 6
fcres = m.get_forecast(H)
mean = np.exp(fcres.predicted_mean)
ci = np.exp(fcres.conf_int(alpha=0.05))
yrs = np.arange(annual.index.max() + 1, annual.index.max() + 1 + H)

fig, ax = plt.subplots()
ax.plot(annual.index, annual.values / 1e12, color="#264653", label="GDP")
ax.plot(yrs, mean / 1e12, color="#e76f51", lw=2, marker="o", label="ARIMA forecast")
ax.fill_between(yrs, ci[:, 0] / 1e12, ci[:, 1] / 1e12, color="#e76f51",
                alpha=0.2, label="95% interval")
ax.set(title="ARIMA(1,1,1) on log annual GDP", xlabel="year",
       ylabel="₹ trillion"); ax.legend(); plt.show()
"""),
        md(r"""
### Diagnose the residuals (this is the step people skip)

If the model captured the structure, the residuals should be **white noise** — no
autocorrelation, roughly normal. The **Ljung-Box** test's null is "residuals are
independent"; we *want* a high p-value. `plot_diagnostics` shows the standardized
residuals, histogram, Q-Q plot, and residual ACF.
"""),
        co("""
fig = m.plot_diagnostics(figsize=(11, 7)); plt.show()
lb = acorr_ljungbox(m.resid[1:], lags=[8], return_df=True)
print("Ljung-Box (lag 8):  p =", round(float(lb['lb_pvalue'].iloc[0]), 3),
      " -> residuals look like white noise" if float(lb['lb_pvalue'].iloc[0]) > 0.05
      else " -> structure remains!")
"""),
        md(r"""
### SARIMA on quarterly NSA: let AIC pick the orders

Rather than agonise over the ACF/PACF, we fix d/D from the stationarity analysis
and grid-search the AR/MA orders, choosing the lowest **AIC** (the hand-rolled
version of `auto_arima`). `src/classical.grid_search_sarima` does exactly this.
"""),
        co("""
grid = C.grid_search_sarima(nsa, d=1, D=1, m=4, max_pq=2, max_PQ=1)
print("Top 5 by AIC:")
print(grid.head(5).to_string())
best = grid.iloc[0]
print("\\nBest:", best['order'], best['seasonal_order'], "AIC", round(best['AIC'], 1))
"""),
        md(r"""
The winner is typically **SARIMA(1,1,1)(0,1,0)₄** — note the seasonal AR/MA are
zero: the seasonal *difference* (D=1) already absorbs the seasonality, leaving the
regular (1,1,1) to model the rest. Parsimony wins on short series. Let's fit it,
diagnose, and forecast 8 quarters with an interval.
"""),
        co("""
order, sorder = best['order'], best['seasonal_order']
mean, lo, hi = C.sarima_forecast_pi(nsa, 8, order=order, seasonal_order=sorder)
future = pd.period_range(nsa.index.max() + 1, periods=8, freq="Q")
fc = pd.Series(mean, index=future)
lo = pd.Series(lo, index=future); hi = pd.Series(hi, index=future)

fig, ax = plt.subplots()
plots.plot_forecast(nsa.iloc[-24:], pred=fc, lower=lo, upper=hi,
                    title=f"SARIMA{order}{sorder} — 8-quarter forecast + 95% PI", ax=ax)
plt.show()
print(fc.round(0))
"""),
        md(r"""
Notice the interval **widens with horizon** and is **asymmetric** in levels (wider
on the upside) — a direct consequence of forecasting in log space. That asymmetry
is *correct* for a multiplicative process and is something naive symmetric
intervals get wrong.

### Final head-to-head: baselines vs ETS vs SARIMA
"""),
        co("""
fcs = {
    "seasonal_naive(4)": B.seasonal_naive(4),
    "drift": B.drift,
    "Holt-Winters": C.ets_forecaster(trend="add", seasonal="add", seasonal_periods=4),
    f"SARIMA{order}{sorder}": C.sarima_forecaster(order, sorder),
    "SARIMA(1,1,1)(1,1,1)4": C.sarima_forecaster((1, 1, 1), (1, 1, 1, 4)),
}
bt.compare(nsa, fcs, initial=40, h=4, step=1, season_length=4).round(3)
"""),
        md(r"""
Both ETS-HW and SARIMA land well under **MASE 0.7**. On *this* short series the
two are close; Holt-Winters often edges it because SARIMA has more parameters to
estimate from few points (the bias–variance trade-off in miniature). The
disciplined answer to "which model?" is never dogma — it's **whatever wins the
backtest**, which is exactly what the capstone (notebook 08) formalises.

---
**Next (05):** the same workflow on the *annual* GDP series end-to-end — a
different frequency and a genuine tiny-sample stress test.
"""),
    )


# ===========================================================================
# Notebook 05 — Annual GDP end-to-end (a tiny-sample stress test)
# ===========================================================================
def nb05():
    return notebook(
        md(r"""
# P1 · 05 — Forecasting *annual* GDP end-to-end (small-sample edition)

So far we've worked the quarterly series. The **annual** real GDP series (1960–2024,
~65 points, *no seasonality*) is a different challenge: a longer span but far fewer
points, dominated by a single exponential trend. It's the classic macro
"small-data" forecast — and a stress test for our workflow.

We run the **same disciplined pipeline** at this new frequency:
transform → baselines → ETS/ARIMA/Theta → backtest → forecast with intervals.
"""),
        co(PREAMBLE + """
from src import baselines as B, backtest as bt, classical as C
annual = data.load_annual()
la = np.log(annual)
print("annual:", annual.index.min(), "->", annual.index.max(), "| n =", len(annual))
"""),
        md(r"""
### No seasonality → simpler models

Annual data has no within-year cycle, so the seasonal machinery (Holt-Winters'
seasonal term, SARIMA's seasonal orders) is irrelevant. The right tools are:

* **Holt's linear trend** (ETS with trend, no seasonal),
* **ARIMA(p,1,q)** on the log (one difference removes the trend),
* **Theta** (a strong, cheap benchmark; `period=1` → non-seasonal).

The baselines simplify too: `seasonal_naive` collapses to `naive`, so we use
`naive`, `drift`, `mean`. MASE uses `season_length=1` (the random-walk scale).
"""),
        co("""
fcs = {
    "naive": B.naive,
    "drift": B.drift,
    "mean":  B.mean,
    "Holt":  C.ets_forecaster(trend="add", seasonal=None),
    "ARIMA(1,1,1)": C.sarima_forecaster((1, 1, 1), (0, 0, 0, 0)),
    "Theta": C.theta_forecaster(period=1),
}
board = bt.compare(annual, fcs, initial=45, h=3, step=1, season_length=1)
board.round(3)
"""),
        md(r"""
> **Reading MASE at multi-step horizons.** Notice *every* model shows MASE > 1.
> That is **expected**, not alarming: the MASE denominator is the **one-step**
> in-sample naive error, but we're forecasting **3 years** ahead — a much harder
> task. So here, judge models by **ranking**, not the 1.0 line: Holt/ARIMA cut the
> error to roughly **a third** of the `naive` row. The absolute "< 1 beats naive"
> rule is exact only for one-step forecasts.

### Forecast the next 6 years with an interval

We take the backtest winner (typically **Holt** or **ARIMA**) and forecast on the
log scale, exponentiating the mean and the 95% band (so the band is asymmetric in
levels — correct for multiplicative growth).
"""),
        co("""
from statsmodels.tsa.arima.model import ARIMA
H = 6
m = ARIMA(la.to_numpy(), order=(1, 1, 1)).fit()
fcres = m.get_forecast(H)
mean = np.exp(fcres.predicted_mean)
ci = np.exp(fcres.conf_int(alpha=0.05))
yrs = np.arange(annual.index.max() + 1, annual.index.max() + 1 + H)

fig, ax = plt.subplots()
ax.plot(annual.index, annual.values / 1e12, color="#264653", label="GDP")
ax.plot(yrs, mean / 1e12, color="#e76f51", lw=2, marker="o", label="ARIMA forecast")
ax.fill_between(yrs, ci[:, 0] / 1e12, ci[:, 1] / 1e12, color="#e76f51",
                alpha=0.2, label="95% interval")
ax.set(title="Annual real GDP — ARIMA(1,1,1) on log, 6-year forecast",
       xlabel="year", ylabel="₹ trillion"); ax.legend(); plt.show()

implied = (np.diff(np.log(np.r_[annual.values[-1], mean])) * 100)
print("Forecast implied annual growth (%):", np.round(implied, 2))
"""),
        md(r"""
The implied growth should sit near the historical **~6%** mean — a sanity check that
the model is extrapolating the real trend, not inventing one.

**Takeaway:** the workflow is *frequency-agnostic*. With only ~65 points, lean
models (Holt, low-order ARIMA, Theta) dominate — there simply isn't data to support
anything fancier. Small samples reward restraint.

---
**Next (06):** two more classical approaches — **Prophet** and the **Theta method** —
on the quarterly series.
"""),
    )


# ===========================================================================
# Notebook 06 — Prophet & Theta
# ===========================================================================
def nb06():
    return notebook(
        md(r"""
# P1 · 06 — Two more classical tools: Prophet & Theta

Beyond ETS and ARIMA, two widely used approaches:

* **Prophet** (Meta) — an *additive decomposition* model:
  $y_t = \text{trend}_t + \text{seasonality}_t + \text{holidays}_t + \varepsilon_t$,
  fit by a Bayesian backend. Popular for being robust and "hands-off".
* **Theta** — decomposes the series into "theta lines" (essentially SES on the
  deseasonalised data plus half the long-run trend). Simple, fast, and famously
  **won the M3 forecasting competition**.

We hold them to the same backtest as everything else.
"""),
        co(PREAMBLE + """
import logging
for _n in ("cmdstanpy", "prophet"):                 # silence the Stan backend
    _lg = logging.getLogger(_n); _lg.setLevel(logging.ERROR); _lg.disabled = True
from src import baselines as B, backtest as bt, classical as C
q = data.load_quarterly(); nsa = q["gdp_nsa"]
"""),
        md(r"""
### Prophet: fit and inspect its decomposition

Prophet wants a two-column frame (`ds` = date, `y` = value). We feed it the
quarter-start timestamps and turn on yearly seasonality (our cycle), with
multiplicative seasonality (the swing grows with the level — same reasoning as our
log transform elsewhere).
"""),
        co("""
from prophet import Prophet
df = pd.DataFrame({"ds": nsa.index.to_timestamp(how="start"),
                   "y": nsa.to_numpy(dtype=float)})
m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
            weekly_seasonality=False, daily_seasonality=False)
m.fit(df)
future = m.make_future_dataframe(periods=8, freq="QS")
forecast = m.predict(future)
fig = m.plot_components(forecast); plt.show()
"""),
        md(r"""
The components panel is Prophet's selling point: it shows the inferred **trend**
(with automatic changepoints) and the **yearly seasonal** shape separately — an
interpretable decomposition you can hand to a stakeholder.

### Prophet's forecast with its native interval
"""),
        co("""
fig, ax = plt.subplots()
hist = nsa.iloc[-24:]
ax.plot(hist.index.to_timestamp(how="start"), hist.values / 1e6,
        color="#264653", label="history")
fut = forecast.tail(8)
ax.plot(fut["ds"], fut["yhat"] / 1e6, color="#e76f51", lw=2, marker="o", label="Prophet")
ax.fill_between(fut["ds"], fut["yhat_lower"] / 1e6, fut["yhat_upper"] / 1e6,
                color="#e76f51", alpha=0.2, label="interval")
ax.set(title="Prophet — 8-quarter forecast", ylabel="real GDP (level)"); ax.legend()
plt.show()
"""),
        md(r"""
### The backtest verdict (Prophet & Theta vs the field)

Prophet refits a Bayesian model each fold, so it's the slowest model here — we use
`step=2` to thin the folds and keep runtime sane (the MASE estimate stays robust).
"""),
        co("""
fcs = {
    "seasonal_naive(4)": B.seasonal_naive(4),
    "Holt-Winters":      C.ets_forecaster(trend="add", seasonal="add", seasonal_periods=4),
    "SARIMA(1,1,1)(0,1,0)4": C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4)),
    "Theta":             C.theta_forecaster(period=4),
    "Prophet":           C.prophet_forecaster(),
}
bt.compare(nsa, fcs, initial=40, h=4, step=2, season_length=4).round(3)
"""),
        md(r"""
Both **Prophet and Theta comfortably beat the baselines** (MASE ≈ 0.73) but land
*behind* the tuned SARIMA/Holt-Winters on this short, clean series. That's a common
real-world pattern: Prophet's strengths — many changepoints, holiday effects, messy
business data with strong multi-seasonality — aren't what a tidy 88-point GDP series
needs. **Theta** punches far above its complexity, which is why it's a respected
benchmark. Lesson: *popular ≠ best for your data* — the backtest decides.

---
**Next (07):** ARIMAX — feeding the model **exogenous** information (deterministic
seasonality and an event dummy).
"""),
    )


# ===========================================================================
# Notebook 07 — ARIMAX (exogenous regressors)
# ===========================================================================
def nb07():
    return notebook(
        md(r"""
# P1 · 07 — ARIMAX: giving ARIMA outside information

Plain (S)ARIMA only looks at the series' own past. **ARIMAX** adds **exogenous
regressors** $X_t$ — external variables that help explain $y_t$:

$$\phi(B)(1-B)^d\,(y_t - \beta^\top X_t) = \theta(B)\,\varepsilon_t.$$

We demonstrate two deterministic (fully reproducible) kinds of exog:

1. **Fourier terms** → smooth seasonality as sine/cosine regressors
   (*dynamic harmonic regression*) — an alternative to seasonal differencing.
2. **An intervention dummy** → a flag for the COVID quarters (2020 Q2/Q3) so the
   shock is *explained*, not baked into the trend.

A real driver (interest rate, monsoon, IIP) would plug into the exact same slot.
"""),
        co(PREAMBLE + """
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src import baselines as B, backtest as bt, classical as C
q = data.load_quarterly(); nsa = q["gdp_nsa"]
"""),
        md(r"""
### 1. Fourier seasonality (dynamic harmonic regression)

Instead of differencing away the season, we add Fourier pairs as regressors and let
ARIMA model the *non*-seasonal dynamics. We backtest it against the
seasonal-differencing SARIMA to see which wins.
"""),
        co("""
fcs = {
    "seasonal_naive(4)":     B.seasonal_naive(4),
    "SARIMA(1,1,1)(0,1,0)4": C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4)),
    "ARIMAX Fourier(2)":     C.arimax_forecaster((1, 1, 1), use_fourier=True, fourier_order=2),
    "ARIMAX Fourier(2)+covid": C.arimax_forecaster((1, 1, 1), use_fourier=True,
                                                   fourier_order=2, covid_dummy=True),
}
bt.compare(nsa, fcs, initial=40, h=4, step=1, season_length=4).round(3)
"""),
        md(r"""
On *this* series, **seasonal differencing (SARIMA) beats Fourier-ARIMAX.** That's an
important, non-obvious lesson:

* **Fourier** assumes a *fixed* seasonal shape and shines when the period is **large**
  (m = 52 weekly, m = 365 daily) where a seasonal difference would be wasteful or
  impossible to estimate.
* **Seasonal differencing** *adapts* to a slowly changing seasonal pattern and is
  ideal for small m (here m = 4). Right tool, right job.

The COVID dummy doesn't improve point accuracy either — which sets up the real
reason to use an intervention dummy: **diagnostics**, not the backtest.

### 2. The intervention dummy's real job: cleaning the 2020 shock

Fit ARIMA on the quarterly **YoY growth** with and without a COVID dummy, and look
at the **2020 residual**. Without the dummy, the −20%+ pandemic crash shows up as a
giant residual that distorts the variance and the AR estimates; with it, the shock
is absorbed by a coefficient and the residuals behave.
"""),
        co("""
g = np.log(nsa).diff(4).dropna() * 100                 # YoY growth, %
g.index = g.index.to_timestamp(how="start")            # DatetimeIndex for clean access
covid = pd.DataFrame(
    {"covid": ((g.index.year == 2020) & (g.index.quarter.isin([2, 3]))).astype(float)},
    index=g.index)

# AR(1) with a constant (trend='c'), with and without the COVID exog
m_no = SARIMAX(g, order=(1, 0, 0), trend="c").fit(disp=False)
m_yes = SARIMAX(g, exog=covid, order=(1, 0, 0), trend="c").fit(disp=False)

resid_no, resid_yes = m_no.resid, m_yes.resid
print("Max |residual| in 2020  without dummy: %.1f pp" % resid_no.loc["2020"].abs().max())
print("Max |residual| in 2020  with dummy   : %.1f pp" % resid_yes.loc["2020"].abs().max())
print("Residual std    without / with       : %.2f / %.2f" % (resid_no.std(), resid_yes.std()))
print("COVID dummy coefficient              : %.1f pp (the modelled shock size)"
      % m_yes.params["covid"])

fig, ax = plt.subplots()
ax.plot(resid_no.index, resid_no.values, label="no dummy", color="#e76f51")
ax.plot(resid_yes.index, resid_yes.values, label="with COVID dummy", color="#2a9d8f")
ax.axhline(0, color="k", lw=0.8)
ax.set(title="AR(1) residuals on YoY growth — the dummy absorbs the 2020 shock",
       ylabel="residual (pp)"); ax.legend(); plt.show()
"""),
        md(r"""
With the dummy, the monstrous 2020 residual collapses and the residual std shrinks —
the model is no longer fighting an event it was never going to predict. **That** is
when exogenous regressors earn their place: encoding *known* external facts (events,
policy changes, campaigns) so the stochastic part of the model stays clean. Whether
they improve out-of-sample *accuracy* is, as always, a question for the backtest.

---
**Next (08):** the capstone — bake **all** eight model families off against each
other and produce the final, defensible forecast.
"""),
    )


# ===========================================================================
# Notebook 08 — Capstone: a defensible forecast (all methods)
# ===========================================================================
def nb08():
    return notebook(
        md(r"""
# P1 · 08 — Capstone: the full bake-off & a defensible forecast

Everything from P1, assembled into the workflow you'd run at work — now with the
**complete** classical slate (baselines, ETS, SARIMA, Theta, Prophet, ARIMAX):

1. **Backtest** every candidate on identical rolling-origin folds.
2. **Select** by MASE.
3. **Refit the winner on all data** and forecast with an interval.
4. **Interpret & caveat** — what could break this.
"""),
        co(PREAMBLE + """
import logging
for _n in ("cmdstanpy", "prophet"):                 # silence the Stan backend
    _lg = logging.getLogger(_n); _lg.setLevel(logging.ERROR); _lg.disabled = True
from src import baselines as B, backtest as bt, classical as C
q = data.load_quarterly(); nsa = q["gdp_nsa"]

candidates = {
    "seasonal_naive(4)":     B.seasonal_naive(4),
    "drift":                 B.drift,
    "Holt-Winters":          C.ets_forecaster(trend="add", seasonal="add", seasonal_periods=4),
    "Holt-Winters (damped)": C.ets_forecaster(trend="add", seasonal="add",
                                              seasonal_periods=4, damped_trend=True),
    "SARIMA(1,1,1)(0,1,0)4": C.sarima_forecaster((1, 1, 1), (0, 1, 0, 4)),
    "SARIMA(1,1,1)(1,1,1)4": C.sarima_forecaster((1, 1, 1), (1, 1, 1, 4)),
    "Theta":                 C.theta_forecaster(period=4),
    "Prophet":               C.prophet_forecaster(),
    "ARIMAX Fourier(2)":     C.arimax_forecaster((1, 1, 1), use_fourier=True, fourier_order=2),
}
"""),
        md("### 1–2. Backtest & select (step=2 — Prophet refits are slow)"),
        co("""
board = bt.compare(nsa, candidates, initial=40, h=4, step=2, season_length=4)
print(board.round(3).to_string())
winner = board.index[0]
print("\\nWinner by MASE:", winner)
"""),
        md("### 3. Refit the winner on ALL data and forecast 8 quarters ahead"),
        co("""
H = 8
future = pd.period_range(nsa.index.max() + 1, periods=H, freq="Q")
if winner.startswith("SARIMA"):
    # parse the order tuples back out of the winning name
    import re
    nums = [int(n) for n in re.findall(r"\\d+", winner)]
    order, sorder = tuple(nums[:3]), tuple(nums[3:7])
    mean, lo, hi = C.sarima_forecast_pi(nsa, H, order=order, seasonal_order=sorder)
else:  # a non-SARIMA winner -> approximate PI via the backtest residual spread
    f = candidates[winner]
    mean = f(nsa, H)
    res = bt.rolling_origin(nsa, f, initial=40, h=4, step=2)
    sd = (res["y_true"] - res["y_pred"]).std()
    lo, hi = mean - 1.96 * sd, mean + 1.96 * sd

fc = pd.Series(np.asarray(mean), index=future)
lo = pd.Series(np.asarray(lo), index=future); hi = pd.Series(np.asarray(hi), index=future)
fig, ax = plt.subplots()
plots.plot_forecast(nsa.iloc[-28:], pred=fc, lower=lo, upper=hi,
                    title=f"Final forecast — {winner}", ax=ax)
plots.save(fig, "p1_final_forecast"); plt.show()

out = pd.DataFrame({"forecast": fc.round(0), "low95": lo.round(0), "high95": hi.round(0)})
out
"""),
        md("### 4. Sanity-check the implied growth"),
        co("""
# YoY implied growth of the forecast vs the historical ~6.6% trend
hist = np.log(nsa).diff(4).dropna() * 100
joined = pd.concat([nsa, fc])
implied = (np.log(joined).diff(4).dropna() * 100).iloc[-H:]
print("Historical YoY growth: mean %.1f%%, last %.1f%%" % (hist.mean(), hist.iloc[-1]))
print("Forecast implied YoY growth (next 8q):")
print(implied.round(2))
"""),
        md(r"""
### What could break this forecast (always write this section)

* **Structural breaks.** COVID (2020 Q2, −26% YoY) shows the model assumes the
  past regime continues. A demonetisation, a pandemic, or a base-year revision
  invalidates it. Classical models *cannot* foresee shocks.
* **Short sample.** 88 quarters is little; the seasonal pattern and trend are
  estimated from ~20 years. Treat far-horizon points with suspicion.
* **Point vs density.** We report a 95% interval, but it assumes the residual
  distribution is stable and (for SARIMA) roughly Gaussian-in-log. Project 3
  replaces this with proper probabilistic / conformal intervals.
* **Only deterministic exog.** ARIMAX (07) used Fourier terms and an event dummy —
  *known* external facts. We used no **data-driven** drivers (rates, monsoon, global
  demand). Project 2 adds feature-based ML to bring those in.

### The office recipe (reusable checklist)

1. Plot the series; **log** if it grows multiplicatively.
2. **Decompose** and measure seasonal strength.
3. **Test stationarity** (ADF + KPSS) → differencing orders.
4. Establish **baselines** + pick a **metric** (MASE) + a **backtest** (rolling origin).
5. Try a **slate** — ETS, (S)ARIMA, Theta, Prophet, ARIMAX; **diagnose residuals**.
6. **Select by backtest**, refit on all data, forecast **with intervals**.
7. Write the **caveats**. A forecast without an interval and a caveat is a guess.

---
**Project 1 complete.** Across 8 notebooks you can now diagnose, transform, model
(the full classical slate), evaluate honestly, and defend a forecast. Project 2
moves to feature-based machine-learning forecasting.
"""),
    )


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "00_first_look.ipynb": nb00,
        "01_decomposition_stationarity.ipynb": nb01,
        "02_baselines_evaluation.ipynb": nb02,
        "03_exponential_smoothing.ipynb": nb03,
        "04_arima_sarima.ipynb": nb04,
        "05_annual_forecast.ipynb": nb05,
        "06_prophet_theta.ipynb": nb06,
        "07_arimax.ipynb": nb07,
        "08_capstone.ipynb": nb08,
    }
    for fname, fn in builders.items():
        nbf.write(fn(), NB_DIR / fname)
        print("wrote", fname)


if __name__ == "__main__":
    main()
