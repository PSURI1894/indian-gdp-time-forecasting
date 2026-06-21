# Code Walkthrough — the `src/` toolkit, line by line

This document explains **every module in `src/`**, block by block, with the
*reasoning* behind each design choice. If `DEEP_DIVE.md` is the "why does the math
work", this is the "why is the code written this way". Read it once end-to-end;
afterwards the notebooks read like prose.

## Architecture in one picture

```
        ┌─────────────┐   f(train, h) -> ŷ[h]    ┌──────────────┐
data.py │  loaders    │ ───────────────────────► │ backtest.py  │
(fetch  │ load_*()    │        the ONE contract  │ rolling_origin│──► metrics.py
 once)  └─────────────┘   every model implements └──────────────┘    (MASE…)
                                  ▲      ▲
                                  │      │
                         baselines.py  classical.py
                         (naive,…)     (ETS, SARIMA)
```

Three deliberate separations:

1. **Fetch vs. use.** `data.py` hits the network *once* and writes CSVs; notebooks
   only ever call the fast, offline `load_*()` functions. Reproducible and quick.
2. **Model vs. evaluator.** Every model — a one-line baseline or a SARIMA — is just
   a function `f(train, h) -> array[h]`. `backtest.py` knows nothing about the
   model internals, so it evaluates all of them *identically*. No special-casing =
   no accidental unfair advantage.
3. **Scoring is its own module.** `metrics.py` has no model or data dependencies, so
   it's trivially testable and reusable.

---

## `src/data.py` — acquisition & caching

### Module docstring & imports
```python
from __future__ import annotations
import json
import urllib.request
from pathlib import Path
import pandas as pd
```
- `from __future__ import annotations` makes type hints lazy strings — lets us
  write `-> pd.DataFrame` cheaply and avoids import-order headaches.
- **Only `json` + `urllib` for networking** (both stdlib) — *justification:* the data
  layer must run the instant `pandas` is installed, with **no API keys** and no
  extra dependency (`requests`) to install or break. Fewer moving parts in the part
  most likely to fail (the network).

### Paths
```python
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
```
- `Path(__file__).resolve()` is the absolute path of `data.py`; `.parents[1]` walks
  up two levels (`src/` → repo root). *Justification:* paths are anchored to the
  **file**, not the current working directory, so `python -m src.data` and a
  notebook three folders deep both resolve `data/` correctly.

### Source identifiers
```python
WB_URL = ("https://api.worldbank.org/v2/country/IND/indicator/"
          "NY.GDP.MKTP.KN?format=json&per_page=20000")
DBNOMICS = ("https://api.db.nomics.world/v22/series/OECD/"
            "DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR/{code}?observations=1")
OECD_NSA = "Q.N.IND.S1.S1.B1GQ._Z._Z._Z.XDC.Q.N.T0102"
OECD_SA  = "Q.Y.IND.S1.S1.B1GQ._Z._Z._Z.XDC.Q.N.T0102"
```
- `per_page=20000` — World Bank paginates; one huge page avoids a paging loop for a
  series that's only ~65 rows.
- The OECD codes differ in **one field**: position 2 is the *adjustment* flag —
  `N` = Neither seasonally nor calendar adjusted (raw, **seasonal**), `Y` = adjusted.
  Keeping them as named constants documents exactly which series we pulled (vital —
  the difference between NSA and SA is the entire point of having both).

### `_get` — the one network primitive
```python
def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()
```
- A custom `User-Agent` header (`_HEADERS`) — some APIs reject the default Python
  agent. A `timeout` so a hung connection fails loudly instead of blocking forever.
  `with ...` guarantees the socket closes. Leading underscore = "private helper".

### `fetch_worldbank_annual`
```python
payload = json.loads(_get(WB_URL))
records = payload[1]              # payload[0] is pagination metadata
rows = [(int(r["date"]), float(r["value"]))
        for r in records if r["value"] is not None]
df = pd.DataFrame(rows, columns=["year", "gdp"]).sort_values("year")
return df.reset_index(drop=True)
```
- World Bank returns a 2-element JSON array: `[metadata, data]` — hence `payload[1]`.
- The comprehension **drops nulls** (`value is not None`) and **coerces types**
  (`date` is a string in the JSON; `int`/`float` make a clean numeric frame).
- `sort_values("year")` — the API returns newest-first; time series must be
  chronological. `reset_index(drop=True)` discards the now-shuffled old index.

### `_fetch_dbnomics_quarterly`
```python
doc = payload["series"]["docs"][0]
periods = doc["period"]          # e.g. '2004-Q2'
values  = doc["value"]
pairs = [(p, v) for p, v in zip(periods, values) if v is not None]
idx = pd.PeriodIndex([p.replace("-", "") for p in (p for p, _ in pairs)], freq="Q")
return pd.Series([v for _, v in pairs], index=idx, name="gdp").sort_index()
```
- DBnomics returns parallel `period` / `value` lists; `zip` re-pairs them and we
  drop nulls in lockstep so they stay aligned.
- `'2004-Q2'.replace("-", "")` → `'2004Q2'`, which `pd.PeriodIndex(..., freq="Q")`
  parses natively. *Justification for a `PeriodIndex` (not a `DatetimeIndex`):* a
  quarter is an **interval**, not an instant. A `Period` knows "2004Q2" spans
  Apr–Jun; `.diff(4)` means "year-on-year" unambiguously; seasonal models read the
  period number directly. Far less error-prone than juggling timestamps.

### `build` — fetch → disk (run once)
```python
out.insert(0, "date", q.index.to_timestamp(how="start"))
out.insert(1, "quarter", q.index.astype(str))
out.to_csv(PROC / "quarterly_real_gdp.csv", index=False)
```
- CSV can't store a `PeriodIndex`, so we serialise it **two ways**: a `date`
  timestamp (quarter start — convenient for plotting libraries) and a `quarter`
  string `'2004Q2'` (the source of truth we rebuild the `PeriodIndex` from).
  Belt-and-braces so the round-trip is lossless.

### Loaders — the fast path notebooks use
```python
def load_quarterly() -> pd.DataFrame:
    df = pd.read_csv(PROC / "quarterly_real_gdp.csv")
    idx = pd.PeriodIndex(df["quarter"], freq="Q")     # '2004Q2' -> Period
    return df.set_index(idx)[["gdp_nsa", "gdp_sa"]]
```
- Reconstructs the `PeriodIndex` from the `quarter` column, so callers get the rich
  index back. `[[...]]` selects just the two value columns (drops the helper
  `date`/`quarter` columns from the returned frame).

---

## `src/metrics.py` — scoring (no model/data deps)

### `_align` — the shared guard
```python
def _align(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred
```
- Every metric calls this first. It coerces lists / Series / arrays to float arrays
  and **fails fast on a length mismatch** — the single most common silent bug in
  forecasting (forecasting `h` steps but scoring against `h-1` actuals). Better a
  loud error than a quietly wrong MAE.

### `mae`, `rmse`, `mape`, `smape`
```python
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mape(y_true, y_pred): return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
def smape(...):           denom = (|y_true| + |y_pred|) / 2; ...
```
- Vectorised NumPy — no Python loops, so it's fast and reads like the formula.
- `float(...)` wraps each result: returns a plain Python float, not a 0-d NumPy
  scalar, so it prints and serialises cleanly in tables.
- `mape` divides by `y_true` — **this is the danger** the docstring warns about: a
  zero or near-zero actual makes it explode. We compute it (reviewers expect it) but
  *rank* models by MASE.

### `mase` — the headline
```python
m = season_length
if len(y_train) <= m:
    raise ValueError("training set too short for the chosen season_length")
scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))   # in-sample seasonal-naive MAE
if scale == 0:
    raise ValueError("naive scale is zero (flat training series)")
return float(np.mean(np.abs(y_true - y_pred)) / scale)
```
- `y_train[m:] - y_train[:-m]` is the vectorised **seasonal-naive error**: every
  point minus the point `m` steps earlier. For `m=4` that's "this quarter vs the
  same quarter last year". Its mean-abs is the **denominator** that scales the
  forecast MAE.
- Two guards: too-short training (can't form the difference) and a flat series
  (`scale == 0` → divide-by-zero). *Justification:* the scale is computed on
  **training** data, never the test set — so MASE is comparable across models and
  never leaks test information into its own denominator.
- Result interpretation lives in the formula: **< 1 beats naive, > 1 worse**.

### `evaluate`
```python
out = {"MAE":…, "RMSE":…, "MAPE":…, "sMAPE":…}
if y_train is not None:
    out["MASE"] = mase(...)
return out
```
- One call → all metrics as a dict (one table row). MASE is **opt-in** because it
  needs the training series; everything else needs only `(y_true, y_pred)`.

---

## `src/baselines.py` — the benchmarks

Each baseline is the *minimum* a real model must beat. All share the
`f(train, h) -> array[h]` contract.

```python
def naive(train, h):  return np.repeat(train.iloc[-1], h)          # last value, flat
def mean(train, h):   return np.repeat(train.mean(), h)            # historical mean
def drift(train, h):
    n = len(train)
    slope = (train.iloc[-1] - train.iloc[0]) / (n - 1)            # avg slope, ends only
    return train.iloc[-1] + slope * np.arange(1, h + 1)           # extend the line
```
- `naive` = the random walk; the honest default for many financial series.
- `drift` draws a straight line through the **first and last** points and extends it
  — equivalent to an ARIMA(0,1,0) with drift. Cheap trend benchmark.
- `mean` is deliberately included to *fail* on trending GDP — it teaches that a
  baseline must match the data's character (it's only sane for stationary series).

```python
def seasonal_naive(season_length):
    m = season_length
    def _f(train, h):
        last_season = train.iloc[-m:].to_numpy()      # the most recent full cycle
        reps = int(np.ceil(h / m))                    # how many cycles to cover h
        return np.tile(last_season, reps)[:h]         # tile, then trim to exactly h
    return _f
```
- A **factory**: `seasonal_naive(4)` *returns* a forecaster. *Justification:* the
  season length is a property of the data, not a per-call argument — baking it in
  keeps the returned function's signature `f(train, h)`, identical to every other
  model, so the backtest treats it the same. `np.tile(...)[:h]` repeats last year's
  four quarters across the horizon and trims any overshoot.

---

## `src/backtest.py` — the honest evaluator

### `rolling_origin` — the core loop
```python
n = len(y); values = y.to_numpy(dtype=float); index = y.index
cutoff = initial
while cutoff + h <= n:                      # stop when a full h-window won't fit
    start = 0 if expanding else cutoff - initial
    train = y.iloc[start:cutoff]            # everything BEFORE the cutoff
    yhat = np.asarray(forecaster(train, h), dtype=float)
    if yhat.shape[0] != h:
        raise ValueError(...)               # contract check: model must return h
    for k in range(h):
        j = cutoff + k                      # the actual we're predicting
        rows.append({"cutoff_idx": cutoff, "cutoff_label": str(index[cutoff-1]),
                     "step_ahead": k+1, "index": str(index[j]),
                     "y_true": values[j], "y_pred": yhat[k]})
    cutoff += step
return pd.DataFrame(rows)
```
Line by line, the anti-leakage design:
- `train = y.iloc[start:cutoff]` — slicing **stops at `cutoff`**, so training data is
  strictly the past. The model literally cannot see the future it's scored on.
- `expanding` toggles the two textbook schemes: `start=0` grows the window
  (use all history); `start=cutoff-initial` slides a fixed window (adapt to recent
  regimes, e.g. post-COVID). One flag, both methods.
- `while cutoff + h <= n` — the loop only runs folds where a **full** `h`-step actual
  exists, so every fold is comparable (no short final fold skewing the average).
- The **shape check** enforces the `f(train,h)` contract at runtime — a model that
  returns the wrong length is caught here, not three tables later.
- We record one **long-format** row per `(cutoff, step_ahead)` — tidy data that
  trivially groups by horizon afterwards.

### `summarize`
```python
return M.evaluate(results["y_true"].to_numpy(), results["y_pred"].to_numpy(),
                  y_train=y_train_for_scale.to_numpy(), season_length=season_length)
```
- Pools **all** out-of-sample predictions across folds into one metric set. The
  `y_train_for_scale` series feeds MASE's denominator (passed in, not recomputed, so
  the scale is identical across every model being compared).

### `summarize_by_horizon`
```python
g = results.groupby("step_ahead")
out = g.apply(lambda d: pd.Series({"MAE": M.mae(d.y_true, d.y_pred),
                                   "RMSE": M.rmse(d.y_true, d.y_pred)}),
              include_groups=False)
```
- Groups by `step_ahead` to show **error growing with horizon** (1-step is easy,
  4-step is hard). `include_groups=False` silences a pandas deprecation and avoids
  feeding the grouping column into the lambda.

### `compare`
```python
for name, f in forecasters.items():
    res = rolling_origin(y, f, initial=initial, h=h, step=step, expanding=expanding)
    rows[name] = summarize(res, y, season_length=season_length)
table = pd.DataFrame(rows).T
sort_key = "MASE" if "MASE" in table.columns else "MAE"
return table.sort_values(sort_key)
```
- Runs **identical folds** for every model (same `initial/h/step`), then stacks the
  metric dicts into a leaderboard sorted by MASE. This function *is* the
  decision-making tool — the capstone calls it and reads row 0 as the winner.

---

## `src/classical.py` — ETS & SARIMA as forecasters

### `_prep` — the log switch
```python
def _prep(train, use_log):
    y = np.asarray(train, dtype=float)
    return np.log(y) if use_log else y
```
- Centralises the log transform so every wrapper handles it the same way and we
  never forget to invert it. Passing a **NumPy array** (not the Series) to
  statsmodels sidesteps index-frequency warnings — we only need the values; the
  backtest tracks the dates separately.

### `ets_forecaster` — Holt / Holt-Winters factory
```python
def ets_forecaster(trend="add", seasonal=None, seasonal_periods=None,
                   damped_trend=False, use_log=True):
    def f(train, h):
        y = _prep(train, use_log)
        model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal,
                    seasonal_periods=seasonal_periods, damped_trend=damped_trend,
                    initialization_method="estimated").fit()
        fc = np.asarray(model.forecast(h), dtype=float)
        return np.exp(fc) if use_log else fc
    return f
```
- A **factory** again: configure the components once (`seasonal="add", seasonal_periods=4`
  → Holt-Winters), get back a plain `f(train, h)`. *Justification:* the backtest must
  **refit on each fold's training data**, so we can't hand it a pre-fit model — we
  hand it a recipe that fits fresh every call.
- `initialization_method="estimated"` lets MLE learn the initial level/trend/seasonal
  states (more robust than the legacy heuristic on short series).
- `np.exp(fc)` inverts the log. (Caveat from `DEEP_DIVE.md`: this is the *median*,
  marginally below the mean — fine for point accuracy.)

### `sarima_forecaster`
```python
model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc = np.asarray(model.forecast(h), dtype=float)
return np.exp(fc) if use_log else fc
```
- `enforce_stationarity=False, enforce_invertibility=False` — *justification:* on
  short samples the MLE optimiser sometimes wanders to the boundary of the
  stationary region; relaxing these constraints lets it **converge** instead of
  erroring, at no practical cost to the forecast. `disp=False` mutes the optimiser's
  iteration log.

### `sarima_forecast_pi` — point **and** interval
```python
fc = model.get_forecast(h)
mean = np.asarray(fc.predicted_mean, dtype=float)
ci = np.asarray(fc.conf_int(alpha=alpha), dtype=float)   # shape (h, 2)
lower, upper = ci[:, 0], ci[:, 1]
if use_log:
    return np.exp(mean), np.exp(lower), np.exp(upper)
```
- `get_forecast` (vs `forecast`) returns a results object carrying the **prediction
  interval**. We exponentiate the bounds too — so in level space the interval is
  **asymmetric** (wider above), which is *correct* for a multiplicative process and
  something naive symmetric ± intervals get wrong.

### `grid_search_sarima` — hand-rolled auto-ARIMA
```python
for p in range(max_pq+1):
  for qq in range(max_pq+1):
    for P in range(max_PQ+1):
      for Q in range(max_PQ+1):
        order, sorder = (p, d, qq), (P, D, Q, m)
        try:
            res = SARIMAX(yv, order=order, seasonal_order=sorder,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            rows.append({"order":order, "seasonal_order":sorder, "AIC":res.aic, "BIC":res.bic})
        except Exception:
            continue
return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
```
- **`d` and `D` are fixed** (from the stationarity analysis), and we only search the
  AR/MA orders — exactly what `auto_arima` does, written explicitly so you see the
  mechanism. AIC rewards fit and **penalises parameters**, so it naturally prefers
  parsimony (which is why `(1,1,1)(0,1,0)₄` wins on this short series).
- `try/except: continue` — some order combos fail to converge; we skip them rather
  than abort the whole search. Returns sorted by AIC so row 0 is the pick.

### `theta_forecaster`
```python
s = pd.Series(np.log(train.to_numpy()) if use_log else train.to_numpy(), index=train.index)
fc = ThetaModel(s, period=period).fit().forecast(h)
return np.exp(fc) if use_log else np.asarray(fc)
```
- We rebuild a `Series` **with the original index** because `ThetaModel`
  deseasonalises internally and needs the period structure. `period=4` for
  quarterly, `period=1` to switch seasonality off (the annual notebook). Same
  log-in / exp-out convention as the other wrappers, so it's drop-in comparable.

### `prophet_forecaster`
```python
def f(train, h):
    from prophet import Prophet                 # lazy import
    ds = train.index.to_timestamp(how="start")
    df = pd.DataFrame({"ds": ds, "y": train.to_numpy()})
    m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                weekly_seasonality=False, daily_seasonality=False).fit(df)
    future = m.make_future_dataframe(periods=h, freq="QS")
    return m.predict(future)["yhat"].to_numpy()[-h:]
```
- **Lazy `import prophet` inside `f`** — *justification:* Prophet pulls a heavy Stan
  backend; importing it lazily means the whole toolkit (and every other model) still
  works if Prophet isn't installed. You only pay for it if you actually use it.
- Prophet speaks **timestamps**, so we convert the `PeriodIndex` to quarter starts;
  `multiplicative` seasonality mirrors our log reasoning (swing grows with level);
  the weekly/daily cycles are switched off (irrelevant to quarterly data). We slice
  `[-h:]` because `predict` returns fitted history **plus** the future.
- (Module top sets `logging.getLogger("cmdstanpy"/"prophet").setLevel(ERROR)` so the
  Stan backend doesn't spam the notebooks.)

### `_build_exog` — deterministic regressors for ARIMAX
```python
future_index = pd.period_range(train_index[-1] + 1, periods=h, freq=freq)
if use_fourier:
    dp = DeterministicProcess(train_index, additional_terms=[Fourier(period, fourier_order)])
    parts_in.append(dp.in_sample()); parts_out.append(dp.out_of_sample(h))
if covid_dummy:
    pulse = lambda idx: pd.DataFrame({"covid": ((idx.year==2020) & idx.quarter.isin([2,3])).astype(float)}, index=idx)
    parts_in.append(pulse(train_index)); parts_out.append(pulse(future_index))
```
- The crux of ARIMAX is that the exog must be **known for the future too** — you
  can't forecast with a regressor you don't have. So everything here builds **two
  aligned blocks**: `in_sample` (length = train) and `out_of_sample(h)` (length = h).
- `DeterministicProcess` guarantees the Fourier terms **continue the same phase** out
  of sample (no seam at the forecast boundary). The COVID `pulse` is purely
  date-driven, so it's trivially 0 in the future — exactly right for a one-off event.
- *Justification for deterministic exog:* fully reproducible, no extra data to fetch
  or align, and it isolates the *mechanism* of ARIMAX. A real driver series would
  replace these blocks with no change to `arimax_forecaster`.

### `arimax_forecaster`
```python
y = _prep(train, use_log)
Xin, Xout = _build_exog(train.index, h, use_fourier, fourier_order, period, covid_dummy)
model = SARIMAX(y, exog=None if Xin is None else Xin.to_numpy(),
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc = model.forecast(h, exog=None if Xout is None else Xout.to_numpy())
```
- Same `SARIMAX` engine as `sarima_forecaster`, now with `exog`. The forecast call
  **must be given `Xout`** — the future regressor values — or it can't project the
  regression part. With Fourier exog and no seasonal order, this *is* dynamic
  harmonic regression; add `covid_dummy=True` to fold in the intervention.

---

## `src/plots.py` — thin, on purpose

```python
def _x(series):
    idx = series.index
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp(how="start")
    return idx
```
- One helper converts a `PeriodIndex` to timestamps **only for plotting**
  (matplotlib wants datetimes on the x-axis) while the data keeps its `Period` index
  everywhere else.

```python
def plot_forecast(train, test=None, pred=None, lower=None, upper=None, title="", ax=None):
    ...
    if lower is not None and upper is not None:
        ax.fill_between(_x(pred), lower.values, upper.values, alpha=0.2, ...)
```
- Optional layers (`test`, `pred`, interval) so the *same* function draws a bare
  history, a backtest overlay, or a final forecast-with-interval. `fill_between`
  shades the prediction interval. *Justification for keeping plotting thin:* the
  notebooks should still **show** the matplotlib so you learn it — these helpers only
  remove the boilerplate that would otherwise be copy-pasted into every cell.

---

## `src/features.py` — series → supervised table (Project 2)

### `feature_row` — the one function that prevents train/serve skew
```python
def feature_row(hist, pos, lags, rolls, fourier_order, period, trend):
    f = {}
    for L in lags:        f[f"lag{L}"] = hist[-L]
    for w in rolls:
        window = hist[-w:]
        f[f"rmean{w}"] = float(np.mean(window)); f[f"rstd{w}"] = float(np.std(window))
    if trend:             f["trend"] = float(len(hist))
    if fourier_order:
        q = pos.quarter
        for k in range(1, fourier_order+1):
            a = 2*np.pi*k*(q-1)/period
            f[f"sin{k}"] = float(np.sin(a)); f[f"cos{k}"] = float(np.cos(a))
    return f
```
- **`hist` is everything observed *before* `pos`** — `hist[-1]` is the latest known
  value. That single convention is the anti-leakage guarantee: a feature literally
  cannot see the target, because the target isn't in `hist`.
- *Justification for one shared function:* this exact code builds **both** the
  training matrix (looping `feature_row(vals[:t], …)`) **and** each step of a
  recursive forecast (`feature_row(growing_hist, …)`). If a feature is ever wrong,
  it's wrong identically in train and serve — so the model never sees a distribution
  at inference it didn't see in training. Train/serve skew is designed out, not
  tested for.
- Seasonality comes from `pos.quarter` (deterministic, known for any future date);
  `trend` is just `len(hist)` — included so §01 can *show* that trees can't use it.

### `build_supervised`
```python
start = max(max(lags), max(rolls) if rolls else 0)
for t in range(start, len(vals)):
    rows.append(feature_row(vals[:t], idx[t], …))
    tgt.append(vals[t] if target=="level" else vals[t] - vals[t-1])
```
- `start` skips the first rows that lack enough history for the longest lag/window.
- The **target switch** is the crux of Project 2: `"level"` predicts $y_t$;
  `"growth"` predicts $y_t-y_{t-1}$ (a first difference of the log = growth rate),
  which is stationary so trees can handle it.

## `src/ml.py` — LightGBM forecasters (recursive, direct, quantile)

### `gbm_recursive_forecaster`
```python
hist = list(s.to_numpy())                    # log-levels, grows as we forecast
for step in range(1, h+1):
    pos = last_period + step
    feat = feature_row(np.asarray(hist), pos, …)
    yhat = float(model.predict(pd.DataFrame([feat])[X.columns])[0])
    level = hist[-1] + yhat if target=="growth" else yhat
    hist.append(level)                       # <-- prediction fed back in
    preds.append(level)
```
- The loop **appends each prediction to `hist`**, so the next iteration's lags are
  computed from forecasts — that's what "recursive" means. `[X.columns]` re-orders the
  one-row frame to match training column order (LightGBM is positional).
- For a growth target we **cumulate**: `level = hist[-1] + yhat`. The back-transform
  `np.exp` happens once at the end.

### `gbm_direct_forecaster`
```python
for k in range(1, h+1):
    for t in range(start, len(vals)-k):
        rows.append(feature_row(vals[:t+1], idx[t+k], …))     # features at t, target at t+k
        tgt.append(vals[t+k] - vals[t] if target=="growth" else vals[t+k])
    models[k] = LGBMRegressor(**p).fit(pd.DataFrame(rows), tgt)
```
- **One model per horizon `k`.** Note `vals[:t+1]` (includes `t`) paired with target
  `t+k`: the model learns to jump `k` steps in one shot, so there's no compounding.
  The `if not rows:` guard handles horizons too long for the sample (returns the last
  value). At forecast time each `models[k]` predicts directly from the latest row.

### `gbm_quantile_forecast`
```python
qs = {"lo": alpha, "mid": 0.5, "hi": 1-alpha}
models = {k: LGBMRegressor(objective="quantile", alpha=a, **p).fit(X, y) for k,a in qs.items()}
```
- Three models trained with the **pinball loss** at the 5th/50th/95th percentiles →
  a median plus a 90% band, rolled forward recursively (each quantile keeps its own
  `hist`). A Project-3 teaser; the docstring flags the crossing / fast-widening
  caveats.

### `DEFAULT_PARAMS`
```python
DEFAULT_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=7,
                      min_child_samples=8, subsample=0.8, colsample_bytree=0.9,
                      reg_lambda=1.0, verbose=-1, random_state=0)
```
- Deliberately **timid** — shallow trees, slow learning, regularisation — because
  ~80 rows overfit instantly. `verbose=-1` mutes LightGBM; `random_state=0` makes
  runs reproducible. The §03 sweep shows even `num_leaves=7` is too generous here.

---

## `src/probabilistic.py` — scoring & building intervals (Project 3)

### Scores
```python
def winkler_score(y_true, lo, hi, alpha):
    width = hi - lo
    below = (lo - y_true) * (y_true < lo)
    above = (y_true - hi) * (y_true > hi)
    return float(np.mean(width + (2/alpha) * (below + above)))
```
- The one score that ranks interval methods *honestly*: width **plus** a miss
  penalty of `2/alpha × shortfall`. `coverage` and `mean_interval_width` report the
  two halves separately; Winkler is the proper combined score. `pinball_loss` is the
  quantile analogue.

### The interval backtest
```python
def rolling_origin_intervals(y, g, initial, h, step=1):
    while cutoff + h <= n:
        mean, lo, hi = g(y.iloc[:cutoff], h)     # interval forecaster
        ... record y_true, mean, lo, hi per step_ahead ...
```
- The probabilistic twin of `backtest.rolling_origin`: the forecaster `g(train,h)`
  returns **(mean, lo, hi)** instead of a point. `summarize_intervals` then turns a
  run into coverage / width / Winkler — so every interval method is judged on
  identical folds, same as the point models.

### Conformal calibration
```python
def conformal_forecast(y, point_forecaster, h, initial, alpha, ...):
    res = backtest_residuals(y, point_forecaster, initial, h, step, use_log)
    q = res.groupby("step_ahead")["resid"].apply(lambda r: np.quantile(np.abs(r), 1-alpha))
    mean = point_forecaster(y, h)
    return mean, mean*np.exp(-q), mean*np.exp(q)     # multiplicative band (log space)
```
- Calibrate the `(1-alpha)` quantile of **out-of-sample** `|residual|` **per horizon**
  (longer horizons → wider), then wrap the point forecast. Residuals are computed in
  **log space**, so `exp(±q)` gives a *multiplicative*, asymmetric-in-levels band —
  correct for GDP. `conformal_split_evaluate` does the honest version: calibrate on
  early folds, measure coverage on later ones.

### Adaptive Conformal Inference
```python
hw = np.quantile(np.abs(resid_hist), 1 - alpha_t)        # working level
miss = int(not (lo <= y_t <= hi))
alpha_t = np.clip(alpha_t + gamma * (alpha - miss), 1e-3, 0.999)   # the ACI update
```
- The whole of ACI is that one update line: a **miss widens** the next interval
  (lowers `alpha_t`), a hit tightens it. It's feedback control on coverage, so the
  long-run rate tracks the target **even when the residual distribution drifts** —
  which point-in-time conformal can't promise on non-exchangeable time series.

---

## How the notebooks compose it (the payoff)

A whole model comparison collapses to a few lines because the contract is uniform:

```python
from src import data, baselines as B, classical as C, backtest as bt
y = data.load_quarterly()["gdp_nsa"]
bt.compare(
    y,
    {"seasonal_naive(4)": B.seasonal_naive(4),
     "Holt-Winters":      C.ets_forecaster(trend="add", seasonal="add", seasonal_periods=4),
     "SARIMA":            C.sarima_forecaster((1,1,1),(0,1,0,4))},
    initial=40, h=4, step=1, season_length=4,
)
```

`compare` refits each model on identical rolling-origin folds and returns the MASE
leaderboard. Swap in a LightGBM model in Project 2 and **this code doesn't change** —
that's the entire reason for the `f(train, h)` contract.
