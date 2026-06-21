# Project 2 — Deep Dive (the math & mechanics)

Why feature-based ML forecasting works, and the two traps that sink it.

---

## 1. The supervised reframing

A forecaster is a function of the past. ML makes that explicit by building, for each
time $t$, a feature vector from information available **before** $t$:

$$X_t = \big[\,y_{t-1},\dots,y_{t-p},\ \bar y_{t-1:t-w},\ \mathrm{sd}_{t-1:t-w},\
            \sin/\cos\ \text{terms},\ \text{trend}\,\big],\qquad
  \hat y_t = g(X_t),$$

and fitting a regressor $g$ (here, gradient-boosted trees). This buys the whole ML
toolbox — nonlinearity, interactions, exogenous variables, multi-series training —
at the price of having to **engineer temporal structure** and **prevent leakage**.

**Leakage** = any feature in $X_t$ that encodes information from time $\ge t$. It
makes notebook metrics look great and production collapse. Defences: features use
only `hist[:t]`; rolling stats are computed on shifted windows; cross-validation is
rolling-origin, never shuffled; scalers/encoders are fit inside each fold.

## 2. Why trees cannot extrapolate (the core trap)

A regression tree predicts the **mean training target of the leaf** an input falls
into. Formally, for a single tree the prediction is

$$\hat g(x) = \sum_{\ell\in\text{leaves}} \bar y_\ell \,\mathbf 1[x \in R_\ell],$$

where $\bar y_\ell$ is an average of *training* targets. A boosted ensemble is a sum
of such trees. Therefore **every prediction is a combination of values seen in
training** — the output is bounded by the training target range. On a trending series
with a *level* target, future inputs fall in the extreme leaf, whose $\bar y_\ell$ is
the old maximum: the forecast **saturates** and cannot make new highs.

**Fix — model a stationary target.** Predict the **growth** $r_t = \log y_t -
\log y_{t-1}$, which fluctuates around a small constant (so the tree interpolates),
then cumulate:

$$\log \hat y_{t+k} = \log y_t + \sum_{j=1}^{k} \hat r_{t+j},\qquad
  \hat y_{t+k} = \exp(\cdot).$$

The *level* is now free to climb arbitrarily while the *model* only ever predicts
in-range growth. This is the ML counterpart of ARIMA's differencing $d$.

*Empirical proof:* level target → MASE **1.74** (worse than naive); growth target →
**~0.82** (beats baselines). Same model, same features.

## 3. Multi-step strategies

To forecast $h$ steps from a one-step learner:

**Recursive (iterated).** Use the model's own prediction as the next lag:
$$\hat y_{t+1}=g(X_t),\quad \hat y_{t+2}=g(\tilde X_{t+1}),\ \dots$$
where $\tilde X$ substitutes predictions for unknown actuals. One model; uses the
latest information; but the feature errors **compound**: $\operatorname{Var}$ of the
$k$-step forecast grows roughly with accumulated one-step errors.

**Direct.** Train a separate model per horizon:
$$\hat y_{t+k} = g_k(X_t),\qquad k=1,\dots,h,$$
fitting $g_k$ on targets $y_{t+k}$ (or cumulative growth $\log y_{t+k}-\log y_t$).
No compounding, but each $g_k$ trains on fewer rows and can't use its own
intermediate predictions. (Hybrid *DirRec* combines both.)

Guidance: recursive for short horizons / scarce data / highly-informative last
value; direct for longer horizons where compounding dominates. *Here, direct wins*
(MASE 0.669 vs 0.819).

## 4. Gradient boosting in one paragraph

Boosting builds an additive model $F_M(x)=\sum_{m=1}^{M}\nu\,h_m(x)$ stage-wise: each
shallow tree $h_m$ is fit to the **negative gradient** of the loss at the current
predictions (for squared error, that's the residuals), shrunk by a learning rate
$\nu$. LightGBM makes this fast with histogram-binned features and leaf-wise growth.
Key small-data knobs: `num_leaves` (capacity — keep small), `learning_rate` +
`n_estimators` (slow learning, many small steps), `min_child_samples`, and
`reg_lambda` / subsampling (regularisation). On ~80 rows, **less capacity wins**
(`num_leaves=3` beat larger trees on the backtest).

## 5. Quantile regression for intervals

Train the model to minimise the **pinball (quantile) loss** at level $\tau$:
$$L_\tau(y,\hat y)=\begin{cases}\tau\,(y-\hat y) & y\ge \hat y\\(\tau-1)(y-\hat y)& y<\hat y\end{cases}$$
Fitting $\tau\in\{0.05,0.5,0.95\}$ yields a median plus a 90% interval — possibly
asymmetric, for free. Caveats: separately-fit quantiles can **cross** (lower > upper)
and recursive quantile forecasts **widen quickly** as growth uncertainty accumulates.
Project 3 fixes coverage properly with **conformal prediction**.

---

## Results (India real GDP, quarterly NSA, h = 4, rolling origin)

| model | MASE | note |
|-------|------|------|
| SARIMA(1,1,1)(0,1,0)₄ | **0.612** | classical still wins on one clean series |
| Holt-Winters | 0.640 | |
| GBM growth (direct) | 0.669 | best ML; no compounding |
| GBM growth (recursive) | 0.819 | compounding hurts |
| GBM **level** (recursive) | **1.74** | the extrapolation trap — worse than naive |
| seasonal_naive(4) | 1.263 | baseline |

Tuning: `num_leaves` ∈ {3,5,7,15,31} → best **3** (MASE 0.744). Less is more on
small data.
