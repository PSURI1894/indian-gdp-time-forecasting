# Project 1 — Deep Dive (the math behind the notebooks)

A compact reference for *why* each technique works. Read alongside the notebooks.

---

## 1. Why log, why growth rates

GDP grows roughly multiplicatively: $GDP_t \approx GDP_{t-1}(1+g)$. Taking logs,

$$\log GDP_t \approx \log GDP_0 + t\,\log(1+g),$$

a straight line with slope $\log(1+g)\approx g$ for small $g$. Two payoffs:

- **Variance stabilisation.** A constant *percentage* wiggle becomes a constant
  *absolute* wiggle in log space, so the series is homoskedastic enough to model.
- **Growth = first difference of log.**
  $\Delta \log GDP_t = \log GDP_t - \log GDP_{t-1} \approx \dfrac{GDP_t - GDP_{t-1}}{GDP_{t-1}}$,
  i.e. the period-over-period growth rate. Differencing the log both detrends and
  gives an interpretable quantity.

## 2. Decomposition

A series is viewed as **trend $T_t$ + seasonal $S_t$ + remainder $R_t$**:

- **Additive** $y_t = T_t + S_t + R_t$ — seasonal swing constant in size.
- **Multiplicative** $y_t = T_t \cdot S_t \cdot R_t$ — swing grows with level.
  Taking logs converts multiplicative → additive, which is why we decompose
  `log(GDP)`.

**STL** (Seasonal-Trend via Loess) estimates $T_t$ and $S_t$ with local regression,
allowing the seasonal shape to evolve and resisting outliers (robust weighting).

**Seasonal strength** (Hyndman):
$$F_S = \max\!\Big(0,\; 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(S_t + R_t)}\Big).$$
Near 1 ⇒ seasonality dominates the remainder ⇒ a seasonal model is essential.
*(India GDP: $F_S \approx 0.74$.)*

## 3. Stationarity & differencing

A series is **(weakly) stationary** if its mean, variance, and autocovariances
don't depend on time. ARIMA models the stationary version, so we difference until
stationary. Two tests with **opposite nulls** (use them together):

| test | $H_0$ | small p (<0.05) ⇒ |
|------|-------|-------------------|
| ADF (Augmented Dickey-Fuller) | unit root (non-stationary) | stationary |
| KPSS | stationary | non-stationary |

- **Regular differencing** $\nabla y_t = y_t - y_{t-1}$ removes a trend (order $d$).
- **Seasonal differencing** $\nabla_m y_t = y_t - y_{t-m}$ removes seasonality
  (order $D$, here $m=4$).

*India GDP diagnosis:* $d=1,\ D=1,\ m=4$.

## 4. Reading ACF / PACF (order selection cheat-sheet)

Computed on the **stationary** (differenced) series:

| pattern | suggests |
|---------|----------|
| ACF cuts off after lag $q$, PACF tails off | **MA(q)** |
| PACF cuts off after lag $p$, ACF tails off | **AR(p)** |
| both tail off | mixed **ARMA(p,q)** |
| spikes at lags $m,2m,\dots$ | seasonal **AR/MA** terms |

In practice: use these for a short-list, then let **AIC** decide.

## 5. Exponential smoothing (ETS) recursions

- **SES** (level only): $\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}$; forecast flat
  $\hat y_{t+h}=\ell_t$.
- **Holt** (+ trend):
  $\ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1}+b_{t-1})$,
  $b_t = \beta(\ell_t-\ell_{t-1}) + (1-\beta)b_{t-1}$,
  forecast $\hat y_{t+h} = \ell_t + h\,b_t$.
- **Holt-Winters** (+ seasonal, additive):
  $\hat y_{t+h} = \ell_t + h\,b_t + s_{t+h-m}$.
- **Damped trend**: replace $h\,b_t$ with $(\phi+\phi^2+\dots+\phi^h)b_t$, $0<\phi<1$,
  so the trend flattens out at long horizons (guards against over-extrapolation).

The smoothing parameters $\alpha,\beta,\gamma,\phi$ are fit by maximum likelihood.

## 6. ARIMA / SARIMA

With backshift operator $B\,y_t = y_{t-1}$, a non-seasonal ARIMA(p,d,q):

$$\phi_p(B)\,(1-B)^d\,y_t = \theta_q(B)\,\varepsilon_t,$$

where $\phi_p,\theta_q$ are the AR and MA polynomials and $\varepsilon_t$ is white
noise. **SARIMA** $(p,d,q)(P,D,Q)_m$ multiplies in seasonal polynomials in $B^m$
and a seasonal difference $(1-B^m)^D$:

$$\phi_p(B)\,\Phi_P(B^m)\,(1-B)^d(1-B^m)^D\,y_t
   = \theta_q(B)\,\Theta_Q(B^m)\,\varepsilon_t.$$

*Winner here:* $(1,1,1)(0,1,0)_4$ — seasonal differencing $(1-B^4)$ alone soaks up
the seasonality, so no seasonal AR/MA is needed (parsimony on a short series).

**Residual diagnostics.** If the model is adequate, residuals are white noise.
**Ljung-Box** tests $H_0$: "no autocorrelation up to lag $k$"; we want a *high*
p-value. Also check the residual histogram / Q-Q for approximate normality.

## 7. Metrics — MASE

$$\text{MASE} = \frac{\frac1H\sum_{h}|y_h-\hat y_h|}
{\frac{1}{n-m}\sum_{t=m+1}^{n}|y_t - y_{t-m}|}.$$

The denominator is the **in-sample seasonal-naive MAE**. Hence:

- **MASE < 1** ⇒ better than the seasonal-naive benchmark,
- **MASE > 1** ⇒ worse — the model isn't earning its complexity.

Scale-free (compare across series), defined even when $y=0$ (unlike MAPE), and
symmetric (unlike MAPE, which over-penalises over-forecasts).

## 8. Rolling-origin backtesting

Single train/test splits are high-variance and can leak the future. Instead, for
cutoffs $t_0 < t_1 < \dots$, train on $[\,0,\,t_i\,]$, forecast $[\,t_i+1,\,t_i+h\,]$,
and average errors over all folds (expanding or sliding window). This estimates
*genuine* out-of-sample skill and is the basis of every model comparison here.

---

## Empirical results (India real GDP, quarterly NSA, h = 4)

| model | MASE | note |
|-------|------|------|
| **SARIMA(1,1,1)(0,1,0)₄** | **0.612** | AIC-selected *and* backtest winner |
| Holt-Winters (damped) | 0.639 | |
| Holt-Winters | 0.640 | |
| SARIMA(1,1,1)(1,1,1)₄ | 0.693 | over-parameterised for the sample |
| drift | 1.014 | baseline |
| seasonal_naive(4) | 1.263 | baseline (the MASE denominator) |

Final 8-quarter forecast implies **~6.6% YoY growth**, matching the historical mean
— a clean sanity check.
