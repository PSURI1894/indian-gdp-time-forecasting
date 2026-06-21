# Project 3 — Deep Dive (the math of uncertainty)

Why each interval method works, and what its guarantee actually buys you.

---

## 1. Scoring a probabilistic forecast

Three desiderata, three numbers.

**Coverage** (calibration). For a nominal $1-\alpha$ interval $[\ell_t,u_t]$,
$$\text{cov} = \frac1N\sum_t \mathbf 1[\ell_t \le y_t \le u_t]\ \approx\ 1-\alpha.$$
Measured **out-of-sample** — in-sample coverage is meaningless.

**Sharpness.** Mean width $\frac1N\sum (u_t-\ell_t)$. Tighter is better *given*
calibration.

**Winkler / interval score** (proper, lower = better):
$$W_\alpha = (u-\ell) + \tfrac{2}{\alpha}(\ell-y)\mathbf 1[y<\ell] + \tfrac{2}{\alpha}(y-u)\mathbf 1[y>u].$$
It is the width **plus** a miss penalty scaled by $2/\alpha$. You cannot game it:
too-narrow → frequent penalised misses; too-wide → large width term. Minimised by
*honest* intervals — that's what "proper" means.

## 2. Quantile regression

To estimate the conditional $\tau$-quantile $q_\tau(x)$, minimise the **pinball
loss**
$$L_\tau(y,\hat q)=\max\!\big(\tau(y-\hat q),\ (\tau-1)(y-\hat q)\big).$$
At the minimiser, $\Pr(y\le \hat q)=\tau$ — exactly the definition of a quantile.
$\tau=0.5$ is symmetric (median, robust to outliers, unlike the mean's squared loss).
LightGBM optimises this directly via `objective="quantile", alpha=τ`.

**Quantile crossing.** Quantiles fit independently aren't guaranteed monotone in
$\tau$, so a 90% prediction can fall *below* the 50% one. Cheap fix: **sort the
predicted quantiles** at each point (a valid rearrangement that never worsens pinball
loss). Calibration is then checked with a **reliability diagram**: plot nominal vs
empirical coverage; the diagonal is perfect, *just above* is safely conservative.

## 3. Conformal prediction

Turn any point predictor $\hat f$ into intervals with **finite-sample, distribution-
free** coverage. Split conformal:

1. Fit $\hat f$ on a proper-training set.
2. On a disjoint **calibration** set, compute conformity scores
   $s_i = |y_i - \hat f(x_i)|$ (here, in log space → relative residuals).
3. Let $\hat q$ be the $\lceil (n+1)(1-\alpha)\rceil/n$ empirical quantile of $\{s_i\}$.
   The interval $\hat f(x)\pm \hat q$ satisfies
   $$\Pr\big(y_{n+1}\in \hat f(x_{n+1})\pm\hat q\big)\ \ge\ 1-\alpha.$$

The guarantee needs only **exchangeability** of the scores — no Gaussianity, no model
correctness. We calibrate **per horizon** (h-step residuals differ) and exponentiate
the log-space band → a **multiplicative** (asymmetric in levels) interval.

## 4. Why plain conformal struggles on time series — and ACI

Exchangeability **fails** for time series: residuals are autocorrelated and their
distribution **drifts** (volatility regimes, structural breaks). A fixed conformal
band calibrated on the past can lose coverage after a shift.

**Adaptive Conformal Inference (ACI)** restores it online. Keep a working level
$\alpha_t$ and, after observing whether interval $t$ missed ($\text{err}_t\in\{0,1\}$),
update
$$\alpha_{t+1}=\alpha_t+\gamma\,(\alpha-\text{err}_t).$$
A miss ($\text{err}_t=1$) **lowers** $\alpha_t$ → a **wider** next interval; a run of
hits raises it → tighter. One can show the long-run empirical miss rate $\to\alpha$
for any $\gamma>0$, **regardless of distribution shift** — coverage is recovered by
control, not by assumption. $\gamma$ trades adaptation speed vs stability.

---

## Results (India real GDP, quarterly NSA, 90% intervals, rolling origin)

| method | coverage | mean width | Winkler (↓) | note |
|--------|----------|-----------|-------------|------|
| SARIMA-Gaussian | ~0.89 | narrow | **best** | model-based; calibrated & sharp |
| Quantile-GBM | ~0.82 | wide | worst | learned tails need more data |
| Conformal (split) | ≥0.90 | medium | competitive | distribution-free guarantee |
| ACI (1-step) | ≈0.90 | adaptive | — | holds coverage under drift |

Takeaway: report an interval, **prove its coverage on a backtest**, and prefer
conformal/ACI whenever you can't trust a distributional assumption.
