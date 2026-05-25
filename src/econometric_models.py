"""Econometric one-step-ahead forecasters for daily log-realized-volatility.

Each class implements the same interface so the rolling-window engine can
treat them uniformly::

    model.fit(y_train)             # estimate parameters
    y_hat = model.forecast(y_hist) # one-step-ahead prediction

``y_train`` and ``y_hist`` are 1-D arrays/Series of *log* realized
volatility (i.e. ``0.5 * log(RV)``).

Implemented models
------------------
* :class:`RandomWalkForecaster` — naive benchmark, ``ŷ_{t+1} = y_t``.
* :class:`ARForecaster` — AR(p) with lag order selected by BIC on the
  training window (``max_p=22`` so it never reaches further back than HAR).
* :class:`HARForecaster` — Corsi (2009): regression on daily / weekly /
  monthly averages of past log-RV.
* :class:`ARFIMAForecaster` — ARFIMA(0, d, 1). ``d`` is estimated via the
  Geweke – Porter-Hudak (GPH) periodogram regression; the fractionally
  differenced series is then fit with an MA(1) and the one-step forecast
  is inverted back to the original scale.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA


ArrayLike = Sequence[float] | np.ndarray | pd.Series


# ---------------------------------------------------------------------------
# Random walk
# ---------------------------------------------------------------------------

class RandomWalkForecaster:
    """Naive benchmark: predict the last observed value."""

    def fit(self, y_train: ArrayLike) -> "RandomWalkForecaster":  # noqa: D401
        return self

    def forecast(self, y_history: ArrayLike) -> float:
        return float(np.asarray(y_history)[-1])

    @property
    def params(self) -> dict:
        return {"model": "RW"}


# ---------------------------------------------------------------------------
# AR(p) with BIC lag selection
# ---------------------------------------------------------------------------

class ARForecaster:
    """AR(p) with lag order picked by BIC up to ``max_p``."""

    def __init__(self, max_p: int = 22, ic: str = "bic") -> None:
        self.max_p = max_p
        self.ic = ic
        self.lags_: list[int] = []
        self.p_: int = 0
        self.const_: float = 0.0
        self.phis_: np.ndarray = np.array([])

    def fit(self, y_train: ArrayLike) -> "ARForecaster":
        y = np.asarray(y_train, dtype=float)
        sel = ar_select_order(y, maxlag=self.max_p, ic=self.ic, glob=False, trend="c")
        lags = sel.ar_lags if sel.ar_lags else [1]
        # ar_select_order with glob=False returns consecutive lags 1..p
        self.lags_ = list(lags)
        self.p_ = max(self.lags_)
        fitted = AutoReg(y, lags=self.lags_, trend="c", old_names=False).fit()
        params = np.asarray(fitted.params)
        # AutoReg with trend='c': params = [const, phi_lag1, phi_lag2, ...]
        self.const_ = float(params[0])
        self.phis_ = params[1:].astype(float)
        return self

    def forecast(self, y_history: ArrayLike) -> float:
        y = np.asarray(y_history, dtype=float)
        pred = self.const_
        for phi, lag in zip(self.phis_, self.lags_):
            pred += phi * y[-lag]
        return float(pred)

    @property
    def params(self) -> dict:
        return {"model": f"AR({self.p_})", "const": self.const_, "phis": self.phis_.tolist()}


# ---------------------------------------------------------------------------
# HAR (Corsi 2009)
# ---------------------------------------------------------------------------

class HARForecaster:
    """HAR-RV (Corsi 2009): regression on daily/weekly/monthly past averages."""

    DAILY = 1
    WEEKLY = 5
    MONTHLY = 22

    def __init__(self) -> None:
        self.coefs_: np.ndarray = np.array([])  # [const, beta_d, beta_w, beta_m]

    def fit(self, y_train: ArrayLike) -> "HARForecaster":
        y = np.asarray(y_train, dtype=float)
        T = len(y)
        if T <= self.MONTHLY + 1:
            raise ValueError(f"HAR needs > {self.MONTHLY + 1} observations, got {T}")
        ser = pd.Series(y)
        rv_d = ser.shift(0).values
        rv_w = ser.rolling(self.WEEKLY).mean().values
        rv_m = ser.rolling(self.MONTHLY).mean().values
        # Use features at time t to predict y_{t+1}.
        # Valid feature index: t = MONTHLY-1 .. T-2. Target index: t+1.
        first = self.MONTHLY - 1
        X = np.column_stack([
            np.ones(T - 1 - first),
            rv_d[first:T - 1],
            rv_w[first:T - 1],
            rv_m[first:T - 1],
        ])
        Y = y[first + 1:T]
        self.coefs_, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return self

    def forecast(self, y_history: ArrayLike) -> float:
        y = np.asarray(y_history, dtype=float)
        rv_d = y[-self.DAILY]
        rv_w = float(np.mean(y[-self.WEEKLY:]))
        rv_m = float(np.mean(y[-self.MONTHLY:]))
        c, bd, bw, bm = self.coefs_
        return float(c + bd * rv_d + bw * rv_w + bm * rv_m)

    @property
    def params(self) -> dict:
        c, bd, bw, bm = self.coefs_
        return {
            "model": "HAR",
            "const": float(c),
            "beta_daily": float(bd),
            "beta_weekly": float(bw),
            "beta_monthly": float(bm),
        }


# ---------------------------------------------------------------------------
# ARFIMA(p, d, q) with GPH-estimated d
# ---------------------------------------------------------------------------

def _gph_d(y: np.ndarray, bandwidth_power: float = 0.5) -> float:
    """Geweke – Porter-Hudak estimator of the fractional integration order.

    Regresses ``log I(λ_j)`` on ``log[4 sin²(λ_j/2)]`` for the lowest
    ``m = T^bandwidth_power`` Fourier frequencies; ``d = -slope``.
    Clipped to ``[0.01, 0.49]`` to keep the model inside the stationary
    long-memory region.
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    T = len(y)
    m = max(8, int(T ** bandwidth_power))
    Y = np.fft.fft(y)
    periodogram = (np.abs(Y[1:m + 1]) ** 2) / (2.0 * np.pi * T)
    lam = 2.0 * np.pi * np.arange(1, m + 1) / T
    x = np.log(4.0 * np.sin(lam / 2.0) ** 2)
    log_I = np.log(periodogram)
    slope, _intercept = np.polyfit(x, log_I, 1)
    d = -slope
    return float(np.clip(d, 0.01, 0.49))


def _frac_diff_weights(d: float, K: int) -> np.ndarray:
    """Coefficients π_k in the expansion (1 - L)^d = Σ π_k L^k.

    π_0 = 1, π_k = π_{k-1} (k - 1 - d) / k.
    """
    pi = np.empty(K + 1, dtype=float)
    pi[0] = 1.0
    for k in range(1, K + 1):
        pi[k] = pi[k - 1] * (k - 1 - d) / k
    return pi


class ARFIMAForecaster:
    """ARFIMA(p, d, q) with d from GPH and ARMA(p, q) on the differenced series.

    The model is::

        (1 - L)^d (y_t - μ) = u_t,    u_t ~ ARMA(p, q)

    Forecast inversion uses the relation
    ``y_{T+1} - μ = -Σ_{k=1}^K π_k (y_{T+1-k} - μ) + u_{T+1}``,
    where ``u_{T+1}`` is the ARMA one-step prediction.
    """

    def __init__(self, p: int = 0, q: int = 1, max_frac_lags: int = 200) -> None:
        self.p = p
        self.q = q
        self.max_frac_lags = max_frac_lags
        self.mu_: float = 0.0
        self.d_: float = 0.0
        self.pi_: np.ndarray = np.array([])
        self.u_forecast_: float = 0.0

    def fit(self, y_train: ArrayLike) -> "ARFIMAForecaster":
        y = np.asarray(y_train, dtype=float)
        self.mu_ = float(y.mean())
        y_dm = y - self.mu_
        self.d_ = _gph_d(y_dm)
        K = min(self.max_frac_lags, len(y_dm) - max(self.p, self.q) - 5)
        self.pi_ = _frac_diff_weights(self.d_, K)
        u = np.convolve(y_dm, self.pi_, mode="valid")  # length T - K
        if self.p == 0 and self.q == 0:
            self.u_forecast_ = float(u.mean())
        else:
            arma = ARIMA(u, order=(self.p, 0, self.q), trend="c").fit(
                method_kwargs={"warn_convergence": False}
            )
            self.u_forecast_ = float(np.asarray(arma.forecast(1))[0])
        return self

    def forecast(self, y_history: ArrayLike) -> float:
        y = np.asarray(y_history, dtype=float)
        y_dm = y - self.mu_
        K = len(self.pi_) - 1
        # y_{T+1} - μ = -Σ_{k=1..K} π_k (y_{T+1-k} - μ) + u_{T+1}
        tail = y_dm[-K:][::-1]  # y_T, y_{T-1}, ..., y_{T-K+1}
        if len(tail) < K:
            # Not enough history — pad with mean (zero, since demeaned).
            pad = np.zeros(K - len(tail))
            tail = np.concatenate([tail, pad])
        inv_lag = -float(np.dot(self.pi_[1:K + 1], tail))
        return inv_lag + self.u_forecast_ + self.mu_

    @property
    def params(self) -> dict:
        return {
            "model": f"ARFIMA({self.p}, d, {self.q})",
            "d": float(self.d_),
            "mu": float(self.mu_),
            "u_forecast": float(self.u_forecast_),
        }


# ---------------------------------------------------------------------------
# ARFIMAX — ARFIMA(0, d, q) with exogenous regressors
# ---------------------------------------------------------------------------

class ARFIMAXForecaster:
    """ARFIMA(0, d, q) plus exogenous regressors, estimated in two stages.

    Step 1 — OLS of ``y`` on ``[1, X]`` gives the exogenous coefficients β
    and a residual series that is purged of the linear macro effect.
    Step 2 — the residual is modelled by a plain :class:`ARFIMAForecaster`
    (GPH d, fractional differencing, MA(q), inversion).

    The one-step forecast adds the two pieces back:
    ``ŷ_{T+1} = β'·[1, x_{T+1}] + û_{T+1}``, where ``x_{T+1}`` is the (already
    1-day-lagged) macro vector known at time T and ``û_{T+1}`` is the residual
    ARFIMA forecast. This is the tractable two-step ARFIMAX of Bucci (2020,
    §3.2); a full state-space MLE would be more efficient but is not needed
    for one-step rolling evaluation.
    """

    def __init__(self, q: int = 1) -> None:
        self.q = q
        self.beta_: np.ndarray = np.array([])
        self.resid_model_: ARFIMAForecaster | None = None

    def fit(self, y_train: ArrayLike, X_train: np.ndarray) -> "ARFIMAXForecaster":
        y = np.asarray(y_train, dtype=float).ravel()
        X = np.asarray(X_train, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xc = np.column_stack([np.ones(len(y)), X])
        self.beta_, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        resid = y - Xc @ self.beta_
        self.resid_model_ = ARFIMAForecaster(p=0, q=self.q).fit(resid)
        self._resid_hist = resid  # cache for the forecast call
        return self

    def forecast(self, y_history: ArrayLike, X_history: np.ndarray, x_next: np.ndarray) -> float:
        y = np.asarray(y_history, dtype=float).ravel()
        X = np.asarray(X_history, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xc = np.column_stack([np.ones(len(y)), X])
        resid_hist = y - Xc @ self.beta_
        x_next = np.asarray(x_next, dtype=float).ravel()
        exog_next = float(np.concatenate([[1.0], x_next]) @ self.beta_)
        u_forecast = float(self.resid_model_.forecast(resid_hist))
        return exog_next + u_forecast

    @property
    def params(self) -> dict:
        return {
            "model": f"ARFIMAX(0, d, {self.q})",
            "beta": self.beta_.tolist(),
            "d": float(self.resid_model_.d_) if self.resid_model_ else None,
        }


__all__ = [
    "RandomWalkForecaster",
    "ARForecaster",
    "HARForecaster",
    "ARFIMAForecaster",
    "ARFIMAXForecaster",
]
