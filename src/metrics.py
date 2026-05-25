"""Forecast-evaluation metrics for daily log-realized-volatility models.

All functions take 1-D NumPy arrays / pandas Series of the *log*
realized-volatility (the target produced by :func:`src.rv_estimator.compute_daily_rv`,
i.e. ``log_rv = 0.5 log RV``).

* :func:`mse` — mean squared error on the log scale (Bucci 2020, Table 4).
* :func:`qlike` — Patton (2011) QLIKE on the *variance* scale.
  Internally we convert ``log_rv -> σ² = exp(2 · log_rv) = RV`` because
  QLIKE is defined on the variance, not on its log.
* :func:`diebold_mariano` — DM test of equal predictive accuracy with the
  Harvey – Leybourne – Newbold (1997) small-sample correction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats as _sps


ArrayLike = Sequence[float] | np.ndarray


def _to_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Point-error metrics
# ---------------------------------------------------------------------------

def mse(actual: ArrayLike, forecast: ArrayLike) -> float:
    a = _to_array(actual)
    f = _to_array(forecast)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {f.shape}")
    return float(np.mean((a - f) ** 2))


def qlike(actual_log_rv: ArrayLike, forecast_log_rv: ArrayLike) -> float:
    """QLIKE loss on the variance scale (Patton 2011).

    For variance σ² with forecast σ̂², the loss is
    ``L(σ², σ̂²) = σ²/σ̂² - log(σ²/σ̂²) - 1`` which is non-negative and
    zero iff σ² = σ̂². Inputs are *log* realized volatility, so we
    exponentiate to get the variances first.
    """
    a = _to_array(actual_log_rv)
    f = _to_array(forecast_log_rv)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {f.shape}")
    sigma2_a = np.exp(2.0 * a)
    sigma2_f = np.exp(2.0 * f)
    ratio = sigma2_a / sigma2_f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


# ---------------------------------------------------------------------------
# Diebold – Mariano with Harvey – Leybourne – Newbold small-sample fix
# ---------------------------------------------------------------------------

def diebold_mariano(
    errors_model: ArrayLike,
    errors_benchmark: ArrayLike,
    h: int = 1,
    loss: str = "squared",
) -> dict:
    """Diebold – Mariano test with HLN small-sample correction.

    Null hypothesis: the two competing forecasts have equal expected loss.
    Two-sided. A negative statistic with a small p-value says the *model*
    has a significantly *smaller* loss than the benchmark.

    Parameters
    ----------
    errors_model, errors_benchmark : array-like
        Out-of-sample forecast errors ``actual - forecast`` for the two
        competing models, aligned 1-to-1.
    h : int, default 1
        Forecast horizon. Long-run variance is estimated with a
        Newey-West kernel using ``h-1`` lags (no lags for h=1).
    loss : {"squared", "absolute"}, default "squared"
        Loss function applied to the errors before differencing.

    Returns
    -------
    dict
        ``{"stat": HLN-adjusted t-statistic,
           "pvalue": two-sided p-value from a t_{T-1} distribution,
           "dm_stat": uncorrected DM-statistic,
           "mean_loss_diff": E[L_model - L_benchmark]}``.
    """
    em = _to_array(errors_model)
    eb = _to_array(errors_benchmark)
    if em.shape != eb.shape:
        raise ValueError(f"shape mismatch: {em.shape} vs {eb.shape}")
    if loss == "squared":
        d = em ** 2 - eb ** 2
    elif loss == "absolute":
        d = np.abs(em) - np.abs(eb)
    else:
        raise ValueError(f"unknown loss '{loss}'")

    T = len(d)
    d_bar = float(d.mean())
    var0 = float(np.mean((d - d_bar) ** 2))
    long_run_var = var0
    for lag in range(1, h):
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        long_run_var += 2.0 * cov
    long_run_var = max(long_run_var, 1e-12)

    dm_stat = d_bar / np.sqrt(long_run_var / T)
    hln_factor = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    hln_stat = dm_stat * hln_factor
    pvalue = float(2.0 * (1.0 - _sps.t.cdf(abs(hln_stat), df=T - 1)))
    return {
        "stat": float(hln_stat),
        "pvalue": pvalue,
        "dm_stat": float(dm_stat),
        "mean_loss_diff": d_bar,
    }


# Stage-6 alias: the project spec imports DM as `dm_test`.
dm_test = diebold_mariano


__all__ = ["mse", "qlike", "diebold_mariano", "dm_test"]
