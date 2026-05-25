"""Rolling-window one-step-ahead forecasting engine.

The engine is model-agnostic: any object with ``fit(y_train)`` and
``forecast(y_history) -> float`` (see :mod:`src.econometric_models`)
plugs in directly.

Two scheduling modes are supported:

* :func:`rolling_forecast` — re-fit *every* step (Bucci's choice).
* :func:`rolling_forecast_periodic` — re-fit every ``refit_every`` steps
  but always feed the latest ``y_history`` to the forecast call. This
  keeps the model parameters fresh enough while saving cost on slow
  estimators (e.g. ARFIMA). It is a standard practical shortcut and is
  acknowledged in the literature (Liu et al. 2015 §3.1, footnote 7).
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class _ForecasterLike(Protocol):
    def fit(self, y_train): ...
    def forecast(self, y_history) -> float: ...


ModelFactory = Callable[[], _ForecasterLike]


def _check_split(T: int, train_window: int, test_size: int) -> int:
    if train_window <= 0:
        raise ValueError("train_window must be positive")
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if train_window + test_size > T:
        raise ValueError(
            f"train_window ({train_window}) + test_size ({test_size}) "
            f"exceeds series length ({T})"
        )
    return T - test_size  # index of the first observation to forecast


def rolling_forecast(
    model_factory: ModelFactory,
    y: pd.Series,
    train_window: int,
    test_size: int,
    desc: str = "",
    refit_every: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    """Rolling-window one-step-ahead forecasts.

    Parameters
    ----------
    model_factory : callable
        Zero-arg callable returning a freshly constructed forecaster.
        A new instance is fit at every (re-)estimation step.
    y : pd.Series
        Full target series indexed by date. Length ``T``.
    train_window : int
        Width of the rolling training window. Constant across steps.
    test_size : int
        Number of out-of-sample one-step forecasts to produce. The
        forecast positions are the last ``test_size`` entries of ``y``.
    desc : str, default ""
        Description shown in the tqdm progress bar.
    refit_every : int, default 1
        Re-estimate the model every ``refit_every`` steps. Between
        re-estimations the *same* fitted model is queried but with an
        updated ``y_history`` — so AR/HAR/ARFIMA still use the latest
        lag values, only the coefficients are slightly stale.
    progress : bool, default True
        Toggle the tqdm bar.

    Returns
    -------
    pd.DataFrame
        Indexed like ``y[-test_size:]`` with columns
        ``[actual, forecast, error]``  where ``error = actual - forecast``.
    """
    y_arr = np.asarray(y.values, dtype=float)
    T = len(y_arr)
    start = _check_split(T, train_window, test_size)

    actuals = np.empty(test_size, dtype=float)
    forecasts = np.empty(test_size, dtype=float)

    model: _ForecasterLike | None = None
    iterator = range(start, T)
    if progress:
        iterator = tqdm(iterator, desc=desc, leave=False, total=test_size)

    for i, t in enumerate(iterator):
        train_slice = y_arr[t - train_window: t]
        if model is None or (i % refit_every == 0):
            model = model_factory()
            model.fit(train_slice)
        forecasts[i] = float(model.forecast(train_slice))
        actuals[i] = y_arr[t]

    test_index = y.index[start:T]
    return pd.DataFrame(
        {"actual": actuals, "forecast": forecasts, "error": actuals - forecasts},
        index=test_index,
    )


def rolling_forecast_periodic(
    model_factory: ModelFactory,
    y: pd.Series,
    train_window: int,
    test_size: int,
    refit_every: int = 22,
    desc: str = "",
    progress: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper for :func:`rolling_forecast` with ``refit_every > 1``."""
    return rolling_forecast(
        model_factory=model_factory,
        y=y,
        train_window=train_window,
        test_size=test_size,
        refit_every=refit_every,
        desc=desc,
        progress=progress,
    )


def rolling_forecast_nn(
    model_factory: ModelFactory,
    y: pd.Series,
    train_window: int = 1585,
    test_size: int = 679,
    refit_every: int = 22,
    desc: str = "NN",
    progress: bool = True,
) -> pd.DataFrame:
    """Rolling-window forecast for neural-network models with periodic refit.

    Identical semantics to :func:`rolling_forecast` but with NN-friendly
    defaults: train window 1585, test 679, refit once per trading month.
    Between refits the fitted model receives the *latest* ``y_history``
    each day, so the lag values fed into the network always come from
    the most recent observations — only the weights are a month stale.
    """
    return rolling_forecast(
        model_factory=model_factory,
        y=y,
        train_window=train_window,
        test_size=test_size,
        refit_every=refit_every,
        desc=desc,
        progress=progress,
    )


def rolling_forecast_exog(
    model_factory: Callable[[], object],
    y: pd.Series,
    X: pd.DataFrame,
    train_window: int,
    test_size: int,
    refit_every: int = 1,
    desc: str = "",
    progress: bool = True,
) -> pd.DataFrame:
    """Rolling-window one-step forecasts for models with exogenous inputs.

    For ARFIMAX / NARX the forecaster exposes ``fit(y_train, X_train)`` and
    ``forecast(y_history, X_history, x_next) -> float``. ``X`` must be aligned
    to ``y`` (same DatetimeIndex) and **already 1-day-lagged** so that row
    ``X[t]`` contains only information available before ``y[t]`` is realised —
    the engine therefore passes ``X[t]`` straight through as ``x_next`` with no
    further shifting.

    Parameters
    ----------
    model_factory : callable
        Zero-arg constructor returning a fresh exog-aware forecaster.
    y : pd.Series
        Target series, length T.
    X : pd.DataFrame
        Lagged exogenous features aligned to ``y``; shape (T, k).
    train_window, test_size : int
        Rolling-window width and number of one-step forecasts.
    refit_every : int, default 1
        Re-estimate every ``refit_every`` steps. Between refits the model is
        re-queried with the latest history/exog (only weights are stale).

    Returns
    -------
    pd.DataFrame
        Indexed like ``y[-test_size:]`` with columns ``[actual, forecast, error]``.
    """
    y_arr = np.asarray(y.values, dtype=float)
    X_arr = np.asarray(X.values, dtype=float)
    T = len(y_arr)
    if len(X_arr) != T:
        raise ValueError(f"X length {len(X_arr)} != y length {T}")
    start = _check_split(T, train_window, test_size)

    actuals = np.empty(test_size, dtype=float)
    forecasts = np.empty(test_size, dtype=float)

    model = None
    iterator = range(start, T)
    if progress:
        iterator = tqdm(iterator, desc=desc, leave=False, total=test_size)

    for i, t in enumerate(iterator):
        y_tr = y_arr[t - train_window: t]
        X_tr = X_arr[t - train_window: t]
        if model is None or (i % refit_every == 0):
            model = model_factory()
            model.fit(y_tr, X_tr)
        forecasts[i] = float(model.forecast(y_tr, X_tr, X_arr[t]))
        actuals[i] = y_arr[t]

    test_index = y.index[start:T]
    return pd.DataFrame(
        {"actual": actuals, "forecast": forecasts, "error": actuals - forecasts},
        index=test_index,
    )


__all__ = [
    "rolling_forecast",
    "rolling_forecast_periodic",
    "rolling_forecast_nn",
    "rolling_forecast_exog",
]
