"""Iterative multistep-ahead forecasts for the project's unified forecaster API.

Every forecaster in :mod:`src.econometric_models` and :mod:`src.neural_models`
follows the same protocol::

    model.fit(y_train)               # estimate parameters
    y_hat = model.forecast(y_history) # one-step-ahead prediction (scalar)

To get a k-step-ahead path we feed each prediction back as the next "observed"
point and call ``forecast`` again. At step t+i the input history is
``[y_real..t, ŷ_{t+1}, ..., ŷ_{t+i-1}]`` — only the *first* point uses real
future data; everything after is the model's own extrapolation.

This is the standard iterative scheme from Bucci (2020, §4) and the dynamic
forecast literature (Marcellino, Stock, Watson 2006). The alternative, direct
multistep, would require a separate model per horizon and is not what Bucci
uses.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class _ForecasterLike(Protocol):
    def fit(self, y_train): ...
    def forecast(self, y_history) -> float: ...


def iterative_forecast(model: _ForecasterLike, y_history, k: int = 5) -> np.ndarray:
    """k iterative one-step-ahead forecasts seeded from ``y_history``.

    The model is queried ``k`` times. Each prediction is appended to the
    history before the next call, so step i uses ``i-1`` synthetic lags.

    Parameters
    ----------
    model : forecaster
        Already-fit instance with a ``forecast(y_history)`` method.
    y_history : 1-D array-like
        Latest observed values. Length must be at least the model's
        maximum lag (22 for HAR; n_lags for AR/neural; max_frac_lags for
        ARFIMA).
    k : int, default 5
        Forecast horizon.

    Returns
    -------
    np.ndarray
        Shape ``(k,)``. ``out[0]`` is the one-step-ahead forecast,
        ``out[k-1]`` the k-step-ahead.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    history = list(np.asarray(y_history, dtype=float))
    forecasts = np.empty(k, dtype=float)
    for i in range(k):
        forecasts[i] = float(model.forecast(np.asarray(history, dtype=float)))
        history.append(forecasts[i])
    return forecasts


def rolling_multistep_forecast(
    model_factory: Callable[[], _ForecasterLike],
    y: pd.Series,
    train_window: int,
    test_size: int,
    k: int = 5,
    refit_every: int = 1,
    desc: str = "",
    progress: bool = True,
) -> pd.DataFrame:
    """Rolling-window iterative k-step-ahead forecasts.

    At each step t in the test region we (re-)fit the model on
    ``y[t-train_window:t]`` and produce a path
    ``(ŷ_{t+1}, ŷ_{t+2}, …, ŷ_{t+k})`` from :func:`iterative_forecast`.
    The actuals ``y_{t+i}`` are aligned against each horizon column.

    Parameters
    ----------
    model_factory : callable
        Zero-arg constructor for a forecaster.
    y : pd.Series
        Full target series indexed by date.
    train_window : int
        Width of the rolling training window.
    test_size : int
        Number of starting positions ``t`` in the test region. Forecasts
        of horizon i are only well-defined when ``t + i <= len(y)``, so
        the last ``k-1`` rows of the returned frame are NaN for horizons
        beyond the available actuals.
    k : int, default 5
        Forecast horizon.
    refit_every : int, default 1
        Re-fit the model every ``refit_every`` starting positions. Slow
        estimators (AR with BIC search, ARFIMA, NN) can use a larger
        value; the forecast still feeds the latest ``y_history`` so
        only the coefficients are stale.

    Returns
    -------
    pd.DataFrame
        Indexed by the *origin* date ``t`` (the last observed day before
        the forecast path begins). Columns ``h1, h2, …, hk`` for the
        forecasts and ``a1, a2, …, ak`` for the corresponding actuals
        (NaN where the test region runs out).
    """
    y_arr = np.asarray(y.values, dtype=float)
    T = len(y_arr)
    if train_window + test_size > T:
        raise ValueError(
            f"train_window ({train_window}) + test_size ({test_size}) "
            f"exceeds series length ({T})"
        )
    start = T - test_size

    forecasts = np.full((test_size, k), np.nan, dtype=float)
    actuals = np.full((test_size, k), np.nan, dtype=float)

    model: _ForecasterLike | None = None
    iterator = range(test_size)
    if progress:
        iterator = tqdm(iterator, desc=desc, leave=False, total=test_size)

    for i in iterator:
        t = start + i  # origin: last observed day index in y_arr
        train_slice = y_arr[t - train_window: t]
        if model is None or (i % refit_every == 0):
            model = model_factory()
            model.fit(train_slice)
        path = iterative_forecast(model, train_slice, k=k)
        forecasts[i, :] = path
        for h in range(k):
            if t + h < T:
                actuals[i, h] = y_arr[t + h]

    origin_dates = y.index[start - 1: T - 1]  # the t-th forecast originates from day t-1's close
    out = pd.DataFrame(index=origin_dates)
    for h in range(k):
        out[f"h{h + 1}"] = forecasts[:, h]
    for h in range(k):
        out[f"a{h + 1}"] = actuals[:, h]
    out.index.name = "origin"
    return out


__all__ = ["iterative_forecast", "rolling_multistep_forecast"]
