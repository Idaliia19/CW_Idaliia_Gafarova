"""Model Confidence Set (Hansen, Lunde, Nason 2011) via ``arch.bootstrap.MCS``.

The MCS controls the family-wise error rate when selecting from a collection
of competing forecasts: with confidence level ``1-α`` it returns the set
``M̂_{1-α}`` that contains the best model with probability ≥ ``1-α``. Models
outside the set are statistically inferior. The bootstrap p-values are
attached to each excluded model in the order in which it was eliminated —
``p_value_i`` is the probability of observing the equivalence test
statistic as large as the one used to drop model ``i``.

We use the stationary-bootstrap variant (Politis & Romano 1994) with
mean block length 12 — Bucci's choice — and the range statistic ("R").
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from arch.bootstrap import MCS


def compute_mcs(
    losses: pd.DataFrame,
    size: float = 0.10,
    reps: int = 10_000,
    block_size: int = 12,
    method: str = "R",
    bootstrap: str = "stationary",
    seed: int = 42,
) -> dict:
    """Run the Model Confidence Set procedure.

    Parameters
    ----------
    losses : pd.DataFrame
        Per-period loss matrix, rows = time, columns = model name.
        For QLIKE/MSE these are non-negative; the MCS works with any
        bounded loss differential, signs do not need to be matched.
    size : float, default 0.10
        Confidence level α. The returned SSM contains the best model
        with probability ``1 - size``. ``size = 0.25`` recovers the
        looser SSM-75.
    reps : int, default 10_000
        Number of bootstrap replications.
    block_size : int, default 12
        Mean block length for the stationary bootstrap (Bucci's choice).
    method : {"R", "max"}, default "R"
        Equivalence statistic. "R" is the range statistic — the default
        in ``arch`` and the more conservative of the two.
    bootstrap : str, default "stationary"
        Bootstrap scheme; "stationary" matches Hansen et al. 2011.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``included`` — list of models in the SSM (lower-is-better losses).
        ``excluded`` — list of models eliminated.
        ``pvalues`` — dict ``{model: p_value}``. For included models the
        elimination p-value is reported by ``arch`` (= 1 if never tested).
        ``ranking`` — models sorted by p-value, lowest p-value first
        (= weakest evidence the model is best).
        ``size`` — the input size used.
    """
    losses_clean = losses.dropna(how="any")
    if losses_clean.shape[0] != losses.shape[0]:
        dropped = losses.shape[0] - losses_clean.shape[0]
        if dropped > 0:
            import warnings as _w
            _w.warn(f"MCS: dropped {dropped} rows with NaN losses before running")

    mcs = MCS(
        losses=losses_clean.values,
        size=size,
        reps=reps,
        block_size=block_size,
        method=method,
        bootstrap=bootstrap,
        seed=seed,
    )
    mcs.compute()

    cols = list(losses_clean.columns)
    # arch.bootstrap.MCS.pvalues is a DataFrame indexed by *original column index*
    # in elimination order (worst eliminated first, last survivor with p = 1.0).
    # The bug previously here was `ravel() + cols[i]`, which mapped elimination-order
    # p-values onto original-order column names — scrambling the mapping. Fix: read
    # the DataFrame's index (integer column position) and map back through ``cols``.
    pvalues_df = mcs.pvalues
    pvalues = {cols[int(idx)]: float(p) for idx, p in pvalues_df.iloc[:, 0].items()}
    included_idx = np.asarray(mcs.included).ravel().astype(int)
    excluded_idx = np.asarray(mcs.excluded).ravel().astype(int)
    included = [cols[i] for i in included_idx]
    excluded = [cols[i] for i in excluded_idx]
    ranking = sorted(pvalues.items(), key=lambda kv: kv[1])

    return {
        "included": included,
        "excluded": excluded,
        "pvalues": pvalues,
        "ranking": ranking,
        "size": size,
    }


def per_period_losses(
    actuals: np.ndarray,
    forecasts: dict[str, np.ndarray],
    loss: str = "mse",
) -> pd.DataFrame:
    """Build the per-period loss matrix from aligned forecast series.

    Parameters
    ----------
    actuals : 1-D array
        Realised target series (``log_rv``).
    forecasts : dict[str, 1-D array]
        Aligned forecasts keyed by model name.
    loss : {"mse", "qlike", "ae"}, default "mse"
        Pointwise loss function. ``mse`` is ``(y - ŷ)²``; ``qlike`` is the
        Patton (2011) variance-scale loss
        ``σ²/σ̂² - log(σ²/σ̂²) - 1`` with ``σ² = exp(2 · log_rv)``;
        ``ae`` is ``|y - ŷ|``.
    """
    a = np.asarray(actuals, dtype=float).ravel()
    cols = {}
    for name, f in forecasts.items():
        f = np.asarray(f, dtype=float).ravel()
        if f.shape != a.shape:
            raise ValueError(f"shape mismatch for {name}: {f.shape} vs {a.shape}")
        if loss == "mse":
            cols[name] = (a - f) ** 2
        elif loss == "qlike":
            sigma2_a = np.exp(2.0 * a)
            sigma2_f = np.exp(2.0 * f)
            ratio = sigma2_a / sigma2_f
            cols[name] = ratio - np.log(ratio) - 1.0
        elif loss == "ae":
            cols[name] = np.abs(a - f)
        else:
            raise ValueError(f"unknown loss '{loss}'")
    return pd.DataFrame(cols)


__all__ = ["compute_mcs", "per_period_losses"]
