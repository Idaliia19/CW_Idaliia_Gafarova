"""Identification of high-volatility ("stress") sub-periods.

Two complementary identification methods:

1.  :func:`identify_stress_periods_quantile` — data-driven. A day is in
    stress if its log-RV is above a rolling 90th-percentile threshold and
    the run of such days is at least ``min_duration`` long. This catches
    every period the *data* says is unusually volatile, including ones
    we did not anticipate.

2.  :func:`identify_stress_periods_explicit` — narrative. A short list of
    macro events with dates set by hand (Russia/Ukraine, SVB / Credit
    Suisse, the August 2024 yen-carry / VIX shock). This lets us speak
    about specific episodes in the writeup.

Both return iterables of date intervals usable with
``df.loc[start:end]`` style slicing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressPeriod:
    start: pd.Timestamp
    end: pd.Timestamp
    name: str

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


def identify_stress_periods_quantile(
    rv_series: pd.Series,
    threshold: float = 0.90,
    min_duration: int = 10,
    window: int = 60,
    label_prefix: str = "Q",
) -> list[StressPeriod]:
    """Stress = consecutive ``min_duration`` days above the rolling quantile.

    Parameters
    ----------
    rv_series : pd.Series
        Daily ``log_rv`` (or RV) indexed by date.
    threshold : float, default 0.90
        Quantile used as the threshold (e.g. 0.90 = 90th percentile).
    min_duration : int, default 10
        Minimum run length of "above threshold" days to count as stress.
    window : int, default 60
        Rolling window for the quantile, in trading days.
    label_prefix : str, default "Q"
        Prefix for auto-generated period names ``Qn``.

    Returns
    -------
    list[StressPeriod]
        One entry per detected episode, in chronological order.
    """
    if not isinstance(rv_series.index, pd.DatetimeIndex):
        raise ValueError("rv_series must be indexed by a DatetimeIndex")
    s = rv_series.dropna()
    rolling_q = s.rolling(window=window, min_periods=max(20, window // 3)).quantile(threshold)
    above = (s > rolling_q).astype(int).fillna(0)

    periods: list[StressPeriod] = []
    in_run = False
    run_start_idx = -1
    for i, val in enumerate(above.values):
        if val == 1 and not in_run:
            in_run = True
            run_start_idx = i
        elif val == 0 and in_run:
            run_end_idx = i - 1
            if run_end_idx - run_start_idx + 1 >= min_duration:
                periods.append(StressPeriod(
                    start=s.index[run_start_idx],
                    end=s.index[run_end_idx],
                    name=f"{label_prefix}{len(periods) + 1}",
                ))
            in_run = False
    if in_run:
        run_end_idx = len(above) - 1
        if run_end_idx - run_start_idx + 1 >= min_duration:
            periods.append(StressPeriod(
                start=s.index[run_start_idx],
                end=s.index[run_end_idx],
                name=f"{label_prefix}{len(periods) + 1}",
            ))
    return periods


def identify_stress_periods_explicit() -> list[StressPeriod]:
    """Hand-picked macro-event windows inside the 2022-04 → 2024-12 test region."""
    return [
        StressPeriod(
            start=pd.Timestamp("2022-02-15"),
            end=pd.Timestamp("2022-04-30"),
            name="Russia/Ukraine + Fed tightening",
        ),
        StressPeriod(
            start=pd.Timestamp("2023-03-08"),
            end=pd.Timestamp("2023-05-15"),
            name="SVB + Credit Suisse banking crisis",
        ),
        StressPeriod(
            start=pd.Timestamp("2024-08-01"),
            end=pd.Timestamp("2024-08-31"),
            name="Yen carry unwind + VIX spike",
        ),
    ]


def overlap_index(
    period: StressPeriod, idx: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Return the subset of ``idx`` that falls inside ``period``."""
    mask = (idx >= period.start) & (idx <= period.end)
    return idx[mask]


__all__ = [
    "StressPeriod",
    "identify_stress_periods_quantile",
    "identify_stress_periods_explicit",
    "overlap_index",
]
