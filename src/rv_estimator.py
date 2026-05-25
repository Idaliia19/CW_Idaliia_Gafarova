"""Daily realized-volatility estimator from 1-min intraday OHLCV.

Methodology follows Andersen-Bollerslev-Diebold-Ebens (2001) and the
sampling-frequency recommendation of Liu-Patton-Sheppard (2015):

  1.  Resample 1-min close prices to 5-min frequency (last close in each
      5-min bucket).
  2.  Log-returns on the 5-min grid.
  3.  Drop the overnight return that crosses the session boundary —
      realized volatility is an intraday quantity.
  4.  RV_t = sum of squared 5-min log-returns inside day t.
  5.  Target = log(sqrt(RV_t)) = 0.5 * log(RV_t), which is approximately
      Gaussian (Andersen et al., Bucci 2020 eq. 1).

The 5-min frequency trades off the variance reduction of finer sampling
against microstructure noise (bid-ask bounce, discrete prices) that
biases 1-min RV upwards.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RESAMPLE_RULE = "5min"
_HALF_DAY_THRESHOLD = 50  # returns/day below this are flagged as half-days
_LOG_RV_PLAUSIBLE = (-8.0, -2.0)  # broad sanity range for daily equity log-RV


def compute_5min_returns(intraday_df: pd.DataFrame) -> pd.Series:
    """Down-sample 1-min closes to 5-min and return intraday log-returns.

    The first 5-min observation of each session is dropped from the
    return series so that no overnight (close-to-open) move contaminates
    RV. Within a session, returns are `log(C_i) - log(C_{i-1})` on the
    5-min grid.

    Parameters
    ----------
    intraday_df : pd.DataFrame
        Tz-aware DatetimeIndex of 1-min bars and a 'close' column.

    Returns
    -------
    pd.Series
        5-min intraday log-returns, named ``ret_5min``, indexed by the
        5-min bucket label.
    """
    if "close" not in intraday_df.columns:
        raise ValueError("intraday_df must contain a 'close' column")
    if not isinstance(intraday_df.index, pd.DatetimeIndex):
        raise ValueError("intraday_df must be indexed by a DatetimeIndex")

    # Last close in each 5-min bucket. Default closed='left', label='left'
    # → bucket [09:30, 09:35) is labelled 09:30 and contains bars 09:30..09:34.
    close_5m = (
        intraday_df["close"]
        .resample(_RESAMPLE_RULE)
        .last()
        .dropna()
    )

    log_ret = np.log(close_5m.astype("float64")).diff()

    # Drop overnight return: keep only rows whose previous timestamp lies
    # on the same calendar date.
    date_idx = pd.Series(log_ret.index.normalize(), index=log_ret.index)
    same_session = date_idx.eq(date_idx.shift(1))
    log_ret = log_ret[same_session]

    log_ret.name = "ret_5min"
    return log_ret


def compute_daily_rv(
    intraday_df: pd.DataFrame,
    half_day_threshold: int = _HALF_DAY_THRESHOLD,
) -> pd.DataFrame:
    """Aggregate 5-min returns into daily realized volatility.

    Parameters
    ----------
    intraday_df : pd.DataFrame
        Tz-aware 1-min OHLCV as returned by :func:`load_intraday` and
        cleaned to regular trading hours.
    half_day_threshold : int, default 50
        Days with fewer than this many 5-min returns are flagged in
        ``is_half_day`` (full sessions yield ~77 returns; half-days
        yield ~41).

    Returns
    -------
    pd.DataFrame
        Indexed by tz-naive calendar date with columns:
        ``rv``                    — sum of squared 5-min log-returns,
        ``log_rv``                — ``0.5 * log(rv)``,
        ``n_5min_returns_used``   — count of 5-min returns per day,
        ``is_half_day``           — bool, ``n_5min_returns_used`` below threshold.
    """
    r5 = compute_5min_returns(intraday_df)

    # Group by the calendar date (tz-naive) so the daily index is clean.
    date_key = pd.Index(r5.index.date, name="date")
    grouped = r5.groupby(date_key)

    rv = grouped.apply(lambda x: float((x ** 2).sum()))
    n_returns = grouped.size().astype("int32")

    log_rv = 0.5 * np.log(rv.replace(0.0, np.nan))

    out = pd.DataFrame(
        {
            "rv": rv,
            "log_rv": log_rv,
            "n_5min_returns_used": n_returns,
            "is_half_day": n_returns < half_day_threshold,
        }
    )
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = "date"

    _validate_log_rv(out["log_rv"])
    logger.info(
        "computed daily RV: %d days, n_5min mean=%.1f, half-days=%d, log_rv mean=%.2f",
        len(out),
        out["n_5min_returns_used"].mean(),
        int(out["is_half_day"].sum()),
        out["log_rv"].mean(),
    )
    return out


def _validate_log_rv(log_rv: pd.Series) -> None:
    """Warn if a non-trivial share of log_rv falls outside the plausible band."""
    lo, hi = _LOG_RV_PLAUSIBLE
    out_of_band = ((log_rv < lo) | (log_rv > hi)).sum()
    if out_of_band:
        share = out_of_band / log_rv.notna().sum()
        if share > 0.01:
            logger.warning(
                "%d days (%.1f%%) have log_rv outside [%.1f, %.1f] — investigate",
                out_of_band,
                share * 100,
                lo,
                hi,
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "data" / "processed" / "AAPL_intraday_clean.parquet"
    if sample_path.exists():
        df = pd.read_parquet(sample_path)
        daily = compute_daily_rv(df)
        print("\nFirst 5 days:")
        print(daily.head().to_string())
        print("\nLast 5 days:")
        print(daily.tail().to_string())
        print("\nlog_rv describe:")
        print(daily["log_rv"].describe().to_string())
        print("\nn_5min_returns_used value counts (top):")
        print(daily["n_5min_returns_used"].value_counts().sort_index(ascending=False).head().to_string())
