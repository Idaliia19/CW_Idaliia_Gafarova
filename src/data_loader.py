"""Intraday OHLCV loader for the Bucci (2020) realized-volatility replication.

The raw files (AAPL.txt, AMZN.txt, JPM.txt) are 1-minute bars with columns
[Date, Time, Open, High, Low, Close, Volume] and no header. Dates use the
US format MM/DD/YYYY (verified by the presence of "12/31/2024" in tails — a
DD/MM/YYYY reading would be impossible since there is no 31st month).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
_DTYPES = {
    "Open": "float32",
    "High": "float32",
    "Low": "float32",
    "Close": "float32",
    "Volume": "int64",
}
_NY_TZ = "America/New_York"


def load_intraday(ticker: str, data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load a single ticker's 1-minute OHLCV file into a tidy DataFrame.

    Parameters
    ----------
    ticker : str
        Ticker symbol; must match the file stem (e.g. "AAPL" -> "AAPL.txt").
    data_dir : str or Path, default "data/raw"
        Directory containing the .txt files.

    Returns
    -------
    pd.DataFrame
        Indexed by tz-aware DatetimeIndex (America/New_York), sorted ascending.
        Columns: [open, high, low, close, volume] with float32 / int64 dtypes.

    Notes
    -----
    Date format is MM/DD/YYYY (US convention). The raw timestamps in the files
    are wall-clock NY times; we localize them to America/New_York to make any
    later resampling DST-aware.
    """
    path = Path(data_dir) / f"{ticker.upper()}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=_COLUMNS,
        dtype={**_DTYPES, "Date": "string", "Time": "string"},
        engine="c",
    )

    timestamp = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%m/%d/%Y %H:%M",
        errors="raise",
    )
    timestamp = timestamp.dt.tz_localize(
        _NY_TZ, ambiguous="NaT", nonexistent="shift_forward"
    )

    df = (
        df.drop(columns=["Date", "Time"])
        .rename(columns=str.lower)
        .set_index(timestamp)
        .sort_index()
    )
    df.index.name = "timestamp"

    n_drop = df.index.isna().sum()
    if n_drop:
        logger.warning("%s: dropping %d rows with invalid (DST) timestamps", ticker, n_drop)
        df = df.loc[df.index.notna()]

    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(
        "Loaded %s: %d rows | %s -> %s | %.1f MB in memory",
        ticker,
        len(df),
        df.index.min(),
        df.index.max(),
        mem_mb,
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    for sym in ("AAPL", "AMZN", "JPM"):
        load_intraday(sym)
