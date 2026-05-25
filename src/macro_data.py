"""Macro-financial exogenous variables for ARFIMAX / NARX models.

Builds a 7-feature daily macro-financial panel from freely available data
(FRED + Yahoo Finance). Five features reconstruct Bucci's (2020) predictors
(MKT, TB, TS, DEF, and INF -- the last a daily breakeven proxy for his
monthly inflation); two further daily volatility proxies *not* in Bucci's
set are added: log VIX and the dollar-index return.

    MKT      S&P 500 daily excess return  (S&P return − daily T-bill)   [Bucci]
    TB       3-month T-bill level (%)                                   [Bucci]
    TS       term spread = 10Y − 3M (%)                                 [Bucci]
    DEF      default spread = DBAA − DAAA (%, daily Moody's yields)      [Bucci]
    INF      5-year breakeven inflation expectation (%)                 [Bucci proxy]
    VIX      log CBOE VIX                                               [added]
    DXY_RET  US dollar index daily return                               [added]

Data-availability notes:
* The daily ICE BAML high-yield OAS (BAMLH0A0HYM2) is returned truncated to
  2023-05-onward by the pandas_datareader FRED endpoint, so it is dropped.
  The default spread (DEF) already proxies credit risk, leaving 7 features.
* We use the *daily* Moody's series DBAA / DAAA rather than the monthly BAA /
  AAA, so DEF is a daily series (the monthly versions ffill to a step
  function).
* Yahoo Finance occasionally returns HTTP 429 (rate-limited). The build
  function downloads all three Yahoo tickers in a single request to minimise
  calls; callers should cache the result (``macro_features.csv``) and load
  from disk on subsequent runs rather than re-downloading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FRED_SERIES = {
    "DGS3MO": "TB",        # 3-month T-bill rate (daily)
    "DGS10": "GS10",       # 10-year Treasury yield (daily)
    "DBAA": "BAA",         # Moody's BAA corporate bond yield (daily)
    "DAAA": "AAA",         # Moody's AAA corporate bond yield (daily)
    "T5YIE": "INF",        # 5-year breakeven inflation expectation (daily)
}

YAHOO_SERIES = {
    "^GSPC": "SPX",        # S&P 500
    "^VIX": "VIX",         # CBOE Volatility Index
    "DX-Y.NYB": "DXY",     # US Dollar Index
}


def download_fred(start: str = "2015-01-01", end: str = "2025-01-31") -> pd.DataFrame:
    """Download the FRED series via pandas_datareader. Index = calendar date."""
    from pandas_datareader import data as pdr

    series = {}
    for fred_code, alias in FRED_SERIES.items():
        try:
            df = pdr.DataReader(fred_code, "fred", start, end)
            series[alias] = df[fred_code]
        except Exception as e:  # noqa: BLE001
            print(f"FRED download failed for {fred_code}: {e}")
    return pd.DataFrame(series)


def download_yahoo(start: str = "2015-01-01", end: str = "2025-01-31") -> pd.DataFrame:
    """Download the Yahoo tickers in a single batched request (fewer 429s)."""
    import yfinance as yf

    tickers = list(YAHOO_SERIES.keys())
    raw = yf.download(
        tickers, start=start, end=end, progress=False, auto_adjust=False
    )
    # Batched multi-ticker download returns a column MultiIndex (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:  # single ticker fallback
        close = raw[["Close"]]
        close.columns = tickers
    out = pd.DataFrame(index=close.index)
    for yf_code, alias in YAHOO_SERIES.items():
        if yf_code in close.columns:
            out[alias] = close[yf_code]
        else:
            print(f"Yahoo series missing in response: {yf_code}")
    return out


def build_macro_features(
    start: str = "2015-01-01", end: str = "2025-01-31"
) -> pd.DataFrame:
    """Combine FRED + Yahoo, derive the 7 macro features, drop NaN rows.

    All level series are forward-filled across non-trading days before the
    feature transforms (FRED publishes on a business-day calendar with the
    occasional gap). Returns are computed *after* the ffill so a holiday
    does not create a spurious zero return.
    """
    fred = download_fred(start, end)
    yahoo = download_yahoo(start, end)
    df = pd.concat([fred, yahoo], axis=1).sort_index().ffill()

    out = pd.DataFrame(index=df.index)
    spx_ret = df["SPX"].pct_change()
    tb_daily = df["TB"] / 100.0 / 252.0          # annual % → daily fraction
    out["MKT"] = spx_ret - tb_daily              # equity excess return
    out["TB"] = df["TB"]                         # short rate level
    out["TS"] = df["GS10"] - df["TB"]            # term spread
    out["DEF"] = df["BAA"] - df["AAA"]           # default spread
    out["INF"] = df["INF"]                       # inflation expectation
    out["VIX"] = np.log(df["VIX"])               # log VIX
    out["DXY_RET"] = df["DXY"].pct_change()      # dollar return

    return out.dropna()


def lag_features(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Lag every column by ``lag`` days to avoid look-ahead. Default 1 day."""
    return df.shift(lag).dropna()


MACRO_COLUMNS = ["MKT", "TB", "TS", "DEF", "INF", "VIX", "DXY_RET"]


__all__ = [
    "download_fred",
    "download_yahoo",
    "build_macro_features",
    "lag_features",
    "FRED_SERIES",
    "YAHOO_SERIES",
    "MACRO_COLUMNS",
]
