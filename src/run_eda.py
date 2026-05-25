"""Stage-1 EDA pipeline — terminal-runnable equivalent of notebooks/01_eda.ipynb.

Usage (from the project root):
    python3 -m src.run_eda

Produces:
    data/processed/<TICKER>_intraday_clean.parquet
    results/figures/fig_price_paths.png
    results/figures/fig_intraday_pattern.png
    results/figures/fig_daily_returns_dist.png
    results/tables/eda_summary.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_intraday  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

TICKERS = ["AAPL", "AMZN", "JPM"]
COLOR = {"AAPL": "#1f77b4", "AMZN": "#ff7f0e", "JPM": "#2ca02c"}


def sanity_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, df in data.items():
        t = df.index
        bars_per_day = df.groupby(t.date).size()
        rows.append({
            "ticker": sym,
            "n_rows": len(df),
            "start": t.min().date(),
            "end": t.max().date(),
            "n_trading_days": int(bars_per_day.size),
            "bars_per_day_mean": round(bars_per_day.mean(), 1),
            "bars_per_day_median": int(bars_per_day.median()),
            "bars_per_day_min": int(bars_per_day.min()),
            "bars_per_day_max": int(bars_per_day.max()),
            "n_duplicates": int(t.duplicated().sum()),
            "n_nan_close": int(df["close"].isna().sum()),
            "earliest_time": str(min(t.time)),
            "latest_time": str(max(t.time)),
            "weekends_present": bool((t.weekday >= 5).any()),
        })
    return pd.DataFrame(rows).set_index("ticker")


def big_minute_returns(df: pd.DataFrame, threshold: float = 0.15) -> pd.DataFrame:
    r = np.log(df["close"]).diff()
    big = r[r.abs() > threshold]
    return big.rename("log_return").to_frame()


def clean_intraday(df: pd.DataFrame) -> pd.DataFrame:
    t = df.index
    minutes = t.hour * 60 + t.minute
    mask = (
        (t.weekday < 5)
        & (minutes >= 9 * 60 + 30)
        & (minutes <= 15 * 60 + 59)
        & df.notna().all(axis=1)
    )
    return df.loc[mask].copy()


def plot_price_paths(data: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, sym in zip(axes, TICKERS):
        dc = data[sym]["close"].resample("1D").last().dropna()
        ax.plot(dc.index, dc.values, color=COLOR[sym], lw=0.9)
        ax.set_ylabel(f"{sym} close (USD)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Daily close prices, 2016-01-04 → 2024-12-31 (split-adjusted)", y=0.995)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_intraday_volume(data: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for sym, df in data.items():
        mod = df.index.hour * 60 + df.index.minute
        prof = df.groupby(mod)["volume"].mean()
        prof = prof / prof.sum()
        ax.plot(prof.index, prof.values * 100, label=sym, color=COLOR[sym], lw=1.4)
    ax.set_xlabel("Minute of day (NY time)")
    ax.set_ylabel("Share of session volume (%)")
    ax.set_title("Average intraday volume profile (regular-hours only)")
    xt = [9 * 60 + 30, 10 * 60, 11 * 60, 12 * 60, 13 * 60, 14 * 60, 15 * 60, 16 * 60]
    ax.set_xticks(xt)
    ax.set_xticklabels(["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"])
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


def plot_daily_return_dist(data: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    daily_close = pd.DataFrame({sym: df["close"].resample("1D").last() for sym, df in data.items()})
    daily_ret = np.log(daily_close).diff().dropna()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, sym in zip(axes, TICKERS):
        r = daily_ret[sym].dropna()
        ax.hist(r, bins=80, density=True, color=COLOR[sym], alpha=0.7, edgecolor="white")
        mu, sd = r.mean(), r.std()
        xs = np.linspace(r.min(), r.max(), 200)
        ax.plot(
            xs,
            np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)),
            "k--",
            lw=1.0,
            label="N(μ̂, σ̂²)",
        )
        ax.set_title(f"{sym}  σ={sd*100:.2f}%  kurt={r.kurt():.1f}")
        ax.set_xlabel("daily log-return")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Distribution of daily log-returns (close-to-close)", y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return daily_ret


def main() -> None:
    for d in (PROCESSED_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    print("\n>>> 1. Loading raw intraday files")
    data = {sym: load_intraday(sym, RAW_DIR) for sym in TICKERS}

    print("\n>>> 2. Sanity table")
    print(sanity_table(data).to_string())

    print("\n>>> 2b. Stock-split / large-move screen (|log-return| > 15 %)")
    for sym, df in data.items():
        big = big_minute_returns(df)
        print(f"  {sym}: {len(big)} bar(s)")
        if len(big):
            print(big.to_string())

    print("\n>>> 3. Building figures")
    plot_price_paths(data, FIG_DIR / "fig_price_paths.png")
    plot_intraday_volume(data, FIG_DIR / "fig_intraday_pattern.png")
    plot_daily_return_dist(data, FIG_DIR / "fig_daily_returns_dist.png")
    print(f"  saved: {sorted(p.name for p in FIG_DIR.glob('*.png'))}")

    print("\n>>> 4. Cleaning + parquet + summary table")
    rows = []
    for sym, df in data.items():
        clean = clean_intraday(df)
        out = PROCESSED_DIR / f"{sym}_intraday_clean.parquet"
        clean.to_parquet(out, compression="snappy")
        n_dropped = len(df) - len(clean)
        print(f"  {sym}: kept {len(clean):,} / {len(df):,} (dropped {n_dropped}) -> {out.name}")

        dc = clean["close"].resample("1D").last().dropna()
        dr = np.log(dc).diff().dropna()
        p5 = clean["close"].resample("5min").last().dropna()
        r5 = np.log(p5).diff()
        date5 = pd.Series(r5.index.normalize(), index=r5.index)
        r5 = r5[date5.eq(date5.shift(1))]
        rv_daily = r5.pow(2).groupby(r5.index.date).sum()
        rows.append({
            "ticker": sym,
            "n_minute_bars": len(clean),
            "n_trading_days": int(dc.size),
            "avg_daily_volume": int(
                clean["volume"].resample("1D").sum().replace(0, np.nan).dropna().mean()
            ),
            "avg_daily_return_bps": round(dr.mean() * 1e4, 3),
            "avg_daily_realized_var": round(rv_daily.mean(), 6),
            "avg_daily_realized_vol_pct": round(np.sqrt(rv_daily.mean()) * 100, 3),
        })

    summary = pd.DataFrame(rows).set_index("ticker")
    summary.to_csv(TABLE_DIR / "eda_summary.csv")
    print("\n>>> Stage-1 summary (results/tables/eda_summary.csv):")
    print(summary.to_string())

    print("\n>>> Done.")


if __name__ == "__main__":
    main()
