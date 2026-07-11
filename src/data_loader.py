"""
data_loader.py
---------------
Populates the risk.db SQLite database with instrument metadata and daily
adjusted-close prices.

Two modes:
  1. live()      -> pulls real data via yfinance (requires internet + `pip install yfinance`)
  2. synthetic()  -> generates a reproducible multi-asset return series with a
                     realistic correlation structure, so the tool runs fully
                     offline for demos, tests, and CI.

Usage:
    python -m src.data_loader --mode synthetic
    python -m src.data_loader --mode live --start 2021-01-01
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "risk.db"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema.sql"

# A representative multi-asset universe: equities, bonds, commodities, credit, FX
UNIVERSE = {
    "SPY": ("S&P 500 ETF", "equity"),
    "EFA": ("MSCI EAFE Developed Mkts ETF", "equity"),
    "EEM": ("MSCI Emerging Mkts ETF", "equity"),
    "TLT": ("20+ Yr Treasury Bond ETF", "bond"),
    "LQD": ("Investment Grade Corp Bond ETF", "credit"),
    "HYG": ("High Yield Corp Bond ETF", "credit"),
    "GLD": ("Gold Trust", "commodity"),
    "DBC": ("Broad Commodity ETF", "commodity"),
    "UUP": ("US Dollar Bullish ETF", "fx"),
}


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def _insert_instruments(conn: sqlite3.Connection) -> None:
    rows = [(t, name, ac, "USD") for t, (name, ac) in UNIVERSE.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO instruments (ticker, name, asset_class, currency) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def _insert_prices(conn: sqlite3.Connection, prices: pd.DataFrame) -> None:
    """prices: DataFrame indexed by date, one column per ticker (adj close)."""
    long_df = prices.reset_index().melt(id_vars=prices.index.name or "index",
                                         var_name="ticker", value_name="adj_close")
    long_df.columns = ["price_date", "ticker", "adj_close"]
    long_df["price_date"] = pd.to_datetime(long_df["price_date"]).dt.strftime("%Y-%m-%d")
    long_df.dropna(subset=["adj_close"], inplace=True)
    conn.executemany(
        "INSERT OR REPLACE INTO prices (ticker, price_date, adj_close) VALUES (?, ?, ?)",
        long_df[["ticker", "price_date", "adj_close"]].itertuples(index=False, name=None),
    )
    conn.commit()


def synthetic(n_days: int = 750, seed: int = 42) -> pd.DataFrame:
    """
    Generates a reproducible synthetic price history for the UNIVERSE with a
    realistic cross-asset correlation structure (equities correlated with each
    other, negatively correlated with bonds, commodities semi-independent, etc.)
    so the whole pipeline runs without network access.
    """
    rng = np.random.default_rng(seed)
    tickers = list(UNIVERSE.keys())
    n = len(tickers)

    # Hand-specified target correlation blocks (rough, illustrative of real markets)
    # order: SPY EFA EEM TLT LQD HYG GLD DBC UUP
    target_corr = np.array([
        [1.00, 0.85, 0.75, -0.30, 0.10, 0.55, 0.05, 0.20, -0.20],
        [0.85, 1.00, 0.80, -0.25, 0.10, 0.50, 0.10, 0.20, -0.25],
        [0.75, 0.80, 1.00, -0.15, 0.15, 0.55, 0.15, 0.30, -0.30],
        [-0.30, -0.25, -0.15, 1.00, 0.55, -0.10, 0.20, -0.05, 0.15],
        [0.10, 0.10, 0.15, 0.55, 1.00, 0.45, 0.10, 0.05, 0.05],
        [0.55, 0.50, 0.55, -0.10, 0.45, 1.00, 0.10, 0.25, -0.15],
        [0.05, 0.10, 0.15, 0.20, 0.10, 0.10, 1.00, 0.35, -0.40],
        [0.20, 0.20, 0.30, -0.05, 0.05, 0.25, 0.35, 1.00, -0.25],
        [-0.20, -0.25, -0.30, 0.15, 0.05, -0.15, -0.40, -0.25, 1.00],
    ])
    annual_vol = np.array([0.17, 0.19, 0.23, 0.13, 0.08, 0.11, 0.15, 0.18, 0.07])
    daily_vol = annual_vol / np.sqrt(252)
    annual_drift = np.array([0.09, 0.06, 0.05, 0.02, 0.04, 0.05, 0.04, 0.02, 0.00])
    daily_drift = annual_drift / 252

    cov = np.outer(daily_vol, daily_vol) * target_corr
    # ensure positive semi-definite (numerical safety)
    cov = (cov + cov.T) / 2
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(n))

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    n_days = len(dates)  # guard against off-by-one edge cases in bdate_range
    z = rng.standard_normal((n_days, n))
    daily_returns = daily_drift + z @ chol.T

    prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "price_date"
    return df


def load_synthetic() -> None:
    conn = get_connection()
    init_schema(conn)
    _insert_instruments(conn)
    prices = synthetic()
    _insert_prices(conn, prices)
    conn.close()
    print(f"Loaded synthetic data for {len(UNIVERSE)} instruments into {DB_PATH}")


def load_live(start: str = "2021-01-01") -> None:
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit(
            "yfinance not installed. Run: pip install yfinance"
        ) from e

    conn = get_connection()
    init_schema(conn)
    _insert_instruments(conn)

    raw = yf.download(list(UNIVERSE.keys()), start=start, auto_adjust=True)["Close"]
    raw.index.name = "price_date"
    _insert_prices(conn, raw)
    conn.close()
    print(f"Loaded live data for {len(UNIVERSE)} instruments into {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load price data into risk.db")
    parser.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    parser.add_argument("--start", default="2021-01-01", help="Start date for live mode")
    args = parser.parse_args()

    if args.mode == "synthetic":
        load_synthetic()
    else:
        load_live(args.start)
