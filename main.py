"""
main.py
-------
CLI entry point for the Portfolio Risk Decomposition Tool.

Example:
    python main.py --portfolio-id demo_balanced --confidence 0.99 --horizon 1

Reads prices + positions from data/risk.db (run src/data_loader.py first if
it doesn't exist yet), computes the full risk decomposition, writes the
results back to SQL, and renders a Markdown report to reports/.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import DB_PATH, UNIVERSE, load_synthetic
from src.risk_metrics import (
    compute_log_returns, shrinkage_covariance, portfolio_volatility,
    diversification_ratio, parametric_var, historical_var, conditional_var,
    portfolio_return_series,
)
from src.risk_decomposition import component_contribution_to_risk, pca_factor_decomposition
from src.report import persist_run, render_markdown_report

# Default demo portfolio: a diversified multi-asset allocation
DEFAULT_WEIGHTS = {
    "SPY": 0.30, "EFA": 0.10, "EEM": 0.10,
    "TLT": 0.20, "LQD": 0.10, "HYG": 0.05,
    "GLD": 0.08, "DBC": 0.04, "UUP": 0.03,
}


def load_prices(conn: sqlite3.Connection, lookback_days: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ticker, price_date, adj_close FROM prices ORDER BY price_date", conn
    )
    if df.empty:
        raise SystemExit("No price data found. Run: python -m src.data_loader --mode synthetic")
    wide = df.pivot(index="price_date", columns="ticker", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    return wide.tail(lookback_days + 1)  # +1 because returns drop the first row


def store_positions(conn: sqlite3.Connection, portfolio_id: str, weights: dict) -> None:
    conn.execute("DELETE FROM positions WHERE portfolio_id = ?", (portfolio_id,))
    conn.executemany(
        "INSERT INTO positions (portfolio_id, ticker, weight) VALUES (?, ?, ?)",
        [(portfolio_id, t, w) for t, w in weights.items()],
    )
    conn.commit()


def run(portfolio_id: str, weights: dict, lookback_days: int, confidence: float,
        horizon_days: int, shrinkage: float) -> None:
    if not DB_PATH.exists():
        print("No database found — generating synthetic demo data first...")
        load_synthetic()

    conn = sqlite3.connect(DB_PATH)
    store_positions(conn, portfolio_id, weights)

    prices = load_prices(conn, lookback_days)
    returns = compute_log_returns(prices)
    w = pd.Series(weights)

    cov = shrinkage_covariance(returns, delta=shrinkage)
    port_vol = portfolio_volatility(w, cov)
    div_ratio = diversification_ratio(w, cov)

    port_returns = portfolio_return_series(returns, w)
    param_var = parametric_var(port_vol, confidence, horizon_days)
    hist_var = historical_var(port_returns, confidence, horizon_days)
    cvar = conditional_var(port_returns, confidence, horizon_days)

    contributions = component_contribution_to_risk(w, cov)
    factors = pca_factor_decomposition(returns, w, n_factors=3)

    run_id = persist_run(conn, portfolio_id, lookback_days, confidence, horizon_days,
                          port_vol, param_var, hist_var, cvar, div_ratio,
                          contributions, factors)

    report_path = render_markdown_report(portfolio_id, run_id, lookback_days, confidence,
                                          horizon_days, port_vol, param_var, hist_var,
                                          cvar, div_ratio, contributions, factors)

    conn.close()

    print(f"\nRisk run #{run_id} complete for portfolio '{portfolio_id}'")
    print(f"  Annualized volatility : {port_vol:.2%}")
    print(f"  {int(confidence*100)}% Parametric VaR ({horizon_days}d) : {param_var:.2%}")
    print(f"  {int(confidence*100)}% Historical VaR ({horizon_days}d) : {hist_var:.2%}")
    print(f"  Expected Shortfall (CVaR)     : {cvar:.2%}")
    print(f"  Diversification ratio         : {div_ratio:.2f}x")
    print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Asset Portfolio Risk Decomposition Tool")
    parser.add_argument("--portfolio-id", default="demo_balanced")
    parser.add_argument("--lookback", type=int, default=500, help="Trading days of history to use")
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--horizon", type=int, default=1, help="VaR horizon in trading days")
    parser.add_argument("--shrinkage", type=float, default=0.2,
                        help="Covariance shrinkage intensity, 0 (sample) to 1 (target)")
    args = parser.parse_args()

    run(args.portfolio_id, DEFAULT_WEIGHTS, args.lookback, args.confidence,
        args.horizon, args.shrinkage)
