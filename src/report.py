"""
report.py
---------
Persists a risk run to the database and renders a Markdown risk report —
the kind of one-pager a risk analyst would attach to a morning risk meeting.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def persist_run(conn: sqlite3.Connection, portfolio_id: str, lookback_days: int,
                 confidence: float, horizon_days: int, port_vol: float,
                 param_var: float, hist_var: float, cvar: float, div_ratio: float,
                 contributions: pd.DataFrame, factors: pd.DataFrame) -> int:
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO risk_runs
           (portfolio_id, lookback_days, confidence_level, horizon_days,
            portfolio_vol_annual, parametric_var, historical_var,
            cvar_expected_shortfall, diversification_ratio)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (portfolio_id, lookback_days, confidence, horizon_days, port_vol,
         param_var, hist_var, cvar, div_ratio),
    )
    run_id = cur.lastrowid

    for ticker, row in contributions.iterrows():
        cur.execute(
            """INSERT INTO risk_contributions
               (run_id, ticker, weight, standalone_vol_annual,
                marginal_contribution, component_contribution, pct_of_total_risk)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, ticker, row["weight"], row["standalone_vol_annual"],
             row["marginal_contribution"], row["component_contribution"],
             row["pct_of_total_risk"]),
        )

    for _, row in factors.iterrows():
        cur.execute(
            """INSERT INTO factor_exposures
               (run_id, factor_id, explained_variance_ratio, portfolio_loading)
               VALUES (?, ?, ?, ?)""",
            (run_id, int(row["factor_id"]), row["explained_variance_ratio"],
             row["portfolio_loading"]),
        )

    conn.commit()
    return run_id


def render_markdown_report(portfolio_id: str, run_id: int, lookback_days: int,
                            confidence: float, horizon_days: int, port_vol: float,
                            param_var: float, hist_var: float, cvar: float,
                            div_ratio: float, contributions: pd.DataFrame,
                            factors: pd.DataFrame) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# Portfolio Risk Report — {portfolio_id}")
    lines.append(f"_Run #{run_id} · generated {ts} · {lookback_days}d lookback · "
                 f"{int(confidence * 100)}% confidence · {horizon_days}d horizon_\n")

    lines.append("## Headline Risk Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Annualized Portfolio Volatility | {port_vol:.2%} |")
    lines.append(f"| Parametric VaR ({int(confidence*100)}%, {horizon_days}d) | {param_var:.2%} |")
    lines.append(f"| Historical VaR ({int(confidence*100)}%, {horizon_days}d) | {hist_var:.2%} |")
    lines.append(f"| CVaR / Expected Shortfall | {cvar:.2%} |")
    lines.append(f"| Diversification Ratio | {div_ratio:.2f}x |\n")

    lines.append("## Risk Contribution by Instrument\n")
    lines.append("Component contributions are an exact Euler decomposition of "
                 "total portfolio volatility — they sum to 100%.\n")
    lines.append("| Ticker | Weight | Standalone Vol | Marginal Contrib. | "
                 "Component Contrib. | % of Total Risk |")
    lines.append("|---|---|---|---|---|---|")
    for ticker, row in contributions.iterrows():
        lines.append(
            f"| {ticker} | {row['weight']:.1%} | {row['standalone_vol_annual']:.1%} | "
            f"{row['marginal_contribution']:.1%} | {row['component_contribution']:.1%} | "
            f"{row['pct_of_total_risk']:.1%} |"
        )
    lines.append("")

    lines.append("## Systematic Factor Decomposition (PCA)\n")
    lines.append("Top principal components of the asset covariance matrix, "
                 "treated as data-driven systematic risk factors.\n")
    lines.append("| Factor | Variance Explained | Portfolio Loading | Dominant Instruments |")
    lines.append("|---|---|---|---|")
    for _, row in factors.iterrows():
        lines.append(
            f"| PC{int(row['factor_id'])} | {row['explained_variance_ratio']:.1%} | "
            f"{row['portfolio_loading']:.3f} | {row['top_drivers']} |"
        )
    lines.append("")

    top_riskiest = contributions.index[0]
    top_pct = contributions.iloc[0]["pct_of_total_risk"]
    lines.append("## Notes\n")
    lines.append(f"- **{top_riskiest}** is the single largest risk driver, "
                 f"contributing {top_pct:.1%} of total portfolio volatility.")
    if div_ratio < 1.3:
        lines.append("- Diversification ratio is relatively low — risk is "
                     "concentrated rather than well spread across holdings.")
    else:
        lines.append("- Diversification ratio indicates meaningful risk reduction "
                     "from cross-asset correlation benefits.")

    report_path = REPORTS_DIR / f"{portfolio_id}_run{run_id}.md"
    report_path.write_text("\n".join(lines))
    return report_path
