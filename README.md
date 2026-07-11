# Multi-Asset Portfolio Risk Decomposition Tool

A Python/SQL tool that answers the question every risk desk asks every morning:
**"where is the risk in this portfolio actually coming from?"**

Given a multi-asset portfolio (equities, government bonds, credit, commodities, FX),
it computes portfolio volatility, parametric and historical VaR, and Expected
Shortfall (CVaR) — then decomposes total risk into exact **per-instrument
contributions** and **systematic factor exposures**, and persists every run to
a SQL database for auditability.

This mirrors the day-to-day workflow of a market/portfolio risk analyst: not
a trading strategy, but the risk-reporting infrastructure that sits underneath
every allocation decision.

## What it computes

| Layer | Method | Output |
|---|---|---|
| **Portfolio risk** | Covariance-based, with Ledoit-Wolf-style shrinkage | Annualized volatility |
| **Tail risk** | Parametric (delta-normal) VaR, historical VaR, CVaR/Expected Shortfall | Loss estimates at chosen confidence & horizon |
| **Instrument attribution** | Euler decomposition (marginal & component contribution to risk) | Exact % of total risk per holding, sums to 100% |
| **Systematic factors** | PCA on the asset covariance matrix | Data-driven risk factors (e.g. "risk-on/off"), portfolio loading per factor |

The Euler decomposition is the key piece: because portfolio volatility is
homogeneous of degree 1 in the weights, `Σ component_contribution_i` sums
**exactly** to total portfolio volatility. That identity is what makes the
per-instrument breakdown a legitimate attribution rather than a heuristic.

## Architecture

```
portfolio-risk-decomposition/
├── data/
│   └── schema.sql            # instruments, prices, positions, risk_runs, risk_contributions, factor_exposures
├── src/
│   ├── data_loader.py        # yfinance (live) or reproducible synthetic multi-asset data → SQLite
│   ├── risk_metrics.py       # returns, shrinkage covariance, portfolio vol, VaR, CVaR
│   ├── risk_decomposition.py # marginal/component risk contribution, PCA factor decomposition
│   └── report.py             # persists results to SQL, renders Markdown risk report
├── tests/
│   └── test_risk_metrics.py  # unit tests against hand-derivable analytical results
├── main.py                   # CLI entry point
└── reports/                  # generated Markdown risk reports land here
```

Everything downstream of `data_loader.py` is pure Python (no I/O), which is
what makes it unit-testable against closed-form answers rather than just
"it runs without crashing."

## Quickstart

```bash
pip install -r requirements.txt

# Generates a reproducible synthetic 9-asset dataset (equities, bonds, credit,
# commodities, FX) — no API key or network access required.
python -m src.data_loader --mode synthetic

# Run the full risk decomposition on the default demo portfolio
python main.py --portfolio-id demo_balanced --confidence 0.99 --horizon 1
```

This writes results to `data/risk.db` and a report to `reports/demo_balanced_run1.md`.

To use real market data instead:
```bash
pip install yfinance
python -m src.data_loader --mode live --start 2021-01-01
```

### Using your own portfolio

Edit `DEFAULT_WEIGHTS` in `main.py`, or extend `main.py` to accept a
`--weights-file positions.csv`. The instrument universe is defined in
`UNIVERSE` in `src/data_loader.py` — add tickers there (with an asset class)
and re-run the loader.

## Example output

Running the demo portfolio (30% SPY, 10% EFA, 10% EEM, 20% TLT, 10% LQD,
5% HYG, 8% GLD, 4% DBC, 3% UUP) against ~2 years of data produces a report like:

```
Annualized volatility          : 9.1%
99% Parametric VaR (1d)        : 1.3%
99% Historical VaR (1d)        : 1.2%
Expected Shortfall (CVaR)      : 1.3%
Diversification ratio          : 1.68x
```

with a full breakdown showing, for example, that SPY at 30% of *weight* can
account for close to half of total portfolio *risk* — the kind of gap between
capital allocation and risk allocation that risk decomposition exists to surface.

See `reports/` after running for the full instrument- and factor-level tables.

## Methodology notes

- **Covariance shrinkage**: the raw sample covariance matrix is noisy,
  especially with a limited lookback window relative to the number of assets.
  A Ledoit-Wolf-style shrinkage toward a constant-correlation target
  (`--shrinkage`, default 0.2) trades a small amount of bias for a large
  reduction in estimation error — standard practice, not just a textbook aside.
- **VaR methods are shown side by side deliberately.** Parametric VaR assumes
  normally distributed returns and will understate tail risk when returns are
  fat-tailed or skewed; historical VaR and CVaR make no such assumption. Showing
  both — and CVaR, which reflects tail *severity* rather than just a cutoff — is
  closer to how a real risk report is built than presenting a single VaR number.
- **PCA factors are a lightweight stand-in for a full factor model** (e.g.
  Barra, Axioma). They require no external factor data and still separate
  systematic from idiosyncratic risk — PC1 in a diversified multi-asset book is
  almost always a recognizable "risk-on/risk-off" factor, which is a useful
  sanity check on the covariance matrix itself.
- **Everything persists to SQL**, not just prints to console. `risk_runs`,
  `risk_contributions`, and `factor_exposures` are separate normalized tables
  so historical risk runs are queryable — e.g. "how did SPY's risk contribution
  trend over the last 20 runs" is a single SQL query, not a re-computation.

## Possible extensions

- Stressed/historical scenario analysis (e.g. replay 2008, 2020 COVID shock)
- Monte Carlo VaR with fat-tailed (Student-t) return simulation
- Incremental VaR for proposed trades (risk impact before executing)
- Replace PCA factors with a proper style factor model (value/momentum/carry)
- Rolling risk dashboards (Streamlit) reading directly from `risk.db`

## Tests

```bash
python -m pytest tests/ -v
```

Tests check the risk engine against hand-derivable answers — e.g. that
component risk contributions sum exactly to portfolio volatility, that CVaR
is always ≥ VaR, and that VaR scales with `sqrt(horizon)` as the model requires.
