-- Portfolio Risk Decomposition Tool — Database Schema
-- SQLite (portable, zero-setup). Swap the CREATE TABLE dialect for Postgres if needed.

DROP TABLE IF EXISTS prices;
DROP TABLE IF EXISTS instruments;
DROP TABLE IF EXISTS positions;
DROP TABLE IF EXISTS risk_runs;
DROP TABLE IF EXISTS risk_contributions;
DROP TABLE IF EXISTS factor_exposures;

-- Master list of tradable instruments across asset classes
CREATE TABLE instruments (
    ticker          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    asset_class     TEXT NOT NULL CHECK (asset_class IN
                        ('equity', 'bond', 'commodity', 'fx', 'credit', 'alt')),
    currency        TEXT NOT NULL DEFAULT 'USD'
);

-- Daily adjusted close prices, long format
CREATE TABLE prices (
    ticker          TEXT NOT NULL REFERENCES instruments(ticker),
    price_date      DATE NOT NULL,
    adj_close       REAL NOT NULL,
    PRIMARY KEY (ticker, price_date)
);

-- Current portfolio holdings (weights should sum to ~1.0, or use market_value)
CREATE TABLE positions (
    portfolio_id    TEXT NOT NULL,
    ticker          TEXT NOT NULL REFERENCES instruments(ticker),
    weight          REAL NOT NULL,
    market_value    REAL,
    PRIMARY KEY (portfolio_id, ticker)
);

-- One row per risk computation (so history of runs is queryable/auditable)
CREATE TABLE risk_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id        TEXT NOT NULL,
    run_timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    lookback_days        INTEGER NOT NULL,
    confidence_level     REAL NOT NULL,
    horizon_days          INTEGER NOT NULL,
    portfolio_vol_annual REAL NOT NULL,
    parametric_var       REAL NOT NULL,
    historical_var        REAL NOT NULL,
    cvar_expected_shortfall REAL NOT NULL,
    diversification_ratio REAL NOT NULL
);

-- Per-instrument risk decomposition results for a given run
CREATE TABLE risk_contributions (
    run_id              INTEGER NOT NULL REFERENCES risk_runs(run_id),
    ticker              TEXT NOT NULL REFERENCES instruments(ticker),
    weight              REAL NOT NULL,
    standalone_vol_annual REAL NOT NULL,
    marginal_contribution REAL NOT NULL,
    component_contribution REAL NOT NULL,
    pct_of_total_risk    REAL NOT NULL,
    PRIMARY KEY (run_id, ticker)
);

-- PCA-based systematic factor exposures for a given run
CREATE TABLE factor_exposures (
    run_id              INTEGER NOT NULL REFERENCES risk_runs(run_id),
    factor_id           INTEGER NOT NULL,
    explained_variance_ratio REAL NOT NULL,
    portfolio_loading    REAL NOT NULL,
    PRIMARY KEY (run_id, factor_id)
);

CREATE INDEX idx_prices_date ON prices(price_date);
CREATE INDEX idx_positions_portfolio ON positions(portfolio_id);
