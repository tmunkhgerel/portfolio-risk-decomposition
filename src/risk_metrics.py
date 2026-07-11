"""
risk_metrics.py
----------------
Core statistical risk engine: turns a price panel + weight vector into
portfolio-level risk numbers (volatility, parametric/historical VaR, CVaR).

All functions are pure (no DB, no I/O) so they're easy to unit test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns from a wide price panel (date index, ticker columns)."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Sample covariance matrix of daily log returns.
    Annualized by scaling with TRADING_DAYS (assumes i.i.d. daily returns).
    """
    cov = returns.cov()
    return cov * TRADING_DAYS if annualize else cov


def shrinkage_covariance(returns: pd.DataFrame, delta: float = 0.2,
                          annualize: bool = True) -> pd.DataFrame:
    """
    Ledoit-Wolf-style shrinkage toward a constant-correlation target.
    Shrinking the sample covariance reduces estimation noise, which matters
    a lot when the number of assets is close to the number of observations —
    standard practice in real risk models, not just an academic nicety.

    delta: shrinkage intensity in [0, 1]. 0 = pure sample cov, 1 = pure target.
    """
    sample = returns.cov()
    n = sample.shape[0]
    var = np.diag(sample).copy()
    avg_corr = (sample.values / np.sqrt(np.outer(var, var))).sum()
    avg_corr = (avg_corr - n) / (n * (n - 1))
    target = avg_corr * np.sqrt(np.outer(var, var))
    np.fill_diagonal(target, var)
    shrunk = delta * target + (1 - delta) * sample.values
    shrunk_df = pd.DataFrame(shrunk, index=sample.index, columns=sample.columns)
    return shrunk_df * TRADING_DAYS if annualize else shrunk_df


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Annualized portfolio volatility: sqrt(w' Sigma w)."""
    w = weights.reindex(cov.index).fillna(0.0).values
    return float(np.sqrt(w @ cov.values @ w))


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """
    Weighted average of standalone vols divided by portfolio vol.
    DR > 1 means diversification is reducing risk below what you'd get if
    everything moved in lockstep; DR = 1 means no diversification benefit.
    """
    w = weights.reindex(cov.index).fillna(0.0)
    standalone_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(w.values @ standalone_vols)
    port_vol = portfolio_volatility(weights, cov)
    return weighted_avg_vol / port_vol if port_vol > 0 else np.nan


def parametric_var(port_vol_annual: float, confidence: float = 0.99,
                    horizon_days: int = 1, portfolio_value: float = 1.0) -> float:
    """
    Variance-covariance (delta-normal) VaR, assuming normally distributed
    returns with zero mean. Scales annual vol down to the target horizon
    via the square-root-of-time rule.
    """
    daily_vol = port_vol_annual / np.sqrt(TRADING_DAYS)
    horizon_vol = daily_vol * np.sqrt(horizon_days)
    z = norm.ppf(confidence)
    return float(z * horizon_vol * portfolio_value)


def historical_var(portfolio_returns: pd.Series, confidence: float = 0.99,
                    horizon_days: int = 1, portfolio_value: float = 1.0) -> float:
    """
    Non-parametric VaR: the empirical (1 - confidence) quantile of historical
    portfolio P&L, scaled to the horizon by sqrt(time). Makes no distributional
    assumption, so it captures skew/fat tails the parametric method misses.
    """
    scaled = portfolio_returns * np.sqrt(horizon_days)
    loss_quantile = -np.quantile(scaled, 1 - confidence)
    return float(loss_quantile * portfolio_value)


def conditional_var(portfolio_returns: pd.Series, confidence: float = 0.99,
                     horizon_days: int = 1, portfolio_value: float = 1.0) -> float:
    """
    Expected Shortfall / CVaR: average loss in the tail beyond the VaR
    threshold. More informative than VaR alone since it reflects tail severity,
    not just the cutoff point — this is what regulators (FRTB) actually use now.
    """
    scaled = portfolio_returns * np.sqrt(horizon_days)
    threshold = np.quantile(scaled, 1 - confidence)
    tail_losses = scaled[scaled <= threshold]
    if len(tail_losses) == 0:
        return float(-threshold * portfolio_value)
    return float(-tail_losses.mean() * portfolio_value)


def portfolio_return_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Historical daily portfolio return series, given asset returns and weights."""
    w = weights.reindex(returns.columns).fillna(0.0)
    return returns.mul(w, axis=1).sum(axis=1)
