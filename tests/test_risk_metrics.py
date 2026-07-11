"""
Unit tests for the risk engine. Uses small hand-built examples with known
analytical answers rather than relying on the synthetic/live data pipeline,
so these tests are fast and fully deterministic.
"""
import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    compute_log_returns, covariance_matrix, portfolio_volatility,
    diversification_ratio, parametric_var, historical_var, conditional_var,
    portfolio_return_series,
)
from src.risk_decomposition import (
    marginal_contribution_to_risk, component_contribution_to_risk,
    pca_factor_decomposition,
)


@pytest.fixture
def toy_returns():
    """Two assets, no correlation, easy-to-verify vol numbers."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(0, 0.01, n)
    b = rng.normal(0, 0.02, n)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"A": a, "B": b}, index=dates)


def test_log_returns_shape():
    prices = pd.DataFrame({"A": [100, 101, 102], "B": [50, 49, 51]})
    rets = compute_log_returns(prices)
    assert len(rets) == 2  # one row lost to differencing


def test_covariance_matrix_diagonal_matches_variance(toy_returns):
    cov = covariance_matrix(toy_returns)
    assert cov.loc["A", "A"] == pytest.approx(toy_returns["A"].var() * 252, rel=1e-6)


def test_portfolio_volatility_matches_manual_quadratic_form(toy_returns):
    cov = covariance_matrix(toy_returns)
    w = pd.Series({"A": 0.5, "B": 0.5})
    port_vol = portfolio_volatility(w, cov)
    # full w' Sigma w, including the sample cross-covariance term (A and B
    # are drawn independently, but a finite sample will have a small non-zero
    # sample covariance, so the cross term can't be dropped from the check)
    expected = np.sqrt(
        0.5**2 * cov.loc["A", "A"] + 0.5**2 * cov.loc["B", "B"]
        + 2 * 0.5 * 0.5 * cov.loc["A", "B"]
    )
    assert port_vol == pytest.approx(expected, rel=1e-6)


def test_diversification_ratio_at_least_one(toy_returns):
    cov = covariance_matrix(toy_returns)
    w = pd.Series({"A": 0.5, "B": 0.5})
    dr = diversification_ratio(w, cov)
    assert dr >= 1.0  # zero correlation must give diversification benefit


def test_component_contributions_sum_to_portfolio_vol(toy_returns):
    cov = covariance_matrix(toy_returns)
    w = pd.Series({"A": 0.5, "B": 0.5})
    contrib = component_contribution_to_risk(w, cov)
    port_vol = portfolio_volatility(w, cov)
    assert contrib["component_contribution"].sum() == pytest.approx(port_vol, rel=1e-6)
    assert contrib["pct_of_total_risk"].sum() == pytest.approx(1.0, rel=1e-6)


def test_marginal_contribution_matches_manual_formula(toy_returns):
    cov = covariance_matrix(toy_returns)
    w = pd.Series({"A": 0.5, "B": 0.5})
    mcr = marginal_contribution_to_risk(w, cov)
    port_vol = portfolio_volatility(w, cov)
    manual = (cov.values @ w.values) / port_vol
    np.testing.assert_allclose(mcr.values, manual, rtol=1e-6)


def test_var_is_positive_and_scales_with_horizon(toy_returns):
    cov = covariance_matrix(toy_returns)
    w = pd.Series({"A": 0.5, "B": 0.5})
    port_vol = portfolio_volatility(w, cov)
    var_1d = parametric_var(port_vol, confidence=0.99, horizon_days=1)
    var_10d = parametric_var(port_vol, confidence=0.99, horizon_days=10)
    assert var_1d > 0
    assert var_10d == pytest.approx(var_1d * np.sqrt(10), rel=1e-6)


def test_cvar_is_at_least_var(toy_returns):
    w = pd.Series({"A": 0.5, "B": 0.5})
    port_returns = portfolio_return_series(toy_returns, w)
    hvar = historical_var(port_returns, confidence=0.99)
    cvar = conditional_var(port_returns, confidence=0.99)
    # Expected shortfall averages losses beyond VaR, so it must be >= VaR
    assert cvar >= hvar


def test_pca_factors_explain_decreasing_variance(toy_returns):
    w = pd.Series({"A": 0.5, "B": 0.5})
    factors = pca_factor_decomposition(toy_returns, w, n_factors=2)
    ratios = factors["explained_variance_ratio"].values
    assert ratios[0] >= ratios[1]
    assert ratios.sum() <= 1.0 + 1e-9


def test_weights_with_missing_ticker_are_ignored_gracefully(toy_returns):
    cov = covariance_matrix(toy_returns)
    w = pd.Series({"A": 0.5, "C": 0.5})  # "C" doesn't exist in cov
    # should not raise; missing weight is treated as 0 exposure
    vol = portfolio_volatility(w, cov)
    assert vol >= 0
