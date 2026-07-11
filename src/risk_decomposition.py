"""
risk_decomposition.py
----------------------
Breaks total portfolio risk down into (a) per-instrument contributions and
(b) systematic factor contributions. This is the core "where is my risk
coming from" analysis a risk desk runs on every book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_metrics import portfolio_volatility


def marginal_contribution_to_risk(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """
    MCR_i = (Sigma w)_i / sigma_p
    Interpretation: the instantaneous change in portfolio vol per unit
    increase in asset i's weight. This is the building block for component
    risk — it answers "if I nudge this position, how much does total risk move?"
    """
    w = weights.reindex(cov.index).fillna(0.0)
    port_vol = portfolio_volatility(w, cov)
    if port_vol == 0:
        return pd.Series(0.0, index=cov.index)
    sigma_w = cov.values @ w.values
    return pd.Series(sigma_w / port_vol, index=cov.index)


def component_contribution_to_risk(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """
    CCR_i = w_i * MCR_i
    Component contributions sum exactly to total portfolio volatility (an
    exact Euler decomposition, since portfolio vol is homogeneous of degree 1
    in weights) — a useful sanity check and the reason this method is standard
    on risk desks over ad-hoc alternatives.
    """
    w = weights.reindex(cov.index).fillna(0.0)
    mcr = marginal_contribution_to_risk(w, cov)
    ccr = w * mcr
    port_vol = portfolio_volatility(w, cov)
    standalone_vol = np.sqrt(np.diag(cov))

    result = pd.DataFrame({
        "weight": w,
        "standalone_vol_annual": standalone_vol,
        "marginal_contribution": mcr,
        "component_contribution": ccr,
        "pct_of_total_risk": ccr / port_vol if port_vol > 0 else 0.0,
    })
    return result.sort_values("pct_of_total_risk", ascending=False)


def pca_factor_decomposition(returns: pd.DataFrame, weights: pd.Series,
                              n_factors: int = 3) -> pd.DataFrame:
    """
    Extracts the top `n_factors` principal components of the asset return
    covariance matrix — a data-driven stand-in for systematic risk factors
    (e.g. PC1 in a multi-asset book is almost always a "risk-on/risk-off"
    factor). Reports how much of total variance each factor explains and how
    exposed the current portfolio is to it.

    This is a lightweight, dependency-free alternative to a full Barra-style
    factor model — same idea (systematic vs idiosyncratic risk), simpler
    machinery, no external factor data required.
    """
    demeaned = returns - returns.mean()
    cov = demeaned.cov().values
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns ascending order; flip to descending (largest variance first)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    total_var = eigvals.sum()
    w = weights.reindex(returns.columns).fillna(0.0).values

    rows = []
    for i in range(min(n_factors, len(eigvals))):
        loading = float(w @ eigvecs[:, i])  # portfolio's exposure to factor i
        rows.append({
            "factor_id": i + 1,
            "explained_variance_ratio": float(eigvals[i] / total_var),
            "portfolio_loading": loading,
            "top_drivers": _top_drivers(eigvecs[:, i], returns.columns, k=3),
        })
    return pd.DataFrame(rows)


def _top_drivers(eigenvector: np.ndarray, tickers: pd.Index, k: int = 3) -> str:
    """Human-readable summary of which instruments dominate a principal component."""
    abs_weights = pd.Series(np.abs(eigenvector), index=tickers)
    top = abs_weights.sort_values(ascending=False).head(k)
    signed = eigenvector[[tickers.get_loc(t) for t in top.index]]
    return ", ".join(f"{t}({'+' if s >= 0 else '-'})" for t, s in zip(top.index, signed))
