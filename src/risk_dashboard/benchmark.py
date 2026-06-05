"""Benchmark and relative-risk analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .metrics import annualized_return


def benchmark_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> dict:
    """Compute benchmark-relative performance and risk metrics."""
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    p = portfolio_returns.loc[common]
    b = benchmark_returns.loc[common]

    beta = p.cov(b) / b.var()
    p_ann = annualized_return(p)
    b_ann = annualized_return(b)
    alpha = p_ann - (risk_free_rate + beta * (b_ann - risk_free_rate))
    active = p - b
    tracking_error = active.std() * np.sqrt(TRADING_DAYS)
    information_ratio = (p_ann - b_ann) / tracking_error if tracking_error != 0 else np.nan

    return {
        "Beta": float(beta),
        "Alpha": float(alpha),
        "Tracking Error": float(tracking_error),
        "Information Ratio": float(information_ratio),
        "Benchmark Correlation": float(p.corr(b)),
        "Benchmark Annualized Return": float(b_ann),
        "Active Annualized Return": float(p_ann - b_ann),
    }


def rolling_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series, window: int) -> pd.Series:
    """Rolling beta versus benchmark."""
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    p = portfolio_returns.loc[common]
    b = benchmark_returns.loc[common]
    rolling_cov = p.rolling(window).cov(b)
    rolling_var = b.rolling(window).var()
    beta = rolling_cov / rolling_var
    beta.name = "Rolling Beta"
    return beta
