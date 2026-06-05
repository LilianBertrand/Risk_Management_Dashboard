"""Risk, performance and model validation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, norm, skew

from .config import TRADING_DAYS


def annualized_return(returns: pd.Series) -> float:
    """Annualized arithmetic return."""
    return float(returns.mean() * TRADING_DAYS)


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    cumulative = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or cumulative <= 0:
        return np.nan
    return cumulative ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized volatility."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def downside_volatility(returns: pd.Series, threshold: float = 0.0) -> float:
    """Annualized downside volatility using returns below a threshold."""
    downside = returns[returns < threshold]
    if downside.empty:
        return np.nan
    return float(downside.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Sharpe ratio based on annualized excess return."""
    vol = annualized_volatility(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (annualized_return(returns) - risk_free_rate) / vol


def sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Sortino ratio based on downside volatility."""
    dvol = downside_volatility(returns)
    if dvol == 0 or np.isnan(dvol):
        return np.nan
    return (annualized_return(returns) - risk_free_rate) / dvol


def max_drawdown(returns: pd.Series):
    """Maximum drawdown and drawdown time series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min()), drawdown


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio = CAGR / absolute maximum drawdown."""
    max_dd, _ = max_drawdown(returns)
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    return cagr(returns) / abs(max_dd)


def historical_var(returns: pd.Series, confidence: float) -> float:
    """Historical VaR using empirical quantiles."""
    return float(np.percentile(returns, (1 - confidence) * 100))


def parametric_var(returns: pd.Series, confidence: float) -> float:
    """Gaussian parametric VaR."""
    alpha = 1 - confidence
    return float(returns.mean() + returns.std() * norm.ppf(alpha))


def cornish_fisher_var(returns: pd.Series, confidence: float) -> float:
    """Cornish-Fisher modified VaR accounting for skewness and kurtosis."""
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    s = skew(returns, nan_policy="omit")
    k = kurtosis(returns, fisher=True, nan_policy="omit")
    z_cf = (
        z
        + (1 / 6) * (z**2 - 1) * s
        + (1 / 24) * (z**3 - 3 * z) * k
        - (1 / 36) * (2 * z**3 - 5 * z) * s**2
    )
    return float(returns.mean() + returns.std() * z_cf)


def expected_shortfall(returns: pd.Series, var_value: float) -> float:
    """Expected Shortfall / Conditional VaR."""
    tail = returns[returns <= var_value]
    return float(tail.mean()) if not tail.empty else np.nan


def monte_carlo_var(returns: pd.Series, confidence: float, n_simulations: int, seed: int = 42):
    """Portfolio-level Monte Carlo VaR using normal simulations."""
    rng = np.random.default_rng(seed)
    simulated = rng.normal(returns.mean(), returns.std(), int(n_simulations))
    var_value = float(np.percentile(simulated, (1 - confidence) * 100))
    es_value = float(simulated[simulated <= var_value].mean())
    return var_value, es_value, simulated


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """EWMA annualized volatility using RiskMetrics-style decay."""
    alpha = 1 - lambda_
    return returns.ewm(alpha=alpha, adjust=False).std() * np.sqrt(TRADING_DAYS)


def rolling_var(returns: pd.Series, confidence: float, window: int) -> pd.Series:
    """Rolling historical VaR."""
    return returns.rolling(window).quantile(1 - confidence)


def rolling_expected_shortfall(returns: pd.Series, confidence: float, window: int) -> pd.Series:
    """Rolling Expected Shortfall."""
    alpha = 1 - confidence

    def _es(x):
        var_value = np.percentile(x, alpha * 100)
        tail = x[x <= var_value]
        return tail.mean() if len(tail) else np.nan

    return returns.rolling(window).apply(_es, raw=False)


def var_backtesting(returns: pd.Series, var_series: pd.Series):
    """VaR backtesting exception count and exception rate."""
    aligned_var = var_series.dropna()
    aligned_returns = returns.loc[aligned_var.index]
    exceptions = aligned_returns < aligned_var
    n = len(aligned_returns)
    x = int(exceptions.sum())
    rate = x / n if n else np.nan
    return exceptions, x, rate, n


def kupiec_pof_test(n_exceptions: int, n_observations: int, expected_rate: float):
    """Kupiec proportion of failures test for VaR exception frequency."""
    if n_observations == 0:
        return np.nan, np.nan
    x = n_exceptions
    n = n_observations
    p = expected_rate
    phat = x / n
    eps = 1e-12
    phat = min(max(phat, eps), 1 - eps)
    p = min(max(p, eps), 1 - eps)
    lr = -2 * (
        (n - x) * np.log((1 - p) / (1 - phat))
        + x * np.log(p / phat)
    )
    p_value = 1 - chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def risk_contribution(asset_returns: pd.DataFrame, weights: np.ndarray):
    """Asset contribution to portfolio volatility."""
    cov = asset_returns.cov() * TRADING_DAYS
    portfolio_vol = float(np.sqrt(weights.T @ cov @ weights))
    marginal = cov @ weights / portfolio_vol
    absolute = weights * marginal
    percent = absolute / portfolio_vol
    return percent, portfolio_vol


def component_var(risk_contrib_pct, portfolio_var: float):
    """Approximate component VaR from risk contribution percentages."""
    return risk_contrib_pct * portfolio_var


def hhi_concentration(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman concentration index."""
    return float(np.sum(weights**2))


def diversification_ratio(asset_returns: pd.DataFrame, weights: np.ndarray) -> float:
    """Diversification ratio = weighted average vol / portfolio vol."""
    asset_vol = asset_returns.std() * np.sqrt(TRADING_DAYS)
    portfolio_vol = np.sqrt(weights.T @ (asset_returns.cov() * TRADING_DAYS) @ weights)
    if portfolio_vol == 0:
        return np.nan
    return float((weights @ asset_vol) / portfolio_vol)


def distribution_stats(returns: pd.Series) -> dict:
    """Distribution diagnostics for portfolio returns."""
    return {
        "Skewness": float(skew(returns, nan_policy="omit")),
        "Excess Kurtosis": float(kurtosis(returns, fisher=True, nan_policy="omit")),
        "Best Day": float(returns.max()),
        "Worst Day": float(returns.min()),
        "Positive Days": float((returns > 0).mean()),
    }
