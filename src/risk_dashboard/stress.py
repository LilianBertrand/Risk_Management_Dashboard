"""Hypothetical, historical and custom stress-testing utilities."""

from __future__ import annotations

import pandas as pd


def hypothetical_stress_testing(portfolio_value: float) -> pd.DataFrame:
    """Simple portfolio-level shock scenarios."""
    scenarios = {
        "Mild correction (-5%)": -0.05,
        "Equity sell-off (-10%)": -0.10,
        "Liquidity shock (-15%)": -0.15,
        "Severe market crash (-20%)": -0.20,
        "Extreme crisis (-30%)": -0.30,
    }
    return custom_stress_testing(portfolio_value, scenarios)


def custom_stress_testing(portfolio_value: float, scenarios: dict[str, float]) -> pd.DataFrame:
    """Apply user-defined percentage shocks to portfolio value."""
    rows = []
    for name, shock in scenarios.items():
        impact = portfolio_value * shock
        rows.append({
            "Scenario": name,
            "Shock": shock,
            "Impact": impact,
            "Portfolio Value After Shock": portfolio_value + impact,
        })
    return pd.DataFrame(rows)


def historical_stress_testing(portfolio_returns, portfolio_value: float) -> pd.DataFrame:
    """Estimate portfolio performance during major historical stress windows."""
    scenarios = {
        "COVID Crash 2020": ("2020-02-19", "2020-03-23"),
        "Rate Shock 2022": ("2022-01-01", "2022-10-15"),
        "Tech Sell-off 2022": ("2021-11-15", "2022-06-15"),
        "Banking Stress 2023": ("2023-03-01", "2023-03-31"),
    }
    rows = []
    for name, (start, end) in scenarios.items():
        period = portfolio_returns.loc[
            (portfolio_returns.index >= pd.to_datetime(start))
            & (portfolio_returns.index <= pd.to_datetime(end))
        ]
        if not period.empty:
            cumulative_return = (1 + period).prod() - 1
            impact = portfolio_value * cumulative_return
            rows.append({
                "Scenario": name,
                "Start": start,
                "End": end,
                "Cumulative Return": cumulative_return,
                "Impact": impact,
                "Portfolio Value After Scenario": portfolio_value + impact,
            })
    return pd.DataFrame(rows)
