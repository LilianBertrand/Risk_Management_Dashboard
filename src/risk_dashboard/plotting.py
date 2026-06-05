"""Plotly chart builders used by the Streamlit app."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def line_chart(df: pd.DataFrame, title: str, y_label: str):
    fig = px.line(df, title=title, labels={"value": y_label, "Date": "Date"})
    fig.update_layout(legend_title_text="")
    return fig


def drawdown_chart(drawdown):
    fig = px.area(drawdown, title="Portfolio Drawdown", labels={"value": "Drawdown", "Date": "Date"})
    fig.update_layout(showlegend=False)
    return fig


def var_distribution_chart(returns, hist_var, param_var, cf_var, mc_var):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=90, name="Daily Returns", histnorm="probability density"))
    for value, label, dash in [
        (hist_var, "Historical VaR", "dash"),
        (param_var, "Gaussian VaR", "dot"),
        (cf_var, "Cornish-Fisher VaR", "dashdot"),
        (mc_var, "Monte Carlo VaR", "longdash"),
    ]:
        fig.add_vline(x=value, line_dash=dash, annotation_text=label)
    fig.update_layout(title="Distribution of Daily Portfolio Returns", xaxis_title="Daily Return", yaxis_title="Density")
    return fig


def backtesting_chart(portfolio_returns, rolling_var, exceptions):
    df = pd.DataFrame({"Portfolio Returns": portfolio_returns, "Rolling VaR": rolling_var}).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Portfolio Returns"], mode="lines", name="Portfolio Returns"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Rolling VaR"], mode="lines", name="Rolling VaR"))
    exception_dates = exceptions[exceptions].index
    fig.add_trace(go.Scatter(x=exception_dates, y=portfolio_returns.loc[exception_dates], mode="markers", name="VaR Exceptions", marker={"size": 8}))
    fig.update_layout(title="VaR Backtesting", xaxis_title="Date", yaxis_title="Daily Return")
    return fig
