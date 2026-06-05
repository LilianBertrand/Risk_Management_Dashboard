"""Market data loading and return preparation."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from .validation import validate_price_frame


@st.cache_data(show_spinner=False, ttl=3600)
def download_prices(tickers: tuple[str, ...], start, end) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = list(tickers)

    prices = prices.dropna(how="all").ffill().dropna()
    return validate_price_frame(prices)


@st.cache_data(show_spinner=False, ttl=3600)
def download_benchmark(benchmark: str, start, end) -> pd.Series:
    """Download adjusted close prices for a benchmark ticker."""
    data = yf.download(
        benchmark,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        series = data["Close"].iloc[:, 0]
    else:
        series = data["Close"]

    series = series.dropna().ffill()
    series.name = benchmark
    if series.empty or len(series) < 260:
        raise ValueError("Not enough benchmark data. Check benchmark ticker and dates.")
    return series


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute arithmetic daily returns."""
    return prices.pct_change().dropna()


def calculate_portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    """Compute portfolio returns from asset returns and portfolio weights."""
    portfolio_returns = returns @ weights
    portfolio_returns.name = "Portfolio"
    return portfolio_returns.dropna()


def align_series(*objects):
    """Align pandas Series/DataFrames on their common date index."""
    common_index = objects[0].index
    for obj in objects[1:]:
        common_index = common_index.intersection(obj.index)
    return [obj.loc[common_index] for obj in objects]
