"""Input validation helpers."""

from __future__ import annotations

import numpy as np


def parse_tickers_and_weights(tickers_text: str, weights_text: str):
    """Parse and validate ticker and weight inputs."""
    tickers = [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]
    weights = np.array([float(w.strip()) for w in weights_text.split(",") if w.strip()], dtype=float)

    if not tickers:
        raise ValueError("At least one ticker is required.")
    if len(tickers) != len(weights):
        raise ValueError("The number of tickers must match the number of weights.")
    if np.any(weights < 0):
        raise ValueError("This dashboard currently supports long-only portfolios. Weights cannot be negative.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-5):
        raise ValueError(f"Portfolio weights must sum to 1. Current sum: {weights.sum():.4f}")

    return tickers, weights


def validate_price_frame(prices):
    """Basic sanity checks for downloaded price data."""
    if prices.empty:
        raise ValueError("No price data was downloaded. Check tickers and dates.")
    if prices.shape[0] < 260:
        raise ValueError("Not enough observations. Use a longer historical period, ideally at least one year.")
    return prices
