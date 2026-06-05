# Portfolio Risk Management Dashboard

A professional Python / Streamlit dashboard for **market risk analytics**, designed for applications in **Risk Management**, **Portfolio Risk**, **Market Risk**, **Asset Management Risk** and **Quantitative Risk Analysis**.

The project analyses a multi-asset portfolio using historical market data and produces a full risk report: performance, downside risk, VaR, Expected Shortfall, VaR backtesting, risk contribution, benchmark-relative metrics and stress testing.

---

## Key Features

### Portfolio Analytics

- Market data download from Yahoo Finance
- User-defined portfolio tickers and weights
- Portfolio value evolution
- Benchmark comparison
- Cumulative active return

### Performance Metrics

- Annualized return
- CAGR
- Annualized volatility
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Best / worst daily return
- Positive-day ratio

### Downside Risk

- Maximum drawdown
- Rolling 30-day volatility
- Rolling 90-day volatility
- EWMA volatility using RiskMetrics-style decay
- Skewness and excess kurtosis diagnostics

### VaR and Tail Risk

- Historical VaR
- Gaussian parametric VaR
- Cornish-Fisher modified VaR
- Monte Carlo VaR
- Expected Shortfall
- Rolling VaR
- Rolling Expected Shortfall

### VaR Model Validation

- VaR exception tracking
- Observed exception rate
- Expected exception rate
- Kupiec Proportion of Failures test
- Backtesting chart with VaR breaches

### Risk Decomposition

- Asset-level risk contribution
- Component VaR approximation
- Herfindahl-Hirschman Index concentration metric
- Diversification ratio
- Correlation matrix

### Benchmark Risk

- Beta
- Alpha
- Tracking Error
- Information Ratio
- Benchmark correlation
- Rolling beta

### Stress Testing

- Hypothetical portfolio shocks
- Historical stress scenarios:
  - COVID Crash 2020
  - Rate Shock 2022
  - Tech Sell-off 2022
  - Banking Stress 2023

### Reporting

- Downloadable CSV risk report
- Downloadable Excel risk pack with multiple sheets

---

## Project Structure

```text
risk_management_dashboard_pro/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── sample_portfolios.csv
│
├── src/
│   └── risk_dashboard/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       ├── plotting.py
│       ├── reporting.py
│       ├── stress.py
│       └── validation.py
│
└── tests/
    └── test_metrics.py
```

This modular structure is more professional than a single script because each file has a clear responsibility:

- `data.py`: data download and return preparation
- `metrics.py`: risk and performance metrics
- `benchmark.py`: benchmark-relative risk metrics
- `stress.py`: stress testing logic
- `plotting.py`: Plotly chart builders
- `reporting.py`: CSV / Excel reporting utilities
- `validation.py`: input validation
- `app.py`: Streamlit user interface

---

## Installation

From the project folder:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

---

## Default Portfolio

The default portfolio is a US large-cap portfolio:

```text
AAPL, MSFT, GOOGL, AMZN, JPM
```

Weights:

```text
25%, 25%, 20%, 15%, 15%
```

Default benchmark:

```text
S&P 500 (^GSPC)
```

You can change all inputs directly in the Streamlit sidebar.

---

## Methodology

### Value-at-Risk

The dashboard includes four VaR methodologies:

1. **Historical VaR** — based on empirical historical returns.
2. **Gaussian VaR** — assumes normally distributed returns.
3. **Cornish-Fisher VaR** — adjusts the normal quantile for skewness and kurtosis.
4. **Monte Carlo VaR** — simulates portfolio-level daily returns.

### Expected Shortfall

Expected Shortfall measures the average loss beyond the VaR threshold. It is useful because VaR only gives a threshold, not the severity of losses beyond that threshold.

### VaR Backtesting

A VaR exception occurs when the realised portfolio return is lower than the estimated rolling VaR.

For example:

- 95% VaR implies an expected exception rate of 5%.
- 99% VaR implies an expected exception rate of 1%.

The dashboard also implements the **Kupiec Proportion of Failures test**, which statistically checks whether the observed number of exceptions is consistent with the theoretical confidence level.

### EWMA Volatility

The dashboard includes an EWMA volatility estimate inspired by the RiskMetrics approach. More recent returns receive more weight than older returns, making the risk estimate more reactive to market stress.

### Risk Contribution

The dashboard decomposes total portfolio volatility into asset-level risk contributions. This is useful because portfolio weights do not necessarily reflect actual risk exposure. A low-weight asset can still contribute heavily to risk if it is highly volatile or highly correlated with other positions.

---

## Interview Pitch

You can present the project like this:

> I developed a professional Python-based risk management dashboard to analyse the risk profile of a multi-asset portfolio. The tool downloads historical market data, computes portfolio returns and calculates key market risk indicators such as volatility, maximum drawdown, historical VaR, Gaussian VaR, Cornish-Fisher VaR, Monte Carlo VaR and Expected Shortfall.

Then add:

> I also implemented rolling risk indicators, EWMA volatility and VaR backtesting with a Kupiec test to validate whether the model remains reliable over time. Finally, I included benchmark-relative metrics, asset-level risk contribution and stress testing to identify the main sources of risk and assess portfolio resilience under adverse market conditions.

---

## CV Description

```text
Built a professional Python market risk dashboard including historical, Gaussian, Cornish-Fisher and Monte Carlo VaR, Expected Shortfall, EWMA volatility, VaR backtesting with Kupiec test, benchmark-relative risk metrics, stress testing and risk contribution decomposition.
```

Alternative version:

```text
Designed a modular Python risk monitoring framework to assess portfolio downside risk, validate VaR models, decompose asset-level risk contribution and evaluate portfolio resilience under historical and hypothetical stress scenarios.
```

---

## Limitations

This project is professional and interview-ready, but it remains an educational risk analytics tool. Main limitations:

- Yahoo Finance data is not institutional-grade.
- Monte Carlo simulation uses a simple normal portfolio-return assumption.
- No dynamic rebalancing is assumed.
- Transaction costs are not included.
- Liquidity risk is simplified.
- No interest-rate curve, credit spread curve or FX factor model is included.

---

## Possible Future Extensions

- Multi-asset correlated Monte Carlo simulation
- GARCH volatility model
- Factor-based stress testing
- Interest-rate duration / convexity module
- Credit spread shock module
- SQL database integration
- Bloomberg / Refinitiv connection
- PDF report generation
- Authentication and deployment on Streamlit Cloud
