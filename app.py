"""Professional Portfolio Risk Management Dashboard.

Run from the project root with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow imports from src/ when running with `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_dashboard.benchmark import benchmark_metrics, rolling_beta
from risk_dashboard.config import (
    DEFAULT_BENCHMARK,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_EWMA_LAMBDA,
    DEFAULT_MONTE_CARLO_SIMULATIONS,
    DEFAULT_PORTFOLIO_VALUE,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_START_DATE,
    DEFAULT_TICKERS,
    DEFAULT_WEIGHTS,
)
from risk_dashboard.data import (
    align_series,
    calculate_portfolio_returns,
    calculate_returns,
    download_benchmark,
    download_prices,
)
from risk_dashboard.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cagr,
    component_var,
    cornish_fisher_var,
    diversification_ratio,
    distribution_stats,
    ewma_volatility,
    expected_shortfall,
    hhi_concentration,
    historical_var,
    kupiec_pof_test,
    max_drawdown,
    monte_carlo_var,
    parametric_var,
    risk_contribution,
    rolling_expected_shortfall,
    rolling_var,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    var_backtesting,
)
from risk_dashboard.plotting import (
    backtesting_chart,
    drawdown_chart,
    line_chart,
    var_distribution_chart,
)
from risk_dashboard.reporting import create_excel_report, create_summary_report
from risk_dashboard.stress import historical_stress_testing, hypothetical_stress_testing
from risk_dashboard.validation import parse_tickers_and_weights


st.set_page_config(
    page_title="Risk Management Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Portfolio Risk Management Dashboard")
st.caption("Professional market risk analytics: VaR, Expected Shortfall, backtesting, benchmark risk, stress testing and concentration analysis.")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Portfolio Setup")
    tickers_input = st.text_input("Tickers", DEFAULT_TICKERS)
    weights_input = st.text_input("Weights", DEFAULT_WEIGHTS)
    benchmark_ticker = st.text_input("Benchmark", DEFAULT_BENCHMARK)

    st.divider()
    start_date = st.date_input("Start date", pd.to_datetime(DEFAULT_START_DATE))
    end_date = st.date_input("End date", pd.to_datetime("today"))
    portfolio_value = st.number_input("Portfolio value", min_value=1000, value=DEFAULT_PORTFOLIO_VALUE, step=10000)

    st.divider()
    confidence_level = st.selectbox("VaR confidence level", [0.95, 0.99], index=0 if DEFAULT_CONFIDENCE_LEVEL == 0.95 else 1)
    risk_free_rate = st.number_input("Annual risk-free rate", min_value=0.0, max_value=0.25, value=DEFAULT_RISK_FREE_RATE, step=0.005, format="%.3f")
    rolling_window = st.number_input("Rolling window", min_value=60, max_value=750, value=DEFAULT_ROLLING_WINDOW, step=10)
    mc_simulations = st.number_input("Monte Carlo simulations", min_value=1000, max_value=100000, value=DEFAULT_MONTE_CARLO_SIMULATIONS, step=1000)
    ewma_lambda = st.slider("EWMA lambda", min_value=0.80, max_value=0.99, value=DEFAULT_EWMA_LAMBDA, step=0.01)


# -----------------------------------------------------------------------------
# Computation
# -----------------------------------------------------------------------------

try:
    tickers, weights = parse_tickers_and_weights(tickers_input, weights_input)

    with st.spinner("Downloading market data and computing risk analytics..."):
        prices = download_prices(tuple(tickers), start_date, end_date)
        asset_returns = calculate_returns(prices)
        portfolio_returns = calculate_portfolio_returns(asset_returns, weights)

        benchmark_prices = download_benchmark(benchmark_ticker, start_date, end_date)
        benchmark_returns = benchmark_prices.pct_change().dropna()

        portfolio_returns, benchmark_returns, asset_returns = align_series(portfolio_returns, benchmark_returns, asset_returns)

        cumulative_portfolio = (1 + portfolio_returns).cumprod() * portfolio_value
        cumulative_benchmark = (1 + benchmark_returns).cumprod() * portfolio_value

        ann_return = annualized_return(portfolio_returns)
        ann_vol = annualized_volatility(portfolio_returns)
        portfolio_cagr = cagr(portfolio_returns)
        sharpe = sharpe_ratio(portfolio_returns, risk_free_rate)
        sortino = sortino_ratio(portfolio_returns, risk_free_rate)
        max_dd, drawdown = max_drawdown(portfolio_returns)
        calmar = calmar_ratio(portfolio_returns)

        hist_var = historical_var(portfolio_returns, confidence_level)
        gaussian_var = parametric_var(portfolio_returns, confidence_level)
        cf_var = cornish_fisher_var(portfolio_returns, confidence_level)
        hist_es = expected_shortfall(portfolio_returns, hist_var)
        mc_var, mc_es, simulated_returns = monte_carlo_var(portfolio_returns, confidence_level, mc_simulations)

        rolling_vol_30 = rolling_volatility(portfolio_returns, 30)
        rolling_vol_90 = rolling_volatility(portfolio_returns, 90)
        ewma_vol = ewma_volatility(portfolio_returns, ewma_lambda)
        rolling_var_series = rolling_var(portfolio_returns, confidence_level, rolling_window)
        rolling_es_series = rolling_expected_shortfall(portfolio_returns, confidence_level, rolling_window)

        exceptions, n_exceptions, exception_rate, n_backtest_obs = var_backtesting(portfolio_returns, rolling_var_series)
        kupiec_lr, kupiec_pvalue = kupiec_pof_test(n_exceptions, n_backtest_obs, 1 - confidence_level)

        risk_contrib_pct, portfolio_vol_from_cov = risk_contribution(asset_returns, weights)
        component_var_values = component_var(risk_contrib_pct, hist_var)
        hhi = hhi_concentration(weights)
        div_ratio = diversification_ratio(asset_returns, weights)
        dist_stats = distribution_stats(portfolio_returns)

        benchmark_results = benchmark_metrics(portfolio_returns, benchmark_returns, risk_free_rate)
        rolling_beta_series = rolling_beta(portfolio_returns, benchmark_returns, rolling_window)

        simple_stress_df = hypothetical_stress_testing(portfolio_value)
        historical_stress_df = historical_stress_testing(portfolio_returns, portfolio_value)

except Exception as exc:
    st.error(f"Unable to run the dashboard: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Executive Summary
# -----------------------------------------------------------------------------

st.subheader("Executive Risk Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annualized Return", f"{ann_return:.2%}")
col2.metric("Annualized Volatility", f"{ann_vol:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
col4.metric("Maximum Drawdown", f"{max_dd:.2%}")

col5, col6, col7, col8 = st.columns(4)
col5.metric(f"Historical VaR {int(confidence_level*100)}%", f"{hist_var:.2%}")
col6.metric(f"Expected Shortfall {int(confidence_level*100)}%", f"{hist_es:.2%}")
col7.metric("Portfolio Beta", f"{benchmark_results['Beta']:.2f}")
col8.metric("Tracking Error", f"{benchmark_results['Tracking Error']:.2%}")

if exception_rate > (1 - confidence_level) * 1.5:
    st.warning("VaR backtesting indicates more exceptions than expected. The model may underestimate tail risk.")
elif exception_rate < (1 - confidence_level) * 0.5:
    st.info("VaR backtesting indicates fewer exceptions than expected. The model may be conservative.")
else:
    st.success("VaR backtesting is broadly consistent with the selected confidence level.")


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tabs = st.tabs([
    "Performance",
    "Downside Risk",
    "VaR & Backtesting",
    "Risk Decomposition",
    "Benchmark",
    "Stress Testing",
    "Report",
])

with tabs[0]:
    st.subheader("Performance and Portfolio Value")
    comparison_df = pd.DataFrame({"Portfolio": cumulative_portfolio, "Benchmark": cumulative_benchmark}).dropna()
    st.plotly_chart(line_chart(comparison_df, "Portfolio Value vs Benchmark", "Value"), use_container_width=True)

    perf_table = pd.DataFrame({
        "Metric": ["Annualized Return", "CAGR", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Best Day", "Worst Day", "Positive Days"],
        "Value": [ann_return, portfolio_cagr, ann_vol, sharpe, sortino, calmar, dist_stats["Best Day"], dist_stats["Worst Day"], dist_stats["Positive Days"]],
    })
    st.dataframe(perf_table, use_container_width=True)

with tabs[1]:
    st.subheader("Downside Risk and Rolling Risk Indicators")
    st.plotly_chart(drawdown_chart(drawdown), use_container_width=True)

    rolling_vol_df = pd.DataFrame({
        "30D Rolling Volatility": rolling_vol_30,
        "90D Rolling Volatility": rolling_vol_90,
        "EWMA Volatility": ewma_vol,
    }).dropna()
    st.plotly_chart(line_chart(rolling_vol_df, "Rolling and EWMA Annualized Volatility", "Volatility"), use_container_width=True)

    st.markdown("**Distribution diagnostics**")
    st.dataframe(pd.DataFrame(dist_stats.items(), columns=["Metric", "Value"]), use_container_width=True)

with tabs[2]:
    st.subheader("VaR, Expected Shortfall and Model Validation")
    var_table = pd.DataFrame({
        "Metric": [
            "Historical VaR", "Gaussian VaR", "Cornish-Fisher VaR", "Monte Carlo VaR",
            "Historical Expected Shortfall", "Monte Carlo Expected Shortfall"
        ],
        "Value": [hist_var, gaussian_var, cf_var, mc_var, hist_es, mc_es]
    })
    st.dataframe(var_table, use_container_width=True)
    st.plotly_chart(var_distribution_chart(portfolio_returns, hist_var, gaussian_var, cf_var, mc_var), use_container_width=True)

    rolling_tail_df = pd.DataFrame({"Rolling VaR": rolling_var_series, "Rolling Expected Shortfall": rolling_es_series}).dropna()
    st.plotly_chart(line_chart(rolling_tail_df, f"Rolling VaR and Expected Shortfall - {rolling_window}D Window", "Daily Return"), use_container_width=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Backtest Observations", f"{n_backtest_obs}")
    col_b.metric("VaR Exceptions", f"{n_exceptions}")
    col_c.metric("Observed Exception Rate", f"{exception_rate:.2%}")
    col_d.metric("Kupiec p-value", f"{kupiec_pvalue:.3f}" if pd.notna(kupiec_pvalue) else "n/a")
    st.plotly_chart(backtesting_chart(portfolio_returns, rolling_var_series, exceptions), use_container_width=True)

with tabs[3]:
    st.subheader("Risk Decomposition and Concentration")
    rc_df = pd.DataFrame({
        "Ticker": tickers,
        "Portfolio Weight": weights,
        "Risk Contribution": risk_contrib_pct,
        "Component VaR": component_var_values,
    })
    st.dataframe(rc_df.style.format({"Portfolio Weight": "{:.2%}", "Risk Contribution": "{:.2%}", "Component VaR": "{:.2%}"}), use_container_width=True)
    st.plotly_chart(px.bar(rc_df, x="Ticker", y="Risk Contribution", text_auto=".2%", title="Asset-Level Risk Contribution"), use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("HHI Concentration", f"{hhi:.3f}")
    col_b.metric("Diversification Ratio", f"{div_ratio:.2f}")
    col_c.metric("Portfolio Vol from Covariance", f"{portfolio_vol_from_cov:.2%}")

    st.plotly_chart(px.imshow(asset_returns.corr(), text_auto=".2f", title="Asset Correlation Matrix", aspect="auto"), use_container_width=True)

with tabs[4]:
    st.subheader("Benchmark and Relative Risk")
    benchmark_table = pd.DataFrame(benchmark_results.items(), columns=["Metric", "Value"])
    st.dataframe(benchmark_table, use_container_width=True)
    st.plotly_chart(line_chart(pd.DataFrame({"Rolling Beta": rolling_beta_series}).dropna(), f"Rolling Beta vs {benchmark_ticker}", "Beta"), use_container_width=True)

    active_returns = portfolio_returns - benchmark_returns
    st.plotly_chart(line_chart(pd.DataFrame({"Active Return Cumulative": (1 + active_returns).cumprod() - 1}), "Cumulative Active Return", "Active Return"), use_container_width=True)

with tabs[5]:
    st.subheader("Stress Testing")
    st.markdown("Hypothetical scenarios apply direct portfolio-level shocks. Historical scenarios replay selected market stress windows on the current portfolio composition.")

    st.dataframe(simple_stress_df.style.format({"Shock": "{:.2%}", "Impact": "€{:,.0f}", "Portfolio Value After Shock": "€{:,.0f}"}), use_container_width=True)
    st.plotly_chart(px.bar(simple_stress_df, x="Scenario", y="Impact", text_auto=".0f", title="Hypothetical Stress Impact"), use_container_width=True)

    if not historical_stress_df.empty:
        st.dataframe(historical_stress_df.style.format({"Cumulative Return": "{:.2%}", "Impact": "€{:,.0f}", "Portfolio Value After Scenario": "€{:,.0f}"}), use_container_width=True)
        st.plotly_chart(px.bar(historical_stress_df, x="Scenario", y="Impact", text_auto=".0f", title="Historical Stress Impact"), use_container_width=True)
    else:
        st.info("Not enough historical data for predefined stress windows. Use an earlier start date.")

with tabs[6]:
    st.subheader("Exportable Risk Report")

    summary_metrics = {
        "Annualized Return": ann_return,
        "CAGR": portfolio_cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Maximum Drawdown": max_dd,
        "Historical VaR": hist_var,
        "Gaussian VaR": gaussian_var,
        "Cornish-Fisher VaR": cf_var,
        "Monte Carlo VaR": mc_var,
        "Expected Shortfall": hist_es,
        "Portfolio Beta": benchmark_results["Beta"],
        "Tracking Error": benchmark_results["Tracking Error"],
        "Information Ratio": benchmark_results["Information Ratio"],
        "HHI Concentration": hhi,
        "Diversification Ratio": div_ratio,
        "VaR Exceptions": n_exceptions,
        "Observed Exception Rate": exception_rate,
        "Kupiec LR Statistic": kupiec_lr,
        "Kupiec p-value": kupiec_pvalue,
    }
    report_df = create_summary_report(summary_metrics)
    weights_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})

    st.dataframe(report_df, use_container_width=True)

    csv_report = report_df.to_csv(index=False).encode("utf-8")
    excel_report = create_excel_report({
        "Summary": report_df,
        "Weights": weights_df,
        "Risk Contribution": rc_df,
        "VaR Table": var_table,
        "Benchmark": pd.DataFrame(benchmark_results.items(), columns=["Metric", "Value"]),
        "Hypothetical Stress": simple_stress_df,
        "Historical Stress": historical_stress_df,
    })

    col_a, col_b = st.columns(2)
    col_a.download_button("Download CSV Risk Report", data=csv_report, file_name="risk_report.csv", mime="text/csv")
    col_b.download_button("Download Excel Risk Pack", data=excel_report, file_name="risk_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with st.expander("Raw data"):
        st.dataframe(prices.tail(20), use_container_width=True)
        st.dataframe(asset_returns.tail(20), use_container_width=True)
