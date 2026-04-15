import streamlit as st
import numpy as np
import pandas as pd
from datetime import date

from src.data.loader import download_data, get_returns
from src.analysis.returns import compute_summary_stats, compute_wealth_index
from src.analysis.risk import (
    compute_rolling_volatility, compute_drawdown,
    compute_max_drawdown, compute_risk_metrics
)
from src.analysis.correlation import (
    compute_correlation, compute_covariance, compute_rolling_correlation
)
from src.analysis.optimization import (
    compute_gmv, compute_tangency, compute_efficient_frontier,
    compute_risk_contribution, portfolio_metrics, portfolio_performance
)
from src.components.charts import (
    plot_wealth_index, plot_histogram, plot_qq,
    plot_rolling_volatility, plot_drawdown,
    plot_correlation_heatmap, plot_rolling_correlation,
    plot_weights, plot_risk_contribution,
    plot_efficient_frontier, plot_portfolio_comparison
)
from src.components.tables import (
    format_summary_stats, format_risk_metrics,
    format_metrics_dict, format_comparison_table
)

st.set_page_config(page_title="Portfolio Analytics", page_icon="📊", layout="wide")

with st.sidebar:
    st.title("📊 Portfolio Analytics")
    st.markdown("---")
    st.subheader("1. Select Stocks")
    ticker_input = st.text_input("Enter 3–10 ticker symbols (comma-separated):", value="AAPL, MSFT, JPM, KO, CVX")
    st.subheader("2. Date Range")
    start_date = st.date_input("Start Date", value=date(2019, 1, 1))
    end_date   = st.date_input("End Date",   value=date.today())
    st.subheader("3. Risk-Free Rate")
    rf_rate = st.number_input("Annualized Risk-Free Rate (%):", min_value=0.0, max_value=20.0, value=2.0, step=0.1) / 100
    st.markdown("---")
    run_button = st.button("🚀 Run Analysis", use_container_width=True)
    with st.expander("ℹ️ About & Methodology"):
        st.markdown("""
        **Data Source:** Yahoo Finance via `yfinance`
        **Key Assumptions:**
        - Simple (arithmetic) returns
        - Annualization factor: 252 trading days
        - Risk-free rate: user-defined (annualized)
        - No short-selling constraints in optimization
        **Methods:**
        - Mean-Variance Optimization (Markowitz)
        - Global Minimum Variance (GMV) Portfolio
        - Maximum Sharpe Ratio (Tangency) Portfolio
        - Efficient Frontier via constrained optimization
        - Risk Contribution Decomposition
        """)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
errors  = []
if len(tickers) < 3:
    errors.append("⚠️ Please enter **at least 3** ticker symbols.")
if len(tickers) > 10:
    errors.append("⚠️ Please enter **no more than 10** ticker symbols.")
if (end_date - start_date).days < 365 * 2:
    errors.append("⚠️ Date range must be **at least 2 years**.")
if start_date >= end_date:
    errors.append("⚠️ Start date must be **before** end date.")
if errors:
    for e in errors:
        st.error(e)
    st.stop()

if run_button or "data" in st.session_state:
    if run_button:
        with st.spinner("📥 Downloading data from Yahoo Finance..."):
            data, valid_tickers, warnings = download_data(tickers, str(start_date), str(end_date))
            st.session_state["data"]          = data
            st.session_state["valid_tickers"] = valid_tickers
            st.session_state["warnings"]      = warnings
            st.session_state["rf_rate"]       = rf_rate

    data          = st.session_state.get("data")
    valid_tickers = st.session_state.get("valid_tickers", [])
    warnings      = st.session_state.get("warnings", [])
    rf_rate       = st.session_state.get("rf_rate", rf_rate)

    for w in warnings:
        st.warning(w)

    if data is None or data.empty:
        st.error("❌ No data available. Please check your tickers and date range.")
        st.stop()
    if len(valid_tickers) < 3:
        st.error("❌ Fewer than 3 valid tickers remain. Please add more tickers.")
        st.stop()

    returns       = get_returns(data)
    stock_returns = returns[valid_tickers]
    bench_returns = returns["^GSPC"]
    all_returns   = returns[valid_tickers + ["^GSPC"]]

    mean_ret  = stock_returns.mean() * 252
    cov_mat   = stock_returns.cov()  * 252
    n_assets  = len(valid_tickers)

    ew_weights = np.ones(n_assets) / n_assets

    gmv_weights, gmv_err = compute_gmv(mean_ret, cov_mat)
    if gmv_err:
        st.error(f"❌ {gmv_err}")
        st.stop()

    tan_weights, tan_err = compute_tangency(mean_ret, cov_mat, rf_rate)
    if tan_err:
        st.error(f"❌ {tan_err}")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Returns & Exploratory",
        "⚠️ Risk Analysis",
        "🔗 Correlation & Covariance",
        "💼 Portfolio Optimization",
        "🔍 Sensitivity Analysis"
    ])

    # ================================================================
    # TAB 1
    # ================================================================
    with tab1:
        st.header("📈 Returns & Exploratory Analysis")

        st.subheader("Summary Statistics")
        stats = compute_summary_stats(all_returns)
        st.dataframe(format_summary_stats(stats), use_container_width=True)

        st.subheader("Annual Extremes")
        ann_ret = all_returns.mean() * 252
        ann_vol = all_returns.std() * np.sqrt(252)
        extremes = pd.DataFrame({
            "Statistic": ["Max Annual Return", "Min Annual Return", "Max Volatility",   "Min Volatility"],
            "Ticker":    [ann_ret.idxmax(),     ann_ret.idxmin(),    ann_vol.idxmax(),   ann_vol.idxmin()],
            "Value":     [f"{ann_ret.max():.2%}",f"{ann_ret.min():.2%}",f"{ann_vol.max():.2%}",f"{ann_vol.min():.2%}"]
        })
        st.dataframe(extremes, use_container_width=True, hide_index=True)

        st.subheader("📥 Download Data")
        csv = data.to_csv().encode("utf-8")
        st.download_button(label="Download Prices as CSV", data=csv, file_name="portfolio_data.csv", mime="text/csv")

        st.markdown("---")

        st.subheader("Cumulative Wealth Index — Growth of $10,000")
        all_cols = valid_tickers + ["^GSPC"]
        selected = st.multiselect("Select assets to display:", options=all_cols, default=all_cols)
        wealth   = compute_wealth_index(all_returns)
        st.plotly_chart(plot_wealth_index(wealth, selected), use_container_width=True)

        st.markdown("---")

        st.subheader("Return Distribution")
        col1, col2 = st.columns([1, 2])
        with col1:
            dist_ticker = st.selectbox("Select stock:", valid_tickers, key="dist_ticker")
            plot_type   = st.radio("Plot type:", ["Histogram", "Q-Q Plot"], key="dist_type")
        with col2:
            if plot_type == "Histogram":
                st.plotly_chart(plot_histogram(stock_returns[dist_ticker], dist_ticker), use_container_width=True)
            else:
                st.plotly_chart(plot_qq(stock_returns[dist_ticker], dist_ticker), use_container_width=True)

    # ================================================================
    # TAB 2
    # ================================================================
    with tab2:
        st.header("⚠️ Risk Analysis")

        st.subheader("Rolling Volatility")
        window = st.select_slider("Rolling window (days):", options=[20, 30, 60, 90, 120], value=30, key="vol_window")
        rolling_vol = compute_rolling_volatility(all_returns, window)
        st.plotly_chart(plot_rolling_volatility(rolling_vol), use_container_width=True)

        st.markdown("---")

        st.subheader("Drawdown Analysis")
        dd_ticker   = st.selectbox("Select stock:", valid_tickers, key="dd_ticker")
        dd_series   = compute_drawdown(stock_returns[dd_ticker])
        max_dd_val  = compute_max_drawdown(stock_returns[dd_ticker])
        max_dd_date = dd_series.idxmin().date()

        col_a, col_b = st.columns(2)
        col_a.metric(label=f"Maximum Drawdown — {dd_ticker}", value=f"{max_dd_val:.2%}")
        col_b.metric(label="Date of Maximum Drawdown",        value=str(max_dd_date))
        st.plotly_chart(plot_drawdown(dd_series, dd_ticker), use_container_width=True)

        st.markdown("---")

        st.subheader("Risk-Adjusted Performance Metrics")
        risk_metrics = compute_risk_metrics(all_returns, rf=rf_rate)
        st.dataframe(format_risk_metrics(risk_metrics), use_container_width=True)

    # ================================================================
    # TAB 3
    # ================================================================
    with tab3:
        st.header("🔗 Correlation & Covariance Analysis")

        st.subheader("Correlation Heatmap")
        corr = compute_correlation(all_returns)
        st.plotly_chart(plot_correlation_heatmap(corr), use_container_width=True)

        corr_pairs = corr.unstack()
        corr_pairs = corr_pairs[
            corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)
        ]
        corr_pairs = corr_pairs[
            corr_pairs.index.get_level_values(0) < corr_pairs.index.get_level_values(1)
        ]
        most_pos = corr_pairs.idxmax()
        most_neg = corr_pairs.idxmin()
        col1, col2 = st.columns(2)
        col1.success(f"**Most Positively Correlated:** {most_pos[0]} & {most_pos[1]} → {corr_pairs.max():.2f}")
        col2.warning(f"**Least Correlated:** {most_neg[0]} & {most_neg[1]} → {corr_pairs.min():.2f}")

        st.markdown("---")

        st.subheader("Covariance Verification")
        cov     = compute_covariance(all_returns)
        std_dev = all_returns.std()
        if len(valid_tickers) >= 2:
            t_a, t_b    = valid_tickers[0], valid_tickers[1]
            cov_formula = corr.loc[t_a, t_b] * std_dev[t_a] * std_dev[t_b]
            cov_actual  = cov.loc[t_a, t_b]
            st.markdown(f"""
            Verifying covariance between **{t_a}** and **{t_b}**:
            - **Formula** `Corr × σ_A × σ_B` = `{cov_formula:.8f}`
            - **Direct**  `Cov(A,B)`          = `{cov_actual:.8f}`
            - **Match:** `{np.isclose(cov_formula, cov_actual)}`
            """)

        st.markdown("---")

        st.subheader("Rolling Correlation")
        all_tickers_list = valid_tickers + ["^GSPC"]
        col1, col2, col3 = st.columns(3)
        with col1:
            t1 = st.selectbox("Ticker 1:", all_tickers_list, index=0, key="rc_t1")
        with col2:
            t2 = st.selectbox("Ticker 2:", all_tickers_list, index=1, key="rc_t2")
        with col3:
            rc_window = st.select_slider("Window (days):", options=[20, 30, 60, 90, 120], value=30, key="rc_window")
        if t1 == t2:
            st.warning("Please select two different tickers.")
        else:
            roll_corr = compute_rolling_correlation(all_returns, t1, t2, rc_window)
            st.plotly_chart(plot_rolling_correlation(roll_corr, t1, t2), use_container_width=True)

        st.markdown("---")

        with st.expander("📊 View Full Covariance Matrix"):
            st.dataframe(cov.style.format("{:.6f}"), use_container_width=True)

    # ================================================================
    # TAB 4
    # ================================================================
    with tab4:
        st.header("💼 Portfolio Optimization")

        st.subheader("🎛️ Custom Portfolio")
        st.caption("Adjust sliders to set your preferred allocation. Weights are automatically normalized.")
        raw_weights = []
        slider_cols = st.columns(n_assets)
        for i, ticker in enumerate(valid_tickers):
            with slider_cols[i]:
                w = st.slider(ticker, 0.0, 1.0, 1.0 / n_assets, 0.01, key=f"w_{ticker}")
                raw_weights.append(w)
        raw_sum = sum(raw_weights)
        custom_weights = np.array(raw_weights) / raw_sum if raw_sum > 0 else np.ones(n_assets) / n_assets

        st.markdown("**Normalized Weights:**")
        weight_display = pd.DataFrame([custom_weights], columns=valid_tickers).map(lambda x: f"{x:.1%}")
        st.dataframe(weight_display, use_container_width=True)

        st.markdown("---")

        ew_metrics   = portfolio_metrics(ew_weights,     stock_returns, rf_rate)
        gmv_metrics  = portfolio_metrics(gmv_weights,    stock_returns, rf_rate)
        tan_metrics  = portfolio_metrics(tan_weights,    stock_returns, rf_rate)
        cust_metrics = portfolio_metrics(custom_weights, stock_returns, rf_rate)
        bench_metrics= portfolio_metrics(np.ones(1), bench_returns.to_frame(), rf_rate)

        def get_max_dd_date(weights, returns_df):
            pr = returns_df.dot(weights)
            w  = (1 + pr).cumprod()
            dd = (w - w.cummax()) / w.cummax()
            return str(dd.idxmin().date())

        # --- Equal Weight ---
        st.subheader("⚖️ Equal-Weight Portfolio (1/N)")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Annual Return",     f"{ew_metrics['Annual Return']:.2%}")
        c2.metric("Annual Volatility", f"{ew_metrics['Annual Volatility']:.2%}")
        c3.metric("Sharpe Ratio",      f"{ew_metrics['Sharpe Ratio']:.2f}")
        c4.metric("Sortino Ratio",     f"{ew_metrics['Sortino Ratio']:.2f}")
        c5.metric("Max Drawdown",      f"{ew_metrics['Max Drawdown']:.2%}")
        c6.metric("Max DD Date",       get_max_dd_date(ew_weights, stock_returns))

        st.markdown("---")

        # --- GMV ---
        st.subheader("🛡️ Global Minimum Variance (GMV) Portfolio")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Annual Return",     f"{gmv_metrics['Annual Return']:.2%}")
        c2.metric("Annual Volatility", f"{gmv_metrics['Annual Volatility']:.2%}")
        c3.metric("Sharpe Ratio",      f"{gmv_metrics['Sharpe Ratio']:.2f}")
        c4.metric("Sortino Ratio",     f"{gmv_metrics['Sortino Ratio']:.2f}")
        c5.metric("Max Drawdown",      f"{gmv_metrics['Max Drawdown']:.2%}")
        c6.metric("Max DD Date",       get_max_dd_date(gmv_weights, stock_returns))

        gmv_weights_df = pd.DataFrame({"Asset": valid_tickers, "Weight": [f"{w:.2%}" for w in gmv_weights]})
        col_w, col_r = st.columns(2)
        with col_w:
            st.plotly_chart(plot_weights(gmv_weights, valid_tickers, "GMV Weights"), use_container_width=True)
            st.dataframe(gmv_weights_df, use_container_width=True, hide_index=True)
        with col_r:
            gmv_prc = compute_risk_contribution(gmv_weights, cov_mat)
            st.plotly_chart(plot_risk_contribution(gmv_prc, valid_tickers, "GMV Risk Contribution"), use_container_width=True)
        st.info("**Risk Contribution:** Each bar shows what fraction of total portfolio variance comes from that asset.")

        st.markdown("---")

        # --- Tangency ---
        st.subheader("🎯 Maximum Sharpe Ratio (Tangency) Portfolio")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Annual Return",     f"{tan_metrics['Annual Return']:.2%}")
        c2.metric("Annual Volatility", f"{tan_metrics['Annual Volatility']:.2%}")
        c3.metric("Sharpe Ratio",      f"{tan_metrics['Sharpe Ratio']:.2f}")
        c4.metric("Sortino Ratio",     f"{tan_metrics['Sortino Ratio']:.2f}")
        c5.metric("Max Drawdown",      f"{tan_metrics['Max Drawdown']:.2%}")
        c6.metric("Max DD Date",       get_max_dd_date(tan_weights, stock_returns))

        tan_weights_df = pd.DataFrame({"Asset": valid_tickers, "Weight": [f"{w:.2%}" for w in tan_weights]})
        col_w, col_r = st.columns(2)
        with col_w:
            st.plotly_chart(plot_weights(tan_weights, valid_tickers, "Tangency Weights"), use_container_width=True)
            st.dataframe(tan_weights_df, use_container_width=True, hide_index=True)
        with col_r:
            tan_prc = compute_risk_contribution(tan_weights, cov_mat)
            st.plotly_chart(plot_risk_contribution(tan_prc, valid_tickers, "Tangency Risk Contribution"), use_container_width=True)
        st.info("**Risk Contribution:** Each bar shows what fraction of total portfolio variance comes from that asset.")

        st.markdown("---")

        # --- Efficient Frontier ---
        st.subheader("📉 Efficient Frontier")
        st.markdown("""
        The **Efficient Frontier** shows portfolios that offer the highest expected return for each level of risk.
        The **Capital Allocation Line (CAL)** connects the risk-free rate to the Tangency portfolio.
        """)
        with st.spinner("Computing efficient frontier..."):
            frontier_vols, frontier_returns = compute_efficient_frontier(mean_ret, cov_mat)

        ew_ret,   ew_vol   = portfolio_performance(ew_weights,     mean_ret, cov_mat)
        gmv_ret,  gmv_vol  = portfolio_performance(gmv_weights,    mean_ret, cov_mat)
        tan_ret,  tan_vol  = portfolio_performance(tan_weights,    mean_ret, cov_mat)
        cust_ret, cust_vol = portfolio_performance(custom_weights, mean_ret, cov_mat)
        bench_ret = bench_returns.mean() * 252
        bench_vol = bench_returns.std()  * np.sqrt(252)

        stock_vols = [np.sqrt(cov_mat.loc[t, t]) for t in valid_tickers]
        stock_rets = [mean_ret[t]                 for t in valid_tickers]

        ef_fig = plot_efficient_frontier(
            frontier_vols=frontier_vols, frontier_returns=frontier_returns,
            gmv=      {"return": gmv_ret,  "vol": gmv_vol},
            tangency= {"return": tan_ret,  "vol": tan_vol},
            ew=       {"return": ew_ret,   "vol": ew_vol},
            custom=   {"return": cust_ret, "vol": cust_vol},
            stock_vols=stock_vols, stock_returns=stock_rets, stock_labels=valid_tickers,
            rf=rf_rate,
            sp500={"return": bench_ret, "vol": bench_vol}
        )
        st.plotly_chart(ef_fig, use_container_width=True)

        st.markdown("---")

        # --- Portfolio Comparison ---
        st.subheader("📊 Portfolio Comparison")
        ew_wealth    = (1 + stock_returns.dot(ew_weights)).cumprod()
        gmv_wealth   = (1 + stock_returns.dot(gmv_weights)).cumprod()
        tan_wealth   = (1 + stock_returns.dot(tan_weights)).cumprod()
        cust_wealth  = (1 + stock_returns.dot(custom_weights)).cumprod()
        bench_wealth = (1 + bench_returns).cumprod()

        wealth_comparison = pd.DataFrame({
            "Equal Weight":  ew_wealth,
            "GMV Portfolio": gmv_wealth,
            "Tangency":      tan_wealth,
            "Custom":        cust_wealth,
            "S&P 500":       bench_wealth
        })
        st.plotly_chart(plot_portfolio_comparison(wealth_comparison), use_container_width=True)

        comparison = format_comparison_table({
            "Equal Weight":  ew_metrics,
            "GMV Portfolio": gmv_metrics,
            "Tangency":      tan_metrics,
            "Custom":        cust_metrics,
            "S&P 500":       bench_metrics
        })
        st.dataframe(comparison, use_container_width=True)

    # ================================================================
    # TAB 5
    # ================================================================
    with tab5:
        st.header("🔍 Estimation Window Sensitivity Analysis")
        st.markdown("""
        Mean-variance optimization is highly sensitive to its inputs.
        This section shows how GMV and Tangency weights change across different lookback windows.
        """)

        total_days   = len(stock_returns)
        all_windows  = {"1 Year": 252, "3 Years": 756, "5 Years": 1260, "Full Sample": total_days}
        avail_windows= {k: v for k, v in all_windows.items() if v <= total_days}

        if len(avail_windows) < 2:
            st.warning("Need at least 2 years of data for sensitivity analysis.")
        else:
            selected_windows = st.multiselect("Select lookback windows to compare:", options=list(avail_windows.keys()), default=list(avail_windows.keys()))

            if selected_windows:
                with st.spinner("Running sensitivity analysis..."):
                    gmv_results = {}
                    tan_results = {}
                    for window_name in selected_windows:
                        n_days   = avail_windows[window_name]
                        subset   = stock_returns.iloc[-n_days:]
                        sub_mean = subset.mean() * 252
                        sub_cov  = subset.cov()  * 252
                        gw, gerr = compute_gmv(sub_mean, sub_cov)
                        tw, terr = compute_tangency(sub_mean, sub_cov, rf_rate)
                        if gw is not None:
                            gr, gv = portfolio_performance(gw, sub_mean, sub_cov)
                            gmv_results[window_name] = {"weights": gw, "return": gr, "vol": gv}
                        if tw is not None:
                            tr, tv = portfolio_performance(tw, sub_mean, sub_cov)
                            tan_results[window_name] = {"weights": tw, "return": tr, "vol": tv, "sharpe": (tr - rf_rate) / tv}

                st.subheader("🛡️ GMV Portfolio — Sensitivity")
                gmv_rows = []
                for wname, res in gmv_results.items():
                    row = {"Window": wname, "Annual Return": f"{res['return']:.2%}", "Annual Volatility": f"{res['vol']:.2%}"}
                    for i, t in enumerate(valid_tickers):
                        row[t] = f"{res['weights'][i]:.2%}"
                    gmv_rows.append(row)
                st.dataframe(pd.DataFrame(gmv_rows).set_index("Window"), use_container_width=True)

                st.subheader("🎯 Tangency Portfolio — Sensitivity")
                tan_rows = []
                for wname, res in tan_results.items():
                    row = {"Window": wname, "Annual Return": f"{res['return']:.2%}", "Annual Volatility": f"{res['vol']:.2%}", "Sharpe Ratio": f"{res['sharpe']:.2f}"}
                    for i, t in enumerate(valid_tickers):
                        row[t] = f"{res['weights'][i]:.2%}"
                    tan_rows.append(row)
                st.dataframe(pd.DataFrame(tan_rows).set_index("Window"), use_container_width=True)

                st.subheader("📊 Weight Comparison Across Windows")
                import plotly.graph_objects as go
                for port_name, results in [("GMV", gmv_results), ("Tangency", tan_results)]:
                    fig = go.Figure()
                    for wname, res in results.items():
                        fig.add_trace(go.Bar(name=wname, x=valid_tickers, y=res["weights"], text=[f"{w:.1%}" for w in res["weights"]], textposition="outside"))
                    fig.update_layout(barmode="group", title=f"{port_name} Weights Across Estimation Windows", xaxis_title="Asset", yaxis_title="Weight", yaxis_tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.title("📊 Interactive Portfolio Analytics")
    st.markdown("""
    Welcome! Use the **sidebar** to configure your analysis:
    1. Enter **3 to 10 stock tickers**
    2. Select a **date range** (minimum 2 years)
    3. Set your **risk-free rate**
    4. Click **🚀 Run Analysis**
    ---
    ### What This App Does
    | Tab | Content |
    |-----|---------|
    | 📈 Returns & Exploratory | Summary stats, wealth index, return distributions |
    | ⚠️ Risk Analysis | Rolling volatility, drawdowns, Sharpe & Sortino ratios |
    | 🔗 Correlation & Covariance | Heatmaps, rolling correlations, covariance matrix |
    | 💼 Portfolio Optimization | GMV, Tangency, Custom portfolio, Efficient Frontier |
    | 🔍 Sensitivity Analysis | How results change across estimation windows |
    """)
