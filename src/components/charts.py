import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_wealth_index(wealth_df: pd.DataFrame, selected: list) -> go.Figure:
    fig = go.Figure()
    for col in selected:
        if col in wealth_df.columns:
            fig.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df[col], mode="lines", name=col))
    fig.update_layout(title="Growth of $10,000 Investment", xaxis_title="Date", yaxis_title="Portfolio Value ($)", yaxis_tickprefix="$", yaxis_tickformat=",.0f", hovermode="x unified")
    return fig

def plot_histogram(returns: pd.Series, ticker: str) -> go.Figure:
    from scipy.stats import norm
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=80, histnorm="probability density", name="Daily Returns", opacity=0.6, marker_color="steelblue"))
    fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, mu, sigma), mode="lines", name="Normal Fit", line=dict(color="red", width=2)))
    fig.update_layout(title=f"Return Distribution — {ticker}", xaxis_title="Daily Return", yaxis_title="Density")
    return fig

def plot_qq(returns: pd.Series, ticker: str) -> go.Figure:
    from scipy import stats
    (osm, osr), (slope, intercept, _) = stats.probplot(returns, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Quantiles", marker=dict(color="steelblue", size=4)))
    fig.add_trace(go.Scatter(x=osm, y=np.array(osm)*slope+intercept, mode="lines", name="Normal Line", line=dict(color="red", width=2)))
    fig.update_layout(title=f"Q-Q Plot — {ticker}", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
    return fig

def plot_rolling_volatility(rolling_vol: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in rolling_vol.columns:
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=col))
    fig.update_layout(title="Rolling Annualized Volatility", xaxis_title="Date", yaxis_title="Volatility", yaxis_tickformat=".1%", hovermode="x unified")
    return fig

def plot_drawdown(drawdown: pd.Series, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", fill="tozeroy", name=ticker, line=dict(color="crimson")))
    fig.update_layout(title=f"Drawdown — {ticker}", xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".1%")
    return fig

def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(), colorscale="RdBu", zmid=0, zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}", showscale=True))
    fig.update_layout(title="Correlation Matrix of Daily Returns")
    return fig

def plot_rolling_correlation(roll_corr: pd.Series, ticker1: str, ticker2: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, mode="lines", name=f"{ticker1} vs {ticker2}", line=dict(color="steelblue")))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(title=f"Rolling Correlation: {ticker1} vs {ticker2}", xaxis_title="Date", yaxis_title="Correlation", yaxis_range=[-1, 1])
    return fig

def plot_weights(weights: np.ndarray, tickers: list, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=tickers, y=weights, marker_color="steelblue", text=[f"{w:.1%}" for w in weights], textposition="outside"))
    fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Weight", yaxis_tickformat=".0%", yaxis_range=[0, max(weights)*1.3])
    return fig

def plot_risk_contribution(prc: np.ndarray, tickers: list, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=tickers, y=prc, marker_color="tomato", text=[f"{p:.1%}" for p in prc], textposition="outside"))
    fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Risk Contribution", yaxis_tickformat=".0%", yaxis_range=[0, max(prc)*1.3])
    return fig

def plot_efficient_frontier(frontier_vols, frontier_returns, gmv, tangency, ew, custom, stock_vols, stock_returns, stock_labels, rf, sp500=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frontier_vols, y=frontier_returns, mode="lines", name="Efficient Frontier", line=dict(color="royalblue", width=2)))
    tan_sharpe = (tangency["return"] - rf) / tangency["vol"]
    cal_x = np.linspace(0, max(frontier_vols)*1.3, 100)
    fig.add_trace(go.Scatter(x=cal_x, y=rf + tan_sharpe*cal_x, mode="lines", name="Capital Allocation Line", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=stock_vols, y=stock_returns, mode="markers+text", name="Individual Stocks", text=stock_labels, textposition="top center", marker=dict(color="black", size=8)))
    for p, label, color, symbol, size in [(ew,"Equal Weight","orange","diamond",10),(gmv,"GMV Portfolio","red","star",14),(tangency,"Tangency","green","star",14),(custom,"Custom","purple","diamond",10)]:
        fig.add_trace(go.Scatter(x=[p["vol"]], y=[p["return"]], mode="markers+text", name=label, text=[label], textposition="top center", marker=dict(color=color, symbol=symbol, size=size)))
    if sp500:
        fig.add_trace(go.Scatter(x=[sp500["vol"]], y=[sp500["return"]], mode="markers+text", name="S&P 500", text=["S&P 500"], textposition="top center", marker=dict(color="gray", symbol="x", size=10)))
    fig.add_trace(go.Scatter(x=[0], y=[rf], mode="markers+text", name="Risk-Free Rate", text=["Rf"], textposition="top center", marker=dict(color="purple", size=8)))
    fig.update_layout(title="Efficient Frontier with Capital Allocation Line", xaxis_title="Annualized Volatility", yaxis_title="Annualized Return", xaxis_tickformat=".1%", yaxis_tickformat=".1%", hovermode="closest")
    return fig

def plot_portfolio_comparison(wealth_df: pd.DataFrame) -> go.Figure:
    colors = {"Equal Weight":"orange","GMV Portfolio":"red","Tangency":"green","Custom":"purple","S&P 500":"gray"}
    fig = go.Figure()
    for col in wealth_df.columns:
        fig.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df[col], mode="lines", name=col, line=dict(color=colors.get(col,"steelblue"))))
    fig.update_layout(title="Portfolio Comparison — Growth of $1", xaxis_title="Date", yaxis_title="Cumulative Return", hovermode="x unified")
    return fig
