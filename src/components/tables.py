import pandas as pd

def format_summary_stats(stats: pd.DataFrame) -> pd.DataFrame:
    fmt = stats.copy()
    for col in ["Annualized Return","Annualized Volatility","Min Daily Return","Max Daily Return"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map("{:.2%}".format)
    for col in ["Skewness","Kurtosis"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map("{:.2f}".format)
    return fmt

def format_risk_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    fmt = metrics.copy()
    for col in ["Annualized Return","Annualized Volatility","Max Drawdown"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map("{:.2%}".format)
    for col in ["Sharpe Ratio","Sortino Ratio"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map("{:.2f}".format)
    return fmt

def format_metrics_dict(metrics: dict) -> pd.DataFrame:
    rows = {
        "Annual Return":     f"{metrics['Annual Return']:.2%}",
        "Annual Volatility": f"{metrics['Annual Volatility']:.2%}",
        "Sharpe Ratio":      f"{metrics['Sharpe Ratio']:.2f}",
        "Sortino Ratio":     f"{metrics['Sortino Ratio']:.2f}",
        "Max Drawdown":      f"{metrics['Max Drawdown']:.2%}",
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])

def format_comparison_table(metrics_dict: dict) -> pd.DataFrame:
    rows = []
    for name, m in metrics_dict.items():
        rows.append({
            "Portfolio":         name,
            "Annual Return":     f"{m['Annual Return']:.2%}",
            "Annual Volatility": f"{m['Annual Volatility']:.2%}",
            "Sharpe Ratio":      f"{m['Sharpe Ratio']:.2f}",
            "Sortino Ratio":     f"{m['Sortino Ratio']:.2f}",
            "Max Drawdown":      f"{m['Max Drawdown']:.2%}",
        })
    return pd.DataFrame(rows).set_index("Portfolio")
