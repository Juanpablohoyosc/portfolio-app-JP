import pandas as pd
import numpy as np

TRADING_DAYS = 252

def compute_rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Compute rolling annualized volatility."""
    return returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)

def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from a return series."""
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    return (wealth - running_max) / running_max

def compute_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown."""
    return compute_drawdown(returns).min()

def compute_risk_metrics(returns: pd.DataFrame, rf: float = 0.02) -> pd.DataFrame:
    """Compute Sharpe and Sortino ratios for each asset."""
    metrics = pd.DataFrame(index=returns.columns)
    ann_return = returns.mean() * TRADING_DAYS
    ann_vol    = returns.std() * np.sqrt(TRADING_DAYS)
    metrics["Annualized Return"]     = ann_return
    metrics["Annualized Volatility"] = ann_vol
    metrics["Sharpe Ratio"]          = (ann_return - rf) / ann_vol
    downside = returns.copy()
    downside[downside > 0] = 0
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)
    metrics["Sortino Ratio"] = (ann_return - rf) / downside_vol
    metrics["Max Drawdown"]  = returns.apply(compute_max_drawdown)
    return metrics
